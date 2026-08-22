"""Application orchestration for point-in-time MOPS fundamental data."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable

import pandas as pd

from application.fundamental_deriver import FundamentalDeriver
from application.fundamental_point_in_time import PointInTimeResolver
from application.fundamental_refresh import RefreshPlanner
from application.fundamental_snapshot import BatchSnapshotResolution
from application.fundamental_snapshot import SnapshotAssembler
from application.filing_ingestion import FilingIngestionService
from application.filing_refresh_planner import FilingRefreshPlanner
from application.fundamental_ports import FilingCatalogPort
from application.fundamental_ports import FilingDocumentPort
from application.fundamental_ports import FilingParserPort
from application.fundamental_ports import FinancialBatchPort
from application.fundamental_ports import FundamentalRepositoryPort
from application.fundamental_ports import FundamentalUnitOfWork
from domain.base_quality_features import BaseQualityFeatureProducer
from domain.errors import DataSourceError
from domain.errors import NoDataError
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_derivation import derive_fundamentals
from domain.fundamental_models import FilingAnnouncement
from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FilingCorrection
from domain.fundamental_models import FilingRecord
from domain.fundamental_models import FundamentalSnapshot


class MopsFundamentalService:
    """Refresh the minimum filing set needed for annual history and TTM facts."""

    def __init__(
        self,
        filing_catalog: FilingCatalogPort,
        repository: FundamentalRepositoryPort | None = None,
        planner: FilingRefreshPlanner | None = None,
        *,
        filing_documents: FilingDocumentPort | None = None,
        filing_parser: FilingParserPort | None = None,
        include_current_observations: bool = False,
        now_provider: Callable[[], dt.datetime] | None = None,
        financial_batch_client: FinancialBatchPort | None = None,
        base_quality_feature_producer: BaseQualityFeatureProducer | None = None,
    ) -> None:
        if repository is None:
            raise ValueError("repository port is required")
        if filing_documents is None:
            raise ValueError("filing document port is required")
        if filing_parser is None:
            raise ValueError("filing parser port is required")
        self.filing_catalog = filing_catalog
        self.repository = repository
        self.filing_ingestion = FilingIngestionService(
            filing_catalog,
            filing_documents,
            filing_parser,
            repository,
        )
        self.planner = planner or FilingRefreshPlanner()
        self.include_current_observations = include_current_observations
        self._now_provider = now_provider or (
            lambda: dt.datetime.now(dt.timezone.utc)
        )
        self._company_sectors: dict[str, str] | None = None
        self.financial_batch_client = financial_batch_client
        self.base_quality_feature_producer = (
            base_quality_feature_producer or BaseQualityFeatureProducer()
        )
        self.refresh_planner = RefreshPlanner(
            filing_catalog,
            include_current_observations=include_current_observations,
        )
        self.point_in_time = PointInTimeResolver(
            include_current_observations=include_current_observations,
            now_provider=self._now_provider,
        )
        self.snapshot_assembler = SnapshotAssembler(
            repository,
            self.base_quality_feature_producer,
        )
        self.fundamental_deriver = FundamentalDeriver(
            repository,
            derive_bundle=lambda *args, **kwargs: derive_fundamentals(
                *args,
                **kwargs,
            ),
        )
        self._daily_financials: dict[str, FundamentalSnapshot] = {}
        self._daily_corrections: dict[
            str,
            dict[tuple[int, int], FilingCorrection],
        ] = {}
        self._individual_scope_fallback_symbols: set[str] = set()

    def reset_scope_fallback_diagnostics(self) -> None:
        self._individual_scope_fallback_symbols.clear()

    def consume_scope_fallback_symbols(self) -> tuple[str, ...]:
        symbols = tuple(sorted(self._individual_scope_fallback_symbols))
        self._individual_scope_fallback_symbols.clear()
        return symbols

    def prepare_daily_batch(self, observed_at: dt.datetime) -> int:
        """Fetch both all-company statements once for the complete daily run."""

        if observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.financial_batch_client is None:
            self._daily_financials = {}
            return 0
        self._daily_financials = self.financial_batch_client.load(
            observed_at=observed_at,
        )
        self.snapshot_assembler.persist_daily_financials(
            self._daily_financials
        )
        logging.info(
            "TWSE financial batch loaded companies=%s observed_at=%s requests=2",
            len(self._daily_financials),
            observed_at.isoformat(),
        )
        return len(self._daily_financials)

    def prepare_daily_revisions(self, observed_at: dt.datetime) -> int:
        """Discover recent market-wide corrections once before symbol loading."""

        if observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        periods = self.refresh_planner.recent_reporting_periods(
            observed_at,
            self._daily_financials,
        )
        self._daily_corrections = {}
        discover = getattr(self.filing_catalog, "discover_corrections", None)
        if not callable(discover):
            return 0

        corrections = discover(periods)
        indexed: dict[str, dict[tuple[int, int], FilingCorrection]] = {}
        for correction in corrections:
            by_period = indexed.setdefault(correction.symbol, {})
            period = correction.fiscal_year, correction.quarter
            existing = by_period.get(period)
            if existing is None or correction.announced_on > existing.announced_on:
                by_period[period] = correction
        self._daily_corrections = indexed
        logging.info(
            "MOPS daily correction scan periods=%s events=%s affected_symbols=%s",
            ",".join("{}Q{}".format(*period) for period in periods),
            len(corrections),
            len(indexed),
        )
        return len(corrections)

    def load_cached_bundles(
        self,
        symbols: list[str],
        *,
        signal_date: str,
        sector: str,
        history_years: int = 5,
    ) -> dict[str, FundamentalBundle]:
        """Preload safe active snapshots while leaving stale entries to ``load``."""

        if history_years < 3:
            raise ValueError("history_years must be at least 3")
        parsed_signal = pd.to_datetime(signal_date, utc=True, errors="raise")
        effective_signal = self.point_in_time.effective_signal_time(
            parsed_signal
        )
        symbol_sectors = {
            symbol: self._resolve_sector(symbol, sector)
            for symbol in dict.fromkeys(str(item).strip() for item in symbols)
            if symbol
        }
        cached = self.repository.load_cached_bundles(
            symbol_sectors,
            as_of=effective_signal.isoformat(),
            derivation_version=FUNDAMENTAL_DERIVATION_VERSION,
        )
        reusable: dict[str, FundamentalBundle] = {}
        for symbol, entry in cached.items():
            if entry.annual_history_years < history_years:
                continue
            if self.planner.is_monthly_refresh_due(
                entry.full_catalog_checked_at_epoch,
                effective_signal.to_pydatetime(),
            ):
                continue
            if self._daily_corrections.get(symbol):
                continue
            current_batch = self.snapshot_assembler.eligible_batch_snapshot(
                self._daily_financials,
                symbol,
                effective_signal,
            )
            if (
                current_batch is not None
                and current_batch.period > entry.latest_period
            ):
                continue

            with self.repository.unit_of_work() as uow:
                resolution = self._resolve_batch_snapshot(
                    symbol,
                    effective_signal,
                    uow=uow,
                )
                bundle = self.snapshot_assembler.attach_base_evidence(
                    self.point_in_time.bundle_as_of(
                        entry.bundle,
                        effective_signal,
                    ),
                    resolution,
                    symbol=symbol,
                    signal_time=effective_signal,
                    uow=uow,
                )
                batch_state = self.repository.load_financial_batch_state(
                    symbol,
                    uow=uow,
                )
                if (
                    current_batch is not None
                    and entry.latest_period >= current_batch.period
                    and batch_state is None
                ):
                    self.snapshot_assembler.save_batch_state(
                        symbol,
                        current_batch,
                        uow=uow,
                    )
            snapshot = dict(bundle.snapshot)
            snapshot["remote_refresh_skipped"] = True
            reusable[symbol] = FundamentalBundle(
                annual_fundamentals=bundle.annual_fundamentals,
                snapshot=snapshot,
                as_of=bundle.as_of,
            )

        logging.debug(
            "MOPS snapshot preload requested=%s candidates=%s reused=%s "
            "fallback=%s derivation_version=%s",
            len(symbol_sectors),
            len(cached),
            len(reusable),
            len(symbol_sectors) - len(reusable),
            FUNDAMENTAL_DERIVATION_VERSION,
            extra={
                "source": "MOPS_CACHE",
                "status": "BULK_PRELOAD",
                "cache_hit": bool(reusable),
            },
        )
        return reusable

    def load(
        self,
        symbol: str,
        *,
        signal_date: str,
        sector: str,
        history_years: int = 5,
    ) -> FundamentalBundle:
        if history_years < 3:
            raise ValueError("history_years must be at least 3")
        parsed_signal = pd.to_datetime(signal_date, utc=True, errors="raise")
        effective_signal = self.point_in_time.effective_signal_time(
            parsed_signal
        )
        local_year = effective_signal.tz_convert("Asia/Taipei").year
        minimum_year = local_year - history_years - 1
        fallback_available_at = self.point_in_time.utc_now().isoformat()
        repository = self.repository
        with repository.unit_of_work() as uow:
            cached_filings = repository.load_filings(
                symbol,
                uow=uow,
            )
            monthly_refresh = (
                not cached_filings
                or self.planner.is_monthly_refresh_due(
                    repository.last_full_refresh_epoch(
                        symbol,
                        uow=uow,
                    ),
                    effective_signal.to_pydatetime(),
                )
            )
            batch_resolution = self._resolve_batch_snapshot(
                symbol,
                effective_signal,
                uow=uow,
            )
            batch_snapshot = batch_resolution.snapshot
            batch_state = repository.load_financial_batch_state(
                symbol,
                uow=uow,
            )
            event_corrected_periods = (
                self.refresh_planner.event_corrected_periods(
                    symbol,
                    cached_filings,
                    self._daily_corrections,
                )
            )
            batch_has_new_period = self.refresh_planner.batch_has_new_period(
                batch_snapshot,
                cached_filings,
                as_of=effective_signal,
            )
            cache_history_complete = (
                self.refresh_planner.has_complete_cached_filing_history(
                    cached_filings,
                    history_years=history_years,
                    as_of=effective_signal,
                )
            )
            if (
                not monthly_refresh
                and not batch_has_new_period
                and not event_corrected_periods
                and cache_history_complete
            ):
                bundle, cache_hit, filing_count = self._load_cached_bundle(
                    symbol,
                    parsed_signal=effective_signal,
                    sector=sector,
                    uow=uow,
                )
                bundle = self.snapshot_assembler.attach_base_evidence(
                    bundle,
                    batch_resolution,
                    symbol=symbol,
                    signal_time=effective_signal,
                    uow=uow,
                )
                if batch_snapshot is not None and batch_state is None:
                    self.snapshot_assembler.save_batch_state(
                        symbol,
                        batch_snapshot,
                        uow=uow,
                    )
                logging.debug(
                    "MOPS fundamentals symbol=%s remote_refresh_skipped=true "
                    "reason=mops_cache_complete revision_event=false "
                    "batch_period=%s filings=%s bundle_cache_hit=%s",
                    symbol,
                    (
                        "{}Q{}".format(
                            batch_snapshot.fiscal_year,
                            batch_snapshot.quarter,
                        )
                        if batch_snapshot is not None
                        else "N/A"
                    ),
                    filing_count,
                    cache_hit,
                    extra={
                        "symbol": symbol,
                        "source": "MOPS_CACHE",
                        "status": "CACHE_HIT",
                        "cache_hit": True,
                    },
                )
                return bundle
            try:
                catalog = self.filing_catalog.discover(symbol)
                catalog = [
                    candidate
                    for candidate in catalog
                    if minimum_year <= candidate.fiscal_year <= local_year
                ]
                if any(
                    candidate.statement_scope == "individual"
                    for candidate in catalog
                ):
                    self._individual_scope_fallback_symbols.add(symbol)
                announcement_years = (
                    list(range(minimum_year, local_year + 1))
                    if not cache_history_complete
                    else [local_year]
                )
                announcements = self.filing_catalog.discover_announcements(
                    symbol,
                    announcement_years,
                )
                dated_catalog = self._apply_availability(
                    catalog,
                    cached_filings,
                    announcements,
                )
                refresh_candidates = (
                    self.refresh_planner.select_refresh_candidates(
                        dated_catalog,
                        parsed_signal,
                    )
                )
                corrected_periods = set(event_corrected_periods)
                if monthly_refresh:
                    corrected_periods.update(
                        self.refresh_planner.find_corrected_periods(
                            refresh_candidates,
                            cached_filings,
                        )
                    )
                plan = self.planner.plan(
                    refresh_candidates,
                    cached_filings,
                    history_years=history_years,
                    corrected_periods=corrected_periods,
                )
                first_observation_seed = sum(
                    candidate.available_at is None for candidate in plan.download
                )
                ingested_filings: list[FilingRecord] = []
                for candidate in plan.download + plan.verify_revision:
                    ingested_filings.append(
                        self.filing_ingestion.ingest(
                            candidate,
                            fallback_available_at=fallback_available_at,
                            uow=uow,
                        )
                    )
                effective_signal = self.point_in_time.advance_live_evaluation_time(
                    effective_signal,
                    ingested_filings,
                )
            except DataSourceError as refresh_error:
                if not cached_filings:
                    raise
                return self._load_cached_fallback(
                    symbol,
                    parsed_signal=self.point_in_time.effective_signal_time(
                        parsed_signal
                    ),
                    sector=sector,
                    uow=uow,
                    refresh_error=refresh_error,
                )

            if monthly_refresh:
                repository.mark_full_refresh(
                    symbol,
                    effective_signal.isoformat(),
                    uow=uow,
                )
            filed_financials = repository.load_filed_financials(
                symbol,
                effective_signal.isoformat(),
                uow=uow,
            )
            if not filed_financials:
                raise NoDataError(
                    "MOPS refresh for {} produced no point-in-time eligible filings; "
                    "empty refresh state and snapshots were not stored".format(symbol),
                    retryable=False,
                )
            resolved_sector = self._resolve_sector(symbol, sector)
            bundle, cache_hit = self.fundamental_deriver.load_or_derive(
                symbol,
                parsed_signal=effective_signal,
                resolved_sector=resolved_sector,
                filed_financials=filed_financials,
                uow=uow,
            )
            bundle = self.point_in_time.bundle_as_of(bundle, effective_signal)
            bundle = self.snapshot_assembler.attach_base_evidence(
                bundle,
                batch_resolution,
                symbol=symbol,
                signal_time=effective_signal,
                uow=uow,
            )
            if batch_snapshot is not None:
                latest_filing_period = max(
                    (filing.fiscal_year, filing.quarter)
                    for filing in filed_financials
                )
                state_is_not_newer = (
                    batch_state is None
                    or batch_snapshot.period >= batch_state[0]
                )
                if (
                    latest_filing_period >= batch_snapshot.period
                    and state_is_not_newer
                ):
                    self.snapshot_assembler.save_batch_state(
                        symbol,
                        batch_snapshot,
                        uow=uow,
                    )
                else:
                    logging.warning(
                        "TWSE financial batch period=%sQ%s is newer than "
                        "MOPS cache period=%sQ%s for symbol=%s; "
                        "batch baseline not advanced",
                        batch_snapshot.fiscal_year,
                        batch_snapshot.quarter,
                        latest_filing_period[0],
                        latest_filing_period[1],
                        symbol,
                    )
            repository.mark_recent_revision_check(
                symbol,
                effective_signal.isoformat(),
                uow=uow,
            )
        logging.debug(
            "MOPS fundamentals symbol=%s filings=%s annual_years=%s "
            "reuse=%s download=%s verify_revision=%s monthly_refresh=%s "
            "first_observation_seed=%s bundle_cache_hit=%s evaluation_at=%s "
            "snapshot_available_at=%s derivation_version=%s",
            symbol,
            len(filed_financials),
            len(bundle.annual_fundamentals),
            len(plan.reuse),
            len(plan.download),
            len(plan.verify_revision),
            monthly_refresh,
            first_observation_seed,
            cache_hit,
            effective_signal.isoformat(),
            bundle.snapshot.get("available_at", "N/A"),
            FUNDAMENTAL_DERIVATION_VERSION,
        )
        return bundle

    def _load_cached_bundle(
        self,
        symbol: str,
        *,
        parsed_signal: pd.Timestamp,
        sector: str,
        uow: FundamentalUnitOfWork,
    ) -> tuple[FundamentalBundle, bool, int]:
        repository = self.repository
        filed_financials = repository.load_filed_financials(
            symbol,
            parsed_signal.isoformat(),
            uow=uow,
        )
        if not filed_financials:
            raise NoDataError(
                "Cached MOPS filings for {} are not point-in-time eligible".format(
                    symbol
                ),
                retryable=False,
            )
        resolved_sector = self._resolve_sector(symbol, sector)
        bundle, cache_hit = self.fundamental_deriver.load_or_derive(
            symbol,
            parsed_signal=parsed_signal,
            resolved_sector=resolved_sector,
            filed_financials=filed_financials,
            uow=uow,
        )
        return (
            self.point_in_time.bundle_as_of(bundle, parsed_signal),
            cache_hit,
            len(filed_financials),
        )

    def _resolve_batch_snapshot(
        self,
        symbol: str,
        signal_time: pd.Timestamp,
        *,
        uow: FundamentalUnitOfWork,
    ) -> BatchSnapshotResolution:
        return self.snapshot_assembler.resolve(
            self._daily_financials,
            symbol,
            signal_time,
            uow=uow,
        )

    def _load_cached_fallback(
        self,
        symbol: str,
        *,
        parsed_signal: pd.Timestamp,
        sector: str,
        uow: FundamentalUnitOfWork,
        refresh_error: DataSourceError,
    ) -> FundamentalBundle:
        repository = self.repository
        filed_financials = repository.load_filed_financials(
            symbol,
            parsed_signal.isoformat(),
            uow=uow,
        )
        if not filed_financials:
            raise refresh_error
        resolved_sector = self._resolve_sector(symbol, sector)
        bundle, cache_hit = self.fundamental_deriver.load_or_derive(
            symbol,
            parsed_signal=parsed_signal,
            resolved_sector=resolved_sector,
            filed_financials=filed_financials,
            uow=uow,
        )
        bundle = self.point_in_time.bundle_as_of(bundle, parsed_signal)
        batch_resolution = self._resolve_batch_snapshot(
            symbol,
            parsed_signal,
            uow=uow,
        )
        bundle = self.snapshot_assembler.attach_base_evidence(
            bundle,
            batch_resolution,
            symbol=symbol,
            signal_time=parsed_signal,
            uow=uow,
        )
        logging.warning(
            "MOPS refresh status=%s symbol=%s cache_fallback=true filings=%s "
            "bundle_cache_hit=%s detail=%s",
            refresh_error.category.value.upper(),
            symbol,
            len(filed_financials),
            cache_hit,
            refresh_error,
        )
        return bundle

    @staticmethod
    def _apply_availability(
        catalog: list[FilingCandidate],
        cached_filings: list[FilingRecord],
        announcements: list[FilingAnnouncement],
    ) -> list[FilingCandidate]:
        return RefreshPlanner.apply_availability(
            catalog,
            cached_filings,
            announcements,
        )

    def _resolve_sector(self, symbol: str, fallback: str) -> str:
        if self._company_sectors is None:
            loader = getattr(
                self.filing_catalog,
                "load_company_sectors",
                None,
            )
            try:
                self._company_sectors = (
                    loader() if callable(loader) else {}
                )
            except DataSourceError as error:
                logging.warning(
                    "Unable to resolve TWSE industry code for %s; using fallback sector=%s: %s",
                    symbol,
                    fallback,
                    error,
                )
                self._company_sectors = {}
        official_sector = self._company_sectors.get(symbol)
        if official_sector is not None:
            return official_sector
        normalized_fallback = str(fallback).strip()
        return normalized_fallback if normalized_fallback.isdigit() else ""

__all__ = ["MopsFundamentalService"]
