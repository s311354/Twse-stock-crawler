"""Application orchestration for point-in-time MOPS fundamental data."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from collections.abc import Callable

import pandas as pd

from application.filing_ingestion import FilingIngestionService
from application.filing_refresh_planner import FilingRefreshPlanner
from application.fundamental_ports import FilingCatalogPort
from application.fundamental_ports import FilingDocumentPort
from application.fundamental_ports import FilingParserPort
from application.fundamental_ports import FinancialBatchPort
from application.fundamental_ports import FundamentalRepositoryPort
from application.fundamental_ports import FundamentalUnitOfWork
from domain.data_provenance import MOPS_PUBLIC_ANNOUNCEMENT
from domain.errors import DataSourceError
from domain.errors import NoDataError
from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_derivation import derive_fundamentals
from domain.fundamental_models import FilingAnnouncement
from domain.fundamental_models import FilingCandidate
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
        self._daily_financials: dict[str, FundamentalSnapshot] = {}

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
        logging.info(
            "TWSE financial batch loaded companies=%s observed_at=%s requests=2",
            len(self._daily_financials),
            observed_at.isoformat(),
        )
        return len(self._daily_financials)

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
        effective_signal = self._effective_signal_time(parsed_signal)
        local_year = parsed_signal.tz_convert("Asia/Taipei").year
        minimum_year = local_year - history_years - 1
        fallback_available_at = self._utc_now().isoformat()
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
                    parsed_signal.to_pydatetime(),
                )
            )
            batch_snapshot = self._eligible_batch_snapshot(symbol, effective_signal)
            batch_state = repository.load_financial_batch_state(
                symbol,
                uow=uow,
            )
            batch_requires_refresh = self._batch_requires_refresh(
                batch_snapshot,
                batch_state,
            )
            if (
                not monthly_refresh
                and not batch_requires_refresh
                and batch_snapshot is not None
                and self._has_complete_cached_history(
                    cached_filings,
                    history_years=history_years,
                    latest_batch_period=batch_snapshot.period,
                    as_of=effective_signal,
                )
            ):
                bundle, cache_hit, filing_count = self._load_cached_bundle(
                    symbol,
                    parsed_signal=effective_signal,
                    sector=sector,
                    uow=uow,
                )
                bundle = self._attach_batch_snapshot(bundle, batch_snapshot)
                if batch_state is None:
                    self._save_batch_state(
                        symbol,
                        batch_snapshot,
                        uow=uow,
                    )
                logging.info(
                    "MOPS fundamentals symbol=%s remote_refresh_skipped=true "
                    "reason=twse_batch_period_unchanged batch_period=%sQ%s "
                    "filings=%s bundle_cache_hit=%s",
                    symbol,
                    batch_snapshot.fiscal_year,
                    batch_snapshot.quarter,
                    filing_count,
                    cache_hit,
                )
                return bundle
            try:
                catalog = self.filing_catalog.discover(symbol)
                catalog = [
                    candidate
                    for candidate in catalog
                    if minimum_year <= candidate.fiscal_year <= local_year
                ]
                announcement_years = (
                    list(range(minimum_year, local_year + 1))
                    if monthly_refresh
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
                refresh_candidates = self._select_refresh_candidates(
                    dated_catalog,
                    parsed_signal,
                )
                corrected_periods = (
                    self._find_corrected_periods(refresh_candidates, cached_filings)
                    if monthly_refresh
                    else set()
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
                effective_signal = self._advance_live_evaluation_time(
                    effective_signal,
                    ingested_filings,
                )
            except DataSourceError as refresh_error:
                if not cached_filings:
                    raise
                return self._load_cached_fallback(
                    symbol,
                    parsed_signal=self._effective_signal_time(parsed_signal),
                    sector=sector,
                    uow=uow,
                    refresh_error=refresh_error,
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
            bundle, cache_hit = self._load_or_derive_bundle(
                symbol,
                parsed_signal=effective_signal,
                resolved_sector=resolved_sector,
                filed_financials=filed_financials,
                uow=uow,
            )
            bundle = self._bundle_as_of(bundle, effective_signal)
            if batch_snapshot is not None:
                bundle = self._attach_batch_snapshot(bundle, batch_snapshot)
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
                    self._save_batch_state(
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
            if monthly_refresh:
                repository.mark_full_refresh(
                    symbol,
                    effective_signal.isoformat(),
                    uow=uow,
                )
        logging.info(
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
        bundle, cache_hit = self._load_or_derive_bundle(
            symbol,
            parsed_signal=parsed_signal,
            resolved_sector=resolved_sector,
            filed_financials=filed_financials,
            uow=uow,
        )
        return (
            self._bundle_as_of(bundle, parsed_signal),
            cache_hit,
            len(filed_financials),
        )

    def _eligible_batch_snapshot(
        self,
        symbol: str,
        signal_time: pd.Timestamp,
    ) -> FundamentalSnapshot | None:
        snapshot = self._daily_financials.get(symbol)
        if snapshot is None:
            return None
        observed_at = pd.to_datetime(snapshot.observed_at, utc=True, errors="coerce")
        if pd.isna(observed_at) or observed_at > signal_time:
            return None
        return snapshot

    @staticmethod
    def _batch_requires_refresh(
        snapshot: FundamentalSnapshot | None,
        state: tuple[tuple[int, int], str] | None,
    ) -> bool:
        if snapshot is None or state is None:
            return False
        prior_period, prior_hash = state
        return snapshot.period > prior_period or (
            snapshot.period == prior_period
            and snapshot.content_sha256 != prior_hash
        )

    def _save_batch_state(
        self,
        symbol: str,
        snapshot: FundamentalSnapshot,
        *,
        uow: FundamentalUnitOfWork,
    ) -> None:
        self.repository.save_financial_batch_state(
            symbol,
            snapshot.fiscal_year,
            snapshot.quarter,
            snapshot.content_sha256,
            snapshot.observed_at,
            uow=uow,
        )

    @staticmethod
    def _has_complete_cached_history(
        cached_filings: list[FilingRecord],
        *,
        history_years: int,
        latest_batch_period: tuple[int, int],
        as_of: pd.Timestamp,
    ) -> bool:
        eligible = []
        for filing in cached_filings:
            available_at = pd.to_datetime(
                filing.available_at,
                utc=True,
                errors="coerce",
            )
            if not pd.isna(available_at) and available_at <= as_of:
                eligible.append(filing)
        if not eligible:
            return False
        periods = {(filing.fiscal_year, filing.quarter) for filing in eligible}
        latest_period = max(periods)
        if latest_period < latest_batch_period:
            return False
        annual_years = {year for year, quarter in periods if quarter == 4}
        if len(annual_years) < history_years:
            return False
        latest_year, latest_quarter = latest_period
        if latest_quarter < 4:
            required = {
                latest_period,
                (latest_year - 1, latest_quarter),
                (latest_year - 1, 4),
            }
            if not required.issubset(periods):
                return False
        return True

    @staticmethod
    def _attach_batch_snapshot(
        bundle: FundamentalBundle,
        batch_snapshot: FundamentalSnapshot,
    ) -> FundamentalBundle:
        snapshot = dict(bundle.snapshot)
        snapshot.update(batch_snapshot.as_snapshot_fields())
        return FundamentalBundle(
            annual_fundamentals=bundle.annual_fundamentals,
            snapshot=snapshot,
            as_of=bundle.as_of,
        )

    def _select_refresh_candidates(
        self,
        candidates: list[FilingCandidate],
        signal_time: pd.Timestamp,
    ) -> list[FilingCandidate]:
        """Separate cache seeding from point-in-time fact eligibility.

        A live run may download a catalog filing without an exact publication
        timestamp. The stored document becomes usable only from its
        FIRST_OBSERVED_DOWNLOAD timestamp; historical runs never download or
        consume that undated candidate.
        """

        selected: list[FilingCandidate] = []
        for candidate in candidates:
            if candidate.available_at is None:
                if self.include_current_observations:
                    selected.append(candidate)
                continue
            available_at = pd.to_datetime(
                candidate.available_at,
                utc=True,
                errors="coerce",
            )
            if not pd.isna(available_at) and available_at <= signal_time:
                selected.append(candidate)
        return selected

    def _effective_signal_time(self, requested: pd.Timestamp) -> pd.Timestamp:
        if not self.include_current_observations:
            return requested
        observed = pd.Timestamp(self._utc_now())
        return max(requested, observed)

    def _advance_live_evaluation_time(
        self,
        current: pd.Timestamp,
        ingested_filings: list[FilingRecord],
    ) -> pd.Timestamp:
        """Make same-run first observations eligible without backdating them."""

        if not self.include_current_observations or not ingested_filings:
            return current
        observed_times = [
            pd.to_datetime(filing.available_at, utc=True, errors="raise")
            for filing in ingested_filings
        ]
        return max(current, *observed_times)

    def _utc_now(self) -> dt.datetime:
        current = self._now_provider()
        if current.tzinfo is None:
            raise ValueError("MOPS now_provider must return a timezone-aware datetime")
        return current.astimezone(dt.timezone.utc)

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
        bundle, cache_hit = self._load_or_derive_bundle(
            symbol,
            parsed_signal=parsed_signal,
            resolved_sector=resolved_sector,
            filed_financials=filed_financials,
            uow=uow,
        )
        bundle = self._bundle_as_of(bundle, parsed_signal)
        batch_snapshot = self._eligible_batch_snapshot(symbol, parsed_signal)
        if batch_snapshot is not None:
            bundle = self._attach_batch_snapshot(bundle, batch_snapshot)
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
    def _bundle_as_of(
        bundle: FundamentalBundle,
        evaluated_at: pd.Timestamp,
    ) -> FundamentalBundle:
        return FundamentalBundle(
            annual_fundamentals=bundle.annual_fundamentals,
            snapshot=bundle.snapshot,
            as_of=evaluated_at.isoformat(),
        )

    def _load_or_derive_bundle(
        self,
        symbol: str,
        *,
        parsed_signal: pd.Timestamp,
        resolved_sector: str,
        filed_financials: list[FiledFinancials],
        uow: FundamentalUnitOfWork,
    ) -> tuple[FundamentalBundle, bool]:
        repository = self.repository
        filing_set_sha256 = repository.filing_set_hash(filed_financials)
        snapshot_cache_key = hashlib.sha256(
            "{}:{}".format(
                FUNDAMENTAL_DERIVATION_VERSION,
                filing_set_sha256,
            ).encode("ascii")
        ).hexdigest()
        bundle = repository.load_fundamental_bundle(
            symbol,
            snapshot_cache_key,
            resolved_sector,
            uow=uow,
        )
        if bundle is not None:
            return bundle, True
        bundle = derive_fundamentals(
            filed_financials,
            signal_date=parsed_signal,
            sector=resolved_sector,
        )
        repository.save_fundamental_bundle(
            symbol,
            snapshot_cache_key,
            resolved_sector,
            bundle,
            uow=uow,
        )
        return bundle, False

    @staticmethod
    def _apply_availability(
        catalog: list[FilingCandidate],
        cached_filings: list[FilingRecord],
        announcements: list[FilingAnnouncement],
    ) -> list[FilingCandidate]:
        latest_cached: dict[tuple[int, int], FilingRecord] = {}
        for filing in sorted(cached_filings, key=lambda item: item.revision_number):
            latest_cached[(filing.fiscal_year, filing.quarter)] = filing
        available_by_period = {
            (int(announcement.fiscal_year), int(announcement.quarter)): str(
                announcement.available_at
            )
            for announcement in announcements
        }
        dated: list[FilingCandidate] = []
        for candidate in catalog:
            period = candidate.fiscal_year, candidate.quarter
            existing = latest_cached.get(period)
            available_at = available_by_period.get(period)
            if available_at is None and existing is not None:
                available_at = existing.available_at
            dated.append(
                FilingCandidate(
                    symbol=candidate.symbol,
                    fiscal_year=candidate.fiscal_year,
                    quarter=candidate.quarter,
                    report_id=candidate.report_id,
                    source_url=candidate.source_url,
                    available_at=available_at,
                    period_end=candidate.period_end,
                    statement_scope=candidate.statement_scope,
                    audit_status=candidate.audit_status,
                    availability_source=(
                        MOPS_PUBLIC_ANNOUNCEMENT
                        if period in available_by_period
                        else existing.availability_source
                        if existing is not None
                        else candidate.availability_source
                    ),
                )
            )
        return dated

    def _find_corrected_periods(
        self,
        candidates: list[FilingCandidate],
        cached_filings: list[FilingRecord],
    ) -> set[tuple[int, int]]:
        latest_cached: dict[tuple[int, int], FilingRecord] = {}
        for filing in sorted(cached_filings, key=lambda item: item.revision_number):
            latest_cached[(filing.fiscal_year, filing.quarter)] = filing
        corrected: set[tuple[int, int]] = set()
        for candidate in candidates:
            period = candidate.fiscal_year, candidate.quarter
            existing = latest_cached.get(period)
            if existing is None:
                continue
            correction = self.filing_catalog.latest_correction(candidate)
            if correction is not None and (
                existing.correction_announced_on is None
                or correction.announced_on > existing.correction_announced_on
            ):
                corrected.add(period)
        return corrected

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

    @staticmethod
    def _select_required_filings(
        filings: list[FilingCandidate],
        history_years: int,
    ) -> list[FilingCandidate]:
        return FilingRefreshPlanner.select_required_filings(filings, history_years)


__all__ = ["MopsFundamentalService"]
