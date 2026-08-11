"""Application orchestration for point-in-time MOPS fundamental data."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from application.filing_ingestion import FilingIngestionService
from application.filing_refresh_planner import FilingRefreshPlanner
from application.fundamental_ports import FilingCatalogPort
from application.fundamental_ports import FilingDocumentPort
from application.fundamental_ports import FilingParserPort
from application.fundamental_ports import FinancialBatchPort
from application.fundamental_ports import FundamentalRepositoryPort
from application.fundamental_ports import FundamentalUnitOfWork
from domain.base_quality_features import BaseQualityFeatureProducer
from domain.data_provenance import MOPS_PUBLIC_ANNOUNCEMENT
from domain.errors import DataSourceError
from domain.errors import NoDataError
from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_derivation import derive_fundamentals
from domain.fundamental_models import FilingAnnouncement
from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FilingCorrection
from domain.fundamental_models import FilingRecord
from domain.fundamental_models import FundamentalSnapshot


@dataclass(frozen=True)
class _BatchSnapshotResolution:
    snapshot: FundamentalSnapshot | None
    source: str
    cache_hit: bool


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
        self._persist_daily_financials()
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
        periods = self._recent_reporting_periods(observed_at)
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
        effective_signal = self._effective_signal_time(parsed_signal)
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
            current_batch = self._eligible_batch_snapshot(
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
                bundle = self._attach_base_evidence(
                    self._bundle_as_of(entry.bundle, effective_signal),
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
                    self._save_batch_state(
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
        effective_signal = self._effective_signal_time(parsed_signal)
        local_year = effective_signal.tz_convert("Asia/Taipei").year
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
            event_corrected_periods = self._event_corrected_periods(
                symbol,
                cached_filings,
            )
            batch_has_new_period = self._batch_has_new_period(
                batch_snapshot,
                cached_filings,
                as_of=effective_signal,
            )
            cache_history_complete = self._has_complete_cached_filing_history(
                cached_filings,
                history_years=history_years,
                as_of=effective_signal,
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
                bundle = self._attach_base_evidence(
                    bundle,
                    batch_resolution,
                    symbol=symbol,
                    signal_time=effective_signal,
                    uow=uow,
                )
                if batch_snapshot is not None and batch_state is None:
                    self._save_batch_state(
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
                refresh_candidates = self._select_refresh_candidates(
                    dated_catalog,
                    parsed_signal,
                )
                corrected_periods = set(event_corrected_periods)
                if monthly_refresh:
                    corrected_periods.update(
                        self._find_corrected_periods(
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
            bundle, cache_hit = self._load_or_derive_bundle(
                symbol,
                parsed_signal=effective_signal,
                resolved_sector=resolved_sector,
                filed_financials=filed_financials,
                uow=uow,
            )
            bundle = self._bundle_as_of(bundle, effective_signal)
            bundle = self._attach_base_evidence(
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

    def _recent_reporting_periods(
        self,
        observed_at: dt.datetime,
    ) -> list[tuple[int, int]]:
        local = observed_at.astimezone(dt.timezone(dt.timedelta(hours=8)))
        completed_quarter = (local.month - 1) // 3
        calendar_latest = (
            (local.year, completed_quarter)
            if completed_quarter > 0
            else (local.year - 1, 4)
        )
        batch_periods = {
            snapshot.period for snapshot in self._daily_financials.values()
        }
        latest = max(batch_periods | {calendar_latest})
        previous = (
            (latest[0], latest[1] - 1)
            if latest[1] > 1
            else (latest[0] - 1, 4)
        )
        return [previous, latest]

    def _event_corrected_periods(
        self,
        symbol: str,
        cached_filings: list[FilingRecord],
    ) -> set[tuple[int, int]]:
        events = self._daily_corrections.get(symbol, {})
        if not events:
            return set()
        latest_cached: dict[tuple[int, int], FilingRecord] = {}
        for filing in sorted(cached_filings, key=lambda item: item.revision_number):
            latest_cached[(filing.fiscal_year, filing.quarter)] = filing
        return {
            period
            for period, correction in events.items()
            if period in latest_cached
            and (
                latest_cached[period].correction_announced_on is None
                or correction.announced_on
                > latest_cached[period].correction_announced_on
            )
        }

    @staticmethod
    def _batch_has_new_period(
        batch_snapshot: FundamentalSnapshot | None,
        cached_filings: list[FilingRecord],
        *,
        as_of: pd.Timestamp,
    ) -> bool:
        if batch_snapshot is None:
            return False
        eligible_periods = []
        for filing in cached_filings:
            available_at = pd.to_datetime(
                filing.available_at,
                utc=True,
                errors="coerce",
            )
            if not pd.isna(available_at) and available_at <= as_of:
                eligible_periods.append((filing.fiscal_year, filing.quarter))
        return not eligible_periods or batch_snapshot.period > max(eligible_periods)

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

    def _resolve_batch_snapshot(
        self,
        symbol: str,
        signal_time: pd.Timestamp,
        *,
        uow: FundamentalUnitOfWork,
    ) -> _BatchSnapshotResolution:
        current = self._eligible_batch_snapshot(symbol, signal_time)
        if current is not None:
            return _BatchSnapshotResolution(
                snapshot=current,
                source="TWSE_OPENAPI_LATEST",
                cache_hit=False,
            )

        history = self.repository.load_financial_batch_snapshots(
            symbol,
            as_of=signal_time.isoformat(),
            uow=uow,
        )
        if not history:
            return _BatchSnapshotResolution(
                snapshot=None,
                source="MOPS_POINT_IN_TIME_BASE_FALLBACK",
                cache_hit=False,
            )

        cached = max(
            history,
            key=lambda item: (item.period, item.observed_at, item.content_sha256),
        )
        logging.debug(
            "TWSE financial batch symbol=%s status=CACHE_FALLBACK "
            "period=%sQ%s observed_at=%s cache_hit=true",
            symbol,
            cached.fiscal_year,
            cached.quarter,
            cached.observed_at,
        )
        return _BatchSnapshotResolution(
            snapshot=cached,
            source="TWSE_OPENAPI_CACHE_FALLBACK",
            cache_hit=True,
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
    def _has_complete_cached_filing_history(
        cached_filings: list[FilingRecord],
        *,
        history_years: int,
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

    def _attach_batch_snapshot(
        self,
        bundle: FundamentalBundle,
        batch_snapshot: FundamentalSnapshot,
        *,
        source: str,
        signal_time: pd.Timestamp,
        uow: FundamentalUnitOfWork,
    ) -> FundamentalBundle:
        self.repository.save_financial_batch_snapshot(batch_snapshot, uow=uow)
        history = self.repository.load_financial_batch_snapshots(
            batch_snapshot.symbol,
            as_of=signal_time.isoformat(),
            uow=uow,
        )
        features = self.base_quality_feature_producer.produce(
            current=batch_snapshot,
            history=history,
            annual_fundamentals=bundle.annual_fundamentals,
            existing_snapshot=bundle.snapshot,
            signal_date=signal_time,
        )
        snapshot = dict(bundle.snapshot)
        batch_fields = batch_snapshot.as_snapshot_fields()
        batch_fields["twse_financial_source"] = source
        snapshot.update(batch_fields)
        snapshot.update(features.as_snapshot_fields())
        snapshot["base_quality_evidence_mode"] = source
        return FundamentalBundle(
            annual_fundamentals=bundle.annual_fundamentals,
            snapshot=snapshot,
            as_of=bundle.as_of,
        )

    def _attach_base_evidence(
        self,
        bundle: FundamentalBundle,
        resolution: _BatchSnapshotResolution,
        *,
        symbol: str,
        signal_time: pd.Timestamp,
        uow: FundamentalUnitOfWork,
    ) -> FundamentalBundle:
        if resolution.snapshot is not None:
            return self._attach_batch_snapshot(
                bundle,
                resolution.snapshot,
                source=resolution.source,
                signal_time=signal_time,
                uow=uow,
            )

        features = self.base_quality_feature_producer.produce(
            current=None,
            history=[],
            annual_fundamentals=bundle.annual_fundamentals,
            existing_snapshot=bundle.snapshot,
            signal_date=signal_time,
        )
        snapshot = dict(bundle.snapshot)
        snapshot.update(features.as_snapshot_fields())
        snapshot["base_quality_evidence_mode"] = resolution.source
        logging.debug(
            "Base quality symbol=%s status=PROVISIONAL source=%s "
            "feature_count=%s cache_hit=false",
            symbol,
            resolution.source,
            len(features.values),
        )
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
        batch_resolution = self._resolve_batch_snapshot(
            symbol,
            parsed_signal,
            uow=uow,
        )
        bundle = self._attach_base_evidence(
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

    def _persist_daily_financials(self) -> None:
        """Seed period history even when a later MOPS refresh is unavailable."""

        save = getattr(self.repository, "save_financial_batch_snapshot", None)
        unit_of_work = getattr(self.repository, "unit_of_work", None)
        if not callable(save) or not callable(unit_of_work):
            return
        with unit_of_work() as uow:
            for snapshot in self._daily_financials.values():
                save(snapshot, uow=uow)

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
        cache_hit = bundle is not None
        if bundle is None:
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
        latest = max(
            filed_financials,
            key=lambda filing: (
                filing.fiscal_year,
                filing.quarter,
                filing.available_at,
                filing.revision_number,
            ),
        )
        available_at = max(
            filed_financials,
            key=lambda filing: pd.to_datetime(
                filing.available_at,
                utc=True,
                errors="raise",
            ),
        ).available_at
        repository.mark_active_fundamental_bundle(
            symbol,
            filing_set_sha256=filing_set_sha256,
            snapshot_cache_key=snapshot_cache_key,
            derivation_version=FUNDAMENTAL_DERIVATION_VERSION,
            available_at=available_at,
            latest_period=(latest.fiscal_year, latest.quarter),
            annual_history_years=self._fast_path_history_years(
                filed_financials
            ),
            uow=uow,
        )
        return bundle, cache_hit

    @staticmethod
    def _fast_path_history_years(
        filings: list[FiledFinancials],
    ) -> int:
        periods = {(filing.fiscal_year, filing.quarter) for filing in filings}
        if not periods:
            return 0
        latest_year, latest_quarter = max(periods)
        if latest_quarter < 4:
            comparable_periods = {
                (latest_year, latest_quarter),
                (latest_year - 1, latest_quarter),
                (latest_year - 1, 4),
            }
            if not comparable_periods.issubset(periods):
                return 0
        return len({year for year, quarter in periods if quarter == 4})

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
