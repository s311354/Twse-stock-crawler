"""Application orchestration for point-in-time MOPS fundamental data."""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Callable

import pandas as pd

from application.fundamental_deriver import FundamentalDeriver
from application.mops_refresh_control import MopsRefreshControl, MopsRefreshPolicy, RefreshDeferred
from application.fundamental_point_in_time import PointInTimeResolver
from application.fundamental_refresh import RefreshPlanner
from application.fundamental_snapshot import BatchSnapshotResolution
from application.fundamental_snapshot import SnapshotAssembler
from application.filing_ingestion import FilingIngestionService
from application.filing_refresh_planner import FilingRefreshPlanner
from application.filing_catalog_refresh import FilingCatalogRefresh
from application.fundamental_ports import FilingCatalogPort
from application.fundamental_ports import FilingDocumentPort
from application.fundamental_ports import FilingParserPort
from application.fundamental_ports import FinancialBatchPort
from application.fundamental_ports import FundamentalRepositoryPort
from application.fundamental_ports import FundamentalUnitOfWork
from domain.base_quality_features import BaseQualityFeatureProducer
from domain.errors import DataSourceError
from domain.errors import DataErrorCategory
from domain.errors import NoDataError
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from domain.fundamental_derivation import CachedFundamentalBundle
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_derivation import derive_fundamentals
from domain.fundamental_models import FilingAnnouncement
from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FilingCorrection
from domain.fundamental_models import FilingRecord
from domain.fundamental_models import FundamentalSnapshot
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION, ixbrl_record
from domain.strategies.low_entry_v6_quality import combine_filing_sources, evaluate_quality, snapshot_record, verification_diagnostics


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
        refresh_policy: MopsRefreshPolicy | None = None,
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
        self.refresh_control = MopsRefreshControl(
            repository, refresh_policy or MopsRefreshPolicy(), self._now_provider,
        )
        for port in {id(filing_catalog): filing_catalog, id(filing_documents): filing_documents}.values():
            configure = getattr(port, "configure_refresh_control", None)
            if callable(configure):
                configure(self.refresh_control)
        self._company_sectors: dict[str, str] | None = None
        self.financial_batch_client = financial_batch_client
        self.base_quality_feature_producer = (
            base_quality_feature_producer or BaseQualityFeatureProducer()
        )
        self.refresh_planner = RefreshPlanner(
            filing_catalog,
            include_current_observations=include_current_observations,
        )
        self.catalog_refresh = FilingCatalogRefresh(
            filing_catalog, repository, self.planner, self.refresh_planner,
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
        self._access_blocked_cache_fallback_symbols: set[str] = set()

    def begin_run(self) -> None:
        self.refresh_control.begin_run()
        reset = getattr(self.filing_catalog, "begin_refresh_run", None)
        if callable(reset):
            reset()

    def migrate_local_cache(self, symbols: list[str], *, signal_date: str, sector: str) -> int:
        """Convert PIT-eligible cached facts before *any* MOPS backfill in this run."""
        signal = self.point_in_time.effective_signal_time(pd.to_datetime(signal_date, utc=True))
        sectors = {symbol: self._resolve_sector(symbol, sector) for symbol in dict.fromkeys(symbols)}
        cached = self.repository.load_cached_bundles(
            sectors, as_of=signal.isoformat(), derivation_version=FUNDAMENTAL_DERIVATION_VERSION,
        )
        migrated = 0
        for symbol in sectors:
            existing = cached.get(symbol)
            if (existing is not None and "les_v6_mops_filings" in existing.bundle.snapshot
                    and existing.bundle.snapshot.get("les_v6_filing_schema_version") == V6_FILING_SCHEMA_VERSION):
                continue
            with self.repository.unit_of_work() as uow:
                try:
                    self._load_cached_bundle(symbol, parsed_signal=signal, sector=sector, uow=uow)
                except NoDataError:
                    continue
                migrated += 1
        return migrated

    def reset_scope_fallback_diagnostics(self) -> None:
        self._individual_scope_fallback_symbols.clear()

    def inspect_local_evidence(self, symbols: list[str], *, signal_date: str, market: str = "TWSE") -> dict[str, dict]:
        """Inspect stored source facts/batch snapshots only; never download to diagnose."""
        if market not in {"TWSE", "TPEX"}:
            raise ValueError("Unsupported fundamental evidence market")
        signal = self.point_in_time.effective_signal_time(pd.to_datetime(signal_date, utc=True))
        result = {}
        with self.repository.unit_of_work() as uow:
            for symbol in dict.fromkeys(symbols):
                filings = self.repository.load_filed_financials(symbol, signal.isoformat(), uow=uow)
                # The batch repository currently contains TWSE observations only.
                # MOPS issuer filings are shared, but a TWSE snapshot cannot
                # certify an OTC issuer or hide its missing-quarter backfill.
                batches = self.repository.load_financial_batch_snapshots(symbol, as_of=signal.isoformat(), uow=uow) if market == "TWSE" else []
                twse = [snapshot_record(item) for item in batches]
                records = combine_filing_sources(
                    [dict(ixbrl_record(symbol, filing), market=market) for filing in filings],
                    twse, as_of=signal,
                )
                diagnostic = verification_diagnostics(records, symbol=symbol, as_of=signal, market=market)
                quality = evaluate_quality(records, symbol=symbol, market=market, as_of=signal)
                diagnostic.update(quality_status=quality.status, quality_reasons=list(quality.reasons))
                result[symbol] = diagnostic
        return result

    def consume_scope_fallback_symbols(self) -> tuple[str, ...]:
        symbols = tuple(sorted(self._individual_scope_fallback_symbols))
        self._individual_scope_fallback_symbols.clear()
        return symbols

    def reset_access_blocked_diagnostics(self) -> None:
        self._access_blocked_cache_fallback_symbols.clear()

    def consume_access_blocked_cache_fallback_symbols(self) -> tuple[str, ...]:
        symbols = tuple(sorted(self._access_blocked_cache_fallback_symbols))
        self._access_blocked_cache_fallback_symbols.clear()
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

        try:
            self.refresh_control.check_host()
            corrections = discover(periods)
        except DataSourceError as error:
            self.refresh_control.failed(
                "__REVISIONS__", error,
                transport_counted=getattr(self.filing_catalog, "refresh_control", None) is self.refresh_control,
            )
            raise
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
        preload_started = time.perf_counter()
        cached = self.repository.load_cached_bundles(
            symbol_sectors,
            as_of=effective_signal.isoformat(),
            derivation_version=FUNDAMENTAL_DERIVATION_VERSION,
        )
        eligible: dict[
            str,
            tuple[CachedFundamentalBundle, FundamentalSnapshot | None],
        ] = {}
        for symbol, entry in cached.items():
            if ("les_v6_mops_filings" not in entry.bundle.snapshot
                    or entry.bundle.snapshot.get("les_v6_filing_schema_version") != V6_FILING_SCHEMA_VERSION):
                continue
            periods = set()
            for filing in entry.bundle.snapshot["les_v6_mops_filings"]:
                period_end = pd.to_datetime(filing.get("period_end"), errors="coerce")
                available_at = pd.to_datetime(filing.get("available_at"), utc=True, errors="coerce")
                if pd.notna(period_end) and pd.notna(available_at) and available_at <= effective_signal:
                    periods.add((period_end.year, period_end.quarter))
            if not periods or not self.planner.history_periods(max(periods), history_years).issubset(periods):
                continue
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
            eligible[symbol] = (entry, current_batch)

        reusable: dict[str, FundamentalBundle] = {}
        eligible_symbols = list(eligible)
        if eligible_symbols:
            with self.repository.unit_of_work() as uow:
                histories = (
                    self.repository.load_financial_batch_snapshots_many(
                        eligible_symbols,
                        as_of=effective_signal.isoformat(),
                        uow=uow,
                    )
                )
                batch_states = self.repository.load_financial_batch_states(
                    eligible_symbols,
                    uow=uow,
                )
                for symbol, (entry, current_batch) in eligible.items():
                    history = histories.get(symbol, [])
                    resolution = self._resolve_batch_snapshot(
                        symbol,
                        effective_signal,
                        uow=uow,
                        history=history,
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
                        history=history,
                        persist_snapshot=False,
                    )
                    batch_state = batch_states.get(symbol)
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
            "fallback=%s derivation_version=%s duration_seconds=%.3f",
            len(symbol_sectors),
            len(cached),
            len(reusable),
            len(symbol_sectors) - len(reusable),
            FUNDAMENTAL_DERIVATION_VERSION,
            time.perf_counter() - preload_started,
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
        fallback_available_at = self.point_in_time.utc_now().isoformat()
        repository = self.repository
        with repository.unit_of_work() as uow:
            cached_filings = repository.load_filings(
                symbol,
                uow=uow,
            )
            cached_catalog = repository.load_filing_catalog(
                symbol, as_of=effective_signal.isoformat(), uow=uow,
            )
            monthly_refresh = (
                not cached_filings
                or self.planner.is_monthly_refresh_due(
                    int(pd.Timestamp(cached_catalog.checked_at).timestamp()) if cached_catalog else repository.last_full_refresh_epoch(
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
            if cached_catalog is not None:
                acknowledged = {
                    (event.fiscal_year, event.quarter): event.announced_on
                    for event in cached_catalog.verified_corrections
                }
                event_corrected_periods = {
                    period for period in event_corrected_periods
                    if self._daily_corrections[symbol][period].announced_on
                    > acknowledged.get(period, "")
                }
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
            if cached_catalog is not None:
                eligible_catalog = self.refresh_planner.select_refresh_candidates(
                    list(cached_catalog.candidates), effective_signal,
                )
                local_plan = self.planner.plan(eligible_catalog, cached_filings, history_years=history_years)
                required_periods = {(item.fiscal_year, item.quarter) for item in local_plan.reuse}
                # Young issuers / catalog gaps do not justify downloading the
                # same catalog daily. Financial coverage remains unchanged.
                pending_corrections = self.catalog_refresh.unverified_periods(
                    cached_catalog, cached_filings,
                )
                cache_history_complete = (
                    bool(local_plan.reuse)
                    and not cached_catalog.announcements_pending
                    and not local_plan.download
                    and not pending_corrections
                    and required_periods.issubset(cached_catalog.scanned_periods)
                )
            # Format migration is local and must not depend on remote availability.
            local_bundle = None
            if cached_filings:
                try:
                    local_bundle = self._load_cached_bundle(
                        symbol, parsed_signal=effective_signal, sector=sector, uow=uow,
                    )
                except NoDataError:
                    pass
            if (
                not monthly_refresh
                and not batch_has_new_period
                and not event_corrected_periods
                and cache_history_complete
                and local_bundle is not None
            ):
                bundle, cache_hit, filing_count = local_bundle
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
                self.refresh_control.start_symbol(symbol, uow=uow)
                catalog_state, plan = self.catalog_refresh.prepare(
                    symbol, state=cached_catalog, cached=cached_filings,
                    signal=effective_signal, history_years=history_years,
                    batch_period=batch_snapshot.period if batch_snapshot else None,
                    events=self._daily_corrections.get(symbol, {}), uow=uow,
                )
                if any(
                    candidate.statement_scope == "individual"
                    for candidate in catalog_state.candidates
                ):
                    self._individual_scope_fallback_symbols.add(symbol)
                first_observation_seed = sum(
                    candidate.available_at is None for candidate in plan.download
                )
                ingested_filings: list[FilingRecord] = []
                pending = plan.verify_revision + list(reversed(plan.download))
                batch = pending[:self.refresh_control.policy.max_filings_per_symbol]
                for candidate in batch:
                    ingested_filings.append(
                        self.filing_ingestion.ingest(
                            candidate,
                            fallback_available_at=fallback_available_at,
                            uow=uow,
                        )
                    )
                    catalog_state = self.catalog_refresh.acknowledge(catalog_state, candidate, uow=uow)
                effective_signal = self.point_in_time.advance_live_evaluation_time(
                    effective_signal,
                    ingested_filings,
                )
                if getattr(self.filing_catalog, "refresh_control", None) is not self.refresh_control:
                    self.refresh_control.response(blocked=False)
            except DataSourceError as refresh_error:
                self.refresh_control.failed(
                    symbol, refresh_error, uow=uow,
                    transport_counted=getattr(self.filing_catalog, "refresh_control", None) is self.refresh_control,
                )
                try:
                    bundle = self._load_cached_fallback(
                        symbol, parsed_signal=self.point_in_time.effective_signal_time(parsed_signal),
                        sector=sector, uow=uow, refresh_error=refresh_error,
                    )
                except DataSourceError:
                    # Keep successful partial downloads and negative-attempt cooldown.
                    uow.commit()
                    raise refresh_error
                return bundle

            filed_financials = repository.load_filed_financials(
                symbol,
                effective_signal.isoformat(),
                uow=uow,
            )
            if not filed_financials:
                self.refresh_control.finish(symbol, "NO_ELIGIBLE_FILINGS", uow=uow)
                uow.commit()
                raise NoDataError(
                    "MOPS refresh for {} produced no point-in-time eligible filings; "
                    "empty refresh state and snapshots were not stored".format(symbol),
                    retryable=False,
                )
            self.refresh_control.finish(
                symbol, "PARTIAL" if len(batch) < len(pending) else "SUCCESS", uow=uow,
            )
            if len(batch) == len(pending):
                repository.mark_full_refresh(symbol, catalog_state.checked_at, uow=uow)
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
            snapshot = dict(bundle.snapshot)
            snapshot.update(
                remote_refresh_skipped=False,
                remote_refresh_status="PARTIAL" if len(batch) < len(pending) else "SUCCESS",
            )
            actual_periods = {(item.fiscal_year, item.quarter) for item in filed_financials}
            expected_periods = self.planner.history_periods(
                max(actual_periods | {
                    (item.fiscal_year, item.quarter)
                    for item in self.refresh_planner.select_refresh_candidates(
                        list(catalog_state.candidates), effective_signal,
                    )
                }), history_years,
            )
            cached_count = len(actual_periods & expected_periods)
            snapshot["history_cache_progress"] = {
                "years": history_years,
                "cached_periods": cached_count,
                "expected_periods": len(expected_periods),
                "status": "COMPLETE" if cached_count == len(expected_periods) else "PARTIAL",
            }
            logging.info(
                "%s symbol=%s quarters=%s/%s pending=%s",
                "建立五年歷史資料" if history_years == 5 else "建立{}年歷史資料".format(history_years),
                symbol, cached_count, len(expected_periods), len(pending) - len(batch),
                extra={"source": "MOPS_CACHE", "symbol": symbol, "status": "HISTORY_CACHE_PROGRESS"},
            )
            bundle = FundamentalBundle(bundle.annual_fundamentals, snapshot, bundle.as_of)
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

    def load_historical(self, symbol: str, *, as_of: dt.datetime, sector: str,
                        market: str = "TWSE") -> FundamentalBundle:
        """Read-only derivation from PIT cache; never promote the cutoff to now.

        Do not activate an old bundle, fetch a current company master, refresh
        catalogs, or reuse an active snapshot assembled after the cutoff.
        """
        if as_of.tzinfo is None:
            raise ValueError("Historical cutoff must be timezone-aware")
        cutoff = pd.Timestamp(as_of).tz_convert("UTC")
        with self.repository.unit_of_work() as uow:
            filings = self.repository.load_filed_financials(symbol, cutoff.isoformat(), uow=uow)
            filings = [item for item in filings if pd.to_datetime(item.available_at, utc=True, errors="coerce") <= cutoff]
            history = self.repository.load_financial_batch_snapshots(symbol, as_of=cutoff.isoformat(), uow=uow) if market == "TWSE" else []
            history = [item for item in history if pd.to_datetime(item.observed_at, utc=True, errors="coerce") <= cutoff]
            # Use the explicitly selected universe, not today's company master.
            resolved_sector = sector
            bundle = derive_fundamentals(filings, signal_date=cutoff, sector=resolved_sector)
            snapshot = dict(bundle.snapshot)
            snapshot["les_v6_mops_filings"] = [dict(ixbrl_record(symbol, item), market=market) for item in filings]
            snapshot["les_v6_filing_schema_version"] = V6_FILING_SCHEMA_VERSION
            snapshot["sector"] = resolved_sector
            snapshot["market"] = market
            bundle = FundamentalBundle(bundle.annual_fundamentals, snapshot, cutoff.isoformat())
            resolution = self.snapshot_assembler.resolve({}, symbol, cutoff, uow=uow, history=history)
            return self.snapshot_assembler.attach_base_evidence(
                bundle, resolution, symbol=symbol, signal_time=cutoff,
                uow=uow, history=history, persist_snapshot=False,
            )

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
        history: list[FundamentalSnapshot] | None = None,
    ) -> BatchSnapshotResolution:
        return self.snapshot_assembler.resolve(
            self._daily_financials,
            symbol,
            signal_time,
            uow=uow,
            history=history,
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
        snapshot = dict(bundle.snapshot)
        snapshot.update(
            remote_refresh_skipped=True,
            remote_refresh_status=getattr(refresh_error, "reason", refresh_error.category.value.upper()),
        )
        bundle = FundamentalBundle(bundle.annual_fundamentals, snapshot, bundle.as_of)
        if isinstance(refresh_error, RefreshDeferred):
            logging.debug("MOPS refresh deferred symbol=%s reason=%s; using PIT cache", symbol, refresh_error.reason)
        elif refresh_error.category is DataErrorCategory.ACCESS_BLOCKED:
            self._access_blocked_cache_fallback_symbols.add(symbol)
            logging.debug(
                "MOPS refresh status=ACCESS_BLOCKED symbol=%s "
                "cache_fallback=true filings=%s bundle_cache_hit=%s detail=%s",
                symbol,
                len(filed_financials),
                cache_hit,
                refresh_error,
                extra={
                    "symbol": symbol,
                    "source": "MOPS_CACHE",
                    "status": "ACCESS_BLOCKED",
                    "cache_hit": True,
                },
            )
        else:
            logging.warning(
                "MOPS refresh status=%s symbol=%s cache_fallback=true "
                "filings=%s bundle_cache_hit=%s detail=%s",
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
