"""Cache-first fundamental refresh coordination."""

from __future__ import annotations

import datetime
import logging
import sqlite3
import time
from typing import Any
from zoneinfo import ZoneInfo

from application.contracts import AnalysisContext
from application.contracts import StockAnalysisRequest
from application.observability import NullRunObserver
from application.mops_refresh_control import RefreshDeferred
from application.ports import RunObserverPort
from domain.fundamental_derivation import FundamentalBundle
from domain.security_identity import SecurityIdentity
from domain.security_identity import resolve_security_identity
from domain.strategies.low_entry_v6_watch import screen_leadership
from infrastructure.crawler.twse_http_client import TwseClientError
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.fundamentals.mops_repository import MopsStorageError
from infrastructure.observability import structured_extra


TAIPEI_TIMEZONE = ZoneInfo("Asia/Taipei")
MOPS_INFRASTRUCTURE_FAILURE_THRESHOLD = 2


class FundamentalRefreshCoordinator:
    """Coordinate cache-first MOPS refresh while preserving fail-closed behavior."""

    def __init__(
        self,
        fundamental_service: Any,
        *,
        observer: RunObserverPort | None = None,
        restriction_repository: Any | None = None,
        now_provider: Any | None = None,
    ) -> None:
        self.fundamental_service = fundamental_service
        self._observer = observer or NullRunObserver()
        self._restriction_repository = restriction_repository
        self._now_provider = now_provider

    def refresh(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
        *,
        analysis_time: datetime.datetime,
    ) -> None:
        refresh_started = time.monotonic()
        crawler = context.resource
        if request.as_of is not None:
            self._replay_cached_fundamentals(context, request)
            return
        select_market = getattr(getattr(self.fundamental_service, "filing_catalog", None), "select_market", None)
        if callable(select_market):
            select_market(request.market_source)
        sector = "24" if request.market_source == "TPEX" else str(request.stocktype)
        stocknumbers = getattr(crawler, "stocknumbers", [])
        stocknames = getattr(crawler, "stocknames", [])
        daily_dates = getattr(crawler, "daily_dates", [])
        setter = getattr(crawler, "set_low_entry_v4_fundamentals", None)
        if not callable(setter):
            self._observer.record(
                "fundamental_refresh_skipped",
                source="MOPS",
                status="UNSUPPORTED",
            )
            return

        localized_time = analysis_time.astimezone(TAIPEI_TIMEZONE)
        # Current lists are dated by observation, not by the last completed bar.
        # Never fetch a present-day list on behalf of a historical requested date.
        batch_client = getattr(self.fundamental_service, "financial_batch_client", None)
        http_client = getattr(batch_client, "http_client", None)
        crawler.trading_restrictions = {}
        if http_client is not None and request.market_source in {"TWSE", "TPEX"}:
            from infrastructure.crawler.trading_restrictions import TradingRestrictionClient, TpexTradingRestrictionClient
            restriction_client = TpexTradingRestrictionClient if request.market_source == "TPEX" else TradingRestrictionClient
            try:
                crawler.trading_restrictions = restriction_client(http_client).load(
                    stocknumbers, session_date=context.market_date.isoformat(), observed_at=localized_time,
                )
                # The complete response set is only known after the final HTTP
                # response, never at request start. Do not backdate the archive.
                completed_at = self._now_provider() if self._now_provider else localized_time
                if completed_at.tzinfo is None:
                    raise ValueError("Restriction observation requires a timezone-aware clock")
                completed_at = completed_at.astimezone(TAIPEI_TIMEZONE)
                if completed_at < localized_time or completed_at.date() != localized_time.date():
                    crawler.trading_restrictions = {}
                else:
                    for item in crawler.trading_restrictions.values():
                        item["available_at"] = completed_at.isoformat()
                    localized_time = completed_at
                if self._restriction_repository is not None:
                    scheduled = getattr(crawler, "iso_scheduled_times", [])
                    bar_date = scheduled[-1] if scheduled else None
                    if bar_date:
                        try:
                            self._restriction_repository.record(
                                crawler.trading_restrictions, market=request.market_source,
                                market_data_date=bar_date, decision_time=localized_time,
                            )
                        except (OSError, sqlite3.Error, ValueError) as error:
                            logging.warning("Restriction archive write failed; current evidence retained: %s", error)
            except TwseClientError as error:
                logging.warning("LES v6 restriction evidence unavailable; entries fail closed: %s", error)
        analysis_signal_date = localized_time.isoformat()
        # Research observations use completed bars and official restrictions;
        # no financial or entry status is rewritten by this pre-refresh pass.
        crawler.les_v6_research_context = {}
        build_history = getattr(crawler, "build_strategy_history", None)
        if callable(build_history):
            watch_histories = []
            for item, stock_no in enumerate(stocknumbers):
                if stock_no and item < len(daily_dates) and daily_dates[item]:
                    try:
                        watch_histories.append(build_history(item))
                    except (ValueError, IndexError) as error:
                        logging.debug("Leadership history unavailable symbol=%s: %s", stock_no, error)
            # Weekends/premarket have no completed bar for context.market_date.
            # Use the batch's scheduled session, retaining stale-symbol checks.
            watch_session = max((h.attrs.get("les_v6_analysis_session") or str(h["Date"].iloc[-1])
                                 for h in watch_histories if len(h)), default="")
            watches = screen_leadership(watch_histories, session_date=watch_session, as_of=analysis_signal_date)
            crawler.les_v6_research_context = {watch["symbol"]: {"watch": watch} for watch in watches}
        begin_run = getattr(self.fundamental_service, "begin_run", None)
        if callable(begin_run):
            begin_run()
        issuer_symbols = []
        for item, stock_no in enumerate(stocknumbers):
            if not stock_no or item >= len(daily_dates) or not daily_dates[item]:
                continue
            stock_name = stocknames[item] if item < len(stocknames) else ""
            issuer_symbols.append(resolve_security_identity(stock_no, stock_name).issuer_symbol)
        migrate = getattr(self.fundamental_service, "migrate_local_cache", None)
        if callable(migrate):
            try:
                migrated = migrate(
                    issuer_symbols, signal_date=analysis_signal_date, sector=sector,
                )
                self._observer.record(
                    "fundamental_local_migration_completed", source="MOPS_CACHE", status="SUCCESS",
                    migrated_symbols=migrated,
                )
            except (MopsStorageError, TwseClientError) as error:
                logging.warning("Local MOPS migration unavailable; retaining per-symbol fail-closed checks: %s", error)
        consecutive_network_failures = 0
        consecutive_storage_failures = 0
        consecutive_schema_failures = 0
        bundles_by_issuer: dict[str, FundamentalBundle] = {}
        reset_scope_fallbacks = getattr(
            self.fundamental_service,
            "reset_scope_fallback_diagnostics",
            None,
        )
        if callable(reset_scope_fallbacks):
            reset_scope_fallbacks()
        reset_access_blocked = getattr(
            self.fundamental_service,
            "reset_access_blocked_diagnostics",
            None,
        )
        if callable(reset_access_blocked):
            reset_access_blocked()

        prepare_daily_batch = getattr(
            self.fundamental_service,
            "prepare_daily_batch",
            None,
        )
        if callable(prepare_daily_batch) and request.market_source == "TWSE":
            try:
                company_count = prepare_daily_batch(localized_time)
                self._observer.record(
                    "financial_batch_loaded",
                    trade_date=context.market_date.isoformat(),
                    source="TWSE_OPENAPI",
                    status="SUCCESS",
                    company_count=company_count,
                )
            except TwseClientError as error:
                self._observer.record(
                    "financial_batch_unavailable",
                    level=logging.WARNING,
                    trade_date=context.market_date.isoformat(),
                    source="TWSE_OPENAPI",
                    status=error.category.value.upper(),
                )
                logging.warning(
                    "TWSE financial batch status=%s unavailable; "
                    "falling back to cache-first MOPS refresh: %s",
                    error.category.value.upper(),
                    error,
                )

        prepare_daily_revisions = getattr(
            self.fundamental_service,
            "prepare_daily_revisions",
            None,
        )
        if callable(prepare_daily_revisions):
            try:
                correction_count = prepare_daily_revisions(localized_time)
                self._observer.record(
                    "fundamental_revision_scan_completed",
                    trade_date=context.market_date.isoformat(),
                    source="MOPS",
                    status="SUCCESS",
                    correction_count=correction_count,
                )
            except TwseClientError as error:
                self._observer.record(
                    "fundamental_revision_scan_unavailable",
                    level=logging.WARNING,
                    trade_date=context.market_date.isoformat(),
                    source="MOPS",
                    status=error.category.value.upper(),
                )
                logging.warning(
                    "MOPS daily correction scan status=%s unavailable; "
                    "complete filing caches remain usable: %s",
                    error.category.value.upper(),
                    error,
                )

        preloaded_bundles: dict[str, FundamentalBundle] = {}
        preload_cached_bundles = getattr(
            self.fundamental_service,
            "load_cached_bundles",
            None,
        )
        if callable(preload_cached_bundles):
            preload_started = time.monotonic()
            try:
                preloaded_bundles = preload_cached_bundles(
                    list(dict.fromkeys(issuer_symbols)),
                    signal_date=analysis_signal_date,
                    sector=sector,
                    history_years=request.mops_history_years,
                )
                self._observer.record(
                    "fundamental_cache_preload_completed",
                    trade_date=context.market_date.isoformat(),
                    source="MOPS_CACHE",
                    status="SUCCESS",
                    cache_hit=bool(preloaded_bundles),
                    requested_symbol_count=len(set(issuer_symbols)),
                    reused_symbol_count=len(preloaded_bundles),
                    duration_seconds=round(
                        time.monotonic() - preload_started,
                        3,
                    ),
                )
            except (MopsStorageError, TwseClientError) as error:
                self._observer.record(
                    "fundamental_cache_preload_failed",
                    level=logging.DEBUG,
                    trade_date=context.market_date.isoformat(),
                    source="MOPS_CACHE",
                    status="BULK_PRELOAD_FAILED",
                    cache_hit=False,
                    requested_symbol_count=len(set(issuer_symbols)),
                    duration_seconds=round(
                        time.monotonic() - preload_started,
                        3,
                    ),
                )
                logging.debug(
                    "MOPS snapshot bulk preload unavailable; falling back to "
                    "per-symbol refresh: %s",
                    error,
                    extra={
                        "source": "MOPS_CACHE",
                        "status": "BULK_PRELOAD_FAILED",
                        "cache_hit": False,
                    },
                )

        control = getattr(self.fundamental_service, "refresh_control", None)
        if control is not None:
            try:
                research = crawler.les_v6_research_context
                candidates = [symbol for symbol, row in research.items() if row["watch"]["priority_eligible"]]
                inspect = getattr(self.fundamental_service, "inspect_local_evidence", None)
                diagnostics = inspect(candidates, signal_date=analysis_signal_date, market=request.market_source) if callable(inspect) else {}
                for symbol, diagnostic in diagnostics.items():
                    research[symbol]["before_refresh"] = diagnostic
                priority_symbols = [symbol for symbol in candidates
                                    if diagnostics.get(symbol, {}).get("backfill_needed")
                                    and not diagnostics[symbol].get("local_review_required")
                                    and diagnostics[symbol].get("quality_status") != "REJECT"]
                control.schedule([symbol for symbol in issuer_symbols if symbol not in preloaded_bundles],
                                 priority_symbols=priority_symbols)
                self._observer.record(
                    "leadership_verification_queue", source="STRATEGY", status="RESEARCH_ONLY",
                    watch_count=len(candidates), priority_backfill_count=len(priority_symbols),
                    priority_symbols=priority_symbols, scheduled_symbols=control.scheduled_order,
                )
            except MopsStorageError as error:
                logging.warning("MOPS batch scheduling unavailable; retaining per-symbol cache checks: %s", error)

        processing_order = [
            index for index, symbol in enumerate(stocknumbers)
            if symbol and index < len(daily_dates) and daily_dates[index]
        ]
        if control is not None:
            priority = {symbol: index for index, symbol in enumerate(control.scheduled_order)}
            processing_order.sort(key=lambda index: priority.get(
                resolve_security_identity(
                    stocknumbers[index], stocknames[index] if index < len(stocknames) else "",
                ).issuer_symbol, len(priority),
            ))
        for processing_offset, item in enumerate(processing_order):
            stock_no = stocknumbers[item]
            if not stock_no or item >= len(daily_dates) or not daily_dates[item]:
                continue
            stock_name = stocknames[item] if item < len(stocknames) else ""
            identity = resolve_security_identity(stock_no, stock_name)
            issuer_symbol = identity.issuer_symbol
            if issuer_symbol != stock_no:
                logging.debug(
                    "Fundamental identity market_symbol=%s issuer_symbol=%s "
                    "security_type=%s",
                    stock_no,
                    issuer_symbol,
                    identity.security_type.value,
                    extra={
                        "symbol": stock_no,
                        "source": "TWSE_SECURITY_MASTER",
                        "status": "IDENTITY_MAPPED",
                    },
                )
            try:
                bundle = bundles_by_issuer.get(issuer_symbol)
                if bundle is None:
                    bundle = preloaded_bundles.get(issuer_symbol)
                    if bundle is None:
                        bundle = self.fundamental_service.load(
                            issuer_symbol,
                            signal_date=analysis_signal_date,
                            sector=sector,
                            history_years=request.mops_history_years,
                        )
                    bundles_by_issuer[issuer_symbol] = bundle
                bundle = self._attach_security_identity(bundle, identity)
                if request.market_source == "TPEX":
                    # MOPS issuer facts are shared, but TWSE batch observations
                    # and restriction lists cannot certify an OTC instrument.
                    snapshot = dict(bundle.snapshot)
                    snapshot["market"] = "TPEX"
                    snapshot["sector"] = "24"
                    snapshot["les_v6_filings"] = [
                        dict(row, market="TPEX")
                        for row in snapshot.get("les_v6_mops_filings", [])
                        if row.get("source") == "MOPS_IXBRL" and row.get("symbol") == issuer_symbol
                    ]
                    bundle = FundamentalBundle(bundle.annual_fundamentals, snapshot, bundle.as_of)
            except RefreshDeferred as error:
                self._mark_unavailable(
                    crawler, stock_no, status=error.reason, detail=str(error),
                    signal_date=analysis_signal_date, account_equity=request.account_equity,
                )
                continue
            except TwseNetworkError as error:
                self._observer.record(
                    "fundamental_refresh_failed",
                    level=logging.ERROR,
                    symbol=stock_no,
                    trade_date=context.market_date.isoformat(),
                    source="MOPS",
                    status=error.category.value.upper(),
                )
                self._mark_unavailable(
                    crawler,
                    stock_no,
                    status=error.category.value.upper(),
                    detail=str(error),
                    signal_date=analysis_signal_date,
                    account_equity=request.account_equity,
                )
                logging.error(
                    "MOPS status=%s market_symbol=%s issuer_symbol=%s detail=%s; "
                    "LES v5 formal BUY disabled.",
                    error.category.value.upper(),
                    stock_no,
                    issuer_symbol,
                    error,
                    extra={
                        "symbol": stock_no,
                        "source": "MOPS",
                        "status": error.category.value.upper(),
                    },
                )
                consecutive_storage_failures = 0
                consecutive_schema_failures = 0
                consecutive_network_failures += 1
                if (
                    consecutive_network_failures
                    >= MOPS_INFRASTRUCTURE_FAILURE_THRESHOLD
                ):
                    logging.error(
                        "MOPS circuit open after %s consecutive network failures; "
                        "skipping remaining symbols; LES v5 formal BUY disabled.",
                        consecutive_network_failures,
                    )
                    self._observer.record(
                        "fundamental_circuit_opened",
                        level=logging.ERROR,
                        source="MOPS",
                        status="NETWORK_ERROR",
                    )
                    self._mark_remaining_unavailable(
                        crawler,
                        [stocknumbers[index] for index in processing_order[processing_offset + 1 :]],
                        status="CIRCUIT_OPEN",
                        detail="MOPS network failure circuit is open",
                        signal_date=analysis_signal_date,
                        account_equity=request.account_equity,
                    )
                    break
                continue
            except TwseClientError as error:
                self._observer.record(
                    "fundamental_refresh_failed",
                    level=logging.ERROR,
                    symbol=stock_no,
                    trade_date=context.market_date.isoformat(),
                    source="MOPS",
                    status=error.category.value.upper(),
                )
                self._mark_unavailable(
                    crawler,
                    stock_no,
                    status=error.category.value.upper(),
                    detail=str(error),
                    signal_date=analysis_signal_date,
                    account_equity=request.account_equity,
                )
                logging.error(
                    "MOPS status=%s market_symbol=%s issuer_symbol=%s detail=%s; "
                    "LES v5 formal BUY disabled.",
                    error.category.value.upper(),
                    stock_no,
                    issuer_symbol,
                    error,
                    extra={
                        "symbol": stock_no,
                        "source": "MOPS",
                        "status": error.category.value.upper(),
                    },
                )
                consecutive_network_failures = 0
                consecutive_storage_failures = 0
                consecutive_schema_failures = (
                    consecutive_schema_failures + 1
                    if isinstance(error, TwseSchemaChangedError)
                    else 0
                )
                if (
                    consecutive_schema_failures
                    >= MOPS_INFRASTRUCTURE_FAILURE_THRESHOLD
                ):
                    logging.error(
                        "MOPS schema circuit open after %s consecutive failures; "
                        "skipping remaining symbols; LES v5 formal BUY disabled.",
                        consecutive_schema_failures,
                    )
                    self._observer.record(
                        "fundamental_circuit_opened",
                        level=logging.ERROR,
                        source="MOPS",
                        status="SCHEMA_CHANGED",
                    )
                    self._mark_remaining_unavailable(
                        crawler,
                        [stocknumbers[index] for index in processing_order[processing_offset + 1 :]],
                        status="CIRCUIT_OPEN",
                        detail="MOPS schema failure circuit is open",
                        signal_date=analysis_signal_date,
                        account_equity=request.account_equity,
                    )
                    break
                continue
            except MopsStorageError as error:
                self._observer.record(
                    "fundamental_refresh_failed",
                    level=logging.ERROR,
                    symbol=stock_no,
                    trade_date=context.market_date.isoformat(),
                    source="MOPS_CACHE",
                    status="STORAGE_ERROR",
                )
                self._mark_unavailable(
                    crawler,
                    stock_no,
                    status="STORAGE_ERROR",
                    detail=str(error),
                    signal_date=analysis_signal_date,
                    account_equity=request.account_equity,
                )
                logging.error(
                    "MOPS storage market_symbol=%s issuer_symbol=%s unavailable; "
                    "LES v5 formal BUY disabled: %s",
                    stock_no,
                    issuer_symbol,
                    error,
                    extra={
                        "symbol": stock_no,
                        "source": "MOPS_CACHE",
                        "status": "STORAGE_ERROR",
                    },
                )
                consecutive_network_failures = 0
                consecutive_schema_failures = 0
                consecutive_storage_failures += 1
                if (
                    consecutive_storage_failures
                    >= MOPS_INFRASTRUCTURE_FAILURE_THRESHOLD
                ):
                    logging.error(
                        "MOPS storage circuit open after %s consecutive failures; "
                        "skipping remaining symbols; LES v5 formal BUY disabled.",
                        consecutive_storage_failures,
                    )
                    self._observer.record(
                        "fundamental_circuit_opened",
                        level=logging.ERROR,
                        source="MOPS_CACHE",
                        status="STORAGE_ERROR",
                    )
                    self._mark_remaining_unavailable(
                        crawler,
                        [stocknumbers[index] for index in processing_order[processing_offset + 1 :]],
                        status="CIRCUIT_OPEN",
                        detail="MOPS storage failure circuit is open",
                        signal_date=analysis_signal_date,
                        account_equity=request.account_equity,
                    )
                    break
                continue

            consecutive_network_failures = 0
            consecutive_storage_failures = 0
            consecutive_schema_failures = 0
            cache_value = bundle.snapshot.get("remote_refresh_skipped")
            cache_hit = (
                cache_value if isinstance(cache_value, bool) else None
            )
            self._observer.record(
                "fundamental_refresh_completed",
                level=logging.DEBUG,
                symbol=stock_no,
                trade_date=context.market_date.isoformat(),
                source="MOPS_CACHE" if cache_hit else "MOPS",
                status="SUCCESS",
                cache_hit=cache_hit,
                available_at=bundle.as_of,
                issuer_symbol=issuer_symbol,
                security_type=identity.security_type.value,
            )
            setter(
                stock_no,
                annual_fundamentals=bundle.annual_fundamentals,
                fundamental_snapshot=bundle.snapshot,
                account_equity=request.account_equity,
                signal_date=bundle.as_of or analysis_signal_date,
            )

        control = getattr(self.fundamental_service, "refresh_control", None)
        if control is not None:
            self._observer.record(
                "fundamental_refresh_control_summary", source="MOPS", status="BOUNDED_REFRESH",
                **control.summary(),
            )

        consume_scope_fallbacks = getattr(
            self.fundamental_service,
            "consume_scope_fallback_symbols",
            None,
        )
        if callable(consume_scope_fallbacks):
            fallback_symbols = tuple(consume_scope_fallbacks())
            if fallback_symbols:
                logging.warning(
                    "MOPS individual scope fallback summary stocks=%s "
                    "symbols=%s; consolidated filing was unavailable for "
                    "one or more selected periods. Use debug mode for "
                    "per-symbol details.",
                    len(fallback_symbols),
                    ",".join(fallback_symbols),
                    extra=structured_extra(
                        source="MOPS",
                        status="INDIVIDUAL_SCOPE_FALLBACK_SUMMARY",
                    ),
                )
                self._observer.record(
                    "fundamental_individual_scope_fallback_summary",
                    level=logging.DEBUG,
                    source="MOPS",
                    status="INDIVIDUAL_SCOPE_FALLBACK_SUMMARY",
                    fallback_symbols=list(fallback_symbols),
                    fallback_symbol_count=len(fallback_symbols),
                )

        consume_access_blocked = getattr(
            self.fundamental_service,
            "consume_access_blocked_cache_fallback_symbols",
            None,
        )
        if callable(consume_access_blocked):
            blocked_symbols = tuple(consume_access_blocked())
            if blocked_symbols:
                logging.warning(
                    "MOPS access blocked cache fallback summary stocks=%s "
                    "symbols=%s; cached filings were reused. Use debug mode "
                    "for per-symbol details.",
                    len(blocked_symbols),
                    ",".join(blocked_symbols),
                    extra=structured_extra(
                        source="MOPS_CACHE",
                        status="ACCESS_BLOCKED_CACHE_FALLBACK_SUMMARY",
                        cache_hit=True,
                    ),
                )
                self._observer.record(
                    "fundamental_access_blocked_cache_fallback_summary",
                    level=logging.DEBUG,
                    source="MOPS_CACHE",
                    status="ACCESS_BLOCKED_CACHE_FALLBACK_SUMMARY",
                    cache_hit=True,
                    fallback_symbols=list(blocked_symbols),
                    fallback_symbol_count=len(blocked_symbols),
                )

        self._observer.record(
            "fundamental_runtime_completed",
            trade_date=context.market_date.isoformat(),
            source="MOPS",
            status="SUCCESS",
            issuer_symbol_count=len(set(issuer_symbols)),
            preloaded_symbol_count=len(preloaded_bundles),
            duration_seconds=round(time.monotonic() - refresh_started, 3),
        )

    def _replay_cached_fundamentals(self, context: AnalysisContext, request: StockAnalysisRequest) -> None:
        crawler = context.resource
        cutoff = request.as_of
        assert cutoff is not None
        sector = "24" if request.market_source == "TPEX" else request.stocktype.name
        bundles: dict[str, FundamentalBundle] = {}
        for index, symbol in enumerate(crawler.stocknumbers):
            if not symbol:
                continue
            name = crawler.stocknames[index] if index < len(crawler.stocknames) else ""
            identity = resolve_security_identity(symbol, name)
            try:
                if identity.issuer_symbol not in bundles:
                    bundles[identity.issuer_symbol] = self.fundamental_service.load_historical(
                        identity.issuer_symbol, as_of=cutoff, sector=sector, market=request.market_source,
                    )
                bundle = self._attach_security_identity(bundles[identity.issuer_symbol], identity)
                crawler.set_low_entry_v4_fundamentals(
                    symbol, bundle.annual_fundamentals, bundle.snapshot,
                    account_equity=request.account_equity, signal_date=cutoff.isoformat(),
                )
            except MopsStorageError as error:
                self._mark_unavailable(crawler, symbol, status="REPLAY_CACHE_UNAVAILABLE", detail=str(error),
                                       signal_date=cutoff.isoformat(), account_equity=request.account_equity)
        self._observer.record("fundamental_historical_replay_completed", source="MOPS_CACHE",
                              status="SUCCESS", cutoff=cutoff.isoformat(), issuer_symbol_count=len(bundles))

    @staticmethod
    def _attach_security_identity(
        bundle: FundamentalBundle,
        identity: SecurityIdentity,
    ) -> FundamentalBundle:
        snapshot = dict(bundle.snapshot)
        snapshot.update(identity.snapshot_fields())
        return FundamentalBundle(
            annual_fundamentals=bundle.annual_fundamentals,
            snapshot=snapshot,
            as_of=bundle.as_of,
        )

    @staticmethod
    def _mark_unavailable(
        crawler: Any,
        stock_no: str,
        *,
        status: str,
        detail: str,
        signal_date: str,
        account_equity: float,
    ) -> None:
        marker = getattr(
            crawler,
            "set_low_entry_v4_fundamentals_unavailable",
            None,
        )
        if callable(marker):
            marker(
                stock_no,
                status=status,
                detail=detail,
                signal_date=signal_date,
                account_equity=account_equity,
            )

    @classmethod
    def _mark_remaining_unavailable(
        cls,
        crawler: Any,
        stocknumbers: list[str],
        *,
        status: str,
        detail: str,
        signal_date: str,
        account_equity: float,
    ) -> None:
        for stock_no in stocknumbers:
            if stock_no:
                cls._mark_unavailable(
                    crawler,
                    stock_no,
                    status=status,
                    detail=detail,
                    signal_date=signal_date,
                    account_equity=account_equity,
                )
