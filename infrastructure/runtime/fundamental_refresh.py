"""Cache-first fundamental refresh coordination."""

from __future__ import annotations

import datetime
import logging
import sqlite3
import time
import weakref
from collections.abc import Mapping
from typing import Any
from zoneinfo import ZoneInfo

from application.contracts import AnalysisContext
from application.contracts import StockAnalysisRequest
from application.daily_performance import measure_stage
from application.daily_work import current_work
from application.observability import NullRunObserver
from application.mops_refresh_control import RefreshDeferred
from application.leadership_screen import screen_once, watch_result_key
from application.ports import RunObserverPort
from domain.fundamental_derivation import FundamentalBundle
from domain.security_identity import SecurityIdentity
from domain.security_identity import resolve_security_identity
from domain.trading_restrictions import restriction_check
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

    @staticmethod
    def _restriction_bar(crawler):
        scheduled = getattr(crawler, "iso_scheduled_times", ())
        return scheduled[-1] if scheduled else max(
            (str(dates[-1])[:10] for dates in getattr(crawler, "daily_dates", ()) if dates), default=None)

    def _collect_restrictions(self, context, *, market, started_at, completion_clock=None):
        """Collect one complete market response set; never retain an old CLEAR.

        The normal refresh and the native daily pre-freeze repair share this
        collector. Production supplies the real clock through bootstrap; legacy
        callers without a provider retain their existing injected start clock.
        """
        crawler = context.resource
        crawler.trading_restrictions = {}
        batch = getattr(self.fundamental_service, "financial_batch_client", None)
        client = getattr(batch, "http_client", None)
        if client is None or market not in {"TWSE", "TPEX"}:
            return started_at
        from infrastructure.crawler.trading_restrictions import TradingRestrictionClient, TpexTradingRestrictionClient
        collector = TpexTradingRestrictionClient if market == "TPEX" else TradingRestrictionClient
        try:
            evidence = collector(client).load(
                getattr(crawler, "stocknumbers", ()),
                session_date=context.market_date.isoformat(), observed_at=started_at)
            clock = completion_clock or self._now_provider
            completed = clock() if clock is not None else started_at
            if (completed.tzinfo is None or completed < started_at
                    or completed.astimezone(TAIPEI_TIMEZONE).date() != started_at.date()):
                return started_at
            completed = completed.astimezone(TAIPEI_TIMEZONE)
            for item in evidence.values():
                item["available_at"] = completed.isoformat()
            bar = self._restriction_bar(crawler)
            symbols = [str(s) for s in getattr(crawler, "stocknumbers", ()) if s]
            # One missing/partial response is not a complete negative list.
            if (not bar or any(restriction_check(evidence.get(s, {}), market=market,
                    market_data_date=bar, decision_time=completed)["status"] == "UNKNOWN" for s in symbols)):
                return completed
            crawler.trading_restrictions = evidence
            if self._restriction_repository is not None and bar:
                try:
                    self._restriction_repository.record(evidence, market=market,
                        market_data_date=bar, decision_time=completed)
                except (OSError, sqlite3.Error, ValueError) as error:
                    logging.warning("Restriction archive write failed; current evidence retained: %s", error)
            return completed
        except (TwseClientError, ValueError, TypeError, AttributeError) as error:
            logging.warning("Restriction evidence unavailable; entries fail closed: %s", error)
            return started_at

    def ensure_current_restrictions(self, context: AnalysisContext) -> None:
        """Repair aged current-daily evidence at most once before cutoff freeze.

        Only a job that entered live fundamental refresh is armed. Replay,
        skip-MOPS and legacy daily paths never gain implicit HTTP requests.
        """
        crawler = context.resource
        settings = getattr(crawler, "app_settings", None)
        market = getattr(crawler, "_restriction_refresh_market", None)
        job = getattr(crawler, "_restriction_refresh_context", None)
        if (context.as_of is not None or market not in {"TWSE", "TPEX"}
                or not isinstance(job, weakref.ReferenceType) or job() is not context
                or getattr(settings, "daily_entry_model", "v6") != "v7"
                or not getattr(settings, "les_v7_native_evidence", False)
                or getattr(crawler, "_restriction_recapture_attempted", False)):
            return
        clock = self._now_provider or (lambda: datetime.datetime.now(TAIPEI_TIMEZONE))
        now = clock()
        if not isinstance(now, datetime.datetime) or now.utcoffset() is None:
            crawler.trading_restrictions = {}
            return
        now = now.astimezone(TAIPEI_TIMEZONE)
        bar = self._restriction_bar(crawler)
        symbols = [str(s) for s in getattr(crawler, "stocknumbers", ()) if s]
        if not bar or not symbols:
            crawler.trading_restrictions = {}
            return  # No evaluable batch: an HTTP request cannot repair bars.
        evidence = getattr(crawler, "trading_restrictions", {})
        if not isinstance(evidence, Mapping):
            evidence = {}
        if (bar and symbols and all(restriction_check(evidence.get(s, {}), market=market,
                market_data_date=bar, decision_time=now)["status"] != "UNKNOWN" for s in symbols)):
            return  # A fresh RESTRICTED observation is valid evidence too.
        crawler._restriction_recapture_attempted = True
        with measure_stage("restriction_pre_freeze_refresh"):
            completed = self._collect_restrictions(context, market=market, started_at=now,
                completion_clock=clock)
        captured = bool(getattr(crawler, "trading_restrictions", {}))
        self._observer.record("restriction_pre_freeze_refresh", source=market,
            status="SUCCESS" if captured else "UNKNOWN", attempted_at=now.isoformat(),
            observed_at=completed.isoformat() if captured else None, symbol_count=len(symbols))

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
        # Current lists are dated by actual observation, not the completed bar.
        crawler._restriction_refresh_market = request.market_source
        crawler._restriction_refresh_context = weakref.ref(context)
        crawler._restriction_recapture_attempted = False
        localized_time = self._collect_restrictions(context,
            market=request.market_source, started_at=localized_time)
        analysis_signal_date = localized_time.isoformat()
        # Research observations use completed bars and official restrictions;
        # no financial or entry status is rewritten by this pre-refresh pass.
        crawler.les_v6_research_context = {}
        build_history = (getattr(crawler, "build_strategy_screen_history", None)
                         or getattr(crawler, "build_strategy_history", None))
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
            watches, watch_key = screen_once(watch_histories, session_date=watch_session,
                                            as_of=analysis_signal_date, screener=screen_leadership)
            crawler.les_v6_research_context = {watch["symbol"]: {"watch": watch, "watch_input_sha256": watch_key,
                                                                "watch_result_sha256": watch_result_key(watch)}
                                             for watch in watches}
            del watch_histories
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
                with measure_stage("fundamental_local_migration"):
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
        retention = getattr(current_work(), "snapshot_retention", None)
        bundles_by_issuer = retention.bundles() if retention is not None else {}
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
                with measure_stage("fundamental_batch_fetch_save"):
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
                with measure_stage("fundamental_revision_scan"):
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

        # Native preloading and per-issuer reuse have the same work lifetime.
        # Use one owner map: decoding a preload only to retain the identical
        # snapshot in a second map otherwise pickles the full financial graph
        # again for every issuer. Legacy dict-returning callers keep their
        # existing separate mutable maps.
        preloaded_bundles = bundles_by_issuer if retention is not None else {}
        preload_cached_bundles = getattr(
            self.fundamental_service,
            "load_cached_bundles",
            None,
        )
        if callable(preload_cached_bundles):
            preload_started = time.monotonic()
            try:
                with measure_stage("fundamental_bulk_preload"):
                    streaming_preload = getattr(self.fundamental_service, "load_cached_bundles_into", None)
                    if retention is not None and callable(streaming_preload):
                        streaming_preload(list(dict.fromkeys(issuer_symbols)),
                            signal_date=analysis_signal_date, sector=sector,
                            history_years=request.mops_history_years,
                            sink=preloaded_bundles.__setitem__)
                    else:
                        loaded = preload_cached_bundles(list(dict.fromkeys(issuer_symbols)),
                            signal_date=analysis_signal_date, sector=sector,
                            history_years=request.mops_history_years)
                        if retention is None:
                            preloaded_bundles = loaded
                        else:
                            preloaded_bundles.update(loaded)
                        del loaded
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
                if retention is not None:
                    # Legacy assignment never publishes a partly returned
                    # preload dict after failure. Discard streamed partials as
                    # well; storage/capacity/hash errors outside these source
                    # exception types still propagate without fallback.
                    preloaded_bundles.clear()
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

        # Later misses populate the shared issuer map. Preserve the actual
        # preload set for scheduling and diagnostics, not its growing contents.
        preloaded_symbols = frozenset(preloaded_bundles)
        control = getattr(self.fundamental_service, "refresh_control", None)
        if control is not None:
            try:
                research = crawler.les_v6_research_context
                candidates = [symbol for symbol, row in research.items() if row["watch"]["priority_eligible"]]
                inspect = getattr(self.fundamental_service, "inspect_local_evidence", None)
                diagnostics = inspect(candidates, signal_date=analysis_signal_date, market=request.market_source, sector=sector) if callable(inspect) else {}
                for symbol, diagnostic in diagnostics.items():
                    research[symbol]["before_refresh"] = diagnostic
                priority_symbols = [symbol for symbol in candidates
                                    if diagnostics.get(symbol, {}).get("backfill_needed")
                                    and not diagnostics[symbol].get("local_review_required")
                                    and diagnostics[symbol].get("quality_status") != "REJECT"]
                control.schedule([symbol for symbol in issuer_symbols if symbol not in preloaded_symbols],
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
                        with measure_stage("fundamental_issuer_refresh"):
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
            native_v7 = (getattr(getattr(crawler, "app_settings", None), "daily_entry_model", "v6") == "v7"
                         and getattr(getattr(crawler, "app_settings", None), "les_v7_native_evidence", False))
            # A live refresh may advance its causal time after a new original
            # is actually ingested. Source supplements must use that same bound,
            # not the earlier request-start clock. Replay never enters this loop.
            source_cutoff = (bundle.as_of or analysis_signal_date) if native_v7 else analysis_signal_date
            bundle = self._attach_v7_sources(bundle, crawler=crawler, issuer=issuer_symbol,
                                             cutoff=source_cutoff, market=request.market_source)
            setter(
                stock_no,
                annual_fundamentals=bundle.annual_fundamentals,
                fundamental_snapshot=bundle.snapshot,
                account_equity=request.account_equity,
                signal_date=bundle.as_of or analysis_signal_date,
            )

        consume_history = getattr(self.fundamental_service, "consume_history_cache_summary", None)
        if callable(consume_history):
            with measure_stage("fundamental_history_summary"):
                summary = consume_history()
            if summary["completed_companies"] or summary["partial_companies"]:
                logging.info(
                    "MOPS history cache summary completed_companies=%s partial_companies=%s "
                    "pending_work=%s; scope=refreshed_companies. Use debug mode for per-symbol details.",
                    summary["completed_companies"], summary["partial_companies"], summary["pending_work"],
                    extra=structured_extra(source="MOPS_CACHE", status="HISTORY_CACHE_SUMMARY", summary=summary),
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
            preloaded_symbol_count=len(preloaded_symbols),
            duration_seconds=round(time.monotonic() - refresh_started, 3),
        )
        if retention is not None:
            # Only per-instrument collected snapshots are needed by strategies;
            # discard preload/issuer encodings without decoding them again.
            preloaded_bundles.clear()
            bundles_by_issuer.clear()

    def _replay_cached_fundamentals(self, context: AnalysisContext, request: StockAnalysisRequest) -> None:
        crawler = context.resource
        cutoff = request.as_of
        assert cutoff is not None
        sector = "24" if request.market_source == "TPEX" else request.stocktype.name
        retention = getattr(current_work(), "snapshot_retention", None)
        bundles = retention.bundles() if retention is not None else {}
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
                bundle = self._attach_v7_sources(bundle, crawler=crawler, issuer=identity.issuer_symbol,
                                                 cutoff=cutoff.isoformat(), market=request.market_source)
                crawler.set_low_entry_v4_fundamentals(
                    symbol, bundle.annual_fundamentals, bundle.snapshot,
                    account_equity=request.account_equity, signal_date=cutoff.isoformat(),
                )
            except MopsStorageError as error:
                self._mark_unavailable(crawler, symbol, status="REPLAY_CACHE_UNAVAILABLE", detail=str(error),
                                       signal_date=cutoff.isoformat(), account_equity=request.account_equity)
        self._observer.record("fundamental_historical_replay_completed", source="MOPS_CACHE",
                              status="SUCCESS", cutoff=cutoff.isoformat(), issuer_symbol_count=len(bundles))
        if retention is not None:
            bundles.clear()

    def _attach_v7_sources(self, bundle, *, crawler, issuer, cutoff, market):
        """R2/R3 local enrichment; reuse bounded work prep, never selections."""
        settings = getattr(crawler, "app_settings", None)
        if getattr(settings, "daily_entry_model", "v6") != "v7":
            return bundle
        import json
        from domain.les_v7.contracts import identity
        from domain.les_v7.quality_policy import is_source_evidence_policy, is_r7_policy
        from domain.les_v7.financial_sources import FINANCIAL_MAPPING
        from domain.les_v7.valuation_earnings import MAPPING_VERSION as EARNINGS_MAPPING
        from domain.les_v7.financial_sources_r5 import SOURCE_REPAIR_POLICY_R5
        captured = getattr(settings, "les_v7_bundle_json", None)
        extra = bundle.snapshot.get("les_v7_context", {})
        strategy = getattr(settings, "les_v7_strategy_policy_json", None)
        explicit = (json.loads(strategy)["payload"]["quality_policy"] if strategy else
                    json.loads(captured)["payload"]["quality_policy"] if captured else extra.get("quality_policy"))
        if explicit is not None and not is_source_evidence_policy(explicit):
            return bundle  # Do not add source work/metadata to an old R1 path.
        reader = getattr(self.fundamental_service, "load_v7_financial_sources", None)
        if not callable(reader):
            return bundle
        records = bundle.snapshot.get("les_v6_mops_filings", bundle.snapshot.get("les_v6_filings", ()))
        # The key touches only immutable source headers, not financial graphs.
        source_repair = bool(getattr(settings, "les_v7_source_repair", False)
            and getattr(settings, "les_v7_rule_baseline", False) and not getattr(crawler, "replay_as_of", None))
        effective_quality = explicit
        if source_repair and effective_quality is None and not captured and not strategy:
            from domain.les_v7.rule_baseline import baseline_policy, current_source_baseline_version
            effective_quality = baseline_policy(market,
                version=current_source_baseline_version(cutoff))["quality_policy"]
        audit_repair_r7 = source_repair and is_r7_policy(effective_quality)
        r7_identity = {}
        if audit_repair_r7:
            from infrastructure.fundamentals.parse_identity import source_fingerprint
            from domain.les_v7.financial_audit_r7 import SOURCE_VERSION
            r7_identity = dict(audit_source_policy=SOURCE_VERSION,
                audit_source_semantics=source_fingerprint((
                    ("domain/les_v7/financial_audit_r7.py", ()),
                    ("domain/les_v7/financial_sources_r5.py", ()),
                    ("domain/fundamental_derivation.py", ()),
                    ("domain/text_normalization.py", ()),
                )))
        key = identity(dict(issuer=issuer, market=market, cutoff=cutoff, mapping=FINANCIAL_MAPPING,
            earnings_mapping=EARNINGS_MAPPING,
            source_repair_policy=SOURCE_REPAIR_POLICY_R5 if source_repair else None,
            sources=[{k: row.get(k) for k in ("content_sha256", "revision_id", "available_at", "period_end")}
                     for row in records], **r7_identity))
        compute = lambda: reader(issuer, as_of=cutoff, records=records, market=market,
            **({"source_repair": True} if source_repair else {}),
            **({"audit_repair_r7": True} if audit_repair_r7 else {}))
        work = current_work()
        with measure_stage("v7_financial_sources"):
            additions = work.memo_json("v7_financial_sources", key, compute) if work is not None else compute()
        snapshot = dict(bundle.snapshot)
        snapshot["les_v7_context"] = dict(extra, **additions)
        return FundamentalBundle(bundle.annual_fundamentals, snapshot, bundle.as_of)

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
