"""Infrastructure adapters for the stock-analysis application ports."""

from __future__ import annotations

import datetime
import logging
import time
from collections.abc import Callable
from typing import Any
from zoneinfo import ZoneInfo

from application.contracts import AnalysisContext
from application.contracts import MarketStatus
from application.contracts import StockAnalysisRequest
from application.contracts import StrategyEvaluation
from application.observability import NullRunObserver
from application.portfolio_allocation import PortfolioAllocationPipeline
from application.portfolio_allocation import PortfolioRuntimeContext
from application.ports import MarketCalendarPort
from application.ports import RunObserverPort
from config.settings import QUERY_WAIT_SECONDS
from domain.fundamental_derivation import FundamentalBundle
from domain.models import PortfolioPosition
from domain.security_identity import SecurityIdentity
from domain.security_identity import resolve_security_identity
from infrastructure.crawler.twse_http_client import TwseClientError
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.fundamentals.mops_repository import MopsStorageError
from infrastructure.observability import structured_extra


TAIPEI_TIMEZONE = ZoneInfo("Asia/Taipei")
MOPS_INFRASTRUCTURE_FAILURE_THRESHOLD = 2


class CrawlerMarketDataAdapter:
    def __init__(
        self,
        crawler_factory: Callable[[int], Any],
        *,
        observer: RunObserverPort | None = None,
    ) -> None:
        self._crawler_factory = crawler_factory
        self._observer = observer or NullRunObserver()

    def fetch(
        self,
        request: StockAnalysisRequest,
        *,
        calendar: MarketCalendarPort,
        market_date: datetime.date,
        is_trading_day: bool,
        include_reference: bool,
    ) -> AnalysisContext:
        trade_date = market_date.isoformat()
        self._observer.record(
            "market_data_fetch_started",
            trade_date=trade_date,
            source="TWSE",
            status=("OPEN" if is_trading_day else "CLOSED"),
            symbol_count=len(request.stocklist),
        )
        crawler = self._crawler_factory(len(request.stocklist))
        crawler.analysis_universe = str(request.stocktype)
        crawler.get_date_times(
            begin_backtrack=request.beginbacktrack,
            end_backtrack=request.endbacktrack,
            calendar=calendar,
            reference_date=market_date,
            include_reference=include_reference,
        )
        try:
            crawler.get_twse_daily_stocks(
                file_name=request.output_file_names,
                stocktype=request.stocktype,
                stocks=request.stocklist,
            )
        finally:
            no_data_dates = sorted(
                getattr(crawler, "twse_no_data_dates", ())
            )
            if no_data_dates:
                self._observer.record(
                    "market_data_no_data_dates_collected",
                    level=logging.DEBUG,
                    source="TWSE",
                    status="NO_DATA",
                    no_data_dates=no_data_dates,
                    no_data_date_count=len(no_data_dates),
                )
            closed_unscheduled_dates = sorted(
                getattr(crawler, "twse_closed_unscheduled_dates", ())
            )
            if closed_unscheduled_dates:
                self._observer.record(
                    "market_closed_unscheduled_dates_collected",
                    level=logging.DEBUG,
                    source="TWSE",
                    status="CLOSED_UNSCHEDULED",
                    closed_unscheduled_dates=closed_unscheduled_dates,
                    closed_unscheduled_date_count=len(
                        closed_unscheduled_dates
                    ),
                )
            missing_tracked_stocks = dict(
                getattr(crawler, "missing_tracked_stock_first_dates", {})
            )
            if missing_tracked_stocks:
                self._observer.record(
                    "market_data_missing_tracked_stocks_collected",
                    level=logging.DEBUG,
                    source="TWSE",
                    status="TRACKED_STOCK_MISSING",
                    missing_tracked_stocks=missing_tracked_stocks,
                    missing_tracked_stock_count=len(missing_tracked_stocks),
                )
        self._observer.record(
            "market_data_fetch_completed",
            trade_date=trade_date,
            source="TWSE",
            status="SUCCESS",
            symbol_count=len(request.stocklist),
            trading_days=len(getattr(crawler, "iso_scheduled_times", [])),
        )
        return AnalysisContext(
            resource=crawler,
            market_date=market_date,
            market_status=(
                MarketStatus.OPEN if is_trading_day else MarketStatus.CLOSED
            ),
        )


class CrawlerFundamentalAdapter:
    """Attach MOPS bundles while preserving fail-closed LES behavior."""

    def __init__(
        self,
        fundamental_service: Any,
        *,
        observer: RunObserverPort | None = None,
    ) -> None:
        self.fundamental_service = fundamental_service
        self._observer = observer or NullRunObserver()

    def refresh(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
        *,
        analysis_time: datetime.datetime,
    ) -> None:
        crawler = context.resource
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
        analysis_signal_date = localized_time.isoformat()
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

        prepare_daily_batch = getattr(
            self.fundamental_service,
            "prepare_daily_batch",
            None,
        )
        if callable(prepare_daily_batch):
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

        issuer_symbols = []
        for item, stock_no in enumerate(stocknumbers):
            if not stock_no or item >= len(daily_dates) or not daily_dates[item]:
                continue
            stock_name = stocknames[item] if item < len(stocknames) else ""
            issuer_symbols.append(
                resolve_security_identity(stock_no, stock_name).issuer_symbol
            )
        preloaded_bundles: dict[str, FundamentalBundle] = {}
        preload_cached_bundles = getattr(
            self.fundamental_service,
            "load_cached_bundles",
            None,
        )
        if callable(preload_cached_bundles):
            try:
                preloaded_bundles = preload_cached_bundles(
                    list(dict.fromkeys(issuer_symbols)),
                    signal_date=analysis_signal_date,
                    sector=str(request.stocktype),
                    history_years=request.mops_history_years,
                )
            except (MopsStorageError, TwseClientError) as error:
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

        for item, stock_no in enumerate(stocknumbers):
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
                            sector=str(request.stocktype),
                            history_years=request.mops_history_years,
                        )
                    bundles_by_issuer[issuer_symbol] = bundle
                bundle = self._attach_security_identity(bundle, identity)
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
                        stocknumbers[item + 1 :],
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
                        stocknumbers[item + 1 :],
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
                        stocknumbers[item + 1 :],
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


class CrawlerStrategyAdapter:
    def __init__(
        self,
        *,
        observer: RunObserverPort | None = None,
        portfolio_allocator: PortfolioAllocationPipeline | None = None,
        portfolio_positions: tuple[PortfolioPosition, ...] = (),
        portfolio_current_cash: float | None = None,
        account_equity: float = 1_000_000.0,
    ) -> None:
        self._observer = observer or NullRunObserver()
        self._portfolio_allocator = portfolio_allocator
        self._portfolio_positions = portfolio_positions
        self._portfolio_current_cash = portfolio_current_cash
        self._account_equity = account_equity

    def evaluate(self, context: AnalysisContext) -> StrategyEvaluation:
        crawler = context.resource
        if self._portfolio_allocator is not None:
            crawler.portfolio_entry_les_by_stock = {
                position.symbol: position.entry_les
                for position in self._portfolio_positions
                if position.entry_les is not None
            }
            crawler.portfolio_entry_dates_by_stock = {
                position.symbol: position.entry_date
                for position in self._portfolio_positions
                if position.entry_date is not None
            }
        maxprofits = crawler.cal_max_profit()
        decision_builder = getattr(
            crawler,
            "build_strategy_decisions",
            None,
        )
        decisions = (
            tuple(decision_builder(maxprofits))
            if callable(decision_builder)
            else ()
        )
        if self._portfolio_allocator is not None and decisions:
            snapshots = getattr(
                crawler,
                "fundamental_snapshots_by_stock",
                {},
            )
            sector_by_symbol = {
                str(symbol): str(snapshot.get("sector", "")).strip()
                for symbol, snapshot in snapshots.items()
                if isinstance(snapshot, dict)
                and str(snapshot.get("sector", "")).strip()
            }
            decisions = self._portfolio_allocator.enrich(
                decisions,
                PortfolioRuntimeContext(
                    account_equity=self._account_equity,
                    positions=self._portfolio_positions,
                    current_cash=self._portfolio_current_cash,
                    sector_by_symbol=sector_by_symbol,
                    default_sector=str(
                        getattr(crawler, "analysis_universe", "UNKNOWN")
                    ),
                ),
            )
        evaluation = StrategyEvaluation(
            maxprofits=maxprofits,
            decisions=decisions,
        )
        self._observer.record(
            "strategy_evaluation_completed",
            trade_date=context.market_date.isoformat(),
            source="STRATEGY",
            status="SUCCESS",
            result_count=len(evaluation.maxprofits),
            pae_status=(
                evaluation.decisions[0].value("PAE Status")
                if evaluation.decisions
                else None
            ),
            pae_mode=(
                evaluation.decisions[0].value("PAE Mode")
                if evaluation.decisions
                else None
            ),
        )
        return evaluation


class CrawlerReportAdapter:
    def __init__(self, *, observer: RunObserverPort | None = None) -> None:
        self._observer = observer or NullRunObserver()

    def generate(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
        evaluation: StrategyEvaluation,
    ) -> None:
        crawler = context.resource
        decision_recorder = getattr(
            crawler,
            "record_strategy_decisions",
            None,
        )
        if evaluation.decisions and callable(decision_recorder):
            decision_recorder(
                file_name=request.output_file_names,
                decisions=evaluation.decisions,
            )
        else:
            crawler.record_analysis_dataset(
                file_name=request.output_file_names,
                maxprofits=evaluation.maxprofits,
            )
        self._observer.record(
            "report_generated",
            trade_date=context.market_date.isoformat(),
            source="REPORT",
            status="SUCCESS",
            output=request.output_file_names,
        )


class CrawlerNotificationAdapter:
    def __init__(self, *, observer: RunObserverPort | None = None) -> None:
        self._observer = observer or NullRunObserver()

    def send(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
        evaluation: StrategyEvaluation,
    ) -> None:
        crawler = context.resource
        strategy_mailer = getattr(
            crawler,
            "smtp_strategy_email",
            None,
        )
        if evaluation.decisions and callable(strategy_mailer):
            strategy_mailer(
                subject=request.subject,
                ccreceiver=request.ccreceiver,
                stocktype=request.stocktype,
                decisions=evaluation.decisions,
            )
        else:
            crawler.smtp_email(
                subject=request.subject,
                ccreceiver=request.ccreceiver,
                stocktype=request.stocktype,
                maxprofits=evaluation.maxprofits,
            )
        self._observer.record(
            "notification_sent",
            trade_date=context.market_date.isoformat(),
            source="SMTP",
            status="SUCCESS",
            receiver_count="configured",
        )


class CrawlerLineChartAdapter:
    def __init__(
        self,
        crawler_factory: Callable[[int], Any],
        calendar: MarketCalendarPort,
        *,
        now_provider: Callable[[], datetime.datetime],
        sleeper: Callable[[float], None] = time.sleep,
        observer: RunObserverPort | None = None,
    ) -> None:
        self._crawler_factory = crawler_factory
        self._calendar = calendar
        self._now_provider = now_provider
        self._sleep = sleeper
        self._observer = observer or NullRunObserver()

    def generate(self, request: StockAnalysisRequest) -> None:
        self._observer.record(
            "line_chart_started",
            source="TWSE",
            status="STARTED",
            symbol_count=len(request.stocklist),
        )
        startofbacktrack = request.endbacktrack - request.period
        backtrack = (
            self._now_provider()
            + datetime.timedelta(days=-request.endbacktrack)
        ).strftime("%Y-%m-%d")
        maxprofitratios: list[list[float]] = [
            [] for _ in range(len(request.stocklist))
        ]

        for back in range(startofbacktrack):
            crawler = self._crawler_factory(len(request.stocklist))
            logging.debug("Line-chart backtrack iteration=%s", back)
            crawler.get_date_times(
                begin_backtrack=startofbacktrack - back,
                end_backtrack=request.endbacktrack - back,
                calendar=self._calendar,
            )
            crawler.get_twse_daily_stocks(
                file_name=request.output_file_names,
                stocktype=request.stocktype,
                stocks=request.stocklist,
            )
            for item, ratio in enumerate(crawler.cal_max_profit_ratio_data()):
                if item < len(maxprofitratios):
                    maxprofitratios[item].append(ratio)

            if back == startofbacktrack - 1:
                crawler.draw_linechart(
                    duration=startofbacktrack,
                    maxprofitratios=maxprofitratios,
                )
                if request.mail:
                    crawler.smtp_img_email(
                        subject=request.subject,
                        ccreceiver=request.ccreceiver,
                        stocktype=request.stocktype,
                        backtrack=backtrack,
                    )
                continue
            self._sleep(QUERY_WAIT_SECONDS)

        logging.debug(
            "The trend of performance indicators for TWSE stock market: %s",
            maxprofitratios,
        )
        self._observer.record(
            "line_chart_completed",
            source="REPORT",
            status="SUCCESS",
            symbol_count=len(request.stocklist),
        )
