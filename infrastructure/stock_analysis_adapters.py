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
from application.ports import MarketCalendarPort
from application.ports import RunObserverPort
from config.settings import QUERY_WAIT_SECONDS
from infrastructure.crawler.twse_http_client import TwseClientError
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.fundamentals.mops_repository import MopsStorageError


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
        crawler.get_date_times(
            begin_backtrack=request.beginbacktrack,
            end_backtrack=request.endbacktrack,
            calendar=calendar,
            reference_date=market_date,
            include_reference=True,
        )
        crawler.get_twse_daily_stocks(
            file_name=request.output_file_names,
            stocktype=request.stocktype,
            stocks=request.stocklist,
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

        for item, stock_no in enumerate(stocknumbers):
            if not stock_no or item >= len(daily_dates) or not daily_dates[item]:
                continue
            try:
                bundle = self.fundamental_service.load(
                    stock_no,
                    signal_date=analysis_signal_date,
                    sector=str(request.stocktype),
                    history_years=request.mops_history_years,
                )
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
                    "MOPS status=%s symbol=%s detail=%s; LES v4 remains WATCH_ONLY.",
                    error.category.value.upper(),
                    stock_no,
                    error,
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
                        "skipping remaining symbols and keeping LES v4 WATCH_ONLY.",
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
                    "MOPS status=%s symbol=%s detail=%s; LES v4 remains WATCH_ONLY.",
                    error.category.value.upper(),
                    stock_no,
                    error,
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
                        "skipping remaining symbols and keeping LES v4 WATCH_ONLY.",
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
                    "MOPS storage symbol=%s unavailable; LES v4 remains WATCH_ONLY: %s",
                    stock_no,
                    error,
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
                        "skipping remaining symbols and keeping LES v4 WATCH_ONLY.",
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
                symbol=stock_no,
                trade_date=context.market_date.isoformat(),
                source="MOPS_CACHE" if cache_hit else "MOPS",
                status="SUCCESS",
                cache_hit=cache_hit,
                available_at=bundle.as_of,
            )
            setter(
                stock_no,
                annual_fundamentals=bundle.annual_fundamentals,
                fundamental_snapshot=bundle.snapshot,
                account_equity=request.account_equity,
                signal_date=bundle.as_of or analysis_signal_date,
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
    def __init__(self, *, observer: RunObserverPort | None = None) -> None:
        self._observer = observer or NullRunObserver()

    def evaluate(self, context: AnalysisContext) -> StrategyEvaluation:
        crawler = context.resource
        evaluation = StrategyEvaluation(maxprofits=crawler.cal_max_profit())
        self._observer.record(
            "strategy_evaluation_completed",
            trade_date=context.market_date.isoformat(),
            source="STRATEGY",
            status="SUCCESS",
            result_count=len(evaluation.maxprofits),
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
            print("Back {}".format(back))
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

        logging.info(
            "The trend of performance indicators for TWSE stock market: %s",
            maxprofitratios,
        )
        self._observer.record(
            "line_chart_completed",
            source="REPORT",
            status="SUCCESS",
            symbol_count=len(request.stocklist),
        )
