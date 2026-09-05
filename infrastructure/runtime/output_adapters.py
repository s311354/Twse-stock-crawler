"""Report, notification, and chart runtime adapters."""

from __future__ import annotations

import datetime
import logging
import time
from collections.abc import Callable
from typing import Any

from application.contracts import AnalysisContext
from application.contracts import StockAnalysisRequest
from application.contracts import StrategyEvaluation
from application.observability import NullRunObserver
from application.ports import MarketCalendarPort
from application.ports import RunObserverPort
from config.settings import QUERY_WAIT_SECONDS


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
