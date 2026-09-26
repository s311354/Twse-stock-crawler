"""Report, notification, and chart runtime adapters."""

from __future__ import annotations

import datetime
import json
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
                file_name=request.report_file_prefix,
                decisions=evaluation.decisions,
            )
        else:
            crawler.record_analysis_dataset(
                file_name=request.report_file_prefix,
                maxprofits=evaluation.maxprofits,
            )
        self._observer.record(
            "report_generated",
            trade_date=context.market_date.isoformat(),
            source="REPORT",
            status="SUCCESS",
            output=request.report_file_prefix,
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
        subject = request.subject
        if request.as_of is not None:
            if not request.notification_enabled or context.as_of != request.as_of:
                raise ValueError("REPLAY_MAIL_NOT_AUTHORIZED_OR_CUTOFF_MISMATCH")
            if not evaluation.decisions:
                raise ValueError("REPLAY_MAIL_REQUIRES_TYPED_DECISIONS")
            for decision in evaluation.decisions:
                try:
                    report = json.loads(decision.value("LES v6 Report"))
                    if (report.get("evaluation_mode") != "HISTORICAL_REPLAY"
                            or datetime.datetime.fromisoformat(report["replay_as_of"].replace("Z", "+00:00")) != request.as_of
                            or datetime.datetime.fromisoformat(report["decision_time"].replace("Z", "+00:00")) != request.as_of):
                        raise ValueError("mismatched replay evidence")
                except (KeyError, TypeError, ValueError, AttributeError) as error:
                    raise ValueError("REPLAY_MAIL_REPORT_CUTOFF_MISMATCH") from error
            subject = f"[HISTORICAL_REPLAY｜歷史重播｜不可下單｜截止 {request.as_of.isoformat()}] {subject}"
        strategy_mailer = getattr(
            crawler,
            "smtp_strategy_email",
            None,
        )
        if evaluation.decisions and callable(strategy_mailer):
            strategy_mailer(
                subject=subject,
                ccreceiver=request.ccreceiver,
                stocktype=request.stocktype,
                decisions=evaluation.decisions,
            )
        else:
            if request.as_of is not None:
                raise ValueError("REPLAY_MAIL_REQUIRES_TYPED_MAILER")
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
        tpex_crawler_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self._crawler_factory = crawler_factory
        self._calendar = calendar
        self._now_provider = now_provider
        self._sleep = sleeper
        self._observer = observer or NullRunObserver()
        self._tpex_crawler_factory = tpex_crawler_factory

    def generate(self, request: StockAnalysisRequest) -> None:
        factory = self._crawler_factory
        if request.market_source == "TPEX":
            if "ALL" in request.stocklist:
                raise ValueError("SEMI line charts require explicit TPEx stock numbers")
            if self._tpex_crawler_factory is None:
                raise ValueError("TPEx crawler factory is required for SEMI")
            factory = self._tpex_crawler_factory
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
            crawler = factory(len(request.stocklist))
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
