"""Deterministic use cases for the daily stock-analysis workflow."""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable
from zoneinfo import ZoneInfo

from application.contracts import AnalysisContext
from application.contracts import MarketStatus
from application.contracts import StockAnalysisRequest
from application.contracts import StrategyEvaluation
from application.historical_replay import SESSION_CLOSE, TAIPEI
from application.ports import FundamentalPort
from application.ports import MarketCalendarPort
from application.ports import MarketDataPort
from application.ports import NotificationPort
from application.ports import ReportPort
from application.ports import StrategyPort


TAIPEI_TIMEZONE = ZoneInfo("Asia/Taipei")
MARKET_DATA_READY_TIME = datetime.time(13, 45)


class FetchMarketData:
    def __init__(
        self,
        market_data: MarketDataPort,
        calendar: MarketCalendarPort,
        today_provider: Callable[[], datetime.date],
        now_provider: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self._market_data = market_data
        self._calendar = calendar
        self._today_provider = today_provider
        self._now_provider = now_provider

    def execute(self, request: StockAnalysisRequest) -> AnalysisContext:
        if request.as_of is None and request.beginbacktrack > 0:
            # Offsets describe a price window, not a historical account or
            # evidence cutoff. Never refresh today's evidence for an old end.
            raise ValueError("HISTORICAL_RANGE_REQUIRES_DATE_REPLAY: use -b START_DATE -e END_DATE with pinned evidence and an explicit historical account")
        if request.replay_mail and request.as_of is None:
            raise ValueError("Replay email requires a historical cutoff")
        if request.as_of is not None and (request.as_of.utcoffset() is None or (request.mail and not request.replay_mail) or request.linechart):
            raise ValueError("Historical replay requires an aware cutoff, explicit replay-mail for email, and no live charts")
        if request.as_of is not None and request.as_of.astimezone(TAIPEI).time() != SESSION_CLOSE:
            raise ValueError("REPLAY_CLOSE_ONLY: historical replay cutoff must be Taipei 13:30")
        market_date = request.as_of.astimezone(TAIPEI).date() if request.as_of else self._today_provider()
        try:
            is_trading_day = self._calendar.is_trading_day(market_date)
        except Exception:
            logging.error(
                "TWSE market status for %s is UNKNOWN; aborting analysis.",
                market_date.isoformat(),
            )
            raise

        market_status = MarketStatus.OPEN if is_trading_day else MarketStatus.CLOSED
        print(
            "{} market status for {}: {}".format(
                request.market_source,
                market_date.isoformat(),
                market_status.value,
            )
        )
        if not is_trading_day:
            logging.info(
                "TWSE is closed today; continuing analysis for OPEN dates "
                "in the requested backtrack range."
            )

        if request.as_of is not None:
            include_reference = is_trading_day
        else:
            include_reference = self._reference_data_is_ready(market_date, is_trading_day=is_trading_day)

        return self._market_data.fetch(
            request,
            calendar=self._calendar,
            market_date=market_date,
            is_trading_day=is_trading_day,
            include_reference=include_reference,
        )

    def _reference_data_is_ready(
        self,
        market_date: datetime.date,
        *,
        is_trading_day: bool,
    ) -> bool:
        if not is_trading_day or self._now_provider is None:
            return True

        current = self._now_provider()
        if current.tzinfo is None:
            current = current.replace(tzinfo=TAIPEI_TIMEZONE)
        else:
            current = current.astimezone(TAIPEI_TIMEZONE)

        if current.date() != market_date:
            return current.date() > market_date
        return current.time().replace(tzinfo=None) >= MARKET_DATA_READY_TIME


class RefreshFundamentals:
    def __init__(
        self,
        fundamentals: FundamentalPort,
        now_provider: Callable[[], datetime.datetime],
    ) -> None:
        self._fundamentals = fundamentals
        self._now_provider = now_provider

    def execute(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
    ) -> None:
        if not request.mops_enabled:
            return
        analysis_time = request.as_of or self._now_provider()
        if analysis_time.tzinfo is None:
            raise ValueError("now_provider must return a timezone-aware datetime")
        self._fundamentals.refresh(
            context,
            request,
            analysis_time=analysis_time,
        )


class EvaluateStrategies:
    def __init__(self, strategies: StrategyPort) -> None:
        self._strategies = strategies

    def execute(self, context: AnalysisContext) -> StrategyEvaluation:
        return self._strategies.evaluate(context)


class GenerateReport:
    def __init__(self, reports: ReportPort) -> None:
        self._reports = reports

    def execute(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
        evaluation: StrategyEvaluation,
    ) -> None:
        self._reports.generate(context, request, evaluation)


class SendNotification:
    def __init__(self, notifications: NotificationPort) -> None:
        self._notifications = notifications

    def execute(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
        evaluation: StrategyEvaluation,
    ) -> None:
        if request.notification_enabled:
            self._notifications.send(context, request, evaluation)
