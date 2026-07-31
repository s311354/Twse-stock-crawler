"""Ports required by the stock-analysis application workflow."""

from __future__ import annotations

import datetime
import logging
from typing import Protocol

from application.contracts import AnalysisContext
from application.contracts import StockAnalysisRequest
from application.contracts import StrategyEvaluation


class MarketCalendarPort(Protocol):
    def is_trading_day(self, target_date: datetime.date) -> bool:
        """Return whether target_date is an official TWSE trading day."""


class MarketDataPort(Protocol):
    def fetch(
        self,
        request: StockAnalysisRequest,
        *,
        calendar: MarketCalendarPort,
        market_date: datetime.date,
        is_trading_day: bool,
    ) -> AnalysisContext:
        """Fetch the requested OHLCV history and return an opaque context."""


class FundamentalPort(Protocol):
    def refresh(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
        *,
        analysis_time: datetime.datetime,
    ) -> None:
        """Refresh point-in-time fundamentals for the fetched symbols."""


class StrategyPort(Protocol):
    def evaluate(self, context: AnalysisContext) -> StrategyEvaluation:
        """Evaluate configured strategies against the fetched data."""


class ReportPort(Protocol):
    def generate(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
        evaluation: StrategyEvaluation,
    ) -> None:
        """Persist or render the analysis report."""


class NotificationPort(Protocol):
    def send(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
        evaluation: StrategyEvaluation,
    ) -> None:
        """Send the generated analysis notification."""


class LineChartPort(Protocol):
    def generate(self, request: StockAnalysisRequest) -> None:
        """Run the legacy multi-window performance chart flow."""


class RunObserverPort(Protocol):
    def start(self, request: StockAnalysisRequest) -> None:
        """Start one observable analysis run."""

    def record(
        self,
        event: str,
        *,
        level: int = logging.INFO,
        symbol: str | None = None,
        trade_date: str | None = None,
        source: str | None = None,
        status: str | None = None,
        attempt: int | None = None,
        cache_hit: bool | None = None,
        **metrics: object,
    ) -> None:
        """Record one structured event."""

    def finish(self, status: str) -> None:
        """Emit the final run summary."""
