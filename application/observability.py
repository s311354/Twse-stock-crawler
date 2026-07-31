"""No-op observability implementation for isolated application tests."""

from __future__ import annotations

import logging

from application.contracts import StockAnalysisRequest


class NullRunObserver:
    def start(self, request: StockAnalysisRequest) -> None:
        pass

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
        pass

    def finish(self, status: str) -> None:
        pass
