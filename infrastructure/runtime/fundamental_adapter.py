"""Thin application-port adapter for fundamental refresh."""

from __future__ import annotations

import datetime
from typing import Any

from application.contracts import AnalysisContext
from application.contracts import StockAnalysisRequest
from application.observability import NullRunObserver
from application.ports import RunObserverPort
from infrastructure.runtime.fundamental_refresh import (
    FundamentalRefreshCoordinator,
)


class CrawlerFundamentalAdapter:
    """Delegate the application port to the infrastructure coordinator."""

    def __init__(
        self,
        fundamental_service: Any,
        *,
        observer: RunObserverPort | None = None,
        coordinator: FundamentalRefreshCoordinator | None = None,
    ) -> None:
        self.fundamental_service = fundamental_service
        self._observer = observer or NullRunObserver()
        self._coordinator = coordinator or FundamentalRefreshCoordinator(
            fundamental_service,
            observer=self._observer,
        )

    def refresh(
        self,
        context: AnalysisContext,
        request: StockAnalysisRequest,
        *,
        analysis_time: datetime.datetime,
    ) -> None:
        self._coordinator.refresh(
            context,
            request,
            analysis_time=analysis_time,
        )
