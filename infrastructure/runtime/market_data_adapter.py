"""TWSE market-data runtime adapter."""

from __future__ import annotations

import datetime
import logging
import time
from collections.abc import Callable
from typing import Any

from application.contracts import AnalysisContext
from application.contracts import MarketStatus
from application.contracts import StockAnalysisRequest
from application.observability import NullRunObserver
from application.ports import MarketCalendarPort
from application.ports import RunObserverPort


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
        fetch_started = time.monotonic()
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
        # Shared multi-owner refreshes add symbols without making them PAE
        # holdings. They still need the same full-market recovery path as an
        # explicit portfolio holding when the selected industry omits them.
        crawler.explicit_stock_numbers = set(request.explicit_symbols).union(
            request.portfolio_symbols
        )
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
            duration_seconds=round(time.monotonic() - fetch_started, 3),
        )
        return AnalysisContext(
            resource=crawler,
            market_date=market_date,
            market_status=(
                MarketStatus.OPEN if is_trading_day else MarketStatus.CLOSED
            ),
        )
