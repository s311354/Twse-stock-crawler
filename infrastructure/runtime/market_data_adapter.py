"""TWSE market-data runtime adapter."""

from __future__ import annotations

import datetime
import json
import logging
import sqlite3
import time
from collections.abc import Callable
from typing import Any
from pathlib import Path

from application.contracts import AnalysisContext
from application.contracts import MarketStatus
from application.contracts import StockAnalysisRequest
from application.observability import NullRunObserver
from application.ports import MarketCalendarPort
from application.ports import RunObserverPort
from application.historical_replay import completed_session
from application.restriction_history import HistoricalRestrictionLedger
from domain.restriction_announcements import BUNDLE_SCHEMA


class CrawlerMarketDataAdapter:
    def __init__(
        self,
        crawler_factory: Callable[[int], Any],
        *,
        observer: RunObserverPort | None = None,
        tpex_crawler_factory: Callable[[int], Any] | None = None,
        restriction_repository: Any | None = None,
    ) -> None:
        self._crawler_factory = crawler_factory
        self._observer = observer or NullRunObserver()
        self._tpex_crawler_factory = tpex_crawler_factory
        self._restriction_repository = restriction_repository

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
            source=request.market_source,
            status=("OPEN" if is_trading_day else "CLOSED"),
            symbol_count=len(request.stocklist),
        )
        factory = self._crawler_factory
        if request.market_source == "TPEX":
            if self._tpex_crawler_factory is None:
                raise ValueError("TPEx crawler factory is required for SEMI")
            factory = self._tpex_crawler_factory
        crawler = factory(len(request.stocklist))
        crawler.analysis_universe = str(request.stocktype)
        # Shared multi-owner refreshes add symbols without making them PAE
        # holdings. They still need the same full-market recovery path as an
        # explicit portfolio holding when the selected industry omits them.
        crawler.explicit_stock_numbers = set(request.explicit_symbols).union(
            request.portfolio_symbols
        )
        date_bounds = {}
        if request.history_start_date is not None or request.history_end_date is not None:
            if request.as_of is None:
                raise ValueError("EXPLICIT_DATE_RANGE_REQUIRES_REPLAY_CUTOFF")
            date_bounds = {"start_date": request.history_start_date, "end_date": request.history_end_date}
        crawler.get_date_times(
            begin_backtrack=request.beginbacktrack,
            end_backtrack=request.endbacktrack,
            calendar=calendar,
            reference_date=market_date,
            include_reference=include_reference,
            **date_bounds,
        )
        historical_ledger = None
        if request.as_of is not None:
            expected = completed_session(request.as_of, calendar).isoformat()
            scheduled = getattr(crawler, "iso_scheduled_times", [])
            if not scheduled or scheduled[-1] != expected:
                raise ValueError("REPLAY_RANGE_MISSES_CUTOFF_SESSION: include the latest completed session in the date range; for integer offsets use -b 0 and a sufficient -e range")
            crawler.replay_as_of = request.as_of.isoformat()
            crawler.trading_restrictions = {}
            if request.replay_restrictions_file:
                archive = json.loads(Path(request.replay_restrictions_file).read_text(encoding="utf-8"))
                if isinstance(archive, dict) and archive.get("schema_version") == BUNDLE_SCHEMA:
                    historical_ledger = HistoricalRestrictionLedger(archive)
                elif not isinstance(archive, dict) or any(not isinstance(v, dict) for v in archive.values()):
                    raise ValueError("REPLAY_RESTRICTIONS_REQUIRE_SYMBOL_MAPPING")
                # The domain restriction gate checks market, complete sources,
                # observation date, freshness and available_at against the cutoff.
                else:
                    crawler.trading_restrictions = archive
            elif self._restriction_repository is not None:
                try:
                    crawler.trading_restrictions = self._restriction_repository.load_as_of(
                        market=request.market_source, market_data_date=expected, as_of=request.as_of,
                    )
                except (OSError, sqlite3.Error, ValueError) as error:
                    logging.warning("Historical restriction archive unavailable; entries fail closed: %s", error)
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
                    source=request.market_source,
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
                    source=request.market_source,
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
                    source=request.market_source,
                    status="TRACKED_STOCK_MISSING",
                    missing_tracked_stocks=missing_tracked_stocks,
                    missing_tracked_stock_count=len(missing_tracked_stocks),
                )
        if request.as_of is not None:
            if historical_ledger is not None:
                # Index-based stock selectors have been resolved by the crawler.
                # All downstream models share this one cutoff-bound reconstruction.
                crawler.trading_restrictions = historical_ledger.reconstruct(
                    (symbol for symbol in crawler.stocknumbers if symbol),
                    market=request.market_source, market_data_date=expected, as_of=request.as_of,
                )
            # This also freezes the decision clock when MOPS is explicitly skipped.
            crawler.low_entry_v4_signal_dates_by_stock = {
                str(symbol): request.as_of.isoformat() for symbol in crawler.stocknumbers if symbol
            }
        self._observer.record(
            "market_data_fetch_completed",
            trade_date=trade_date,
            source=request.market_source,
            status="SUCCESS",
            symbol_count=getattr(crawler, "stocklistsize", len(request.stocklist)),
            trading_days=len(getattr(crawler, "iso_scheduled_times", [])),
            duration_seconds=round(time.monotonic() - fetch_started, 3),
        )
        return AnalysisContext(
            resource=crawler,
            market_date=market_date,
            market_status=(
                MarketStatus.OPEN if is_trading_day else MarketStatus.CLOSED
            ),
            as_of=request.as_of,
        )
