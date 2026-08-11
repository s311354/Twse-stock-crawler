"""Structured logging and per-run telemetry for the TWSE workflow."""

from __future__ import annotations

import contextvars
import datetime
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable
from threading import Lock
from typing import TextIO

from application.contracts import StockAnalysisRequest


_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "twse_run_id",
    default="",
)
STRUCTURED_FIELDS = (
    "run_id",
    "symbol",
    "trade_date",
    "source",
    "status",
    "attempt",
    "cache_hit",
)


def structured_extra(**fields: object) -> dict[str, object]:
    values = {
        "run_id": _RUN_ID.get() or None,
        "symbol": None,
        "trade_date": None,
        "source": None,
        "status": None,
        "attempt": None,
        "cache_hit": None,
    }
    values.update(fields)
    return values


class JsonLogFormatter(logging.Formatter):
    """Render every log record as one machine-readable JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field_name in STRUCTURED_FIELDS:
            payload[field_name] = getattr(record, field_name, None)
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        metrics = getattr(record, "metrics", None)
        if metrics:
            payload["metrics"] = metrics
        summary = getattr(record, "summary", None)
        if summary:
            payload["summary"] = summary
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )


class StructuredContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        defaults = structured_extra()
        for field_name, value in defaults.items():
            if not hasattr(record, field_name):
                setattr(record, field_name, value)
        return True


def configure_structured_logging(
    level: str,
    *,
    stream: TextIO | None = None,
) -> None:
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    handler.addFilter(StructuredContextFilter())
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


class StructuredRunObserver:
    def __init__(
        self,
        run_id: str,
        *,
        logger: logging.Logger | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id cannot be empty")
        self.run_id = run_id
        self._logger = logger or logging.getLogger("twse.run")
        self._monotonic = monotonic
        self._started_at: float | None = None
        self._events: Counter[str] = Counter()
        self._statuses: Counter[str] = Counter()
        self._sources: Counter[str] = Counter()
        self._symbols: set[str] = set()
        self._cache_hits = 0
        self._cache_misses = 0
        self._requested_symbols = 0
        self._no_data_dates: set[str] = set()
        self._closed_unscheduled_dates: set[str] = set()
        self._missing_tracked_stocks: dict[str, str] = {}
        self._lock = Lock()

    def start(self, request: StockAnalysisRequest) -> None:
        _RUN_ID.set(self.run_id)
        self._started_at = self._monotonic()
        self._requested_symbols = len(request.stocklist)
        self.record(
            "run_started",
            source="APPLICATION",
            status="STARTED",
            symbol_count=self._requested_symbols,
            mail_enabled=request.mail,
            mops_enabled=request.mops_enabled,
        )

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
        with self._lock:
            self._events[event] += 1
            if status:
                self._statuses[status] += 1
            if source:
                self._sources[source] += 1
            if symbol:
                self._symbols.add(symbol)
            if cache_hit is True:
                self._cache_hits += 1
            elif cache_hit is False:
                self._cache_misses += 1
            if source == "TWSE" and status == "NO_DATA":
                if trade_date:
                    self._no_data_dates.add(str(trade_date))
                observed_dates = metrics.get("no_data_dates", ())
                if isinstance(observed_dates, (list, tuple, set)):
                    self._no_data_dates.update(
                        str(value) for value in observed_dates if value
                    )
            if source == "TWSE" and status == "CLOSED_UNSCHEDULED":
                if trade_date:
                    self._closed_unscheduled_dates.add(str(trade_date))
                observed_dates = metrics.get(
                    "closed_unscheduled_dates",
                    (),
                )
                if isinstance(observed_dates, (list, tuple, set)):
                    self._closed_unscheduled_dates.update(
                        str(value) for value in observed_dates if value
                    )
            if source == "TWSE" and status == "TRACKED_STOCK_MISSING":
                observed_stocks = metrics.get("missing_tracked_stocks", {})
                if isinstance(observed_stocks, dict):
                    for stock_no, first_date in observed_stocks.items():
                        symbol_text = str(stock_no)
                        date_text = str(first_date)
                        previous = self._missing_tracked_stocks.get(symbol_text)
                        if previous is None or date_text < previous:
                            self._missing_tracked_stocks[symbol_text] = date_text

        extra = structured_extra(
            run_id=self.run_id,
            symbol=symbol,
            trade_date=trade_date,
            source=source,
            status=status,
            attempt=attempt,
            cache_hit=cache_hit,
        )
        extra["event"] = event
        extra["metrics"] = metrics
        self._logger.log(level, event, extra=extra)

    def finish(self, status: str) -> None:
        duration_seconds = (
            max(0.0, self._monotonic() - self._started_at)
            if self._started_at is not None
            else 0.0
        )
        summary = self.snapshot()
        summary.update(
            {
                "final_status": status,
                "duration_seconds": round(duration_seconds, 3),
            }
        )
        extra = structured_extra(
            run_id=self.run_id,
            source="APPLICATION",
            status=status,
        )
        extra["event"] = "run_summary"
        extra["summary"] = summary
        self._logger.info("run_summary", extra=extra)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "requested_symbols": self._requested_symbols,
                "observed_symbols": len(self._symbols),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "no_data_date_count": len(self._no_data_dates),
                "no_data_dates": sorted(self._no_data_dates),
                "closed_unscheduled_date_count": len(
                    self._closed_unscheduled_dates
                ),
                "closed_unscheduled_dates": sorted(
                    self._closed_unscheduled_dates
                ),
                "missing_tracked_stock_count": len(
                    self._missing_tracked_stocks
                ),
                "missing_tracked_stocks": dict(
                    sorted(self._missing_tracked_stocks.items())
                ),
                "events": dict(sorted(self._events.items())),
                "statuses": dict(sorted(self._statuses.items())),
                "sources": dict(sorted(self._sources.items())),
            }
