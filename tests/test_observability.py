import io
import json
import logging

from application.contracts import StockAnalysisRequest
from infrastructure.observability import StructuredRunObserver
from infrastructure.observability import configure_structured_logging


def request() -> StockAnalysisRequest:
    return StockAnalysisRequest(
        stocklist=["2330", "2317"],
        stocktype="ELEC",
        output_file_names="shirong",
        endbacktrack=3,
    )


def test_structured_observer_emits_required_fields_and_summary() -> None:
    output = io.StringIO()
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    try:
        configure_structured_logging("INFO", stream=output)
        times = iter([10.0, 12.3456])
        observer = StructuredRunObserver(
            "run-observer-test",
            monotonic=lambda: next(times),
        )

        observer.start(request())
        observer.record(
            "fundamental_refresh_completed",
            symbol="2330",
            trade_date="2026-07-20",
            source="MOPS_CACHE",
            status="SUCCESS",
            attempt=1,
            cache_hit=True,
        )
        observer.record(
            "market_data_no_data_dates_collected",
            level=logging.DEBUG,
            source="TWSE",
            status="NO_DATA",
            no_data_dates=["2025-10-06", "2025-10-07"],
        )
        observer.record(
            "market_data_missing_tracked_stocks_collected",
            level=logging.DEBUG,
            source="TWSE",
            status="TRACKED_STOCK_MISSING",
            missing_tracked_stocks={
                "2327": "2025-08-14",
                "3321": "2025-09-08",
            },
        )
        observer.record(
            "market_closed_unscheduled_dates_collected",
            level=logging.DEBUG,
            source="TWSE",
            status="CLOSED_UNSCHEDULED",
            closed_unscheduled_dates=["2026-07-10"],
        )
        logging.getLogger("legacy").warning("legacy warning")
        observer.finish("SUCCESS")
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)

    records = [
        json.loads(line)
        for line in output.getvalue().splitlines()
        if line.strip()
    ]
    fundamental = next(
        record
        for record in records
        if record.get("event") == "fundamental_refresh_completed"
    )
    legacy = next(
        record for record in records if record["message"] == "legacy warning"
    )
    summary = next(
        record for record in records if record.get("event") == "run_summary"
    )

    assert fundamental["run_id"] == "run-observer-test"
    assert fundamental["symbol"] == "2330"
    assert fundamental["trade_date"] == "2026-07-20"
    assert fundamental["source"] == "MOPS_CACHE"
    assert fundamental["status"] == "SUCCESS"
    assert fundamental["attempt"] == 1
    assert fundamental["cache_hit"] is True
    assert legacy["run_id"] == "run-observer-test"
    assert {
        "run_id",
        "symbol",
        "trade_date",
        "source",
        "status",
        "attempt",
        "cache_hit",
    }.issubset(legacy)
    assert summary["summary"]["requested_symbols"] == 2
    assert summary["summary"]["observed_symbols"] == 1
    assert summary["summary"]["cache_hits"] == 1
    assert summary["summary"]["cache_misses"] == 0
    assert summary["summary"]["no_data_date_count"] == 2
    assert summary["summary"]["no_data_dates"] == [
        "2025-10-06",
        "2025-10-07",
    ]
    assert summary["summary"]["closed_unscheduled_date_count"] == 1
    assert summary["summary"]["closed_unscheduled_dates"] == [
        "2026-07-10"
    ]
    assert summary["summary"]["missing_tracked_stock_count"] == 2
    assert summary["summary"]["missing_tracked_stocks"] == {
        "2327": "2025-08-14",
        "3321": "2025-09-08",
    }
    assert summary["summary"]["final_status"] == "SUCCESS"
    assert summary["summary"]["duration_seconds"] == 2.346


def test_release_logging_filters_diagnostics_but_keeps_operational_summary() -> None:
    output = io.StringIO()
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    try:
        configure_structured_logging("INFO", stream=output)
        logging.debug("per-symbol diagnostic")
        logging.info("operational summary")
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)

    records = [
        json.loads(line)
        for line in output.getvalue().splitlines()
        if line.strip()
    ]
    assert [record["message"] for record in records] == ["operational summary"]


def test_debug_logging_emits_diagnostics() -> None:
    output = io.StringIO()
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    try:
        configure_structured_logging("DEBUG", stream=output)
        logging.debug("per-symbol diagnostic")
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)

    records = [
        json.loads(line)
        for line in output.getvalue().splitlines()
        if line.strip()
    ]
    assert [record["message"] for record in records] == ["per-symbol diagnostic"]
