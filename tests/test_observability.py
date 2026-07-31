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
    assert summary["summary"]["final_status"] == "SUCCESS"
    assert summary["summary"]["duration_seconds"] == 2.346
