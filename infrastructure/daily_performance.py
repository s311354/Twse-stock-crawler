"""Small durable performance receipt independent of logging verbosity."""
import datetime as dt
import json
import os
from pathlib import Path
import tempfile


def retain_daily_performance(request, context, metrics, *, run_id):
    repository = getattr(getattr(context, "resource", None), "csv_repository", None)
    directory = getattr(repository, "output_dir", None)
    if directory is None:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema="daily-performance/v1", run_id=run_id,
                   recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                   report_prefix=request.report_file_prefix,
                   signal_session=str(getattr(context, "market_date", "")),
                   timing_semantics="NESTED_STAGES_NOT_ADDITIVE_RECEIPT_WRITE_EXCLUDED", **metrics)
    target = directory / (request.report_file_prefix + "_performance.json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=directory, encoding="utf-8", delete=False) as stream:
            temporary = stream.name
            json.dump(receipt, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return target
