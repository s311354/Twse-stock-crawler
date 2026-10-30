"""Synthetic receipts only; bounded aggregation never certifies fake evidence."""
import json

import pytest

from application.replay_contracts import digest
from examples import compare_les_r20 as comparison
from infrastructure.research_archive import write_once, write_bytes_once


def shards(tmp_path):
    programs = {"synthetic.py": digest("SYNTHETIC_ONLY")}
    # The archive hashes bind actual bytes, not a version label.
    import hashlib
    programs["synthetic.py"] = hashlib.sha256(b"SYNTHETIC_ONLY").hexdigest()
    common = dict(programs=programs, versions={"code": "SYNTHETIC_ONLY"},
        source_bundle_sha256="SYNTHETIC_ONLY", evaluation_cutoff="2030-01-01T00:00:00Z")
    write_once(tmp_path / "job.json", dict(common, workers=2, symbols=["SYNTHETIC0", "SYNTHETIC1"]))
    for index in range(2):
        child = tmp_path / f"shard-{index:02d}"
        symbol = f"SYNTHETIC{index}"
        row = dict(symbol=symbol, signal_date="2029-12-31", state="SOURCE_OR_RECOMPUTATION_GAP",
            error="SYNTHETIC_ONLY_NO_REAL_EVIDENCE")
        write_once(child / "rows/0000.json", row)
        write_once(child / "job.json", dict(common, symbols=[symbol]))
        value = dict(synthetic_only=True, padding="x" * 100_000)
        write_once(child / "comparison.json", value)
        write_once(child / "complete.json", dict(comparison_sha256=digest(value), row_hashes=[digest(row)]))
        write_bytes_once(child / "programs/synthetic.py", b"SYNTHETIC_ONLY")
        write_bytes_once(child / "report_evidence.sqlite3", b"SYNTHETIC_ONLY_NOT_A_REAL_DATABASE")
    return tmp_path


def test_aggregation_loads_only_small_receipts_and_one_row_preserving_gaps(tmp_path, monkeypatch):
    root = shards(tmp_path)
    original = json.loads
    def bounded(value, *args, **kwargs):
        assert len(value) < 10_000  # Never parse a full child comparison document.
        return original(value, *args, **kwargs)
    monkeypatch.setattr(comparison.json, "loads", bounded)
    assert comparison.aggregate_shards(root) == 0
    result = original((root / "comparison-index.json").read_text())
    assert result["summary"]["total"] == 2
    assert result["summary"]["states"] == {"SOURCE_OR_RECOMPUTATION_GAP": 2}
    assert result["summary"]["r20_decisions"] == {}
    assert result["full_differences_retained_in_rows"]
    assert (root / "shard-00/comparison.json").exists()
    assert comparison.aggregate_shards(root) == 0  # Deterministic/idempotent index.


@pytest.mark.parametrize("damage", ["row", "archive", "comparison", "missing"])
def test_damaged_completed_shard_cannot_publish_complete_index(tmp_path, damage):
    root = shards(tmp_path)
    if damage == "missing":
        (root / "shard-00/rows/0000.json").unlink()
    else:
        target = {"row": "rows/0000.json", "archive": "programs/synthetic.py",
                  "comparison": "comparison.json"}[damage]
        path = root / "shard-00" / target
        path.write_bytes(path.read_bytes() + b" ")
        if damage == "row":
            value = json.loads(path.read_text()); value["symbol"] = "TAMPERED_SYNTHETIC"
            path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="R20_SHARD_"):
        comparison.aggregate_shards(root)
    assert not (root / "complete-index.json").exists()
