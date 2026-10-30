"""SYNTHETIC ONLY: paired research contracts, not official coverage or trades."""
from copy import deepcopy
from dataclasses import dataclass
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from application.replay_contracts import digest
from domain.les_v7.contracts import identity
from domain.les_v7.report import validate_report
from examples import compare_les_r24 as module
from tests.test_v7_r14_comparison import fixture

CUTOFF = "2026-10-28T14:00:00Z"


def test_new_quality_proof_path_keeps_exact_values_and_selects_at_saved_source_cutoff():
    saved, filings, _ = fixture()
    before = deepcopy(filings)
    loader = Mock(return_value={"financial_audits_r7": {}})
    new, receipt = module.enrich_filings(saved, filings, companion_loader=loader, cutoff=CUTOFF)
    assert new == filings == before
    assert receipt["original_values_and_times_unchanged"]
    assert receipt["added_audit_proof_count"] == 0  # Absent source is not PASS.
    assert loader.call_args.kwargs["as_of"] == saved["diagnostics"]["native_evidence"]["source_packet"]["cutoff"]
    assert loader.call_args.kwargs["audit_repair_r7"] is True


def test_complete_native_recalculation_preserves_account_raw_values_and_unknowns():
    saved, filings, _ = fixture()
    before = deepcopy(saved)
    old, new, row = module.compare_report(saved, filings, filings, cutoff=CUTOFF)
    assert saved == before
    for report in (old, new):
        validate_report(report)
        assert not report["pae_authorization"]["live_order_authorized"]
        assert report["diagnostics"]["evidence_binding"]["financials_hash"] == identity(filings)
        assert report["diagnostics"]["evidence_binding"]["bars_hash"] == saved["diagnostics"]["evidence_binding"]["bars_hash"]
    assert old["pae_authorization"]["account"] == new["pae_authorization"]["account"]
    assert row["same_bars_account_source_packet"] and row["business"] and row["complete"]
    assert new["diagnostics"]["quality"]["financial_safety"]["state"] == "UNKNOWN"
    assert new["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    # Old current observation must not be dated forward merely to get a PASS.
    gates = {g["gate_id"]: g["state"] for g in new["diagnostics"]["gate_receipts"]}
    assert gates["TRADING_STATUS"] == "UNKNOWN"


@pytest.mark.parametrize("damage", ["old_hash", "new_value", "future", "account"])
def test_mismatched_sources_and_unverified_account_are_not_repaired_by_policy(damage):
    saved, filings, _ = fixture()
    changed = deepcopy(filings)
    if damage == "old_hash":
        with pytest.raises(ValueError, match="SAVED_FINANCIAL_VECTOR_HASH_REQUIRED"):
            module.compare_report(saved, filings[:-1], filings, cutoff=CUTOFF)
    elif damage == "new_value":
        changed[-1]["cfo"] = 1e99
        with pytest.raises(ValueError, match="ORIGINAL_FINANCIAL_OPERAND_CHANGED"):
            module.compare_report(saved, filings, changed, cutoff=CUTOFF)
    elif damage == "future":
        with pytest.raises(ValueError, match="BEFORE_RELEASE"):
            module.compare_report(saved, filings, changed, cutoff="2026-10-27T00:00:00Z")
    else:
        saved.pop("pae_authorization")
        with pytest.raises(ValueError, match="EXPLICIT_SAVED_ACCOUNT_REQUIRED"):
            module.compare_report(saved, filings, filings, cutoff=CUTOFF)


@pytest.mark.parametrize("damage", [None, "hash", "future", "wrong_field", "raw_hash", "coherent_projection", "path_escape"])
def test_separate_current_capture_is_bound_without_backdating(tmp_path, damage):
    from domain.les_v7.contracts import timestamp
    from examples.capture_les_repair_sources import capture
    transport = Mock()
    transport.request_bytes.return_value = SimpleNamespace(content=b"[]", status_code=200, attempts=1)
    stamps = iter(timestamp(f"2026-10-28T13:50:0{i}Z") for i in range(5))
    value = capture(["6672"], tmp_path / "capture", transport=transport, clock=lambda: next(stamps))
    body = {k: v for k, v in value.items() if k != "content_sha256"}
    if damage == "future": body["by_symbol"]["6672"]["restriction_evidence"]["available_at"] = "2027-01-01T00:00:00Z"
    if damage == "wrong_field": body["by_symbol"]["6672"]["quality"] = {"status": "PASS"}
    if damage == "raw_hash":
        body["metadata"]["documents"][0]["content_sha256"] = "a" * 64
    if damage == "coherent_projection": body["by_symbol"]["6672"]["restriction_evidence"]["restricted"] = True
    if damage == "path_escape": body["metadata"]["documents"][0]["path"] = "../outside.json"
    value = dict(body, content_sha256=digest(body))
    if damage == "hash": value["content_sha256"] = "a" * 64
    path = tmp_path / "capture" / "test-overrides.json"
    path.write_text(json.dumps(value))
    if damage is None:
        assert module.load_overrides(path, cutoff=CUTOFF, symbols=["6672"]) == value
    else:
        with pytest.raises(ValueError):
            module.load_overrides(path, cutoff=CUTOFF, symbols=["6672"])


def test_scope_gaps_are_retained_for_both_pending_issuers_not_quality_filtered(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for i, symbol in enumerate(("6672", "6789")):
        body = {"symbol": symbol, "missing_report": "SYNTHETIC_ONLY"}
        path = source / "shard-00" / "rows" / f"{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body))
        rows.append(dict(symbol=symbol, row_path=str(path.relative_to(source)), row_sha256=digest(body)))
    index = {"records": rows}
    (source / "comparison-index.json").write_text(json.dumps(index))
    (source / "complete-index.json").write_text(json.dumps({"index_sha256": digest(index)}))

    @dataclass
    class Versions:
        source: str = "SYNTHETIC_ONLY"

    guard = Mock(versions=Versions())
    guard.retain.return_value = {}
    monkeypatch.setattr(module, "ResearchVersionGuard", lambda: guard)
    restorer = Mock()
    monkeypatch.setattr(module.sources, "HashMatchedFilings", lambda *a: restorer)
    out = tmp_path / "result"
    result = module.main(["--comparison-root", str(source), "--fundamental-db", str(tmp_path / "unused.sqlite3"),
        "--output-dir", str(out), "--evaluation-cutoff", CUTOFF])
    assert result == 2
    summary = json.loads((out / "complete-index.json").read_text())["summary"]
    assert summary["total"] == 2
    assert summary["states"] == {"SOURCE_OR_RECOMPUTATION_GAP": 2}
    assert summary["r24"]["decisions"] == {}
    assert [json.loads(p.read_text())["symbol"] for p in sorted((out / "rows").glob("*.json"))] == ["6672", "6789"]
    with pytest.raises(FileExistsError):
        module.main(["--comparison-root", str(source), "--fundamental-db", str(tmp_path / "unused.sqlite3"),
            "--output-dir", str(out), "--evaluation-cutoff", CUTOFF])


@pytest.mark.parametrize("count,index", [(0, 0), (17, 0), (2, 2), (3, 0)])
def test_shard_selection_is_bounded_and_never_drops_unassigned_stocks(count, index):
    with pytest.raises(ValueError, match="SHARD_RANGE_INVALID"):
        module.select_shard(["6672", "6789"], index=index, count=count)


def test_shard_partition_is_fixed_order_disjoint_and_complete():
    symbols = [str(i) for i in range(436)]
    shards = [module.select_shard(symbols, index=i, count=4) for i in range(4)]
    assert all(len(shard) == 109 for shard in shards)
    assert len({s for shard in shards for s in shard}) == 436
    assert all(shards[i] == symbols[i::4] for i in range(4))


@pytest.mark.parametrize("damage", [None, "cutoff", "account_policy", "row_hash", "store_hash", "program_hash"])
def test_deterministic_merge_checks_partition_identities_and_all_retained_hashes(tmp_path, monkeypatch, damage):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    entries = []
    for i, symbol in enumerate(("6672", "6789")):
        value = dict(symbol=symbol, synthetic_only=True)
        path = source / "shard-00/rows" / f"{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        entries.append(dict(symbol=symbol, row_path=str(path.relative_to(source)), row_sha256=digest(value)))
    index = dict(records=entries)
    (source / "comparison-index.json").write_text(json.dumps(index))
    (source / "complete-index.json").write_text(json.dumps(dict(index_sha256=digest(index))))

    @dataclass
    class Versions:
        source: str = "SYNTHETIC_ONLY"

    guard = Mock(versions=Versions())
    guard.retain.return_value = {}
    monkeypatch.setattr(module, "ResearchVersionGuard", lambda: guard)
    monkeypatch.setattr(module.sources, "HashMatchedFilings", lambda *a: Mock())
    for i in range(2):
        assert module.main(["--comparison-root", str(source), "--fundamental-db", str(tmp_path / "unused"),
            "--output-dir", str(output / f"shard-{i:02d}"), "--evaluation-cutoff", CUTOFF,
            "--shard-count", "2", "--shard-index", str(i)]) == 2
    child = output / "shard-01"
    if damage in {"cutoff", "account_policy"}:
        job = json.loads((child / "job.json").read_text())
        job[damage] = "SYNTHETIC_CONFLICT"
        (child / "job.json").write_text(json.dumps(job))
    elif damage == "row_hash":
        (child / "rows/0000.json").write_text('{"symbol":"6672"}')
    elif damage == "store_hash":
        with (child / "report_evidence.sqlite3").open("ab") as stream:
            stream.write(b"SYNTHETIC_CORRUPTION")
    elif damage == "program_hash":
        (child / "programs/compare_les_r24.py").write_text("SYNTHETIC_CHANGED_PROGRAM")
    if damage is not None:
        with pytest.raises(ValueError, match="SHARD_"):
            module.merge_shards(output, comparison_root=source)
        assert not (output / "complete-index.json").exists()
    else:
        assert module.main(["--comparison-root", str(source), "--merge-shards", str(output)]) == 2
        result = json.loads((output / "comparison-index.json").read_text())
        assert result["full_original_universe_processed"]
        assert [r["symbol"] for r in result["records"]] == ["6672", "6789"]
        assert result["summary"]["states"] == {"SOURCE_OR_RECOMPUTATION_GAP": 2}
        before = (output / "complete-index.json").read_bytes()
        assert module.merge_shards(output, comparison_root=source) == 2
        assert (output / "complete-index.json").read_bytes() == before
