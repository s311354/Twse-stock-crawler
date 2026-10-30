"""Synthetic, offline backfill regressions; never historical certification."""
from copy import deepcopy
import datetime as dt
import json
import sqlite3

import pytest

from application.replay_contracts import canonical_json, digest
from infrastructure.les_v7_training_backfill import backfill_training_evidence, merge_market_backfill
from infrastructure.les_v7_training_source import FINAL_CLOSE, load_training_source
from tests.test_v7_training_source import fixture as source_fixture


START = dt.date(2026, 7, 30)


def inputs():
    parent = source_fixture()
    parent.update(
        filings={"3014": [{"content_sha256": "SYNTHETIC_ORIGINAL", "available_at": "2026-07-01T00:00:00+08:00"}]},
        raw_documents={"SYNTHETIC_ORIGINAL": {"body_utf8": "SYNTHETIC TEST ONLY original financial bytes"}},
        batches={"SYNTHETIC_BATCH": {"available_at": "2026-07-01T00:00:00+08:00"}},
        restrictions=[{"symbol": "3014", "evidence": {"status": "UNKNOWN"}}],
        restriction_ledger={"status": "MISSING_RESTRICTION_EVIDENCE"},
        kelly_calibrations=[{"status": "UNKNOWN", "reason": "SYNTHETIC_NO_CALIBRATION"}],
        preparation_gaps=["SYNTHETIC_EXISTING_GAP"],
    )
    supplement = deepcopy(parent)
    supplement["start"] = START.isoformat()
    supplement["bars"] = deepcopy(parent["bars"])
    for record in supplement["bars"]:
        record["date"] = "20260730"
    supplement["calendar"]["days"]["2026-07-30"] = "OPEN"
    # A market-only repair cannot silently adopt fresh finance or safety data.
    supplement["filings"] = {"3014": [{"content_sha256": "SYNTHETIC_NEW_NOT_REQUESTED"}]}
    supplement["raw_documents"] = {"SYNTHETIC_NEW_NOT_REQUESTED": {"body_utf8": "SYNTHETIC other bytes"}}
    supplement["restrictions"] = [{"symbol": "3014", "evidence": {"status": "PASS"}}]
    supplement["kelly_calibrations"] = [{"status": "PASS", "win_probability": .99}]
    return parent, supplement


def merge(parent, supplement, *, start=START):
    return merge_market_backfill(digest(parent), parent, supplement, history_start=start)


def test_backfill_is_deterministic_detached_and_preserves_original_sources():
    parent, supplement = inputs()
    before_parent, before_supplement = deepcopy(parent), deepcopy(supplement)
    result, report = merge(parent, supplement)
    assert merge(parent, supplement) == (result, report)
    assert parent == before_parent and supplement == before_supplement
    assert result["start"] == START.isoformat() and result["end"] == parent["end"]
    for key in ("filings", "raw_documents", "batches", "restrictions", "restriction_ledger", "kelly_calibrations"):
        assert result[key] == parent[key]
    assert result["calendar"]["documents"] == parent["calendar"]["documents"]
    assert result["calendar"]["announcements"] == parent["calendar"]["announcements"]
    result["filings"]["3014"][0]["content_sha256"] = "mutated_output"
    result["bars"][0]["data"] = "mutated_output"
    assert parent == before_parent and supplement == before_supplement


def test_exact_records_deduplicate_without_collapsing_distinct_source_headers():
    parent, supplement = inputs()
    supplement["bars"] += deepcopy(parent["bars"])
    supplement["bars"] += deepcopy(supplement["bars"])
    other_source = deepcopy(parent["bars"][0])
    other_source["market_type"] = "ALLBUT0999"
    supplement["bars"].append(other_source)
    result, _ = merge(parent, supplement)
    record_ids = [digest(row) for row in result["bars"]]
    assert len(record_ids) == len(set(record_ids)) == 5
    assert digest(other_source) in record_ids
    assert all(digest(row) in record_ids for row in parent["bars"])


def test_conflicting_prices_are_retained_and_adapter_excludes_them(tmp_path):
    parent, supplement = inputs()
    conflict = deepcopy(parent["bars"][0])
    conflict["data"][0][8] = "100.00"
    supplement["bars"].append(conflict)
    result, _ = merge(parent, supplement)
    assert digest(parent["bars"][0]) in {digest(row) for row in result["bars"]}
    assert digest(conflict) in {digest(row) for row in result["bars"]}
    path = tmp_path / "synthetic-evidence.json"
    path.write_text(canonical_json({"evidence_id": digest(result), "payload": result}))
    source = load_training_source(path, evaluation_cutoff="2026-08-05T13:30:00+08:00", availability_policy=FINAL_CLOSE)
    assert source["payload"]["diagnostics"]["reason_counts"]["CONFLICTING_MARKET_ROWS"] == 1
    assert not any(row["Date"] == "2026-08-03" for history in source["payload"]["histories"] for row in history["bars"])


def test_new_calendar_dates_remain_unknown_despite_quotes_or_supplement_open():
    parent, supplement = inputs()
    parent["calendar"]["days"]["2026-08-04"] = "UNKNOWN"
    result, _ = merge(parent, supplement)
    assert result["calendar"]["days"]["2026-08-04"] == "UNKNOWN"
    assert result["calendar"]["days"]["2026-07-30"] == "UNKNOWN"
    assert result["calendar"]["days"]["2026-07-31"] == "UNKNOWN"
    assert result["calendar"]["days"]["2026-08-01"] == "UNKNOWN"
    assert result["calendar"]["days"]["2026-08-03"] == "OPEN"


@pytest.mark.parametrize("key,value", [
    ("schema", "replay-evidence/v1"),
    ("market", "TPEX"),
    ("sector", "24"),
    ("symbols", ["3014"]),
    ("price_unit", "TWD_1000"),
    ("volume_unit", "LOTS"),
    ("amount_unit", "TWD_1000"),
    ("market_policy", "SYNTHETIC_OTHER_VINTAGE"),
])
def test_source_identity_mismatch_rejected(key, value):
    parent, supplement = inputs()
    supplement[key] = value
    with pytest.raises(ValueError):
        merge(parent, supplement)


def test_parent_evidence_hash_mismatch_rejected():
    parent, supplement = inputs()
    with pytest.raises(ValueError):
        merge_market_backfill("0" * 64, parent, supplement, history_start=START)


def test_history_window_cannot_move_start_forward():
    parent, supplement = inputs()
    with pytest.raises(ValueError):
        merge(parent, supplement, start=dt.date(2026, 8, 4))


@pytest.mark.parametrize("outside_date", ["20260729", "20260806"])
def test_source_bars_outside_exact_pinned_window_rejected(outside_date):
    parent, supplement = inputs()
    supplement["bars"][0]["date"] = outside_date
    with pytest.raises(ValueError, match="BACKFILL_BAR_OUTSIDE_PINNED_RANGE"):
        merge(parent, supplement)


def test_source_cannot_extend_pinned_end_date():
    parent, supplement = inputs()
    supplement["end"] = "2026-08-06"
    with pytest.raises(ValueError, match="BACKFILL_BOUNDED_PAST_RANGE_REQUIRED"):
        merge(parent, supplement)


@pytest.mark.parametrize("change,reason", [
    ({"source": "TPEX"}, "BACKFILL_STOCK_SOURCE_OR_UNIVERSE_MISMATCH"),
    ({"dataset": "SYNTHETIC_OTHER_DATASET"}, "BACKFILL_MARKET_DATASET_UNSUPPORTED"),
    ({"data": [["2330"]]}, "BACKFILL_STOCK_SOURCE_OR_UNIVERSE_MISMATCH"),
])
def test_stock_source_or_scope_mismatch_rejected(change, reason):
    parent, supplement = inputs()
    supplement["bars"][0].update(change)
    with pytest.raises(ValueError, match=reason):
        merge(parent, supplement)


def test_benchmark_source_mismatch_rejected():
    parent, supplement = inputs()
    supplement["bars"][1]["source"] = "SYNTHETIC_OTHER_BENCHMARK"
    with pytest.raises(ValueError, match="BACKFILL_BENCHMARK_SOURCE_MISMATCH"):
        merge(parent, supplement)


def test_action_marker_is_source_bound_pending_not_no_action():
    parent, supplement = inputs()
    marked = supplement["bars"][0]["data"][0]
    marked[9] = "<p>X</p>"
    result, report = merge(parent, supplement)
    markers = report["pending_action_markers"]
    assert len(markers) == 1
    assert markers[0]["symbol"] == "3014"
    assert markers[0]["date"] == "20260730"
    assert markers[0]["source_row_sha256"] == digest(marked)
    assert "VERIFIED_NO_ACTIONS" not in canonical_json(report)
    assert marked in next(row["data"] for row in result["bars"] if row["date"] == "20260730" and row["dataset"] == "STOCK_TABLE")


def test_non_raw_or_undeclared_parent_vintage_rejected():
    parent, supplement = inputs()
    parent["market_policy"] = supplement["market_policy"] = "SYNTHETIC_UNKNOWN_VINTAGE"
    with pytest.raises(ValueError):
        merge(parent, supplement)


def test_public_backfill_reads_local_cache_and_publishes_new_immutable_dataset(tmp_path, monkeypatch):
    parent = source_fixture()
    evidence = tmp_path / "original.json"
    original_bytes = canonical_json({"evidence_id": digest(parent), "payload": parent}).encode()
    evidence.write_bytes(original_bytes)
    cache = tmp_path / "synthetic-market.sqlite3"
    with sqlite3.connect(cache) as connection:
        connection.execute("CREATE TABLE market_history_cache (trade_date TEXT,dataset TEXT,market_type TEXT,schema_version INTEGER,payload_json TEXT,cached_at_epoch INTEGER)")
        for record in parent["bars"]:
            copied = deepcopy(record)
            copied["date"] = "20260730"
            connection.execute("INSERT INTO market_history_cache VALUES (?,?,?,?,?,?)", (
                copied["date"], copied["dataset"], copied["market_type"], 1, json.dumps(copied["data"]), 1))
    cache_bytes = cache.read_bytes()
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("network forbidden"))
    store = tmp_path / "synthetic-datasets"
    with pytest.raises(ValueError, match="SYNTHETIC_BACKFILL_REQUIRES_EXPLICIT_OPT_IN"):
        backfill_training_evidence(evidence, market_cache=cache, history_start=START, store=store)
    assert not store.exists()
    first = backfill_training_evidence(evidence, market_cache=cache, history_start=START, store=store, allow_synthetic=True)
    second = backfill_training_evidence(evidence, market_cache=cache, history_start=START, store=store, allow_synthetic=True)
    assert first == second
    assert first["evidence_id"] != digest(parent)
    assert first["parent_evidence_id"] == digest(parent)
    assert evidence.read_bytes() == original_bytes and cache.read_bytes() == cache_bytes
    stored = json.loads((store / first["evidence_id"] / "evidence.json").read_text())
    assert stored["evidence_id"] == digest(stored["payload"])
    assert stored["payload"]["purpose"] == "SYNTHETIC_TEST_ONLY"
    assert stored["payload"]["start"] == START.isoformat()
    assert stored["payload"]["calendar"]["days"]["2026-07-30"] == "UNKNOWN"
    assert stored["payload"]["filings"] == parent["filings"]
    manifest = json.loads((store / first["evidence_id"] / "manifest.json").read_text())
    assert manifest["parent_dataset_id"] == digest(parent)
    report = json.loads((store / first["evidence_id"] / "market_backfill.json").read_text())
    assert report["historical_information_set_certified"] is False
    assert report["original_observation_receipts_added"] is False


def bounded_public_inputs(tmp_path, *, stock_rows=None):
    """Tiny SYNTHETIC cache with a real SQLite reader; no production files."""
    parent = source_fixture()
    evidence = tmp_path / "synthetic-original.json"
    evidence.write_text(canonical_json({"evidence_id": digest(parent), "payload": parent}))
    cache = tmp_path / "synthetic-bounded-cache.sqlite3"
    with sqlite3.connect(cache) as connection:
        connection.execute("CREATE TABLE market_history_cache (trade_date TEXT,dataset TEXT,market_type TEXT,schema_version INTEGER,payload_json TEXT,cached_at_epoch INTEGER,PRIMARY KEY(trade_date,dataset,market_type,schema_version))")
        for original in parent["bars"]:
            data = stock_rows if original["dataset"] == "STOCK_TABLE" and stock_rows is not None else original["data"]
            connection.execute("INSERT INTO market_history_cache VALUES (?,?,?,?,?,?)", (
                "20260730", original["dataset"], original["market_type"], 1, json.dumps(data), 1))
    return evidence, cache, tmp_path / "unpublished-synthetic-store", parent


def test_cache_day_byte_limit_rejects_before_any_publication(tmp_path, monkeypatch):
    import infrastructure.les_v7_training_backfill as module
    evidence, cache, store, _ = bounded_public_inputs(tmp_path)
    monkeypatch.setattr(module, "MAX_CACHE_DAY_BYTES", 10)
    with pytest.raises(ValueError, match="BACKFILL_CACHE_DAY_OR_COUNT_LIMIT"):
        backfill_training_evidence(evidence, market_cache=cache, history_start=START, store=store, allow_synthetic=True)
    assert not store.exists()


def test_selected_row_byte_limit_rejects_before_any_publication(tmp_path, monkeypatch):
    import infrastructure.les_v7_training_backfill as module
    row = deepcopy(source_fixture()["bars"][0]["data"][0])
    row[1] = "SYNTHETIC_OVERSIZED_SELECTED_NAME" * 30
    evidence, cache, store, _ = bounded_public_inputs(tmp_path, stock_rows=[row])
    monkeypatch.setattr(module, "MAX_CACHE_ROW_BYTES", 256)
    with pytest.raises(ValueError, match="BACKFILL_SELECTED_ROW_SIZE_OR_COUNT_LIMIT"):
        backfill_training_evidence(evidence, market_cache=cache, history_start=START, store=store, allow_synthetic=True)
    assert not store.exists()


def test_selected_market_aggregate_byte_limit_rejects_during_cache_read(tmp_path, monkeypatch):
    import infrastructure.les_v7_training_backfill as module
    _, cache, store, parent = bounded_public_inputs(tmp_path)
    monkeypatch.setattr(module, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="BACKFILL_SELECTED_MARKET_SIZE_LIMIT"):
        module._cached_market_records(cache, parent, START)
    assert not store.exists()


def test_complete_output_envelope_limit_rejects_before_any_publication(tmp_path, monkeypatch):
    import infrastructure.les_v7_training_backfill as module
    evidence, cache, store, _ = bounded_public_inputs(tmp_path)
    # Permit the exact original envelope and small selected cache records, but
    # not the larger repaired envelope (including provenance and new dates).
    monkeypatch.setattr(module, "MAX_BYTES", evidence.stat().st_size)
    with pytest.raises(ValueError, match="BACKFILL_EVIDENCE_OUTPUT_SIZE_LIMIT"):
        backfill_training_evidence(evidence, market_cache=cache, history_start=START, store=store, allow_synthetic=True)
    assert not store.exists()


def test_large_unrelated_stock_rows_never_expand_in_python_or_breadth(tmp_path, monkeypatch):
    import infrastructure.les_v7_training_backfill as module
    import infrastructure.replay_evidence as replay_evidence
    selected = deepcopy(source_fixture()["bars"][0]["data"][0])
    unrelated = deepcopy(selected)
    unrelated[0], unrelated[1] = "2330", "SYNTHETIC_UNRELATED_MUST_STAY_IN_SQLITE" * 300
    evidence, cache, store, _ = bounded_public_inputs(tmp_path, stock_rows=[selected, unrelated])
    original_loads = json.loads

    def assert_selected_parse_only(value, *args, **kwargs):
        text = value.decode() if isinstance(value, bytes) else value
        assert "SYNTHETIC_UNRELATED_MUST_STAY_IN_SQLITE" not in text
        return original_loads(value, *args, **kwargs)

    def no_general_collector(*args, **kwargs):
        pytest.fail("general collector or sector breadth expansion forbidden")

    monkeypatch.setattr(module.json, "loads", assert_selected_parse_only)
    monkeypatch.setattr(replay_evidence, "collect_evidence", no_general_collector)
    monkeypatch.setattr(replay_evidence, "derive_twse_market_breadth", no_general_collector)
    monkeypatch.setattr(module, "MAX_CACHE_ROW_BYTES", 256)
    result = backfill_training_evidence(evidence, market_cache=cache, history_start=START, store=store, allow_synthetic=True)
    payload = json.loads((store / result["evidence_id"] / "evidence.json").read_text())["payload"]
    repaired_rows = [row for record in payload["bars"] if record["dataset"] == "STOCK_TABLE" for row in record["data"]]
    assert all(row[0] in {"3014", "6834"} for row in repaired_rows)
    assert any(record["date"] == "20260730" for record in payload["bars"])
