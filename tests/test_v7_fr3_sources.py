"""SYNTHETIC native-port orchestration only, never real evidence or fit results."""
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from application.replay_contracts import digest
from infrastructure import les_v7_fr3_sources as sources


class Documents:
    def check(self):
        pass

    def read(self, reference):
        return b"SYNTHETIC_TEST_ONLY"

    def externalize(self, value, **kwargs):
        return deepcopy(value)


@pytest.fixture
def ports(tmp_path, monkeypatch):
    seed = tmp_path / "old" / "freezes" / ("a" * 64) / "manifest.json"
    seed.parent.mkdir(parents=True)
    seed.write_bytes(b"SYNTHETIC_OLD_IMMUTABLE_SCOPE")
    parent = dict(universe_record={"source_day": "2026-08-07", "SYNTHETIC": True},
        universe=["3010"], calendar_evidence_id="b" * 64, calendar_sha256="c" * 64,
        market="TWSE", sector="13", history_start="2026-08-01", history_end="2026-10-22",
        batch_size=1, dataset_kind="RESEARCH", reference_evidence_policy=sources.FINAL_REFERENCE,
        chunks=[dict(index=0, symbols=["3010"], evidence_id="a" * 64)])
    original = dict(freeze_id="a" * 64, body=parent)
    calendar = {"SYNTHETIC_TEST_ONLY": True}
    sessions = [f"2026-07-{i:02d}" for i in range(1, 32)] + [f"2026-08-{i:02d}" for i in range(1, 32)]
    days = {d: "OPEN" for d in sessions}
    market = dict(start="2026-07-01", end="2026-07-31", status="COMPLETE", gaps=[],
        calendar_sha256=digest(calendar), documents=[{"date": d} for d in sessions if d < "2026-08-01"])
    actions = dict(start="2026-07-01", end="2026-10-22", SYNTHETIC_TEST_ONLY=True)
    build = dict(seed={"reference_policy": {"version": "SYNTHETIC_FR3"}},
        evaluation_cutoff="2026-10-22T13:30:00+08:00", event_start="2026-07-01",
        event_end="2026-08-01", max_events=4096)
    old_bar = dict(date="20260801", dataset="STOCK_TABLE", data=[["3010", "OLD_UNCHANGED"]])
    payload = dict(symbols=["3010"], bars=[old_bar], calendar={"SYNTHETIC_OLD": True},
        corporate_actions_final={"SYNTHETIC_OLD": True}, market_supplement_final={"SYNTHETIC_OLD": True})
    emitted, added = [], [dict(date="20260701", dataset="STOCK_TABLE", data=[["3010", "NEW"]])]
    def verify(path, root, **kwargs):
        assert kwargs.get("verify_sources") is True
        return deepcopy(original) if Path(path) == seed else json.loads(Path(path).read_text())
    monkeypatch.setattr(sources, "_verify_freeze", verify)
    monkeypatch.setattr(sources, "validate_build_config", lambda *_: None)
    monkeypatch.setattr(sources, "is_fr3_reference", lambda p: p.get("version") == "SYNTHETIC_FR3")
    monkeypatch.setattr(sources, "load_final_calendar", lambda p: (deepcopy(calendar), Documents()))
    monkeypatch.setattr(sources, "load_final_market", lambda p: (deepcopy(market), Documents()))
    monkeypatch.setattr(sources, "load_actions_final", lambda p: deepcopy(actions))
    monkeypatch.setattr(sources, "select_final_calendar", lambda *a, **k: (
        dict(sessions=deepcopy(sessions), days=deepcopy(days)),
        dict(unknown_dates=[d for d, s in days.items() if s == "UNKNOWN"])))
    monkeypatch.setattr(sources, "EvidenceDocuments", lambda *_: Documents())
    monkeypatch.setattr(sources, "_load_bound_scope_shard", lambda *a: (seed, "a" * 64, {}))
    monkeypatch.setattr(sources, "repair_reference_evidence", lambda *a, **k: seed)
    monkeypatch.setattr(sources, "load_evidence", lambda *a, **k: ("b" * 64, {}))
    monkeypatch.setattr(sources, "_repartition_shards", lambda *a: iter([
        (0, ["3010"], deepcopy(payload), [{"evidence_path": str(seed)}], [Documents()])]))
    @contextmanager
    def index(*args):
        yield SimpleNamespace(records=lambda symbols: deepcopy(added))
    monkeypatch.setattr(sources, "_prefix_index", index)
    def retain(body, root, **kwargs):
        emitted.append(deepcopy(body))
        return Path(root) / digest(body) / "evidence.json"
    monkeypatch.setattr(sources, "create_dataset", retain)
    return dict(seed=seed, original=original, calendar=calendar, market=market, actions=actions,
        build=build, days=days, sessions=sessions, emitted=emitted, added=added, payload=payload,
        output=tmp_path / "new", calendar_path=tmp_path / "calendar.json",
        market_path=tmp_path / "market.json", actions_path=tmp_path / "actions" / "id" / "manifest.json")


def run(ports, **options):
    return sources.prepend_fr3_scope(ports["seed"], ports["calendar_path"], ports["market_path"],
        ports["actions_path"], build_config=ports["build"], output_root=options.pop("output", ports["output"]),
        guard_factory=options.pop("guard_factory", lambda: SimpleNamespace(check=lambda: None)), **options)


def test_prepend_keeps_old_values_and_sources_and_publishes_completion_last(ports):
    before = ports["seed"].read_bytes()
    original = deepcopy(ports["original"])
    report = run(ports)
    body = json.loads(Path(report["scope_file"]).read_text())["body"]
    emitted = ports["emitted"][0]
    assert body["history_start"] == "2026-07-01"
    assert body["universe"] == original["body"]["universe"]
    assert body["universe_record"] == original["body"]["universe_record"]
    assert emitted["bars"][0] == ports["payload"]["bars"][0]
    assert emitted["bars"][1:] == ports["added"]
    assert emitted["original_market_supplement_before_fr3_expansion"] == {"SYNTHETIC_OLD": True}
    assert report["prefix_open_sessions"] == 31
    assert report["mutable_market_cache_read"] is report["model_fitted"] is False
    assert ports["seed"].read_bytes() == before and ports["original"] == original
    assert list(ports["output"].glob("*/completion.json"))
    assert run(ports) == report


@pytest.mark.parametrize("damage", ["gap", "partial", "missing_day", "duplicate", "overlap",
    "calendar_binding", "unknown", "actions", "premature"])
def test_incomplete_or_conflicting_prefix_cannot_publish_a_dataset(ports, damage):
    if damage == "gap": ports["market"]["gaps"] = [{"reason": "SYNTHETIC_MISSING"}]
    elif damage == "partial": ports["market"]["status"] = "PARTIAL"
    elif damage == "missing_day": ports["market"]["documents"].pop()
    elif damage == "duplicate": ports["market"]["documents"].append(ports["market"]["documents"][0])
    elif damage == "overlap": ports["market"]["end"] = "2026-08-01"
    elif damage == "calendar_binding": ports["market"]["calendar_sha256"] = "0" * 64
    elif damage == "unknown": ports["days"]["2026-07-02"] = "UNKNOWN"
    elif damage == "actions": ports["actions"]["start"] = "2026-07-02"
    else: ports["build"]["event_end"] = "2026-08-31"
    with pytest.raises(ValueError, match="FR3_PREFIX_"):
        run(ports)
    assert not ports["output"].exists() and not ports["emitted"]


def test_derived_record_cannot_overlap_old_history_even_after_original_validation(ports):
    ports["added"][0]["date"] = "20260801"
    with pytest.raises(ValueError, match="RECORD_OVERLAPS"):
        run(ports)
    assert not list(ports["output"].glob("*/completion.json"))
    assert not list(ports["output"].glob("*/scope/freezes/*/manifest.json"))


def test_corrupt_transport_original_stops_before_output(ports, monkeypatch):
    def corrupt(*_): raise ValueError("DOCUMENT_BODY_HASH_MISMATCH")
    monkeypatch.setattr(sources, "load_final_market", corrupt)
    with pytest.raises(ValueError, match="DOCUMENT_BODY_HASH_MISMATCH"):
        run(ports)
    assert not ports["output"].exists()


def test_old_scope_and_non_fr3_input_cannot_be_rebased(ports):
    with pytest.raises(ValueError, match="PRESERVE_SEED"):
        run(ports, output=ports["seed"].parent.parent.parent / "new")
    ports["build"]["seed"]["reference_policy"]["version"] = "OLD"
    with pytest.raises(ValueError, match="EXPLICIT_FR3"):
        run(ports)


def test_parent_change_before_completion_prevents_scope_publication(ports, monkeypatch):
    original_verify = sources._verify_freeze
    calls = []
    def verify(path, *a, **k):
        result = original_verify(path, *a, **k)
        if Path(path) == ports["seed"]:
            calls.append(path)
            if len(calls) > 1: result["freeze_id"] = "CHANGED_SYNTHETIC_PARENT"
        return result
    monkeypatch.setattr(sources, "_verify_freeze", verify)
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        run(ports)
    assert not list(ports["output"].glob("*/completion.json"))
    assert not list(ports["output"].glob("*/scope/freezes/*/manifest.json"))


def test_runtime_change_during_source_assembly_never_publishes_completion(ports):
    calls = []
    def check():
        calls.append(None)
        if len(calls) == 3:
            raise ValueError("REPLAY_VERSION_CHANGED_DURING_JOB")
    with pytest.raises(ValueError, match="VERSION_CHANGED"):
        run(ports, guard_factory=lambda: SimpleNamespace(check=check))
    assert not list(ports["output"].glob("*/completion.json"))
    assert not list(ports["output"].glob("*/scope/freezes/*/manifest.json"))


def test_calendar_semantics_repair_gets_new_id_without_overwriting_official_bytes(tmp_path, monkeypatch):
    old = tmp_path / "old" / "collections" / ("a" * 64) / "calendar.json"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"SYNTHETIC_IMMUTABLE_OFFICIAL_BYTES")
    calendar = dict(SYNTHETIC_TEST_ONLY=True, documents={})
    def load(path):
        return (deepcopy(calendar) if Path(path) == old else json.loads(Path(path).read_text())["payload"], Documents())
    monkeypatch.setattr(sources, "load_final_calendar", load)
    monkeypatch.setattr(sources, "EvidenceDocuments", lambda *_: Documents())
    result = sources.retain_corrected_fr3_calendar(old, tmp_path / "new")
    value = json.loads(result.read_text())
    assert value["calendar_id"] == digest(value["payload"]) != digest(calendar)
    assert value["payload"]["parent_calendar_id"] == digest(calendar)
    assert value["payload"]["annual_parser_version"] == "OFFICIAL_ANNUAL_CURRENT_SESSION_R2"
    assert value["payload"]["documents"] == calendar["documents"]
    assert old.read_bytes() == b"SYNTHETIC_IMMUTABLE_OFFICIAL_BYTES"
    assert sources.retain_corrected_fr3_calendar(old, tmp_path / "new") == result
    completion = json.loads((result.parent / "completion.json").read_text())
    assert completion["official_source_bytes_changed"] is False


def test_expansion_plan_refuses_unknown_calendar_before_publication(ports, monkeypatch, tmp_path):
    from infrastructure import les_v7_fr3_training as training
    parent = tmp_path / "parent"; parent.mkdir()
    body = dict(build_config={"SYNTHETIC": True}, training_config={"SYNTHETIC": True}, symbols=["3010"])
    request = dict(payload=body, sha256=digest(body))
    (parent / "fr3-training-request.json").write_text(json.dumps(request))
    monkeypatch.setattr(training, "verify_fr3_scope_training", lambda p: dict(request_id=request["sha256"]))
    ports["days"]["2026-07-02"] = "UNKNOWN"
    with pytest.raises(ValueError, match="COMPLETE_VERIFIED_CALENDAR"):
        sources.freeze_fr3_expansion_settings(parent, ports["calendar_path"],
            history_start="2026-07-01", through="2026-10-22", output_root=ports["output"])
    assert not ports["output"].exists()


def test_expansion_plan_never_uses_a_future_collection_cutoff(ports, monkeypatch, tmp_path):
    from infrastructure import les_v7_fr3_training as training
    parent = tmp_path / "parent"; parent.mkdir()
    body = dict(build_config={"SYNTHETIC": True}, training_config={"SYNTHETIC": True}, symbols=["3010"])
    request = dict(payload=body, sha256=digest(body))
    (parent / "fr3-training-request.json").write_text(json.dumps(request))
    monkeypatch.setattr(training, "verify_fr3_scope_training", lambda p: dict(request_id=request["sha256"]))
    monkeypatch.setattr(sources, "utc_now", lambda: "2026-10-21T13:30:00+08:00")
    with pytest.raises(ValueError, match="REAL_CUTOFF_REQUIRED"):
        sources.freeze_fr3_expansion_settings(parent, ports["calendar_path"],
            history_start="2026-07-01", through="2026-10-22", output_root=ports["output"])
    assert not ports["output"].exists()


def test_ephemeral_prefix_index_parses_each_raw_day_once_and_preserves_native_batches(monkeypatch):
    calls = []
    rows = {"2021-01-04": [["3011", "first"], ["3010", "second"], ["OTHER", "not fixed"]],
            "2021-01-05": [["3011", "third"]]}
    def parse(raw, day):
        calls.append((raw, day))
        return deepcopy(rows[day]), 123.45
    monkeypatch.setattr(sources, "verified_document", lambda d, **kw: d["SYNTHETIC_BYTES"])
    monkeypatch.setattr(sources, "parse_final_market", parse)
    market = dict(documents=[dict(date=d, document={"SYNTHETIC_BYTES": d.encode()}) for d in rows])
    with sources._prefix_index(market, Documents(), ["3010", "3011"]) as index:
        database = Path(index.connection.execute("PRAGMA database_list").fetchone()[2])
        single = index.records(["3010"])
        assert single == [
            dict(date="20210104", dataset="STOCK_TABLE", market_type="ALLBUT0999", source="TWSE", data=[["3010", "second"]]),
            dict(date="20210104", dataset="TAIEX", market_type="IND", source="TWSE_TAIEX", data="123.45"),
            dict(date="20210105", dataset="STOCK_TABLE", market_type="ALLBUT0999", source="TWSE", data=[]),
            dict(date="20210105", dataset="TAIEX", market_type="IND", source="TWSE_TAIEX", data="123.45")]
        assert index.records(["3010", "3011"])[0]["data"] == rows["2021-01-04"][:2]
        assert index.records(["3011"])[2]["data"] == [["3011", "third"]]
        assert database.exists() and len(calls) == 2
        single[0]["data"][0][1] = "mutated"
        assert index.records(["3010"])[0]["data"][0][1] == "second"
    assert not database.exists()  # No reusable trust cache or live project SQLite.


@pytest.mark.parametrize("damage", ["duplicate", "row_limit", "source_hash"])
def test_prefix_index_boundaries_do_not_turn_invalid_input_into_market_rows(monkeypatch, damage):
    rows = [["3010", "SYNTHETIC"]]
    if damage == "duplicate": rows *= 2
    if damage == "row_limit": monkeypatch.setattr(sources, "MAX_PREFIX_SELECTED_ROWS", 0)
    def document(*a, **k):
        if damage == "source_hash": raise ValueError("DOCUMENT_BODY_HASH_MISMATCH")
        return b"SYNTHETIC_TEST_ONLY"
    monkeypatch.setattr(sources, "verified_document", document)
    monkeypatch.setattr(sources, "parse_final_market", lambda *a: (rows, 100.))
    with pytest.raises(ValueError):
        with sources._prefix_index(dict(documents=[dict(date="2021-01-04", document={})]), Documents(), ["3010"]):
            pytest.fail("SYNTHETIC invalid source was exposed")
