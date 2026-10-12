"""Research-only parse/index persistence; exact parity, not timing assertions."""
import base64
from copy import deepcopy
from dataclasses import replace
import datetime as dt
import hashlib
import json
import sqlite3

import pytest

from application.replay_contracts import digest, aware_timestamp
from application.research_evidence import filing_binding
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.replay_evidence import load_evidence
from infrastructure.replay_runtime import build_replay_engine, current_versions
from infrastructure.research_archive import create_dataset
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.research_job import ResearchJob
from infrastructure.storage.research_cache import SQLiteFilingHistory, epoch_microseconds
from tests.test_research_archive import raw_fixture, add_raw, proof
from tests.test_strict_replay import request
from interface.research import main


def history():
    payload = raw_fixture()
    first = add_raw(payload)
    second = deepcopy(first)
    original = payload["raw_documents"][first["content_sha256"]]
    content = base64.b64decode(original["body_base64"]).replace(b"5,000", b"6,000")
    sha = hashlib.sha256(content).hexdigest()
    second.update(content_sha256=sha, revision_number=2, source_version="test-revision-two")
    second["publication_proof"] = proof(filing_binding("2330", second), publication="2026-09-11T13:30:00+08:00", precision="TIMESTAMP")
    second["publication_proof"]["original_published_at"] = "2026-08-01T00:00:00+08:00"
    payload["raw_documents"][sha] = dict(original, sha256=sha, body_base64=base64.b64encode(content).decode())
    payload["filings"]["2330"].append(second)
    return payload


def adapter(tmp_path, payload, *, version="current", parser=parse_ixbrl):
    return SQLiteFilingHistory(tmp_path / "parse.sqlite3", evidence_id=digest(payload),
                              version=version, parse_filing=parser)


def test_restart_reuses_parses_and_queries_cutoff_without_future_facts(tmp_path):
    payload = history()
    calls = []
    def parse(body):
        calls.append(body)
        return parse_ixbrl(body)
    cutoff = aware_timestamp("2026-09-11T13:29:59.999999+08:00")
    cache = adapter(tmp_path, payload, parser=parse)
    rows, errors = cache.load(payload, ("2330",), cutoff)
    assert errors == {"2330": []}
    assert len(calls) == 2
    assert rows["2330"][0]["facts"][0]["numeric_value"] == 5_000_000
    assert rows["2330"][1]["facts"] == []
    restarted = adapter(tmp_path, payload, parser=parse)
    assert restarted.load(payload, ("2330",), cutoff) == (rows, errors)
    at = aware_timestamp("2026-09-11T05:30:00+00:00")
    assert restarted.load(payload, ("2330",), at)[0]["2330"][1]["facts"][0]["numeric_value"] == 6_000_000
    assert len(calls) == 2
    assert restarted.stats["sqlite_index_hit"] == 2
    assert epoch_microseconds(at) == epoch_microseconds("2026-09-11T13:30:00+08:00")
    with pytest.raises(ValueError): epoch_microseconds(dt.datetime(2026, 9, 11))
    with sqlite3.connect(cache.path) as conn:
        plan = conn.execute("EXPLAIN QUERY PLAN SELECT ordinal FROM research_filing_versions "
            "WHERE version=? AND evidence_id=? AND symbol=? AND available_at_epoch_us<=?",
            (cache.version, cache.evidence_id, "2330", epoch_microseconds(at))).fetchall()
        assert any("research_filings_cutoff" in row[3] for row in plan)


def test_code_model_parameter_versions_isolate_cache_and_dataset_metadata(tmp_path):
    payload = history()
    calls = []
    def parse(body):
        calls.append(True)
        return parse_ixbrl(body)
    at = aware_timestamp("2026-09-11T13:30:00+08:00")
    for version in ("current", "current", "new-model", "new-code", "new-parameters"):
        adapter(tmp_path, payload, version=version, parser=parse).load(payload, ("2330",), at)
    assert len(calls) == 8
    changed = deepcopy(payload)
    changed["filings"]["2330"][0]["publication_proof"]["published_at"] = "2026-09-12"
    reused = adapter(tmp_path, changed, parser=parse)
    rows, _ = reused.load(changed, ("2330",), at)
    assert rows["2330"][0]["facts"] == []
    assert reused.stats["sqlite_ixbrl_hit"] == 2  # Same bytes, independent new publication metadata.
    assert len(calls) == 8


@pytest.mark.parametrize("damage", ["facts", "missing_parse", "available_at", "missing_version"])
def test_corrupt_derived_data_is_rebuilt_not_used_as_evidence(tmp_path, damage):
    payload = history()
    cache = adapter(tmp_path, payload)
    at = aware_timestamp("2026-09-10T13:30:00+08:00")
    baseline = cache.load(payload, ("2330",), at)
    with sqlite3.connect(cache.path) as conn:
        if damage == "facts":
            conn.execute("UPDATE research_parses SET facts_json='[]'")
        elif damage == "missing_parse":
            conn.execute("DELETE FROM research_parses")
        elif damage == "available_at":
            conn.execute("UPDATE research_filing_versions SET available_at_epoch_us=0")
        else:
            conn.execute("DELETE FROM research_filing_versions WHERE ordinal=1")
    assert adapter(tmp_path, payload).load(payload, ("2330",), at) == baseline


def test_interruption_rolls_back_index_and_parse_writes(tmp_path):
    payload = history()
    calls = 0
    def interrupted(body):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError("simulated interruption")
        return parse_ixbrl(body)
    at = aware_timestamp("2026-09-11T13:30:00+08:00")
    cache = adapter(tmp_path, payload, parser=interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        cache.load(payload, ("2330",), at)
    with sqlite3.connect(cache.path) as conn:
        assert not conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert len(adapter(tmp_path, payload).load(payload, ("2330",), at)[0]["2330"]) == 2


def test_foreign_database_is_never_modified(tmp_path):
    payload = history()
    cache = adapter(tmp_path, payload)
    with sqlite3.connect(cache.path) as conn:
        conn.execute("CREATE TABLE filings (symbol TEXT)")
        conn.execute("INSERT INTO filings VALUES ('do-not-touch')")
    before = cache.path.read_bytes()
    with pytest.raises(ValueError, match="FOREIGN_DATABASE"):
        cache.load(payload, ("2330",), aware_timestamp("2026-09-11T13:30:00+08:00"))
    assert cache.path.read_bytes() == before


def test_native_cached_and_uncached_daily_results_revision_selection_match(tmp_path):
    payload = history()
    path = create_dataset(payload, tmp_path / "datasets")
    req = replace(request(tmp_path, payload, day="2026-09-10"), evidence_id=path.parent.name,
                  versions=current_versions())
    plain, _ = build_replay_engine(path)
    cached, _ = build_replay_engine(path, parse_cache_path=tmp_path / "parse.sqlite3")
    results = []
    for day in ("2026-09-10", "2026-09-11"):
        req = replace(req, as_of=aware_timestamp(day + "T13:30:00+08:00"), history_end=dt.date.fromisoformat(day))
        a = plain.run_with_coverage(replace(req, state_path=tmp_path / "plain.sqlite3"))
        b = cached.run_with_coverage(replace(req, state_path=tmp_path / "cached.sqlite3"))
        assert a == b
        selected = cached.evidence.select(req)
        chosen = [row["revision_number"] for row in selected.evidence["filings"]["2330"] if row["selected"]]
        assert chosen == [1 if day.endswith("10") else 2]
        results.append(b)
    assert results[0][0]["diagnostics"]["excluded_evidence"]["2330"][0]["reason"] == "AFTER_CUTOFF"


def test_scope_precedence_and_conflicts_keep_original_semantics(tmp_path):
    payload = history()
    first, second = payload["filings"]["2330"]
    second["statement_scope"] = "individual"
    second["receipt"]["report_id"] = "A"
    second["publication_proof"] = proof(filing_binding("2330", second),
        publication="2026-09-11T13:30:00+08:00", precision="TIMESTAMP")
    second["publication_proof"]["original_published_at"] = "2026-08-01T00:00:00+08:00"
    path = create_dataset(payload, tmp_path / "datasets")
    req = replace(request(tmp_path, payload), evidence_id=path.parent.name)
    engine, _ = build_replay_engine(path, parse_cache_path=tmp_path / "parse.sqlite3")
    receipts = engine.evidence.select(req).evidence["filings"]["2330"]
    assert len(receipts) == 2
    assert [row["statement_scope"] for row in receipts if row["selected"]] == ["consolidated"]
    conflicted = history()
    left, right = conflicted["filings"]["2330"]
    right["revision_number"] = 1
    right["publication_proof"] = proof(filing_binding("2330", right))
    path2 = create_dataset(conflicted, tmp_path / "datasets")
    engine2, _ = build_replay_engine(path2, parse_cache_path=tmp_path / "parse.sqlite3")
    selected = engine2.evidence.select(replace(req, evidence_id=path2.parent.name))
    assert any(row["reason"] == "CONFLICTING_FILING_VERSION" for row in selected.diagnostics["excluded_evidence"]["2330"])


def test_shared_cache_job_restart_no_cache_and_document_checks(tmp_path, monkeypatch):
    payload = history()
    path = create_dataset(payload, tmp_path / "datasets")
    req = replace(request(tmp_path, payload, day="2026-09-10"), versions=current_versions(), evidence_id=path.parent.name)
    cache_path = tmp_path / "shared.sqlite3"
    with ResearchJob(path, tmp_path / "job", parse_cache_path=cache_path) as job:
        first = job.run_day(req)
    def forbidden(_): raise AssertionError("Warm SQLite cache must not reparse")
    monkeypatch.setattr("infrastructure.replay_runtime.parse_ixbrl", forbidden)
    with ResearchJob(path, tmp_path / "job", parse_cache_path=cache_path) as resumed:
        next_req = replace(req, as_of=aware_timestamp("2026-09-11T13:30:00+08:00"), history_end=dt.date(2026, 9, 11))
        resumed.run_day(next_req)
        assert resumed.cache_stats["sqlite_index_hit"] == 1
    monkeypatch.setattr("infrastructure.replay_runtime.parse_ixbrl", parse_ixbrl)
    with ResearchJob(path, tmp_path / "off", cache_enabled=False, parse_cache_path=tmp_path / "never.sqlite3") as off:
        assert off.run_day(req).result == first.result
    assert not (tmp_path / "never.sqlite3").exists()
    eid, frozen = load_evidence(path)
    documents = EvidenceDocuments.for_evidence(path, eid)
    documents.objects.path(next(iter(frozen["raw_documents"]))).unlink()
    with pytest.raises(ValueError, match="SOURCE_DOCUMENT_UNAVAILABLE"):
        ResearchJob(path, tmp_path / "job", parse_cache_path=cache_path)


def test_cli_shared_cache_contract_and_rebuild_after_deletion(tmp_path, capsys):
    payload = history()
    path = create_dataset(payload, tmp_path / "datasets")
    req = replace(request(tmp_path, payload), versions=current_versions(), evidence_id=path.parent.name)
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps([req.inputs()]))
    cache = tmp_path / "shared.sqlite3"
    for name in ("first", "second"):
        assert main(["job-capture", "--evidence", str(path), "--job", str(tmp_path / name),
                     "--requests", str(requests), "--parse-cache", str(cache)]) == 0
        capsys.readouterr()
        if name == "first": cache.unlink()
    with ResearchJob(path, tmp_path / "second", parse_cache_path=cache) as job:
        assert job.reproduce()["equal"]


def test_single_date_cli_and_public_api_use_same_cached_engine(tmp_path):
    from interface.cli import main as replay_cli
    payload = history()
    path = create_dataset(payload, tmp_path / "datasets")
    holdings = tmp_path / "holdings.json"
    holdings.write_text("[]")
    cache = tmp_path / "shared.sqlite3"
    assert replay_cli(["2330", "-b", payload["start"], "-e", "2026-09-11",
        "--replay-id", "test", "--replay-evidence-file", str(path), "--replay-parse-cache", str(cache),
        "--replay-state-path", str(tmp_path / "cli.sqlite3"), "--replay-output-dir", str(tmp_path / "reports"),
        "--pae-holdings-file", str(holdings), "--pae-current-cash", "1000000", "--account-equity", "1000000"]) == 0
    engine, eid = build_replay_engine(path, parse_cache_path=cache)
    req = replace(request(tmp_path, payload), versions=current_versions(), evidence_id=eid)
    assert json.loads(next((tmp_path / "reports").glob("*.json")).read_text()) == engine.run(req)


def test_indexed_engine_rejects_versions_changed_after_open(tmp_path, monkeypatch):
    import infrastructure.replay_runtime as runtime
    payload = history()
    path = create_dataset(payload, tmp_path / "datasets")
    engine, eid = build_replay_engine(path, parse_cache_path=tmp_path / "parse.sqlite3")
    versions = current_versions()
    monkeypatch.setattr(runtime, "current_versions", lambda: replace(versions, code="changed"))
    req = replace(request(tmp_path, payload), evidence_id=eid, versions=versions)
    with pytest.raises(ValueError, match="VERSION_CHANGED_DURING_JOB"):
        engine.run(req)


def test_database_failure_uses_uncached_semantics_and_unsupported_schema_rejected(tmp_path, monkeypatch):
    payload = history()
    cutoff = aware_timestamp("2026-09-11T13:30:00+08:00")
    cache = adapter(tmp_path, payload)
    expected = cache.load(payload, ("2330",), cutoff)
    with monkeypatch.context() as patch:
        def locked(*args, **kwargs): raise sqlite3.OperationalError("database locked")
        patch.setattr(sqlite3, "connect", locked)
        assert digest(cache.load(payload, ("2330",), cutoff)) == digest(expected)
    assert cache.stats["sqlite_fallback"] == 1
    with sqlite3.connect(cache.path) as conn:
        conn.execute("PRAGMA user_version=999")
    before = cache.path.read_bytes()
    with pytest.raises(ValueError, match="UNSUPPORTED_RESEARCH_CACHE_SCHEMA"):
        cache.load(payload, ("2330",), cutoff)
    assert cache.path.read_bytes() == before


def test_temporary_document_failure_is_not_persisted_as_a_negative_cache(tmp_path):
    payload = history()
    path = create_dataset(payload, tmp_path / "datasets")
    eid, frozen = load_evidence(path)
    documents = EvidenceDocuments.for_evidence(path, eid)
    available = False
    def reader(reference):
        if not available: raise ValueError("SOURCE_DOCUMENT_UNAVAILABLE:" + reference["sha256"])
        return documents.read(reference)
    cache = SQLiteFilingHistory(tmp_path / "parse.sqlite3", evidence_id=eid, version="test",
                               parse_filing=parse_ixbrl, document_reader=reader)
    at = aware_timestamp("2026-09-11T13:30:00+08:00")
    with pytest.raises(ValueError, match="SOURCE_DOCUMENT_UNAVAILABLE"):
        cache.load(frozen, ("2330",), at)
    available = True
    records, errors = cache.load(frozen, ("2330",), at)
    assert len(records["2330"]) == 2
    assert errors == {"2330": []}
