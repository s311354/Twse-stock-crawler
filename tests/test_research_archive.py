"""SYNTHETIC evidence, isolated under tmp_path; never promoted to production."""
import base64
from copy import deepcopy
from dataclasses import asdict, replace
import datetime as dt
import hashlib
import json
import sqlite3

import pandas as pd
import pytest

from application.historical_replay import replay_close_cutoff
from application.replay_contracts import digest, calendar_window
from application.replay_inputs import PinnedReplayEvidence as Evidence
from application.research_evidence import (
    RAW_EVIDENCE_SCHEMA, materialize_filings as materialize, filing_binding, calendar_at as calendar_policy, verify_publication,
)
from application.research_audit import coverage_audit, compare_results
from domain.fundamental_derivation import derive_fundamentals, FiledFact, FiledFinancials
from domain.strategies.low_entry_v6_quality import normalize_quarters
from domain.strategies.portfolio_allocation_v2 import PersistenceObservationV2
from infrastructure.daily_research import retain_research, reproduce_research
from infrastructure.replay_evidence import load_evidence
from infrastructure.replay_runtime import current_versions, build_replay_engine
from infrastructure.research_archive import create_dataset, write_once
from infrastructure.storage.replay_state import SQLiteReplayState
from interface.research import main
from tests.test_strict_replay import payload_fixture, request, VERSIONS
from tests.fundamentals.test_ixbrl_units import document
from infrastructure.calendar.twse_calendar import TwseCalendar
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl


def annual_calendar(data):
    return TwseCalendar(fetcher=lambda: data)


def PinnedReplayEvidence(evidence_id, payload):
    return Evidence(evidence_id, payload, parse_filing=parse_ixbrl, annual_calendar=annual_calendar)


def materialize_filings(payload):
    return materialize(payload, parse_filing=parse_ixbrl)


def calendar_at(calendar, cutoff, **kwargs):
    return calendar_policy(calendar, cutoff, annual_calendar=annual_calendar, **kwargs)


def proof(binding, *, publication="2026-08-01", precision="DATE"):
    body = "SYNTHETIC TEST ONLY " + json.dumps(binding, sort_keys=True) + " " + publication
    return dict(status="VERIFIED", verified_by="SYNTHETIC_TEST_PROVIDER",
        verification_method="TEST EXACT VERSION BINDING", verified_at="2026-09-20T08:00:00+08:00",
        binding=binding, published_at=publication, publication_precision=precision, quotes=[body],
        document=dict(url="https://www.twse.com.tw/test-only", body_utf8=body,
            sha256=hashlib.sha256(body.encode()).hexdigest(), retrieved_at="2026-09-20T08:00:00+08:00"))


def raw_fixture():
    payload = payload_fixture()
    payload.update(schema=RAW_EVIDENCE_SCHEMA, filings={}, raw_documents={}, breadth_sources={},
        market_policy="SYNTHETIC_TEST_ONLY", purpose="SYNTHETIC_TEST_ONLY")
    for year in {day[:4] for day in payload["calendar"]["days"]}:
        annual = dict(stat="ok", fields=["日期", "名稱", "說明"],
                      data=[[year+"-01-01", "元旦休市", "休市"]])
        binding = dict(kind="ANNUAL_CALENDAR", market="TWSE", year=int(year), content_sha256=digest(annual))
        payload["calendar"].setdefault("documents", {})[year] = dict(payload=annual,
            publication_proof=proof(binding, publication=f"{int(year)-1}-12-01"))
    return payload


def add_raw(payload, *, revision=1, proof_present=True, published="2026-08-01"):
    content = document()
    sha = hashlib.sha256(content).hexdigest()
    row = dict(fiscal_year=2026, quarter=2, revision_number=revision,
        available_at="2026-09-20T08:00:00+08:00", availability_source="FIRST_OBSERVED_DOWNLOAD",
        content_sha256=sha, source_url="https://mops.twse.com.tw/test-only", period_end="2026-06-30",
        statement_scope="consolidated", source_version=f"test-source-r{revision}",
        receipt=dict(fetched_at="2026-09-20T08:00:00+08:00", report_id="C"))
    if proof_present: row["publication_proof"] = proof(filing_binding("2330", row), publication=published)
    payload["filings"] = {"2330": [row]}
    payload["raw_documents"][sha] = dict(url=row["source_url"], sha256=sha,
        retrieved_at=row["receipt"]["fetched_at"], body_base64=base64.b64encode(content).decode())
    return row


def test_verified_late_collection_date_precision_and_no_deadline_backdating(tmp_path):
    payload = raw_fixture(); add_raw(payload)
    rows, errors = materialize_filings(payload)
    assert not errors["2330"]
    assert rows["2330"][0]["available_at"] == "2026-08-02T00:00:00+08:00"
    assert rows["2330"][0]["receipt"]["fetched_at"].startswith("2026-09-20")
    # Raw parser retains TWD scale and per-share units (not old cached facts).
    facts = rows["2330"][0]["facts"]
    assert facts[0]["numeric_value"] == 5_000_000
    assert facts[1]["numeric_value"] == -.14
    before = PinnedReplayEvidence(digest(payload), payload).select(request(tmp_path, payload, day="2026-09-11"))
    assert before.evidence["filings"]["2330"][0]["selected"]
    no_proof = deepcopy(payload); no_proof["filings"]["2330"][0].pop("publication_proof")
    excluded = PinnedReplayEvidence(digest(no_proof), no_proof).select(request(tmp_path, no_proof))
    assert not excluded.evidence["filings"]["2330"]
    assert excluded.diagnostics["excluded_evidence"]["2330"][0]["reason"] == "AFTER_CUTOFF"
    assert dt.datetime.fromisoformat(rows["2330"][0]["available_at"]) > replay_close_cutoff("2026-08-01")


def test_latest_revision_cannot_borrow_original_publication_or_changed_hash():
    payload = raw_fixture(); row = add_raw(payload, revision=2)
    _, errors = materialize_filings(payload)
    assert errors["2330"][0]["reason"] == "REVISION_PUBLICATION_ORDER_UNVERIFIED"
    row["publication_proof"]["original_published_at"] = "2026-07-01T10:00:00+08:00"
    row["publication_proof"]["binding"]["content_sha256"] = "a"*64
    assert materialize_filings(payload)[1]["2330"][0]["reason"] == "ORIGINAL_VERSION_BINDING_NOT_VERIFIED"


def test_calendar_emergency_unknown_and_no_bars_are_different(tmp_path):
    payload = raw_fixture()
    event = dict(date="2026-09-10", status="CLOSED", market="TWSE")
    event["proof"] = proof(dict(kind="CALENDAR_EXCEPTION", **event), publication="2026-09-09")
    payload["calendar"]["announcements"] = [event]
    selected = PinnedReplayEvidence(digest(payload), payload).select(request(tmp_path, payload))
    assert "2026-09-10" not in selected.sessions
    assert "2026-09-10" not in selected.diagnostics["missing_bars"]["2330"]
    payload["bars"] = [b for b in payload["bars"] if b["date"] != "20260911"]
    selected = PinnedReplayEvidence(digest(payload), payload).select(request(tmp_path, payload))
    assert "2026-09-11" in selected.diagnostics["missing_bars"]["2330"]
    payload["calendar"]["documents"] = {}
    selected = PinnedReplayEvidence(digest(payload), payload).select(request(tmp_path, payload))
    assert "2026-09-11" in selected.diagnostics["calendar_unknown_dates"]
    assert "2026-09-10" not in selected.diagnostics["calendar_unknown_dates"]


def test_calendar_mapping_cannot_override_original_document():
    payload = raw_fixture(); payload["calendar"]["days"]["2026-09-11"] = "CLOSED"
    days, _ = calendar_at(payload["calendar"], replay_close_cutoff("2026-09-11"), strict=True)
    assert days["2026-09-11"] == "OPEN"


def test_prelisting_requires_source_proof_not_first_seen_bar(tmp_path):
    payload = raw_fixture()
    event = dict(symbol="2330", market="TWSE", kind="PRE_LISTING", start=payload["start"], end="2026-09-09")
    payload["security_lifecycle"] = {"2330": [dict(event, proof=proof(event))]}
    payload["bars"] = [b for b in payload["bars"] if b["date"] >= "20260910"]
    selected = PinnedReplayEvidence(digest(payload), payload).select(request(tmp_path, payload))
    assert not selected.diagnostics["missing_bars"]["2330"]
    assert selected.diagnostics["bar_exemptions"]["2330"]
    payload["security_lifecycle"] = {}
    selected = PinnedReplayEvidence(digest(payload), payload).select(request(tmp_path, payload))
    assert selected.diagnostics["missing_bars"]["2330"]


def test_v2_calibration_requires_sample_start_version_and_cutoff(tmp_path):
    payload = raw_fixture()
    calibration = dict(symbol="2330", market="TWSE", available_at="2026-08-01T10:00:00+08:00",
        training_start="2025-01-01T13:30:00+08:00", training_end="2026-07-31T13:30:00+08:00",
        source_id="synthetic", source_sha256="b"*64, calibration_version="test-v1",
        half_kelly_weight=.1, sample_count=200, method="empirical-test", model_version=VERSIONS.model,
        parameter_sha256=VERSIONS.parameters)
    payload["kelly_calibrations"] = [calibration]
    selected = PinnedReplayEvidence(digest(payload), payload).select(request(tmp_path, payload))
    assert selected.diagnostics["kelly"]["2330"]["status"] == "PASS"
    for key, value in (("training_start", None), ("training_end", "2026-10-01T13:30:00+08:00"), ("calibration_version", None)):
        changed = deepcopy(payload); changed["kelly_calibrations"][0][key] = value
        assert not PinnedReplayEvidence(digest(changed), changed).select(request(tmp_path, changed)).calibrations


def test_dataset_stable_identity_immutable_and_repair_lineage(tmp_path):
    payload = raw_fixture(); first = create_dataset(payload, tmp_path, collected_at="2026-09-20T00:00:00Z")
    original = first.read_bytes()
    assert create_dataset(payload, tmp_path, collected_at="2026-09-28T00:00:00Z") == first
    assert first.read_bytes() == original
    changed = deepcopy(payload); changed["preparation_gaps"].append("KNOWN_MISSING_SOURCE")
    second = create_dataset(changed, tmp_path, parent=first)
    assert first != second and first.read_bytes() == original
    manifest = json.loads((second.parent/"manifest.json").read_text())
    assert manifest["parent_dataset_id"] == first.parent.name
    with pytest.raises(ValueError, match="IMMUTABLE"):
        write_once(first, {"overwrite": True})
    load_evidence(first)


def test_snapshot_restore_isolated_exact_and_observations_not_duplicated(tmp_path):
    payload = payload_fixture(); store = SQLiteReplayState()
    req = request(tmp_path, payload, day="2026-09-10")
    with store.session(req, ("2026-09-10", "2026-09-11")) as session:
        session.record_session_v2(req.as_of.date(), (PersistenceObservationV2("2330", "3014", True),))
        session.finish({"test": "synthetic"})
    snapshot = store.export_snapshot(req)
    next_day = replace(req, as_of=replay_close_cutoff("2026-09-11"), history_end=dt.date(2026,9,11), state_path=tmp_path/"fork.sqlite3")
    before = req.state_path.read_bytes()
    store.restore_snapshot(next_day, snapshot); store.restore_snapshot(next_day, snapshot)
    assert store.export_snapshot(next_day) == snapshot
    assert req.state_path.read_bytes() == before
    with pytest.raises(ValueError, match="IDENTITY"):
        store.restore_snapshot(replace(next_day, evidence_id="different"), snapshot)
    with sqlite3.connect(next_day.state_path) as connection:
        assert connection.execute("SELECT count(*) FROM replay_observations").fetchone()[0] == 1


def test_daily_interruption_retry_recomputation_and_api_cli_parity(tmp_path, monkeypatch):
    payload = raw_fixture(); evidence = create_dataset(payload, tmp_path/"datasets")
    req = replace(request(tmp_path, payload), versions=current_versions())
    pre = SQLiteReplayState.empty_snapshot(req)
    original = __import__("infrastructure.daily_research", fromlist=["write_once"]).write_once
    def interrupt(path, value):
        if path.name == "complete.json": raise RuntimeError("simulated interruption after commit")
        original(path, value)
    monkeypatch.setattr("infrastructure.daily_research.write_once", interrupt)
    with pytest.raises(RuntimeError): retain_research(evidence, req, tmp_path/"runs", predecessor=pre)
    monkeypatch.setattr("infrastructure.daily_research.write_once", original)
    folder = retain_research(evidence, req, tmp_path/"runs", predecessor=pre)
    assert retain_research(evidence, req, tmp_path/"runs", predecessor=pre) == folder
    assert not req.state_path.exists()  # Caller/formal state is never changed.
    assert reproduce_research(folder, evidence)["equal"]
    actual = json.loads((folder/"result.json").read_text())
    engine, _ = build_replay_engine(evidence)
    expected = engine.run(replace(req, state_path=tmp_path/"api.sqlite3"))
    assert actual == expected
    from interface.cli import main as replay_cli
    holdings = tmp_path/"holdings.json"; holdings.write_text("[]")
    assert replay_cli(["2330", "-b", req.history_start.isoformat(), "-e", req.history_end.isoformat(),
        "--replay-id", req.replay_id, "--replay-evidence-file", str(evidence),
        "--replay-state-path", str(tmp_path/"cli.sqlite3"), "--replay-output-dir", str(tmp_path/"cli"),
        "--pae-holdings-file", str(holdings), "--pae-current-cash", "1000000", "--account-equity", "1000000"]) == 0
    assert json.loads(next((tmp_path/"cli").glob("*.json")).read_text()) == actual


def test_compare_does_not_hide_research_differences():
    left = dict(replay_id="a", decisions=[dict(score=70, gates={"quality":"UNKNOWN"})], rows=[])
    right = dict(left, replay_id="b")
    assert compare_results(left, right, different_execution=True)["equal"]
    right = deepcopy(right); right["decisions"][0]["gates"]["quality"] = "PASS"
    assert not compare_results(left, right, different_execution=True)["equal"]


def test_derivation_available_time_and_scope_are_conservative():
    recent = FiledFinancials(2026,1,"2026-05-01T10:00:00Z",1,())
    prior = FiledFinancials(2025,4,"2026-06-01T10:00:00Z",2,())
    result = derive_fundamentals([recent,prior], signal_date="2026-08-01T00:00:00Z", sector="24")
    assert result.snapshot["available_at"] == prior.available_at
    individual = replace(prior, statement_scope="individual")
    result = derive_fundamentals([recent,individual], signal_date="2026-08-01T00:00:00Z", sector="24")
    assert result.annual_fundamentals.empty
    assert result.snapshot["available_at"] == recent.available_at
    periods = {pd.Period("2026Q1"):dict(flow_basis="QUARTER",revenue=100),
               pd.Period("2026Q2"):dict(flow_basis="YTD",revenue=250)}
    assert normalize_quarters(periods)[pd.Period("2026Q2")]["revenue"] is None


def test_future_raw_filing_and_batch_do_not_change_native_research(tmp_path):
    from dataclasses import fields
    from domain.fundamental_models import FundamentalSnapshot
    payload = raw_fixture()
    before = create_dataset(payload, tmp_path / "datasets")
    req = replace(request(tmp_path, payload), versions=current_versions())
    baseline = build_replay_engine(before)[0].run(req)
    changed = deepcopy(payload)
    add_raw(changed, proof_present=False)  # First observed after evaluation cutoff.
    batch = {field.name: None for field in fields(FundamentalSnapshot)}
    batch.update(symbol="2330", company_name="SYNTHETIC", fiscal_year=2026, quarter=2,
                 observed_at="2026-09-20T08:00:00+08:00")
    changed["batches"] = {"2330": [asdict(FundamentalSnapshot(**batch))]}
    after = create_dataset(changed, tmp_path / "datasets", parent=before)
    actual = build_replay_engine(after)[0].run(replace(req, evidence_id=after.parent.name,
                                                     state_path=tmp_path / "after.sqlite3"))
    assert actual["rows"] == baseline["rows"]
    assert actual["decisions"] == baseline["decisions"]
    assert len(actual["diagnostics"]["excluded_evidence"]["2330"]) == 2


def test_restore_interruption_leaves_no_destination_and_retry_is_safe(tmp_path, monkeypatch):
    payload = raw_fixture()
    req = request(tmp_path, payload)
    store = SQLiteReplayState()
    snapshot = store.empty_snapshot(req)
    initialize = store._initialize
    def fail(connection):
        initialize(connection)
        raise RuntimeError("test interruption")
    monkeypatch.setattr(store, "_initialize", fail)
    with pytest.raises(RuntimeError): store.restore_snapshot(req, snapshot)
    assert not req.state_path.exists()
    monkeypatch.setattr(store, "_initialize", initialize)
    store.restore_snapshot(req, snapshot)
    assert store.export_snapshot(req) == snapshot


def test_unknown_calendar_blocks_research_without_hiding_diagnostic_scores(tmp_path):
    payload = raw_fixture(); payload["calendar"]["documents"] = {}
    evidence = create_dataset(payload, tmp_path / "datasets")
    req = replace(request(tmp_path, payload), versions=current_versions())
    engine, _ = build_replay_engine(evidence)
    result = engine.run(req)
    assert result["decisions"][0]["hard_gates"]["calendar"] == "UNKNOWN"
    assert result["decisions"][0]["hard_gates"]["market_history"] == "UNKNOWN"
    assert not result["decisions"][0]["research_entry_eligible"]
    assert result["pae_state"]["observations"] == []
    assert result["decisions"][0]["raw"]["les_score"] is not None
    audit = coverage_audit(req, engine.evidence.select(req), result)
    assert "published_report" not in audit["condition_failure_counts"]


def test_empty_restriction_list_is_not_clear(tmp_path):
    payload = raw_fixture(); payload["restrictions"] = []; payload["restriction_ledger"] = None
    selected = PinnedReplayEvidence(digest(payload), payload).select(request(tmp_path, payload))
    assert selected.diagnostics["restrictions"]["2330"]["status"] == "UNKNOWN"
    assert selected.market.trading_restrictions == {}


def test_missing_benchmark_and_bar_lifecycle_conflict_block_research(tmp_path):
    payload = raw_fixture()
    event = dict(symbol="2330", market="TWSE", kind="SUSPENDED", start="2026-09-10", end="2026-09-11")
    payload["security_lifecycle"] = {"2330": [dict(event, proof=proof(event))]}
    payload["bars"] = [r for r in payload["bars"] if r["dataset"] != "TAIEX"]
    evidence = create_dataset(payload, tmp_path/"datasets")
    req = replace(request(tmp_path, payload), versions=current_versions())
    result = build_replay_engine(evidence)[0].run(req)
    assert result["diagnostics"]["missing_taiex_dates"]
    assert any(r["reason"] == "BAR_CONFLICTS_WITH_SECURITY_LIFECYCLE" for r in result["diagnostics"]["invalid_bars"])
    assert result["decisions"][0]["hard_gates"]["market_context"] == "UNKNOWN"
    assert not result["decisions"][0]["research_entry_eligible"]
