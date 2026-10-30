from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from interface import daily_oos_collection as collection


TZ = ZoneInfo("Asia/Taipei")


class Calendar:
    def __init__(self, result=True, error=None):
        self.result, self.error = result, error

    def is_trading_day(self, day):
        if self.error:
            raise self.error
        return self.result


def test_daily_command_is_no_mail_and_has_fixed_range_and_account():
    command = collection._daily_command(Path("/isolated/output"), python="python315")
    assert command[:4] == ["python315", str(collection.PROJECT_ROOT / "main.py"), "-o", "shirong"]
    assert "--no-mail" in command and "-m" not in command
    assert command[command.index("-e") + 1] == "700"
    assert command[command.index("--account-equity") + 1] == "1000000"
    assert command[command.index("--pae-current-cash") + 1] == "1000000"


def test_child_environment_is_isolated_and_declares_empty_research_account(tmp_path):
    root = tmp_path / "collection"
    env = collection._child_environment(tmp_path / "day", root)
    assert env["TWSE_DATA_DIR"] == str(tmp_path / "day")
    assert env["TWSE_PAE_PERSISTENCE_PATH"] == str(root / "pae" / "r19" / "portfolio.sqlite3")
    assert env["TWSE_PAE_HOLDINGS_JSON"] == "[]"
    assert env["TWSE_PAE_CURRENT_CASH"] == "1000000"
    assert env["TWSE_ACCOUNT_EQUITY"] == "1000000"
    assert env["TWSE_RUN_LOCK_PATH"] == str(collection.SOURCE_DATA / "stockdataanalysis.lock")
    assert env["TWSE_PAE_PERSISTENCE_PATH"] != str(collection.SOURCE_DATA / "portfolio_allocation.sqlite3")
    assert all(not env.get(key) for key in ("TWSE_SMTP_EMAIL", "TWSE_SMTP_PASSWORD", "TWSE_SMTP_RECEIVERS"))
    assert not env["TWSE_LES_V7_BUNDLE"] and not env["TWSE_LES_V7_STRATEGY_POLICY"]
    assert env["TWSE_LES_V7_DIAGNOSTICS_ONLY"] == "false"


def test_collection_skips_official_closed_day_without_running_or_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "SOURCE_DATA", tmp_path / "data")
    result = collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 26, 16, 0, tzinfo=TZ),
        calendar=Calendar(False), runner=lambda *args, **kwargs: pytest.fail("closed day must not run"))
    assert result["status"] == "SKIPPED_CLOSED"
    assert not (tmp_path / "data" / "les-oos-collections").exists()


def test_collection_fails_closed_when_official_calendar_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "SOURCE_DATA", tmp_path / "data")
    with pytest.raises(collection.CollectionError, match="OFFICIAL_TWSE_CALENDAR_UNKNOWN"):
        collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 26, 16, 0, tzinfo=TZ),
            calendar=Calendar(error=OSError("offline")), runner=lambda *args, **kwargs: None)
    assert not (tmp_path / "data" / "les-oos-collections").exists()


def test_collection_refuses_to_run_before_close_window(tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "SOURCE_DATA", tmp_path / "data")
    with pytest.raises(collection.CollectionError, match="BEFORE_POST_CLOSE"):
        collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 26, 15, 30, tzinfo=TZ),
            calendar=Calendar(True), runner=lambda *args, **kwargs: pytest.fail("too early"))


def test_existing_receipt_is_hash_verified_and_identity_bound(tmp_path):
    root = tmp_path / "series"
    session = root / "sessions" / "2026-10-26"
    session.mkdir(parents=True)
    identity = collection._sealed_identity({"collection_id": "test-oos", "schema": collection.IDENTITY_SCHEMA})
    (root / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (session / "artifact.json").write_text("{}", encoding="utf-8")
    manifest = {"session_date": session.name, "identity_sha256": identity["identity_sha256"]}
    manifest["capture_id"] = collection.digest(manifest)
    (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    receipt = {"schema": collection.COLLECTION_SCHEMA, "status": "SOURCE_CAPTURED",
        "identity_sha256": identity["identity_sha256"], "session_date": session.name,
        "capture_id": manifest["capture_id"],
        "files": {name: collection.file_sha256(session / name) for name in ("artifact.json", "manifest.json")}}
    (session / "complete.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert collection._verified_receipt(session) == receipt
    (session / "artifact.json").write_text("changed", encoding="utf-8")
    with pytest.raises(collection.CollectionError, match="NO_OVERWRITE"):
        collection._verified_receipt(session)


def test_collection_identity_requires_exact_elec_selector_range(tmp_path):
    bad = tmp_path / "stocklist"
    bad.write_text("0;1;2", encoding="utf-8")
    with pytest.raises(collection.CollectionError, match="STOCKLIST_0_435"):
        collection.collection_identity("test-oos", stocklist_path=bad)


def setup_collection(tmp_path, monkeypatch):
    """SYNTHETIC ONLY: exercise publication, never source/model efficacy."""
    monkeypatch.setattr(collection, "SOURCE_DATA", tmp_path / "data")
    monkeypatch.setattr(collection, "EXPECTED_STOCKLIST_SIZE", 2)
    monkeypatch.setattr(collection, "collection_identity", lambda *args, **kwargs: {
        "schema": collection.IDENTITY_SCHEMA, "collection_id": "test-oos", "versions": {"code": "synthetic"},
        "account_inputs": {"equity": 1e6, "cash": 1e6, "holdings": []}})
    monkeypatch.setattr(collection, "_validate_native_outputs", lambda *args, **kwargs: {
        "scope_status": "COMPLETE", "r19_baseline": collection.R19_VERSION, "report_count": 2,
        "selected_symbols": ["SYNTHETIC_A", "SYNTHETIC_B"],
        "current_session_symbols": ["SYNTHETIC_A", "SYNTHETIC_B"]})
    from infrastructure import research_objects
    class SyntheticGuard:
        versions = None
        def retain(self, objects):
            return {"source": "synthetic", "environment": "synthetic", "parameters": "synthetic"}
        def check(self):
            pass
    monkeypatch.setattr(research_objects, "ResearchVersionGuard", SyntheticGuard)
    monkeypatch.setattr(collection, "asdict", lambda value: {"code": "synthetic"})
    return tmp_path / "data" / "les-oos-collections" / "test-oos"


def test_collection_success_retry_and_next_day_are_immutable(tmp_path, monkeypatch):
    root = setup_collection(tmp_path, monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="SYNTHETIC_ONLY", stderr="")
    first = collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 26, 16, tzinfo=TZ),
                                     calendar=Calendar(), runner=run)
    receipt_path = root / "sessions" / "2026-10-26" / "complete.json"
    original = receipt_path.read_bytes()
    retry = collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 26, 17, tzinfo=TZ),
                                     calendar=Calendar(), runner=run)
    assert first["status"] == "SOURCE_CAPTURED" and retry["status"] == "ALREADY_CAPTURED"
    assert len(calls) == 1 and receipt_path.read_bytes() == original
    manifest = json.loads((receipt_path.parent / "manifest.json").read_text())
    assert manifest["bank_status"]["fill_observations"] == "NOT_CAPTURED"
    assert not manifest["cost_status"]["full_cost_oos_eligibility"]
    assert manifest["outputs"]["universe_status"] == "FROZEN"
    assert manifest["evaluation_cutoff_policy"].endswith("NOT_13_30_REPLAY")
    collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 27, 16, tzinfo=TZ),
                             calendar=Calendar(), runner=run)
    assert len(calls) == 2 and receipt_path.read_bytes() == original


@pytest.mark.parametrize("day", [26, 27])
def test_interrupted_run_preserved_and_blocks_same_or_later_day(tmp_path, monkeypatch, day):
    root = setup_collection(tmp_path, monkeypatch)
    with pytest.raises(collection.CollectionError, match="DAILY_RUN_FAILED"):
        collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 26, 16, tzinfo=TZ),
            calendar=Calendar(), runner=lambda *args, **kwargs: SimpleNamespace(returncode=4, stdout="", stderr="busy"))
    attempts = list((root / "sessions").glob(".*.attempt-*"))
    assert len(attempts) == 1 and not list((root / "sessions").glob("*/complete.json"))
    with pytest.raises(collection.CollectionError, match="INCOMPLETE.*REQUIRES_REVIEW"):
        collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, day, 17, tzinfo=TZ),
            calendar=Calendar(), runner=lambda *args, **kwargs: pytest.fail("must not mutate PAE twice"))


def test_changed_version_rejected_before_running(tmp_path, monkeypatch):
    root = setup_collection(tmp_path, monkeypatch)
    root.mkdir(parents=True)
    (root / "identity.json").write_text(json.dumps(collection._sealed_identity({"old": True})))
    with pytest.raises(collection.CollectionError, match="IDENTITY_CHANGED"):
        collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 26, 16, tzinfo=TZ),
            calendar=Calendar(), runner=lambda *args, **kwargs: pytest.fail("version drift"))


def test_collection_rejects_naive_time():
    with pytest.raises(collection.CollectionError, match="TIMEZONE_AWARE"):
        collection.collect_daily(now=dt.datetime(2026, 10, 26, 16), calendar=Calendar())


def test_collection_lock_does_not_allow_two_collectors(tmp_path, monkeypatch):
    root = setup_collection(tmp_path, monkeypatch)
    root.mkdir(parents=True)
    with (root / ".collector.lock").open("a+b") as lock:
        collection.fcntl.flock(lock.fileno(), collection.fcntl.LOCK_EX | collection.fcntl.LOCK_NB)
        with pytest.raises(collection.CollectionError, match="ALREADY_RUNNING"):
            collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 26, 16, tzinfo=TZ),
                calendar=Calendar(), runner=lambda *args, **kwargs: pytest.fail("must not run"))


def test_native_artifact_linkage_preserves_unknown_and_rejects_projection_tamper(tmp_path, monkeypatch):
    """SYNTHETIC transport fixture, not a valid financial/model source."""
    from domain.les_v7 import report as report_module
    from infrastructure.storage.daily_candidate_archive import build_candidate, publish_snapshot
    from infrastructure.storage.report_evidence import ReportEvidenceWriter
    from tests.repositories.test_daily_candidate_archive import report as synthetic_report
    # Native model validation has its own source-to-report tests. Here the spy
    # lets this small synthetic fixture exercise serialization/linkage only.
    validations = []
    monkeypatch.setattr(report_module, "validate_report", lambda value: validations.append(value["symbol"]))
    report = synthetic_report()
    report["model_version"] = "7.0.0"
    report["envelope"]["event"]["cutoff_at"] = "2026-10-20T16:00:00+08:00"
    report["envelope"]["strategy"]["strategy_decision"] = "DATA_INSUFFICIENT"
    report["diagnostics"]["native_evidence"] = {"source_packet": {"synthetic_only": True}}
    report["diagnostics"]["rule_baseline"]["policy"]["policy_version"] = collection.R19_VERSION
    with ReportEvidenceWriter(tmp_path / "report_evidence.sqlite3") as store:
        reference = store.retain(report)
    (tmp_path / "shirong_les_v7_reports.json").write_text(json.dumps({
        "model": "low_entry_score_v7", "storage_schema": "daily-reports/v2", "research_only": True,
        "execution_enabled": False, "reports": [reference]}))
    candidate = build_candidate(report, reference, {"EREM 分數": None, "EREM 決策訊號": "HOLD", "EREM 理由": "SYNTHETIC"})
    path = publish_snapshot(tmp_path, "shirong", [candidate])
    result = collection._validate_native_outputs(tmp_path, signal_date="2026-10-20")
    assert result["scope_status"] == "PARTIAL" and validations == ["SYNTHETIC_ONLY"]
    assert candidate["candidate_state"] == "BLOCKED_DATA_INSUFFICIENT"
    candidate["score"]["final_score"] = 99.
    publish_snapshot(tmp_path, "shirong", [candidate])
    path.unlink()  # isolated synthetic fixture only
    with pytest.raises(collection.CollectionError, match="content mismatch"):
        collection._validate_native_outputs(tmp_path, signal_date="2026-10-20")


def test_universe_change_cannot_silently_complete_new_day(tmp_path, monkeypatch):
    root = setup_collection(tmp_path, monkeypatch)
    run = lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="SYNTHETIC", stderr="")
    collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 26, 16, tzinfo=TZ),
                             calendar=Calendar(), runner=run)
    (root / "universe.json").write_text(json.dumps({"symbols": ["CHANGED_SYNTHETIC_ONLY"]}))
    with pytest.raises(collection.CollectionError, match="UNIVERSE_CHANGED"):
        collection.collect_daily(collection_id="test-oos", now=dt.datetime(2026, 10, 27, 16, tzinfo=TZ),
                                 calendar=Calendar(), runner=run)
    assert not (root / "sessions" / "2026-10-27" / "complete.json").exists()
