"""SYNTHETIC ONLY: full native R15/R16 comparison and immutable publication."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
import json
import sqlite3

import pytest

from application.replay_contracts import digest
from domain.les_v7.contracts import timestamp
from domain.les_v7.report import validate_report
from examples import compare_les_r16 as comparison
from examples.compare_les_r13 import file_hash
from tests.test_v7_r14_comparison import fixture


def at():
    from domain.les_v7.policy_r16 import AVAILABLE_AT
    return (timestamp(AVAILABLE_AT) + timedelta(seconds=1)).isoformat()


def test_native_comparison_uses_r15_r16_with_exact_sources_account_and_full_differences():
    from domain.les_v7.policy_r15 import BASELINE_VERSION as R15
    from domain.les_v7.policy_r16 import BASELINE_VERSION as R16
    saved, filings, _ = fixture()
    before = deepcopy(saved)
    old, new, row = comparison.compare_report(saved, filings, evaluation_cutoff=at())
    assert saved == before
    for report, version in ((old, R15), (new, R16)):
        validate_report(report)
        assert report["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == version
        assert report["pae_authorization"]["live_order_authorized"] is False
        assert report["pae_authorization"]["account"] == old["pae_authorization"]["account"]
    assert row["state"] == "FULL_NATIVE_RECOMPUTED" and row["account_inputs_equal"] is True
    assert row["r15"] == comparison.business(old) and row["r16"] == comparison.business(new)
    assert row["full_report_differences"] == list(comparison.differences(old, new))
    assert row["differences"] == list(comparison.differences(row["r15"], row["r16"]))
    assert old["diagnostics"]["quality"] == new["diagnostics"]["quality"]
    assert old["diagnostics"]["native_evidence"]["source_packet"] == new["diagnostics"]["native_evidence"]["source_packet"]
    assert row["source_snapshot_cutoff"] != row["evaluation_cutoff"]
    assert row["raw_source_identity"]["financials_hash"] == row["financials_sha256"]


def test_pre_release_and_financial_vector_mismatch_are_rejected():
    from domain.les_v7.policy_r16 import AVAILABLE_AT
    saved, filings, _ = fixture()
    with pytest.raises(ValueError, match="BEFORE_POLICY_RELEASE"):
        comparison.compare_report(saved, filings, evaluation_cutoff=(
            timestamp(AVAILABLE_AT) - timedelta(microseconds=1)).isoformat())
    with pytest.raises(ValueError, match="FINANCIAL_VECTOR_HASH_REQUIRED"):
        comparison.compare_report(saved, filings[:-1], evaluation_cutoff=at())


def test_missing_saved_account_not_fabricated_and_source_packet_not_refreshed():
    saved, filings, _ = fixture()
    source = deepcopy(saved["diagnostics"]["native_evidence"]["source_packet"])
    old, new, _ = comparison.compare_report(saved, filings, evaluation_cutoff=at())
    for report in old, new:
        packet = report["diagnostics"]["native_evidence"]["source_packet"]
        assert packet["restriction_evidence"] == source["restriction_evidence"]
        assert packet["market_rows"] == source["market_rows"]
        gates = {row["gate_id"]: row["state"] for row in report["diagnostics"]["gate_receipts"]}
        assert gates["TRADING_STATUS"] == "UNKNOWN"
    saved["pae_authorization"].pop("account")
    with pytest.raises(ValueError, match="EXPLICIT_SAVED_ACCOUNT_REQUIRED"):
        comparison.compare_report(saved, filings, evaluation_cutoff=at())


def test_stale_daily_rejections_remain_blocking_after_recomputation():
    from application.low_entry_v7_daily import finalize_daily_report
    saved, filings, _ = fixture()
    copy = deepcopy(saved)
    copy.pop("pae_authorization")
    source = finalize_daily_report(copy, account=saved["pae_authorization"]["account"],
                                   daily_input_errors=["STALE_MARKET_DATA"])
    old, new, row = comparison.compare_report(source, filings, evaluation_cutoff=at())
    for report in old, new:
        assert report["envelope"]["score"]["final_score"] is None
        assert "STALE_MARKET_DATA" in report["envelope"]["strategy"]["reason_codes"]
        assert report["pae_authorization"]["authorized"] is False
        assert report["diagnostics"].get("reference_entry_route") is None
    assert row["r15"]["daily_input_errors"] == row["r16"]["daily_input_errors"]


def test_business_projection_keeps_all_new_conditional_plan_fields_and_unknown_extensions(monkeypatch):
    saved = dict(diagnostics={})
    # Exercise the added projection without depending on an in-flight policy
    # implementation. The complete native projection is checked above.
    monkeypatch.setattr(comparison.previous, "business", lambda report: dict(
        diagnostic_extensions=deepcopy(report["diagnostics"])))
    branch = dict(selected_route="EARLY_PULLBACK_FLOOR_R16", plan=dict(
        entry_basis="STRUCTURAL_CONDITIONAL_BAND", signal_close=110., entry=101.,
        entry_band=[100., 101.], risk=dict(state="UNKNOWN"), next_open_only=True),
        cap=.005, risk_budget=.0005, simulated_fill_count=0)
    saved["diagnostics"]["r16_route_evaluation"] = branch
    saved["diagnostics"]["future_r16_diagnostic"] = dict(state="UNKNOWN", no_imputation=True)
    projection = comparison.business(saved)
    assert projection["r16_route_evaluation"] == branch
    assert projection["diagnostic_extensions"]["r16_route_evaluation"] == branch
    assert projection["diagnostic_extensions"]["future_r16_diagnostic"]["state"] == "UNKNOWN"


def test_full_differences_preserve_accounts_gates_source_hashes_score_caps_and_absent_null():
    old = dict(account=dict(sha="SYNTHETIC_OLD", cash=100.), gates=dict(state="UNKNOWN"),
        plan=dict(entry=101., stop=99., proof="SYNTHETIC_A"), score=62., cap=.005, absent=None)
    new = dict(account=dict(sha="SYNTHETIC_NEW", cash=90.), gates=dict(state="FAIL"),
        plan=dict(entry=102., stop=98., proof="SYNTHETIC_B"), score=61., cap=0., extra=None)
    changes = {row["path"]: row for row in comparison.differences(old, new)}
    assert set(changes) == {"account.sha", "account.cash", "gates.state", "plan.entry", "plan.stop",
                            "plan.proof", "score", "cap", "absent", "extra"}
    assert changes["absent"]["before_present"] is True and changes["absent"]["after_present"] is False
    assert changes["extra"]["before_present"] is False and changes["extra"]["after_present"] is True
    assert list(comparison.differences(old, deepcopy(old))) == []


def test_program_capture_binds_all_helper_bytes(tmp_path):
    paths = [tmp_path / f"SYNTHETIC_{i}.py" for i in range(4)]
    for i, path in enumerate(paths): path.write_text(f"# SYNTHETIC ONLY {i}\n")
    expected = comparison.program_bindings(paths)
    comparison.check_program_bindings(expected, paths)
    paths[2].write_text("# SYNTHETIC MODIFICATION\n")
    with pytest.raises(ValueError, match="PROGRAM_CHANGED"):
        comparison.check_program_bindings(expected, paths)
    assert len(comparison.PROGRAM_PATHS) == 4


def test_summary_retains_whole_scope_unknown_and_conditional_signals_are_not_fills():
    result = dict(evaluation_cutoff="2030-01-01T00:00:00Z", source_bundle_sha256="SYNTHETIC_ONLY",
        summary=dict(total=3, states=dict(FULL_NATIVE_RECOMPUTED=2, SOURCE_OR_RECOMPUTATION_GAP=1),
                     r15_decisions=dict(NO_BUY=2), r16_decisions=dict(NO_BUY=1, BUY_SMALL=1)))
    output = comparison.markdown(result)
    assert "缺口：1 檔" in output and "| BUY_SMALL | 0 | 1 |" in output
    assert "過期仍 UNKNOWN" in output and "條件式 BUY_SMALL 是尚未成交" in output
    assert "日內高低價觸帶不作成交" in output and "本次未模擬任何成交" in output
    assert "不是歷史當時部署、獲利或成交證明" in output
    assert "並非自足可攜原始資料集" in output


def command_args(bundle, destination):
    return ["--report-bundle", str(bundle), "--expected-bundle-sha256", file_hash(bundle),
            "--fundamental-db", str(bundle.parent / "synthetic.sqlite3"),
            "--evaluation-cutoff", "2030-01-01T00:00:00Z", "--output-dir", str(destination)]


def test_existing_attempt_never_appended_or_overwritten(tmp_path):
    bundle = tmp_path / "synthetic.json"
    bundle.write_text(json.dumps(dict(reports=[dict(symbol="SYNTHETIC")])))
    destination = tmp_path / "old-attempt"
    destination.mkdir()
    marker = destination / "preserve.json"
    marker.write_text('{"synthetic_only":true}')
    with pytest.raises(FileExistsError): comparison.main(command_args(bundle, destination))
    assert marker.read_text() == '{"synthetic_only":true}'
    assert not (destination / "complete.json").exists()


def test_bad_source_hash_rejected_before_output_creation(tmp_path):
    bundle = tmp_path / "synthetic.json"
    bundle.write_text(json.dumps(dict(reports=[dict(symbol="SYNTHETIC")])))
    destination = tmp_path / "not-created"
    args = command_args(bundle, destination)
    args[args.index("--expected-bundle-sha256") + 1] = "0" * 64
    with pytest.raises(ValueError, match="SOURCE_BUNDLE_HASH_MISMATCH"): comparison.main(args)
    assert not destination.exists()


@pytest.mark.parametrize("symbols", [[], ["SYNTHETIC", "SYNTHETIC"]])
def test_scope_never_drops_or_deduplicates_stocks_silently(tmp_path, symbols):
    bundle = tmp_path / "synthetic.json"
    bundle.write_text(json.dumps(dict(reports=[dict(symbol=symbol) for symbol in symbols])))
    destination = tmp_path / "not-created"
    with pytest.raises(ValueError, match="SCOPE_EMPTY_OR_DUPLICATE"):
        comparison.main(command_args(bundle, destination))
    assert not destination.exists()


def test_result_store_integrity_and_identity_are_captured(tmp_path):
    path = tmp_path / "synthetic.sqlite3"
    with sqlite3.connect(path) as connection: connection.execute("CREATE TABLE synthetic(value)")
    result = comparison._store_receipt(path)
    assert result == dict(name=path.name, sha256=file_hash(path), size_bytes=path.stat().st_size,
                          quick_check="ok", checkpoint="CLOSED_WRITER")


def mocked_job(monkeypatch, *, mismatch=False, modification=False):
    """SYNTHETIC plumbing; native integration is covered separately above."""
    @dataclass
    class Versions:
        code: str = "SYNTHETIC_CODE"
        parameters: str = "SYNTHETIC_PARAMETERS"
        model: str = "SYNTHETIC_MODEL"
    class Guard:
        def __init__(self): self.versions = Versions()
        def check(self):
            if modification: raise ValueError("REPLAY_VERSION_CHANGED_DURING_JOB")
            return self.versions
        def retain(self, objects):
            return {key: objects.put_json(dict(synthetic_only=True, kind=key))
                    for key in ("source", "environment", "parameters")}
    class Work:
        def check_version(self): return Versions(code="SYNTHETIC_CHANGED" if mismatch else "SYNTHETIC_CODE")
    @contextmanager
    def opened(**kwargs): yield Work()
    class Restorer:
        def __init__(self, *args): self.counts = {}; self.closed = False
        def restore(self, value): raise ValueError("SYNTHETIC_SOURCE_NOT_VERIFIED")
        def close(self): self.closed = True
    monkeypatch.setattr(comparison, "ResearchVersionGuard", Guard)
    monkeypatch.setattr(comparison, "open_daily_work", opened)
    monkeypatch.setattr(comparison.source_runtime, "HashMatchedFilings", Restorer)
    monkeypatch.setattr(comparison, "validate_report", lambda value: None)
    monkeypatch.setattr(comparison, "iter_report_bundle", lambda *args: iter([
        dict(symbol="SYNTHETIC1", synthetic_only=True), dict(symbol="SYNTHETIC2", synthetic_only=True)]))


def test_gaps_preserved_runtime_captured_and_completion_published_last(tmp_path, monkeypatch):
    bundle = tmp_path / "synthetic.json"
    bundle.write_text(json.dumps(dict(reports=[dict(symbol="SYNTHETIC1"), dict(symbol="SYNTHETIC2")])))
    mocked_job(monkeypatch)
    destination = tmp_path / "new-attempt"
    assert comparison.main(command_args(bundle, destination)) == 0
    result = json.loads((destination / "comparison.json").read_text())
    assert result["universe_symbols"] == ["SYNTHETIC1", "SYNTHETIC2"]
    assert len(result["records"]) == 2
    assert all(row["state"] == "SOURCE_OR_RECOMPUTATION_GAP" for row in result["records"])
    assert all(row["error"] == "SYNTHETIC_SOURCE_NOT_VERIFIED" for row in result["records"])
    assert result["summary"]["r15_decisions"] == result["summary"]["r16_decisions"] == {}
    receipt = json.loads((destination / "complete.json").read_text())
    assert receipt["status"] == "PARTIAL_GAPS_RETAINED"
    assert receipt["comparison_sha256"] == file_hash(destination / "comparison.json")
    assert receipt["result_store"]["sha256"] == file_hash(destination / "report_evidence.sqlite3")
    runtime = json.loads((destination / "runtime-objects/manifest.json").read_text())
    assert runtime["versions"] == result["versions"] and runtime["references"] == result["runtime_references"]
    objects = comparison.ResearchObjects(destination / "runtime-objects/objects")
    for key, sha in runtime["references"].items():
        assert objects.get_json(sha) == dict(synthetic_only=True, kind=key)
    programs = json.loads((destination / "programs/manifest.json").read_text())
    assert len(programs["files"]) == 4
    for entry in programs["files"]:
        assert file_hash(destination / "programs" / entry["repository_path"]) == entry["sha256"]


@pytest.mark.parametrize("mismatch,modification", [(True, False), (False, True)])
def test_version_drift_never_publishes_complete(tmp_path, monkeypatch, mismatch, modification):
    bundle = tmp_path / "synthetic.json"
    bundle.write_text(json.dumps(dict(reports=[dict(symbol="SYNTHETIC1"), dict(symbol="SYNTHETIC2")])))
    mocked_job(monkeypatch, mismatch=mismatch, modification=modification)
    destination = tmp_path / "interrupted"
    with pytest.raises(ValueError, match="VERSION"):
        comparison.main(command_args(bundle, destination))
    assert not (destination / "complete.json").exists()
    assert not (destination / "comparison.json").exists()


def test_source_copy_receipt_binds_bundle_and_readonly_backup_bytes(tmp_path):
    bundle = tmp_path / "synthetic.json"
    bundle.write_text('{"synthetic_only":true}')
    store = tmp_path / "synthetic.sqlite3"
    store.write_bytes(b"SYNTHETIC ONLY")
    expected = dict(schema="les-report-local-sqlite-backup/v1", method="SQLITE_BACKUP_READ_ONLY_SOURCE",
        bundle_sha256=file_hash(bundle), local_bundle=str(bundle), local_store=str(store),
        local_store_sha256=file_hash(store))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(expected))
    assert comparison._copy_receipt(receipt, bundle, file_hash(bundle)) == expected
    store.write_bytes(b"SYNTHETIC MODIFIED")
    with pytest.raises(ValueError, match="SOURCE_COPY_RECEIPT_INVALID"):
        comparison._copy_receipt(receipt, bundle, file_hash(bundle))
