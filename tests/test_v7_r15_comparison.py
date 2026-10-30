"""SYNTHETIC ONLY: native R14/R15 comparison and complete branch retention."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path

import pytest

from domain.les_v7.contracts import timestamp
from domain.les_v7.report import validate_report
from examples import compare_les_r15 as comparison
from examples.compare_les_r13 import file_hash
from tests.test_v7_r14_comparison import fixture


def at():
    from domain.les_v7.policy_r15 import AVAILABLE_AT
    return (timestamp(AVAILABLE_AT) + timedelta(seconds=1)).isoformat()


def test_full_native_comparison_uses_both_policies_and_same_explicit_sources_account():
    from domain.les_v7.policy_r14 import BASELINE_VERSION as R14
    from domain.les_v7.policy_r15 import BASELINE_VERSION as R15
    saved, filings, _ = fixture()
    before = deepcopy(saved)
    old, new, record = comparison.compare_report(saved, filings, evaluation_cutoff=at())
    for report, version in ((old, R14), (new, R15)):
        validate_report(report)
        assert report["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == version
        assert report["pae_authorization"]["live_order_authorized"] is False
    assert saved == before
    assert record["state"] == "FULL_NATIVE_RECOMPUTED"
    assert record["account_inputs_equal"] is True
    assert record["r14"] == comparison.business(old)
    assert record["r15"] == comparison.business(new)
    assert old["diagnostics"]["quality"] == new["diagnostics"]["quality"]
    assert old["diagnostics"]["native_evidence"]["source_packet"] == new["diagnostics"]["native_evidence"]["source_packet"]
    assert record["r15"]["r15_route_evaluation"] == new["diagnostics"]["r15_route_evaluation"]
    assert record["r15"]["reference_entry_route"] == new["diagnostics"].get("reference_entry_route")
    routes = new["diagnostics"]["r15_route_evaluation"]
    selected = routes["selected_route"].lower()
    for key, value in routes[selected]["score"].items():
        assert new["envelope"]["score"][key] == value
    if selected == "core":
        assert new["envelope"]["strategy"]["strategy_decision"] == "BUY"
    else:
        assert new["envelope"]["strategy"]["strategy_decision"] != "BUY"
    assert record["full_report_differences"] == list(comparison.differences(old, new))
    assert record["raw_source_identity"]["financials_hash"] == record["financials_sha256"]
    assert record["raw_source_identity"]["source_cutoff"] != record["evaluation_cutoff"]


def test_pre_release_evaluation_and_mismatched_financial_vector_rejected():
    from domain.les_v7.policy_r15 import AVAILABLE_AT
    saved, filings, _ = fixture()
    with pytest.raises(ValueError, match="BEFORE_POLICY_RELEASE"):
        comparison.compare_report(saved, filings, evaluation_cutoff=(
            timestamp(AVAILABLE_AT) - timedelta(microseconds=1)).isoformat())
    with pytest.raises(ValueError, match="FINANCIAL_VECTOR_HASH_REQUIRED"):
        comparison.compare_report(saved, filings[:-1], evaluation_cutoff=at())


def test_saved_daily_rejections_and_stale_evidence_are_not_promoted_to_pass():
    from application.low_entry_v7_daily import finalize_daily_report
    saved, filings, _ = fixture()
    source = deepcopy(saved)
    source.pop("pae_authorization")
    source = finalize_daily_report(source, account=saved["pae_authorization"]["account"],
                                   daily_input_errors=["STALE_MARKET_DATA"])
    old, new, record = comparison.compare_report(source, filings, evaluation_cutoff=at())
    for report in old, new:
        assert "STALE_MARKET_DATA" in report["envelope"]["strategy"]["reason_codes"]
        assert report["pae_authorization"]["authorized"] is False
        assert report["envelope"]["score"]["final_score"] is None
        gates = {gate["gate_id"]: gate for gate in report["diagnostics"]["gate_receipts"]}
        assert gates["TRADING_STATUS"]["state"] == "UNKNOWN"
    assert record["r14"]["daily_input_errors"] == record["r15"]["daily_input_errors"]


def test_missing_explicit_saved_account_is_never_fabricated():
    saved, filings, _ = fixture()
    saved["pae_authorization"].pop("account")
    with pytest.raises(ValueError, match="EXPLICIT_SAVED_ACCOUNT_REQUIRED"):
        comparison.compare_report(saved, filings, evaluation_cutoff=at())


def test_projection_preserves_both_route_scores_admissions_risks_and_unknown_new_keys():
    saved, _, _ = fixture()
    before = deepcopy(saved)
    routes = dict(selected_route="NONE", source_state="UNKNOWN", pattern_state="KNOWN_ABSENT",
        core=dict(score=dict(raw_score=73., final_score=58.), admission=dict(core_buy_allowed=False,
            core_small_allowed=False, core_buy_cap=0., core_small_cap=0.), risk=dict(state="UNKNOWN")),
        tactical=dict(score=dict(raw_score=73., final_score=65.), admission=dict(tactical_small_allowed=False,
            tactical_total_cap=.0025, tactical_risk_fraction=.0005), risk=dict(state="FAIL")))
    saved["diagnostics"]["r15_route_evaluation"] = routes
    saved["diagnostics"]["reference_entry_route"] = None
    saved["diagnostics"]["future_computational_field"] = dict(state="UNKNOWN", value=None, points=0.)
    result = comparison.business(saved)
    assert result["r15_route_evaluation"] == routes
    assert result["reference_entry_route"] is None
    assert result["diagnostic_extensions"]["r15_route_evaluation"] == routes
    assert result["diagnostic_extensions"]["future_computational_field"] == dict(
        state="UNKNOWN", value=None, points=0.)
    for key, value in before.items():
        if key != "diagnostics": assert saved[key] == value
    assert saved["diagnostics"]["r15_route_evaluation"] == routes


def test_full_difference_keeps_source_ids_cash_hard_gates_and_both_route_fields():
    before = dict(account=dict(cash=10., account_hash="SYNTHETIC_OLD"), gate=dict(state="UNKNOWN"),
                  tactical=dict(score=62., cap=.0025), source=dict(proof="SYNTHETIC_OLD"), absent=None)
    after = dict(account=dict(cash=0., account_hash="SYNTHETIC_NEW"), gate=dict(state="FAIL"),
                 tactical=dict(score=58., cap=0.), source=dict(proof="SYNTHETIC_NEW"), extra=None)
    changes = {row["path"]: row for row in comparison.differences(before, after)}
    assert set(changes) == {"account.cash", "account.account_hash", "gate.state", "tactical.score",
                            "tactical.cap", "source.proof", "absent", "extra"}
    assert changes["absent"]["before_present"] is True and changes["absent"]["after_present"] is False
    assert changes["extra"]["before_present"] is False and changes["extra"]["after_present"] is True
    assert list(comparison.differences(before, deepcopy(before))) == []


def test_harness_and_all_imported_example_helpers_are_content_bound(tmp_path):
    files = [tmp_path / name for name in ("new.py", "old.py", "helper.py")]
    for i, path in enumerate(files): path.write_text(f"# SYNTHETIC ONLY {i}\n")
    expected = comparison.program_bindings(files)
    comparison.check_program_bindings(expected, files)
    files[1].write_text("# SYNTHETIC MODIFICATION\n")
    with pytest.raises(ValueError, match="PROGRAM_CHANGED"):
        comparison.check_program_bindings(expected, files)


def test_human_summary_retains_whole_scope_gaps_and_no_performance_claims():
    result = dict(evaluation_cutoff="2030-01-01T00:00:00Z", source_bundle_sha256="SYNTHETIC_ONLY",
        summary=dict(total=3, states=dict(FULL_NATIVE_RECOMPUTED=2, SOURCE_OR_RECOMPUTATION_GAP=1),
                     r14_decisions=dict(NO_BUY=2), r15_decisions=dict(WATCH_ONLY=1, BUY_SMALL=1)))
    output = comparison.markdown(result)
    assert "缺口：1 檔" in output and "| BUY_SMALL | 0 | 1 |" in output
    assert "core／tactical" in output and "過期仍 UNKNOWN" in output
    assert "不是歷史當時部署、獲利或成交證明" in output
    assert "並非自足可攜原始資料集" in output


def command_args(bundle, destination):
    return ["--report-bundle", str(bundle), "--expected-bundle-sha256", file_hash(bundle),
            "--fundamental-db", str(bundle.parent / "synthetic.sqlite3"),
            "--evaluation-cutoff", "2030-01-01T00:00:00Z", "--output-dir", str(destination)]


def test_cli_never_appends_or_overwrites_an_existing_attempt(tmp_path):
    bundle = tmp_path / "synthetic.json"
    bundle.write_text(json.dumps(dict(reports=[dict(symbol="SYNTHETIC")])))
    destination = tmp_path / "already-present"
    destination.mkdir()
    marker = destination / "preserve.json"
    marker.write_text('{"synthetic_only":true}')
    before = marker.read_bytes()
    with pytest.raises(FileExistsError):
        comparison.main(command_args(bundle, destination))
    assert marker.read_bytes() == before
    assert not (destination / "complete.json").exists()


def test_cli_retains_source_gaps_and_only_publishes_partial_receipt(tmp_path, monkeypatch):
    """SYNTHETIC plumbing only, not a substitute for native integration above."""
    bundle = tmp_path / "synthetic.json"
    bundle.write_text(json.dumps(dict(reports=[dict(symbol="SYNTHETIC1"), dict(symbol="SYNTHETIC2")])))
    saved = [dict(symbol="SYNTHETIC1", synthetic_only=True), dict(symbol="SYNTHETIC2", synthetic_only=True)]
    class Restorer:
        def __init__(self, *args): self.counts = {}; self.closed = False
        def restore(self, value): raise ValueError("SYNTHETIC_SOURCE_NOT_VERIFIED")
        def close(self): self.closed = True
    @dataclass
    class Versions:
        code: str = "SYNTHETIC"
        parameters: str = "SYNTHETIC"
        model: str = "SYNTHETIC"
    class Work:
        def check_version(self): return Versions()
    @contextmanager
    def opened(**kwargs): yield Work()
    monkeypatch.setattr(comparison.previous, "HashMatchedFilings", Restorer)
    monkeypatch.setattr(comparison, "iter_report_bundle", lambda *args: iter(saved))
    monkeypatch.setattr(comparison, "validate_report", lambda value: None)
    monkeypatch.setattr(comparison, "open_daily_work", opened)
    destination = tmp_path / "new-attempt"
    assert comparison.main(command_args(bundle, destination)) == 0
    result = json.loads((destination / "comparison.json").read_text())
    assert result["universe_symbols"] == ["SYNTHETIC1", "SYNTHETIC2"]
    assert len(result["records"]) == 2
    assert all(row["state"] == "SOURCE_OR_RECOMPUTATION_GAP" for row in result["records"])
    assert all(row["error"] == "SYNTHETIC_SOURCE_NOT_VERIFIED" for row in result["records"])
    assert result["summary"]["r14_decisions"] == result["summary"]["r15_decisions"] == {}
    assert json.loads((destination / "complete.json").read_text())["status"] == "PARTIAL_GAPS_RETAINED"
    manifest = json.loads((destination / "programs/manifest.json").read_text())
    assert len(manifest["files"]) == 3
    for entry in manifest["files"]:
        assert file_hash(destination / "programs" / entry["repository_path"]) == entry["sha256"]
