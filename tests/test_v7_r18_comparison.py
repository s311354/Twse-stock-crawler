"""SYNTHETIC ONLY: native comparison identity and durable resumable publication."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
import fcntl
import json

import pytest

from application.replay_contracts import digest
from domain.les_v7.contracts import timestamp
from domain.les_v7.report import validate_report
from examples import compare_les_r18 as comparison
from examples.compare_les_r13 import file_hash
from infrastructure.storage.report_evidence import ReportEvidenceWriter
from tests.test_v7_r14_comparison import fixture


def at():
    from domain.les_v7.policy_r18 import AVAILABLE_AT
    return (timestamp(AVAILABLE_AT) + timedelta(seconds=1)).isoformat()


def test_native_comparison_exact_raw_sources_account_costs_and_all_differences():
    from domain.les_v7.policy_r17 import BASELINE_VERSION as R17
    from domain.les_v7.policy_r18 import BASELINE_VERSION as R18
    saved, filings, _ = fixture()
    original = deepcopy(saved)
    old, new, row = comparison.compare_report(saved, filings, evaluation_cutoff=at())
    assert saved == original
    for report, version in ((old, R17), (new, R18)):
        validate_report(report)
        assert report["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == version
        assert report["pae_authorization"]["live_order_authorized"] is False
        assert report["pae_authorization"]["account"] == old["pae_authorization"]["account"]
        assert timestamp(report["envelope"]["event"]["cutoff_at"]) == timestamp(at())
    assert row["r17"] == comparison.business(old) and row["r18"] == comparison.business(new)
    assert row["full_report_differences"] == list(comparison.differences(old, new))
    assert row["differences"] == list(comparison.differences(row["r17"], row["r18"]))
    assert old["diagnostics"]["quality"] == new["diagnostics"]["quality"]
    assert old["diagnostics"]["native_evidence"]["source_packet"] == new["diagnostics"]["native_evidence"]["source_packet"]
    assert row["common_cost_inputs_sha256"] == digest(row["common_cost_inputs"])
    assert row["common_cost_inputs"]["reference_shares"] == 1
    assert row["common_cost_inputs"]["cost_policy"]["buy_rate"] > 0
    assert row["common_cost_inputs"]["price_grid"]
    assert row["raw_source_identity"]["financials_hash"] == row["financials_sha256"]
    assert row["source_snapshot_cutoff"] != row["evaluation_cutoff"]


def test_missing_account_financial_vector_and_pre_release_cannot_be_repaired_by_comparator():
    from domain.les_v7.policy_r18 import AVAILABLE_AT
    saved, filings, _ = fixture()
    with pytest.raises(ValueError, match="BEFORE_POLICY_RELEASE"):
        comparison.compare_report(saved, filings, evaluation_cutoff=(timestamp(AVAILABLE_AT) - timedelta(seconds=1)).isoformat())
    with pytest.raises(ValueError, match="FINANCIAL_VECTOR_HASH_REQUIRED"):
        comparison.compare_report(saved, filings[:-1], evaluation_cutoff=at())
    saved["pae_authorization"].pop("account")
    with pytest.raises(ValueError, match="EXPLICIT_SAVED_ACCOUNT_REQUIRED"):
        comparison.compare_report(saved, filings, evaluation_cutoff=at())


def test_frozen_restriction_not_refreshed_and_unknown_not_fabricated_pass():
    saved, filings, _ = fixture()
    source = deepcopy(saved["diagnostics"]["native_evidence"]["source_packet"])
    old, new, _ = comparison.compare_report(saved, filings, evaluation_cutoff=at())
    for report in old, new:
        packet = report["diagnostics"]["native_evidence"]["source_packet"]
        assert packet["restriction_evidence"] == source["restriction_evidence"]
        assert packet["market_rows"] == source["market_rows"]
        gates = {gate["gate_id"]: gate["state"] for gate in report["diagnostics"]["gate_receipts"]}
        assert gates["TRADING_STATUS"] == "UNKNOWN"
        assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}


def test_business_and_full_diff_do_not_hide_new_plans_scores_or_safety(monkeypatch):
    monkeypatch.setattr(comparison.previous, "business", lambda report: dict(
        diagnostic_extensions=deepcopy(report["diagnostics"])))
    source = dict(diagnostics=dict(r18_route_evaluation=dict(
        selected_route="SUPPORT_RESISTANCE_LIMIT_R18", plan=dict(entry=101., stop=99., target=105.),
        conditional=True, maximum_weight=.01, no_fill_without_matching_price=True), future_branch=dict(state="UNKNOWN")))
    output = comparison.business(source)
    assert output["r18_route_evaluation"] == source["diagnostics"]["r18_route_evaluation"]
    assert output["diagnostic_extensions"]["future_branch"]["state"] == "UNKNOWN"
    old = dict(account=dict(cash=100., hash="SYNTHETIC_A"), quality="UNKNOWN", score=65., cap=.01,
               plan=dict(stop=99.), absent=None)
    new = dict(account=dict(cash=90., hash="SYNTHETIC_B"), quality="FAIL", score=60., cap=0.,
               plan=dict(stop=98.), extra=None)
    changes = {row["path"]: row for row in comparison.differences(old, new)}
    assert set(changes) == {"account.cash", "account.hash", "quality", "score", "cap", "plan.stop", "absent", "extra"}
    assert changes["absent"]["before_present"] is True and changes["absent"]["after_present"] is False
    assert changes["extra"]["before_present"] is False and changes["extra"]["after_present"] is True


def test_cost_identity_binds_arithmetic_not_policy_publication_clock():
    source = dict(diagnostics=dict(rule_baseline=dict(policy=dict(reference_policy=dict(
        reference_shares=1, cost_path_policy=dict(instrument_type="ORDINARY_STOCK", market="TWSE",
        price_grid="SYNTHETIC_GRID", coordinate_policy="SYNTHETIC_UNADJUSTED", slippage_bps=0.,
        available_at="2030-01-01T00:00:00Z", cost_policy=dict(available_at="2030-01-01T00:00:00Z",
        buy_rate=.001425, sell_rate=.004425, minimum_buy_fee=0., minimum_sell_fee=0., rounding="ROUND_HALF_UP")))))))
    changed = deepcopy(source)
    path = changed["diagnostics"]["rule_baseline"]["policy"]["reference_policy"]["cost_path_policy"]
    path["available_at"] = path["cost_policy"]["available_at"] = "2030-01-02T00:00:00Z"
    assert comparison._cost_operands(changed) == comparison._cost_operands(source)
    path["cost_policy"]["buy_rate"] = 0.
    assert comparison._cost_operands(changed) != comparison._cost_operands(source)


def test_summary_preserves_scope_and_disclaims_fills_deployment_profit():
    result = dict(evaluation_cutoff="2030-01-01T00:00:00Z", source_bundle_sha256="SYNTHETIC",
        summary=dict(total=3, states=dict(FULL_NATIVE_RECOMPUTED=2, SOURCE_OR_RECOMPUTATION_GAP=1),
                     r17_decisions=dict(NO_BUY=2), r18_decisions=dict(NO_BUY=1, BUY_SMALL=1)))
    output = comparison.markdown(result)
    assert "缺口：1 檔" in output and "| BUY_SMALL | 0 | 1 |" in output
    assert "價格未符合不得成交" in output and "本次未模擬任何成交" in output
    assert "不是歷史當時部署、獲利或成交證明" in output
    assert "並非自足可攜原始資料集" in output


def test_resumable_rows_require_exact_source_hash_prefix_chain_and_scope(tmp_path):
    refs = [dict(symbol="SYNTHETIC1", report_sha256="a" * 64), dict(symbol="SYNTHETIC2", report_sha256="b" * 64)]
    row = dict(symbol="SYNTHETIC1", original_report_sha256="a" * 64, state="SOURCE_OR_RECOMPUTATION_GAP", error="SYNTHETIC")
    receipt = comparison._row_receipt(row, job_id="SYNTHETIC_JOB", index=0, previous_sha256="SYNTHETIC_JOB")
    comparison.write_once(tmp_path / "rows/0000.json", receipt)
    loaded, chain = comparison._load_rows(tmp_path, job_id="SYNTHETIC_JOB", references=refs)
    assert loaded == [row] and chain == receipt["row_sha256"]
    with pytest.raises(ValueError, match="ROW_IDENTITY_MISMATCH"):
        comparison._load_rows(tmp_path, job_id="SYNTHETIC_OTHER_JOB", references=refs)
    changed_refs = deepcopy(refs)
    changed_refs[0]["report_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="ROW_IDENTITY_MISMATCH"):
        comparison._load_rows(tmp_path, job_id="SYNTHETIC_JOB", references=changed_refs)
    comparison.write_once(tmp_path / "rows/0002.json", receipt)
    with pytest.raises(ValueError, match="ROW_SEQUENCE_INVALID"):
        comparison._load_rows(tmp_path, job_id="SYNTHETIC_JOB", references=refs)


def command_args(bundle, destination, fundamental):
    return ["--report-bundle", str(bundle), "--expected-bundle-sha256", file_hash(bundle),
        "--expected-report-store-sha256", file_hash(bundle.parent / "report_evidence.sqlite3"),
        "--fundamental-db", str(fundamental), "--expected-fundamental-sha256", file_hash(fundamental),
        "--evaluation-cutoff", "2030-01-01T00:00:00Z", "--expected-count", "2", "--output-dir", str(destination)]


def synthetic_sources(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    values = [dict(symbol=f"SYNTHETIC{i}", signal_date="2029-12-31", model_version="SYNTHETIC",
                   parameter_version="SYNTHETIC", synthetic_only=True) for i in (1, 2)]
    with ReportEvidenceWriter(source / "report_evidence.sqlite3") as writer:
        refs = [writer.retain(value) for value in values]
    bundle = source / "synthetic.json"
    bundle.write_text(json.dumps(dict(reports=refs)))
    fundamental = source / "synthetic-fundamental.sqlite3"
    fundamental.write_bytes(b"SYNTHETIC MOCKED DATABASE ONLY")
    return bundle, fundamental, values


def mocked_job(monkeypatch, *, interrupt_second=False, gap=False):
    @dataclass
    class Versions:
        code: str = "SYNTHETIC_CODE"
        parameters: str = "SYNTHETIC_PARAMETERS"
        model: str = "SYNTHETIC_MODEL"
    class Guard:
        def __init__(self): self.versions = Versions()
        def check(self): return self.versions
        def retain(self, objects):
            return {key: objects.put_json(dict(synthetic_only=True, kind=key))
                    for key in ("source", "environment", "parameters")}
    class Work:
        def check_version(self): return Versions()
    @contextmanager
    def opened(**kwargs): yield Work()
    class Restorer:
        def __init__(self, *args): self.counts = {}
        def restore(self, value):
            if gap: raise ValueError("SYNTHETIC_SOURCE_GAP")
            return [], dict(synthetic_only=True)
        def close(self): pass
    calls, control = [], dict(interrupt=interrupt_second)
    def compare(saved, filings, *, evaluation_cutoff):
        calls.append(saved["symbol"])
        if control["interrupt"] and saved["symbol"] == "SYNTHETIC2":
            raise RuntimeError("SYNTHETIC_INTERRUPTION")
        account = dict(value=1000000., available_cash=1000000., positions=[], as_of=evaluation_cutoff)
        def report(version):
            return dict(symbol=saved["symbol"], signal_date=saved["signal_date"], model_version="SYNTHETIC",
                parameter_version=version, synthetic_only=True, business=dict(strategy=dict(strategy_decision="NO_BUY"),
                synthetic_only=True, version=version), diagnostics={},
                envelope=dict(event=dict(cutoff_at=evaluation_cutoff)),
                pae_authorization=dict(account=account, account_sha256=digest(account), live_order_authorized=False))
        old, new = report("SYNTHETIC_R17"), report("SYNTHETIC_R18")
        before, after = old["business"], new["business"]
        return old, new, dict(symbol=saved["symbol"], state="FULL_NATIVE_RECOMPUTED",
            original_report_sha256=digest(saved), evaluation_cutoff=evaluation_cutoff,
            source_snapshot_cutoff="2029-12-31T13:30:00+08:00", calendar_source_present=False,
            evaluation_account_sha256=digest(account), r17=before, r18=after,
            differences=list(comparison.differences(before, after)),
            full_report_differences=list(comparison.differences(old, new)))
    monkeypatch.setattr(comparison, "ResearchVersionGuard", Guard)
    monkeypatch.setattr(comparison, "open_daily_work", opened)
    monkeypatch.setattr(comparison.source_runtime, "HashMatchedFilings", Restorer)
    monkeypatch.setattr(comparison, "validate_report", lambda value: None)
    monkeypatch.setattr(comparison, "verify_report_valuation_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparison, "business", lambda report: deepcopy(report["business"]))
    monkeypatch.setattr(comparison, "compare_report", compare)
    return calls, control


def test_full_scope_native_publication_is_idempotent_and_never_authorizes_orders(tmp_path, monkeypatch):
    bundle, fundamental, _ = synthetic_sources(tmp_path)
    calls, _ = mocked_job(monkeypatch)
    destination = tmp_path / "attempt"
    args = command_args(bundle, destination, fundamental)
    assert comparison.main(args) == 0
    result = json.loads((destination / "comparison.json").read_text())
    assert result["summary"]["states"] == dict(FULL_NATIVE_RECOMPUTED=2)
    assert result["summary"]["r17_decisions"] == result["summary"]["r18_decisions"] == dict(NO_BUY=2)
    assert result["order_authority"] is False and result["simulated_fill_count"] == 0
    before = {path: file_hash(destination / path) for path in ("comparison.json", "comparison.md", "complete.json", "rows/0000.json")}
    with pytest.raises(FileExistsError): comparison.main(args)
    assert comparison.main([*args, "--resume"]) == 0
    assert calls == ["SYNTHETIC1", "SYNTHETIC2"]
    assert before == {path: file_hash(destination / path) for path in before}


def test_committed_row_survives_interruption_and_restart_does_not_recompute_it(tmp_path, monkeypatch):
    bundle, fundamental, _ = synthetic_sources(tmp_path)
    calls, control = mocked_job(monkeypatch, interrupt_second=True)
    destination = tmp_path / "attempt"
    args = command_args(bundle, destination, fundamental)
    with pytest.raises(RuntimeError, match="SYNTHETIC_INTERRUPTION"): comparison.main(args)
    assert (destination / "rows/0000.json").exists() and not (destination / "rows/0001.json").exists()
    assert not (destination / "complete.json").exists()
    original_row_hash = file_hash(destination / "rows/0000.json")
    control["interrupt"] = False
    assert comparison.main([*args, "--resume"]) == 0
    assert calls == ["SYNTHETIC1", "SYNTHETIC2", "SYNTHETIC2"]
    assert file_hash(destination / "rows/0000.json") == original_row_hash
    assert json.loads((destination / "complete.json").read_text())["status"] == "COMPLETE"


@pytest.mark.parametrize("stage", ["row", "completion"])
def test_interrupt_after_cas_commit_or_result_publication_recovers_without_overwrite(tmp_path, monkeypatch, stage):
    bundle, fundamental, _ = synthetic_sources(tmp_path)
    calls, _ = mocked_job(monkeypatch)
    destination = tmp_path / "attempt"
    args = command_args(bundle, destination, fundamental)
    write_once = comparison.write_once
    failed = False
    def publish(path, value):
        nonlocal failed
        if not failed and ((stage == "row" and path.name == "0000.json")
                           or (stage == "completion" and path.name == "complete.json")):
            failed = True
            raise RuntimeError("SYNTHETIC_PUBLICATION_INTERRUPTION")
        return write_once(path, value)
    monkeypatch.setattr(comparison, "write_once", publish)
    with pytest.raises(RuntimeError, match="PUBLICATION_INTERRUPTION"): comparison.main(args)
    assert (destination / "report_evidence.sqlite3").exists()
    assert not (destination / "complete.json").exists()
    prior_result = file_hash(destination / "comparison.json") if stage == "completion" else None
    assert comparison.main([*args, "--resume"]) == 0
    assert calls.count("SYNTHETIC1") == (2 if stage == "row" else 1)
    if prior_result is not None: assert file_hash(destination / "comparison.json") == prior_result


def test_gaps_preserved_and_cannot_be_promoted_to_complete(tmp_path, monkeypatch):
    bundle, fundamental, _ = synthetic_sources(tmp_path)
    mocked_job(monkeypatch, gap=True)
    destination = tmp_path / "attempt"
    args = command_args(bundle, destination, fundamental)
    assert comparison.main(args) == 0
    result = json.loads((destination / "comparison.json").read_text())
    assert result["summary"]["states"] == dict(SOURCE_OR_RECOMPUTATION_GAP=2)
    assert result["summary"]["r17_decisions"] == result["summary"]["r18_decisions"] == {}
    assert json.loads((destination / "complete.json").read_text())["status"] == "PARTIAL_GAPS_RETAINED"


@pytest.mark.parametrize("which", ["bundle", "store", "fundamental"])
def test_changed_source_pin_rejected_before_output(tmp_path, monkeypatch, which):
    bundle, fundamental, _ = synthetic_sources(tmp_path)
    mocked_job(monkeypatch)
    destination = tmp_path / "not-created"
    args = command_args(bundle, destination, fundamental)
    target = {"bundle": bundle, "store": bundle.parent / "report_evidence.sqlite3", "fundamental": fundamental}[which]
    target.write_bytes(target.read_bytes() + b"SYNTHETIC_CORRUPTION")
    with pytest.raises(ValueError, match="HASH_MISMATCH"): comparison.main(args)
    assert not destination.exists()


def test_resume_rejects_scope_code_or_saved_result_mutation(tmp_path, monkeypatch):
    bundle, fundamental, _ = synthetic_sources(tmp_path)
    mocked_job(monkeypatch)
    destination = tmp_path / "attempt"
    args = command_args(bundle, destination, fundamental)
    assert comparison.main(args) == 0
    changed = list(args)
    changed[changed.index("--evaluation-cutoff") + 1] = "2030-01-02T00:00:00Z"
    with pytest.raises(ValueError, match="JOB_IDENTITY_MISMATCH"): comparison.main([*changed, "--resume"])
    row_path = destination / "rows/0000.json"
    row = json.loads(row_path.read_text())
    row["payload"]["row"]["r18"]["strategy"]["strategy_decision"] = "BUY_SMALL"
    row_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="ROW_IDENTITY_MISMATCH"): comparison.main([*args, "--resume"])
    assert json.loads((destination / "complete.json").read_text())["status"] == "COMPLETE"
    # An old receipt remains immutable, but corrupted content cannot be read as
    # a verified completion by the public resume entry point.


def test_exact_universe_count_and_new_output_directory_required(tmp_path, monkeypatch):
    bundle, fundamental, _ = synthetic_sources(tmp_path)
    mocked_job(monkeypatch)
    destination = tmp_path / "not-created"
    args = command_args(bundle, destination, fundamental)
    args[args.index("--expected-count") + 1] = "436"
    with pytest.raises(ValueError, match="EXPECTED_SCOPE_MISMATCH"): comparison.main(args)
    assert not destination.exists()


def test_resume_revalidates_full_reports_even_if_row_hash_chain_was_recomputed(tmp_path, monkeypatch):
    bundle, fundamental, _ = synthetic_sources(tmp_path)
    mocked_job(monkeypatch)
    destination = tmp_path / "attempt"
    args = command_args(bundle, destination, fundamental)
    assert comparison.main(args) == 0
    path = destination / "rows/0001.json"
    receipt = json.loads(path.read_text())
    receipt["payload"]["row"]["r18"]["strategy"]["strategy_decision"] = "BUY_SMALL"
    receipt["row_sha256"] = digest(receipt["payload"])
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="NATIVE_RESULT_MISMATCH"):
        comparison.main([*args, "--resume"])


def test_concurrent_resume_is_nonblocking_and_does_not_truncate_lock(tmp_path, monkeypatch):
    bundle, fundamental, _ = synthetic_sources(tmp_path)
    mocked_job(monkeypatch)
    destination = tmp_path / "attempt"
    args = command_args(bundle, destination, fundamental)
    assert comparison.main(args) == 0
    lock_path = destination / ".comparison.lock"
    lock_path.write_bytes(b"SYNTHETIC LOCK METADATA")
    with lock_path.open("r+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): comparison.main([*args, "--resume"])
    assert lock_path.read_bytes() == b"SYNTHETIC LOCK METADATA"
