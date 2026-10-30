"""SYNTHETIC ONLY: bounded saved-input arithmetic, not a full LES backtest."""
import json
import sqlite3

import pytest

from application.replay_contracts import canonical_json, digest
from domain.les_v7.valuation_price import evaluate_price_valuation
from domain.les_v7.valuation_recovery_r10 import evaluate_recovery_price_valuation
from examples.compare_les_r13 import compare_bundle, main
from infrastructure.storage.report_evidence import ReportEvidenceReader, ReportEvidenceWriter


def inputs():
    return dict(current_eps=8., prior_eps=4., current_revenue=1200., prior_revenue=1000.,
                current_margin=.20, prior_margin=.15, current_price=100., current_pe=20.,
                recent_pe_median=20., eps_basis="DILUTED")


def diagnostic():
    annual = evaluate_price_valuation(**inputs())
    return evaluate_recovery_price_valuation(legacy_result=annual,
        comparables=dict(status="INSUFFICIENT", reasons=["SYNTHETIC_INSUFFICIENT_HISTORY"],
                         issuer_month_count=3, peer_count=15), recovery=None)


def report(symbol="2330", valuation=None):
    valuation = diagnostic() if valuation is None else valuation
    return dict(symbol=symbol, signal_date="2026-10-14", model_version="SYNTHETIC_ONLY",
        parameter_version="SYNTHETIC_R10", envelope=dict(event=dict(cutoff_at="2026-10-14T13:30:00+08:00"),
            score=dict(raw_score=80., final_score=70., total_penalty=10.),
            strategy=dict(strategy_decision="NO_BUY")),
        diagnostics=dict(native_evidence=dict(produced_receipts=dict(VALUATION=dict(gate_id="VALUATION",
            state=valuation["status"], valuation_diagnostics=valuation,
            input_hash=digest("SYNTHETIC_INPUT"), input_ids=[digest("SYNTHETIC_INPUT_ID")],
            proof_record_ids=[digest("SYNTHETIC_PROOF_ID")],
            source_available_at="2026-10-14T13:00:00+08:00")))),
        financial_graph=dict(unrelated_originals=[dict(n=i, note="SYNTHETIC_ONLY" * 20) for i in range(100)]))


def bundle(tmp_path, reports):
    store = tmp_path / "report_evidence.sqlite3"
    with ReportEvidenceWriter(store) as writer:
        refs = [writer.retain(row) for row in reports]
    path = tmp_path / "reports.json"
    path.write_text(canonical_json(dict(model="SYNTHETIC_ONLY", reports=refs)))
    return path, refs


def test_comparison_recomputes_math_only_and_never_expands_full_financial_graph(tmp_path, monkeypatch):
    old = report()
    path, refs = bundle(tmp_path, [old])
    before = path.read_bytes()
    monkeypatch.setattr(ReportEvidenceReader, "resolve", lambda *args: pytest.fail("Full report expanded"))
    result = compare_bundle(path)
    assert path.read_bytes() == before
    assert result["summary"]["report_count"] == 1
    row = result["records"][0]
    assert row["comparison_state"] == "RECOMPUTED_ARITHMETIC"
    assert row["source"]["report_reference"] == refs[0]
    assert row["old"]["method"] == "ORIGINAL_R1_FALLBACK"
    assert row["new"]["policy_version"].endswith("R13")
    assert row["persisted_final_score"] == 70.
    assert row["new_final_score"] is None and row["new_strategy_decision"] is None
    assert result["new_buy_count"] is None and result["new_buy_small_count"] is None
    assert result["full_report_hash_verified"] is False
    assert result["financial_originals_reverified"] is False
    assert result["historical_information_set_certified"] is False
    assert row["valuation_deduction_difference"] == row["new"]["grade"]["points"] - row["old"]["grade"]["points"]


def test_supplemental_input_policy_does_not_change_arithmetic_verification(tmp_path):
    value = diagnostic()
    value.update(input_policy=dict(note="SYNTHETIC_WRAPPER"), input_policy_id=digest("WRAPPER"),
                 entry_quality_required=True, quality_dependency="RESULT_ID_BINDING")
    path, _ = bundle(tmp_path, [report(valuation=value)])
    assert compare_bundle(path)["records"][0]["comparison_state"] == "RECOMPUTED_ARITHMETIC"


def test_original_r1_and_r10_inputs_both_recompute_without_claiming_new_cutoff(tmp_path):
    original = evaluate_price_valuation(**inputs())
    path, _ = bundle(tmp_path, [report(valuation=original)])
    row = compare_bundle(path)["records"][0]
    assert row["comparison_state"] == "RECOMPUTED_ARITHMETIC"
    assert row["old"]["method"] == "V7_REPORTED_EPS_PRICE_RETURN_R1"
    assert row["cutoff_at"] == "2026-10-14T13:30:00+08:00"
    # Original R1 did not retain a comparable availability status. Do not
    # manufacture NOT_AVAILABLE to select the favorable R13 fallback.
    assert row["new"]["status"] == "UNKNOWN"
    assert row["new"]["bear_hard_check"] is None


@pytest.mark.parametrize("change", ["fair", "unknown_extra", "receipt_state"])
def test_stored_arithmetic_tampering_never_produces_new_grade(tmp_path, change):
    old = report()
    receipt = old["diagnostics"]["native_evidence"]["produced_receipts"]["VALUATION"]
    if change == "fair": receipt["valuation_diagnostics"]["scenarios"]["BASE"]["fair_price"] += 1.
    elif change == "unknown_extra": receipt["valuation_diagnostics"]["unverified_override"] = True
    else: receipt["state"] = "PASS"
    path, _ = bundle(tmp_path, [old])
    row = compare_bundle(path)["records"][0]
    assert row["comparison_state"] == "REJECTED_STORED_ARITHMETIC"
    assert row["new"] is None and row["valuation_deduction_difference"] is None


@pytest.mark.parametrize("status", ["FAIL", "UNKNOWN"])
def test_missing_readiness_inputs_preserve_every_row_and_native_verdict(tmp_path, status):
    missing = dict(policy_version="V7_RECOVERY_INDUSTRY_PRICE_RETURN_R10", status=status,
                   reasons=["SYNTHETIC_REQUIRED_SOURCE_NOT_VERIFIED"])
    path, _ = bundle(tmp_path, [report(), report("2454", valuation=missing)])
    result = compare_bundle(path)
    assert result["summary"]["report_count"] == 2
    row = result["records"][1]
    assert row["symbol"] == "2454" and row["old"]["status"] == status
    assert row["comparison_state"] == "INPUTS_UNAVAILABLE" and row["new"] is None


def test_explicit_null_receipt_is_retained_as_unknown_not_dropped_or_replaced(tmp_path):
    missing = report("3036A")
    missing["diagnostics"]["native_evidence"]["produced_receipts"]["VALUATION"] = None
    path, _ = bundle(tmp_path, [report(), missing])
    result = compare_bundle(path)
    assert result["summary"]["report_count"] == 2
    assert result["summary"]["old_valuation_grades"]["UNKNOWN"] == 1
    row = result["records"][1]
    assert row["symbol"] == "3036A" and row["comparison_state"] == "INPUTS_UNAVAILABLE"
    assert row["old"]["status"] is None and row["old"]["grade"] is None
    assert row["new"] is None and row["valuation_deduction_difference"] is None
    assert row["source"]["valuation_receipt_sha256"] == digest(None)


def test_unknown_comparables_remain_unknown_under_r13(tmp_path):
    value = evaluate_recovery_price_valuation(legacy_result=evaluate_price_valuation(**inputs()),
        comparables=dict(status="UNKNOWN", reasons=["SYNTHETIC_SOURCE_CONFLICT"]), recovery=None)
    path, _ = bundle(tmp_path, [report(valuation=value)])
    row = compare_bundle(path)["records"][0]
    assert row["comparison_state"] == "RECOMPUTED_ARITHMETIC"
    assert row["new"]["status"] == "UNKNOWN"
    assert row["new"]["grade"]["state"] == "UNKNOWN"


def test_partial_node_hash_and_metadata_binding_errors_abort(tmp_path):
    path, refs = bundle(tmp_path, [report()])
    data = json.loads(path.read_text())
    data["reports"][0]["symbol"] = "FORGED"
    path.write_text(canonical_json(data))
    with pytest.raises(ValueError, match="METADATA_BINDING_MISMATCH"):
        compare_bundle(path)
    data["reports"][0] = refs[0]
    path.write_text(canonical_json(data))
    with sqlite3.connect(tmp_path / "report_evidence.sqlite3") as connection:
        connection.execute("UPDATE report_objects SET body=? WHERE sha256=?", (b"bad-zlib", refs[0]["root"][1]))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        compare_bundle(path)


def test_public_cli_write_once_idempotency_and_conflicts(tmp_path, capsys):
    path, _ = bundle(tmp_path, [report()])
    output = tmp_path / "comparison"
    argv = ["--report-bundle", str(path), "--output-dir", str(output)]
    assert main(argv) == 0
    first = {p.name: p.read_bytes() for p in output.iterdir()}
    assert main(argv) == 0
    assert {p.name: p.read_bytes() for p in output.iterdir()} == first
    receipt = json.loads((output / "complete.json").read_text())
    value = json.loads((output / "comparison.json").read_text())
    assert receipt["comparison_id"] == value["comparison_id"]
    assert "No new LES score" in (output / "comparison.md").read_text()
    (output / "comparison.md").write_text("SYNTHETIC_TAMPER")
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        main(argv)
