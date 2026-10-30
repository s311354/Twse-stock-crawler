"""SYNTHETIC ONLY: bounded technical policy changes, no market/fill claims."""
from copy import deepcopy

import pytest

from application.low_entry_v7_daily import project_report
from domain.les_v7.contracts import GateReceipt, ReferencePlan, identity, timestamp
from domain.les_v7.engine import evaluate
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (BASELINE_POLICY, LEGACY_BASELINE_POLICY, R2_BASELINE_POLICY,
    R3_BASELINE_POLICY, baseline_policy, score_baseline, select_baseline_context)
from tests.strategies.test_v7_rule_baseline import CUTOFF, request, unit_features


def pullback():
    f = unit_features()
    f.update(close=100., open=99.9, high=100.1, low=98., support=97.5,
        ema10=100.1, ema20=100., ema20_slope=.1, drawdown60=-.01, position60=.5,
        drawdown20=-.05, down_volume_contracts=True)
    return f


@pytest.mark.parametrize("field,value,selected", [
    ("ema10", 99.5, True), ("ema10", 99.4999, False),
    ("close", 98.5, True), ("close", 98.4999, False)])
def test_r4_bounded_pullback_inclusive_atr_edges_without_expanding_old_versions(field, value, selected):
    f = pullback(); f[field] = value; before = deepcopy(f)
    assert select_baseline_context(f).context == ("TREND_PULLBACK" if selected else "NO_CLEAR_CONTEXT")
    if selected:
        assert select_baseline_context(f, policy_version=R3_BASELINE_POLICY).context == "NO_CLEAR_CONTEXT"
    assert f == before


@pytest.mark.parametrize("slope", [0., -.01, None])
def test_relaxed_short_ema_branch_requires_positive_known_medium_slope(slope):
    f = pullback(); f.update(ema10=99.5, ema20_slope=slope)
    assert select_baseline_context(f).context == "NO_CLEAR_CONTEXT"


@pytest.mark.parametrize("field", ["ema10", "ema20", "ema20_slope", "atr14", "close"])
def test_relaxed_branch_respects_availability_masks(field):
    f = pullback(); f.update(ema10=99.5, close=98.5, score_availability={field: False})
    assert select_baseline_context(f).context == "NO_CLEAR_CONTEXT"


@pytest.mark.parametrize("field,value", [("support_intact", False), ("no_new_low2", False),
    ("drawdown20", -.16), ("drawdown20", -.005)])
def test_relaxed_context_never_removes_support_low_or_drawdown_requirements(field, value):
    f = pullback(); f.update(ema10=99.5); f[field] = value
    assert select_baseline_context(f).context == "NO_CLEAR_CONTEXT"


def test_fixed_prescore_priority_and_old_flat_slope_branch_are_preserved():
    from tests.strategies.test_v7_rule_baseline_r2 import relative_leader
    f = relative_leader(); f["drawdown20"] = -.02
    assert select_baseline_context(f).context == "LEADERSHIP_CONTINUATION"
    f = pullback(); f["ema20_slope"] = 0.
    assert select_baseline_context(f).context == "TREND_PULLBACK"
    assert select_baseline_context(unit_features()).context == "REVERSAL"


def test_wider_close_tolerance_also_requires_positive_ema20_slope():
    f = pullback(); f.update(close=98.5, ema20_slope=0.)
    assert select_baseline_context(f).context == "NO_CLEAR_CONTEXT"


def outcome(*, failed_gate=None, unknown_gate=None, risks=(), native=None, p=None):
    f = unit_features()
    prepared = dict(policy=baseline_policy(), features=f, selection=select_baseline_context(f).record())
    p = p or ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL",
        101., 99., 109., 2., (101., 101.5), 1, .1, .1)
    gates = [GateReceipt(gate, "SYNTHETIC_ONLY", "FAIL" if gate == failed_gate else
        "UNKNOWN" if gate == unknown_gate else "PASS", identity({}), timestamp(CUTOFF).isoformat(), (),
        ("SYNTHETIC_ONLY",), ("SYNTHETIC_ONLY",), timestamp(CUTOFF).isoformat())
        for gate in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK")]
    return score_baseline(prepared, plan=p, quality=dict(status="PASS", quality_score=80.), gates=gates,
        native_features=dict(spread=.003, adv=50_000_000., **(native or {})), risks=list(risks),
        standard=not risks and failed_gate is None and unknown_gate is None,
        starter_gate=not risks, confirmed_gate=not risks)


@pytest.mark.parametrize("gate", ["VALUATION", "TRADING_STATUS", "DATA_INTEGRITY", "SAFETY", "STRESS_RISK"])
def test_known_gate_fail_is_zero_capped_no_buy_without_fake_technical_veto(gate):
    result = outcome(failed_gate=gate, risks=(gate + "_FAIL",))
    assert result["score"]["raw_score"] == 94
    assert result["score"]["final_score"] == 0 and result["decision"] == "NO_BUY"
    assert result["caps"][gate + "_FAIL"] == 0
    assert result["technical_vetoes"] == [] and "BASELINE_HARD_VETO" not in result["caps"]


def test_execution_fail_stays_zero_capped_even_if_all_external_gate_receipts_say_pass():
    p = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL",
        101., 99., 102., 2., (101., 101.5), 1, .1, .1)
    result = outcome(risks=("EXECUTION_FAIL",), p=p)
    assert p.net_rr < 2
    assert result["caps"]["BASELINE_EXECUTION_FAIL"] == 0
    assert result["score"]["final_score"] == 0 and result["decision"] == "NO_BUY"
    assert result["technical_vetoes"] == []


def test_actual_technical_support_failure_remains_a_veto():
    result = outcome(risks=("SUPPORT_FAILURE",), native=dict(close=99., support=100.))
    assert "SUPPORT_FAILURE" in result["technical_vetoes"]
    assert result["caps"]["BASELINE_HARD_VETO"] == 0 and result["decision"] == "NO_BUY"


def test_unknown_receipt_cannot_be_compensated_by_high_score():
    result = outcome(unknown_gate="TRADING_STATUS")
    assert result["score"]["raw_score"] == 94
    assert result["score"]["final_score"] <= 61 and result["decision"] == "DATA_INSUFFICIENT"


def receipt_report(*, version=BASELINE_POLICY, gate_states=None):
    bars, context = request(); context["rule_baseline_policy_version"] = version
    first = evaluate(bars, context)
    for gate, inputs in first["diagnostics"]["gate_bindings"].items():
        if gate == "PRICE_DISCOVERY_PAYOFF": continue
        context["receipts"][gate] = dict(gate_id=gate, gate_version="SYNTHETIC_ONLY_EXTERNAL_PORT",
            state=(gate_states or {}).get(gate, "PASS"), input_hash=identity(inputs),
            source_available_at=timestamp(CUTOFF).isoformat(), proof_record_ids=["SYNTHETIC_ONLY"],
            input_ids=["SYNTHETIC_ONLY"], checked_at=timestamp(CUTOFF).isoformat(), reason_codes=[])
    return evaluate(bars, context)


def test_r4_projection_separates_financial_reject_from_falling_knife_and_keeps_no_buy():
    report = receipt_report(gate_states=dict(VALUATION="FAIL")); validate_report(report)
    row = project_report(report)
    assert row["低點決策"] == "NO_BUY" and row["LES v7 Final Score"] == 0
    assert row["Falling Knife Veto"] is False and row["Falling Knife State"] == "CLEAR"
    assert row["No Veto"] is True  # Technical/source Veto is not the Valuation Gate.


def test_r4_projection_trading_fail_is_not_a_falling_knife_but_no_veto_fails():
    row = project_report(receipt_report(gate_states=dict(TRADING_STATUS="FAIL")))
    assert row["低點決策"] == "NO_BUY" and row["No Veto"] is False
    assert row["Falling Knife Veto"] is False and row["Falling Knife State"] == "CLEAR"


@pytest.mark.parametrize("version", [LEGACY_BASELINE_POLICY, R2_BASELINE_POLICY, R3_BASELINE_POLICY])
def test_old_saved_veto_projection_and_formula_still_validate_under_old_policy(version):
    report = receipt_report(version=version, gate_states=dict(VALUATION="FAIL")); validate_report(report)
    assert "technical_vetoes" not in report["diagnostics"]["rule_baseline"]
    row = project_report(report)
    assert row["Falling Knife Veto"] is True and row["Falling Knife State"] == "VETO"


def test_rehashed_r4_technical_veto_or_execution_ledger_cannot_be_forged():
    original = receipt_report()
    for kind in ("veto", "ledger"):
        report = deepcopy(original)
        saved = report["diagnostics"]["rule_baseline"]
        if kind == "veto": saved["technical_vetoes"] = ["INVENTED"]
        else:
            condition = next(c for c in saved["score_evidence"]["conditions"] if c["id"] == "execution:net_rr")
            condition["earned_points"] = 3.
            condition["sources"][0]["operands"] = dict(cost_up=0.)
        with pytest.raises(ValueError, match="RECOMPUTATION_MISMATCH"):
            validate_report(report)


def test_valuation_quality_and_risk_policy_do_not_change_in_r4():
    old, new = baseline_policy(version=R3_BASELINE_POLICY), baseline_policy()
    for key in ("valuation_policy", "quality_policy", "config", "model_spec", "native_risk_limits"):
        assert old[key] == new[key]
    old_reference, new_reference = deepcopy(old["reference_policy"]), deepcopy(new["reference_policy"])
    old_reference["cost_path_policy"].pop("available_at")
    new_reference["cost_path_policy"].pop("available_at")
    old_reference["cost_path_policy"]["cost_policy"].pop("available_at")
    new_reference["cost_path_policy"]["cost_policy"].pop("available_at")
    assert old_reference == new_reference
