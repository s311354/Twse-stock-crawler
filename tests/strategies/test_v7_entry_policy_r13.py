"""SYNTHETIC ONLY: R13 admission arithmetic; no real-source/return claim."""
from copy import deepcopy
from dataclasses import replace

import pytest

# Follow the application's plugin initialization order before importing the
# shared PAE admission module; do not change its legacy import graph here.
import domain.strategies  # noqa: F401

from domain.les_v7.contracts import GateReceipt, ReferencePlan, identity
from domain.les_v7.entry_policy_r13 import assess, policy, valuation_grade
from domain.les_v7.entry_policy_r9 import assess as old_assess, cap_for
from domain.les_v7.policy_r13 import AVAILABLE_AT, BASELINE_VERSION
from domain.les_v7.rule_baseline import baseline_policy, score_baseline, select_baseline_context
from domain.les_v7.valuation_relaxed_r13 import evaluate_annual_price_valuation, evaluate_relaxed_price_valuation
from tests.strategies.test_v7_rule_baseline import unit_features


def quality(score=80.):
    return dict(profile_id="QG_V7_REPORTED_PERIOD_R6", status="PASS" if score >= 65 else "WATCH_QUALITY",
        quality_score=score, data_confidence=100., coverage_total=1., core_inputs_pass=True,
        core_reported_pass=True, mandatory_safety_pass=True, native_quality_complete=True,
        financial_safety=dict(state="PASS"))


def valuation(*, gap=.4, growth=.05, price=101.):
    """Backsolve a declared synthetic EPS operand, never edit a verdict or fair value."""
    args = dict(current_eps=1., prior_eps=1 / (1 + growth), current_revenue=100 * (1 + growth),
        prior_revenue=100., current_margin=.2, prior_margin=.2, current_price=price,
        current_pe=20., recent_pe_median=20., eps_basis="DILUTED")
    per_eps = evaluate_annual_price_valuation(**args)["scenarios"]["BEAR"]["fair_price"]
    eps = price * (1 - gap) / per_eps
    args.update(current_eps=eps, prior_eps=eps / (1 + growth))
    result = evaluate_relaxed_price_valuation(legacy_result=evaluate_annual_price_valuation(**args),
        comparables=dict(status="INSUFFICIENT", reasons=["SYNTHETIC_ONLY_MISSING_COMPARABLES"]))
    return dict(gate_id="VALUATION", state=result["status"], valuation_diagnostics=result,
                provenance="SYNTHETIC_ONLY_NOT_SOURCE_CERTIFICATION")


def boundary_valuation(price):
    result = evaluate_relaxed_price_valuation(legacy_result=evaluate_annual_price_valuation(
        current_eps=2814669., prior_eps=1., current_revenue=105., prior_revenue=100.,
        current_margin=.2, prior_margin=.2, current_price=price, current_pe=20.,
        recent_pe_median=10., eps_basis="DILUTED"),
        comparables=dict(status="INSUFFICIENT", reasons=["SYNTHETIC_ONLY_MISSING_COMPARABLES"]))
    return dict(gate_id="VALUATION", state=result["status"], valuation_diagnostics=result)


def gates(receipt, **states):
    return [GateReceipt(gate, "SYNTHETIC_ONLY", states.get(gate, receipt["state"] if gate == "VALUATION" else "PASS"),
        identity({}), AVAILABLE_AT, (), ("SYNTHETIC_ONLY",), ("SYNTHETIC_ONLY",), AVAILABLE_AT)
        for gate in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK")]


def admission(receipt=None, *, q=None, states=None, old=False):
    receipt = receipt if receipt is not None else valuation()
    evaluator = old_assess if old else assess
    return evaluator(q if q is not None else quality(), gates(receipt, **(states or {})), receipt,
        symbol="SYNTHETIC", cutoff=AVAILABLE_AT, current_price=receipt["valuation_diagnostics"]["inputs"]["current_price"])


def outcome(*, q=None, receipt=None, states=None, risks=(), version=BASELINE_VERSION):
    f = unit_features(); receipt = receipt if receipt is not None else valuation()
    prepared = dict(policy=baseline_policy(version=version), features=f,
        selection=select_baseline_context(f, policy_version=version).record())
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL",
        101., 99., 109., 2., (101., 101.5), 1, .1, .1)
    return score_baseline(prepared, plan=plan, quality=q if q is not None else quality(),
        gates=gates(receipt, **(states or {})), native_features=dict(close=101., spread=.003, adv=100_000_000.),
        risks=list(risks), standard=True, starter_gate=True, confirmed_gate=True,
        valuation_receipt=receipt, entry_binding=dict(symbol="SYNTHETIC", cutoff=AVAILABLE_AT))


@pytest.mark.parametrize("price,state,warning,small,multiplier", [
    (20453125., "ALLOWED", False, False, 1.),
    (20453125.01, "ALLOWED_WITH_WARNING", True, True, .5),
    (24171875., "ALLOWED_WITH_WARNING", True, True, .5),
    (24171875.01, "BLOCKED", False, False, 0.),
])
def test_exact_35_45_boundaries_grade_without_relabeling_native_receipt(price, state, warning, small, multiplier):
    receipt = boundary_valuation(price); before = deepcopy(receipt)
    grade = valuation_grade(receipt, current_price=price)
    assert (grade["state"], grade["bear_warning"], grade["small_only"], grade["multiplier"]) == (state, warning, small, multiplier)
    assert receipt == before
    assert grade["native_state"] == receipt["state"]
    if warning:
        assert receipt["state"] == "FAIL" and grade["points"] == 0.


@pytest.mark.parametrize("gap,growth,band,points,price_multiplier", [
    (.4, .05, "PASS", 0., 1.),
    (.4, -.05, "MILD", 5., .75),
    (.44, -.05, "MODERATE", 10., .5),
    (.4, -.10, "HIGH", 15., .25),
])
def test_bear_warning_half_cap_is_applied_once_separately_from_price_band(gap, growth, band, points, price_multiplier):
    receipt = valuation(gap=gap, growth=growth); result = admission(receipt)
    grade = result["valuation"]
    assert grade["state"] == "ALLOWED_WITH_WARNING" and grade["native_state"] == "FAIL"
    assert (grade["band"], grade["points"]) == (band, points)
    assert grade["multiplier"] == price_multiplier * .5
    assert result["small_total_cap"] == pytest.approx(.05 * price_multiplier * .5)
    assert result["small_allowed"] is True and result["buy_allowed"] is False
    assert cap_for(result, "BUY_SMALL") == result["small_total_cap"]
    assert admission(receipt) == result  # retries do not multiply a retained cap again


@pytest.mark.parametrize("score,tier,cap", [(80., "HIGH", .025), (75., "HIGH", .025),
    (74.9999, "STANDARD", .0125), (65., "STANDARD", .0125), (64.9999, "LIMITED", .01), (60., "LIMITED", .01)])
def test_quality_tiers_change_total_caps_not_les_points(score, tier, cap):
    result = outcome(q=quality(score)); full = outcome(q=quality())
    assert result["score"] == full["score"]
    assert result["penalties"] == full["penalties"] == {}
    assert result["entry_admission"]["quality"]["tier"] == tier
    assert result["entry_admission"]["small_total_cap"] == cap
    assert result["entry_admission"]["buy_allowed"] is False
    assert result["decision"] == "BUY_SMALL"


def test_bear_warning_is_sizing_only_and_does_not_reduce_technical_score():
    normal, warning = outcome(receipt=valuation(gap=.34)), outcome(receipt=valuation(gap=.4))
    assert normal["score"] == warning["score"]
    assert normal["penalties"] == warning["penalties"] == {}
    assert normal["decision"] == "BUY" and warning["decision"] == "BUY_SMALL"
    assert normal["entry_admission"]["small_total_cap"] == .05
    assert warning["entry_admission"]["small_total_cap"] == .025


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "STRESS_RISK"])
@pytest.mark.parametrize("state,expected", [("FAIL", "BLOCKED"), ("UNKNOWN", "UNKNOWN")])
def test_unknown_and_hard_failure_cannot_be_compensated_by_high_score(gate, state, expected):
    result = outcome(states={gate: state})
    assert result["entry_admission"]["state"] == expected
    assert result["decision"] not in {"BUY", "BUY_SMALL"}
    assert not result["entry_admission"]["buy_allowed"] and not result["entry_admission"]["small_allowed"]


@pytest.mark.parametrize("key,value", [("status", "UNKNOWN"), ("status", "DATA_INSUFFICIENT"),
    ("mandatory_safety_pass", False), ("native_quality_complete", False), ("data_confidence", 64.999),
    ("coverage_total", .74999), ("financial_safety", dict(state="UNKNOWN")),
    ("financial_safety", dict(state="FAIL")), ("status", "REJECT")])
def test_incomplete_or_rejected_quality_is_not_a_lower_quality_tier(key, value):
    q = quality(); q[key] = value
    result = outcome(q=q)
    assert result["decision"] not in {"BUY", "BUY_SMALL"}
    assert not result["entry_admission"]["small_allowed"]


def test_below_quality_floor_failed_execution_and_veto_remain_blocked():
    assert outcome(q=quality(59.999))["decision"] == "NO_BUY"
    for reason in ("EXECUTION_FAIL", "SUPPORT_FAILURE", "SAFETY_FAIL"):
        assert outcome(risks=[reason])["decision"] == "NO_BUY"
    assert outcome(risks=["VALUATION_FAIL"])["decision"] == "BUY_SMALL"


def test_old_admission_contract_does_not_accept_r13_economics():
    result = admission(old=True)
    assert result["state"] == "UNKNOWN"
    assert not result["buy_allowed"] and not result["small_allowed"]
    from domain.les_v7.rule_baseline import GRADED_BASELINE_POLICY
    assert outcome(version=GRADED_BASELINE_POLICY)["decision"] == "DATA_INSUFFICIENT"


@pytest.mark.parametrize("damage", ["fair", "loss", "hard_gate", "state", "version", "input_price", "missing_inputs", "fallback", "policy"])
def test_rehashed_or_edited_valuation_cannot_authorize_entry(damage):
    receipt = valuation(); d = receipt["valuation_diagnostics"]
    if damage == "fair": d["scenarios"]["BASE"]["fair_price"] *= 2
    elif damage == "loss": d["bear_permanent_loss"] = 0.
    elif damage == "hard_gate": d["gate_checks"]["BEAR_LOSS_LTE_HARD_MAXIMUM"] = False
    elif damage == "state": receipt["state"] = "PASS"
    elif damage == "version": d["policy_version"] = "SYNTHETIC_UNSUPPORTED"
    elif damage == "input_price": d["inputs"]["current_price"] += 1.
    elif damage == "missing_inputs": d.pop("inputs")
    elif damage == "fallback": d["fallback_result"]["anchor"]["scenario_anchor"] *= 2
    else: d["policy"]["required_return"] = 0.; d["policy_id"] = identity(d["policy"])
    assert valuation_grade(receipt, current_price=101.)["state"] in {"UNKNOWN", "BLOCKED"}
    assert outcome(receipt=receipt)["decision"] not in {"BUY", "BUY_SMALL"}


def test_missing_duplicate_or_mismatched_gate_set_is_not_complete():
    receipt = valuation(); good = gates(receipt)
    for broken in (good[:-1], good[:-1] + [good[0]], [replace(g, state="PASS") if g.gate_id == "VALUATION" else g for g in good]):
        result = assess(quality(), broken, receipt, symbol="SYNTHETIC", cutoff=AVAILABLE_AT, current_price=101.)
        assert result["state"] == "UNKNOWN" and not result["small_allowed"]


def test_policy_and_input_records_are_detached_from_admission():
    p = policy(); p["valuation_bands"][0]["points"] = 999
    assert policy()["valuation_bands"][0]["points"] == 5.
    q, receipt = quality(), valuation(); before = deepcopy((q, receipt))
    assess(q, gates(receipt), receipt, symbol="SYNTHETIC", cutoff=AVAILABLE_AT, current_price=101.)
    assert (q, receipt) == before
