"""SYNTHETIC ONLY: approved R18 admission arithmetic, not market evidence."""
from copy import deepcopy

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.entry_policy_r18 import assess, policy, tactical_valuation_grade, valuation_grade
from domain.les_v7.policy_r18 import (AVAILABLE_AT, EARLY_PULLBACK, PRICE_DISCOVERY,
    SUPPORT_RESISTANCE_LIMIT, risk_limits, tactical_risk_limits, technical_policy)
from tests.strategies.test_v7_entry_policy_r13 import gates, quality
from tests.strategies.test_v7_entry_policy_r15 import admission as previous, valuation


def admission(receipt=None, *, score=80., states=None, variant=None, q=None):
    r = receipt if receipt is not None else valuation()
    return assess(q if q is not None else quality(score), gates(r, **(states or {})), r,
        symbol="SYNTHETIC", cutoff=AVAILABLE_AT,
        current_price=r["valuation_diagnostics"]["inputs"]["current_price"],
        setup_variant=variant)


@pytest.mark.parametrize("ratio,band,points,cap", [
    (.8, "PASS", 0., .01), (1.00001, "MILD", 2., .01), (1.10, "MILD", 2., .01),
    (1.10001, "MODERATE", 4., .0075), (1.25, "MODERATE", 4., .0075),
    (1.25001, "HIGH", 6., .005)])
def test_r18_only_changes_verified_tactical_warning_points(ratio, band, points, cap):
    r = valuation(ratio=ratio, peer_only=True); before = deepcopy(r)
    old = previous(r); new = admission(r)
    grade = new["tactical_grade"]
    assert (grade["band"], grade["price_points"], grade["price_total_cap"]) == (band, points, cap)
    assert grade["bear_points"] == (3. if grade["bear_warning"] else 0.)
    assert grade["points"] == points + grade["bear_points"]
    assert new["tactical_total_cap"] == old["tactical_total_cap"]
    assert r == before and grade["native_state"] == r["state"]


@pytest.mark.parametrize("score,allowed", [(59.999, False), (60., True), (64.999, True), (65., True)])
def test_tactical_quality_floor_is_60_without_core_buy_upgrade(score, allowed):
    result = admission(valuation(ratio=2.), score=score)
    assert result["small_allowed"] is allowed
    assert result["tactical_small_allowed"] is allowed
    assert not result["buy_allowed"] and result["quality"]["score"] == score
    assert result["tactical_points"] == 9.
    if 60 <= score < 65:
        assert not result["core_buy_allowed"] and result["quality"]["tier"] == "LIMITED"
        assert previous(valuation(ratio=2.), score=score)["small_allowed"] is False


@pytest.mark.parametrize("field,value", [("data_confidence", 64.999), ("coverage_total", .74999),
    ("core_inputs_pass", False), ("core_reported_pass", False),
    ("mandatory_safety_pass", False), ("native_quality_complete", False)])
def test_quality_evidence_floor_unchanged(field, value):
    q = quality(60.); q[field] = value
    result = admission(q=q)
    assert result["state"] == "UNKNOWN" and not result["small_allowed"]


@pytest.mark.parametrize("variant,cap,risk", [
    (EARLY_PULLBACK, .005, .0005), (PRICE_DISCOVERY, .0025, .00025),
    (SUPPORT_RESISTANCE_LIMIT, .01, .0005)])
def test_conditional_variants_are_small_only_with_unchanged_loss_budgets(variant, cap, risk):
    result = admission(valuation(ratio=.7), score=60., variant=variant)
    assert result["small_allowed"] and not result["buy_allowed"]
    assert not result["core_buy_allowed"]
    assert result["small_total_cap"] == cap == result["route_total_cap"]
    assert result["tactical_risk_fraction"] == result["risk_fraction"] == risk
    assert result["position_limits_affect_score"] is False


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "STRESS_RISK"])
@pytest.mark.parametrize("state", ["UNKNOWN", "FAIL"])
@pytest.mark.parametrize("variant", [None, SUPPORT_RESISTANCE_LIMIT])
def test_no_source_financial_restriction_or_stop_risk_concession(gate, state, variant):
    result = admission(valuation(ratio=2.), states={gate: state}, variant=variant)
    assert result["state"] == ("UNKNOWN" if state == "UNKNOWN" else "BLOCKED")
    assert not result["small_allowed"] and not result["buy_allowed"]
    assert not result["tactical_small_allowed"] and not result["core_buy_allowed"]


@pytest.mark.parametrize("damage", ["fair", "state", "policy", "components", "nonprice", "missing"])
def test_mutated_receipt_never_becomes_compensable(damage):
    r = valuation(ratio=2.); d = r["valuation_diagnostics"]
    if damage == "fair": d["scenarios"]["BASE"]["fair_price"] *= 2
    elif damage == "state": r["state"] = "PASS"
    elif damage == "policy": d["policy"]["required_return"] = 0.
    elif damage == "components": d["comparables"]["peers"]["count"] = 999
    elif damage == "nonprice": d["reasons"].append("MAJOR_FINANCIAL_RISK")
    else: d["inputs"]["current_eps"] = None
    result = admission(r)
    assert result["core_grade"]["state"] == result["tactical_grade"]["state"] == "UNKNOWN"
    assert not result["buy_allowed"] and not result["small_allowed"]


def test_core_economics_and_old_public_policy_are_unchanged():
    for ratio in (.7, 1.05, 1.2, 2.):
        r = valuation(ratio=ratio); old = previous(r); new = admission(r)
        assert {k: v for k, v in new["core_grade"].items() if k != "admission_policy_id"} == {
            k: v for k, v in old["core_grade"].items() if k != "admission_policy_id"}
        assert new["core_points"] == old["core_points"]
        assert new["core_buy_allowed"] == old["core_buy_allowed"]
        assert new["core_buy_cap"] == old["core_buy_cap"]
    assert previous(valuation(ratio=2.))["tactical_points"] == 12.
    from domain.les_v7.policy_r17 import tactical_risk_limits as old_limits
    assert old_limits()["maximum_stop_atr"] == 2. and old_limits()["minimum_net_rr"] == 1.3


def test_new_policy_is_detached_deterministic_and_no_score_size_feedback():
    result = admission(valuation(ratio=2.), variant=SUPPORT_RESISTANCE_LIMIT)
    assert result == admission(valuation(ratio=2.), variant=SUPPORT_RESISTANCE_LIMIT)
    assert result["policy_id"] == result["admission_policy_id"] == identity(policy())
    assert result["assessment_id"] == identity({k: v for k, v in result.items() if k != "assessment_id"})
    fixed = policy(); fixed["tactical"]["price_bands"][0]["points"] = -100
    assert policy()["tactical"]["price_bands"][0]["points"] == 2.
    assert tactical_risk_limits() == dict(maximum_stop_atr=2.5, minimum_net_rr=1.2,
        total_plan_loss_fraction=.0005, maximum_total_weight=.01)
    assert risk_limits()["maximum_stop_atr"] == 1.8
    assert technical_policy()["buy_minimum"] == 70. and technical_policy()["small_minimum"] == 60.
    assert technical_policy()["position_limits_affect_score"] is False
    with pytest.raises(ValueError, match="R18_UNKNOWN_SETUP_VARIANT"):
        admission(variant="UNVERIFIED_HIGH_SCORE")


def test_incomplete_gate_set_and_gate_receipt_mismatch_fail_closed():
    r = valuation(); q = quality(); args = dict(symbol="SYNTHETIC", cutoff=AVAILABLE_AT,
        current_price=r["valuation_diagnostics"]["inputs"]["current_price"])
    for receipts in (gates(r)[:-1], [*gates(r), gates(r)[0]], gates(r, VALUATION="UNKNOWN")):
        result = assess(q, receipts, r, **args)
        assert result["state"] == "UNKNOWN" and not result["small_allowed"]


def test_negative_eps_is_not_relabelled_as_a_price_warning():
    from domain.les_v7.valuation_recovery_r15 import evaluate_recovery_price_valuation
    from tests.strategies.test_v7_valuation_r15 import components
    r = valuation(); args = dict(r["valuation_diagnostics"]["inputs"], current_eps=0.)
    d = evaluate_recovery_price_valuation(legacy_result=dict(inputs=args), comparables=components())
    r.update(state=d["status"], valuation_diagnostics=d)
    result = admission(r)
    assert result["tactical_grade"]["state"] == "BLOCKED"
    assert not result["small_allowed"] and not result["buy_allowed"]
