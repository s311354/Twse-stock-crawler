"""SYNTHETIC ONLY: separate core/tactical grades and absolute total caps."""
from copy import deepcopy

import pytest
import domain.strategies  # noqa: F401 -- existing plugin initialization order

from domain.les_v7.contracts import identity
from domain.les_v7.entry_policy_r15 import (TACTICAL_SETUP, assess, policy,
    tactical_valuation_grade, valuation_grade)
from domain.les_v7.policy_r15 import AVAILABLE_AT
from domain.les_v7.valuation_recovery_r15 import evaluate_recovery_price_valuation
from tests.strategies.test_v7_entry_policy_r13 import gates, quality
from tests.strategies.test_v7_valuation_r15 import components


def valuation(*, ratio=.8, peer_only=False, growth=.05):
    args = dict(current_eps=10., prior_eps=10. / (1 + growth),
        current_revenue=100. * (1 + growth), prior_revenue=100.,
        current_margin=.2, prior_margin=.2, current_price=100., current_pe=20.,
        recent_pe_median=20., eps_basis="DILUTED")
    c = components(issuer_count=3 if peer_only else 24, issuer_pe=20., peer_pe=20.)
    initial = evaluate_recovery_price_valuation(legacy_result=dict(inputs=args), comparables=c)
    args["current_price"] = ratio * initial["scenarios"]["BASE"]["fair_price"]
    result = evaluate_recovery_price_valuation(legacy_result=dict(inputs=args), comparables=c)
    return dict(gate_id="VALUATION", state=result["status"], valuation_diagnostics=result,
                provenance="SYNTHETIC_ONLY_NOT_SOURCE_CERTIFICATION")


def admission(receipt=None, *, score=80., states=None, variant=None):
    r = receipt or valuation()
    return assess(quality(score), gates(r, **(states or {})), r, symbol="SYNTHETIC", cutoff=AVAILABLE_AT,
        current_price=r["valuation_diagnostics"]["inputs"]["current_price"], setup_variant=variant)


@pytest.mark.parametrize("ratio,band,price_points,price_cap", [
    (.8, "PASS", 0., .01), (1.00001, "MILD", 3., .01), (1.10, "MILD", 3., .01),
    (1.10001, "MODERATE", 5., .0075), (1.25, "MODERATE", 5., .0075), (1.25001, "HIGH", 8., .005)])
def test_price_warnings_and_bear_warning_sum_points_but_take_minimum_absolute_cap(ratio, band, price_points, price_cap):
    result = admission(valuation(ratio=ratio, peer_only=True)); grade = result["tactical_grade"]
    assert (grade["band"], grade["price_total_cap"]) == (band, price_cap)
    assert grade["points"] == price_points + (4. if grade["bear_warning"] else 0.)
    assert grade["points"] == grade["price_points"] + grade["bear_points"]
    assert grade["method_total_cap"] == .0125
    assert result["tactical_total_cap"] == min(.025, .01, price_cap,
        .0025 if grade["bear_warning"] else 1., .0125)
    assert grade["multiplier"] == 1. and result["risk_fraction"] == .0005
    assert result == admission(valuation(ratio=ratio, peer_only=True))


def test_tactical_economic_bear_limit_is_not_a_core_buy_upgrade():
    r = valuation(ratio=2.)
    result = admission(r)
    assert r["state"] == "FAIL"
    assert result["core_grade"]["state"] == "BLOCKED" and not result["core_buy_allowed"]
    assert result["core_points"] == 0.
    assert result["tactical_points"] == 12. and result["tactical_small_allowed"]
    assert result["tactical_total_cap"] == .0025
    assert not result["buy_allowed"] and result["small_allowed"]


def test_core_keeps_original_price_penalties_and_quality_caps():
    for ratio in (.7, 1.05, 1.2):
        r = valuation(ratio=ratio)
        result = admission(r)
        assert result["core_grade"] == valuation_grade(r, current_price=r["valuation_diagnostics"]["inputs"]["current_price"])
        assert result["buy_allowed"] == result["core_buy_allowed"]
        assert result["buy_total_cap"] == result["core_buy_cap"]
    assert admission(valuation(ratio=.7))["core_buy_allowed"]
    assert admission(valuation(ratio=.7), variant=TACTICAL_SETUP)["core_buy_allowed"] is False


@pytest.mark.parametrize("score,allowed", [(64.999, False), (65., True), (100., True)])
def test_tactical_quality_floor_does_not_relabel_limited_quality_or_alter_points(score, allowed):
    result = admission(valuation(ratio=2.), score=score)
    assert result["tactical_small_allowed"] is allowed
    assert result["tactical_points"] == 12.
    assert result["quality"]["score"] == score


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "STRESS_RISK"])
@pytest.mark.parametrize("state", ["UNKNOWN", "FAIL"])
def test_hard_unknown_and_fail_are_never_compensated(gate, state):
    result = admission(valuation(ratio=2.), states={gate: state})
    assert not result["buy_allowed"] and not result["small_allowed"]
    assert not result["core_buy_allowed"] and not result["tactical_small_allowed"]
    assert result["state"] == ("UNKNOWN" if state == "UNKNOWN" else "BLOCKED")


@pytest.mark.parametrize("damage", ["fair", "state", "policy", "components", "nonprice", "missing"])
def test_exact_recomputation_rejects_mutated_receipts_in_both_routes(damage):
    r = valuation(ratio=2.); original = deepcopy(r)
    d = r["valuation_diagnostics"]
    if damage == "fair": d["scenarios"]["BASE"]["fair_price"] *= 2
    elif damage == "state": r["state"] = "PASS"
    elif damage == "policy": d["policy"]["required_return"] = 0.
    elif damage == "components": d["comparables"]["peers"]["count"] = 999
    elif damage == "nonprice": d["reasons"].append("MAJOR_FINANCIAL_RISK")
    else: d["inputs"]["current_eps"] = None
    result = admission(r)
    assert result["core_grade"]["state"] == result["tactical_grade"]["state"] == "UNKNOWN"
    assert not result["buy_allowed"] and not result["small_allowed"]
    assert r != original


def test_known_nonpositive_annual_prerequisite_stays_blocked_not_economic_warning():
    r = valuation(); args = dict(r["valuation_diagnostics"]["inputs"], current_eps=0.)
    d = evaluate_recovery_price_valuation(legacy_result=dict(inputs=args), comparables=components())
    r.update(state=d["status"], valuation_diagnostics=d)
    result = admission(r)
    assert result["tactical_grade"]["state"] == "BLOCKED"
    assert not result["small_allowed"] and not result["buy_allowed"]


def test_admission_identity_and_policy_detachment_are_deterministic():
    result = admission()
    assert result["assessment_id"] == identity({k: v for k, v in result.items() if k != "assessment_id"})
    fixed = policy(); fixed["tactical"]["price_bands"][0]["points"] = -100
    assert policy()["tactical"]["price_bands"][0]["points"] == 3.
    with pytest.raises(ValueError, match="R15_UNKNOWN_SETUP_VARIANT"):
        admission(variant="UNVERIFIED_VARIANT")
