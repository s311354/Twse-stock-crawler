"""SYNTHETIC ONLY: R10 policy equations, not price predictions or live trades."""
from copy import deepcopy
from math import prod

import pytest

from domain.les_v7.contracts import canonical, identity
from domain.les_v7.valuation_price import evaluate_price_valuation, policy as original_policy
from domain.les_v7.valuation_recovery_r10 import policy, evaluate_recovery_price_valuation


def legacy(**changes):
    args = dict(current_eps=12., prior_eps=10., current_revenue=1200., prior_revenue=1000.,
        current_margin=.2, prior_margin=.15, current_price=50., current_pe=5., recent_pe_median=12., eps_basis="DILUTED")
    args.update(changes)
    return evaluate_price_valuation(**args)


def comparable(**changes):
    result = dict(status="VERIFIED", reasons=[], industry="半導體業", issuer_pe_median=25.,
                  peer_pe_median=35., peer_count=10, issuer_month_count=24)
    result.update(changes)
    return result


def evaluate(*, old=None, peers=None, recovery=None):
    return evaluate_recovery_price_valuation(legacy_result=old or legacy(),
        comparables=peers or comparable(), recovery=recovery)


def test_original_r1_identity_and_outputs_remain_unchanged():
    before = legacy(); fixed = original_policy(); before_id = identity(fixed)
    evaluate(); assert before == legacy() and identity(original_policy()) == before_id
    assert before["anchor"]["scenario_anchor"] == 10.
    assert before["scenarios"]["BEAR"]["years"] == 3
    assert before["scenarios"]["BULL"]["years"] == 7


def test_policy_is_detached_and_scenarios_use_latest_annual_anchor_same_five_year_horizon():
    fixed = policy(); fixed["base"]["years"] = 99
    assert policy()["base"]["years"] == 5
    result = evaluate()
    assert result["anchor"]["scenario_anchor"] == 12.
    assert "market_implied_earnings_ceiling" not in result["anchor"]
    assert result["scenarios"]["BASE"]["exit_pe"] == 30.
    rates = [.03 + (.20 - .03) * .5 ** year for year in range(5)]
    assert result["scenarios"]["BASE"]["annual_growth"] == pytest.approx(rates)
    for name, factor, pe, growth in (
        ("BASE", 1., 30., rates), ("BEAR", .9, 24., [max(-.1, rate - .05) for rate in rates]),
        ("BULL", 1.1, 36., [min(.25, rate + .05) for rate in rates])):
        scenario = result["scenarios"][name]
        assert scenario["years"] == 5
        assert scenario["terminal_eps"] == pytest.approx(12 * factor * prod(1 + rate for rate in growth))
        assert scenario["future_price"] == pytest.approx(scenario["terminal_eps"] * pe)
        assert scenario["fair_price"] == pytest.approx(scenario["future_price"] / 1.1 ** 5)
    assert result["scenarios"]["BULL"]["future_price"] > result["scenarios"]["BASE"]["future_price"]
    assert result["bear_permanent_loss"] == result["discounted_bear_value_shortfall"]
    assert "NOT_DRAWDOWN_OR_LOSS_PROBABILITY" in result["bear_metric"]
    canonical(result)


def test_recovery_weight_is_comparable_ytd_revenue_oi_not_annualized_half_year_eps():
    recovery = dict(status="VERIFIED", reasons=[], values=dict(current_revenue=130., prior_revenue=100.,
        current_operating_income=25., prior_operating_income=20., current_cfo=5.))
    old = legacy(current_eps=11., current_revenue=1100.)
    result = evaluate(old=old, recovery=recovery)
    assert result["observed_growth"]["recent_recovery"] == pytest.approx(.25)
    assert result["observed_growth"]["blended"] == pytest.approx(.75 * .10 + .25 * .25)
    assert result["anchor"]["scenario_anchor"] == 11.
    assert result["inputs"]["current_eps"] == 11.
    assert "interim_eps" not in result["inputs"]


@pytest.mark.parametrize("status", ["NO_RECOVERY", "NOT_AVAILABLE"])
def test_missing_or_failed_recovery_has_no_recovery_credit(status):
    before = evaluate()
    assert evaluate(recovery=dict(status=status, reasons=["SYNTHETIC"]))["scenarios"] == before["scenarios"]


@pytest.mark.parametrize("status", ["INSUFFICIENT", "NOT_AVAILABLE"])
def test_optional_comparable_gap_keeps_complete_original_r1_economics(status):
    old = legacy(current_price=200., current_pe=20.)
    result = evaluate(old=old, peers=dict(status=status, reasons=["SYNTHETIC_MISSING"]))
    assert result["selected_method"] == "ORIGINAL_R1_FALLBACK"
    assert result["fallback_result"] == old
    for key in ("status", "inputs", "anchor", "scenarios", "gate_checks", "bear_permanent_loss", "reasons"):
        assert result[key] == old[key]


@pytest.mark.parametrize("where", ["comparables", "recovery"])
def test_conflicting_source_never_uses_favorable_fallback(where):
    args = {"peers" if where == "comparables" else "recovery": dict(status="UNKNOWN", reasons=["SOURCE_CONFLICT"])}
    result = evaluate(**args)
    assert result["status"] == "UNKNOWN" and result["scenarios"] is None


@pytest.mark.parametrize("industry,growth,pe", [("半導體業", .2, 30.), ("電子零組件業", .2, 30.),
    ("電腦及週邊設備業", .15, 25.), ("光電業", .15, 25.), ("通信網路業", .15, 25.),
    ("電子通路業", .1, 20.), ("其他電子業", .1, 20.)])
def test_sector_caps_and_basic_haircut_are_explicit(industry, growth, pe):
    result = evaluate(old=legacy(current_eps=20., current_revenue=2000., eps_basis="BASIC"),
                      peers=comparable(industry=industry, issuer_pe_median=50., peer_pe_median=60.))
    assert result["anchor"]["scenario_anchor"] == 18.
    assert result["scenarios"]["BASE"]["annual_growth"][0] == pytest.approx(growth)
    assert result["scenarios"]["BASE"]["exit_pe"] == pe


@pytest.mark.parametrize("bad", [None, True, "25", float("nan"), float("inf"), -1.])
def test_invalid_comparable_number_never_produces_scenario(bad):
    result = evaluate(peers=comparable(peer_pe_median=bad))
    assert result["status"] == "UNKNOWN" and result["scenarios"] is None
    canonical(result)


def test_known_negative_annual_prerequisite_is_not_replaced_by_recent_recovery():
    old = legacy(current_eps=-1.)
    result = evaluate(old=old)
    assert result["status"] == "FAIL" and result["scenarios"] is None
    assert "NON_POSITIVE_CURRENT_EPS" in result["reasons"]


def test_input_objects_are_not_changed_by_arithmetic():
    old, peers = legacy(), comparable(); before = deepcopy((old, peers))
    evaluate_recovery_price_valuation(legacy_result=old, comparables=peers)
    assert (old, peers) == before
