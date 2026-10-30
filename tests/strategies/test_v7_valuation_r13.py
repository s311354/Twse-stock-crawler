"""SYNTHETIC ONLY: approved R13 assumptions, not investment-performance proof."""
from copy import deepcopy
from math import prod

import pytest

from domain.les_v7.contracts import canonical, identity
from domain.les_v7.valuation_price import evaluate_price_valuation, policy as original_policy
from domain.les_v7.valuation_recovery_r10 import evaluate_recovery_price_valuation, policy as recovery_policy
from domain.les_v7.valuation_relaxed_r13 import (
    ANNUAL_METHOD, RECOVERY_METHOD, evaluate_annual_price_valuation,
    evaluate_relaxed_price_valuation, policy,
)


def inputs(**changes):
    values = dict(current_eps=12., prior_eps=10., current_revenue=1200., prior_revenue=1000.,
        current_margin=.2, prior_margin=.15, current_price=50., current_pe=20., recent_pe_median=12., eps_basis="DILUTED")
    return dict(values, **changes)


def comparable(**changes):
    return dict(dict(status="VERIFIED", reasons=[], industry="半導體業", issuer_pe_median=25.,
        peer_pe_median=35., peer_count=10, issuer_month_count=24), **changes)


def evaluate(*, args=None, peers=None, recovery=None):
    return evaluate_relaxed_price_valuation(legacy_result=evaluate_price_valuation(**(args or inputs())),
        comparables=peers if peers is not None else comparable(), recovery=recovery)


def test_old_policy_identities_and_results_are_unchanged():
    old = evaluate_price_valuation(**inputs())
    r10 = evaluate_recovery_price_valuation(legacy_result=old, comparables=comparable())
    ids = identity(original_policy()), identity(recovery_policy())
    evaluate(); evaluate(peers=dict(status="INSUFFICIENT", reasons=["SYNTHETIC"]))
    assert ids == (identity(original_policy()), identity(recovery_policy()))
    assert old == evaluate_price_valuation(**inputs())
    assert r10 == evaluate_recovery_price_valuation(legacy_result=old, comparables=comparable())
    assert old["anchor"]["scenario_anchor"] == 2.5
    assert old["policy"]["required_return"] == r10["policy"]["required_return"] == .10


def test_policy_is_detached_complete_and_never_claims_validated_performance():
    fixed = policy(); before = identity(fixed)
    fixed["annual_fallback"]["base"]["years"] = 99
    fixed["industry_profiles"]["半導體業"]["maximum_exit_pe"] = 999
    assert identity(policy()) == before
    assert policy()["annual_fallback"]["base"]["years"] == 5
    assert policy()["required_return"] == .08
    assert policy()["performance_validated"] is False


@pytest.mark.parametrize("basis,anchor", [("DILUTED", 12.), ("BASIC", 10.8)])
def test_unified_annual_anchor_not_prior_or_market_implied_while_fallback_stresses_stay(basis, anchor):
    args = inputs(eps_basis=basis)
    old = evaluate_price_valuation(**args)
    result = evaluate_annual_price_valuation(**args)
    assert result["selected_method"] == ANNUAL_METHOD
    assert result["anchor"]["scenario_anchor"] == anchor
    assert "market_implied_earnings_ceiling" not in result["anchor"]
    for name, years in (("BASE", 5), ("BEAR", 3), ("BULL", 7)):
        scenario, prior = result["scenarios"][name], old["scenarios"][name]
        assert scenario["years"] == years
        for key in ("growth", "eps_factor", "exit_pe"):
            assert scenario[key] == prior[key]
        expected_terminal = anchor * (1 + scenario["growth"]) ** years * scenario["eps_factor"]
        assert scenario["terminal_eps"] == pytest.approx(expected_terminal)
        assert scenario["fair_price"] == pytest.approx(expected_terminal * scenario["exit_pe"] / 1.08 ** years)
    assert result["scenarios"]["BASE"]["exit_pe"] <= 20
    assert result["bear_permanent_loss"] == result["discounted_bear_value_shortfall"]
    canonical(result)


@pytest.mark.parametrize("status", ["INSUFFICIENT", "NOT_AVAILABLE"])
def test_missing_comparables_select_explicit_new_fallback_not_unchanged_r1(status):
    result = evaluate(peers=dict(status=status, reasons=["SYNTHETIC_GAP"]))
    annual = evaluate_annual_price_valuation(**inputs())
    assert result["selected_method"] == ANNUAL_METHOD
    assert result["fallback_result"] == annual
    for key in ("anchor", "scenarios", "status", "reasons", "gate_checks", "bear_permanent_loss"):
        assert result[key] == annual[key]
    assert result["policy_id"] != identity(original_policy())


def test_comparable_recovery_growth_multiples_stresses_and_five_years_unchanged_only_discount_changes():
    recent = dict(status="VERIFIED", reasons=[], values=dict(current_revenue=130., prior_revenue=100.,
        current_operating_income=25., prior_operating_income=20., current_cfo=5.))
    old = evaluate_recovery_price_valuation(legacy_result=evaluate_price_valuation(**inputs()),
        comparables=comparable(), recovery=recent)
    result = evaluate(recovery=recent)
    assert result["selected_method"] == RECOVERY_METHOD
    assert result["observed_growth"] == old["observed_growth"]
    assert result["anchor"] == old["anchor"]
    for name, scenario in result["scenarios"].items():
        for key in ("years", "growth", "annual_growth", "eps_factor", "exit_pe", "terminal_eps", "future_price", "price_cagr"):
            assert scenario[key] == old["scenarios"][name][key]
        assert scenario["fair_price"] == pytest.approx(scenario["future_price"] / 1.08 ** 5)
        assert scenario["terminal_eps"] == pytest.approx(12 * scenario["eps_factor"] * prod(1 + r for r in scenario["annual_growth"]))
    assert result["scenarios"]["BULL"]["future_price"] >= result["scenarios"]["BASE"]["future_price"] >= result["scenarios"]["BEAR"]["future_price"]


@pytest.mark.parametrize("where", ["comparables", "recovery"])
@pytest.mark.parametrize("status", ["UNKNOWN", "CONFLICT", "UNRECOGNIZED"])
def test_unknown_conflict_or_invalid_status_never_chooses_favorable_fallback(where, status):
    args = {"peers" if where == "comparables" else "recovery": dict(status=status, reasons=["SYNTHETIC_SOURCE_CONFLICT"])}
    result = evaluate(**args)
    assert result["status"] == "UNKNOWN" and result["scenarios"] is None
    canonical(result)


@pytest.mark.parametrize("field", ["current_eps", "prior_eps", "current_revenue", "prior_revenue", "current_margin", "prior_margin", "current_price", "current_pe", "recent_pe_median"])
@pytest.mark.parametrize("value,status", [(None, "UNKNOWN"), (True, "UNKNOWN"), (float("nan"), "UNKNOWN"), (0., "FAIL"), (-1., "FAIL")])
def test_all_positive_annual_and_market_prerequisites_remain(field, value, status):
    result = evaluate(args=inputs(**{field: value}))
    assert result["status"] == status and result["scenarios"] is None
    canonical(result)


@pytest.mark.parametrize("gap,price,soft,hard,band", [
    (.35, 20453125., True, True, "NORMAL"),
    (.35, 20453125.01, False, True, "SMALL_ONLY_WARNING"),
    (.45, 24171875., False, True, "SMALL_ONLY_WARNING"),
    (.45, 24171875.01, False, False, "HARD_LIMIT_EXCEEDED"),
])
def test_exact_decimal_bear_boundaries_keep_warning_fail_not_pass(gap, price, soft, hard, band):
    # EPS=19683*13*11 cancels 1.08**3 exactly; both boundary
    # prices are integers. No binary epsilon weakens the approved limits.
    result = evaluate_annual_price_valuation(**inputs(current_eps=2814669., prior_eps=1.,
        current_revenue=105., prior_revenue=100., current_price=price, recent_pe_median=10.))
    assert result["gate_checks"]["BEAR_LOSS_LTE_MAXIMUM"] is soft
    assert result["gate_checks"]["BEAR_LOSS_LTE_HARD_MAXIMUM"] is hard
    assert result["bear_shortfall_band"] == band
    assert result["discounted_bear_value_shortfall"] == pytest.approx(gap, abs=1e-8)
    if not soft:
        assert result["status"] == "FAIL"
        assert "BEAR_LOSS_LTE_MAXIMUM" in result["reasons"]


def test_inputs_are_not_mutated_and_diagnostic_values_cannot_override_arithmetic():
    old, peers = evaluate_price_valuation(**inputs()), comparable()
    before = deepcopy((old, peers))
    old["anchor"]["scenario_anchor"] = 99999.
    result = evaluate_relaxed_price_valuation(legacy_result=old, comparables=peers)
    assert result["anchor"]["scenario_anchor"] == 12.
    assert peers == before[1]
    assert old["anchor"]["scenario_anchor"] == 99999.
