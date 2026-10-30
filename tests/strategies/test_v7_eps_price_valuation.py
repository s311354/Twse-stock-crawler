"""SYNTHETIC ONLY: price-scenario arithmetic, not source repair or returns."""
from copy import deepcopy
import json

import pytest

from domain.les_v7.contracts import canonical, identity
from domain.les_v7.valuation_price import POLICY_VERSION, evaluate_price_valuation, policy


def inputs(**changes):
    result = dict(current_eps=12., prior_eps=10., current_revenue=1200., prior_revenue=1000.,
        current_margin=.2, prior_margin=.15, current_price=50., current_pe=5.,
        recent_pe_median=12., eps_basis="DILUTED")
    result.update(changes)
    return result


def evaluate(**changes):
    supplied = inputs()
    supplied.update(changes)
    return evaluate_price_valuation(**supplied)


def test_policy_is_exact_detached_versioned_and_has_no_observed_dividend_or_dilution_zero():
    fixed = policy(); before = identity(fixed)
    assert fixed["policy_version"] == POLICY_VERSION
    assert fixed["required_return"] == .1 and fixed["maximum_bear_permanent_loss"] == .35
    assert fixed["dividends"] == "EXCLUDED_NOT_IMPUTED"
    assert fixed["dilution"] == "NOT_SEPARATELY_MODELED_NO_OBSERVED_ZERO_CLAIM"
    assert fixed["performance_validated"] is False
    fixed["base"]["maximum_growth"] = 999
    assert identity(policy()) == before
    result = evaluate()
    assert identity(result["policy"]) == result["policy_id"]
    assert "dividends_paid" not in canonical(result) and "annual_share_dilution" not in canonical(result)
    assert json.loads(canonical(result)) == result


def test_reported_eps_growth_price_ceiling_and_all_scenario_equations():
    result = evaluate()
    assert result["status"] == "PASS"
    assert result["anchor"]["market_implied_earnings_ceiling"] == 10.
    assert result["anchor"]["scenario_anchor"] == 10.
    assert result["observed_growth"] == pytest.approx(dict(eps=.2, revenue=.2))
    expected = {"BASE": (5, .1, 1., 12.), "BEAR": (3, .05, .85, 8.4), "BULL": (7, .15, 1.1, 14.4)}
    for name, (years, growth, factor, pe) in expected.items():
        scenario = result["scenarios"][name]
        terminal = 10 * (1 + growth) ** years * factor
        future = terminal * pe
        assert scenario["terminal_eps"] == pytest.approx(terminal)
        assert scenario["future_price"] == pytest.approx(future)
        assert scenario["fair_price"] == pytest.approx(future / 1.1 ** years)
        assert scenario["price_cagr"] == pytest.approx((future / 50) ** (1 / years) - 1)


def test_basic_eps_haircut_is_declared_assumption_not_invented_diluted_eps():
    diluted, basic = evaluate(), evaluate(eps_basis="BASIC")
    assert basic["inputs"]["current_eps"] == diluted["inputs"]["current_eps"]
    assert basic["anchor"]["scenario_anchor"] == .9 * diluted["anchor"]["scenario_anchor"]
    assert basic["assumptions"]["basic_eps_haircut"] == .9
    assert basic["anchor"]["basis"] == "BASIC"
    assert "diluted_eps" not in basic["inputs"]


@pytest.mark.parametrize("field", ["current_eps", "prior_eps", "current_revenue", "prior_revenue",
    "current_margin", "prior_margin", "current_price", "current_pe", "recent_pe_median"])
@pytest.mark.parametrize("bad", [None, True, "12", float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_operands_are_unknown_not_filled(field, bad):
    result = evaluate(**{field: bad})
    assert result["status"] == "UNKNOWN" and result["scenarios"] is None
    assert result["inputs"][field] is None
    assert "INVALID_OR_MISSING_" + field.upper() in result["reasons"]
    canonical(result)  # Nonfinite input never leaks into a persisted JSON result.


@pytest.mark.parametrize("field", ["current_eps", "prior_eps", "current_revenue", "prior_revenue",
    "current_margin", "prior_margin", "current_price", "current_pe", "recent_pe_median"])
@pytest.mark.parametrize("bad", [0., -1.])
def test_known_nonpositive_conditions_fail_not_missing(field, bad):
    result = evaluate(**{field: bad})
    assert result["status"] == "FAIL" and result["scenarios"] is None
    assert "NON_POSITIVE_" + field.upper() in result["reasons"]


def test_known_negative_history_is_not_dropped_for_more_favorable_latest_eps():
    result = evaluate(prior_eps=-10.)
    assert result["status"] == "FAIL" and result["anchor"] is None
    assert "NON_POSITIVE_PRIOR_EPS" in result["reasons"]
    result = evaluate(prior_eps=0., current_pe=None)
    assert result["status"] == "FAIL"
    assert "INVALID_OR_MISSING_CURRENT_PE" in result["reasons"]


def test_growth_and_exit_pe_caps_are_fixed_and_no_second_margin_growth_credit():
    high = evaluate(recent_pe_median=99.)
    assert high["scenarios"]["BASE"]["exit_pe"] == 20.
    assert high["scenarios"]["BEAR"]["exit_pe"] == 14.
    assert high["scenarios"]["BULL"]["exit_pe"] == 24.
    low = evaluate(recent_pe_median=1., current_revenue=500.)
    assert low["scenarios"]["BASE"]["exit_pe"] == 5.
    assert low["scenarios"]["BEAR"]["exit_pe"] == 5.
    assert low["scenarios"]["BASE"]["growth"] == -.1
    assert low["scenarios"]["BEAR"]["growth"] == -.1
    assert evaluate(current_margin=.8)["scenarios"] == evaluate()["scenarios"]


def test_gate_has_all_original_strict_hurdles_and_known_failure_is_not_unknown():
    result = evaluate(current_price=200., current_pe=20.)
    assert result["status"] == "FAIL"
    assert not result["gate_checks"]["BASE_FAIR_PRICE_GT_CURRENT_PRICE"]
    assert not result["gate_checks"]["BASE_PRICE_CAGR_GTE_REQUIRED_RETURN"]
    assert not result["gate_checks"]["BEAR_LOSS_LTE_MAXIMUM"]
    # A base fair price mathematically equal to spot does not pass the strict >.
    equal = evaluate(current_price=120., current_pe=12.)
    assert equal["scenarios"]["BASE"]["fair_price"] == pytest.approx(120.)
    assert equal["gate_checks"]["BASE_FAIR_PRICE_GT_CURRENT_PRICE"] is False


def test_inputs_are_unchanged_and_unknown_eps_basis_cannot_silently_select_basic():
    supplied = inputs(); before = deepcopy(supplied)
    evaluate_price_valuation(**supplied)
    assert supplied == before
    assert evaluate(eps_basis="UNKNOWN")["status"] == "UNKNOWN"


def test_finite_inputs_with_unrepresentable_derived_numbers_fail_closed_and_keep_json_finite():
    result = evaluate(current_eps=1e308, prior_eps=1e-308)
    assert result["status"] == "UNKNOWN"
    assert result["reasons"] == ["NUMERICAL_RESULT_NOT_FINITE"]
    assert result["scenarios"] is None and result["observed_growth"] is None
    canonical(result)
