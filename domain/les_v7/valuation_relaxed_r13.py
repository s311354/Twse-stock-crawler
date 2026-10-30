"""User-approved R13 annual-EPS assumptions, not calibrated return forecasts.

Original R1/R10 arithmetic and identities remain unchanged. The new fallback
uses verified latest annual EPS rather than a prior/market-implied minimum;
method-specific growth, multiples, stresses and horizons are retained. Proofs,
cutoff selection and positive annual prerequisites still belong to the port.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, localcontext
from math import isfinite

from domain.les_v7.contracts import identity
from domain.les_v7.policy_r13 import AVAILABLE_AT, VALUATION_VERSION
from domain.les_v7.valuation_price import evaluate_price_valuation, policy as original_policy
from domain.les_v7.valuation_recovery_r10 import (
    evaluate_recovery_price_valuation, policy as recovery_policy,
)


POLICY_VERSION = VALUATION_VERSION
ANNUAL_METHOD = "R13_ANNUAL_FALLBACK"
RECOVERY_METHOD = "RECOVERY_INDUSTRY_R13"


def policy():
    """Detached complete contract; a warning is not a native valuation PASS."""
    fixed = recovery_policy()
    fallback = original_policy()
    fallback.update(policy_version="V7_ANNUAL_EPS_PRICE_RETURN_R13_FALLBACK", available_at=AVAILABLE_AT,
        required_return=.08, eps_anchor=fixed["eps_anchor"],
        market_implied_earnings="DIAGNOSTIC_ONLY_NOT_ANCHOR_OR_EPS_SUBSTITUTE")
    fixed.update(schema="les-v7-price-return-valuation/v3", policy_version=POLICY_VERSION,
        available_at=AVAILABLE_AT, required_return=.08, annual_fallback=fallback,
        maximum_bear_discounted_shortfall=.45,
        bear_shortfall_bands=dict(normal_maximum=.35, warning_maximum=.45,
            warning="BUY_SMALL_ONLY_TOTAL_WEIGHT_MULTIPLIER_0.5", above_warning="BLOCK",
            warning_native_status="FAIL_NOT_PASS"))
    fixed["comparables"]["insufficient"] = "EXPLICIT_R13_ANNUAL_FALLBACK"
    fixed["formula"]["fair_price"] = "FUTURE_PRICE / (1 + REQUIRED_RETURN) ** SCENARIO_YEARS"
    fixed["pass_conditions"].append("BEAR_LOSS_LTE_HARD_MAXIMUM")
    return fixed


def _bind_policy(result):
    fixed = policy()
    result.update(schema=fixed["schema"], policy_version=POLICY_VERSION,
        policy_id=identity(fixed), policy=fixed, bear_metric=fixed["bear_metric"],
        discounted_bear_value_shortfall=None, bear_shortfall_band=None)
    return result


def _unknown(result, reason):
    result.update(status="UNKNOWN", reasons=[reason], scenarios=None, anchor=None,
        gate_checks=None, bear_permanent_loss=None, discounted_bear_value_shortfall=None,
        bear_shortfall_band=None)
    return result


def _discount_and_grade(result):
    """Rebuild exact scenario arithmetic from operands, never rounded fair values."""
    if not result.get("scenarios") or not result.get("anchor"):
        return result
    try:
        with localcontext() as context:
            context.prec = 34
            price = Decimal(str(result["inputs"]["current_price"]))
            anchor = Decimal(str(result["anchor"]["scenario_anchor"]))
            discount = 1 + Decimal(str(result["policy"]["required_return"]))
            exact = {}
            for name, scenario in result["scenarios"].items():
                years = scenario["years"]
                rates = scenario.get("annual_growth", [scenario["growth"]] * years)
                terminal = anchor * Decimal(str(scenario["eps_factor"]))
                for rate in rates:
                    terminal *= 1 + Decimal(str(rate))
                future = terminal * Decimal(str(scenario["exit_pe"]))
                fair = future / discount ** years
                ratio = float(future / price)
                if not all(isfinite(float(value)) and value > 0 for value in (terminal, future, fair)) or not isfinite(ratio) or ratio <= 0:
                    raise ValueError("R13_NUMERICAL_RESULT_NOT_FINITE")
                scenario.update(terminal_eps=float(terminal), future_price=float(future), fair_price=float(fair),
                                price_cagr=ratio ** (1 / years) - 1)
                exact[name] = fair
            loss = max(Decimal(0), 1 - exact["BEAR"] / price)
            checks = dict(BASE_PRICE_CAGR_GTE_REQUIRED_RETURN=exact["BASE"] >= price,
                BASE_FAIR_PRICE_GT_CURRENT_PRICE=exact["BASE"] > price,
                BEAR_LOSS_LTE_MAXIMUM=loss <= Decimal(".35"),
                BEAR_LOSS_LTE_HARD_MAXIMUM=loss <= Decimal(".45"))
            band = ("NORMAL" if checks["BEAR_LOSS_LTE_MAXIMUM"] else
                    "SMALL_ONLY_WARNING" if checks["BEAR_LOSS_LTE_HARD_MAXIMUM"] else "HARD_LIMIT_EXCEEDED")
        result.update(status="PASS" if all(checks.values()) else "FAIL",
            reasons=[key for key, passed in checks.items() if not passed], gate_checks=checks,
            bear_permanent_loss=float(loss), discounted_bear_value_shortfall=float(loss),
            bear_shortfall_band=band)
    except (KeyError, TypeError, ValueError, ArithmeticError, OverflowError) as error:
        return _unknown(result, str(error) or "R13_NUMERICAL_RESULT_NOT_FINITE")
    return result


def evaluate_annual_price_valuation(*, current_eps, prior_eps, current_revenue, prior_revenue,
                                  current_margin, prior_margin, current_price, current_pe,
                                  recent_pe_median, eps_basis):
    """R13 fallback: new annual anchor/8%, unchanged R1 growth/PE/horizons."""
    result = _bind_policy(evaluate_price_valuation(current_eps=current_eps, prior_eps=prior_eps,
        current_revenue=current_revenue, prior_revenue=prior_revenue, current_margin=current_margin,
        prior_margin=prior_margin, current_price=current_price, current_pe=current_pe,
        recent_pe_median=recent_pe_median, eps_basis=eps_basis))
    result["selected_method"] = ANNUAL_METHOD
    inputs = result["inputs"]
    # The frozen evaluator sanitizes all explicit operands and distinguishes
    # known non-positive premises from absent/invalid evidence. Do not replace
    # either with recovery or a favorable prior original.
    if any(value is None for value in inputs.values()) or any(
            value <= 0 for key, value in inputs.items() if key != "eps_basis"):
        result["scenarios"] = None
        return result
    if not result.get("scenarios"):
        return result
    with localcontext() as context:
        context.prec = 34
        anchor = float(Decimal(str(inputs["current_eps"])) * Decimal(".9" if eps_basis == "BASIC" else "1"))
    result["anchor"] = dict(current_reported_eps=inputs["current_eps"],
        prior_reported_eps=inputs["prior_eps"], before_basic_haircut=inputs["current_eps"],
        scenario_anchor=anchor, basis=eps_basis)
    return _discount_and_grade(result)


def evaluate_relaxed_price_valuation(*, legacy_result, comparables, recovery=None):
    """Use R10 comparable economics or the explicit new R13 annual fallback.

    A supplied conflict/UNKNOWN cannot select the more favorable method. The
    caller must validate original proofs before invoking this arithmetic.
    """
    inputs = legacy_result.get("inputs", {})
    try:
        annual = evaluate_annual_price_valuation(**inputs)
    except TypeError:
        return _unknown(_bind_policy(deepcopy(legacy_result)), "R13_ANNUAL_INPUTS_UNVERIFIED")
    if not isinstance(comparables, Mapping) or comparables.get("status") not in {"VERIFIED", "INSUFFICIENT", "NOT_AVAILABLE", "UNKNOWN"}:
        return _unknown(annual, "R13_COMPARABLE_STATUS_UNVERIFIED")
    if recovery is not None and (not isinstance(recovery, Mapping) or recovery.get("status") not in {"VERIFIED", "NOT_AVAILABLE", "NO_RECOVERY", "UNKNOWN"}):
        return _unknown(annual, "R13_RECOVERY_STATUS_UNVERIFIED")
    result = evaluate_recovery_price_valuation(legacy_result=annual, comparables=comparables, recovery=recovery)
    if not annual.get("scenarios"):
        # R10's peer branch does not require official daily PE operands. R13
        # explicitly retains every annual/quote prerequisite: even verified
        # peers cannot turn absent/failed prerequisites into valid scenarios.
        result.update(annual)
        result["selected_method"] = "R13_ANNUAL_PREREQUISITE_FAILURE"
        return result
    selected = result.get("selected_method")
    _bind_policy(result)
    result["selected_method"] = RECOVERY_METHOD if selected == "RECOVERY_INDUSTRY_R10" else (
        "R13_ANNUAL_PREREQUISITE_FAILURE" if selected == "ORIGINAL_R1_PREREQUISITE_FAILURE" else ANNUAL_METHOD)
    return _discount_and_grade(result)
