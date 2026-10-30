"""R14 independent recovery and multiple components, not a calibrated forecast.

Source selection belongs to the receipt producer. This deterministic arithmetic
never certifies naked medians, annualizes interim EPS, or uses an unverified
component in place of a conflicting one. Historical R10/R13 remain unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, localcontext
from math import isfinite

from domain.les_v7.contracts import identity
from domain.les_v7.policy_r14 import AVAILABLE_AT, VALUATION_VERSION
from domain.les_v7.valuation_price import _number
from domain.les_v7.valuation_recovery_r10 import INDUSTRIES
from domain.les_v7.valuation_relaxed_r13 import (
    evaluate_annual_price_valuation as _annual_prerequisites, policy as r13_policy,
)


POLICY_VERSION = VALUATION_VERSION
COMPONENT_SCHEMA = "les-v7-comparable-components/v1"
FULL_METHOD = "RECOVERY_FULL_COMPARABLES_R14"
PEER_METHOD = "RECOVERY_PEER_ONLY_R14"
ANNUAL_METHOD = "RECOVERY_ANNUAL_MULTIPLE_FALLBACK_R14"


def policy():
    fixed = r13_policy()
    fixed.update(schema="les-v7-price-return-valuation/v4", policy_version=POLICY_VERSION,
        available_at=AVAILABLE_AT,
        recovery_multiple_dependency="INDEPENDENT_VERIFIED_COMPONENTS_NOT_JOINT_COVERAGE",
        scenario_horizons="ALL_FIVE_YEARS_WITH_COMMON_GROWTH_DECAY",
        unknown_component="CONFLICT_OR_UNVERIFIED_PRESENT_EVIDENCE_BLOCKS_NO_FAVORABLE_FALLBACK",
        annual_fallback=dict(multiple="MIN_INDUSTRY_CAP_R13_CAP20_MAX5_OFFICIAL_ROLLING60_MEDIAN",
            official_pe_basis="DATED_OFFICIAL_REFERENCE_NOT_ANNUAL_EPS_COMPARABLE_OR_EARNINGS_SUBSTITUTE",
            unknown_industry_growth_cap=.10, unknown_industry_pe_cap=20.,
            missing_recovery="ANNUAL_GROWTH_ONLY_NO_RECOVERY_CREDIT"))
    fixed["comparables"].update(component_schema=COMPONENT_SCHEMA,
        selection="FULL_ELSE_VERIFIED_PEERS_ELSE_EXPLICIT_ANNUAL_MULTIPLE_FALLBACK",
        insufficient="INDEPENDENT_RECOVERY_RETAINED_COMMON_FIVE_YEAR_SCENARIOS",
        peer_only=dict(minimum_peer_count=10, maximum_exit_pe="INDUSTRY_CAP",
            weighting="PEER_ANNUAL_MEDIAN_ONLY_NO_OFFICIAL_PE_POOLING",
            small_only=True, total_weight_cap=.025, method_position_multiplier=.50))
    return fixed


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return None if isinstance(value, float) and not isfinite(value) else value


def _unknown(result, reason):
    result.update(status="UNKNOWN", reasons=[reason], scenarios=None, anchor=None,
        gate_checks=None, bear_permanent_loss=None, discounted_bear_value_shortfall=None,
        bear_shortfall_band=None)
    return result


def _component_state(comparables):
    if (not isinstance(comparables, Mapping) or comparables.get("schema") != COMPONENT_SCHEMA
            or comparables.get("status") not in {"VERIFIED", "PARTIAL", "INSUFFICIENT", "UNKNOWN"}):
        raise ValueError("R14_COMPARABLE_COMPONENT_STATUS_UNVERIFIED")
    if comparables["status"] == "UNKNOWN":
        raise ValueError("R14_COMPARABLE_SOURCE_UNKNOWN:" + ";".join(comparables.get("reasons", ())))
    ready = []
    for name, minimum in (("issuer_history", 24), ("peers", 10)):
        part = comparables.get(name)
        if (not isinstance(part, Mapping) or part.get("status") not in {"VERIFIED", "INSUFFICIENT"}
                or type(part.get("count")) is not int or part["count"] < 0
                or part.get("minimum") != minimum):
            raise ValueError("R14_COMPARABLE_COMPONENT_BINDING_INVALID:" + name)
        actual = part["count"] >= minimum
        if (part["status"] == "VERIFIED") != actual:
            raise ValueError("R14_COMPARABLE_COMPONENT_COVERAGE_CONFLICT:" + name)
        if actual and (_number(part.get("median")) is None or part["median"] <= 0):
            raise ValueError("R14_COMPARABLE_PE_INVALID:" + name)
        count_key = "issuer_month_count" if name == "issuer_history" else "peer_count"
        if count_key in comparables and comparables[count_key] != part["count"]:
            raise ValueError("R14_COMPARABLE_COMPONENT_COUNT_CONFLICT:" + name)
        ready.append(actual)
    expected = "VERIFIED" if all(ready) else "PARTIAL" if ready[1] else "INSUFFICIENT"
    if comparables["status"] != expected:
        raise ValueError("R14_COMPARABLE_COMPONENT_SUMMARY_CONFLICT")
    return ready


def evaluate_annual_price_valuation(**inputs):
    """Explicit no-comparables entry using the same common five-year equations."""
    missing = dict(schema=COMPONENT_SCHEMA, status="INSUFFICIENT", reasons=["R14_COMPARABLE_CONTEXT_MISSING"],
        issuer_history=dict(status="INSUFFICIENT", count=0, minimum=24, median=None, sources=[]),
        peers=dict(status="INSUFFICIENT", count=0, minimum=10, median=None, sources=[]))
    return evaluate_recovery_price_valuation(legacy_result=dict(inputs=inputs), comparables=missing)


def evaluate_recovery_price_valuation(*, legacy_result, comparables, recovery=None):
    """Rebuild from annual operands plus independently selected source components."""
    try:
        annual = _annual_prerequisites(**legacy_result.get("inputs", {}))
    except (AttributeError, TypeError):
        annual = dict(inputs={}, status="UNKNOWN", reasons=["R14_ANNUAL_INPUTS_UNVERIFIED"],
                      assumptions={}, scenarios=None, anchor=None)
    fixed = policy()
    result = dict(schema=fixed["schema"], policy_version=POLICY_VERSION, policy_id=identity(fixed), policy=fixed,
        inputs=deepcopy(annual.get("inputs", {})), status=annual["status"], reasons=list(annual["reasons"]),
        assumptions=deepcopy(annual.get("assumptions", {})), anchor=None, observed_growth=None,
        scenarios=None, gate_checks=None, bear_permanent_loss=None, discounted_bear_value_shortfall=None,
        bear_shortfall_band=None, bear_metric=fixed["bear_metric"],
        comparables=_plain(comparables), recovery=_plain(recovery or dict(status="NOT_AVAILABLE", reasons=[])),
        selected_method=ANNUAL_METHOD, method_small_only=False, method_position_multiplier=1.,
        prerequisite_result_id=identity(annual), multiple_selection=None)
    # A valid peer/recovery cannot repair missing or known failed prerequisites.
    if not annual.get("scenarios"):
        return result
    try:
        issuer_ready, peer_ready = _component_state(comparables)
        recent = result["recovery"]
        if not isinstance(recent, Mapping) or recent.get("status") not in {"VERIFIED", "NO_RECOVERY", "NOT_AVAILABLE", "UNKNOWN"}:
            raise ValueError("R14_RECOVERY_STATUS_UNVERIFIED")
        if recent["status"] == "UNKNOWN":
            raise ValueError("R14_RECOVERY_SOURCE_UNKNOWN:" + ";".join(recent.get("reasons", ())))
        industry = comparables.get("industry")
        industry_profile_defined = industry in INDUSTRIES
        if industry is not None and not industry_profile_defined:
            if (not isinstance(industry, str) or not industry.strip()
                    or comparables.get("industry_source_status") != "VERIFIED_OFFICIAL_CLASSIFICATION"
                    or comparables.get("industry_profile_status") != "NOT_DEFINED"):
                raise ValueError("R14_INDUSTRY_PROFILE_UNVERIFIED")
        elif comparables.get("industry_profile_status") not in (None, "DEFINED"):
            raise ValueError("R14_INDUSTRY_PROFILE_STATUS_CONFLICT")
        maximum_growth, maximum_pe = INDUSTRIES.get(industry, (.10, 20.))
        inputs = result["inputs"]
        if industry_profile_defined and issuer_ready and peer_ready:
            method, small_only, multiplier = FULL_METHOD, False, 1.
            raw_pe = .50 * comparables["issuer_history"]["median"] + .50 * comparables["peers"]["median"]
            pe = min(maximum_pe, max(5., raw_pe))
            basis = "SAME_BASIS_ANNUAL_ISSUER_AND_PEER_MEDIANS_50_50"
        elif industry_profile_defined and peer_ready:
            method, small_only, multiplier = PEER_METHOD, True, .50
            raw_pe = comparables["peers"]["median"]
            pe = min(maximum_pe, max(5., raw_pe))
            basis = "SAME_BASIS_ANNUAL_PEER_MEDIAN_ONLY_NO_INVENTED_ISSUER_MONTHS"
        else:
            method, small_only, multiplier = ANNUAL_METHOD, False, 1.
            raw_pe = inputs["recent_pe_median"]
            pe = min(maximum_pe, 20., max(5., raw_pe))
            basis = "R13_OFFICIAL_ROLLING_REFERENCE_NOT_ANNUAL_COMPARABLE_MEDIAN"
        with localcontext() as context:
            context.prec = 34
            values = {key: Decimal(str(value)) for key, value in inputs.items() if key != "eps_basis"}
            annual_growth = min(float(values["current_eps"] / values["prior_eps"] - 1),
                                float(values["current_revenue"] / values["prior_revenue"] - 1))
            blended, recent_growth = annual_growth, None
            if recent["status"] == "VERIFIED":
                r = recent.get("values", {})
                if any(_number(r.get(key)) is None or r[key] <= 0 for key in
                       ("current_revenue", "prior_revenue", "current_operating_income", "prior_operating_income", "current_cfo")):
                    raise ValueError("R14_RECOVERY_OPERAND_INVALID")
                recent_growth = min(float(Decimal(str(r["current_" + field])) / Decimal(str(r["prior_" + field])) - 1)
                                    for field in ("revenue", "operating_income"))
                if recent_growth <= 0:
                    raise ValueError("R14_RECOVERY_CONDITION_CONFLICT")
                blended = .75 * annual_growth + .25 * recent_growth
            if not all(isfinite(value) for value in (annual_growth, blended, pe)):
                raise ValueError("R14_NUMERICAL_RESULT_NOT_FINITE")
            growth = min(maximum_growth, max(-.10, blended))
            rates = [.03 + (growth - .03) * .5 ** year for year in range(5)]
            anchor = values["current_eps"] * Decimal(".9" if inputs["eps_basis"] == "BASIC" else "1")
            scenarios, exact = {}, {}
            for name, factor, exit_pe, scenario_rates in (
                ("BASE", 1., pe, rates),
                ("BEAR", .90, max(5., pe * .80), [max(-.10, rate - .05) for rate in rates]),
                ("BULL", 1.10, min(maximum_pe * 1.20, pe * 1.20), [min(maximum_growth + .05, rate + .05) for rate in rates])):
                terminal = anchor * Decimal(str(factor))
                for rate in scenario_rates:
                    terminal *= 1 + Decimal(str(rate))
                future = terminal * Decimal(str(exit_pe)); fair = future / Decimal("1.08") ** 5
                ratio = float(future / values["current_price"])
                if not all(isfinite(float(value)) and value > 0 for value in (terminal, future, fair)) or not isfinite(ratio) or ratio <= 0:
                    raise ValueError("R14_NUMERICAL_RESULT_NOT_FINITE")
                scenarios[name] = dict(years=5, annual_growth=scenario_rates, growth=scenario_rates[0], eps_factor=factor,
                    exit_pe=exit_pe, terminal_eps=float(terminal), future_price=float(future), fair_price=float(fair),
                    price_cagr=ratio ** .2 - 1)
                exact[name] = fair
            loss = max(Decimal(0), 1 - exact["BEAR"] / values["current_price"])
            checks = dict(BASE_PRICE_CAGR_GTE_REQUIRED_RETURN=exact["BASE"] >= values["current_price"],
                BASE_FAIR_PRICE_GT_CURRENT_PRICE=exact["BASE"] > values["current_price"],
                BEAR_LOSS_LTE_MAXIMUM=loss <= Decimal(".35"), BEAR_LOSS_LTE_HARD_MAXIMUM=loss <= Decimal(".45"))
            band = "NORMAL" if checks["BEAR_LOSS_LTE_MAXIMUM"] else "SMALL_ONLY_WARNING" if checks["BEAR_LOSS_LTE_HARD_MAXIMUM"] else "HARD_LIMIT_EXCEEDED"
        result.update(selected_method=method, method_small_only=small_only, method_position_multiplier=multiplier,
            status="PASS" if all(checks.values()) else "FAIL", reasons=[key for key, passed in checks.items() if not passed],
            anchor=dict(current_reported_eps=inputs["current_eps"], prior_reported_eps=inputs["prior_eps"],
                before_basic_haircut=inputs["current_eps"], scenario_anchor=float(anchor), basis=inputs["eps_basis"]),
            observed_growth=dict(eps=float(values["current_eps"] / values["prior_eps"] - 1),
                revenue=float(values["current_revenue"] / values["prior_revenue"] - 1), annual=annual_growth,
                recent_recovery=recent_growth, blended=blended, bounded_first_year=growth),
            scenarios=scenarios, gate_checks=checks, bear_permanent_loss=float(loss),
            discounted_bear_value_shortfall=float(loss), bear_shortfall_band=band,
            multiple_selection=dict(basis=basis, industry=industry, raw_reference_pe=raw_pe, exit_pe=pe,
                issuer_history_required=method == FULL_METHOD, component_result_id=identity(comparables),
                industry_profile_status="DEFINED" if industry_profile_defined else "NOT_DEFINED",
                fallback_reason="INDUSTRY_PROFILE_NOT_DEFINED" if industry is not None and not industry_profile_defined else
                    "COMPARABLE_COVERAGE_INSUFFICIENT" if method == ANNUAL_METHOD else None))
    except (KeyError, TypeError, ValueError, ArithmeticError, OverflowError) as error:
        return _unknown(result, str(error) or "R14_NUMERICAL_RESULT_NOT_FINITE")
    return result


# Consistent with the R13 arithmetic port for independent admission rechecks.
evaluate_relaxed_price_valuation = evaluate_recovery_price_valuation
