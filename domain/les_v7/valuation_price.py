"""Frozen price-return scenarios using reported, comparable per-share earnings.

The source port must independently verify the exact current/comparative original
and its availability. This arithmetic does not certify a financial source, infer
shares, or assert dividends/dilution are zero. It is an uncalibrated scenario
policy, not an estimated probability or a guarantee of future fair value.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from math import isfinite

from domain.les_v7.contracts import identity


POLICY_VERSION = "V7_REPORTED_EPS_PRICE_RETURN_R1"
AVAILABLE_AT = "2026-10-14T00:56:31Z"


def policy() -> dict:
    """Return a detached, complete fixed policy; publication belongs to its port."""
    return dict(
        schema="les-v7-price-return-valuation/v1", policy_version=POLICY_VERSION, available_at=AVAILABLE_AT,
        return_basis="PRICE_RETURN_ONLY", performance_validated=False,
        source_requirement="VERIFIED_SAME_ORIGINAL_CURRENT_COMPARATIVE_EPS_REVENUE_MARGIN",
        maximum_annual_age_days=550,
        eps_anchor="MIN_CURRENT_PRIOR_REPORTED_EPS_AND_MARKET_IMPLIED_EARNINGS_CEILING",
        market_implied_earnings="CURRENT_PRICE_DIVIDED_BY_OFFICIAL_CURRENT_PE_NOT_REPORTED_EPS",
        eps_basis_policy=dict(diluted="PREFERRED_VERIFIED_REPORTED", basic="EXPLICIT_FALLBACK_HAIRCUT"),
        basic_eps_haircut=.90, dividends="EXCLUDED_NOT_IMPUTED",
        dilution="NOT_SEPARATELY_MODELED_NO_OBSERVED_ZERO_CLAIM",
        growth_basis="MIN_REPORTED_EPS_GROWTH_AND_REVENUE_GROWTH",
        required_return=.10, maximum_bear_permanent_loss=.35,
        base=dict(years=5, minimum_growth=-.10, maximum_growth=.10,
                  eps_factor=1., minimum_exit_pe=5., maximum_exit_pe=20.),
        bear=dict(years=3, growth_increment=-.05, minimum_growth=-.10,
                  eps_factor=.85, exit_pe_multiplier=.70, minimum_exit_pe=5.),
        bull=dict(years=7, growth_increment=.05, maximum_growth=.15,
                  eps_factor=1.10, exit_pe_multiplier=1.20, maximum_exit_pe=25.),
        formula=dict(
            terminal_eps="HAIRCUT_ANCHOR * (1 + SCENARIO_GROWTH) ** YEARS * EPS_FACTOR",
            future_price="TERMINAL_EPS * EXIT_PE",
            fair_price="FUTURE_PRICE / (1 + REQUIRED_RETURN) ** YEARS",
            price_cagr="(FUTURE_PRICE / CURRENT_PRICE) ** (1 / YEARS) - 1",
            bear_loss="MAX(0, 1 - BEAR_FAIR_PRICE / CURRENT_PRICE)"),
        pass_conditions=["BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE",
                         "BEAR_LOSS_LTE_MAXIMUM"],
        comparison_arithmetic="DECIMAL_34_DIGIT_EQUIVALENT_PRICE_INEQUALITIES_NO_THRESHOLD_RELAXATION",
    )


def _number(value):
    # Do not accept booleans, coercible strings, NaN/Inf, or missing observations.
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value) else None


def evaluate_price_valuation(*, current_eps, prior_eps, current_revenue, prior_revenue,
                             current_margin, prior_margin, current_price, current_pe,
                             recent_pe_median, eps_basis) -> dict:
    """Compute only from explicit finite operands; upstream owns proof validation.

    Margins are required positive business-quality evidence but do not multiply
    EPS growth again. Thus a margin change is not credited twice. Reported EPS
    includes historical per-share effects; no share count is reverse engineered.
    """
    fixed = policy()
    inputs = {key: _number(value) for key, value in dict(
        current_eps=current_eps, prior_eps=prior_eps,
        current_revenue=current_revenue, prior_revenue=prior_revenue,
        current_margin=current_margin, prior_margin=prior_margin,
        current_price=current_price, current_pe=current_pe,
        recent_pe_median=recent_pe_median).items()}
    inputs["eps_basis"] = eps_basis if eps_basis in ("DILUTED", "BASIC") else None
    missing = ["INVALID_OR_MISSING_" + key.upper() for key, value in inputs.items() if value is None]
    failed = ["NON_POSITIVE_" + key.upper() for key, value in inputs.items()
              if key != "eps_basis" and value is not None and value <= 0]
    result = dict(schema=fixed["schema"], policy_version=POLICY_VERSION, policy_id=identity(fixed),
        policy=fixed, inputs=inputs, status="FAIL" if failed else "UNKNOWN" if missing else None,
        reasons=failed + missing, assumptions=dict(
            return_basis=fixed["return_basis"], dividends=fixed["dividends"], dilution=fixed["dilution"],
            source_verification="REQUIRED_UPSTREAM_NOT_INFERRED", performance_validated=False,
            basic_eps_haircut=fixed["basic_eps_haircut"] if eps_basis == "BASIC" else None),
        anchor=None, observed_growth=None, scenarios=None, bear_permanent_loss=None,
        gate_checks=None)
    if failed or missing:
        return result

    # A binary rounding artifact must not turn mathematically equal fair value
    # into strictly greater value. Bind precision and compare equivalent future
    # price inequalities, without easing any risk or return threshold.
    with localcontext() as decimal_context:
        decimal_context.prec = 34
        numeric = {key: Decimal(str(value)) for key, value in inputs.items() if key != "eps_basis"}
        eps_growth = float(numeric["current_eps"] / numeric["prior_eps"] - 1)
        revenue_growth = float(numeric["current_revenue"] / numeric["prior_revenue"] - 1)
        market_ceiling_exact = numeric["current_price"] / numeric["current_pe"]
        before_haircut_exact = min(numeric["current_eps"], numeric["prior_eps"], market_ceiling_exact)
        anchor_exact = before_haircut_exact * Decimal(str(fixed["basic_eps_haircut"] if eps_basis == "BASIC" else 1.))
    market_ceiling, before_haircut, anchor = map(float, (market_ceiling_exact, before_haircut_exact, anchor_exact))
    if not all(isfinite(value) for value in (eps_growth, revenue_growth, market_ceiling, before_haircut, anchor)):
        result.update(status="UNKNOWN", reasons=["NUMERICAL_RESULT_NOT_FINITE"])
        return result
    base = fixed["base"]
    growth = min(base["maximum_growth"], max(base["minimum_growth"], min(eps_growth, revenue_growth)))
    pe = min(base["maximum_exit_pe"], max(base["minimum_exit_pe"], inputs["recent_pe_median"]))
    operands = dict(
        BASE=(base["years"], growth, base["eps_factor"], pe),
        BEAR=(fixed["bear"]["years"], max(fixed["bear"]["minimum_growth"], growth + fixed["bear"]["growth_increment"]),
              fixed["bear"]["eps_factor"], max(fixed["bear"]["minimum_exit_pe"], pe * fixed["bear"]["exit_pe_multiplier"])),
        BULL=(fixed["bull"]["years"], min(fixed["bull"]["maximum_growth"], growth + fixed["bull"]["growth_increment"]),
              fixed["bull"]["eps_factor"], min(fixed["bull"]["maximum_exit_pe"], pe * fixed["bull"]["exit_pe_multiplier"])),
    )
    scenarios, exact = {}, {}
    with localcontext() as decimal_context:
        decimal_context.prec = 34
        discount = 1 + Decimal(str(fixed["required_return"]))
        for name, (years, scenario_growth, eps_factor, exit_pe) in operands.items():
            terminal = anchor_exact * (1 + Decimal(str(scenario_growth))) ** years * Decimal(str(eps_factor))
            future = terminal * Decimal(str(exit_pe))
            fair = future / discount ** years
            terminal_eps, future_price, fair_price = map(float, (terminal, future, fair))
            ratio = float(future / numeric["current_price"])
            if not all(isfinite(value) and value > 0 for value in (terminal_eps, future_price, fair_price, ratio)):
                result.update(status="UNKNOWN", reasons=["NUMERICAL_RESULT_NOT_FINITE"])
                return result
            scenarios[name] = dict(years=years, growth=scenario_growth, eps_factor=eps_factor, exit_pe=exit_pe,
                terminal_eps=terminal_eps, future_price=future_price, fair_price=fair_price,
                price_cagr=ratio ** (1 / years) - 1)
            exact[name] = dict(future=future, fair=fair, required_future=numeric["current_price"] * discount ** years)
        loss_exact = max(Decimal(0), 1 - exact["BEAR"]["fair"] / numeric["current_price"])
        loss = float(loss_exact)
        checks = dict(BASE_PRICE_CAGR_GTE_REQUIRED_RETURN=exact["BASE"]["future"] >= exact["BASE"]["required_future"],
            BASE_FAIR_PRICE_GT_CURRENT_PRICE=exact["BASE"]["future"] > exact["BASE"]["required_future"],
            BEAR_LOSS_LTE_MAXIMUM=loss_exact <= Decimal(str(fixed["maximum_bear_permanent_loss"])))
    result.update(status="PASS" if all(checks.values()) else "FAIL",
        reasons=[name for name, passed in checks.items() if not passed],
        anchor=dict(current_reported_eps=inputs["current_eps"], prior_reported_eps=inputs["prior_eps"],
            market_implied_earnings_ceiling=market_ceiling, before_basic_haircut=before_haircut,
            scenario_anchor=anchor, basis=eps_basis),
        observed_growth=dict(eps=eps_growth, revenue=revenue_growth), scenarios=scenarios,
        bear_permanent_loss=loss, gate_checks=checks)
    return result
