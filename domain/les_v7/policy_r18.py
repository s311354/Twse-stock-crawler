"""R18 graded tactical entry; unknown evidence and loss budgets are unchanged.

These are user-approved investment assumptions, not fitted probabilities or a
guarantee of a low point. Publication of the candidate is not daily activation.
"""
from domain.les_v7.policy_r15 import risk_limits
from domain.les_v7.policy_r17 import EARLY_PULLBACK, PRICE_DISCOVERY


BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R18"
DECISION_VERSION = "RULE_BASED_V6_HYBRID_R18"
AVAILABLE_AT = "2026-10-17T04:03:00Z"
# Explicit research is available first. The daily selector is activated only
# after the full pinned-universe comparison and regressions have completed.
DAILY_ACTIVATED_AT = "2026-10-17T05:49:19Z"
ADMISSION_VERSION = "V7_GRADED_CONDITIONAL_LIMIT_R18"
PLAN_VERSION = "V7_SUPPORT_RESISTANCE_LIMIT_R18"
SUPPORT_RESISTANCE_LIMIT = "SUPPORT_RESISTANCE_LIMIT_R18"
# Public alias used by plan/routing code; both names identify one small-only route.
SUPPORT_RESISTANCE = SUPPORT_RESISTANCE_LIMIT


def tactical_risk_limits():
    return dict(maximum_stop_atr=2.5, minimum_net_rr=1.2,
                total_plan_loss_fraction=.0005, maximum_total_weight=.01)


def route_limits(variant):
    if variant == EARLY_PULLBACK:
        return dict(maximum_total_weight=.005, total_plan_loss_fraction=.0005)
    if variant == PRICE_DISCOVERY:
        return dict(maximum_total_weight=.0025, total_plan_loss_fraction=.00025)
    limits = tactical_risk_limits()
    return dict(maximum_total_weight=limits["maximum_total_weight"],
                total_plan_loss_fraction=limits["total_plan_loss_fraction"])


def technical_policy():
    return dict(version="SOURCE_BOUND_CONDITIONAL_LIMIT_UNIFIED_FACTOR_SCORE_R18",
        fixed_factor_budget=100, buy_minimum=70., small_minimum=60.,
        soft_conditions="EXISTING_FACTOR_POINTS_ONLY_NO_SECOND_BOOLEAN_VETO_OR_DEDUCTION",
        duplicate_deductions_removed=["no_clear_context", "trigger_not_ready", "unconfirmed_starter"],
        causal_structure_and_plan_required=True, unknown_sources_compensable=False,
        neutral_route=dict(variant=SUPPORT_RESISTANCE_LIMIT, signal="BUY_SMALL_ONLY",
                           entry="CONFIRMED_SOURCE_BOUND_SUPPORT_RESISTANCE_CONDITIONAL_LIMIT",
                           unmatched_price="NO_FILL"),
        risk_and_valuation_parameters="R18_TACTICAL_ONLY_CORE_UNCHANGED_R17",
        position_limits_affect_score=False)
