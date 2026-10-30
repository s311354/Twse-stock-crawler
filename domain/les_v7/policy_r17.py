"""R17 scores technical strength once; source and plan safety remain absolute.

No new valuation, RR, ATR, cost, weight or probability assumptions are introduced.
The release clock is distinct from the dates of the input price observations.
"""
from domain.les_v7.policy_r15 import risk_limits, tactical_risk_limits

BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R17"
DECISION_VERSION = "RULE_BASED_V6_HYBRID_R17"
# Frozen publication clock after initial native/source/causality checks.
AVAILABLE_AT = "2026-10-16T15:16:03Z"
ADMISSION_VERSION = "V7_UNIFIED_SOFT_TECHNICAL_R17"
PLAN_VERSION = "V7_SETUP_STRUCTURE_H20_R17"
EARLY_PULLBACK = "EARLY_PULLBACK_FLOOR_R17"
PRICE_DISCOVERY = "PRICE_DISCOVERY_MEASURED_TARGET_R17"


def route_limits(variant):
    if variant == EARLY_PULLBACK:
        return dict(maximum_total_weight=.005, total_plan_loss_fraction=.0005)
    if variant == PRICE_DISCOVERY:
        return dict(maximum_total_weight=.0025, total_plan_loss_fraction=.00025)
    limits = tactical_risk_limits()
    return dict(maximum_total_weight=limits["maximum_total_weight"],
                total_plan_loss_fraction=limits["total_plan_loss_fraction"])


def technical_policy():
    return dict(version="SOURCE_BOUND_STRUCTURE_UNIFIED_FACTOR_SCORE_R17",
        fixed_factor_budget=100, buy_minimum=72., small_minimum=62.,
        soft_conditions="EXISTING_FACTOR_POINTS_ONLY_NO_SECOND_BOOLEAN_VETO_OR_DEDUCTION",
        duplicate_deductions_removed=["no_clear_context", "trigger_not_ready", "unconfirmed_starter"],
        causal_structure_and_plan_required=True, unknown_sources_compensable=False,
        risk_and_valuation_parameters="UNCHANGED_R16", position_limits_affect_score=False)
