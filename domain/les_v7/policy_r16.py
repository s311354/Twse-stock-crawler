"""Approved R16 causal conditional starters; not fitted probabilities."""

from domain.les_v7.policy_r15 import risk_limits, tactical_risk_limits

BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R16"
DECISION_VERSION = "RULE_BASED_V6_HYBRID_R16"
# Actual publication clock, after initial causal/native/source regression.
AVAILABLE_AT = "2026-10-16T11:22:27Z"
ADMISSION_VERSION = "V7_CONDITIONAL_STRUCTURE_TOTAL_CAP_R16"
PLAN_VERSION = "V7_SETUP_STRUCTURE_H20_R16"

EARLY_PULLBACK = "EARLY_PULLBACK_FLOOR_R16"
PRICE_DISCOVERY = "PRICE_DISCOVERY_MEASURED_TARGET_R16"


def route_limits(variant):
    if variant == EARLY_PULLBACK:
        return dict(maximum_total_weight=.005, total_plan_loss_fraction=.0005)
    if variant == PRICE_DISCOVERY:
        return dict(maximum_total_weight=.0025, total_plan_loss_fraction=.00025)
    limits = tactical_risk_limits()
    return dict(maximum_total_weight=limits["maximum_total_weight"],
                total_plan_loss_fraction=limits["total_plan_loss_fraction"])
