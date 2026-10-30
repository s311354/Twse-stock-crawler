"""Approved R14 component selection; neither fitted odds nor a profit guarantee."""

from domain.les_v7.policy_r13 import risk_limits as _r13_risk_limits


BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R14"
DECISION_VERSION = "RULE_BASED_V6_HYBRID_R14"
# Actual release clock, frozen after the initial offline component checks.
AVAILABLE_AT = "2026-10-16T06:28:12Z"
VALUATION_VERSION = "V7_RECOVERY_COMPONENT_PRICE_RETURN_R14"
INPUT_VERSION = "V7_PRICE_INPUT_READINESS_R14"
ADMISSION_VERSION = "V7_GRADED_ENTRY_AND_TOTAL_CAP_R14"
PLAN_VERSION = "V7_SETUP_STRUCTURE_H20_R14"


def risk_limits():
    """Eligibility limits remain R13; new method sizing is independent of score."""
    return _r13_risk_limits()
