"""R20 changes technical attainment only; R19 admission/plan/risk stay frozen.

These fixed curves are uncalibrated policy assumptions, not bottom guarantees
or probabilities. Publication and daily activation must not be backdated.
"""
from domain.les_v7.policy_r19 import technical_policy as previous_technical_policy

BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R20"
DECISION_VERSION = "RULE_BASED_V6_HYBRID_R20"
AVAILABLE_AT = "2026-10-28T02:16:09Z"
DAILY_ACTIVATED_AT = "2026-10-28T02:52:11Z"


def technical_policy():
    return dict(previous_technical_policy(),
        version="SOURCE_BOUND_EARLY_STABILIZATION_UNIFIED_FACTOR_SCORE_R20",
        score_changes="PARTIAL_CONFIRMED_FLOOR_AND_CONTINUOUS_STOP_ATTAINMENT",
        risk_and_valuation_parameters="UNCHANGED_R19")
