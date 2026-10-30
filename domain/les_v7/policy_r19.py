"""R19 widens reference entry bands within the unchanged native risk budget.

This is an explicit research assumption, not a fitted return forecast.  R18
archives retain their exact policy and arithmetic.
"""
from domain.les_v7.policy_r18 import risk_limits, route_limits, tactical_risk_limits
from domain.les_v7.policy_r18 import technical_policy as previous_technical_policy

BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R19"
DECISION_VERSION = "RULE_BASED_V6_HYBRID_R19"
AVAILABLE_AT = "2026-10-22T13:56:53Z"
DAILY_ACTIVATED_AT = "2026-10-22T14:13:21Z"
PLAN_VERSION = "V7_RISK_BOUNDED_ENTRY_R19"


def technical_policy():
    result = previous_technical_policy()
    result.update(version="SOURCE_BOUND_RISK_BOUNDED_ENTRY_UNIFIED_FACTOR_SCORE_R19",
        entry_band_policy="SIGNAL_CLOSE_PLUS_ONE_ATR_CLIPPED_AT_UNCHANGED_NATIVE_RISK",
        risk_and_valuation_parameters="UNCHANGED_R18")
    return result
