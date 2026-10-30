"""User-approved daily R13 assumptions, not fitted probabilities or guarantees.

Keep identity separate from earlier baselines. Release time is frozen after
initial offline checks, before final deployment verification.
"""

BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R13"
DECISION_VERSION = "RULE_BASED_V6_HYBRID_R13"
# Actual publication clock after focused old/new arithmetic and risk checks.
AVAILABLE_AT = "2026-10-16T00:21:12Z"
VALUATION_VERSION = "V7_ANNUAL_EPS_GRADED_PRICE_RETURN_R13"
INPUT_VERSION = "V7_PRICE_INPUT_READINESS_R13"
ADMISSION_VERSION = "V7_GRADED_ENTRY_AND_TOTAL_CAP_R13"
PLAN_VERSION = "V7_SETUP_STRUCTURE_H20_R13"


def risk_limits():
    """One R13 source for eligibility; scoring curves remain unchanged."""
    return dict(maximum_spread=.005, maximum_reference_order_adv=.01, maximum_stop_atr=1.8,
        starter_stop_atr=dict(REVERSAL=1.8, TREND_PULLBACK=1.8, LEADERSHIP_CONTINUATION=1.8),
        minimum_net_rr=dict(REVERSAL=1.8, TREND_PULLBACK=1.6, LEADERSHIP_CONTINUATION=1.6),
        source_bound_hard_gates_required=True, buy_quality_minimum=65., small_quality_minimum=60.)
