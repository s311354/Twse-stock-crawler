"""Approved R15 core/tactical separation; assumptions are not fitted odds."""

from domain.les_v7.policy_r14 import risk_limits as _r14_risk_limits


BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R15"
DECISION_VERSION = "RULE_BASED_V6_HYBRID_R15"
# Actual release clock after initial native, source, risk and compatibility checks.
AVAILABLE_AT = "2026-10-16T09:23:52Z"
VALUATION_VERSION = "V7_ROLE_BOUND_RECOVERY_PRICE_RETURN_R15"
INPUT_VERSION = "V7_PRICE_INPUT_READINESS_R15"
ADMISSION_VERSION = "V7_CORE_TACTICAL_TOTAL_CAP_R15"
PLAN_VERSION = "V7_SETUP_STRUCTURE_H20_R15"


def risk_limits():
    """Core risk limits stay R14; tactical limits are explicit and separate."""
    return _r14_risk_limits()


def tactical_risk_limits():
    return dict(maximum_stop_atr=2., minimum_net_rr=1.3,
                total_plan_loss_fraction=.0005, maximum_total_weight=.01)
