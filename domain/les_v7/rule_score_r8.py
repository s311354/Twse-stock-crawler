"""Uncapped technical assessment; never a substitute for entry Hard Gates.

The deductions are declared rule assumptions, not calibrated win probabilities.
Missing factors retain zero earned points and the fixed 100-point denominator.
"""
from domain.strategies.low_entry_v6_decision import compose_score


VERSION = "V7_UNCAPPED_TECHNICAL_PENALTIES_R1"


def adjustment_policy():
    return dict(version=VERSION, formula="MAX(0, RAW_MINUS_EXPLICIT_PENALTIES)",
        score_role="TECHNICAL_ASSESSMENT_NOT_ENTRY_AUTHORIZATION", safety_caps_enabled=False,
        deductions=dict(unconfirmed_starter=5., trigger_not_ready=10., no_clear_context=10.),
        overlap_policy="CONTEXT_ELSE_TRIGGER_ELSE_CONFIRMATION_ONE_DEDUCTION",
        market_policy="UNCHANGED_CONFIG_RISK_OFF_DEDUCTION",
        required_entry_gates="UNCHANGED_NON_COMPENSABLE", performance_validated=False)


def compose_penalty_score(ledger, penalties):
    # Reuse condition/budget/finite-value validation, without supplying any cap.
    outcome = compose_score(ledger, penalties, [])
    outcome.update(policy_version=VERSION, safety_cap=None,
                   final_score=outcome["adjusted_score"])
    return outcome
