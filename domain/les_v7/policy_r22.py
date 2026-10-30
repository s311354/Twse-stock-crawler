"""Forward-only soft-score/source semantics, not a promise of bottom timing."""
from domain.les_v7.policy_r21 import technical_policy as previous_technical_policy

BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R22"
DECISION_VERSION = "RULE_BASED_PROGRESSIVE_CONFIRMATION_R22"
AVAILABLE_AT = "2026-10-28T06:30:13Z"
DAILY_ACTIVATED_AT = "2026-10-28T06:42:55Z"
SOURCE_SELECTION_VERSION = "V7_STRUCTURE_SOURCE_SEMANTICS_R22"


def technical_policy():
    return dict(previous_technical_policy(), version="PROGRESSIVE_CONFIRMATION_SOURCE_SEMANTICS_R22",
        score_changes="BOUNDED_PREBREAK_CREDIT_ONLY_WITH_VERIFIED_HELD_SUPPORT",
        source_selection=SOURCE_SELECTION_VERSION,
        soft_missing="ZERO_POINTS_NOT_SECOND_STRUCTURE_VETO",
        hard_sources_and_risk="UNCHANGED_R19")
