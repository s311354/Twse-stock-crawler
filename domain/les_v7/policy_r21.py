"""R21 low-zone scoring interpretation; no new permission to buy falling prices."""
from domain.les_v7.policy_r20 import technical_policy as previous_technical_policy

BASELINE_VERSION = "V7_RULE_BASED_BASELINE_R21"
DECISION_VERSION = "RULE_BASED_CONTEXT_AWARE_HYBRID_R21"
AVAILABLE_AT = "2026-10-28T05:52:32Z"
DAILY_ACTIVATED_AT = "2026-10-28T06:10:33Z"


def technical_policy():
    return dict(previous_technical_policy(), version="LOW_ZONE_OBSERVATION_AND_HORIZON_COVERAGE_R21",
        score_changes="LOW_ZONE_SHORT_TREND_INTERPRETATION_INDEPENDENT_INDUSTRY_HORIZONS",
        entry_permission="UNCHANGED_R19_NOT_GRANTED_BY_SCORING_INTERPRETATION")
