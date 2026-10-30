"""Interpret a known low-zone observation before scoring, not as entry proof.

R20 treated every NO_CLEAR_CONTEXT observation with the general EMA20 trend
curve, including low-zone recovery observations. R21 uses the existing reversal
EMA5/extension rules there; weights, penalties, plans and hard risks do not change.
"""
from collections.abc import Mapping

from domain.les_v7.contracts import identity
from domain.les_v7.rule_score_r20 import score_factors_r20, score_policy as previous_policy
from domain.strategies.low_entry_v6_hybrid import ContextSelection, NO_CLEAR_CONTEXT, known_features
from domain.strategies.low_entry_v6_quality import number

VERSION = "V7_LOW_ZONE_OBSERVATION_SCORE_R21"


def score_policy():
    return dict(previous_policy(), version=VERSION,
        observation_interpretation=dict(history_and_market_required=True,
            maximum_position60=.4, maximum_drawdown60=-.06,
            selected_before_score=True, entry_context_changed=False,
            observed_low_zone="EXISTING_REVERSAL_EMA5_CURVES_NO_OVERSOLD_BONUS",
            missing_policy="NO_DEFAULT_NEUTRAL_OR_POSITIVE_POINTS"))


def interpretation(features, selection):
    f = known_features(features)
    position, drawdown = (number(f.get(key)) for key in ("position60", "drawdown60"))
    position = position if position is not None and 0 <= position <= 1 else None
    drawdown = drawdown if drawdown is not None and -1 < drawdown <= 0 else None
    ready = f.get("history_ready") is True and f.get("market_ready") is True
    low_zone = ((position is not None and position <= .4) or
                (drawdown is not None and drawdown <= -.06))
    early = selection.context == NO_CLEAR_CONTEXT and ready and low_zone
    return dict(version=VERSION, formal_context=selection.context,
        scoring_context="REVERSAL" if early else selection.context,
        interpretation="LOW_ZONE_OBSERVATION_NOT_ENTRY_SETUP" if early else "FORMAL_CONTEXT",
        inputs=dict(position60=position, drawdown60=drawdown, history_and_market_ready=ready),
        entry_permission_granted=False, score_or_account_used_for_selection=False)


def score_factors_r21(features, selection, plan, native_features, *, config, policy,
                      legacy_resilience=False):
    block = policy.get("score_policy", policy) if isinstance(policy, Mapping) else None
    if not isinstance(block, Mapping) or identity(dict(block)) != identity(score_policy()):
        raise ValueError("V7_R21_SCORE_POLICY_MISMATCH")
    audit = interpretation(features, selection)
    scoring_selection = ContextSelection(audit["scoring_context"], selection.evidence)
    result = score_factors_r20(features, scoring_selection, plan, native_features,
        config=config, policy=previous_policy(), legacy_resilience=legacy_resilience)
    result["valid"] = result["recognized"] = selection.context != NO_CLEAR_CONTEXT
    result["score_evidence"].update(policy_version=VERSION, scoring_interpretation=audit)
    return result
