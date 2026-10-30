"""Finite progress toward a prior high, not fictitious breakout confirmation."""
from collections.abc import Mapping
from math import fsum

from domain.les_v7.contracts import identity
from domain.les_v7.rule_score_r21 import score_factors_r21, score_policy as previous_policy
from domain.strategies.low_entry_v6_hybrid import known_features
from domain.strategies.low_entry_v6_quality import number

VERSION = "V7_HELD_SUPPORT_PREBREAK_PROGRESS_R22"


def score_policy():
    return dict(previous_policy(), version=VERSION,
        close_confirmation=dict(zero_atr=-.5, full_atr=.5,
            maximum_prebreak_fraction=.5, required_true=["history_ready", "market_ready", "support_intact"],
            unchanged_context="LEADERSHIP_CONTINUATION", missing_policy="NO_CREDIT_OR_REDISTRIBUTION"))


def score_factors_r22(features, selection, plan, native_features, *, config, policy,
                      legacy_resilience=False):
    block = policy.get("score_policy", policy) if isinstance(policy, Mapping) else None
    fixed = score_policy()
    if not isinstance(block, Mapping) or identity(dict(block)) != identity(fixed):
        raise ValueError("V7_R22_SCORE_POLICY_MISMATCH")
    result = score_factors_r21(features, selection, plan, native_features,
        config=config, policy=previous_policy(), legacy_resilience=legacy_resilience)
    evidence = result["score_evidence"]
    rule = fixed["close_confirmation"]
    f = known_features(features)
    if selection.context != rule["unchanged_context"]:
        for condition in evidence["conditions"]:
            if condition["id"] != "structure:close_confirmation":
                continue
            # Keep the original known/missing condition and existing breakout
            # points. Additional progress needs every source below; a missing
            # flag is never assumed held, and a broken floor gets no bonus.
            operands = {key: f.get(key) for key in (*rule["required_true"],
                "close", "previous_high", "atr14", "support", "low")}
            close, high, atr, support, low = (number(operands[k]) for k in
                ("close", "previous_high", "atr14", "support", "low"))
            ready = (condition["observed"] and all(f.get(k) is True for k in rule["required_true"])
                and all(v is not None for v in (close, high, atr, support, low))
                and min(close, high, atr, support, low) > 0 and support <= low < close)
            distance = (close - high) / atr if ready else None
            bonus_fraction = (min(1., max(0., (distance - rule["zero_atr"]) /
                (rule["full_atr"] - rule["zero_atr"]))) if ready else 0.)
            fraction = max(condition["fraction"] or 0., bonus_fraction)
            previous_fraction = condition["fraction"]
            condition.update(rule=VERSION + ":HELD_SUPPORT_PROGRESS_NOT_BREAKOUT_PROOF",
                fraction=fraction if condition["observed"] else None,
                earned_points=condition["weight"] * fraction if condition["observed"] else 0.,
                sources=[dict(kind="R22_ADDITIONAL_PREBREAK_PROGRESS", operands=operands,
                    eligible=bool(ready), distance_atr=distance, previous_fraction=previous_fraction,
                    additional_fraction=bonus_fraction, entry_permission_granted=False)],
                additional_dependencies=list(operands))
    for group, component in evidence["components"].items():
        rows = [c for c in evidence["conditions"] if c["group"] == group]
        component["earned_points"] = fsum(c["earned_points"] for c in rows)
        result["factor_budgets"][group]["earned"] = component["earned_points"]
    raw = fsum(c["earned_points"] for c in evidence["conditions"]) if evidence["available_weight"] else None
    evidence.update(policy_version=VERSION, formal_score=raw, earned_points=raw)
    return result
