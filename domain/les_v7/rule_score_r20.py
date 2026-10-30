"""Bounded, causal partial credit; never substitute points for Hard Gates."""
from collections.abc import Mapping
from math import fsum

from domain.les_v7.contracts import identity
from domain.les_v7.rule_score_r4 import score_factors_r4, score_policy as previous_policy
from domain.strategies.low_entry_v6_hybrid import known_features
from domain.strategies.low_entry_v6_quality import number

SCORE_POLICY_VERSION = "V7_EARLY_STABILIZATION_CONTINUOUS_STOP_R20"


def score_policy():
    return dict(previous_policy(), version=SCORE_POLICY_VERSION,
        early_floor=dict(max_credit=.5, maximum_lower_low_atr=.5,
            required_true=["support_intact", "no_new_low2"], maximum_atr_expansion=1.,
            pivots="CAUSALLY_CONFIRMED_DISTINCT_SWING_LOWS_ONLY"),
        stop_scoring="FULL_TO_1_5_ATR_LINEAR_TO_ZERO_AT_2_5_ATR",
        stop_zero_atr=2.5, hard_limits="UNCHANGED_R19_NOT_REPLACED_BY_SCORE")


def _bool(value):
    return value if type(value) is bool else None


def early_floor_credit(features, policy):
    """A lower/flat floor gets at most half credit after verified stabilization.

    A confirmed higher low retains full credit without new dependencies. Missing
    evidence is not a failed observation and cannot earn the partial credit.
    """
    f = known_features(features)
    higher = _bool(f.get("higher_low"))
    if higher is True:
        return 1., ["higher_low"]
    fields = ["higher_low", "swing_low", "previous_swing_low", "atr14",
              "atr_expansion", *policy["required_true"]]
    values = [number(f.get(key)) for key in fields[1:5]]
    flags = [_bool(f.get(key)) for key in policy["required_true"]]
    if higher is None or any(v is None for v in (*values, *flags)):
        return None, fields
    low, previous, atr, expansion = values
    if (min(low, previous, atr) <= 0 or any(flag is False for flag in flags)
            or expansion <= 0 or expansion > policy["maximum_atr_expansion"]):
        return 0., fields
    # Do not give a full higher-low score to a contradiction in source operands.
    if low > previous:
        return 0., fields
    fraction = max(0., 1. - (previous - low) / (policy["maximum_lower_low_atr"] * atr))
    return policy["max_credit"] * fraction, fields


def continuous_stop_credit(stop_atr, stop_fraction, policy):
    if stop_atr is None or stop_fraction is None:
        return None
    if stop_atr <= 0 or stop_fraction <= 0 or stop_fraction > policy["max_stop_fraction"]:
        return 0.
    return min(1., max(0., (policy["stop_zero_atr"] - stop_atr) /
                          (policy["stop_zero_atr"] - policy["max_stop_atr"])))


def score_factors_r20(features, selection, plan, native_features, *, config, policy,
                      legacy_resilience=False):
    block = policy.get("score_policy", policy) if isinstance(policy, Mapping) else None
    fixed = score_policy()
    if not isinstance(block, Mapping) or identity(dict(block)) != identity(fixed):
        raise ValueError("V7_R20_SCORE_POLICY_MISMATCH")
    # Reuse the exact validated R4 plan/cost/coverage calculation. Only two
    # bounded slots change; other factor conditions remain byte-for-byte equal.
    result = score_factors_r4(features, selection, plan, native_features,
        config=config, policy=previous_policy(), legacy_resilience=legacy_resilience)
    evidence = result["score_evidence"]
    for condition in evidence["conditions"]:
        if condition["id"] == "structure:low_structure" and selection.context != "LEADERSHIP_CONTINUATION":
            fraction, fields = early_floor_credit(features, fixed["early_floor"])
            condition.update(dependencies=fields, sources=[dict(kind="CAUSAL_EARLY_FLOOR_R20",
                operands={key: known_features(features).get(key) for key in fields})])
        elif condition["id"] == "execution:stop":
            measured = result["execution_score"]["measured"]
            fraction = continuous_stop_credit(measured["stop_distance_atr"],
                                               measured["stop_distance_fraction"], fixed)
        else:
            continue
        observed = condition["observed"] and fraction is not None
        condition.update(rule=fixed["version"] + ":" + condition["id"],
            observed=observed, status="OBSERVED" if observed else "UNKNOWN",
            fraction=fraction if observed else None,
            earned_points=condition["weight"] * fraction if observed else 0.,
            available_weight=condition["weight"] if observed else 0.)
    for group, component in evidence["components"].items():
        rows = [c for c in evidence["conditions"] if c["group"] == group]
        component.update(earned_points=fsum(c["earned_points"] for c in rows),
                         available_weight=fsum(c["available_weight"] for c in rows))
        result["factor_budgets"][group].update(earned=component["earned_points"],
                                               available_weight=component["available_weight"])
    available = fsum(c["available_weight"] for c in evidence["conditions"])
    raw = fsum(c["earned_points"] for c in evidence["conditions"]) if available else None
    evidence.update(policy_version=fixed["version"], formal_score=raw, earned_points=raw,
        available_weight=available, coverage_percent=available,
        status="COMPLETE" if available == 100 else "PARTIAL" if available else "UNAVAILABLE",
        missing_fields=sorted({field for c in evidence["conditions"] if c["status"] != "OBSERVED"
                               for field in c["dependencies"]}))
    result["execution_score"].update(policy_version=fixed["version"],
        stop_scoring=fixed["stop_scoring"], stop_zero_atr=fixed["stop_zero_atr"],
        **evidence["components"]["execution"])
    return result
