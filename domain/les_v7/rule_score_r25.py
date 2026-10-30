"""Normal turnover earns partial participation credit, not evidence of a bottom.

Only the amount-ratio source in the existing MAX pool changes. OBV, coverage,
all other slots and the fixed 100-point denominator are preserved exactly.
"""
from collections.abc import Mapping
from math import fsum

from domain.les_v7.contracts import identity
from domain.les_v7.rule_score_r22 import score_factors_r22, score_policy as previous_policy
from domain.strategies.low_entry_v6_hybrid import known_features
from domain.strategies.low_entry_v6_quality import number

VERSION = "V7_NORMAL_TURNOVER_PARTICIPATION_R25"


def score_policy():
    return dict(previous_policy(), version=VERSION,
        amount_participation=dict(zero_ratio=.5, full_ratio=1.5,
            normal_ratio=1., normal_fraction=.5,
            pooling="MAX_WITH_UNCHANGED_FIXED_COVERAGE",
            missing_policy="ZERO_POINTS_NO_REDISTRIBUTION",
            interpretation="PARTICIPATION_NOT_BREAKOUT_OR_ENTRY_PROOF"))


def score_factors_r25(features, selection, plan, native_features, *, config, policy,
                      legacy_resilience=False):
    block = policy.get("score_policy", policy) if isinstance(policy, Mapping) else None
    fixed = score_policy()
    if not isinstance(block, Mapping) or identity(dict(block)) != identity(fixed):
        raise ValueError("V7_R25_SCORE_POLICY_MISMATCH")
    result = score_factors_r22(features, selection, plan, native_features,
        config=config, policy=previous_policy(), legacy_resilience=legacy_resilience)
    evidence = result["score_evidence"]
    ratio = number(known_features(features).get("amount_ratio"))
    curve = fixed["amount_participation"]
    for row in evidence["conditions"]:
        if row["id"] != "demand:participation":
            continue
        # Preserve the old source's missing/availability verdict. A stale value
        # hidden by a mask cannot be made usable by the new curve.
        for source in row["sources"]:
            if source["field"] == "amount_ratio" and source["fraction"] is not None and ratio is not None:
                source["fraction"] = min(1., max(0., (ratio - curve["zero_ratio"]) /
                    (curve["full_ratio"] - curve["zero_ratio"])))
                source["operands"] = dict(amount_ratio=ratio)
        known = [s["fraction"] for s in row["sources"] if s["fraction"] is not None]
        coverage = len(known) / len(row["sources"])
        fraction = max(known) * coverage if known and row["observed"] else None
        row.update(rule=VERSION + ":MAX_WITH_FIXED_COVERAGE",
            fraction=fraction, earned_points=row["weight"] * fraction if fraction is not None else 0.)
    for group, component in evidence["components"].items():
        component["earned_points"] = fsum(c["earned_points"] for c in evidence["conditions"] if c["group"] == group)
        result["factor_budgets"][group]["earned"] = component["earned_points"]
    raw = fsum(c["earned_points"] for c in evidence["conditions"]) if evidence["available_weight"] else None
    evidence.update(policy_version=VERSION, formal_score=raw, earned_points=raw)
    return result
