"""R4-only execution scoring from the one bound native reference plan.

The first ninety points retain the existing fixed Hybrid formulas. Execution
points are measured scenario quality, never a replacement for native Hard Gates
or an account-bound live order quota. Missing sources earn no redistributed points.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from math import fsum, isclose, isfinite

from domain.les_v7.contracts import ReferencePlan, identity
from domain.strategies.low_entry_v6_execution_score import validate_score_budget
from domain.strategies.low_entry_v6_hybrid import known_features, score_factors
from domain.strategies.low_entry_v6_quality import number
from domain.strategies.low_entry_v6_score_evidence import valid_price_anchor


SCORE_POLICY_VERSION = "V7_BOUND_REFERENCE_EXECUTION_R1"
CONTINUOUS_STOP_SCORE_POLICY_VERSION = "V7_BOUND_REFERENCE_EXECUTION_R2_CONTINUOUS_STOP"


def score_policy() -> dict:
    return dict(version=SCORE_POLICY_VERSION, rr_full_credit=2.,
        execution_weights=dict(stop=4, net_rr=3, spread=1, liquidity=2),
        max_stop_fraction=.08, max_stop_atr=1.5, max_spread=.005, max_reference_order_adv=.01)


def continuous_stop_score_policy() -> dict:
    """Isolated score-only candidate; all R1 limits and factor budgets remain fixed."""
    return dict(score_policy(), version=CONTINUOUS_STOP_SCORE_POLICY_VERSION,
        stop_scoring="LINEAR_UNUSED_RISK_BUDGET")


def _number(value):
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value) else None


def _metric(features, field):
    masks = features.get("score_availability", {})
    if not isinstance(masks, Mapping) or field in masks and masks[field] is not True:
        return None
    return _number(features.get(field))


def _continuous_stop_fraction(stop_atr, stop_fraction, policy):
    if stop_atr is None or stop_fraction is None:
        return None
    utilization = max(stop_atr / policy["max_stop_atr"],
                      stop_fraction / policy["max_stop_fraction"])
    return min(1., max(0., 1. - utilization))


def score_factors_r4(features, selection, plan: ReferencePlan | None, native_features, *, config, policy,
                     legacy_resilience=False):
    """Use source-bound plan costs/ATR; no cached passed flags or V6 payoff RR.

    ``policy`` may be the complete baseline policy or its score_policy block.
    The caller still independently validates the plan, sources and all gates.
    """
    block = policy.get("score_policy", policy) if isinstance(policy, Mapping) else None
    if not isinstance(block, Mapping):
        raise ValueError("V7_BOUND_REFERENCE_SCORE_POLICY_MISMATCH")
    candidate = block.get("version") == CONTINUOUS_STOP_SCORE_POLICY_VERSION
    fixed = continuous_stop_score_policy() if candidate else score_policy()
    if identity(dict(block)) != identity(fixed):
        raise ValueError("V7_BOUND_REFERENCE_SCORE_POLICY_MISMATCH")
    if not isinstance(native_features, Mapping):
        raise ValueError("V7_NATIVE_SCORE_FEATURES_REQUIRED")
    if plan is not None:
        if not isinstance(plan, ReferencePlan): raise ValueError("V7_BOUND_REFERENCE_PLAN_REQUIRED")
        # Validate its complete typed operands even if a caller has bypassed the
        # frozen dataclass constructor. Its content identity includes both costs.
        from domain.les_v7.plan_r19 import plan_from_record
        plan_from_record(asdict(plan))
    f = known_features(features)
    atr = number(f.get("atr14"))
    alias = dict(entry_price=plan.entry if plan else None, stop_price=plan.stop if plan else None,
        target_price=plan.target if plan else None, stop_source="V7_BOUND_H20_STRUCTURE",
        stop_distance_atr=(plan.entry - plan.stop) / atr if plan and atr is not None and atr > 0 else None,
        stop_distance_fraction=(plan.entry - plan.stop) / plan.entry if plan else None)
    # The temporary V6 execution slots are discarded. The unchanged first90
    # conditions are retained, not normalized or independently re-scored.
    result = score_factors(f, selection, alias, config=config, legacy_resilience=legacy_resilience)
    evidence = result["score_evidence"]
    conditions = [condition for condition in evidence["conditions"] if condition["group"] != "execution"]
    anchor = valid_price_anchor(f)
    plan_id = plan.plan_id if plan is not None else None
    stop_atr, stop_fraction = (_number(plan.stop_atr), _number((plan.entry - plan.stop) / plan.entry)) if plan else (None, None)
    spread, adv = _metric(native_features, "spread"), _metric(native_features, "adv")
    if spread is not None and spread < 0: spread = None
    if adv is not None and adv <= 0: adv = None
    participation = _number(plan.entry * plan.reference_shares / adv) if plan and adv is not None else None
    risk = _number(plan.entry - plan.stop + plan.cost_stop) if plan else None
    reward = _number(plan.target - plan.entry - plan.cost_up) if plan and plan.target is not None else None
    rr = _number(plan.net_rr) if plan and risk is not None and reward is not None else None
    stop_credit = (_continuous_stop_fraction(stop_atr, stop_fraction, fixed) if candidate else
        float(stop_atr <= fixed["max_stop_atr"] and stop_fraction <= fixed["max_stop_fraction"])
        if stop_atr is not None and stop_fraction is not None else None)
    values = dict(
        stop=stop_credit,
        net_rr=min(1., max(0., rr / fixed["rr_full_credit"])) if rr is not None else None,
        spread=float(spread <= fixed["max_spread"]) if spread is not None else None,
        liquidity=float(participation <= fixed["max_reference_order_adv"]) if participation is not None else None)
    measured = dict(stop_distance_atr=stop_atr, stop_distance_fraction=stop_fraction, net_reward_risk=rr,
        spread=spread, adv=adv, reference_order_adv=participation,
        net_risk_per_share=risk, net_reward_per_share=reward,
        cost_up_per_share=plan.cost_up if plan else None, cost_stop_per_share=plan.cost_stop if plan else None)
    dependencies = dict(stop=["@reference_plan.entry", "@reference_plan.stop", "@reference_plan.atr"],
        net_rr=["@reference_plan.entry", "@reference_plan.stop", "@reference_plan.target",
                "@reference_plan.cost_up", "@reference_plan.cost_stop"],
        spread=["native:spread"], liquidity=["@reference_plan.reference_shares", "@reference_plan.entry", "native:adv"])
    operands = dict(stop=dict(entry=plan.entry, stop=plan.stop, atr=plan.atr) if plan else None,
        net_rr=dict(entry=plan.entry, stop=plan.stop, target=plan.target, cost_up=plan.cost_up, cost_stop=plan.cost_stop) if plan else None,
        spread=dict(spread=spread), liquidity=dict(entry=plan.entry, reference_shares=plan.reference_shares, adv=adv) if plan else None)
    for name, weight in fixed["execution_weights"].items():
        fraction = values[name]
        observed = anchor and fraction is not None
        conditions.append(dict(id="execution:" + name, group="execution", weight=weight,
            dependencies=dependencies[name], rule=fixed["version"] + ":" + name.upper(),
            observed=observed, status="OBSERVED" if observed else "UNKNOWN", fraction=fraction if observed else None,
            earned_points=weight * fraction if observed else 0., available_weight=weight if observed else 0.,
            sources=[dict(kind="BOUND_NATIVE_REFERENCE_EXECUTION", plan_id=plan_id,
                          operands=operands[name], status="OBSERVED" if observed else "UNKNOWN")]))
    if not isclose(fsum(condition["weight"] for condition in conditions), 100., rel_tol=0., abs_tol=1e-10):
        raise ValueError("V7_BOUND_REFERENCE_SCORE_BUDGET_MISMATCH")
    validate_score_budget({group: [(condition["fraction"], condition["weight"]) for condition in conditions
                                  if condition["group"] == group] for group in evidence["components"]})
    components = {group: dict(total_weight=component["total_weight"],
        earned_points=fsum(c["earned_points"] for c in conditions if c["group"] == group),
        available_weight=fsum(c["available_weight"] for c in conditions if c["group"] == group))
        for group, component in evidence["components"].items()}
    available = fsum(component["available_weight"] for component in components.values())
    raw = fsum(component["earned_points"] for component in components.values()) if available else None
    evidence.update(policy_version=fixed["version"], conditions=conditions, components=components,
        formal_score=raw, earned_points=raw, estimated_score=None, available_weight=available,
        coverage_percent=available, status="COMPLETE" if available == 100 else "PARTIAL" if available else "UNAVAILABLE",
        missing_fields=sorted({field for c in conditions if c["status"] != "OBSERVED" for field in
            (tuple(source["field"] for source in c["sources"] if source.get("fraction") is None)
             if c["sources"] and "field" in c["sources"][0] else c["dependencies"])}))
    result["factor_budgets"] = {group: dict(budget=component["total_weight"], earned=component["earned_points"],
        available_weight=component["available_weight"], condition_ids=[c["id"] for c in conditions if c["group"] == group])
        for group, component in components.items()}
    execution_score = dict(policy_version=fixed["version"],
        weights=dict(fixed["execution_weights"]), limits={key: fixed[key] for key in (
            "rr_full_credit", "max_stop_fraction", "max_stop_atr", "max_spread", "max_reference_order_adv")},
        measured=measured, rr_basis="BOUND_REFERENCE_PLAN_COSTS_AND_LEGAL_GRID", plan_id=plan_id,
        sizing_basis="REFERENCE_ORDER_SUPPORT_NOT_LIVE_QUOTA", **components["execution"])
    if candidate:
        execution_score["stop_scoring"] = fixed["stop_scoring"]
    result["execution_score"] = execution_score
    return result
