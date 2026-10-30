"""SYNTHETIC ONLY: R4 fixed score arithmetic, not source repairs or fills."""
from copy import deepcopy
from dataclasses import replace

import pytest

from domain.les_v7.contracts import ReferencePlan, canonical, identity
from domain.les_v7.rule_score_r4 import (CONTINUOUS_STOP_SCORE_POLICY_VERSION, SCORE_POLICY_VERSION,
    continuous_stop_score_policy, score_factors_r4, score_policy)
from domain.strategies.low_entry_v6_hybrid import select_context, score_factors
from domain.strategies.low_entry_v6_signals import relaxed_hybrid_config
from tests.strategies.test_v7_rule_baseline import unit_features


CONFIG = relaxed_hybrid_config()


def plan(rr=2., **changes):
    p = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL", 100., 98.,
                      100.5 + max(rr, 0.) * 2.5, 2., (100., 100.5), 1, .5 - min(rr, 0.) * 2.5, .5)
    return replace(p, **changes)


def score(*, p=None, features=None, native=None, policy=None):
    f = unit_features() if features is None else features
    return score_factors_r4(f, select_context(f), p, dict(spread=.003, adv=50_000_000.) if native is None else native,
                           config=CONFIG, policy=score_policy() if policy is None else policy)


def condition(result, name):
    return next(c for c in result["score_evidence"]["conditions"] if c["id"] == "execution:" + name)


def test_policy_is_detached_and_requires_exact_frozen_budget_and_limits():
    fixed = score_policy(); before = identity(fixed)
    fixed["execution_weights"]["net_rr"] = 99
    assert identity(score_policy()) == before
    with pytest.raises(ValueError, match="SCORE_POLICY_MISMATCH"):
        score(p=plan(), policy=fixed)
    assert score(p=plan(), policy=dict(score_policy=score_policy())) == score(p=plan())


def test_first_ninety_conditions_are_unchanged_and_all_summaries_reconcile():
    f, p = unit_features(), plan()
    before = deepcopy(f)
    legacy = score_factors(f, select_context(f), dict(entry_price=p.entry, stop_price=p.stop, target_price=p.target,
        stop_distance_atr=(p.entry-p.stop)/f["atr14"], stop_distance_fraction=(p.entry-p.stop)/p.entry), config=CONFIG)
    result = score(features=f, p=p)
    assert f == before
    first90 = lambda x: [c for c in x["score_evidence"]["conditions"] if c["group"] != "execution"]
    assert first90(result) == first90(legacy)
    evidence = result["score_evidence"]
    assert sum(c["weight"] for c in evidence["conditions"]) == pytest.approx(100)
    assert result["execution_score"]["weights"] == dict(stop=4, net_rr=3, spread=1, liquidity=2)
    assert result["execution_score"]["earned_points"] == 10
    assert evidence["formal_score"] == pytest.approx(sum(c["earned_points"] for c in evidence["conditions"]))
    assert evidence["coverage_percent"] == pytest.approx(sum(c["available_weight"] for c in evidence["conditions"]))
    for group, summary in result["factor_budgets"].items():
        assert summary["earned"] == pytest.approx(sum(c["earned_points"] for c in evidence["conditions"] if c["group"] == group))
    assert result["execution_score"]["plan_id"] == p.plan_id


@pytest.mark.parametrize("rr,points", [(1.8, 2.7), (1.9, 2.85), (2., 3.), (3., 3.), (0., 0.), (-1., 0.)])
def test_rr_credit_is_continuous_monotonic_and_uses_bound_costs(rr, points):
    p = plan(rr)
    result = score(p=p)
    assert result["execution_score"]["measured"]["net_reward_risk"] == pytest.approx(p.net_rr)
    c = condition(result, "net_rr")
    assert c["earned_points"] == pytest.approx(points)
    assert c["status"] == "OBSERVED" and c["available_weight"] == 3
    assert c["sources"][0]["operands"]["cost_up"] == p.cost_up
    assert c["sources"][0]["operands"]["cost_stop"] == p.cost_stop


def test_cost_or_atr_identity_changes_are_not_masked_by_v6_cached_rr_pass_or_atr():
    f = unit_features(); f.update(net_rr=999., rr_pass=True, atr14=100.)
    p = plan(1.9)
    changed = replace(p, cost_up=p.cost_up + .25)
    result, other = score(features=f, p=p), score(features=f, p=changed)
    assert condition(other, "net_rr")["earned_points"] < condition(result, "net_rr")["earned_points"]
    assert result["execution_score"]["plan_id"] != other["execution_score"]["plan_id"]
    assert result["execution_score"]["measured"]["stop_distance_atr"] == p.stop_atr
    assert condition(score(features=f, p=replace(p, atr=.5)), "stop")["earned_points"] == 0


@pytest.mark.parametrize("p", [None, plan(target=None)])
def test_missing_plan_or_target_earns_no_rr_without_fabricating_target_or_renormalizing(p):
    result = score(p=p)
    c = condition(result, "net_rr")
    assert c["status"] == "UNKNOWN" and c["fraction"] is None
    assert c["earned_points"] == c["available_weight"] == 0
    assert result["score_evidence"]["total_weight"] == 100
    assert result["execution_score"]["measured"]["net_reward_risk"] is None


@pytest.mark.parametrize("field", ["spread", "adv"])
@pytest.mark.parametrize("bad", [None, True, "0.001", float("nan"), float("inf"), -1.])
def test_invalid_native_metrics_are_unknown_not_replaced_with_v6_good_values(field, bad):
    native = dict(spread=.003, adv=50_000_000., liquidity_pass=True, spread_pass=True)
    native[field] = bad
    result = score(p=plan(), native=native)
    c = condition(result, "spread" if field == "spread" else "liquidity")
    assert c["status"] == "UNKNOWN" and c["earned_points"] == c["available_weight"] == 0
    canonical(result)


@pytest.mark.parametrize("field", ["spread", "adv"])
def test_native_availability_masks_are_respected(field):
    native = dict(spread=.003, adv=50_000_000., score_availability={field: False})
    c = condition(score(p=plan(), native=native), "spread" if field == "spread" else "liquidity")
    assert c["status"] == "UNKNOWN"


def test_measured_bad_execution_is_known_zero_not_missing_and_limits_remain_fixed():
    p = plan()
    result = score(p=replace(p, atr=1.), native=dict(spread=.006, adv=100.))
    for name in ("stop", "spread", "liquidity"):
        c = condition(result, name)
        assert c["earned_points"] == 0 and c["status"] == "OBSERVED"
        assert c["available_weight"] == c["weight"]
    assert condition(score(p=replace(p, stop=92., atr=6.)), "stop")["earned_points"] == 4
    assert condition(score(p=replace(p, stop=91.9, atr=6.)), "stop")["earned_points"] == 0
    assert condition(score(p=replace(p, stop=97., atr=2.)), "stop")["earned_points"] == 4
    assert condition(score(p=replace(p, stop=96.99, atr=2.)), "stop")["earned_points"] == 0


def test_continuous_stop_score_is_separate_and_preserves_all_r1_hard_limits():
    baseline, candidate = score_policy(), continuous_stop_score_policy()
    assert candidate["version"] == CONTINUOUS_STOP_SCORE_POLICY_VERSION
    assert candidate["execution_weights"] == baseline["execution_weights"]
    assert {key: candidate[key] for key in ("max_stop_atr", "max_stop_fraction", "max_spread", "max_reference_order_adv")} == {
        key: baseline[key] for key in ("max_stop_atr", "max_stop_fraction", "max_spread", "max_reference_order_adv")}
    near = score(p=replace(plan(), stop=99.4), policy=candidate)
    middle = score(p=replace(plan(), stop=98.5), policy=candidate)
    at_cap = score(p=replace(plan(), stop=97., atr=2.), policy=candidate)
    beyond_cap = score(p=replace(plan(), stop=96.99, atr=2.), policy=candidate)
    assert condition(near, "stop")["earned_points"] == pytest.approx(3.2)
    assert condition(middle, "stop")["earned_points"] == pytest.approx(2.)
    assert condition(at_cap, "stop")["earned_points"] == 0.
    assert condition(beyond_cap, "stop")["earned_points"] == 0.
    assert near["execution_score"]["stop_scoring"] == "LINEAR_UNUSED_RISK_BUDGET"
    assert near["execution_score"]["policy_version"] == CONTINUOUS_STOP_SCORE_POLICY_VERSION
    # R1 remains threshold-scored; the candidate is opt-in only.
    baseline = score(p=replace(plan(), stop=98.5))
    assert condition(baseline, "stop")["earned_points"] == 4.
    assert "stop_scoring" not in baseline["execution_score"]


def test_continuous_stop_score_missing_stop_evidence_remains_unknown():
    candidate = continuous_stop_score_policy()
    result = score(p=None, policy=candidate)
    stop = condition(result, "stop")
    assert stop["status"] == "UNKNOWN" and stop["fraction"] is None
    assert stop["earned_points"] == stop["available_weight"] == 0.


def test_invalid_current_price_anchor_cannot_get_execution_only_positive_score():
    f = unit_features(); f["close"] = None
    result = score(features=f, p=plan())
    assert all(c["status"] == "UNKNOWN" and c["earned_points"] == 0 for c in result["score_evidence"]["conditions"])
    assert result["score_evidence"]["formal_score"] is None
