"""Synthetic arithmetic only: no historical source repair or return claim."""
from copy import deepcopy
from dataclasses import replace

import pytest

from domain.les_v7.rule_score_r20 import (continuous_stop_credit, early_floor_credit,
    score_factors_r20, score_policy)
from domain.strategies.low_entry_v6_hybrid import ContextSelection
from tests.strategies.test_v7_rule_baseline import unit_features
from tests.strategies.test_v7_rule_score_r4 import CONFIG, plan


def floor(**updates):
    return dict(higher_low=False, swing_low=99.9, previous_swing_low=100.,
        atr14=2., atr_expansion=.9, support_intact=True, no_new_low2=True, **updates)


def test_early_floor_is_partial_causal_and_monotonic_not_a_higher_low():
    policy = score_policy()["early_floor"]
    for low, expected in ((98.9, 0.), (99., 0.), (99.5, .25), (100., .5)):
        value = floor(); value["swing_low"] = low
        assert early_floor_credit(value, policy)[0] == pytest.approx(expected)
    assert early_floor_credit(dict(higher_low=True), policy) == (1., ["higher_low"])
    value = floor(); value["swing_low"] = 101.  # Contradictory known false flag.
    assert early_floor_credit(value, policy)[0] == 0.


@pytest.mark.parametrize("field", list(floor()))
def test_missing_or_masked_floor_source_never_earns_partial_points(field):
    value = floor(); value[field] = None
    assert early_floor_credit(value, score_policy()["early_floor"])[0] is None
    value = floor(); value["score_availability"] = {field: False}
    assert early_floor_credit(value, score_policy()["early_floor"])[0] is None


@pytest.mark.parametrize("field,bad", [("support_intact", False), ("no_new_low2", False),
    ("atr_expansion", 1.01), ("atr14", 0.), ("swing_low", 0.)])
def test_known_failed_stabilization_is_observed_zero(field, bad):
    value = floor(); value[field] = bad
    assert early_floor_credit(value, score_policy()["early_floor"])[0] == 0.


@pytest.mark.parametrize("atr,expected", [(1., 1.), (1.5, 1.), (1.50001, .99999),
    (1.75, .75), (2., .5), (2.25, .25), (2.5, 0.), (3., 0.)])
def test_stop_is_continuous_and_bounded_without_changing_risk_limits(atr, expected):
    policy = score_policy()
    assert continuous_stop_credit(atr, .04, policy) == pytest.approx(expected)
    assert continuous_stop_credit(atr, .080001, policy) == 0.
    assert continuous_stop_credit(None, .04, policy) is None


def test_exact_fixed_policy_only_and_unmodified_conditions_remain_identical():
    features = unit_features(); features.update(floor())
    before = deepcopy(features)
    selected = ContextSelection("REVERSAL", ())
    p = replace(plan(), atr=1.)
    new = score_factors_r20(features, selected, p, dict(spread=.003, adv=50_000_000.),
        config=CONFIG, policy=score_policy())
    # Both use the same context, not competing context scores.
    from domain.les_v7.rule_score_r4 import score_factors_r4, score_policy as r4_policy
    old = score_factors_r4(features, selected, p, dict(spread=.003, adv=50_000_000.),
        config=CONFIG, policy=r4_policy())
    assert features == before
    changed = {"structure:low_structure", "execution:stop"}
    assert [c for c in old["score_evidence"]["conditions"] if c["id"] not in changed] == [
        c for c in new["score_evidence"]["conditions"] if c["id"] not in changed]
    ledger = new["score_evidence"]
    assert sum(c["weight"] for c in ledger["conditions"]) == pytest.approx(100)
    assert ledger["formal_score"] == pytest.approx(sum(c["earned_points"] for c in ledger["conditions"]))
    assert ledger["available_weight"] == pytest.approx(sum(c["available_weight"] for c in ledger["conditions"]))
    assert new["execution_score"]["plan_id"] == old["execution_score"]["plan_id"]
    assert new["execution_score"]["measured"] == old["execution_score"]["measured"]
    bad = score_policy(); bad["early_floor"]["max_credit"] = 1.
    with pytest.raises(ValueError, match="R20_SCORE_POLICY_MISMATCH"):
        score_factors_r20(features, selected, p, {}, config=CONFIG, policy=bad)


def test_leadership_structure_and_missing_plan_do_not_gain_floor_or_execution_points():
    f = unit_features(); f.update(floor())
    selected = ContextSelection("LEADERSHIP_CONTINUATION", ())
    new = score_factors_r20(f, selected, None, {}, config=CONFIG, policy=score_policy())
    rows = {c["id"]: c for c in new["score_evidence"]["conditions"]}
    assert "R20" not in rows["structure:low_structure"]["rule"]
    assert rows["execution:stop"]["status"] == "UNKNOWN"
    assert rows["execution:stop"]["earned_points"] == 0.
