"""Synthetic score mechanics; no predictive or performance claim."""
from copy import deepcopy

import pytest

from domain.les_v7.rule_score_r21 import score_factors_r21, score_policy as old_policy
from domain.les_v7.rule_score_r22 import score_factors_r22, score_policy
from domain.strategies.low_entry_v6_hybrid import ContextSelection
from tests.strategies.test_v7_rule_score_r4 import CONFIG, plan
from tests.strategies.test_v7_rule_baseline import unit_features


def score(changes=None, context="REVERSAL"):
    f = dict(unit_features(), history_ready=True, market_ready=True,
        support_intact=True, support=95., low=99., close=100., previous_high=100., atr14=2.)
    f.update(changes or {})
    before = deepcopy(f)
    s = ContextSelection(context, ())
    args = (f, s, plan(), dict(spread=.003, adv=1e8))
    old = score_factors_r21(*args, config=CONFIG, policy=old_policy())
    new = score_factors_r22(*args, config=CONFIG, policy=score_policy())
    assert before == f
    return old, new


@pytest.mark.parametrize("distance,fraction", [(-.5, 0.), (-.25, .25), (0., .5), (.25, .75), (.5, 1.)])
def test_continuous_confirmation_has_fixed_budget_not_entry_permission(distance, fraction):
    old, new = score(dict(previous_high=100.-2.*distance))
    rows = {c["id"]: c for c in new["score_evidence"]["conditions"]}
    row = rows["structure:close_confirmation"]
    assert row["fraction"] == pytest.approx(fraction)
    assert row["weight"] == 4.8
    assert row["sources"][0]["entry_permission_granted"] is False
    assert new["score_evidence"]["formal_score"] - old["score_evidence"]["formal_score"] <= 2.4 + 1e-12
    assert sum(c["weight"] for c in rows.values()) == pytest.approx(100)
    assert new["score_evidence"]["coverage_percent"] == old["score_evidence"]["coverage_percent"]
    for key, previous in old["score_evidence"]["components"].items():
        if key != "structure": assert previous == new["score_evidence"]["components"][key]


@pytest.mark.parametrize("changes", [dict(support_intact=False), dict(support_intact=None),
    dict(support=101.), dict(low=94.), dict(history_ready=False), dict(market_ready=False),
    dict(previous_high=None), dict(atr14=None), dict(close=None),
    dict(score_availability={"support_intact": False})])
def test_no_extra_credit_with_unknown_or_broken_support(changes):
    old, new = score(changes)
    assert old["score_evidence"]["formal_score"] == new["score_evidence"]["formal_score"]


def test_leadership_does_not_get_another_breakout_bonus():
    old, new = score(context="LEADERSHIP_CONTINUATION")
    assert old["score_evidence"]["conditions"] == new["score_evidence"]["conditions"]


def test_policy_tampering_rejected():
    policy = score_policy(); policy["close_confirmation"]["zero_atr"] = -5
    with pytest.raises(ValueError, match="R22_SCORE_POLICY_MISMATCH"):
        score_factors_r22({}, ContextSelection("REVERSAL", ()), None, {}, config=CONFIG, policy=policy)
