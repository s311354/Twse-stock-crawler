"""Synthetic mechanics only: no bottom, profit or calibrated-odds claim."""
from copy import deepcopy

import pytest

from domain.les_v7.rule_score_r22 import score_factors_r22, score_policy as old_policy
from domain.les_v7.rule_score_r25 import score_factors_r25, score_policy
from domain.strategies.low_entry_v6_hybrid import ContextSelection
from tests.strategies.test_v7_rule_score_r4 import CONFIG, plan
from tests.strategies.test_v7_rule_baseline import unit_features


def pair(changes=None):
    f = dict(unit_features(), history_ready=True, market_ready=True,
        amount_ratio=1., obv_slope5=-1., volume_ma20_strict=100.)
    f.update(changes or {})
    before = deepcopy(f)
    args = (f, ContextSelection("TREND_PULLBACK", ()), plan(), dict(spread=.003, adv=1e8))
    old = score_factors_r22(*args, config=CONFIG, policy=old_policy())
    new = score_factors_r25(*args, config=CONFIG, policy=score_policy())
    assert f == before
    return old, new


@pytest.mark.parametrize("ratio,fraction", [(0., 0.), (.5, 0.), (.75, .25),
    (1., .5), (1.25, .75), (1.5, 1.), (2., 1.)])
def test_normal_turnover_continuous_and_bounded_in_same_ten_point_slot(ratio, fraction):
    old, new = pair(dict(amount_ratio=ratio))
    previous = {c["id"]: c for c in old["score_evidence"]["conditions"]}
    rows = {c["id"]: c for c in new["score_evidence"]["conditions"]}
    row = rows["demand:participation"]
    assert row["weight"] == 10. and row["fraction"] == pytest.approx(fraction)
    assert row["earned_points"] == pytest.approx(10 * fraction)
    assert sum(c["weight"] for c in rows.values()) == pytest.approx(100)
    assert old["score_evidence"]["coverage_percent"] == new["score_evidence"]["coverage_percent"]
    for key in rows:
        if key != "demand:participation": assert rows[key] == previous[key]
    assert old["execution_score"] == new["execution_score"]


@pytest.mark.parametrize("changes", [dict(amount_ratio=None),
    dict(score_availability={"amount_ratio": False})])
def test_missing_turnover_keeps_coverage_and_no_imputation(changes):
    old, new = pair(changes)
    a, b = (next(c for c in x["score_evidence"]["conditions"] if c["id"] == "demand:participation")
            for x in (old, new))
    assert b["fraction"] == a["fraction"] == 0.
    assert b["available_weight"] == a["available_weight"] == 5.
    assert b["sources"][0]["status"] == "UNKNOWN"


def test_both_sources_missing_stays_unknown_and_strong_obv_not_double_counted():
    old, new = pair(dict(amount_ratio=None, obv_slope5=None))
    assert old["score_evidence"]["formal_score"] == new["score_evidence"]["formal_score"]
    row = next(c for c in new["score_evidence"]["conditions"] if c["id"] == "demand:participation")
    assert row["status"] == "UNKNOWN" and row["fraction"] is None
    assert row["available_weight"] == row["earned_points"] == 0.
    old, new = pair(dict(obv_slope5=20.))
    assert old["score_evidence"]["formal_score"] == new["score_evidence"]["formal_score"]


def test_policy_tampering_is_rejected():
    policy = score_policy(); policy["amount_participation"]["zero_ratio"] = -1.
    with pytest.raises(ValueError, match="R25_SCORE_POLICY_MISMATCH"):
        score_factors_r25({}, ContextSelection("REVERSAL", ()), None, {}, config=CONFIG, policy=policy)
