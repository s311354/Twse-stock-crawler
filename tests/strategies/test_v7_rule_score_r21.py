"""Synthetic interpretation, not investment-performance evidence."""
from copy import deepcopy

import pytest

from domain.les_v7.rule_score_r21 import interpretation, score_factors_r21, score_policy
from domain.strategies.low_entry_v6_hybrid import ContextSelection
from tests.strategies.test_v7_rule_baseline import unit_features
from tests.strategies.test_v7_rule_score_r4 import CONFIG, plan


def low_zone():
    return dict(unit_features(), position60=.2, drawdown60=-.2,
        ema20_slope=-.2, ema5_slope=.1, ema20=110., ema5=100., history_ready=True, market_ready=True)


def test_low_zone_is_selected_before_scoring_not_an_entry_setup_or_oversold_bonus():
    f = low_zone(); before = deepcopy(f)
    selection = ContextSelection("NO_CLEAR_CONTEXT", ())
    value = score_factors_r21(f, selection, plan(), dict(spread=.003, adv=1e8), config=CONFIG, policy=score_policy())
    audit = value["score_evidence"]["scoring_interpretation"]
    assert audit["formal_context"] == "NO_CLEAR_CONTEXT"
    assert audit["scoring_context"] == "REVERSAL"
    assert audit["interpretation"] == "LOW_ZONE_OBSERVATION_NOT_ENTRY_SETUP"
    assert not audit["entry_permission_granted"] and not value["recognized"]
    rows = {c["id"]: c for c in value["score_evidence"]["conditions"]}
    assert rows["trend:ema_progress"]["dependencies"] == ["ema5_slope"]
    assert rows["trend:extension"]["dependencies"] == ["close", "ema5", "atr14"]
    assert sum(c["weight"] for c in rows.values()) == pytest.approx(100.)
    assert f == before and selection.context == "NO_CLEAR_CONTEXT"


@pytest.mark.parametrize("changes", [dict(history_ready=False), dict(market_ready=False),
    dict(position60=.8, drawdown60=-.01), dict(position60=None, drawdown60=None)])
def test_missing_or_non_low_zone_does_not_get_reversal_interpretation(changes):
    f = low_zone(); f.update(changes)
    value = interpretation(f, ContextSelection("NO_CLEAR_CONTEXT", ()))
    assert value["scoring_context"] == "NO_CLEAR_CONTEXT"


@pytest.mark.parametrize("context", ["REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION"])
def test_qualified_formal_context_is_not_reselected_to_maximize_points(context):
    value = interpretation(low_zone(), ContextSelection(context, ()))
    assert value["scoring_context"] == context
    assert value["interpretation"] == "FORMAL_CONTEXT"


def test_missing_short_features_earn_no_automatic_recovery_credit():
    f = low_zone(); f.update(ema5_slope=None, ema5=None, rsi_change=-3.)
    result = score_factors_r21(f, ContextSelection("NO_CLEAR_CONTEXT", ()), None, {},
        config=CONFIG, policy=score_policy())
    rows = {c["id"]: c for c in result["score_evidence"]["conditions"]}
    assert rows["trend:ema_progress"]["status"] == "UNKNOWN"
    assert rows["trend:extension"]["earned_points"] == 0.
    assert rows["momentum:rsi_direction"]["earned_points"] == 0.
    assert rows["execution:net_rr"]["status"] == "UNKNOWN"
