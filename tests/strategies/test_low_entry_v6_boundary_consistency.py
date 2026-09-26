"""Numerical boundary consistency, not relaxed thresholds or market forecasts."""
from math import inf, nextafter

import pytest

from domain.strategies.low_entry_v6_signals import (
    SETUPS, decide, risk_pass, score_setups, trade_plan,
)
from tests.strategies.test_low_entry_score_v6 import CONFIG, quality, setup_features


def boundary_features(setup="REVERSAL", **changes):
    return setup_features(setup, **dict({
        "high": 100.4, "support": 97.4, "atr14": 2.,
        "swing_high": 120., "hh20": 130., "hh60": 140.,
    }, **changes))


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("offset,points", [(0., 10.), (-1e-7, 6.)])
def test_atr_score_and_risk_gate_agree_at_boundary(setup, offset, points):
    f = boundary_features(setup, support=nextafter(97.4, -inf) + offset)
    plan = trade_plan(f, config=CONFIG)
    assert score_setups(f, plan, config=CONFIG)[setup]["components"]["execution"] == points
    assert risk_pass(setup, "BUY", f, plan, "NEUTRAL", CONFIG)[0] == (offset == 0.)


@pytest.mark.parametrize("setup,minimum", [("REVERSAL", 2.), ("TREND_PULLBACK", 1.8)])
@pytest.mark.parametrize("delta,points", [(-1e-13, 10.), (-1e-7, 7.)])
def test_net_rr_score_uses_same_numerical_equality_as_risk(setup, minimum, delta, points):
    f = boundary_features(setup, high=100., support=99.)
    plan = trade_plan(f, config=CONFIG)
    entry_cost = 100. * (1 + CONFIG.buy_fee_rate)
    net_loss = entry_cost - plan["stop_price"] * (1 - CONFIG.sell_cost_rate)
    # Solve for the target at exactly the configured after-cost RR boundary.
    f["swing_high"] = (entry_cost + minimum * net_loss) / (1 - CONFIG.sell_cost_rate) + delta
    plan = trade_plan(f, config=CONFIG)
    assert score_setups(f, plan, config=CONFIG)[setup]["components"]["execution"] == points
    assert risk_pass(setup, "BUY", f, plan, "NEUTRAL", CONFIG)[0] == (delta == -1e-13)


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("high,atr,support", [(100.4, 2., 97.4), (100., 10., 95.)])
def test_roundoff_does_not_replace_valid_stop_or_downgrade_entry_tier(setup, high, atr, support):
    f = boundary_features(
        setup, high=high, atr14=atr, support=nextafter(support, -inf),
        consolidation_confirmed=True, consolidation_support=98.,
        breakout_retest_confirmed=True, breakout_retest_support=98.,
        breakout_retest_level=99., breakout_retest_age=1.,
    )
    plan = trade_plan(f, config=CONFIG, setup=setup)
    assert plan["stop_source"] == "CONFIRMED_SWING_LOW"
    assert score_setups(f, trade_plan(f, config=CONFIG), config=CONFIG)[setup]["entry_tier"] == "CONFIRMED"
    # A real excess still requires the separately confirmed newer structure.
    f["support"] -= 1e-7
    replaced = trade_plan(f, config=CONFIG, setup=setup)
    assert replaced["stop_source"] == (
        "CONFIRMED_BREAKOUT_RETEST" if setup == "LEADERSHIP_CONTINUATION"
        else "CONFIRMED_CONSOLIDATION_LOW"
    )


def threshold_features(**changes):
    return boundary_features(**dict({
        "support": nextafter(97.4, -inf), "clv": .2,
        "cmf_improving": False, "up_volume_ratio": .9,
        "rsi_cross40": False, "macd_strict": -1., "macd_signal_strict": 0.,
    }, **changes))


def test_float_neighbor_cannot_turn_valid_75_point_buy_into_71_point_small_buy():
    expected = decide(quality(), threshold_features(support=97.4), config=CONFIG)
    actual = decide(quality(), threshold_features(), config=CONFIG)
    assert actual["signal"] == expected["signal"] == "BUY"
    assert actual["score"] == expected["score"] == 75.
    assert actual["plan"]["stop_price"] == pytest.approx(expected["plan"]["stop_price"])
    assert actual["plan"]["target_price"] == expected["plan"]["target_price"]


@pytest.mark.parametrize("quality_changes,feature_changes", [
    ({"status": "REJECT"}, {}),
    ({"status": "DATA_INSUFFICIENT"}, {}),
    ({}, {"restricted": None}),
    ({}, {"restricted": True}),
    ({}, {"new_lows3": 2., "clv": -.7}),
    ({}, {"market_ready": False}),
])
def test_corrected_score_never_overrides_safety_gates(quality_changes, feature_changes):
    result = decide(quality(**quality_changes), threshold_features(**feature_changes), config=CONFIG)
    assert result["signal"] not in {"BUY", "BUY_SMALL"}
    assert result["risk_fraction"] == 0
