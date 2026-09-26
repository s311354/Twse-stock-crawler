"""Bounded entry eligibility preserves both structural and account-risk limits."""
from copy import deepcopy

import pytest

from domain.strategies import low_entry_v6_signals as signals
from domain.strategies.low_entry_score_v6 import assess_execution
from tests.strategies.test_low_entry_score_v6 import (
    ACCOUNT, CONFIG, execution_report, quality, setup_features,
)


@pytest.mark.parametrize("setup", signals.SETUPS)
@pytest.mark.parametrize("distance,fraction,passed", [
    (3., .08, True), (2.9, .079, True),
    (3.000001, .08, False), (3., .080001, False),
    (None, .08, False), (float("nan"), .08, False), (0., .08, False),
    (3., None, False), (3., float("nan"), False), (3., 0., False),
])
def test_small_requires_both_finite_atr_and_percentage_stop_bounds(setup, distance, fraction, passed):
    f = setup_features(setup)
    if all(signals.number(v) is not None and v > 0 for v in (distance, fraction)):
        # Match ratios to actual prices/ATR, not unrelated cached fields.
        f.update(high=100., atr14=100. * fraction / distance,
                 swing_high=120., hh20=125., hh60=130.)
        f["support"] = 100. * (1 - fraction) + .3 * f["atr14"]
        plan = signals.trade_plan(f, config=CONFIG)
    else:
        plan = dict(signals.trade_plan(f, config=CONFIG), stop_distance_atr=distance,
                    stop_distance_fraction=fraction)
    observed, cap, _ = signals.risk_pass(setup, "BUY_SMALL", f, plan, "NEUTRAL", CONFIG)
    assert observed is passed and cap == 3.


@pytest.mark.parametrize("setup", signals.SETUPS)
def test_missing_percentage_evidence_does_not_default_to_zero_risk(setup):
    f = setup_features(setup)
    plan = signals.trade_plan(f, config=CONFIG)
    plan.pop("stop_distance_fraction")
    assert signals.risk_pass(setup, "BUY_SMALL", f, plan, "NEUTRAL", CONFIG)[0] is False


@pytest.mark.parametrize("fraction", [None, float("nan"), float("inf"), 0., -.01, .080001])
def test_percentage_stop_configuration_cannot_exceed_bounded_limit(fraction):
    with pytest.raises(ValueError):
        signals.LowEntryV6Config(starter_maximum_stop_fraction=fraction)


def test_three_atr_eligibility_reduces_shares_without_raising_account_risk():
    sizes = []
    for distance in (1.7, 2.9):
        f = setup_features("REVERSAL", support=100.4 + .6 - 2. * distance,
                           swing_high=115., hh20=120., hh60=125., break_previous_high=False)
        decision = signals.decide(quality(), f, config=CONFIG)
        assert decision["signal"] == "BUY_SMALL"
        assert decision["plan"]["stop_distance_atr"] == pytest.approx(distance)
        assert decision["risk_fraction"] == .0025
        assert decision["plan"]["stop_price"] == pytest.approx(f["support"] - .3 * f["atr14"])
        shares = signals.size_position(
            decision, portfolio_value=1_000_000., available_cash=1_000_000.,
            intended_weight=1., portfolio_capacity_weight=1., amount_ma20=100_000_000., config=CONFIG,
        )
        per_share_risk = decision["plan"]["entry_price"] - decision["plan"]["stop_price"]
        assert 0 < shares * per_share_risk <= 1_000_000. * .0025
        sizes.append(shares)
    assert sizes[1] < sizes[0]


def test_execution_rejects_percentage_limit_crossing_even_when_atr_and_rr_still_pass():
    f = setup_features("REVERSAL", high=100., close=99.8, support=93., atr14=3.,
                       swing_high=125., hh20=130., hh60=140.)
    decision = signals.decide(quality(), f, config=CONFIG)
    assert decision["signal"] == "BUY_SMALL"
    assert decision["plan"]["stop_distance_fraction"] == pytest.approx(.079)
    report = deepcopy(execution_report())
    report.update(technical=f, decision=decision)
    report["entry"].update(decision["plan"], trigger_price=f["high"])
    original = deepcopy(report)
    fresh_quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00",
                       price=100.05, bid=100., ask=100.05, restricted=False)
    args = dict(now=fresh_quote["timestamp"], trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    assert assess_execution(report, fresh_quote, **args)["action"] == "ELIGIBLE"

    gap_quote = dict(fresh_quote, price=100.2, bid=100.15, ask=100.2)
    gap_plan = signals.trade_plan(f, gap_quote["price"], config=CONFIG, setup=decision["setup"])
    assert gap_plan["stop_price"] == report["entry"]["stop_price"]
    assert gap_plan["stop_distance_atr"] < 3. and gap_plan["reward_risk"] > 2.
    assert gap_plan["stop_distance_fraction"] > .08
    assert gap_plan["liquidity_pass"] is True
    assert assess_execution(report, gap_quote, **args) == {
        "action": "SKIP", "shares": 0, "reason": "REPRICED_RISK_OR_LIQUIDITY_FAIL",
    }
    assert report == original


def shallow_pullback(**changes):
    return setup_features("TREND_PULLBACK", **{
        "drawdown60": -.04, "drawdown20": -.015, "rs_near_high": False, **changes,
    })


@pytest.mark.parametrize("depth", [-.01, -.015, -.02])
def test_one_to_two_percent_pullback_near_averages_is_starter_only(depth):
    f = shallow_pullback(drawdown20=depth)
    result = signals.decide(quality(), f, config=CONFIG)
    setup = result["setups"]["TREND_PULLBACK"]
    assert setup["recovery_checks"]["SHALLOW_PULLBACK_LOCATION"]
    assert result["setup"] == "TREND_PULLBACK" and result["score"] >= CONFIG.buy_score_minimum
    assert result["signal"] == "BUY_SMALL" and result["entry_tier"] == "STARTER_ONLY"
    assert setup["strict_valid"] is False
    assert "STARTER_ONLY_SETUP" in result["entry_checks"]["BUY"]["blockers"]


@pytest.mark.parametrize("changes,blocker", [
    (dict(drawdown20=-.009999), "PULLBACK_DEPTH"),
    (dict(ema20=96., ema10=96.5, vwap20=96., ema20_extension=2.), "SHALLOW_PULLBACK_LOCATION"),
    (dict(ema20_extension=1.000001), "SHALLOW_PULLBACK_LOCATION"),
])
def test_shallow_pullback_cannot_skip_depth_or_average_location(changes, blocker):
    result = signals.decide(quality(), shallow_pullback(**changes), config=CONFIG)
    setup = result["setups"]["TREND_PULLBACK"]
    assert setup["valid"] is False and blocker in setup["blocking_conditions"]
    assert result["signal"] not in signals.EXECUTABLE_SIGNALS


@pytest.mark.parametrize("target", [None, 120.])
@pytest.mark.parametrize("source,limit", [("CONFIRMED_SWING_LOW", 1.), ("CONFIRMED_BREAKOUT_RETEST", 1.5)])
def test_leader_small_extension_boundary_applies_with_or_without_target(target, source, limit):
    f = setup_features("LEADERSHIP_CONTINUATION", ema20_extension=limit)
    plan = dict(signals.trade_plan(f, config=CONFIG), stop_source=source,
                target_price=target, reward_risk=3.)
    assert signals.risk_pass("LEADERSHIP_CONTINUATION", "BUY_SMALL", f, plan, "NEUTRAL", CONFIG)[0]
    assert not signals.risk_pass("LEADERSHIP_CONTINUATION", "BUY_SMALL", dict(f, ema20_extension=limit + .000001),
                                 plan, "NEUTRAL", CONFIG)[0]


@pytest.mark.parametrize("changes", [dict(breakout_extension=1.250001), dict(support_intact=False),
                                     dict(ema20_extension=None), dict(breakout_extension=float("nan"))])
@pytest.mark.parametrize("target", [None, 120.])
def test_known_target_does_not_bypass_leader_extension_or_support(changes, target):
    f = setup_features("LEADERSHIP_CONTINUATION", **changes)
    plan = dict(signals.trade_plan(f, config=CONFIG), target_price=target, reward_risk=3.)
    assert not signals.risk_pass("LEADERSHIP_CONTINUATION", "BUY_SMALL", f, plan, "NEUTRAL", CONFIG)[0]


def test_confirmed_breakout_cannot_promote_overextended_leader_with_known_target():
    f = setup_features("LEADERSHIP_CONTINUATION", support=95.5, ema20=94., ema10=96.,
                       ema20_extension=3., hh20=112., hh60=120., swing_high=115., breakout_extension=-6.)
    starter, confirmed, _ = signals.triggers("LEADERSHIP_CONTINUATION", f)
    assert starter and confirmed
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["plan"]["target_price"] == 112.
    assert result["plan"]["reward_risk"] > 1.8
    assert result["plan"]["stop_distance_atr"] < 3.
    assert result["plan"]["stop_distance_fraction"] < .08
    assert result["signal"] not in signals.EXECUTABLE_SIGNALS
    assert "LEADERSHIP_EXTENSION_OR_SUPPORT_FAIL" in result["entry_checks"]["BUY_SMALL"]["blockers"]
