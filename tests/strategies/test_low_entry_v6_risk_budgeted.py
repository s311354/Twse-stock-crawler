"""Broader starters remain small, conditional, and bounded by structural risk."""
from copy import deepcopy

import pytest

from domain.strategies import low_entry_v6_signals as signals
from domain.strategies.low_entry_score_v6 import assess_execution
from tests.strategies.test_low_entry_score_v6 import (
    ACCOUNT, CONFIG, execution_report, quality, setup_features,
)


@pytest.mark.parametrize("setup", signals.SETUPS)
def test_wider_structural_stop_allows_only_small_not_full_buy(setup):
    # High score cannot overcome the new standard BUY cap (1.8 ATR).
    f = setup_features(setup, support=97.2, swing_high=115., hh20=120., hh60=125.)
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["score"] >= CONFIG.buy_score_minimum
    assert result["signal"] == "BUY_SMALL"
    assert result["plan"]["stop_distance_atr"] == pytest.approx(1.9)
    assert result["risk_fraction"] == .0025
    # Inspect the requested setup independently: the common risk ranking may
    # select another eligible starter, whose BUY also lacks a confirmed shape.
    assert result["setup_decisions"][setup]["entry_checks"]["BUY"]["blockers"] == ["STOP_DISTANCE_EXCEEDS_LIMIT"]
    assert "STOP_DISTANCE_EXCEEDS_LIMIT" in result["entry_checks"]["BUY"]["blockers"]
    assert result["entry_checks"]["BUY_SMALL"]["blockers"] == []


@pytest.mark.parametrize("setup", signals.SETUPS)
@pytest.mark.parametrize("distance,passed", [(3., True), (3.000001, False), (0, False), (None, False)])
def test_starter_stop_boundary_is_finite_and_inclusive(setup, distance, passed):
    f = setup_features(setup, swing_high=120., hh20=125., hh60=130.)
    if distance is not None:
        f["support"] = f["high"] + .3 * f["atr14"] - distance * f["atr14"]
    else:
        f["atr14"] = None
    plan = signals.trade_plan(f, config=CONFIG)
    observed, cap, _ = signals.risk_pass(setup, "BUY_SMALL", f, plan, "NEUTRAL", CONFIG)
    assert observed is passed and cap == 3.


@pytest.mark.parametrize("setup", signals.SETUPS)
@pytest.mark.parametrize("valuation,extra", [("NEUTRAL", 0), ("EXPENSIVE", .2)])
def test_cost_aware_rr_boundary_for_wider_starters(setup, valuation, extra):
    f = setup_features(setup, support=97.6)
    plan = signals.trade_plan(f, config=CONFIG)
    minimum = CONFIG.starter_minimum_net_rr + extra
    entry_cost = plan["entry_price"] * (1 + CONFIG.buy_fee_rate)
    loss = entry_cost - plan["stop_price"] * (1 - CONFIG.sell_cost_rate)
    plan["target_price"] = (entry_cost + minimum * loss) / (1 - CONFIG.sell_cost_rate)
    assert signals.risk_pass(setup, "BUY_SMALL", f, plan, valuation, CONFIG) == (True, 3., minimum)
    plan["target_price"] -= .000001
    assert signals.risk_pass(setup, "BUY_SMALL", f, plan, valuation, CONFIG)[0] is False


@pytest.mark.parametrize("setup", signals.SETUPS)
def test_riskoff_keeps_stricter_stop_and_confirmation(setup):
    f = setup_features(setup, market_regime="RISK_OFF", support=97.6)
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["signal"] not in signals.EXECUTABLE_SIGNALS
    plan = signals.trade_plan(f, config=CONFIG)
    assert signals.risk_pass(setup, "BUY_SMALL", f, plan, "NEUTRAL", CONFIG)[1] == pytest.approx(1.53)


@pytest.mark.parametrize("setup", signals.SETUPS)
def test_score_execution_component_keeps_tight_stop_budget(setup):
    f = setup_features(setup, support=97.6)
    plan = signals.trade_plan(f, config=CONFIG)
    wide = signals.score_setups(f, plan, config=CONFIG)[setup]
    wider_features = dict(f, support=f["high"] + .3 * f["atr14"] - 1.800001 * f["atr14"])
    wider_plan = signals.trade_plan(wider_features, config=CONFIG)
    too_wide = signals.score_setups(wider_features, wider_plan, config=CONFIG)[setup]
    assert wide["score"] - too_wide["score"] == pytest.approx(4)
    assert wide["components"]["execution"] - too_wide["components"]["execution"] == 4
    assert wide["valid"] == too_wide["valid"]


def test_wider_stop_reduces_shares_instead_of_increasing_account_risk():
    args = dict(portfolio_value=1_000_000, available_cash=1_000_000, intended_weight=1.,
                portfolio_capacity_weight=1., amount_ma20=100_000_000, config=CONFIG)
    shares = []
    for support in (98.6, 97.6):
        result = signals.decide(quality(), setup_features("REVERSAL", support=support,
                                                        break_previous_high=False), config=CONFIG)
        assert result["signal"] == "BUY_SMALL"
        size = signals.size_position(result, **args)
        risk = result["plan"]["entry_price"] - result["plan"]["stop_price"]
        assert 0 < size * risk <= 1_000_000 * .0025
        shares.append(size)
    assert shares[1] < shares[0]


def test_leadership_can_use_multihorizon_rs_not_only_peak_resilience():
    f = setup_features("LEADERSHIP_CONTINUATION", resilience=-.01, break_previous_high=False,
                       hh20=100.2, support=97.6)
    starter, confirmed, _ = signals.triggers("LEADERSHIP_CONTINUATION", f)
    assert starter and not confirmed
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["signal"] == "BUY_SMALL"
    # The alternative cannot bypass a Risk-Off market's positive-resilience gate.
    assert signals.decide(quality(), dict(f, market_regime="RISK_OFF"), config=CONFIG)["signal"] == "ARMED"


@pytest.mark.parametrize("change", [dict(rs20=0), dict(rs60=None), dict(rs60=0), dict(rs_slope5=0),
                                   dict(rs_near_high=False), dict(support_intact=False), dict(no_new_low2=False),
                                   dict(ema20_extension=1.01), dict(breakout_extension=1.26)])
def test_alternative_leadership_needs_all_evidence_and_no_chasing(change):
    f = setup_features("LEADERSHIP_CONTINUATION", **dict(resilience=-.01, break_previous_high=False, hh20=100.2, **change))
    assert signals.triggers("LEADERSHIP_CONTINUATION", f)[0] is False


@pytest.mark.parametrize("updates,reason", [(dict(support=93.), "STOP_DISTANCE_EXCEEDS_LIMIT"),
                                           (dict(swing_high=101.), "RR_BELOW_MINIMUM"),
                                           (dict(no_new_low2=False, undercut_reclaim=False), "TRIGGER_NOT_CONFIRMED")])
def test_entry_attempts_explain_blockers_without_hiding_other_setups(monkeypatch, updates, reason):
    f = setup_features("REVERSAL", **updates)
    scored = signals.score_setups(f, signals.trade_plan(f, config=CONFIG))
    candidate = dict(scored["REVERSAL"], valid=True, strict_valid=False, entry_tier="STARTER_ONLY", score=80.)
    monkeypatch.setattr(signals, "score_setups", lambda *args, **kwargs: {"REVERSAL": candidate})
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["signal"] == "ARMED"
    assert f"BUY_SMALL:{reason}" in result["reasons"]
    assert reason in result["setup_decisions"]["REVERSAL"]["entry_checks"]["BUY_SMALL"]["blockers"]


def test_wider_starter_is_rechecked_at_real_execution_price():
    report = deepcopy(execution_report())
    f = setup_features("REVERSAL", support=97.2, swing_high=109.)
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["signal"] == "BUY_SMALL"
    report.update(technical=f, decision=result)
    report["entry"].update(result["plan"])
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00",
                 price=100.5, bid=100.4, ask=100.5, restricted=False)
    args = dict(now=quote["timestamp"], trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    assert assess_execution(report, quote, **args)["action"] == "ELIGIBLE"
    assert assess_execution(report, dict(quote, price=101., bid=100.9, ask=101.), **args)["reason"] == "REPRICED_RISK_OR_LIQUIDITY_FAIL"
    assert assess_execution(report, dict(quote, timestamp="2026-09-14T09:08+08:00"), **args)["reason"] == "STALE_OR_MISSING_QUOTE"
    assert assess_execution(report, quote, **dict(args, portfolio={}))["shares"] == 0


@pytest.mark.parametrize("limit", [1.199, 3.00001, float("nan"), float("inf")])
def test_stop_config_cannot_be_used_for_unbounded_relaxation(limit):
    with pytest.raises(ValueError):
        signals.LowEntryV6Config(starter_maximum_stop_atr=limit)
