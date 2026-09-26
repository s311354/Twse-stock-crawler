"""A causal held-retest may fund a small leader; it cannot manufacture a BUY."""
from copy import deepcopy

import pytest

from domain.strategies import low_entry_v6_signals as signals
from domain.strategies.low_entry_score_v6 import assess_execution
from tests.strategies.test_low_entry_score_v6 import (
    ACCOUNT, CONFIG, execution_report, quality, setup_features,
)


def retest_features(**changes):
    f = setup_features(
        "LEADERSHIP_CONTINUATION", open=100., close=101., high=101.4, low=99.8,
        support=90., stop_support=90., atr14=2., ema20=98.5, ema10=99.5,
        ema20_extension=1.25, hh20=101.2, hh60=101.2, swing_high=100.,
        breakout_extension=-.1, break_previous_high=False,
        breakout_retest_confirmed=True, breakout_retest_support=99.6,
        breakout_retest_level=100., breakout_retest_age=2.,
    )
    return dict(f, **changes)


def retest_report(**changes):
    report = deepcopy(execution_report())
    f = retest_features(**changes)
    decision = signals.decide(quality(), f, config=CONFIG)
    report.update(technical=f, decision=decision)
    report["entry"].update(decision["plan"], trigger_price=f["high"])
    return report


def execution_args():
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00",
                 price=101.5, bid=101.4, ask=101.5, restricted=False)
    args = dict(now=quote["timestamp"], trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    return quote, args


@pytest.mark.parametrize("confirmed_trigger", [False, True])
def test_confirmed_retest_replaces_old_support_only_for_small_leader(confirmed_trigger):
    f = retest_features(break_previous_high=confirmed_trigger)
    old = signals.decide(quality(), dict(f, breakout_retest_confirmed=False), config=CONFIG)
    result = signals.decide(quality(), f, config=CONFIG)
    assert old["signal"] == "ARMED"
    assert old["plan"]["stop_distance_atr"] == pytest.approx(6.)
    assert result["signal"] == "BUY_SMALL"
    assert result["setup"] == "LEADERSHIP_CONTINUATION"
    assert result["score"] >= CONFIG.buy_score_minimum
    assert result["entry_tier"] == "STARTER_ONLY"
    assert result["plan"]["stop_source"] == "CONFIRMED_BREAKOUT_RETEST"
    assert result["plan"]["stop_support"] == 99.6
    assert result["plan"]["stop_price"] == pytest.approx(99.)
    assert result["plan"]["stop_distance_atr"] == pytest.approx(1.2)
    assert result["plan"]["target_price"] is None  # No invented upside / RR.
    assert result["risk_fraction"] == .0025


@pytest.mark.parametrize("setup", [None, "REVERSAL", "TREND_PULLBACK"])
def test_new_retest_anchor_cannot_leak_into_other_setups(setup):
    plan = signals.trade_plan(retest_features(), config=CONFIG, setup=setup)
    assert plan["stop_source"] == "CONFIRMED_SWING_LOW"
    assert plan["stop_price"] == 89.4


@pytest.mark.parametrize("changes", [
    dict(breakout_retest_confirmed=False), dict(breakout_retest_confirmed=None),
    dict(breakout_retest_support=None), dict(breakout_retest_support=101.),
    dict(breakout_retest_level=None), dict(breakout_retest_age=None),
    dict(breakout_retest_age=0.), dict(breakout_retest_age=1.5), dict(breakout_retest_age=6.),
    dict(low=99.5), dict(close=99.9),
])
def test_missing_expired_or_broken_retest_cannot_shrink_stop(changes):
    result = signals.decide(quality(), retest_features(**changes), config=CONFIG)
    assert result["signal"] not in signals.EXECUTABLE_SIGNALS
    assert result["plan"]["stop_source"] == "CONFIRMED_SWING_LOW"


@pytest.mark.parametrize("changes", [dict(ema20_extension=1.50001), dict(breakout_extension=1.25001),
                                     dict(rs20=0.), dict(rs_slope5=-.001), dict(support_intact=False)])
def test_held_retest_requires_non_chasing_trend_and_relative_strength(changes):
    assert not signals.triggers("LEADERSHIP_CONTINUATION", retest_features(**changes))[0]


def test_known_nearby_resistance_still_blocks_a_confirmed_retest():
    f = retest_features(swing_high=102.)
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["plan"]["target_price"] == 102.
    assert result["plan"]["reward_risk"] == pytest.approx(.25)
    assert result["signal"] == "ARMED"
    assert "BUY_SMALL:RR_BELOW_MINIMUM" in result["reasons"]


@pytest.mark.parametrize("status", ["REJECT", "WATCH_QUALITY", "PROVISIONAL", "DATA_INSUFFICIENT"])
def test_retest_cannot_override_quality_or_data(status):
    result = signals.decide(quality(status=status), retest_features(), config=CONFIG)
    assert result["signal"] not in signals.EXECUTABLE_SIGNALS
    assert result["risk_fraction"] == 0


@pytest.mark.parametrize("changes", [dict(restricted=None), dict(spread=None), dict(amount_ma20=1.),
                                     dict(history_ready=False), dict(market_ready=False),
                                     dict(new_lows3=2, clv=-.6, undercut_reclaim=False)])
def test_retest_cannot_override_hard_veto_liquidity_or_history(changes):
    result = signals.decide(quality(), retest_features(**changes), config=CONFIG)
    assert result["signal"] not in signals.EXECUTABLE_SIGNALS


def test_retest_execution_keeps_published_stop_and_rechecks_fresh_quote_gap_and_cash():
    report = retest_report()
    original = deepcopy(report)
    quote, args = execution_args()
    execution = assess_execution(report, quote, **args)
    assert execution["action"] == "ELIGIBLE" and not execution["is_fill"]
    assert execution["stop_price"] == report["entry"]["stop_price"]
    assert 0 < execution["shares"] * (execution["reference_price"] - execution["stop_price"]) <= ACCOUNT["value"] * .0025
    assert assess_execution(report, dict(quote, price=103., bid=102.9, ask=103.), **args)["reason"] == "REPRICED_RISK_OR_LIQUIDITY_FAIL"
    assert assess_execution(report, dict(quote, timestamp="2026-09-14T09:08+08:00"), **args)["reason"] == "STALE_OR_MISSING_QUOTE"
    assert assess_execution(report, quote, **dict(args, portfolio=dict(ACCOUNT, available_cash=0)))["shares"] == 0
    assert report == original


def test_intraday_gap_cannot_activate_a_different_structural_anchor():
    # The old stop qualified at signal close. A later gap must not silently move
    # it up to the retest support to make execution risk look smaller.
    report = retest_report(support=99., stop_support=99., ema10=100.5, ema20=100., ema20_extension=.5)
    assert report["decision"]["signal"] in signals.EXECUTABLE_SIGNALS
    assert report["entry"]["stop_source"] == "CONFIRMED_SWING_LOW"
    quote, args = execution_args()
    quote.update(price=102.2, bid=102.1, ask=102.2)
    plan = signals.trade_plan(report["technical"], 102.2, config=CONFIG, setup="LEADERSHIP_CONTINUATION")
    assert plan["stop_source"] == "CONFIRMED_SWING_LOW"
    assert plan["stop_price"] == report["entry"]["stop_price"]
    assert assess_execution(report, quote, **args)["reason"] == "REPRICED_RISK_OR_LIQUIDITY_FAIL"


@pytest.mark.parametrize("field,value", [("stop_price", 100.), ("stop_source", "CONFIRMED_SWING_LOW")])
def test_changed_published_stop_is_not_executable(field, value):
    report = retest_report()
    report["entry"][field] = value
    quote, args = execution_args()
    assert assess_execution(report, quote, **args)["reason"] == "PUBLISHED_STOP_MISMATCH"


def test_quality_block_does_not_hide_relative_strength_or_bad_reward_risk():
    f = setup_features("LEADERSHIP_CONTINUATION", swing_high=101., support=97.6)
    result = signals.decide(quality(status="WATCH_QUALITY", score=63.32), f, config=CONFIG)
    observation = result["technical_observation"]
    assert result["signal"] == "WATCH_ONLY" and result["risk_fraction"] == 0
    assert observation["patterns"] == ["RELATIVE_STRENGTH"]
    assert observation["is_entry_signal"] is False
    assert {"WATCH_QUALITY", "RR_BELOW_MINIMUM"} <= set(observation["blocking_reasons"])


def test_strong_but_illiquid_overextended_leader_is_visible_not_bought():
    f = setup_features("LEADERSHIP_CONTINUATION", spread=None, support=90.,
                       ema20_extension=2.8, breakout_extension=1.35)
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["signal"] == "SKIP" and result["risk_fraction"] == 0
    observation = result["technical_observation"]
    assert "RELATIVE_STRENGTH" in observation["patterns"]
    assert {"MISSING_TWO_SIDED_QUOTE", "STOP_DISTANCE_EXCEEDS_LIMIT", "WAIT_FOR_CONTROLLED_RETEST"} <= set(observation["blocking_reasons"])


def test_reversal_observation_does_not_promote_quality_reject():
    result = signals.decide(quality(status="REJECT", score=42.65), setup_features("REVERSAL", ema5=99.), config=CONFIG)
    assert result["signal"] == "NO_BUY"
    assert "LOW_ZONE_RECOVERY" in result["technical_observation"]["patterns"]
    assert "QUALITY_REJECT" in result["technical_observation"]["blocking_reasons"]


@pytest.mark.parametrize("changes", [dict(volume=0.), dict(amount_ma20=1.), dict(history_ready=False), dict(market_ready=False)])
def test_technical_observations_cannot_claim_false_strength_from_bad_history(changes):
    result = signals.decide(quality(status="WATCH_QUALITY"), retest_features(**changes), config=CONFIG)
    assert result["technical_observation"]["patterns"] == []


def test_unsupported_sector_updates_final_observation_reasons(monkeypatch):
    import pandas as pd
    from domain.strategies import low_entry_score_v6 as model
    from tests.strategies.test_low_entry_score_v6 import bars, financials

    data = bars()
    controlled = pd.DataFrame([dict(retest_features(), pe=20.)] * len(data), index=data.index)
    monkeypatch.setattr(model, "build_features", lambda *_: controlled)
    prepared = model.attach_low_entry_v6_context(data, symbol="2330", sector="UNSUPPORTED",
                                                as_of="2026-09-11T15:00+08:00", filings=financials(), portfolio=ACCOUNT)
    report = model.LowEntryScoreV6Strategy().run_latest(prepared).iloc[-1]["LES v6 Report"]
    assert report["decision"]["signal"] == "NO_MODEL"
    assert report["sizing"]["indicative_shares"] == 0
    observation = report["decision"]["technical_observation"]
    assert observation["patterns"] == ["RELATIVE_STRENGTH"]
    assert "SECTOR_UNSUPPORTED" in observation["blocking_reasons"]
