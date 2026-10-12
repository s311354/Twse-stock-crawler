"""Bounded recovery alternatives recognize setups without granting full-size BUY."""

import pandas as pd
import pytest

import domain.strategies.low_entry_score_v6 as strategy_module
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.strategies.low_entry_v6_signals import (
    EXECUTABLE_SIGNALS, decide, score_setups, trade_plan,
)
from tests.strategies.test_low_entry_score_v6 import ACCOUNT, CONFIG, bars, quality, setup_features


def evidence(path):
    if path == "absolute":
        return "REVERSAL", setup_features(
            "REVERSAL", rs_slope5=-.001, rs_improving=False,
            resilience_improving=False, consolidation_confirmed=False)
    if path == "price_recovery":
        return "REVERSAL", setup_features(
            "REVERSAL", atr_stable=False, atr_expansion=1.1,
            macd_improving=False, rsi_cross40=False)
    if path == "longer_trend":
        return "TREND_PULLBACK", setup_features(
            "TREND_PULLBACK", ema10=99.1, ema50=96., reclaim5=True,
            position60=.6, drawdown60=-.04)
    return "LEADERSHIP_CONTINUATION", setup_features(
        "LEADERSHIP_CONTINUATION", rs_slope5=-.001)


def scored(path, **changes):
    setup, f = evidence(path)
    f.update(changes)
    return score_setups(f, trade_plan(f, config=CONFIG), config=CONFIG)[setup]


@pytest.mark.parametrize("path", ["absolute", "price_recovery", "longer_trend", "held_leader"])
def test_new_recovery_paths_are_small_only_even_with_high_scores(path):
    setup, f = evidence(path)
    s = scored(path)
    assert s["valid"] and not s["strict_valid"]
    assert s["entry_tier"] == "STARTER_ONLY"
    assert s["score"] >= CONFIG.buy_score_minimum
    result = decide(quality(), f, config=CONFIG)
    assert result["setup"] == setup
    assert result["signal"] == "BUY_SMALL"
    assert result["plan"]["stop_source"] == "CONFIRMED_SWING_LOW"


@pytest.mark.parametrize("changes", [
    {"no_new_low2": False}, {"higher_low": False, "atr_expansion": 1.1}, {"support_intact": False},
    {"market_regime": "RISK_OFF"}, {"r5": 0.}, {"macd_improving": False},
    {"clv": .4}, {"cmf_improving": False, "obv_slope5": -1., "up_volume_ratio": .8},
])
def test_absolute_recovery_needs_independent_floor_and_strong_demand(changes):
    assert not scored("absolute", **changes)["valid"]


@pytest.mark.parametrize("changes", [
    {"ema50": None}, {"ema50": 100.}, {"ema20_slope": -.01},
    {"higher_low": None}, {"no_new_low2": False}, {"support_intact": False},
    {"reclaim5": False, "break_previous_high": False}, {"atr_expansion": 1.251},
])
def test_longer_trend_alternative_requires_reclaim_and_controlled_volatility(changes):
    assert not scored("longer_trend", **changes)["valid"]


@pytest.mark.parametrize("changes", [
    {"rs60": None}, {"rs60": 0.}, {"rs20": 0.}, {"rs_near_high": False},
    {"rs_improving": False}, {"no_new_low2": False}, {"higher_low": False},
    {"support_intact": False}, {"atr_expansion": 1.251},
])
def test_pausing_leader_requires_medium_term_strength_and_a_held_floor(changes):
    assert not scored("held_leader", **changes)["valid"]


def test_price_or_rsi_recovery_does_not_award_missing_macd_points():
    s = scored("price_recovery")
    assert s["components"]["momentum"] == 4.  # Only the observed MACD line cross.
    assert not scored("price_recovery", break_previous_high=False)["valid"]
    assert not scored("price_recovery", clv=.4)["valid"]
    assert scored("price_recovery", break_previous_high=False, rsi_cross40=True)["valid"]


@pytest.mark.parametrize("path", ["absolute", "price_recovery", "longer_trend", "held_leader"])
def test_recovery_cannot_override_quality_veto_data_liquidity_or_stop_limits(path):
    _, f = evidence(path)
    for status in ("REJECT", "DATA_INSUFFICIENT", "PROVISIONAL", "WATCH_QUALITY"):
        assert decide(quality(status=status), f, config=CONFIG)["signal"] not in EXECUTABLE_SIGNALS
    for changes in (
        {"new_lows3": 2., "clv": -.7}, {"history_ready": False}, {"market_ready": False},
        {"support": None}, {"support": 80.}, {"amount_ma20": 1.},
        {"restricted": None}, {"restricted": True}, {"spread": None}, {"spread": .02},
        {"swing_high": 100.5, "hh20": 100.6, "hh60": 100.7},
    ):
        assert decide(quality(), dict(f, **changes), config=CONFIG)["signal"] not in EXECUTABLE_SIGNALS


def test_confirmed_setups_keep_full_buy_without_score_inflation():
    for setup in ("REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION"):
        f = setup_features(setup)
        assert decide(quality(), f, config=CONFIG)["signal"] == "BUY"


@pytest.mark.parametrize("path", ["absolute", "price_recovery", "longer_trend", "held_leader"])
def test_recovery_report_and_next_session_execution_share_the_same_evidence(path, monkeypatch):
    setup, f = evidence(path)
    f["pe"] = None
    # Supply controlled indicators/quality, but exercise real production industry
    # admission, serialization, sizing, five-gate validation and execution checks.
    monkeypatch.setattr(strategy_module, "build_features", lambda data, shared: pd.DataFrame([f]))
    complete_quality = quality(coverage=dict.fromkeys(
        ("balance_sheet", "profitability", "growth", "return_on_capital", "cash_flow"), 100.))
    monkeypatch.setattr(strategy_module, "evaluate_quality", lambda *args, **kwargs: complete_quality)
    data = strategy_module.attach_low_entry_v6_context(
        bars(1), symbol="2330", sector="24", as_of="2026-09-11T15:00+08:00",
        filings=[], portfolio=ACCOUNT)
    # Preserve the legacy recovery-tier contract; unified policy is tested separately.
    row = strategy_module.LowEntryScoreV6Strategy(config=CONFIG).run_latest(data).iloc[-1]
    report = row["LES v6 Report"]
    assert report["industry_profile"]["mode"] == "ACTIVE"
    assert report["decision"]["signal"] == "BUY_SMALL"
    assert report["decision"]["setup"] == setup
    assert report["sizing"]["indicative_shares"] > 0
    assert report_failure(report) is None
    assert set(entry_gate_checks(report).values()) == {"PASS"}
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00",
                 price=100.5, bid=100.4, ask=100.5, restricted=False)
    args = dict(now="2026-09-14T09:10:30+08:00", trading_sessions=["2026-09-11", "2026-09-14"],
                portfolio=ACCOUNT)
    execute = strategy_module.assess_execution
    result = execute(report, quote, **args)
    assert result["action"] == "ELIGIBLE" and result["shares"] > 0
    assert not result["is_fill"]
    assert execute(report, dict(quote, timestamp="2026-09-14T09:08+08:00"), **args)["action"] == "SKIP"
    assert execute(report, dict(quote, price=110., ask=110.), **args)["action"] == "SKIP"
    assert execute(report, quote, **dict(args, portfolio=dict(ACCOUNT, available_cash=0)))["action"] == "SKIP"
