"""Offline raw-bar regression for the starter-only confirmed-breakout retest."""
import json

import numpy as np
import pandas as pd
import pytest

from domain.features import SharedFeaturePipeline
from domain.strategies.low_entry_score_v6 import (
    LowEntryScoreV6Strategy, assess_execution, attach_low_entry_v6_context,
)
from domain.strategies.low_entry_v6_features import build_features
from domain.strategies.low_entry_v6_report import report_failure
from domain.strategies.low_entry_v6_signals import LowEntryV6Config, trade_plan
from tests.strategies.test_low_entry_score_v6 import ACCOUNT, CONFIG, financials


def retest_bars():
    """Synthetic uptrend with an old distant floor, breakout, then held retest.

    The last bar tests 99.8 below the frozen breakout level of 100 and closes
    above it. Its high exceeds known resistance, so no target is fabricated.
    All features and scores are derived from these bars, never injected.
    """
    count = 120
    close = np.r_[np.linspace(85., 99.2, count - 2), 100.8, 100.8]
    data = pd.DataFrame({
        "Date": pd.bdate_range(end="2026-09-11", periods=count),
        "Open": close - .3, "High": close + .8, "Low": close - .8, "Close": close,
        "Volume": 1_000_000., "Amount": close * 1_000_000., "Trades": 500,
        "Bid": close - .05, "Ask": close + .05, "PE": 20.,
        "TAIEX Close": np.linspace(20000., 20200., count), "Restricted": False, "Complete": True,
    })
    data.loc[105, "Low"] = 92.  # Confirmed well before the breakout, not a new tight stop.
    data.loc[118, ["Open", "High", "Low", "Close", "Volume"]] = [99.6, 100.9, 99.3, 100.8, 1_500_000.]
    data.loc[119, ["Open", "High", "Low", "Close", "Volume"]] = [100.2, 101., 99.8, 100.8, 1_200_000.]
    data["Amount"] = data["Close"] * data["Volume"]
    return data


def raw_report(data):
    prepared = attach_low_entry_v6_context(
        data, symbol="2330", name="合成回測案例", sector="ELEC",
        as_of="2026-09-11T15:00:00+08:00", filings=financials(), portfolio=ACCOUNT,
    )
    # Freeze the earlier leadership slots to isolate causal retest/execution parity.
    result = LowEntryScoreV6Strategy(config=LowEntryV6Config(leadership_policy_version=1)).run_latest(prepared)
    assert len(result) == 1
    # Exercise the same JSON-safe frozen evidence consumed by execution adapters.
    return json.loads(json.dumps(result.iloc[0]["LES v6 Report"], allow_nan=False))


@pytest.fixture
def published_retest(active_industry_profiles):
    return raw_report(retest_bars())


def quote(**changes):
    return dict({
        "symbol": "2330", "timestamp": "2026-09-14T09:10:00+08:00",
        "price": 101.05, "bid": 101., "ask": 101.05, "restricted": False,
    }, **changes)


def execute(report, fresh_quote, *, now=None):
    return assess_execution(
        report, fresh_quote, now=now or fresh_quote["timestamp"],
        trading_sessions=["2026-09-11", "2026-09-14", "2026-09-15"], portfolio=ACCOUNT,
    )


def test_raw_bars_establish_starter_retest_and_preserve_stop_through_execution(published_retest):
    data = retest_bars()
    features = build_features(data, SharedFeaturePipeline().build(data))
    breakout, retest = features.iloc[-2], features.iloc[-1]
    assert len(data) >= 100 and data["Complete"].all()
    assert breakout["close"] > breakout["hh20"] == pytest.approx(100.)
    assert breakout["volume_ratio"] >= 1.1
    assert not breakout["breakout_retest_confirmed"]
    assert retest["breakout_retest_confirmed"]
    assert retest["breakout_retest_age"] == 1
    assert retest["breakout_retest_support"] == pytest.approx(99.8)
    assert retest["swing_low"] == pytest.approx(92.)

    report = published_retest
    decision, entry = report["decision"], report["entry"]
    assert report["quality"]["status"] == "PASS" and report["quality"]["score"] >= 75
    assert report["downtrend_gate"]["hard_veto"] is False
    assert decision["setup"] == "LEADERSHIP_CONTINUATION"
    assert decision["signal"] == "BUY_SMALL" and decision["entry_tier"] == "STARTER_ONLY"
    assert "STRUCTURAL_ANCHOR_STARTER_ONLY" in decision["entry_checks"]["BUY"]["blockers"]
    assert decision["risk_gate"] == "PASS" and decision["risk_fraction"] == CONFIG.small_risk_fraction
    assert decision["score"] >= decision["required_score"]
    old_plan = trade_plan(retest.to_dict(), config=CONFIG)
    assert old_plan["stop_distance_atr"] > CONFIG.starter_maximum_stop_atr
    assert entry["stop_source"] == "CONFIRMED_BREAKOUT_RETEST"
    assert entry["stop_support"] == retest["breakout_retest_support"]
    assert entry["stop_price"] == pytest.approx(99.8 - CONFIG.stop_buffer_atr * retest["atr14"])
    assert 0 < entry["stop_distance_atr"] <= CONFIG.starter_maximum_stop_atr
    assert entry["target_price"] is None and entry["reward_risk"] is None
    assert decision["rr_policy"] == "PRICE_DISCOVERY_EXTENSION_SUPPORT"
    assert entry["net_reward_risk"] is None
    assert entry["trigger_price"] == data.iloc[-1]["High"]
    assert entry["is_fill"] is False and report_failure(report) is None

    frozen = json.dumps(report, sort_keys=True)
    result = execute(report, quote())
    assert result["action"] == "ELIGIBLE" and result["signal"] == "BUY_SMALL"
    assert result["shares"] > 0 and result["is_fill"] is False
    assert result["stop_price"] == entry["stop_price"]
    assert json.dumps(report, sort_keys=True) == frozen


@pytest.mark.parametrize("damage", ["breakout_volume", "retest_floor"])
def test_broken_raw_retest_cannot_borrow_the_new_stop_or_execute(damage):
    data = retest_bars()
    if damage == "breakout_volume":
        data.loc[118, "Volume"] = 1_000_000.
        data["Amount"] = data["Close"] * data["Volume"]
    else:
        data.loc[119, "Low"] = 99.  # Below the breakout level minus its frozen 0.3 ATR.
    report = raw_report(data)

    assert report["technical"]["breakout_retest_confirmed"] is False
    assert report["entry"]["stop_source"] == "CONFIRMED_SWING_LOW"
    assert report["entry"]["stop_distance_atr"] > CONFIG.starter_maximum_stop_atr
    assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}
    assert report["sizing"]["indicative_shares"] == 0
    assert execute(report, quote())["action"] == "SKIP"


@pytest.mark.parametrize("changes,now,reason", [
    ({"price": 101., "ask": 101.}, None, "TRIGGER_NOT_REACHED"),
    ({}, "2026-09-14T09:11:01+08:00", "STALE_OR_MISSING_QUOTE"),
    ({"price": 105., "bid": 104.95, "ask": 105.}, None, "REPRICED_RISK_OR_LIQUIDITY_FAIL"),
    ({"timestamp": "2026-09-11T15:01:00+08:00"}, None, "OUTSIDE_NEXT_SESSION"),
    ({"timestamp": "2026-09-15T09:10:00+08:00"}, None, "OUTSIDE_NEXT_SESSION"),
])
def test_published_retest_still_requires_fresh_next_session_trigger_and_risk(published_retest, changes, now, reason):
    result = execute(published_retest, quote(**changes), now=now)
    assert result == {"action": "SKIP", "shares": 0, "reason": reason}
