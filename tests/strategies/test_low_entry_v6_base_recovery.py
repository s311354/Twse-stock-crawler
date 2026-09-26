"""Completed compact bases broaden eligibility without upgrading missing evidence."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from domain.strategies.low_entry_v6_features import _consolidation_features
from domain.strategies.low_entry_v6_signals import decide, trade_plan, size_position
from domain.strategies.low_entry_score_v6 import assess_execution, attach_low_entry_v6_context, LowEntryScoreV6Strategy
from domain.strategies.low_entry_v6_report import report_failure
from tests.strategies.test_low_entry_score_v6 import ACCOUNT, CONFIG, execution_report, financials, quality, setup_features


def base_bars():
    # First floor is held for two completed sessions despite a lower daily low
    # on the last session. Prior swing support is already known and improving.
    return pd.DataFrame({
        "open": [99., 99.4, 99.3], "high": [100., 100.1, 100.4],
        "low": [98., 98.5, 98.2], "close": [99.2, 99.5, 100.],
        "volume": 1_000_000., "atr14": 2., "support": 95.,
        "history_ready": True, "market_ready": True,
        "support_intact": True, "higher_low": True,
    })


def base_features(**changes):
    return setup_features("REVERSAL", **dict({
        "support": 95., "low": 98.2, "consolidation_confirmed": True,
        "consolidation_support": 98., "no_new_low2": False,
        "rs_improving": False, "rs_slope5": -.001, "rs20": -.02,
        "resilience_improving": False,
    }, **changes))


def test_daily_low_oscillation_does_not_invalidate_a_held_compact_floor():
    data = base_bars()
    assert data.low.iloc[-1] < data.low.iloc[-2]
    result = _consolidation_features(data)
    assert result.consolidation_confirmed.tolist() == [False, False, True]
    assert result.consolidation_support.iloc[-1] == 98.
    assert result.consolidation_support.iloc[:2].isna().all()


@pytest.mark.parametrize("column,value", [
    ("open", np.nan), ("high", np.nan), ("low", 97.99),
    ("close", 98.4), ("volume", 0.), ("atr14", 0.),
    ("support_intact", False), ("market_ready", False), ("history_ready", False),
])
def test_broken_or_incomplete_base_never_produces_support(column, value):
    data = base_bars()
    data.loc[1, column] = value
    result = _consolidation_features(data)
    assert not result.consolidation_confirmed.any()
    assert result.consolidation_support.isna().all()


@pytest.mark.parametrize("column,value", [("higher_low", False), ("support", 98.), ("atr14", 1.)])
def test_base_requires_preexisting_higher_support_and_compact_range(column, value):
    data = base_bars()
    data.loc[0, column] = value
    assert not _consolidation_features(data).consolidation_confirmed.any()


def test_base_atr_is_frozen_and_future_bars_cannot_confirm_early():
    data = base_bars()
    data.loc[0, "atr14"] = 1.
    data.loc[1:, "atr14"] = 20.
    assert not _consolidation_features(data).consolidation_confirmed.any()
    data = pd.concat([base_bars(), base_bars()], ignore_index=True)
    data.index = pd.bdate_range("2026-01-01", periods=len(data))
    complete = _consolidation_features(data)
    for end in range(1, len(data) + 1):
        pd.testing.assert_frame_equal(complete.iloc[:end], _consolidation_features(data.iloc[:end]))
    assert not _consolidation_features(data.drop(columns="support")).consolidation_confirmed.any()


def test_absolute_recovery_can_be_small_without_claiming_relative_leadership():
    f = base_features()
    result = decide(quality(), f, config=CONFIG)
    assert result["signal"] == "BUY_SMALL" and result["setup"] == "REVERSAL"
    assert result["entry_tier"] == "STARTER_ONLY" and result["risk_fraction"] == .0025
    assert result["setups"]["REVERSAL"]["components"]["relative_strength"] == 0
    assert result["score"] >= 65
    assert result["plan"]["stop_source"] == "CONFIRMED_CONSOLIDATION_LOW"
    assert result["plan"]["stop_price"] == pytest.approx(97.4)
    assert result["plan"]["target_price"] == 108.
    assert "STARTER_ONLY_SETUP" in result["entry_checks"]["BUY"]["blockers"]
    shares = size_position(result, portfolio_value=ACCOUNT["value"], available_cash=ACCOUNT["available_cash"],
                           intended_weight=ACCOUNT["intended_weight"], portfolio_capacity_weight=ACCOUNT["capacity_weight"],
                           amount_ma20=f["amount_ma20"], config=CONFIG)
    assert 0 < shares * (100.4 - 97.4) <= ACCOUNT["value"] * .0025


@pytest.mark.parametrize("changes", [
    {"consolidation_confirmed": False}, {"consolidation_support": None},
    {"consolidation_support": 98.3}, {"higher_low": False}, {"support_intact": False},
    {"ema5_slope": 0.}, {"macd_improving": False}, {"r5": 0.}, {"clv": .4},
    {"vwap20": 100.}, {"market_regime": "RISK_OFF"},
    {"cmf_improving": False, "obv_slope5": -1., "up_volume_ratio": .9},
])
def test_absolute_recovery_is_not_a_blanket_relative_strength_waiver(changes):
    result = decide(quality(), base_features(**changes), config=CONFIG)
    assert result["signal"] not in {"BUY", "BUY_SMALL"}


@pytest.mark.parametrize("changes,expected", [
    ({"new_lows3": 2., "clv": -.7}, "NO_BUY"),
    ({"amount_ma20": 1_000_000.}, "SKIP"), ({"spread": None}, "SKIP"),
    ({"restricted": True}, "SKIP"), ({"history_ready": False}, "DATA_INSUFFICIENT"),
    ({"rs_slope5": None}, "DATA_INSUFFICIENT"),
])
def test_base_cannot_bypass_veto_data_or_liquidity(changes, expected):
    assert decide(quality(), base_features(**changes), config=CONFIG)["signal"] == expected


@pytest.mark.parametrize("status", ["REJECT", "DATA_INSUFFICIENT", "PROVISIONAL", "WATCH_QUALITY"])
def test_base_cannot_replace_financial_quality(status):
    assert decide(quality(status=status), base_features(), config=CONFIG)["signal"] not in {"BUY", "BUY_SMALL"}


def test_base_never_raises_targets_or_bypasses_small_quality_floor():
    f = base_features(swing_high=101., hh20=110., hh60=120.)
    result = decide(quality(), f, config=CONFIG)
    assert result["signal"] == "ARMED" and result["plan"]["target_price"] == 101.
    assert "RR_BELOW_MINIMUM" in result["entry_checks"]["BUY_SMALL"]["blockers"]
    assert decide(quality(score=74.9), base_features(), config=CONFIG)["signal"] == "BUY_SMALL"
    assert decide(quality(score=69.9), base_features(), config=CONFIG)["signal"] == "WATCH_ONLY"


@pytest.mark.parametrize("setup", ["REVERSAL", "TREND_PULLBACK"])
def test_short_base_stop_is_starter_only_even_with_confirmed_high_score(setup):
    f = setup_features(setup, support=95., low=98.2, consolidation_support=98.,
                       consolidation_confirmed=True, rs20=.1, rs_improving=True)
    result = decide(quality(), f, config=CONFIG)
    assert result["signal"] == "BUY_SMALL" and result["score"] >= 75
    assert result["entry_tier"] == "STARTER_ONLY"
    assert result["plan"]["stop_source"] == "CONFIRMED_CONSOLIDATION_LOW"


def test_base_anchor_does_not_change_a_tight_existing_stop_or_leadership_model():
    f = base_features(support=97.5)
    assert trade_plan(f, config=CONFIG, setup="REVERSAL")["stop_source"] == "CONFIRMED_SWING_LOW"
    assert trade_plan(base_features(), config=CONFIG, setup="LEADERSHIP_CONTINUATION")["stop_source"] == "CONFIRMED_SWING_LOW"


def test_live_execution_keeps_base_stop_and_target_frozen_and_rechecks_rr_cash_quote():
    f = base_features()
    result = decide(quality(), f, config=CONFIG)
    report = deepcopy(execution_report())
    report.update(technical=f, decision=result)
    report["entry"].update(result["plan"], trigger_price=f["high"])
    original = deepcopy(report)
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00",
                 price=100.45, bid=100.4, ask=100.45, restricted=False)
    args = dict(now=quote["timestamp"], trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    eligible = assess_execution(report, quote, **args)
    assert eligible["action"] == "ELIGIBLE" and eligible["is_fill"] is False
    assert eligible["stop_price"] == original["entry"]["stop_price"]
    assert assess_execution(report, dict(quote, price=100.4), **args)["reason"] == "TRIGGER_NOT_REACHED"
    assert assess_execution(report, dict(quote, price=104., bid=103.9, ask=104.), **args)["reason"] == "REPRICED_RISK_OR_LIQUIDITY_FAIL"
    assert assess_execution(report, dict(quote, timestamp="2026-09-14T09:00+08:00"), **args)["reason"] == "STALE_OR_MISSING_QUOTE"
    assert assess_execution(report, quote, **dict(args, portfolio=dict(ACCOUNT, available_cash=0)))["reason"] == "ZERO_CAPACITY_OR_MISSING_PORTFOLIO_DATA"
    assert report == original


def recovery_bars():
    """Synthetic correction, higher low, recovery and held base; no injected score."""
    close = np.r_[np.linspace(110., 120., 60), np.linspace(120., 90., 20),
                  90. + np.sin(np.arange(20)), [92., 94., 93., 95., 96., 95., 94., 95., 94.5, 97., 97.6, 98.5]]
    count = len(close)
    data = pd.DataFrame({
        "Date": pd.bdate_range(end="2026-09-11", periods=count),
        "Open": close - .3, "High": close + .5, "Low": close - 1., "Close": close,
        "Volume": 1_000_000., "Amount": close * 1_000_000., "Trades": 500,
        "Bid": close - .05, "Ask": close + .05, "PE": 20.,
        "TAIEX Close": np.linspace(20000., 35000., count), "Restricted": False, "Complete": True,
    })
    data.loc[count - 3:, "Low"] = [96.3, 96.8, 96.5]
    data.loc[count - 1, "High"] = 98.8
    # Recovery is real but does not yet outperform a faster-rising benchmark.
    data.loc[count - 21:, "TAIEX Close"] = np.linspace(30000., 39000., 21)
    return data


def raw_recovery_report(data):
    prepared = attach_low_entry_v6_context(data, symbol="2330", sector="ELEC",
                                           as_of="2026-09-11T15:00+08:00", filings=financials(), portfolio=ACCOUNT)
    report = LowEntryScoreV6Strategy().run_latest(prepared).iloc[0]["LES v6 Report"]
    return json.loads(json.dumps(report, allow_nan=False))


@pytest.mark.usefixtures("active_industry_profiles")
def test_raw_bars_to_quality_to_serialized_base_signal_to_next_session_execution():
    report = raw_recovery_report(recovery_bars())
    f, d, entry = (report[key] for key in ("technical", "decision", "entry"))
    assert f["consolidation_confirmed"] is True and f["no_new_low2"] is True
    assert f["rs_improving"] is False and f["rs_slope5"] < 0
    assert d["signal"] == "BUY_SMALL" and d["setup"] == "REVERSAL" and d["score"] >= 75
    assert d["entry_tier"] == "STARTER_ONLY" and report_failure(report) is None
    assert entry["stop_source"] == "CONFIRMED_CONSOLIDATION_LOW"
    assert entry["stop_support"] == 96.3 and entry["trigger_price"] == 98.8
    assert entry["target_price"] == 120.5 and entry["is_fill"] is False
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00",
                 price=98.85, bid=98.8, ask=98.85, restricted=False)
    result = assess_execution(report, quote, now=quote["timestamp"],
                              trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    assert result["action"] == "ELIGIBLE" and result["stop_price"] == entry["stop_price"]
    assert result["shares"] > 0 and result["is_fill"] is False


@pytest.mark.parametrize("column,value", [("Low", 95.), ("Volume", 0.), ("TAIEX Close", np.nan)])
def test_raw_damaged_base_cannot_borrow_tighter_stop(column, value):
    data = recovery_bars()
    data.loc[len(data) - 2, column] = value
    report = raw_recovery_report(data)
    assert report["technical"]["consolidation_confirmed"] is False
    assert report["entry"]["stop_source"] != "CONFIRMED_CONSOLIDATION_LOW"
    assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}
