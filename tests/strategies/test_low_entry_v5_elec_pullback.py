"""Guarded local-pullback entries, not ticker-specific hindsight overrides."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from domain.features import SharedFeaturePipeline
from domain.quality_v5 import FinalQualityStatus
from domain.strategies.low_entry_score_v4 import attach_low_entry_v4_context
from domain.strategies.low_entry_score_v5 import LowEntryScoreV5Config
from domain.strategies.low_entry_score_v5 import LowEntryScoreV5Strategy
from tests.strategies.test_low_entry_score_v5 import SIGNAL_DATE
from tests.strategies.test_low_entry_score_v5 import _annual_fundamentals
from tests.strategies.test_low_entry_score_v5 import _quality_snapshot


def _history(sector="24"):
    close = np.r_[
        np.linspace(60, 100, 120),
        [102, 103, 104, 106, 107, 105, 103, 101, 99, 100, 102],
    ]
    size = len(close)
    data = pd.DataFrame({
        "Date": pd.bdate_range(end="2026-07-21", periods=size),
        "Open": close - 0.6, "High": close + 0.5,
        "Low": close - 2, "Close": close, "Volume": 1_000_000.0,
        "Amount": 100_000_000.0, "Trades": 5_000.0,
        "Bid": close - 0.05, "Ask": close + 0.05, "PE": 10.0,
        "TAIEX Close": np.linspace(20_000, 21_000, size),
    })
    data.loc[size - 1, ["High", "Low"]] = [102.1, 100.7]
    return attach_low_entry_v4_context(
        data, annual_fundamentals=_annual_fundamentals(),
        fundamental_snapshot=_quality_snapshot(sector=sector),
        signal_date=SIGNAL_DATE,
    )


def _inputs(sector="24"):
    strategy = LowEntryScoreV5Strategy()
    data = _history(sector)
    components = strategy.pipeline.components
    features = SharedFeaturePipeline(components.market.shared_feature_config()).build(data)
    indicators = components.market.calculate_indicators(data, features)
    components.market.evaluate_market_data(data, indicators, dict(data.attrs))
    quality = components.fundamentals.evaluate_quality(data, dict(data.attrs))
    valuation = pd.Series("SUPPORTIVE", index=data.index)
    return strategy, indicators, quality, valuation


@pytest.mark.usefixtures("active_industry_profiles")
def test_elec_local_reclaim_opens_only_conditional_starter_with_repriced_risk():
    data = _history()
    baseline = LowEntryScoreV5Strategy(
        LowEntryScoreV5Config(elec_trend_pullback_enabled=False)
    ).run(data).iloc[-1]
    revised = LowEntryScoreV5Strategy().run(data).iloc[-1]

    assert baseline["低點決策"] not in {"BUY", "BUY_SMALL", "EARLY_BUY"}
    assert "ELECTrendPullback" not in baseline["低點理由"]
    assert baseline["Risk Gate"] == "FAIL"
    assert revised["低點分數"] == baseline["低點分數"]  # no score inflation
    assert revised["60日位置"] > 0.5
    assert revised["Quality Pullback Starter Trigger"] == 0
    assert revised["Broad Rally Tier"] == "NONE"
    assert revised["低點決策"] == "BUY_SMALL"
    assert revised["Position Fraction"] == pytest.approx(1 / 8)
    assert revised["Entry Rule"] == "NEXT_SESSION_BREAKS_SIGNAL_HIGH"
    assert revised["Entry Candidate"] == 102.5  # next valid TWSE tick
    assert revised["Risk Gate"] == "PASS"
    assert revised["RewardRisk"] >= 1.7
    assert 0 < revised["Entry Candidate"] - revised["策略停損"] <= 1.8 * revised["ATR14"]
    assert "ELECTrendPullback=YES；ELECRepricedRisk=PASS" in revised["低點理由"]
    assert "ELECStarterRequired=55；ELECStarterRawRequired=60" in revised["低點理由"]
    assert revised["建議股數"] > 0


@pytest.mark.parametrize("sector", ["15", "21", "", None])
def test_non_electronics_and_unknown_sector_never_opt_into_local_reclaim(sector):
    strategy, indicators, quality, valuation = _inputs(sector)
    panic = strategy.pipeline.components.panic.evaluate_panic(indicators, quality, valuation)
    assert not panic["elec_trend_pullback_trigger"].any()
    if not sector:
        assert indicators["is_electronics"].isna().all()


@pytest.mark.parametrize("column,value", [
    ("ema50", 200.0), ("drawdown20", 0.0), ("drawdown20", -0.20),
    ("ema20", 80.0), ("ema20", 110.0), ("atr14", 0.0),
    ("ema5", 103.0), ("clv", -0.5), ("volume_ratio", 0.20),
    ("new_low_count3", 2), ("previous_low20", 103.0),
])
def test_reclaim_rejects_chasing_continued_decline_and_missing_confirmation(column, value):
    strategy, indicators, quality, valuation = _inputs()
    indicators.loc[indicators.index[-1], column] = value
    panic = strategy.pipeline.components.panic.evaluate_panic(indicators, quality, valuation)
    assert not panic["elec_trend_pullback_trigger"].iloc[-1]


@pytest.mark.parametrize("override", [
    {"final_score": 79.99}, {"final_status": FinalQualityStatus.PROVISIONAL},
    {"base_data_confidence": 74.99}, {"extended_data_confidence": 69.99},
    {"critical_field_coverage": 69.99}, {"hard_rejects": ("MATERIAL_ISSUE",)},
])
def test_local_reclaim_preserves_quality_and_evidence_floors(override):
    strategy, indicators, quality, valuation = _inputs()
    panic = strategy.pipeline.components.panic.evaluate_panic(
        indicators, replace(quality, **override), valuation
    )
    assert not panic["elec_trend_pullback_trigger"].any()


@pytest.mark.parametrize("override", [
    {"score": 54.99}, {"raw_score": 59.99},
    {"valuation": "FAIL"}, {"valuation": "UNKNOWN"},
    {"falling_state": "PANIC_WATCH"}, {"falling_state": "REVERSAL_WATCH"},
    {"market_gate": "FAIL"}, {"broad_rally_risk_gate": "UNKNOWN"},
    {"elec_repriced_risk_gate": "FAIL"}, {"elec_repriced_risk_gate": "UNKNOWN"},
    {"elec_trend_pullback_trigger": False},
])
def test_local_reclaim_does_not_override_execution_or_valuation_gates(override):
    strategy, _, quality, _ = _inputs()
    values = dict(
        score=60.0, raw_score=65.0, valuation="SUPPORTIVE", falling_state="CLEAR",
        next_day_confirmation=False, trigger=False, risk_gate="FAIL", market_gate="PASS",
        required_score=75.0, broad_rally_trigger=False, broad_rally_risk_gate="PASS",
        elec_trend_pullback_trigger=True, elec_repriced_risk_gate="PASS",
    )
    values.update(override)
    decision = strategy.pipeline.components.decision.resolve_decision(
        quality=quality, **{key: pd.Series([value]) for key, value in values.items()}
    )
    assert decision.iloc[-1] not in {"BUY", "BUY_SMALL", "EARLY_BUY"}


def test_missing_repriced_risk_cannot_authorize_local_starter():
    strategy, _, quality, _ = _inputs()
    decision = strategy.pipeline.components.decision.resolve_decision(
        quality=quality, score=pd.Series([65.0]), raw_score=pd.Series([65.0]),
        valuation=pd.Series(["PASS"]), falling_state=pd.Series(["CLEAR"]),
        next_day_confirmation=pd.Series([False]), trigger=pd.Series([False]),
        risk_gate=pd.Series(["PASS"]), market_gate=pd.Series(["PASS"]),
        required_score=pd.Series([75.0]), broad_rally_trigger=pd.Series([False]),
        broad_rally_risk_gate=pd.Series(["PASS"]),
        elec_trend_pullback_trigger=pd.Series([True]),
    )
    assert decision.iloc[-1] == "ARMED"


@pytest.mark.parametrize("enabled,normal_trigger,expected", [
    (False, False, "ARMED"), (True, True, "BUY"),
])
def test_local_starter_respects_disable_switch_and_preserves_ordinary_buy(
    enabled, normal_trigger, expected
):
    _, _, quality, _ = _inputs()
    strategy = LowEntryScoreV5Strategy(
        LowEntryScoreV5Config(elec_trend_pullback_enabled=enabled)
    )
    decision = strategy.pipeline.components.decision.resolve_decision(
        quality=quality, score=pd.Series([80.0]), raw_score=pd.Series([80.0]),
        valuation=pd.Series(["PASS"]), falling_state=pd.Series(["CLEAR"]),
        next_day_confirmation=pd.Series([False]), trigger=pd.Series([normal_trigger]),
        risk_gate=pd.Series(["PASS"]), market_gate=pd.Series(["PASS"]),
        required_score=pd.Series([75.0]), broad_rally_trigger=pd.Series([False]),
        broad_rally_risk_gate=pd.Series(["PASS"]),
        elec_trend_pullback_trigger=pd.Series([True]),
        elec_repriced_risk_gate=pd.Series(["PASS"]),
    )
    assert decision.iloc[-1] == expected


def test_local_reclaim_is_causal_and_does_not_require_future_confirmation():
    strategy, indicators, quality, valuation = _inputs()
    evaluator = strategy.pipeline.components.panic
    complete = evaluator.evaluate_panic(indicators, quality, valuation)
    for size in (len(indicators) - 2, len(indicators) - 1, len(indicators)):
        prefix = evaluator.evaluate_panic(indicators.iloc[:size], quality, valuation.iloc[:size])
        pd.testing.assert_series_equal(
            prefix["elec_trend_pullback_trigger"],
            complete["elec_trend_pullback_trigger"].iloc[:size],
        )


@pytest.mark.parametrize("column", ["Bid", "Ask", "Amount"])
def test_real_pipeline_skips_missing_execution_data(column):
    data = _history()
    data.loc[data.index[-1], column] = np.nan
    latest = LowEntryScoreV5Strategy().run(data).iloc[-1]
    assert latest["低點決策"] not in {"BUY", "BUY_SMALL", "EARLY_BUY"}
