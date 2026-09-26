import numpy as np
import pandas as pd
import pytest

from domain.features import SharedFeaturePipeline
from domain.strategies.low_entry_v6_features import _breakout_retest_features, build_features
from domain.strategies.low_entry_v6_signals import LowEntryV6Config, trade_plan


def evidence(count=10):
    """Controlled feature evidence with a fresh breakout at row 2."""
    f = pd.DataFrame({
        "open": 99.7, "high": 100., "low": 99.6, "close": 99.8,
        "volume": 1_000_000., "hh20": 100., "atr14": 2., "clv": .2,
        "volume_ratio": 1., "ema20_slope": .1, "rs20": .05,
        "history_ready": True, "market_ready": True,
    }, index=range(count))
    f.loc[2, ["open", "high", "low", "close", "clv", "volume_ratio"]] = [99.9, 101.2, 99.5, 101.1, .9, 1.2]
    f.loc[3:, ["open", "high", "low", "close", "hh20"]] = [101.2, 101.7, 100.8, 101.5, 101.7]
    return f


def retest(f, position=4, low=99.6):
    f.loc[position, ["open", "high", "low", "close", "clv"]] = [100., 101., low, 100.8, .7]
    return f


def bars(lows):
    count = len(lows)
    close = np.full(count, 110.)
    return pd.DataFrame({
        "Date": pd.bdate_range("2026-01-01", periods=count),
        "Open": close, "High": close + 1, "Low": lows, "Close": close,
        "Volume": 1_000_000., "Amount": close * 1_000_000., "Trades": 500,
        "Bid": close - .05, "Ask": close + .05, "PE": 20.,
        "TAIEX Close": 20000. + np.arange(count), "Restricted": False,
    })


def features(data):
    return build_features(data, SharedFeaturePipeline().build(data))


def test_retest_uses_frozen_level_and_atr_and_only_post_breakout_lows():
    f = retest(evidence(), low=99.7)
    f.loc[1, "low"] = 90.
    f.loc[2, "low"] = 95.
    f.loc[3, "low"] = 99.5
    f.loc[4, ["atr14", "hh20"]] = [20., 120.]
    result = _breakout_retest_features(f)
    assert result.loc[4, "breakout_retest_confirmed"]
    assert result.loc[4, "breakout_retest_support"] == 99.5
    assert result.loc[4, "breakout_retest_level"] == 100.
    assert result.loc[4, "breakout_retest_age"] == 2
    assert not result.loc[2, "breakout_retest_confirmed"]
    assert result.loc[2].drop("breakout_retest_confirmed").isna().all()


@pytest.mark.parametrize("age,expected", [(1, True), (5, True), (6, False)])
def test_retest_has_one_through_five_session_window(age, expected):
    result = _breakout_retest_features(retest(evidence(), position=2 + age))
    assert bool(result.loc[2 + age, "breakout_retest_confirmed"]) is expected


@pytest.mark.parametrize("column,value", [
    ("close", 99.9), ("low", 99.39), ("low", np.nan), ("open", np.nan),
    ("high", 100.), ("volume", 0.), ("history_ready", False), ("market_ready", False),
])
def test_intermediate_break_or_missing_bar_permanently_invalidates_event(column, value):
    f = retest(evidence())
    f.loc[3, column] = value
    result = _breakout_retest_features(f)
    assert not result["breakout_retest_confirmed"].any()
    assert result.drop(columns="breakout_retest_confirmed").isna().all().all()


@pytest.mark.parametrize("column,value", [
    ("hh20", np.nan), ("atr14", np.nan), ("atr14", 0.), ("clv", .49),
    ("volume_ratio", 1.09), ("ema20_slope", 0.), ("rs20", 0.),
    ("history_ready", False), ("market_ready", False),
])
def test_missing_or_weak_breakout_evidence_cannot_seed_retest(column, value):
    f = retest(evidence())
    f.loc[2, column] = value
    result = _breakout_retest_features(f)
    assert not result["breakout_retest_confirmed"].any()


def test_breakout_requires_fresh_crossover_not_existing_above_level_price():
    f = retest(evidence())
    f.loc[1, ["high", "close"]] = [101.2, 101.1]
    assert not _breakout_retest_features(f)["breakout_retest_confirmed"].any()


@pytest.mark.parametrize("low,expected", [(99.4, True), (99.399, False), (101., True), (101.001, False)])
def test_retest_low_bounds_are_inclusive_and_frozen(low, expected):
    f = evidence()
    f.loc[4, ["open", "high", "low", "close", "clv", "atr14"]] = [101.2, 102., low, 101.8, .8, 20.]
    assert bool(_breakout_retest_features(f).loc[4, "breakout_retest_confirmed"]) is expected


@pytest.mark.parametrize("column,value", [("clv", .39), ("clv", np.nan), ("open", 100.9)])
def test_retest_requires_held_bullish_demand_confirmation(column, value):
    f = retest(evidence())
    f.loc[4, column] = value
    assert not _breakout_retest_features(f).loc[4, "breakout_retest_confirmed"]


def test_newest_breakout_replaces_previous_event_without_same_bar_confirmation():
    f = evidence()
    f.loc[3, ["open", "high", "low", "close", "hh20", "clv", "volume_ratio"]] = [101.2, 102., 100.8, 101.9, 101.2, .9, 1.2]
    f.loc[4, ["open", "high", "low", "close", "hh20", "clv"]] = [101.3, 102., 101., 101.8, 102., .7]
    result = _breakout_retest_features(f)
    assert not result.loc[3, "breakout_retest_confirmed"]
    assert result.loc[4, "breakout_retest_confirmed"]
    assert result.loc[4, "breakout_retest_level"] == 101.2
    assert result.loc[4, "breakout_retest_support"] == 101.
    assert result.loc[4, "breakout_retest_age"] == 1


def test_confirmed_retest_is_not_carried_to_an_unconfirmed_following_bar():
    result = _breakout_retest_features(retest(evidence()))
    assert result.loc[4, "breakout_retest_confirmed"]
    assert not result.loc[5, "breakout_retest_confirmed"]
    assert result.loc[5].drop("breakout_retest_confirmed").isna().all()


def test_retest_prefixes_and_nondefault_indices_are_invariant():
    f = retest(evidence())
    f.index = pd.bdate_range("2026-01-01", periods=len(f))
    whole = _breakout_retest_features(f)
    for end in range(1, len(f) + 1):
        pd.testing.assert_frame_equal(whole.iloc[:end], _breakout_retest_features(f.iloc[:end]))
    missing = _breakout_retest_features(f.drop(columns="rs20"))
    assert not missing["breakout_retest_confirmed"].any()


def test_equal_low_shelf_is_visible_only_after_two_completed_right_bars():
    lows = np.full(75, 105.)
    lows[65:71] = [104., 100., 103., 100., 104., 105.]
    f = features(bars(lows))
    assert pd.isna(f.loc[66, "swing_low"])
    assert pd.isna(f.loc[67, "swing_low"])
    assert f.loc[68, "swing_low"] == 100.
    assert f.loc[70, "swing_low"] == 100.


def test_flat_low_plateaus_do_not_create_structural_support():
    f = features(bars(np.full(75, 100.)))
    assert f["swing_low"].isna().all()
    assert not f["higher_low"].any()


def test_duplicate_equal_low_events_preserve_previous_distinct_floor():
    lows = np.full(80, 105.)
    lows[60] = 95.
    lows[66] = 100.
    lows[68] = 100.
    f = features(bars(lows))
    assert f.loc[68, "swing_low"] == 100.
    assert f.loc[68, "previous_swing_low"] == 95.
    assert f.loc[70, "swing_low"] == 100.
    assert f.loc[70, "previous_swing_low"] == 95.
    assert f.loc[70, "higher_low"]


def test_new_equal_low_bottom_replaces_broken_stale_support_and_stop_anchor():
    lows = np.full(78, 55.)
    lows[65] = 50.8
    data = bars(lows)
    data[["Open", "High", "Close"]] = [56., 57., 56.]
    data.loc[68:73, ["Open", "High", "Low", "Close"]] = [
        [51.5, 51.8, 50.1, 50.1],
        [49.55, 49.55, 48.3, 48.5],
        [47.5, 48., 46.7, 47.4],
        [47.5, 47.8, 46.7, 46.7],
        [47.2, 47.7, 47., 47.7],
        [48.35, 49.3, 48.25, 48.75],
    ]
    f = features(data)
    assert f.loc[67, "support"] == 50.8
    assert f.loc[68, "support_break"]
    assert f.loc[71, "support"] == 50.8  # Two completed right bars not yet available.
    assert f.loc[72, "support"] == 46.7
    assert f.loc[73, "support"] == f.loc[73, "stop_support"] == 46.7
    assert f.loc[73, "previous_swing_low"] == 50.8
    assert not f.loc[73, "higher_low"]
    assert f.loc[73, "support_intact"]
    plan = trade_plan(f.loc[73].to_dict(), config=LowEntryV6Config())
    assert plan["stop_price"] == pytest.approx(46.7 - .3 * f.loc[73, "atr14"])
    assert plan["stop_distance_atr"] == pytest.approx((49.3 - plan["stop_price"]) / f.loc[73, "atr14"])


def test_equal_high_resistance_remains_strict_and_full_features_are_causal():
    lows = np.full(75, 105.)
    lows[65:71] = [104., 100., 103., 100., 104., 105.]
    data = bars(lows)
    data.loc[66, "High"] = 115.
    data.loc[68, "High"] = 115.
    whole = features(data)
    assert whole["swing_high"].isna().all()
    for end in (65, 67, 68, 69, 70, 71, 74):
        pd.testing.assert_frame_equal(whole.iloc[:end], features(data.iloc[:end]))
