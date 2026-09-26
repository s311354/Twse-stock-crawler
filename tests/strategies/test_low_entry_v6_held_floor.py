"""A held trading range is not the same thing as monotonically rising lows."""
import numpy as np
import pandas as pd
import pytest

from domain.features import SharedFeaturePipeline
from domain.industry_admission import industry_entry_signal
from domain.industry_profiles import IndustryProfileRegistry
from domain.strategies.low_entry_v6_features import build_features
from domain.strategies.low_entry_v6_signals import decide, EXECUTABLE_SIGNALS
from tests.strategies.test_low_entry_score_v6 import CONFIG, bars, quality, setup_features


def range_bars():
    data = bars(80)
    data[["Open", "High", "Low", "Close"]] = [100., 102., 99., 101.]
    data.loc[63, "Low"] = 95.
    data.loc[77:79, "Low"] = [98., 99., 98.5]
    return data


def compute(data):
    return build_features(data, SharedFeaturePipeline().build(data))


@pytest.mark.parametrize("lows", [[98., 99., 98.5], [95., 95., 95.]])
def test_oscillating_or_equal_lows_above_known_floor_are_stable(lows):
    data = range_bars()
    data.loc[77:79, "Low"] = lows
    f = compute(data)
    assert f["no_new_low2"].iloc[-1]
    assert not f["new_low"].iloc[-2:].any()
    assert f["new_lows3"].iloc[-1] == 0


def test_new_low_is_compared_with_prior_floor_and_needs_two_held_sessions():
    data = range_bars()
    data.loc[77:79, "Low"] = [94.9, 95.2, 95.1]
    f = compute(data)
    assert f["new_low"].iloc[-3:].tolist() == [True, False, False]
    assert f["no_new_low2"].iloc[-3:].tolist() == [False, False, True]
    # A new floor is never treated as already held on the day of the break.
    data.loc[79, "Low"] = 94.8
    f = compute(data)
    assert f["new_low"].iloc[-1] and not f["no_new_low2"].iloc[-1]


def test_missing_lookback_and_short_history_cannot_pass_stabilization():
    for count in (1, 20, 21):
        assert not compute(range_bars().iloc[:count])["no_new_low2"].any()
    assert compute(range_bars().iloc[:22])["no_new_low2"].iloc[-1]
    data = range_bars()
    data.loc[65, "Low"] = np.nan
    assert not compute(data)["no_new_low2"].iloc[-1]


@pytest.mark.parametrize("field,value", [("Low", np.nan), ("Volume", 0.),
                                        ("High", 98.), ("Close", 103.)])
def test_invalid_recent_ohlcv_does_not_look_like_a_stable_base(field, value):
    data = range_bars()
    data.loc[78, field] = value
    f = compute(data)
    assert not f["history_ready"].iloc[-1]
    assert not f["no_new_low2"].iloc[-1]


def test_floor_features_use_only_completed_prefix_data():
    data = range_bars()
    full = compute(data)
    columns = ["new_low", "new_lows3", "no_new_low2"]
    for count in (20, 21, 22, 64, 77, 78, 79):
        prefix = compute(data.iloc[:count].copy())
        pd.testing.assert_frame_equal(full.iloc[:count][columns], prefix[columns])
    data.loc[79, "Low"] = 1.
    pd.testing.assert_frame_equal(full.iloc[:79][columns], compute(data).iloc[:79][columns])


@pytest.mark.parametrize("setup", ["REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION"])
def test_all_setups_consume_held_floor_without_waiving_quality_veto_or_risk(setup):
    stable = bool(compute(range_bars())["no_new_low2"].iloc[-1])
    f = setup_features(setup, no_new_low2=stable)
    assert decide(quality(), f, config=CONFIG)["signal"] in EXECUTABLE_SIGNALS
    for status in ("REJECT", "DATA_INSUFFICIENT", "PROVISIONAL", "WATCH_QUALITY"):
        assert decide(quality(status=status), f, config=CONFIG)["signal"] not in EXECUTABLE_SIGNALS
    for changes in ({"new_lows3": 2., "clv": -.7}, {"history_ready": False},
                    {"support": None}, {"support": 80.}, {"amount_ma20": 1.},
                    {"restricted": None}, {"spread": None}):
        assert decide(quality(), dict(f, **changes), config=CONFIG)["signal"] not in EXECUTABLE_SIGNALS


@pytest.mark.parametrize("sector", ["ELEC", "SEMI", "ELECCOMP", "COMP"])
def test_active_industries_preserve_held_floor_model_signals(sector):
    f = setup_features("REVERSAL", no_new_low2=bool(compute(range_bars())["no_new_low2"].iloc[-1]))
    result = decide(quality(), f, config=CONFIG)
    assert result["signal"] in EXECUTABLE_SIGNALS
    assert industry_entry_signal(result["signal"], IndustryProfileRegistry.resolve(sector)) == result["signal"]
