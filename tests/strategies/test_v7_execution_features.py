"""SYNTHETIC ONLY: causal trailing inputs, not market performance evidence."""
from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from domain.les_v7.contracts import ReferencePlan
from domain.les_v7.execution_features import (
    EXECUTION_FEATURE_PROFILE, FEATURE_NAMES, FEATURE_BOUNDS, FRACTAL_DERIVED_NAMES,
    compute_execution_features, profile,
)
from domain.les_v7.features import compute_features
from domain.les_v7.fractals import is_active_fractal_feature


def fixture(count=160):
    index = np.arange(count, dtype=float)
    close = np.exp(np.linspace(math.log(80), math.log(100), count) + .015 * np.sin(index * .31))
    gap = .012 * np.sin(index * .73) + .002
    opened = np.r_[close[0], close[:-1] * np.exp(gap[1:])]
    bars = pd.DataFrame(dict(Date=pd.bdate_range(end="2026-09-14", periods=count).strftime("%Y-%m-%d"),
        Open=opened, High=np.maximum(opened, close) * 1.02,
        Low=np.minimum(opened, close) * .98, Close=close, Volume=np.full(count, 10000.),
        Complete=[True] * count))
    bars["Available At"] = bars["Date"] + "T13:30:00+08:00"
    last = float(close[-1])
    base = dict(close=last, atr=3., dfa512=.6, higuchi256=1.4, higuchi512=1.6)
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_EXECUTION_FEATURE_TEST_ONLY", "test", "REVERSAL",
        last, last - 5., last + 10., 3., (last, last + .75), 1, 0., 0.)
    return bars, base, plan, str(bars["Date"].iloc[-1]) + "T13:30:00+08:00"


def test_declared_profile_is_fresh_bounded_and_all_fractal_derived_inputs_are_ablated():
    first = profile(); first["feature_names"].clear(); first["feature_bounds"][0][0] = 999.
    second = profile()
    assert second["version"] == EXECUTION_FEATURE_PROFILE
    assert len(second["feature_names"]) == len(second["feature_bounds"]) == 16
    assert tuple(second["feature_names"]) == FEATURE_NAMES
    assert tuple(map(tuple, second["feature_bounds"])) == FEATURE_BOUNDS
    assert second["required_close_count"] == 121
    assert {name for name in FEATURE_NAMES if is_active_fractal_feature(name)} == set(FRACTAL_DERIVED_NAMES)


def test_exact_gap_statistics_current_band_fractions_and_interactions_do_not_mutate_inputs():
    bars, base, plan, cutoff = fixture()
    before, before_base = bars.copy(deep=True), deepcopy(base)
    result, audit = compute_execution_features(bars, cutoff=cutoff, base_features=base, plan=plan)
    gap = np.log(bars["Open"].to_numpy()[1:]) - np.log(bars["Close"].to_numpy()[:-1])
    returns = np.diff(np.log(bars["Close"].to_numpy()))
    lower, upper = (math.log(value) - math.log(base["close"]) for value in plan.band)
    assert audit["status"] == "COMPUTED" and audit["profile"] == EXECUTION_FEATURE_PROFILE
    for window in (5, 20, 60):
        assert result[f"execution_gap_mean{window}"] == float(np.mean(gap[-window:]))
        assert result[f"execution_gap_std{window}"] == float(np.std(gap[-window:]))
    for window in (20, 60, 120):
        assert result[f"execution_band_fraction{window}"] == float(np.mean((gap[-window:] >= lower) & (gap[-window:] <= upper)))
    assert result["execution_close_vol_ratio5_60"] == float(np.std(returns[-5:])) / float(np.std(returns[-60:]))
    assert result[FRACTAL_DERIVED_NAMES[0]] == pytest.approx(.1 * result["execution_gap_vol_ratio5_60"])
    assert result[FRACTAL_DERIVED_NAMES[1]] == pytest.approx(-.1 * result["execution_band_fraction20"])
    assert result[FRACTAL_DERIVED_NAMES[2]] == pytest.approx(.1 * result["execution_band_fraction60"])
    assert result["execution_band_width_fraction"] == (plan.band[1] - plan.band[0]) / base["close"]
    pd.testing.assert_frame_equal(bars, before); assert base == before_base


def test_future_bars_and_not_yet_available_past_bar_are_selected_before_statistics():
    bars, base, plan, cutoff = fixture()
    expected = compute_execution_features(bars, cutoff=cutoff, base_features=base, plan=plan)
    future = bars.iloc[[-1]].copy(deep=True)
    future["Date"] = "2026-09-15"; future["Available At"] = "2026-09-15T13:30:00+08:00"
    future[["Open", "High", "Low", "Close"]] = [9999., 20000., 1., 10000.]
    extended = pd.concat([bars, future], ignore_index=True)
    assert compute_execution_features(extended, cutoff=cutoff, base_features=base, plan=plan) == expected
    delayed = bars.copy(deep=True)
    delayed.loc[delayed.index[-2], "Available At"] = "2026-09-15T13:30:00+08:00"
    selected = delayed.drop(delayed.index[-2])
    assert compute_execution_features(delayed, cutoff=cutoff, base_features=base, plan=plan) == compute_execution_features(
        selected, cutoff=cutoff, base_features=base, plan=plan)


def test_insufficient_history_and_missing_plan_never_invent_zero_fill_features():
    bars, base, plan, cutoff = fixture()
    result, audit = compute_execution_features(bars.iloc[-120:], cutoff=cutoff, base_features=base, plan=plan)
    assert audit["status"] == "NOT_READY" and all(value is None for value in result.values())
    assert "INSUFFICIENT_EXECUTION_FEATURE_HISTORY" in audit["reasons"]
    result, audit = compute_execution_features(bars, cutoff=cutoff, base_features=base, plan=None)
    assert audit["status"] == "NOT_READY" and all(value is None for value in result.values())
    assert "EXECUTION_REFERENCE_PLAN_NOT_READY" in audit["reasons"]


def test_missing_fractal_remains_not_ready_but_observed_zero_gap_variance_is_not_missing_data():
    bars, base, plan, cutoff = fixture()
    missing = dict(base, dfa512=None)
    result, audit = compute_execution_features(bars, cutoff=cutoff, base_features=missing, plan=plan)
    assert audit["status"] == "NOT_READY" and result[FRACTAL_DERIVED_NAMES[0]] is None
    bars.loc[bars.index[1:], "Open"] = bars["Close"].to_numpy()[:-1]
    result, audit = compute_execution_features(bars, cutoff=cutoff, base_features=base, plan=plan)
    assert audit["status"] == "COMPUTED" and result["execution_gap_vol_ratio5_60"] == 0
    assert result[FRACTAL_DERIVED_NAMES[0]] == 0
    assert audit["variance_floor_applied"] == ["gap60"]
    assert audit["volatility_denominator_floor"] == profile()["volatility_denominator_floor"] == 1e-8


def test_band_is_the_actual_plan_band_and_known_empty_empirical_fraction_is_legitimate_zero():
    bars, base, plan, cutoff = fixture()
    high_band = replace(plan, band=(base["close"] * 2, base["close"] * 2.1))
    result, audit = compute_execution_features(bars, cutoff=cutoff, base_features=base, plan=high_band)
    assert audit["status"] == "COMPUTED"
    assert all(result[f"execution_band_fraction{window}"] == 0 for window in (20, 60, 120))


@pytest.mark.parametrize("change,code", [("close", "CUTOFF_MISMATCH"), ("atr", "ATR_MISMATCH")])
def test_causal_base_values_must_match_the_selected_view_and_reference_plan(change, code):
    bars, base, plan, cutoff = fixture(); base[change] += 1
    with pytest.raises(ValueError, match=code):
        compute_execution_features(bars, cutoff=cutoff, base_features=base, plan=plan)


def test_support_is_not_clipped_to_make_an_extreme_band_look_in_distribution():
    bars, base, plan, cutoff = fixture()
    result, audit = compute_execution_features(bars, cutoff=cutoff, base_features=base,
        plan=replace(plan, band=(base["close"], base["close"] * 20)))
    assert result["execution_band_width_fraction"] == 19
    assert audit["status"] == "OOD" and audit["outside_support"] == ["execution_band_width_fraction"]


def test_importing_or_computing_optional_profile_keeps_legacy_feature_outputs_unchanged():
    bars, base, plan, cutoff = fixture()
    expected = compute_features(bars)
    compute_execution_features(bars, cutoff=cutoff, base_features=base, plan=plan)
    assert compute_features(bars) == expected
    assert not set(FEATURE_NAMES) & set(expected[0])
