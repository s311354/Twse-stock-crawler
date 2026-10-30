"""SYNTHETIC_ONLY causal distribution features; no forecasting skill claim."""
from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from domain.les_v7.execution_atom_features import (
    FEATURE_NAMES, FEATURE_BOUNDS, FRACTAL_DERIVED_NAMES, profile,
    _legal_price_count, _mass, _from_selected_bars, compute_execution_features,
)
from domain.les_v7.features import compute_features
from domain.les_v7.fractals import is_active_fractal_feature
from tests.strategies.test_v7_execution_features import fixture as original_fixture


def fixture(count=180, zero=False):
    data, base, plan, cutoff = original_fixture(count)
    if zero:
        data.loc[data.index[1:], "Open"] = data["Close"].to_numpy()[:-1]
    native = compute_features(data)[0]
    base = dict(base, atr=native["atr"])
    plan = replace(plan, atr=base["atr"], band=(base["close"], base["close"] + .25 * base["atr"]))
    return data, base, plan, cutoff


def test_exact_fixed_profile_is_fresh_bounded_and_all_derived_fractals_are_recognized():
    first = profile(); first["feature_names"].clear()
    definition = profile()
    assert len(definition["feature_names"]) == len(definition["feature_bounds"]) == 11
    assert definition["feature_names"] == list(FEATURE_NAMES)
    assert definition["required_close_count"] == 135
    assert {name for name in FEATURE_NAMES if is_active_fractal_feature(name)} == set(FRACTAL_DERIVED_NAMES)


def test_complete_inputs_compute_without_mutation_and_match_native_atr():
    data, base, plan, cutoff = fixture()
    old_data, old_base = data.copy(deep=True), deepcopy(base)
    values, audit = compute_execution_features(data, cutoff=cutoff, base_features=base, plan=plan)
    assert audit["status"] == "COMPUTED"
    assert audit["native_atr"]["last_value"] == base["atr"] == plan.atr
    assert audit["native_atr"]["selected_history_count"] == len(data)
    assert all(lower <= values[name] <= upper for name, (lower, upper) in zip(FEATURE_NAMES, FEATURE_BOUNDS))
    assert values[FRACTAL_DERIVED_NAMES[0]] == pytest.approx((base["dfa512"] - .5) * values["execution_atr_band_logit60"])
    pd.testing.assert_frame_equal(data, old_data); assert base == old_base


def test_future_extremes_and_unavailable_past_bars_are_excluded_before_statistics():
    data, base, plan, cutoff = fixture()
    expected = compute_execution_features(data, cutoff=cutoff, base_features=base, plan=plan)
    future = data.iloc[[-1]].copy(); future["Date"] = "2026-09-15"
    future["Available At"] = "2026-09-15T13:30:00+08:00"
    future[["Open", "High", "Low", "Close"]] = [10000., 20000., .01, 15000.]
    assert compute_execution_features(pd.concat([data, future], ignore_index=True),
        cutoff=cutoff, base_features=base, plan=plan) == expected
    delayed = data.copy(deep=True)
    delayed.loc[delayed.index[-2], "Available At"] = "2026-09-15T13:30:00+08:00"
    selected = delayed.drop(delayed.index[-2]); new_atr = compute_features(selected)[0]["atr"]
    new_base, new_plan = dict(base, atr=new_atr), replace(plan, atr=new_atr)
    assert compute_execution_features(delayed, cutoff=cutoff, base_features=new_base, plan=new_plan) == compute_execution_features(
        selected, cutoff=cutoff, base_features=new_base, plan=new_plan)


def test_zero_gap_atom_uses_closed_band_edges_and_declared_jeffreys_prior():
    data, base, plan, cutoff = fixture(zero=True)
    included, audit = compute_execution_features(data, cutoff=cutoff, base_features=base,
        plan=replace(plan, band=(base["close"], base["close"])))
    assert audit["status"] == "COMPUTED"
    for window in (20, 60, 120):
        assert included[f"execution_atom_band_logit{window}"] == math.log(2 * window + 1)
        assert included[f"execution_atr_band_logit{window}"] == math.log(2 * window + 1)
        assert audit["kernel_windows"]["atom"][str(window)]["zero_count"] == window
    excluded, outside = compute_execution_features(data, cutoff=cutoff, base_features=base,
        plan=replace(plan, band=(base["close"] - .2, base["close"] - .1)))
    assert outside["status"] == "COMPUTED"
    assert excluded["execution_atom_band_logit60"] == -math.log(121)


def test_kernel_adds_distance_information_when_r1_inside_band_fraction_is_identical():
    near = .006 + .0007 * np.sin(np.arange(20))
    far = .05 + .0007 * np.sin(np.arange(20))
    assert np.mean((near >= 0) & (near <= .005)) == np.mean((far >= 0) & (far <= .005)) == 0
    left, left_audit = _mass(near, np.zeros(20, dtype=bool), 0., .005)
    right, _ = _mass(far, np.zeros(20, dtype=bool), 0., .005)
    assert left > right
    assert left_audit["mode"] == "NONZERO_GAUSSIAN_KERNEL"
    atom, proof = _mass(np.repeat(.01, 20), np.zeros(20, dtype=bool), .01, .01)
    assert atom == 1. and proof["mode"] == "DEGENERATE_NONZERO_EMPIRICAL_ATOMS"


def test_historical_gaps_use_each_preceding_atr_instead_of_current_atr():
    data, base, plan, cutoff = fixture()
    # Independent scalar reconstruction of the native seed and recursion.
    atr, ranges, prior = [float("nan")], [], None
    for index in range(1, len(data)):
        row, previous_close = data.iloc[index], float(data["Close"].iloc[index - 1])
        value = max(row["High"] - row["Low"], abs(row["High"] - previous_close), abs(row["Low"] - previous_close))
        ranges.append(value)
        if prior is None and len(ranges) == 14: prior = sum(ranges) / 14
        elif prior is not None: prior = (prior * 13 + value) / 14
        atr.append(float("nan") if prior is None else prior)
    gap = data["Open"].to_numpy()[1:] - data["Close"].to_numpy()[:-1]
    normalized = gap[-60:] / np.asarray(atr[:-1])[-60:]
    wrong_current = gap[-60:] / atr[-1]
    _, audit = compute_execution_features(data, cutoff=cutoff, base_features=base, plan=plan)
    observed = audit["kernel_windows"]["atr"]["60"]["nonzero_std"]
    assert observed == pytest.approx(float(np.std(normalized)), abs=1e-14)
    assert abs(observed - float(np.std(wrong_current))) > 1e-5


@pytest.mark.parametrize("band,expected", [((9.99,10.05),3),((49.95,50.1),3),((99.9,100.5),3),
    ((499.5,501.),3),((999.,1005.),3),((10.01,10.04),0),((100.,100.),1)])
def test_legal_tick_span_matches_native_grid_boundaries(band, expected):
    assert _legal_price_count(band) == expected


def test_insufficient_history_missing_inputs_and_atr_reseeding_never_impute():
    data, base, plan, cutoff = fixture()
    values, audit = compute_execution_features(data.iloc[-134:], cutoff=cutoff, base_features=base, plan=plan)
    assert audit["status"] == "NOT_READY" and all(value is None for value in values.values())
    values, audit = compute_execution_features(data, cutoff=cutoff, base_features=dict(base, dfa512=None), plan=plan)
    assert audit["status"] == "NOT_READY" and values[FRACTAL_DERIVED_NAMES[0]] is None
    with pytest.raises(ValueError, match="NATIVE_ATR_BINDING"):
        _from_selected_bars(data.iloc[-135:], base_features=base, plan=plan)
    values, audit = compute_execution_features(data, cutoff=cutoff, base_features=base, plan=None)
    assert audit["status"] == "NOT_READY" and all(value is None for value in values.values())


def test_outside_tick_support_is_reported_without_clipping():
    data, base, plan, cutoff = fixture()
    values, audit = compute_execution_features(data, cutoff=cutoff, base_features=base,
        plan=replace(plan, band=(.01, 1e8)))
    assert audit["status"] == "OOD"
    assert "execution_legal_band_log1p_prices" in audit["outside_support"]
    assert values["execution_legal_band_log1p_prices"] > dict(zip(FEATURE_NAMES, FEATURE_BOUNDS))["execution_legal_band_log1p_prices"][1]
