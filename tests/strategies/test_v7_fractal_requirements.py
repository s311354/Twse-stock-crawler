"""Synthetic regressions only; fractal history is not approval or trading odds."""
import math

import numpy as np
import pandas as pd
import pytest

from domain.les_v7.contracts import ReferencePlan
from domain.les_v7.features import compute_features, select_bars
from domain.les_v7.fractals import dfa, higuchi, fractal_history_requirements, is_active_fractal_feature
from domain.les_v7.localization import FrozenBundle, predict
from tests.strategies.test_low_entry_score_v7 import CUTOFF, bars, bundle, seal


REQUIRED_CLOSES = {
    "dfa512": 513,
    "dfa1024": 1025,
    "delta_dfa512_5": 518,
    "delta_dfa1024_5": 1030,
    "higuchi256": 256,
    "higuchi512": 512,
}


@pytest.mark.parametrize("count", [-1, True, 513., "513", None])
def test_invalid_close_count_is_not_silently_coerced(count):
    with pytest.raises(ValueError, match="NONNEGATIVE_CLOSE_COUNT_REQUIRED"):
        fractal_history_requirements(count)


def test_zero_history_lists_all_dependencies_without_estimates():
    report = fractal_history_requirements(0)
    assert set(report) == set(REQUIRED_CLOSES)
    for name, required in REQUIRED_CLOSES.items():
        assert report[name]["missing_close_count"] == required
        assert report[name]["status"] == "INSUFFICIENT_HISTORY"


def test_empty_feature_audit_retains_explicit_warmup_requirements():
    features, audit = compute_features(pd.DataFrame())
    assert features == {} and audit["status"] == "NO_BARS"
    assert audit["fractal_history"] == fractal_history_requirements(0)


def test_relative_atr_keeps_raw_price_units_out_of_the_model_feature():
    data = select_bars(bars(540), CUTOFF)
    original, _ = compute_features(data)
    scaled = data.copy()
    for name in ('Open', 'High', 'Low', 'Close'):
        scaled[name] *= 10
    actual, _ = compute_features(scaled)
    assert actual['atr'] == pytest.approx(10 * original['atr'])
    assert actual['atr_fraction'] == pytest.approx(original['atr_fraction'])
    assert original['atr_fraction'] == pytest.approx(original['atr'] / original['close'])


@pytest.mark.parametrize("feature,required", REQUIRED_CLOSES.items())
def test_history_requirements_are_close_counts_not_calendar_days(feature, required):
    before = fractal_history_requirements(required - 1)[feature]
    assert before == {
        "available_close_count": required - 1,
        "required_close_count": required,
        "missing_close_count": 1,
        "status": "INSUFFICIENT_HISTORY",
    }
    ready = fractal_history_requirements(required)[feature]
    assert ready == {
        "available_close_count": required,
        "required_close_count": required,
        "missing_close_count": 0,
        "status": "SUFFICIENT_HISTORY",
    }
    assert fractal_history_requirements(required + 7)[feature]["missing_close_count"] == 0


@pytest.mark.parametrize("name", REQUIRED_CLOSES)
def test_all_active_fractal_features_include_lagged_dfa(name):
    assert is_active_fractal_feature(name)


@pytest.mark.parametrize("name", ["close", "momentum", "mfdfa", "lppl", "delta_rsi", "context_REVERSAL"])
def test_shadow_and_non_fractal_features_are_not_active(name):
    assert not is_active_fractal_feature(name)


@pytest.mark.parametrize("feature,before_count,ready_count", [
    ("dfa512", 512, 513),
    ("delta_dfa512_5", 517, 518),
    ("dfa1024", 1024, 1025),
    ("delta_dfa1024_5", 1029, 1030),
])
def test_actual_estimators_preserve_missing_values_until_full_warmup(feature, before_count, ready_count):
    incomplete, before_audit = compute_features(select_bars(bars(before_count), CUTOFF))
    complete, ready_audit = compute_features(select_bars(bars(ready_count), CUTOFF))
    assert incomplete[feature] is None  # Never pad prices or substitute a smaller window.
    assert before_audit["fractal_history"] == fractal_history_requirements(before_count)
    assert ready_audit["fractal_history"] == fractal_history_requirements(ready_count)
    assert complete[feature] is not None and math.isfinite(complete[feature])


def test_history_diagnostics_do_not_change_original_numeric_window_formulas():
    data = select_bars(bars(1030), CUTOFF)
    features, _ = compute_features(data)
    log_closes = np.log(data.Close.to_numpy())
    returns = np.diff(log_closes)
    for window in (512, 1024):
        current = dfa(returns[-window:])
        previous = dfa(returns[-window - 5:-5])
        assert current.value is not None and previous.value is not None
        assert features[f"dfa{window}"] == current.value
        assert features[f"delta_dfa{window}_5"] == current.value - previous.value
    for window in (256, 512):
        assert features[f"higuchi{window}"] == higuchi(log_closes[-window:]).value


def test_post_cutoff_close_cannot_complete_delta_dfa_warmup():
    data = bars(517)
    expected, expected_audit = compute_features(select_bars(data, CUTOFF))
    future = data.tail(1).assign(Date="2026-10-13", Close=data.Close.iloc[-1] + .1)
    # Keep valid OHLC; the new close is excluded by its session cutoff, not invalid data.
    extended = pd.concat([data, future], ignore_index=True)
    selected = select_bars(extended, CUTOFF)
    actual, actual_audit = compute_features(selected)
    assert len(extended) == 518 and len(selected) == 517
    assert actual == expected and actual_audit == expected_audit
    assert actual["delta_dfa512_5"] is None
    assert actual_audit["fractal_history"]["delta_dfa512_5"]["missing_close_count"] == 1


def test_close_observed_after_cutoff_cannot_complete_dfa_warmup():
    data = bars(513)
    data["Available At"] = "2026-10-12T13:30:00+08:00"
    data.loc[data.index[-1], "Available At"] = "2026-10-12T13:30:01+08:00"
    selected = select_bars(data, CUTOFF)
    features, audit = compute_features(selected)
    assert len(selected) == 512 and features["dfa512"] is None
    assert audit["fractal_history"]["dfa512"]["missing_close_count"] == 1


def test_sufficient_history_does_not_claim_numerically_valid_estimate():
    data = bars(518).assign(Open=100., High=102., Low=98., Close=100.)
    features, audit = compute_features(select_bars(data, CUTOFF))
    assert audit["fractal_history"]["delta_dfa512_5"]["status"] == "SUFFICIENT_HISTORY"
    assert audit["fractal"]["dfa512"]["status"] == "ZERO_FLUCTUATION"
    assert features["dfa512"] is None and features["delta_dfa512_5"] is None


def _delta_bundle(*, name="delta_dfa512_5", no_fractal_baseline=False, validated=False):
    record = bundle()
    payload = record["payload"]
    payload["feature_names"] = [name]
    payload["feature_bounds"] = [[-2., 2.]]
    if not no_fractal_baseline:
        payload.pop("fractal_mode")
        payload.pop("baseline_validation_report_id")
    if validated:
        payload["fractal_null_validation_report_id"] = "SYNTHETIC_ONLY_NOT_MARKET_VALIDATED"
    return seal(record)


def _plan():
    return ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL",
                         100., 97., 108., 2., (100., 100.5), 1, .3, .3)


@pytest.mark.parametrize("name", ["dfa512", "higuchi256", "delta_dfa512_5"])
def test_active_fractal_model_cannot_claim_no_fractal_baseline(name):
    with pytest.raises(ValueError, match="FRACTAL_FEATURES_IN_NO_FRACTAL_BASELINE"):
        FrozenBundle.load(_delta_bundle(name=name, no_fractal_baseline=True))


@pytest.mark.parametrize("name", ["dfa512", "higuchi256", "delta_dfa512_5"])
def test_active_fractal_model_requires_same_null_validation(name):
    model = FrozenBundle.load(_delta_bundle(name=name))
    forecast = predict(model, dict(close=100., low=98., atr=2., **{name: .05}),
                       _plan(), cutoff=CUTOFF)
    assert forecast.status == "NOT_READY"
    assert forecast.reasons == ("FRACTAL_NULL_VALIDATION_NOT_PROVIDED",)
    assert forecast.quantiles is None and forecast.p_entry is None


def test_validated_delta_only_synthetic_model_can_use_existing_inference_path():
    model = FrozenBundle.load(_delta_bundle(validated=True))
    forecast = predict(model, dict(close=100., low=98., atr=2., delta_dfa512_5=.05),
                       _plan(), cutoff=CUTOFF)
    assert forecast.status == "VALID"


def test_missing_delta_feature_does_not_invent_a_numeric_estimate():
    model = FrozenBundle.load(_delta_bundle(validated=True))
    forecast = predict(model, dict(close=100., low=98., atr=2., delta_dfa512_5=None),
                       _plan(), cutoff=CUTOFF)
    assert forecast.status == "OOD"
    assert forecast.reasons == ("UNSUPPORTED_FEATURE_VALIDITY_MASK",)
    assert forecast.quantiles is None and forecast.p_entry is None
