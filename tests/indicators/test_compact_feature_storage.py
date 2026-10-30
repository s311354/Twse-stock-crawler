"""Synthetic exact numeric storage/feature equivalence; no market claims."""
from dataclasses import replace
import pickle

import numpy as np
import pandas as pd
import pytest

from domain.features import FeatureFrame, SharedFeatureConfig, SharedFeaturePipeline
from domain.les_v7 import features as v7
from tests.indicators.test_shared_feature_pipeline import _history
from tests.strategies.test_low_entry_score_v7 import CUTOFF, bars


def legacy_frame(frame, index):
    return pd.DataFrame({str(column): frame[column].tolist() for column in frame},
                        index=pd.Index(tuple(index.tolist())))


@pytest.mark.parametrize("size", [0, 1, 21, 240, 700])
def test_compact_storage_preserves_every_native_value_dtype_and_index(size):
    history = _history(size)
    history.index = history.index * 3 + 7
    if size > 1:
        history.loc[history.index[1], ["Close", "Volume", "TAIEX Close"]] = np.nan
    pipeline = SharedFeaturePipeline()
    calculated = pipeline._calculate(history)
    expected = legacy_frame(calculated, calculated.index)
    stored = FeatureFrame.from_pandas(calculated, config=pipeline.config, source_data=history)
    pd.testing.assert_frame_equal(stored.to_pandas(), expected, check_exact=True)
    for column in calculated:
        pd.testing.assert_series_equal(stored.series(column),
            pd.Series(calculated[column].tolist(), index=expected.index, name=column), check_exact=True)
    assert stored.source_fingerprint == pipeline.build(history).source_fingerprint
    assert all(isinstance(column.buffer, bytes) for column in stored._values)
    stored.assert_compatible(history)


@pytest.mark.parametrize("dtype", ["bool", "int8", "int32", "uint8", "uint64", "float16", "float32", "float64", "complex64"])
def test_numeric_storage_keeps_scalar_inference_not_source_width(dtype):
    values = [0, 1, 2] if dtype != "bool" else [False, True, False]
    if dtype == "uint64":
        values[-1] = 2 ** 63 + 1
    frame = pd.DataFrame({"value": np.asarray(values, dtype=dtype)})
    history = _history(3)
    stored = FeatureFrame.from_pandas(frame, config=SharedFeatureConfig(), source_data=history)
    expected = legacy_frame(frame, frame.index)
    pd.testing.assert_frame_equal(stored.to_pandas(), expected, check_exact=True)
    pd.testing.assert_series_equal(stored.series("value"), expected["value"], check_exact=True)


@pytest.mark.parametrize("values", [pd.Series([1., pd.NA, 3.], dtype="Float64"),
                                    pd.Series([True, pd.NA, False], dtype="boolean"),
                                    pd.Series(["first", None, "last"], dtype="str"),
                                    pd.Series(pd.date_range("2026-01-01", periods=3, tz="Asia/Taipei"))])
def test_non_numpy_columns_keep_legacy_inference(values):
    frame = pd.DataFrame({"value": values})
    stored = FeatureFrame.from_pandas(frame, config=SharedFeatureConfig(), source_data=_history(3))
    pd.testing.assert_frame_equal(stored.to_pandas(), legacy_frame(frame, frame.index), check_exact=True)


def test_numeric_build_does_not_box_values_with_tolist(monkeypatch):
    history = _history(240)
    calculated = SharedFeaturePipeline()._calculate(history)
    def forbidden(series):
        raise AssertionError("numeric feature storage must not box each scalar")
    monkeypatch.setattr(pd.Series, "tolist", forbidden)
    stored = FeatureFrame.from_pandas(calculated, config=SharedFeatureConfig(), source_data=history)
    assert len(stored.to_pandas()) == 240


def test_bytes_backing_is_immutable_and_pickle_replace_and_legacy_values_work():
    history = _history(240)
    stored = SharedFeaturePipeline().build(history)
    expected = stored.to_pandas()
    for value in (stored, pickle.loads(pickle.dumps(stored)), replace(stored)):
        for column in value._values:
            view = np.frombuffer(column.buffer, dtype=column.dtype)
            with pytest.raises(ValueError):
                view.setflags(write=True)
        mutable = value.to_pandas()
        mutable.iloc[0, 0] = -100.
        mutable_series = value.series("close")
        mutable_series.iloc[0] = -200.
        pd.testing.assert_frame_equal(value.to_pandas(), expected, check_exact=True)
    legacy = replace(stored, _values=tuple(tuple(expected[column].tolist()) for column in expected))
    restored = pickle.loads(pickle.dumps(legacy))
    pd.testing.assert_frame_equal(restored.to_pandas(), expected, check_exact=True)
    pd.testing.assert_series_equal(restored.series("close"), expected["close"], check_exact=True)
    history.loc[0, "Close"] = -300.
    pd.testing.assert_frame_equal(stored.to_pandas(), expected, check_exact=True)


def legacy_resilience(market, close, rs20):
    result = np.nan
    for i in range(20, len(market)):
        m, stock = market.iloc[i - 20:i + 1], close.iloc[i - 20:i + 1]
        result = np.nan
        if m.notna().all():
            peak = int(np.argmax(m.to_numpy()))
            result = (np.log(stock.iloc[-1] / stock.iloc[peak]) - np.log(m.iloc[-1] / m.iloc[peak])
                      if peak < 20 and m.iloc[-1] < m.iloc[peak] else rs20.iloc[i])
    return result


@pytest.mark.parametrize("size", [1, 20, 21, 61, 240, 700])
@pytest.mark.parametrize("scenario", ["down", "rising", "ties", "missing_last_window", "missing_before_window"])
def test_final_resilience_is_exact_to_full_history_loop(size, scenario, monkeypatch):
    data = bars(size)
    market = 10000. + np.arange(size, dtype=float)
    if scenario in {"down", "ties"}:
        market[-min(15, size):] -= np.arange(min(15, size)) * 10.
    if scenario == "ties" and size >= 21:
        market[-21:-19] = max(market[-21:]) + 10.
    if scenario == "missing_last_window":
        market[-1] = np.nan
    if scenario == "missing_before_window" and size > 21:
        market[-22] = np.nan
    data["TAIEX Close"] = market
    actual = v7.compute_features(v7.select_bars(data, CUTOFF))
    monkeypatch.setattr(v7, "_latest_resilience", legacy_resilience)
    expected = v7.compute_features(v7.select_bars(data, CUTOFF))
    assert actual == expected  # Every published feature and diagnostic, not only resilience.


def test_cutoff_excludes_future_rows_with_compact_storage_and_latest_resilience():
    data = bars(240)
    selected = v7.select_bars(data, CUTOFF)
    expected = v7.compute_features(selected)
    future = data.iloc[-1:].copy()
    future["Date"] = "2026-10-13"
    future[["Open", "High", "Low", "Close"]] *= 10
    combined = pd.concat((data, future), ignore_index=True)
    assert v7.compute_features(v7.select_bars(combined, CUTOFF)) == expected


def test_latest_resilience_checks_one_bounded_window_not_all_history(monkeypatch):
    history = _history(700)
    market, close = history["TAIEX Close"], history["Close"]
    rs20 = close / close.shift(20) - market / market.shift(20)
    calls = []
    original = pd.Series.notna
    def tracked(series):
        calls.append(len(series))
        return original(series)
    monkeypatch.setattr(pd.Series, "notna", tracked)
    v7._latest_resilience(market, close, rs20)
    assert calls == [21]
