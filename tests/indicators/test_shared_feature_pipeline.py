from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from domain.features import FEATURE_PIPELINE_VERSION
from domain.features import SharedFeaturePipeline
from domain.strategies.expected_return_exit_v3 import (
    ExpectedReturnExitV3Strategy,
)
from domain.strategies.low_entry_score_v4_1 import LowEntryScoreV41Strategy


def test_shared_features_are_deterministic_and_vectorized() -> None:
    history = _history()
    pipeline = SharedFeaturePipeline()

    first = pipeline.build(history)
    second = pipeline.build(history)
    features = first.to_pandas()

    pd.testing.assert_frame_equal(features, second.to_pandas())
    assert first.pipeline_version == FEATURE_PIPELINE_VERSION
    assert features.iloc[-1]["ema20"] == pytest.approx(
        history["Close"]
        .ewm(span=20, adjust=False, min_periods=1)
        .mean()
        .iloc[-1]
    )
    assert features.iloc[-1]["macd_histogram"] == pytest.approx(
        features.iloc[-1]["macd"] - features.iloc[-1]["macd_signal"]
    )
    assert features.iloc[-1]["obv"] > features.iloc[0]["obv"]
    assert features.iloc[-1]["vwap20_typical"] > 0
    assert features.iloc[-1]["vwap60_close"] > 0


def test_feature_frame_is_immutable_and_returns_copies() -> None:
    features = SharedFeaturePipeline().build(_history())
    original_close = features.series("close").iloc[-1]

    mutable_frame = features.to_pandas()
    mutable_series = features.series("close")
    mutable_frame.loc[mutable_frame.index[-1], "close"] = -1
    mutable_series.iloc[-1] = -2

    assert features.series("close").iloc[-1] == original_close
    with pytest.raises(FrozenInstanceError):
        features.pipeline_version = "changed"  # type: ignore[misc]


def test_feature_frame_rejects_a_different_strategy_index() -> None:
    history = _history()
    features = SharedFeaturePipeline().build(history)
    shifted = history.copy()
    shifted.index = shifted.index + 1

    with pytest.raises(ValueError, match="index does not match"):
        features.assert_compatible(shifted)

    changed = history.copy()
    changed.loc[changed.index[-1], "Close"] += 1
    with pytest.raises(ValueError, match="source data does not match"):
        features.assert_compatible(changed)


def test_feature_aware_strategy_path_matches_standalone_plugin_api() -> None:
    history = _history()
    features = SharedFeaturePipeline().build(history)

    for strategy in (
        LowEntryScoreV41Strategy(),
        ExpectedReturnExitV3Strategy(),
    ):
        standalone = strategy.run(history)
        shared = strategy.run_with_features(history, features)
        pd.testing.assert_frame_equal(standalone, shared)


def _history(size: int = 240) -> pd.DataFrame:
    close = pd.Series(
        100 + np.arange(size) * 0.1 + np.sin(np.arange(size) / 5),
        dtype="float64",
    )
    volume = pd.Series(
        1_000 + (np.arange(size) % 7) * 50,
        dtype="float64",
    )
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=size, freq="B"),
            "Open": close - 0.3,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": volume,
            "Amount": close * volume,
            "Trades": np.full(size, 500),
            "Bid": close - 0.05,
            "Ask": close + 0.05,
            "PE": np.full(size, 18.0),
            "TAIEX Close": 20_000 + np.arange(size) * 2,
        }
    )
