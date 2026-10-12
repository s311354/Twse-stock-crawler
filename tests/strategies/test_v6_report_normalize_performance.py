"""Same JSON-safe report semantics; no score/gate or missing-value changes."""
from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType

import numpy as np
import pandas as pd

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance
from domain.strategies import low_entry_score_v6 as module
from domain.strategies.low_entry_v6_quality import number
from tests.strategies.test_daily_v6_pipeline import daily_market


def reference_clean(value):
    # The pre-optimization implementation, not regenerated expected output.
    if isinstance(value, Mapping):
        return {str(key): reference_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [reference_clean(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and number(value) is None:
        return None
    return value


def test_normalization_is_exact_for_native_numpy_and_readonly_values():
    value = MappingProxyType({"proof": ({2: np.int64(7), "ratio": np.float64(-0.),
        "none": None, "flag": np.bool_(False)},), "other": [np.nan, np.inf, -np.inf, 0., -0.,
        True, False, 1, 1., "N/A", np.float32(1.25), np.array(1.5)]})
    assert module._clean(value) == reference_clean(value)
    from application.replay_contracts import canonical_json
    assert canonical_json(module._clean(value)) == canonical_json(reference_clean(value))


def test_complete_daily_result_equals_original_normalization(monkeypatch):
    market = daily_market()
    original = deepcopy(market.fundamental_snapshots)
    expected = None
    with monkeypatch.context() as patch:
        patch.setattr(module, "_clean", reference_clean)
        expected = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]])
    probe = DailyPerformance()
    with probe.activate():
        actual = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]])
    assert actual == expected
    assert market.fundamental_snapshots == original
    for stage in ("les_feature_extension", "les_decision", "les_portfolio_valuation", "les_report_normalize"):
        assert probe.calls[stage] == 1
    assert probe.counters["strategy_output_rows"] == 1
