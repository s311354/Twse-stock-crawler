"""Synthetic parity fixtures: decoupling is not a trading-performance claim."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_entry import DailyEntryService
from application.daily_valuation import DailyValuationProvider
from application.low_entry_v6_daily import V6_PROJECTED_COLUMNS, overlay_daily_v6
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from application.strategy_contracts import StrategyContractError
from domain.models import PortfolioPosition
from domain.strategies.low_entry_score_v5 import LOW_ENTRY_OUTPUT_COLUMNS, LOW_ENTRY_STRATEGY_NAME
from domain.strategies.low_entry_v5_pipeline import LowEntryV5DecisionPipeline
from tests.strategies.test_daily_v6_pipeline import daily_market
from tests.strategies.test_low_entry_score_v5 import _annual_fundamentals, _quality_snapshot


class PreviousComposition(AnalysisDatasetBuilder):
    def evaluate_low_entry(self, prepared):
        legacy = self.strategy_execution.evaluate(
            LOW_ENTRY_STRATEGY_NAME, LOW_ENTRY_OUTPUT_COLUMNS, prepared,
        )
        return overlay_daily_v6(prepared.histories, legacy)


def market_with_sidecar(kind):
    market = daily_market()
    if kind == "missing":
        return market
    snapshots = deepcopy(market.fundamental_snapshots)
    snapshots["2330"].update(_quality_snapshot())
    if kind == "reject":
        snapshots["2330"]["parent_equity"] = -1
        snapshots["2330"]["going_concern"] = True
    elif kind == "future":
        snapshots["2330"]["available_at"] = "2026-09-12T15:00:00+08:00"
    return replace(
        market, annual_fundamentals={"2330": _annual_fundamentals()},
        fundamental_snapshots=snapshots,
    )


@pytest.mark.parametrize("kind", ["missing", "complete", "reject"])
@pytest.mark.parametrize("held", [False, True])
def test_v6_erem_sidecar_and_full_pae_result_equal_previous_composition(kind, held):
    market = market_with_sidecar(kind)
    before_builder, after_builder = PreviousComposition(), AnalysisDatasetBuilder()
    before = before_builder.build_decisions(market, [[1, 1, 1, 1, .01]])
    after = after_builder.build_decisions(market, [[1, 1, 1, 1, .01]])
    prepared = after_builder.prepare_strategy_inputs(market)
    sidecar = DailyValuationProvider().evaluate(prepared.histories[0], prepared.feature_frames[0])
    available = {field.name for field in before[0].fields}
    names = (set(V6_PROJECTED_COLUMNS) | set(sidecar)) & available
    names |= {name for name in available if name.startswith("EREM ")}
    for name in names:
        assert after[0].field(name) == before[0].field(name), name
    context = PortfolioRuntimeContext(
        account_equity=1_000_000., current_cash=990_330. if held else 1_000_000.,
        positions=(PortfolioPosition("2330", 100),) if held else (), default_sector="24",
    )
    left, right = PortfolioAllocationV2Pipeline(), PortfolioAllocationV2Pipeline()
    assert left._build_request(before, context, before[0].evaluated_at) == right._build_request(
        after, context, after[0].evaluated_at,
    )
    before_pae, after_pae = left.enrich(before, context), right.enrich(after, context)
    for field in before_pae[0].fields:
        if field.name.startswith("PAE "):
            assert after_pae[0].field(field.name) == field, field.name


def test_future_sidecar_evidence_remains_rejected_by_typed_contract():
    market = market_with_sidecar("future")
    for builder in (PreviousComposition(), AnalysisDatasetBuilder()):
        with pytest.raises(StrategyContractError, match="future point-in-time evidence"):
            builder.build_decisions(market, [[1, 1, 1, 1, .01]])


def test_daily_v6_does_not_execute_v5_entry_pipeline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Daily v6 must not execute the v5 entry pipeline")
    monkeypatch.setattr(LowEntryV5DecisionPipeline, "run", forbidden)
    monkeypatch.setattr(LowEntryV5DecisionPipeline, "run_with_features", forbidden)
    decisions = AnalysisDatasetBuilder().build_decisions(daily_market(), [[1, 1, 1, 1, .01]])
    assert decisions[0].value("LES Model Version") == "6.0.0"
    assert decisions[0].value("LES v5 Trigger") is None
    assert decisions[0].value("動能分數") is None
    assert decisions[0].value("LES v6 Score") is not None


def test_sidecar_cannot_overwrite_entry_owned_fields():
    class InvalidSidecar:
        def evaluate(self, history, features):
            return {"LES v6 Score": 100.}
    builder = AnalysisDatasetBuilder(daily_entry=DailyEntryService(valuation_provider=InvalidSidecar()))
    with pytest.raises(ValueError, match="DAILY_ENTRY_FIELD_OWNERSHIP_CONFLICT:LES v6 Score"):
        builder.build_decisions(daily_market(), [[1, 1, 1, 1, .01]])


@pytest.mark.parametrize("size", [5, 19, 20, 61])
def test_sidecar_positive_pe_window_and_fallback_match_full_v5(size):
    market = market_with_sidecar("complete")
    builder = AnalysisDatasetBuilder()
    history = builder.build_history(market, 0).tail(size).copy()
    history["PE"] = np.linspace(40., 10., size)
    history.loc[history.index[0], "PE"] = -1
    prepared = builder.strategy_execution.prepare([history])
    actual = DailyValuationProvider().evaluate(prepared.histories[0], prepared.feature_frames[0])
    expected = builder.strategy_execution.evaluate(LOW_ENTRY_STRATEGY_NAME, LOW_ENTRY_OUTPUT_COLUMNS, prepared)
    pd.testing.assert_series_equal(
        pd.Series(actual), expected.iloc[0].reindex(actual).rename(None),
        check_names=False, check_dtype=False,
    )


def test_sidecar_does_not_mutate_history_or_shared_features():
    prepared = AnalysisDatasetBuilder().prepare_strategy_inputs(market_with_sidecar("complete"))
    history, features = prepared.histories[0], prepared.feature_frames[0]
    before = history.copy(deep=True)
    shared = features.to_pandas()
    DailyValuationProvider().evaluate(history, features)
    pd.testing.assert_frame_equal(before, history)
    pd.testing.assert_frame_equal(shared, features.to_pandas())
