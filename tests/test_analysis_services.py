import pandas as pd
import pytest

import stockanalysis
from application.analysis_dataset import AnalysisDatasetBuilder
from application.market_history import MarketHistoryAssembler
from application.market_history import MarketHistoryInput
from application.strategy_execution import StrategyExecutionService
from domain.features import SharedFeaturePipeline
from domain.strategies.low_entry_score_v4 import ANNUAL_FUNDAMENTALS_ATTR
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_SNAPSHOT_ATTR
from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from domain.strategy import Strategy
from infrastructure.legacy_crawler_facade import LegacyCrawlerFacade


def market_history_input() -> MarketHistoryInput:
    return MarketHistoryInput(
        stock_count=1,
        stock_numbers=["2330"],
        stock_names=["台積電"],
        dates=[["2026-07-24", "2026-07-25"]],
        opens=[[990_000.0, 1_000_000.0]],
        highs=[[1_010_000.0, 1_020_000.0]],
        lows=[[980_000.0, 995_000.0]],
        closes=[[1_000_000.0, 1_010_000.0]],
        volumes=[[20_000.0, 22_000.0]],
        trade_counts=[[]],
        trade_values=[[20_000_000.0, 22_220_000.0]],
        bids=[[999_000.0, 1_009_000.0]],
        asks=[[1_001_000.0, 1_011_000.0]],
        pe_ratios=[[20.0, 20.2]],
        taiex_closes=[[23_000.0, 23_200.0]],
        annual_fundamentals={
            "2330": pd.DataFrame({"fiscal_year": [2025]})
        },
        fundamental_snapshots={"2330": {"ttm_eps": 45.0}},
        fundamental_statuses={},
        fundamental_status_details={},
        signal_dates={"2330": "2026-07-25T13:30:00+08:00"},
        account_equity=1_000_000.0,
        fallback_date="2026-07-25",
    )


def test_market_history_assembler_preserves_context_and_aligns_optional_data() -> None:
    history = MarketHistoryAssembler().build(
        market_history_input(),
        0,
    )

    assert history["Date"].tolist() == [
        "2026-07-24",
        "2026-07-25",
    ]
    assert history["Trades"].isna().all()
    assert history.attrs[SIGNAL_DATE_ATTR] == (
        "2026-07-25T13:30:00+08:00"
    )
    assert history.attrs[PRICE_SCALE_ATTR] == 1000.0
    assert history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR] == {
        "ttm_eps": 45.0
    }
    assert history.attrs[ANNUAL_FUNDAMENTALS_ATTR].equals(
        pd.DataFrame({"fiscal_year": [2025]})
    )


def test_market_history_assembler_rejects_misaligned_required_data() -> None:
    market = market_history_input()
    market.opens[0].pop()

    with pytest.raises(
        ValueError,
        match="Open history length must match Close",
    ):
        MarketHistoryAssembler().build(market, 0)


class ScoreStrategy(Strategy):
    name = "score"

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("shared feature execution should be used")

    def run_with_features(self, data, features):
        features.assert_compatible(data)
        result = data.copy()
        result["score"] = range(len(result))
        return result


def test_strategy_execution_builds_features_once_per_history_and_uses_latest() -> None:
    builds = 0
    factory_calls = 0

    class CountingPipeline:
        def __init__(self) -> None:
            nonlocal factory_calls
            factory_calls += 1
            self.delegate = SharedFeaturePipeline()

        def build(self, history):
            nonlocal builds
            builds += 1
            return self.delegate.build(history)

    histories = [
        MarketHistoryAssembler().build(market_history_input(), 0)
        for _ in range(2)
    ]
    service = StrategyExecutionService(
        strategy_resolver=lambda name: ScoreStrategy(),
        feature_pipeline_factory=CountingPipeline,
    )

    prepared = service.prepare(histories)
    result = service.evaluate("score", ["score"], prepared)

    assert factory_calls == 1
    assert builds == 2
    assert result["score"].tolist() == [1, 1]


def test_dataset_builder_validates_profit_rows_before_strategy_execution() -> None:
    with pytest.raises(
        ValueError,
        match="maxprofits must match market stock_count",
    ):
        AnalysisDatasetBuilder().build(market_history_input(), [])


def test_twse_crawler_keeps_extracted_legacy_facade_api() -> None:
    assert issubclass(stockanalysis.TwseCrawker, LegacyCrawlerFacade)
    assert "build_analysis_dataset" not in stockanalysis.TwseCrawker.__dict__
    crawler = stockanalysis.TwseCrawker(0)
    assert callable(crawler.cal_max_profit)
    assert callable(crawler.record_analysis_dataset)
    assert callable(crawler.smtp_email)
