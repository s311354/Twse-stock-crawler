"""Build the daily analysis dataset from market history and strategy plugins."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from application.market_history import MarketHistoryAssembler
from application.market_history import MarketHistoryInput
from application.strategy_execution import PreparedStrategyInputs
from application.strategy_execution import StrategyExecutionService
from application.strategy_contracts import StrategyDecisionAssembler
from domain.features import FeatureFrame
from domain.model_contracts import StrategyDecision
from domain.models import ProfitRow
from domain.models import SignalRow
from domain.services import compute_signal_features
from domain.strategies.expected_return_exit_v3 import (
    EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS,
)
from domain.strategies.expected_return_exit_v3 import (
    EXPECTED_RETURN_EXIT_STRATEGY_NAME,
)
from domain.strategies.low_entry_score_v5 import LOW_ENTRY_OUTPUT_COLUMNS
from domain.strategies.low_entry_score_v5 import LOW_ENTRY_STRATEGY_NAME
from domain.strategies.portfolio_allocation_v2 import (
    PAE_V2_OUTPUT_COLUMNS,
)


PROFIT_COLUMNS = [
    "無限次交易",
    "交易一次",
    "至多五次交易",
    "無限次交易(手續費$NTD300)",
    "利潤比",
]
SIGNAL_COLUMNS = [
    "Trend Score",
    "MOM",
    "CS",
    "Stop_Loss",
    "Take_Profit",
]
BASE_COLUMNS = [
    "日期",
    "證券代號",
    "證券名稱",
    "開盤",
    "最高",
    "最低",
    "收盤",
    "無限次交易(手續費$NTD300)",
    "利潤比",
    *SIGNAL_COLUMNS,
]
CORE_ANALYSIS_COLUMNS = (
    BASE_COLUMNS
    + LOW_ENTRY_OUTPUT_COLUMNS
    + EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS
)
ANALYSIS_COLUMNS = (
    CORE_ANALYSIS_COLUMNS
    + list(PAE_V2_OUTPUT_COLUMNS)
)


class AnalysisDatasetBuilder:
    """Coordinate pure history assembly and feature-aware strategy execution."""

    def __init__(
        self,
        *,
        history_assembler: MarketHistoryAssembler | None = None,
        strategy_execution: StrategyExecutionService | None = None,
        decision_assembler: StrategyDecisionAssembler | None = None,
    ) -> None:
        self.history_assembler = (
            history_assembler or MarketHistoryAssembler()
        )
        self.strategy_execution = (
            strategy_execution or StrategyExecutionService()
        )
        self.decision_assembler = (
            decision_assembler or StrategyDecisionAssembler()
        )

    def build_history(
        self,
        market: MarketHistoryInput,
        item: int,
    ) -> pd.DataFrame:
        return self.history_assembler.build(market, item)

    def prepare_strategy_inputs(
        self,
        market: MarketHistoryInput,
        histories: Sequence[pd.DataFrame] | None = None,
        feature_frames: Sequence[FeatureFrame] | None = None,
    ) -> PreparedStrategyInputs:
        effective_histories = (
            list(histories)
            if histories is not None
            else self.history_assembler.build_all(market)
        )
        if len(effective_histories) != market.stock_count:
            raise ValueError(
                "Strategy histories must match market stock_count"
            )
        prepared = self.strategy_execution.prepare(
            effective_histories,
            feature_frames,
        )
        if len(prepared.feature_frames) != market.stock_count:
            raise ValueError(
                "FeatureFrames must match market stock_count"
            )
        return prepared

    def build_signal_features(
        self,
        market: MarketHistoryInput,
    ) -> list[SignalRow]:
        rows: list[SignalRow] = []
        for item in range(market.stock_count):
            features = compute_signal_features(
                closes=list(market.closes[item]),
                highs=list(market.highs[item]),
                lows=list(market.lows[item]),
            )
            rows.append(
                [
                    features.trend_score,
                    features.mom,
                    features.cs,
                    features.stop_loss,
                    features.take_profit,
                ]
            )
        return rows

    def evaluate_low_entry(
        self,
        prepared: PreparedStrategyInputs,
    ) -> pd.DataFrame:
        return self.strategy_execution.evaluate(
            LOW_ENTRY_STRATEGY_NAME,
            LOW_ENTRY_OUTPUT_COLUMNS,
            prepared,
        )

    def evaluate_expected_return_exit(
        self,
        prepared: PreparedStrategyInputs,
    ) -> pd.DataFrame:
        return self.strategy_execution.evaluate(
            EXPECTED_RETURN_EXIT_STRATEGY_NAME,
            EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS,
            prepared,
        )

    def build(
        self,
        market: MarketHistoryInput,
        maxprofits: Sequence[ProfitRow],
    ) -> pd.DataFrame:
        decisions = self.build_decisions(market, maxprofits)
        return self.project_decisions(decisions)

    def build_decisions(
        self,
        market: MarketHistoryInput,
        maxprofits: Sequence[ProfitRow],
    ) -> tuple[StrategyDecision, ...]:
        analysis = self._build_analysis_frame(market, maxprofits)
        return self.decision_assembler.assemble(analysis, market)

    def project_decisions(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> pd.DataFrame:
        return self.decision_assembler.project(
            decisions,
            ANALYSIS_COLUMNS,
        )

    def _build_analysis_frame(
        self,
        market: MarketHistoryInput,
        maxprofits: Sequence[ProfitRow],
    ) -> pd.DataFrame:
        if len(maxprofits) != market.stock_count:
            raise ValueError("maxprofits must match market stock_count")

        analysis = pd.DataFrame(maxprofits, columns=PROFIT_COLUMNS)
        signal_features = pd.DataFrame(
            self.build_signal_features(market),
            columns=SIGNAL_COLUMNS,
        )
        prepared = self.prepare_strategy_inputs(market)
        low_entry = self.evaluate_low_entry(prepared)
        expected_return_exit = self.evaluate_expected_return_exit(prepared)
        analysis = pd.concat(
            [
                analysis,
                signal_features,
                low_entry,
                expected_return_exit,
            ],
            axis=1,
        )
        analysis["日期"] = [
            dates[-1] if dates else market.fallback_date
            for dates in market.dates
        ]
        analysis["證券代號"] = list(market.stock_numbers)
        analysis["證券名稱"] = list(market.stock_names)
        analysis["開盤"] = [
            values[-1] if values else None
            for values in market.opens
        ]
        analysis["最高"] = [
            values[-1] if values else None
            for values in market.highs
        ]
        analysis["最低"] = [
            values[-1] if values else None
            for values in market.lows
        ]
        analysis["收盤"] = [
            values[-1] if values else None
            for values in market.closes
        ]
        return analysis[CORE_ANALYSIS_COLUMNS]


__all__ = [
    "AnalysisDatasetBuilder",
    "ANALYSIS_COLUMNS",
    "BASE_COLUMNS",
    "CORE_ANALYSIS_COLUMNS",
    "PROFIT_COLUMNS",
    "SIGNAL_COLUMNS",
]
