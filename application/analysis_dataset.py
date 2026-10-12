"""Build the daily analysis dataset from market history and strategy plugins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pandas as pd

from application.market_history import MarketHistoryAssembler
from application.market_history import MarketHistoryInput
from application.low_entry_v6_daily import V6_DAILY_COLUMNS
from application.daily_entry import DailyEntryService
from application.daily_valuation import DailyValuationProvider
from application.low_entry_v6_daily import build_daily_v6
from application.daily_performance import measure_stage
from application.evidence_context import EvidenceContext
from application.leadership_screen import screen_once
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from application.strategy_execution import PreparedStrategyInputs
from application.strategy_execution import StrategyExecutionService
from application.strategy_contracts import StrategyDecisionAssembler
from domain.features import FeatureFrame
from domain.features import SharedFeaturePipeline
from domain.strategy import get_strategy
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
CORE_ANALYSIS_COLUMNS = list(dict.fromkeys(
    BASE_COLUMNS
    + LOW_ENTRY_OUTPUT_COLUMNS
    + V6_DAILY_COLUMNS
    + EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS
))
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
        daily_entry: DailyEntryService | None = None,
        optimized_daily: bool = True,
        stock_executor=None,
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
        self.daily_entry = daily_entry or DailyEntryService(reuse_watch=optimized_daily)
        self.optimized_daily = optimized_daily
        # Optional infrastructure port. Only stock-local, stateless native
        # computation crosses it; portfolio allocation stays in the caller.
        self.stock_executor = stock_executor

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
            include_normalized=self.optimized_daily,
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
        with measure_stage("les_and_valuation"):
            result = self.daily_entry.evaluate(prepared)
        # Keep the CSV column contract. Uncomputed legacy technical diagnostics
        # remain missing, never fabricated as zero or borrowed from v6 factors.
        return result.reindex(columns=list(dict.fromkeys(
            list(LOW_ENTRY_OUTPUT_COLUMNS) + V6_DAILY_COLUMNS,
        )))

    def evaluate_expected_return_exit(
        self,
        prepared: PreparedStrategyInputs,
    ) -> pd.DataFrame:
        with measure_stage("erem"):
            return self.strategy_execution.evaluate(
                EXPECTED_RETURN_EXIT_STRATEGY_NAME,
                EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS,
                prepared, latest_only=self.optimized_daily,
            )

    def _daily_components(self, market):
        if self.stock_executor is not None and (
                not self.optimized_daily or type(self) is not AnalysisDatasetBuilder
                or type(self.history_assembler) is not MarketHistoryAssembler
                or type(self.strategy_execution) is not StrategyExecutionService
                or self.strategy_execution._strategy_resolver is not get_strategy
                or self.strategy_execution._feature_pipeline_factory is not SharedFeaturePipeline
                or type(self.daily_entry) is not DailyEntryService
                or self.daily_entry.entry_evaluator is not build_daily_v6
                or type(self.daily_entry.valuation_provider) is not DailyValuationProvider
                or self.daily_entry.valuation_provider.config != DailyValuationProvider().config):
            raise ValueError("PARALLEL_REQUIRES_NATIVE_DAILY_COMPONENTS")
        if (not self.optimized_daily or type(self).evaluate_low_entry is not AnalysisDatasetBuilder.evaluate_low_entry
                or type(self.history_assembler) is not MarketHistoryAssembler):
            # Legacy subclass orchestration still receives its promised attrs.
            prepared = self.prepare_strategy_inputs(market)
            return self.evaluate_low_entry(prepared), self.evaluate_expected_return_exit(prepared)
        with measure_stage("history_assembly"):
            histories = [self.history_assembler.build_screen(market, item) for item in range(market.stock_count)]
        sessions = [pd.to_datetime(h.attrs.get("les_v6_analysis_session") or
                    (h["Date"].iloc[-1] if len(h) else None), errors="coerce") for h in histories]
        cutoffs = [pd.to_datetime(h.attrs.get(SIGNAL_DATE_ATTR), utc=True, errors="coerce") for h in histories]
        watches, _ = screen_once(histories,
            session_date=max((s for s in sessions if pd.notna(s)), default=pd.NaT),
            as_of=max((c for c in cutoffs if pd.notna(c)), default=pd.NaT))
        del histories, sessions, cutoffs
        entries, exits = [], []
        if self.stock_executor is not None:
            from application.daily_stock import isolate_stock_market
            jobs = ((item, isolate_stock_market(market, item), watch) for item, watch in enumerate(watches))
            with measure_stage("stock_parallel_wall"):
                results = self.stock_executor.evaluate(jobs)
            for item in range(market.stock_count):
                entry, exit = results[item]
                entries.append(entry)
                exits.append(exit)
        else:
            with EvidenceContext(market) as evidence:
                for item, watch in enumerate(watches):
                    entry, exit = self.stock_components(market, item, watch, evidence)
                    entries.append(entry)
                    exits.append(exit)
        entry_columns = list(dict.fromkeys(list(LOW_ENTRY_OUTPUT_COLUMNS) + V6_DAILY_COLUMNS))
        return (pd.concat(entries, ignore_index=True) if entries else pd.DataFrame(columns=entry_columns),
                pd.concat(exits, ignore_index=True) if exits else pd.DataFrame(columns=EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS))

    def stock_components(self, market, item, watch, evidence):
        """Shared serial/worker kernel; no persistence, network or allocation."""
        with measure_stage("history_assembly"):
            history = self.history_assembler.build(market, item, include_evidence=False)
        with evidence.stock(item, history) as packet:
            prepared = replace(self.strategy_execution.prepare([history], include_normalized=True),
                               watches=(watch,), evidence=(packet,))
            return self.evaluate_low_entry(prepared), self.evaluate_expected_return_exit(prepared)

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
        low_entry, expected_return_exit = self._daily_components(market)
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
