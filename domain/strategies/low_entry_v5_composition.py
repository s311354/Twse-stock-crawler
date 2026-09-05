"""Composition root for Low Entry Model v5 domain components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from domain.strategies.low_entry_v5_decision import LowEntryV5DecisionEvaluator
from domain.strategies.low_entry_v5_fundamentals import (
    LowEntryV5FundamentalEvaluator,
)
from domain.strategies.low_entry_v5_market import LowEntryV5MarketAdapter
from domain.strategies.low_entry_v5_market import LowEntryV5MarketEvaluator
from domain.strategies.low_entry_v5_panic import LowEntryV5PanicEvaluator
from domain.strategies.low_entry_v5_panic import LowEntryV5QualityEligibility
from domain.strategies.low_entry_v5_pipeline import LowEntryV5PipelineComponents
from domain.strategies.low_entry_v5_report import LowEntryV5ReportBuilder
from domain.strategies.low_entry_v5_scoring import LowEntryV5ScoringEvaluator
from domain.valuation_v5 import LowEntryV5ValuationProducer


@dataclass(frozen=True)
class LowEntryV5CompositionRoot:
    """Build one coherent, replaceable LES v5 component graph."""

    config: Any
    valuation_producer: LowEntryV5ValuationProducer
    market_adapter: LowEntryV5MarketAdapter
    output_columns: Sequence[str]

    def build(self) -> LowEntryV5PipelineComponents:
        quality_policy = LowEntryV5QualityEligibility(self.config)
        return LowEntryV5PipelineComponents(
            market=LowEntryV5MarketEvaluator(
                self.config,
                self.market_adapter,
            ),
            fundamentals=LowEntryV5FundamentalEvaluator(
                self.config.v5_quality,
                self.valuation_producer,
                self.market_adapter.positive_float,
            ),
            panic=LowEntryV5PanicEvaluator(
                self.config,
                quality_policy,
            ),
            scoring=LowEntryV5ScoringEvaluator(self.config),
            decision=LowEntryV5DecisionEvaluator(
                self.config,
                self.market_adapter,
                quality_policy,
            ),
            report=LowEntryV5ReportBuilder(list(self.output_columns)),
        )


__all__ = ["LowEntryV5CompositionRoot"]
