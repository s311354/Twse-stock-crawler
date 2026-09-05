"""Lazy compatibility facade for pre-composition LES v5 callers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from domain.quality_v5 import LowEntryV5QualityResult
from domain.strategies.low_entry_v5_panic import LowEntryV5QualityEligibility
from domain.strategies.low_entry_v5_pipeline import LowEntryV5PipelineComponents
from domain.strategies.low_entry_v5_report import LowEntryV5ReportBuilder
from domain.valuation_v5 import LowEntryV5ValuationResult


class LowEntryV5CompatibilityFacade:
    """Expose the former RuleEngine surface without owning trading rules."""

    _QUALITY_METHODS = {
        "_conditional_quality_starter_eligible": (
            "conditional_quality_starter_eligible"
        ),
        "_broad_rally_quality_eligible": "broad_rally_quality_eligible",
        "_broad_rally_full_buy_quality_eligible": (
            "broad_rally_full_buy_quality_eligible"
        ),
        "_supportive_non_broad_quality_eligible": (
            "supportive_non_broad_quality_eligible"
        ),
        "_quality_pullback_starter_eligible": (
            "quality_pullback_starter_eligible"
        ),
    }

    def __init__(
        self,
        components: LowEntryV5PipelineComponents,
        *,
        config: Any,
        valuation_producer: object,
        market_adapter: object,
    ) -> None:
        self.config = config
        self.valuation_producer = valuation_producer
        self.market_adapter = market_adapter
        self.market_evaluator = components.market
        self.fundamental_evaluator = components.fundamentals
        self.panic_evaluator = components.panic
        self.scoring_evaluator = components.scoring
        self.decision_evaluator = components.decision
        self.report_builder = components.report
        quality_policy = getattr(components.panic, "quality_policy", None)
        self.quality_policy = quality_policy or LowEntryV5QualityEligibility(
            config
        )

    def __getattr__(self, name: str) -> Any:
        quality_method = self._QUALITY_METHODS.get(name)
        if quality_method is not None:
            return getattr(self.quality_policy, quality_method)
        for component in (
            self.market_evaluator,
            self.fundamental_evaluator,
            self.panic_evaluator,
            self.scoring_evaluator,
            self.decision_evaluator,
            self.report_builder,
        ):
            try:
                return getattr(component, name)
            except AttributeError:
                continue
        raise AttributeError(
            "{} has no compatibility member {!r}".format(
                type(self).__name__,
                name,
            )
        )

    @staticmethod
    def _quality_metric_passed(
        quality: LowEntryV5QualityResult,
        metric_name: str,
    ) -> bool:
        return LowEntryV5QualityEligibility.quality_metric_passed(
            quality,
            metric_name,
        )

    @staticmethod
    def scenario_value(
        result: LowEntryV5ValuationResult,
        scenario_name: str,
        field_name: str,
    ) -> object:
        return LowEntryV5ReportBuilder.scenario_value(
            result,
            scenario_name,
            field_name,
        )

    @staticmethod
    def cycle_review_value(review: object, field_name: str) -> object:
        return LowEntryV5ReportBuilder.cycle_review_value(review, field_name)

    @staticmethod
    def valuation_confidence(
        quality: LowEntryV5QualityResult,
        result: LowEntryV5ValuationResult,
    ) -> float:
        return LowEntryV5ReportBuilder.valuation_confidence(quality, result)


@dataclass(frozen=True)
class _CompatibilityRulesAdapter:
    rules: LowEntryV5CompatibilityFacade

    def __getattr__(self, name: str) -> Any:
        return getattr(self.rules, name)


class LowEntryV5MarketRules(_CompatibilityRulesAdapter):
    """Compatibility alias for the former market rules wrapper."""


class LowEntryV5FundamentalRules(_CompatibilityRulesAdapter):
    """Compatibility alias for the former fundamental rules wrapper."""


class LowEntryV5PanicRules(_CompatibilityRulesAdapter):
    """Compatibility alias for the former panic rules wrapper."""


class LowEntryV5ScoringRules(_CompatibilityRulesAdapter):
    """Compatibility alias for the former scoring rules wrapper."""


class LowEntryV5DecisionRules(_CompatibilityRulesAdapter):
    """Compatibility alias for the former decision rules wrapper."""


class LowEntryV5ReportRules(_CompatibilityRulesAdapter):
    """Compatibility alias for the former report rules wrapper."""


__all__ = [
    "LowEntryV5CompatibilityFacade",
    "LowEntryV5DecisionRules",
    "LowEntryV5FundamentalRules",
    "LowEntryV5MarketRules",
    "LowEntryV5PanicRules",
    "LowEntryV5ReportRules",
    "LowEntryV5ScoringRules",
]
