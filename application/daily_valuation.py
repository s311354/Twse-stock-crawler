"""Independent PIT quality/valuation sidecar; never executes an entry plugin."""
from __future__ import annotations

import math

import pandas as pd

from domain.features import FeatureFrame
from domain.numeric_history import metadata_free_frame
from domain.strategies.low_entry_score_v5 import LowEntryScoreV5Config
from domain.strategies.low_entry_v5_fundamentals import LowEntryV5FundamentalEvaluator
from domain.strategies.low_entry_v5_report import LowEntryV5ReportBuilder
from domain.valuation_v5 import LowEntryV5ValuationProducer


def _positive_float(value, fallback):
    # Preserve the legacy valuation price-unit policy exactly.
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) and parsed > 0 else fallback


class DailyValuationProvider:
    """Reuse v5 fundamental formulas, without v5 technical/decision orchestration.

    Legacy source names remain truthful: these are v5-versioned fundamental
    formulas, not v6 quality or a fabricated new valuation model.
    """

    def __init__(self, config: LowEntryScoreV5Config | None = None):
        self.config = config or LowEntryScoreV5Config()
        self.fundamentals = LowEntryV5FundamentalEvaluator(
            self.config.v5_quality, LowEntryV5ValuationProducer(self.config.valuation),
            _positive_float,
        )
        self.projection = LowEntryV5ReportBuilder([])

    def evaluate(self, history: pd.DataFrame, features: FeatureFrame, *, evidence=None) -> dict:
        features.assert_compatible(history)
        context = dict(history.attrs)
        if evidence is not None:
            evidence.validate(history)
            context.update(evidence.fundamental_attrs())
        history = metadata_free_frame(history)
        shared = features.to_pandas()
        # Only valuation dependencies. Preserve the 60-session positive-PE
        # median (20 observations minimum), including last-PE fallback.
        indicators = shared.reindex(columns=("close", "pe")).copy()
        indicators["pe_median60"] = indicators["pe"].where(
            indicators["pe"] > 0,
        ).rolling(self.config.range_window, min_periods=20).median()
        quality = self.fundamentals.evaluate_quality(history, context)
        if history.empty:
            from domain.valuation_v5 import LowEntryV5ValuationResult, ValuationProductionStatus
            result = LowEntryV5ValuationResult(
                status=ValuationProductionStatus.UNKNOWN, signal_date=None,
                current_price=None,
                passive_benchmark_expected_return=self.config.valuation.passive_benchmark_expected_return,
                required_return=(self.config.valuation.passive_benchmark_expected_return
                                 + self.config.valuation.required_excess_return),
                reasons=("MISSING_MARKET_HISTORY",),
            )
            valuation = pd.Series("UNKNOWN", index=pd.RangeIndex(1))
        else:
            valuation, result = self.fundamentals.evaluate_valuation(indicators.tail(1), context)
        projected = self.projection.fundamental_fields(quality, valuation, result)
        # The shared projector accepts Series for legacy full-history reports;
        # this independent sidecar publishes one scalar snapshot only.
        return {key: value.iloc[-1] if isinstance(value, pd.Series) else value
                for key, value in projected.items()}
