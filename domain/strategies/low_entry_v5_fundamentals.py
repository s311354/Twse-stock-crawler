"""Fundamental quality and valuation component for Low Entry Model v5."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

import pandas as pd

from domain.quality_v5 import LowEntryV5QualityConfig
from domain.quality_v5 import LowEntryV5QualityResult
from domain.quality_v5 import evaluate_v5_quality
from domain.strategies.low_entry_score_v4 import ANNUAL_FUNDAMENTALS_ATTR
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_SNAPSHOT_ATTR
from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from domain.valuation_v5 import LowEntryV5ValuationProducer
from domain.valuation_v5 import LowEntryV5ValuationResult
from domain.valuation_v5 import ValuationProductionStatus


class LowEntryV5FundamentalEvaluator:
    """Evaluate point-in-time quality and valuation evidence."""

    def __init__(
        self,
        quality_config: LowEntryV5QualityConfig,
        valuation_producer: LowEntryV5ValuationProducer,
        positive_float: Callable[[Any, float], float],
    ) -> None:
        self.quality_config = quality_config
        self.valuation_producer = valuation_producer
        self.positive_float = positive_float

    def evaluate_quality(
        self,
        data: pd.DataFrame,
        context: dict[str, Any],
    ) -> LowEntryV5QualityResult:
        annual = context.get(ANNUAL_FUNDAMENTALS_ATTR)
        annual = annual if isinstance(annual, pd.DataFrame) else pd.DataFrame()
        snapshot = context.get(FUNDAMENTAL_SNAPSHOT_ATTR)
        snapshot = snapshot if isinstance(snapshot, Mapping) else {}
        signal_date = context.get(SIGNAL_DATE_ATTR)
        if signal_date is None and "Date" in data and not data.empty:
            signal_date = data["Date"].iloc[-1]
        return evaluate_v5_quality(
            annual,
            snapshot,
            signal_date,
            self.quality_config,
        )

    def evaluate_valuation(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, LowEntryV5ValuationResult]:
        annual = context.get(ANNUAL_FUNDAMENTALS_ATTR)
        annual = annual if isinstance(annual, pd.DataFrame) else pd.DataFrame()
        snapshot = context.get(FUNDAMENTAL_SNAPSHOT_ATTR)
        snapshot = snapshot if isinstance(snapshot, Mapping) else {}
        price_scale = self.positive_float(
            context.get(PRICE_SCALE_ATTR),
            1.0,
        )
        current_price = indicators["close"].iloc[-1] / price_scale
        recent_pe = indicators["pe_median60"].iloc[-1]
        if pd.isna(recent_pe):
            recent_pe = indicators["pe"].iloc[-1]
        result = self.valuation_producer.produce(
            annual,
            snapshot,
            signal_date=context.get(SIGNAL_DATE_ATTR),
            current_price=current_price,
            recent_pe_median=recent_pe,
        )
        if result.status is ValuationProductionStatus.UNKNOWN:
            status = "UNKNOWN"
        elif result.gate_passed:
            status = "PASS"
        elif result.supportive_gate_passed:
            status = "SUPPORTIVE"
        else:
            status = "FAIL"
        return pd.Series(status, index=indicators.index), result


__all__ = ["LowEntryV5FundamentalEvaluator"]
