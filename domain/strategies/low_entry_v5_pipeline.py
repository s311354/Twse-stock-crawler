"""Composable decision pipeline contracts for Low Entry Model v5."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import Protocol

import pandas as pd

from domain.features import FeatureFrame
from domain.features import SharedFeatureConfig
from domain.features import SharedFeaturePipeline
from domain.quality_v5 import LowEntryV5QualityResult
from domain.security_identity import ISSUER_SYMBOL_FIELD
from domain.security_identity import SECURITY_TYPE_FIELD
from domain.security_identity import SecurityType
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_SNAPSHOT_ATTR
from domain.valuation_v5 import LowEntryV5ValuationResult


class MarketComponent(Protocol):
    def shared_feature_config(self) -> SharedFeatureConfig: ...

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame: ...

    def evaluate_market_data(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series]: ...

    def build_trade_plan(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame: ...

    def build_broad_rally_plan(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame: ...

    def evaluate_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]: ...

    def evaluate_broad_rally_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]: ...


class FundamentalComponent(Protocol):
    def evaluate_quality(
        self,
        data: pd.DataFrame,
        context: dict[str, Any],
    ) -> LowEntryV5QualityResult: ...

    def evaluate_valuation(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, LowEntryV5ValuationResult]: ...


class PanicComponent(Protocol):
    def evaluate_panic(
        self,
        indicators: pd.DataFrame,
        quality: LowEntryV5QualityResult,
        valuation: pd.Series,
    ) -> dict[str, pd.Series]: ...


class ScoringComponent(Protocol):
    def score_components(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        valuation: pd.Series,
    ) -> pd.DataFrame: ...

    def falling_knife_penalty(
        self,
        indicators: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]: ...

    def required_score(
        self,
        quality: LowEntryV5QualityResult,
        risk_off: pd.Series,
        index: pd.Index,
    ) -> pd.Series: ...


class DecisionComponent(Protocol):
    def evaluate_trigger(self, indicators: pd.DataFrame) -> pd.Series: ...

    def next_day_confirmation(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        falling_state: pd.Series,
    ) -> pd.Series: ...

    def resolve_decision(
        self,
        *,
        score: pd.Series,
        quality: LowEntryV5QualityResult,
        valuation: pd.Series,
        falling_state: pd.Series,
        next_day_confirmation: pd.Series,
        trigger: pd.Series,
        risk_gate: pd.Series,
        market_gate: pd.Series,
        required_score: pd.Series,
        broad_rally_trigger: pd.Series,
        broad_rally_risk_gate: pd.Series,
        raw_score: pd.Series,
        broad_rally_full_buy_trigger: pd.Series | None = None,
        broad_rally_absorption_trigger: pd.Series | None = None,
        multiday_panic_reversal_trigger: pd.Series | None = None,
        recent_rally_retest_trigger: pd.Series | None = None,
        recent_rally_reclaim_trigger: pd.Series | None = None,
        quality_pullback_trigger: pd.Series | None = None,
        broad_rally_active: pd.Series | None = None,
    ) -> pd.Series: ...

    def position_fraction(
        self,
        decision: pd.Series,
        quality: LowEntryV5QualityResult,
        risk_off: pd.Series,
        broad_rally_full_buy: pd.Series | None = None,
    ) -> pd.Series: ...

    def cap_position_shares(
        self,
        plan: pd.DataFrame,
        position_fraction: pd.Series,
        context: dict[str, Any],
    ) -> pd.Series: ...

    def enforce_executable_position(
        self,
        decision: pd.Series,
        recommended_shares: pd.Series,
    ) -> pd.Series: ...


class ReportComponent(Protocol):
    def empty_result(self) -> dict[str, object]: ...

    def assemble_result(
        self,
        *,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        market_gate: pd.Series,
        market_reasons: pd.Series,
        plan: pd.DataFrame,
        quality: LowEntryV5QualityResult,
        valuation: pd.Series,
        valuation_result: LowEntryV5ValuationResult,
        panic: dict[str, pd.Series],
        components: pd.DataFrame,
        penalty: pd.Series,
        risk_off: pd.Series,
        required_score: pd.Series,
        trigger: pd.Series,
        risk_gate: pd.Series,
        risk_reasons: pd.Series,
        broad_rally_risk_gate: pd.Series,
        broad_rally_risk_reasons: pd.Series,
        next_day_confirmation: pd.Series,
        decision: pd.Series,
        position_fraction: pd.Series,
        recommended_shares: pd.Series,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class LowEntryV5PipelineComponents:
    """Replaceable policies used by one v5 orchestration pipeline."""

    market: MarketComponent
    fundamentals: FundamentalComponent
    panic: PanicComponent
    scoring: ScoringComponent
    decision: DecisionComponent
    report: ReportComponent


class LowEntryV5DecisionPipeline:
    """Arrange model components without owning any trading rule."""

    def __init__(self, components: LowEntryV5PipelineComponents) -> None:
        self.components = components

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        features = SharedFeaturePipeline(
            self.components.market.shared_feature_config()
        ).build(data)
        return self.run_with_features(data, features)

    def run_with_features(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame([self.components.report.empty_result()])

        context = dict(data.attrs)
        snapshot = context.get(FUNDAMENTAL_SNAPSHOT_ATTR)
        snapshot = snapshot if isinstance(snapshot, Mapping) else {}
        security_type = str(
            snapshot.get(SECURITY_TYPE_FIELD) or SecurityType.COMMON.value
        ).upper()
        issuer_symbol = str(snapshot.get(ISSUER_SYMBOL_FIELD) or "N/A")
        security_supported = security_type == SecurityType.COMMON.value
        frame = data.copy()
        indicators = self.components.market.calculate_indicators(frame, features)
        market_gate, market_reasons = (
            self.components.market.evaluate_market_data(
                frame,
                indicators,
                context,
            )
        )
        plan = self.components.market.build_trade_plan(indicators, context)
        broad_rally_plan = self.components.market.build_broad_rally_plan(
            indicators,
            plan,
            context,
        )
        quality = self.components.fundamentals.evaluate_quality(frame, context)
        valuation, valuation_result = (
            self.components.fundamentals.evaluate_valuation(
                indicators,
                context,
            )
        )
        panic = self.components.panic.evaluate_panic(
            indicators,
            quality,
            valuation,
        )
        score_components = self.components.scoring.score_components(
            indicators,
            plan,
            valuation,
        )
        penalty, risk_off = self.components.scoring.falling_knife_penalty(
            indicators
        )
        score = (score_components.sum(axis=1) + penalty).clip(0, 100)
        raw_score = score_components.sum(axis=1)
        required_score = self.components.scoring.required_score(
            quality,
            risk_off,
            frame.index,
        )
        trigger = self.components.decision.evaluate_trigger(indicators)
        risk_gate, risk_reasons = self.components.market.evaluate_risk_gate(
            indicators,
            plan,
            market_gate,
        )
        broad_rally_risk_gate, broad_rally_risk_reasons = (
            self.components.market.evaluate_broad_rally_risk_gate(
                indicators,
                broad_rally_plan,
                market_gate,
            )
        )
        next_day_confirmation = self.components.decision.next_day_confirmation(
            indicators,
            plan,
            panic["state"],
        )
        decision = self.components.decision.resolve_decision(
            score=score,
            quality=quality,
            valuation=valuation,
            falling_state=panic["state"],
            next_day_confirmation=next_day_confirmation,
            trigger=trigger,
            risk_gate=risk_gate,
            market_gate=market_gate,
            required_score=required_score,
            broad_rally_trigger=panic["broad_rally_trigger"],
            broad_rally_risk_gate=broad_rally_risk_gate,
            raw_score=raw_score,
            broad_rally_full_buy_trigger=panic[
                "broad_rally_full_buy_trigger"
            ],
            broad_rally_absorption_trigger=panic[
                "broad_rally_absorption_trigger"
            ],
            multiday_panic_reversal_trigger=panic[
                "multiday_panic_reversal_trigger"
            ],
            recent_rally_retest_trigger=panic[
                "recent_rally_retest_trigger"
            ],
            recent_rally_reclaim_trigger=panic[
                "recent_rally_reclaim_trigger"
            ],
            quality_pullback_trigger=panic[
                "quality_pullback_trigger"
            ],
            broad_rally_active=panic["broad_rally_active"],
        )
        if not security_supported:
            decision[:] = "NO_MODEL"
        broad_full_selected = (
            (decision == "BUY")
            & panic["broad_rally_full_buy_trigger"]
        )
        broad_selected = (decision == "BUY_SMALL") | broad_full_selected
        output_plan = plan.copy()
        for column in output_plan.columns:
            output_plan[column] = output_plan[column].where(
                ~broad_selected,
                broad_rally_plan[column],
            )
        output_risk_gate = risk_gate.where(
            ~broad_selected,
            broad_rally_risk_gate,
        )
        output_risk_reasons = risk_reasons.where(
            ~broad_selected,
            broad_rally_risk_reasons,
        )
        position_fraction = self.components.decision.position_fraction(
            decision,
            quality,
            risk_off,
            broad_full_selected,
        )
        recommended_shares = self.components.decision.cap_position_shares(
            output_plan,
            position_fraction,
            context,
        )
        decision = self.components.decision.enforce_executable_position(
            decision,
            recommended_shares,
        )
        position_fraction = self.components.decision.position_fraction(
            decision,
            quality,
            risk_off,
            broad_full_selected,
        )
        recommended_shares = self.components.decision.cap_position_shares(
            output_plan,
            position_fraction,
            context,
        )

        result = self.components.report.assemble_result(
            data=frame,
            indicators=indicators,
            market_gate=market_gate,
            market_reasons=market_reasons,
            plan=output_plan,
            quality=quality,
            valuation=valuation,
            valuation_result=valuation_result,
            panic=panic,
            components=score_components,
            penalty=penalty,
            risk_off=risk_off,
            required_score=required_score,
            trigger=trigger,
            risk_gate=output_risk_gate,
            risk_reasons=output_risk_reasons,
            broad_rally_risk_gate=broad_rally_risk_gate,
            broad_rally_risk_reasons=broad_rally_risk_reasons,
            next_day_confirmation=next_day_confirmation,
            decision=decision,
            position_fraction=position_fraction,
            recommended_shares=recommended_shares,
        )
        result["Security Type"] = security_type
        result["Issuer Symbol"] = issuer_symbol
        if not security_supported:
            result["低點理由"] = (
                "SecurityType={}；Issuer={}；"
                "MODEL_UNSUPPORTED_FOR_SECURITY_TYPE；".format(
                    security_type,
                    issuer_symbol,
                )
                + result["低點理由"].astype(str)
            )
        return result


__all__ = [
    "DecisionComponent",
    "FundamentalComponent",
    "LowEntryV5DecisionPipeline",
    "LowEntryV5PipelineComponents",
    "MarketComponent",
    "PanicComponent",
    "ReportComponent",
    "ScoringComponent",
]
