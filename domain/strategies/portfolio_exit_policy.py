"""Holding exit confirmation and target-reduction policy for PAE."""

from __future__ import annotations

from typing import Any

from domain.strategies.portfolio_allocation_contracts import AllocationSecurity
from domain.strategies.portfolio_allocation_contracts import (
    ExitConfirmationAssessment,
)
from domain.strategies.portfolio_allocation_contracts import PortfolioAction
from domain.strategies.portfolio_candidate_evaluator import SecurityEvaluation


class ExitPolicy:
    """Convert hard and confirmed soft exit evidence into holding actions."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def base_holding_action(
        self,
        holding: AllocationSecurity,
        evaluation: SecurityEvaluation,
        has_alternative: bool,
    ) -> tuple[PortfolioAction, str, ExitConfirmationAssessment]:
        assessment = self.exit_confirmation_assessment(holding, evaluation)
        if evaluation.hard_reject:
            return (
                PortfolioAction.EXIT_AND_REALLOCATE
                if has_alternative
                else PortfolioAction.EXIT_TO_CASH,
                "FUNDAMENTAL_HARD_REJECT",
                assessment,
            )
        if assessment.strategy_stop_breached:
            return (
                PortfolioAction.EXIT_AND_REALLOCATE
                if has_alternative
                else PortfolioAction.EXIT_TO_CASH,
                "STRATEGY_STOP_BREACHED",
                assessment,
            )
        if (
            evaluation.data_insufficient
            or not holding.erem_data_sufficient
            or holding.erem_holding_confidence is None
        ):
            return (
                PortfolioAction.HOLD_DATA_INSUFFICIENT,
                "MISSING_DATA_CANNOT_FORCE_EXIT",
                assessment,
            )
        confidence = holding.erem_holding_confidence
        if confidence < self.config.erem_exit_threshold:
            if (
                confidence < self.config.erem_severe_exit_threshold
                and assessment.trend_broken
            ):
                return (
                    PortfolioAction.EXIT_AND_REALLOCATE
                    if has_alternative
                    else PortfolioAction.EXIT_TO_CASH,
                    "SEVERE_EREM_AND_TREND_BREAK",
                    assessment,
                )
            if assessment.confirmation_count == 3:
                return (
                    PortfolioAction.EXIT_AND_REALLOCATE
                    if has_alternative
                    else PortfolioAction.EXIT_TO_CASH,
                    "CONFIRMED_EREM_EXIT",
                    assessment,
                )
            if assessment.confirmation_count >= 2:
                if assessment.high_quality_protected:
                    return (
                        PortfolioAction.REDUCE_25,
                        "QUALITY_PROTECTED_CONFIRMED_REDUCE",
                        assessment,
                    )
                return (
                    PortfolioAction.EXIT_AND_REALLOCATE
                    if has_alternative
                    else PortfolioAction.EXIT_TO_CASH,
                    "CONFIRMED_EREM_EXIT",
                    assessment,
                )
            return (
                PortfolioAction.HOLD_REVIEW,
                "EREM_EXIT_WARNING_AWAITING_CONFIRMATION",
                assessment,
            )
        if confidence < self.config.erem_reduce_threshold:
            if assessment.high_quality_protected and not assessment.trend_broken:
                return (
                    PortfolioAction.HOLD_NO_ADD,
                    "QUALITY_PROTECTED_HOLD_NO_ADD",
                    assessment,
                )
            return (
                PortfolioAction.REDUCE_25,
                "EREM_REDUCE_25",
                assessment,
            )
        return PortfolioAction.HOLD, "INCUMBENCY_PROTECTED", assessment

    def exit_confirmation_assessment(
        self,
        holding: AllocationSecurity,
        evaluation: SecurityEvaluation,
    ) -> ExitConfirmationAssessment:
        high_quality = bool(
            holding.quality_score is not None
            and holding.quality_score >= self.config.quality_protection_score
            and holding.data_confidence is not None
            and holding.data_confidence
            >= self.config.quality_protection_data_confidence
            and not evaluation.hard_reject
        )
        stop_breached = bool(
            holding.current_price is not None
            and holding.strategy_stop is not None
            and holding.strategy_stop > 0
            and holding.current_price <= holding.strategy_stop
        )
        persistent = bool(
            holding.erem_holding_confidence is not None
            and holding.erem_holding_confidence
            < self.config.erem_exit_threshold
            and holding.erem_exit_streak
            >= self.config.erem_exit_persistence_required
        )
        structure_weak = bool(
            holding.erem_distribution_warning is True
            or holding.erem_higher_low is False
        )
        trend_broken = bool(
            holding.current_price is not None
            and holding.erem_ema20 is not None
            and holding.erem_ema20_slope is not None
            and holding.current_price < holding.erem_ema20
            and holding.erem_ema20_slope < 0
            and structure_weak
        )
        valuation_broken = bool(
            holding.valuation_confidence is not None
            and holding.valuation_confidence
            >= self.config.valuation_confirmation_confidence
            and evaluation.cer is not None
            and evaluation.cer
            < evaluation.required_return
            - self.config.valuation_shortfall_confirmation
        )
        return ExitConfirmationAssessment(
            high_quality_protected=high_quality,
            strategy_stop_breached=stop_breached,
            erem_exit_persistent=persistent,
            trend_broken=trend_broken,
            valuation_broken=valuation_broken,
        )

    @staticmethod
    def base_target_weight(
        current_weight: float,
        action: PortfolioAction,
    ) -> float:
        if action in {PortfolioAction.REDUCE_25, PortfolioAction.ROTATE_25}:
            return current_weight * 0.75
        if action is PortfolioAction.ROTATE_50:
            return current_weight * 0.50
        if action in {
            PortfolioAction.ROTATE_75,
            PortfolioAction.FAST_TRACK_ROTATE,
        }:
            return current_weight * 0.25
        if action in {
            PortfolioAction.EXIT,
            PortfolioAction.EXIT_AND_REALLOCATE,
            PortfolioAction.EXIT_TO_CASH,
        }:
            return 0.0
        return current_weight


__all__ = ["ExitPolicy"]
