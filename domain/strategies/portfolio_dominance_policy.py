"""Pairwise quality/CER dominance and replacement policy for PAE."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from domain.strategies.portfolio_allocation_contracts import (
    AllocationLineDecision,
)
from domain.strategies.portfolio_allocation_contracts import AllocationSecurity
from domain.strategies.portfolio_allocation_contracts import DominanceAssessment
from domain.strategies.portfolio_allocation_contracts import PersistenceEvidence
from domain.strategies.portfolio_allocation_contracts import PortfolioAction
from domain.strategies.portfolio_allocation_contracts import (
    PortfolioAllocationRequest,
)
from domain.strategies.portfolio_candidate_evaluator import SecurityEvaluation
from domain.strategies.portfolio_exit_policy import ExitPolicy
from domain.strategies.portfolio_weight_allocator import WeightAllocator


class DominancePolicy:
    """Require challengers to beat an incumbent on both quality and CER."""

    def __init__(
        self,
        config: Any,
        exit_policy: ExitPolicy,
        weight_allocator: WeightAllocator,
    ) -> None:
        self.config = config
        self.exit_policy = exit_policy
        self.weight_allocator = weight_allocator

    def apply_dual_dominance(
        self,
        request: PortfolioAllocationRequest,
        evaluations: dict[str, SecurityEvaluation],
    ) -> tuple[
        dict[str, SecurityEvaluation],
        dict[str, DominanceAssessment],
    ]:
        updated = dict(evaluations)
        best_assessment: dict[str, DominanceAssessment] = {}
        for challenger in request.candidates:
            evaluation = evaluations[challenger.symbol]
            if not evaluation.absolute_strategic_candidate:
                updated[challenger.symbol] = replace(
                    evaluation,
                    strategic_candidate=False,
                    executable_candidate=False,
                    dual_dominance_candidate=False,
                    capital_claim=0.0,
                )
                continue
            assessments = [
                self.dominance_assessment(
                    holding,
                    evaluations[holding.symbol],
                    challenger,
                    evaluation,
                )
                for holding in request.holdings
            ]
            passing = [item for item in assessments if item.dual_gate == "PASS"]
            evidence_pool = passing or assessments
            if evidence_pool:
                best_assessment[challenger.symbol] = sorted(
                    evidence_pool,
                    key=self.dominance_evidence_rank,
                )[0]
            dual_pass = bool(passing)
            updated[challenger.symbol] = replace(
                evaluation,
                strategic_candidate=dual_pass,
                executable_candidate=(
                    dual_pass and evaluation.executable_candidate
                ),
                dual_dominance_candidate=dual_pass,
                capital_claim=(evaluation.capital_claim if dual_pass else 0.0),
                reasons=(
                    evaluation.reasons
                    if dual_pass
                    else evaluation.reasons
                    + ("QUALITY_AND_CER_DUAL_DOMINANCE_NOT_MET",)
                ),
            )
        return updated, best_assessment

    @staticmethod
    def dominance_evidence_rank(
        assessment: DominanceAssessment,
    ) -> tuple[float, float, str]:
        quality_surplus = (
            assessment.quality_advantage
            - assessment.required_quality_advantage
            if assessment.quality_advantage is not None
            else float("-inf")
        )
        cer_surplus = (
            assessment.excess_opportunity
            if assessment.excess_opportunity is not None
            else float("-inf")
        )
        return (
            -min(quality_surplus, cer_surplus * 100),
            -cer_surplus,
            assessment.incumbent_symbol,
        )

    def best_challenger_for_holding(
        self,
        holding: AllocationSecurity,
        current: SecurityEvaluation,
        executable: list[AllocationSecurity],
        evaluations: dict[str, SecurityEvaluation],
    ) -> tuple[AllocationSecurity, DominanceAssessment] | None:
        passing: list[tuple[AllocationSecurity, DominanceAssessment]] = []
        for challenger in executable:
            alternative = evaluations[challenger.symbol]
            assessment = self.dominance_assessment(
                holding,
                current,
                challenger,
                alternative,
            )
            if assessment.dual_gate == "PASS":
                passing.append((challenger, assessment))
        if not passing:
            return None
        return sorted(
            passing,
            key=lambda item: (
                -float(item[0].quality_score or 0.0),
                -float(item[1].excess_opportunity or 0.0),
                item[0].symbol,
            ),
        )[0]

    def dominance_assessment(
        self,
        holding: AllocationSecurity,
        current: SecurityEvaluation,
        challenger: AllocationSecurity,
        alternative: SecurityEvaluation,
    ) -> DominanceAssessment:
        incumbent_quality = holding.quality_score
        challenger_quality = challenger.quality_score
        quality_advantage = (
            float(challenger_quality) - float(incumbent_quality)
            if challenger_quality is not None and incumbent_quality is not None
            else None
        )
        if challenger_quality is None:
            quality_gate = "UNKNOWN"
        elif challenger_quality < self.config.minimum_candidate_quality_score:
            quality_gate = "FAIL"
        elif current.hard_reject:
            quality_gate = "PASS"
        elif incumbent_quality is None:
            quality_gate = "UNKNOWN"
        elif quality_advantage is not None and quality_advantage >= (
            self.config.minimum_quality_advantage - self.config.epsilon
        ):
            quality_gate = "PASS"
        else:
            quality_gate = "FAIL"

        switch_hurdle = self.switch_hurdle(holding, challenger)
        opportunity_gap: float | None = None
        replacement_threshold: float | None = None
        excess: float | None = None
        if current.cer is not None and alternative.cer is not None:
            opportunity_gap = float(alternative.cer) - float(current.cer)
            replacement_threshold = float(current.cer) + switch_hurdle
            excess = opportunity_gap - switch_hurdle
            cer_gate = "PASS" if excess > self.config.epsilon else "FAIL"
        else:
            cer_gate = "UNKNOWN"

        if quality_gate == "PASS" and cer_gate == "PASS":
            dual_gate = "PASS"
        elif "FAIL" in {quality_gate, cer_gate}:
            dual_gate = "FAIL"
        else:
            dual_gate = "UNKNOWN"
        return DominanceAssessment(
            incumbent_symbol=holding.symbol,
            challenger_symbol=challenger.symbol,
            incumbent_quality_score=incumbent_quality,
            challenger_quality_score=challenger_quality,
            quality_advantage=quality_advantage,
            required_quality_advantage=self.config.minimum_quality_advantage,
            quality_gate=quality_gate,
            opportunity_gap=opportunity_gap,
            switch_hurdle=switch_hurdle,
            replacement_threshold=replacement_threshold,
            excess_opportunity=excess,
            cer_gate=cer_gate,
            dual_gate=dual_gate,
        )

    def switch_hurdle(
        self,
        holding: AllocationSecurity,
        challenger: AllocationSecurity,
    ) -> float:
        confidence = float(holding.erem_holding_confidence or 0)
        transaction_friction = (
            2 * self.config.commission_rate_each_side
            + self.config.transaction_tax_rate
            + self.config.spread_cost_rate
            + self.config.slippage_rate
        ) / self.config.expected_holding_years
        confidence_difference = max(
            float(holding.data_confidence or 0)
            - float(challenger.data_confidence or 0),
            0.0,
        )
        model_error = (
            self.config.maximum_model_error_buffer
            * confidence_difference
            / 100
        )
        return (
            self.config.base_replacement_buffer
            + transaction_friction
            + model_error
            + self.incumbency_premium(confidence)
        )

    @staticmethod
    def with_dominance(
        line: AllocationLineDecision,
        assessment: DominanceAssessment | None,
    ) -> AllocationLineDecision:
        if assessment is None:
            return line
        return replace(
            line,
            dominance_incumbent_symbol=assessment.incumbent_symbol,
            incumbent_quality_score=assessment.incumbent_quality_score,
            candidate_quality_score=assessment.challenger_quality_score,
            quality_advantage=assessment.quality_advantage,
            required_quality_advantage=assessment.required_quality_advantage,
            quality_dominance_gate=assessment.quality_gate,
            cer_dominance_gate=assessment.cer_gate,
            dual_dominance_gate=assessment.dual_gate,
        )

    def replacement_test(
        self,
        holding: AllocationSecurity,
        current: SecurityEvaluation,
        challenger: AllocationSecurity,
        alternative: SecurityEvaluation,
        request: PortfolioAllocationRequest,
        assessment: DominanceAssessment,
    ) -> AllocationLineDecision:
        switch_hurdle = float(assessment.switch_hurdle or 0.0)
        opportunity_gap = float(assessment.opportunity_gap or 0.0)
        excess = float(assessment.excess_opportunity or 0.0)
        persistence = self.persistence_passed(
            request.persistence,
            holding.symbol,
            challenger.symbol,
        )
        obvious = self.obvious_opportunity(challenger, alternative, excess)
        if assessment.quality_gate != "PASS":
            action = PortfolioAction.HOLD
            reason = "CHALLENGER_LACKS_REQUIRED_QUALITY_ADVANTAGE"
        elif assessment.cer_gate != "PASS":
            action = PortfolioAction.HOLD
            reason = "CHALLENGER_BELOW_SWITCH_HURDLE"
        elif not (persistence or obvious):
            action = PortfolioAction.HOLD
            reason = "PERSISTENCE_NOT_CONFIRMED"
        elif obvious:
            action = PortfolioAction.FAST_TRACK_ROTATE
            reason = "OBVIOUS_OPPORTUNITY_FAST_PATH"
        elif excess > 0.05:
            action = PortfolioAction.ROTATE_75
            reason = "PERSISTENT_EXCESS_OPPORTUNITY_ABOVE_5PP"
        elif excess > 0.02:
            action = PortfolioAction.ROTATE_50
            reason = "PERSISTENT_EXCESS_OPPORTUNITY_ABOVE_2PP"
        else:
            action = PortfolioAction.ROTATE_25
            reason = "PERSISTENT_POSITIVE_EXCESS_OPPORTUNITY"
        current_weight = self.weight_allocator.current_weight(
            holding,
            request.account_equity,
        )
        return self.with_dominance(
            AllocationLineDecision(
                symbol=holding.symbol,
                role="HOLDING",
                action=action,
                evaluation=current,
                current_weight=current_weight,
                target_weight=self.exit_policy.base_target_weight(
                    current_weight,
                    action,
                ),
                alternative_symbol=challenger.symbol,
                alternative_cer=alternative.cer,
                opportunity_gap=opportunity_gap,
                switch_hurdle=switch_hurdle,
                replacement_threshold=assessment.replacement_threshold,
                excess_opportunity=excess,
                persistence_passed=persistence,
                obvious_opportunity=obvious,
                reason=reason,
            ),
            assessment,
        )

    def persistence_passed(
        self,
        persistence: tuple[PersistenceEvidence, ...],
        incumbent: str,
        challenger: str,
    ) -> bool:
        for evidence in persistence:
            if (
                evidence.incumbent_symbol == incumbent
                and evidence.challenger_symbol == challenger
            ):
                values = evidence.recent_excess_opportunities[
                    -self.config.persistence_sessions :
                ]
                return (
                    len(values) >= self.config.persistence_sessions
                    and sum(value > 0 for value in values)
                    >= self.config.persistence_required
                )
        return False

    def obvious_opportunity(
        self,
        candidate: AllocationSecurity,
        evaluation: SecurityEvaluation,
        excess: float,
    ) -> bool:
        return bool(
            candidate.quality_score is not None
            and candidate.quality_score >= self.config.obvious_quality_score
            and candidate.data_confidence is not None
            and candidate.data_confidence >= self.config.obvious_data_confidence
            and candidate.valuation_confidence is not None
            and candidate.valuation_confidence
            >= self.config.obvious_valuation_confidence
            and evaluation.cer is not None
            and evaluation.cer
            >= evaluation.required_return
            + self.config.obvious_required_return_spread
            and candidate.les_decision.upper() == "BUY"
            and candidate.risk_gate.upper() == "PASS"
            and candidate.falling_knife_state.upper() != "HARD_NO_BUY"
            and excess >= self.config.obvious_excess_opportunity
        )

    def incumbency_premium(self, confidence: float) -> float:
        if confidence >= 85:
            return self.config.strong_hold_incumbency_premium
        if confidence >= 70:
            return self.config.upper_hold_incumbency_premium
        if confidence >= 55:
            return self.config.lower_hold_incumbency_premium
        if confidence >= 40:
            return self.config.reduce_incumbency_premium
        return 0.0


__all__ = ["DominancePolicy"]
