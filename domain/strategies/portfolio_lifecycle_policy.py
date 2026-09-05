"""COLD_START, BUILDING and STEADY_STATE lifecycle policy for PAE."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from domain.strategies.portfolio_allocation_contracts import (
    AllocationLineDecision,
)
from domain.strategies.portfolio_allocation_contracts import AllocationSecurity
from domain.strategies.portfolio_allocation_contracts import DominanceAssessment
from domain.strategies.portfolio_allocation_contracts import PortfolioAction
from domain.strategies.portfolio_allocation_contracts import (
    PortfolioAllocationRequest,
)
from domain.strategies.portfolio_allocation_contracts import PortfolioMode
from domain.strategies.portfolio_allocation_contracts import resolve_portfolio_mode
from domain.strategies.portfolio_candidate_evaluator import SecurityEvaluation
from domain.strategies.portfolio_dominance_policy import DominancePolicy
from domain.strategies.portfolio_exit_policy import ExitPolicy
from domain.strategies.portfolio_weight_allocator import WeightAllocator


class LifecyclePolicy:
    """Plan allocation lines according to the current holding-count mode."""

    def __init__(
        self,
        config: Any,
        dominance_policy: DominancePolicy,
        exit_policy: ExitPolicy,
        weight_allocator: WeightAllocator,
    ) -> None:
        self.config = config
        self.dominance_policy = dominance_policy
        self.exit_policy = exit_policy
        self.weight_allocator = weight_allocator

    @staticmethod
    def resolve(active_holding_count: int) -> PortfolioMode:
        return resolve_portfolio_mode(active_holding_count)

    def build_lines(
        self,
        mode: PortfolioMode,
        request: PortfolioAllocationRequest,
        evaluations: dict[str, SecurityEvaluation],
        executable: list[AllocationSecurity],
        dominance_by_candidate: dict[str, DominanceAssessment],
    ) -> list[AllocationLineDecision]:
        if mode is PortfolioMode.COLD_START:
            return self.cold_start_lines(
                request,
                evaluations,
                executable,
                dominance_by_candidate,
            )
        if mode is PortfolioMode.BUILDING:
            return self.building_lines(request, evaluations, executable)
        return self.steady_state_lines(
            request,
            evaluations,
            executable,
            dominance_by_candidate,
        )

    def cold_start_lines(
        self,
        request: PortfolioAllocationRequest,
        evaluations: dict[str, SecurityEvaluation],
        executable: list[AllocationSecurity],
        dominance_by_candidate: dict[str, DominanceAssessment],
    ) -> list[AllocationLineDecision]:
        ranked = self.weight_allocator.rank_by_capital_claim(
            executable,
            evaluations,
        )
        first = self.weight_allocator.first_candidate_with_allocation_room(
            request,
            [],
            ranked,
        )
        selected = {first.symbol} if first is not None else set()
        lines: list[AllocationLineDecision] = []
        for security in request.candidates:
            evaluation = evaluations[security.symbol]
            if security.symbol in selected:
                action = self.entry_action(security, evaluation)
                reason = "COLD_START_EXECUTABLE"
            elif evaluation.data_insufficient:
                action = PortfolioAction.DATA_INSUFFICIENT
                reason = "CANDIDATE_DATA_INSUFFICIENT"
            elif evaluation.strategic_candidate:
                action = PortfolioAction.STRATEGIC_WATCH
                reason = "STRATEGIC_NOT_EXECUTABLE_OR_SLOT_UNAVAILABLE"
            else:
                action = PortfolioAction.WATCH
                reason = "ABSOLUTE_HURDLE_OR_HARD_GATE_NOT_MET"
            line = AllocationLineDecision(
                symbol=security.symbol,
                role="CANDIDATE",
                action=action,
                evaluation=evaluation,
                current_weight=0.0,
                target_weight=0.0,
                reason=reason,
            )
            lines.append(
                self.dominance_policy.with_dominance(
                    line,
                    dominance_by_candidate.get(security.symbol),
                )
            )
        return lines

    def building_lines(
        self,
        request: PortfolioAllocationRequest,
        evaluations: dict[str, SecurityEvaluation],
        executable: list[AllocationSecurity],
    ) -> list[AllocationLineDecision]:
        lines: list[AllocationLineDecision] = []
        for holding in request.holdings:
            evaluation = evaluations[holding.symbol]
            current_weight = self.weight_allocator.current_weight(
                holding,
                request.account_equity,
            )
            action, reason, exit_assessment = (
                self.exit_policy.base_holding_action(
                    holding,
                    evaluation,
                    bool(executable),
                )
            )
            lines.append(
                AllocationLineDecision(
                    symbol=holding.symbol,
                    role="HOLDING",
                    action=action,
                    evaluation=evaluation,
                    current_weight=current_weight,
                    target_weight=self.exit_policy.base_target_weight(
                        current_weight,
                        action,
                    ),
                    high_quality_protected=(
                        exit_assessment.high_quality_protected
                    ),
                    strategy_stop_breached=(
                        exit_assessment.strategy_stop_breached
                    ),
                    erem_exit_persistent=(
                        exit_assessment.erem_exit_persistent
                    ),
                    trend_broken=exit_assessment.trend_broken,
                    valuation_broken=exit_assessment.valuation_broken,
                    exit_confirmation_count=(
                        exit_assessment.confirmation_count
                    ),
                    reason=reason,
                )
            )

        ranked = self.weight_allocator.rank_by_capital_claim(
            executable,
            evaluations,
        )
        selected = self.weight_allocator.first_candidate_with_allocation_room(
            request,
            lines,
            ranked,
        )
        portfolio_blocked = selected is None and bool(executable)
        for candidate in request.candidates:
            evaluation = evaluations[candidate.symbol]
            if selected is not None and candidate.symbol == selected.symbol:
                action = self.entry_action(candidate, evaluation)
                reason = "BUILDING_EXECUTABLE_SECOND_POSITION"
            elif evaluation.data_insufficient:
                action = PortfolioAction.DATA_INSUFFICIENT
                reason = "CANDIDATE_DATA_INSUFFICIENT"
            elif portfolio_blocked and evaluation.executable_candidate:
                action = PortfolioAction.WATCH
                reason = "BUILDING_PORTFOLIO_COMPLEMENTARITY_OR_RISK_LIMIT"
            elif evaluation.strategic_candidate:
                action = PortfolioAction.STRATEGIC_WATCH
                reason = "BUILDING_NOT_SELECTED_OR_PORTFOLIO_LIMIT"
            else:
                action = PortfolioAction.WATCH
                reason = "ABSOLUTE_HURDLE_OR_HARD_GATE_NOT_MET"
            line = AllocationLineDecision(
                symbol=candidate.symbol,
                role="CANDIDATE",
                action=action,
                evaluation=evaluation,
                current_weight=0.0,
                target_weight=0.0,
                reason=reason,
            )
            if portfolio_blocked and evaluation.executable_candidate:
                line = replace(line, portfolio_risk_gate="FAIL")
            lines.append(line)
        return lines

    def steady_state_lines(
        self,
        request: PortfolioAllocationRequest,
        evaluations: dict[str, SecurityEvaluation],
        executable: list[AllocationSecurity],
        dominance_by_candidate: dict[str, DominanceAssessment],
    ) -> list[AllocationLineDecision]:
        current_weights = {
            item.symbol: self.weight_allocator.current_weight(
                item,
                request.account_equity,
            )
            for item in request.holdings
        }
        lines: list[AllocationLineDecision] = []
        rotation_tests: list[tuple[float, str, AllocationLineDecision]] = []
        selected_candidates: set[str] = set()

        for holding in request.holdings:
            evaluation = evaluations[holding.symbol]
            current_weight = current_weights[holding.symbol]
            holding_challenger = (
                self.dominance_policy.best_challenger_for_holding(
                    holding,
                    evaluation,
                    executable,
                    evaluations,
                )
            )
            action, reason, exit_assessment = (
                self.exit_policy.base_holding_action(
                    holding,
                    evaluation,
                    holding_challenger is not None,
                )
            )
            line = AllocationLineDecision(
                symbol=holding.symbol,
                role="HOLDING",
                action=action,
                evaluation=evaluation,
                current_weight=current_weight,
                target_weight=self.exit_policy.base_target_weight(
                    current_weight,
                    action,
                ),
                high_quality_protected=exit_assessment.high_quality_protected,
                strategy_stop_breached=exit_assessment.strategy_stop_breached,
                erem_exit_persistent=exit_assessment.erem_exit_persistent,
                trend_broken=exit_assessment.trend_broken,
                valuation_broken=exit_assessment.valuation_broken,
                exit_confirmation_count=exit_assessment.confirmation_count,
                reason=reason,
            )
            if action in {
                PortfolioAction.EXIT,
                PortfolioAction.EXIT_AND_REALLOCATE,
                PortfolioAction.EXIT_TO_CASH,
            } and holding_challenger is not None:
                challenger, assessment = holding_challenger
                selected_candidates.add(challenger.symbol)
                line = replace(
                    self.dominance_policy.with_dominance(line, assessment),
                    alternative_symbol=challenger.symbol,
                    alternative_cer=evaluations[challenger.symbol].cer,
                    opportunity_gap=assessment.opportunity_gap,
                    switch_hurdle=assessment.switch_hurdle,
                    replacement_threshold=assessment.replacement_threshold,
                    excess_opportunity=assessment.excess_opportunity,
                )

            if (
                action in {PortfolioAction.HOLD, PortfolioAction.HOLD_NO_ADD}
                and holding_challenger is not None
                and evaluation.cer is not None
            ):
                challenger, assessment = holding_challenger
                tested = self.dominance_policy.replacement_test(
                    holding,
                    evaluation,
                    challenger,
                    evaluations[challenger.symbol],
                    request,
                    assessment,
                )
                if tested.excess_opportunity is not None:
                    seat_defense = tested.replacement_threshold or float("inf")
                    rotation_tests.append(
                        (seat_defense, holding.symbol, tested)
                    )
                line = tested
            lines.append(line)

        active_after_forced_exits = sum(
            line.target_weight > self.config.epsilon for line in lines
        )
        available_slots = (
            self.config.max_stocks
            - active_after_forced_exits
            - len(selected_candidates)
        )
        qualified_rotations = [
            item
            for item in rotation_tests
            if item[2].action
            in {
                PortfolioAction.ROTATE_25,
                PortfolioAction.ROTATE_50,
                PortfolioAction.ROTATE_75,
                PortfolioAction.FAST_TRACK_ROTATE,
            }
        ]
        if qualified_rotations and available_slots >= 0:
            _, symbol, chosen = sorted(
                qualified_rotations,
                key=lambda item: (
                    item[0],
                    -float(item[2].quality_advantage or 0.0),
                    -float(item[2].excess_opportunity or 0.0),
                    item[1],
                ),
            )[0]
            if available_slots == 0:
                chosen = replace(
                    chosen,
                    target_weight=0.0,
                    reason=chosen.reason + ";FULL_PORTFOLIO_ATOMIC_ROTATION",
                )
            lines = [
                chosen
                if line.symbol == symbol
                else self.hold_rotation_candidate(line)
                for line in lines
            ]
            if chosen.alternative_symbol:
                selected_candidates.add(chosen.alternative_symbol)
            available_slots -= 1
        else:
            lines = [self.hold_rotation_candidate(line) for line in lines]

        lines = [self.resolve_incumbent_add(line, request) for line in lines]

        for candidate in self.weight_allocator.rank_by_capital_claim(
            executable,
            evaluations,
        ):
            if available_slots <= 0:
                break
            if candidate.symbol not in selected_candidates:
                selected_candidates.add(candidate.symbol)
                available_slots -= 1

        for candidate in request.candidates:
            evaluation = evaluations[candidate.symbol]
            if candidate.symbol in selected_candidates:
                action = self.entry_action(candidate, evaluation)
                reason = "EXECUTABLE_CAPITAL_RECIPIENT"
            elif evaluation.data_insufficient:
                action = PortfolioAction.DATA_INSUFFICIENT
                reason = "CANDIDATE_DATA_INSUFFICIENT"
            elif evaluation.strategic_candidate:
                action = PortfolioAction.STRATEGIC_WATCH
                reason = "STRATEGIC_NOT_SELECTED"
            elif evaluation.absolute_strategic_candidate:
                action = PortfolioAction.WATCH
                reason = "QUALITY_AND_CER_DUAL_DOMINANCE_NOT_MET"
            else:
                action = PortfolioAction.WATCH
                reason = "ABSOLUTE_HURDLE_OR_HARD_GATE_NOT_MET"
            line = AllocationLineDecision(
                symbol=candidate.symbol,
                role="CANDIDATE",
                action=action,
                evaluation=evaluation,
                current_weight=0.0,
                target_weight=0.0,
                reason=reason,
            )
            lines.append(
                self.dominance_policy.with_dominance(
                    line,
                    dominance_by_candidate.get(candidate.symbol),
                )
            )
        return lines

    def resolve_incumbent_add(
        self,
        line: AllocationLineDecision,
        request: PortfolioAllocationRequest,
    ) -> AllocationLineDecision:
        if line.role != "HOLDING" or line.action is not PortfolioAction.HOLD:
            return line
        holding = next(
            item for item in request.holdings if item.symbol == line.symbol
        )
        if line.evaluation.executable_candidate:
            action = self.entry_action(holding, line.evaluation)
            return replace(
                line,
                action=action,
                reason=line.reason + ";INCUMBENT_ELIGIBLE_TO_ADD",
            )
        return replace(
            line,
            action=PortfolioAction.HOLD_NO_ADD,
            reason=line.reason + ";WEAK_LES_IS_NOT_A_SELL_SIGNAL",
        )

    def entry_action(
        self,
        security: AllocationSecurity,
        evaluation: SecurityEvaluation,
    ) -> PortfolioAction:
        """Map an executable candidate to a size-aware entry action."""
        if security.les_decision.upper() == "BUY_SMALL":
            return PortfolioAction.STARTER
        if (
            float(security.quality_score or 0.0)
            >= self.config.obvious_quality_score
            and evaluation.opportunity_edge
            >= self.config.obvious_required_return_spread
        ):
            return PortfolioAction.STRONG_ADD
        return PortfolioAction.ADD

    @staticmethod
    def hold_rotation_candidate(
        line: AllocationLineDecision,
    ) -> AllocationLineDecision:
        if line.action not in {
            PortfolioAction.ROTATE_25,
            PortfolioAction.ROTATE_50,
            PortfolioAction.ROTATE_75,
            PortfolioAction.FAST_TRACK_ROTATE,
        }:
            return line
        return replace(
            line,
            action=PortfolioAction.HOLD,
            target_weight=line.current_weight,
            portfolio_risk_gate="FAIL",
            reason="PORTFOLIO_SLOT_LIMIT_OR_STRONGER_SEAT_SELECTED",
        )


__all__ = ["LifecyclePolicy"]
