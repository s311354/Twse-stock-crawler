"""Holding-count lifecycle orchestration for PAE v2."""

from __future__ import annotations

from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioActionV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioAllocationLineV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioAllocationRequestV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioAllocationV2Config,
)
from domain.strategies.portfolio_allocation_v2_contracts import PortfolioModeV2
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioSecurityV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    SecurityEvaluationV2,
)
from domain.strategies.portfolio_allocation_v2_dominance_policy import (
    PortfolioDominancePolicyV2,
)
from domain.strategies.portfolio_allocation_v2_evaluator import (
    PortfolioSecurityEvaluatorV2,
)
from domain.strategies.portfolio_allocation_v2_exit_policy import (
    PortfolioExitPolicyV2,
)
from domain.strategies.portfolio_allocation_v2_weight_allocator import (
    PortfolioWeightAllocatorV2,
)


class PortfolioLifecyclePolicyV2:
    """Build decision lines for each normative v2 holding-count mode."""

    def __init__(
        self,
        config: PortfolioAllocationV2Config,
        evaluator: PortfolioSecurityEvaluatorV2,
        dominance: PortfolioDominancePolicyV2,
        exits: PortfolioExitPolicyV2,
        weights: PortfolioWeightAllocatorV2,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.dominance = dominance
        self.exits = exits
        self.weights = weights

    def build_lines(
        self,
        mode: PortfolioModeV2,
        request: PortfolioAllocationRequestV2,
        evaluations: dict[str, SecurityEvaluationV2],
        holding_lines: list[PortfolioAllocationLineV2],
    ) -> tuple[list[PortfolioAllocationLineV2], tuple[str, ...]]:
        if mode is PortfolioModeV2.POLICY_BREACH_REDUCE_ONLY:
            return (
                self.exits.breach_lines(holding_lines),
                ("POLICY_BREACH_HOLDINGS_GT_4",),
            )
        if mode is PortfolioModeV2.DIRECT_SELECTION:
            return (
                self.direct_selection_lines(
                    request,
                    evaluations,
                    holding_lines,
                ),
                ("PAE_NOT_TRIGGERED_HOLDINGS_LT_3",),
            )
        if mode is PortfolioModeV2.PAE_ACTIVE_OPEN_SLOT:
            return (
                self.open_slot_lines(request, evaluations, holding_lines),
                ("PAE_TRIGGERED_AT_3_HOLDINGS",),
            )
        return (
            self.full_portfolio_lines(request, evaluations, holding_lines),
            ("PAE_TRIGGERED_AT_4_HOLDINGS",),
        )

    def direct_selection_lines(
        self,
        request: PortfolioAllocationRequestV2,
        evaluations: dict[str, SecurityEvaluationV2],
        holding_lines: list[PortfolioAllocationLineV2],
    ) -> list[PortfolioAllocationLineV2]:
        lines = list(holding_lines)
        slots = max(0, 3 - len(request.holdings))
        cash_capacity = self.weights.cash_capacity(request)
        candidates = sorted(
            (
                item
                for item in request.candidates
                if self.evaluator.direct_selection_executable(item)
            ),
            key=lambda item: (
                {"BUY": 0, "EARLY_BUY": 1, "BUY_SMALL": 2}.get(
                    item.les_decision.upper(),
                    3,
                ),
                -float(item.quality_score or 0.0),
                item.stock_id,
            ),
        )
        selected: set[str] = set()
        for security in candidates:
            if slots <= 0 or cash_capacity <= self.config.epsilon:
                break
            evaluation = evaluations[security.stock_id]
            target, blocking = self.weights.target_weight(
                request,
                security,
                cash_capacity,
                excluded_holding=None,
            )
            if target <= self.config.epsilon:
                continue
            selected.add(security.stock_id)
            slots -= 1
            cash_capacity -= target
            lines.append(
                self.weights.candidate_line(
                    security,
                    evaluation,
                    PortfolioActionV2.STARTER,
                    target,
                    ("LES_DIRECT_SELECTION_NON_BINDING",),
                    blocking,
                    portfolio_improvement=None,
                )
            )
        lines.extend(
            self.weights.inactive_candidate_line(
                item,
                evaluations[item.stock_id],
            )
            for item in request.candidates
            if item.stock_id not in selected
        )
        return lines

    def open_slot_lines(
        self,
        request: PortfolioAllocationRequestV2,
        evaluations: dict[str, SecurityEvaluationV2],
        holding_lines: list[PortfolioAllocationLineV2],
    ) -> list[PortfolioAllocationLineV2]:
        lines = list(holding_lines)
        selected: PortfolioSecurityV2 | None = None
        selected_target = 0.0
        for security in self.weights.rank_candidates(
            request.candidates,
            evaluations,
        ):
            target, _ = self.weights.target_weight(
                request,
                security,
                self.weights.cash_capacity(request),
                excluded_holding=None,
            )
            if target <= self.config.epsilon:
                continue
            if not self.weights.portfolio_improves(
                request,
                evaluations,
                security,
                target,
                excluded_holding=None,
            ):
                continue
            selected = security
            selected_target = target
            break
        for security in request.candidates:
            evaluation = evaluations[security.stock_id]
            if selected is not None and security.stock_id == selected.stock_id:
                lines.append(
                    self.weights.candidate_line(
                        security,
                        evaluation,
                        PortfolioActionV2.STARTER,
                        selected_target,
                        ("OPEN_SLOT_ADMISSION_PASS",),
                        (),
                        portfolio_improvement=True,
                    )
                )
            else:
                lines.append(
                    self.weights.inactive_candidate_line(security, evaluation)
                )
        return lines

    def full_portfolio_lines(
        self,
        request: PortfolioAllocationRequestV2,
        evaluations: dict[str, SecurityEvaluationV2],
        holding_lines: list[PortfolioAllocationLineV2],
    ) -> list[PortfolioAllocationLineV2]:
        candidates = self.weights.rank_candidates(
            request.candidates,
            evaluations,
        )
        best: tuple[
            float,
            PortfolioSecurityV2,
            PortfolioSecurityV2,
            float,
            float,
            bool,
        ] | None = None
        for challenger in candidates:
            for incumbent in request.holdings:
                comparison = self.dominance.replacement_comparison(
                    request,
                    challenger,
                    incumbent,
                    evaluations,
                )
                if comparison is None:
                    continue
                excess, hurdle, fast_path = comparison
                target, _ = self.weights.target_weight(
                    request,
                    challenger,
                    self.weights.cash_capacity(request)
                    + incumbent.current_weight,
                    excluded_holding=incumbent,
                )
                if target <= self.config.epsilon:
                    continue
                if not self.weights.portfolio_improves(
                    request,
                    evaluations,
                    challenger,
                    target,
                    excluded_holding=incumbent,
                ):
                    continue
                if best is None or excess > best[0]:
                    best = (
                        excess,
                        challenger,
                        incumbent,
                        target,
                        hurdle,
                        fast_path,
                    )
        if best is None:
            return list(holding_lines) + [
                self.weights.inactive_candidate_line(
                    item,
                    evaluations[item.stock_id],
                )
                for item in request.candidates
            ]

        excess, challenger, incumbent, target, hurdle, fast_path = best
        atomic_id = self.dominance.atomic_group_id(
            request,
            incumbent,
            challenger,
        )
        lines: list[PortfolioAllocationLineV2] = []
        for line in holding_lines:
            if line.stock_id == incumbent.stock_id:
                lines.append(
                    self.exits.replace_line(
                        line,
                        action=PortfolioActionV2.EXIT_AND_REALLOCATE,
                        target_weight=0.0,
                        incumbent_stock_id=incumbent.stock_id,
                        atomic_group_id=atomic_id,
                        execution_status="CONDITIONAL",
                        reason_codes=("ATOMIC_REPLACEMENT_REQUIRED",),
                    )
                )
            else:
                lines.append(line)
        for security in request.candidates:
            evaluation = evaluations[security.stock_id]
            if security.stock_id == challenger.stock_id:
                action = (
                    PortfolioActionV2.FAST_TRACK_ROTATE
                    if fast_path
                    else PortfolioActionV2.ROTATE_75
                )
                lines.append(
                    PortfolioAllocationLineV2(
                        stock_id=security.stock_id,
                        role="CANDIDATE",
                        action=action,
                        current_weight=0.0,
                        target_weight=target,
                        evaluation=evaluation,
                        incumbent_stock_id=incumbent.stock_id,
                        switch_hurdle=hurdle,
                        excess_opportunity=excess,
                        quality_dominance=True,
                        cer_dominance=True,
                        persistence_pass=not fast_path,
                        fast_path_pass=fast_path,
                        portfolio_improvement=True,
                        atomic_group_id=atomic_id,
                        execution_status="CONDITIONAL",
                        reason_codes=(
                            "FAST_PATH_PASS"
                            if fast_path
                            else "DUAL_DOMINANCE_AND_PERSISTENCE_PASS",
                        ),
                    )
                )
            else:
                lines.append(
                    self.weights.inactive_candidate_line(security, evaluation)
                )
        return lines


__all__ = ["PortfolioLifecyclePolicyV2"]
