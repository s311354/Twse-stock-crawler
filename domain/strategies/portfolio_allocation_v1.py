"""Portfolio Allocation Engine v1 strategy plugin.

The aggregate ``allocate`` API is authoritative. ``run`` is a compatibility
adapter for the existing DataFrame-based Strategy plugin boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from domain.strategy import Strategy
from domain.strategies.portfolio_candidate_evaluator import (
    PortfolioCandidateEvaluator,
)
from domain.strategies.portfolio_candidate_evaluator import SecurityEvaluation

if TYPE_CHECKING:
    import pandas as pd


from domain.strategies.portfolio_allocation_contracts import (
    AllocationLineDecision,
)
from domain.strategies.portfolio_allocation_contracts import AllocationSecurity
from domain.strategies.portfolio_allocation_contracts import DominanceAssessment
from domain.strategies.portfolio_allocation_contracts import (
    ExitConfirmationAssessment,
)
from domain.strategies.portfolio_allocation_contracts import PAE_V1_MODEL_VERSION
from domain.strategies.portfolio_allocation_contracts import (
    PORTFOLIO_ALLOCATION_OUTPUT_COLUMNS,
)
from domain.strategies.portfolio_allocation_contracts import (
    PORTFOLIO_ALLOCATION_STRATEGY_NAME,
)
from domain.strategies.portfolio_allocation_contracts import PersistenceEvidence
from domain.strategies.portfolio_allocation_contracts import PortfolioAction
from domain.strategies.portfolio_allocation_contracts import (
    PortfolioAllocationConfig,
)
from domain.strategies.portfolio_allocation_contracts import (
    PortfolioAllocationDecision,
)
from domain.strategies.portfolio_allocation_contracts import (
    PortfolioAllocationRequest,
)
from domain.strategies.portfolio_allocation_contracts import PortfolioMode
from domain.strategies.portfolio_allocation_contracts import resolve_portfolio_mode
from domain.strategies.portfolio_dominance_policy import DominancePolicy
from domain.strategies.portfolio_exit_policy import ExitPolicy
from domain.strategies.portfolio_lifecycle_policy import LifecyclePolicy
from domain.strategies.portfolio_weight_allocator import WeightAllocator

class PortfolioAllocationEngine:
    """Orchestrate independent PAE policies behind the stable aggregate API."""

    def __init__(
        self,
        config: PortfolioAllocationConfig | None = None,
        *,
        candidate_evaluator: PortfolioCandidateEvaluator | None = None,
        lifecycle_policy: LifecyclePolicy | None = None,
        dominance_policy: DominancePolicy | None = None,
        exit_policy: ExitPolicy | None = None,
        weight_allocator: WeightAllocator | None = None,
    ) -> None:
        self.config = config or PortfolioAllocationConfig()
        self.candidate_evaluator = candidate_evaluator or PortfolioCandidateEvaluator(
            self.config
        )
        self.weight_allocator = weight_allocator or WeightAllocator(self.config)
        self.exit_policy = exit_policy or ExitPolicy(self.config)
        self.dominance_policy = dominance_policy or DominancePolicy(
            self.config,
            self.exit_policy,
            self.weight_allocator,
        )
        self.lifecycle_policy = lifecycle_policy or LifecyclePolicy(
            self.config,
            self.dominance_policy,
            self.exit_policy,
            self.weight_allocator,
        )

    def allocate(
        self,
        request: PortfolioAllocationRequest,
    ) -> PortfolioAllocationDecision:
        if len(request.holdings) > self.config.max_stocks:
            raise ValueError("Current holdings exceed max_stocks")
        current_total = request.current_cash / request.account_equity + sum(
            self.weight_allocator.current_weight(
                item,
                request.account_equity,
            )
            for item in request.holdings
        )
        if not np.isclose(
            current_total,
            1.0,
            atol=self.config.portfolio_balance_tolerance,
        ):
            raise ValueError(
                "Current holdings plus cash must reconcile to account_equity"
            )

        mode = self.lifecycle_policy.resolve(len(request.holdings))
        required_return = self.candidate_evaluator.required_return(request)
        securities = request.holdings + request.candidates
        evaluations = {
            item.symbol: self.candidate_evaluator.evaluate(
                item,
                required_return,
            )
            for item in securities
        }
        dominance_by_candidate: dict[str, DominanceAssessment] = {}
        if mode is PortfolioMode.STEADY_STATE:
            evaluations, dominance_by_candidate = (
                self.dominance_policy.apply_dual_dominance(
                    request,
                    evaluations,
                )
            )
        strategic = self.candidate_evaluator.rank(
            request.candidates,
            evaluations,
            predicate="strategic_candidate",
        )
        executable = self.candidate_evaluator.rank(
            request.candidates,
            evaluations,
            predicate="executable_candidate",
        )
        best_strategic = strategic[0] if strategic else None
        best_executable = executable[0] if executable else None

        lines = self.lifecycle_policy.build_lines(
            mode,
            request,
            evaluations,
            executable,
            dominance_by_candidate,
        )
        lines = self.weight_allocator.allocate_weights(request, lines)
        target_cash = max(0.0, 1.0 - sum(line.target_weight for line in lines))
        candidate_allocated = any(
            line.role == "CANDIDATE"
            and line.target_weight > self.config.epsilon
            for line in lines
        )
        portfolio_cer = self.weight_allocator.portfolio_cer(lines)
        sector_weights, cluster_weights = self.weight_allocator.exposure_weights(
            securities,
            lines,
        )
        if mode is PortfolioMode.COLD_START and not executable:
            reason = "CASH_ONLY_NO_EXECUTABLE_CANDIDATE"
        elif mode is PortfolioMode.COLD_START and not candidate_allocated:
            reason = "CASH_ONLY_PORTFOLIO_LIMIT"
        elif mode is PortfolioMode.BUILDING and not strategic:
            reason = "BUILDING_NO_ABSOLUTE_STRATEGIC_CANDIDATE"
        elif mode is PortfolioMode.BUILDING and not executable:
            reason = "BUILDING_NO_EXECUTABLE_CANDIDATE"
        elif mode is PortfolioMode.BUILDING and not candidate_allocated:
            reason = "BUILDING_NO_PORTFOLIO_FEASIBLE_SECOND_POSITION"
        elif mode is PortfolioMode.STEADY_STATE and not strategic:
            reason = "ALLOCATION_COMPLETE_NO_DUAL_DOMINANCE_CANDIDATE"
        elif mode is PortfolioMode.STEADY_STATE and not executable:
            reason = "ALLOCATION_COMPLETE_NO_EXECUTABLE_DUAL_DOMINANCE_CANDIDATE"
        else:
            reason = "ALLOCATION_COMPLETE"
        return PortfolioAllocationDecision(
            mode=mode,
            model_version=PAE_V1_MODEL_VERSION,
            required_return=required_return,
            best_strategic_alternative=(
                best_strategic.symbol if best_strategic else None
            ),
            best_executable_alternative=(
                best_executable.symbol if best_executable else None
            ),
            lines=tuple(lines),
            target_cash_weight=target_cash,
            portfolio_expected_cer=portfolio_cer,
            sector_weights=tuple(sorted(sector_weights.items())),
            correlation_cluster_weights=tuple(sorted(cluster_weights.items())),
            reason=reason,
        )

    def _required_return(self, request: PortfolioAllocationRequest) -> float:
        return self.candidate_evaluator.required_return(request)

    def _evaluate(
        self,
        security: AllocationSecurity,
        required_return: float,
    ) -> SecurityEvaluation:
        return self.candidate_evaluator.evaluate(security, required_return)

    @staticmethod
    def _rank(
        securities: tuple[AllocationSecurity, ...],
        evaluations: dict[str, SecurityEvaluation],
        *,
        predicate: str,
    ) -> list[AllocationSecurity]:
        return PortfolioCandidateEvaluator.rank(
            securities,
            evaluations,
            predicate=predicate,
        )

    def _apply_dual_dominance(self, *args, **kwargs):
        return self.dominance_policy.apply_dual_dominance(*args, **kwargs)

    @staticmethod
    def _dominance_evidence_rank(assessment: DominanceAssessment):
        return DominancePolicy.dominance_evidence_rank(assessment)

    def _cold_start_lines(self, *args, **kwargs):
        return self.lifecycle_policy.cold_start_lines(*args, **kwargs)

    def _building_lines(self, *args, **kwargs):
        return self.lifecycle_policy.building_lines(*args, **kwargs)

    def _first_candidate_with_allocation_room(self, *args, **kwargs):
        return self.weight_allocator.first_candidate_with_allocation_room(
            *args,
            **kwargs,
        )

    @staticmethod
    def _rank_by_capital_claim(*args, **kwargs):
        return WeightAllocator.rank_by_capital_claim(*args, **kwargs)

    def _steady_state_lines(self, *args, **kwargs):
        return self.lifecycle_policy.steady_state_lines(*args, **kwargs)

    def _best_challenger_for_holding(self, *args, **kwargs):
        return self.dominance_policy.best_challenger_for_holding(
            *args,
            **kwargs,
        )

    def _dominance_assessment(self, *args, **kwargs):
        return self.dominance_policy.dominance_assessment(*args, **kwargs)

    def _switch_hurdle(self, *args, **kwargs):
        return self.dominance_policy.switch_hurdle(*args, **kwargs)

    @staticmethod
    def _with_dominance(*args, **kwargs):
        return DominancePolicy.with_dominance(*args, **kwargs)

    def _base_holding_action(self, *args, **kwargs):
        return self.exit_policy.base_holding_action(*args, **kwargs)

    def _exit_confirmation_assessment(self, *args, **kwargs):
        return self.exit_policy.exit_confirmation_assessment(*args, **kwargs)

    def _resolve_incumbent_add(self, *args, **kwargs):
        return self.lifecycle_policy.resolve_incumbent_add(*args, **kwargs)

    def _replacement_test(self, *args, **kwargs):
        return self.dominance_policy.replacement_test(*args, **kwargs)

    @staticmethod
    def _hold_rotation_candidate(*args, **kwargs):
        return LifecyclePolicy.hold_rotation_candidate(*args, **kwargs)

    def _persistence_passed(
        self,
        request: PortfolioAllocationRequest,
        incumbent: str,
        challenger: str,
    ) -> bool:
        return self.dominance_policy.persistence_passed(
            request.persistence,
            incumbent,
            challenger,
        )

    def _obvious_opportunity(self, *args, **kwargs):
        return self.dominance_policy.obvious_opportunity(*args, **kwargs)

    def _incumbency_premium(self, confidence: float) -> float:
        return self.dominance_policy.incumbency_premium(confidence)

    @staticmethod
    def _base_target_weight(*args, **kwargs):
        return ExitPolicy.base_target_weight(*args, **kwargs)

    def _allocate_weights(self, *args, **kwargs):
        return self.weight_allocator.allocate_weights(*args, **kwargs)

    @staticmethod
    def _current_weight(*args, **kwargs):
        return WeightAllocator.current_weight(*args, **kwargs)

    @staticmethod
    def _portfolio_cer(*args, **kwargs):
        return WeightAllocator.portfolio_cer(*args, **kwargs)

    @staticmethod
    def _exposure_weights(*args, **kwargs):
        return WeightAllocator.exposure_weights(*args, **kwargs)


class PortfolioAllocationV1Strategy(Strategy):
    """Strategy plugin facade for Portfolio Allocation Engine v1."""

    name = PORTFOLIO_ALLOCATION_STRATEGY_NAME
    version = PAE_V1_MODEL_VERSION
    description = (
        "Lifecycle-aware quality-protected allocation across up to four holdings "
        "with steady-state quality-plus-CER dominance and confirmed exits"
    )

    def __init__(self, config: PortfolioAllocationConfig | None = None) -> None:
        self.engine = PortfolioAllocationEngine(config)
        from domain.strategies.portfolio_allocation_dataframe import (
            PortfolioAllocationDataFrameAdapter,
        )

        self.dataframe_adapter = PortfolioAllocationDataFrameAdapter(
            self.engine
        )

    def allocate(
        self,
        request: PortfolioAllocationRequest,
    ) -> PortfolioAllocationDecision:
        return self.engine.allocate(request)

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.dataframe_adapter.run(data)

    @classmethod
    def _security_from_row(cls, row: pd.Series) -> AllocationSecurity:
        from domain.strategies.portfolio_allocation_dataframe import (
            PortfolioAllocationDataFrameAdapter,
        )

        return PortfolioAllocationDataFrameAdapter.security_from_row(row)

    @staticmethod
    def _row_symbol(row: pd.Series) -> str:
        from domain.strategies.portfolio_allocation_dataframe import (
            PortfolioAllocationDataFrameAdapter,
        )

        return PortfolioAllocationDataFrameAdapter.row_symbol(row)

    @classmethod
    def _row_number(cls, row: pd.Series, *names: str) -> float | None:
        from domain.strategies.portfolio_allocation_dataframe import (
            PortfolioAllocationDataFrameAdapter,
        )

        return PortfolioAllocationDataFrameAdapter.row_number(row, *names)

    @staticmethod
    def _row_bool(row: pd.Series, name: str) -> bool:
        from domain.strategies.portfolio_allocation_dataframe import (
            PortfolioAllocationDataFrameAdapter,
        )

        return PortfolioAllocationDataFrameAdapter.row_bool(row, name)

    @staticmethod
    def _row_optional_bool(row: pd.Series, name: str) -> bool | None:
        from domain.strategies.portfolio_allocation_dataframe import (
            PortfolioAllocationDataFrameAdapter,
        )

        return PortfolioAllocationDataFrameAdapter.row_optional_bool(
            row,
            name,
        )

    @classmethod
    def _frame_number(cls, data: pd.DataFrame, name: str) -> float | None:
        from domain.strategies.portfolio_allocation_dataframe import (
            PortfolioAllocationDataFrameAdapter,
        )

        return PortfolioAllocationDataFrameAdapter.frame_number(data, name)

    @staticmethod
    def _optional_number(value: object) -> float | None:
        from domain.strategies.portfolio_allocation_dataframe import (
            PortfolioAllocationDataFrameAdapter,
        )

        return PortfolioAllocationDataFrameAdapter.optional_number(value)


__all__ = [
    "AllocationLineDecision",
    "AllocationSecurity",
    "DominancePolicy",
    "DominanceAssessment",
    "ExitPolicy",
    "ExitConfirmationAssessment",
    "LifecyclePolicy",
    "PAE_V1_MODEL_VERSION",
    "PORTFOLIO_ALLOCATION_OUTPUT_COLUMNS",
    "PORTFOLIO_ALLOCATION_STRATEGY_NAME",
    "PersistenceEvidence",
    "PortfolioAction",
    "PortfolioAllocationConfig",
    "PortfolioAllocationDecision",
    "PortfolioAllocationEngine",
    "PortfolioAllocationRequest",
    "PortfolioAllocationV1Strategy",
    "PortfolioCandidateEvaluator",
    "PortfolioMode",
    "SecurityEvaluation",
    "WeightAllocator",
    "resolve_portfolio_mode",
]
