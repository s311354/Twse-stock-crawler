"""Holding exit and reduce-only policy for PAE v2."""

from __future__ import annotations

from dataclasses import asdict

from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioActionV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioAllocationLineV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioSecurityV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    SecurityEvaluationV2,
)


class PortfolioExitPolicyV2:
    """Convert published EREM and hard-risk evidence into holding actions."""

    def holding_line(
        self,
        security: PortfolioSecurityV2,
        evaluation: SecurityEvaluationV2,
    ) -> PortfolioAllocationLineV2:
        hard_exit = (
            security.fundamental_hard_exit
            or security.strategy_stop_breached
            or security.cyclical_state.upper() == "CYCLICAL_DOWNTURN_CONFIRMED"
        )
        if hard_exit:
            action = PortfolioActionV2.EXIT_TO_CASH
            target = 0.0
            reason = "FUNDAMENTAL_OR_HARD_RISK_EXIT"
        elif not security.erem_data_sufficient:
            action = PortfolioActionV2.HOLD_DATA_INSUFFICIENT
            target = security.current_weight
            reason = "MISSING_EREM_DOES_NOT_FORCE_EXIT"
        elif security.erem_decision.upper() == "EXIT":
            action = PortfolioActionV2.EXIT_TO_CASH
            target = 0.0
            reason = "EREM_EXIT"
        elif security.erem_decision.upper() == "REDUCE_25":
            action = PortfolioActionV2.TRIM_25
            target = security.current_weight * 0.75
            reason = "EREM_REDUCE"
        elif security.cyclical_state.upper() == "CYCLICAL_PEAK_CONFIRMED":
            action = PortfolioActionV2.TRIM_TO_TARGET
            target = security.current_weight * 0.75
            reason = "CYCLICAL_PEAK_CONFIRMED"
        elif security.cyclical_state.upper() == "CYCLICAL_DATA_INSUFFICIENT":
            action = PortfolioActionV2.HOLD_REVIEW
            target = security.current_weight
            reason = "CYCLICAL_DATA_INSUFFICIENT"
        else:
            action = PortfolioActionV2.HOLD
            target = security.current_weight
            reason = "EREM_HOLD"
        return PortfolioAllocationLineV2(
            stock_id=security.stock_id,
            role="HOLDING",
            action=action,
            current_weight=security.current_weight,
            target_weight=target,
            evaluation=evaluation,
            execution_status=(
                "READY"
                if action
                in {
                    PortfolioActionV2.EXIT_TO_CASH,
                    PortfolioActionV2.TRIM_25,
                    PortfolioActionV2.TRIM_TO_TARGET,
                }
                else "NOT_REQUIRED"
            ),
            reason_codes=(reason,),
        )

    def breach_lines(
        self,
        holding_lines: list[PortfolioAllocationLineV2],
    ) -> list[PortfolioAllocationLineV2]:
        protected = {
            PortfolioActionV2.EXIT_TO_CASH,
            PortfolioActionV2.TRIM_25,
            PortfolioActionV2.TRIM_TO_TARGET,
        }
        return [
            line
            if line.action in protected
            else self.replace_line(
                line,
                action=PortfolioActionV2.POLICY_BREACH_REDUCE_ONLY,
                reason_codes=("POLICY_BREACH_HOLDINGS_GT_4",),
            )
            for line in holding_lines
        ]

    @staticmethod
    def replace_line(
        line: PortfolioAllocationLineV2,
        **changes: object,
    ) -> PortfolioAllocationLineV2:
        values = asdict(line)
        values["evaluation"] = line.evaluation
        values.update(changes)
        return PortfolioAllocationLineV2(**values)


__all__ = ["PortfolioExitPolicyV2"]
