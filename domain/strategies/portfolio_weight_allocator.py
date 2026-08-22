"""Portfolio feasibility checks and deterministic weight allocation for PAE."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from domain.strategies.portfolio_allocation_contracts import (
    AllocationLineDecision,
)
from domain.strategies.portfolio_allocation_contracts import AllocationSecurity
from domain.strategies.portfolio_allocation_contracts import PortfolioAction
from domain.strategies.portfolio_allocation_contracts import (
    PortfolioAllocationRequest,
)
from domain.strategies.portfolio_candidate_evaluator import SecurityEvaluation


class WeightAllocator:
    """Apply cash, name, sector, cluster and liquidity allocation limits."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def first_candidate_with_allocation_room(
        self,
        request: PortfolioAllocationRequest,
        lines: list[AllocationLineDecision],
        ranked: list[AllocationSecurity],
    ) -> AllocationSecurity | None:
        if not ranked:
            return None
        sector_weights: dict[str, float] = {}
        cluster_weights: dict[str, float] = {}
        security_by_symbol = {
            item.symbol: item
            for item in request.holdings + request.candidates
        }
        for line in lines:
            if line.target_weight <= self.config.epsilon:
                continue
            security = security_by_symbol[line.symbol]
            sector_weights[security.sector] = (
                sector_weights.get(security.sector, 0.0)
                + line.target_weight
            )
            cluster_weights[security.correlation_cluster] = (
                cluster_weights.get(security.correlation_cluster, 0.0)
                + line.target_weight
            )

        released_weight = sum(
            max(line.current_weight - line.target_weight, 0.0)
            for line in lines
            if line.role == "HOLDING"
        )
        portfolio_room = max(
            0.0,
            1.0
            - self.config.minimum_cash_reserve
            - sum(line.target_weight for line in lines),
        )
        deployable_cash = max(
            0.0,
            request.current_cash / request.account_equity
            + released_weight
            - self.config.minimum_cash_reserve,
        )
        available = min(portfolio_room, deployable_cash)
        if available <= self.config.epsilon:
            return None

        for candidate in ranked:
            name_cap = min(
                self.config.single_stock_weight_limit,
                candidate.maximum_weight
                if candidate.maximum_weight is not None
                else 1.0,
                candidate.liquidity_weight_limit
                if candidate.liquidity_weight_limit is not None
                else 1.0,
            )
            sector_room = max(
                0.0,
                self.config.sector_weight_limit
                - sector_weights.get(candidate.sector, 0.0),
            )
            cluster_room = max(
                0.0,
                self.config.correlation_cluster_weight_limit
                - cluster_weights.get(candidate.correlation_cluster, 0.0),
            )
            allocation_room = min(
                available,
                name_cap,
                sector_room,
                cluster_room,
            )
            if allocation_room > self.config.epsilon:
                return candidate
        return None

    @staticmethod
    def rank_by_capital_claim(
        securities: list[AllocationSecurity],
        evaluations: dict[str, SecurityEvaluation],
    ) -> list[AllocationSecurity]:
        return sorted(
            securities,
            key=lambda item: (
                -float(item.quality_score or 0.0),
                -evaluations[item.symbol].capital_claim,
                -(evaluations[item.symbol].cer or float("-inf")),
                item.symbol,
            ),
        )

    def allocate_weights(
        self,
        request: PortfolioAllocationRequest,
        lines: list[AllocationLineDecision],
    ) -> list[AllocationLineDecision]:
        securities = {
            item.symbol: item
            for item in request.holdings + request.candidates
        }
        updated = list(lines)
        sector_weights: dict[str, float] = {}
        cluster_weights: dict[str, float] = {}
        for line in updated:
            if line.target_weight <= self.config.epsilon:
                continue
            security = securities[line.symbol]
            sector_weights[security.sector] = (
                sector_weights.get(security.sector, 0.0) + line.target_weight
            )
            cluster_weights[security.correlation_cluster] = (
                cluster_weights.get(security.correlation_cluster, 0.0)
                + line.target_weight
            )

        portfolio_room = max(
            0.0,
            1.0
            - self.config.minimum_cash_reserve
            - sum(line.target_weight for line in updated),
        )
        released_weight = sum(
            max(line.current_weight - line.target_weight, 0.0)
            for line in updated
            if line.role == "HOLDING"
        )
        deployable_cash = max(
            0.0,
            request.current_cash / request.account_equity
            + released_weight
            - self.config.minimum_cash_reserve,
        )
        available = min(portfolio_room, deployable_cash)
        recipient_indexes = [
            index
            for index, line in enumerate(updated)
            if line.action
            in {
                PortfolioAction.STRONG_ADD,
                PortfolioAction.ADD,
                PortfolioAction.STARTER,
            }
            and line.evaluation.capital_claim > 0
        ]
        total_claim = sum(
            updated[index].evaluation.capital_claim
            for index in recipient_indexes
        )
        original_available = available
        failed_recipients: set[str] = set()
        for index in sorted(
            recipient_indexes,
            key=lambda item: (
                -updated[item].evaluation.capital_claim,
                updated[item].symbol,
            ),
        ):
            line = updated[index]
            security = securities[line.symbol]
            requested = (
                original_available
                * line.evaluation.capital_claim
                / total_claim
                if total_claim > self.config.epsilon
                else 0.0
            )
            current_target = line.target_weight
            name_cap = min(
                self.config.single_stock_weight_limit,
                security.maximum_weight
                if security.maximum_weight is not None
                else 1.0,
                security.liquidity_weight_limit
                if security.liquidity_weight_limit is not None
                else 1.0,
            )
            sector_remaining = max(
                0.0,
                self.config.sector_weight_limit
                - sector_weights.get(security.sector, 0.0),
            )
            cluster_remaining = max(
                0.0,
                self.config.correlation_cluster_weight_limit
                - cluster_weights.get(security.correlation_cluster, 0.0),
            )
            addition = min(
                requested,
                available,
                max(0.0, name_cap - current_target),
                sector_remaining,
                cluster_remaining,
            )
            if addition <= self.config.epsilon:
                replacement_action = (
                    PortfolioAction.HOLD_NO_ADD
                    if line.role == "HOLDING"
                    else PortfolioAction.WATCH
                )
                updated[index] = replace(
                    line,
                    action=replacement_action,
                    portfolio_risk_gate="FAIL",
                    reason="PORTFOLIO_RISK_CAP_OR_CASH_RESERVE",
                )
                failed_recipients.add(line.symbol)
                continue
            target = current_target + addition
            updated[index] = replace(line, target_weight=target)
            available -= addition
            sector_weights[security.sector] = (
                sector_weights.get(security.sector, 0.0) + addition
            )
            cluster_weights[security.correlation_cluster] = (
                cluster_weights.get(security.correlation_cluster, 0.0)
                + addition
            )
        if failed_recipients:
            updated = [
                replace(
                    line,
                    action=PortfolioAction.HOLD,
                    target_weight=line.current_weight,
                    portfolio_risk_gate="FAIL",
                    reason="ALTERNATIVE_FAILED_PORTFOLIO_RISK_GATE",
                )
                if line.role == "HOLDING"
                and line.alternative_symbol in failed_recipients
                and line.action
                in {
                    PortfolioAction.ROTATE_25,
                    PortfolioAction.ROTATE_50,
                    PortfolioAction.ROTATE_75,
                    PortfolioAction.FAST_TRACK_ROTATE,
                }
                else line
                for line in updated
            ]
        return updated

    @staticmethod
    def current_weight(
        security: AllocationSecurity,
        account_equity: float,
    ) -> float:
        if security.current_weight > 0:
            return security.current_weight
        return security.market_value / account_equity

    @staticmethod
    def portfolio_cer(
        lines: list[AllocationLineDecision],
    ) -> float | None:
        known = [
            line
            for line in lines
            if line.target_weight > 0 and line.evaluation.cer is not None
        ]
        denominator = sum(line.target_weight for line in known)
        if denominator <= 0:
            return None
        return sum(
            line.target_weight * float(line.evaluation.cer) for line in known
        ) / denominator

    @staticmethod
    def exposure_weights(
        securities: tuple[AllocationSecurity, ...],
        lines: list[AllocationLineDecision],
    ) -> tuple[dict[str, float], dict[str, float]]:
        by_symbol = {item.symbol: item for item in securities}
        sectors: dict[str, float] = {}
        clusters: dict[str, float] = {}
        for line in lines:
            if line.target_weight <= 0:
                continue
            security = by_symbol[line.symbol]
            sectors[security.sector] = (
                sectors.get(security.sector, 0.0) + line.target_weight
            )
            clusters[security.correlation_cluster] = clusters.get(
                security.correlation_cluster,
                0.0,
            ) + line.target_weight
        return sectors, clusters


__all__ = ["WeightAllocator"]
