"""Cash, position sizing and portfolio-improvement policy for PAE v2."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

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
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioSecurityV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    SecurityEvaluationV2,
)


class PortfolioWeightAllocatorV2:
    """Apply risk caps and prove a proposed allocation improves the portfolio."""

    def __init__(self, config: PortfolioAllocationV2Config) -> None:
        self.config = config

    def validate_balance(self, request: PortfolioAllocationRequestV2) -> None:
        total = request.available_cash / request.account_equity + sum(
            item.current_weight for item in request.holdings
        )
        if not np.isclose(total, 1.0, atol=self.config.balance_tolerance):
            raise ValueError("holdings plus available_cash must reconcile")

    def target_weight(
        self,
        request: PortfolioAllocationRequestV2,
        security: PortfolioSecurityV2,
        available_weight: float,
        excluded_holding: PortfolioSecurityV2 | None,
    ) -> tuple[float, tuple[str, ...]]:
        mandatory = {
            "LES_POSITION_CAP_MISSING": security.les_position_cap,
            "RISK_BUDGET_WEIGHT_MISSING": security.risk_budget_weight,
            "LIQUIDITY_WEIGHT_CAP_MISSING": security.liquidity_weight_cap,
        }
        missing = tuple(name for name, value in mandatory.items() if value is None)
        if missing:
            return 0.0, missing
        sector_weight = sum(
            item.current_weight
            for item in request.holdings
            if item is not excluded_holding
            and item.sector_id == security.sector_id
        )
        cluster_weight = sum(
            item.current_weight
            for item in request.holdings
            if item is not excluded_holding
            and item.correlation_cluster_id == security.correlation_cluster_id
        )
        caps = [
            float(security.les_position_cap),
            float(security.risk_budget_weight),
            float(security.liquidity_weight_cap),
            self.config.absolute_single_stock_max,
            self.config.starter_weight_max,
            max(available_weight, 0.0),
        ]
        if security.half_kelly_weight is not None:
            caps.append(float(security.half_kelly_weight))
        if self.config.sector_cap_enabled:
            caps.append(max(self.config.sector_weight_max - sector_weight, 0.0))
        if security.correlation_cluster_id.upper() in {"", "UNKNOWN", "N/A"}:
            caps.append(self.config.unknown_correlation_starter_cap)
        else:
            caps.append(
                max(
                    self.config.correlation_cluster_weight_max - cluster_weight,
                    0.0,
                )
            )
        execution_factor = {
            "BUY_SMALL": self.config.buy_small_execution_factor,
            "EARLY_BUY": self.config.early_buy_execution_factor,
            "BUY": self.config.buy_execution_factor,
        }.get(security.les_decision.upper(), 0.0)
        return max(min(caps) * execution_factor, 0.0), ()

    def cash_capacity(self, request: PortfolioAllocationRequestV2) -> float:
        return max(
            request.available_cash / request.account_equity
            - self.minimum_cash_weight(request),
            0.0,
        )

    def minimum_cash_weight(
        self,
        request: PortfolioAllocationRequestV2,
    ) -> float:
        return max(
            self.config.minimum_cash_weight,
            request.reserved_cash / request.account_equity,
            request.near_term_liability_reserve / request.account_equity,
        )

    def rank_candidates(
        self,
        candidates: tuple[PortfolioSecurityV2, ...],
        evaluations: dict[str, SecurityEvaluationV2],
    ) -> list[PortfolioSecurityV2]:
        return sorted(
            (
                item
                for item in candidates
                if evaluations[item.stock_id].executable_candidate
            ),
            key=lambda item: (
                -evaluations[item.stock_id].capital_claim,
                -float(item.quality_score or 0.0),
                item.stock_id,
            ),
        )

    @staticmethod
    def candidate_line(
        security: PortfolioSecurityV2,
        evaluation: SecurityEvaluationV2,
        action: PortfolioActionV2,
        target_weight: float,
        reasons: tuple[str, ...],
        blocking: tuple[str, ...],
        portfolio_improvement: bool | None,
    ) -> PortfolioAllocationLineV2:
        return PortfolioAllocationLineV2(
            stock_id=security.stock_id,
            role="CANDIDATE",
            action=action,
            current_weight=0.0,
            target_weight=target_weight,
            evaluation=evaluation,
            portfolio_improvement=portfolio_improvement,
            execution_status="READY",
            reason_codes=reasons,
            blocking_reasons=blocking,
        )

    @staticmethod
    def inactive_candidate_line(
        security: PortfolioSecurityV2,
        evaluation: SecurityEvaluationV2,
    ) -> PortfolioAllocationLineV2:
        reason = (
            "OPEN_SLOT_NOT_REQUIRED_TO_FILL"
            if evaluation.executable_candidate
            else "NO_NEW_CAPITAL"
        )
        return PortfolioAllocationLineV2(
            stock_id=security.stock_id,
            role="CANDIDATE",
            action=PortfolioActionV2.NO_ACTION,
            current_weight=0.0,
            target_weight=0.0,
            evaluation=evaluation,
            execution_status="NOT_REQUIRED",
            reason_codes=(reason,),
            blocking_reasons=evaluation.blocking_reasons,
        )

    @staticmethod
    def cash_after(lines: list[PortfolioAllocationLineV2]) -> float:
        return max(0.0, 1.0 - sum(line.target_weight for line in lines))

    def portfolio_improves(
        self,
        request: PortfolioAllocationRequestV2,
        evaluations: dict[str, SecurityEvaluationV2],
        candidate: PortfolioSecurityV2,
        target_weight: float,
        excluded_holding: PortfolioSecurityV2 | None,
    ) -> bool:
        holding_evaluations = [
            evaluations[item.stock_id]
            for item in request.holdings
            if item is not excluded_holding
        ]
        if any(item.cer is None for item in holding_evaluations):
            return False
        candidate_evaluation = evaluations[candidate.stock_id]
        if candidate_evaluation.cer is None:
            return False
        cash_return = (
            self.config.cash_expected_return
            if request.cash_expected_return is None
            else request.cash_expected_return
        )
        current_cer = sum(
            item.current_weight * float(evaluations[item.stock_id].cer)
            for item in request.holdings
            if evaluations[item.stock_id].cer is not None
        ) + (request.available_cash / request.account_equity) * cash_return
        remaining = [
            item for item in request.holdings if item is not excluded_holding
        ]
        cash_after = max(
            0.0,
            1.0 - sum(item.current_weight for item in remaining) - target_weight,
        )
        post_cer = sum(
            item.current_weight * float(evaluations[item.stock_id].cer)
            for item in remaining
        ) + target_weight * candidate_evaluation.cer + cash_after * cash_return
        friction = self.annualized_portfolio_friction(target_weight)
        delta_cer = post_cer - current_cer - friction

        before_quality = self.portfolio_quality(request.holdings)
        after_quality = self.portfolio_quality(
            tuple(remaining) + (self.with_weight(candidate, target_weight),)
        )
        quality_not_worse = (
            before_quality is None
            or (
                after_quality is not None
                and after_quality + self.config.epsilon >= before_quality
            )
        )
        return bool(
            delta_cer > self.config.minimum_delta_portfolio_cer
            and quality_not_worse
            and cash_after + self.config.epsilon
            >= self.minimum_cash_weight(request)
        )

    def annualized_portfolio_friction(self, traded_weight: float) -> float:
        friction = (
            self.config.sell_commission
            + self.config.sell_transaction_tax
            + self.config.sell_spread_cost
            + self.config.sell_slippage
            + self.config.buy_commission
            + self.config.buy_spread_cost
            + self.config.buy_slippage
        )
        annualized = (
            (1 + friction) ** (1 / self.config.expected_holding_years) - 1
        )
        return traded_weight * annualized

    @staticmethod
    def portfolio_quality(
        securities: tuple[PortfolioSecurityV2, ...],
    ) -> float | None:
        weighted = [
            (item.current_weight, item.quality_score)
            for item in securities
            if item.current_weight > 0 and item.quality_score is not None
        ]
        total = sum(weight for weight, _ in weighted)
        if total <= 0:
            return None
        return sum(weight * float(score) for weight, score in weighted) / total

    @staticmethod
    def with_weight(
        security: PortfolioSecurityV2,
        weight: float,
    ) -> PortfolioSecurityV2:
        values = asdict(security)
        values["current_weight"] = weight
        return PortfolioSecurityV2(**values)


__all__ = ["PortfolioWeightAllocatorV2"]
