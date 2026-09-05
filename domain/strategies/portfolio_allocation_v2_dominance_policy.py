"""Dual-dominance, switch-hurdle and persistence policy for PAE v2."""

from __future__ import annotations

import hashlib

from domain.strategies.portfolio_allocation_v2_contracts import (
    PersistenceObservationV2,
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
from domain.strategies.portfolio_allocation_v2_utils import (
    normalized_confidence,
)


class PortfolioDominancePolicyV2:
    """Compare challengers to incumbents without choosing portfolio weights."""

    def __init__(self, config: PortfolioAllocationV2Config) -> None:
        self.config = config

    def persistence_observations(
        self,
        request: PortfolioAllocationRequestV2,
        evaluations: dict[str, SecurityEvaluationV2],
    ) -> tuple[PersistenceObservationV2, ...]:
        observations: list[PersistenceObservationV2] = []
        for incumbent in request.holdings:
            incumbent_eval = evaluations[incumbent.stock_id]
            for challenger in request.candidates:
                challenger_eval = evaluations[challenger.stock_id]
                observations.append(
                    PersistenceObservationV2(
                        incumbent_stock_id=incumbent.stock_id,
                        challenger_stock_id=challenger.stock_id,
                        passed=self.pair_crosses_replacement_hurdle(
                            challenger,
                            challenger_eval,
                            incumbent,
                            incumbent_eval,
                        ),
                    )
                )
        return tuple(observations)

    def pair_crosses_replacement_hurdle(
        self,
        challenger: PortfolioSecurityV2,
        challenger_eval: SecurityEvaluationV2,
        incumbent: PortfolioSecurityV2,
        incumbent_eval: SecurityEvaluationV2,
    ) -> bool:
        if not challenger_eval.executable_candidate:
            return False
        if challenger.quality_score is None or incumbent.quality_score is None:
            return False
        quality_dominance = (
            challenger.quality_status.upper() == "PASS"
            and float(challenger.quality_score)
            >= float(incumbent.quality_score)
            + self.config.minimum_quality_advantage
            and normalized_confidence(challenger.quality_confidence)
            >= self.config.minimum_quality_confidence
        )
        if not quality_dominance:
            return False
        if challenger_eval.cer is None or incumbent_eval.cer is None:
            return False
        return bool(
            challenger_eval.cer
            - incumbent_eval.cer
            - self.switch_hurdle(challenger, incumbent)
            > 0
        )

    def replacement_comparison(
        self,
        request: PortfolioAllocationRequestV2,
        challenger: PortfolioSecurityV2,
        incumbent: PortfolioSecurityV2,
        evaluations: dict[str, SecurityEvaluationV2],
    ) -> tuple[float, float, bool] | None:
        challenger_eval = evaluations[challenger.stock_id]
        incumbent_eval = evaluations[incumbent.stock_id]
        if not challenger_eval.executable_candidate:
            return None
        if challenger.quality_score is None or incumbent.quality_score is None:
            return None
        quality_dominance = (
            challenger.quality_status.upper() == "PASS"
            and float(challenger.quality_score)
            >= float(incumbent.quality_score)
            + self.config.minimum_quality_advantage
            and normalized_confidence(challenger.quality_confidence)
            >= self.config.minimum_quality_confidence
        )
        if not quality_dominance:
            return None
        if challenger_eval.cer is None or incumbent_eval.cer is None:
            return None
        hurdle = self.switch_hurdle(challenger, incumbent)
        excess = challenger_eval.cer - incumbent_eval.cer - hurdle
        if excess <= 0:
            return None
        fast_path = self.fast_path(challenger, challenger_eval, excess)
        persistent = self.persistence_passed(
            request,
            incumbent.stock_id,
            challenger.stock_id,
        )
        if not fast_path and not persistent:
            return None
        if not fast_path and excess < self.config.full_replacement_minimum_excess:
            return None
        return excess, hurdle, fast_path

    def switch_hurdle(
        self,
        challenger: PortfolioSecurityV2,
        incumbent: PortfolioSecurityV2,
    ) -> float:
        friction = (
            self.config.sell_commission
            + self.config.sell_transaction_tax
            + self.config.sell_spread_cost
            + self.config.sell_slippage
            + self.config.buy_commission
            + self.config.buy_spread_cost
            + self.config.buy_slippage
        )
        annualized_friction = (
            (1 + friction) ** (1 / self.config.expected_holding_years) - 1
        )
        confidence_disadvantage = max(
            normalized_confidence(incumbent.data_confidence)
            - normalized_confidence(challenger.data_confidence),
            0.0,
        )
        model_error = self.config.model_error_lambda * confidence_disadvantage
        confidence = float(incumbent.erem_holding_confidence or 0.0)
        if confidence >= 85:
            premium = 0.03
        elif confidence >= 70:
            premium = 0.02
        elif confidence >= 55:
            premium = 0.01
        elif confidence >= 40:
            premium = 0.005
        else:
            premium = 0.0
        return (
            self.config.base_replacement_buffer
            + annualized_friction
            + model_error
            + premium
        )

    def persistence_passed(
        self,
        request: PortfolioAllocationRequestV2,
        incumbent: str,
        challenger: str,
    ) -> bool:
        evidence = next(
            (
                item
                for item in request.persistence
                if item.incumbent_stock_id == incumbent
                and item.challenger_stock_id == challenger
            ),
            None,
        )
        if evidence is None:
            return False
        recent = evidence.recent_passes[
            -self.config.persistence_window_sessions :
        ]
        return sum(bool(item) for item in recent) >= (
            self.config.persistence_required_sessions
        )

    def fast_path(
        self,
        security: PortfolioSecurityV2,
        evaluation: SecurityEvaluationV2,
        excess: float,
    ) -> bool:
        return bool(
            security.les_decision.upper() in {"BUY", "EARLY_BUY"}
            and security.risk_gate.upper() == "PASS"
            and security.falling_knife_state.upper() == "CLEAR"
            and security.liquidity_status.upper() == "PASS"
            and float(security.quality_score or 0.0)
            >= self.config.obvious_quality_score
            and normalized_confidence(security.quality_confidence)
            >= self.config.obvious_quality_confidence
            and normalized_confidence(security.data_confidence)
            >= self.config.obvious_data_confidence
            and normalized_confidence(security.valuation_confidence)
            >= self.config.obvious_valuation_confidence
            and evaluation.cer is not None
            and evaluation.cer
            >= evaluation.required_return
            + self.config.obvious_required_return_spread
            and excess >= self.config.obvious_excess_opportunity
        )

    @staticmethod
    def atomic_group_id(
        request: PortfolioAllocationRequestV2,
        incumbent: PortfolioSecurityV2,
        challenger: PortfolioSecurityV2,
    ) -> str:
        raw = "{}|{}|{}|{}".format(
            request.portfolio_id,
            request.decision_time.isoformat(),
            incumbent.stock_id,
            challenger.stock_id,
        )
        return "PAE2-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


__all__ = ["PortfolioDominancePolicyV2"]
