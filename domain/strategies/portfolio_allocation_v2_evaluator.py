"""Security eligibility and expected-return evaluation for PAE v2."""

from __future__ import annotations

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


class PortfolioSecurityEvaluatorV2:
    """Evaluate CER and hard eligibility gates without allocating capital."""

    def __init__(self, config: PortfolioAllocationV2Config) -> None:
        self.config = config

    def required_return(self, request: PortfolioAllocationRequestV2) -> float:
        benchmark = (
            self.config.benchmark_expected_return
            if request.benchmark_expected_return is None
            else request.benchmark_expected_return
        )
        cash_return = (
            self.config.cash_expected_return
            if request.cash_expected_return is None
            else request.cash_expected_return
        )
        return max(
            self.config.strategy_minimum_return,
            benchmark + self.config.required_alpha,
            cash_return + self.config.equity_risk_premium,
        )

    def evaluate(
        self,
        security: PortfolioSecurityV2,
        required_return: float,
    ) -> SecurityEvaluationV2:
        missing = tuple(
            name
            for name in (
                "quality_score",
                "quality_confidence",
                "data_confidence",
                "bear_cagr",
                "base_cagr",
                "bull_cagr",
                "valuation_confidence",
            )
            if getattr(security, name) is None
        )
        cer: float | None = None
        if not missing:
            scenario_return = (
                self.config.bear_probability * float(security.bear_cagr)
                + self.config.base_probability * float(security.base_cagr)
                + self.config.bull_probability * float(security.bull_cagr)
            )
            spread = max(
                float(security.bull_cagr) - float(security.bear_cagr),
                0.0,
            )
            valuation_penalty = (
                self.config.valuation_uncertainty_lambda
                * spread
                * (1 - normalized_confidence(security.valuation_confidence))
            )
            data_penalty = self.config.maximum_data_confidence_penalty * (
                1 - normalized_confidence(security.data_confidence)
            )
            cer = (
                scenario_return
                - valuation_penalty
                - data_penalty
                - security.business_uncertainty_penalty
            )

        from domain.les_v7.entry_policy_r9 import pae_admission
        admission = pae_admission(security)
        quality_pass = (admission["quality"]["evidence_ready"] and admission["quality"]["small_cap"] > 0
                        if admission else security.quality_status.upper() == "PASS")
        valuation_pass = (admission["valuation"]["state"] in {"ALLOWED", "ALLOWED_WITH_WARNING"}
                          if admission else security.valuation_gate.upper() == "PASS")
        minimum_confidence = .65 if admission else self.config.minimum_quality_confidence
        reasons: list[str] = []
        if security.fundamental_hard_exit:
            reasons.append("FUNDAMENTAL_HARD_EXIT")
        if not quality_pass:
            reasons.append("QUALITY_REJECT")
        if security.valuation_status.upper() != "COMPLETE":
            reasons.append("VALUATION_INCOMPLETE")
        if not valuation_pass:
            reasons.append("VALUATION_FAIL")
        quality_confidence_pass = (
            security.quality_confidence is not None
            and normalized_confidence(security.quality_confidence)
            >= minimum_confidence
        )
        if not quality_confidence_pass:
            reasons.append("QUALITY_CONFIDENCE_LOW_OR_UNKNOWN")
        if cer is None:
            reasons.append("CER_UNKNOWN")
        elif cer < required_return:
            reasons.append("CER_BELOW_REQUIRED_RETURN")

        strategic = bool(
            not security.fundamental_hard_exit
            and quality_pass
            and security.quality_score is not None
            and quality_confidence_pass
            and security.valuation_status.upper() == "COMPLETE"
            and valuation_pass
            and cer is not None
            and cer >= required_return
        )
        executable = bool(
            strategic
            and (not admission or admission.get("buy_allowed" if security.les_decision.upper() == "BUY" else "small_allowed") is True)
            and security.les_decision.upper()
            in {"BUY", "EARLY_BUY", "BUY_SMALL"}
            and security.risk_gate.upper() == "PASS"
            and security.falling_knife_state.upper() == "CLEAR"
            and security.liquidity_status.upper() == "PASS"
            and security.executable_shares > 0
        )
        if strategic and not executable:
            reasons.append("LES_OR_EXECUTION_GATE_NOT_EXECUTABLE")
        edge = None if cer is None else cer - required_return
        execution_factor = {
            "BUY_SMALL": self.config.buy_small_execution_factor,
            "EARLY_BUY": self.config.early_buy_execution_factor,
            "BUY": self.config.buy_execution_factor,
        }.get(security.les_decision.upper(), 0.0)
        capital_claim = (
            max(float(edge or 0.0), 0.0)
            * normalized_confidence(security.quality_confidence)
            * normalized_confidence(security.valuation_confidence)
            * execution_factor
            if executable
            else 0.0
        )
        return SecurityEvaluationV2(
            stock_id=security.stock_id,
            cer=cer,
            required_return=required_return,
            opportunity_edge=edge,
            capital_claim=capital_claim,
            strategic_candidate=strategic,
            executable_candidate=executable,
            data_quality="INSUFFICIENT" if missing else "COMPLETE",
            missing_fields=missing,
            blocking_reasons=tuple(reasons),
        )

    def direct_selection_executable(
        self,
        security: PortfolioSecurityV2,
    ) -> bool:
        """Apply LES hard gates while PAE remains non-binding."""
        from domain.les_v7.entry_policy_r9 import pae_admission
        admission = pae_admission(security)
        if admission:
            return bool(admission.get("buy_allowed" if security.les_decision.upper() == "BUY" else "small_allowed")
                and security.les_decision.upper() in {"BUY", "BUY_SMALL"}
                and not security.fundamental_hard_exit and security.risk_gate.upper() == "PASS"
                and security.falling_knife_state.upper() == "CLEAR" and security.liquidity_status.upper() == "PASS"
                and security.executable_shares > 0)
        return bool(
            not security.fundamental_hard_exit
            and security.quality_status.upper() == "PASS"
            and security.quality_score is not None
            and security.quality_confidence is not None
            and normalized_confidence(security.quality_confidence)
            >= self.config.minimum_quality_confidence
            and security.valuation_status.upper() == "COMPLETE"
            and security.valuation_gate.upper() == "PASS"
            and security.les_decision.upper()
            in {"BUY", "EARLY_BUY", "BUY_SMALL"}
            and security.risk_gate.upper() == "PASS"
            and security.falling_knife_state.upper() == "CLEAR"
            and security.liquidity_status.upper() == "PASS"
            and security.executable_shares > 0
        )


__all__ = ["PortfolioSecurityEvaluatorV2"]
