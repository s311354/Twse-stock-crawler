"""Candidate return and eligibility evaluation for PAE v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


_EXECUTABLE_LES_DECISIONS = frozenset({"BUY", "BUY_SMALL"})
_ACCEPTABLE_QUALITY_STATUSES = frozenset(
    {"QUALITY_CONFIRMED", "QUALITY_PROVISIONAL"}
)
_STARTER_QUALITY_STATUSES = frozenset(
    {"QUALITY_CONFIRMED", "QUALITY_PROVISIONAL", "QUALITY_WATCH"}
)
_HARD_REJECT_STATUSES = frozenset(
    {"HARD_NO_BUY", "QUALITY_REJECT", "BASE_REJECT", "EXTENDED_REJECT"}
)


class CandidateEvaluationConfig(Protocol):
    benchmark_expected_return: float
    cash_expected_return: float
    absolute_minimum_return: float
    required_alpha: float
    equity_risk_premium: float
    bear_probability: float
    base_probability: float
    bull_probability: float
    valuation_uncertainty_lambda: float
    maximum_data_confidence_penalty: float
    minimum_candidate_quality_score: float
    buy_execution_factor: float
    buy_small_execution_factor: float


class CandidateSecurity(Protocol):
    symbol: str
    sector: str
    correlation_cluster: str
    quality_status: str
    quality_score: float | None
    data_confidence: float | None
    valuation_status: str
    valuation_gate: str
    bear_cagr: float | None
    base_cagr: float | None
    bull_cagr: float | None
    valuation_confidence: float | None
    les_decision: str
    risk_gate: str
    falling_knife_state: str
    executable_shares: float
    business_uncertainty_penalty: float
    fundamental_hard_reject: bool


class CandidateEvaluationRequest(Protocol):
    benchmark_expected_return: float | None
    cash_expected_return: float | None


@dataclass(frozen=True)
class SecurityEvaluation:
    symbol: str
    quality_score: float | None
    cer: float | None
    required_return: float
    opportunity_edge: float
    scenario_return: float | None
    scenario_spread: float | None
    uncertainty_penalty: float | None
    absolute_strategic_candidate: bool
    strategic_candidate: bool
    executable_candidate: bool
    dual_dominance_candidate: bool | None
    capital_claim: float
    data_insufficient: bool
    hard_reject: bool
    reasons: tuple[str, ...]


class PortfolioCandidateEvaluator:
    """Calculate CER and absolute candidate eligibility for one security."""

    def __init__(self, config: CandidateEvaluationConfig) -> None:
        self.config = config

    def required_return(self, request: CandidateEvaluationRequest) -> float:
        benchmark = (
            request.benchmark_expected_return
            if request.benchmark_expected_return is not None
            else self.config.benchmark_expected_return
        )
        cash_return = (
            request.cash_expected_return
            if request.cash_expected_return is not None
            else self.config.cash_expected_return
        )
        return max(
            self.config.absolute_minimum_return,
            benchmark + self.config.required_alpha,
            cash_return + self.config.equity_risk_premium,
        )

    def evaluate(
        self,
        security: CandidateSecurity,
        required_return: float,
    ) -> SecurityEvaluation:
        reasons: list[str] = []
        hard_reject = (
            security.fundamental_hard_reject
            or security.quality_status.upper() in _HARD_REJECT_STATUSES
        )
        if hard_reject:
            reasons.append("QUALITY_HARD_REJECT")

        scenario_values = (
            security.bear_cagr,
            security.base_cagr,
            security.bull_cagr,
        )
        valuation_complete = security.valuation_status.upper() == "COMPLETE"
        confidence_complete = (
            security.data_confidence is not None
            and security.valuation_confidence is not None
        )
        scenarios_complete = all(value is not None for value in scenario_values)
        if not valuation_complete:
            reasons.append("VALUATION_NOT_COMPLETE")
        if not scenarios_complete:
            reasons.append("SCENARIO_CAGR_MISSING")
        if not confidence_complete:
            reasons.append("CONFIDENCE_MISSING")

        scenario_return: float | None = None
        scenario_spread: float | None = None
        uncertainty_penalty: float | None = None
        cer: float | None = None
        if valuation_complete and scenarios_complete and confidence_complete:
            bear, base, bull = (float(value) for value in scenario_values)
            scenario_return = (
                self.config.bear_probability * bear
                + self.config.base_probability * base
                + self.config.bull_probability * bull
            )
            scenario_spread = max(0.0, bull - bear)
            valuation_penalty = (
                self.config.valuation_uncertainty_lambda
                * scenario_spread
                * (1 - float(security.valuation_confidence) / 100)
            )
            data_penalty = self.config.maximum_data_confidence_penalty * (
                1 - float(security.data_confidence) / 100
            )
            uncertainty_penalty = (
                valuation_penalty
                + data_penalty
                + security.business_uncertainty_penalty
            )
            cer = scenario_return - uncertainty_penalty

        les_decision = security.les_decision.upper()
        is_starter = les_decision == "BUY_SMALL"
        quality_acceptable = security.quality_status.upper() in (
            _STARTER_QUALITY_STATUSES
            if is_starter
            else _ACCEPTABLE_QUALITY_STATUSES
        )
        classification_known = (
            security.sector.strip().upper() not in {"", "UNKNOWN", "N/A"}
            and security.correlation_cluster.strip().upper()
            not in {"", "UNKNOWN", "N/A"}
        )
        if not classification_known:
            reasons.append("PORTFOLIO_CLASSIFICATION_MISSING")
        accepted_valuation_gates = (
            {"PASS", "SUPPORTIVE"} if is_starter else {"PASS"}
        )
        valuation_pass = (
            security.valuation_gate.upper() in accepted_valuation_gates
        )
        absolute_strategic_candidate = bool(
            not hard_reject
            and quality_acceptable
            and security.quality_score is not None
            and security.quality_score
            >= self.config.minimum_candidate_quality_score
            and classification_known
            and valuation_complete
            and valuation_pass
            and cer is not None
            and cer >= required_return
        )
        strategic_candidate = absolute_strategic_candidate
        executable_candidate = bool(
            strategic_candidate
            and les_decision in _EXECUTABLE_LES_DECISIONS
            and security.risk_gate.upper() == "PASS"
            and security.falling_knife_state.upper() == "CLEAR"
            and security.executable_shares > 0
        )
        edge = max((cer or 0.0) - required_return, 0.0)
        execution_factor = {
            "BUY": self.config.buy_execution_factor,
            "BUY_SMALL": self.config.buy_small_execution_factor,
        }.get(les_decision, 0.0)
        capital_claim = (
            edge
            * float(security.data_confidence or 0) / 100
            * float(security.valuation_confidence or 0) / 100
            * execution_factor
            if executable_candidate
            else 0.0
        )
        data_insufficient = bool(
            not hard_reject
            and (
                not valuation_complete
                or not scenarios_complete
                or not confidence_complete
                or not classification_known
                or security.quality_status.upper() == "DATA_INSUFFICIENT"
            )
        )
        return SecurityEvaluation(
            symbol=security.symbol,
            quality_score=security.quality_score,
            cer=cer,
            required_return=required_return,
            opportunity_edge=edge,
            scenario_return=scenario_return,
            scenario_spread=scenario_spread,
            uncertainty_penalty=uncertainty_penalty,
            absolute_strategic_candidate=absolute_strategic_candidate,
            strategic_candidate=strategic_candidate,
            executable_candidate=executable_candidate,
            dual_dominance_candidate=None,
            capital_claim=capital_claim,
            data_insufficient=data_insufficient,
            hard_reject=hard_reject,
            reasons=tuple(reasons),
        )

    @staticmethod
    def rank(
        securities: tuple[CandidateSecurity, ...],
        evaluations: dict[str, SecurityEvaluation],
        *,
        predicate: str,
    ) -> list[CandidateSecurity]:
        eligible = [
            item
            for item in securities
            if bool(getattr(evaluations[item.symbol], predicate))
        ]
        return sorted(
            eligible,
            key=lambda item: (
                -float(item.quality_score or 0.0),
                -(evaluations[item.symbol].cer or float("-inf")),
                item.symbol,
            ),
        )


__all__ = [
    "CandidateEvaluationConfig",
    "CandidateEvaluationRequest",
    "CandidateSecurity",
    "PortfolioCandidateEvaluator",
    "SecurityEvaluation",
]
