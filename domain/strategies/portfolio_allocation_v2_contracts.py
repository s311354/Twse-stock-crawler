"""Versioned contracts for Portfolio Allocation Engine v2."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


PAE_V2_MODEL_VERSION = "2.0.0"
PAE_V2_SCHEMA_VERSION = "pae.portfolio_decision.v2"
PAE_V2_STRATEGY_NAME = "portfolio_allocation_v2"
PAE_V2_OUTPUT_COLUMNS = (
    "PAE Status",
    "PAE Mode",
    "PAE Binding",
    "PAE Portfolio Input",
    "PAE Role",
    "PAE CER",
    "PAE Quality Score",
    "PAE Required Return",
    "PAE Opportunity Edge",
    "PAE Strategic Candidate",
    "PAE Executable Candidate",
    "PAE Capital Claim",
    "PAE Current Weight",
    "PAE Target Weight",
    "PAE Weight Change",
    "PAE Target Cash Weight",
    "PAE Portfolio Expected CER",
    "PAE Decision",
    "PAE Reason",
    "PAE Blocking Reasons",
    "PAE Incumbent",
    "PAE Switch Hurdle",
    "PAE Excess Opportunity",
    "PAE Quality Dominance Gate",
    "PAE CER Dominance Gate",
    "PAE Persistence Gate",
    "PAE Obvious Opportunity",
    "PAE Portfolio Improvement",
    "PAE Execution Status",
    "PAE Atomic Group",
    "PAE Data Quality",
    "PAE Missing Fields",
    "PAE Portfolio Reason",
    "PAE Model Version",
    "PAE Schema Version",
)


class PortfolioModeV2(str, Enum):
    DIRECT_SELECTION = "DIRECT_SELECTION"
    PAE_ACTIVE_OPEN_SLOT = "PAE_ACTIVE_OPEN_SLOT"
    PAE_ACTIVE_FULL = "PAE_ACTIVE_FULL"
    POLICY_BREACH_REDUCE_ONLY = "POLICY_BREACH_REDUCE_ONLY"


class PortfolioBindingV2(str, Enum):
    BINDING = "BINDING"
    NON_BINDING = "NON_BINDING"


class PortfolioActionV2(str, Enum):
    HOLD = "HOLD"
    HOLD_NO_ADD = "HOLD_NO_ADD"
    HOLD_CASH = "HOLD_CASH"
    HOLD_REVIEW = "HOLD_REVIEW"
    HOLD_DATA_INSUFFICIENT = "HOLD_DATA_INSUFFICIENT"
    STARTER = "STARTER"
    ADD = "ADD"
    STRONG_ADD = "STRONG_ADD"
    TRIM_25 = "TRIM_25"
    TRIM_TO_TARGET = "TRIM_TO_TARGET"
    EXIT_TO_CASH = "EXIT_TO_CASH"
    EXIT_AND_REALLOCATE = "EXIT_AND_REALLOCATE"
    ROTATE_25 = "ROTATE_25"
    ROTATE_50 = "ROTATE_50"
    ROTATE_75 = "ROTATE_75"
    FAST_TRACK_ROTATE = "FAST_TRACK_ROTATE"
    DEFER_ROTATION = "DEFER_ROTATION"
    NO_ACTION = "NO_ACTION"
    SKIP = "SKIP"
    POLICY_BREACH_REDUCE_ONLY = "POLICY_BREACH_REDUCE_ONLY"


def resolve_portfolio_mode_v2(holding_count: int) -> PortfolioModeV2:
    """Resolve the normative PAE v2 mode from effective equity holdings."""
    if holding_count < 0:
        raise ValueError("holding_count cannot be negative")
    if holding_count < 3:
        return PortfolioModeV2.DIRECT_SELECTION
    if holding_count == 3:
        return PortfolioModeV2.PAE_ACTIVE_OPEN_SLOT
    if holding_count == 4:
        return PortfolioModeV2.PAE_ACTIVE_FULL
    return PortfolioModeV2.POLICY_BREACH_REDUCE_ONLY


def binding_for_mode(mode: PortfolioModeV2) -> PortfolioBindingV2:
    if mode is PortfolioModeV2.DIRECT_SELECTION:
        return PortfolioBindingV2.NON_BINDING
    return PortfolioBindingV2.BINDING


@dataclass(frozen=True)
class PortfolioAllocationV2Config:
    maximum_holdings: int = 4
    minimum_cash_weight: float = 0.10
    absolute_single_stock_max: float = 0.35
    sector_weight_max: float = 0.50
    correlation_cluster_weight_max: float = 0.50
    sector_cap_enabled: bool = True
    strategy_minimum_return: float = 0.10
    benchmark_expected_return: float = 0.07
    required_alpha: float = 0.03
    cash_expected_return: float = 0.02
    equity_risk_premium: float = 0.06
    bear_probability: float = 0.25
    base_probability: float = 0.50
    bull_probability: float = 0.25
    valuation_uncertainty_lambda: float = 0.25
    maximum_data_confidence_penalty: float = 0.03
    base_replacement_buffer: float = 0.03
    minimum_quality_advantage: float = 0.0
    minimum_quality_confidence: float = 0.70
    persistence_window_sessions: int = 3
    persistence_required_sessions: int = 2
    sell_commission: float = 0.001425
    sell_transaction_tax: float = 0.003
    sell_spread_cost: float = 0.001
    sell_slippage: float = 0.001
    buy_commission: float = 0.001425
    buy_spread_cost: float = 0.001
    buy_slippage: float = 0.001
    expected_holding_years: float = 3.0
    model_error_lambda: float = 0.02
    starter_weight_max: float = 0.10
    unknown_correlation_starter_cap: float = 0.05
    strong_add_minimum_weight_gap: float = 0.05
    minimum_delta_portfolio_cer: float = 0.0
    full_replacement_minimum_excess: float = 0.05
    obvious_quality_score: float = 85.0
    obvious_quality_confidence: float = 0.85
    obvious_data_confidence: float = 0.85
    obvious_valuation_confidence: float = 0.85
    obvious_required_return_spread: float = 0.07
    obvious_excess_opportunity: float = 0.05
    buy_small_execution_factor: float = 0.50
    early_buy_execution_factor: float = 0.65
    buy_execution_factor: float = 1.00
    balance_tolerance: float = 0.01
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.maximum_holdings != 4:
            raise ValueError("PAE v2 maximum_holdings must be 4")
        probabilities = (
            self.bear_probability
            + self.base_probability
            + self.bull_probability
        )
        if not np.isclose(probabilities, 1.0):
            raise ValueError("scenario probabilities must sum to 1.0")
        if self.expected_holding_years <= 0:
            raise ValueError("expected_holding_years must be positive")
        if not (
            1
            <= self.persistence_required_sessions
            <= self.persistence_window_sessions
        ):
            raise ValueError("invalid persistence requirement")
        for name in (
            "minimum_cash_weight",
            "absolute_single_stock_max",
            "sector_weight_max",
            "correlation_cluster_weight_max",
            "starter_weight_max",
            "unknown_correlation_starter_cap",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class PortfolioSecurityV2:
    stock_id: str
    sector_id: str = "UNKNOWN"
    correlation_cluster_id: str = "UNKNOWN"
    quality_status: str = "UNKNOWN"
    quality_score: float | None = None
    quality_confidence: float | None = None
    data_confidence: float | None = None
    valuation_status: str = "UNKNOWN"
    valuation_gate: str = "UNKNOWN"
    bear_cagr: float | None = None
    base_cagr: float | None = None
    bull_cagr: float | None = None
    valuation_confidence: float | None = None
    business_uncertainty_penalty: float = 0.0
    les_decision: str = "DATA_INSUFFICIENT"
    risk_gate: str = "UNKNOWN"
    falling_knife_state: str = "UNKNOWN"
    liquidity_status: str = "UNKNOWN"
    executable_shares: float = 0.0
    les_position_cap: float | None = None
    risk_budget_weight: float | None = None
    liquidity_weight_cap: float | None = None
    half_kelly_weight: float | None = None
    erem_decision: str = "DATA_INSUFFICIENT"
    erem_holding_confidence: float | None = None
    erem_data_sufficient: bool = False
    current_weight: float = 0.0
    current_price: float | None = None
    stop_price: float | None = None
    fundamental_hard_exit: bool = False
    strategy_stop_breached: bool = False
    cyclical_state: str = "CYCLICAL_CLEAR"

    def __post_init__(self) -> None:
        if not self.stock_id.strip():
            raise ValueError("stock_id cannot be empty")
        if self.current_weight < 0 or self.current_weight > 1:
            raise ValueError("current_weight must be between 0 and 1")
        if self.executable_shares < 0:
            raise ValueError("executable_shares cannot be negative")
        for name in (
            "les_position_cap",
            "risk_budget_weight",
            "liquidity_weight_cap",
            "half_kelly_weight",
        ):
            value = getattr(self, name)
            if value is not None and not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in (
            "quality_score",
            "quality_confidence",
            "data_confidence",
            "valuation_confidence",
        ):
            value = getattr(self, name)
            if value is not None and not 0 <= float(value) <= 100:
                raise ValueError(f"{name} must be between 0 and 100")


@dataclass(frozen=True)
class PersistenceEvidenceV2:
    incumbent_stock_id: str
    challenger_stock_id: str
    recent_passes: tuple[bool, ...]


@dataclass(frozen=True)
class PersistenceObservationV2:
    incumbent_stock_id: str
    challenger_stock_id: str
    passed: bool


@dataclass(frozen=True)
class PortfolioAllocationRequestV2:
    portfolio_id: str
    decision_time: dt.datetime
    account_equity: float
    available_cash: float
    holdings: tuple[PortfolioSecurityV2, ...] = ()
    candidates: tuple[PortfolioSecurityV2, ...] = ()
    reserved_cash: float = 0.0
    near_term_liability_reserve: float = 0.0
    benchmark_expected_return: float | None = None
    cash_expected_return: float | None = None
    persistence: tuple[PersistenceEvidenceV2, ...] = ()

    def __post_init__(self) -> None:
        if not self.portfolio_id.strip():
            raise ValueError("portfolio_id cannot be empty")
        if self.decision_time.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        if self.account_equity <= 0:
            raise ValueError("account_equity must be positive")
        for name in (
            "available_cash",
            "reserved_cash",
            "near_term_liability_reserve",
        ):
            value = float(getattr(self, name))
            if value < 0 or value > self.account_equity:
                raise ValueError(f"{name} must be within account equity")
        symbols = [item.stock_id for item in self.holdings + self.candidates]
        if len(symbols) != len(set(symbols)):
            raise ValueError("holdings and candidates must have unique symbols")


@dataclass(frozen=True)
class SecurityEvaluationV2:
    stock_id: str
    cer: float | None
    required_return: float
    opportunity_edge: float | None
    capital_claim: float
    strategic_candidate: bool
    executable_candidate: bool
    data_quality: str
    missing_fields: tuple[str, ...]
    blocking_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PortfolioAllocationLineV2:
    stock_id: str
    role: str
    action: PortfolioActionV2
    current_weight: float
    target_weight: float
    evaluation: SecurityEvaluationV2
    incumbent_stock_id: str | None = None
    switch_hurdle: float | None = None
    excess_opportunity: float | None = None
    quality_dominance: bool | None = None
    cer_dominance: bool | None = None
    persistence_pass: bool | None = None
    fast_path_pass: bool | None = None
    portfolio_improvement: bool | None = None
    atomic_group_id: str | None = None
    execution_status: str = "NOT_REQUIRED"
    reason_codes: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ()

    @property
    def weight_change(self) -> float:
        return self.target_weight - self.current_weight


@dataclass(frozen=True)
class PortfolioAllocationDecisionV2:
    schema_version: str
    model_version: str
    portfolio_id: str
    decision_time: dt.datetime
    mode: PortfolioModeV2
    binding: PortfolioBindingV2
    required_return: float
    holding_count_before: int
    holding_count_after: int
    cash_weight_before: float
    cash_weight_after: float
    lines: tuple[PortfolioAllocationLineV2, ...]
    reason_codes: tuple[str, ...]
    input_hash: str
    config_hash: str

    def to_frame(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "Stock Code": line.stock_id,
                    "PAE Status": "SUCCESS",
                    "PAE Mode": self.mode.value,
                    "PAE Binding": self.binding.value,
                    "PAE Role": line.role,
                    "PAE CER": line.evaluation.cer,
                    "PAE Required Return": self.required_return,
                    "PAE Opportunity Edge": line.evaluation.opportunity_edge,
                    "PAE Strategic Candidate": (
                        line.evaluation.strategic_candidate
                    ),
                    "PAE Executable Candidate": (
                        line.evaluation.executable_candidate
                    ),
                    "PAE Capital Claim": line.evaluation.capital_claim,
                    "PAE Current Weight": line.current_weight,
                    "PAE Target Weight": line.target_weight,
                    "PAE Weight Change": line.weight_change,
                    "PAE Target Cash Weight": self.cash_weight_after,
                    "PAE Decision": line.action.value,
                    "PAE Reason": ";".join(line.reason_codes),
                    "PAE Blocking Reasons": ";".join(line.blocking_reasons),
                    "PAE Incumbent": line.incumbent_stock_id,
                    "PAE Switch Hurdle": line.switch_hurdle,
                    "PAE Excess Opportunity": line.excess_opportunity,
                    "PAE Quality Dominance Gate": line.quality_dominance,
                    "PAE CER Dominance Gate": line.cer_dominance,
                    "PAE Persistence Gate": line.persistence_pass,
                    "PAE Obvious Opportunity": line.fast_path_pass,
                    "PAE Portfolio Improvement": line.portfolio_improvement,
                    "PAE Execution Status": line.execution_status,
                    "PAE Atomic Group": line.atomic_group_id,
                    "PAE Data Quality": line.evaluation.data_quality,
                    "PAE Missing Fields": ";".join(
                        line.evaluation.missing_fields
                    ),
                    "PAE Portfolio Reason": ";".join(self.reason_codes),
                    "PAE Model Version": self.model_version,
                    "PAE Schema Version": self.schema_version,
                }
                for line in self.lines
            ]
        )


__all__ = [
    "PAE_V2_MODEL_VERSION",
    "PAE_V2_OUTPUT_COLUMNS",
    "PAE_V2_SCHEMA_VERSION",
    "PAE_V2_STRATEGY_NAME",
    "PersistenceEvidenceV2",
    "PersistenceObservationV2",
    "PortfolioActionV2",
    "PortfolioAllocationDecisionV2",
    "PortfolioAllocationLineV2",
    "PortfolioAllocationRequestV2",
    "PortfolioAllocationV2Config",
    "PortfolioBindingV2",
    "PortfolioModeV2",
    "PortfolioSecurityV2",
    "SecurityEvaluationV2",
    "binding_for_mode",
    "resolve_portfolio_mode_v2",
]
