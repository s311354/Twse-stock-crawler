"""Versioned data contracts shared by Portfolio Allocation Engine policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


PORTFOLIO_ALLOCATION_STRATEGY_NAME = "portfolio_allocation_v1"
PAE_V1_MODEL_VERSION = "1.2.1"
PORTFOLIO_ALLOCATION_OUTPUT_COLUMNS = [
    "PAE Status",
    "PAE Mode",
    "PAE Portfolio Input",
    "PAE Role",
    "PAE CER",
    "PAE Quality Score",
    "PAE Required Return",
    "PAE Opportunity Edge",
    "PAE Absolute Strategic Candidate",
    "PAE Strategic Candidate",
    "PAE Executable Candidate",
    "PAE Dual Dominance Candidate",
    "PAE Capital Claim",
    "PAE Best Strategic Alternative",
    "PAE Best Executable Alternative",
    "PAE Alternative",
    "PAE Alternative CER",
    "PAE Opportunity Gap",
    "PAE Switch Hurdle",
    "PAE Replacement Threshold",
    "PAE Excess Opportunity",
    "PAE Dominance Incumbent",
    "PAE Incumbent Quality Score",
    "PAE Candidate Quality Score",
    "PAE Quality Advantage",
    "PAE Required Quality Advantage",
    "PAE Quality Dominance Gate",
    "PAE CER Dominance Gate",
    "PAE Dual Dominance Gate",
    "PAE Persistence Gate",
    "PAE Obvious Opportunity",
    "PAE Portfolio Risk Gate",
    "PAE High Quality Protected",
    "PAE Strategy Stop Breached",
    "PAE EREM Exit Persistent",
    "PAE Trend Broken",
    "PAE Valuation Broken",
    "PAE Exit Confirmation Count",
    "PAE Current Weight",
    "PAE Target Weight",
    "PAE Weight Change",
    "PAE Target Cash Weight",
    "PAE Portfolio Expected CER",
    "PAE Decision",
    "PAE Reason",
    "PAE Portfolio Reason",
    "PAE Model Version",
]

class PortfolioMode(str, Enum):
    COLD_START = "COLD_START"
    BUILDING = "BUILDING"
    STEADY_STATE = "STEADY_STATE"


def resolve_portfolio_mode(active_holding_count: int) -> PortfolioMode:
    """Classify the portfolio lifecycle from the current effective holdings."""
    if active_holding_count < 0:
        raise ValueError("active_holding_count must not be negative")
    if active_holding_count == 0:
        return PortfolioMode.COLD_START
    if active_holding_count == 1:
        return PortfolioMode.BUILDING
    return PortfolioMode.STEADY_STATE


class PortfolioAction(str, Enum):
    STRONG_ADD = "STRONG_ADD"
    ADD = "ADD"
    STARTER = "STARTER"
    HOLD = "HOLD"
    HOLD_REVIEW = "HOLD_REVIEW"
    HOLD_NO_ADD = "HOLD_NO_ADD"
    HOLD_DATA_INSUFFICIENT = "HOLD_DATA_INSUFFICIENT"
    WATCH = "WATCH"
    STRATEGIC_WATCH = "STRATEGIC_WATCH"
    REDUCE_25 = "REDUCE_25"
    ROTATE_25 = "ROTATE_25"
    ROTATE_50 = "ROTATE_50"
    ROTATE_75 = "ROTATE_75"
    FAST_TRACK_ROTATE = "FAST_TRACK_ROTATE"
    EXIT = "EXIT"
    EXIT_AND_REALLOCATE = "EXIT_AND_REALLOCATE"
    EXIT_TO_CASH = "EXIT_TO_CASH"
    CASH = "CASH"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"


@dataclass(frozen=True)
class PortfolioAllocationConfig:
    """Versioned policy parameters; rates are decimal fractions."""

    max_stocks: int = 4
    minimum_cash_reserve: float = 0.10
    single_stock_weight_limit: float = 0.35
    # stockdataanalysis.sh evaluates one industry profile per run, so a 50%
    # broad-sector cap would force half the account into cash without creating
    # cross-sector diversification.  Keep the configurable gate for a future
    # multi-industry allocator, but disable it by default for profile runs.
    sector_weight_limit: float = 1.00
    correlation_cluster_weight_limit: float = 0.50
    bear_probability: float = 0.25
    base_probability: float = 0.50
    bull_probability: float = 0.25
    valuation_uncertainty_lambda: float = 0.25
    maximum_data_confidence_penalty: float = 0.03
    absolute_minimum_return: float = 0.10
    benchmark_expected_return: float = 0.07
    required_alpha: float = 0.03
    cash_expected_return: float = 0.02
    equity_risk_premium: float = 0.06
    minimum_candidate_quality_score: float = 70.0
    # A challenger must be no worse in quality. Requiring an arbitrary +3
    # points blocked otherwise superior risk-adjusted return opportunities.
    minimum_quality_advantage: float = 0.0
    # Keep a meaningful model-error buffer so relaxed quality parity does not
    # turn small estimated-return differences into excessive rotation.
    base_replacement_buffer: float = 0.03
    commission_rate_each_side: float = 0.001425
    transaction_tax_rate: float = 0.003
    spread_cost_rate: float = 0.001
    slippage_rate: float = 0.001
    expected_holding_years: float = 3.0
    maximum_model_error_buffer: float = 0.02
    strong_hold_incumbency_premium: float = 0.03
    upper_hold_incumbency_premium: float = 0.02
    lower_hold_incumbency_premium: float = 0.01
    reduce_incumbency_premium: float = 0.005
    buy_small_execution_factor: float = 0.40
    buy_execution_factor: float = 1.00
    persistence_sessions: int = 3
    persistence_required: int = 2
    obvious_quality_score: float = 85.0
    obvious_data_confidence: float = 85.0
    obvious_valuation_confidence: float = 85.0
    obvious_required_return_spread: float = 0.07
    obvious_excess_opportunity: float = 0.05
    quality_protection_score: float = 80.0
    quality_protection_data_confidence: float = 70.0
    valuation_confirmation_confidence: float = 70.0
    valuation_shortfall_confirmation: float = 0.05
    erem_exit_threshold: float = 40.0
    erem_severe_exit_threshold: float = 30.0
    erem_reduce_threshold: float = 55.0
    erem_exit_persistence_required: int = 2
    erem_partial_exit_confirmation_required: int = 2
    erem_full_exit_confirmation_required: int = 3
    portfolio_balance_tolerance: float = 0.01
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        probabilities = (
            self.bear_probability
            + self.base_probability
            + self.bull_probability
        )
        if not np.isclose(probabilities, 1.0):
            raise ValueError("Scenario probabilities must sum to 1.0")
        if self.max_stocks < 1:
            raise ValueError("max_stocks must be positive")
        if self.expected_holding_years <= 0:
            raise ValueError("expected_holding_years must be positive")
        if self.persistence_sessions < 1:
            raise ValueError("persistence_sessions must be positive")
        if not 1 <= self.persistence_required <= self.persistence_sessions:
            raise ValueError(
                "persistence_required must be within persistence_sessions"
            )
        if not 0 <= self.minimum_candidate_quality_score <= 100:
            raise ValueError(
                "minimum_candidate_quality_score must be between 0 and 100"
            )
        if not 0 <= self.minimum_quality_advantage <= 100:
            raise ValueError(
                "minimum_quality_advantage must be between 0 and 100"
            )
        for name in (
            "quality_protection_score",
            "quality_protection_data_confidence",
            "valuation_confirmation_confidence",
            "erem_exit_threshold",
            "erem_severe_exit_threshold",
            "erem_reduce_threshold",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 100:
                raise ValueError("{} must be between 0 and 100".format(name))
        if self.erem_severe_exit_threshold > self.erem_exit_threshold:
            raise ValueError(
                "erem_severe_exit_threshold cannot exceed erem_exit_threshold"
            )
        if self.erem_exit_threshold > self.erem_reduce_threshold:
            raise ValueError(
                "erem_exit_threshold cannot exceed erem_reduce_threshold"
            )
        if self.erem_exit_persistence_required < 1:
            raise ValueError("erem_exit_persistence_required must be positive")
        if not (
            1
            <= self.erem_partial_exit_confirmation_required
            < self.erem_full_exit_confirmation_required
            <= 3
        ):
            raise ValueError(
                "EREM exit confirmation thresholds must satisfy "
                "1 <= partial < full <= 3"
            )
        for name in (
            "minimum_cash_reserve",
            "single_stock_weight_limit",
            "sector_weight_limit",
            "correlation_cluster_weight_limit",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError("{} must be between 0 and 1".format(name))


@dataclass(frozen=True)
class AllocationSecurity:
    """Published upstream decisions consumed by PAE without recomputation."""

    symbol: str
    sector: str = "UNKNOWN"
    correlation_cluster: str = "UNKNOWN"
    quality_status: str = "DATA_INSUFFICIENT"
    quality_score: float | None = None
    data_confidence: float | None = None
    valuation_status: str = "UNKNOWN"
    valuation_gate: str = "UNKNOWN"
    bear_cagr: float | None = None
    base_cagr: float | None = None
    bull_cagr: float | None = None
    valuation_confidence: float | None = None
    les_score: float | None = None
    les_decision: str = "DATA_INSUFFICIENT"
    risk_gate: str = "UNKNOWN"
    falling_knife_state: str = "DATA_INSUFFICIENT"
    executable_shares: float = 0.0
    erem_score: float | None = None
    erem_holding_confidence: float | None = None
    erem_exit_streak: int = 0
    erem_ema20: float | None = None
    erem_ema20_slope: float | None = None
    erem_distribution_warning: bool | None = None
    erem_higher_low: bool | None = None
    erem_decision: str = "DATA_INSUFFICIENT"
    erem_data_sufficient: bool = False
    current_shares: float = 0.0
    average_cost: float | None = None
    market_value: float = 0.0
    current_weight: float = 0.0
    current_price: float | None = None
    strategy_stop: float | None = None
    maximum_weight: float | None = None
    liquidity_weight_limit: float | None = None
    business_uncertainty_penalty: float = 0.0
    fundamental_hard_reject: bool = False

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty")
        for name in ("data_confidence", "valuation_confidence", "quality_score"):
            value = getattr(self, name)
            if value is not None and not 0 <= float(value) <= 100:
                raise ValueError("{} must be between 0 and 100".format(name))
        for name in ("current_weight", "maximum_weight", "liquidity_weight_limit"):
            value = getattr(self, name)
            if value is not None and not 0 <= float(value) <= 1:
                raise ValueError("{} must be between 0 and 1".format(name))
        for name in (
            "current_shares",
            "market_value",
            "executable_shares",
            "business_uncertainty_penalty",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError("{} cannot be negative".format(name))
        if int(self.erem_exit_streak) < 0:
            raise ValueError("erem_exit_streak cannot be negative")


@dataclass(frozen=True)
class PersistenceEvidence:
    incumbent_symbol: str
    challenger_symbol: str
    recent_excess_opportunities: tuple[float, ...]


@dataclass(frozen=True)
class PortfolioAllocationRequest:
    account_equity: float
    current_cash: float
    holdings: tuple[AllocationSecurity, ...] = ()
    candidates: tuple[AllocationSecurity, ...] = ()
    persistence: tuple[PersistenceEvidence, ...] = ()
    benchmark_expected_return: float | None = None
    cash_expected_return: float | None = None

    def __post_init__(self) -> None:
        if self.account_equity <= 0:
            raise ValueError("account_equity must be positive")
        if self.current_cash < 0:
            raise ValueError("current_cash cannot be negative")
        if self.current_cash > self.account_equity:
            raise ValueError("current_cash cannot exceed account_equity")
        symbols = [item.symbol for item in self.holdings + self.candidates]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Holdings and candidates must have unique symbols")


@dataclass(frozen=True)
class DominanceAssessment:
    """Pairwise quality and CER evidence for one challenger and incumbent."""

    incumbent_symbol: str
    challenger_symbol: str
    incumbent_quality_score: float | None
    challenger_quality_score: float | None
    quality_advantage: float | None
    required_quality_advantage: float
    quality_gate: str
    opportunity_gap: float | None
    switch_hurdle: float | None
    replacement_threshold: float | None
    excess_opportunity: float | None
    cer_gate: str
    dual_gate: str


@dataclass(frozen=True)
class ExitConfirmationAssessment:
    """Independent evidence used to confirm a soft EREM exit warning."""

    high_quality_protected: bool = False
    strategy_stop_breached: bool = False
    erem_exit_persistent: bool = False
    trend_broken: bool = False
    valuation_broken: bool = False

    @property
    def confirmation_count(self) -> int:
        return sum(
            (
                self.erem_exit_persistent,
                self.trend_broken,
                self.valuation_broken,
            )
        )


@dataclass(frozen=True)
class AllocationLineDecision:
    symbol: str
    role: str
    action: PortfolioAction
    evaluation: SecurityEvaluation
    current_weight: float
    target_weight: float
    alternative_symbol: str | None = None
    alternative_cer: float | None = None
    opportunity_gap: float | None = None
    switch_hurdle: float | None = None
    replacement_threshold: float | None = None
    excess_opportunity: float | None = None
    dominance_incumbent_symbol: str | None = None
    incumbent_quality_score: float | None = None
    candidate_quality_score: float | None = None
    quality_advantage: float | None = None
    required_quality_advantage: float | None = None
    quality_dominance_gate: str = "NOT_APPLICABLE"
    cer_dominance_gate: str = "NOT_APPLICABLE"
    dual_dominance_gate: str = "NOT_APPLICABLE"
    persistence_passed: bool = False
    obvious_opportunity: bool = False
    portfolio_risk_gate: str = "PASS"
    high_quality_protected: bool = False
    strategy_stop_breached: bool = False
    erem_exit_persistent: bool = False
    trend_broken: bool = False
    valuation_broken: bool = False
    exit_confirmation_count: int = 0
    reason: str = ""

    @property
    def weight_change(self) -> float:
        return self.target_weight - self.current_weight


@dataclass(frozen=True)
class PortfolioAllocationDecision:
    mode: PortfolioMode
    model_version: str
    required_return: float
    best_strategic_alternative: str | None
    best_executable_alternative: str | None
    lines: tuple[AllocationLineDecision, ...]
    target_cash_weight: float
    portfolio_expected_cer: float | None
    sector_weights: tuple[tuple[str, float], ...]
    correlation_cluster_weights: tuple[tuple[str, float], ...]
    reason: str

    def to_frame(self) -> pd.DataFrame:
        from domain.strategies.portfolio_allocation_dataframe import (
            decision_to_frame,
        )

        return decision_to_frame(self)


__all__ = [
    "AllocationLineDecision",
    "AllocationSecurity",
    "DominanceAssessment",
    "ExitConfirmationAssessment",
    "PAE_V1_MODEL_VERSION",
    "PORTFOLIO_ALLOCATION_OUTPUT_COLUMNS",
    "PORTFOLIO_ALLOCATION_STRATEGY_NAME",
    "PersistenceEvidence",
    "PortfolioAction",
    "PortfolioAllocationConfig",
    "PortfolioAllocationDecision",
    "PortfolioAllocationRequest",
    "PortfolioMode",
    "resolve_portfolio_mode",
]
