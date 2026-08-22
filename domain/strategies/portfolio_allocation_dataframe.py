"""Pandas compatibility boundary for Portfolio Allocation Engine v1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.strategies.portfolio_allocation_v1 import AllocationSecurity
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationDecision
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationEngine
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationRequest


def decision_to_frame(decision: PortfolioAllocationDecision) -> pd.DataFrame:
    """Project a typed allocation decision onto the legacy report schema."""

    return pd.DataFrame(
        [
            {
                "Stock Code": line.symbol,
                "PAE Status": "SUCCESS",
                "PAE Mode": decision.mode.value,
                "PAE Portfolio Input": None,
                "PAE Role": line.role,
                "PAE CER": line.evaluation.cer,
                "PAE Quality Score": line.evaluation.quality_score,
                "PAE Required Return": decision.required_return,
                "PAE Opportunity Edge": line.evaluation.opportunity_edge,
                "PAE Absolute Strategic Candidate": (
                    line.evaluation.absolute_strategic_candidate
                ),
                "PAE Strategic Candidate": (
                    line.evaluation.strategic_candidate
                ),
                "PAE Executable Candidate": (
                    line.evaluation.executable_candidate
                ),
                "PAE Dual Dominance Candidate": (
                    line.evaluation.dual_dominance_candidate
                ),
                "PAE Capital Claim": line.evaluation.capital_claim,
                "PAE Best Strategic Alternative": (
                    decision.best_strategic_alternative
                ),
                "PAE Best Executable Alternative": (
                    decision.best_executable_alternative
                ),
                "PAE Alternative": line.alternative_symbol,
                "PAE Alternative CER": line.alternative_cer,
                "PAE Opportunity Gap": line.opportunity_gap,
                "PAE Switch Hurdle": line.switch_hurdle,
                "PAE Replacement Threshold": line.replacement_threshold,
                "PAE Excess Opportunity": line.excess_opportunity,
                "PAE Dominance Incumbent": line.dominance_incumbent_symbol,
                "PAE Incumbent Quality Score": line.incumbent_quality_score,
                "PAE Candidate Quality Score": line.candidate_quality_score,
                "PAE Quality Advantage": line.quality_advantage,
                "PAE Required Quality Advantage": (
                    line.required_quality_advantage
                ),
                "PAE Quality Dominance Gate": line.quality_dominance_gate,
                "PAE CER Dominance Gate": line.cer_dominance_gate,
                "PAE Dual Dominance Gate": line.dual_dominance_gate,
                "PAE Persistence Gate": line.persistence_passed,
                "PAE Obvious Opportunity": line.obvious_opportunity,
                "PAE Portfolio Risk Gate": line.portfolio_risk_gate,
                "PAE High Quality Protected": line.high_quality_protected,
                "PAE Strategy Stop Breached": line.strategy_stop_breached,
                "PAE EREM Exit Persistent": line.erem_exit_persistent,
                "PAE Trend Broken": line.trend_broken,
                "PAE Valuation Broken": line.valuation_broken,
                "PAE Exit Confirmation Count": line.exit_confirmation_count,
                "PAE Current Weight": line.current_weight,
                "PAE Target Weight": line.target_weight,
                "PAE Weight Change": line.weight_change,
                "PAE Target Cash Weight": decision.target_cash_weight,
                "PAE Portfolio Expected CER": decision.portfolio_expected_cer,
                "PAE Decision": line.action.value,
                "PAE Reason": line.reason,
                "PAE Portfolio Reason": decision.reason,
                "PAE Model Version": decision.model_version,
            }
            for line in decision.lines
        ]
    )


class PortfolioAllocationDataFrameAdapter:
    """Translate the Strategy DataFrame API to the typed PAE aggregate API."""

    def __init__(self, engine: PortfolioAllocationEngine) -> None:
        self.engine = engine

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data.copy()
        account_equity = self.frame_number(data, "Account Equity")
        if account_equity is None:
            account_equity = self.optional_number(
                data.attrs.get("account_equity")
            )
        if account_equity is None or account_equity <= 0:
            raise ValueError(
                "portfolio_allocation_v1 requires Account Equity or "
                "data.attrs['account_equity']"
            )
        current_cash = self.frame_number(data, "Current Cash")
        if current_cash is None:
            current_cash = (
                self.optional_number(data.attrs.get("current_cash")) or 0.0
            )

        holdings: list[AllocationSecurity] = []
        candidates: list[AllocationSecurity] = []
        for _, row in data.iterrows():
            security = self.security_from_row(row)
            role = str(row.get("Portfolio Role", "CANDIDATE")).strip().upper()
            if role == "HOLDING":
                holdings.append(security)
            elif role == "CANDIDATE":
                candidates.append(security)
            else:
                raise ValueError("Portfolio Role must be HOLDING or CANDIDATE")
        decision = self.engine.allocate(
            PortfolioAllocationRequest(
                account_equity=account_equity,
                current_cash=current_cash,
                holdings=tuple(holdings),
                candidates=tuple(candidates),
            )
        )
        output = decision_to_frame(decision).set_index("Stock Code")
        result = data.copy()
        symbols = result.apply(self.row_symbol, axis=1)
        for column in output.columns:
            result[column] = symbols.map(output[column])
        return result

    @classmethod
    def security_from_row(cls, row: pd.Series) -> AllocationSecurity:
        symbol = cls.row_symbol(row)
        hard_rejects = str(
            row.get("Quality Hard Rejects", "NONE")
        ).strip().upper()
        return AllocationSecurity(
            symbol=symbol,
            sector=str(row.get("Sector", "UNKNOWN")),
            correlation_cluster=str(
                row.get("Correlation Cluster", "UNKNOWN")
            ),
            quality_status=str(
                row.get(
                    "Quality Status",
                    row.get(
                        "LES v5 Final Quality Status",
                        "DATA_INSUFFICIENT",
                    ),
                )
            ),
            quality_score=cls.row_number(
                row,
                "Quality Score",
                "LES v5 Final Quality Score",
            ),
            data_confidence=cls.row_number(
                row,
                "Data Confidence",
                "Base Data Confidence",
            ),
            valuation_status=str(
                row.get(
                    "Valuation Status",
                    row.get("Valuation Model Status", "UNKNOWN"),
                )
            ),
            valuation_gate=str(row.get("Valuation Gate", "UNKNOWN")),
            bear_cagr=cls.row_number(
                row,
                "Bear CAGR",
                "Bear Case Expected CAGR",
            ),
            base_cagr=cls.row_number(
                row,
                "Base CAGR",
                "Base Case Expected CAGR",
            ),
            bull_cagr=cls.row_number(
                row,
                "Bull CAGR",
                "Bull Case Expected CAGR",
            ),
            valuation_confidence=cls.row_number(row, "Valuation Confidence"),
            les_score=cls.row_number(row, "LES Score", "低點分數"),
            les_decision=str(
                row.get(
                    "LES Decision",
                    row.get("低點決策", "DATA_INSUFFICIENT"),
                )
            ),
            risk_gate=str(row.get("Risk Gate", "UNKNOWN")),
            falling_knife_state=str(
                row.get("Falling Knife State", "DATA_INSUFFICIENT")
            ),
            executable_shares=(
                cls.row_number(row, "Executable Shares", "建議股數") or 0.0
            ),
            erem_score=cls.row_number(row, "EREM Score", "EREM 分數"),
            erem_holding_confidence=cls.row_number(
                row,
                "EREM Holding Confidence",
            ),
            erem_exit_streak=int(
                cls.row_number(row, "EREM Exit Streak Count") or 0
            ),
            erem_ema20=cls.row_number(row, "EREM EMA20"),
            erem_ema20_slope=cls.row_number(row, "EREM EMA20 Slope"),
            erem_distribution_warning=cls.row_optional_bool(
                row,
                "EREM Distribution Warning",
            ),
            erem_higher_low=cls.row_optional_bool(row, "EREM Higher Low"),
            erem_decision=str(
                row.get(
                    "EREM Decision",
                    row.get("EREM 決策訊號", "DATA_INSUFFICIENT"),
                )
            ),
            erem_data_sufficient=(
                cls.row_bool(row, "EREM Data Sufficient")
                if "EREM Data Sufficient" in row.index
                else cls.row_number(row, "EREM Holding Confidence") is not None
            ),
            current_shares=cls.row_number(row, "Current Shares") or 0.0,
            average_cost=cls.row_number(row, "Average Cost"),
            market_value=(
                cls.row_number(
                    row,
                    "Current Market Value",
                    "Market Value",
                )
                or 0.0
            ),
            current_weight=(
                cls.row_number(
                    row,
                    "Current Portfolio Weight",
                    "Current Weight",
                )
                or 0.0
            ),
            current_price=cls.row_number(
                row,
                "Current Price",
                "收盤",
                "Close",
            ),
            strategy_stop=cls.row_number(row, "Strategy Stop", "策略停損"),
            maximum_weight=cls.row_number(row, "Maximum Weight"),
            liquidity_weight_limit=cls.row_number(
                row,
                "Liquidity Weight Limit",
            ),
            business_uncertainty_penalty=(
                cls.row_number(row, "Business Uncertainty Penalty") or 0.0
            ),
            fundamental_hard_reject=(
                cls.row_bool(row, "Fundamental Hard Reject")
                or hard_rejects not in {"", "NONE", "NAN"}
            ),
        )

    @staticmethod
    def row_symbol(row: pd.Series) -> str:
        value = row.get("Stock Code", row.get("證券代號", ""))
        symbol = str(value).strip()
        if not symbol:
            raise ValueError(
                "Each portfolio row requires Stock Code or 證券代號"
            )
        return symbol

    @classmethod
    def row_number(cls, row: pd.Series, *names: str) -> float | None:
        for name in names:
            if name in row.index:
                value = cls.optional_number(row[name])
                if value is not None:
                    return value
        return None

    @staticmethod
    def row_bool(row: pd.Series, name: str) -> bool:
        value = row.get(name, False)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def row_optional_bool(row: pd.Series, name: str) -> bool | None:
        if name not in row.index or pd.isna(row[name]):
            return None
        value = row[name]
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "pass"}:
            return True
        if text in {"0", "false", "no", "n", "fail"}:
            return False
        return None

    @classmethod
    def frame_number(cls, data: pd.DataFrame, name: str) -> float | None:
        if name not in data:
            return None
        values = pd.to_numeric(data[name], errors="coerce").dropna()
        return float(values.iloc[0]) if not values.empty else None

    @staticmethod
    def optional_number(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None


__all__ = [
    "PortfolioAllocationDataFrameAdapter",
    "decision_to_frame",
]
