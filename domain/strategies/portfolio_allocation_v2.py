"""Portfolio Allocation Engine v2 public facade and strategy adapter.

The decision engine delegates calculation and policy work to focused domain
components. Existing imports remain stable for callers and persisted contracts.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from domain.strategy import Strategy
from domain.strategies.portfolio_allocation_dataframe import (
    PortfolioAllocationDataFrameAdapter,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PAE_V2_MODEL_VERSION,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PAE_V2_OUTPUT_COLUMNS,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PAE_V2_SCHEMA_VERSION,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PAE_V2_STRATEGY_NAME,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PersistenceEvidenceV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PersistenceObservationV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioActionV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioAllocationDecisionV2,
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
    PortfolioBindingV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import PortfolioModeV2
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioSecurityV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    SecurityEvaluationV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import binding_for_mode
from domain.strategies.portfolio_allocation_v2_contracts import (
    resolve_portfolio_mode_v2,
)
from domain.strategies.portfolio_allocation_v2_dominance_policy import (
    PortfolioDominancePolicyV2,
)
from domain.strategies.portfolio_allocation_v2_evaluator import (
    PortfolioSecurityEvaluatorV2,
)
from domain.strategies.portfolio_allocation_v2_exit_policy import (
    PortfolioExitPolicyV2,
)
from domain.strategies.portfolio_allocation_v2_lifecycle_policy import (
    PortfolioLifecyclePolicyV2,
)
from domain.strategies.portfolio_allocation_v2_utils import (
    normalized_confidence,
)
from domain.strategies.portfolio_allocation_v2_utils import (
    stable_dataclass_hash,
)
from domain.strategies.portfolio_allocation_v2_weight_allocator import (
    PortfolioWeightAllocatorV2,
)

if TYPE_CHECKING:
    import pandas as pd


class PortfolioAllocationEngineV2:
    """Compose focused PAE v2 policies in the normative decision order."""

    def __init__(
        self,
        config: PortfolioAllocationV2Config | None = None,
        *,
        evaluator: PortfolioSecurityEvaluatorV2 | None = None,
        dominance: PortfolioDominancePolicyV2 | None = None,
        exits: PortfolioExitPolicyV2 | None = None,
        weights: PortfolioWeightAllocatorV2 | None = None,
        lifecycle: PortfolioLifecyclePolicyV2 | None = None,
    ) -> None:
        self.config = config or PortfolioAllocationV2Config()
        self.evaluator = evaluator or PortfolioSecurityEvaluatorV2(self.config)
        self.dominance = dominance or PortfolioDominancePolicyV2(self.config)
        self.exits = exits or PortfolioExitPolicyV2()
        self.weights = weights or PortfolioWeightAllocatorV2(self.config)
        self.lifecycle = lifecycle or PortfolioLifecyclePolicyV2(
            self.config,
            self.evaluator,
            self.dominance,
            self.exits,
            self.weights,
        )

    def allocate(
        self,
        request: PortfolioAllocationRequestV2,
    ) -> PortfolioAllocationDecisionV2:
        self.weights.validate_balance(request)
        mode = resolve_portfolio_mode_v2(len(request.holdings))
        binding = binding_for_mode(mode)
        required_return = self.evaluator.required_return(request)
        evaluations = {
            item.stock_id: self.evaluator.evaluate(item, required_return)
            for item in request.holdings + request.candidates
        }
        holding_lines = [
            self.exits.holding_line(item, evaluations[item.stock_id])
            for item in request.holdings
        ]
        lines, reason_codes = self.lifecycle.build_lines(
            mode,
            request,
            evaluations,
            holding_lines,
        )

        cash_after = self.weights.cash_after(lines)
        holding_count_after = sum(
            line.target_weight > self.config.epsilon for line in lines
        )
        if (
            mode is not PortfolioModeV2.POLICY_BREACH_REDUCE_ONLY
            and holding_count_after > self.config.maximum_holdings
        ):
            raise AssertionError("post-trade holding count exceeds four")
        return PortfolioAllocationDecisionV2(
            schema_version=PAE_V2_SCHEMA_VERSION,
            model_version=PAE_V2_MODEL_VERSION,
            portfolio_id=request.portfolio_id,
            decision_time=request.decision_time,
            mode=mode,
            binding=binding,
            required_return=required_return,
            holding_count_before=len(request.holdings),
            holding_count_after=holding_count_after,
            cash_weight_before=request.available_cash / request.account_equity,
            cash_weight_after=cash_after,
            lines=tuple(lines),
            reason_codes=reason_codes,
            input_hash=stable_dataclass_hash(request),
            config_hash=stable_dataclass_hash(self.config),
        )

    def persistence_observations(
        self,
        request: PortfolioAllocationRequestV2,
    ) -> tuple[PersistenceObservationV2, ...]:
        """Evaluate pair evidence before applying the historical gate."""
        required_return = self.evaluator.required_return(request)
        evaluations = {
            item.stock_id: self.evaluator.evaluate(item, required_return)
            for item in request.holdings + request.candidates
        }
        return self.dominance.persistence_observations(request, evaluations)

    # Compatibility wrappers for existing diagnostic callers.
    def _required_return(self, request: PortfolioAllocationRequestV2) -> float:
        return self.evaluator.required_return(request)

    def _evaluate(
        self,
        security: PortfolioSecurityV2,
        required_return: float,
    ) -> SecurityEvaluationV2:
        return self.evaluator.evaluate(security, required_return)

    @staticmethod
    def _confidence(value: float | None) -> float:
        return normalized_confidence(value)


class PortfolioAllocationV2Strategy(Strategy):
    """Strategy plugin facade for PAE v2."""

    name = PAE_V2_STRATEGY_NAME
    version = PAE_V2_MODEL_VERSION
    description = (
        "Opportunity-cost allocator binding at three or four holdings with "
        "strict open-slot and atomic replacement contracts"
    )

    def __init__(self, config: PortfolioAllocationV2Config | None = None) -> None:
        self.engine = PortfolioAllocationEngineV2(config)

    def allocate(
        self,
        request: PortfolioAllocationRequestV2,
    ) -> PortfolioAllocationDecisionV2:
        return self.engine.allocate(request)

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd

        if data.empty:
            return data.copy()
        account_equity = PortfolioAllocationDataFrameAdapter.frame_number(
            data,
            "Account Equity",
        ) or PortfolioAllocationDataFrameAdapter.optional_number(
            data.attrs.get("account_equity")
        )
        available_cash = PortfolioAllocationDataFrameAdapter.frame_number(
            data,
            "Current Cash",
        )
        if available_cash is None:
            available_cash = PortfolioAllocationDataFrameAdapter.optional_number(
                data.attrs.get("current_cash")
            )
        if account_equity is None or available_cash is None:
            raise ValueError("PAE v2 requires account_equity and current_cash")
        decision_time = self._decision_time(data)
        holdings: list[PortfolioSecurityV2] = []
        candidates: list[PortfolioSecurityV2] = []
        for _, row in data.iterrows():
            security = self._security_from_row(row)
            role = str(row.get("Portfolio Role", "CANDIDATE")).upper()
            if role == "HOLDING":
                holdings.append(security)
            elif role == "CANDIDATE":
                candidates.append(security)
            else:
                raise ValueError("Portfolio Role must be HOLDING or CANDIDATE")
        decision = self.allocate(
            PortfolioAllocationRequestV2(
                portfolio_id=str(data.attrs.get("portfolio_id", "default")),
                decision_time=decision_time,
                account_equity=account_equity,
                available_cash=available_cash,
                holdings=tuple(holdings),
                candidates=tuple(candidates),
                reserved_cash=float(data.attrs.get("reserved_cash", 0.0)),
                near_term_liability_reserve=float(
                    data.attrs.get("near_term_liability_reserve", 0.0)
                ),
            )
        )
        output = decision.to_frame().set_index("Stock Code")
        result = data.copy()
        symbols = result.apply(
            PortfolioAllocationDataFrameAdapter.row_symbol,
            axis=1,
        )
        for column in output.columns:
            result[column] = symbols.map(output[column])
        return result

    @classmethod
    def _security_from_row(cls, row: pd.Series) -> PortfolioSecurityV2:
        base = PortfolioAllocationDataFrameAdapter.security_from_row(row)
        quality_status = {
            "QUALITY_CONFIRMED": "PASS",
            "QUALITY_PROVISIONAL": "PROVISIONAL",
            "QUALITY_WATCH": "WATCH_QUALITY",
            "QUALITY_REJECT": "REJECT",
        }.get(base.quality_status.upper(), base.quality_status.upper())
        liquidity_status = str(row.get("Liquidity Status", "UNKNOWN"))
        return PortfolioSecurityV2(
            stock_id=base.symbol,
            sector_id=base.sector,
            correlation_cluster_id=base.correlation_cluster,
            quality_status=quality_status,
            quality_score=base.quality_score,
            quality_confidence=(
                PortfolioAllocationDataFrameAdapter.row_number(
                    row,
                    "Quality Confidence",
                )
                or base.data_confidence
            ),
            data_confidence=base.data_confidence,
            valuation_status=base.valuation_status,
            valuation_gate=base.valuation_gate,
            bear_cagr=base.bear_cagr,
            base_cagr=base.base_cagr,
            bull_cagr=base.bull_cagr,
            valuation_confidence=base.valuation_confidence,
            business_uncertainty_penalty=base.business_uncertainty_penalty,
            les_decision=base.les_decision,
            risk_gate=base.risk_gate,
            falling_knife_state=base.falling_knife_state,
            liquidity_status=liquidity_status,
            executable_shares=base.executable_shares,
            les_position_cap=base.maximum_weight,
            risk_budget_weight=PortfolioAllocationDataFrameAdapter.row_number(
                row,
                "Risk Budget Weight",
            ),
            liquidity_weight_cap=base.liquidity_weight_limit,
            half_kelly_weight=PortfolioAllocationDataFrameAdapter.row_number(
                row,
                "Half Kelly Weight",
            ),
            erem_decision=base.erem_decision,
            erem_holding_confidence=base.erem_holding_confidence,
            erem_data_sufficient=base.erem_data_sufficient,
            current_weight=base.current_weight,
            current_price=base.current_price,
            stop_price=base.strategy_stop,
            fundamental_hard_exit=base.fundamental_hard_reject,
            strategy_stop_breached=(
                base.current_price is not None
                and base.strategy_stop is not None
                and base.current_price <= base.strategy_stop
            ),
            cyclical_state=str(row.get("Cyclical State", "CYCLICAL_CLEAR")),
        )

    @staticmethod
    def _decision_time(data: pd.DataFrame) -> dt.datetime:
        import pandas as pd

        configured = data.attrs.get("decision_time")
        if configured is not None:
            parsed = dt.datetime.fromisoformat(str(configured))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed
        for name in ("Date", "日期"):
            if name in data:
                values = pd.to_datetime(data[name], errors="coerce").dropna()
                if not values.empty:
                    parsed = values.max().to_pydatetime()
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=dt.timezone.utc)
                    return parsed
        raise ValueError("PAE v2 requires decision_time or Date/日期")


__all__ = [
    "PAE_V2_MODEL_VERSION",
    "PAE_V2_OUTPUT_COLUMNS",
    "PAE_V2_SCHEMA_VERSION",
    "PAE_V2_STRATEGY_NAME",
    "PersistenceEvidenceV2",
    "PersistenceObservationV2",
    "PortfolioActionV2",
    "PortfolioAllocationDecisionV2",
    "PortfolioAllocationEngineV2",
    "PortfolioAllocationLineV2",
    "PortfolioAllocationRequestV2",
    "PortfolioAllocationV2Config",
    "PortfolioAllocationV2Strategy",
    "PortfolioBindingV2",
    "PortfolioModeV2",
    "PortfolioSecurityV2",
    "SecurityEvaluationV2",
    "binding_for_mode",
    "resolve_portfolio_mode_v2",
]
