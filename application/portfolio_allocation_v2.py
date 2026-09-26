"""Production adapter from published stock decisions to PAE v2."""

from __future__ import annotations

import datetime
import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from application.portfolio_contracts import PAE_INPUT_COLD_START
from application.portfolio_contracts import PAE_INPUT_EXPLICIT
from application.portfolio_contracts import PAE_INPUT_MIXED
from application.portfolio_contracts import PAE_INPUT_SYMBOL_ONLY
from application.portfolio_contracts import PortfolioAllocationInputError
from application.portfolio_contracts import PortfolioPersistenceError
from application.portfolio_contracts import PortfolioRuntimeContext
from domain.model_contracts import DecisionField
from domain.model_contracts import FieldSource
from domain.model_contracts import FieldUnit
from domain.model_contracts import FieldValue
from domain.model_contracts import StrategyDecision
from domain.models import PortfolioPosition
from domain.strategies.portfolio_allocation_v2 import PAE_V2_MODEL_VERSION
from domain.strategies.portfolio_allocation_v2 import PAE_V2_OUTPUT_COLUMNS
from domain.strategies.portfolio_allocation_v2 import PAE_V2_SCHEMA_VERSION
from domain.strategies.portfolio_allocation_v2 import PersistenceEvidenceV2
from domain.strategies.portfolio_allocation_v2 import PersistenceObservationV2
from domain.strategies.portfolio_allocation_v2 import PortfolioAllocationDecisionV2
from domain.strategies.portfolio_allocation_v2 import PortfolioAllocationEngineV2
from domain.strategies.portfolio_allocation_v2 import PortfolioAllocationRequestV2
from domain.strategies.portfolio_allocation_v2 import PortfolioBindingV2
from domain.strategies.portfolio_allocation_v2 import PortfolioSecurityV2
from domain.strategies.portfolio_allocation_v2 import binding_for_mode
from domain.strategies.portfolio_allocation_v2 import resolve_portfolio_mode_v2


class PortfolioPersistenceV2Port(Protocol):
    def load_recent_v2(
        self,
        incumbent_symbols: Sequence[str],
        challenger_symbols: Sequence[str],
        *,
        through_date: datetime.date,
        limit_per_pair: int,
    ) -> tuple[PersistenceEvidenceV2, ...]: ...

    def record_session_v2(
        self,
        trade_date: datetime.date,
        observations: Sequence[PersistenceObservationV2],
    ) -> None: ...


class NullPortfolioPersistenceV2:
    def load_recent_v2(
        self,
        incumbent_symbols: Sequence[str],
        challenger_symbols: Sequence[str],
        *,
        through_date: datetime.date,
        limit_per_pair: int,
    ) -> tuple[PersistenceEvidenceV2, ...]:
        return ()

    def record_session_v2(
        self,
        trade_date: datetime.date,
        observations: Sequence[PersistenceObservationV2],
    ) -> None:
        return None


class PortfolioAllocationV2Pipeline:
    """Evaluate PAE v2 once over the complete point-in-time stock universe."""

    def __init__(
        self,
        *,
        engine: PortfolioAllocationEngineV2 | None = None,
        persistence: PortfolioPersistenceV2Port | None = None,
        portfolio_id: str = "default",
    ) -> None:
        self.engine = engine or PortfolioAllocationEngineV2()
        self.persistence = persistence or NullPortfolioPersistenceV2()
        self.portfolio_id = portfolio_id

    def enrich(
        self,
        decisions: Sequence[StrategyDecision],
        context: PortfolioRuntimeContext,
    ) -> tuple[StrategyDecision, ...]:
        base_decisions = tuple(decisions)
        if not base_decisions:
            return ()
        portfolio_at = max(item.evaluated_at for item in base_decisions)
        try:
            request, trade_date, input_mode = self._build_request(
                base_decisions,
                context,
                portfolio_at,
            )
            result = self._evaluate_with_persistence(request, trade_date)
        except (
            PortfolioAllocationInputError,
            PortfolioPersistenceError,
            ValueError,
        ) as error:
            return self._append_unavailable(
                base_decisions,
                context,
                portfolio_at,
                str(error),
            )
        return self._append_result(
            base_decisions,
            result,
            request,
            portfolio_at,
            input_mode,
        )

    def _build_request(
        self,
        decisions: tuple[StrategyDecision, ...],
        context: PortfolioRuntimeContext,
        portfolio_at: datetime.datetime,
    ) -> tuple[PortfolioAllocationRequestV2, datetime.date, str]:
        if context.account_equity <= 0:
            raise PortfolioAllocationInputError("INVALID_ACCOUNT_EQUITY")
        by_symbol = {
            self._text(item.value("證券代號")): item for item in decisions
        }
        if "" in by_symbol:
            raise PortfolioAllocationInputError("MISSING_STOCK_CODE")
        configured = {item.symbol: item for item in context.positions}
        missing = sorted(set(configured) - set(by_symbol))
        if missing:
            raise PortfolioAllocationInputError(
                "CONFIGURED_HOLDINGS_MISSING_FROM_ANALYSIS={}".format(
                    ",".join(missing)
                )
            )

        positions, available_cash, input_mode = self._resolve_positions(
            by_symbol,
            context,
        )
        holdings = tuple(
            self._security(
                by_symbol[position.symbol],
                context,
                position=position,
            )
            for position in positions
        )
        candidates = tuple(
            self._security(item, context, position=None)
            for symbol, item in by_symbol.items()
            if symbol not in configured
        )
        trade_date = max(self._decision_date(item) for item in decisions)
        decision_time = portfolio_at
        if decision_time.tzinfo is None:
            decision_time = decision_time.replace(tzinfo=datetime.timezone.utc)
        return (
            PortfolioAllocationRequestV2(
                portfolio_id=self.portfolio_id,
                decision_time=decision_time,
                account_equity=context.account_equity,
                available_cash=available_cash,
                holdings=holdings,
                candidates=candidates,
            ),
            trade_date,
            input_mode,
        )

    def _resolve_positions(
        self,
        by_symbol: Mapping[str, StrategyDecision],
        context: PortfolioRuntimeContext,
    ) -> tuple[tuple[PortfolioPosition, ...], float, str]:
        if not context.positions:
            cash = (
                context.account_equity
                if context.current_cash is None
                else context.current_cash
            )
            return (), cash, PAE_INPUT_COLD_START

        prices: dict[str, float] = {}
        for position in context.positions:
            price = self._decision_price(by_symbol[position.symbol], "收盤")
            if price is None or price <= 0:
                raise PortfolioAllocationInputError(
                    "HOLDING_{}_MISSING_CURRENT_PRICE".format(position.symbol)
                )
            prices[position.symbol] = price

        explicit = tuple(
            item for item in context.positions if item.shares is not None
        )
        symbol_only = tuple(
            item for item in context.positions if item.shares is None
        )
        explicit_value = sum(
            float(item.shares or 0.0) * prices[item.symbol]
            for item in explicit
        )
        if context.current_cash is not None:
            available_cash = context.current_cash
        elif symbol_only:
            available_cash = (
                context.account_equity * self.engine.config.minimum_cash_weight
            )
        else:
            available_cash = context.account_equity - explicit_value

        tolerance = context.account_equity * self.engine.config.balance_tolerance
        unassigned = context.account_equity - available_cash - explicit_value
        if unassigned < -tolerance:
            raise PortfolioAllocationInputError(
                "HOLDING_MARKET_VALUE_EXCEEDS_ACCOUNT_EQUITY"
            )
        if symbol_only and unassigned <= tolerance:
            raise PortfolioAllocationInputError(
                "NO_CAPITAL_AVAILABLE_FOR_SYMBOL_ONLY_HOLDINGS"
            )
        equal_value = unassigned / len(symbol_only) if symbol_only else 0.0
        resolved = tuple(
            replace(item, shares=equal_value / prices[item.symbol])
            if item.shares is None
            else item
            for item in context.positions
        )
        if explicit and symbol_only:
            input_mode = PAE_INPUT_MIXED
        elif symbol_only:
            input_mode = PAE_INPUT_SYMBOL_ONLY
        else:
            input_mode = PAE_INPUT_EXPLICIT
        return resolved, max(available_cash, 0.0), input_mode

    def _evaluate_with_persistence(
        self,
        request: PortfolioAllocationRequestV2,
        trade_date: datetime.date,
    ) -> PortfolioAllocationDecisionV2:
        if not request.holdings or not request.candidates:
            return self.engine.allocate(request)
        incumbents = tuple(item.stock_id for item in request.holdings)
        challengers = tuple(item.stock_id for item in request.candidates)
        observations = self.engine.persistence_observations(request)
        self.persistence.record_session_v2(trade_date, observations)
        evidence = self.persistence.load_recent_v2(
            incumbents,
            challengers,
            through_date=trade_date,
            limit_per_pair=self.engine.config.persistence_window_sessions,
        )
        return self.engine.allocate(replace(request, persistence=evidence))

    def _security(
        self,
        decision: StrategyDecision,
        context: PortfolioRuntimeContext,
        *,
        position: PortfolioPosition | None,
    ) -> PortfolioSecurityV2:
        symbol = self._text(decision.value("證券代號"))
        price = self._decision_price(decision, "收盤")
        if position is not None and price is None:
            raise PortfolioAllocationInputError(
                "HOLDING_{}_MISSING_CURRENT_PRICE".format(symbol)
            )
        sector_map = context.sector_by_symbol or {}
        sector = self._classification(
            position.sector if position is not None else None,
            sector_map.get(symbol),
            context.default_sector,
        )
        cluster = self._classification(
            position.correlation_cluster if position is not None else None,
            sector,
        )
        base_confidence = self._number(
            decision.value("Base Data Confidence")
        )
        extended_confidence = self._number(
            decision.value("Extended Data Confidence")
        )
        confidences = [
            item
            for item in (base_confidence, extended_confidence)
            if item is not None
        ]
        data_confidence = min(confidences) if confidences else None
        is_v6 = decision.value("LES Model Version") == "6.0.0"
        if is_v6:
            data_confidence = self._number(decision.value("Data Confidence"))
        quality_confidence = self._number(
            decision.value("Quality Confidence")
        )
        if quality_confidence is None:
            quality_confidence = data_confidence
        hard_rejects = self._text(
            decision.value("Quality Hard Rejects", "NONE")
        ).upper()
        quality_status = self._quality_status(
            decision.value("Quality Status" if is_v6 else "LES v5 Final Quality Status")
        )
        if is_v6:
            quality_confidence = data_confidence
        erem_reason = self._text(decision.value("EREM 理由", ""))
        holding_confidence = self._number(
            decision.value("EREM Holding Confidence")
        )
        position_cap = self._positive_number(
            decision.value("Position Fraction")
        )
        recommended_shares = self._number(decision.value("建議股數")) or 0.0
        # LES publishes Position Fraction after its risk and execution caps.
        # Reusing that final cap cannot increase PAE sizing when the optional
        # component-level caps are not yet exposed as separate fields.
        risk_budget = self._positive_number(
            decision.value("Risk Budget Weight")
        ) or position_cap
        liquidity_cap = self._positive_number(
            decision.value("Liquidity Weight Cap")
        ) or position_cap
        current_shares = (
            float(position.shares or 0.0) if position is not None else 0.0
        )
        current_weight = (
            current_shares * float(price or 0.0) / context.account_equity
        )
        stop_price = self._decision_price(decision, "策略停損")
        return PortfolioSecurityV2(
            stock_id=symbol,
            sector_id=sector,
            correlation_cluster_id=cluster,
            quality_status=quality_status,
            quality_score=self._number(
                decision.value("Quality Score" if is_v6 else "LES v5 Final Quality Score")
            ),
            quality_confidence=quality_confidence,
            data_confidence=data_confidence,
            valuation_status=self._text(
                decision.value("Valuation Model Status", "UNKNOWN")
            ),
            valuation_gate=self._text(
                decision.value("Valuation Gate", "UNKNOWN")
            ),
            bear_cagr=self._number(
                decision.value("Bear Case Expected CAGR")
            ),
            base_cagr=self._number(
                decision.value("Base Case Expected CAGR")
            ),
            bull_cagr=self._number(
                decision.value("Bull Case Expected CAGR")
            ),
            valuation_confidence=self._number(
                decision.value("Valuation Confidence")
            ),
            business_uncertainty_penalty=(
                self._number(decision.value("Business Uncertainty Penalty"))
                or 0.0
            ),
            les_decision=self._text(
                decision.value("低點決策", "DATA_INSUFFICIENT")
            ),
            risk_gate=self._text(decision.value("Risk Gate", "UNKNOWN")),
            falling_knife_state=self._text(
                decision.value("Falling Knife State", "UNKNOWN")
            ),
            liquidity_status=self._liquidity_status(
                decision,
                recommended_shares,
            ),
            executable_shares=recommended_shares,
            les_position_cap=position_cap,
            risk_budget_weight=risk_budget,
            liquidity_weight_cap=liquidity_cap,
            half_kelly_weight=self._positive_number(
                decision.value("Half Kelly Weight")
            ),
            erem_decision=self._text(
                decision.value("EREM 決策訊號", "DATA_INSUFFICIENT")
            ),
            erem_holding_confidence=holding_confidence,
            erem_data_sufficient=bool(
                holding_confidence is not None
                and "資料不足" not in erem_reason
                and "DATA_INSUFFICIENT" not in erem_reason.upper()
            ),
            current_weight=current_weight,
            current_price=price,
            stop_price=stop_price,
            fundamental_hard_exit=(
                hard_rejects not in {"", "NONE", "N/A", "NAN"}
            ),
            strategy_stop_breached=bool(
                position is not None
                and price is not None
                and stop_price is not None
                and price <= stop_price
            ),
            cyclical_state=self._text(
                decision.value("Cyclical State", "CYCLICAL_CLEAR")
            ),
        )

    def _append_result(
        self,
        decisions: tuple[StrategyDecision, ...],
        result: PortfolioAllocationDecisionV2,
        request: PortfolioAllocationRequestV2,
        portfolio_at: datetime.datetime,
        input_mode: str,
    ) -> tuple[StrategyDecision, ...]:
        securities = {
            item.stock_id: item for item in request.holdings + request.candidates
        }
        cash_return = (
            self.engine.config.cash_expected_return
            if request.cash_expected_return is None
            else request.cash_expected_return
        )
        portfolio_cer = result.cash_weight_after * cash_return + sum(
            line.target_weight * float(line.evaluation.cer or 0.0)
            for line in result.lines
        )
        rows: dict[str, dict[str, object]] = {}
        for row in result.to_frame().to_dict(orient="records"):
            symbol = str(row["Stock Code"])
            row["PAE Portfolio Input"] = input_mode
            row["PAE Quality Score"] = securities[symbol].quality_score
            row["PAE Portfolio Expected CER"] = portfolio_cer
            rows[symbol] = row
        return tuple(
            self._append_fields(
                item,
                rows[self._text(item.value("證券代號"))],
                portfolio_at,
            )
            for item in decisions
        )

    def _append_unavailable(
        self,
        decisions: tuple[StrategyDecision, ...],
        context: PortfolioRuntimeContext,
        portfolio_at: datetime.datetime,
        reason: str,
    ) -> tuple[StrategyDecision, ...]:
        holdings = {item.symbol for item in context.positions}
        mode = resolve_portfolio_mode_v2(len(holdings))
        binding = binding_for_mode(mode)
        input_mode = self._input_mode(context.positions)
        output: list[StrategyDecision] = []
        for decision in decisions:
            symbol = self._text(decision.value("證券代號"))
            row = {name: None for name in PAE_V2_OUTPUT_COLUMNS}
            row.update(
                {
                    "PAE Status": "DATA_INSUFFICIENT",
                    "PAE Mode": mode.value,
                    "PAE Binding": binding.value,
                    "PAE Portfolio Input": input_mode,
                    "PAE Role": "HOLDING" if symbol in holdings else "CANDIDATE",
                    "PAE Decision": (
                        "HOLD_DATA_INSUFFICIENT"
                        if symbol in holdings
                        else "NO_ACTION"
                    ),
                    "PAE Reason": reason,
                    "PAE Blocking Reasons": reason,
                    "PAE Portfolio Reason": reason,
                    "PAE Model Version": PAE_V2_MODEL_VERSION,
                    "PAE Schema Version": PAE_V2_SCHEMA_VERSION,
                }
            )
            output.append(self._append_fields(decision, row, portfolio_at))
        return tuple(output)

    def _append_fields(
        self,
        decision: StrategyDecision,
        values: Mapping[str, object],
        portfolio_at: datetime.datetime,
    ) -> StrategyDecision:
        existing = tuple(
            field
            for field in decision.fields
            if field.name not in PAE_V2_OUTPUT_COLUMNS
        )
        fields = tuple(
            DecisionField(
                name=name,
                value=self._field_value(values.get(name)),
                source=FieldSource.PAE_V2,
                available_at=portfolio_at,
                confidence=(
                    0.0 if self._field_value(values.get(name)) is None else 100.0
                ),
                unit=self._unit(name),
                model_version=PAE_V2_MODEL_VERSION,
            )
            for name in PAE_V2_OUTPUT_COLUMNS
        )
        return replace(
            decision,
            evaluated_at=portfolio_at,
            fields=existing + fields,
        )

    @staticmethod
    def _quality_status(value: object) -> str:
        text = PortfolioAllocationV2Pipeline._text(value).upper()
        return {
            "QUALITY_CONFIRMED": "PASS",
            "QUALITY_PROVISIONAL": "PROVISIONAL",
            "QUALITY_WATCH": "WATCH_QUALITY",
            "QUALITY_REJECT": "REJECT",
        }.get(text, text or "UNKNOWN")

    @staticmethod
    def _liquidity_status(
        decision: StrategyDecision,
        executable_shares: float,
    ) -> str:
        configured = PortfolioAllocationV2Pipeline._text(
            decision.value("Liquidity Status")
        ).upper()
        if configured in {"PASS", "FAIL", "UNKNOWN"}:
            return configured
        reasons = PortfolioAllocationV2Pipeline._text(
            decision.value("Risk Gate Reasons")
        ).upper()
        if "LIQUIDITY" in reasons:
            return "FAIL"
        return "PASS" if executable_shares > 0 else "UNKNOWN"

    @staticmethod
    def _input_mode(positions: tuple[PortfolioPosition, ...]) -> str:
        if not positions:
            return PAE_INPUT_COLD_START
        explicit = any(item.shares is not None for item in positions)
        symbol_only = any(item.shares is None for item in positions)
        if explicit and symbol_only:
            return PAE_INPUT_MIXED
        return PAE_INPUT_SYMBOL_ONLY if symbol_only else PAE_INPUT_EXPLICIT

    @staticmethod
    def _decision_date(decision: StrategyDecision) -> datetime.date:
        value = str(decision.value("日期", "")).strip()[:10]
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            return decision.evaluated_at.date()

    @staticmethod
    def _decision_price(
        decision: StrategyDecision,
        field_name: str,
    ) -> float | None:
        value = PortfolioAllocationV2Pipeline._number(
            decision.value(field_name)
        )
        if value is None:
            return None
        try:
            unit = decision.field(field_name).unit
        except KeyError:
            unit = FieldUnit.TEXT
        return value / 1000.0 if unit is FieldUnit.TWD_SCALED_1000 else value

    @staticmethod
    def _classification(*values: object) -> str:
        for value in values:
            text = PortfolioAllocationV2Pipeline._text(value)
            if text and text.upper() not in {"N/A", "NONE", "UNKNOWN"}:
                return text
        return "UNKNOWN"

    @staticmethod
    def _text(value: object) -> str:
        return "" if value is None else str(value).strip()

    @staticmethod
    def _number(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    @classmethod
    def _positive_number(cls, value: object) -> float | None:
        parsed = cls._number(value)
        return parsed if parsed is not None and parsed > 0 else None

    @staticmethod
    def _field_value(value: object) -> FieldValue | None:
        if value is None:
            return None
        scalar = value.item() if hasattr(value, "item") else value
        if isinstance(scalar, float) and not math.isfinite(scalar):
            return None
        if isinstance(
            scalar,
            (str, bool, int, float, datetime.date, datetime.datetime),
        ):
            return scalar
        return str(scalar)

    @staticmethod
    def _unit(name: str) -> FieldUnit:
        if "Version" in name:
            return FieldUnit.VERSION
        if name in {
            "PAE Strategic Candidate",
            "PAE Executable Candidate",
            "PAE Quality Dominance Gate",
            "PAE CER Dominance Gate",
            "PAE Persistence Gate",
            "PAE Obvious Opportunity",
            "PAE Portfolio Improvement",
        }:
            return FieldUnit.BOOLEAN
        if name == "PAE Quality Score":
            return FieldUnit.SCORE
        if any(
            token in name
            for token in (
                "CER",
                "Return",
                "Edge",
                "Hurdle",
                "Weight",
                "Capital Claim",
            )
        ):
            return FieldUnit.RATIO
        return FieldUnit.TEXT


__all__ = [
    "NullPortfolioPersistenceV2",
    "PortfolioAllocationV2Pipeline",
    "PortfolioPersistenceV2Port",
]
