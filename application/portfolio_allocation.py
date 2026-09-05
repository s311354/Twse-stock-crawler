"""Application pipeline that converts daily decisions into PAE v1 outputs."""

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
from application.portfolio_contracts import PersistenceObservation
from application.portfolio_contracts import PortfolioAllocationInputError
from application.portfolio_contracts import PortfolioPersistenceError
from application.portfolio_contracts import PortfolioRuntimeContext
from domain.model_contracts import DecisionField
from domain.model_contracts import FieldSource
from domain.model_contracts import FieldUnit
from domain.model_contracts import FieldValue
from domain.model_contracts import StrategyDecision
from domain.models import PortfolioPosition
from domain.strategies.portfolio_allocation_v1 import AllocationSecurity
from domain.strategies.portfolio_allocation_v1 import PAE_V1_MODEL_VERSION
from domain.strategies.portfolio_allocation_v1 import PersistenceEvidence
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationDecision
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationEngine
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationRequest
from domain.strategies.portfolio_allocation_v1 import resolve_portfolio_mode
from domain.strategies.portfolio_allocation_v1 import (
    PORTFOLIO_ALLOCATION_OUTPUT_COLUMNS,
)


class PortfolioPersistencePort(Protocol):
    def load_recent(
        self,
        incumbent_symbols: Sequence[str],
        challenger_symbols: Sequence[str],
        *,
        through_date: datetime.date,
        limit_per_pair: int,
    ) -> tuple[PersistenceEvidence, ...]:
        """Load completed-session evidence for all requested symbol pairs."""

    def record_session(
        self,
        trade_date: datetime.date,
        observations: Sequence[PersistenceObservation],
    ) -> None:
        """Atomically upsert one completed session of pair evidence."""


class NullPortfolioPersistence:
    def load_recent(
        self,
        incumbent_symbols: Sequence[str],
        challenger_symbols: Sequence[str],
        *,
        through_date: datetime.date,
        limit_per_pair: int,
    ) -> tuple[PersistenceEvidence, ...]:
        return ()

    def record_session(
        self,
        trade_date: datetime.date,
        observations: Sequence[PersistenceObservation],
    ) -> None:
        return None


class PortfolioAllocationPipeline:
    """Evaluate PAE once across the complete daily stock universe."""

    def __init__(
        self,
        *,
        engine: PortfolioAllocationEngine | None = None,
        persistence: PortfolioPersistencePort | None = None,
    ) -> None:
        self.engine = engine or PortfolioAllocationEngine()
        self.persistence = persistence or NullPortfolioPersistence()

    def enrich(
        self,
        decisions: Sequence[StrategyDecision],
        context: PortfolioRuntimeContext,
    ) -> tuple[StrategyDecision, ...]:
        base_decisions = tuple(decisions)
        if not base_decisions:
            return ()
        portfolio_at = max(decision.evaluated_at for decision in base_decisions)
        try:
            request, trade_date, input_mode = self._build_request(
                base_decisions,
                context,
            )
            final = self._evaluate_with_persistence(request, trade_date)
        except (PortfolioAllocationInputError, PortfolioPersistenceError, ValueError) as error:
            return self._append_unavailable(
                base_decisions,
                context,
                portfolio_at,
                str(error),
            )
        return self._append_result(
            base_decisions,
            final,
            portfolio_at,
            input_mode,
        )

    def _build_request(
        self,
        decisions: tuple[StrategyDecision, ...],
        context: PortfolioRuntimeContext,
    ) -> tuple[PortfolioAllocationRequest, datetime.date, str]:
        if context.account_equity <= 0:
            raise PortfolioAllocationInputError("INVALID_ACCOUNT_EQUITY")
        by_symbol = {
            self._text(decision.value("證券代號")): decision
            for decision in decisions
        }
        if "" in by_symbol:
            raise PortfolioAllocationInputError("MISSING_STOCK_CODE")
        positions = {position.symbol: position for position in context.positions}
        missing = sorted(set(positions) - set(by_symbol))
        if missing:
            raise PortfolioAllocationInputError(
                "CONFIGURED_HOLDINGS_MISSING_FROM_ANALYSIS={}".format(
                    ",".join(missing)
                )
            )

        resolved_positions, current_cash, input_mode = self._resolve_positions(
            by_symbol,
            context,
        )
        holdings = tuple(
            self._security(
                by_symbol[position.symbol],
                context,
                position=position,
            )
            for position in resolved_positions
        )
        candidates = tuple(
            self._security(decision, context, position=None)
            for symbol, decision in by_symbol.items()
            if symbol not in positions
        )
        trade_date = max(self._decision_date(item) for item in decisions)
        return (
            PortfolioAllocationRequest(
                account_equity=context.account_equity,
                current_cash=current_cash,
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
        """Value explicit shares or synthesize a declared equal-weight baseline."""
        if not context.positions:
            current_cash = (
                context.account_equity
                if context.current_cash is None
                else context.current_cash
            )
            return (), current_cash, PAE_INPUT_COLD_START

        priced: dict[str, float] = {}
        for position in context.positions:
            price = self._current_price(by_symbol[position.symbol])
            if price is None or price <= 0:
                raise PortfolioAllocationInputError(
                    "HOLDING_{}_MISSING_CURRENT_PRICE".format(position.symbol)
                )
            priced[position.symbol] = price

        explicit = tuple(
            position for position in context.positions if position.shares is not None
        )
        symbol_only = tuple(
            position for position in context.positions if position.shares is None
        )
        explicit_value = sum(
            float(position.shares or 0.0) * priced[position.symbol]
            for position in explicit
        )
        if context.current_cash is not None:
            current_cash = context.current_cash
        elif symbol_only:
            current_cash = (
                context.account_equity * self.engine.config.minimum_cash_reserve
            )
        else:
            current_cash = context.account_equity - explicit_value

        tolerance = (
            context.account_equity * self.engine.config.portfolio_balance_tolerance
        )
        available = context.account_equity - current_cash - explicit_value
        if available < -tolerance:
            raise PortfolioAllocationInputError(
                "HOLDING_MARKET_VALUE_EXCEEDS_ACCOUNT_EQUITY"
            )
        if symbol_only and available <= tolerance:
            raise PortfolioAllocationInputError(
                "NO_CAPITAL_AVAILABLE_FOR_SYMBOL_ONLY_HOLDINGS"
            )

        equal_value = available / len(symbol_only) if symbol_only else 0.0
        resolved = tuple(
            replace(
                position,
                shares=equal_value / priced[position.symbol],
            )
            if position.shares is None
            else position
            for position in context.positions
        )
        if explicit and symbol_only:
            input_mode = PAE_INPUT_MIXED
        elif symbol_only:
            input_mode = PAE_INPUT_SYMBOL_ONLY
        else:
            input_mode = PAE_INPUT_EXPLICIT
        return resolved, max(0.0, current_cash), input_mode

    def _evaluate_with_persistence(
        self,
        request: PortfolioAllocationRequest,
        trade_date: datetime.date,
    ) -> PortfolioAllocationDecision:
        if not request.holdings or not request.candidates:
            return self.engine.allocate(request)

        incumbents = tuple(item.symbol for item in request.holdings)
        challengers = tuple(item.symbol for item in request.candidates)
        prior = self.persistence.load_recent(
            incumbents,
            challengers,
            through_date=trade_date - datetime.timedelta(days=1),
            limit_per_pair=self.engine.config.persistence_sessions,
        )
        preliminary = self.engine.allocate(replace(request, persistence=prior))
        observations = tuple(
            PersistenceObservation(
                incumbent_symbol=line.symbol,
                challenger_symbol=str(line.alternative_symbol),
                excess_opportunity=float(line.excess_opportunity),
            )
            for line in preliminary.lines
            if line.role == "HOLDING"
            and line.alternative_symbol
            and line.excess_opportunity is not None
        )
        if not observations:
            return preliminary
        self.persistence.record_session(trade_date, observations)
        current = self.persistence.load_recent(
            incumbents,
            challengers,
            through_date=trade_date,
            limit_per_pair=self.engine.config.persistence_sessions,
        )
        return self.engine.allocate(replace(request, persistence=current))

    def _security(
        self,
        decision: StrategyDecision,
        context: PortfolioRuntimeContext,
        *,
        position: PortfolioPosition | None,
    ) -> AllocationSecurity:
        symbol = self._text(decision.value("證券代號"))
        current_price = self._current_price(decision)
        if position is not None and current_price is None:
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
        base_confidence = self._number(decision.value("Base Data Confidence"))
        extended_confidence = self._number(
            decision.value("Extended Data Confidence")
        )
        confidences = [
            value
            for value in (base_confidence, extended_confidence)
            if value is not None
        ]
        data_confidence = min(confidences) if confidences else None
        hard_rejects = self._text(
            decision.value("Quality Hard Rejects", "NONE")
        ).upper()
        quality_status = self._text(
            decision.value(
                "LES v5 Final Quality Status",
                "DATA_INSUFFICIENT",
            )
        )
        erem_reason = self._text(decision.value("EREM 理由", ""))
        erem_holding_confidence = self._number(
            decision.value("EREM Holding Confidence")
        )
        position_fraction = self._number(decision.value("Position Fraction"))
        current_shares = (
            float(position.shares)
            if position is not None and position.shares is not None
            else 0.0
        )
        market_value = current_shares * float(current_price or 0.0)
        return AllocationSecurity(
            symbol=symbol,
            sector=sector,
            correlation_cluster=cluster,
            quality_status=quality_status,
            quality_score=self._number(
                decision.value("LES v5 Final Quality Score")
            ),
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
            les_score=self._number(decision.value("低點分數")),
            les_decision=self._text(
                decision.value("低點決策", "DATA_INSUFFICIENT")
            ),
            risk_gate=self._text(decision.value("Risk Gate", "UNKNOWN")),
            falling_knife_state=self._text(
                decision.value("Falling Knife State", "DATA_INSUFFICIENT")
            ),
            executable_shares=self._number(decision.value("建議股數")) or 0.0,
            erem_score=self._number(decision.value("EREM 分數")),
            erem_holding_confidence=erem_holding_confidence,
            erem_exit_streak=int(
                self._number(decision.value("EREM Exit Streak Count")) or 0
            ),
            erem_ema20=self._decision_price(decision, "EREM EMA20"),
            erem_ema20_slope=self._number(
                decision.value("EREM EMA20 Slope")
            ),
            erem_distribution_warning=self._optional_bool(
                decision.value("EREM Distribution Warning")
            ),
            erem_higher_low=self._optional_bool(
                decision.value("EREM Higher Low")
            ),
            erem_decision=self._text(
                decision.value("EREM 決策訊號", "DATA_INSUFFICIENT")
            ),
            erem_data_sufficient=bool(
                erem_holding_confidence is not None
                and "資料不足" not in erem_reason
                and "DATA_INSUFFICIENT" not in erem_reason.upper()
            ),
            current_shares=current_shares,
            average_cost=(position.average_cost if position is not None else None),
            market_value=market_value,
            current_weight=(market_value / context.account_equity),
            current_price=current_price,
            strategy_stop=self._decision_price(decision, "策略停損"),
            maximum_weight=(
                position.maximum_weight
                if position is not None
                else (
                    position_fraction
                    if position_fraction is not None and position_fraction > 0
                    else None
                )
            ),
            fundamental_hard_reject=(
                hard_rejects not in {"", "NONE", "N/A", "NAN"}
            ),
        )

    def _append_result(
        self,
        decisions: tuple[StrategyDecision, ...],
        result: PortfolioAllocationDecision,
        portfolio_at: datetime.datetime,
        input_mode: str,
    ) -> tuple[StrategyDecision, ...]:
        rows = {}
        for row in result.to_frame().to_dict(orient="records"):
            row["PAE Portfolio Input"] = input_mode
            rows[str(row["Stock Code"])] = row
        return tuple(
            self._append_fields(decision, rows[self._text(decision.value("證券代號"))], portfolio_at)
            for decision in decisions
        )

    def _append_unavailable(
        self,
        decisions: tuple[StrategyDecision, ...],
        context: PortfolioRuntimeContext,
        portfolio_at: datetime.datetime,
        reason: str,
    ) -> tuple[StrategyDecision, ...]:
        holdings = {position.symbol for position in context.positions}
        input_mode = self._input_mode(context.positions)
        mode = resolve_portfolio_mode(len(holdings)).value
        target_cash = (
            context.current_cash / context.account_equity
            if context.current_cash is not None and context.account_equity > 0
            else (1.0 if not holdings else None)
        )
        output: list[StrategyDecision] = []
        for decision in decisions:
            symbol = self._text(decision.value("證券代號"))
            row = {name: None for name in PORTFOLIO_ALLOCATION_OUTPUT_COLUMNS}
            row.update(
                {
                    "PAE Status": "DATA_INSUFFICIENT",
                    "PAE Mode": mode,
                    "PAE Portfolio Input": input_mode,
                    "PAE Role": "HOLDING" if symbol in holdings else "CANDIDATE",
                    "PAE Target Cash Weight": target_cash,
                    "PAE Decision": (
                        "HOLD_DATA_INSUFFICIENT"
                        if symbol in holdings
                        else "DATA_INSUFFICIENT"
                    ),
                    "PAE Reason": reason,
                    "PAE Portfolio Reason": reason,
                    "PAE Model Version": PAE_V1_MODEL_VERSION,
                }
            )
            output.append(self._append_fields(decision, row, portfolio_at))
        return tuple(output)

    @staticmethod
    def _input_mode(positions: tuple[PortfolioPosition, ...]) -> str:
        if not positions:
            return PAE_INPUT_COLD_START
        has_explicit = any(position.shares is not None for position in positions)
        has_symbol_only = any(position.shares is None for position in positions)
        if has_explicit and has_symbol_only:
            return PAE_INPUT_MIXED
        if has_symbol_only:
            return PAE_INPUT_SYMBOL_ONLY
        return PAE_INPUT_EXPLICIT

    def _append_fields(
        self,
        decision: StrategyDecision,
        values: Mapping[str, object],
        portfolio_at: datetime.datetime,
    ) -> StrategyDecision:
        existing = tuple(
            field
            for field in decision.fields
            if field.name not in PORTFOLIO_ALLOCATION_OUTPUT_COLUMNS
        )
        pae_fields = tuple(
            DecisionField(
                name=name,
                value=self._field_value(values.get(name)),
                source=FieldSource.PAE_V1,
                available_at=portfolio_at,
                confidence=(0.0 if self._field_value(values.get(name)) is None else 100.0),
                unit=self._unit(name),
                model_version=PAE_V1_MODEL_VERSION,
            )
            for name in PORTFOLIO_ALLOCATION_OUTPUT_COLUMNS
        )
        return replace(
            decision,
            evaluated_at=portfolio_at,
            fields=existing + pae_fields,
        )

    @staticmethod
    def _current_price(decision: StrategyDecision) -> float | None:
        return PortfolioAllocationPipeline._decision_price(decision, "收盤")

    @staticmethod
    def _decision_price(
        decision: StrategyDecision,
        field_name: str,
    ) -> float | None:
        value = PortfolioAllocationPipeline._number(decision.value(field_name))
        if value is None:
            return None
        try:
            unit = decision.field(field_name).unit
        except KeyError:
            unit = FieldUnit.TEXT
        if unit is FieldUnit.TWD_SCALED_1000:
            return value / 1000.0
        return value

    @staticmethod
    def _decision_date(decision: StrategyDecision) -> datetime.date:
        value = str(decision.value("日期", "")).strip()[:10]
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            return decision.evaluated_at.date()

    @staticmethod
    def _classification(*values: object) -> str:
        for value in values:
            text = PortfolioAllocationPipeline._text(value)
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

    @staticmethod
    def _optional_bool(value: object) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in {"true", "yes", "pass"}:
                return True
            if text in {"false", "no", "fail"}:
                return False
            return None
        return bool(numeric) if math.isfinite(numeric) else None

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
        if any(token in name for token in ("Candidate", "Gate", "Opportunity")) and name in {
            "PAE Absolute Strategic Candidate",
            "PAE Strategic Candidate",
            "PAE Executable Candidate",
            "PAE Dual Dominance Candidate",
            "PAE Persistence Gate",
            "PAE Obvious Opportunity",
        }:
            return FieldUnit.BOOLEAN
        if name in {
            "PAE High Quality Protected",
            "PAE Strategy Stop Breached",
            "PAE EREM Exit Persistent",
            "PAE Trend Broken",
            "PAE Valuation Broken",
        }:
            return FieldUnit.BOOLEAN
        if name == "PAE Exit Confirmation Count":
            return FieldUnit.COUNT
        if name in {
            "PAE Quality Score",
            "PAE Incumbent Quality Score",
            "PAE Candidate Quality Score",
            "PAE Quality Advantage",
            "PAE Required Quality Advantage",
        }:
            return FieldUnit.SCORE
        if any(
            token in name
            for token in (
                "CER",
                "Return",
                "Edge",
                "Gap",
                "Hurdle",
                "Threshold",
                "Weight",
                "Capital Claim",
            )
        ):
            return FieldUnit.RATIO
        return FieldUnit.TEXT


__all__ = [
    "NullPortfolioPersistence",
    "PAE_INPUT_COLD_START",
    "PAE_INPUT_EXPLICIT",
    "PAE_INPUT_MIXED",
    "PAE_INPUT_SYMBOL_ONLY",
    "PersistenceObservation",
    "PortfolioAllocationInputError",
    "PortfolioAllocationPipeline",
    "PortfolioPersistenceError",
    "PortfolioPersistencePort",
    "PortfolioRuntimeContext",
]
