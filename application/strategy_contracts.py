"""Application boundary for assembling and validating model decisions."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import pandas as pd

from application.market_history import MarketHistoryInput
from domain.model_contracts import DecisionField
from domain.model_contracts import FieldSource
from domain.model_contracts import FieldUnit
from domain.model_contracts import FieldValue
from domain.model_contracts import StrategyDecision
from domain.strategies.expected_return_exit_v3 import EREM_V3_MODEL_VERSION


DAILY_ANALYSIS_CONTRACT_NAME = "daily_analysis_v1"


class StrategyContractError(ValueError):
    """Raised when a plugin result cannot satisfy the report contract."""


@dataclass(frozen=True)
class StrategyDecisionSchema:
    """Required fields and cross-field invariants for daily decisions."""

    required_fields: tuple[str, ...] = (
        "日期",
        "證券代號",
        "證券名稱",
        "收盤",
        "LES v5 Final Quality Status",
        "Base Quality Status",
        "Base Available Weight",
        "Valuation Gate",
        "Valuation Model Status",
        "Valuation Model Version",
        "LES Model Version",
        "低點決策",
        "EREM 決策訊號",
    )
    complete_valuation_fields: tuple[str, ...] = (
        "Valuation Confidence",
        "Bear Case Expected CAGR",
        "Base Case Expected CAGR",
        "Bull Case Expected CAGR",
        "Base Case Fair Value",
        "Bear Case Fair Value",
        "Bear Case Permanent Loss",
    )
    non_null_fields: tuple[str, ...] = (
        "證券代號",
        "證券名稱",
        "收盤",
        "LES v5 Final Quality Status",
        "Base Quality Status",
        "Base Available Weight",
        "Valuation Gate",
        "Valuation Model Status",
        "Valuation Model Version",
        "LES Model Version",
        "低點決策",
        "EREM 決策訊號",
    )

    def validate(self, decision: StrategyDecision) -> None:
        names = {field.name for field in decision.fields}
        missing = [
            name for name in self.required_fields if name not in names
        ]
        if missing:
            raise StrategyContractError(
                "StrategyDecision is missing required fields: {}".format(
                    ", ".join(missing)
                )
            )

        null_fields = [
            name
            for name in self.non_null_fields
            if decision.value(name) is None
        ]
        if null_fields:
            raise StrategyContractError(
                "StrategyDecision has null contract fields: {}".format(
                    ", ".join(null_fields)
                )
            )

        future_fields = [
            field.name
            for field in decision.fields
            if field.available_at > decision.evaluated_at
        ]
        if future_fields:
            raise StrategyContractError(
                "StrategyDecision contains future point-in-time evidence: {}"
                .format(", ".join(sorted(future_fields)))
            )

        self._validate_base_coverage(decision)
        self._validate_valuation(decision)

    @staticmethod
    def _validate_base_coverage(decision: StrategyDecision) -> None:
        if decision.value("Base Quality Status") != "BASE_PASS":
            return
        coverage = decision.value("Base Available Weight")
        if coverage is None:
            raise StrategyContractError(
                "BASE_PASS requires Base Available Weight evidence"
            )
        if float(coverage) < 70.0:
            raise StrategyContractError(
                "BASE_PASS cannot use Base Available Weight below 70"
            )

    def _validate_valuation(self, decision: StrategyDecision) -> None:
        status = decision.value("Valuation Model Status")
        entry_decision = decision.value("低點決策")
        if entry_decision == "BUY":
            if status != "COMPLETE":
                raise StrategyContractError(
                    "BUY decisions require complete valuation evidence"
                )
            if decision.value("Valuation Gate") != "PASS":
                raise StrategyContractError(
                    "BUY decisions require Valuation Gate=PASS"
                )
        if status != "COMPLETE":
            return
        missing = [
            name
            for name in self.complete_valuation_fields
            if decision.value(name) is None
        ]
        if missing:
            raise StrategyContractError(
                "COMPLETE valuation is missing scenario evidence: {}".format(
                    ", ".join(missing)
                )
            )
        version = decision.field("Valuation Model Version")
        if version.source is not FieldSource.LES_V5_VALUATION:
            raise StrategyContractError(
                "Valuation Model Version must come from LES_V5_VALUATION"
            )


class StrategyDecisionAssembler:
    """Convert one validated analysis frame into immutable decisions."""

    def __init__(
        self,
        schema: StrategyDecisionSchema | None = None,
    ) -> None:
        self.schema = schema or StrategyDecisionSchema()

    def assemble(
        self,
        dataset: pd.DataFrame,
        market: MarketHistoryInput,
    ) -> tuple[StrategyDecision, ...]:
        if len(dataset) != market.stock_count:
            raise StrategyContractError(
                "Analysis dataset must match market stock_count"
            )

        decisions: list[StrategyDecision] = []
        for item, (_, row) in enumerate(dataset.iterrows()):
            symbol = str(market.stock_numbers[item]).strip()
            evaluated_at = self._evaluated_at(market, item, symbol)
            snapshot = market.fundamental_snapshots.get(symbol, {})
            extended_at = self._timestamp(
                snapshot.get("available_at"),
                fallback=evaluated_at,
            )
            base_at = self._timestamp(
                snapshot.get("twse_financial_observed_at"),
                fallback=extended_at,
            )
            market_at = self._market_available_at(market, item)
            base_source = self._base_source(snapshot)
            fields = tuple(
                self._build_field(
                    name=str(name),
                    raw_value=value,
                    row=row,
                    evaluated_at=evaluated_at,
                    market_at=market_at,
                    base_at=base_at,
                    extended_at=extended_at,
                    base_source=base_source,
                    has_fundamental_snapshot=bool(snapshot),
                )
                for name, value in row.items()
            )
            decision = StrategyDecision(
                strategy_name=DAILY_ANALYSIS_CONTRACT_NAME,
                evaluated_at=evaluated_at,
                fields=fields,
            )
            self.schema.validate(decision)
            decisions.append(decision)
        return tuple(decisions)

    @staticmethod
    def project(
        decisions: Sequence[StrategyDecision],
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if not decisions:
            return pd.DataFrame(columns=list(columns or ()))
        output_columns = list(
            columns or [field.name for field in decisions[0].fields]
        )
        rows = [
            {
                name: decision.value(name)
                for name in output_columns
            }
            for decision in decisions
        ]
        return pd.DataFrame(rows, columns=output_columns)

    def _build_field(
        self,
        *,
        name: str,
        raw_value: object,
        row: pd.Series,
        evaluated_at: datetime.datetime,
        market_at: datetime.datetime,
        base_at: datetime.datetime,
        extended_at: datetime.datetime,
        base_source: FieldSource,
        has_fundamental_snapshot: bool,
    ) -> DecisionField[FieldValue]:
        value = self._normalize_value(raw_value)
        source = self._source_for(
            name,
            base_source,
            has_fundamental_snapshot,
        )
        available_at = self._available_at_for(
            name=name,
            source=source,
            evaluated_at=evaluated_at,
            market_at=market_at,
            base_at=base_at,
            extended_at=extended_at,
        )
        return DecisionField(
            name=name,
            value=value,
            source=source,
            available_at=available_at,
            confidence=self._confidence_for(name, row, value),
            unit=self._unit_for(name),
            model_version=self._model_version_for(name, row, source),
        )

    @classmethod
    def _evaluated_at(
        cls,
        market: MarketHistoryInput,
        item: int,
        symbol: str,
    ) -> datetime.datetime:
        signal_date = market.signal_dates.get(symbol)
        if signal_date:
            return cls._timestamp(signal_date)
        dates = market.dates[item]
        fallback_date = dates[-1] if dates else market.fallback_date
        return cls._timestamp(
            "{}T13:30:00+08:00".format(fallback_date)
        )

    @classmethod
    def _market_available_at(
        cls,
        market: MarketHistoryInput,
        item: int,
    ) -> datetime.datetime:
        dates = market.dates[item]
        trade_date = dates[-1] if dates else market.fallback_date
        return cls._timestamp(
            "{}T13:30:00+08:00".format(trade_date)
        )

    @staticmethod
    def _timestamp(
        value: object,
        fallback: datetime.datetime | None = None,
    ) -> datetime.datetime:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            if fallback is None:
                raise StrategyContractError(
                    "Decision evidence requires a valid available_at"
                )
            return fallback
        return parsed.to_pydatetime()

    @staticmethod
    def _normalize_value(value: object) -> FieldValue | None:
        if value is None:
            return None
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
        scalar = value.item() if hasattr(value, "item") else value
        if isinstance(scalar, datetime.datetime):
            return scalar
        if isinstance(scalar, datetime.date):
            return scalar
        if isinstance(scalar, (str, bool, int, float)):
            return scalar
        return str(scalar)

    @staticmethod
    def _source_for(
        name: str,
        base_source: FieldSource,
        has_fundamental_snapshot: bool,
    ) -> FieldSource:
        if name in {
            "日期",
            "證券代號",
            "證券名稱",
            "開盤",
            "最高",
            "最低",
            "收盤",
        }:
            return FieldSource.TWSE_MARKET
        if name in {
            "無限次交易",
            "交易一次",
            "至多五次交易",
            "無限次交易(手續費$NTD300)",
            "利潤比",
        }:
            return FieldSource.PROFIT_ANALYSIS
        if name in {
            "Trend Score",
            "MOM",
            "CS",
            "Stop_Loss",
            "Take_Profit",
        }:
            return FieldSource.SIGNAL_FEATURES
        if name.startswith("EREM "):
            return FieldSource.EREM_V3
        if name.startswith("Valuation ") or name in {
            "Passive Benchmark Expected Return",
            "Required Valuation Return",
            "Base Case Revenue Growth",
            "Base Case Operating Margin",
            "Base Case Exit PE",
            "Base Case Expected CAGR",
            "Base Case Fair Value",
            "Bear Case Expected CAGR",
            "Bear Case Fair Value",
            "Bear Case Permanent Loss",
            "Bear Case Permanent Loss Acceptable",
            "Bull Case Expected CAGR",
            "Bull Case Fair Value",
        }:
            return FieldSource.LES_V5_VALUATION
        if name.startswith("Base "):
            return base_source
        if name.startswith("Extended ") or name in {
            "Critical Field Coverage",
            "Effective Extended Weight",
            "Quality Hard Rejects",
            "Quality Warnings",
        }:
            return (
                FieldSource.MOPS_IXBRL
                if has_fundamental_snapshot
                else FieldSource.UNKNOWN
            )
        return FieldSource.LES_V5

    @staticmethod
    def _base_source(
        snapshot: Mapping[str, object],
    ) -> FieldSource:
        sources = str(
            snapshot.get("base_quality_feature_sources", "")
        )
        has_twse = (
            bool(snapshot.get("twse_financial_observed_at"))
            or "TWSE_OPENAPI_HISTORY" in sources
        )
        has_mops = "MOPS_POINT_IN_TIME_ANNUAL_FALLBACK" in sources
        if has_twse and has_mops:
            return FieldSource.MIXED_FUNDAMENTALS
        if has_twse:
            return FieldSource.TWSE_FINANCIAL_BATCH
        if has_mops:
            return FieldSource.MOPS_IXBRL
        return FieldSource.UNKNOWN

    @staticmethod
    def _available_at_for(
        *,
        name: str,
        source: FieldSource,
        evaluated_at: datetime.datetime,
        market_at: datetime.datetime,
        base_at: datetime.datetime,
        extended_at: datetime.datetime,
    ) -> datetime.datetime:
        if source is FieldSource.LES_V5_VALUATION:
            return max(market_at, base_at, extended_at)
        if name.startswith("Base "):
            if source is FieldSource.MIXED_FUNDAMENTALS:
                return max(base_at, extended_at)
            return base_at
        if source in {
            FieldSource.TWSE_MARKET,
            FieldSource.PROFIT_ANALYSIS,
            FieldSource.SIGNAL_FEATURES,
            FieldSource.EREM_V3,
        }:
            return market_at
        if source in {
            FieldSource.MOPS_IXBRL,
            FieldSource.MIXED_FUNDAMENTALS,
        }:
            return extended_at
        if source is FieldSource.LES_V5:
            return max(market_at, base_at, extended_at)
        return evaluated_at

    @staticmethod
    def _confidence_for(
        name: str,
        row: Mapping[str, object],
        value: FieldValue | None,
    ) -> float:
        if value is None:
            return 0.0
        if name.startswith("Base "):
            return StrategyDecisionAssembler._score_value(
                row.get("Base Data Confidence")
            )
        if name.startswith("Extended ") or name in {
            "Critical Field Coverage",
            "Effective Extended Weight",
            "Quality Hard Rejects",
            "Quality Warnings",
        }:
            return StrategyDecisionAssembler._score_value(
                row.get("Extended Data Confidence")
            )
        if name.startswith("Valuation ") or "Case " in name:
            return (
                100.0
                if row.get("Valuation Model Status") == "COMPLETE"
                else 0.0
            )
        return 100.0

    @staticmethod
    def _score_value(value: object) -> float:
        try:
            if value is None or bool(pd.isna(value)):
                return 0.0
            return min(100.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _model_version_for(
        name: str,
        row: Mapping[str, object],
        source: FieldSource,
    ) -> str:
        if source is FieldSource.TWSE_MARKET:
            return "twse-daily-v1"
        if source is FieldSource.PROFIT_ANALYSIS:
            return "profit-analysis-v1"
        if source is FieldSource.SIGNAL_FEATURES:
            return "signal-features-v1"
        if source is FieldSource.EREM_V3:
            return EREM_V3_MODEL_VERSION
        if source is FieldSource.LES_V5_VALUATION:
            return StrategyDecisionAssembler._version_value(
                row.get("Valuation Model Version"),
                "les-v5-valuation-unknown",
            )
        if source is FieldSource.TWSE_FINANCIAL_BATCH:
            return "twse-base-quality-v1"
        if source is FieldSource.MOPS_IXBRL:
            return "mops-extended-quality-v1"
        if source is FieldSource.MIXED_FUNDAMENTALS:
            return "mixed-fundamental-quality-v1"
        if source is FieldSource.UNKNOWN:
            return "unknown-evidence-v1"
        return StrategyDecisionAssembler._version_value(
            row.get("LES Model Version"),
            "les-v5-unknown",
        )

    @staticmethod
    def _version_value(value: object, fallback: str) -> str:
        try:
            if value is None or bool(pd.isna(value)):
                return fallback
        except (TypeError, ValueError):
            return fallback
        text = str(value).strip()
        return text or fallback

    @staticmethod
    def _unit_for(name: str) -> FieldUnit:
        lowered = name.lower()
        if name == "日期":
            return FieldUnit.DATE
        if "version" in lowered or "版本" in name:
            return FieldUnit.VERSION
        if name in {
            "開盤",
            "最高",
            "最低",
            "收盤",
            "Stop_Loss",
            "Take_Profit",
            "策略停損",
            "策略停利",
            "EREM EMA20",
            "EREM EMA50",
            "EREM EMA200",
        } or "Fair Value" in name or "UpperBound" in name or "Highest" in name or "Lowest" in name or "VWAP" in name:
            return FieldUnit.TWD_SCALED_1000
        if "股數" in name:
            return FieldUnit.SHARES
        if "Count" in name or name == "EREM Holding Days":
            return FieldUnit.COUNT
        if name == "EREM Entry Weight":
            return FieldUnit.RATIO
        if "Trigger" in name or name in {
            "No Veto",
            "Entry Candidate",
            "Bear Case Permanent Loss Acceptable",
        }:
            return FieldUnit.BOOLEAN
        if "Score" in name or "Confidence" in name or "Coverage" in name or "Weight" in name or "分數" in name:
            return FieldUnit.SCORE
        if any(
            token in name
            for token in (
                "Ratio",
                "Return",
                "CAGR",
                "Loss",
                "Fraction",
                "RS20",
                "MOM",
                "CS",
                "Slope",
            )
        ):
            return FieldUnit.RATIO
        return FieldUnit.TEXT


__all__ = [
    "DAILY_ANALYSIS_CONTRACT_NAME",
    "StrategyContractError",
    "StrategyDecisionAssembler",
    "StrategyDecisionSchema",
]
