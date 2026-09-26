"""Immutable, versioned contracts for model decisions and evidence."""

from __future__ import annotations

import datetime
import enum
from dataclasses import dataclass
from typing import Generic
from typing import TypeVar


STRATEGY_DECISION_SCHEMA_VERSION = "1.0.0"


class FieldSource(str, enum.Enum):
    """Stable producer identifiers used for field-level lineage."""

    TWSE_MARKET = "TWSE_MARKET"
    TPEX_MARKET = "TPEX_MARKET"
    TWSE_FINANCIAL_BATCH = "TWSE_FINANCIAL_BATCH"
    MOPS_IXBRL = "MOPS_IXBRL"
    MIXED_FUNDAMENTALS = "MIXED_FUNDAMENTALS"
    PROFIT_ANALYSIS = "PROFIT_ANALYSIS"
    SIGNAL_FEATURES = "SIGNAL_FEATURES"
    LES_V5 = "LES_V5"
    LES_V6 = "LES_V6"
    LES_V5_VALUATION = "LES_V5_VALUATION"
    EREM_V3 = "EREM_V3"
    PAE_V1 = "PAE_V1"
    PAE_V2 = "PAE_V2"
    UNKNOWN = "UNKNOWN"


class FieldUnit(str, enum.Enum):
    """Canonical units carried by every report/model field."""

    TEXT = "text"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    COUNT = "count"
    SCORE = "score"
    RATIO = "ratio"
    PERCENT = "percent"
    TWD_SCALED_1000 = "TWD_x1000"
    SHARES = "shares"
    VERSION = "version"


FieldValue = str | int | float | bool | datetime.date | datetime.datetime
T = TypeVar("T", bound=FieldValue)


@dataclass(frozen=True)
class DecisionField(Generic[T]):
    """One model value with complete point-in-time provenance."""

    name: str
    value: T | None
    source: FieldSource
    available_at: datetime.datetime
    confidence: float
    unit: FieldUnit
    model_version: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("DecisionField name cannot be empty")
        if self.available_at.tzinfo is None:
            raise ValueError(
                "DecisionField available_at must be timezone-aware"
            )
        if not 0.0 <= float(self.confidence) <= 100.0:
            raise ValueError(
                "DecisionField confidence must be between 0 and 100"
            )
        if not self.model_version.strip():
            raise ValueError(
                "DecisionField model_version cannot be empty"
            )


@dataclass(frozen=True)
class StrategyDecision:
    """Validated per-stock decision consumed by storage and renderers."""

    strategy_name: str
    evaluated_at: datetime.datetime
    fields: tuple[DecisionField[FieldValue], ...]
    schema_version: str = STRATEGY_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.strategy_name.strip():
            raise ValueError("strategy_name cannot be empty")
        if self.evaluated_at.tzinfo is None:
            raise ValueError(
                "StrategyDecision evaluated_at must be timezone-aware"
            )
        if not self.schema_version.strip():
            raise ValueError("schema_version cannot be empty")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            duplicates = sorted(
                name for name in set(names) if names.count(name) > 1
            )
            raise ValueError(
                "StrategyDecision contains duplicate fields: {}".format(
                    ", ".join(duplicates)
                )
            )

    def field(self, name: str) -> DecisionField[FieldValue]:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError("StrategyDecision field '{}' is missing".format(name))

    def value(self, name: str, default: object = None) -> object:
        try:
            return self.field(name).value
        except KeyError:
            return default


__all__ = [
    "DecisionField",
    "FieldSource",
    "FieldUnit",
    "FieldValue",
    "STRATEGY_DECISION_SCHEMA_VERSION",
    "StrategyDecision",
]
