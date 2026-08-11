"""Application contracts shared by stock-analysis use cases."""

from __future__ import annotations

import datetime
import enum
from dataclasses import dataclass
from typing import Any

from domain.models import ProfitRow
from domain.models import StockSelector
from domain.model_contracts import StrategyDecision


class MarketStatus(str, enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class StockAnalysisRequest:
    stocklist: list[StockSelector]
    stocktype: Any
    output_file_names: str
    endbacktrack: int
    beginbacktrack: int = 0
    subject: str = "None"
    ccreceiver: str | None = None
    linechart: bool = False
    period: int = 7
    mail: bool = False
    mops_enabled: bool = True
    mops_history_years: int = 5
    account_equity: float = 1_000_000.0


@dataclass(frozen=True)
class AnalysisContext:
    """Opaque market-data resource passed between application use cases."""

    resource: object
    market_date: datetime.date
    market_status: MarketStatus


@dataclass(frozen=True)
class StrategyEvaluation:
    maxprofits: list[ProfitRow]
    decisions: tuple[StrategyDecision, ...] = ()
