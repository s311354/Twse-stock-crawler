"""Application contracts shared by stock-analysis use cases."""

from __future__ import annotations

import datetime
import enum
from dataclasses import dataclass
from typing import Any

from domain.models import ProfitRow
from domain.models import StockSelector
from domain.models import Stocktype
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
    portfolio_symbols: tuple[str, ...] = ()
    explicit_symbols: tuple[str, ...] = ()
    # Internal PIT cutoff derived from date -e; not a user-facing CLI option.
    as_of: datetime.datetime | None = None
    replay_restrictions_file: str | None = None
    replay_mail: bool = False
    history_start_date: datetime.date | None = None
    history_end_date: datetime.date | None = None

    @property
    def notification_enabled(self) -> bool:
        return self.mail and (self.as_of is None or self.replay_mail)

    @property
    def report_file_prefix(self) -> str:
        if self.as_of is None:
            return self.output_file_names
        return f"{self.output_file_names}_replay_{self.as_of.strftime('%Y%m%dT%H%M%S%z')}"

    @property
    def market_source(self) -> str:
        return "TPEX" if self.stocktype == Stocktype.SEMI else "TWSE"


@dataclass(frozen=True)
class AnalysisContext:
    """Opaque market-data resource passed between application use cases."""

    resource: object
    market_date: datetime.date
    market_status: MarketStatus
    as_of: datetime.datetime | None = None


@dataclass(frozen=True)
class StrategyEvaluation:
    maxprofits: list[ProfitRow]
    decisions: tuple[StrategyDecision, ...] = ()
