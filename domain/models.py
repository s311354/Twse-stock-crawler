"""Domain models shared across TWSE stock analysis layers."""

from __future__ import annotations

import enum
import datetime
import math
from dataclasses import dataclass
from typing import TypeAlias

StockRow: TypeAlias = list[str]
StockRows: TypeAlias = list[StockRow]
StockSelector: TypeAlias = int | str
Ohlc: TypeAlias = tuple[float, float, float, float]
ProfitRow: TypeAlias = list[float | int | None]
SignalRow: TypeAlias = list[float | int | None]


@dataclass(frozen=True)
class StockIdentity:
    stock_no: str
    stock_name: str


@dataclass(frozen=True)
class PortfolioPosition:
    """User-supplied portfolio state; missing shares mean equal-weight input."""

    symbol: str
    shares: float | None = None
    average_cost: float | None = None
    entry_les: float | None = None
    entry_decision: str | None = None
    entry_date: datetime.date | None = None
    sector: str | None = None
    correlation_cluster: str | None = None
    maximum_weight: float | None = None

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip()
        if not symbol:
            raise ValueError("PortfolioPosition symbol cannot be empty")
        object.__setattr__(self, "symbol", symbol)
        if self.shares is not None:
            if not math.isfinite(float(self.shares)) or float(self.shares) <= 0:
                raise ValueError("PortfolioPosition shares must be positive")
        for name in ("average_cost", "entry_les", "maximum_weight"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("PortfolioPosition {} must be finite".format(name))
        if self.average_cost is not None and self.average_cost < 0:
            raise ValueError("PortfolioPosition average_cost cannot be negative")
        if self.entry_les is not None and not 0 <= self.entry_les <= 100:
            raise ValueError("PortfolioPosition entry_les must be between 0 and 100")
        if self.entry_decision is not None:
            entry_decision = (
                str(self.entry_decision)
                .strip()
                .upper()
                .replace("-", "_")
                .replace(" ", "_")
            )
            if entry_decision not in {"BUY", "EARLY_BUY", "BUY_SMALL"}:
                raise ValueError(
                    "PortfolioPosition entry_decision must be BUY, EARLY_BUY, or BUY_SMALL"
                )
            object.__setattr__(self, "entry_decision", entry_decision)
        if self.entry_date is not None:
            entry_date = self.entry_date
            if isinstance(entry_date, datetime.datetime):
                entry_date = entry_date.date()
            elif isinstance(entry_date, str):
                try:
                    entry_date = datetime.date.fromisoformat(entry_date.strip())
                except ValueError as error:
                    raise ValueError(
                        "PortfolioPosition entry_date must use YYYY-MM-DD"
                    ) from error
            if not isinstance(entry_date, datetime.date):
                raise ValueError(
                    "PortfolioPosition entry_date must be a date or YYYY-MM-DD"
                )
            object.__setattr__(self, "entry_date", entry_date)
        if self.maximum_weight is not None and not 0 < self.maximum_weight <= 1:
            raise ValueError("PortfolioPosition maximum_weight must be within (0, 1]")
        for name in ("sector", "correlation_cluster"):
            value = getattr(self, name)
            if value is not None:
                normalized = str(value).strip()
                object.__setattr__(self, name, normalized or None)


@dataclass(frozen=True)
class TWSEStock:
    stock_no: str
    stock_name: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    pe: float | None = None
    trades: int | None = None
    amount: float | None = None
    bid: float | None = None
    ask: float | None = None


class TwseColumns:
    STOCK_NO = 0
    NAME = 1
    VOLUME = 2
    TRADE_COUNT = 3
    TRADE_VALUE = 4
    OPEN = 5
    HIGH = 6
    LOW = 7
    CLOSE = 8
    CHANGE_SIGN = 9
    PRICE_CHANGE = 10
    BID_PRICE = 11
    BID_VOLUME = 12
    ASK_PRICE = 13
    ASK_VOLUME = 14
    PE = 15

    REQUIRED_WIDTH = PE + 1


class Stocktype(enum.Enum):
    VEH = 12,
    ELEC = 13,
    BUI = 14,
    AIR = 15,
    CHEMI = 21,
    BIO = 22,
    SEMI = 24,
    COMP = 25,
    COMM = 27

    def __str__(self) -> str:
        return self.name
