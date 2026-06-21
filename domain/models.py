"""Domain models shared across TWSE stock analysis layers."""

from __future__ import annotations

import enum
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
class TWSEStock:
    stock_no: str
    stock_name: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    pe: float | None = None


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
    SEMI = 14,
    AIR = 15,
    BIO = 22,
    COMM = 27

    def __str__(self) -> str:
        return self.name
