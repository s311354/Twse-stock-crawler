"""Domain models shared across TWSE stock analysis layers."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TypeAlias

StockRow: TypeAlias = list[str]
StockRows: TypeAlias = list[StockRow]
Ohlc: TypeAlias = tuple[float, float, float, float]
ProfitRow: TypeAlias = list[float | int | None]
SignalRow: TypeAlias = list[float | int | None]


@dataclass(frozen=True)
class StockIdentity:
    stock_no: str
    stock_name: str


class Stocktype(enum.Enum):
    VEH = 12,
    ELEC = 13,
    SEMI = 14,
    AIR = 15,
    BIO = 22,
    COMM = 27

    def __str__(self) -> str:
        return self.name
