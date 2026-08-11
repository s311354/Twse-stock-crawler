"""Application contract for incremental TWSE market-history caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from domain.models import StockRows


MARKET_HISTORY_SCHEMA_VERSION = 1


class MarketHistoryCacheError(RuntimeError):
    """Raised when the market-history cache cannot be read or written."""


@dataclass(frozen=True)
class CachedMarketDay:
    """A complete market day; partial days are never returned to callers."""

    trade_date: str
    stock_rows: StockRows
    taiex_close: float


class MarketHistoryCachePort(Protocol):
    """Persistence boundary for immutable TWSE daily market payloads."""

    def load_many(
        self,
        trade_dates: list[str],
        *,
        market_type: str,
        schema_version: int = MARKET_HISTORY_SCHEMA_VERSION,
    ) -> dict[str, CachedMarketDay]:
        """Load complete cached days with one repository query."""

    def upsert_day(
        self,
        day: CachedMarketDay,
        *,
        market_type: str,
        schema_version: int = MARKET_HISTORY_SCHEMA_VERSION,
    ) -> None:
        """Atomically persist stock rows and TAIEX for one trading day."""


__all__ = [
    "CachedMarketDay",
    "MARKET_HISTORY_SCHEMA_VERSION",
    "MarketHistoryCacheError",
    "MarketHistoryCachePort",
]
