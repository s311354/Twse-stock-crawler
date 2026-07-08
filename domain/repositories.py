"""Repository abstractions for stock time-series data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class StockRepository(ABC):
    """Persistence boundary for stock datasets.

    Concrete implementations decide whether data lands in CSV, SQLite, DuckDB,
    or another backend. Callers only depend on deterministic save/load behavior.
    """

    @abstractmethod
    def save(self, symbol: str, stocks: pd.DataFrame) -> None:
        """Persist one symbol's stock rows, replacing existing rows for symbol."""

    @abstractmethod
    def load(self, symbol: str) -> pd.DataFrame:
        """Load one symbol's stock rows. Missing symbols return an empty frame."""
