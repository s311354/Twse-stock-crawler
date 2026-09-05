"""Version-neutral application contracts shared by portfolio pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from domain.models import PortfolioPosition


class PortfolioAllocationInputError(ValueError):
    """Raised when runtime portfolio state cannot be valued safely."""


class PortfolioPersistenceError(RuntimeError):
    """Raised when cross-session opportunity evidence is unavailable."""


@dataclass(frozen=True)
class PersistenceObservation:
    """Legacy PAE v1 opportunity observation kept for storage compatibility."""

    incumbent_symbol: str
    challenger_symbol: str
    excess_opportunity: float


@dataclass(frozen=True)
class PortfolioRuntimeContext:
    """Version-neutral portfolio state supplied to an allocation pipeline."""

    account_equity: float
    positions: tuple[PortfolioPosition, ...] = ()
    current_cash: float | None = None
    sector_by_symbol: Mapping[str, str] | None = None
    default_sector: str = "UNKNOWN"


PAE_INPUT_COLD_START = "COLD_START_ALL_CASH"
PAE_INPUT_EXPLICIT = "EXPLICIT_SHARES"
PAE_INPUT_SYMBOL_ONLY = "SYMBOL_ONLY_EQUAL_WEIGHT"
PAE_INPUT_MIXED = "MIXED_EXPLICIT_AND_EQUAL_WEIGHT"


__all__ = [
    "PAE_INPUT_COLD_START",
    "PAE_INPUT_EXPLICIT",
    "PAE_INPUT_MIXED",
    "PAE_INPUT_SYMBOL_ONLY",
    "PersistenceObservation",
    "PortfolioAllocationInputError",
    "PortfolioPersistenceError",
    "PortfolioRuntimeContext",
]
