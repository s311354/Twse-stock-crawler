"""Stable names and enums shared by Low Entry Model v5 components."""

from __future__ import annotations

from enum import Enum


LOW_ENTRY_STRATEGY_NAME = "low_entry_score_v5"
LES_V5_MODEL_VERSION = "5.0.0"
LES_V5_PARAMETER_VERSION = (
    "quality-first-panic-reversal-2026-09-05-v26-elec-trend-pullback"
)


class FallingKnifeState(str, Enum):
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    PANIC_WATCH = "PANIC_WATCH"
    REVERSAL_WATCH = "REVERSAL_WATCH"
    CLEAR = "CLEAR"


class BroadRallyTier(str, Enum):
    NONE = "NONE"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class ValuationStatus(str, Enum):
    PASS = "PASS"
    SUPPORTIVE = "SUPPORTIVE"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


__all__ = [
    "BroadRallyTier",
    "FallingKnifeState",
    "LES_V5_MODEL_VERSION",
    "LES_V5_PARAMETER_VERSION",
    "LOW_ENTRY_STRATEGY_NAME",
    "ValuationStatus",
]
