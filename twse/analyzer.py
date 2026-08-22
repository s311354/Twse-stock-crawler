"""Backward-compatible facade for TWSE analysis helpers.

Core calculations live in :mod:`domain.services`; this module preserves the
existing public import path for callers that still use ``twse.analyzer``.
"""

from domain.services import LowEntryDecision
from domain.services import SignalFeatures
from domain.services import compute_signal_features
from domain.services import evaluate_low_entry
from domain.services import max_profit
from domain.services import max_profit_k_transactions
from domain.services import max_profit_unlimited
from domain.services import max_profit_with_fee

__all__ = [
    "LowEntryDecision",
    "SignalFeatures",
    "compute_signal_features",
    "evaluate_low_entry",
    "max_profit",
    "max_profit_k_transactions",
    "max_profit_unlimited",
    "max_profit_with_fee",
]
