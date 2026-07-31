"""Backward-compatible aliases for domain filing records."""

from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FilingCorrection
from domain.fundamental_models import FilingRecord

MopsFilingCandidate = FilingCandidate
MopsFiling = FilingRecord
MopsCorrection = FilingCorrection

__all__ = [
    "MopsCorrection",
    "MopsFiling",
    "MopsFilingCandidate",
]
