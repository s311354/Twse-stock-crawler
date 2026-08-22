"""Point-in-time policy for fundamental evaluation."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import pandas as pd

from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FilingRecord


class PointInTimeResolver:
    """Apply one explicit availability cutoff across the refresh pipeline."""

    def __init__(
        self,
        *,
        include_current_observations: bool,
        now_provider: Callable[[], dt.datetime],
    ) -> None:
        self._include_current_observations = include_current_observations
        self._now_provider = now_provider

    def effective_signal_time(self, requested: pd.Timestamp) -> pd.Timestamp:
        if not self._include_current_observations:
            return requested
        return max(requested, pd.Timestamp(self.utc_now()))

    def advance_live_evaluation_time(
        self,
        current: pd.Timestamp,
        ingested_filings: list[FilingRecord],
    ) -> pd.Timestamp:
        if not self._include_current_observations or not ingested_filings:
            return current
        observed_times = [
            pd.to_datetime(filing.available_at, utc=True, errors="raise")
            for filing in ingested_filings
        ]
        return max(current, *observed_times)

    def utc_now(self) -> dt.datetime:
        current = self._now_provider()
        if current.tzinfo is None:
            raise ValueError(
                "MOPS now_provider must return a timezone-aware datetime"
            )
        return current.astimezone(dt.timezone.utc)

    @staticmethod
    def bundle_as_of(
        bundle: FundamentalBundle,
        evaluated_at: pd.Timestamp,
    ) -> FundamentalBundle:
        return FundamentalBundle(
            annual_fundamentals=bundle.annual_fundamentals,
            snapshot=bundle.snapshot,
            as_of=evaluated_at.isoformat(),
        )


__all__ = ["PointInTimeResolver"]
