"""Deterministic cache-first planning for MOPS filing refreshes."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FilingRecord


FilingPeriod = tuple[int, int]


@dataclass(frozen=True)
class FilingRefreshPlan:
    """Partition required filings into mutually exclusive refresh actions."""

    reuse: list[FilingRecord]
    download: list[FilingCandidate]
    verify_revision: list[FilingCandidate]


class FilingRefreshPlanner:
    """Build a five-year seed and minimize subsequent MOPS downloads."""

    DAILY_VERIFY_PERIODS = 2

    def plan(
        self,
        candidates: list[FilingCandidate],
        cached_filings: list[FilingRecord],
        *,
        history_years: int,
        corrected_periods: set[FilingPeriod] | None = None,
    ) -> FilingRefreshPlan:
        if history_years < 1:
            raise ValueError("history_years must be at least one")

        selected = self.select_required_filings(candidates, history_years)
        latest_cached = self._latest_by_period(cached_filings)
        recent_periods = {
            self._period(candidate)
            for candidate in selected[-self.DAILY_VERIFY_PERIODS :]
        }
        corrected = corrected_periods or set()
        reuse: list[FilingRecord] = []
        download: list[FilingCandidate] = []
        verify_revision: list[FilingCandidate] = []

        for candidate in selected:
            period = self._period(candidate)
            cached = latest_cached.get(period)
            if cached is None:
                download.append(candidate)
            elif period in recent_periods or period in corrected:
                verify_revision.append(candidate)
            else:
                reuse.append(cached)

        return FilingRefreshPlan(
            reuse=reuse,
            download=download,
            verify_revision=verify_revision,
        )

    @staticmethod
    def is_monthly_refresh_due(
        last_refresh_epoch: int | None,
        signal_time: dt.datetime,
    ) -> bool:
        """Return true once per Taipei calendar month, including first seed."""

        if signal_time.tzinfo is None:
            raise ValueError("signal_time must be timezone-aware")
        if last_refresh_epoch is None:
            return True
        taipei = dt.timezone(dt.timedelta(hours=8))
        last_refresh = dt.datetime.fromtimestamp(last_refresh_epoch, dt.timezone.utc)
        return last_refresh.astimezone(taipei).strftime("%Y-%m") != signal_time.astimezone(
            taipei
        ).strftime("%Y-%m")

    @staticmethod
    def select_required_filings(
        filings: list[FilingCandidate],
        history_years: int,
    ) -> list[FilingCandidate]:
        """Select annual history plus the periods needed for latest TTM metrics."""

        if not filings:
            return []
        filings_by_period = {
            (candidate.fiscal_year, candidate.quarter): candidate
            for candidate in filings
        }
        latest = max(
            filings,
            key=lambda candidate: (candidate.fiscal_year, candidate.quarter),
        )
        annual = sorted(
            (candidate for candidate in filings if candidate.quarter == 4),
            key=lambda candidate: candidate.fiscal_year,
        )[-history_years:]
        selected = {
            (candidate.fiscal_year, candidate.quarter): candidate
            for candidate in annual
        }
        selected[(latest.fiscal_year, latest.quarter)] = latest
        if latest.quarter < 4:
            for period in (
                (latest.fiscal_year - 1, latest.quarter),
                (latest.fiscal_year - 1, 4),
            ):
                if period in filings_by_period:
                    selected[period] = filings_by_period[period]
        return [selected[key] for key in sorted(selected)]

    @staticmethod
    def _latest_by_period(
        filings: list[FilingRecord],
    ) -> dict[FilingPeriod, FilingRecord]:
        latest: dict[FilingPeriod, FilingRecord] = {}
        for filing in sorted(
            filings,
            key=lambda item: (
                item.fiscal_year,
                item.quarter,
                item.revision_number,
            ),
        ):
            latest[(filing.fiscal_year, filing.quarter)] = filing
        return latest

    @staticmethod
    def _period(candidate: FilingCandidate) -> FilingPeriod:
        return candidate.fiscal_year, candidate.quarter


__all__ = ["FilingRefreshPlan", "FilingRefreshPlanner"]
