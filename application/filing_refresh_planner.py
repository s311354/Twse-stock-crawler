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
        selected_periods = {self._period(item) for item in selected}
        selected.extend(item for item in candidates
                        if self._period(item) in (corrected_periods or set())
                        and self._period(item) not in selected_periods)
        latest_cached = self._latest_by_period(cached_filings)
        corrected = corrected_periods or set()
        reuse: list[FilingRecord] = []
        download: list[FilingCandidate] = []
        verify_revision: list[FilingCandidate] = []

        for candidate in selected:
            period = self._period(candidate)
            cached = latest_cached.get(period)
            if cached is None:
                download.append(candidate)
            elif period in corrected:
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
    def is_daily_revision_check_due(
        last_check_epoch: int | None,
        signal_time: dt.datetime,
    ) -> bool:
        """Return true at most once per Taipei calendar day."""

        if signal_time.tzinfo is None:
            raise ValueError("signal_time must be timezone-aware")
        if last_check_epoch is None:
            return True
        taipei = dt.timezone(dt.timedelta(hours=8))
        last_check = dt.datetime.fromtimestamp(last_check_epoch, dt.timezone.utc)
        return last_check.astimezone(taipei).date() != signal_time.astimezone(
            taipei
        ).date()

    @staticmethod
    def select_required_filings(
        filings: list[FilingCandidate],
        history_years: int,
    ) -> list[FilingCandidate]:
        """All quarters of five completed fiscal years, plus current YTD.

        Select only catalog evidence; never invent a missing filing or count
        five scattered annual reports as five complete years of quarters.
        """

        if history_years < 1:
            raise ValueError("history_years must be at least one")
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
        required = FilingRefreshPlanner.history_periods(
            (latest.fiscal_year, latest.quarter), history_years,
        )
        return [filings_by_period[key] for key in sorted(required & filings_by_period.keys())]

    @staticmethod
    def history_periods(latest: FilingPeriod, history_years: int) -> set[FilingPeriod]:
        if history_years < 1 or latest[1] not in {1, 2, 3, 4}:
            raise ValueError("Invalid history years or fiscal quarter")
        last_full_year = latest[0] if latest[1] == 4 else latest[0] - 1
        first_index = (last_full_year - history_years + 1) * 4
        last_index = latest[0] * 4 + latest[1] - 1
        return {(index // 4, index % 4 + 1) for index in range(first_index, last_index + 1)}

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
