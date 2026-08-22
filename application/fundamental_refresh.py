"""Refresh decisions for cache-first point-in-time fundamentals."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from application.fundamental_ports import FilingCatalogPort
from domain.data_provenance import MOPS_PUBLIC_ANNOUNCEMENT
from domain.fundamental_models import FilingAnnouncement
from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FilingCorrection
from domain.fundamental_models import FilingRecord
from domain.fundamental_models import FundamentalSnapshot


class RefreshPlanner:
    """Resolve refresh triggers without performing network or storage I/O."""

    def __init__(
        self,
        filing_catalog: FilingCatalogPort,
        *,
        include_current_observations: bool,
    ) -> None:
        self._filing_catalog = filing_catalog
        self._include_current_observations = include_current_observations

    @staticmethod
    def recent_reporting_periods(
        observed_at: dt.datetime,
        daily_financials: dict[str, FundamentalSnapshot],
    ) -> list[tuple[int, int]]:
        local = observed_at.astimezone(dt.timezone(dt.timedelta(hours=8)))
        completed_quarter = (local.month - 1) // 3
        calendar_latest = (
            (local.year, completed_quarter)
            if completed_quarter > 0
            else (local.year - 1, 4)
        )
        batch_periods = {
            snapshot.period for snapshot in daily_financials.values()
        }
        latest = max(batch_periods | {calendar_latest})
        previous = (
            (latest[0], latest[1] - 1)
            if latest[1] > 1
            else (latest[0] - 1, 4)
        )
        return [previous, latest]

    @staticmethod
    def event_corrected_periods(
        symbol: str,
        cached_filings: list[FilingRecord],
        daily_corrections: dict[
            str,
            dict[tuple[int, int], FilingCorrection],
        ],
    ) -> set[tuple[int, int]]:
        events = daily_corrections.get(symbol, {})
        if not events:
            return set()
        latest_cached: dict[tuple[int, int], FilingRecord] = {}
        for filing in sorted(
            cached_filings,
            key=lambda item: item.revision_number,
        ):
            latest_cached[(filing.fiscal_year, filing.quarter)] = filing
        return {
            period
            for period, correction in events.items()
            if period in latest_cached
            and (
                latest_cached[period].correction_announced_on is None
                or correction.announced_on
                > latest_cached[period].correction_announced_on
            )
        }

    @staticmethod
    def batch_has_new_period(
        batch_snapshot: FundamentalSnapshot | None,
        cached_filings: list[FilingRecord],
        *,
        as_of: pd.Timestamp,
    ) -> bool:
        if batch_snapshot is None:
            return False
        eligible_periods = []
        for filing in cached_filings:
            available_at = pd.to_datetime(
                filing.available_at,
                utc=True,
                errors="coerce",
            )
            if not pd.isna(available_at) and available_at <= as_of:
                eligible_periods.append((filing.fiscal_year, filing.quarter))
        return (
            not eligible_periods
            or batch_snapshot.period > max(eligible_periods)
        )

    @staticmethod
    def has_complete_cached_filing_history(
        cached_filings: list[FilingRecord],
        *,
        history_years: int,
        as_of: pd.Timestamp,
    ) -> bool:
        eligible = []
        for filing in cached_filings:
            available_at = pd.to_datetime(
                filing.available_at,
                utc=True,
                errors="coerce",
            )
            if not pd.isna(available_at) and available_at <= as_of:
                eligible.append(filing)
        if not eligible:
            return False
        periods = {(item.fiscal_year, item.quarter) for item in eligible}
        latest_period = max(periods)
        annual_years = {year for year, quarter in periods if quarter == 4}
        if len(annual_years) < history_years:
            return False
        latest_year, latest_quarter = latest_period
        if latest_quarter < 4:
            required = {
                latest_period,
                (latest_year - 1, latest_quarter),
                (latest_year - 1, 4),
            }
            if not required.issubset(periods):
                return False
        return True

    def select_refresh_candidates(
        self,
        candidates: list[FilingCandidate],
        signal_time: pd.Timestamp,
    ) -> list[FilingCandidate]:
        """Select only filings eligible at the signal evaluation time."""

        selected: list[FilingCandidate] = []
        for candidate in candidates:
            if candidate.available_at is None:
                if self._include_current_observations:
                    selected.append(candidate)
                continue
            available_at = pd.to_datetime(
                candidate.available_at,
                utc=True,
                errors="coerce",
            )
            if not pd.isna(available_at) and available_at <= signal_time:
                selected.append(candidate)
        return selected

    @staticmethod
    def apply_availability(
        catalog: list[FilingCandidate],
        cached_filings: list[FilingRecord],
        announcements: list[FilingAnnouncement],
    ) -> list[FilingCandidate]:
        latest_cached: dict[tuple[int, int], FilingRecord] = {}
        for filing in sorted(
            cached_filings,
            key=lambda item: item.revision_number,
        ):
            latest_cached[(filing.fiscal_year, filing.quarter)] = filing
        available_by_period = {
            (int(item.fiscal_year), int(item.quarter)): str(item.available_at)
            for item in announcements
        }
        dated: list[FilingCandidate] = []
        for candidate in catalog:
            period = candidate.fiscal_year, candidate.quarter
            existing = latest_cached.get(period)
            available_at = available_by_period.get(period)
            if available_at is None and existing is not None:
                available_at = existing.available_at
            dated.append(
                FilingCandidate(
                    symbol=candidate.symbol,
                    fiscal_year=candidate.fiscal_year,
                    quarter=candidate.quarter,
                    report_id=candidate.report_id,
                    source_url=candidate.source_url,
                    available_at=available_at,
                    period_end=candidate.period_end,
                    statement_scope=candidate.statement_scope,
                    audit_status=candidate.audit_status,
                    availability_source=(
                        MOPS_PUBLIC_ANNOUNCEMENT
                        if period in available_by_period
                        else existing.availability_source
                        if existing is not None
                        else candidate.availability_source
                    ),
                )
            )
        return dated

    def find_corrected_periods(
        self,
        candidates: list[FilingCandidate],
        cached_filings: list[FilingRecord],
    ) -> set[tuple[int, int]]:
        latest_cached: dict[tuple[int, int], FilingRecord] = {}
        for filing in sorted(
            cached_filings,
            key=lambda item: item.revision_number,
        ):
            latest_cached[(filing.fiscal_year, filing.quarter)] = filing
        corrected: set[tuple[int, int]] = set()
        for candidate in candidates:
            period = candidate.fiscal_year, candidate.quarter
            existing = latest_cached.get(period)
            if existing is None:
                continue
            correction = self._filing_catalog.latest_correction(candidate)
            if correction is not None and (
                existing.correction_announced_on is None
                or correction.announced_on > existing.correction_announced_on
            ):
                corrected.add(period)
        return corrected


__all__ = ["RefreshPlanner"]
