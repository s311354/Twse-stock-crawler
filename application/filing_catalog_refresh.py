"""Persist discovery progress independently of bounded iXBRL downloads."""
from dataclasses import replace
from collections.abc import Iterable

import pandas as pd

from application.filing_refresh_planner import FilingRefreshPlan, FilingRefreshPlanner
from application.fundamental_ports import FilingCatalogPort, FundamentalRepositoryPort, FundamentalUnitOfWork
from application.fundamental_refresh import RefreshPlanner
from domain.fundamental_models import FilingCandidate, FilingCatalogState, FilingCorrection, FilingRecord


class FilingCatalogRefresh:
    def __init__(self, catalog: FilingCatalogPort, repository: FundamentalRepositoryPort,
                 planner: FilingRefreshPlanner, eligibility: RefreshPlanner):
        self.catalog = catalog
        self.repository = repository
        self.planner = planner
        self.eligibility = eligibility

    def prepare(
        self, symbol: str, *, state: FilingCatalogState | None,
        cached: list[FilingRecord], signal: pd.Timestamp, history_years: int,
        batch_period: tuple[int, int] | None,
        events: dict[tuple[int, int], FilingCorrection], uow: FundamentalUnitOfWork,
    ) -> tuple[FilingCatalogState, FilingRefreshPlan]:
        """Network operations share the caller's request budget and SQLite UoW.

        Save after each successful discovery step so a quota/guard interruption
        resumes locally next run. Catalog observation never dates a filing.
        """
        monthly = state is None or self.planner.is_monthly_refresh_due(
            int(pd.Timestamp(state.checked_at).timestamp()), signal.to_pydatetime(),
        )
        periods = {(item.fiscal_year, item.quarter) for item in state.candidates} if state else set()
        new_batch = (batch_period is not None and batch_period not in periods
                     and (state is None or batch_period != state.observed_batch_period))
        discover = monthly or new_batch or bool(set(events) - periods)
        if discover:
            candidates = self.catalog.discover(symbol)
            local_year = signal.tz_convert("Asia/Taipei").year
            minimum_year = local_year - history_years - 1
            candidates = [item for item in candidates if minimum_year <= item.fiscal_year <= local_year]
            old = state
            state = FilingCatalogState(
                symbol=symbol, checked_at=signal.isoformat(), candidates=tuple(candidates),
                scanned_periods=() if monthly else old.scanned_periods,
                corrections=old.corrections if old else (),
                verified_corrections=old.verified_corrections if old else (),
                announcements_pending=True, observed_batch_period=batch_period,
            )
            self.repository.save_filing_catalog(state, uow=uow)

        assert state is not None  # First observation always discovers a catalog.
        if state.announcements_pending:
            # Only missing periods need announcement provenance lookup.
            cached_periods = {(item.fiscal_year, item.quarter) for item in cached}
            years = sorted({item.fiscal_year + (item.quarter == 4)
                            for item in self.planner.select_required_filings(list(state.candidates), history_years)
                            if (item.fiscal_year, item.quarter) not in cached_periods})
            announcements = self.catalog.discover_announcements(symbol, years) if years else []
            state = replace(
                state, announcements_pending=False,
                candidates=tuple(self.eligibility.apply_availability(list(state.candidates), cached, announcements)),
            )
            self.repository.save_filing_catalog(state, uow=uow)

        eligible = self.eligibility.select_refresh_candidates(list(state.candidates), signal)
        required = self.planner.select_required_filings(eligible, history_years)
        cached_periods = {(item.fiscal_year, item.quarter) for item in cached}
        scanned = set(state.scanned_periods)
        # Corrections of older years are scanned once per catalog month, not
        # once per unfinished backfill. Client caches each all-market period.
        for candidate in required:
            period = candidate.fiscal_year, candidate.quarter
            if period not in cached_periods or period in scanned:
                continue
            correction = self.catalog.latest_correction(candidate)
            corrections = self._merge(state.corrections, [correction] if correction else [], signal)
            scanned.add(period)
            state = replace(state, scanned_periods=tuple(sorted(scanned)), corrections=corrections)
            self.repository.save_filing_catalog(state, uow=uow)
        state = replace(state, corrections=self._merge(state.corrections, events.values(), signal))
        self.repository.save_filing_catalog(state, uow=uow)
        return state, self.planner.plan(
            eligible, cached, history_years=history_years,
            corrected_periods=self.unverified_periods(state, cached),
        )

    @staticmethod
    def unverified_periods(
        state: FilingCatalogState, cached: list[FilingRecord],
    ) -> set[tuple[int, int]]:
        """Notice acknowledgement may be in either the filing or planner state."""
        verified = {
            (item.fiscal_year, item.quarter): item.announced_on
            for item in state.verified_corrections
        }
        latest = FilingRefreshPlanner._latest_by_period(cached)
        corrected = set()
        for event in state.corrections:
            period = event.fiscal_year, event.quarter
            filing_notice = latest[period].correction_announced_on if period in latest else None
            if event.announced_on > max(verified.get(period, ""), filing_notice or ""):
                corrected.add(period)
        return corrected

    def acknowledge(self, state: FilingCatalogState, candidate: FilingCandidate,
                    *, uow: FundamentalUnitOfWork) -> FilingCatalogState:
        """A verified identical hash need not be downloaded again for this notice."""
        period = candidate.fiscal_year, candidate.quarter
        events = [item for item in state.corrections if (item.fiscal_year, item.quarter) == period]
        prior = [item for item in state.verified_corrections if (item.fiscal_year, item.quarter) != period]
        state = replace(state, verified_corrections=tuple(prior + events),
                        scanned_periods=tuple(sorted(set(state.scanned_periods) | {period})))
        self.repository.save_filing_catalog(state, uow=uow)
        return state

    @staticmethod
    def _merge(existing: Iterable[FilingCorrection], events: Iterable[FilingCorrection],
               signal: pd.Timestamp) -> tuple[FilingCorrection, ...]:
        latest: dict[tuple[int, int], FilingCorrection] = {}
        for event in (*existing, *events):
            if pd.Timestamp(event.announced_on).date() > signal.tz_convert("Asia/Taipei").date():
                continue
            period = event.fiscal_year, event.quarter
            if period not in latest or event.announced_on > latest[period].announced_on:
                latest[period] = event
        return tuple(latest[key] for key in sorted(latest))
