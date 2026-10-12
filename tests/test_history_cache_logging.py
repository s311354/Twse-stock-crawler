"""MOPS progress details are debug-only; release receives one run summary."""
import datetime as dt
import logging

import pandas as pd
import pytest

from application.contracts import AnalysisContext, StockAnalysisRequest
from application.mops_refresh_control import MopsRefreshPolicy
from domain.fundamental_derivation import FundamentalBundle
from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.fundamentals.mops_client import FilingAnnouncement
from tests.fundamentals.test_filing_refresh_planner import (
    _FakeMopsClient, _candidate, _fundamental_service,
)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_progress_is_debug_only_and_summary_is_consumed(tmp_path, caplog, level):
    repo = MopsRepository(tmp_path / "mops")
    candidates = [_candidate(year, quarter) for year in range(2021, 2026) for quarter in range(1, 5)]
    announcements = [FilingAnnouncement(symbol="2330", fiscal_year=c.fiscal_year,
        quarter=c.quarter, available_at=c.available_at, title="financial report") for c in candidates]
    service = _fundamental_service(_FakeMopsClient(repo, candidates, announcements), repo,
        refresh_policy=MopsRefreshPolicy(max_filings_per_symbol=4, cooldown_seconds=0))
    service._company_sectors = {"2330": "24"}
    service.begin_run()
    with caplog.at_level(level):
        service.load("2330", signal_date="2026-07-21T13:30:00+08:00", sector="ELEC", history_years=5)
    details = [r for r in caplog.records if getattr(r, "status", None) == "HISTORY_CACHE_PROGRESS"]
    assert len(details) == (1 if level == logging.DEBUG else 0)
    assert service.consume_history_cache_summary() == dict(completed_companies=0, partial_companies=1, pending_work=16)
    assert service.consume_history_cache_summary() == dict(completed_companies=0, partial_companies=0, pending_work=0)
    service._history_cache_progress.update({"2330": (False, 6), "2317": (True, 0)})
    service._history_cache_progress["2330"] = (True, 0)
    assert service.consume_history_cache_summary()["completed_companies"] == 2
    service._history_cache_progress["2330"] = (False, 6)
    service.begin_run()
    assert service.consume_history_cache_summary()["partial_companies"] == 0


def test_coordinator_emits_one_summary_after_all_companies(caplog):
    class Crawler:
        stocknumbers = ["2330", "2317"]
        stocknames = ["A", "B"]
        daily_dates = [["2026-08-03"], ["2026-08-03"]]

        def set_low_entry_v4_fundamentals(self, *args, **kwargs):
            pass

    class Service:
        def __init__(self):
            self.loaded = []
            self.consumed = False

        def load(self, symbol, **kwargs):
            self.loaded.append(symbol)
            return FundamentalBundle(pd.DataFrame(), {"available_at": "2026-08-03T13:45:00+08:00"})

        def consume_history_cache_summary(self):
            assert len(self.loaded) == 2
            assert not self.consumed
            self.consumed = True
            return dict(completed_companies=1, partial_companies=1, pending_work=6)

    service = Service()
    with caplog.at_level(logging.INFO):
        FundamentalRefreshCoordinator(service).refresh(
            AnalysisContext(Crawler(), dt.date(2026, 8, 3), "OPEN"),
            StockAnalysisRequest(stocklist=["2330", "2317"], stocktype="ELEC", output_file_names="test", endbacktrack=3),
            analysis_time=dt.datetime.fromisoformat("2026-08-03T13:45:00+08:00"),
        )
    summaries = [r for r in caplog.records if getattr(r, "status", None) == "HISTORY_CACHE_SUMMARY"]
    assert len(summaries) == 1
    assert summaries[0].summary == dict(completed_companies=1, partial_companies=1, pending_work=6)
    assert summaries[0].source == "MOPS_CACHE"
