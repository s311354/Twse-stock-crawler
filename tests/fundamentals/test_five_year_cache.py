"""Offline cache-first history, revision, and SQLite regression coverage."""

import datetime as dt
import sqlite3

import pandas as pd
import pytest

import application.fundamental_service as service_module
from application.filing_refresh_planner import FilingRefreshPlanner
from application.fundamental_refresh import RefreshPlanner
from application.mops_refresh_control import MopsRefreshPolicy
from domain.errors import AccessBlockedError
from domain.fundamental_models import FilingAnnouncement, FilingCatalogState, FilingCorrection, FilingDocument
from infrastructure.fundamentals.mops_repository import MopsRepository
from tests.fundamentals.test_filing_refresh_planner import (
    _FakeMopsClient, _candidate, _fact, _filing, _fundamental_service,
)


def history():
    return [_candidate(year, quarter) for year in range(2021, 2026)
            for quarter in (1, 2, 3, 4)] + [_candidate(2026, 1), _candidate(2026, 2)]


class HistoryClient(_FakeMopsClient):
    def __init__(self, repo, candidates):
        super().__init__(repo, candidates, [FilingAnnouncement(
            item.symbol, item.fiscal_year, item.quarter, item.available_at, "財務報告",
        ) for item in candidates])


def load(repo, client, day, *, batch_size=100):
    service = _fundamental_service(client, repo, refresh_policy=MopsRefreshPolicy(
        max_filings_per_symbol=batch_size, cooldown_seconds=0,
    ))
    service._company_sectors = {"2330": "24"}
    return service.load("2330", signal_date=day, sector="ELEC", history_years=5)


def test_history_window_keeps_all_twenty_quarters_and_current_ytd():
    planner = FilingRefreshPlanner()
    candidates = [_candidate(2020, 4)] + history()
    plan = planner.plan(candidates, [], history_years=5)
    assert len(plan.download) == 22
    assert plan.download == history()
    assert planner.plan(list(reversed(candidates)), [], history_years=5) == plan
    # A missing old quarter is a genuine gap, not repaired with an annual report.
    cached = [_filing(item) for item in history()]
    assert RefreshPlanner.has_complete_cached_filing_history(
        cached, history_years=5, as_of=pd.Timestamp("2026-07-21", tz="UTC"),
    )
    assert not RefreshPlanner.has_complete_cached_filing_history(
        cached[1:], history_years=5, as_of=pd.Timestamp("2026-07-21", tz="UTC"),
    )
    assert planner.select_required_filings(candidates[2:], 5) == history()[1:]


def test_five_year_backfill_resumes_across_processes_without_rediscovery(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    client = HistoryClient(repo, history())
    for offset in range(6):
        bundle = load(MopsRepository(repo.root), client, f"2026-07-{21 + offset}T13:30:00+08:00", batch_size=4)
        assert len(client.downloaded_periods) == min((offset + 1) * 4, 22)
        assert client.discovery_calls == 1
        assert bundle.snapshot["remote_refresh_status"] == ("PARTIAL" if offset < 5 else "SUCCESS")
        assert bundle.snapshot["history_cache_progress"] == {
            "years": 5, "cached_periods": min((offset + 1) * 4, 22), "expected_periods": 22,
            "status": "PARTIAL" if offset < 5 else "COMPLETE",
        }
    assert len(set(client.downloaded_periods)) == 22
    assert len(repo.load_filed_financials("2330", "2026-07-27T06:00:00Z")) == 22
    assert repo.last_full_refresh_epoch("2330") is not None
    load(MopsRepository(repo.root), client, "2026-07-27T13:30:00+08:00")
    assert client.discovery_calls == 1
    assert len(client.downloaded_periods) == 22


@pytest.mark.parametrize("changed_hash", [False, True])
def test_monthly_old_correction_hash_controls_snapshot_reuse(tmp_path, monkeypatch, changed_hash):
    repo = MopsRepository(tmp_path / "mops")

    class Client(HistoryClient):
        changed = False

        def download(self, candidate):
            document = super().download(candidate)
            if self.changed and (candidate.fiscal_year, candidate.quarter) == (2022, 4):
                return FilingDocument(candidate, document.content + b"-corrected", "2026-08-01T05:00:00Z")
            return document

    client = Client(repo, history())
    derivations = []
    original = service_module.derive_fundamentals

    def derive(*args, **kwargs):
        derivations.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(service_module, "derive_fundamentals", derive)
    load(repo, client, "2026-07-21T13:30:00+08:00")
    old = repo.latest_filing("2330", 2022, 4)
    client.corrections = [FilingCorrection("2330", 2022, 4, "20260725", "更正財報")]
    client.changed = changed_hash
    # Daily recent-period scan must not download historical years.
    service = _fundamental_service(client, repo)
    service._company_sectors = {"2330": "24"}
    service.prepare_daily_revisions(dt.datetime(2026, 7, 26, tzinfo=dt.timezone.utc))
    service.load("2330", signal_date="2026-07-26T13:30:00+08:00", sector="ELEC")
    assert client.correction_scan_periods == [[(2026, 1), (2026, 2)]]
    assert len(client.downloaded_periods) == 22
    load(repo, client, "2026-08-01T13:30:00+08:00")
    assert client.discovery_calls == 2
    assert client.downloaded_periods[22:] == [(2022, 4)]
    assert len(derivations) == (2 if changed_hash else 1)
    revised = repo.latest_filing("2330", 2022, 4)
    assert revised.revision_number == old.revision_number + int(changed_hash)
    if changed_hash:
        before = repo.load_filed_financials("2330", "2026-07-31T23:59:59Z")
        assert all(item.content_sha256 != revised.content_sha256 for item in before)
    else:
        assert revised == old  # Verification does not rewrite original availability.
    # An identical notice is acknowledged even when content was unchanged.
    for day in ("2026-08-02T13:30:00+08:00", "2026-09-01T13:30:00+08:00"):
        load(MopsRepository(repo.root), client, day)
    assert client.downloaded_periods[22:] == [(2022, 4)]
    assert len(derivations) == (2 if changed_hash else 1)


def test_monthly_correction_scan_resumes_after_blocking_without_restarting_catalog(tmp_path):
    repo = MopsRepository(tmp_path / "mops")

    class Client(HistoryClient):
        calls = []
        blocked = False

        def latest_correction(self, candidate):
            period = candidate.fiscal_year, candidate.quarter
            self.calls.append(period)
            if self.blocked and period == (2021, 3):
                raise AccessBlockedError("MOPS security guard")
            return super().latest_correction(candidate)

    client = Client(repo, history())
    load(repo, client, "2026-07-21T13:30:00+08:00")
    client.calls = []
    client.blocked = True
    load(repo, client, "2026-08-01T13:30:00+08:00")
    state = repo.load_filing_catalog("2330", as_of="2026-08-02T06:00:00Z")
    assert state.scanned_periods == ((2021, 1), (2021, 2))
    client.blocked = False
    load(MopsRepository(repo.root), client, "2026-08-02T13:30:00+08:00")
    assert client.discovery_calls == 2
    assert client.calls[:4] == [(2021, 1), (2021, 2), (2021, 3), (2021, 3)]
    assert len(client.downloaded_periods) == 22


def test_incomplete_catalog_does_not_claim_five_year_financial_coverage(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    client = HistoryClient(repo, [_candidate(2025, 4)])
    load(repo, client, "2026-07-21T13:30:00+08:00")
    load(MopsRepository(repo.root), client, "2026-07-22T13:30:00+08:00")
    assert client.discovery_calls == 1  # No repetitive daily catalog retry for unavailable history.
    assert not RefreshPlanner.has_complete_cached_filing_history(
        repo.load_filings("2330"), history_years=5, as_of=pd.Timestamp("2026-07-22", tz="UTC"),
    )


def test_catalog_roundtrip_asof_and_company_single_connection_transaction(tmp_path, monkeypatch):
    repo = MopsRepository(tmp_path / "mops")
    repo._ensure_schema()
    connect = repo._connect
    connections = []
    statements = []

    def counted_connect():
        connection = connect()
        connections.append(connection)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(repo, "_connect", counted_connect)
    state = FilingCatalogState("2330", "2026-07-21T05:30:00Z", tuple(history()))
    with repo.unit_of_work() as uow:
        repo.save_filing_catalog(state, uow=uow)
        repo.save_revision(
            _candidate(2025, 4), b"one", [_fact(100)],
            available_at="2026-02-01T00:00:00Z", fetched_at="2026-02-01T00:00:00Z", uow=uow,
        )
        repo.save_revision(
            _candidate(2025, 3), b"empty", [],
            available_at="2026-02-01T00:00:00Z", fetched_at="2026-02-01T00:00:00Z", uow=uow,
        )
        filings = repo.load_filed_financials("2330", "2026-07-22T00:00:00Z", uow=uow)
        assert len(filings) == 2 and filings[0].facts == ()  # LEFT JOIN preserves no-fact filings.
        assert repo.load_filing_catalog("2330", as_of="2026-07-22T00:00:00Z", uow=uow) == state
        assert repo.load_filing_catalog("2330", as_of="2026-07-20T00:00:00Z", uow=uow) is None
        query = next(sql for sql in statements if "LEFT JOIN facts" in sql)
        # SQLite may prefer the period-order index for tiny samples. Verify
        # the epoch predicate is indexable without forcing production's planner.
        indexed = query.replace("filings AS f", "filings AS f INDEXED BY filings_symbol_available_epoch")
        explain = uow.connection.execute("EXPLAIN QUERY PLAN " + indexed).fetchall()
        assert any("filings_symbol_available_epoch" in str(row) for row in explain)
        assert any("available_at_epoch<?" in str(row) for row in explain)
        assert "julianday" not in query.lower()
    assert len(connections) == 1
    assert sum(sql == "BEGIN" for sql in statements) == 1
    assert sum(sql == "COMMIT" for sql in statements) == 1
    assert sum("LEFT JOIN facts" in sql for sql in statements) == 1


def test_catalog_and_filing_roll_back_together(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    with pytest.raises(RuntimeError, match="abort"):
        with repo.unit_of_work() as uow:
            repo.save_filing_catalog(FilingCatalogState("2330", "2026-07-21T00:00:00Z", tuple(history())), uow=uow)
            repo.save_revision(
                _candidate(2025, 4), b"one", [_fact(100)],
                available_at="2026-02-01T00:00:00Z", fetched_at="2026-02-01T00:00:00Z", uow=uow,
            )
            raise RuntimeError("abort")
    assert repo.load_filings("2330") == []
    assert repo.load_filing_catalog("2330", as_of="2026-07-22T00:00:00Z") is None


def test_announcement_block_preserves_catalog_for_next_run(tmp_path):
    repo = MopsRepository(tmp_path / "mops")

    class Client(HistoryClient):
        blocked = True

        def discover_announcements(self, symbol, years):
            if self.blocked:
                raise AccessBlockedError("MOPS security guard")
            return super().discover_announcements(symbol, years)

    client = Client(repo, history())
    with pytest.raises(AccessBlockedError):
        load(repo, client, "2026-07-21T13:30:00+08:00")
    state = repo.load_filing_catalog("2330", as_of="2026-07-22T00:00:00Z")
    assert state.announcements_pending
    assert len(state.candidates) == 22
    assert repo.load_filings("2330") == []
    client.blocked = False
    load(MopsRepository(repo.root), client, "2026-07-22T13:30:00+08:00")
    assert client.discovery_calls == 1
    assert len(repo.load_filings("2330")) == 22


def test_service_borrows_one_company_connection_for_entire_refresh(tmp_path, monkeypatch):
    repo = MopsRepository(tmp_path / "mops")
    repo._ensure_schema()
    original = repo._connect
    calls = []

    def connect():
        calls.append(1)
        return original()

    monkeypatch.setattr(repo, "_connect", connect)
    load(repo, HistoryClient(repo, history()), "2026-07-21T13:30:00+08:00")
    assert len(calls) == 1


def test_legacy_database_migrates_epoch_and_preserves_asof_boundary(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    repo.root.mkdir()
    with sqlite3.connect(repo.database_path) as connection:
        connection.execute("""
            CREATE TABLE filings (
                id INTEGER PRIMARY KEY, symbol TEXT, fiscal_year INTEGER, quarter INTEGER,
                report_id TEXT, available_at TEXT, fetched_at TEXT, revision_number INTEGER,
                content_sha256 TEXT, source_url TEXT, raw_path TEXT
            )
        """)
        connection.execute("INSERT INTO filings VALUES (1, '2330', 2025, 4, 'C', ?, ?, 1, 'hash', 'source', 'raw')",
                           ("2026-02-10T18:16:19+08:00", "2026-02-10T18:16:19+08:00"))
    assert repo.load_filed_financials("2330", "2026-02-10T10:16:18Z") == []
    assert len(repo.load_filed_financials("2330", "2026-02-10T10:16:19Z")) == 1
    with repo.company_connection() as connection:
        epoch = connection.execute("SELECT available_at_epoch, typeof(available_at_epoch) FROM filings").fetchone()
        assert epoch == (int(pd.Timestamp("2026-02-10T10:16:19Z").timestamp()), "integer")
        assert connection.execute("SELECT COUNT(*) FROM filing_catalog_cache").fetchone()[0] == 0
