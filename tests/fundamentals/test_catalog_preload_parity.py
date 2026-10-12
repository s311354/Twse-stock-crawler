"""Synthetic catalog work completeness never implies financial completeness."""
from dataclasses import replace

import pandas as pd
import pytest

from application.daily_performance import DailyPerformance
from domain.fundamental_models import FilingCorrection
from domain.strategies.low_entry_v6_quality import evaluate_quality
from tests.fundamentals.test_incremental_snapshot_refresh import seeded, CUTOFF, prohibit_facts
from tests.fundamentals.test_filing_refresh_planner import _candidate


def test_partial_catalog_preload_matches_individual_load_in_every_field(tmp_path, monkeypatch):
    repo, client, service = seeded(tmp_path)
    reference = service.load("2330", signal_date=CUTOFF, sector="24")
    monkeypatch.setattr(repo, "load_filed_financials", prohibit_facts)
    before_discoveries = client.discovery_calls
    probe = DailyPerformance()
    with probe.activate():
        preloaded = service.load_cached_bundles(["2330"], signal_date=CUTOFF, sector="24")
    result = preloaded["2330"]
    assert result.snapshot == reference.snapshot and result.as_of == reference.as_of
    pd.testing.assert_frame_equal(result.annual_fundamentals, reference.annual_fundamentals, check_exact=True)
    quality = evaluate_quality(result.snapshot["les_v6_mops_filings"], symbol="2330", market="TWSE", as_of=CUTOFF)
    assert quality.status != "PASS"
    assert client.discovery_calls == before_discoveries
    assert probe.counters["fundamental_snapshot_decodes"] == 1
    assert probe.counters["fundamental_catalog_queries"] == 1


@pytest.mark.parametrize("change", ["announcement", "scan", "download", "correction", "monthly", "future_catalog"])
def test_pending_or_future_catalog_cannot_enable_preload(tmp_path, monkeypatch, change):
    repo, _, service = seeded(tmp_path)
    state = repo.load_filing_catalog("2330", as_of=CUTOFF)
    if change == "announcement":
        state = replace(state, announcements_pending=True)
    elif change == "scan":
        state = replace(state, scanned_periods=())
    elif change == "download":
        state = replace(state, candidates=state.candidates + (_candidate(2025, 3),))
    elif change == "correction":
        event = FilingCorrection(symbol="2330", fiscal_year=2023, quarter=4,
                                 announced_on="20260720", description="SYNTHETIC correction")
        state = replace(state, corrections=(event,), verified_corrections=())
        service._daily_corrections = {"2330": {(2023, 4): event}}
    elif change == "monthly":
        state = replace(state, checked_at="2026-06-01T00:00:00+00:00")
    else:
        state = replace(state, checked_at="2026-07-22T00:00:00+00:00")
    repo.save_filing_catalog(state)
    monkeypatch.setattr(repo.queries, "_decode_bundle", prohibit_facts)
    assert service.load_cached_bundles(["2330"], signal_date=CUTOFF, sector="24") == {}


def test_acknowledged_correction_is_not_refreshed_again(tmp_path):
    repo, _, service = seeded(tmp_path)
    event = FilingCorrection(symbol="2330", fiscal_year=2023, quarter=4,
                             announced_on="20260720", description="SYNTHETIC identical revision")
    state = repo.load_filing_catalog("2330", as_of=CUTOFF)
    repo.save_filing_catalog(replace(state, corrections=(event,), verified_corrections=(event,)))
    service._daily_corrections = {"2330": {(2023, 4): event}}
    reference = service.load("2330", signal_date=CUTOFF, sector="24")
    result = service.load_cached_bundles(["2330"], signal_date=CUTOFF, sector="24")["2330"]
    assert result.snapshot == reference.snapshot


def test_catalog_query_is_bounded_single_connection_and_cutoff_filtered(tmp_path, monkeypatch):
    repo, _, _ = seeded(tmp_path)
    repo.queries._BULK_SYMBOL_CHUNK_SIZE = 2
    original, connections, statements = repo._connect, [], []
    def connect():
        connection = original()
        connections.append(connection)
        connection.set_trace_callback(statements.append)
        return connection
    monkeypatch.setattr(repo, "_connect", connect)
    symbols = ["2330", "EMPTY1", "EMPTY2", "EMPTY3", "EMPTY4", "2330"]
    with repo.unit_of_work() as uow:
        result = repo.load_filing_catalogs_many(symbols, as_of=CUTOFF, uow=uow)
    assert len(connections) == 1
    selects = [sql for sql in statements if sql.startswith("SELECT")]
    assert len(selects) == 3
    assert all("facts" not in sql and "julianday" not in sql and "snapshot_json" not in sql for sql in selects)
    assert result == {"2330": repo.load_filing_catalog("2330", as_of=CUTOFF)}
    assert repo.load_filing_catalogs_many(symbols, as_of="2026-01-01T00:00:00Z") == {}


def test_repository_without_bulk_catalog_extension_remains_conservative(tmp_path, monkeypatch):
    repo, client, service = seeded(tmp_path)
    reference = service.load("2330", signal_date=CUTOFF, sector="24")
    discoveries = client.discovery_calls
    monkeypatch.setattr(repo, "load_filing_catalogs_many", None)
    assert service.load_cached_bundles(["2330"], signal_date=CUTOFF, sector="24") == {}
    assert service.load("2330", signal_date=CUTOFF, sector="24").snapshot == reference.snapshot
    assert client.discovery_calls == discoveries
