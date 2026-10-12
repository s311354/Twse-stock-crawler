"""Synthetic/offline contracts for exact-selection incremental refresh."""

import datetime as dt
from dataclasses import replace

import pandas as pd
import pytest

import application.fundamental_deriver as deriver_module
from application.mops_refresh_control import MopsRefreshPolicy
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from domain.fundamental_models import FilingAnnouncement, FilingCorrection
from domain.strategies.low_entry_v6_quality import evaluate_quality
from infrastructure.fundamentals.mops_repository import MopsRepository
from tests.fundamentals.test_filing_refresh_planner import (
    _FakeMopsClient, _candidate, _fact, _fundamental_service,
)


CUTOFF = "2026-07-21T05:30:00+00:00"


def seeded(tmp_path, *, complete=False):
    repo = MopsRepository(tmp_path / "synthetic-mops")
    quarters = (1, 2, 3, 4) if complete else (4,)
    candidates = [_candidate(year, quarter) for year in range(2021, 2026) for quarter in quarters]
    announcements = [FilingAnnouncement(
        symbol="2330", fiscal_year=item.fiscal_year, quarter=item.quarter,
        available_at=item.available_at, title="SYNTHETIC 財務報告",
    ) for item in candidates]
    client = _FakeMopsClient(repo, candidates, announcements)
    svc = _fundamental_service(client, repo)
    svc._company_sectors = {"2330": "24"}
    svc.load("2330", signal_date=CUTOFF, sector="24")
    return repo, client, svc


def cached(svc, cutoff=CUTOFF, sector="24"):
    with svc.repository.unit_of_work() as uow:
        return svc._load_cached_bundle(
            "2330", parsed_signal=pd.Timestamp(cutoff), sector=sector, uow=uow,
        )[0]


def prohibit_facts(*args, **kwargs):
    raise AssertionError("unchanged selection must not reload facts or parse raw units")


def test_partial_snapshot_reuses_exact_selection_without_facts(tmp_path, monkeypatch):
    repo, client, svc = seeded(tmp_path)
    before = cached(svc)
    diagnostic = svc.inspect_local_evidence(["2330"], signal_date=CUTOFF)
    # The stored annual original is available: inspect its required reported
    # fields locally, rather than download diagnostic-only missing quarters.
    assert not diagnostic["2330"]["backfill_needed"]
    assert diagnostic["2330"]["diagnostic_missing_periods"]
    assert diagnostic["2330"]["local_review_required"]
    assert diagnostic["2330"]["quality_status"] != "PASS"
    monkeypatch.setattr(repo, "load_filed_financials", prohibit_facts)
    monkeypatch.setattr(svc.filing_ingestion.parser, "parse", prohibit_facts)
    monkeypatch.setattr(svc.fundamental_deriver, "_derive_bundle", prohibit_facts)
    after = cached(svc)
    pd.testing.assert_frame_equal(before.annual_fundamentals, after.annual_fundamentals)
    assert before.snapshot == after.snapshot
    assert svc.inspect_local_evidence(["2330"], signal_date=CUTOFF) == diagnostic
    assert svc.migrate_local_cache(["2330"], signal_date=CUTOFF, sector="24") == 0
    svc.load("2330", signal_date="2026-07-22T05:30:00+00:00", sector="24")
    assert len(client.downloaded_periods) == 5


def test_future_revision_changes_only_later_cutoff_selection(tmp_path, monkeypatch):
    repo, _, svc = seeded(tmp_path)
    before = cached(svc)
    repo.save_revision(
        _candidate(2025, 4), b"synthetic-future-revision", [_fact(999.)],
        available_at="2026-07-23T05:00:00+00:00", fetched_at="2026-07-23T05:01:00+00:00",
    )
    with monkeypatch.context() as patch:
        patch.setattr(repo, "load_filed_financials", prohibit_facts)
        assert cached(svc).snapshot == before.snapshot
    calls = []
    original = repo.load_filed_financials
    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(repo, "load_filed_financials", counted)
    later = cached(svc, "2026-07-23T05:30:00+00:00")
    assert calls == ["2330"]
    assert len(later.snapshot["les_v6_mops_filings"]) == 6
    assert later.snapshot["available_at"] == "2026-07-23T05:00:00+00:00"
    calls.clear()
    assert cached(svc, "2026-07-23T05:30:00+00:00").snapshot == later.snapshot
    assert calls == []


@pytest.mark.parametrize("change", ["derivation", "sector", "schema", "provenance"])
def test_changed_identity_or_format_reads_facts_once(tmp_path, monkeypatch, change):
    repo, _, svc = seeded(tmp_path)
    sector = "24"
    if change == "derivation":
        monkeypatch.setattr(deriver_module, "FUNDAMENTAL_DERIVATION_VERSION", "synthetic-next-version")
    elif change == "sector":
        svc._company_sectors["2330"] = "25"
        sector = "25"
    elif change == "schema":
        filings = repo.load_filings_many(["2330"], as_of=CUTOFF)["2330"]
        key = svc.fundamental_deriver.snapshot_cache_key(repo.filing_set_hash(filings))
        bundle = repo.load_fundamental_bundle("2330", key, "24")
        repo.save_fundamental_bundle("2330", key, "24", replace(
            bundle, snapshot=dict(bundle.snapshot, les_v6_filing_schema_version="old"),
        ))
    else:
        filing = repo.load_filings("2330")[0]
        repo.update_availability_provenance(
            filing, available_at="2026-07-20T00:00:00+00:00", availability_source="FIRST_OBSERVED_DOWNLOAD",
        )
    calls = []
    original = repo.load_filed_financials
    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(repo, "load_filed_financials", counted)
    before = cached(svc, sector=sector)
    assert calls == ["2330"]
    calls.clear()
    assert cached(svc, sector=sector).snapshot == before.snapshot
    assert calls == []


def test_bulk_preload_rejects_stale_active_selection(tmp_path, monkeypatch):
    repo, _, svc = seeded(tmp_path, complete=True)
    stale = repo.load_cached_bundle_headers({"2330": "24"}, as_of=CUTOFF,
                                    derivation_version=FUNDAMENTAL_DERIVATION_VERSION)
    repo.save_revision(_candidate(2025, 4), b"synthetic-new", [_fact(999.)],
                       available_at="2026-07-22T05:00:00+00:00", fetched_at="2026-07-22T05:00:00+00:00")
    # Defensive validation even if a stale active pointer survives a legacy writer.
    monkeypatch.setattr(repo, "load_cached_bundle_headers", lambda *args, **kwargs: stale)
    monkeypatch.setattr(repo, "load_filed_financials", prohibit_facts)
    assert svc.load_cached_bundles(["2330"], signal_date="2026-07-22T05:30:00+00:00", sector="24") == {}


def test_daily_scan_only_two_periods_and_monthly_catalog_does_not_reparse(tmp_path, monkeypatch):
    repo, client, svc = seeded(tmp_path)
    scanned = []
    original = client.latest_correction
    def counted(candidate):
        scanned.append((candidate.fiscal_year, candidate.quarter))
        return original(candidate)
    monkeypatch.setattr(client, "latest_correction", counted)
    monkeypatch.setattr(repo, "load_filed_financials", prohibit_facts)
    monkeypatch.setattr(svc.filing_ingestion.parser, "parse", prohibit_facts)
    svc.begin_run()
    svc.prepare_daily_revisions(dt.datetime.fromisoformat("2026-07-22T05:30:00+00:00"))
    svc.load("2330", signal_date="2026-07-22T05:30:00+00:00", sector="24")
    assert client.correction_scan_periods == [[(2026, 1), (2026, 2)]]
    assert scanned == []
    assert client.discovery_calls == 1
    svc.begin_run()
    svc.load("2330", signal_date="2026-08-01T05:30:00+00:00", sector="24")
    assert client.discovery_calls == 2
    assert set(scanned) == {(year, 4) for year in range(2021, 2026)}
    assert len(client.downloaded_periods) == 5


def test_identical_hash_correction_acknowledgement_keeps_cached_facts(tmp_path, monkeypatch):
    repo, client, svc = seeded(tmp_path)
    svc._daily_corrections = {"2330": {(2023, 4): FilingCorrection(
        symbol="2330", fiscal_year=2023, quarter=4, announced_on="20260722", description="SYNTHETIC 更正",
    )}}
    monkeypatch.setattr(repo, "load_filed_financials", prohibit_facts)
    monkeypatch.setattr(svc.filing_ingestion.parser, "parse", prohibit_facts)
    svc.begin_run()
    svc.load("2330", signal_date="2026-07-22T05:30:00+00:00", sector="24")
    assert client.downloaded_periods[-1] == (2023, 4)
    assert len(client.downloaded_periods) == 6
    svc.begin_run()
    svc.load("2330", signal_date="2026-07-23T05:30:00+00:00", sector="24")
    assert len(client.downloaded_periods) == 6


def test_deferred_backfill_reuses_partial_snapshot_without_admitting_entry(tmp_path, monkeypatch):
    repo, _, svc = seeded(tmp_path)
    monkeypatch.setattr(repo, "load_filed_financials", prohibit_facts)
    svc.refresh_control.policy = MopsRefreshPolicy(max_symbols=0, cooldown_seconds=0)
    svc.begin_run()
    # A monthly check is due, so the remote guard must actually be exercised.
    result = svc.load("2330", signal_date="2026-08-01T05:30:00+00:00", sector="24")
    assert result.snapshot["remote_refresh_status"] == "SYMBOL_BUDGET_EXHAUSTED"
    quality = evaluate_quality(result.snapshot["les_v6_mops_filings"], symbol="2330", market="TWSE", as_of=result.as_of)
    assert quality.status != "PASS"


def test_bulk_metadata_and_preload_share_one_connection_and_no_facts(tmp_path, monkeypatch):
    repo, _, svc = seeded(tmp_path, complete=True)
    # Force small chunks to prove the SQL parameter bound without large fixtures.
    repo.queries._BULK_SYMBOL_CHUNK_SIZE = 2
    symbols = ["2330", "EMPTY1", "EMPTY2", "EMPTY3", "EMPTY4"]
    connections = []
    statements = []
    original = repo._connect
    def counted():
        connection = original()
        connections.append(connection)
        connection.set_trace_callback(statements.append)
        return connection
    monkeypatch.setattr(repo, "_connect", counted)
    monkeypatch.setattr(repo, "load_filed_financials", prohibit_facts)
    with repo.unit_of_work() as uow:
        metadata = repo.load_filings_many(symbols, as_of=CUTOFF, uow=uow)
        entries = repo.load_cached_bundles(dict.fromkeys(symbols, "24"), as_of=CUTOFF,
                                          derivation_version=FUNDAMENTAL_DERIVATION_VERSION, uow=uow)
    assert len(connections) == 1
    assert len(metadata["2330"]) == 20
    assert all(metadata[symbol] == [] for symbol in symbols[1:])
    assert set(entries) == {"2330"}
    selects = [statement.lower() for statement in statements if statement.lstrip().lower().startswith("select")]
    assert len(selects) == 6  # Three bounded metadata + three snapshot queries.
    assert all("facts" not in statement and "julianday" not in statement for statement in selects)
    connections.clear()
    statements.clear()
    svc.load_cached_bundles(symbols, signal_date=CUTOFF, sector="24")
    assert len(connections) == 1


def test_metadata_hash_matches_facts_hash_at_cutoff(tmp_path):
    repo, _, _ = seeded(tmp_path)
    repo.save_revision(_candidate(2025, 4), b"synthetic-future", [_fact(999.)],
                       available_at="2026-07-23T05:00:00+00:00", fetched_at="2026-07-23T05:00:00+00:00")
    for cutoff, count in [(CUTOFF, 5), ("2026-07-23T05:30:00+00:00", 6)]:
        with repo.unit_of_work() as uow:
            metadata = repo.load_filings_many(["2330"], as_of=cutoff, uow=uow)["2330"]
            financials = repo.load_filed_financials("2330", cutoff, uow=uow)
        assert len(metadata) == len(financials) == count
        assert repo.filing_set_hash(metadata) == repo.filing_set_hash(financials)


def test_hash_alignment_does_not_change_legacy_fact_order_for_empty_report(tmp_path):
    repo = MopsRepository(tmp_path / "synthetic-mops")
    first = repo.save_revision(_candidate(2025, 4), b"synthetic-consolidated", [_fact(1000.)],
                               available_at=CUTOFF, fetched_at=CUTOFF)
    empty = repo.save_revision(replace(_candidate(2025, 4), report_id="A", statement_scope="individual"),
                               b"synthetic-empty-individual", [], available_at=CUTOFF, fetched_at=CUTOFF)
    with repo.unit_of_work() as uow:
        metadata = repo.load_filings_many(["2330"], as_of=CUTOFF, uow=uow)["2330"]
        financials = repo.load_filed_financials("2330", CUTOFF, uow=uow)
    assert [item.filing_id for item in metadata] == [first.filing_id, empty.filing_id]
    # Original LEFT JOIN semantics: NULL fact IDs precede populated reports.
    assert [item.filing_id for item in financials] == [empty.filing_id, first.filing_id]
    assert repo.filing_set_hash(metadata) == repo.filing_set_hash(financials)
