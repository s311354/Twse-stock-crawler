"""Offline/synthetic header-first refresh: no new financial scoring policy."""
from dataclasses import replace
import sqlite3

import pandas as pd
import pytest

from application.daily_performance import DailyPerformance
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
from domain.strategies.low_entry_v6_quality import evaluate_quality
from infrastructure.fundamentals.mops_schema import MopsSchemaMigrator
from tests.fundamentals.test_incremental_snapshot_refresh import CUTOFF, seeded, cached, prohibit_facts


def test_headers_use_covering_index_not_large_snapshot_table_rows(tmp_path):
    repo, _, _ = seeded(tmp_path)
    statements = []
    with repo.unit_of_work() as uow:
        uow.connection.set_trace_callback(statements.append)
        result = repo.load_cached_bundle_headers({"2330": "24"}, as_of=CUTOFF,
            derivation_version=FUNDAMENTAL_DERIVATION_VERSION, uow=uow)
        uow.connection.set_trace_callback(None)
        query = next(sql for sql in statements if "INDEXED BY fundamental_snapshots_header" in sql)
        plan = uow.connection.execute("EXPLAIN QUERY PLAN " + query).fetchall()
        assert any("COVERING INDEX fundamental_snapshots_header" in row[3] for row in plan)
        before = uow.connection.execute(query).fetchall()
        after = uow.connection.execute(query.replace("INDEXED BY fundamental_snapshots_header",
            "INDEXED BY sqlite_autoindex_fundamental_snapshots_1")).fetchall()
        assert before == after
        columns = [row[2] for row in uow.connection.execute("PRAGMA index_info(fundamental_snapshots_header)")]
        assert columns == ["symbol", "filing_set_sha256", "sector", "filing_schema_version", "has_mops_filings"]
        assert "annual_json" not in columns and "snapshot_json" not in columns
    assert "2330" in result


def test_legacy_cache_creates_header_index_once_without_rewriting_payload(tmp_path):
    repo, _, _ = seeded(tmp_path)
    with repo.unit_of_work() as uow:
        before = uow.connection.execute("SELECT annual_json,snapshot_json,created_at_epoch FROM fundamental_snapshots").fetchall()
        uow.connection.execute("DROP INDEX fundamental_snapshots_header")
    probe = DailyPerformance()
    with probe.activate():
        reopened = type(repo)(repo.root)
        reopened.load_cached_bundle_headers({"2330": "24"}, as_of=CUTOFF,
            derivation_version=FUNDAMENTAL_DERIVATION_VERSION)
        reopened.load_cached_bundle_headers({"2330": "24"}, as_of=CUTOFF,
            derivation_version=FUNDAMENTAL_DERIVATION_VERSION)
    with reopened.unit_of_work() as uow:
        assert uow.connection.execute("SELECT annual_json,snapshot_json,created_at_epoch FROM fundamental_snapshots").fetchall() == before
    assert probe.counters["fundamental_header_indexes_created"] == 1
    assert probe.counters["fundamental_snapshot_decodes"] == 0


def test_current_migration_and_ineligible_preload_never_decode_json(tmp_path, monkeypatch):
    repo, _, service = seeded(tmp_path)
    state = repo.load_filing_catalog("2330", as_of=CUTOFF)
    repo.save_filing_catalog(replace(state, announcements_pending=True))
    monkeypatch.setattr(repo, "load_filed_financials", prohibit_facts)
    monkeypatch.setattr(repo.queries, "_decode_bundle", prohibit_facts)
    probe = DailyPerformance()
    with probe.activate():
        assert service.migrate_local_cache(["2330"], signal_date=CUTOFF, sector="24") == 0
        assert service.load_cached_bundles(["2330"], signal_date=CUTOFF, sector="24") == {}
    assert probe.counters["fundamental_migration_header_skips"] == 1
    assert probe.counters["fundamental_preload_header_rejections"] == 1
    assert probe.counters["fundamental_snapshot_decodes"] == 0


@pytest.mark.parametrize("complete", [False, True])
def test_header_and_legacy_paths_have_identical_full_outputs(tmp_path, monkeypatch, complete):
    repo, _, service = seeded(tmp_path, complete=complete)
    def run():
        probe = DailyPerformance()
        with probe.activate():
            assert service.migrate_local_cache(["2330"], signal_date=CUTOFF, sector="24") == 0
            preloaded = service.load_cached_bundles(["2330"], signal_date=CUTOFF, sector="24")
            result = preloaded.get("2330") or cached(service)
            quality = evaluate_quality(result.snapshot["les_v6_mops_filings"],
                                       symbol="2330", market="TWSE", as_of=CUTOFF)
        return result, quality, probe
    optimized, quality, probe = run()
    with monkeypatch.context() as patch:
        patch.setattr(repo, "load_cached_bundle_headers", None)
        reference, reference_quality, before = run()
    assert optimized.snapshot == reference.snapshot
    assert optimized.as_of == reference.as_of
    pd.testing.assert_frame_equal(optimized.annual_fundamentals, reference.annual_fundamentals)
    assert quality == reference_quality
    assert quality.status != "PASS"  # Synthetic fixtures are not sufficient evidence.
    assert probe.counters["fundamental_snapshot_decodes"] == 1
    assert before.counters["fundamental_snapshot_decodes"] == 2


def test_legacy_index_is_lazy_and_preserves_payload_and_times(tmp_path):
    repo, _, service = seeded(tmp_path)
    with repo.unit_of_work() as uow:
        before = uow.connection.execute("SELECT annual_json,snapshot_json,created_at_epoch FROM fundamental_snapshots").fetchall()
        uow.connection.execute("UPDATE fundamental_snapshots SET filing_schema_version=NULL,has_mops_filings=NULL")
    probe = DailyPerformance()
    with probe.activate():
        assert service.migrate_local_cache(["2330"], signal_date=CUTOFF, sector="24") == 0
        assert service.migrate_local_cache(["2330"], signal_date=CUTOFF, sector="24") == 0
    with repo.unit_of_work() as uow:
        after = uow.connection.execute("SELECT annual_json,snapshot_json,created_at_epoch FROM fundamental_snapshots").fetchall()
    assert before == after
    assert probe.counters["fundamental_legacy_format_inspections"] == 1
    assert probe.counters["fundamental_snapshot_decodes"] == 1
    assert probe.counters["fundamental_migration_header_skips"] == 1


def test_format_index_requires_transaction_and_rolls_back(tmp_path):
    repo, _, service = seeded(tmp_path)
    header = repo.load_cached_bundle_headers({"2330": "24"}, as_of=CUTOFF,
                                            derivation_version=FUNDAMENTAL_DERIVATION_VERSION)["2330"]
    with repo.unit_of_work() as uow:
        uow.connection.execute("UPDATE fundamental_snapshots SET filing_schema_version=NULL,has_mops_filings=NULL")
    bundle = cached(service)
    with pytest.raises(ValueError, match="REQUIRES_INSPECTION_TRANSACTION"):
        repo.record_fundamental_bundle_format("2330", header.snapshot_cache_key, "24", bundle)
    with pytest.raises(RuntimeError, match="interrupted"):
        with repo.unit_of_work() as uow:
            repo.record_fundamental_bundle_format("2330", header.snapshot_cache_key, "24", bundle, uow=uow)
            raise RuntimeError("interrupted")
    current = repo.load_cached_bundle_headers({"2330": "24"}, as_of=CUTOFF,
                                             derivation_version=FUNDAMENTAL_DERIVATION_VERSION)["2330"]
    assert current.has_mops_filings is None


def test_format_index_cannot_certify_a_different_payload(tmp_path):
    repo, _, service = seeded(tmp_path)
    header = repo.load_cached_bundle_headers({"2330": "24"}, as_of=CUTOFF,
        derivation_version=FUNDAMENTAL_DERIVATION_VERSION)["2330"]
    bundle = cached(service)
    with repo.unit_of_work() as uow:
        uow.connection.execute("UPDATE fundamental_snapshots SET filing_schema_version=NULL,has_mops_filings=NULL")
        wrong = replace(bundle, snapshot=dict(bundle.snapshot, les_v6_filing_schema_version="invented"))
        repo.record_fundamental_bundle_format("2330", header.snapshot_cache_key, "24", wrong, uow=uow)
    current = repo.load_cached_bundle_headers({"2330": "24"}, as_of=CUTOFF,
        derivation_version=FUNDAMENTAL_DERIVATION_VERSION)["2330"]
    assert current.has_mops_filings is None


def test_current_header_with_wrong_cache_key_does_not_skip_migration(tmp_path, monkeypatch):
    repo, _, service = seeded(tmp_path)
    real = repo.load_cached_bundle_headers
    def wrong_key(*args, **kwargs):
        rows = real(*args, **kwargs)
        return {symbol: replace(entry, snapshot_cache_key="wrong-key") for symbol, entry in rows.items()}
    monkeypatch.setattr(repo, "load_cached_bundle_headers", wrong_key)
    probe = DailyPerformance()
    with probe.activate():
        assert service.migrate_local_cache(["2330"], signal_date=CUTOFF, sector="24") == 1
    assert probe.counters["fundamental_migration_header_skips"] == 0
    assert probe.counters["fundamental_snapshot_decodes"] == 1


def test_header_query_is_bounded_and_never_selects_payload_or_facts(tmp_path, monkeypatch):
    repo, _, _ = seeded(tmp_path)
    repo.queries._BULK_SYMBOL_CHUNK_SIZE = 2
    statements, connections = [], []
    original = repo._connect
    def connect():
        connection = original()
        connections.append(connection)
        connection.set_trace_callback(statements.append)
        return connection
    monkeypatch.setattr(repo, "_connect", connect)
    with repo.unit_of_work() as uow:
        result = repo.load_cached_bundle_headers(dict.fromkeys(["2330", "a", "b", "c", "d"], "24"),
            as_of=CUTOFF, derivation_version=FUNDAMENTAL_DERIVATION_VERSION, uow=uow)
    assert len(connections) == 1
    selects = [s.lower() for s in statements if s.lstrip().lower().startswith("select")]
    assert len(selects) == 3
    assert all(not any(term in s for term in ("snapshot_json", "annual_json", "facts", "json_extract", "julianday")) for s in selects)
    assert set(result) == {"2330"}
    assert result["2330"].filing_schema_version == V6_FILING_SCHEMA_VERSION
    for params in [dict(as_of="2026-01-01T00:00:00Z", derivation_version=FUNDAMENTAL_DERIVATION_VERSION),
                   dict(as_of=CUTOFF, derivation_version="other-version")]:
        assert repo.load_cached_bundle_headers({"2330": "24"}, **params) == {}
    assert repo.load_cached_bundle_headers({"2330": "25"}, as_of=CUTOFF,
                                           derivation_version=FUNDAMENTAL_DERIVATION_VERSION) == {}


def test_header_does_not_replace_full_content_validation(tmp_path, monkeypatch):
    repo, _, service = seeded(tmp_path, complete=True)
    real_load = repo.load_fundamental_bundle
    def invalid_content(*args, **kwargs):
        bundle = real_load(*args, **kwargs)
        return replace(bundle, snapshot=dict(bundle.snapshot, les_v6_filing_schema_version="old"))
    monkeypatch.setattr(repo, "load_fundamental_bundle", invalid_content)
    assert service.load_cached_bundles(["2330"], signal_date=CUTOFF, sector="24") == {}


def test_sqlite_schema_upgrade_does_not_scan_or_rewrite_legacy_json(tmp_path):
    path = tmp_path / "legacy-synthetic.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE fundamental_snapshots(symbol TEXT,filing_set_sha256 TEXT,
                     sector TEXT,annual_json TEXT,snapshot_json TEXT,created_at_epoch INTEGER,
                     PRIMARY KEY(symbol,filing_set_sha256,sector))""")
        conn.execute("INSERT INTO fundamental_snapshots VALUES ('2330','test','24','[]','NOT_PARSED',123)")
    statements = []
    def connect():
        conn = sqlite3.connect(path)
        conn.set_trace_callback(statements.append)
        return conn
    MopsSchemaMigrator(connect).ensure()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT snapshot_json,created_at_epoch,has_mops_filings FROM fundamental_snapshots").fetchone() == ("NOT_PARSED", 123, None)
    assert not any("update fundamental_snapshots" in s.lower() or "json_extract" in s.lower() for s in statements)


def test_new_writer_updates_format_marker_atomically(tmp_path):
    repo, _, service = seeded(tmp_path)
    header = repo.load_cached_bundle_headers({"2330": "24"}, as_of=CUTOFF,
        derivation_version=FUNDAMENTAL_DERIVATION_VERSION)["2330"]
    bundle = cached(service)
    changed = dict(bundle.snapshot)
    changed.pop("les_v6_mops_filings")
    repo.save_fundamental_bundle("2330", header.snapshot_cache_key, "24", replace(bundle, snapshot=changed))
    updated = repo.load_cached_bundle_headers({"2330": "24"}, as_of=CUTOFF,
        derivation_version=FUNDAMENTAL_DERIVATION_VERSION)["2330"]
    assert updated.has_mops_filings is False
    assert service.migrate_local_cache(["2330"], signal_date=CUTOFF, sector="24") == 1
