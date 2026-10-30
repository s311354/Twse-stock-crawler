"""SYNTHETIC ONLY: immutable source preprocessing, not investment evidence."""
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256

import pytest

from application.daily_performance import DailyPerformance
from application.daily_work import DailyWorkContext
from domain.les_v7.financial_sources import audit_proof, financial_supplement
from infrastructure.fundamentals import financial_companions as companions
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.storage.research_cache import SQLiteRawParseCache
from tests.fundamentals.test_filing_refresh_planner import _candidate
from tests.fundamentals.test_taxonomy_cash_debt import cash_document
from tests.fundamentals.test_v7_financial_sources import original

CUTOFF = "2026-10-16T06:00:00Z"


@contextmanager
def work_scope(path=None, *, enabled=True, version="SYNTHETIC_WORK_A", max_bytes=64 * 1024 * 1024):
    if path is None:
        work = DailyWorkContext(version, enabled=enabled, max_bytes=max_bytes)
        try:
            with work.activate():
                yield work
        finally:
            work.close()
    else:
        with SQLiteRawParseCache(path) as disk:
            work = DailyWorkContext(version, enabled=enabled, parse_memo=disk, max_bytes=max_bytes)
            try:
                with work.activate():
                    yield work
            finally:
                work.close()


def reference(source, *, symbol="2330", market="TWSE"):
    return dict(financial_audit=audit_proof(source, symbol=symbol, market=market, original_verified=True),
                financial_supplement=financial_supplement(source, symbol=symbol, market=market, original_verified=True))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("market", ["TWSE", "TPEX"])
def test_all_fields_and_python_container_types_match_uncached_reference(enabled, market):
    source = original()
    expected = reference(source, market=market)
    probe = DailyPerformance()
    with probe.activate(), work_scope(enabled=enabled) as work:
        first = companions.verified_source_companions(source, symbol="2330", market=market)
        second = companions.verified_source_companions(source, symbol="2330", market=market)
        assert first == second == expected
        assert first is not second
        assert isinstance(first["financial_audit"]["contexts"][0]["dimensions"], tuple)
        first["financial_audit"]["audit"]["audit_opinion"] = "SYNTHETIC POISON"
        first["financial_supplement"]["fields"]["parent_equity"] = 999999
        assert companions.verified_source_companions(source, symbol="2330", market=market) == expected
        if enabled:
            packets = [value for (kind, _, _), (value, _) in work._values.items()
                       if kind == "financial_companion"]
            with pytest.raises(TypeError, match="READ_ONLY"):
                packets[0]["value"]["financial_audit"]["status"] = "PASS"
    assert probe.counters["financial_companion_builds"] == (1 if enabled else 3)
    assert probe.counters["financial_companion_cache_hits"] == (2 if enabled else 0)


def test_same_original_reuses_companion_across_job_versions_but_not_derivation_change(tmp_path, monkeypatch):
    source, path = original(), tmp_path / "SYNTHETIC-parses.sqlite3"
    expected = reference(source)
    first = DailyPerformance()
    with first.activate(), work_scope(path):
        assert companions.verified_source_companions(source, symbol="2330") == expected
    warm = DailyPerformance()
    with warm.activate(), work_scope(path, version="SYNTHETIC_UNRELATED_RENDERER_CHANGE"):
        assert companions.verified_source_companions(source, symbol="2330") == expected
    assert first.counters["financial_companion_builds"] == 1
    assert warm.counters["financial_companion_builds"] == 0
    assert warm.counters["financial_companion_disk_hits"] == 1
    monkeypatch.setattr(companions, "companion_semantics", lambda: "SYNTHETIC_NEW_MAPPING")
    changed = DailyPerformance()
    with changed.activate(), work_scope(path):
        assert companions.verified_source_companions(source, symbol="2330") == expected
    assert changed.counters["financial_companion_builds"] == 1


@pytest.mark.parametrize("metadata", [
    dict(available_at="2026-09-01T00:00:00Z"), dict(revision_number=99),
    dict(content_sha256="f" * 64), dict(source_url="https://example.test/SYNTHETIC_REPLACEMENT"),
    dict(statement_scope="individual"), dict(period_end="2026-09-30"),
    dict(taxonomy_schema_refs=("https://example.test/SYNTHETIC_UNKNOWN.xsd",)),
])
def test_source_metadata_is_not_reused_merely_because_body_hash_matches(tmp_path, metadata):
    source, probe = original(), DailyPerformance()
    with probe.activate(), work_scope(tmp_path / "SYNTHETIC-parses.sqlite3"):
        companions.verified_source_companions(source, symbol="2330")
        changed = replace(source, **metadata)
        actual = companions.verified_source_companions(changed, symbol="2330")
        assert actual == reference(changed)
    assert probe.counters["financial_companion_builds"] == 2


def repository(tmp_path):
    root = tmp_path / "SYNTHETIC-mops"
    repo = MopsRepository(root)
    body = cash_document().replace(b"</body>", b'<ix:nonNumeric name="tifrs-ar:UnqualifiedOpinionAbstract" '
        b'contextRef="YTD">Y</ix:nonNumeric></body>')
    filing = repo.save_revision(_candidate(2026, 2), body, parse_ixbrl(body),
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:01:00Z")
    return repo, body, filing


@pytest.mark.parametrize("damage", ["changed", "missing"])
def test_companion_cache_cannot_hide_corrupt_or_missing_original(tmp_path, damage):
    repo, _, filing = repository(tmp_path)
    probe = DailyPerformance()
    with probe.activate(), work_scope(tmp_path / "SYNTHETIC-parses.sqlite3"):
        first = repo.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2),))
        assert first["financial_audits"]
        path = repo.raw_path(filing)
        if damage == "changed":
            path.write_bytes(b"SYNTHETIC CORRUPT ORIGINAL")
        else:
            path.unlink()
        assert repo.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2),)) == dict(
            financial_audits={}, financial_supplements={})
    assert probe.counters["financial_companion_builds"] == 1


def test_each_cutoff_reselects_versions_and_does_not_mutate_original_cache(tmp_path):
    repo, body, old = repository(tmp_path)
    future_body = body.replace(b"</body>", b"<!-- SYNTHETIC FUTURE REVISION --></body>")
    future = repo.save_revision(_candidate(2026, 2), future_body, parse_ixbrl(future_body),
        available_at="2026-10-17T00:00:00Z", fetched_at="2026-10-17T00:01:00Z")
    original_db = sha256(repo.database_path.read_bytes()).hexdigest()
    with work_scope(enabled=False):
        expected = repo.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2),), source_repair=True)
    with work_scope(tmp_path / "SYNTHETIC-parses.sqlite3"):
        assert repo.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2),), source_repair=True) == expected
        later = repo.load_v7_financial_sources("2330", "2026-10-18T00:00:00Z", periods=((2026, 2),), source_repair=True)
        assert set(later["financial_audits"]) == {old.content_sha256, future.content_sha256}
        assert repo.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2),), source_repair=True) == expected
    assert original_db == sha256(repo.database_path.read_bytes()).hexdigest()


def test_warm_companion_never_skips_cutoff_bound_signed_source_selection(tmp_path, monkeypatch):
    from infrastructure.fundamentals.signed_audit import SignedAuditStore
    repo, _, _ = repository(tmp_path)
    calls = []
    def select(store, row, *, cutoff):
        calls.append((row["content_sha256"], cutoff))
        return None
    monkeypatch.setattr(SignedAuditStore, "select", select)
    later = "2026-10-16T07:00:00Z"
    probe = DailyPerformance()
    with probe.activate(), work_scope(tmp_path / "SYNTHETIC-parses.sqlite3"):
        first = repo.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2),), source_repair=True)
        second = repo.load_v7_financial_sources("2330", later, periods=((2026, 2),), source_repair=True)
    assert first == second
    assert [cutoff for _, cutoff in calls] == [CUTOFF, later]
    assert probe.counters["financial_companion_builds"] == 1
    assert probe.counters["financial_companion_cache_hits"] == 1


def test_immutable_work_cache_has_a_hard_capacity_and_releases_on_close():
    source, probe = original(), DailyPerformance()
    with probe.activate(), work_scope(max_bytes=512) as work:
        for _ in range(2):
            assert companions.verified_source_companions(source, symbol="2330") == reference(source)
        assert work.stats["immutable_bytes"] <= 256
    assert work.stats["immutable_bytes"] == work.stats["immutable_entries"] == 0
    assert probe.counters["financial_companion_builds"] == 2
