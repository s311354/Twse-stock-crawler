"""Synthetic working-set checks through the public native work boundary."""
from types import SimpleNamespace

import pytest

from application.daily_performance import DailyPerformance
from infrastructure import daily_work


@pytest.fixture(autouse=True)
def synthetic_version_guard(monkeypatch):
    monkeypatch.setattr(daily_work, "ResearchVersionGuard", lambda: SimpleNamespace(
        versions=SimpleNamespace(), check=lambda: None))
    monkeypatch.setattr(daily_work, "asdict", lambda _: {"code": "SYNTHETIC"})


def test_disk_capacity_is_independent_of_memory_and_default_is_unchanged(tmp_path):
    with daily_work.open_daily_work(parse_cache_path=tmp_path / "default.sqlite3",
                                    snapshot_spool_parent=tmp_path) as work:
        assert work.parse_memo.max_bytes == 512 * 1024 * 1024
        assert work.max_bytes == 64 * 1024 * 1024
    with daily_work.native_work_scope(tmp_path / "large.sqlite3",
                                      snapshot_spool_parent=tmp_path,
                                      parse_cache_max_bytes=4 * 1024 ** 3) as work:
        assert work.parse_memo.max_bytes == 4 * 1024 ** 3
        assert work.max_bytes == 64 * 1024 * 1024
        # A nested adapter must borrow the existing owner, not replace its
        # cache or create a second store with a competing capacity.
        unused = tmp_path / "unused.sqlite3"
        with daily_work.native_work_scope(unused, parse_cache_max_bytes=1) as borrowed:
            assert borrowed is work
            assert borrowed.parse_memo.max_bytes == 4 * 1024 ** 3
        assert not unused.exists()


@pytest.mark.parametrize("disk_bytes,expected_builds,expected_second_hits", [
    (2 * 1024, 16, 0),
    (32 * 1024, 8, 8),
])
def test_two_sequential_days_reuse_entire_source_working_set_when_it_fits(
        tmp_path, disk_bytes, expected_builds, expected_second_hits):
    """A scaled ELEC-like scan exposes FIFO thrashing without large fixtures.

    Each day reads the same sources in the same order. Raw and companion rows
    together exceed the smaller disk budget; the larger budget holds both.
    The small work memory budget stays fixed across both cases.
    """
    path = tmp_path / "source-preprocessing.sqlite3"
    calls = {"raw": 0, "companion": 0}
    results, probes = [], []
    for day in range(2):
        probe = DailyPerformance()
        rows = []
        with probe.activate(), daily_work.open_daily_work(
                parse_cache_path=path, parse_cache_max_bytes=disk_bytes,
                max_bytes=512, snapshot_spool_parent=tmp_path,
                parse_cache_compression=False) as work:
            for source in range(8):
                def raw():
                    calls["raw"] += 1
                    return [{"facts": "raw-" + str(source) + "x" * 1000}]

                def companion():
                    calls["companion"] += 1
                    return {"proof": "companion-" + str(source) + "y" * 1000}

                rows.append((work.raw_parse(str(source), raw, parser_version="synthetic-v1"),
                             work.raw_companion(str(source), companion,
                                                derivation_version="synthetic-v1")))
            assert work.parse_memo.bytes <= disk_bytes
            assert work.max_bytes == 512
        results.append(rows)
        probes.append(probe)
    assert results[0] == results[1]
    assert calls == {"raw": expected_builds, "companion": expected_builds}
    assert probes[1].counters["raw_parse_disk_hits"] == expected_second_hits
    assert probes[1].counters["financial_companion_disk_hits"] == expected_second_hits
    # Public consumers can mutate one result without poisoning later reads.
    results[0][0][0][0]["facts"] = "changed"
    assert results[1][0][0][0]["facts"].startswith("raw-0")


def test_larger_budget_preserves_semantic_invalidation_and_corrupt_row_repair(tmp_path):
    calls = []
    path = tmp_path / "source-preprocessing.sqlite3"

    def parse():
        calls.append("raw")
        return [{"facts": len(calls)}]

    with daily_work.open_daily_work(parse_cache_path=path, parse_cache_max_bytes=32 * 1024,
                                    snapshot_spool_parent=tmp_path) as work:
        original = work.raw_parse("source-A", parse, parser_version="v1")
        assert work.raw_parse("source-A", parse, parser_version="v1") == original
        assert work.raw_parse("source-B", parse, parser_version="v1") != original
        assert work.raw_parse("source-A", parse, parser_version="v2") != original
        disk = work.parse_memo
        with disk.connection:
            disk.connection.execute("UPDATE research_parses SET facts_json='[{}]' "
                                    "WHERE content_sha256='source-A' AND version LIKE '%v1'")
        assert work.raw_parse("source-A", parse, parser_version="v1") != original
        assert disk.stats["raw_parse_disk_miss_corrupt"] == 1
    assert calls == ["raw"] * 4
