"""Synthetic, isolated source-cache identities; no network or official state."""
import hashlib
import json

import pytest

from application.daily_performance import DailyPerformance
from application.daily_work import DailyWorkContext
from infrastructure.fundamentals import parse_identity
from infrastructure.storage.research_cache import SQLiteRawParseCache


def run_work(path, work_version, parser_version, compute, *, enabled=True):
    with SQLiteRawParseCache(path) as disk:
        work = DailyWorkContext(work_version, enabled=enabled, parse_memo=disk)
        try:
            with work.activate():
                return work.raw_parse("SYNTHETIC-original-hash", compute, parser_version=parser_version)
        finally:
            work.close()


def test_raw_disk_reuse_is_independent_of_renderer_model_and_full_job_version(tmp_path):
    calls = []
    def parse():
        calls.append("parse")
        return [{"facts": [{"concept": "SYNTHETIC", "value": 1}]}]
    path = tmp_path / "synthetic.sqlite3"
    version = parse_identity.semantic_raw_parse_version("mapping-v1")
    first = run_work(path, "whole-program-before-renderer-edit", version, parse)
    first[0]["facts"].clear()
    second = run_work(path, "whole-program-after-renderer-edit", version, parse)
    assert second == [{"facts": [{"concept": "SYNTHETIC", "value": 1}]}]
    assert calls == ["parse"]
    assert run_work(path, "other-model", version, parse, enabled=False) == second
    assert calls == ["parse", "parse"]


@pytest.mark.parametrize("changed", ["parser.py", "mapping.py", "schema.py"])
def test_actual_semantic_source_change_invalidates_without_handwritten_version_bump(tmp_path, monkeypatch, changed):
    for name in ("parser.py", "mapping.py", "schema.py", "renderer.py"):
        (tmp_path / name).write_text("VERSION = 1\n")
    monkeypatch.setattr(parse_identity, "_ROOT", tmp_path)
    monkeypatch.setattr(parse_identity, "_RAW_SOURCES", tuple((name, ()) for name in
        ("parser.py", "mapping.py", "schema.py")))
    initial = parse_identity.semantic_raw_parse_version("same-explicit-label")
    (tmp_path / "renderer.py").write_text("VERSION = 200\n")
    assert parse_identity.semantic_raw_parse_version("same-explicit-label") == initial
    (tmp_path / changed).write_text("VERSION = 200\n")
    after = parse_identity.semantic_raw_parse_version("same-explicit-label")
    assert after != initial
    calls = []
    compute = lambda: calls.append(1) or [{"SYNTHETIC": 1}]
    path = tmp_path / "synthetic.sqlite3"
    run_work(path, "same-job", initial, compute)
    run_work(path, "same-job", after, compute)
    assert calls == [1, 1]


def test_scoped_adapter_fingerprint_ignores_unrelated_function_and_checkout_path(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    (first / "adapter.py").write_text("def selected(x):\n    return x + 1\n\ndef unrelated():\n    return 2\n")
    (second / "adapter.py").write_text("def unrelated():\n    return 999\n\n\ndef selected(x):\n    return x + 1\n")
    sources = (("adapter.py", ("selected",)),)
    assert parse_identity.source_fingerprint(sources, root=first) == parse_identity.source_fingerprint(sources, root=second)
    (second / "adapter.py").write_text("def selected(x):\n    return x + 2\n")
    assert parse_identity.source_fingerprint(sources, root=first) != parse_identity.source_fingerprint(sources, root=second)


def test_actual_parser_label_and_runtime_remain_part_of_identity(monkeypatch):
    first = parse_identity.semantic_raw_parse_version("mapping-v1")
    assert first != parse_identity.semantic_raw_parse_version("mapping-v2")
    monkeypatch.setattr(parse_identity.sys, "version_info", (3, 15, 99, "final", 0))
    assert first != parse_identity.semantic_raw_parse_version("mapping-v1")


def test_cache_reuse_does_not_disable_complete_job_version_guard(tmp_path):
    path = tmp_path / "synthetic.sqlite3"
    run_work(path, "old-job", "same-parser", lambda: [])
    checks = []
    def check():
        checks.append(1)
        if len(checks) == 2:
            raise ValueError("REPLAY_VERSION_CHANGED_DURING_JOB")
    with SQLiteRawParseCache(path) as disk:
        work = DailyWorkContext("new-job", parse_memo=disk, check_version=check)
        with pytest.raises(ValueError, match="VERSION_CHANGED"), work.activate():
            assert work.raw_parse("SYNTHETIC-original-hash", lambda: pytest.fail("cache missed"),
                                  parser_version="same-parser") == []
        assert checks == [1, 1]
        work.close()


def test_oversized_parse_does_not_flush_small_working_set_and_reports_reason(tmp_path):
    probe = DailyPerformance()
    with probe.activate(), SQLiteRawParseCache(tmp_path / "synthetic.sqlite3", max_bytes=50) as disk:
        assert disk.memo("ixbrl", ("parser", "small"), lambda: [{"v": 1}]) == [{"v": 1}]
        large = [{"v": "x" * 1000}]
        assert disk.memo("ixbrl", ("parser", "large"), lambda: large) == large
        assert disk.memo("ixbrl", ("parser", "small"), lambda: pytest.fail("small row evicted")) == [{"v": 1}]
        assert disk.bytes <= disk.max_bytes
        assert disk.stats["raw_parse_disk_oversized_bypass"] == 1
        assert disk.stats["raw_parse_disk_miss_absent"] == 2
        assert disk.stats["raw_parse_disk_hits"] == 1
        assert disk.stats["disk_bytes"] > 0
    assert probe.counters["raw_parse_disk_evictions"] == 0
    assert probe.counters["raw_parse_disk_oversized_bypass"] == 1


def test_corrupt_row_is_recomputed_and_corruption_is_not_a_pass(tmp_path):
    with SQLiteRawParseCache(tmp_path / "synthetic.sqlite3") as disk:
        disk.memo("ixbrl", ("parser", "sha"), lambda: [{"v": 1}])
        with disk.connection:
            disk.connection.execute("UPDATE research_parses SET facts_json='[{}]' WHERE content_sha256='sha'")
        assert disk.memo("ixbrl", ("parser", "sha"), lambda: [{"v": 2}]) == [{"v": 2}]
        assert disk.stats["raw_parse_disk_miss_corrupt"] == 1
        row = disk.connection.execute("SELECT facts_json,facts_sha256 FROM research_parses").fetchone()
        assert hashlib.sha256(row[0].encode()).hexdigest() == row[1]
        assert json.loads(row[0]) == [{"v": 2}]


def test_companions_have_separate_namespace_detached_results_and_semantic_key(tmp_path):
    with SQLiteRawParseCache(tmp_path / "synthetic.sqlite3") as disk:
        assert disk.memo("ixbrl", ("same", "sha"), lambda: [{"raw": True}]) == [{"raw": True}]
        assert disk.memo("financial_companion", ("same", "sha"), lambda: [{"companion": True}]) == [{"companion": True}]
        work = DailyWorkContext("program-A", parse_memo=disk)
        first = work.raw_companion("metadata-sha", lambda: {"source": {"version": 1}}, derivation_version="v1")
        first["source"]["version"] = 999
        assert work.raw_companion("metadata-sha", lambda: pytest.fail("recomputed"),
                                  derivation_version="v1") == {"source": {"version": 1}}
        assert work.raw_companion("metadata-sha", lambda: {"source": {"version": 2}},
                                  derivation_version="v2") == {"source": {"version": 2}}
        work.close()
        with pytest.raises(ValueError, match="CLOSED"):
            work.raw_companion("metadata-sha", lambda: {}, derivation_version="v1")


def test_zero_capacity_and_failed_compute_cannot_publish_parse(tmp_path):
    with SQLiteRawParseCache(tmp_path / "synthetic.sqlite3", max_bytes=0) as disk:
        assert disk.memo("ixbrl", ("v1", "sha"), lambda: [{"v": 1}]) == [{"v": 1}]
        assert disk.bytes == 0
        assert disk.connection.execute("SELECT COUNT(*) FROM research_parses").fetchone()[0] == 0
        with pytest.raises(RuntimeError, match="interrupted"):
            disk.memo("ixbrl", ("v1", "sha"), lambda: (_ for _ in ()).throw(RuntimeError("interrupted")))
        assert disk.connection.execute("SELECT COUNT(*) FROM research_parses").fetchone()[0] == 0
