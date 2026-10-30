"""Synthetic raw-parse ownership and slotted fact compatibility; no network."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
import hashlib
import os
import pickle
import weakref

import pytest

from application.daily_performance import DailyPerformance
from application.daily_work import DailyWorkContext
from domain.fundamental_derivation import FiledFact, FiledFinancials
from infrastructure.fundamentals import mops_queries
from infrastructure.fundamentals.mops_repository import MopsRepository
from tests.fundamentals.test_ixbrl_units import document, seed_legacy
from tests.fundamentals.test_mops_refresh_control import SIGNAL


def source(tmp_path, name="SYNTHETIC.xml"):
    content = document()
    path = tmp_path / name
    path.write_bytes(content)
    stat = path.stat()
    return path, hashlib.sha256(content).hexdigest(), (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


@contextmanager
def work_scope(*, enabled=True, max_bytes=64 * 1024 * 1024):
    work = DailyWorkContext("synthetic-lifetime", enabled=enabled, max_bytes=max_bytes)
    try:
        with work.activate():
            yield work
    finally:
        work.close()


@pytest.fixture(autouse=True)
def legacy_cache_isolation():
    mops_queries._raw_normalized_facts.cache_clear()
    yield
    mops_queries._raw_normalized_facts.cache_clear()


@pytest.mark.parametrize("enabled", [True, False])
def test_native_parse_never_uses_or_populates_legacy_lru(tmp_path, enabled):
    path, sha, signature = source(tmp_path)
    read = mops_queries._raw_normalized_facts
    legacy = read(path, sha, signature, "parser-v1")
    before = read.cache_info()
    probe = DailyPerformance()
    with probe.activate(), work_scope(enabled=enabled):
        first = read(path, sha, signature, "parser-v1")
        second = read(path, sha, signature, "parser-v1")
    assert first == second == legacy
    assert read.cache_info() == before
    assert probe.counters["raw_original_reads"] == (1 if enabled else 2)
    assert probe.counters["raw_unit_parse_calls"] == (1 if enabled else 2)
    assert probe.counters["raw_parse_cache_hits"] == (1 if enabled else 0)


@pytest.mark.parametrize("change", ["path", "parser", "stat", "hash"])
def test_native_parse_keys_include_source_stat_parser_and_hash(tmp_path, change):
    path, sha, signature = source(tmp_path)
    read = mops_queries._raw_normalized_facts
    probe = DailyPerformance()
    with probe.activate(), work_scope():
        first = read(path, sha, signature, "parser-v1")
        parser = "parser-v1"
        if change == "path":
            path, _, _ = source(tmp_path, "SYNTHETIC-other.xml")
        elif change == "parser":
            parser = "parser-v2"
        elif change == "stat":
            # Pass the old caller signature deliberately. The current source
            # stat is checked before a memory hit; replacements cannot hide.
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        else:
            sha = "0" * 64
        second = read(path, sha, signature, parser)
    assert probe.counters["raw_original_reads"] == 2
    if change == "hash":
        assert second is None
        assert probe.counters["raw_unit_hash_rejections"] == 1
    else:
        assert second == first
        assert probe.counters["raw_unit_parse_calls"] == 2
    assert read.cache_info().currsize == 0


@pytest.mark.parametrize("damage", ["changed", "missing"])
def test_native_cache_cannot_hide_missing_or_corrupt_original(tmp_path, damage):
    repository = MopsRepository(tmp_path / "SYNTHETIC-mops")
    stored, _ = seed_legacy(repository)
    with work_scope():
        first = repository.load_filed_financials("3016", SIGNAL)[0]
        assert all(fact.numeric_value is not None for fact in first.facts)
        path = repository.raw_path(stored)
        if damage == "changed":
            path.write_bytes(document(currency="USD"))
        else:
            path.unlink()
        second = repository.load_filed_financials("3016", SIGNAL)[0]
        assert all(fact.numeric_value is None and fact.unit is None for fact in second.facts)
        assert second.content_sha256 == first.content_sha256
        assert second.available_at == first.available_at


def test_work_parse_eviction_counts_full_graph_and_has_no_global_owner(tmp_path):
    path, sha, signature = source(tmp_path)
    read = mops_queries._raw_normalized_facts
    first = read(path, sha, signature, "parser-v1")
    capacity = mops_queries._raw_facts_size(first) + 1000
    read.cache_clear()
    probe = DailyPerformance()
    with probe.activate(), work_scope(max_bytes=capacity * 2) as work:
        read(path, sha, signature, "parser-v1")
        other, _, other_signature = source(tmp_path, "SYNTHETIC-second.xml")
        read(other, sha, other_signature, "parser-v1")
        read(path, sha, signature, "parser-v1")
        assert work.stats["immutable_entries"] == 1
        assert work.stats["immutable_bytes"] <= capacity
    assert probe.counters["daily_work_evictions"] == 2
    assert probe.counters["raw_unit_parse_calls"] == 3
    assert work.stats["immutable_entries"] == work.stats["immutable_bytes"] == 0
    assert read.cache_info().currsize == 0


def test_close_releases_native_facts_and_prevents_repopulation(tmp_path, monkeypatch):
    class ObservedFact(FiledFact):
        __slots__ = ("__weakref__",)
    references = []
    def parse(*args):
        fact = ObservedFact("SYNTHETIC", 1., None, None, None, None)
        references.append(weakref.ref(fact))
        return (fact,), ()
    monkeypatch.setattr(mops_queries, "_read_raw_normalized_facts", parse)
    path, sha, signature = source(tmp_path)
    with work_scope() as work:
        mops_queries._raw_normalized_facts(path, sha, signature, "parser-v1")
        assert references[0]() is not None
        assert mops_queries._raw_normalized_facts.cache_info().currsize == 0
    gc.collect()
    assert references[0]() is None
    for call in (
        lambda: work.memo_immutable("raw_parse", "key", parse, lambda _: 1),
        lambda: work.memo_json("raw_parse", "key", lambda: []),
        lambda: work.raw_parse("sha", lambda: [], parser_version="parser-v1"),
    ):
        with pytest.raises(ValueError, match="DAILY_WORK_CONTEXT_CLOSED"):
            call()


@pytest.mark.parametrize("options", [{"enabled": False}, {"max_bytes": 0}])
def test_disabled_or_zero_capacity_cache_does_not_walk_values_for_sizing(options):
    with work_scope(**options) as work:
        assert work.memo_immutable("raw_parse", "source-key", lambda: (1,),
                                  lambda _: pytest.fail("unretained graph was traversed")) == (1,)
        assert work.stats["immutable_entries"] == 0


def test_native_raw_cache_keeps_cutoff_selection_outside_parse_memo(tmp_path):
    repository = MopsRepository(tmp_path / "SYNTHETIC-mops")
    stored, _ = seed_legacy(repository)
    with work_scope():
        first = repository.load_filed_financials("3016", SIGNAL)
        assert first[0].available_at == stored.available_at
        assert repository.load_filed_financials("3016", "2026-08-09T23:59:59Z") == []
        assert repository.load_filed_financials("3016", SIGNAL) == first


@pytest.mark.parametrize("kind", ["fact", "financials"])
def test_slotted_financial_records_preserve_dataclass_and_pickle_contracts(kind):
    fact = FiledFact("SYNTHETIC", 1., None, "2026-01-01", "2026-03-31", None,
                     dimensions=(("dimension", "member"),), unit="iso4217:TWD", context_id="Q1")
    value = fact if kind == "fact" else FiledFinancials(2026, 1, "2026-05-01T10:00:00Z", 1, (fact,))
    assert not hasattr(value, "__dict__")
    assert pickle.loads(pickle.dumps(value)) == value
    assert asdict(pickle.loads(pickle.dumps(value))) == asdict(value)
    changed = replace(value, **({"numeric_value": 2.} if kind == "fact" else {"revision_number": 2}))
    assert changed != value
    assert pickle.loads(pickle.dumps(changed)) == changed
    # Restore an actual pre-slots dictionary-state pickle, rather than testing
    # only the new serializer. The factory creates the same class without init.
    class LegacyPayload:
        def __reduce__(self):
            return _uninitialized, (type(value),), value.__getstate__()
    restored = pickle.loads(pickle.dumps(LegacyPayload()))
    assert restored == value and asdict(restored) == asdict(value)


def test_older_pickle_missing_optional_fields_keeps_conservative_defaults():
    fact = FiledFact.__new__(FiledFact)
    fact.__setstate__(dict(concept="SYNTHETIC", numeric_value=1., text_value=None,
                           start_date=None, end_date=None, instant=None))
    assert fact == FiledFact("SYNTHETIC", 1., None, None, None, None)
    financials = FiledFinancials.__new__(FiledFinancials)
    financials.__setstate__(dict(fiscal_year=2026, quarter=1, available_at="2026-05-01T10:00:00Z",
                                  revision_number=1, facts=(fact,)))
    assert financials.availability_source == "LEGACY_UNVERIFIED"
    assert financials.taxonomy_schema_refs == ()
    with pytest.raises(ValueError, match="MISSING_FINANCIAL_PICKLE_FIELD"):
        FiledFact.__new__(FiledFact).__setstate__({"concept": "SYNTHETIC"})
    with pytest.raises(ValueError, match="INVALID_FINANCIAL_PICKLE_FIELDS"):
        FiledFact.__new__(FiledFact).__setstate__(dict(fact.__getstate__(), unrecognized=True))


def _uninitialized(kind):
    return kind.__new__(kind)
