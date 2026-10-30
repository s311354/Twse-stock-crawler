"""Temporary native retention is lossless, bounded and fail-closed; synthetic."""
from contextlib import contextmanager
from dataclasses import replace
import errno
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance
from application.daily_work import DailyWorkContext
from application.snapshot_retention import snapshot_materialize, snapshot_metadata
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION, FundamentalBundle
from infrastructure.legacy_crawler_facade import LegacyCrawlerFacade
from infrastructure.snapshot_retention import SnapshotRetentionMap, SnapshotRetentionStorageError, WorkSnapshotRetention
from tests.test_daily_entry_components import market_with_sidecar


@contextmanager
def retained_work(**capacity):
    work = DailyWorkContext("synthetic-retention")
    owner = WorkSnapshotRetention(work.version, **capacity)
    work.snapshot_retention = owner
    try:
        with work.activate():
            yield work, owner
    finally:
        work.close()
        owner.close()


def payload():
    return dict(sector="24", available_at="2026-09-11T07:00:00Z",
                twse_financial_observed_at="2026-09-10T05:00:00Z",
                base_quality_feature_sources={"cash": ["SYNTHETIC_ONLY"]},
                tuple_values=(1, None), list_values=[True, "臺灣"],
                number=np.float64(3.25), count=np.int64(7), flag=np.bool_(True),
                timestamp=pd.Timestamp("2026-08-01T13:30:00+08:00"),
                missing=pd.NA, array=np.array([1., 2.], dtype=np.float32))


@pytest.mark.parametrize("memory_bytes", [0, 8 * 1024 * 1024])
def test_snapshot_roundtrip_preserves_types_metadata_bool_binding_and_source(memory_bytes):
    probe = DailyPerformance()
    source = payload()
    with probe.activate(), retained_work(memory_bytes=memory_bytes) as (_, owner):
        mapping = owner.snapshots()
        mapping.retain("2330", source, cutoff="2026-09-11T15:00:00+08:00", market="TWSE")
        metadata = snapshot_metadata(mapping, "2330")
        assert bool(metadata) and metadata["sector"] == "24"
        assert "number" not in metadata
        assert "array" not in metadata and "tuple_values" not in metadata
        assert "2330" in mapping and list(mapping) == ["2330"]
        assert probe.counters["snapshot_materializations"] == 0
        metadata["base_quality_feature_sources"]["cash"].clear()
        assert snapshot_metadata(mapping, "2330")["base_quality_feature_sources"]["cash"] == ["SYNTHETIC_ONLY"]
        source["list_values"].clear()
        restored = snapshot_materialize(mapping, "2330", cutoff="2026-09-11T07:00:00Z", market="TWSE")
        assert restored["list_values"] == [True, "臺灣"]
        assert type(restored["tuple_values"]) is tuple and type(restored["list_values"]) is list
        assert type(restored["number"]) is np.float64 and type(restored["flag"]) is np.bool_
        assert type(restored["count"]) is np.int64 and type(restored["timestamp"]) is pd.Timestamp
        assert restored["missing"] is pd.NA
        assert restored["array"].dtype == np.dtype("float32")
        np.testing.assert_array_equal(restored["array"], np.array([1., 2.], dtype=np.float32))
        restored["list_values"].clear()
        assert mapping["2330"]["list_values"] == [True, "臺灣"]
        with pytest.raises(ValueError, match="BINDING_MISMATCH"):
            snapshot_materialize(mapping, "2330", cutoff="2026-09-12T07:00:00Z", market="TWSE")
        with pytest.raises(ValueError, match="BINDING_MISMATCH"):
            snapshot_materialize(mapping, "2330", cutoff="2026-09-11T07:00:00Z", market="TPEX")
        heavy = owner.snapshots()
        heavy["EMPTY-METADATA"] = {"only_heavy": [1, 2, 3]}
        assert bool(snapshot_metadata(heavy, "EMPTY-METADATA"))
        assert not snapshot_metadata(heavy, "missing")


@pytest.mark.parametrize("damage", ["hash", "size", "missing", "symlink"])
def test_corrupt_spool_refuses_without_live_fallback_or_unpickle(tmp_path, damage, monkeypatch):
    with retained_work(memory_bytes=0) as (_, owner):
        mapping = owner.snapshots()
        mapping["2330"] = payload()
        descriptor = mapping._entries["2330"]
        path = Path(owner._directory.name) / (descriptor.sha256 + ".bin")
        if damage == "hash":
            encoded = path.read_bytes()
            path.write_bytes(bytes([encoded[0] ^ 1]) + encoded[1:])
        elif damage == "size":
            path.write_bytes(b"oversized" * descriptor.size)
        elif damage == "missing":
            path.unlink()
        else:
            outside = tmp_path / "NOT_A_SNAPSHOT.bin"
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
        monkeypatch.setattr("infrastructure.snapshot_retention.pickle.loads",
                            lambda _: pytest.fail("unverified bytes decoded"))
        with pytest.raises(ValueError, match="SNAPSHOT_RETENTION"):
            mapping["2330"]


def test_capacity_refcounts_and_cleanup_do_not_retain_unreachable_encodings():
    with retained_work(memory_bytes=0, max_spool_bytes=4096) as (_, owner):
        left, right = owner.snapshots(), owner.snapshots()
        source = {"sector": "24", "values": [1, 2]}
        left["2330"] = source
        right["2330"] = source
        directory = Path(owner._directory.name)
        assert owner.stats["objects"] == 1
        left.clear()
        assert owner.stats["objects"] == 1
        right.clear()
        assert owner.stats["objects"] == owner.stats["spooled_bytes"] == 0
        assert list(directory.iterdir()) == []
        with pytest.raises(ValueError, match="SNAPSHOT_TOO_LARGE"):
            previous_limit = owner.max_snapshot_bytes
            owner.max_snapshot_bytes = 8
            try:
                left["2330"] = source
            finally:
                owner.max_snapshot_bytes = previous_limit
        with pytest.raises(ValueError, match="CAPACITY_EXCEEDED"):
            left["2330"] = {"large": "SYNTHETIC" * 1000}
    assert not directory.exists()
    assert owner.stats["closed"] and owner.stats["objects"] == 0
    with pytest.raises(ValueError, match="CLOSED"):
        left["2330"]


def test_source_graph_is_detached_and_released_after_retention():
    class Observed(dict):
        pass
    with retained_work() as (_, owner):
        source = Observed(sector="24", values=[1])
        reference = weakref.ref(source)
        mapping = owner.snapshots()
        mapping["2330"] = dict(source)
        del source
        gc.collect()
        assert reference() is None
        assert mapping["2330"]["values"] == [1]


def test_descriptor_metadata_does_not_retain_unused_original_strings_or_arrays():
    with retained_work(memory_bytes=0) as (_, owner):
        mapping = owner.snapshots()
        source = payload()
        source["unused_original"] = "SYNTHETIC_RAW_DOCUMENT" * 100_000
        mapping["2330"] = source
        metadata = mapping._entries["2330"].metadata
        assert set(metadata) == {"sector", "available_at", "twse_financial_observed_at", "base_quality_feature_sources"}
        assert "unused_original" not in metadata and "array" not in metadata
        assert mapping["2330"]["unused_original"] == source["unused_original"]


def test_work_identity_and_closed_work_cannot_share_retention():
    with retained_work() as (work, owner):
        mapping = owner.snapshots()
        mapping["2330"] = {"sector": "24"}
        foreign = DailyWorkContext("different-code")
        try:
            with foreign.activate(), pytest.raises(ValueError, match="WORK_MISMATCH"):
                snapshot_metadata(mapping, "2330")
        finally:
            foreign.close()
        work.close()
        with pytest.raises(ValueError, match="DAILY_WORK_CONTEXT_CLOSED"):
            mapping["2331"] = {"sector": "24"}


@pytest.mark.parametrize("fail", [False, True])
def test_open_daily_work_closes_private_spool_on_success_and_failure(monkeypatch, fail):
    from infrastructure.daily_work import open_daily_work
    monkeypatch.setattr("infrastructure.daily_work.ResearchVersionGuard",
                        lambda: SimpleNamespace(versions=SimpleNamespace(), check=lambda: None))
    monkeypatch.setattr("infrastructure.daily_work.asdict", lambda _: {"code": "SYNTHETIC"})
    directory = None
    try:
        with open_daily_work() as work:
            owner = work.snapshot_retention
            owner.memory_bytes = 0
            mapping = owner.snapshots()
            mapping["2330"] = payload()
            directory = Path(owner._directory.name)
            if fail:
                raise RuntimeError("SYNTHETIC_INTERRUPTION")
    except RuntimeError:
        assert fail
    assert directory is not None and not directory.exists()
    assert owner.stats["closed"]


def test_lazy_snapshot_pipeline_matches_complete_native_decisions():
    market = market_with_sidecar("complete")
    expected = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]])
    probe = DailyPerformance()
    with probe.activate(), retained_work() as (_, owner):
        mapping = owner.snapshots()
        for symbol, snapshot in market.fundamental_snapshots.items():
            mapping.retain(symbol, snapshot, cutoff=market.signal_dates[symbol], market=market.market_source)
        actual = AnalysisDatasetBuilder().build_decisions(replace(market, fundamental_snapshots=mapping),
                                                         [[1, 1, 1, 1, .01]])
        assert actual == expected
        assert probe.counters["snapshot_materializations"] == 1


def test_native_facade_retains_but_legacy_setter_keeps_plain_dict():
    class Facade(LegacyCrawlerFacade):
        def __init__(self):
            self.annual_fundamentals_by_stock = {}
            self.fundamental_snapshots_by_stock = {}
            self.fundamental_status_by_stock = {}
            self.fundamental_status_detail_by_stock = {}
            self.low_entry_v4_signal_dates_by_stock = {}
            self.market_source = "TWSE"
    annual = pd.DataFrame({"year": [2025], "revenue": [100.]})
    legacy = Facade()
    legacy.set_low_entry_v4_fundamentals("2330", annual, payload(), signal_date="2026-09-11T07:00:00Z")
    assert type(legacy.fundamental_snapshots_by_stock) is dict
    crawler = Facade()
    with retained_work() as (_, owner):
        crawler.set_low_entry_v4_fundamentals("2330", annual, payload(), signal_date="2026-09-11T07:00:00Z")
        assert isinstance(crawler.fundamental_snapshots_by_stock, SnapshotRetentionMap)
        assert snapshot_metadata(crawler.fundamental_snapshots_by_stock, "2330")["sector"] == "24"
        assert crawler.fundamental_snapshots_by_stock.owner is owner
    with retained_work(), pytest.raises(ValueError, match="CLOSED"):
        crawler.set_low_entry_v4_fundamentals("2331", annual, payload(), signal_date="2026-09-11T07:00:00Z")


@pytest.mark.parametrize("complete", [False, True])
def test_incremental_preload_matches_legacy_bundle_without_reloading_facts(tmp_path, monkeypatch, complete):
    from tests.fundamentals.test_incremental_snapshot_refresh import seeded, CUTOFF, prohibit_facts
    repository, client, service = seeded(tmp_path, complete=complete)
    reference = service.load_cached_bundles(["2330"], signal_date=CUTOFF, sector="24")["2330"]
    monkeypatch.setattr(repository, "load_filed_financials", prohibit_facts)
    before = client.discovery_calls
    with retained_work() as (_, owner):
        bundles = owner.bundles()
        assert service.load_cached_bundles_into(["2330"], signal_date=CUTOFF, sector="24",
                                                sink=bundles.__setitem__) == ("2330",)
        result = bundles["2330"]
        assert result.snapshot == reference.snapshot and result.as_of == reference.as_of
        pd.testing.assert_frame_equal(result.annual_fundamentals, reference.annual_fundamentals, check_exact=True)
        assert client.discovery_calls == before


def test_incremental_preload_releases_hydrated_bundle_before_next_issuer(tmp_path, monkeypatch):
    from tests.fundamentals.test_incremental_snapshot_refresh import seeded, CUTOFF
    repository, _, service = seeded(tmp_path, complete=True)
    with repository.unit_of_work() as uow:
        original = service._load_active_entries({"2330": "24"}, as_of=CUTOFF,
            derivation_version=FUNDAMENTAL_DERIVATION_VERSION,
            uow=uow)["2330"]
        selected = repository.load_filings_many(["2330"], as_of=CUTOFF, uow=uow)["2330"]
    # Synthetic cloned issuer keys exercise lifetime, not company certification.
    headers = {symbol: replace(original, symbol=symbol) for symbol in ("SYNTHETIC-A", "SYNTHETIC-B")}
    monkeypatch.setattr(service, "_load_active_entries", lambda *args, **kwargs: headers)
    monkeypatch.setattr(repository, "load_filings_many", lambda symbols, **kwargs: {symbol: selected for symbol in symbols})
    hydrate, references = service._hydrate_entry, []
    def tracked(entry, *, uow):
        assert not any(reference() is not None for reference in references)
        actual = replace(hydrate(original, uow=uow), symbol=entry.symbol)
        references.append(weakref.ref(actual.bundle))
        return actual
    monkeypatch.setattr(service, "_hydrate_entry", tracked)
    with retained_work() as (_, owner):
        retained = owner.bundles()
        assert service.load_cached_bundles_into(list(headers), signal_date=CUTOFF, sector="24",
                                                sink=retained.__setitem__) == tuple(headers)
        assert not any(reference() is not None for reference in references)
        assert len(retained) == 2


def test_incremental_preload_sink_failure_propagates_without_fallback(tmp_path):
    from tests.fundamentals.test_incremental_snapshot_refresh import seeded, CUTOFF
    _, client, service = seeded(tmp_path)
    before = client.discovery_calls
    def broken(*args):
        raise ValueError("SYNTHETIC_RETENTION_FAILURE")
    with pytest.raises(ValueError, match="SYNTHETIC_RETENTION_FAILURE"):
        service.load_cached_bundles_into(["2330"], signal_date=CUTOFF, sector="24", sink=broken)
    assert client.discovery_calls == before


def test_explicit_spool_parent_owns_only_new_private_bytes(tmp_path):
    parent = tmp_path / "selected-volume"
    parent.mkdir()
    original = parent / "frozen_bundle.json"
    original.write_text("SYNTHETIC_IMMUTABLE_ORIGINAL")
    with retained_work(memory_bytes=0, spool_parent=parent) as (_, owner):
        mapping = owner.snapshots()
        mapping["2330"] = payload()
        directory = Path(owner._directory.name)
        assert directory.parent == parent
        assert directory.name.startswith("twse-native-snapshots-")
        assert mapping["2330"]["list_values"] == [True, "臺灣"]
    assert not directory.exists()
    assert original.read_text() == "SYNTHETIC_IMMUTABLE_ORIGINAL"


@pytest.mark.parametrize("memory_bytes", [0, 4096])
def test_analysis_scopes_release_snapshots_but_preserve_parent_maps_and_memo(memory_bytes):
    with retained_work(memory_bytes=memory_bytes, max_spool_bytes=4096) as (work, owner):
        # An empty parent catches accidental equality-based map removal.
        empty_parent, parent = owner.snapshots(), owner.snapshots()
        parent["shared"] = {"values": [1, 2]}
        parent_stats = owner.stats.copy()
        work.memo_json("synthetic", "parse", lambda: {"retained": 1})
        for day in range(30):
            with owner.scope():
                mapping = owner.snapshots()
                mapping["shared"] = {"values": [1, 2]}
                mapping[str(day)] = {"new-day": day, "values": [3]}
                bundles = owner.bundles()
                bundles["2330"] = FundamentalBundle(
                    pd.DataFrame({"year": [2025]}), {"values": [4]}, "2026-09-01T05:30:00Z")
                assert len(bundles["2330"].annual_fundamentals) == 1
            assert owner.stats == parent_stats
            assert owner._maps == [empty_parent, parent]
            assert parent["shared"] == {"values": [1, 2]}
            assert not bundles._annual
            with pytest.raises(ValueError, match="SCOPE_CLOSED"):
                mapping["resurrected"] = {}
            assert work.memo_json("synthetic", "parse", lambda: pytest.fail("memo lost")) == {"retained": 1}


def test_nested_analysis_scopes_release_only_their_maps_on_exception():
    with retained_work(memory_bytes=0) as (_, owner):
        with owner.scope():
            outer = owner.snapshots()
            outer["shared"] = {"value": 1}
            with pytest.raises(RuntimeError, match="SYNTHETIC"), owner.scope():
                inner = owner.snapshots()
                inner["shared"] = {"value": 1}
                inner["unique"] = {"value": 2}
                raise RuntimeError("SYNTHETIC")
            assert owner.stats["objects"] == 1
            assert outer["shared"] == {"value": 1}
            assert owner._maps == [outer]
        assert owner.stats["objects"] == owner.stats["spooled_bytes"] == 0
        assert not owner._maps
        assert not list(Path(owner._directory.name).iterdir())


def test_spool_reserve_fails_before_allocating_or_publishing_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr("infrastructure.snapshot_retention.disk_usage", lambda path: SimpleNamespace(free=10))
    with retained_work(memory_bytes=0, spool_parent=tmp_path, min_free_bytes=1024) as (_, owner):
        mapping = owner.snapshots()
        with pytest.raises(SnapshotRetentionStorageError) as caught:
            mapping["2330"] = {"value": 1}
        error = caught.value
        assert error.code == "SNAPSHOT_RETENTION_SPACE_REQUIRED"
        assert error.path == str(tmp_path)
        assert error.filename == str(tmp_path)
        assert error.free_bytes == 10 and error.required_bytes > 1024
        assert error.errno == errno.ENOSPC
        assert "free_bytes=10" in str(error)
        assert owner._directory is None and not mapping
        assert owner.stats["objects"] == owner.stats["spooled_bytes"] == 0
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EDQUOT])
def test_failed_spool_write_removes_partial_bytes_and_allows_verified_retry(tmp_path, monkeypatch, error_number):
    real_open = Path.open
    class FailedWrite:
        def __init__(self, output):
            self.output = output
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.output.close()
        def write(self, encoded):
            self.output.write(encoded[:7])
            raise OSError(error_number, "SYNTHETIC_FULL_VOLUME")
    def fail_open(path, *args, **kwargs):
        output = real_open(path, *args, **kwargs)
        return FailedWrite(output) if path.suffix == ".bin" and args == ("xb",) else output
    with retained_work(memory_bytes=0, spool_parent=tmp_path) as (_, owner):
        mapping = owner.snapshots()
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", fail_open)
            with pytest.raises(SnapshotRetentionStorageError) as caught:
                mapping["2330"] = payload()
        assert caught.value.code == "SNAPSHOT_RETENTION_STORAGE_FULL"
        assert caught.value.__cause__.errno == error_number
        assert not list(Path(owner._directory.name).iterdir())
        assert not mapping and not owner._references
        assert owner.stats["objects"] == owner.stats["spooled_bytes"] == 0
        mapping["2330"] = payload()
        assert mapping["2330"]["list_values"] == [True, "臺灣"]


def test_native_work_scope_forwards_spool_parent_and_borrows_existing_owner(tmp_path, monkeypatch):
    from infrastructure.daily_work import native_work_scope
    monkeypatch.setattr("infrastructure.daily_work.ResearchVersionGuard",
                        lambda: SimpleNamespace(versions=SimpleNamespace(), check=lambda: None))
    monkeypatch.setattr("infrastructure.daily_work.asdict", lambda _: {"code": "SYNTHETIC"})
    parent = tmp_path / "selected-volume" / "private-work"
    with native_work_scope(snapshot_spool_parent=parent) as work:
        owner = work.snapshot_retention
        assert owner.spool_parent == parent and parent.is_dir()
        owner.memory_bytes = 0
        with native_work_scope(snapshot_spool_parent=tmp_path / "unused") as borrowed:
            assert borrowed is work and borrowed.snapshot_retention is owner
            with owner.scope():
                mapping = owner.snapshots()
                mapping["2330"] = payload()
                directory = Path(owner._directory.name)
                assert directory.parent == parent
            assert owner.stats["objects"] == 0 and not owner.stats["closed"]
        assert not (tmp_path / "unused").exists()
    assert not directory.exists() and owner.stats["closed"]

