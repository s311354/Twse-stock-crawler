"""SYNTHETIC ONLY: metadata paths never hydrate the financial universe."""
from dataclasses import replace
import json
import pickle

import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance
from application.daily_stock import isolate_stock_market
from application.evidence_context import EvidenceContext, ReadOnlyEvidence
from application.market_history import MarketHistoryAssembler
from application.snapshot_retention import snapshot_metadata
from infrastructure.snapshot_retention import WorkSnapshotRetention
from tests.test_daily_computation_performance import two_stock_market
from tests.strategies.test_daily_v7_pipeline import executable_synthetic_market


def retained_market(source, owner):
    retained = owner.snapshots()
    for symbol, snapshot in source.fundamental_snapshots.items():
        retained.retain(symbol, snapshot, cutoff=source.signal_dates.get(symbol), market=source.market_source)
    return replace(source, fundamental_snapshots=retained)


@pytest.mark.parametrize("v7", [False, True])
def test_daily_full_results_equal_and_each_source_hydrates_once(v7):
    source = executable_synthetic_market() if v7 else two_stock_market()
    profits = [[1, 1, 1, 1, .01]] * source.stock_count
    expected = AnalysisDatasetBuilder().build_decisions(source, profits)
    owner = WorkSnapshotRetention("synthetic-consumer")
    try:
        market = retained_market(source, owner)
        probe = DailyPerformance()
        with probe.activate():
            actual = AnalysisDatasetBuilder().build_decisions(market, profits)
        assert actual == expected  # Includes every LES/EREM field and provenance.
        assert probe.counters["snapshot_materializations"] == len(source.fundamental_snapshots)
        if v7:
            report = json.loads(actual[0].value("LES v7 Report"))
            assert report["pae_authorization"]["authorized"] is True
    finally:
        owner.close()


def test_screen_and_sector_metadata_never_decode_even_when_heavy_only(monkeypatch):
    source = two_stock_market()
    owner = WorkSnapshotRetention("synthetic-metadata")
    try:
        market = retained_market(source, owner)
        market.fundamental_snapshots.retain("2317", {"proof_only": [{"value": 5.}]},
                                            cutoff=source.signal_dates.get("2317"), market=source.market_source)
        def forbidden(*args, **kwargs):
            raise AssertionError("Metadata consumers must not materialize financials")
        monkeypatch.setattr(owner, "_materialize", forbidden)
        for item in range(market.stock_count):
            MarketHistoryAssembler().build_screen(market, item)
            MarketHistoryAssembler().build(market, item, include_evidence=False)
        assert bool(snapshot_metadata(market.fundamental_snapshots, "2317")) is True
        assert bool(snapshot_metadata(market.fundamental_snapshots, "ABSENT")) is False
        assert set(market.fundamental_snapshots) == {"2330", "2317"}
    finally:
        owner.close()


def test_worker_transport_materializes_one_stock_once_and_has_no_spool_reference():
    source = two_stock_market()
    owner = WorkSnapshotRetention("synthetic-transport", memory_bytes=0)
    try:
        market = retained_market(source, owner)
        probe = DailyPerformance()
        with probe.activate():
            packet = isolate_stock_market(market, 0)
        assert probe.counters["snapshot_materializations"] == 1
        assert type(packet.fundamental_snapshots) is dict
        assert packet.fundamental_snapshots == {"2330": source.fundamental_snapshots["2330"]}
        transported = pickle.loads(pickle.dumps(packet))
        assert transported.fundamental_snapshots == packet.fundamental_snapshots
        assert all("snapshot_retention" not in str(key) for key in transported.fundamental_snapshots)
    finally:
        owner.close()


def test_evidence_packet_refuses_retained_cutoff_and_market_mismatch():
    source = two_stock_market()
    owner = WorkSnapshotRetention("synthetic-bindings")
    try:
        market = retained_market(source, owner)
        for changed in (replace(market, signal_dates={"2330": "2026-09-12T15:00+08:00"}),
                        replace(market, market_source="TPEX")):
            history = MarketHistoryAssembler().build(changed, 0, include_evidence=False)
            with EvidenceContext(changed) as context, pytest.raises(ValueError, match="BINDING_MISMATCH"):
                with context.stock(0, history):
                    pass
    finally:
        owner.close()


def test_json_native_flag_is_readonly_and_does_not_change_missing_or_float_semantics():
    from datetime import date
    assert ReadOnlyEvidence({"x": [{"value": None}, {"value": float("nan")}]}).json_native is True
    dated = ReadOnlyEvidence({"x": [{"date": date(2026, 10, 12)}]})
    assert dated.json_native is False
    with pytest.raises(TypeError, match="READ_ONLY_EVIDENCE"):
        dated._json_native = True
