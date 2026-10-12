"""Metadata performance boundaries: exact native semantics, no invented evidence."""
from copy import deepcopy
from dataclasses import replace
import json
import weakref

import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_entry import DailyEntryService
from application.daily_valuation import DailyValuationProvider
from application.evidence_context import EVIDENCE_CONTEXT_ID_ATTR, EvidenceContext, ReadOnlyEvidence
from application.leadership_screen import screen_once
from application.low_entry_v6_daily import daily_report
from application.low_entry_v6_daily import build_daily_v6
from application.market_history import MarketHistoryAssembler
from domain.features import SharedFeaturePipeline
from domain.numeric_history import metadata_free_frame
from domain.strategies.expected_return_exit_v3 import ExpectedReturnExitV3Strategy
from domain.strategies.low_entry_score_v4 import ANNUAL_FUNDAMENTALS_ATTR, FUNDAMENTAL_SNAPSHOT_ATTR, SIGNAL_DATE_ATTR
from tests.test_daily_computation_performance import two_stock_market
from tests.test_daily_entry_components import market_with_sidecar


class CopyTrap:
    def __deepcopy__(self, memo):
        raise AssertionError("Numeric calculation deep-copied financial attrs")


def test_public_constructor_copies_values_dtypes_and_index_without_attrs():
    source = pd.DataFrame({"price": [1., float("nan")], "flag": pd.Series([True, None], dtype="boolean")})
    source.index = pd.Index([7, 17], name="bar")
    source.attrs["raw_evidence"] = CopyTrap()
    clean = metadata_free_frame(source)
    assert clean.attrs == {}
    pd.testing.assert_frame_equal(clean, pd.DataFrame(source, copy=True), check_exact=True)
    clean.loc[7, "price"] = 99
    assert source.to_numpy()[0, 0] == 1.
    assert isinstance(source.attrs["raw_evidence"], CopyTrap)


def test_features_fingerprint_watch_les_and_erem_never_copy_unrelated_attrs():
    history = MarketHistoryAssembler().build(market_with_sidecar("complete"), 0)
    history.attrs["unused_raw_evidence"] = CopyTrap()
    numeric = metadata_free_frame(history)
    pipeline = SharedFeaturePipeline()
    native, normalized = pipeline.build_price_pair(history)
    native.assert_compatible(history)
    pd.testing.assert_frame_equal(native.to_pandas(), pipeline.build(numeric).to_pandas(), check_exact=True)
    erem = ExpectedReturnExitV3Strategy()
    pd.testing.assert_frame_equal(erem.run_latest_with_features(history, native),
                                  erem.run_with_features(numeric, native).tail(1), check_exact=True)
    assert erem.run_latest_with_features(history, native).attrs == {}
    watches, _ = screen_once([history], session_date="2026-09-11", as_of=history.attrs[SIGNAL_DATE_ATTR])
    report = daily_report(history, watch=watches[0], features=normalized)
    assert report["technical"]["close"] == 96.7
    assert report["quality"]["status"] == "PASS"


def test_stock_evidence_is_bound_readonly_and_annual_is_copy_on_read():
    market = market_with_sidecar("complete")
    history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
    with EvidenceContext(market) as context:
        with context.stock(0, history) as packet:
            assert FUNDAMENTAL_SNAPSHOT_ATTR not in history.attrs
            assert ANNUAL_FUNDAMENTALS_ATTR not in history.attrs
            assert isinstance(history.attrs[EVIDENCE_CONTEXT_ID_ATTR], str)
            assert deepcopy(packet.snapshot) is packet.snapshot
            original = packet.snapshot["les_v6_filings"][0]["revenue"]
            with pytest.raises(TypeError):
                packet.snapshot["les_v6_filings"][0]["revenue"] = -1
            with pytest.raises(TypeError):
                packet.snapshot["les_v6_filings"][0] = {}
            market.fundamental_snapshots["2330"]["les_v6_filings"][0]["revenue"] = -1
            assert packet.snapshot["les_v6_filings"][0]["revenue"] == original
            annual = packet.annual()
            annual.iloc[0, 0] = -1
            assert packet.annual().iloc[0, 0] != -1
            changed = metadata_free_frame(history)
            changed.attrs = dict(history.attrs, **{SIGNAL_DATE_ATTR: "2026-09-12T15:00:00+08:00"})
            with pytest.raises(ValueError, match="BINDING_MISMATCH"):
                packet.validate(changed)
            changed.attrs = dict(history.attrs, les_v6_daily_identity={"symbol": "2317"})
            with pytest.raises(ValueError, match="BINDING_MISMATCH"):
                packet.validate(changed)
            changed.attrs = dict(history.attrs, les_v6_daily_identity={"symbol": "2330", "market": "TPEX"})
            with pytest.raises(ValueError, match="BINDING_MISMATCH"):
                packet.validate(changed)
            with pytest.raises(ValueError, match="CLOSED_OR_ACTIVE"):
                with context.stock(0, history):
                    pass
    with pytest.raises(ValueError, match="CLOSED_OR_ACTIVE"):
        with context.stock(0, history):
            pass


def test_binding_is_work_local_and_preserves_exact_evidence_values():
    market = market_with_sidecar("complete")
    def identity(market):
        history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
        with EvidenceContext(market) as context, context.stock(0, history) as packet:
            return packet.evidence_id
    assert identity(market) != identity(market)
    annual = market.annual_fundamentals["2330"].copy()
    annual.loc[annual.index[0], "revenue"] += .000000001
    market = replace(market, annual_fundamentals={"2330": annual})
    history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
    with EvidenceContext(market) as context, context.stock(0, history) as packet:
        pd.testing.assert_frame_equal(packet.annual(), annual, check_exact=True)
        assert packet.snapshot["les_v6_filings"][0]["revision_id"] == market.fundamental_snapshots["2330"]["les_v6_filings"][0]["revision_id"]
        first = packet.evidence_id
    with EvidenceContext(market) as context, context.stock(0, history) as packet:
        assert packet.evidence_id != first


def test_screen_is_small_but_has_identical_full_universe_signature_and_output():
    market = two_stock_market()
    assembler = MarketHistoryAssembler()
    full = assembler.build_all(market)
    light = [assembler.build_screen(market, i) for i in range(market.stock_count)]
    assert all(len(frame) == 61 and FUNDAMENTAL_SNAPSHOT_ATTR not in frame.attrs for frame in light)
    options = {"session_date": "2026-09-11", "as_of": "2026-09-11T15:00+08:00"}
    assert screen_once(full, **options) == screen_once(light, **options)


def test_legacy_injected_entry_and_valuation_receive_native_json_without_mutating_source():
    from domain.market_breadth import MARKET_BREADTH_HISTORY_ATTR
    def check(history):
        snapshot = history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR]
        assert isinstance(snapshot, dict) and isinstance(snapshot["les_v6_filings"], list)
        json.dumps(snapshot)
        assert history.attrs[MARKET_BREADTH_HISTORY_ATTR] == {
            "2026-09-10": {"as_of": "2026-09-10", "source_final": True, "sample": 10}}
    def entry(histories):
        for history in histories:
            check(history)
        return build_daily_v6(histories)
    class LegacyValuation(DailyValuationProvider):
        def evaluate(self, history, features):
            check(history)
            result = super().evaluate(history, features)
            history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR]["les_v6_filings"].clear()
            return result
    market = replace(market_with_sidecar("complete"), market_breadth_snapshots={
        "2026-09-10": {"as_of": "2026-09-10", "source_final": True, "sample": 10},
        "2026-09-12": {"as_of": "2026-09-12", "source_final": True, "sample": 99}})
    actual = AnalysisDatasetBuilder(daily_entry=DailyEntryService(
        entry_evaluator=entry, valuation_provider=LegacyValuation())).build_decisions(market, [[1, 1, 1, 1, .01]])
    expected = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]])
    assert actual == expected
    assert market.fundamental_snapshots["2330"]["les_v6_filings"]


@pytest.mark.parametrize("malformed", [False, True])
def test_historical_json_restriction_contract_is_unchanged(malformed):
    from application.restriction_history import HistoricalRestrictionLedger
    from tests.test_restriction_announcements import bundle, close_replay_market
    market = close_replay_market()
    cutoff = market.signal_dates["2330"]
    restrictions = HistoricalRestrictionLedger(bundle()).reconstruct(
        ["2330"], market="TWSE", as_of=cutoff, market_data_date="2026-09-11")
    if malformed:
        # A caller-supplied malformed ledger must not be silently repaired to
        # PASS by converting every tuple into a JSON list in a validator.
        restrictions["2330"]["coverage"] = tuple(restrictions["2330"]["coverage"])
    market = replace(market, trading_restrictions=restrictions, replay_as_of=cutoff)
    results = [AnalysisDatasetBuilder(optimized_daily=flag).build_decisions(market, [[1, 1, 1, 1, .01]])
               for flag in (False, True)]
    assert results[0] == results[1]
    report = json.loads(results[1][0].value("LES v6 Report"))
    assert report["data"]["restriction_check"]["status"] == ("UNKNOWN" if malformed else "CLEAR")


def test_daily_releases_financial_packets_between_stocks_even_when_entry_fails(monkeypatch):
    from application.daily_entry import DailyEntryService
    references = []
    actual = DailyEntryService.evaluate
    def tracked(service, prepared):
        assert not any(reference() is not None for reference in references)
        packet = prepared.evidence[0]
        references.append(weakref.ref(packet.snapshot))
        assert all(FUNDAMENTAL_SNAPSHOT_ATTR not in frame.attrs for frame in prepared.histories)
        return actual(service, prepared)
    monkeypatch.setattr(DailyEntryService, "evaluate", tracked)
    market = two_stock_market()
    decisions = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]] * 2)
    assert len(decisions) == len(references) == 2
    assert not any(reference() is not None for reference in references)
    assert all("evidence_context_id" not in row.value("LES v6 Report") for row in decisions)


def test_context_failure_releases_scope_and_gc_observer_never_tunes_gc():
    import gc
    from examples.benchmark_daily_computation import observe_gc
    before = (gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks))
    with pytest.raises(RuntimeError), observe_gc():
        raise RuntimeError("synthetic")
    assert (gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks)) == before
    market = market_with_sidecar("complete")
    history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
    with EvidenceContext(market) as context:
        with pytest.raises(RuntimeError), context.stock(0, history):
            raise RuntimeError("synthetic")
        with context.stock(0, history) as packet:
            packet.validate(history)


@pytest.mark.parametrize("case", ["complete", "missing", "future_revision", "reject", "missing_restriction"])
def test_external_context_matches_all_legacy_business_fields(case):
    market = market_with_sidecar("complete")
    snapshots = deepcopy(market.fundamental_snapshots)
    if case == "missing":
        snapshots = {}
    elif case == "future_revision":
        future = dict(snapshots["2330"]["les_v6_filings"][-1], revision_id="future",
                      available_at="2026-09-12T15:00+08:00", total_equity=-1)
        snapshots["2330"]["les_v6_filings"].append(future)
    elif case == "reject":
        snapshots["2330"]["going_concern"] = True
    snapshots.get("2330", {}).update({"retained_raw_metadata": [{"context": {"source": "SYNTHETIC", "fields": [1, 2, 3]}}] * 100})
    market = replace(market, fundamental_snapshots=snapshots,
                     trading_restrictions={} if case == "missing_restriction" else market.trading_restrictions)
    before = AnalysisDatasetBuilder(optimized_daily=False).build_decisions(market, [[1, 1, 1, 1, .01]])
    after = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]])
    assert after == before
    report = json.loads(after[0].value("LES v6 Report"))
    if case in {"missing", "missing_restriction", "reject"}:
        assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}
