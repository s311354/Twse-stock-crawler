"""Synthetic graph reuse regressions; no live evidence or account mutation."""
from copy import deepcopy
from dataclasses import replace
import json
import weakref

import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance
from application.daily_work import DailyWorkContext
from application.evidence_context import EvidenceContext, StockEvidence
from application.low_entry_v6_daily import daily_report
from application.market_history import MarketHistoryAssembler
from domain.strategies import low_entry_v6_quality as quality
from domain.strategies.low_entry_v6_valuation import valuation_context
from tests.strategies.test_daily_v6_pipeline import daily_market
from tests.strategies.test_low_entry_v6_comparative_ttm import originals


def market_with_originals():
    records, _ = originals()
    return replace(daily_market(), fundamental_snapshots={
        "2330": {"sector": "24", "les_v6_filings": records}})


@pytest.mark.parametrize("cache", [False, True])
def test_one_cutoff_shares_selection_normalization_and_reported_inputs(cache, monkeypatch):
    market = market_with_originals()
    expected = AnalysisDatasetBuilder(optimized_daily=False).build_decisions(market, [[1, 1, 1, 1, .01]])
    calls = {}
    for name in ("_eligible", "normalize_quarters_with_diagnostics", "reported_period_inputs"):
        original = getattr(quality, name)
        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(quality, name, counted)
    probe = DailyPerformance()
    work = DailyWorkContext("synthetic-only", enabled=cache)
    try:
        with work.activate(), probe.activate():
            actual = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]])
    finally:
        work.close()
    assert actual == expected  # complete report, gates, scores, sizing, EREM
    assert calls == {"_eligible": 1, "normalize_quarters_with_diagnostics": 1, "reported_period_inputs": 1}
    assert probe.counters["quality_prepare_selection_builds"] == 1
    assert probe.counters["quality_prepare_quarters_builds"] == 1
    assert probe.counters["quality_prepare_reported_builds"] == 1


def test_every_cutoff_reselects_and_only_three_intermediates_live(monkeypatch):
    market = market_with_originals()
    market = replace(market, signal_dates={"2330": "2026-09-13T13:30:00+08:00"})
    future = deepcopy(market.fundamental_snapshots["2330"]["les_v6_filings"][-1])
    future.update(available_at="2026-09-12T09:00:00+08:00", revision_id="new-revision", cfo=-100)
    market.fundamental_snapshots["2330"]["les_v6_filings"].append(future)
    original = quality._eligible
    calls = []
    def selected(*args, **kwargs):
        calls.append(args[3])
        return original(*args, **kwargs)
    monkeypatch.setattr(quality, "_eligible", selected)
    history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
    with EvidenceContext(market) as context, context.stock(0, history) as packet:
        records = packet.snapshot["les_v6_filings"]
        for cutoff in ("2026-09-11T13:30:00+08:00", "2026-09-12T13:30:00+08:00",
                       "2026-09-11T13:30:00+08:00"):
            options = dict(symbol="2330", market="TWSE", as_of=cutoff)
            # Byte-equivalent native JSON, including all provenance diagnostics.
            actual = quality.evaluate_quality(records, **options, preprocessing=packet.quality_preprocessing)
            baseline = quality.evaluate_quality(records, **options)
            assert json.loads(json.dumps(vars(actual))) == json.loads(json.dumps(vars(baseline)))
            assert json.loads(json.dumps(quality.verification_diagnostics(
                records, **options, preprocessing=packet.quality_preprocessing))) == json.loads(json.dumps(
                    quality.verification_diagnostics(records, **options)))
            assert set(packet._quality_values) <= {"selection", "quarters", "reported"}
            with pytest.raises(TypeError, match="READ_ONLY_EVIDENCE"):
                packet._quality_values["selection"][0].clear()
    # Three optimized cutoff selections plus six no-preprocessing comparisons.
    assert len(calls) == 9


@pytest.mark.parametrize("boundary", ["policy", "work_version"])
def test_scoped_preprocessing_invalidates_policy_and_work_versions(boundary, monkeypatch):
    market = daily_market()
    history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
    probe = DailyPerformance()
    work = DailyWorkContext("first")
    try:
        with probe.activate(), work.activate(), EvidenceContext(market) as context, context.stock(0, history) as packet:
            options = dict(symbol="2330", market="TWSE", as_of=packet.cutoff,
                           preprocessing=packet.quality_preprocessing)
            first = quality.evaluate_quality(packet.snapshot["les_v6_filings"], **options)
            if boundary == "policy":
                monkeypatch.setattr(quality, "QUALITY_POLICY_VERSION", "different-policy")
            else:
                work.version = "second"
            second = quality.evaluate_quality(packet.snapshot["les_v6_filings"], **options)
            # Raw formulas/financial input did not change; provenance policy may.
            assert second.score == first.score
            assert len(packet._quality_values) == 3
    finally:
        work.close()
    assert probe.counters["quality_prepare_selection_builds"] == 2


@pytest.mark.parametrize("change", ["future", "old_history", "unrelated_snapshot"])
def test_unselected_graph_changes_reuse_diagnostics_only_after_reselection(change):
    market = daily_market()
    work, probe = DailyWorkContext("same-code"), DailyPerformance()
    def run():
        history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
        with EvidenceContext(market) as context, context.stock(0, history) as packet:
            return daily_report(history, watch={}, evidence=packet)
    try:
        with work.activate(), probe.activate():
            first = run()
            records = market.fundamental_snapshots["2330"]["les_v6_filings"]
            if change == "unrelated_snapshot":
                market.fundamental_snapshots["2330"]["unused_metadata"] = ["not-consumed"] * 100
            else:
                record = deepcopy(records[-1])
                if change == "future":
                    record.update(available_at="2026-09-12T15:00:00+08:00", revision_id="future", cfo=-9999)
                else:
                    record.update(period_end="2020-03-31", available_at="2020-05-01T00:00:00Z", revision_id="old")
                records.append(record)
            assert run() == first
    finally:
        work.close()
    assert probe.counters["quality_prepare_selection_builds"] == 2
    assert probe.counters["quality_diagnostics_cache_hits"] == 1
    assert probe.calls["quality_gate"] == 2


@pytest.mark.parametrize("damage", ["missing_date", "ambiguous", "incompatible_latest", "missing_cfo", "wrong_proof"])
def test_failures_and_unknown_attribution_match_uncached_original(damage):
    market = market_with_originals()
    records = market.fundamental_snapshots["2330"]["les_v6_filings"]
    if damage == "missing_date":
        records[0].pop("available_at")
    elif damage == "ambiguous":
        records.append(dict(records[-1], revision_id="conflicting-same-time", cfo=-9))
    elif damage == "incompatible_latest":
        records[-1]["unit"] = "dollar"
    elif damage == "missing_cfo":
        records[-1]["cfo"] = None
    else:
        records[-1]["field_evidence"]["cfo"]["revision_id"] = "different-proof"
    before = deepcopy(records)
    expected = AnalysisDatasetBuilder(optimized_daily=False).build_decisions(market, [[1, 1, 1, 1, .01]])
    work = DailyWorkContext("synthetic-only")
    try:
        with work.activate():
            results = [AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]]) for _ in range(2)]
    finally:
        work.close()
    assert results == [expected, expected]
    assert records == before
    report = json.loads(results[0][0].value("LES v6 Report"))
    assert report["quality"]["status"] == "DATA_INSUFFICIENT"
    assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}


def test_valuation_preserves_its_stricter_incomparable_history_contract():
    market = daily_market()
    records = market.fundamental_snapshots["2330"]["les_v6_filings"]
    records[-3]["scope"] = "individual"
    history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
    with EvidenceContext(market) as context, context.stock(0, history) as packet:
        options = dict(symbol="2330", market="TWSE", as_of=packet.cutoff)
        records = packet.snapshot["les_v6_filings"]
        q = quality.evaluate_quality(records, **options, preprocessing=packet.quality_preprocessing)
        assert q.integrity == "PASS"
        assert q.metrics["excluded_quality_inputs"]
        actual = valuation_context(records, **options, quality=q, preprocessing=packet.quality_preprocessing)
        assert actual == valuation_context(records, **options, quality=q)
        assert actual["reason"] == "INCOMPARABLE_OR_MISSING_VALUATION_SOURCES"


def test_wrong_source_binding_and_later_than_packet_cutoff_are_rejected():
    market = daily_market()
    history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
    with EvidenceContext(market) as context, context.stock(0, history) as packet:
        records = packet.snapshot["les_v6_filings"]
        for wrong in (dict(records=deepcopy(market.fundamental_snapshots["2330"]["les_v6_filings"])),
                      dict(symbol="2317"), dict(market="TPEX")):
            options = dict(records=records, symbol="2330", market="TWSE", as_of=packet.cutoff,
                           preprocessing=packet.quality_preprocessing)
            options.update(wrong)
            assert quality.evaluate_quality(**options).reasons == ("QUALITY_PREPROCESSING_BINDING_MISMATCH",)
        with pytest.raises(ValueError, match="CUTOFF_MISMATCH"):
            packet.diagnostics_content_key(symbol="2330", market="TWSE", as_of="2026-09-12T13:30:00+08:00")


def test_preprocessing_graphs_are_released_after_each_stock(monkeypatch):
    from tests.test_evidence_context import two_stock_market
    references = []
    original = StockEvidence.quality_preprocessing
    def tracked(packet, kind, compute, **kwargs):
        value = original(packet, kind, compute, **kwargs)
        # Each intermediate's first member is a ReadOnlyEvidence mapping.
        references.append(weakref.ref(value[0]))
        return value
    monkeypatch.setattr(StockEvidence, "quality_preprocessing", tracked)
    market = two_stock_market()
    results = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]] * 2)
    assert len(results) == 2
    assert references
    assert all(reference() is None for reference in references)
