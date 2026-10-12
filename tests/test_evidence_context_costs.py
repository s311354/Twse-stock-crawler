"""Lossless preprocessing costs and cache isolation, without live requests."""
from copy import copy, deepcopy
from dataclasses import replace
import json
import operator

import pytest

from application.daily_performance import DailyPerformance
from application.daily_work import DailyWorkContext
from application.evidence_context import EvidenceContext, ReadOnlyEvidence
from application.low_entry_v6_daily import daily_report
from application.market_history import MarketHistoryAssembler
from application.replay_contracts import canonical_json, digest
from tests.strategies.test_daily_v6_pipeline import daily_market


def test_readonly_graph_encodes_without_materializing_a_second_graph():
    proof = {"status": "UNKNOWN", "source": "SYNTHETIC_ONLY", "values": [1, None, "臺灣"]}
    source = {"left": proof, "right": proof, "flag": False}
    frozen = ReadOnlyEvidence(source)
    assert frozen["left"] is frozen["right"]
    assert canonical_json(frozen) == canonical_json(source)
    assert copy(frozen) is deepcopy(frozen) is frozen
    for mutate in (
        lambda: operator.setitem(frozen, "flag", True),
        lambda: operator.delitem(frozen, "flag"),
        lambda: frozen.update(flag=True), lambda: frozen.clear(),
        lambda: frozen.pop("flag"), lambda: frozen.popitem(),
        lambda: frozen.setdefault("new", 1),
        lambda: operator.ior(frozen, {"flag": True}),
        lambda: frozen.__init__({"flag": True}),
        lambda: setattr(frozen, "_initialized", False),
    ):
        with pytest.raises(TypeError, match="READ_ONLY_EVIDENCE"):
            mutate()
    proof["values"][0] = 999
    assert frozen["left"]["values"][0] == 1
    detached = json.loads(canonical_json(frozen))
    detached["left"]["values"][0] = -1
    assert detached["right"]["values"][0] == frozen["left"]["values"][0] == 1


def test_captured_filing_content_is_hashed_once_and_includes_actual_proofs():
    market = daily_market()
    proof = {"status": "PENDING_VERIFICATION", "revision_id": "unchanged-claim"}
    market.fundamental_snapshots["2330"]["les_v6_filings"][-1]["proof_for_test"] = proof
    assembler = MarketHistoryAssembler()
    probe = DailyPerformance()
    with probe.activate(), EvidenceContext(market) as context:
        history = assembler.build(market, 0, include_evidence=False)
        with context.stock(0, history) as packet:
            expected = digest(market.fundamental_snapshots["2330"]["les_v6_filings"])
            assert packet.filings_content_key() == packet.filings_content_key() == expected
            proof["status"] = "CONFLICT"
            assert packet.filings_content_key() == expected
        with context.stock(0, history) as packet:
            assert packet.filings_content_key() != expected
    assert probe.counters["quality_diagnostics_input_hashes"] == 2


@pytest.mark.parametrize("enabled", [True, False])
def test_same_packet_retry_never_rehashes_or_reuses_account_decisions(enabled):
    market = daily_market()
    history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
    expected = daily_report(MarketHistoryAssembler().build(market, 0), watch={})
    probe = DailyPerformance()
    work = DailyWorkContext("synthetic-only", enabled=enabled)
    try:
        with probe.activate(), work.activate(), EvidenceContext(market) as context, context.stock(0, history) as packet:
            first = daily_report(history, watch={}, evidence=packet)
            assert first == expected
            first["research"]["verification"]["field_checks"].clear()
            second = daily_report(history, watch={}, evidence=packet)
            assert second == expected
    finally:
        work.close()
    assert probe.counters["quality_diagnostics_input_hashes"] == (1 if enabled else 0)
    assert probe.counters["quality_diagnostics_cache_hits"] == (1 if enabled else 0)
    assert probe.calls["quality_gate"] == 2


@pytest.mark.parametrize("change", ["proof", "cutoff", "market", "policy"])
def test_diagnostic_memo_invalidates_every_relevant_boundary(change, monkeypatch):
    from application import low_entry_v6_daily as daily

    market = daily_market()
    # A claimed original SHA is not a cache identity for all source metadata.
    market.fundamental_snapshots["2330"]["les_v6_filings"][-1]["content_sha256"] = "a" * 64
    work = DailyWorkContext("same-code")
    probe = DailyPerformance()
    def run(market):
        history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
        with EvidenceContext(market) as context, context.stock(0, history) as packet:
            return daily_report(history, watch={}, evidence=packet)
    try:
        with work.activate(), probe.activate():
            run(market)
            if change == "proof":
                market.fundamental_snapshots["2330"]["les_v6_filings"][-1]["evidence_note"] = "changed-proof"
            elif change == "cutoff":
                market = replace(market, signal_dates={"2330": "2026-09-12T13:30:00+08:00"})
            elif change == "market":
                market = replace(market, market_source="TPEX")
            else:
                monkeypatch.setattr(daily, "QUALITY_POLICY_VERSION", "different-validator-policy")
            actual = run(market)
            # No-cache path still produces the entire same business result.
            with DailyWorkContext("same-code", enabled=False).activate():
                assert run(market) == actual
    finally:
        work.close()
    assert probe.counters["quality_diagnostics_cache_misses"] == 2
    assert probe.counters["quality_diagnostics_cache_hits"] == 0
