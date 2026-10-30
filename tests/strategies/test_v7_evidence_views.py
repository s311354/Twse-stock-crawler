"""SYNTHETIC ONLY: immutable native v7 evidence, no market/performance claim."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from types import MappingProxyType

import numpy as np
import pytest

from application.evidence_context import EvidenceContext, ReadOnlyEvidence
from application.low_entry_v7_daily import daily_report, plain
from application.market_history import MarketHistoryAssembler
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.engine import evaluate
from tests.strategies.test_daily_v7_pipeline import prepared_synthetic_market, executable_synthetic_market
from tests.strategies.test_low_entry_score_v7 import bars, bundle, filings, CUTOFF


def _context():
    model = bundle()
    return dict(symbol="SYNTHETIC_ONLY", market="TWSE", as_of=CUTOFF,
        filings=filings(), model_bundle=model, quality_policy=model["payload"]["quality_policy"],
        receipts={}, external_evidence_ids=[])


def test_v7_native_and_mutable_context_reports_have_complete_equality():
    context = _context()
    before = deepcopy(context)
    data = bars()
    expected = evaluate(data, context)
    immutable = ReadOnlyEvidence(context)
    actual = evaluate(data, immutable)
    assert actual == expected
    assert context == before
    # The report is independently JSON-native, not a view of the source packet.
    actual["diagnostics"]["quality"]["reason_codes"].append("OUTPUT_ONLY")
    assert evaluate(data, immutable) == expected
    context["filings"][-1]["field_evidence"]["cfo"]["contexts"][0]["numeric_value"] = -999.
    assert evaluate(data, immutable) == expected
    assert evaluate(data, context) != expected
    with pytest.raises(TypeError, match="READ_ONLY_EVIDENCE"):
        immutable["filings"][-1]["field_evidence"]["cfo"]["status"] = "FORGED"


def test_v7_native_engine_does_not_deepcopy_financial_proof_dicts(monkeypatch):
    import copy
    copied = []
    original = copy._deepcopy_dispatch[dict]

    def counted(value, memo, deepcopy=deepcopy):
        if "content_sha256" in value or "field_evidence" in value:
            copied.append(id(value))
        return original(value, memo, deepcopy)

    context = _context()
    frozen = ReadOnlyEvidence(context)
    monkeypatch.setitem(copy._deepcopy_dispatch, dict, counted)
    expected = evaluate(bars(), context)
    mutable_copies = len(copied)
    copied.clear()
    assert evaluate(bars(), frozen) == expected
    assert mutable_copies >= 200  # stable call-count regression, not wall time
    assert copied == []


def test_v7_plain_shares_only_native_frozen_proofs_and_normalizes_scalar_dates():
    values = {"proof": {"source": "SYNTHETIC_ONLY", "contexts": [{"value": 5.}]}}
    native = ReadOnlyEvidence(values)
    assert plain(native, preserve_frozen=True) is native
    rows = (native,)
    assert plain(rows, preserve_frozen=True) is rows
    dated = ReadOnlyEvidence(dict(proof=native["proof"], available_at=datetime.fromisoformat(CUTOFF)))
    normalized = plain(dated, preserve_frozen=True)
    assert normalized["proof"] is native["proof"]
    assert normalized["available_at"] == CUTOFF
    assert isinstance(normalized, ReadOnlyEvidence)
    # A read-only facade does not make its underlying caller data immutable.
    detached = plain(MappingProxyType(values), preserve_frozen=True)
    values["proof"]["contexts"][0]["value"] = -1.
    assert detached["proof"]["contexts"][0]["value"] == 5.


def test_v7_json_native_boundary_does_not_rewalk_proof_tree(monkeypatch):
    frozen = ReadOnlyEvidence({"source": "SYNTHETIC_ONLY", "proof": {"contexts": [{"value": 5.}]}})
    assert frozen.json_native is True

    def forbidden(*args, **kwargs):
        raise AssertionError("Already-verified JSON-native proofs must not be walked again")

    monkeypatch.setattr(ReadOnlyEvidence, "items", forbidden)
    assert plain(frozen, preserve_frozen=True) is frozen
    rows = (frozen,)
    assert plain(rows, preserve_frozen=True) is rows


def test_v7_normalization_keeps_scalar_types_and_canonical_json():
    raw = dict(boolean=True, count=1, amount=1.5, missing=None, text="1",
        numpy_count=np.int64(7), nested=[dict(count=2, available_at=datetime.fromisoformat(CUTOFF))])
    expected = plain(raw)
    actual = plain(ReadOnlyEvidence(raw), preserve_frozen=True)
    assert canonical(actual) == canonical(expected)
    assert identity(actual) == identity(expected)
    assert type(actual["boolean"]) is bool
    assert type(actual["count"]) is int
    assert type(actual["amount"]) is float
    assert type(actual["numpy_count"]) is int
    assert actual["missing"] is None
    assert actual["text"] == "1"
    # Circular input is unsupported rather than silently removed from hashes.
    cyclic = {}; cyclic["self"] = cyclic
    with pytest.raises(RecursionError):
        plain(cyclic, preserve_frozen=True)


def test_daily_packet_reuses_filing_nodes_and_matches_mutable_adapter(monkeypatch):
    from domain.strategies.low_entry_score_v7 import LowEntryScoreV7Strategy
    market = executable_synthetic_market()
    assembler = MarketHistoryAssembler()
    expected = daily_report(assembler.build(market, 0))
    original = LowEntryScoreV7Strategy.evaluate
    captured = []

    def inspected(self, data, *, context=None, features=None):
        captured.append(context)
        return original(self, data, context=context, features=features)

    monkeypatch.setattr(LowEntryScoreV7Strategy, "evaluate", inspected)
    history = assembler.build(market, 0, include_evidence=False)
    with EvidenceContext(market) as work, work.stock(0, history) as packet:
        actual = daily_report(history, evidence=packet)
        context = captured[-1]
        assert isinstance(context, ReadOnlyEvidence)
        assert all(row is original for row, original in
                   zip(context["filings"], packet.snapshot["les_v6_filings"], strict=True))
        assert context["filings"][-1]["field_evidence"] is packet.snapshot["les_v6_filings"][-1]["field_evidence"]
    assert actual == expected
    assert actual["pae_authorization"]["authorized"] is True


def test_daily_packet_date_scalar_normalization_keeps_canonical_identity():
    market = prepared_synthetic_market()
    snapshots = deepcopy(market.fundamental_snapshots)
    row = snapshots["SYNTHETIC_ONLY"]["les_v6_filings"][-1]
    row["available_at"] = datetime.fromisoformat(row["available_at"])
    for proof in row["field_evidence"].values():
        proof["available_at"] = row["available_at"]
    row["adjusted_share_growth_evidence"]["available_at"] = row["available_at"]
    market = replace(market, fundamental_snapshots=snapshots)
    assembler = MarketHistoryAssembler()
    expected = daily_report(assembler.build(market, 0))
    history = assembler.build(market, 0, include_evidence=False)
    with EvidenceContext(market) as work, work.stock(0, history) as packet:
        actual = daily_report(history, evidence=packet)
    assert actual == expected
    assert actual["envelope"]["quality"]["status"] == "PASS"


@pytest.mark.parametrize("invalid", [None, "bad-date", "2026-08-01"])
def test_native_packet_invalid_availability_remains_fail_closed(invalid):
    market = prepared_synthetic_market()
    snapshots = deepcopy(market.fundamental_snapshots)
    snapshots["SYNTHETIC_ONLY"]["les_v6_filings"][-1]["available_at"] = invalid
    market = replace(market, fundamental_snapshots=snapshots)
    assembler = MarketHistoryAssembler()
    expected = daily_report(assembler.build(market, 0))
    history = assembler.build(market, 0, include_evidence=False)
    with EvidenceContext(market) as work, work.stock(0, history) as packet:
        actual = daily_report(history, evidence=packet)
    assert actual == expected
    assert actual["envelope"]["score"]["final_score"] is None
    assert "INVALID_FINANCIAL_AVAILABILITY" in actual["diagnostics"]["daily_input_errors"]
    assert actual["pae_authorization"]["authorized"] is False


def test_native_packet_future_filing_and_changed_account_are_not_cache_hits():
    market = executable_synthetic_market()
    assembler = MarketHistoryAssembler()
    expected = daily_report(assembler.build(market, 0))
    snapshots = deepcopy(market.fundamental_snapshots)
    future = deepcopy(snapshots["SYNTHETIC_ONLY"]["les_v6_filings"][-1])
    future["available_at"] = "2026-10-13T10:00:00+08:00"
    future["cfo"] = -1e9
    snapshots["SYNTHETIC_ONLY"]["les_v6_filings"].append(future)
    market = replace(market, fundamental_snapshots=snapshots)
    history = assembler.build(market, 0, include_evidence=False)
    with EvidenceContext(market) as work, work.stock(0, history) as packet:
        assert daily_report(history, evidence=packet) == expected
        history.attrs["les_v6_portfolio"] = dict(history.attrs["les_v6_portfolio"], available_cash=1000.)
        changed = daily_report(history, evidence=packet)
        assert changed["envelope"] == expected["envelope"]
        assert changed["pae_authorization"]["authorization_id"] != expected["pae_authorization"]["authorization_id"]
        assert changed["pae_authorization"]["indicative_shares"] < expected["pae_authorization"]["indicative_shares"]
