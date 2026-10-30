"""SYNTHETIC official-format originals: R15 role quarantine, no live repair."""
from copy import deepcopy

import pytest

from application.evidence_context import ReadOnlyEvidence
from domain.les_v7.contracts import identity
from domain.les_v7.valuation_recovery_r10 import (validate_comparable_components,
    validate_comparable_components_r15, validate_comparables)
from domain.les_v7.valuation_recovery_r15 import ANNUAL_METHOD, evaluate_recovery_price_valuation
from tests.strategies.test_v7_valuation_r13 import inputs
import tests.test_v7_valuation_r10_receipts as fixtures


CUTOFF = "2026-10-20T13:30:00+08:00"


def select(value, selector=validate_comparable_components_r15):
    return selector(value, symbol="1000", market="TWSE", cutoff=CUTOFF,
        eps_basis="DILUTED", fiscal_year=2025, scope="consolidated",
        annual_source=value["issuer"]["annual_filings"][-1],
        expected_session="2026-10-15", current_price=120.)


def context(monkeypatch, count=12):
    monkeypatch.setattr(fixtures, "SYMBOLS", [str(1000 + index) for index in range(count)])
    return fixtures.context(cutoff=CUTOFF)


@pytest.mark.parametrize("damage", ["identity", "quote", "annual_hash", "annual_revision", "annual_unit"])
def test_third_party_exception_is_quarantined_only_in_r15_with_original_references(monkeypatch, damage):
    value = context(monkeypatch); raw = value["peers"][0]
    if damage == "identity": raw["security_evidence"]["instrument"]["sector"] = "UNVERIFIED_EDIT"
    elif damage == "quote": raw["market_rows"][-1]["Close"] = 999.
    elif damage == "annual_hash": raw["annual_filings"][-1]["earnings_source"]["content_sha256"] = "0" * 64
    elif damage == "annual_revision":
        conflict = deepcopy(raw["annual_filings"][-1]); conflict["revision_id"] = "CONFLICT"
        raw["annual_filings"].append(conflict)
    else: raw["annual_filings"][-1]["earnings_source"]["field_evidence"]["diluted_eps"]["contexts"][0]["unit"] = "iso4217:TWD"
    before = deepcopy(value); result = select(value)
    assert result["status"] == "VERIFIED" and result["peer_count"] == 10
    assert result["peers"]["median"] == 10.
    q = result["peer_quarantine"][0]
    assert q["symbol"] == raw["symbol"] and q["role"] == "THIRD_PARTY_PEER" and q["reason"]
    assert q["block_id"] == identity(raw) and q["annual_sources"][-1]["content_sha256"]
    assert result["fixed_universe_symbols"] == value["universe_symbols"]
    assert select(value, validate_comparable_components)["status"] == "UNKNOWN"
    assert select(value, validate_comparables)["status"] == "UNKNOWN"
    assert value == before and select(ReadOnlyEvidence(value)) == result


def test_nine_valid_peers_cannot_borrow_quarantined_source_or_claim_full_coverage(monkeypatch):
    value = context(monkeypatch, count=11)
    value["peers"][0]["market_rows"][-1]["Close"] = 999.
    result = select(value)
    assert result["status"] == "INSUFFICIENT" and result["peer_count"] == 9
    assert result["peers"]["status"] == "INSUFFICIENT"
    valuation = evaluate_recovery_price_valuation(legacy_result=dict(inputs=inputs()), comparables=result)
    assert valuation["selected_method"] == ANNUAL_METHOD


@pytest.mark.parametrize("damage", ["own_identity", "own_quote", "own_annual", "own_history", "universe", "cutoff"])
def test_own_or_fixed_universe_conflicts_remain_unknown(monkeypatch, damage):
    value = context(monkeypatch)
    if damage == "own_identity": value["issuer"]["security_evidence"]["instrument"]["sector"] = "EDIT"
    elif damage == "own_quote": value["issuer"]["market_rows"][-1]["Close"] = 121.
    elif damage == "own_annual": value["issuer"]["annual_filings"][-1]["earnings_source"]["content_sha256"] = "0" * 64
    elif damage == "own_history": value["issuer"]["annual_filings"][0]["earnings_source"]["content_sha256"] = "0" * 64
    elif damage == "universe": value["peers"].append(deepcopy(value["peers"][0]))
    else: value["cutoff"] = "2027-01-01T00:00:00Z"
    result = select(value)
    assert result["status"] == "UNKNOWN"
    assert result["issuer_history"]["status"] == "UNKNOWN"


def test_future_revision_is_excluded_by_cutoff_not_quarantined(monkeypatch):
    value = context(monkeypatch); before = select(value)
    future = deepcopy(value["peers"][0]["annual_filings"][-1])
    future.update(available_at="2027-01-01T00:00:00Z", revision_id="FUTURE")
    value["peers"][0]["annual_filings"].append(future)
    after = select(value)
    assert after["peers"] == before["peers"]
    assert after["peer_quarantine"] == before["peer_quarantine"] == []
