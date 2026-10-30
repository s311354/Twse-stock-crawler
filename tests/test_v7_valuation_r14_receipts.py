"""SYNTHETIC sources only: R14 component bindings and old receipt parity."""
from copy import deepcopy

import pandas as pd
import pytest

from application.evidence_context import ReadOnlyEvidence
from domain.les_v7.contracts import identity
from domain.les_v7.policy_r14 import AVAILABLE_AT, INPUT_VERSION
from domain.les_v7.valuation_eps_receipt import COMPONENT_PRODUCER, produce_price_valuation_receipt
from domain.les_v7.valuation_input_policy import input_policy, require_input_policy
from domain.les_v7.valuation_recovery_r10 import validate_comparable_components, validate_comparables
from domain.les_v7.valuation_recovery_r14 import ANNUAL_METHOD, FULL_METHOD, PEER_METHOD, policy
from tests.test_v7_eps_price_receipts import item
from tests.test_v7_valuation_r10_receipts import context, recent, SYMBOLS
from infrastructure.crawler.security_master import capture_security_master, select_security_master
from tests.test_security_master import synthetic_instruments, synthetic_issuers


CUTOFF = "2026-10-20T13:30:00+08:00"


def ctx():
    return context(cutoff=CUTOFF)


def classified_context(industry="資訊服務業"):
    """A fresh source-bound official-format classification, never a flag edit."""
    value = ctx()
    rows = [("股票", [f"{symbol}　合成普通股", f"TW000{symbol}008", "1994/09/05", "上市", industry, "ESVUFR", ""])
            for symbol in SYMBOLS]
    issuers = [dict(公司代號=symbol, 公司簡稱="合成普通股", 公司名稱="合成股份有限公司",
                    營利事業統一編號="12345678", 上市日期="19940905") for symbol in SYMBOLS]
    master = capture_security_master(market="TWSE", instrument_content=synthetic_instruments(rows=rows),
        issuer_content=synthetic_issuers(rows=issuers), fetched_at="2026-10-15T06:00:00Z")
    for block in (value["issuer"], *value["peers"]):
        block["security_evidence"] = select_security_master(master, block["symbol"], CUTOFF)
    return value


def select(data, *, component=True):
    original = data["issuer"]["annual_filings"][-1]
    select_function = validate_comparable_components if component else validate_comparables
    return select_function(data, symbol=SYMBOLS[0], market="TWSE", cutoff=CUTOFF,
        eps_basis="DILUTED", fiscal_year=2025, scope="consolidated", annual_source=original,
        expected_session="2026-10-15", current_price=120.)


def data(context_value=None):
    context_value = ctx() if context_value is None else context_value
    value = item(context_value["issuer"]["annual_filings"][-1])
    value.update(symbol=SYMBOLS[0], cutoff=CUTOFF, policy=policy(),
        input_policy=input_policy(version=INPUT_VERSION), features=dict(close=120.))
    value["bars"]["Date"] = "2026-10-15"
    value["packet"]["r10_valuation_context"] = context_value
    return value


def partial(*, peers=True):
    value = ctx()
    value["issuer"]["annual_filings"] = value["issuer"]["annual_filings"][-1:]
    if not peers:
        missing = value["peers"].pop()
        value["exclusions"].append(dict(symbol=missing["symbol"], reason="SYNTHETIC_MISSING_ORIGINAL"))
    return value


def test_complete_component_selection_keeps_r10_default_output_unchanged():
    value = ctx(); before = deepcopy(value); old = select(value, component=False)
    new = select(value)
    assert old == select(value, component=False)
    assert new["status"] == "VERIFIED"
    assert new["issuer_history"]["count"] == old["issuer_month_count"] == 36
    assert new["peers"]["count"] == old["peer_count"] == 10
    assert new["issuer_history"]["median"] == old["issuer_pe_median"] == 10.
    assert new["peers"]["median"] == old["peer_pe_median"] == 10.
    assert new["issuer_history"]["sources"] == old["monthly_sources"]
    assert new["peers"]["sources"] == old["peer_sources"]
    assert value == before and select(ReadOnlyEvidence(value)) == new


def test_short_issuer_history_retains_authenticated_ten_peer_sources():
    value = partial(); old = select(value, component=False); new = select(value)
    assert old["status"] == "INSUFFICIENT" and "peer_pe_median" not in old
    assert new["status"] == "PARTIAL"
    assert new["issuer_history"]["status"] == "INSUFFICIENT"
    assert new["issuer_history"]["count"] < 24
    assert new["peers"]["status"] == "VERIFIED" and new["peers"]["count"] == 10
    assert new["peers"]["median"] == 10. and len(new["peers"]["sources"]) == 10
    assert all(row["annual"]["available_at"] < CUTOFF for row in new["peers"]["sources"])


def test_verified_official_classification_without_profile_keeps_full_component_sources():
    value = classified_context(); before = deepcopy(value)
    selected = select(value)
    assert selected["status"] == "VERIFIED"
    assert selected["industry"] == "資訊服務業"
    assert selected["industry_source_status"] == "VERIFIED_OFFICIAL_CLASSIFICATION"
    assert selected["industry_profile_status"] == "NOT_DEFINED"
    assert selected["issuer_history"]["count"] == 36 and selected["peers"]["count"] == 10
    assert len(selected["issuer_history"]["sources"]) == 36 and len(selected["peers"]["sources"]) == 10
    # The retained R10/R13 selector keeps its exact serialized old fallback.
    assert select(value, component=False) == dict(status="INSUFFICIENT", reasons=["R10_INDUSTRY_PROFILE_NOT_DEFINED"])
    assert value == before


def test_native_unmapped_official_industry_uses_approved_default_and_keeps_independent_recovery():
    value = data(classified_context()); value["packet"]["filings"] = [recent()]
    result = produce_price_valuation_receipt(**value)
    d = result["valuation_diagnostics"]
    assert result["state"] != "UNKNOWN" and d["selected_method"] == ANNUAL_METHOD
    assert d["comparables"]["industry"] == "資訊服務業"
    assert d["comparables"]["issuer_history"]["count"] == 36 and d["comparables"]["peers"]["count"] == 10
    assert d["recovery"]["status"] == "VERIFIED"
    assert d["observed_growth"]["recent_recovery"] == pytest.approx(.25)
    assert d["scenarios"]["BASE"]["exit_pe"] <= 20.
    assert d["observed_growth"]["bounded_first_year"] <= .10
    assert d["method_small_only"] is False
    assert d["multiple_selection"]["fallback_reason"] == "INDUSTRY_PROFILE_NOT_DEFINED"


@pytest.mark.parametrize("damage", ["issuer_annual", "peer_annual", "peer_quote", "identity", "cutoff", "revision"])
def test_unmapped_policy_profile_never_skips_supplied_source_validation(damage):
    value = classified_context()
    if damage == "issuer_annual": value["issuer"]["annual_filings"][-1]["earnings_source"]["content_sha256"] = "0" * 64
    elif damage == "peer_annual": value["peers"][0]["annual_filings"][-1]["earnings_source"]["content_sha256"] = "0" * 64
    elif damage == "peer_quote": value["peers"][0]["market_rows"][-1]["Close"] = 999.
    elif damage == "identity": value["issuer"]["security_evidence"] = None
    elif damage == "cutoff": value["cutoff"] = "2027-01-01T00:00:00Z"
    else:
        conflict = deepcopy(value["peers"][0]["annual_filings"][-1]); conflict["revision_id"] = "SYNTHETIC_CONFLICT"
        value["peers"][0]["annual_filings"].append(conflict)
    selected = select(value)
    assert selected["status"] == "UNKNOWN"
    assert produce_price_valuation_receipt(**data(value))["state"] == "UNKNOWN"


def test_unmapped_industry_with_conflicting_ytd_source_still_blocks_recovery_and_valuation():
    value = data(classified_context()); row = recent()
    row["comparative_ytd_field_evidence"]["revenue"]["content_sha256"] = "0" * 64
    value["packet"]["filings"] = [row]
    assert produce_price_valuation_receipt(**value)["state"] == "UNKNOWN"


@pytest.mark.parametrize("damage", ["peer_hash", "issuer_hash", "quote_hash", "cutoff", "duplicates", "annual_conflict", "unknown_industry"])
def test_present_source_conflicts_stay_unknown_not_partial_or_fallback(damage):
    value = partial()
    if damage == "peer_hash": value["peers"][0]["annual_filings"][-1]["earnings_source"]["content_sha256"] = "0" * 64
    elif damage == "issuer_hash": value["issuer"]["annual_filings"][-1]["earnings_source"]["content_sha256"] = "0" * 64
    elif damage == "quote_hash": value["issuer"]["market_rows"][-1]["Close"] = 121.
    elif damage == "cutoff": value["cutoff"] = "2027-01-01T00:00:00Z"
    elif damage == "duplicates": value["peers"].append(deepcopy(value["peers"][0]))
    elif damage == "annual_conflict":
        row = deepcopy(value["issuer"]["annual_filings"][-1]); row["revision_id"] = "SYNTHETIC_CONFLICT"
        value["issuer"]["annual_filings"].append(row)
    else: value["issuer"]["security_evidence"]["instrument"]["sector"] = "UNVERIFIED"
    result = select(value)
    assert result["status"] == "UNKNOWN"
    assert result["issuer_history"]["status"] == result["peers"]["status"] == "UNKNOWN"


def test_future_original_is_excluded_from_component_medians_and_selected_proofs():
    value = partial(); before = select(value)
    future = deepcopy(value["issuer"]["annual_filings"][-1]); future.update(
        available_at="2027-01-01T00:00:00Z", revision_id="SYNTHETIC_FUTURE")
    value["issuer"]["annual_filings"].insert(0, future)
    after = select(value)
    assert {k: v for k, v in before.items() if k != "context_id"} == {k: v for k, v in after.items() if k != "context_id"}


def test_native_full_and_peer_only_receipts_are_source_bound_with_distinct_method_sizing():
    full = produce_price_valuation_receipt(**data())
    peer = produce_price_valuation_receipt(**data(partial()))
    assert full["gate_version"] == peer["gate_version"] == COMPONENT_PRODUCER
    assert full["valuation_diagnostics"]["selected_method"] == FULL_METHOD
    diagnostic = peer["valuation_diagnostics"]
    assert diagnostic["selected_method"] == PEER_METHOD
    assert diagnostic["method_small_only"] is True and diagnostic["method_position_multiplier"] == .5
    assert diagnostic["comparables"]["peers"]["sources"]
    assert diagnostic["input_policy_id"] == identity(input_policy(version=INPUT_VERSION))
    assert pd.Timestamp(peer["source_available_at"]) >= pd.Timestamp(diagnostic["comparables"]["available_at"])


def test_native_recovery_remains_effective_when_both_comparable_components_are_short():
    value = data(partial(peers=False)); value["packet"]["filings"] = [recent()]
    result = produce_price_valuation_receipt(**value)
    diagnostic = result["valuation_diagnostics"]
    assert diagnostic["selected_method"] == ANNUAL_METHOD
    assert diagnostic["comparables"]["status"] == "INSUFFICIENT"
    assert diagnostic["recovery"]["status"] == "VERIFIED"
    assert diagnostic["observed_growth"]["recent_recovery"] == pytest.approx(.25)
    assert {row["years"] for row in diagnostic["scenarios"].values()} == {5}


@pytest.mark.parametrize("damage", ["missing_annual", "current_pe", "prior_eps", "companion", "ytd_conflict", "quote_conflict"])
def test_native_r14_does_not_fill_missing_or_conflicting_required_sources(damage):
    value = data(partial())
    if damage == "missing_annual": value["packet"]["valuation_filings"] = []
    elif damage == "current_pe": value["bars"]["PE"] = None
    elif damage == "prior_eps": value["packet"]["valuation_filings"][0]["earnings_source"]["comparative_ytd_field_evidence"].pop("diluted_eps")
    elif damage == "companion": value["packet"]["valuation_filings"][0]["earnings_source"]["content_sha256"] = "0" * 64
    elif damage == "ytd_conflict":
        row = recent(); row["comparative_ytd_field_evidence"]["revenue"]["content_sha256"] = "0" * 64
        value["packet"]["filings"] = [row]
    else: value["features"]["close"] = 121.
    assert produce_price_valuation_receipt(**value)["state"] == "UNKNOWN"


def test_absent_optional_context_retains_explicit_fallback_not_fabricated_issuer_months():
    value = data(); value["packet"].pop("r10_valuation_context")
    result = produce_price_valuation_receipt(**value)["valuation_diagnostics"]
    assert result["selected_method"] == ANNUAL_METHOD
    assert result["comparables"]["issuer_history"]["count"] == 0
    assert result["comparables"]["peers"]["count"] == 0
    assert result["multiple_selection"]["industry"] is None


def test_r14_release_and_exact_policy_identity_are_enforced():
    value = data(); value["cutoff"] = (pd.Timestamp(AVAILABLE_AT) - pd.Timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        produce_price_valuation_receipt(**value)
    value = data(); value["policy"]["base"]["years"] = 1
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        produce_price_valuation_receipt(**value)
    fixed = input_policy(version=INPUT_VERSION)
    assert require_input_policy(fixed, cutoff=CUTOFF) == fixed
