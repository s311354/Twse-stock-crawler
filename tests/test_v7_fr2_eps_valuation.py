"""SYNTHETIC ONLY: EPS research-policy bindings, not market repair or returns."""
from copy import deepcopy

import pandas as pd
import pytest

from application.evidence_context import ReadOnlyEvidence
from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.fr2_eps_valuation import INPUT_SCHEMA, assess_eps_valuation
from domain.les_v7.valuation_eps_receipt import INDEPENDENT_PRODUCER
from domain.les_v7.valuation_price import policy
from tests.test_v7_eps_price_receipts import CUTOFF, source


def case(*, price=30., pe=6., row=None):
    row = source() if row is None else row
    fields = ("symbol", "market", "source", "scope", "currency", "unit", "period_end",
              "flow_basis", "available_at", "revision_id", "content_sha256", "filing_schema",
              "revenue", "operating_income", "net_income", "diluted_shares", "dividends")
    slim = {key: row.get(key) for key in fields}
    slim["field_evidence"] = {key: row.get("field_evidence", {}).get(key, {})
                             for key in ("revenue", "operating_income", "net_income", "diluted_shares", "dividends")}
    dates = pd.bdate_range(end="2026-10-14", periods=60)
    rows = [dict(Date=day.date().isoformat(), Close=price, PE=15.) for day in dates]
    rows[-1]["PE"] = pe
    cutoff = timestamp(CUTOFF).isoformat()
    quality = dict(core_inputs_pass=True, status="PASS", quality_score=80.)
    binding = dict(symbol="SYNTHETIC", market="TWSE", cutoff=cutoff, price=price,
                   quality_result_id=identity(quality), bars_hash=identity(rows))
    packet = dict(schema="les-v7-native-source/v1", symbol="SYNTHETIC", market="TWSE", cutoff=cutoff,
        expected_session="2026-10-14", historical_replay=False, valuation_filings=[slim], market_rows=rows,
        market_evidence=dict(symbol="SYNTHETIC", market="TWSE", session_date="2026-10-14", source="TWSE",
            price_unit="TWD", volume_unit="shares", amount_unit="TWD", source_final=True,
            source_available_at=cutoff, bars_sha256=identity(rows), source_record_ids=[identity(rows)]))
    original = dict(envelope=dict(event=dict(security_id="SYNTHETIC", cutoff_at=cutoff),
            model=dict(bundle_sha256="SYNTHETIC_ONLY_UNCHANGED_FROZEN_MODEL")),
        diagnostics=dict(native_evidence=dict(source_packet=packet), quality=quality,
            features=dict(close=price), reference_plan=dict(SYNTHETIC_ONLY="UNCHANGED"),
            gate_bindings=dict(SAFETY=binding, VALUATION=binding)))
    inputs = dict(schema=INPUT_SCHEMA, original_report_id=identity(original), symbol="SYNTHETIC",
                  market="TWSE", cutoff=cutoff, valuation_filings=[deepcopy(row)])
    return original, inputs


def run(*, price=30., pe=6., row=None):
    return assess_eps_valuation(*case(price=price, pe=pe, row=row))


def test_verified_price_only_without_dividend_or_share_dependencies_keeps_original_unchanged():
    original, inputs = case()
    before = canonical(dict(original=original, inputs=inputs))
    result = assess_eps_valuation(original, inputs)
    assessment, receipt = result["assessment"], result["receipt"]
    assert receipt["state"] == assessment["native_state"] == "PASS"
    assert receipt["gate_version"] == INDEPENDENT_PRODUCER
    assert assessment["return_basis"] == "PRICE_RETURN_ONLY"
    assert assessment["valuation_policy_id"] == identity(policy())
    assert assessment["receipt_id"] == identity(receipt)
    assert assessment["scenario_id"] == identity(assessment["snapshot"])
    assert assessment["selected_annual_ids"] == [identity(inputs["valuation_filings"][0])]
    assumptions = assessment["snapshot"]["assumptions"]
    assert assumptions["dividends"] == "EXCLUDED_NOT_IMPUTED"
    assert assumptions["dilution"] == "NOT_SEPARATELY_MODELED_NO_OBSERVED_ZERO_CLAIM"
    assert inputs["valuation_filings"][0].get("dividends") is None
    assert inputs["valuation_filings"][0].get("diluted_shares") is None
    assert canonical(dict(original=original, inputs=inputs)) == before
    result["assessment"]["snapshot"]["inputs"]["current_eps"] = 999.
    assert assess_eps_valuation(original, inputs)["assessment"]["snapshot"]["inputs"]["current_eps"] == 12.


def test_readonly_source_graph_is_supported_without_mutating_or_thawing_the_report():
    original, inputs = case()
    expected = assess_eps_valuation(original, inputs)
    assert assess_eps_valuation(ReadOnlyEvidence(original), ReadOnlyEvidence(inputs)) == expected


def test_complete_base_only_failed_checks_allow_warning_but_no_bear_relaxation():
    annual = source(current_eps=9., prior_eps=10., money_overrides={("revenue", 2025): 90_000_000.})
    result = run(price=50., pe=5., row=annual)
    assessment = result["assessment"]
    assert assessment["native_state"] == "FAIL" and assessment["warning_eligible"] is True
    assert set(assessment["reason_codes"]) == {"BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE"}
    assert assessment["snapshot"]["gate_checks"]["BEAR_LOSS_LTE_MAXIMUM"] is True
    assert assessment["snapshot"]["bear_permanent_loss"] <= .35
    failed = run(price=200., pe=20.)["assessment"]
    assert failed["native_state"] == "FAIL" and failed["warning_eligible"] is False
    assert "BEAR_LOSS_LTE_MAXIMUM" in failed["reason_codes"]


@pytest.mark.parametrize("eps", [0., -1.])
def test_verified_nonpositive_eps_is_financial_failure_not_missing_or_warning(eps):
    result = run(row=source(current_eps=eps))["assessment"]
    assert result["native_state"] == "FAIL" and result["warning_eligible"] is False
    assert result["reason_codes"] == ["NON_POSITIVE_CURRENT_EPS"]


@pytest.mark.parametrize("damage", ["earnings_missing", "comparative_missing", "eps_proof_hash",
    "eps_unit", "annual_money_comparison", "future_comparison", "source_companion"])
def test_missing_or_conflicting_necessary_earnings_proof_stays_unknown(damage):
    original, inputs = case()
    row = inputs["valuation_filings"][0]
    if damage == "earnings_missing": row.pop("earnings_source")
    elif damage == "comparative_missing": row["earnings_source"]["comparative_ytd_fields"].pop("diluted_eps")
    elif damage == "eps_proof_hash": row["earnings_source"]["field_evidence"]["diluted_eps"]["content_sha256"] = "0" * 64
    elif damage == "eps_unit": row["earnings_source"]["field_evidence"]["diluted_eps"]["contexts"][0]["unit"] = "iso4217:TWD"
    elif damage == "annual_money_comparison": row["comparative_ytd_field_evidence"]["revenue"]["content_sha256"] = "0" * 64
    elif damage == "future_comparison": row["earnings_source"]["comparative_ytd_field_evidence"]["diluted_eps"]["available_at"] = "2027-01-01T00:00:00Z"
    else: row["earnings_source"]["scope"] = "individual"
    assessment = assess_eps_valuation(original, inputs)["assessment"]
    assert assessment["native_state"] == "UNKNOWN" and assessment["warning_eligible"] is False


@pytest.mark.parametrize("field,value", [("schema", "OTHER"), ("original_report_id", "0" * 64),
    ("symbol", "OTHER"), ("market", "TPEX"), ("cutoff", "2026-10-15T13:30:00+08:00")])
def test_supplement_cannot_change_report_event_or_schema(field, value):
    original, inputs = case()
    inputs[field] = value
    with pytest.raises(ValueError, match="FR2_EPS_"):
        assess_eps_valuation(original, inputs)


@pytest.mark.parametrize("field,value", [("symbol", "OTHER"), ("scope", "individual"),
    ("content_sha256", "0" * 64), ("revision_id", "OTHER"), ("unit", "TWD"),
    ("available_at", "2026-03-11T05:00:00Z"), ("period_end", "2024-12-31"),
    ("source_url", "https://mopsov.twse.com.tw/OTHER_ORIGINAL.html")])
def test_existing_original_identity_time_scope_and_source_cannot_be_substituted(field, value):
    original, inputs = case()
    inputs["valuation_filings"][0][field] = value
    with pytest.raises(ValueError, match="FR2_EPS_"):
        assess_eps_valuation(original, inputs)


@pytest.mark.parametrize("damage", ["amount", "proof", "omitted_latest"])
def test_original_financial_values_proofs_and_latest_version_are_not_discarded(damage):
    original, inputs = case()
    row = inputs["valuation_filings"][0]
    if damage == "amount": row["revenue"] += 1.
    elif damage == "proof": row["field_evidence"]["operating_income"]["contexts"][0]["numeric_value"] += 1.
    else: inputs["valuation_filings"] = []
    with pytest.raises(ValueError, match="FR2_EPS_"):
        assess_eps_valuation(original, inputs)


def test_adding_future_revision_cannot_affect_cutoff_selected_economics_or_receipt():
    original, inputs = case()
    before = assess_eps_valuation(original, inputs)
    future = deepcopy(inputs["valuation_filings"][0])
    future.update(available_at="2027-01-01T00:00:00Z", revision_id="FUTURE_OTHER_VERSION")
    future["earnings_source"]["fields"]["diluted_eps"] = 999.
    inputs["valuation_filings"].append(future)
    after = assess_eps_valuation(original, inputs)
    assert after["receipt"] == before["receipt"]
    for key in before["assessment"]:
        if key != "valuation_inputs_id":
            assert after["assessment"][key] == before["assessment"][key]


@pytest.mark.parametrize("damage", ["future", "hash", "missing", "current_pe"])
def test_original_missing_or_invalid_quote_remains_unknown_not_positive_pe_fallback(damage):
    original, inputs = case()
    packet = original["diagnostics"]["native_evidence"]["source_packet"]
    if damage == "future": packet["market_evidence"]["source_available_at"] = "2027-01-01T00:00:00Z"
    elif damage == "hash": packet["market_evidence"]["bars_sha256"] = "0" * 64
    elif damage == "missing": packet["market_evidence"] = {}
    else:
        packet["market_rows"][-1]["PE"] = None
        packet["market_evidence"].update(bars_sha256=identity(packet["market_rows"]), source_record_ids=[identity(packet["market_rows"])])
    inputs["original_report_id"] = identity(original)
    assessment = assess_eps_valuation(original, inputs)["assessment"]
    assert assessment["native_state"] == "UNKNOWN" and assessment["warning_eligible"] is False


def test_valuation_does_not_turn_missing_quality_into_source_failure_or_entry_authority():
    original, inputs = case()
    original["diagnostics"]["quality"].update(status="DATA_INSUFFICIENT", core_inputs_pass=False)
    inputs["original_report_id"] = identity(original)
    before = deepcopy(original)
    result = assess_eps_valuation(original, inputs)
    assert result["assessment"]["native_state"] == "PASS"
    assert result["assessment"]["snapshot"]["entry_quality_required"] is True
    assert original == before
    assert "authorized" not in canonical(result)


def test_same_time_conflicting_eps_companion_cannot_be_selected_away():
    original, inputs = case()
    conflict = deepcopy(inputs["valuation_filings"][0])
    conflict["earnings_source"]["fields"]["diluted_eps"] = 999.
    inputs["valuation_filings"].append(conflict)
    assessment = assess_eps_valuation(original, inputs)["assessment"]
    assert assessment["native_state"] == "UNKNOWN" and assessment["warning_eligible"] is False
    assert assessment["reason_codes"] == ["CONFLICTING_ANNUAL_EPS_VERSIONS"]


@pytest.mark.parametrize("field,value", [("available_at", "UNVERIFIED_TIME"), ("period_end", "UNVERIFIED_PERIOD")])
def test_identical_malformed_original_provenance_is_unknown_not_silently_discarded(field, value):
    original, inputs = case()
    original["diagnostics"]["native_evidence"]["source_packet"]["valuation_filings"][0][field] = value
    inputs["valuation_filings"][0][field] = value
    inputs["original_report_id"] = identity(original)
    assessment = assess_eps_valuation(original, inputs)["assessment"]
    assert assessment["native_state"] == "UNKNOWN" and assessment["warning_eligible"] is False
    assert assessment["reason_codes"] == ["ANNUAL_EPS_AVAILABILITY_UNVERIFIED"]


def test_no_originals_is_unknown_not_fabricated_annual_or_constant_eps():
    original, inputs = case()
    original["diagnostics"]["native_evidence"]["source_packet"]["valuation_filings"] = []
    inputs["valuation_filings"] = []
    inputs["original_report_id"] = identity(original)
    assessment = assess_eps_valuation(original, inputs)["assessment"]
    assert assessment["native_state"] == "UNKNOWN" and assessment["selected_annual_ids"] == []
    assert assessment["reason_codes"] == ["ANNUAL_EPS_ORIGINAL_MISSING"]


def test_source_inputs_cannot_backdate_the_new_price_policy():
    original, inputs = case()
    early = "2026-10-13T05:30:00Z"
    original["envelope"]["event"]["cutoff_at"] = early
    for binding in original["diagnostics"]["gate_bindings"].values(): binding["cutoff"] = early
    original["diagnostics"]["native_evidence"]["source_packet"]["cutoff"] = early
    inputs.update(cutoff=early, original_report_id=identity(original))
    with pytest.raises(ValueError, match="VALUATION_NOT_AVAILABLE_AT_CUTOFF"):
        assess_eps_valuation(original, inputs)


@pytest.mark.parametrize("comparative", [False, True])
@pytest.mark.parametrize("damage", ["proof_url", "proof_revision", "companion_revision", "missing_revision"])
def test_selected_eps_proof_url_and_revision_number_must_bind_the_same_original(comparative, damage):
    original, inputs = case()
    annual = inputs["valuation_filings"][0]
    companion = annual["earnings_source"]
    proof = companion["comparative_ytd_field_evidence" if comparative else "field_evidence"]["diluted_eps"]
    if damage == "proof_url": proof["source_url"] = "https://mopsov.twse.com.tw/OTHER.html"
    elif damage == "proof_revision": proof["revision_number"] += 1
    elif damage == "companion_revision": companion["revision_number"] += 1
    else: annual.pop("revision_number")
    result = assess_eps_valuation(original, inputs)
    assert result["assessment"]["native_state"] == "UNKNOWN"
    assert result["assessment"]["warning_eligible"] is False
    assert "UNVERIFIED" in canonical(result["assessment"]["reason_codes"])


def test_unselected_basic_eps_metadata_does_not_override_verified_diluted_method():
    original, inputs = case()
    inputs["valuation_filings"][0]["earnings_source"]["field_evidence"]["basic_eps"]["source_url"] = "OTHER"
    assert assess_eps_valuation(original, inputs)["assessment"]["native_state"] == "PASS"


@pytest.mark.parametrize("companion", [False, True])
def test_missing_projected_top_source_url_is_unknown_not_a_conflicting_source(companion):
    original, inputs = case()
    row = inputs["valuation_filings"][0]
    row.pop("source_url")
    if not companion: row.pop("earnings_source")
    assessed = assess_eps_valuation(original, inputs)["assessment"]
    assert assessed["native_state"] == "UNKNOWN" and assessed["warning_eligible"] is False
    assert assessed["reason_codes"] == ["ANNUAL_EPS_SOURCE_OR_SCOPE_UNVERIFIED"]


def test_original_wrong_annual_issuer_is_not_rewritten_as_the_outer_security():
    original, inputs = case()
    packet = original["diagnostics"]["native_evidence"]["source_packet"]
    # Model the retained unsupported-class mapping exactly: the outer target
    # and its cutoff are valid, but the bound annual document is another issuer.
    original["envelope"]["event"]["security_id"] = "8112A"
    packet["symbol"] = packet["market_evidence"]["symbol"] = "8112A"
    for binding in original["diagnostics"]["gate_bindings"].values(): binding["symbol"] = "8112A"
    packet["valuation_filings"][0]["symbol"] = "8112"
    inputs["valuation_filings"][0]["symbol"] = "8112"
    inputs["valuation_filings"][0].pop("source_url")
    inputs["valuation_filings"][0].pop("earnings_source")
    inputs.update(symbol="8112A", original_report_id=identity(original))
    before = canonical(dict(original=original, supplement=inputs))
    assessed = assess_eps_valuation(original, inputs)["assessment"]
    assert assessed["native_state"] == "UNKNOWN" and assessed["warning_eligible"] is False
    assert assessed["reason_codes"] == ["ANNUAL_EPS_SOURCE_OR_SCOPE_UNVERIFIED"]
    assert canonical(dict(original=original, supplement=inputs)) == before
