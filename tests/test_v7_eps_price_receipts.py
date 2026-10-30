"""SYNTHETIC ONLY: exact EPS-source gates, not real entry or performance."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from domain.fundamental_derivation import FiledFact
from domain.les_v7.contracts import identity
from domain.les_v7.valuation_earnings import valuation_earnings
from domain.les_v7.valuation_eps_receipt import produce_price_valuation_receipt, PRODUCER, _eps, _amount
from domain.les_v7.valuation_price import policy
from domain.strategies.low_entry_v6_filings import ixbrl_record
from tests.fundamentals.test_v7_valuation_earnings_sources import original, EPS_UNIT


CUTOFF = "2026-10-14T13:30:00+08:00"


def source(*, basic_only=False, current_eps=12., prior_eps=10., money_overrides=None):
    base = original()
    facts = [replace(f, numeric_value=current_eps if f.end_date == "2025-12-31" else prior_eps)
             for f in base.facts if not basic_only or "Diluted" not in f.concept]
    for year, revenue in ((2025, 120000000.), (2024, 100000000.)):
        for field, concept, value in (("revenue", "ifrs-full:Revenue", revenue),
                ("operating_income", "ifrs-full:ProfitLossFromOperatingActivities", revenue * .3)):
            value = (money_overrides or {}).get((field, year), value)
            facts.append(FiledFact(concept, value, None, f"{year}-01-01", f"{year}-12-31", None,
                unit="iso4217:TWD", context_id="SYNTHETIC_ANNUAL_MONEY"))
    filing = replace(base, facts=tuple(facts), content_sha256=identity(dict(synthetic_only=True, facts=[str(f) for f in facts])))
    row = dict(ixbrl_record("SYNTHETIC", filing), market="TWSE")
    row["earnings_source"] = valuation_earnings(filing, symbol="SYNTHETIC", original_verified=True, as_of=CUTOFF)
    return row


def item(row=None):
    bars = pd.DataFrame(dict(PE=[15.] * 59 + [6.]))
    return dict(packet=dict(valuation_filings=[source() if row is None else row]),
        symbol="SYNTHETIC", market="TWSE", cutoff=CUTOFF, features=dict(close=30.),
        quality=dict(core_inputs_pass=True), bars=bars, binding=dict(SYNTHETIC_ONLY=True),
        market_source=dict(source_available_at=CUTOFF, source="TWSE"), policy=policy())


def evaluate(data):
    return produce_price_valuation_receipt(**data)


def test_reported_eps_without_dividends_or_shares_is_evaluated_with_its_distinct_method():
    data = item(); before = deepcopy(data["packet"])
    result = evaluate(data)
    assert result["state"] == "PASS" and result["gate_version"] == PRODUCER
    diagnostic = result["valuation_diagnostics"]
    assert diagnostic["inputs"]["current_eps"] == 12.
    assert diagnostic["inputs"]["prior_eps"] == 10.
    assert diagnostic["assumptions"]["dividends"] == "EXCLUDED_NOT_IMPUTED"
    assert diagnostic["assumptions"]["dilution"] == "NOT_SEPARATELY_MODELED_NO_OBSERVED_ZERO_CLAIM"
    assert data["packet"] == before
    assert data["packet"]["valuation_filings"][0].get("dividends") is None
    assert data["packet"]["valuation_filings"][0].get("diluted_shares") is None


def test_genuinely_unreported_diluted_eps_uses_explicit_basic_haircut_not_renamed_eps():
    result = evaluate(item(source(basic_only=True)))
    diagnostic = result["valuation_diagnostics"]
    assert result["state"] == "PASS"
    assert diagnostic["inputs"]["eps_basis"] == "BASIC"
    assert diagnostic["assumptions"]["basic_eps_haircut"] == .9
    assert diagnostic["anchor"]["scenario_anchor"] == pytest.approx(4.5)


def test_missing_reason_cannot_override_retained_reported_diluted_eps_contexts():
    data = item()
    earnings = data["packet"]["valuation_filings"][0]["earnings_source"]
    for values, proofs in (("fields", "field_evidence"),
                           ("comparative_ytd_fields", "comparative_ytd_field_evidence")):
        earnings[values]["diluted_eps"] = None
        earnings[proofs]["diluted_eps"]["reason"] = "TOTAL_EPS_CONCEPT_NOT_REPORTED"
    result = evaluate(data)
    assert result["state"] == "UNKNOWN"
    assert result["reason_codes"] == ["DILUTED_EPS_CURRENT_OR_COMPARATIVE_UNVERIFIED"]


@pytest.mark.parametrize("comparative", [False, True])
@pytest.mark.parametrize("damage", ["status", "category", "mapping", "basis", "unit",
    "contexts", "source_url", "scope", "revision_number", "availability", "value", "missing_field", "missing_binding"])
def test_basic_fallback_requires_complete_same_original_nonreporting_proofs(comparative, damage):
    data = item(source(basic_only=True))
    earnings = data["packet"]["valuation_filings"][0]["earnings_source"]
    values_key, proofs_key = (("comparative_ytd_fields", "comparative_ytd_field_evidence")
                             if comparative else ("fields", "field_evidence"))
    proof = earnings[proofs_key]["diluted_eps"]
    if damage == "status": proof["status"] = "RESOLVED"
    elif damage == "category": proof["category"] = "CONFLICT"
    elif damage == "mapping": proof["mapping_version"] = "UNVERIFIED_MAPPING"
    elif damage == "basis": proof["basis"] = "YTD"
    elif damage == "unit": proof["unit"] = "thousand"
    elif damage == "contexts": proof["contexts"] = [dict(numeric_value=999.)]
    elif damage == "source_url": proof["source_url"] = "https://mopsov.twse.com.tw/OTHER_ORIGINAL.html"
    elif damage == "scope": proof["scope"] = "individual"
    elif damage == "revision_number": proof["revision_number"] = 999
    elif damage == "availability": proof["available_at"] = "2027-01-01T00:00:00Z"
    elif damage == "value": earnings[values_key]["diluted_eps"] = "not-a-number"
    elif damage == "missing_field": earnings[values_key].pop("diluted_eps")
    else:
        earnings.pop("revision_number")
        proof.pop("revision_number")
    result = evaluate(data)
    assert result["state"] == "UNKNOWN"
    assert result["reason_codes"] == ["DILUTED_EPS_CURRENT_OR_COMPARATIVE_UNVERIFIED"]


@pytest.mark.parametrize("damage", ["unit", "qname", "dimension", "value", "hash", "scope", "current_time", "prior_time"])
def test_unverified_diluted_operand_cannot_be_replaced_by_better_basic_eps(damage):
    data = item(); row = data["packet"]["valuation_filings"][0]
    earnings = row["earnings_source"]
    proof = earnings["field_evidence"]["diluted_eps"]
    if damage == "unit": proof["contexts"][0]["unit"] = "iso4217:TWD"
    elif damage == "qname": proof["contexts"][0]["concept"] += "FromContinuingOperations"
    elif damage == "dimension": proof["contexts"][0]["dimensions"] = [["SYNTHETIC", "SUBSIDIARY"]]
    elif damage == "value": earnings["fields"]["diluted_eps"] = 100.
    elif damage == "hash": proof["content_sha256"] = "0" * 64
    elif damage == "scope": proof["scope"] = "individual"
    elif damage == "current_time": proof["available_at"] = "2026-11-01T00:00:00Z"
    else: earnings["comparative_ytd_field_evidence"]["diluted_eps"]["available_at"] = "2025-03-01T00:00:00Z"
    result = evaluate(data)
    assert result["state"] == "UNKNOWN"
    assert "DILUTED_EPS_CURRENT_OR_COMPARATIVE_UNVERIFIED" in result["reason_codes"]


@pytest.mark.parametrize("group", ["comparative_ytd_fields", "comparative_ytd_field_evidence"])
def test_missing_annual_comparison_is_not_filled_from_another_original(group):
    data = item(); row = data["packet"]["valuation_filings"][0]
    row["earnings_source"][group].pop("diluted_eps")
    assert evaluate(data)["state"] == "UNKNOWN"


@pytest.mark.parametrize("field", ["revenue", "operating_income"])
def test_money_comparison_must_belong_to_the_same_annual_original(field):
    data = item(); row = data["packet"]["valuation_filings"][0]
    row["comparative_ytd_field_evidence"][field]["content_sha256"] = "0" * 64
    assert evaluate(data)["state"] == "UNKNOWN"


@pytest.mark.parametrize("current,prior", [(-1., 10.), (12., 0.), (0., 10.)])
def test_known_nonpositive_eps_is_financial_fail_not_missing_data(current, prior):
    result = evaluate(item(source(current_eps=current, prior_eps=prior)))
    assert result["state"] == "FAIL"
    assert any(reason.startswith("NON_POSITIVE_") for reason in result["reason_codes"])


def test_verified_negative_eps_is_not_relabelled_unknown_when_quote_or_quality_is_absent():
    data = item(source(current_eps=-1.))
    data["bars"]["PE"] = None
    data["quality"]["core_inputs_pass"] = False
    result = evaluate(data)
    assert result["state"] == "FAIL"
    assert "NON_POSITIVE_CURRENT_EPS" in result["reason_codes"]
    assert result["valuation_diagnostics"]["observed_inputs"]["current_eps"] == -1.


@pytest.mark.parametrize("comparative", [False, True])
@pytest.mark.parametrize("retained", [1e-10, -1e-10])
def test_eps_rounding_tolerance_cannot_change_verified_zero_to_positive_or_negative(comparative, retained):
    data = item(source(**{"prior_eps" if comparative else "current_eps": 0.}))
    earnings = data["packet"]["valuation_filings"][0]["earnings_source"]
    values_key = "comparative_ytd_fields" if comparative else "fields"
    earnings[values_key]["diluted_eps"] = retained
    original, _ = _eps(earnings, "diluted_eps", 2024 if comparative else 2025, comparative=comparative)
    assert original == 0.
    result = evaluate(data)
    assert result["state"] == "FAIL"
    key = "PRIOR_EPS" if comparative else "CURRENT_EPS"
    assert result["reason_codes"] == ["NON_POSITIVE_" + key]
    assert result["valuation_diagnostics"]["observed_inputs"][key.lower()] == 0.


@pytest.mark.parametrize("field", ["revenue", "operating_income"])
@pytest.mark.parametrize("comparative", [False, True])
@pytest.mark.parametrize("retained", [.1, -.1])
def test_money_rounding_tolerance_cannot_replace_verified_zero_with_retained_amount(field, comparative, retained):
    year = 2024 if comparative else 2025
    data = item(source(money_overrides={(field, year): 0.}))
    row = data["packet"]["valuation_filings"][0]
    if comparative: row["comparative_ytd_fields"][field] = retained
    else: row[field] = retained
    original, _ = _amount(row, field, pd.Period("2025Q4"), CUTOFF, comparative=comparative)
    assert original == 0.
    result = evaluate(data)
    assert result["state"] == "FAIL"
    key = ("PRIOR_" if comparative else "CURRENT_") + field.upper()
    assert result["reason_codes"] == ["NON_POSITIVE_REPORTED_" + key]


@pytest.mark.parametrize("comparative", [False, True])
@pytest.mark.parametrize("retained", [0., 1e-10, -1e-10])
def test_tiny_positive_original_eps_is_not_rounded_to_zero_by_retained_value(comparative, retained):
    exact = 1e-11
    data = item(source(**{"prior_eps" if comparative else "current_eps": exact}))
    earnings = data["packet"]["valuation_filings"][0]["earnings_source"]
    values_key = "comparative_ytd_fields" if comparative else "fields"
    earnings[values_key]["diluted_eps"] = retained
    original, _ = _eps(earnings, "diluted_eps", 2024 if comparative else 2025, comparative=comparative)
    assert original == exact and original > 0
    assert not any(reason.startswith("NON_POSITIVE_") for reason in evaluate(data)["reason_codes"])


@pytest.mark.parametrize("field", ["revenue", "operating_income"])
@pytest.mark.parametrize("comparative", [False, True])
@pytest.mark.parametrize("retained", [0., .1, -.1])
def test_tiny_positive_original_money_is_not_rounded_to_zero_by_retained_value(field, comparative, retained):
    year = 2024 if comparative else 2025
    data = item(source(money_overrides={(field, year): .001}))  # TWD, not TWD thousands.
    row = data["packet"]["valuation_filings"][0]
    if comparative: row["comparative_ytd_fields"][field] = retained
    else: row[field] = retained
    original, _ = _amount(row, field, pd.Period("2025Q4"), CUTOFF, comparative=comparative)
    assert original == .000001 and original > 0


def test_newer_invalid_revision_is_not_ignored_and_future_revision_is_excluded():
    data = item(); before = evaluate(data)
    future = deepcopy(data["packet"]["valuation_filings"][0])
    future.update(available_at="2026-11-01T00:00:00Z", revision_id="SYNTHETIC_FUTURE")
    data["packet"]["valuation_filings"].append(future)
    assert evaluate(data) == before
    future["available_at"] = "2026-10-14T01:00:00Z"
    assert evaluate(data)["state"] == "UNKNOWN"  # no same-source companion for the newer version


def test_same_time_conflicting_originals_remain_unknown():
    data = item(); conflict = deepcopy(data["packet"]["valuation_filings"][0])
    conflict["revision_id"] = "SYNTHETIC_CONFLICT"
    data["packet"]["valuation_filings"].append(conflict)
    assert "CONFLICTING_ANNUAL_EPS_VERSIONS" in evaluate(data)["reason_codes"]


@pytest.mark.parametrize("value", [None, True, 0., float("nan")])
def test_current_quote_pe_missing_is_not_replaced_by_old_positive_pe(value):
    data = item(); data["bars"] = pd.DataFrame(dict(PE=[15.] * 59 + [value]))
    assert evaluate(data)["state"] == "UNKNOWN"


def test_latest_annual_missing_earnings_source_does_not_fall_back_to_old_year():
    data = item(); data["packet"]["valuation_filings"][0].pop("earnings_source")
    assert "ANNUAL_EPS_SOURCE_COMPANION_UNVERIFIED" in evaluate(data)["reason_codes"]


def test_old_annual_and_unverified_quality_inputs_remain_unknown():
    data = item(); data["cutoff"] = "2028-10-14T13:30:00+08:00"
    assert "ANNUAL_EPS_PERIOD_INCOMPLETE_OR_STALE" in evaluate(data)["reason_codes"]
    data = item(); data["quality"]["core_inputs_pass"] = False
    assert evaluate(data)["state"] == "UNKNOWN"


def test_unknown_backdated_or_modified_policy_cannot_be_executed():
    data = item(); data["policy"]["required_return"] = 0.
    with pytest.raises(ValueError, match="VALUATION_POLICY_MISMATCH"):
        evaluate(data)
    data = item(); data["cutoff"] = "2026-10-13T23:46:41+08:00"
    with pytest.raises(ValueError, match="VALUATION_NOT_AVAILABLE_AT_CUTOFF"):
        evaluate(data)
