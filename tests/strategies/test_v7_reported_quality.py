"""SYNTHETIC ONLY: R3/native equivalence, never repaired official evidence."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from application.evidence_context import ReadOnlyEvidence
from domain.fundamental_derivation import FiledFact
from domain.les_v7.contracts import identity
from domain.les_v7.financial_sources import ixbrl_record_r2
from domain.les_v7.quality_reported import (
    PROFILE, evaluate_financials_reported, validate_reported_result,
)
from domain.strategies.low_entry_v6_quality import evaluate_quality, QUALITY_BUDGETS
from tests.strategies.test_low_entry_v6_comparative_ttm import originals, CUTOFF


def source_records(*, opinion="UnqualifiedOpinionAbstract", negative_cfo=False, negative_profit=False,
                   zero_revenue=False, zero_current_liabilities=False):
    """Typed original facts, exact comparative contexts and a real cash bridge."""
    _, filings = originals()
    records = []
    for filing in filings:
        end, start = filing.period_end, f"{filing.fiscal_year}-01-01"
        facts = list(filing.facts)
        cfo = next(f.numeric_value for f in facts
                   if f.concept == "ifrs-full:CashFlowsFromUsedInOperatingActivities" and f.end_date == end)
        if negative_cfo:
            cfo = -abs(cfo)
            facts = [replace(f, numeric_value=cfo) if f.concept == "ifrs-full:CashFlowsFromUsedInOperatingActivities"
                     and f.end_date == end else f for f in facts]
        if negative_profit:
            facts = [replace(f, numeric_value=-abs(f.numeric_value)) if f.concept == "ifrs-full:ProfitLoss"
                     and f.end_date == end else f for f in facts]
        if zero_revenue:
            facts = [replace(f, numeric_value=0.) if f.concept == "ifrs-full:Revenue"
                     and f.end_date == end else f for f in facts]
        if zero_current_liabilities:
            facts = [replace(f, numeric_value=0.) if f.concept == "ifrs-full:CurrentLiabilities"
                     and f.instant == end else f for f in facts]
        for concept, value in (
            ("tifrs-SCF:NetCashFlowsFromUsedInInvestingActivities", -cfo),
            ("tifrs-SCF:CashFlowsFromUsedInFinancingActivities", 0.),
            ("ifrs-full:EffectOfExchangeRateChangesOnCashAndCashEquivalents", 0.),
            ("ifrs-full:IncreaseDecreaseInCashAndCashEquivalents", 0.),
        ):
            facts.append(FiledFact(concept, value, None, start, end, None,
                unit="iso4217:TWD", context_id="SYNTHETIC_ONLY_CURRENT_CASH_FLOW"))
        for concept, instant in (
            ("tifrs-SCF:CashAndCashEquivalentsAtBeginningOfPeriod", end),
            ("tifrs-SCF:CashAndCashEquivalentsAtEndOfPeriod", end),
        ):
            facts.append(FiledFact(concept, 1000_000., None, None, None, instant,
                unit="iso4217:TWD", context_id="SYNTHETIC_ONLY_CURRENT_CASH_ANCHOR"))
        facts.append(FiledFact("tifrs-ar:" + opinion, None, "Y", None, end, None,
            context_id="SYNTHETIC_ONLY_AUDIT"))
        filing = replace(filing, facts=tuple(facts),
            content_sha256=identity(dict(synthetic_only=True, facts=[str(f) for f in facts])),
            source_url="https://mops.twse.com.tw/SYNTHETIC_ONLY_R3",
            taxonomy_schema_refs=("https://mops.twse.com.tw/tifrs-ci-cr-2026-03-31.xsd",))
        records.append(ixbrl_record_r2("2330", filing, original_verified=True))
    return records


def evaluate(records, *, cutoff=CUTOFF, policy=None, safety_receipt=None):
    from domain.les_v7.quality_policy import r3_quality_policy
    return evaluate_financials_reported(records, symbol="2330", market="TWSE", cutoff=cutoff,
        policy=policy or r3_quality_policy(), safety_receipt=safety_receipt)


def core(result):
    return (result["status"], result["quality_score"], result["data_confidence"],
            result["coverage_total"], result["coverage_by_domain"], result["metric_ledger"],
            result["reported_result"]["metrics"]["quality_verification"])


def test_r3_reuses_native_reported_quality_fixed_budget_and_dc():
    rows = source_records()
    before = deepcopy(rows)
    result = evaluate(rows)
    native = evaluate_quality(rows, symbol="2330", market="TWSE", as_of=CUTOFF)
    assert result["profile_id"] == PROFILE and result["mapping_version"] == "QG_MAP_R3"
    assert result["status"] == native.status == "PASS"
    assert result["quality_score"] == native.score and result["data_confidence"] == native.confidence
    assert result["reported_result"]["components"] == native.components
    assert result["coverage_by_domain"] == {key: value / 100. for key, value in native.coverage.items()}
    assert sum(item["budget"] for item in result["metric_ledger"]) == pytest.approx(100)
    assert result["reported_result"]["metrics"]["quality_budgets"] == QUALITY_BUDGETS
    assert result["core_reported_pass"] and result["core_inputs_pass"] and result["native_quality_complete"]
    assert result["scoring_basis"] == "REPORTED_SAME_PERIOD"
    assert rows == before
    validate_reported_result(result)
    json.dumps(result, allow_nan=False)


def test_one_current_original_is_sufficient_without_fabricating_ttm():
    rows = source_records()
    full, one = evaluate(rows), evaluate(rows[-1:])
    assert core(one) == core(full)
    assert one["status"] == "PASS" and one["core_reported_pass"] and one["core_inputs_pass"]
    assert not one["core_ttm_pass"] and all(value is None for value in one["core_ttm"].values())
    assert one["missing_quarters"]  # diagnostic gap, not a manufactured quarter
    validate_reported_result(one)


def test_known_ttm_accounting_review_does_not_change_reported_core():
    rows = source_records()
    baseline = evaluate(rows)
    rows[-1]["accounting_basis_evidence"].update(status="UNKNOWN", reason="ACCOUNTING_POLICY_EXCEPTION_REQUIRES_REVIEW")
    changed = evaluate(rows)
    assert core(changed) == core(baseline)
    assert changed["core_ttm"]["cfo"] is None
    validate_reported_result(changed)


@pytest.mark.parametrize("field", ["interest_bearing_debt", "income_tax", "pretax_income", "interest_expense", "parent_equity", "parent_net_income", "intangible_capex"])
def test_extension_source_missing_never_contaminates_native_core(field):
    rows = source_records()
    baseline = evaluate(rows)
    rows[-1][field] = None
    rows[-1]["field_evidence"].pop(field, None)
    result = evaluate(rows)
    assert core(result) == core(baseline)
    assert result["core_inputs_pass"]
    validate_reported_result(result)


@pytest.mark.parametrize("field", ["revenue", "cost_of_revenue", "operating_income", "net_income", "cfo"])
def test_missing_required_reported_flow_never_passes(field):
    rows = source_records()
    rows[-1][field] = None
    result = evaluate(rows)
    assert result["status"] == "DATA_INSUFFICIENT" and not result["core_reported_pass"]
    assert result["core_reported"][field] is None
    validate_reported_result(result)


@pytest.mark.parametrize("field", ["total_assets", "current_assets", "non_current_assets", "total_liabilities", "current_liabilities", "total_equity"])
def test_missing_required_balance_never_passes(field):
    rows = source_records()
    rows[-1][field] = None
    result = evaluate(rows)
    assert result["status"] == "DATA_INSUFFICIENT"
    assert not result["core_inputs_pass"]
    validate_reported_result(result)


@pytest.mark.parametrize("key,value", [("content_sha256", "wrong"), ("revision_id", "other"),
    ("unit", "dollar"), ("scope", "individual"), ("basis", "QUARTER"), ("start_date", "2025-04-01")])
def test_comparative_proof_failure_never_falls_back_to_other_original(key, value):
    rows = source_records()
    rows[-1]["comparative_ytd_field_evidence"]["revenue"][key] = value
    result = evaluate(rows)
    assert result["status"] == "DATA_INSUFFICIENT"
    assert result["reported_inputs"]["comparative"]["revenue"] is None
    assert result["core_reported_pass"] and result["core_inputs_pass"]
    assert not result["native_quality_complete"]
    assert result["coverage_by_domain"]["profitability"] == pytest.approx(1/3)
    validate_reported_result(result)


def test_future_original_and_revision_cannot_change_earlier_cutoff():
    rows = source_records()
    expected = evaluate(rows)
    future = deepcopy(rows[-1])
    future.update(available_at="2026-10-03T00:00:00Z", revision_id="SYNTHETIC_ONLY_FUTURE", cfo=10**12)
    assert evaluate(rows + [future]) == expected


def test_known_nonpositive_cfo_is_observed_bad_quality_not_missing_data():
    result = evaluate(source_records(negative_cfo=True))
    assert result["core_reported_pass"] and result["core_inputs_pass"]
    assert result["status"] != "DATA_INSUFFICIENT"
    assert result["coverage_total"] == 1.
    cfo = next(item for item in result["metric_ledger"] if item["metric_id"] == "cfo_positive")
    assert cfo["available"] and cfo["earned_points"] == 0
    validate_reported_result(result)


def test_known_nonpositive_profit_ratio_is_not_meaningful_zero_not_missing():
    result = evaluate(source_records(negative_profit=True))
    conversion = next(item for item in result["metric_ledger"] if item["metric_id"] == "cash_conversion")
    assert conversion["available"] and conversion["value_status"] == "NOT_MEANINGFUL"
    assert conversion["earned_points"] == 0
    assert result["coverage_total"] == 1.
    assert result["core_reported_pass"]
    validate_reported_result(result)


def test_missing_capex_is_bounded_penalty_not_zero_payment_or_dynamic_denominator():
    rows = source_records()
    rows[-1]["capex"] = None
    result = evaluate(rows)
    assert result["coverage_total"] == .95
    assert result["missing_penalty_points"] == 1.25
    assert result["quality_score"] == result["earned_points"] - 1.25
    assert result["financial_inputs"]["capex_reported"] is None
    assert result["financial_inputs"]["fcf_reported"] is None
    assert sum(item["budget"] for item in result["metric_ledger"]) == pytest.approx(100)
    validate_reported_result(result)


def test_insufficient_domain_coverage_is_explicit_data_insufficient():
    rows = source_records()
    rows[-1]["comparative_year_end_fields"]["total_assets"] = None
    rows[-1]["comparative_year_end_fields"]["total_equity"] = None
    result = evaluate(rows)
    assert result["status"] == "DATA_INSUFFICIENT"
    assert result["core_reported_pass"] and result["core_inputs_pass"]
    assert result["coverage_total"] == .8 and result["coverage_by_domain"]["return_on_capital"] == 0
    validate_reported_result(result)


def test_old_generic_mapper_unknown_does_not_replace_verified_audit_classification():
    rows = source_records()
    baseline = evaluate(rows)
    rows[-1].update(auditor_opinion="UNKNOWN", audit_review_required=True, material_accounting_issue=None)
    result = evaluate(rows)
    assert core(result) == core(baseline)
    assert result["reported_result"]["optional"]["auditor_opinion"] == result["financial_safety"]["audit_classification"]["audit_opinion"]
    validate_reported_result(result)


@pytest.mark.parametrize("opinion", ["AdverseOpinionAbstract", "DisclaimerOfOpinionAbstract"])
def test_confirmed_source_audit_hard_risk_is_not_hidden_by_high_quality(opinion):
    result = evaluate(source_records(opinion=opinion))
    assert result["status"] == "REJECT" and not result["mandatory_safety_pass"]
    validate_reported_result(result)


def test_missing_audit_proof_or_market_pass_receipt_cannot_substitute_financial_proof():
    rows = source_records()
    rows[-1].pop("audit_evidence")
    class FakeMarketPass:
        state = "PASS"
    result = evaluate(rows, safety_receipt=FakeMarketPass())
    assert result["status"] == "DATA_INSUFFICIENT" and not result["mandatory_safety_pass"]
    assert result["financial_safety"]["state"] == "UNKNOWN"
    validate_reported_result(result)


def test_rehashed_promote_of_unknown_financial_receipt_is_still_rejected():
    rows = source_records()
    rows[-1].pop("audit_evidence")
    result = evaluate(rows)
    assert result["financial_safety"]["state"] == "UNKNOWN"
    assert result["reported_result"]["status"] == "PASS" and result["core_reported_pass"]
    result["financial_safety"]["state"] = "PASS"
    result.update(status="PASS", core_inputs_pass=True, mandatory_safety_pass=True)
    identity(result)
    with pytest.raises(ValueError, match="FINANCIAL_PROOF_RECEIPT_MISMATCH"):
        validate_reported_result(result)


@pytest.mark.parametrize("damage", ["cash_value", "cash_context", "audit", "available_at"])
def test_stored_source_packet_must_reproduce_complete_financial_receipt(damage):
    result = evaluate(source_records())
    result = deepcopy(result)
    source = result["financial_safety_source"]
    if damage == "cash_value": source["cash_ending"] += 100.
    elif damage == "cash_context": source["field_evidence"]["cash_ending"]["contexts"][0]["numeric_value"] += 100_000.
    elif damage == "audit": source["audit_evidence"]["contexts"][0]["text_value"] = "N"
    else: source["available_at"] = "2026-10-03T00:00:00Z"
    with pytest.raises(ValueError, match="V7_REPORTED_QUALITY_INVALID"):
        validate_reported_result(result)


def test_generic_qualified_audit_is_not_promoted_like_proven_nonmaterial_scope_exception():
    result = evaluate(source_records(opinion="QualifiedOpinionAbstract"))
    assert result["status"] == "DATA_INSUFFICIENT"
    assert result["financial_safety"]["state"] == "UNKNOWN"
    assert result["reported_result"]["optional"]["audit_review_required"] is True
    validate_reported_result(result)


def test_readonly_evidence_and_detached_diagnostics_cannot_pollute_next_date():
    rows = source_records()
    baseline = evaluate(rows)
    frozen = ReadOnlyEvidence(dict(records=rows))["records"]
    result = evaluate(frozen)
    # The immutable context stores sequences as tuples; JSON business content
    # remains identical across mutable/read-only inputs without dropping fields.
    assert identity(result) == identity(baseline)
    assert result["financial_inputs"] is result["reported_result"]["metrics"]
    published = json.loads(json.dumps(result, allow_nan=False))
    published["reported_result"]["metrics"]["quality_raw_score"] = 1.
    result["field_checks"]["2026Q2:revenue"]["value"] = 1.
    assert identity(evaluate(frozen)) == identity(baseline)


@pytest.mark.parametrize("key", ["policy_sha256", "source_mapping_sha256", "capex_rule"])
def test_modified_registered_policy_is_config_failure_not_fake_pass(key):
    from domain.les_v7.quality_policy import r3_quality_policy
    policy = r3_quality_policy()
    policy[key] = "SYNTHETIC_ONLY_TAMPER"
    result = evaluate(source_records(), policy=policy)
    assert result["status"] == "CONFIG_INCOMPLETE"
    validate_reported_result(result)


@pytest.mark.parametrize("damage", ["score", "dc", "coverage", "penalty", "core", "native_score", "native_budget", "native_penalty", "native_dc", "ledger"])
def test_rehashed_tampering_cannot_rewrite_reported_quality_business_arithmetic(damage):
    result = evaluate(source_records())
    if damage == "score": result["quality_score"] -= 1
    elif damage == "dc": result["data_confidence"] -= 1
    elif damage == "coverage": result["coverage_total"] -= .01
    elif damage == "penalty": result["missing_penalty_points"] += 1
    elif damage == "core": result["core_inputs_pass"] = False
    elif damage == "native_score": result["reported_result"]["score"] -= 1
    elif damage == "native_budget": result["reported_result"]["metrics"]["quality_budgets"]["cash_flow"] = 20
    elif damage == "native_penalty": result["reported_result"]["metrics"]["quality_penalty"] += 1
    elif damage == "native_dc": result["reported_result"]["metrics"]["quality_dc_inputs"]["verification"] -= 1
    else: result["metric_ledger"][0]["earned_points"] -= 1
    # Hashes are not an excuse to ignore business arithmetic or source floors.
    identity(result)
    with pytest.raises(ValueError, match="V7_REPORTED_QUALITY_INVALID"):
        validate_reported_result(result)


def test_insufficient_result_still_validates_partial_arithmetic_not_only_pass():
    rows = source_records()
    rows[-1]["cfo"] = None
    result = evaluate(rows)
    assert result["status"] == "DATA_INSUFFICIENT"
    result["reported_result"]["metrics"]["quality_metric_checks"]["cfo_positive"]["points"] = 1.
    result["metric_ledger"] = [dict(item, earned_points=1.) if item["metric_id"] == "cfo_positive" else item
                               for item in result["metric_ledger"]]
    with pytest.raises(ValueError, match="V7_REPORTED_QUALITY_INVALID"):
        validate_reported_result(result)


@pytest.mark.parametrize("component", ["freshness", "reliability"])
def test_rebound_dc_total_cannot_conceal_wrong_time_or_source_component(component):
    result = evaluate(source_records())
    metrics = result["reported_result"]["metrics"]
    metrics["quality_dc_inputs"][component] -= 10.
    result["reported_result"]["confidence"] = sum(
        metrics["quality_dc_weights"][key] * value for key, value in metrics["quality_dc_inputs"].items())
    result["data_confidence"] = result["reported_result"]["confidence"]
    # All numerical totals and inner/outer projections agree. It is still not
    # the actual source reliability or native cutoff freshness policy.
    with pytest.raises(ValueError, match="V7_REPORTED_QUALITY_INVALID:DC_ARITHMETIC"):
        validate_reported_result(result)
