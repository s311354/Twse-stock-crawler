"""SYNTHETIC ONLY: bounded missing-data policy; never official repaired data."""
from copy import deepcopy
from dataclasses import replace

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.quality_policy import default_quality_policy, r3_quality_policy
from domain.les_v7.quality_reported import evaluate_financials_reported, validate_reported_result
from domain.strategies.low_entry_v6_quality import (
    QUALITY_R4_POLICY_VERSION, evaluate_quality, quality_evidence_complete,
)
from tests.strategies.test_v7_reported_quality import source_records, evaluate as r3
from tests.strategies.test_low_entry_v6_comparative_ttm import CUTOFF


def r4(rows, **kwargs):
    return evaluate_financials_reported(rows, symbol="2330", market="TWSE", cutoff=CUTOFF,
        policy=default_quality_policy(), **kwargs)


def missing_opening(rows):
    for field in ("total_assets", "total_equity"):
        rows[-1]["comparative_year_end_fields"].pop(field, None)
        rows[-1]["comparative_year_end_field_evidence"].pop(field, None)
    return rows


def missing_capex(rows):
    rows[-1]["capex"] = None
    rows[-1]["field_evidence"].pop("capex", None)
    return rows


@pytest.mark.parametrize("case,expected", [
    ("complete", "0fc67ea22fb6c754a69f6c5d2e8ac1a03a5c17b25074ee880025b8002920083c"),
    ("negative_cfo", "1defbc1891cf547abc124ca2f75d4f57cbbd905e93a13a7cb54877d239b14c62"),
    ("negative_profit", "5dea9aff174873f219c75f35fc71498485784b39584c8cf711c48f2798b68797"),
    ("missing_capex", "ed2f762441d130c040f008165bb0a759d7790d34d3deaac96631617cf8b050bc"),
    ("missing_opening", "506d5387ed16f21d6f392dd5cf633d66fc1dcc1cc77c6390da9be33d3ae58306"),
])
def test_full_r3_result_hashes_captured_before_r4_remain_unchanged(case, expected):
    rows = source_records(**{case: True}) if case.startswith("negative_") else source_records()
    if case == "missing_capex": missing_capex(rows)
    if case == "missing_opening": missing_opening(rows)
    assert identity(r3(rows)) == expected


def test_complete_business_arithmetic_is_identical_except_registered_native_policy():
    rows = source_records()
    baseline, result = r3(rows), r4(rows)
    assert result["profile_id"] == "QG_V7_REPORTED_PERIOD_R4"
    assert result["mapping_version"] == "QG_MAP_R4"
    for key in ("status", "quality_score", "earned_points", "missing_penalty_points", "data_confidence",
                "coverage_total", "coverage_by_domain", "metric_ledger", "financial_safety", "field_checks"):
        assert result[key] == baseline[key]
    before, after = deepcopy(baseline["reported_result"]), deepcopy(result["reported_result"])
    for key in ("quality_policy_version", "quality_scoring_policy", "quality_coverage_minima"):
        before["metrics"].pop(key)
        after["metrics"].pop(key)
    assert before == after  # All native business and diagnostic fields, not a cherry-picked score.
    validate_reported_result(result)


@pytest.mark.parametrize("also_missing_capex", [False, True])
def test_bounded_optional_return_gaps_are_penalized_without_domain_veto(also_missing_capex):
    rows = missing_opening(source_records())
    if also_missing_capex: missing_capex(rows)
    assert r3(rows)["status"] == "DATA_INSUFFICIENT"
    result = r4(rows)
    assert result["status"] == "PASS" and result["core_inputs_pass"]
    assert result["coverage_total"] == (.75 if also_missing_capex else .8)
    assert result["coverage_by_domain"]["return_on_capital"] == 0.
    assert result["missing_penalty_points"] == (6.25 if also_missing_capex else 5.)
    assert result["quality_score"] == result["earned_points"] - result["missing_penalty_points"]
    assert result["financial_inputs"]["roe_reported"] is None
    if also_missing_capex: assert result["financial_inputs"]["fcf_reported"] is None
    assert sum(row["budget"] for row in result["metric_ledger"]) == pytest.approx(100.)
    validate_reported_result(result)


def test_unbounded_optional_gaps_keep_the_total_evidence_floor():
    rows = missing_capex(missing_opening(source_records()))
    rows[-1]["comparative_ytd_fields"]["net_income"] = None
    result = r4(rows)
    assert result["coverage_total"] < .75
    assert result["status"] == "DATA_INSUFFICIENT" and result["quality_score"] is not None
    assert result["core_inputs_pass"] and not result["native_quality_complete"]
    validate_reported_result(result)


@pytest.mark.parametrize("zero,metrics", [
    ("zero_current_liabilities", ["current_ratio"]),
    ("zero_revenue", [name + "_" + part for name in ("gross_margin", "operating_margin", "net_margin")
                      for part in ("positive", "history", "stability")]),
])
def test_verified_zero_denominators_are_known_not_meaningful_without_invented_ratios(zero, metrics):
    rows = source_records(**{zero: True})
    result = r4(rows)
    assert result["coverage_total"] == 1.
    assert result["status"] != "DATA_INSUFFICIENT"
    ledger = {item["metric_id"]: item for item in result["metric_ledger"]}
    for name in metrics:
        assert ledger[name]["value_status"] == "NOT_MEANINGFUL"
        assert ledger[name]["available"] and ledger[name]["earned_points"] == 0.
    for name in ("current_ratio",) if zero == "zero_current_liabilities" else ("gross_margin", "operating_margin", "net_margin"):
        assert result["financial_inputs"][name] is None  # No Infinity, zero-value imputation or full points.
    validate_reported_result(result)


def test_zero_current_revenue_does_not_hide_missing_comparison_sources():
    rows = source_records(zero_revenue=True)
    rows[-1]["comparative_ytd_fields"]["revenue"] = None
    result = r4(rows)
    ledger = {item["metric_id"]: item for item in result["metric_ledger"]}
    assert ledger["gross_margin_positive"]["value_status"] == "NOT_MEANINGFUL"
    assert not ledger["gross_margin_history"]["available"]
    assert ledger["gross_margin_history"]["value_status"] == "MISSING_OR_INCOMPARABLE"
    assert result["coverage_total"] < 1.
    validate_reported_result(result)


@pytest.mark.parametrize("field", ["revenue", "cost_of_revenue", "operating_income", "net_income", "cfo",
                                  "total_assets", "current_liabilities", "total_equity"])
def test_missing_mandatory_source_never_uses_optional_penalty_authority(field):
    rows = source_records()
    rows[-1][field] = None
    result = r4(rows)
    assert result["status"] == "DATA_INSUFFICIENT" and not result["core_inputs_pass"]
    validate_reported_result(result)


@pytest.mark.parametrize("damage", ["audit_missing", "cash_missing", "audit_hard_risk"])
def test_financial_safety_is_independent_of_a_high_or_finite_quality_score(damage):
    rows = source_records(opinion="AdverseOpinionAbstract" if damage == "audit_hard_risk" else "UnqualifiedOpinionAbstract")
    if damage == "audit_missing": rows[-1].pop("audit_evidence")
    if damage == "cash_missing": rows[-1].pop("cash_ending")
    result = r4(rows)
    assert result["status"] == ("REJECT" if damage == "audit_hard_risk" else "DATA_INSUFFICIENT")
    assert not result["mandatory_safety_pass"]
    validate_reported_result(result)


def test_r4_future_versions_and_input_mutation_are_isolated():
    rows = source_records()
    frozen = deepcopy(rows)
    baseline = r4(rows)
    future = deepcopy(rows[-1])
    future.update(available_at="2026-10-03T00:00:00Z", revision_id="SYNTHETIC_ONLY_FUTURE", cfo=10**12)
    assert r4(rows + [future]) == baseline
    assert rows == frozen


def test_native_r4_completeness_requires_explicit_policy_not_legacy_metadata_contract():
    rows = source_records()
    native = evaluate_quality(rows, symbol="2330", market="TWSE", as_of=CUTOFF, policy_version=QUALITY_R4_POLICY_VERSION)
    assert quality_evidence_complete(native, policy_version=QUALITY_R4_POLICY_VERSION)
    assert not quality_evidence_complete(native)
    assert not quality_evidence_complete(replace(native, metrics={}), policy_version=QUALITY_R4_POLICY_VERSION)
    with pytest.raises(ValueError, match="UNREGISTERED_NATIVE_QUALITY_POLICY"):
        evaluate_quality(rows, symbol="2330", market="TWSE", as_of=CUTOFF, policy_version="unregistered")


@pytest.mark.parametrize("damage", ["policy", "coverage_floor", "score", "not_meaningful_points"])
def test_rehashed_r4_tampering_still_fails_business_validation(damage):
    result = r4(source_records(zero_current_liabilities=True))
    metrics = result["financial_inputs"]
    if damage == "policy": metrics["quality_scoring_policy"]["optional_coverage_policy"] = "ANYTHING_PASSES"
    elif damage == "coverage_floor": metrics["quality_minimum_available_weight"] = 0.
    elif damage == "score": result["quality_score"] += 1.
    else: metrics["quality_metric_checks"]["current_ratio"]["fraction"] = 1.
    identity(result)
    with pytest.raises(ValueError, match="V7_REPORTED_QUALITY_INVALID"):
        validate_reported_result(result)


def test_coherent_zero_revenue_score_rewrite_cannot_override_original_operands():
    result = r4(source_records())
    native, metrics = result["reported_result"], result["financial_inputs"]
    metrics["quality_reported_inputs"]["current"]["revenue"] = 0.
    metrics["quality_field_checks"]["2026Q2:revenue"]["value"] = 0.
    metrics["revenue_reported"] = result["core_reported"]["revenue"] = 0.
    for name, check in metrics["quality_metric_checks"].items():
        if check["group"] == "profitability":
            check.update(fraction=0., points=0., available=True, status="NOT_MEANINGFUL")
            projected = next(row for row in result["metric_ledger"] if row["metric_id"] == name)
            projected.update(mapped_score=0., earned_points=0., value_status="NOT_MEANINGFUL")
    native["components"]["profitability"] = 0.
    score = sum(native["components"].values())
    native["score"] = metrics["quality_raw_score"] = metrics["quality_final_score"] = score
    result["quality_score"] = result["earned_points"] = score
    identity(result)  # All numerical/projection ledgers agree; original proof does not.
    with pytest.raises(ValueError, match="NOT_MEANINGFUL_SOURCE"):
        validate_reported_result(result)
