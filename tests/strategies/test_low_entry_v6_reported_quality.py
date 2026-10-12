"""SYNTHETIC reported-period policy tests, not proof of completed real data."""
from copy import deepcopy
from dataclasses import replace

import pytest

from domain.strategies.low_entry_v6_quality import evaluate_quality, quality_evidence_complete, verification_diagnostics
from tests.strategies.test_low_entry_v6_comparative_ttm import CUTOFF, originals


def evaluate(rows):
    return evaluate_quality(rows, symbol="2330", market="TWSE", as_of=CUTOFF)


def core(q):
    return q.status, q.score, q.confidence, q.components, q.coverage, q.metrics["quality_verification"]


def test_ttm_review_never_changes_reported_score_or_measured_confidence():
    rows, _ = originals()
    baseline = evaluate(rows)
    rows[-1]["accounting_basis_evidence"].update(status="UNKNOWN", reason="ACCOUNTING_POLICY_EXCEPTION_REQUIRES_REVIEW")
    q = evaluate(rows)
    assert q.metrics["cfo_ttm"] is None and baseline.metrics["cfo_ttm"] is not None
    assert core(q) == core(baseline)
    assert q.metrics["quality_required_missing"] == []
    assert quality_evidence_complete(q)
    audit = verification_diagnostics(rows, symbol="2330", as_of=CUTOFF)
    assert not audit["unresolved_fields"] and not audit["backfill_needed"]
    assert audit["diagnostic_unresolved_fields"]  # no quarter values were fabricated


def test_one_verified_original_has_all_scoring_inputs_without_eight_quarter_gate():
    rows, _ = originals()
    baseline = evaluate(rows)
    before = deepcopy(rows[-1])
    q = evaluate([rows[-1]])
    assert rows[-1] == before
    assert core(q) == core(baseline)
    assert q.metrics["revenue_ttm"] is None
    assert q.metrics["revenue_reported"] == rows[-1]["revenue"]
    assert q.metrics["quality_reported_basis"] == "YTD"
    assert quality_evidence_complete(q)
    audit = verification_diagnostics([rows[-1]], symbol="2330", as_of=CUTOFF)
    assert audit["missing_periods"] == [] and audit["unresolved_fields"] == []
    assert audit["diagnostic_missing_periods"]
    assert audit["next_action"] == "NORMAL_MODEL_EVALUATION"


@pytest.mark.parametrize("key,value", [("content_sha256", "wrong"), ("revision_id", "other"),
    ("available_at", "2026-10-03T00:00Z"), ("unit", "dollar"), ("scope", "individual"),
    ("basis", "QUARTER"), ("start_date", "2025-04-01")])
def test_failed_canonical_comparison_does_not_fallback_to_other_original(key, value):
    rows, _ = originals()
    rows[-1]["comparative_ytd_field_evidence"]["revenue"][key] = value
    q = evaluate(rows)
    assert q.metrics["quality_reported_inputs"]["comparative"]["revenue"] is None
    assert q.metrics["revenue_growth"] is None
    assert q.status == "PROVISIONAL" and not quality_evidence_complete(q)
    assert q.coverage["profitability"] == pytest.approx(100 / 3)
    assert "2026Q2:comparative:revenue" in verification_diagnostics(rows, symbol="2330", as_of=CUTOFF)["unresolved_fields"]


def test_reported_returns_use_opening_instant_not_ttm_and_do_not_annualize_values():
    rows, _ = originals()
    q = evaluate(rows)
    assert q.metrics["roe"] == q.metrics["roe_reported"] == pytest.approx(rows[-1]["net_income"] / 7000)
    assert q.metrics["roe_ttm"] == pytest.approx(q.metrics["net_income_ttm"] / 7000)
    assert q.metrics["roe"] != q.metrics["roe_ttm"]
    assert q.metrics["quality_reported_inputs"]["duration_fraction"] == .5
    assert q.metrics["fcf_reported"] == rows[-1]["cfo"] - abs(rows[-1]["capex"])
    assert q.metrics["cash_conversion"] == rows[-1]["cfo"] / rows[-1]["net_income"]


def test_verified_bad_opening_balance_identity_is_still_hard_reject():
    rows, _ = originals()
    row = rows[-1]
    row["comparative_year_end_fields"]["total_assets"] *= 2
    proof = row["comparative_year_end_field_evidence"]["total_assets"]
    for context in proof["contexts"]:
        if context["instant"] == "2025-12-31":
            context["numeric_value"] *= 2
    assert evaluate(rows).reasons == ("HISTORICAL_BALANCE_SHEET_IDENTITY_FAIL",)


def test_missing_single_return_source_does_not_erase_other_verified_return():
    rows, _ = originals()
    rows[-1]["comparative_year_end_fields"]["total_equity"] = None
    q = evaluate(rows)
    assert q.metrics["roe_reported"] is None and q.metrics["roa_reported"] is not None
    assert q.coverage["return_on_capital"] == 50
    assert q.metrics["quality_available_weight"] == 90
    assert q.metrics["quality_penalty"] == 2.5


def test_current_source_tampering_and_future_revision_do_not_grant_permission():
    rows, _ = originals()
    baseline = evaluate(rows)
    future = deepcopy(rows[-1])
    future.update(available_at="2026-10-03T00:00Z", revision_id="future", net_income=10**12)
    assert evaluate(rows + [future]) == baseline
    changed = deepcopy(baseline.metrics)
    changed["revenue_reported"] += 1
    assert not quality_evidence_complete(replace(baseline, metrics=changed))


def test_all_basis_policies_still_leave_missing_critical_values_unknown():
    rows, _ = originals()
    rows[-1]["cfo"] = None
    q = evaluate(rows)
    assert q.status == "DATA_INSUFFICIENT"
    assert q.metrics["quality_required_missing"] == ["cfo_reported"]
    assert q.metrics["cfo_ttm"] is None
    assert q.metrics["fcf_reported"] is None
    assert not quality_evidence_complete(q)


def test_canonical_quarter_never_uses_ytd_comparison_or_year_opening_as_quarter_sources():
    rows, _ = originals()
    rows[-1]["flow_basis"] = "QUARTER"
    q = evaluate(rows)
    inputs = q.metrics["quality_reported_inputs"]
    assert inputs["opening_period"] == "2026Q1"
    assert all(value is None for value in inputs["opening"].values())
    assert all(value is None for value in inputs["comparative"].values())
    assert q.status == "DATA_INSUFFICIENT"


def test_legacy_missing_comparison_is_attributed_to_its_actual_period_not_current():
    from tests.strategies.test_low_entry_score_v6 import financials
    rows = financials()[-1:]
    audit = verification_diagnostics(rows, symbol="2330", as_of=CUTOFF)
    core = {check["dependency_id"]: check for check in audit["field_checks"] if check["required_by_quality"]}
    comparison = core["2026Q2:comparative:revenue"]
    opening = core["2026Q2:opening:total_assets"]
    assert comparison["period"] == "2025Q2" and comparison["category"] == "MISSING_DATA"
    assert opening["period"] == "2026Q1" and opening["category"] == "MISSING_DATA"
    assert audit["backfill_needed"] and audit["local_review_required"]
