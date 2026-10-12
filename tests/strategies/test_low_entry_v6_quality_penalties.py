"""SYNTHETIC policy regressions; not real filings, fills or performance claims."""
from copy import deepcopy
from dataclasses import replace

import pytest

from domain.strategies.low_entry_v6_decision import common_safety_caps
from domain.strategies.low_entry_v6_quality import (
    QUALITY_BUDGETS, QUALITY_MISSING_PENALTY_RATE, QUALITY_POLICY_VERSION,
    evaluate_quality, quality_evidence_complete, verification_diagnostics,
)
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.strategies.low_entry_v6_signals import decide
from tests.strategies.test_low_entry_score_v6 import CONFIG, execution_report, features, financials


def evaluate(rows):
    return evaluate_quality(rows, symbol="2330", market="TWSE", as_of="2026-09-11T13:30:00+08:00")


def test_fixed_budget_and_explicit_penalties_do_not_redistribute_missing_points():
    rows = financials()
    rows[-1]["capex"] = None
    before = deepcopy(rows)
    q = evaluate(rows)
    assert q.metrics["quality_policy_version"] == QUALITY_POLICY_VERSION
    assert q.metrics["quality_raw_score"] == sum(q.components.values())
    assert q.metrics["quality_available_weight"] == pytest.approx(sum(
        QUALITY_BUDGETS[k] * coverage / 100 for k, coverage in q.coverage.items()))
    missing = 100 - q.metrics["quality_available_weight"]
    assert q.metrics["quality_penalty"] == missing * QUALITY_MISSING_PENALTY_RATE
    assert q.score == q.metrics["quality_raw_score"] - q.metrics["quality_penalty"]
    assert q.score < evaluate(financials()).score and rows == before
    assert q.metrics["fcf_ttm"] is None  # Gate PASS does not invent an FCF value.


@pytest.mark.parametrize("field", ["capex", "income_tax", "pretax_income", "cash", "interest_bearing_debt"])
def test_bounded_soft_gap_can_pass_quality_without_an_automatic_les_cap(field):
    rows = financials()
    rows[-1][field] = None
    q = evaluate(rows)
    assert q.status == "PASS" and quality_evidence_complete(q)
    assert ("QUALITY_EVIDENCE_PENALTY_APPLIED" in q.warnings) == (field == "capex")
    if field != "capex":
        assert q.score == evaluate(financials()).score
        assert q.confidence == evaluate(financials()).confidence
    assert common_safety_caps(q, features(), industry_mode="ACTIVE") == []


@pytest.mark.parametrize("field", ["revenue", "cost_of_revenue", "operating_income", "net_income", "cfo"])
def test_critical_reported_source_gap_remains_unknown_even_when_other_scores_are_high(field):
    rows = financials()
    rows[-1][field] = None
    q = evaluate(rows)
    assert q.status == "DATA_INSUFFICIENT" and not quality_evidence_complete(q)
    assert field + "_reported" in q.metrics["quality_required_missing"]
    assert decide(q, features(), config=CONFIG)["signal"] == "DATA_INSUFFICIENT"


def test_combined_soft_gaps_still_require_bounded_total_coverage_and_actual_score():
    rows = financials()
    rows[-1].update(capex=None, interest_bearing_debt=None)
    q = evaluate(rows)
    assert q.metrics["quality_available_weight"] == 95
    assert q.metrics["quality_penalty"] == 1.25
    assert q.status == "PASS" and q.score == pytest.approx(evaluate(financials()).score - 6.25)
    # Diagnostic-only gaps cannot change fixed core scoring or coverage.
    rows[-1]["basic_eps"] = None
    assert evaluate(rows).score == q.score
    assert evaluate(rows).confidence == q.confidence
    # Prior equity removes ROE's 10-point core dependency, not ROIC's old budget.
    rows[-2]["total_equity"] = None
    q = evaluate(rows)
    assert q.metrics["quality_available_weight"] == 85
    assert q.status == "PASS" and quality_evidence_complete(q)
    # Loss of both return inputs violates the dimension floor, even though
    # overall computable weight still meets the 75-point floor.
    rows[-2]["total_assets"] = None
    q = evaluate(rows)
    assert q.metrics["quality_available_weight"] == 75
    assert q.coverage["return_on_capital"] == 0
    assert q.status == "PROVISIONAL" and not quality_evidence_complete(q)


def test_diagnostic_peer_evidence_never_changes_score_or_confidence():
    rows = financials()
    latest = rows[-1]
    latest["quality_peer_evidence"] = {name: dict(percentile=0., basis="TTM", status="VERIFIED",
        symbol="2330", market="TWSE", scope="consolidated", period="2026Q2", universe_id="SYNTHETIC_PEERS",
        source_hashes=["synthetic-peer-original"], metric_value=value, available_at=latest["available_at"])
        for name, value in (("gross_margin", .5), ("operating_margin", .3), ("net_margin", .25))}
    baseline = evaluate(rows)
    rows[-1]["quality_peer_evidence"].pop("gross_margin")
    after = evaluate(rows)
    assert after.score == baseline.score and after.confidence == baseline.confidence
    assert after.coverage == baseline.coverage
    assert baseline.metrics["quality_diagnostic_checks"]["gross_margin_peer"]["value"] == 0
    assert after.metrics["quality_diagnostic_checks"]["gross_margin_peer"]["value"] is None


def test_diagnostic_historical_scope_gap_is_retained_not_a_reported_score_penalty():
    rows = financials()
    rows[-8]["scope"] = "individual"
    q = evaluate(rows)
    assert q.status == "PASS" and q.integrity == "PASS"
    assert q.metrics["revenue_ttm"] is not None and q.metrics["revenue_growth"] is not None
    assert q.metrics["excluded_quality_inputs"][0]["period"] == "2024Q3"
    audit = verification_diagnostics(rows, symbol="2330", as_of="2026-09-11")
    assert audit["excluded_quality_inputs"] == q.metrics["excluded_quality_inputs"]
    assert audit["unusable_periods"] == ["2024Q3"] and audit["missing_periods"] == []
    checks = [c for c in audit["field_checks"] if c["period"] == "2024Q3"]
    assert checks and all(c["category"] == "INCOMPARABLE" and c["value"] is None for c in checks)
    assert not audit["unresolved_fields"] and audit["diagnostic_unresolved_fields"]


@pytest.mark.parametrize("field,value", [("total_equity", 0), ("total_assets", 20_000),
    ("auditor_opinion", "ADVERSE"), ("going_concern", True), ("material_accounting_issue", True)])
def test_confirmed_hard_risks_remain_noncompensable(field, value):
    rows = financials()
    rows[-1][field] = value
    q = evaluate(rows)
    assert q.status == "REJECT" and q.integrity == "FAIL"
    assert any(c["maximum"] == 0 for c in common_safety_caps(q, features(), industry_mode="ACTIVE"))


def test_future_revision_does_not_change_penalty_coverage_or_selected_sources():
    rows = financials()
    rows[-1]["capex"] = None
    q = evaluate(rows)
    future = dict(rows[-1], revision_id="future", available_at="2026-09-12T00:00:00Z", capex=1.)
    assert evaluate(rows + [future]) == q


def published(q):
    report = execution_report()
    report["quality"] = dict(status=q.status, score=q.score, coverage=q.coverage,
                             metrics=q.metrics, components=q.components)
    report["data"].update(confidence=q.confidence, integrity=q.integrity)
    return report


def test_scoring_report_and_execution_share_same_bounded_evidence_floor():
    rows = financials()
    rows[-1]["capex"] = None
    q = evaluate(rows)
    report = published(q)
    assert entry_gate_checks(report)["quality"] == "PASS"
    assert report_failure(report, config=CONFIG) is None
    for key in ("quality_penalty", "quality_available_weight", "quality_final_score"):
        tampered = deepcopy(q.metrics)
        tampered[key] += 1
        bad = replace(q, metrics=tampered)
        assert not quality_evidence_complete(bad)
        assert entry_gate_checks(published(bad))["quality"] == "FAIL"
        assert report_failure(published(bad), config=CONFIG) == "MISSING_CRITICAL_DATA_OR_QUALITY_FAIL"
    tampered = deepcopy(q.metrics)
    tampered["quality_penalties"][0]["points"] = 0
    assert not quality_evidence_complete(replace(q, metrics=tampered))


def test_balanced_quality_score_and_confidence_floor_is_shared_not_renderer_only():
    rows = financials()
    rows[-1].update(capex=None, interest_bearing_debt=None, basic_eps=None)
    q = evaluate(rows)
    assert q.status == "PASS" and q.score >= 65
    # Boundary check of the still-supported legacy contract. A versioned DC
    # ledger must not be changed independently of its measured input checks.
    q = replace(q, score=65, confidence=65, metrics={})
    assert quality_evidence_complete(q)
    assert entry_gate_checks(published(q))["quality"] == "PASS"
    assert report_failure(published(q), config=CONFIG) is None
    bad = replace(q, confidence=64.999)
    assert not quality_evidence_complete(bad)
    assert entry_gate_checks(published(bad))["quality"] == "FAIL"
    assert report_failure(published(bad), config=CONFIG) == "MISSING_CRITICAL_DATA_OR_QUALITY_FAIL"


def test_previous_penalty_policy_cannot_be_relabelled_as_new_execution_evidence():
    q = evaluate(financials())
    metrics = dict(q.metrics, quality_policy_version="2026-10-04-v6-bounded-evidence-penalty-v2")
    assert not quality_evidence_complete(replace(q, metrics=metrics))


def test_required_gap_diagnostics_preserve_period_and_revision_reason():
    rows = financials()
    rows[-1].update(flow_basis="YTD", flow_derivation_policy="SAME_DOCUMENT_ONLY")
    q = evaluate(rows)
    assert q.status == "PROVISIONAL"
    for field in ("revenue", "cost_of_revenue", "operating_income", "net_income", "cfo"):
        assert q.metrics[field + "_reported"] == rows[-1][field]
        assert q.metrics["quality_ttm_inputs"][field + "_ttm"]["gaps"][-1]["reason"] == "REVISION_BASIS_UNVERIFIED"
    assert q.metrics["quality_required_missing"] == []
    assert q.coverage["growth"] == q.coverage["return_on_capital"] == 0
    assert not quality_evidence_complete(q)
