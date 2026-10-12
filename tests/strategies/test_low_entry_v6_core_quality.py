"""SYNTHETIC core-policy tests, not real filings or investment performance."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from domain.strategies.low_entry_v6_quality import (
    QUALITY_BUDGETS, QUALITY_DIAGNOSTIC_ONLY, QUALITY_POLICY_VERSION,
    evaluate_quality, quality_evidence_complete, verification_diagnostics,
)
from tests.strategies.quality_evidence_fixtures import bind_synthetic_contexts
from tests.strategies.test_low_entry_score_v6 import financials


def evaluate(rows):
    return evaluate_quality(rows, symbol="2330", market="TWSE", as_of="2026-09-11T13:30:00+08:00")


def core(q):
    return (q.status, q.score, q.confidence, q.integrity, q.components, q.coverage,
            q.metrics["quality_verification"], q.metrics["quality_penalties"], q.reasons)


def test_removed_items_do_not_change_score_coverage_dc_or_required_sources():
    rows = [bind_synthetic_contexts(row) for row in financials()]
    complete = evaluate(rows)
    for row in rows:
        for field in ("basic_eps", "interest_bearing_debt", "cash", "income_tax", "pretax_income"):
            row[field] = None
        row.pop("quality_peer_evidence", None)
    before = deepcopy(rows)
    partial = evaluate(rows)
    assert rows == before
    assert core(partial) == core(complete)
    assert quality_evidence_complete(partial)
    assert partial.metrics["roic"] is None
    assert partial.metrics["positive_eps_quarters"] is None
    assert partial.metrics["quality_missing_weight"] == 0
    assert set(QUALITY_DIAGNOSTIC_ONLY).isdisjoint(partial.metrics["quality_metric_checks"])
    assert all(not d["scored"] and not d["affects_gate"] for d in partial.metrics["quality_diagnostic_checks"].values())
    assert not any(key.split(":")[1] in {"basic_eps", "interest_bearing_debt", "cash", "income_tax", "pretax_income"}
                   for key in partial.metrics["quality_verification"]["dependencies"])


def test_five_fixed_budgets_are_fully_assessed_without_diagnostic_inputs():
    q = evaluate(financials())
    checks = q.metrics["quality_metric_checks"]
    assert sum(d["weight"] for d in checks.values()) == pytest.approx(100)
    for group, budget in QUALITY_BUDGETS.items():
        assert sum(d["weight"] for d in checks.values() if d["group"] == group) == pytest.approx(budget)
    assert checks["roe"]["weight"] == checks["roa"]["weight"] == 10
    assert checks["revenue_growth"]["weight"] == checks["operating_income_growth"]["weight"] == 7.5
    assert checks["net_income_growth"]["weight"] == 5
    assert set(q.coverage.values()) == {100}


def test_diagnostic_gaps_are_retained_but_do_not_request_quality_remediation():
    rows = financials()
    for row in rows:
        for field in ("basic_eps", "income_tax", "pretax_income", "cash", "interest_bearing_debt"):
            row[field] = None
    audit = verification_diagnostics(rows, symbol="2330", as_of="2026-09-11T13:30:00+08:00")
    diagnostic_gaps = [r for r in audit["field_gaps"] if r["field"] in
                       {"basic_eps", "income_tax", "pretax_income", "cash", "interest_bearing_debt"}]
    assert diagnostic_gaps and all(not r["required_by_quality"] for r in diagnostic_gaps)
    assert audit["unresolved_fields"] == [] and not audit["local_review_required"]
    assert audit["next_action"] == "NORMAL_MODEL_EVALUATION"


@pytest.mark.parametrize("field", ["revenue", "cost_of_revenue", "operating_income", "net_income", "cfo"])
def test_necessary_reported_evidence_still_blocks_without_optional_score_items(field):
    rows = financials()
    rows[-1][field] = None
    q = evaluate(rows)
    assert q.status == "DATA_INSUFFICIENT"
    assert not quality_evidence_complete(q)
    assert field + "_reported" in q.metrics["quality_required_missing"]


def test_missing_capex_still_reduces_fcf_score_not_assumed_zero():
    rows = financials()
    baseline = evaluate(rows)
    rows[-1]["capex"] = None
    partial = evaluate(rows)
    assert partial.metrics["fcf_ttm"] is None
    assert partial.score < baseline.score
    assert partial.metrics["quality_available_weight"] == 95
    assert partial.metrics["quality_penalty"] == 1.25


@pytest.mark.parametrize("field,value", [("auditor_opinion", "ADVERSE"), ("going_concern", True),
                                          ("material_accounting_issue", True)])
def test_diagnostic_removal_never_ignores_confirmed_hard_risk(field, value):
    rows = financials()
    rows[-1][field] = value
    assert evaluate(rows).status == "REJECT"


def test_new_policy_roundtrips_but_old_policy_or_changed_weights_cannot_execute():
    q = evaluate(financials())
    roundtrip = replace(q, metrics=json.loads(json.dumps(q.metrics)))
    assert quality_evidence_complete(roundtrip)
    for old in ("2026-10-04-v6-field-evidence-ttm-quality-v5", "fake-policy"):
        assert not quality_evidence_complete(replace(q, metrics=dict(q.metrics, quality_policy_version=old)))
    changed = deepcopy(q.metrics)
    changed["quality_scoring_policy"]["return_weights"]["roe"] = 20
    assert not quality_evidence_complete(replace(q, metrics=changed))
    assert q.metrics["quality_policy_version"] == QUALITY_POLICY_VERSION
