"""Direct plugin callers must not bypass audit review through optional fields."""
import pytest

from domain.strategies.low_entry_v6_quality import evaluate_quality
from tests.strategies.test_low_entry_score_v6 import financials


def assessed(**fields):
    records = financials()
    records[-1].update(fields)
    return evaluate_quality(records, symbol="2330", market="TWSE", as_of="2026-09-11T15:00:00+08:00")


@pytest.mark.parametrize("fields", [
    {"auditor_opinion": "UNKNOWN"}, {"auditor_opinion": "QUALIFIED"},
    {"auditor_opinion": "QUALIFIED_UNKNOWN"}, {"audit_review_required": True},
])
def test_explicit_unresolved_audit_blocks_without_zeroing_calculated_quality(fields):
    result = assessed(**fields)
    assert result.score >= 75
    assert result.status == "DATA_INSUFFICIENT"
    assert result.reasons == ("AUDIT_REVIEW_REQUIRED",)


@pytest.mark.parametrize("fields", [
    {}, {"auditor_opinion": None}, {"auditor_opinion": "UNMODIFIED"},
    {"auditor_opinion": "QUALIFIED_SCOPE_LIMITATION", "audit_review_required": False},
])
def test_absent_optional_audit_or_resolved_scope_is_not_unresolved_known_evidence(fields):
    assert assessed(**fields).status == "PASS"


@pytest.mark.parametrize("fields", [
    {"auditor_opinion": "QUALIFIED_MATERIAL_MISSTATEMENT"},
    {"material_accounting_issue": True}, {"going_concern": True},
])
def test_hard_risk_beats_review_grace_or_numeric_score(fields):
    result = assessed(audit_review_required=True, **fields)
    assert result.status == "REJECT"
    assert result.reasons == ("CONFIRMED_OPTIONAL_HARD_RISK",)


def test_future_qualification_revision_cannot_rewrite_earlier_quality():
    records = financials()
    records.append(dict(records[-1], available_at="2026-09-12T09:00:00+08:00",
                        revision_id="future-audit-review", auditor_opinion="QUALIFIED"))
    before = evaluate_quality(records, symbol="2330", market="TWSE", as_of="2026-09-11T15:00:00+08:00")
    after = evaluate_quality(records, symbol="2330", market="TWSE", as_of="2026-09-12T10:00:00+08:00")
    assert before.status == "PASS"
    assert after.status == "DATA_INSUFFICIENT"
