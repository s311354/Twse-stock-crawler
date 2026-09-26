"""An audit qualification is not synonymous with a proven misstatement."""
from dataclasses import replace

import pytest

from domain.fundamental_derivation import (
    FiledFact, FiledFinancials, _derive_audit_evidence, _qualified_basis_text,
    derive_fundamentals,
)


def filing(body, opinion="QualifiedOpinionAbstract"):
    facts = tuple(FiledFact(f"tifrs-ar:{concept}", None, text, None, None, None)
                  for concept, text in ((opinion, "Y"), ("AccountantsReportBody", body)))
    return FiledFinancials(2026, 2, "2026-08-16T00:00:00+08:00", 1, facts)


@pytest.mark.parametrize("body", [
    "依本會計師之意見，除保留意見之基礎段所述事項外。"
    "保留意見之基礎：本會計師未能接觸部分非重要子公司之財務資訊及管理階層，"
    "無法取得足夠查核證據，無法判斷是否須調整。"
    "本會計師相信取得足夠證據以作為表示保留意見之基礎。"
    "關鍵查核事項：除保留意見之基礎段所述事項外。"
    "管理階層與治理單位對合併財務報告之責任：確保報告不存在重大不實表達。"
    "會計師查核合併財務報告之責任：辨認重大錯報風險。",
    "Except for the matter described in the 'Basis for Qualified Opinion' section. "
    "Basis for Qualified Opinion: We were unable to access information from certain subsidiaries. "
    "Key Audit Matters: As described in the Basis for Qualified Opinion section. "
    "Responsibilities of management: Prevent material misstatement. "
    "Auditor's responsibilities: Assess material misstatement risks.",
])
def test_inaccessible_evidence_is_unknown_not_a_proven_material_misstatement(body):
    evidence = _derive_audit_evidence(filing(body))
    assert evidence["current_audit_opinion"] == "UNKNOWN"
    assert evidence["material_accounting_issue"] is None
    assert evidence["audit_review_required"] is True
    basis = _qualified_basis_text(body, current_fiscal_year=2026)
    assert "重大不實" not in basis and "material misstatement" not in basis
    assert "未能接觸" in basis or "unable to access" in basis
    # Flag survives the full derived snapshot used by the daily v6 adapter.
    bundle = derive_fundamentals([filing(body)], signal_date="2026-08-17T13:30:00+08:00", sector="ELEC")
    assert bundle.snapshot["audit_review_required"] is True


@pytest.mark.parametrize("boundary", [
    "關鍵查核事項", "管理階層對財務報告之責任", "會計師核閱合併財務報告之責任",
    "Other Matters", "Emphasis of Matter", "Management’s Responsibilities",
    "Auditor’s Responsibilities", "Responsibilities of the auditor",
])
def test_scope_limitation_does_not_absorb_other_report_sections(boundary):
    body = ("保留結論之基礎：部分非重要 子公司未經會計師核閱。"
            + boundary + "：重大錯報、重大不實表達、material misstatement。")
    evidence = _derive_audit_evidence(filing(body))
    assert evidence["current_audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert evidence["material_accounting_issue"] is False
    assert evidence["audit_review_required"] is False


@pytest.mark.parametrize("basis", [
    "存貨未提列減損而重大高估，構成重大錯報。",
    "Inventory is materially misstated due to failure to recognize impairment.",
])
def test_actual_basis_misstatement_remains_hard_risk(basis):
    evidence = _derive_audit_evidence(filing("保留意見之基礎：" + basis + "關鍵查核事項：收入認列。"))
    assert evidence["current_audit_opinion"] == "QUALIFIED_MATERIAL_MISSTATEMENT"
    assert evidence["material_accounting_issue"] is True
    assert evidence["audit_review_required"] is False


def test_no_actual_basis_heading_does_not_classify_cross_reference():
    body = "除保留意見之基礎段所述外。管理階層對財務報告之責任：避免重大錯報。"
    evidence = _derive_audit_evidence(filing(body))
    assert evidence["current_audit_opinion"] == "UNKNOWN"
    assert evidence["material_accounting_issue"] is None


def test_inaccessible_component_cannot_be_cleared_as_routine_unreviewed_component():
    body = "保留結論之基礎：部分非重要子公司未經核閱，且本會計師無法取得財務資訊。"
    evidence = _derive_audit_evidence(filing(body))
    assert evidence["audit_review_required"] is True
    assert evidence["material_accounting_issue"] is None


def test_current_unmodified_and_prior_qualified_basis_stay_separate():
    evidence = _derive_audit_evidence(filing(
        "本期無保留意見。前一年度保留意見之基礎：存貨重大高估。"
        "管理階層對財務報告之責任：避免重大錯報。",
        opinion="UnqualifiedOpinionAbstract",
    ))
    assert evidence["current_audit_opinion"] == "UNMODIFIED"
    assert evidence["comparative_audit_opinion"] == "QUALIFIED_MATERIAL_MISSTATEMENT"
    assert evidence["material_accounting_issue"] is False
    assert evidence["audit_review_required"] is False


def test_conflicting_abstracts_require_review():
    report = filing("無保留意見。")
    report = replace(report, facts=report.facts + (
        FiledFact("tifrs-ar:UnqualifiedOpinionAbstract", None, "Y", None, None, None),
    ))
    assert _derive_audit_evidence(report)["audit_review_required"] is True
