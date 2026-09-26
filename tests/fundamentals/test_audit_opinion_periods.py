"""Current audit decisions must not inherit comparative-period opinions."""

from dataclasses import replace
from pathlib import Path

import pytest

from domain.data_provenance import OFFICIAL_FILING_TIMESTAMP
from domain.fundamental_derivation import (
    FiledFact,
    FiledFinancials,
    _derive_audit_evidence,
    derive_fundamentals,
)
from domain.fundamental_models import FilingCandidate
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_repository import MopsRepository


def fact(concept: str, text: str | None, end: str | None = None) -> FiledFact:
    return FiledFact(
        f"tifrs-ar:{concept}", None, text, None, end, None,
        context_id=f"period-{end}" if end else None,
    )


def filing(*facts: FiledFact) -> FiledFinancials:
    return FiledFinancials(
        2025, 2, "2025-08-10T18:00:00+08:00", 1, tuple(facts),
        period_end="2025-06-30",
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_context_dates_separate_structured_opinions(reverse: bool) -> None:
    facts = (
        fact("UnqualifiedOpinionAbstract", "Y", "2025-06-30"),
        fact("QualifiedOpinionAbstract", "Y", "2024-06-30"),
        fact("AccountantsReportBody", "保留結論之基礎：非重要子公司未經核閱。", "2024-06-30"),
    )
    result = _derive_audit_evidence(filing(*(facts[::-1] if reverse else facts)))

    assert result["current_audit_opinion"] == "UNMODIFIED"
    assert result["comparative_audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert result["audit_opinion_source"] == "ABSTRACT"
    assert result["comparative_audit_opinion_source"] == "ABSTRACT_COMPARATIVE"
    assert result["material_accounting_issue"] is False
    assert result["audit_review_required"] is False


def test_prior_structured_opinion_wins_over_prior_narrative() -> None:
    result = _derive_audit_evidence(filing(
        fact("UnqualifiedOpinionAbstract", "Y", "2025-06-30"),
        fact("AdverseOpinionAbstract", "Y", "2024-06-30"),
        fact("AccountantsReportBody", "前期無保留意見。本期無保留意見。", "2025-06-30"),
    ))
    assert result["current_audit_opinion"] == "UNMODIFIED"
    assert result["comparative_audit_opinion"] == "ADVERSE"
    assert result["material_accounting_issue"] is False


@pytest.mark.parametrize("text", [None, "N", "0", "false", "UNKNOWN", "N/A", "未知"])
def test_present_but_unusable_abstract_does_not_enable_body_fallback(text: str | None) -> None:
    result = _derive_audit_evidence(filing(
        fact("UnqualifiedOpinionAbstract", text, "2025-06-30"),
        fact("AccountantsReportBody", "本期無保留意見。", "2025-06-30"),
    ))
    assert result["current_audit_opinion"] == "UNKNOWN"
    assert result["audit_opinion_source"] == "ABSTRACT_UNKNOWN"
    assert result["material_accounting_issue"] is None
    assert result["audit_review_required"] is True


def test_prior_abstract_does_not_disable_current_body_fallback() -> None:
    result = _derive_audit_evidence(filing(
        fact("QualifiedOpinionAbstract", "Y", "2024-06-30"),
        fact("AccountantsReportBody", "本期無保留結論。", "2025-06-30"),
    ))
    assert result["current_audit_opinion"] == "UNMODIFIED"
    assert result["audit_opinion_source"] == "REPORT_BODY_FALLBACK"
    assert result["comparative_audit_opinion"] == "QUALIFIED_UNKNOWN"


@pytest.mark.parametrize("body", [
    "前期保留結論之基礎：非重要子公司未經核閱。",
    "本報告敘述管理階層對財務報告之責任。",
])
def test_no_positive_current_opinion_evidence_stays_unknown(body: str) -> None:
    result = _derive_audit_evidence(filing(fact("AccountantsReportBody", body)))
    assert result["current_audit_opinion"] == "UNKNOWN"
    assert result["material_accounting_issue"] is None


@pytest.mark.parametrize("prior,current", [
    ("前期", "本期"),
    ("民國113年度", "民國114年度"),
    ("Previous year: ", "Current period: "),
])
def test_long_prior_section_and_short_current_transition(prior: str, current: str) -> None:
    body = (
        prior + "比較財務報告。" * 12
        + "保留結論之基礎：非重要子公司未經核閱。"
        + current + "無保留意見。"
    )
    result = _derive_audit_evidence(filing(fact("AccountantsReportBody", body)))
    assert result["current_audit_opinion"] == "UNMODIFIED"
    assert result["comparative_audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"


def test_prior_material_basis_does_not_contaminate_current_scope_basis() -> None:
    result = _derive_audit_evidence(filing(
        fact("QualifiedOpinionAbstract", "Y"),
        fact("AccountantsReportBody", (
            "本期保留結論之基礎：非重要子公司未經核閱。"
            "前期保留意見之基礎：存貨重大高估。"
        )),
    ))
    assert result["current_audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert result["comparative_audit_opinion"] == "QUALIFIED_MATERIAL_MISSTATEMENT"
    assert result["material_accounting_issue"] is False


@pytest.mark.parametrize("concept,body,expected", [
    ("QualifiedOpinionAbstract", "前期無保留意見。本期保留意見之基礎：存貨重大高估。", "QUALIFIED_MATERIAL_MISSTATEMENT"),
    ("AdverseOpinionAbstract", "前期無保留意見。", "ADVERSE"),
    ("DisclaimerOfOpinionAbstract", "前期無保留意見。", "DISCLAIMER"),
])
def test_current_hard_risk_is_not_cleared_by_prior_unmodified(
    concept: str, body: str, expected: str,
) -> None:
    result = _derive_audit_evidence(filing(
        fact(concept, "Y", "2025-06-30"),
        fact("UnqualifiedOpinionAbstract", "Y", "2024-06-30"),
        fact("AccountantsReportBody", body, "2025-06-30"),
    ))
    assert result["current_audit_opinion"] == expected
    assert result["comparative_audit_opinion"] == "UNMODIFIED"


def test_prior_going_concern_does_not_become_current_flag() -> None:
    result = _derive_audit_evidence(filing(
        fact("UnqualifiedOpinionAbstract", "Y", "2025-06-30"),
        fact("AccountantsReportBody", "繼續經營重大不確定性。", "2024-06-30"),
    ))
    assert result["going_concern"] is False


def test_current_going_concern_remains_even_with_unqualified_abstract() -> None:
    result = _derive_audit_evidence(filing(
        fact("UnqualifiedOpinionAbstract", "Y"),
        fact("AccountantsReportBody", "前期無保留意見。本期繼續經營重大不確定性。"),
    ))
    assert result["going_concern"] is True


def test_dated_current_facts_win_over_undated_and_unrelated_contexts() -> None:
    result = _derive_audit_evidence(filing(
        replace(fact("UnqualifiedOpinionAbstract", "Y"), instant="2025-06-30"),
        fact("AdverseOpinionAbstract", "Y"),
        fact("AdverseOpinionAbstract", "Y", "2025-12-31"),
        fact("AdverseOpinionAbstract", "Y", "invalid"),
        replace(fact("AdverseOpinionAbstract", "Y", "2025-06-30"), dimensions=(("entity", "subsidiary"),)),
    ))
    assert result["current_audit_opinion"] == "UNMODIFIED"
    assert result["comparative_audit_opinion"] is None


def test_latest_prior_period_selected_without_merging_all_history() -> None:
    result = _derive_audit_evidence(filing(
        fact("UnqualifiedOpinionAbstract", "Y", "2025-06-30"),
        fact("UnqualifiedOpinionAbstract", "Y", "2025-03-31"),
        fact("AdverseOpinionAbstract", "Y", "2024-06-30"),
    ))
    assert result["comparative_audit_opinion"] == "UNMODIFIED"


def test_empty_prior_body_does_not_invent_comparative_evidence() -> None:
    result = _derive_audit_evidence(filing(
        fact("UnqualifiedOpinionAbstract", "Y", "2025-06-30"),
        fact("AccountantsReportBody", None, "2024-06-30"),
    ))
    assert result["comparative_audit_opinion"] is None
    assert result["comparative_audit_opinion_source"] == "NONE"


def test_independent_current_facts_do_not_inherit_each_others_prior_labels() -> None:
    result = _derive_audit_evidence(filing(
        fact("AccountantsReportBody", "前期無保留意見。", "2025-06-30"),
        fact("QualifiedOpinionAbstract", "保留意見之基礎：存貨重大高估。", "2025-06-30"),
    ))
    assert result["current_audit_opinion"] == "QUALIFIED_MATERIAL_MISSTATEMENT"
    assert result["comparative_audit_opinion"] == "UNMODIFIED"

    body_only = _derive_audit_evidence(filing(
        fact("AccountantsReportBody", "前期無保留意見。", "2025-06-30"),
        fact("AccountantsReportBody", "保留意見之基礎：存貨重大高估。", "2025-06-30"),
    ))
    assert body_only["current_audit_opinion"] == "QUALIFIED_MATERIAL_MISSTATEMENT"


def test_invalid_filing_period_fails_closed() -> None:
    result = _derive_audit_evidence(replace(
        filing(fact("UnqualifiedOpinionAbstract", "Y", "2025-06-30")),
        period_end="not-a-date",
    ))
    assert result["current_audit_opinion"] == "UNKNOWN"
    assert result["material_accounting_issue"] is None


def test_parser_sqlite_snapshot_preserve_contexts_and_point_in_time(tmp_path: Path) -> None:
    content = """<html><body><ix:resources>
<xbrli:context id="current"><xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-06-30</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="prior"><xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-06-30</xbrli:endDate></xbrli:period></xbrli:context>
</ix:resources>
<ix:nonNumeric name="tifrs-ar:UnqualifiedOpinionAbstract" contextRef="current">Ｙ</ix:nonNumeric>
<ix:nonNumeric name="tifrs-ar:QualifiedOpinionAbstract" contextRef="prior">Y</ix:nonNumeric>
<ix:nonNumeric name="tifrs-ar:AccountantsReportBody" contextRef="prior">保留結論之基礎：非重要<br/>子公司未經核閱。</ix:nonNumeric>
</body></html>""".encode("utf-8")
    timestamp = "2025-08-10T18:00:00+08:00"
    candidate = FilingCandidate(
        symbol="2330", fiscal_year=2025, quarter=2, report_id="C",
        source_url="https://example.test/2025Q2.html", available_at=timestamp,
        availability_source=OFFICIAL_FILING_TIMESTAMP,
        period_end="2025-06-30", audit_status="reviewed",
    )
    repository = MopsRepository(tmp_path / "mops")
    saved = repository.save_revision(
        candidate, content, parse_ixbrl(content),
        available_at=timestamp, fetched_at=timestamp,
    )
    repository = MopsRepository(tmp_path / "mops")
    assert repository.load_filed_financials("2330", "2025-08-10T17:59:59+08:00") == []
    financials = repository.load_filed_financials("2330", timestamp)
    assert {item.context_id for item in financials[0].facts} == {"current", "prior"}
    snapshot = derive_fundamentals(financials, signal_date=timestamp, sector="24").snapshot
    assert snapshot["current_audit_opinion"] == "REVIEWED_UNMODIFIED"
    assert snapshot["comparative_audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert snapshot["audit_opinion_source"] == "ABSTRACT"
    assert snapshot["comparative_audit_opinion_source"] == "ABSTRACT_COMPARATIVE"
    assert snapshot["audit_report_text_raw"] == "保留結論之基礎：非重要\n子公司未經核閱。"
    assert snapshot["content_sha256"] == saved.content_sha256
    assert repository.raw_path(saved).read_bytes() == content
