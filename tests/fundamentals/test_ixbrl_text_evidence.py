"""Text evidence survives parser, SQLite, and snapshot normalization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from domain.data_provenance import OFFICIAL_FILING_TIMESTAMP
from domain.fundamental_derivation import derive_fundamentals
from domain.fundamental_models import FilingCandidate
from domain.text_normalization import normalize_cjk_text
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_repository import MopsRepository


@pytest.mark.parametrize(
    ("report_html", "raw_text"),
    [
        (
            "保留結論之基礎：部分非重要&nbsp;\n子公司未經會計師核&#x202f;閱。",
            "保留結論之基礎：部分非重要\u00a0\n子公司未經會計師核\u202f閱。",
        ),
        (
            "保留結論之基礎：部分非重要<br/>子公司未經會計師核\u200b閱。",
            "保留結論之基礎：部分非重要\n子公司未經會計師核\u200b閱。",
        ),
    ],
)
def test_ixbrl_text_evidence_round_trips_through_sqlite(
    tmp_path: Path, report_html: str, raw_text: str,
) -> None:
    content = f"""<html><body><ix:header><ix:resources>
<xbrli:context id="FY2025"><xbrli:period>
<xbrli:startDate>2025-01-01</xbrli:startDate>
<xbrli:endDate>2025-12-31</xbrli:endDate>
</xbrli:period></xbrli:context></ix:resources></ix:header>
<ix:nonFraction name="ifrs-full:Revenue" contextRef="FY2025" unitRef="TWD" scale="3">1,234</ix:nonFraction>
<ix:nonNumeric name="tifrs-ar:QualifiedOpinionAbstract" contextRef="FY2025"> Ｙ </ix:nonNumeric>
<ix:nonNumeric name="tifrs-ar:AdverseOpinionAbstract" contextRef="FY2025"> Ｎ </ix:nonNumeric>
<ix:nonNumeric name="tifrs-ar:AccountantsReportBody" contextRef="FY2025">{report_html}</ix:nonNumeric>
</body></html>""".encode("utf-8")
    available_at = "2026-02-10T18:00:00+08:00"
    candidate = FilingCandidate(
        symbol="2330",
        fiscal_year=2025,
        quarter=4,
        report_id="C",
        source_url="https://example.test/2330-2025Q4.html",
        available_at=available_at,
        availability_source=OFFICIAL_FILING_TIMESTAMP,
    )
    facts = parse_ixbrl(content)
    assert facts[0].numeric_value == 1_234_000
    assert facts[-1].text_value == raw_text

    repository = MopsRepository(tmp_path / "mops")
    filing = repository.save_revision(
        candidate, content, facts,
        available_at=available_at,
        fetched_at=available_at,
    )
    # Reopen the repository to verify persisted evidence, not in-memory objects.
    repository = MopsRepository(tmp_path / "mops")
    assert repository.raw_path(filing).read_bytes() == content
    assert filing.content_sha256 == hashlib.sha256(content).hexdigest()
    assert repository.load_facts(filing.filing_id)[-1].text_value == raw_text
    assert repository.load_filed_financials(
        "2330", "2026-02-10T17:59:59+08:00"
    ) == []
    financials = repository.load_filed_financials("2330", available_at)
    assert financials[0].facts[-1].text_value == raw_text

    bundle = derive_fundamentals(
        financials, signal_date=available_at, sector="24"
    )
    expected = "保留結論之基礎:部分非重要子公司未經會計師核閱。"
    assert bundle.snapshot["audit_report_text_raw"] == raw_text
    assert bundle.snapshot["audit_report_text_normalized"] == expected
    assert bundle.snapshot["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert bundle.snapshot["material_accounting_issue"] is False
    assert bundle.snapshot["available_at"] == available_at
    assert bundle.snapshot["revision_number"] == filing.revision_number
    assert bundle.snapshot["content_sha256"] == filing.content_sha256

    repository.save_fundamental_bundle("2330", "text-test", "24", bundle)
    cached = repository.load_fundamental_bundle("2330", "text-test", "24")
    assert cached is not None
    assert cached.snapshot["audit_report_text_raw"] == raw_text
    assert cached.snapshot["audit_report_text_normalized"] == expected


def test_ixbrl_layout_keeps_english_word_boundaries() -> None:
    content = """<ix:nonNumeric name="tifrs-ar:AccountantsReportBody" contextRef="x"><p>basis</p><p>for qualified<br>opinion: 非重要<span>子</span><span>公司</span></p></ix:nonNumeric>"""
    fact = parse_ixbrl(content)[0]

    assert fact.text_value == "basis\nfor qualified\nopinion: 非重要子公司\n"
    assert normalize_cjk_text(fact.text_value) == (
        "basis for qualified opinion: 非重要子公司"
    )


def test_ixbrl_empty_text_and_numeric_layout_are_unchanged() -> None:
    content = """<ix:nonNumeric name="tifrs-ar:AccountantsReportBody" contextRef="x"> &nbsp;\n </ix:nonNumeric>
<ix:nonFraction name="ifrs-full:Revenue" contextRef="x" scale="3" sign="-">1,<br/>234</ix:nonFraction>"""
    empty, numeric = parse_ixbrl(content)

    assert empty.text_value is None
    assert numeric.text_value is None
    assert numeric.numeric_value == -1_234_000
