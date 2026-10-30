"""SYNTHETIC ONLY: versioned language support, not certified issuer evidence."""
from dataclasses import replace

import pytest

from domain.fundamental_derivation import FiledFact, _derive_audit_evidence
from domain.les_v7.financial_sources import audit_proof
from domain.les_v7.quality_policy import (default_quality_policy, native_financial_safety,
    r2_quality_policy, REPORTED_SOURCE_MAPPING, financial_safety_source)
from domain.strategies.low_entry_v6_filings import ixbrl_record
from tests.strategies.test_v7_quality_policy import native_original

CUTOFF = "2026-10-12T13:30:00+08:00"


def qualified_original(basis):
    _, source = native_original("QualifiedOpinionAbstract")
    body = ("Review report: consolidated financial statements as of June 30, 2026. "
            "Basis for Qualified Conclusion: " + basis + " Responsibilities of management: "
            "preparation of financial statements free from material misstatement.")
    source = replace(source, facts=source.facts + (FiledFact("tifrs-ar:AccountantsReportBody", None,
        body, None, "2026-06-30", None, context_id="SYNTHETIC_ONLY_REPORTED_AUDIT"),))
    row = ixbrl_record("SYNTHETIC_ONLY", source)
    row["audit_evidence"] = audit_proof(source, symbol=row["symbol"], original_verified=True)
    return source, row


def safety(row, policy):
    return native_financial_safety(row, symbol=row["symbol"], market="TWSE", cutoff=CUTOFF, policy=policy)


@pytest.mark.parametrize("description", ["nonsignificant subsidiaries", "nonsignificant consolidated subsidiaries"])
def test_r3_supports_exact_unreviewed_phrase_without_changing_legacy_classification(description):
    source, row = qualified_original("Certain " + description + " and equity-method investments "
        "were based on unreviewed financial statements of those companies.")
    before = _derive_audit_evidence(source)
    assert before["audit_opinion"] == "UNKNOWN"
    assert row["audit_evidence"]["audit"]["audit_opinion"] == "UNKNOWN"
    assert safety(row, r2_quality_policy())["state"] == "UNKNOWN"
    result = safety(row, default_quality_policy())
    assert result["state"] == "PASS", result
    assert result["audit_classification"]["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert result["audit_classification"]["audit_review_required"] is False
    assert result["audit_classification"]["material_accounting_issue"] is False
    assert _derive_audit_evidence(source) == before
    assert row["audit_evidence"]["audit"]["audit_opinion"] == "UNKNOWN"


@pytest.mark.parametrize("basis,expected", [
    ("Certain nonsignificant subsidiaries had unreviewed financial statements; we were unable to obtain evidence.", "UNKNOWN"),
    ("Certain nonsignificant subsidiaries had unreviewed financial statements; inventory is materially misstated.", "FAIL"),
    ("Certain nonsignificant subsidiaries were consolidated.", "UNKNOWN"),
    ("The company uses unreviewed financial statements.", "UNKNOWN"),
])
def test_r3_language_support_is_not_a_general_qualified_opinion_bypass(basis, expected):
    _, row = qualified_original(basis)
    assert safety(row, default_quality_policy())["state"] == expected


def test_r3_does_not_require_complete_diagnostic_debt_but_still_validates_cash():
    _, source = native_original()
    source = replace(source, facts=tuple(f for f in source.facts if f.concept not in
        {"ifrs-full:Borrowings", "ifrs-full:CurrentLeaseLiabilities", "ifrs-full:NoncurrentLeaseLiabilities"}))
    row = ixbrl_record("SYNTHETIC_ONLY", source)
    row["audit_evidence"] = audit_proof(source, symbol=row["symbol"], original_verified=True)
    assert row["interest_bearing_debt"] is None
    assert safety(row, r2_quality_policy())["state"] == "UNKNOWN"
    assert safety(row, default_quality_policy())["state"] == "PASS"
    row["field_evidence"]["cash_beginning"]["content_sha256"] = "f" * 64
    assert safety(row, default_quality_policy())["state"] == "UNKNOWN"


def test_r3_manifest_is_deeply_frozen_and_r2_template_hash_is_preserved():
    with pytest.raises(TypeError):
        REPORTED_SOURCE_MAPPING["native_scoring_manifest"][0] = ("tampered", 1)
    assert r2_quality_policy()["policy_sha256"] == "6526f90f7764fa92e2e6c2281c7438fb6b937c6f26dc870fcdf583817def906a"


def test_thin_financial_packet_cannot_mutate_the_callers_original():
    _, row = qualified_original("Certain nonsignificant consolidated subsidiaries used unreviewed financial statements.")
    before = safety(row, default_quality_policy())
    packet = financial_safety_source(row)
    packet["audit_evidence"]["contexts"][0]["text_value"] = "SYNTHETIC_TAMPER"
    packet["field_evidence"]["cash_beginning"]["content_sha256"] = "f" * 64
    assert safety(row, default_quality_policy()) == before
