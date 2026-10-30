"""SYNTHETIC ONLY: signed-source policy tests, not issuer/signature certification."""
from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.signed_audit import (
    ANCHORS, SCHEMA, POLICY, validate_signed_source, audit_proof_r6, report_body_ties,
)
from domain.les_v7.quality_policy import r5_quality_policy, r6_quality_policy, native_financial_safety
from domain.les_v7.quality_reported import evaluate_financials_reported, validate_reported_result
from infrastructure.fundamentals.signed_audit import SignedAuditStore
from tests.fundamentals.test_v7_audit_cash_r5 import original, repair, QUALIFIED_BODY, ENGLISH

CUTOFF = "2026-10-14T10:00:00Z"


def fixture():
    row, filing = original(english=ENGLISH.replace("June 30", "March 31"))
    row = repair(row, filing)
    pages = {"1": "公司 股票代碼 SYNTHETIC_ONLY",
             "4": QUALIFIED_BODY + "\n會計師 SyntheticSigner\n中華民國115年8月10日"}
    anchors = {}
    for index, (field, (code, label, basis)) in enumerate(ANCHORS.items(), 6):
        value = row[field]
        tail = f"{value:.0f}" if basis == "INSTANT" or field == "cfo" else f"1 1 2 2 {value:.0f} 3 4 4"
        quote = f"{code} {label} {tail}"
        pages[str(index)] = f"公司 合併財務報表 民國115年1月1日至6月30日 單位：新台幣仟元\n{quote}"
        anchors[field] = dict(page=index, quote=quote, basis=basis,
            value_index=0 if basis == "INSTANT" or field == "cfo" else 4, value=value)
    content = b"%PDF-SYNTHETIC_ONLY_NOT_A_REAL_REPORT"
    sha = hashlib.sha256(content).hexdigest()
    packet = dict(schema=SCHEMA, policy=POLICY, symbol=row["symbol"], market="TWSE", scope=row["scope"],
        currency="TWD", unit="thousand", period_end=row["period_end"], filing_sha256=row["content_sha256"],
        source_url="https://synthetic.invalid/SYNTHETIC_ONLY.pdf", reviewed_issuer_host="synthetic.invalid", issuer_name="公司",
        fetched_at="2026-10-14T09:20:00Z", verified_at="2026-10-14T09:21:00Z", available_at="2026-10-14T09:21:00Z",
        availability_basis="ACTUAL_FETCH_AND_SOURCE_TIEOUT_NO_HISTORICAL_BACKDATE", pdf_sha256=sha,
        body_ref=dict(schema="research-document-ref/v1", sha256=sha, size_bytes=len(content)),
        signature_review=dict(method="VISUAL_SIGNATURES_AND_STAMPS_NOT_CRYPTOGRAPHIC_CERTIFICATION", pdf_sha256=sha,
            reviewer="SYNTHETIC_ONLY", reviewed_at="2026-10-14T09:21:00Z", signed_date="2026-08-10", signers=["SyntheticSigner"], pages=[4]),
        auditor_report_date="2026-08-10", pages=pages, extracted_pages_sha256=identity(pages), audit_pages=[4], financial_anchors=anchors)
    packet["source_id"] = identity(packet)
    return row, filing, packet, content


def reseal(packet):
    packet["extracted_pages_sha256"] = identity(packet["pages"])
    packet["source_id"] = identity({k: v for k, v in packet.items() if k != "source_id"})
    return packet


def test_signed_precedence_recomputes_native_quality_preserving_conflict_and_qualification():
    row, filing, packet, _ = fixture()
    before = deepcopy(row)
    proof = audit_proof_r6(filing, row, packet, symbol=row["symbol"], cutoff=CUTOFF)
    fixed = dict(row, audit_evidence_r6=proof)
    old = evaluate_financials_reported([row], symbol=row["symbol"], market="TWSE", cutoff=CUTOFF, policy=r5_quality_policy())
    new = evaluate_financials_reported([fixed], symbol=row["symbol"], market="TWSE", cutoff=CUTOFF, policy=r6_quality_policy())
    assert old["financial_safety"]["state"] == "UNKNOWN"
    assert new["financial_safety"]["state"] == "PASS", new["reason_codes"]
    assert new["financial_safety"]["audit_classification"]["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert new["financial_safety"]["source_available_at"] == "2026-10-14T09:21:00+00:00"
    assert proof["retained_translation_conflicts"][0]["context"]["text_value"] == ENGLISH.replace("June 30", "March 31")
    assert new["quality_score"] == old["quality_score"]
    assert new["data_confidence"] == old["data_confidence"]
    validate_reported_result(new)
    assert row == before


def test_missing_signed_evidence_keeps_the_original_conflict_unknown():
    row, *_ = fixture()
    result = native_financial_safety(row, symbol=row["symbol"], market="TWSE", cutoff=CUTOFF, policy=r6_quality_policy())
    assert result["state"] == "UNKNOWN"


def test_frozen_native_evidence_matches_mutable_input_without_pollution():
    from application.evidence_context import ReadOnlyEvidence
    row, filing, packet, _ = fixture()
    proof = audit_proof_r6(filing, row, packet, symbol=row["symbol"], cutoff=CUTOFF)
    fixed = dict(row, audit_evidence_r6=proof)
    frozen = ReadOnlyEvidence(fixed)
    expected = evaluate_financials_reported([fixed], symbol=row["symbol"], market="TWSE", cutoff=CUTOFF, policy=r6_quality_policy())
    actual = evaluate_financials_reported([frozen], symbol=row["symbol"], market="TWSE", cutoff=CUTOFF, policy=r6_quality_policy())
    assert identity(actual) == identity(expected)
    validate_reported_result(actual)
    assert identity(frozen) == identity(fixed)


def test_additional_hard_risk_in_signed_pdf_cannot_be_lost_as_an_inserted_line():
    row, filing, packet, _ = fixture()
    packet["pages"]["4"] += "\n繼續經營重大不確定性\n本公司有重大繼續經營風險。"
    reseal(packet)
    proof = audit_proof_r6(filing, row, packet, symbol=row["symbol"], cutoff=CUTOFF)
    assert native_financial_safety(dict(row, audit_evidence_r6=proof), symbol=row["symbol"], market="TWSE",
        cutoff=CUTOFF, policy=r6_quality_policy())["state"] == "FAIL"


@pytest.mark.parametrize("damage", ["symbol", "scope", "filing", "currency", "time", "date", "signer", "anchors", "column", "value", "unit", "report_period", "hash"])
def test_rehashed_packet_cannot_bypass_item_by_item_binding(damage):
    row, filing, packet, _ = fixture()
    if damage == "symbol": packet["symbol"] = "OTHER_SYNTHETIC"
    elif damage == "scope": packet["scope"] = "individual"
    elif damage == "filing": packet["filing_sha256"] = "f" * 64
    elif damage == "currency": packet["currency"] = "USD"
    elif damage == "time": packet["fetched_at"] = "2026-10-14T11:00:00Z"
    elif damage == "date": packet["auditor_report_date"] = "2026-08-11"
    elif damage == "signer": packet["signature_review"]["signers"] = ["MissingSigner"]
    elif damage == "anchors": packet["financial_anchors"].pop("cfo")
    elif damage == "column": packet["financial_anchors"]["revenue"]["value_index"] = 0
    elif damage == "value": packet["financial_anchors"]["net_income"]["value"] += 1
    elif damage == "unit": packet["pages"]["6"] = packet["pages"]["6"].replace("仟元", "元")
    elif damage == "report_period": packet["pages"]["6"] = packet["pages"]["6"].replace("6月30", "3月31")
    elif damage == "hash": packet["signature_review"]["pdf_sha256"] = "a" * 64
    reseal(packet)
    with pytest.raises(ValueError, match="SIGNED_REPORT"):
        audit_proof_r6(filing, row, packet, symbol=row["symbol"], cutoff=CUTOFF)


def test_late_corroboration_is_not_backdated_and_policy_is_cutoff_bound():
    row, filing, packet, _ = fixture()
    for cutoff in ("2026-08-14T10:00:00Z", "2026-10-14T09:20:30Z"):
        with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
            audit_proof_r6(filing, row, packet, symbol=row["symbol"], cutoff=cutoff)
    # Isolate evidence timing from the finalized policy's activation timing.
    from domain.les_v7.quality_policy import validate_quality_policy
    cutoff = "2026-10-14T09:54:59Z"
    validate_quality_policy(r6_quality_policy(), cutoff)
    packet.update(fetched_at="2026-10-14T09:54:00Z", verified_at="2026-10-14T09:55:00Z", available_at="2026-10-14T09:55:00Z")
    packet["signature_review"]["reviewed_at"] = packet["verified_at"]
    reseal(packet)
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        audit_proof_r6(filing, row, packet, symbol=row["symbol"], cutoff=cutoff)


def test_body_matching_cannot_delete_or_replace_material_amounts():
    body = QUALIFIED_BODY + "營收123456，資產234567。"
    assert report_body_ties(body, "PDF page heading\n" + body + "page footer")
    assert not report_body_ties(body, body.replace("123456", "123457"))
    assert not report_body_ties(body, body.replace("營收123456，", ""))


def test_source_precedence_cannot_override_a_retained_translation_hard_risk():
    row, filing, packet, _ = fixture()
    facts = tuple(replace(f, text_value=f.text_value + "\nMaterial uncertainty related to going concern\nThere is material uncertainty about going concern.")
                  if f.concept.endswith("AccountantsReportBody") and f.text_value.startswith("Introduction") else f for f in filing.facts)
    filing = replace(filing, facts=facts)
    proof = audit_proof_r6(filing, row, packet, symbol=row["symbol"], cutoff=CUTOFF)
    assert proof["audit"]["going_concern"] is True
    assert proof["reason"] == "ORIGINAL_FINANCIAL_HARD_RISK_RETAINED"
    assert native_financial_safety(dict(row, audit_evidence_r6=proof), symbol=row["symbol"], market="TWSE",
        cutoff=CUTOFF, policy=r6_quality_policy())["state"] == "FAIL"


def test_store_is_immutable_rechecks_pdf_bytes_and_extraction_and_never_creates_early_receipt(tmp_path, monkeypatch):
    import infrastructure.fundamentals.signed_audit as module
    row, _, packet, content = fixture()
    pages = [packet["pages"].get(str(i), "") for i in range(1, 13)]
    monkeypatch.setattr(module, "extract_pages", lambda content: pages)
    store = SignedAuditStore(tmp_path)
    published = store.publish(content, packet, row, cutoff=CUTOFF)
    assert store.publish(content, packet, row, cutoff=CUTOFF) == published
    assert store.select(row, cutoff="2026-10-14T09:20:59Z") is None
    assert store.select(row, cutoff=CUTOFF) == published
    monkeypatch.setattr(module, "extract_pages", lambda content: ["CORRUPT"] * 12)
    with pytest.raises(ValueError, match="EXTRACTION_DOES_NOT_MATCH"):
        store.select(row, cutoff=CUTOFF)
    path = store.documents.objects.path(published["pdf_sha256"])
    path.write_bytes(b"%PDF-SYNTHETIC_ONLY_TAMPER")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        store.select(row, cutoff=CUTOFF)
