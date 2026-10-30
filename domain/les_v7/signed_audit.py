"""R6 signed-report precedence, not a Chinese-only or issuer-specific exception.

The source port verifies retained PDF bytes and extraction. This pure layer
rechecks the source binding, full report-body tie-out and financial anchors.
Original translations, qualification and hard risks remain in the proof.
"""
from collections.abc import Mapping
from dataclasses import asdict, replace
from difflib import SequenceMatcher
import re
from urllib.parse import urlsplit

import pandas as pd

from domain.fundamental_derivation import _AUDIT_CONCEPTS
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.financial_sources_r5 import (
    _BODY, _body_bindings_r5, classify_audit_contexts_r5, audit_proof_r5,
)
from domain.strategies.low_entry_v6_quality_evidence import validate_field
from domain.text_normalization import normalize_cjk_text

POLICY = "SOURCE_BOUND_SIGNED_REPORT_PRECEDENCE_R1"
AVAILABLE_AT = "2026-10-14T09:53:30Z"  # Post-verification activation, never a historical deployment claim.
SCHEMA = "signed-financial-report/v1"
ANCHORS = {
    "total_assets": ("1XXX", "資產總計", "INSTANT"),
    "total_liabilities": ("2XXX", "負債總計", "INSTANT"),
    "total_equity": ("3XXX", "權益總計", "INSTANT"),
    "revenue": ("4000", "營業收入", "YTD"),
    "operating_income": ("6900", "營業利益(損失)", "YTD"),
    "net_income": ("8200", "本期淨利(淨損)", "YTD"),
    "cfo": ("", "營業活動之淨現金流入(流出)", "YTD"),
}


def compact(text):
    # Only typographic normalization, never numerical/date substitution.
    return re.sub(r"\s+", "", normalize_cjk_text(text)).translate(
        str.maketrans({"（": "(", "）": ")", ",": "，"}))


def report_body_ties(body, pdf_text):
    """Every original character must tie in order; only PDF extras are allowed.

    Page headers/footers may insert text, but no original deletion/replacement
    or numerical rounding is accepted. The complete PDF report is separately
    classified so an additional PDF hard risk is never lost in those extras.
    """
    left, right = compact(body), compact(pdf_text)
    if len(left) < 80 or len(right) > len(left) * 1.5:
        return False
    return all(tag in {"equal", "insert"} for tag, *_ in
               SequenceMatcher(None, left, right, autojunk=False).get_opcodes())


def validate_signed_source(packet, row, *, cutoff):
    if not isinstance(packet, Mapping) or packet.get("schema") != SCHEMA or packet.get("policy") != POLICY:
        raise ValueError("SIGNED_REPORT_POLICY_OR_SCHEMA_INVALID")
    body = {key: value for key, value in packet.items() if key != "source_id"}
    if identity(body) != packet.get("source_id"):
        raise ValueError("SIGNED_REPORT_CONTENT_ID_INVALID")
    for key in ("symbol", "market", "scope", "currency", "unit", "period_end"):
        if packet.get(key) != row.get(key):
            raise ValueError("SIGNED_REPORT_BINDING_INVALID:" + key)
    if packet.get("filing_sha256") != row.get("content_sha256"):
        raise ValueError("SIGNED_REPORT_FILING_VERSION_MISMATCH")
    fetched, verified = timestamp(packet["fetched_at"]), timestamp(packet["verified_at"])
    if (verified < fetched or timestamp(packet["available_at"]) != verified
            or verified > timestamp(cutoff) or timestamp(AVAILABLE_AT) > timestamp(cutoff)
            or timestamp(row["available_at"]) > timestamp(cutoff)):
        raise ValueError("SIGNED_REPORT_NOT_AVAILABLE_AT_CUTOFF")
    if packet.get("availability_basis") != "ACTUAL_FETCH_AND_SOURCE_TIEOUT_NO_HISTORICAL_BACKDATE":
        raise ValueError("SIGNED_REPORT_AVAILABILITY_UNVERIFIED")
    ref = packet.get("body_ref", {})
    if (ref.get("schema") != "research-document-ref/v1" or ref.get("sha256") != packet.get("pdf_sha256")
            or not isinstance(ref.get("size_bytes"), int) or ref["size_bytes"] <= 0):
        raise ValueError("SIGNED_REPORT_ORIGINAL_REFERENCE_INVALID")
    url = urlsplit(packet["source_url"])
    if (url.scheme != "https" or url.hostname != packet.get("reviewed_issuer_host")
            or url.username or url.password or not url.path.endswith(".pdf")):
        raise ValueError("SIGNED_REPORT_ISSUER_SOURCE_INVALID")
    visual = packet.get("signature_review", {})
    if (visual.get("method") != "VISUAL_SIGNATURES_AND_STAMPS_NOT_CRYPTOGRAPHIC_CERTIFICATION"
            or visual.get("pdf_sha256") != packet["pdf_sha256"] or not visual.get("reviewer")
            or visual.get("signed_date") != packet.get("auditor_report_date")
            or not visual.get("signers") or not visual.get("pages")
            or not set(visual["pages"]).issubset(packet.get("audit_pages", ()))
            or timestamp(visual["reviewed_at"]) > verified or timestamp(visual["reviewed_at"]) < fetched):
        raise ValueError("SIGNED_REPORT_SIGNATURE_REVIEW_UNVERIFIED")
    pages = packet["pages"]
    if not isinstance(pages, Mapping) or identity(pages) != packet.get("extracted_pages_sha256"):
        raise ValueError("SIGNED_REPORT_EXTRACTION_INVALID")
    cover = compact(pages["1"])
    if compact(packet["issuer_name"]) not in cover or str(packet["symbol"]) not in cover:
        raise ValueError("SIGNED_REPORT_ISSUER_IDENTITY_MISMATCH")
    audit = "\n".join(pages[str(p)] for p in packet["audit_pages"])
    signed = pd.Timestamp(packet["auditor_report_date"])
    if (signed.date() > pd.Timestamp(fetched).tz_convert("Asia/Taipei").date()
            or f"{signed.year - 1911}年{signed.month}月{signed.day}日" not in compact(audit)):
        raise ValueError("SIGNED_REPORT_SIGNATURE_DATE_MISMATCH")
    if any(compact(name) not in compact(audit) for name in visual["signers"]):
        raise ValueError("SIGNED_REPORT_SIGNER_MISMATCH")
    if set(packet.get("financial_anchors", {})) != set(ANCHORS):
        raise ValueError("SIGNED_REPORT_REQUIRED_FINANCIAL_ANCHORS_MISSING")
    period = pd.Period(row["period_end"], freq="Q")
    if period.quarter == 4:
        raise ValueError("SIGNED_REPORT_ANNUAL_TABLE_LAYOUT_NOT_IMPLEMENTED")
    for field, (code, label, basis) in ANCHORS.items():
        anchor = packet["financial_anchors"][field]
        page = pages[str(anchor["page"])]
        prefix = compact(page.split(anchor["quote"])[0]) if anchor["quote"] in page else ""
        if (anchor.get("basis") != basis or compact(label) not in compact(anchor["quote"])
                or code and not anchor["quote"].lstrip().startswith(code)
                or compact("單位：新台幣仟元") not in prefix or compact(packet["issuer_name"]) not in prefix
                or ("合併" if row["scope"] == "consolidated" else "個體") not in prefix
                or f"{period.year - 1911}年" not in prefix
                or f"{period.end_time.month}月{period.end_time.day}日" not in prefix
                or basis == "YTD" and "1月1日" not in prefix):
            raise ValueError("SIGNED_REPORT_TABLE_CONTEXT_UNVERIFIED:" + field)
        # Validate both the observed table-column selection and canonical
        # original fact. Never accept a caller-provided PASS or a numeric plug.
        tail = anchor["quote"].split(label, 1)[-1]
        values = [float(token.replace(",", "").replace(" ", "").replace("$", "").replace("(", "-").replace(")", ""))
                  for token in re.findall(r"\(\s*\$?\s*\d[\d,]*\s*\)|\d[\d,]*", tail)]
        index = anchor.get("value_index")
        expected_index = 0 if basis == "INSTANT" or field == "cfo" else 4
        check = validate_field(row, field, period=period, cutoff=pd.Timestamp(timestamp(cutoff)), instant=basis == "INSTANT")
        if (type(index) is not int or index != expected_index or index >= len(values)
                or check["value"] is None or check["evidence_level"] != "CANONICAL_ORIGINAL"
                or not all(check["checks"].values()) or check["value"] != values[index]
                or anchor.get("value") != values[index]):
            raise ValueError("SIGNED_REPORT_FINANCIAL_TIEOUT_FAILED:" + field)
    return audit


def audit_proof_r6(filing, row, packet, *, symbol, market="TWSE", cutoff):
    pdf_text = validate_signed_source(packet, row, cutoff=cutoff)
    period = pd.Period(row["period_end"], freq="Q")
    signed = packet["auditor_report_date"]
    facts = tuple(f for f in filing.facts if f.concept in _AUDIT_CONCEPTS and not f.dimensions
                  and (f.end_date or f.instant) == signed)
    matching = tuple(f for f in facts if f.concept == _BODY and f.text_value
                     and report_body_ties(f.text_value, pdf_text)
                     and _body_bindings_r5((f,), period, filing.statement_scope))
    if len(matching) != 1:
        raise ValueError("SIGNED_REPORT_COMPLETE_BODY_TIEOUT_REQUIRED")
    base = tuple(f for f in facts if f.concept != _BODY)
    bindings = dict(method=POLICY, financial_period_end=row["period_end"], auditor_report_date=signed,
                    body_bindings=_body_bindings_r5(matching, period, filing.statement_scope),
                    signed_source_id=packet["source_id"])
    # This is a PDF-sourced interpretation, never a rewritten iXBRL original.
    pdf_fact = replace(matching[0], text_value=pdf_text, context_id="SIGNED_PDF_AUDIT_PAGES")
    audit = classify_audit_contexts_r5(filing, (*base, pdf_fact), bindings)
    original_audit = classify_audit_contexts_r5(filing, facts, bindings)
    original_hard_risk = (original_audit["audit_opinion"] in {"ADVERSE", "DISCLAIMER", "QUALIFIED_MATERIAL_MISSTATEMENT"}
            or original_audit["going_concern"] is True or original_audit["material_accounting_issue"] is True)
    if original_hard_risk:
        audit = dict(original_audit)  # Preserve FAIL semantics, not merely UNKNOWN.
    excluded = [dict(context=asdict(f), body_sha256=identity(f.text_value),
        reason="CONFLICTING_TRANSLATION_RETAINED_SIGNED_PDF_PRECEDENCE_NOT_ORIGINAL_CORRECTION")
        for f in facts if f.concept == _BODY and f not in matching]
    old = audit_proof_r5(filing, symbol=symbol, market=market, original_verified=True)
    return dict(old, schema="financial-audit-proof/v3", policy_version=POLICY,
        status="VERIFIED_REPORTED", reason="ORIGINAL_FINANCIAL_HARD_RISK_RETAINED" if original_hard_risk else
            "SIGNED_REPORT_FULL_BODY_AND_FINANCIAL_ANCHORS_VERIFIED",
        audit=audit, contexts=[asdict(f) for f in facts], period_binding=bindings, result_id=identity(audit),
        signed_source=packet, evidence_available_at=packet["available_at"],
        original_r5_proof_id=identity(old), retained_translation_conflicts=excluded,
        selected_original_body_sha256=identity(matching[0].text_value))
