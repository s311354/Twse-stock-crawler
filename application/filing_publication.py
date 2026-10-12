"""Publication reconstruction and comparability audit, without changing models.

Source semantics require an explicit reviewer attestation. A hash, a current
catalog, or a board approval date alone cannot establish historical vintage.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import fields

from application.replay_contracts import aware_timestamp, digest
from application.research_evidence import filing_binding, verified_document, verify_publication, select_filing_versions
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.strategies.low_entry_v6_filings import ixbrl_record
from domain.strategies.low_entry_v6_quality import (BALANCE_FIELDS, CAPITAL_BALANCE_FIELDS, FLOW_FIELDS, evaluate_quality,
                                                  financial_field_checks, verification_diagnostics)

PROOF_SCHEMA = "filing-publication/v2"
REVIEW_CHECKS = ("company_period_scope", "official_document_identity", "revision_identity",
                 "content_version_match", "revision_history")
NEXT_ACTION = {
    "ORIGINAL_NOT_ACQUIRED": "ACQUIRE_OFFICIAL_ORIGINAL_WITH_RECEIPT",
    "DOCUMENT_PENDING_PUBLICATION_PROOF": "COLLECT_OFFICIAL_PUBLICATION_AND_REVISION_RECORDS",
    "PUBLICATION_VERSION_UNVERIFIED": "REVIEW_EXACT_DOCUMENT_VERSION_LINK",
    "PUBLISHED_AFTER_CUTOFF": "USE_PROVEN_PREDECESSOR_OR_LATER_CUTOFF",
    "ORIGINAL_VERSION_UNRECOVERABLE": "REQUEST_OFFICIAL_ARCHIVED_VERSION_NOT_CURRENT_REPLACEMENT",
    "VERSION_CONFLICT": "RESOLVE_OFFICIAL_VERSION_CONFLICT",
    "PARSE_FAILED": "REVIEW_ORIGINAL_AND_PARSER",
    "VERIFIED_AVAILABLE": "CHECK_COMPARABILITY_AND_NATIVE_GATES",
    "OBSERVED_AVAILABLE": "KEEP_OBSERVATION_BOUNDARY_NO_EARLIER_BACKFILL",
    "POLICY_AVAILABLE": "ASSUMPTION_ONLY_COLLECT_EXACT_IXBRL_VERSION_PROOF",
    "POLICY_AFTER_CUTOFF": "PDF_UPLOAD_POLICY_NOT_AVAILABLE_AT_CUTOFF",
}


def verify_filing_publication(symbol, item, *, document_reader=None, _depth=0) -> str:
    """Verify the existing contract, plus v2 exact-version/revision evidence.

The legacy reviewed proof remains readable. New repair imports require v2.
Every edge is a reviewed assertion backed by preserved official source bytes;
the software verifies integrity/linkage, not a cryptographic publisher signature.
"""
    if _depth > 16:
        raise ValueError("REVISION_CHAIN_TOO_DEEP")
    proof = item["publication_proof"]
    if proof.get("status") == "CONFLICT":
        raise ValueError("CONFLICTING_PUBLICATION_PROOFS")
    if proof.get("schema") not in {None, PROOF_SCHEMA}:
        raise ValueError("UNSUPPORTED_FILING_PUBLICATION_SCHEMA")
    available = verify_publication(proof, binding=filing_binding(symbol, item),
                                   document_reader=document_reader)
    if proof.get("schema") != PROOF_SCHEMA:
        original = proof.get("original_published_at")
        if item.get("revision_number", 1) > 1 and (not original or
                aware_timestamp(original) > aware_timestamp(available)):
            raise ValueError("REVISION_PUBLICATION_ORDER_UNVERIFIED")
        return available
    identity = proof["version_identity"]
    if (not identity.get("official_document_id") or not identity.get("revision_id")
            or identity.get("revision_number") != item.get("revision_number", 1)
            or identity.get("kind") not in {"ORIGINAL", "REVISION"}
            or not all(proof.get("review_checks", {}).get(k) is True for k in REVIEW_CHECKS)):
        raise ValueError("VERSION_IDENTITY_REVIEW_REQUIRED")
    if item.get("period_end") and available[:10] < item["period_end"]:
        raise ValueError("PUBLICATION_PREDATES_REPORTING_PERIOD")
    reviewed = aware_timestamp(proof["verified_at"])
    if reviewed < aware_timestamp(proof["document"]["retrieved_at"]):
        raise ValueError("REVIEW_PREDATES_SOURCE_RETRIEVAL")
    chain = proof.get("version_chain", [])
    if not chain or chain[0]["from_id"] != identity["official_document_id"]:
        raise ValueError("VERSION_CHAIN_REQUIRED")
    node, visited = identity["official_document_id"], set()
    for edge in chain:
        if edge["from_id"] != node or node in visited or not edge.get("to_id"):
            raise ValueError("BROKEN_VERSION_CHAIN")
        # Reuse source/quote validation without pretending an edge has a release date.
        body = verified_document(edge["document"], document_reader=document_reader)
        text = edge["document"].get("extracted_text", body.decode("utf-8", errors="replace"))
        if (not edge.get("quotes") or any(not q or q not in text for q in edge["quotes"])
                or ("extracted_text" in edge["document"] and not edge.get("extraction_reviewed"))
                or reviewed < aware_timestamp(edge["document"]["retrieved_at"])):
            raise ValueError("VERSION_CHAIN_SOURCE_UNVERIFIED")
        visited.add(node)
        node = edge["to_id"]
    if node != item["content_sha256"] or identity["revision_id"] not in visited | {node}:
        raise ValueError("VERSION_CHAIN_CONTENT_OR_REVISION_MISMATCH")
    if chain[-1]["document"]["sha256"] != item["content_sha256"]:
        # Direct hash attestation is permitted only when the official source
        # actually quotes that hash, not merely the import JSON's assertion.
        if item["content_sha256"] not in " ".join(chain[-1]["quotes"]):
            raise ValueError("VERSION_CHAIN_NOT_BOUND_TO_ORIGINAL")
    predecessor = proof.get("predecessor")
    if identity["kind"] == "ORIGINAL":
        if item.get("revision_number", 1) > 1 or predecessor:
            raise ValueError("REVISION_CANNOT_BORROW_ORIGINAL_DATE")
    else:
        if not predecessor:
            raise ValueError("REVISION_PREDECESSOR_PROOF_REQUIRED")
        previous_binding = predecessor["binding"]
        current_binding = filing_binding(symbol, item)
        if (any(previous_binding.get(k) != current_binding[k] for k in
                ("symbol", "fiscal_year", "quarter", "report_id", "statement_scope"))
                or previous_binding.get("content_sha256") == item["content_sha256"]
                or identity.get("supersedes_sha256") != previous_binding.get("content_sha256")):
            raise ValueError("REVISION_PREDECESSOR_IDENTITY_MISMATCH")
        if predecessor.get("schema") != PROOF_SCHEMA:
            raise ValueError("REVISION_PREDECESSOR_REQUIRES_V2_PROOF")
        previous_item = dict(previous_binding, revision_number=predecessor["version_identity"]["revision_number"],
                             receipt=dict(report_id=previous_binding["report_id"]), publication_proof=predecessor)
        previous_at = verify_filing_publication(symbol, previous_item, document_reader=document_reader, _depth=_depth+1)
        if aware_timestamp(previous_at) >= aware_timestamp(available):
            raise ValueError("REVISION_PUBLICATION_ORDER_UNVERIFIED")
        if proof.get("original_published_at") != predecessor.get("original_published_at", predecessor["published_at"]):
            raise ValueError("ORIGINAL_PUBLICATION_MISMATCH")
    return available


def comparability(symbol, item, facts, available):
    """Expose native adapter inputs/context provenance; do not invent ratios."""
    names = {f.name for f in fields(FiledFinancials)} - {"facts", "available_at"}
    filing = FiledFinancials(**{k: v for k, v in item.items() if k in names}, available_at=available,
                            facts=tuple(FiledFact(**f) for f in facts))
    record = ixbrl_record(symbol, filing)
    import pandas as pd
    period = pd.Period(year=filing.fiscal_year, quarter=filing.quarter, freq="Q")
    checks = financial_field_checks({period: record}, {period}, symbol=symbol)
    missing = [r["field"] for r in checks if r["status"] == "UNKNOWN"]
    contexts = [dict(concept=f["concept"], context_id=f.get("context_id"),
                     start_date=f.get("start_date"), end_date=f.get("end_date"), instant=f.get("instant"),
                     unit=f.get("unit"), dimensions=f.get("dimensions", ())) for f in facts
                if f.get("numeric_value") is not None]
    return dict(status="FIELDS_MISSING_OR_INCOMPARABLE" if missing else "CONTEXTS_RESOLVED",
                missing_fields=missing, record=record, contexts=contexts, field_checks=checks,
                required_field_coverage=(len(checks)-len(missing))/len(checks),
                scope=item["statement_scope"], flow_basis=record["flow_basis"],
                revision=item["content_sha256"], source_url=item.get("source_url"))


def audit_filings(payload, symbols, as_of, *, parse_filing, document_reader=None):
    """Keep publication eligibility separate from parsing and model completeness."""
    from dataclasses import asdict
    cutoff = aware_timestamp(as_of)
    if not set(symbols).issubset(payload["symbols"]):
        raise ValueError("PUBLICATION_AUDIT_SYMBOL_OUTSIDE_DATASET")
    rows = []
    for symbol in symbols:
        for item in payload.get("filings", {}).get(symbol, []):
            proof = item.get("publication_proof") or {}
            row = dict(symbol=symbol, period=f"{item['fiscal_year']}Q{item['quarter']}",
                       fiscal_year=item['fiscal_year'], quarter=item['quarter'],
                       content_sha256=item["content_sha256"], source_version=item["source_version"],
                       revision_number=item["revision_number"], statement_scope=item["statement_scope"],
                       original_present=False, published_at=proof.get("published_at"),
                       original_published_at=proof.get("original_published_at", proof.get("published_at")),
                       revision_published_at=proof.get("published_at") if proof.get("version_identity", {}).get("kind") == "REVISION" else None,
                       publication_precision=proof.get("publication_precision"),
                       verified_at=proof.get("verified_at"), fetched_at=item["receipt"]["fetched_at"],
                       stored_available_at=item["available_at"], stored_availability_source=item["availability_source"],
                       proof_status=proof.get("status", "MISSING"), version_binding_status="UNVERIFIED",
                       publication_proof_sha256=digest(proof) if proof else None,
                       version_identity=proof.get("version_identity"), source_url=item.get("source_url"),
                       effective_available_at=None, eligible_at_cutoff=False, selected=False,
                       comparability=dict(status="NOT_EVALUATED"), reason=None)
            try:
                original = payload.get("raw_documents", {}).get(item["content_sha256"])
                if original is None:
                    raise ValueError("ORIGINAL_NOT_ACQUIRED")
                content = verified_document(original, document_reader=document_reader)
                if original["sha256"] != item["content_sha256"]:
                    raise ValueError("FILING_ORIGINAL_HASH_MISMATCH")
                row["original_present"] = True
            except (ValueError, KeyError, TypeError, OSError) as error:
                row.update(status="ORIGINAL_NOT_ACQUIRED", reason=str(error))
                rows.append(row)
                continue
            observed = max(aware_timestamp(item["available_at"]), aware_timestamp(item["receipt"]["fetched_at"]))
            available = observed.isoformat()
            policy = item.get("publication_time_policy")
            if item.get("publication_proof") is not None:
                try:
                    available = verify_filing_publication(symbol, item, document_reader=document_reader)
                    row.update(version_binding_status="VERIFIED", effective_available_at=available)
                    row["status"] = "VERIFIED_AVAILABLE" if aware_timestamp(available) <= cutoff else "PUBLISHED_AFTER_CUTOFF"
                except (ValueError, KeyError, TypeError) as error:
                    row.update(status="VERSION_CONFLICT" if proof.get("status") == "CONFLICT" else
                               "PUBLICATION_VERSION_UNVERIFIED", reason=str(error))
            elif policy:
                from application.pdf_upload_policy import verify_pdf_upload_policy
                try:
                    assessed = verify_pdf_upload_policy(symbol, item, document_reader=document_reader)
                    available = assessed["available_at"]
                    row.update(publication_time_policy=assessed, proof_status="USER_ACCEPTED_ASSUMPTION",
                               version_binding_status="ASSUMED_NOT_VERIFIED", effective_available_at=available,
                               status="POLICY_AVAILABLE" if aware_timestamp(available) <= cutoff else "POLICY_AFTER_CUTOFF")
                except (ValueError, KeyError, TypeError) as error:
                    row.update(status="PUBLICATION_VERSION_UNVERIFIED", reason=str(error))
            else:
                row.update(status="OBSERVED_AVAILABLE" if observed <= cutoff else "DOCUMENT_PENDING_PUBLICATION_PROOF",
                           effective_available_at=available)
            row["eligible_at_cutoff"] = row["status"] in {"VERIFIED_AVAILABLE", "OBSERVED_AVAILABLE", "POLICY_AVAILABLE"}
            if item.get("period_end") and item["period_end"] > cutoff.date().isoformat():
                row.update(eligible_at_cutoff=False, reason="FUTURE_FISCAL_PERIOD")
            try:
                facts = [asdict(f) for f in parse_filing(content)]
                if not facts:
                    raise ValueError("NO_XBRL_FACTS")
                from domain.financial_taxonomy import ixbrl_schema_refs
                row["comparability"] = comparability(symbol, dict(item,
                    taxonomy_schema_refs=ixbrl_schema_refs(content)), facts, available)
            except (ValueError, KeyError, TypeError) as error:
                row.update(status="PARSE_FAILED", eligible_at_cutoff=False, reason=str(error))
            rows.append(row)
    financials, quality = {}, {}
    for symbol in symbols:
        selected = []
        eligible = [r for r in rows if r["symbol"] == symbol and r["eligible_at_cutoff"]]
        indices, conflicts = select_filing_versions([dict(r, available_at=r['effective_available_at']) for r in eligible])
        for conflict in conflicts:
            for i in conflict['indices']:
                eligible[i].update(status="VERSION_CONFLICT", eligible_at_cutoff=False, reason=conflict['reason'])
        for i in indices:
            eligible[i]['selected'] = True
            selected.append(eligible[i]['comparability']['record'])
        financials[symbol] = verification_diagnostics(selected, symbol=symbol, as_of=cutoff, market=payload["market"])
        quality[symbol] = asdict(evaluate_quality(selected, symbol=symbol, as_of=cutoff, market=payload["market"]))
        financials[symbol]["missing_original_periods"] = [p for p in financials[symbol]["missing_periods"]
            if not any(r["symbol"] == symbol and r["period"] == p for r in rows)]
        financials[symbol]["diagnostic_missing_original_periods"] = [
            p for p in financials[symbol].get("diagnostic_missing_periods", [])
            if not any(r["symbol"] == symbol and r["period"] == p for r in rows)]
    for row in rows:
        row["next_action"] = NEXT_ACTION[row["status"]]
        local = row["comparability"].get("field_checks", [])
        if not local:
            category = "MISSING_DATA" if row["status"] == "ORIGINAL_NOT_ACQUIRED" and str(row["reason"]).startswith((
                "ORIGINAL_NOT_ACQUIRED", "SOURCE_DOCUMENT_UNAVAILABLE")) else "PARSE_FAILED" if row["status"] == "PARSE_FAILED" else "PENDING_VERIFICATION"
            local = [dict(symbol=row["symbol"], period=row["period"], field=field,
                          status="UNKNOWN", category=category, reason=row["reason"] or row["status"], value=None)
                     for field in (*FLOW_FIELDS, *BALANCE_FIELDS, *CAPITAL_BALANCE_FIELDS)]
        row["field_checks"] = []
        for check in local:
            check = dict(check, content_sha256=row["content_sha256"], source_url=row["source_url"],
                         publication_status=row["status"], selected=row["selected"])
            if not row["eligible_at_cutoff"] and row["original_present"] and row["status"] != "PARSE_FAILED":
                check.update(status="UNKNOWN", category="PENDING_VERIFICATION", reason=row["reason"] or row["status"],
                             parsed_value=check.get("value"), value=None)
            row["field_checks"].append(check)
    # A missing SELECTED quarter is not necessarily an absent original. Explain
    # blocked candidates instead of labelling publication gaps as download gaps.
    for symbol, financial in financials.items():
        financial["missing_period_reasons"] = {}
        financial["diagnostic_missing_period_reasons"] = {}
        core_periods = set(financial["missing_periods"])
        diagnostic_periods = set(financial.get("diagnostic_missing_periods", []))
        for period in sorted(core_periods | diagnostic_periods):
            candidates = [r for r in rows if r["symbol"] == symbol and r["period"] == period]
            if not candidates:
                detail = dict(category="MISSING_DATA", reason="NO_ORIGINAL_FILING_IN_DATASET", candidates=[])
            else:
                statuses = {r["status"] for r in candidates}
                category = ("MISSING_DATA" if statuses == {"ORIGINAL_NOT_ACQUIRED"} and all(
                    c["category"] == "MISSING_DATA" for r in candidates for c in r["field_checks"])
                else "PARSE_FAILED" if statuses == {"PARSE_FAILED"} else "PENDING_VERIFICATION")
                detail = dict(category=category, reason="NO_USABLE_VERSION_AT_CUTOFF",
                    candidates=[{k: r[k] for k in ("content_sha256", "status", "reason", "effective_available_at", "source_url")} for r in candidates])
            if period in core_periods:
                financial["missing_period_reasons"][period] = detail
            if period in diagnostic_periods:
                financial["diagnostic_missing_period_reasons"][period] = detail
            for check in financial["field_checks"]:
                if check["period"] == period:
                    check.update(category=detail["category"], reason=detail["reason"], version_candidates=detail["candidates"])
        financial["field_gaps"] = [c for c in financial["field_checks"] if c["status"] == "UNKNOWN"]
    # Inventory includes rejected originals, not just survivors of the gate.
    gaps = [r for f in financials.values() for r in f["field_gaps"]]
    for row in rows:
        if not row["selected"]:
            gaps.extend(r for r in row["field_checks"] if r["status"] == "UNKNOWN")
    return dict(schema="filing-publication-audit/v1", field_audit_schema="financial-field-audit/v1",
                dataset_id=digest(payload), as_of=cutoff.isoformat(),
                status="PARTIAL" if any(r["status"] == "POLICY_AVAILABLE" or not r["eligible_at_cutoff"] or
                    r["comparability"]["status"] != "CONTEXTS_RESOLVED" for r in rows) or
                    any(f["missing_periods"] or f["unresolved_fields"] or
                        f.get("diagnostic_missing_periods") or f.get("diagnostic_unresolved_fields")
                        for f in financials.values()) else "COMPLETE",
                rows=rows, financials=financials, native_quality=quality, field_gaps=gaps,
                gap_counts=dict(Counter(r["category"] for r in gaps)),
                gap_count_unit="FIELD_PERIOD_VERSION_CHECK_NOT_UNIQUE_MISSING_DOCUMENTS",
                counts=dict(Counter(r["status"] for r in rows)),
                scope="FINANCIALS_ONLY_NOT_DATASET_OR_MODEL_COMPLETENESS")


def render_filing_audit(report):
    """Human-readable audit artifact, not an extra trading-report UI panel."""
    labels = {"MISSING_DATA": "缺資料", "PARSE_FAILED": "解析失敗",
              "INCOMPARABLE": "不可比較", "PENDING_VERIFICATION": "待核實"}
    def cell(value):
        return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lines = ["# 財報逐欄缺口稽核", "", f"Dataset: `{report['dataset_id']}`",
             f"Cutoff: `{report['as_of']}`", f"Status: `{report['status']}`", "",
             "本摘要只稽核財報證據；Quality 結果不代表 LES 進場、流動性或交易限制已通過。", "",
             "| 股票 | 原生 Quality | 品質分數 | Data Confidence |", "| --- | --- | --- | --- |"]
    for symbol, quality in sorted(report["native_quality"].items()):
        lines.append("| " + " | ".join(cell(v) for v in (symbol, quality["status"], quality["score"], quality["confidence"])) + " |")
    lines += ["", "## 缺口與修復方向", "", "同一季度的不同原文版本與無可用版本的缺口分開列示；數量不是缺失文件數。", "",
              "| 股票 | 季度 | 欄位 | 分類 | 原因 | 原文 SHA256 |", "| --- | --- | --- | --- | --- | --- |"]
    for gap in report["field_gaps"]:
        lines.append("| " + " | ".join(cell(v) for v in (gap["symbol"], gap["period"], gap["field"],
            labels[gap["category"]], gap["reason"], gap.get("content_sha256"))) + " |")
    lines += ["", "## 驗收政策", "", "原文有值但漏讀：修解析；未取得原文：沿用 bounded filing-collect 補收；內容、公布時間或修訂基礎未核實：保持 UNKNOWN。",
              "單季金額優先讀取原文 exact context；僅同份文件、同口徑、同單位的累計金額可安全相減。跨文件基礎未證實不相減；EPS 不以累計相減，資產負債表使用時點值。",
              "修復必須保留原文、雜湊、時間與版本鏈；驗收不以 BUY／BUY_SMALL 或獲利數量為標準。", ""]
    return "\n".join(lines)


def apply_reviews(payload, reviews, *, document_reader=None):
    """Append reviewed proofs to a COPY; preserve receipts and stored availability."""
    result = deepcopy(payload)
    changes = []
    for review in reviews:
        binding = review.get("binding", {})
        candidates = [r for r in result["filings"].get(binding.get("symbol"), [])
                      if filing_binding(binding["symbol"], r) == binding]
        if len(candidates) != 1 or review.get("schema") != PROOF_SCHEMA:
            raise ValueError("REPAIR_REQUIRES_V2_EXACT_EXISTING_FILING_BINDING")
        row = candidates[0]
        verified = dict(row, publication_proof=review)
        verify_filing_publication(binding["symbol"], verified, document_reader=document_reader)
        old = row.get("publication_proof")
        if old and old != review:
            # Do not choose an earlier/better assertion or overwrite a conflict.
            alternatives = old.get("candidates", [old]) if old.get("status") == "CONFLICT" else [old]
            alternatives = {digest(p): p for p in [*alternatives, review]}
            row["publication_proof"] = dict(status="CONFLICT", candidates=[alternatives[k] for k in sorted(alternatives)])
        else:
            row["publication_proof"] = deepcopy(review)
        changes.append(dict(binding=binding, before_proof_sha256=digest(old),
                            after_proof_sha256=digest(row["publication_proof"]), reason="OFFICIAL_PUBLICATION_RECONSTRUCTION"))
    return result, changes
