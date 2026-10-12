"""Research-only publication audit, reviewed repair and paired native replay."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

from application.filing_publication import apply_reviews, audit_filings
from application.replay_contracts import digest
from application.research_audit import compare_results
from application.research_evidence import RAW_EVIDENCE_SCHEMA, verified_document
from infrastructure.daily_research import request_from_document
from infrastructure.filing_evidence_collection import collection_plan, classify_source
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.replay_evidence import load_evidence
from infrastructure.research_archive import create_dataset, write_once
from infrastructure.research_documents import EvidenceDocuments


def open_dataset(path):
    eid, payload = load_evidence(path)
    if payload["schema"] != RAW_EVIDENCE_SCHEMA:
        raise ValueError("PUBLICATION_RECONSTRUCTION_REQUIRES_RAW_V2")
    return eid, payload, EvidenceDocuments.for_evidence(path, eid)


def audit_dataset(path, symbols, as_of, *, collection_file=None):
    _, payload, docs = open_dataset(path)
    result = audit_filings(payload, symbols, as_of, parse_filing=parse_ixbrl, document_reader=docs.read)
    from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
    from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
    from infrastructure.replay_runtime import current_versions
    result["calculation_identity"] = dict(filing_schema=V6_FILING_SCHEMA_VERSION,
        fundamental_derivation=FUNDAMENTAL_DERIVATION_VERSION, runtime_versions=asdict(current_versions()),
        executed="NATIVE_QUALITY_ONLY_NOT_LES_EREM_PAE_OR_EXECUTION")
    result["collection_plan"] = collection_plan(payload, result)
    if collection_file:
        collection_file = Path(collection_file)
        collection = json.loads(collection_file.read_text())
        sources = EvidenceDocuments(collection_file.parent.parent / "objects")
        observations = []
        for receipt in collection["results"]:
            if "receipt_sha256" in receipt and receipt["receipt_sha256"] != digest({k:v for k,v in receipt.items() if k != "receipt_sha256"}):
                raise ValueError("COLLECTION_RECEIPT_MISMATCH")
            item = dict(request=receipt["request"], status=receipt["status"], reason=receipt.get("reason"))
            if receipt.get("document"):
                doc = receipt["document"]
                content = verified_document(doc, document_reader=sources.read)
                item.update(classify_source(receipt["request"], content, doc["url"]), document=doc,
                            captured_status=receipt["status"])
            observations.append(item)
        result["collected_sources"] = observations
        result["collection_policy"] = "CAPTURED_RECORDS_ARE_REVIEW_LEADS_NOT_VERIFIED_VERSION_BINDINGS"
    return result


def audit_difference(before, after):
    def key(row):
        return row["symbol"], row["period"], row["content_sha256"]
    old, new = ({key(r): r for r in report["rows"]} for report in (before, after))
    improved = [dict(symbol=k[0], period=k[1], content_sha256=k[2],
                     before=old[k]["status"], after=new[k]["status"])
                for k in old.keys() & new.keys() if not old[k]["eligible_at_cutoff"] and new[k]["eligible_at_cutoff"]]
    assumed = any(r.get("proof_status") == "USER_ACCEPTED_ASSUMPTION" for r in after["rows"])
    def fields(report):
        return {(symbol, r["period"], r["field"]): r for symbol, financial in report["financials"].items()
                for r in financial.get("field_checks", [])}
    old_fields, new_fields = fields(before), fields(after)
    changed_fields = [dict(symbol=k[0], period=k[1], field=k[2],
        before={f: old_fields[k].get(f) for f in ("status", "category", "reason", "value", "content_sha256")},
        after={f: new_fields[k].get(f) for f in ("status", "category", "reason", "value", "content_sha256")})
        for k in sorted(old_fields.keys() & new_fields.keys()) if any(old_fields[k].get(f) != new_fields[k].get(f)
            for f in ("status", "category", "reason", "value", "content_sha256"))]
    return dict(old_dataset_id=before["dataset_id"], new_dataset_id=after["dataset_id"],
                as_of=before["as_of"], improved_count=len(improved), improved=sorted(improved, key=lambda r: (r['symbol'],r['period'])),
                unresolved_count=sum(not r["eligible_at_cutoff"] for r in after["rows"]),
                unresolved=[{k: r[k] for k in ("symbol", "period", "status", "reason", "next_action")}
                            for r in after["rows"] if not r["eligible_at_cutoff"]],
                financials_before=before["financials"], financials_after=after["financials"],
                changed_fields=changed_fields, native_quality_before=before.get("native_quality"),
                native_quality_after=after.get("native_quality"),
                attribution=("PUBLICATION_TIME_POLICY_CHANGE_NOT_VERIFIED_REPAIR" if assumed else
                             "EVIDENCE_REPAIR_NOT_MODEL_IMPROVEMENT"), dataset_completeness="NOT_CERTIFIED")


def review_repair(evidence, review_file, *, as_of, symbols, collection_file=None, accept_pdf_upload_policy=False):
    """Validate before publishing; external source roots are explicit file inputs."""
    eid, payload, old_docs = open_dataset(evidence)
    review = json.loads(Path(review_file).read_text())
    schema = "filing-pdf-upload-review/v1" if accept_pdf_upload_policy else "filing-repair/v1"
    if review.get("schema") != schema or review.get("parent_dataset_id") != eid:
        raise ValueError("REPAIR_PARENT_OR_SCHEMA_MISMATCH")
    if not isinstance(review.get("reviews"), list):
        raise ValueError("EXPLICIT_REVIEW_LIST_REQUIRED")
    review_docs = EvidenceDocuments(Path(review_file).parent / "objects")
    collection, collection_docs = None, None
    if collection_file:
        collection_file = Path(collection_file)
        collection = json.loads(collection_file.read_text())
        if collection.get("schema") != "filing-source-collection/v1":
            raise ValueError("INVALID_COLLECTION_SCHEMA")
        collection_docs = EvidenceDocuments(collection_file.parent.parent / "objects")
    def read(reference):
        for store in (old_docs, review_docs, collection_docs):
            if store is not None and store.objects.path(reference["sha256"]).exists():
                return store.read(reference)
        return old_docs.read(reference)
    working = deepcopy(payload)
    recovered = []
    if collection:
        for receipt in collection["results"]:
            if "receipt_sha256" in receipt and receipt["receipt_sha256"] != digest({k:v for k,v in receipt.items() if k != "receipt_sha256"}):
                raise ValueError("COLLECTION_RECEIPT_MISMATCH")
            doc = receipt.get("document")
            if doc:
                verified_document(doc, document_reader=read)
            if receipt["status"] == "ORIGINAL_RECOVERED":
                sha = receipt["request"]["expected_sha256"]
                if doc["sha256"] != sha:
                    raise ValueError("RECOVERED_ORIGINAL_HASH_MISMATCH")
                if sha not in working["raw_documents"]:
                    working["raw_documents"][sha] = doc
                    recovered.append(sha)
                else:
                    try:
                        verified_document(working["raw_documents"][sha], document_reader=old_docs.read)
                    except ValueError as error:
                        if not str(error).startswith("SOURCE_DOCUMENT_UNAVAILABLE"):
                            raise
                        # Keep the original receipt, retain the new recovery
                        # receipt separately, and copy exact bytes to a NEW store.
                        recovered.append(sha)
    if accept_pdf_upload_policy:
        from application.pdf_upload_policy import apply_pdf_upload_policy
        working, changes = apply_pdf_upload_policy(working, review, document_reader=read)
    else:
        working, changes = apply_reviews(working, review["reviews"], document_reader=read)
    # This provenance affects dataset identity, never stored download metadata.
    if changes or recovered:
        working["publication_reconstruction"] = dict(parent_dataset_id=eid,
            review_sha256=digest(review), changes=changes, recovered_documents=recovered,
            source_collection=collection, unresolved=review.get("unresolved", []))
        if accept_pdf_upload_policy:
            working["publication_reconstruction"]["policy"] = "PDF_UPLOAD_USER_ASSUMPTION_NOT_VERIFIED_VERSION"
    before = audit_filings(payload, symbols, as_of, parse_filing=parse_ixbrl, document_reader=old_docs.read)
    after = audit_filings(working, symbols, as_of, parse_filing=parse_ixbrl, document_reader=read)
    report = dict(schema="filing-repair-verification/v1", before=before, after=after,
        difference=audit_difference(before, after), verified_reviews=0 if accept_pdf_upload_policy else len(changes),
        recovered_documents=len(recovered), publishable=bool(changes or recovered),
        status="VALIDATED_PARTIAL_REPAIR" if changes or recovered else "NO_VERIFIED_REPAIR")
    if accept_pdf_upload_policy:
        report.update(policy_assumptions=len(changes), status="USER_ACCEPTED_PDF_UPLOAD_POLICY" if changes else "NO_POLICY_CHANGES")
    return working, read, report


def publish_repair(evidence, review_file, store, **kwargs):
    working, reader, report = review_repair(evidence, review_file, **kwargs)
    if not report["publishable"]:
        return None, report
    assumed = kwargs.get("accept_pdf_upload_policy", False)
    path = create_dataset(working, Path(store), parent=Path(evidence), document_reader=reader,
                          purpose="PDF_UPLOAD_POLICY_ASSUMPTION_NOT_STRICT_VERSION_PROOF" if assumed else
                                  "PUBLICATION_RECONSTRUCTION_NOT_COMPLETENESS_CERTIFICATION")
    # References externalized by publication are part of the new canonical ID.
    after = audit_dataset(path, kwargs["symbols"], kwargs["as_of"])
    report.update(after=after, difference=audit_difference(report["before"], after), evidence=str(path),
                  dataset_id=path.parent.name)
    write_once(path.parent / ("publication-policy.json" if assumed else "publication-repair.json"), report)
    return path, report


def compare_datasets(before_file, after_file, request_documents, output, *, cache_enabled=True, initial_state=False):
    """Same native engine, pinned models/accounts/windows, independent PAE chains.

    Both start with explicitly empty predecessor state and replay the SAME full
    request sequence. No hidden rebase of a formal or old-dataset predecessor.
    """
    from infrastructure.research_job import ResearchJob
    if not request_documents or not initial_state:
        raise ValueError("EXPLICIT_REQUEST_SEQUENCE_AND_INITIAL_STATE_REQUIRED")
    output = Path(output)
    # Separate jobs remain resumable; changing inputs/code is rejected natively.
    with ResearchJob(before_file, output / "before", cache_enabled=cache_enabled) as left, \
            ResearchJob(after_file, output / "after", cache_enabled=cache_enabled) as right:
        days = []
        for document in request_documents:
            request = request_from_document(document, output / "unused.sqlite3")
            if request.evidence_id != left.evidence.evidence_id:
                raise ValueError("COMPARISON_REQUEST_MUST_REFERENCE_PARENT_DATASET")
            a = left.run_day(request)
            b = right.run_day(replace(request, evidence_id=right.evidence.evidence_id))
            old = audit_dataset(before_file, request.symbols, request.as_of)
            new = audit_dataset(after_file, request.symbols, request.as_of)
            days.append(dict(as_of=request.as_of.isoformat(), account_sha256=digest(document["account"]),
                result_difference=compare_results(a.result, b.result),
                coverage_before=a.coverage, coverage_after=b.coverage,
                financial_difference=audit_difference(old, new),
                decisions_before=a.result.get("decisions"), decisions_after=b.result.get("decisions"),
                allocations_before=a.result.get("allocation_plans"), allocations_after=b.result.get("allocation_plans"),
                before_receipt=a.receipt, after_receipt=b.receipt))
        report = dict(schema="publication-repair-comparison/v1", old_dataset_id=left.evidence.evidence_id,
            new_dataset_id=right.evidence.evidence_id, versions=request.inputs()["versions"], days=days,
            attribution=("PUBLICATION_TIME_POLICY_CHANGE_NOT_VERIFIED_REPAIR" if any(
                d["financial_difference"]["attribution"] == "PUBLICATION_TIME_POLICY_CHANGE_NOT_VERIFIED_REPAIR" for d in days)
                else "EVIDENCE_REPAIR_NOT_MODEL_IMPROVEMENT"), predecessor_policy="EXPLICIT_EMPTY_THEN_SAME_REQUEST_SEQUENCE",
            initial_state_is_assumption=True, live_order_authorized=False, is_fill=False)
    write_once(output / "comparison.json", report)
    return report
