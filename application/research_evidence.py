"""Verified historical evidence policies; hashes are integrity, not authenticity.

No downloads, guessed publication deadlines, or model decisions belong here.
Provider attestations must bind the exact original document/version, not merely
the company and fiscal period. Unverified records retain first observation time.
"""
from __future__ import annotations

import datetime as dt
from copy import deepcopy
from typing import Protocol

from application.replay_contracts import aware_timestamp, digest
from domain.evidence_publication import (
    DOCUMENT_REF_SCHEMA, RAW_EVIDENCE_SCHEMA, DocumentReader, OFFICIAL_HOSTS,
    calendar_at, validate_document_ref, verified_document, verify_publication,
)


class FilingHistoryIndex(Protocol):
    def load(self, payload: dict, symbols: tuple[str, ...], as_of: dt.datetime) -> tuple[dict, dict]: ...


def filing_binding(symbol: str, record: dict) -> dict:
    return dict(symbol=symbol, fiscal_year=record["fiscal_year"], quarter=record["quarter"],
                report_id=record["receipt"]["report_id"], statement_scope=record["statement_scope"],
                source_version=record["source_version"], content_sha256=record["content_sha256"])


def select_filing_versions(records: list[dict]) -> tuple[list[int], list[dict]]:
    """Shared period/scope/version ordering AFTER caller cutoff validation."""
    selected, conflicts = [], []
    for period in sorted({(r["fiscal_year"], r["quarter"]) for r in records}):
        candidates = [i for i, r in enumerate(records) if (r["fiscal_year"], r["quarter"]) == period]
        candidates = ([i for i in candidates if records[i]["statement_scope"] == "consolidated"] or
                      [i for i in candidates if records[i]["statement_scope"] == "individual"])
        if not candidates:
            continue
        def order(i):
            return aware_timestamp(records[i]["available_at"]), records[i]["revision_number"]
        latest = max(map(order, candidates))
        winners = [i for i in candidates if order(i) == latest]
        if len({records[i]["content_sha256"] for i in winners}) > 1:
            conflicts.append(dict(period=f"{period[0]}Q{period[1]}", indices=winners, reason="CONFLICTING_FILING_VERSION"))
        else:
            selected.append(winners[0])
    return selected, conflicts


def publication_identity_sha256(proof: dict) -> str:
    """Hash a verified proof independent of inline vs referenced byte storage.

    Retain URLs, content hashes, times, bindings and version-chain metadata.
    Call only after proof verification has checked every original document.
    """
    def identity(value):
        if isinstance(value, list):
            return [identity(item) for item in value]
        if isinstance(value, dict):
            is_document = all(key in value for key in ("url", "sha256", "retrieved_at"))
            return {key: identity(child) for key, child in value.items()
                    if not (is_document and key in {"body_ref", "body_base64", "body_utf8"})}
        return value
    return digest(identity(proof))


def materialize_filings(payload: dict, *, parse_filing, cache=None, version="",
                       document_reader: DocumentReader | None = None) -> tuple[dict, dict]:
    """Reparse immutable original bytes with CURRENT parser, never old features."""
    from dataclasses import asdict

    records, errors = {}, {}
    for symbol, items in payload.get("filings", {}).items():
        records[symbol], errors[symbol] = [], []
        for item in items:
            row = deepcopy(item)
            proof = row.pop("publication_proof", None)
            policy = row.pop("publication_time_policy", None)
            row.pop("source_version", None)
            try:
                original = payload["raw_documents"][item["content_sha256"]]
                content = verified_document(original, document_reader=document_reader)
                if original["sha256"] != item["content_sha256"]:
                    raise ValueError("FILING_ORIGINAL_HASH_MISMATCH")
                from domain.financial_taxonomy import ixbrl_schema_refs
                # Metadata is re-established from immutable bytes, not a stale
                # or caller-supplied taxonomy assertion in a stored manifest.
                row["taxonomy_schema_refs"] = list(ixbrl_schema_refs(content))
                row["receipt"] = dict(row["receipt"], original_availability=dict(
                    available_at=item["available_at"], availability_source=item["availability_source"],
                    fetched_at=item["receipt"]["fetched_at"], source_version=item.get("source_version")))
                # A later repair must supersede an assumption, not the other
                # way around. Invalid/conflicting proofs must never fall back.
                if proof is not None:
                    from application.filing_publication import verify_filing_publication
                    row["available_at"] = verify_filing_publication(symbol, item, document_reader=document_reader)
                    row["availability_source"] = "OFFICIAL_FILING_TIMESTAMP"
                    row["receipt"]["publication_verification"] = dict(
                        status="VERIFIED", proof_identity_sha256=publication_identity_sha256(proof),
                        published_at=proof["published_at"],
                        publication_precision=proof["publication_precision"],
                        verified_at=proof["verified_at"], version_identity=proof.get("version_identity"),
                        source_document_sha256=proof["document"]["sha256"])
                elif policy:
                    from application.pdf_upload_policy import PDF_UPLOAD_SOURCE, verify_pdf_upload_policy
                    assessed = verify_pdf_upload_policy(symbol, item, document_reader=document_reader)
                    row["available_at"] = assessed["available_at"]
                    row["availability_source"] = PDF_UPLOAD_SOURCE
                    row["receipt"] = dict(row["receipt"], publication_policy=dict(
                        schema=policy["schema"], status=policy["status"],
                        policy_sha256=digest(policy), pdf_filename=policy["pdf_filename"],
                        source_document_sha256=policy["document"]["sha256"], **assessed))
                else:
                    row["available_at"] = max(aware_timestamp(row["available_at"]),
                                               aware_timestamp(row["receipt"]["fetched_at"])).isoformat()
                    row["availability_source"] = "FIRST_OBSERVED_DOWNLOAD"
                def parse():
                    return [asdict(f) for f in parse_filing(content)]
                row["facts"] = (cache.memo("ixbrl", (version, item["content_sha256"]), parse)
                                if cache is not None else parse())
                records[symbol].append(row)
            except (KeyError, ValueError, TypeError) as error:
                detail = dict(content_sha256=item.get("content_sha256"), fiscal_year=item.get("fiscal_year"),
                              quarter=item.get("quarter"), reason=str(error))
                if str(error) == "CONFLICTING_PUBLICATION_PROOFS":
                    from application.filing_publication import verify_filing_publication
                    try:
                        times = [aware_timestamp(verify_filing_publication(symbol, dict(item, publication_proof=p),
                            document_reader=document_reader)) for p in proof["candidates"]]
                        detail["conflict_available_at"] = min(times).isoformat()
                    except (KeyError, ValueError, TypeError):
                        detail["conflict_available_at"] = None
                errors[symbol].append(detail)
    return records, errors
