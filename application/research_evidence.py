"""Verified historical evidence policies; hashes are integrity, not authenticity.

No downloads, guessed publication deadlines, or model decisions belong here.
Provider attestations must bind the exact original document/version, not merely
the company and fiscal period. Unverified records retain first observation time.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
from copy import deepcopy
from collections.abc import Callable
from typing import Protocol
from urllib.parse import urlparse

from application.replay_contracts import aware_timestamp, digest
from domain.restriction_announcements import publication_bounds

RAW_EVIDENCE_SCHEMA = "replay-evidence/v2"
DOCUMENT_REF_SCHEMA = "research-document-ref/v1"
DocumentReader = Callable[[dict], bytes]


class FilingHistoryIndex(Protocol):
    def load(self, payload: dict, symbols: tuple[str, ...], as_of: dt.datetime) -> tuple[dict, dict]: ...


OFFICIAL_HOSTS = {"www.twse.com.tw", "twse.com.tw", "wwwc.twse.com.tw",
                  "doc.twse.com.tw",
                  "mops.twse.com.tw", "mopsov.twse.com.tw", "mopsfin.twse.com.tw",
                  "www.tpex.org.tw", "tpex.org.tw", "investoredu.twse.com.tw"}


def validate_document_ref(reference: dict) -> None:
    """References are content identities, never filesystem paths or fetch URLs."""
    if (not isinstance(reference, dict)
            or set(reference) != {"schema", "sha256", "size_bytes"}
            or reference.get("schema") != DOCUMENT_REF_SCHEMA
            or not isinstance(reference.get("sha256"), str)
            or len(reference["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in reference["sha256"])
            or type(reference.get("size_bytes")) is not int
            or reference["size_bytes"] < 0):
        raise ValueError("INVALID_SOURCE_DOCUMENT_REFERENCE")


def verified_document(document: dict, *, document_reader: DocumentReader | None = None) -> bytes:
    url = urlparse(document["url"])
    if url.scheme != "https" or url.hostname not in OFFICIAL_HOSTS or url.username or url.password:
        raise ValueError("OFFICIAL_EVIDENCE_SOURCE_REQUIRED")
    if sum(key in document for key in ("body_ref", "body_base64", "body_utf8")) != 1:
        raise ValueError("AMBIGUOUS_SOURCE_DOCUMENT_BODY")
    if "body_ref" in document:
        reference = document["body_ref"]
        validate_document_ref(reference)
        if reference["sha256"] != document["sha256"]:
            raise ValueError("SOURCE_DOCUMENT_HASH_MISMATCH")
        if document_reader is None:
            raise ValueError("SOURCE_DOCUMENT_READER_REQUIRED")
        body = document_reader(reference)
        if len(body) != reference["size_bytes"]:
            raise ValueError("SOURCE_DOCUMENT_SIZE_MISMATCH")
    else:
        body = (base64.b64decode(document["body_base64"], validate=True) if "body_base64" in document
                else document["body_utf8"].encode("utf-8"))
    if hashlib.sha256(body).hexdigest() != document["sha256"]:
        raise ValueError("SOURCE_DOCUMENT_HASH_MISMATCH")
    aware_timestamp(document["retrieved_at"])
    return body


def verify_publication(proof: dict, *, binding: dict, document_reader: DocumentReader | None = None) -> str:
    """Reviewed source assertion, with exact hash/version binding and quotes.

Date-only releases become usable at the NEXT Taipei midnight. Human/provider
review is explicit; a checksum or a download URL alone never proves vintage.
"""
    if (proof.get("status") != "VERIFIED" or not proof.get("verified_by")
            or not proof.get("verification_method") or proof.get("binding") != binding):
        raise ValueError("ORIGINAL_VERSION_BINDING_NOT_VERIFIED")
    aware_timestamp(proof["verified_at"])
    document = proof["document"]
    body = verified_document(document, document_reader=document_reader)
    # PDF evidence may retain reviewed extracted text alongside the original.
    text = document.get("extracted_text", body.decode("utf-8", errors="replace"))
    quotes = proof.get("quotes", [])
    if not quotes or any(not q or q not in text for q in quotes):
        raise ValueError("PUBLICATION_PROOF_NOT_IN_SOURCE")
    if "extracted_text" in document and not proof.get("extraction_reviewed"):
        raise ValueError("EXTRACTED_SOURCE_REVIEW_REQUIRED")
    first, upper = publication_bounds(proof)
    if aware_timestamp(document["retrieved_at"]) < first:
        raise ValueError("DOCUMENT_PREDATES_PUBLICATION")
    return upper.isoformat()


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


def calendar_at(calendar: dict, cutoff: dt.datetime, *, strict: bool, annual_calendar=None, verifier=None) -> tuple[dict, list]:
    """Annual schedule plus dated, reviewed emergency announcements.

Legacy v1 schedules retain their documented final-archive policy. In v2 an
unverified annual schedule cannot certify OPEN. Unknown days are retained and
block research eligibility rather than being silently treated as weekends.
"""
    days, excluded = {}, []
    verify = verifier or verify_publication
    annual = calendar.get("documents", {})
    verified_years = {}
    if strict:
        for year, doc in annual.items():
            try:
                # Annual schedule data is preserved exactly; receipt is not an
                # original release timestamp. A reviewed publication is needed.
                proof = doc["publication_proof"]
                binding = dict(kind="ANNUAL_CALENDAR", market=calendar["source"].split("_")[0],
                               year=int(year), content_sha256=digest(doc["payload"]))
                if aware_timestamp(verify(proof, binding=binding)) <= cutoff:
                    if annual_calendar is None:
                        raise ValueError("RAW_CALENDAR_ADAPTER_REQUIRED")
                    verified_years[int(year)] = annual_calendar(doc["payload"])
            except (KeyError, ValueError, TypeError) as error:
                excluded.append(dict(year=year, reason="UNVERIFIED_ANNUAL_CALENDAR", detail=str(error)))
    for day, status in calendar.get("days", {}).items():
        if not strict:
            days[day] = status
        elif int(day[:4]) in verified_years:
            days[day] = verified_years[int(day[:4])].trading_day_status(dt.date.fromisoformat(day)).value.upper()
        else:
            days[day] = "UNKNOWN"
    announcements = {}
    for event in calendar.get("announcements", []):
        try:
            day, status = event["date"], event["status"]
            dt.date.fromisoformat(day)
            if status not in {"OPEN", "CLOSED", "UNKNOWN"}:
                raise ValueError("INVALID_CALENDAR_STATUS")
            binding = dict(kind="CALENDAR_EXCEPTION", market=event["market"], date=day, status=status)
            if event["market"] != calendar["source"].split("_")[0]:
                raise ValueError("CALENDAR_MARKET_MISMATCH")
            at = aware_timestamp(verify(event["proof"], binding=binding))
            if at > cutoff:
                excluded.append(dict(date=day, reason="CALENDAR_ANNOUNCEMENT_AFTER_CUTOFF"))
                continue
            announcements.setdefault(day, []).append((at, status))
        except (KeyError, ValueError, TypeError) as error:
            excluded.append(dict(date=event.get("date"), reason=str(error)))
    for day, events in announcements.items():
        last = max(at for at, _ in events)
        states = {status for at, status in events if at == last}
        days[day] = states.pop() if len(states) == 1 else "UNKNOWN"
    return days, excluded
