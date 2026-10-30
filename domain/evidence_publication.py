"""Pure original-document and publication policies shared by all consumers.

This is the existing strict-replay verification core, extracted downward so
native domain rules do not depend on application or transport adapters. A
content hash establishes integrity, not official authenticity or vintage.
Raw calendar interpretation remains an injected, no-I/O adapter.
"""
from __future__ import annotations

import base64
from collections.abc import Callable
import datetime as dt
import hashlib
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from domain.les_v7.contracts import identity as digest
from domain.restriction_announcements import publication_bounds

RAW_EVIDENCE_SCHEMA = "replay-evidence/v2"
DOCUMENT_REF_SCHEMA = "research-document-ref/v1"
DocumentReader = Callable[[dict], bytes]
_TAIPEI = ZoneInfo("Asia/Taipei")

OFFICIAL_HOSTS = {"www.twse.com.tw", "twse.com.tw", "wwwc.twse.com.tw",
                  "doc.twse.com.tw", "openapi.twse.com.tw", "isin.twse.com.tw",
                  "mops.twse.com.tw", "mopsov.twse.com.tw", "mopsfin.twse.com.tw",
                  "www.tpex.org.tw", "tpex.org.tw", "investoredu.twse.com.tw"}


def aware_timestamp(value: str | dt.datetime) -> dt.datetime:
    """Preserve replay-contract parsing and its exact Taipei representation."""
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(parsed, dt.datetime) or parsed.utcoffset() is None:
        raise ValueError("REPLAY_REQUIRES_TIMEZONE")
    return parsed.astimezone(_TAIPEI)


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
