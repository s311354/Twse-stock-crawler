"""Bounded, cache-first R17 current-calendar capture using retained originals.

The existing annual/news/PDF collector remains the transport/validation owner.
No current observation is relabelled as an original historical publication.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from domain.evidence_publication import verified_document
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.session_coverage_r17 import POLICY, SCHEMA
from infrastructure.les_v7_calendar_final import (SCHEMA as FINAL_SCHEMA, _closure, _pdf_text,
    collect_final_calendars, load_final_calendar)
from infrastructure.research_archive import write_once


def unavailable(*, market, observed_at, reason):
    value = dict(schema=SCHEMA, policy=POLICY, market=market, source=market + "_OFFICIAL",
        observed_at=timestamp(observed_at).isoformat(), documents={}, closure_queries={}, announcements=[],
        diagnostics=[reason], status="UNKNOWN", original_historical_available_at=None,
        historical_information_set_certified=False,
        availability_basis="ACTUAL_CURRENT_OBSERVATION_NOT_HISTORICAL_PUBLICATION")
    return dict(value, evidence_id=identity(value))


def prepare_current_calendar(payload, documents, *, observed_at, extract_text=None):
    """Detach verified raw bytes once for the pure cutoff verifier/worker port."""
    if payload.get("schema") != FINAL_SCHEMA or payload.get("source") != "TWSE_OFFICIAL":
        raise ValueError("R17_CURRENT_CALENDAR_ORIGINAL_SCHEMA_REQUIRED")
    observed = timestamp(observed_at)

    def inline(document):
        body = verified_document(document, document_reader=documents.read)
        if len(body) > 2 * 1024 * 1024 or timestamp(document["retrieved_at"]) > observed:
            raise ValueError("R17_ORIGINAL_TOO_LARGE_OR_AFTER_OBSERVATION")
        result = deepcopy(document)
        for key in ("body_ref", "body_utf8", "body_base64"):
            result.pop(key, None)
        result["body_base64"] = base64.b64encode(body).decode("ascii")
        return result

    def row(value):
        return dict(payload=deepcopy(value["payload"]), payload_sha256=value["payload_sha256"],
                    document=inline(value["document"]))

    announcements = []
    for event in payload.get("announcements", []):
        try:
            # This independently checks source detail, actual PDF bytes,
            # extracted title/release/effective dates and original hashes.
            released = _closure(event, documents, extract_text=extract_text or _pdf_text)
            text = (extract_text or _pdf_text)(verified_document(event["document"], document_reader=documents.read))
            pdf = inline(event["document"])
            pdf["extracted_text"] = text
            detail_doc = inline(event["detail_document"])
            detail = json.loads(verified_document(event["detail_document"], document_reader=documents.read))
            binding = dict(kind="CALENDAR_EXCEPTION", market="TWSE", date=event["date"], status="CLOSED")
            proof = dict(status="VERIFIED", verified_by="R17_CURRENT_CALENDAR_SOURCE_CHECKER",
                verification_method="AUTOMATED_OFFICIAL_RAW_PDF_AND_DETAIL_DATE_TITLE_EFFECTIVE_DATE_CHECKS_R17",
                verified_at=observed.isoformat(), binding=binding, published_at=released,
                publication_precision="TIMESTAMP", extraction_reviewed=True, quotes=[text], document=pdf)
            announcements.append(dict(news_id=event["news_id"], date=event["date"], market="TWSE", status="CLOSED",
                detail=dict(payload=detail, payload_sha256=identity(detail), document=detail_doc), proof=proof))
        except (OSError, KeyError, ValueError, TypeError, IndexError) as error:
            # Do not drop an unresolved closure from an otherwise complete query.
            announcements.append(dict(news_id=event.get("news_id"), date=event.get("date"), market="TWSE",
                status="UNKNOWN", verification_error=str(error)))
    value = dict(schema=SCHEMA, policy=POLICY, market="TWSE", source="TWSE_OFFICIAL",
        observed_at=observed.isoformat(), documents={year: row(item) for year, item in payload.get("documents", {}).items()},
        closure_queries={year: row(item) for year, item in payload.get("closure_queries", {}).items()},
        announcements=announcements, original_historical_available_at=None,
        historical_information_set_certified=False,
        availability_basis="ACTUAL_CURRENT_OBSERVATION_NOT_HISTORICAL_PUBLICATION",
        original_collection_id=identity(payload), diagnostics=deepcopy(payload.get("collection_errors", [])),
        source_authenticity="EXISTING_OFFICIAL_CAPTURE_AND_SOURCE_BOUND_ORIGINAL_RECHECK_NOT_HASH_ALONE")
    return dict(value, evidence_id=identity(value))


def capture_current_calendar(root, *, market, start, end, client, clock):
    """One market/year/date capture, not per-symbol fetching or weekday repair.

    A same-day pointer reuses immutable annual/query/PDF originals. It is never
    reused to cover a session closing after the actual query observation, and
    corrupted cached references fail rather than silently switching sources.
    """
    observed = timestamp(clock())
    if market != "TWSE":
        return unavailable(market=market, observed_at=observed,
            reason="R17_MARKET_SPECIFIC_CALENDAR_CAPTURE_UNAVAILABLE")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last-first).days > 3660:
        raise ValueError("R17_CALENDAR_CAPTURE_RANGE_INVALID")
    years = tuple(range(first.year, last.year+1))
    root = Path(root)
    capture_root = root / market
    day = observed.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
    pointer = capture_root / "daily" / (day + "-through-" + last.isoformat() + "-" + "-".join(map(str, years)) + ".json")
    if pointer.exists():
        with pointer.open("rb") as stream:
            raw = stream.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ValueError("R17_CALENDAR_POINTER_TOO_LARGE")
        cached = json.loads(raw)
        cache_id = cached.pop("cache_id", None)
        if (cached.get("schema") != "les-v7-current-calendar-cache/v1" or cached.get("market") != market
                or cached.get("years") != list(years) or timestamp(cached["observed_at"]) > observed
                or timestamp(cached["observed_at"]).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat() != day
                or cached.get("through_session") != last.isoformat() or cache_id != identity(cached)
                or not isinstance(cached.get("calendar_id"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", cached["calendar_id"])):
            raise ValueError("R17_CALENDAR_POINTER_INVALID")
        reference = Path(cached["manifest"])
        if reference.is_absolute() or ".." in reference.parts:
            raise ValueError("R17_CALENDAR_POINTER_PATH_INVALID")
        if reference.parts != ("collections", cached["calendar_id"], "calendar.json"):
            raise ValueError("R17_CALENDAR_POINTER_SOURCE_BINDING_MISMATCH")
        payload, documents = load_final_calendar(capture_root / reference)
        if identity(payload) != cached.get("calendar_id") or timestamp(cached["observed_at"]) > observed:
            raise ValueError("R17_CALENDAR_CACHED_ORIGINAL_MISMATCH")
        return prepare_current_calendar(payload, documents, observed_at=cached["observed_at"])
    manifest = collect_final_calendars(capture_root, years=years, client=client, max_requests=20,
        now=lambda: timestamp(clock()).isoformat())
    payload, documents = load_final_calendar(manifest)
    completed = timestamp(clock()).isoformat()
    selected = prepare_current_calendar(payload, documents, observed_at=completed)
    cache = dict(schema="les-v7-current-calendar-cache/v1", market=market, years=list(years),
        through_session=last.isoformat(), observed_at=completed, calendar_id=identity(payload),
        manifest=str(manifest.relative_to(capture_root)))
    write_once(pointer, dict(cache, cache_id=identity(cache)))
    return selected
