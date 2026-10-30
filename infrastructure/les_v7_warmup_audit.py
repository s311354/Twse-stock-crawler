"""Source-bound listing lifecycle and normalized-price warmup diagnostics.

The current official listing table is a retrospective source, not proof of an
original historical observation. This audit never fills prices, treats a first
saved quote as a listing date, or makes a model/trading eligibility decision.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import date, datetime, time, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from application.replay_contracts import aware_timestamp, digest
from application.research_evidence import verified_document
from domain.errors import AccessBlockedError
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments

URL = "https://www.twse.com.tw/rwd/zh/company/newlisting"
PROFILE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
SCHEMA = "les-v7-official-final-listing/v1"
POLICY = "OFFICIAL_FINAL_LISTING_LIFECYCLE_R1"
MAX_BYTES = 4 * 1024 * 1024
MAX_ROWS = 4096
TAIPEI = ZoneInfo("Asia/Taipei")
FIELDS = ["公司代號", "公司簡稱", "申請日期", "董事長", "申請時股本(仟元)",
          "上市審議委員會審議日期", "交易所董事會通過上市日期", "上市契約報請主管機關備查日期",
          "證期局核准上市契約日期", "股票上市買賣日期", "承銷商", "承銷價", "備註"]
LIMITATIONS = ["CURRENT_FINAL_LISTING_TABLE_NOT_ORIGINAL_HISTORICAL_OBSERVATION",
               "MISSING_ROW_DOES_NOT_PROVE_UNLISTED", "PREVIOUS_MARKET_HISTORY_NOT_CERTIFIED",
               "NO_PRICE_OR_SECURITY_IDENTITY_IMPUTATION"]


def _json(content):
    if not isinstance(content, bytes) or not 0 < len(content) <= MAX_BYTES:
        raise ValueError("LISTING_RESPONSE_SIZE_LIMIT")
    text = content.decode("utf-8-sig")
    if any(marker in text[:8192].lower() for marker in
           ("access denied", "access blocked", "request rejected", "因為安全性考量")):
        raise AccessBlockedError("TWSE listing source blocked", retryable=False)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("DUPLICATE_LISTING_JSON_KEY")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("NON_FINITE_LISTING_SOURCE_VALUE:" + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def parse_listing_final(content):
    """Validate the exact official table; retain pending dates as unknown."""
    raw = _json(content)
    if (not isinstance(raw, dict) or raw.get("stat") != "OK"
            or raw.get("title") != "最近上市公司" or raw.get("fields") != FIELDS
            or not isinstance(raw.get("data"), list) or not 1 <= len(raw["data"]) <= MAX_ROWS
            or type(raw.get("total")) is not int or raw["total"] != len(raw["data"])):
        raise ValueError("LISTING_COMPLETE_TABLE_OR_COLUMN_MAPPING_MISMATCH")
    entries = []
    seen = set()
    for row in raw["data"]:
        if (not isinstance(row, list) or len(row) != len(FIELDS)
                or not isinstance(row[0], str) or not re.fullmatch(r"[0-9A-Z]{4,10}", row[0])
                or not isinstance(row[1], str) or not row[1].strip()
                or not isinstance(row[9], str) or not isinstance(row[12], str)
                or row[0] in seen):
            raise ValueError("LISTING_ROW_IDENTITY_OR_WIDTH_CONFLICT")
        seen.add(row[0])
        day = None
        if row[9]:
            matched = re.fullmatch(r"(\d{2,3})\.(\d{2})\.(\d{2})", row[9])
            if not matched:
                raise ValueError("LISTING_ROC_EFFECTIVE_DATE_UNRECOGNIZED")
            year, month, number = map(int, matched.groups())
            day = date(year + 1911, month, number).isoformat()
        entries.append(dict(symbol=row[0], name=row[1], market="TWSE", listing_date=day,
            remark=row[12], status="VERIFIED_REPORTED" if day else "PENDING_VERIFICATION",
            source_row_sha256=digest(row)))
    return dict(source_scope="CURRENT_OFFICIAL_LISTING_TABLE_NOT_ALL_HISTORICAL_SECURITIES",
                fields=FIELDS, row_count=len(entries), entries=sorted(entries, key=lambda row: row["symbol"]),
                raw_sha256=hashlib.sha256(content).hexdigest())


def _packet_id(packet):
    body = dict(schema=SCHEMA, policy=POLICY, market="TWSE", url=packet["document"]["url"],
                raw_sha256=packet["document"]["sha256"], query=packet["query"])
    if "profiles" in packet:
        profiles = packet["profiles"]
        body["profiles"] = dict(url=profiles["document"]["url"],
            raw_sha256=profiles["document"]["sha256"], query=profiles["query"])
    return digest(body)


def parse_listed_profiles_final(content):
    """Extract only lifecycle/identity fields from the exact official feed.

    Current financial/share-count fields are retained in the original bytes but
    never become historical research inputs. A table's production date is not
    its original publication or historical observation time.
    """
    raw = _json(content)
    fields = {"公司代號", "公司名稱", "公司簡稱", "上市日期", "營利事業統一編號"}
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_ROWS:
        raise ValueError("LISTED_PROFILE_ARRAY_OR_ROW_BOUND_MISMATCH")
    entries, seen = [], set()
    for row in raw:
        if (not isinstance(row, dict) or not fields.issubset(row)
                or any(not isinstance(row[field], str) for field in fields)
                or not re.fullmatch(r"[0-9A-Z]{4,10}", row["公司代號"])
                or not row["公司名稱"].strip() or not row["公司簡稱"].strip()
                or not re.fullmatch(r"\d{8}", row["營利事業統一編號"])
                or row["公司代號"] in seen):
            raise ValueError("LISTED_PROFILE_IDENTITY_OR_DUPLICATE_CONFLICT")
        seen.add(row["公司代號"])
        day = None
        if row["上市日期"]:
            if not re.fullmatch(r"\d{8}", row["上市日期"]):
                raise ValueError("LISTED_PROFILE_EFFECTIVE_DATE_UNRECOGNIZED")
            day = datetime.strptime(row["上市日期"], "%Y%m%d").date().isoformat()
        entries.append(dict(symbol=row["公司代號"], name=row["公司簡稱"], company_name=row["公司名稱"],
            issuer_id=row["營利事業統一編號"], market="TWSE", listing_date=day, remark="",
            status="VERIFIED_REPORTED" if day else "PENDING_VERIFICATION",
            source_kind="CURRENT_OFFICIAL_LISTED_COMPANY_PROFILE", source_row_sha256=digest(row)))
    return dict(source_scope="CURRENT_OFFICIAL_LISTED_COMPANY_PROFILES_NOT_HISTORICAL_MEMBERSHIP",
        lifecycle_fields=sorted(fields), row_count=len(entries), entries=sorted(entries, key=lambda row: row["symbol"]),
        raw_sha256=hashlib.sha256(content).hexdigest())


def verify_listing_final(packet, *, document_reader=None):
    if (not isinstance(packet, dict) or packet.get("schema") != SCHEMA or packet.get("policy") != POLICY
            or packet.get("market") != "TWSE" or packet.get("historical_available_at") is not None
            or packet.get("historical_information_set_certified") is not False
            or packet.get("limitations") != LIMITATIONS or packet.get("query") != {"response": "json"}
            or not isinstance(packet.get("document"), dict)):
        raise ValueError("EXPLICIT_FINAL_LISTING_PACKET_REQUIRED")
    document = packet["document"]
    if (document.get("url") != URL + "?response=json"
            or packet.get("fetched_at") != document.get("retrieved_at")):
        raise ValueError("LISTING_SOURCE_OR_TIME_BINDING_MISMATCH")
    aware_timestamp(packet["fetched_at"])
    original = verified_document(document, document_reader=document_reader)
    if packet.get("parsed") != parse_listing_final(original):
        raise ValueError("LISTING_ORIGINAL_AND_PARSED_RESULT_MISMATCH")
    if "profiles" in packet:
        profiles = packet["profiles"]
        if (not isinstance(profiles, dict) or profiles.get("query") != {}
                or not isinstance(profiles.get("document"), dict)
                or profiles["document"].get("url") != PROFILE_URL
                or profiles.get("fetched_at") != profiles["document"].get("retrieved_at")
                or profiles.get("historical_available_at") is not None
                or profiles.get("historical_information_set_certified") is not False):
            raise ValueError("LISTED_PROFILE_SOURCE_OR_TIME_BINDING_MISMATCH")
        aware_timestamp(profiles["fetched_at"])
        profile_content = verified_document(profiles["document"], document_reader=document_reader)
        if profiles.get("parsed") != parse_listed_profiles_final(profile_content):
            raise ValueError("LISTED_PROFILE_ORIGINAL_AND_PARSED_RESULT_MISMATCH")
    if packet.get("evidence_id") != _packet_id(packet):
        raise ValueError("LISTING_PACKET_ID_MISMATCH")
    return deepcopy(packet)


def collect_listing_final(root, *, client=None, max_requests=1, now=utc_now, include_profiles=False):
    """Capture one source, or explicitly two; failures publish no completion."""
    if (type(include_profiles) is not bool or type(max_requests) is not int
            or max_requests != (2 if include_profiles else 1)):
        raise ValueError("LISTING_SINGLE_SOURCE_REQUEST_BUDGET_REQUIRED")
    transport = client or TwseHttpClient(TwseHttpConfig(timeout_seconds=25, max_attempts=1,
        min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
    if transport.config.allow_insecure_ssl_fallback or transport.config.min_request_interval_seconds < 1:
        raise ValueError("SECURE_RATE_LIMITED_LISTING_TRANSPORT_REQUIRED")
    root = Path(root)
    documents, calls = EvidenceDocuments(root / "objects"), 0

    def reserve():
        nonlocal calls
        if calls >= max_requests:
            raise ValueError("LISTING_REQUEST_BUDGET_EXHAUSTED")
        calls += 1

    query = {"response": "json"}
    response = transport.request_bytes("GET", URL, params=query, before_attempt=reserve,
        context="TWSE official FINAL listing lifecycle", content_validator=parse_listing_final)
    parsed = parse_listing_final(response.content)
    fetched = now()
    document = dict(url=URL + "?" + urlencode(query), retrieved_at=fetched,
        sha256=parsed["raw_sha256"], body_ref=documents.store(response.content),
        representation="ORIGINAL_TRANSPORT_BYTES")
    packet = dict(schema=SCHEMA, policy=POLICY, market="TWSE", query=query, parsed=parsed,
        document=document, fetched_at=fetched, historical_available_at=None,
        historical_information_set_certified=False, limitations=LIMITATIONS)
    if include_profiles:
        response = transport.request_bytes("GET", PROFILE_URL, params=None, before_attempt=reserve,
            context="TWSE official FINAL listed-company lifecycle", content_validator=parse_listed_profiles_final)
        parsed_profile = parse_listed_profiles_final(response.content)
        profile_fetched = now()
        profile_document = dict(url=PROFILE_URL, retrieved_at=profile_fetched,
            sha256=parsed_profile["raw_sha256"], body_ref=documents.store(response.content),
            representation="ORIGINAL_TRANSPORT_BYTES")
        packet["profiles"] = dict(query={}, parsed=parsed_profile, document=profile_document,
            fetched_at=profile_fetched, historical_available_at=None, historical_information_set_certified=False)
    packet["evidence_id"] = _packet_id(packet)
    path = root / packet["evidence_id"] / "manifest.json"
    if path.exists():
        load_listing_final(path)  # Keep the first actual observation receipt.
    else:
        verify_listing_final(packet, document_reader=documents.read)
        write_once(path, packet)
    return path


def load_listing_final(path):
    path = Path(path)
    with path.open("rb") as stream:
        content = stream.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("LISTING_PACKET_SIZE_LIMIT")
    packet = json.loads(content)
    documents = EvidenceDocuments(path.parent.parent / "objects")
    documents.verify(packet)
    result = verify_listing_final(packet, document_reader=documents.read)
    documents.check()
    return result, documents


def audit_normalized_warmup(histories, calendar, listing_packet, *, cutoff, required=513,
                            document_reader=None):
    """Retain every supplied stock and distinguish missing from impossible data.

    Only cutoff-valid, complete normalized prices on verified OPEN dates count.
    Listing-table absence remains pending. The earliest archived 513th close is
    a diagnostic, not permission to use its later prices at this cutoff.
    """
    at = aware_timestamp(cutoff).astimezone(TAIPEI)
    if type(required) is not int or not 1 <= required <= 4096:
        raise ValueError("WARMUP_REQUIREMENT_OUT_OF_BOUNDS")
    if (not isinstance(histories, list) or len(histories) > 2048
            or not isinstance(calendar, dict) or not isinstance(calendar.get("days"), dict)):
        raise ValueError("BOUNDED_NORMALIZED_WARMUP_INPUT_REQUIRED")
    days = calendar["days"]
    if any(state not in {"OPEN", "CLOSED", "UNKNOWN"} for state in days.values()):
        raise ValueError("WARMUP_CALENDAR_STATUS_INVALID")
    listed = {}
    profiles = {}
    listing_id = None
    if listing_packet is not None:
        verified = verify_listing_final(listing_packet, document_reader=document_reader)
        listed = {row["symbol"]: row for row in verified["parsed"]["entries"]}
        profiles = {row["symbol"]: row for row in verified.get("profiles", {}).get("parsed", {}).get("entries", [])}
        listing_id = verified["evidence_id"]
    rows, symbols = [], set()
    for history in histories:
        symbol = history.get("symbol")
        if not isinstance(symbol, str) or not symbol or symbol in symbols or not isinstance(history.get("bars"), list):
            raise ValueError("WARMUP_HISTORY_IDENTITY_CONFLICT")
        symbols.add(symbol)
        listing, profile = listed.get(symbol), profiles.get(symbol)
        if listing and profile:
            if listing["listing_date"] and profile["listing_date"] and listing["listing_date"] != profile["listing_date"]:
                # Retain the conflict per stock so unrelated gaps can still be
                # audited. Neither date can exclude bars or prove an upper bound.
                listing = dict(symbol=symbol, market="TWSE", listing_date=None, status="CONFLICT",
                    conflict_reason="LISTING_EFFECTIVE_DATE_SOURCE_CONFLICT", sources=[listing, profile])
            else:
                # A missing date needs a positive independently verified report.
                listing = dict(listing, issuer_id=profile["issuer_id"], company_name=profile["company_name"],
                    profile_source_row_sha256=profile["source_row_sha256"],
                    listing_date=listing["listing_date"] or profile["listing_date"],
                    status="VERIFIED_REPORTED" if listing["listing_date"] or profile["listing_date"] else "PENDING_VERIFICATION")
        elif profile:
            listing = profile
        listing_day = listing.get("listing_date") if listing else None
        observations, excluded = {}, Counter()
        conflicts = set()
        for bar in history["bars"]:
            try:
                day = date.fromisoformat(bar["Date"]).isoformat()
                close = bar["Close"]
                available = aware_timestamp(bar["Available At"])
                if (type(close) not in {float, int} or not math.isfinite(close) or close <= 0
                        or bar.get("Complete") is not True):
                    excluded["INVALID_OR_INCOMPLETE_PRICE"] += 1
                    continue
                if days.get(day, "UNKNOWN") != "OPEN":
                    excluded["CALENDAR_NOT_VERIFIED_OPEN"] += 1
                    continue
                if listing_day is not None and day < listing_day:
                    excluded["PRICE_BEFORE_VERIFIED_TWSE_LISTING_REQUIRES_TRANSITION_EVIDENCE"] += 1
                    continue
                # A normalized final close must not claim availability before
                # the source session close, including intraday cutoff audits.
                close_at = datetime.combine(date.fromisoformat(day), time(13, 30), TAIPEI)
                if available < close_at:
                    excluded["PRICE_AVAILABILITY_PRECEDES_CLOSE"] += 1
                    continue
                value = close, available
                if day in observations and observations[day] != value:
                    conflicts.add(day)
                observations[day] = value
            except (KeyError, ValueError, TypeError):
                excluded["INVALID_PRICE_DATE_OR_AVAILABILITY"] += 1
        for day in conflicts:
            observations.pop(day, None)
        archived = sorted(observations)
        usable = sorted(day for day, (_, available) in observations.items()
                        if day <= at.date().isoformat() and available <= at)
        future = sum(day > at.date().isoformat() or available > at for day, (_, available) in observations.items())
        upper, unknown, complete_range = None, [], False
        if listing_day is not None:
            first = date.fromisoformat(listing_day)
            if first > at.date():
                upper, complete_range = 0, True
            else:
                # Bound the diagnostic before producing a multi-decade list.
                span = (at.date() - first).days + 1
                if span <= 3660:
                    expected = [(first + timedelta(days=index)).isoformat() for index in range(span)]
                    unknown = [day for day in expected if days.get(day, "UNKNOWN") == "UNKNOWN"]
                    complete_range = not unknown
                    if complete_range:
                        upper = sum(days[day] == "OPEN" and
                            datetime.combine(date.fromisoformat(day), time(13, 30), TAIPEI) <= at for day in expected)
        missing = max(0, required - len(usable))
        if listing and listing.get("status") == "CONFLICT":
            status = "CONFLICTING_LISTING_EVIDENCE"
        elif not missing:
            status = "WARMUP_COMPLETE"
        elif listing_day is None:
            status = "PENDING_LISTING_VERIFICATION"
        elif upper is not None and upper < required:
            status = ("NOT_YET_TWSE_LISTED_AT_CUTOFF" if upper == 0 and listing_day > at.date().isoformat()
                      else "SHORT_TWSE_LISTED_HISTORY_UNFILLABLE_AT_CUTOFF")
        elif "櫃轉市" in listing.get("remark", ""):
            status = "PREVIOUS_MARKET_HISTORY_REQUIRES_VERIFIED_TRANSITION"
        else:
            status = "VALID_PRICE_HISTORY_SHORT_EXTEND_WINDOW_OR_VERIFY_GAPS"
        rows.append(dict(symbol=symbol, market=history.get("market"), security_type=history.get("security_type"),
            status=status, valid_closes_at_cutoff=len(usable), missing_closes=missing,
            first_valid_close_at_or_before_cutoff=usable[0] if usable else None,
            first_valid_close_in_archive=archived[0] if archived else None,
            first_required_close_date_in_archive=archived[required - 1] if len(archived) >= required else None,
            archived_valid_closes=len(archived), future_or_late_prices_excluded=future,
            listing_evidence=deepcopy(listing), official_open_session_upper_bound_since_listing=upper,
            calendar_complete_since_listing=complete_range, unknown_calendar_days_since_listing=len(unknown),
            conflicting_dates=sorted(conflicts), excluded=dict(sorted(excluded.items()))))
    rows.sort(key=lambda row: row["symbol"])
    result = dict(schema="les-v7-normalized-warmup-audit/v1", cutoff=at.isoformat(), required_closes=required,
        listing_evidence_id=listing_id, calendar_sha256=digest(calendar),
        histories_sha256=digest(histories), historical_information_set_certified=False,
        rows=rows, symbol_count=len(rows), status_counts=dict(sorted(Counter(row["status"] for row in rows).items())),
        limitations=["NOT_A_MODEL_OR_TRADE_ELIGIBILITY_RESULT", "NO_SYNTHETIC_PRICE_FILLING",
                     "FINAL_LISTING_LIFECYCLE_NOT_HISTORICAL_PUBLICATION_PROOF"])
    result["audit_id"] = digest(result)
    return result


def audit_scope_warmup(freeze_path, listing_path, *, cutoff, root, required=513):
    """Verify frozen shards, normalize one at a time, publish a new audit.

    Future archived dates appear only as clearly labelled readiness diagnostics;
    they never contribute to the requested cutoff's close count. No model is
    fitted, candidate removed, or existing source/archive overwritten.
    """
    from infrastructure.les_v7_dataset_scope import _verify_freeze
    from infrastructure.les_v7_training_source import FINAL_CLOSE, load_training_source
    freeze_path = Path(freeze_path)
    source_root = freeze_path.parents[2]
    frozen = _verify_freeze(freeze_path, source_root)
    body = frozen["body"]
    at = aware_timestamp(cutoff).astimezone(TAIPEI)
    if not body["history_start"] <= at.date().isoformat() <= body["history_end"]:
        raise ValueError("WARMUP_AUDIT_CUTOFF_OUTSIDE_FROZEN_SCOPE")
    listing, documents = load_listing_final(listing_path)
    rows, references, calendar_days_id = [], [], None
    for chunk in body["chunks"]:
        source = load_training_source(source_root / "evidence" / chunk["evidence_id"] / "evidence.json",
            evaluation_cutoff=body["history_end"] + "T13:30:00+08:00",
            availability_policy=FINAL_CLOSE, reference_evidence_policy=body.get("reference_evidence_policy"))
        payload = source["payload"]
        if payload["source_evidence_ids"] != [chunk["evidence_id"]]:
            raise ValueError("WARMUP_AUDIT_SOURCE_CHANGED_AFTER_FREEZE_VERIFICATION")
        calendar = payload["calendars"][body["market"]]
        days_id = digest(calendar["days"])
        if calendar_days_id is not None and calendar_days_id != days_id:
            raise ValueError("WARMUP_AUDIT_SHARD_CALENDAR_CONFLICT")
        calendar_days_id = days_id
        audit = audit_normalized_warmup(payload["histories"], calendar, listing,
            cutoff=cutoff, required=required, document_reader=documents.read)
        rows.extend(audit["rows"])
        references.append(dict(evidence_id=chunk["evidence_id"], audit_id=audit["audit_id"],
            histories_sha256=audit["histories_sha256"], calendar_sha256=audit["calendar_sha256"],
            normalized_source_sha256=digest(payload)))
        del source, payload, calendar, audit
    rows.sort(key=lambda row: row["symbol"])
    if [row["symbol"] for row in rows] != body["universe"]:
        raise ValueError("WARMUP_AUDIT_FIXED_UNIVERSE_MISMATCH")
    from infrastructure.storage.les_v7_training import read_json_bounded, MAX_DATASET_BYTES
    if read_json_bounded(freeze_path, MAX_DATASET_BYTES) != frozen:
        raise ValueError("WARMUP_AUDIT_FREEZE_CHANGED_DURING_JOB")
    documents.check()
    root = Path(root)
    retained = EvidenceDocuments(root / "objects")
    listing_copy = retained.externalize(listing, document_reader=documents.read)
    result = dict(schema="les-v7-scope-warmup-audit/v1", scope_freeze_id=frozen["freeze_id"],
        scope_id=body["scope_id"], cutoff=at.isoformat(), required_closes=required,
        history_start=body["history_start"], history_end=body["history_end"],
        listing_evidence_id=listing["evidence_id"], calendar_days_sha256=calendar_days_id,
        source_chunks=references, rows=rows, symbol_count=len(rows),
        status_counts=dict(sorted(Counter(row["status"] for row in rows).items())),
        ordinary_complete=sum(row["security_type"] == "COMMON" and row["missing_closes"] == 0 for row in rows),
        ordinary_incomplete=sum(row["security_type"] == "COMMON" and row["missing_closes"] > 0 for row in rows),
        listing_packet=listing_copy, historical_information_set_certified=False,
        research_only=True, fitting_performed=False, approved_for_live=False, execution_action="NO_ACTION",
        limitations=["WARMUP_COVERAGE_NOT_MODEL_OR_TRADE_ELIGIBILITY",
            "FUTURE_ARCHIVE_READINESS_DATES_NOT_USED_AT_EARLIER_CUTOFF",
            "FIXED_FINAL_UNIVERSE_NOT_HISTORICAL_MEMBERSHIP",
            "RAW_PRICE_COORDINATE_RESETS_AND_OTHER_NATIVE_GATES_REMAIN"])
    identifier = digest(result)
    path = root / "audits" / identifier / "manifest.json"
    if path.exists():
        earlier = read_json_bounded(path, MAX_DATASET_BYTES)
        if earlier.get("audit_id") != identifier or earlier.get("body") != result:
            raise ValueError("WARMUP_AUDIT_IMMUTABLE_CONFLICT")
    else:
        write_once(path, dict(audit_id=identifier, body=result, completed_at=utc_now()))
    return path
