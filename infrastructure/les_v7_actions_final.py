"""Bounded official FINAL price-coordinate evidence for retrospective research.

These are today's official historical result tables, NOT historical publication
receipts. Three exact categories are covered; this packet neither proves trading
permission nor certifies every company action or historical information set.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode

from application.replay_contracts import aware_timestamp, digest
from application.research_evidence import verified_document
from domain.errors import AccessBlockedError, NetworkError
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments

SCHEMA = "les-v7-official-final-actions/v1"
POLICY = "OFFICIAL_FINAL_ACTIONS_RECONSTRUCTION_R1"
MAX_BYTES = 16 * 1024 * 1024
MAX_REQUESTS = 24
MAX_ROWS = 200_000
TWT49U_READ_TIMEOUT_SECONDS = 60
FEEDS = {
    "TWT49U": ("/exRight/TWT49U", "EX_RIGHTS_AND_DIVIDENDS", "資料日期", "股票代號"),
    "TWTAUU": ("/reducation/TWTAUU", "CAPITAL_REDUCTION_RESUMPTION", "恢復買賣日期", "股票代號"),
    "TWTB8U": ("/change/TWTB8U", "PAR_VALUE_CHANGE_RESUMPTION", "恢復買賣日期", "股票代號"),
}
CATEGORY_SCOPE = [item[1] for item in FEEDS.values()]
REQUIRED_FIELDS = {
    "TWT49U": ["資料日期", "股票代號", "股票名稱", "除權息前收盤價", "除權息參考價", "權值+息值", "權/息"],
    "TWTAUU": ["恢復買賣日期", "股票代號", "名稱", "停止買賣前收盤價格", "恢復買賣參考價", "減資原因", "詳細資料"],
    "TWTB8U": ["恢復買賣日期", "股票代號", "名稱", "停止買賣前收盤價格", "恢復買賣參考價", "詳細資料"],
}
LIMITATIONS = ["NOT_ORIGINAL_HISTORICAL_VINTAGE", "HISTORICAL_PUBLICATION_TIME_UNVERIFIED",
              "HALTS_AND_TRADING_RESTRICTIONS_NOT_CERTIFIED", "NOT_ALL_COMPANY_ACTION_CATEGORIES",
              "NO_PRICE_OR_SHARE_ADJUSTMENTS_IMPLEMENTED"]


def _range(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last - first).days >= 3660 or first.isoformat() != start or last.isoformat() != end:
        raise ValueError("BOUNDED_CANONICAL_ACTION_RANGE_REQUIRED")
    return first, last


def _official_date(value):
    if not isinstance(value, str):
        raise ValueError("ACTION_EFFECTIVE_DATE_REQUIRED")
    match = re.fullmatch(r"(\d{3,4})(?:年|/)(\d{1,2})(?:月|/)(\d{1,2})(?:日)?", value.strip())
    if not match:
        raise ValueError("UNRECOGNIZED_OFFICIAL_ACTION_DATE")
    year, month, day = map(int, match.groups())
    return date(year + 1911 if year < 1911 else year, month, day).isoformat()


def _json(content):
    if not isinstance(content, bytes) or not 0 < len(content) <= MAX_BYTES:
        raise ValueError("ACTION_RESPONSE_SIZE_LIMIT")
    text = content.decode("utf-8-sig")
    lower = text[:8192].lower()
    if any(marker in lower for marker in ("access denied", "access blocked", "forbidden", "因為安全性考量")):
        raise AccessBlockedError("TWSE action source blocked", retryable=False)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("DUPLICATE_ACTION_JSON_KEY")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("NON_FINITE_ACTION_SOURCE_VALUE:" + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def parse_actions_response(feed, content, *, start, end):
    """Verify exact range and schema; an empty OK table is not a missing feed."""
    first, last = _range(start, end)
    if feed not in FEEDS:
        raise ValueError("UNRECOGNIZED_ACTION_FEED")
    payload = _json(content)
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        raise ValueError("ACTION_RESPONSE_NOT_COMPLETE_OK_TABLE")
    compact_start, compact_end = start.replace("-", ""), end.replace("-", "")
    if feed == "TWTB8U":
        params = payload.get("params", {})
        if (params.get("startDate"), params.get("endDate"), params.get("controller"), params.get("action")) != (
                compact_start, compact_end, "change", feed):
            raise ValueError("ACTION_RESPONSE_RANGE_OR_FEED_MISMATCH")
    elif (payload.get("strDate"), payload.get("endDate")) != (compact_start, compact_end):
        raise ValueError("ACTION_RESPONSE_RANGE_OR_FEED_MISMATCH")
    fields, rows = payload.get("fields"), payload.get("data")
    _, category, date_field, symbol_field = FEEDS[feed]
    if (not isinstance(fields, list) or any(not isinstance(f, str) for f in fields)
            or len(set(fields)) != len(fields) or any(f not in fields for f in REQUIRED_FIELDS[feed])
            or not isinstance(rows, list) or len(rows) > MAX_ROWS):
        raise ValueError("ACTION_RESPONSE_SCHEMA_OR_ROW_BOUND_MISMATCH")
    for name in ("total", "totalCount", "totleCount"):
        if name in payload and (type(payload[name]) is not int or payload[name] != len(rows)):
            raise ValueError("ACTION_RESPONSE_PARTIAL_OR_PAGED")
    actions = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(fields):
            raise ValueError("ACTION_RESPONSE_ROW_WIDTH_MISMATCH")
        day, symbol = _official_date(row[fields.index(date_field)]), row[fields.index(symbol_field)]
        if (not isinstance(symbol, str) or not re.fullmatch(r"[0-9A-Z]{4,10}", symbol.strip())
                or not first.isoformat() <= day <= last.isoformat()):
            raise ValueError("ACTION_ROW_IDENTITY_OR_DATE_OUTSIDE_QUERY")
        actions.append(dict(symbol=symbol.strip(), date=day, category=category, feed=feed, raw_row_sha256=digest(row)))
    # Whole transport bytes remain retained, including ancillary current financial
    # fields in TWT49U; none of those values become historical financial inputs.
    return dict(feed=feed, category=category, start=start, end=end, status="COMPLETE_OFFICIAL_FINAL_TABLE",
                field_names=deepcopy(fields), row_count=len(rows), actions=sorted(actions, key=digest),
                raw_sha256=hashlib.sha256(content).hexdigest(), source_scope="ALL_TWSE_INSTRUMENTS")


def _packet_id(packet):
    return digest(dict(schema=SCHEMA, policy=POLICY, market="TWSE", start=packet["start"], end=packet["end"],
        feeds=[dict(feed=r["feed"], url=r["document"]["url"], raw_sha256=r["document"]["sha256"],
                    query=r["query"]) for r in packet["feeds"]]))


def _verify_record(record, *, start, end, document_reader=None):
    feed, document = record["feed"], record["document"]
    if feed not in FEEDS:
        raise ValueError("UNRECOGNIZED_ACTION_FEED")
    params = dict(startDate=start.replace("-", ""), endDate=end.replace("-", ""), response="json")
    url = "https://www.twse.com.tw/rwd/zh" + FEEDS[feed][0] + "?" + urlencode(params)
    if (document.get("url") != url or record.get("query") != params
            or record.get("source_scope") != "ALL_TWSE_INSTRUMENTS"
            or record.get("historical_available_at") is not None
            or record.get("fetched_at") != document.get("retrieved_at")):
        raise ValueError("ACTION_DOCUMENT_SOURCE_OR_TIME_BINDING_MISMATCH")
    if aware_timestamp(record["fetched_at"]) < aware_timestamp(end + "T13:30:00+08:00"):
        raise ValueError("FINAL_ACTION_CAPTURE_PRECEDES_RANGE_COMPLETION")
    content = verified_document(document, document_reader=document_reader)
    parsed = parse_actions_response(feed, content, start=start, end=end)
    if record.get("parsed") != parsed:
        raise ValueError("ACTION_ORIGINAL_AND_PARSED_RESULT_MISMATCH")


def verify_packet(packet, *, document_reader=None):
    """Re-read every bound original; checksums never certify historical timing."""
    if (not isinstance(packet, dict) or packet.get("schema") != SCHEMA or packet.get("policy") != POLICY
            or packet.get("market") != "TWSE" or packet.get("historical_available_at") is not None
            or packet.get("historical_information_set_certified") is not False
            or packet.get("category_scope") != CATEGORY_SCOPE or packet.get("limitations") != LIMITATIONS
            or not isinstance(packet.get("feeds"), list) or any(not isinstance(r, dict) for r in packet["feeds"])
            or [r.get("feed") for r in packet["feeds"]] != list(FEEDS)):
        raise ValueError("EXPLICIT_FINAL_ACTION_PACKET_REQUIRED")
    _range(packet["start"], packet["end"])
    for record in packet["feeds"]:
        _verify_record(record, start=packet["start"], end=packet["end"], document_reader=document_reader)
    if packet.get("evidence_id") != _packet_id(packet):
        raise ValueError("ACTION_PACKET_ID_MISMATCH")
    return deepcopy(packet)


def collect_actions_final(start, end, root, *, client=None, max_requests=MAX_REQUESTS, resume=False):
    """Bounded capture; optional range-bound feed receipts survive interruption.

    Preparation uses a single-attempt transport shared with its global budget.
    Retry one transient network failure here for that transport, charging every
    attempt through ``before_attempt``. A blocked/invalid source is never retried
    or replaced by a partial table. No completion manifest exists until all
    three original sources pass the unchanged packet checks.
    """
    _range(start, end)
    if type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS:
        raise ValueError("BOUNDED_ACTION_REQUEST_BUDGET_REQUIRED")
    if type(resume) is not bool:
        raise ValueError("BOOLEAN_ACTION_RESUME_REQUIRED")
    client = client or TwseHttpClient(TwseHttpConfig(timeout_seconds=25, max_attempts=2,
        min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
    if client.config.allow_insecure_ssl_fallback or client.config.min_request_interval_seconds < 1:
        raise ValueError("SECURE_RATE_LIMITED_ACTION_TRANSPORT_REQUIRED")
    root = Path(root)
    documents, requests = EvidenceDocuments(root / "objects"), 0

    def reserve():
        nonlocal requests
        if requests >= max_requests:
            raise ValueError("ACTION_REQUEST_BUDGET_EXHAUSTED")
        requests += 1

    records = []
    capture_id = digest(dict(schema=SCHEMA, policy=POLICY, start=start, end=end))
    for feed, (route, _, _, _) in FEEDS.items():
        checkpoint = root / "captures" / capture_id / (feed + ".json")
        if resume and checkpoint.exists():
            if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size > 3 * MAX_BYTES:
                raise ValueError("ACTION_CAPTURE_RECEIPT_PATH_OR_SIZE_INVALID")
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("feed") != feed:
                raise ValueError("ACTION_CAPTURE_RECEIPT_FEED_MISMATCH")
            _verify_record(record, start=start, end=end, document_reader=documents.read)
            records.append(record)
            continue
        query = dict(startDate=start.replace("-", ""), endDate=end.replace("-", ""), response="json")
        url = "https://www.twse.com.tw/rwd/zh" + route
        # Do not multiply the retry count of the collector's normal transport.
        attempts = 2 if client.config.max_attempts == 1 else 1
        # This whole-range table can span several years. Give its body a
        # separate read allowance; connect timeout and the shared calendar /
        # market transport configuration remain unchanged on success or error.
        options = (dict(read_timeout_seconds=TWT49U_READ_TIMEOUT_SECONDS)
                   if feed == "TWT49U" and client.config.timeout_seconds < TWT49U_READ_TIMEOUT_SECONDS else {})
        for attempt in range(attempts):
            try:
                response = client.request_bytes("GET", url, params=query, before_attempt=reserve,
                    context="TWSE official FINAL action " + feed, content_validator=lambda value: _json(value),
                    **options)
                break
            except NetworkError as error:
                if not error.retryable or attempt + 1 == attempts:
                    raise
                # Preparation shares a long-lived transport with calendar and
                # market collection. Do not repeat a stalled pooled session.
                reset = getattr(client, 'reset_session', None)
                if callable(reset):
                    reset()
                time.sleep(max(client.config.min_request_interval_seconds,
                               1, min(5, client.config.backoff_seconds)))
        parsed = parse_actions_response(feed, response.content, start=start, end=end)
        fetched_at = utc_now()
        document = dict(url=url + "?" + urlencode(query), retrieved_at=fetched_at,
                        sha256=parsed["raw_sha256"], body_ref=documents.store(response.content))
        record = dict(feed=feed, query=query, source_scope="ALL_TWSE_INSTRUMENTS", parsed=parsed,
                      fetched_at=fetched_at, historical_available_at=None, document=document)
        _verify_record(record, start=start, end=end, document_reader=documents.read)
        if resume:
            write_once(checkpoint, record)
        records.append(record)
    packet = dict(schema=SCHEMA, policy=POLICY, market="TWSE", start=start, end=end, feeds=records,
                  historical_available_at=None, historical_information_set_certified=False,
                  category_scope=CATEGORY_SCOPE, limitations=LIMITATIONS)
    packet["evidence_id"] = _packet_id(packet)
    manifest = root / packet["evidence_id"] / "manifest.json"
    if manifest.exists():
        # Preserve the FIRST collection receipts when unchanged business content
        # is collected again. The retrieval time is not an invented release time.
        load_actions_final(manifest)
    else:
        verify_packet(packet, document_reader=documents.read)
        write_once(manifest, packet)
    return manifest


def load_actions_final(manifest_path):
    path = Path(manifest_path)
    if path.stat().st_size > 3 * MAX_BYTES:
        raise ValueError("ACTION_PACKET_SIZE_LIMIT")
    packet = json.loads(path.read_text(encoding="utf-8"))
    documents = EvidenceDocuments(path.parent.parent / "objects")
    return verify_packet(packet, document_reader=documents.read)


def build_final_coordinate_coverages(histories, packet, *, pending_markers=(), document_reader=None):
    """Maximal quiet segments, excluding events and EVERY unresolved raw marker.

    The limited three-category claim is explicit. No action is adjusted away;
    event-crossing H20 windows do not qualify under the raw-price policy.
    """
    packet = verify_packet(packet, document_reader=document_reader)
    first, last = _range(packet["start"], packet["end"])
    dates = {}
    for record in packet["feeds"]:
        for row in record["parsed"]["actions"]:
            dates.setdefault(row["symbol"], set()).add(row["date"])
    for row in pending_markers:
        day = row["date"]
        if len(day) == 8 and day.isdigit():
            day = day[:4] + "-" + day[4:6] + "-" + day[6:]
        date.fromisoformat(day)
        dates.setdefault(row["symbol"], set()).add(day)
    fetched_at = max((r["fetched_at"] for r in packet["feeds"]), key=aware_timestamp)
    evidence_ids = [packet["evidence_id"], *[r["document"]["sha256"] for r in packet["feeds"]]]
    output = []
    for history in histories:
        if history.get("market") != "TWSE" or not history.get("coordinate_id") or not history.get("symbol"):
            raise ValueError("SOURCE_BOUND_TWSE_HISTORY_REQUIRED")
        excluded = set(dates.get(history["symbol"], set()))
        for bar in history.get("bars", []):
            if bar.get("Corporate Action State") not in {None, "NO_ACTION"}:
                excluded.add(bar["Date"])
        cursor = first
        for bound in [*sorted(date.fromisoformat(d) for d in excluded if start_in_range(d, first, last)), last + timedelta(days=1)]:
            end = bound - timedelta(days=1)
            if cursor <= end:
                output.append(dict(schema="les-v7-coordinate-coverage/v1", policy=POLICY, status="VERIFIED_NO_ACTIONS",
                    coordinate_id=history["coordinate_id"], symbol=history["symbol"], market="TWSE",
                    start_session=cursor.isoformat(), end_session=end.isoformat(), evidence_ids=list(evidence_ids),
                    available_at=fetched_at, fetched_at=fetched_at, historical_available_at=None,
                    category_scope=list(CATEGORY_SCOPE), limitations=list(LIMITATIONS),
                    historical_information_set_certified=False, actions_packet_id=packet["evidence_id"]))
            cursor = bound + timedelta(days=1)
    return deepcopy(output)


def start_in_range(day, first, last):
    return first <= date.fromisoformat(day) <= last
