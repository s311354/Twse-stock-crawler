"""Explicit retrospective official-final calendars, never PIT publication proof.

The annual transport bytes and complete official closure-news queries are kept.
This module does not change strict calendar_at(), fabricate original availability,
or infer a closure from absent price bars. Final reconstruction is an opt-in
research convention and must not be supplied to a live/strict-replay contract.
"""
from __future__ import annotations

import datetime as dt
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlencode, urlparse
from zoneinfo import ZoneInfo

from application.replay_contracts import aware_timestamp, digest
from application.research_evidence import verified_document
from domain.errors import DataSourceError
from infrastructure.calendar.twse_calendar import TwseCalendar, TWSE_CALENDAR_URL
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments

SCHEMA = "les-v7-official-final-calendar/v1"
POLICY = "FINAL_OFFICIAL_CALENDAR_RECONSTRUCTION_R1"
HOST = "https://www.twse.com.tw"
NEWS_LIST = HOST + "/rwd/zh/news/newsList"
NEWS_DETAIL = HOST + "/rwd/zh/news/newsDetail"
TAIPEI = ZoneInfo("Asia/Taipei")
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
CLOSURE_TITLE = re.compile(r"^臺灣證券交易所集中交易市場(\d{3})年(\d{1,2})月(\d{1,2})日休市一天$")


def _json_document(document, documents):
    body = verified_document(document, document_reader=documents.read)
    if len(body) > MAX_DOCUMENT_BYTES:
        raise ValueError("FINAL_CALENDAR_DOCUMENT_TOO_LARGE")
    return json.loads(body)


def _official_url(document, path, params):
    parsed = urlparse(document["url"])
    if (parsed.scheme != "https" or parsed.netloc != "www.twse.com.tw"
            or parsed.path != path or parse_qs(parsed.query) != {k: [str(v)] for k, v in params.items()}):
        raise ValueError("FINAL_CALENDAR_SOURCE_QUERY_MISMATCH")


def _annual(row, year, documents):
    document = row["document"]
    _official_url(document, "/holidaySchedule/holidaySchedule", dict(response="json", date=f"{year}0101"))
    payload = _json_document(document, documents)
    if (row["payload"] != payload or row["payload_sha256"] != digest(payload)
            or payload.get("stat") != "ok" or payload.get("queryYear") != year
            or payload.get("date") != f"{year}0101"
            or payload.get("title") != f"{year-1911} 年市場開休市日期"
            or payload.get("fields") != ["日期", "名稱", "說明"]
            or payload.get("total") != len(payload.get("data", [])) or not payload.get("data")):
        raise ValueError("FINAL_ANNUAL_PAYLOAD_BINDING_MISMATCH")
    if any(not isinstance(item, list) or len(item) != 3
           or dt.date.fromisoformat(item[0]).year != year for item in payload["data"]):
        raise ValueError("FINAL_ANNUAL_DATE_RANGE_MISMATCH")
    calendar = TwseCalendar(fetcher=lambda: payload)
    if calendar.trading_day_status(dt.date(year, 1, 1)).value == "unknown":
        raise ValueError("FINAL_ANNUAL_YEAR_COVERAGE_MISMATCH")
    return calendar


def _closure_query(row, year, documents):
    document = row["document"]
    _official_url(document, "/rwd/zh/news/newsList",
                  dict(startDate=year, endDate=year, tag="休市", response="json"))
    payload = _json_document(document, documents)
    if (row["payload"] != payload or row["payload_sha256"] != digest(payload)
            or payload.get("stat") != "ok" or payload.get("fields") != ["項次", "標題", "日期", "zhId", "enId"]
            or payload.get("totalCount") != len(payload.get("data", []))):
        raise ValueError("FINAL_CLOSURE_QUERY_INCOMPLETE_OR_MISMATCH")
    found = {}
    for item in payload["data"]:
        if not isinstance(item, list) or len(item) != 5:
            raise ValueError("FINAL_CLOSURE_QUERY_ROW_INVALID")
        match = CLOSURE_TITLE.fullmatch(item[1])
        if not match:
            continue  # Other news mentioning closure is preserved, not misclassified.
        event_date = dt.date(int(match[1])+1911, int(match[2]), int(match[3])).isoformat()
        if int(event_date[:4]) != year or not re.fullmatch(r"[0-9a-f]{32}", item[3]):
            raise ValueError("FINAL_CLOSURE_QUERY_DATE_OR_ID_MISMATCH")
        if item[3] in found and found[item[3]] != event_date:
            raise ValueError("FINAL_CLOSURE_QUERY_CONFLICT")
        found[item[3]] = event_date
    return found


def _pdf_text(content):
    from infrastructure.filing_pdf_comparison import extract_pdf_pages
    pages, _ = extract_pdf_pages(content)
    return "\n".join(pages)


def _closure(event, documents, *, extract_text):
    if event.get("market") != "TWSE" or event.get("status") != "CLOSED":
        raise ValueError("FINAL_CLOSURE_MARKET_OR_STATUS_MISMATCH")
    day = dt.date.fromisoformat(event["date"])
    _official_url(event["detail_document"], "/rwd/zh/news/newsDetail", dict(id=event["news_id"], response="json"))
    detail = _json_document(event["detail_document"], documents)
    if detail.get("stat") != "ok" or not re.fullmatch(r"\d{12}", detail.get("date", "")):
        raise ValueError("FINAL_CLOSURE_DETAIL_INVALID")
    released = dt.datetime.strptime(detail["date"], "%Y%m%d%H%M").replace(tzinfo=TAIPEI)
    title = f"臺灣證券交易所集中交易市場{day.year-1911}年{day.month}月{day.day}日休市一天"
    table = detail["tables"][0]
    if table.get("title") != title or table.get("fields") != ["text", "html", "pdf", "lang"]:
        raise ValueError("FINAL_CLOSURE_DETAIL_TITLE_MISMATCH")
    path = table["data"][2]
    if not re.fullmatch(r"/news/news/tsecnews/[0-9a-f]{32}\.pdf", path):
        raise ValueError("FINAL_CLOSURE_PDF_PATH_INVALID")
    if event["document"]["url"] != HOST + "/staticFiles" + path:
        raise ValueError("FINAL_CLOSURE_PDF_BINDING_MISMATCH")
    content = verified_document(event["document"], document_reader=documents.read)
    if len(content) > MAX_DOCUMENT_BYTES or not content.startswith(b"%PDF-"):
        raise ValueError("FINAL_CLOSURE_PDF_REQUIRED")
    text = extract_text(content)
    compact = re.sub(r"\s+", "", text)
    release_text = f"中華民國{released.year-1911}年{released.month}月{released.day}日"
    effective_text = f"{day.year-1911}年{day.month}月{day.day}日集中交易市場全日休市"
    if (title not in compact or release_text not in compact or effective_text not in compact
            or released.date() > day
            or event["declared_published_at"] != released.isoformat()):
        raise ValueError("FINAL_CLOSURE_SOURCE_TEXT_OR_RELEASE_MISMATCH")
    if any(aware_timestamp(event[key]["retrieved_at"]) < released
           for key in ("document", "detail_document")):
        raise ValueError("FINAL_CLOSURE_DOCUMENT_PREDATES_RELEASE")
    return released.isoformat()


def select_final_calendar(calendar, cutoff, *, documents, start, end, evidence_id, extract_text=None):
    """Pure revalidation/selection of FINAL data; original known time stays null.

    availability_by_date is explicitly the retrospective close convention. It
    is not a historical observation receipt or strict historical publication.
    """
    cutoff = aware_timestamp(cutoff) if isinstance(cutoff, str) else cutoff
    if cutoff.tzinfo is None:
        raise ValueError("FINAL_CALENDAR_CUTOFF_MUST_BE_AWARE")
    first = dt.date.fromisoformat(start) if isinstance(start, str) else start
    last = dt.date.fromisoformat(end) if isinstance(end, str) else end
    if (first > last or (last-first).days > 3660
            or dt.datetime.combine(last, dt.time(13,30), TAIPEI) > cutoff):
        raise ValueError("FINAL_CALENDAR_RANGE_AFTER_CUTOFF")
    if (calendar.get("schema") != SCHEMA or calendar.get("source") != "TWSE_OFFICIAL"
            or calendar.get("policy") != POLICY or calendar.get("historical_information_set_certified") is not False
            or calendar.get("original_historical_available_at") is not None):
        raise ValueError("EXPLICIT_RETROSPECTIVE_CALENDAR_REQUIRED")
    annual, query_ids, excluded = {}, {}, []
    for year in range(first.year, last.year+1):
        try:
            annual[year] = _annual(calendar["documents"][str(year)], year, documents)
            query_ids[year] = _closure_query(calendar["closure_queries"][str(year)], year, documents)
        except (KeyError, ValueError, TypeError) as error:
            annual.pop(year, None)
            excluded.append(dict(year=year, reason="FINAL_ANNUAL_OR_CLOSURE_COVERAGE_UNVERIFIED", detail=str(error)))
    states, source_versions = {}, {}
    for index in range((last-first).days+1):
        day = first + dt.timedelta(days=index)
        states[day.isoformat()] = (annual[day.year].trading_day_status(day).value.upper()
                                   if day.year in annual else "UNKNOWN")
        if day.year in annual:
            source_versions[day.isoformat()] = [calendar["documents"][str(day.year)]["document"]["sha256"],
                                                calendar["closure_queries"][str(day.year)]["document"]["sha256"]]
    received = {}
    for event in calendar.get("announcements", []):
        day = event.get("date")
        if not isinstance(day, str) or day not in states:
            continue
        try:
            year = int(day[:4])
            if query_ids.get(year, {}).get(event["news_id"]) != day:
                raise ValueError("FINAL_CLOSURE_NOT_BOUND_TO_QUERY")
            _closure(event, documents, extract_text=extract_text or _pdf_text)
            received.setdefault(day, []).append(event["news_id"])
            states[day] = "CLOSED"
            source_versions.setdefault(day, []).extend([event["detail_document"]["sha256"], event["document"]["sha256"]])
            if len(received[day]) > 1:
                raise ValueError("FINAL_CLOSURE_DUPLICATE_OR_CONFLICT")
        except (KeyError, ValueError, TypeError) as error:
            states[day] = "UNKNOWN"
            excluded.append(dict(date=day, reason="FINAL_CLOSURE_UNVERIFIED", detail=str(error)))
    for year, items in query_ids.items():
        for news_id, day in items.items():
            if day in states and news_id not in received.get(day, []):
                states[day] = "UNKNOWN"
                excluded.append(dict(date=day, reason="FINAL_CLOSURE_DOCUMENT_MISSING", news_id=news_id))
    availability = {day: (dt.datetime.combine(dt.date.fromisoformat(day), dt.time(13,30), TAIPEI).isoformat()
                          if status != "UNKNOWN" else None) for day, status in states.items()}
    result = dict(source="TWSE_OFFICIAL", sessions=sorted(day for day, status in states.items() if status == "OPEN"),
                  days=states, available_at=max((v for v in availability.values() if v), default=None),
                  availability_by_date=availability, evidence_ids=[evidence_id, digest(calendar)],
                  publication_events=[], scope_start=first.isoformat(), scope_end=last.isoformat(),
                  availability_policy=POLICY, original_historical_available_at=None,
                  historical_information_set_certified=False, source_versions_by_date=source_versions,
                  reconstruction_limitations=["FINAL_ANNUAL_VERSION_NOT_ORIGINAL_OBSERVATION",
                      "EMERGENCY_SUPPLEMENT_FROM_COMPLETE_OFFICIAL_NEWS_QUERY_NOT_GLOBAL_ARCHIVE_CERTIFICATION"])
    return result, dict(unknown_dates=[day for day, state in states.items() if state == "UNKNOWN"],
                       excluded=excluded, policy=POLICY, historical_information_set_certified=False,
                       collection_errors=calendar.get("collection_errors", []))


def collect_final_calendars(root, *, years=(2023,2024,2025,2026), client=None, max_requests=20,
                            now=utc_now, previous=None, refresh_years=()):
    """Manual, bounded official capture; ACCESS_BLOCKED stops all later requests."""
    if not 1 <= max_requests <= 20 or not years or len(set(years)) != len(years):
        raise ValueError("INVALID_FINAL_CALENDAR_COLLECTION_BOUNDS")
    if any(type(year) is not int or not 2000 <= year <= 2100 for year in years):
        raise ValueError("INVALID_FINAL_CALENDAR_YEAR")
    if (not isinstance(refresh_years, (tuple, list, set))
            or any(type(year) is not int or year not in years for year in refresh_years)):
        raise ValueError("INVALID_FINAL_CALENDAR_REFRESH_YEAR")
    root = Path(root)
    store = EvidenceDocuments(root / "objects")
    transport = client or TwseHttpClient(config=TwseHttpConfig(timeout_seconds=20, max_attempts=1,
                         min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
    payload = dict(schema=SCHEMA, source="TWSE_OFFICIAL", policy=POLICY,
                   historical_information_set_certified=False, original_historical_available_at=None,
                   documents={}, closure_queries={}, announcements=[], collection_errors=[])
    if previous is not None:
        earlier, earlier_documents = load_final_calendar(previous)
        if (earlier.get("schema") != SCHEMA or earlier.get("policy") != POLICY
                or earlier.get("source") != "TWSE_OFFICIAL"
                or earlier.get("historical_information_set_certified") is not False
                or earlier.get("original_historical_available_at") is not None):
            raise ValueError("FINAL_CALENDAR_REPAIR_POLICY_MISMATCH")
        # New immutable source identity; never alter the partial original.
        preserved = store.externalize(deepcopy(earlier), document_reader=earlier_documents.read)
        payload.update(documents={y:r for y,r in preserved["documents"].items() if int(y) in years},
            closure_queries={y:r for y,r in preserved["closure_queries"].items() if int(y) in years},
            announcements=[e for e in preserved["announcements"] if int(e["date"][:4]) in years],
            parent_calendar_id=digest(earlier), previous_collection_errors=earlier.get("collection_errors", []))
        # A complete old news query cannot prove that no closure was announced
        # since capture. Refresh explicitly selected years into a NEW manifest;
        # the parent and its original transport bytes remain immutable.
        for year in refresh_years:
            payload["documents"].pop(str(year), None)
            payload["closure_queries"].pop(str(year), None)
    count, blocked = 0, False

    def capture(url, params=None):
        nonlocal count, blocked
        if blocked:
            raise ValueError("FINAL_CALENDAR_REFRESH_STOPPED_ACCESS_BLOCKED")
        if count >= max_requests:
            raise ValueError("FINAL_CALENDAR_REQUEST_LIMIT")
        count += 1
        response = transport.request_bytes("GET", url, params=params, context="Final calendar evidence capture")
        body = response.content
        if len(body) > MAX_DOCUMENT_BYTES:
            raise ValueError("FINAL_CALENDAR_DOCUMENT_TOO_LARGE")
        lowered = body[:8192].lower()
        if any(marker in lowered for marker in (b"access denied", b"request rejected", b"security policy",
                                                "安全性".encode(), "安全阻擋".encode(), b"access blocked")):
            blocked = True
            raise ValueError("FINAL_CALENDAR_ACCESS_BLOCKED")
        source_url = url + ("?" + urlencode(params) if params else "")
        return dict(url=source_url, sha256=hashlib.sha256(body).hexdigest(), retrieved_at=now(),
                    body_ref=store.store(body), representation="ORIGINAL_TRANSPORT_BYTES")

    def failure(stage, error, **binding):
        nonlocal blocked
        category = error.category.value if isinstance(error, DataSourceError) else None
        if category == "access_blocked" or "ACCESS_BLOCKED" in str(error):
            blocked = True
        payload["collection_errors"].append(dict(stage=stage, reason=str(error), category=category, **binding))

    pending, refreshed_queries = [], {}
    for year in sorted(years):
        try:
            row = payload["documents"].get(str(year))
            if row is None:
                document = capture(TWSE_CALENDAR_URL, dict(response="json", date=f"{year}0101"))
                raw = _json_document(document, store)
                row = dict(payload=raw, payload_sha256=digest(raw), document=document)
            _annual(row, year, store)
            payload["documents"][str(year)] = row
        except (OSError, ValueError, TypeError, KeyError, DataSourceError) as error:
            failure("annual", error, year=year)
        try:
            row = payload["closure_queries"].get(str(year))
            if row is None:
                document = capture(NEWS_LIST, dict(startDate=year, endDate=year, tag="休市", response="json"))
                raw = _json_document(document, store)
                row = dict(payload=raw, payload_sha256=digest(raw), document=document)
            found = _closure_query(row, year, store)
            payload["closure_queries"][str(year)] = row
            if year in refresh_years:
                refreshed_queries[year] = set(found)
            pending.extend(sorted(found.items()))
        except (OSError, ValueError, TypeError, KeyError, DataSourceError) as error:
            failure("closure_query", error, year=year)
    # A refreshed complete query replaces membership, not unchanged verified
    # transport bytes. Re-fetching every old PDF on every day would starve the
    # bounded market budget. Newly listed IDs still require complete evidence.
    payload["announcements"] = [event for event in payload["announcements"]
        if int(event["date"][:4]) not in refreshed_queries
        or event["news_id"] in refreshed_queries[int(event["date"][:4])]]
    for news_id, day in pending:
        try:
            existing = [e for e in payload["announcements"] if e["news_id"] == news_id]
            if existing:
                if len(existing) != 1 or existing[0]["date"] != day:
                    raise ValueError("FINAL_CLOSURE_REPAIR_BINDING_CONFLICT")
                _closure(existing[0], store, extract_text=_pdf_text)
                continue
            detail_doc = capture(NEWS_DETAIL, dict(id=news_id, response="json"))
            detail = _json_document(detail_doc, store)
            path = detail["tables"][0]["data"][2]
            if not re.fullmatch(r"/news/news/tsecnews/[0-9a-f]{32}\.pdf", path):
                raise ValueError("FINAL_CLOSURE_PDF_PATH_INVALID")
            document = capture(HOST + "/staticFiles" + path)
            declared = dt.datetime.strptime(detail["date"], "%Y%m%d%H%M").replace(tzinfo=TAIPEI).isoformat()
            event = dict(news_id=news_id, date=day, status="CLOSED", market="TWSE",
                         detail_document=detail_doc, document=document, declared_published_at=declared,
                         original_historical_available_at=None, policy=POLICY)
            _closure(event, store, extract_text=_pdf_text)
            payload["announcements"].append(event)
        except (OSError, ValueError, TypeError, KeyError, DataSourceError) as error:
            failure("closure_document", error, date=day, news_id=news_id)
    payload["announcements"].sort(key=lambda item: (item["date"], item["news_id"]))
    if (previous is not None and count == 0 and not payload["collection_errors"]
            and not earlier.get("collection_errors")
            and sorted(earlier["documents"]) == sorted(str(year) for year in years)
            and sorted(earlier["closure_queries"]) == sorted(str(year) for year in years)
            and earlier["announcements"] == payload["announcements"]):
        return Path(previous)
    source_id = digest(payload)
    path = root / "collections" / source_id / "calendar.json"
    write_once(path, dict(calendar_id=source_id, payload=payload))
    write_once(path.parent / "completion.json", dict(schema="les-v7-final-calendar-collection/v1",
        calendar_id=source_id, request_count=count, max_requests=max_requests, completed_at=now(),
        status="PARTIAL" if payload["collection_errors"] else "SOURCE_CAPTURED_RETROSPECTIVE_ONLY"))
    return path


def load_final_calendar(path):
    path = Path(path)
    with path.open("rb") as stream:
        content = stream.read(2 * 1024 * 1024 + 1)
    if len(content) > 2 * 1024 * 1024:
        raise ValueError("FINAL_CALENDAR_MANIFEST_TOO_LARGE")
    record = json.loads(content)
    if record.get("calendar_id") != digest(record.get("payload")):
        raise ValueError("FINAL_CALENDAR_ID_MISMATCH")
    root = path.parent.parent.parent
    documents = EvidenceDocuments(root / "objects")
    documents.verify(record["payload"])
    return record["payload"], documents
