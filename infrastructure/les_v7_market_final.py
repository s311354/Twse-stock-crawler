"""Bounded manual official final-bar warmup collection, never PIT observation."""
from datetime import date
import json
from pathlib import Path

from application.replay_contracts import digest
from application.research_evidence import verified_document
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.storage.les_v7_training import read_json_bounded
from twstockcrawler import TwStockCrawler

URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
SCHEMA = "les-v7-official-final-market/v1"
MAX_BODY = 16 * 1024 * 1024
STOCK_FIELDS = ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價",
                "漲跌(+/-)", "漲跌價差", "最後揭示買價", "最後揭示買量", "最後揭示賣價", "最後揭示賣量", "本益比"]


def parse_final_market(content, day):
    """Use the existing native stock/TAIEX adapters, with exact query binding."""
    if len(content) > MAX_BODY:
        raise ValueError("FINAL_MARKET_RESPONSE_TOO_LARGE")
    payload = json.loads(content)
    if (not isinstance(payload, dict) or payload.get("stat") != "OK"
            or payload.get("date") != day.replace("-", "")
            or payload.get("params", {}).get("date") != day.replace("-", "")
            or payload["params"].get("type") != "ALLBUT0999"):
        raise ValueError("FINAL_MARKET_DATE_SCHEMA_OR_QUERY_UNVERIFIED")
    tables = [t for t in payload.get("tables", []) if isinstance(t, dict) and t.get("fields") == STOCK_FIELDS]
    if len(tables) != 1 or any(not isinstance(row, list) or len(row) != len(STOCK_FIELDS) for row in tables[0].get("data", [])):
        raise ValueError("FINAL_MARKET_PRICE_VOLUME_UNIT_OR_COLUMN_MAPPING_UNVERIFIED")
    # Construction performs no network request or SMTP/configuration lookup.
    crawler = TwStockCrawler(http_client=TwseHttpClient(TwseHttpConfig(
        max_attempts=1, allow_insecure_ssl_fallback=False)))
    rows = crawler._extract_stock_rows(payload, day)
    benchmark = crawler._extract_taiex_close(payload, day)
    return rows, benchmark


def collect_final_market(calendar_path, *, start, end, root, max_requests=128,
                         client=None, now=utc_now, progress=None, previous=None):
    """Retain raw transport bytes; resume completed days without requests.

    Missing/failed responses remain gaps. They never amend the calendar or
    overwrite the project market cache. A consecutive failure limit and hard
    request budget prevent uncontrolled history refresh.
    """
    from infrastructure.les_v7_calendar_final import load_final_calendar, select_final_calendar
    if (type(start) is not date or type(end) is not date or start > end
            or (end - start).days > 3660 or type(max_requests) is not int
            or not 1 <= max_requests <= 256):
        raise ValueError("FINAL_MARKET_BOUNDED_RANGE_AND_REQUEST_BUDGET_REQUIRED")
    calendar, calendar_documents = load_final_calendar(calendar_path)
    days, _ = select_final_calendar(calendar, now(), documents=calendar_documents,
        start=start, end=end, evidence_id=digest(calendar))
    root = Path(root)
    store = EvidenceDocuments(root / "objects")
    client = client or TwseHttpClient(TwseHttpConfig(timeout_seconds=20,
        max_attempts=1, min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
    if client.config.allow_insecure_ssl_fallback or client.config.min_request_interval_seconds < 1:
        raise ValueError("SECURE_RATE_LIMITED_FINAL_MARKET_TRANSPORT_REQUIRED")
    request = dict(schema=SCHEMA, start=start.isoformat(), end=end.isoformat(),
        calendar_sha256=digest(calendar), market="TWSE", endpoint=URL,
        policy="PINNED_FINAL_DAILY_ARCHIVE_NOT_ORIGINAL_VINTAGE")
    task = root / "jobs" / digest(request)
    write_once(task / "request.json", request)
    previous_id = None
    if previous is not None:
        # Range expansion reuses VERIFIED originals, not today's replacement
        # values. A corrupt parent fails before any network request/publication.
        earlier, earlier_store = load_final_market(previous)
        if earlier["policy"] != request["policy"]:
            raise ValueError("FINAL_MARKET_PREVIOUS_POLICY_MISMATCH")
        previous_id = digest(earlier)
        for entry in earlier["documents"]:
            if days["days"].get(entry["date"]) != "OPEN":
                continue
            checkpoint = task / "days" / (entry["date"] + ".json")
            if checkpoint.exists():
                if read_json_bounded(checkpoint, MAX_BODY) != entry:
                    raise ValueError("FINAL_MARKET_PREVIOUS_CHECKPOINT_CONFLICT")
            else:
                preserved = store.externalize(entry, document_reader=earlier_store.read)
                write_once(checkpoint, preserved)
    documents, gaps, calls, failures = [], [], 0, 0
    def reserve():
        nonlocal calls
        if calls >= max_requests:
            raise ValueError("FINAL_MARKET_REQUEST_BUDGET_EXHAUSTED")
        calls += 1
    for day, status in sorted(days["days"].items()):
        if status != "OPEN":
            if status == "UNKNOWN":
                gaps.append(dict(date=day, reason="CALENDAR_UNKNOWN"))
            continue
        checkpoint = task / "days" / (day + ".json")
        if checkpoint.exists():
            entry = read_json_bounded(checkpoint, MAX_BODY)
            parse_final_market(verified_document(entry["document"], document_reader=store.read), day)
            documents.append(entry)
            continue
        if calls >= max_requests or failures >= 3:
            gaps.append(dict(date=day, reason="REQUEST_BUDGET_OR_CONSECUTIVE_FAILURE_LIMIT"))
            continue
        try:
            params = dict(date=day.replace("-", ""), type="ALLBUT0999", response="json")
            response = client.request_bytes("GET", URL, params=params,
                context="Manual v7 retrospective warmup " + day, before_attempt=reserve)
            rows, benchmark = parse_final_market(response.content, day)
            fetched = now()
            reference = store.store(response.content)
            entry = dict(date=day, market="TWSE", row_count=len(rows), taiex_close=benchmark,
                query=params, historical_available_at=None,
                document=dict(url=URL + "?date=" + params["date"] + "&type=ALLBUT0999&response=json",
                    sha256=reference["sha256"], body_ref=reference, retrieved_at=fetched))
            write_once(checkpoint, entry)
            documents.append(entry)
            failures = 0
            if progress:
                progress(day, calls, len(documents))
        except (ValueError, OSError, RuntimeError) as error:
            failures += 1
            gaps.append(dict(date=day, reason=type(error).__name__, detail=str(error)))
            # No retry or bypass after a security/transport access block.
            if "AccessBlocked" in type(error).__name__ or "ACCESS_BLOCKED" in str(error):
                failures = 3
    body = dict(**request, documents=documents, gaps=gaps,
        status="COMPLETE" if not gaps else "PARTIAL", historical_information_set_certified=False,
        original_observation_receipts_available=False, research_only=True, approved_for_live=False)
    if previous_id is not None:
        # A checked retry is not a new capture merely because its immediate
        # parent would be itself. Compare the complete semantic body: calendar,
        # range, source bytes AND retrieval metadata, gaps and status all remain
        # significant. Only lineage pointers are excluded from this comparison.
        lineage = {"parent_manifest_id", "parent_calendar_sha256"}
        if ({key: value for key, value in body.items() if key not in lineage}
                == {key: value for key, value in earlier.items() if key not in lineage}):
            checked, _ = load_final_market(previous)
            if checked != earlier:
                raise ValueError("FINAL_MARKET_PREVIOUS_CHANGED_DURING_COLLECTION")
            return Path(previous)
        body["parent_manifest_id"] = previous_id
        body["parent_calendar_sha256"] = earlier["calendar_sha256"]
    identifier = digest(body)
    manifest = root / "manifests" / identifier / "manifest.json"
    if not manifest.exists():
        write_once(manifest, dict(manifest_id=identifier, body=body, created_at=now()))
    load_final_market(manifest)
    return manifest


def load_final_market(path):
    path = Path(path)
    record = read_json_bounded(path, MAX_BODY)
    body = record["body"]
    if (record.get("manifest_id") != digest(body) or body.get("schema") != SCHEMA
            or body.get("market") != "TWSE" or body.get("endpoint") != URL
            or body.get("historical_information_set_certified") is not False
            or body.get("approved_for_live") is not False):
        raise ValueError("FINAL_MARKET_MANIFEST_BINDING_INVALID")
    store = EvidenceDocuments(path.parents[2] / "objects")
    seen = set()
    for entry in body["documents"]:
        day = entry["date"]
        if day in seen or not body["start"] <= day <= body["end"]:
            raise ValueError("FINAL_MARKET_DATE_DUPLICATE_OR_OUT_OF_RANGE")
        seen.add(day)
        expected_query = dict(date=day.replace("-", ""), type="ALLBUT0999", response="json")
        if (entry.get("query") != expected_query or entry.get("historical_available_at") is not None
                or entry["document"]["url"] != URL + "?date=" + expected_query["date"] + "&type=ALLBUT0999&response=json"):
            raise ValueError("FINAL_MARKET_DOCUMENT_QUERY_OR_AVAILABILITY_MISMATCH")
        raw = verified_document(entry["document"], document_reader=store.read)
        rows, benchmark = parse_final_market(raw, day)
        if entry["row_count"] != len(rows) or entry["taiex_close"] != benchmark:
            raise ValueError("FINAL_MARKET_DERIVED_SUMMARY_MISMATCH")
    return body, store


def final_market_records(body, documents, symbols):
    """Bounded selected rows only; caller binds the entire source manifest."""
    symbols = set(symbols)
    if not 1 <= len(symbols) <= 32:
        raise ValueError("BOUNDED_FINAL_MARKET_SYMBOLS_REQUIRED")
    records = []
    for entry in body["documents"]:
        rows, benchmark = parse_final_market(
            verified_document(entry["document"], document_reader=documents.read), entry["date"])
        day = entry["date"].replace("-", "")
        records.extend([dict(date=day, dataset="STOCK_TABLE", market_type="ALLBUT0999",
            source="TWSE", data=[row for row in rows if str(row[0]) in symbols]),
            dict(date=day, dataset="TAIEX", market_type="IND", source="TWSE_TAIEX", data=str(benchmark))])
    return records
