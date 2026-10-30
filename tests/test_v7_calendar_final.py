"""Synthetic contract regressions, not official historical market evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from application.replay_contracts import canonical_json, digest
from application.research_evidence import calendar_at
from domain.errors import AccessBlockedError, NetworkError
from infrastructure.les_v7_calendar_final import (
    HOST, NEWS_LIST, NEWS_DETAIL, POLICY, SCHEMA, collect_final_calendars,
    load_final_calendar, select_final_calendar,
)
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.crawler.twse_http_client import TwseBinaryResponse

STAMP = "2026-10-13T00:00:00+00:00"
NEWS_ID = "a" * 32
PDF_ID = "b" * 32
TITLE = "臺灣證券交易所集中交易市場115年7月10日休市一天"
PDF_TEXT = "中華民國115年7月9日\n" + TITLE + "\n115年7月10日集中交易市場全日休市"


def annual():
    return dict(stat="ok", queryYear=2026, date="20260101", title="115 年市場開休市日期",
                fields=["日期", "名稱", "說明"], data=[["2026-01-01", "元旦", "依規定放假1日。"]], total=1)


def query():
    return dict(stat="ok", fields=["項次", "標題", "日期", "zhId", "enId"], totalCount=1,
                data=[[1, TITLE, "115年07月09日", NEWS_ID, "c"*32]])


def detail():
    return dict(stat="ok", date="202607092011", tables=[dict(title=TITLE,
                fields=["text", "html", "pdf", "lang"],
                data=["", "", f"/news/news/tsecnews/{PDF_ID}.pdf", "c"*32])])


class Client:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def request_bytes(self, method, url, *, params=None, **kwargs):
        self.calls.append((method, url, params))
        if self.error:
            raise self.error
        body = (b"%PDF-SYNTHETIC" if url.endswith(".pdf") else
                canonical_json(annual() if "holidaySchedule" in url else
                               query() if url == NEWS_LIST else detail()).encode())
        return TwseBinaryResponse(body, 200, 1)


def packet(tmp_path):
    from urllib.parse import urlencode
    store = EvidenceDocuments(tmp_path / "objects")

    def doc(url, value, params=None):
        body = value if isinstance(value, bytes) else canonical_json(value).encode()
        return dict(url=url+("?"+urlencode(params) if params else ""),
                    sha256=hashlib.sha256(body).hexdigest(), retrieved_at=STAMP, body_ref=store.store(body))

    a, q, d = annual(), query(), detail()
    calendar = dict(schema=SCHEMA, source="TWSE_OFFICIAL", policy=POLICY,
        historical_information_set_certified=False, original_historical_available_at=None,
        documents={"2026": dict(payload=a, payload_sha256=digest(a),
            document=doc(HOST+"/holidaySchedule/holidaySchedule", a, dict(response="json", date="20260101")))},
        closure_queries={"2026": dict(payload=q, payload_sha256=digest(q),
            document=doc(NEWS_LIST, q, dict(startDate=2026,endDate=2026,tag="休市",response="json")))},
        announcements=[dict(news_id=NEWS_ID, date="2026-07-10", status="CLOSED", market="TWSE",
            declared_published_at="2026-07-09T20:11:00+08:00", original_historical_available_at=None,
            document=doc(HOST+f"/staticFiles/news/news/tsecnews/{PDF_ID}.pdf", b"%PDF-SYNTHETIC"),
            detail_document=doc(NEWS_DETAIL,d,dict(id=NEWS_ID,response="json")))])
    return calendar, store


def select(calendar, store, **kwargs):
    return select_final_calendar(calendar, "2026-08-07T13:30:00+08:00", documents=store,
        start="2026-07-09", end="2026-07-13", evidence_id="d"*64,
        extract_text=lambda _: PDF_TEXT, **kwargs)


def test_explicit_retrospective_final_calendar_with_real_capture_times(tmp_path):
    calendar, store = packet(tmp_path)
    before = deepcopy(calendar)
    out, audit = select(calendar, store)
    assert out["sessions"] == ["2026-07-09", "2026-07-13"]
    assert out["days"]["2026-07-10"] == "CLOSED"
    assert out["days"]["2026-07-11"] == "CLOSED"
    assert not audit["unknown_dates"]
    assert out["availability_by_date"]["2026-07-10"] == "2026-07-10T13:30:00+08:00"
    assert out["original_historical_available_at"] is None
    assert out["historical_information_set_certified"] is False
    assert calendar == before
    assert calendar["documents"]["2026"]["document"]["retrieved_at"] == STAMP


def test_explicit_refresh_recollects_current_year_into_new_immutable_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr('infrastructure.les_v7_calendar_final._pdf_text', lambda _: PDF_TEXT)
    first_client = Client()
    first = collect_final_calendars(tmp_path, years=(2026,), client=first_client, now=lambda: STAMP)
    original = first.read_bytes()
    client = Client()
    later = '2026-10-14T00:00:00+00:00'
    second = collect_final_calendars(tmp_path, years=(2026,), client=client,
        now=lambda: later, previous=first, refresh_years=(2026,))
    assert len(client.calls) == 2  # Unchanged verified detail/PDF bytes reused.
    assert second != first and first.read_bytes() == original
    refreshed, _ = load_final_calendar(second)
    assert refreshed['parent_calendar_id']
    assert refreshed['closure_queries']['2026']['document']['retrieved_at'] == later


def test_no_policy_opt_in_or_false_certification_is_rejected(tmp_path):
    calendar, store = packet(tmp_path)
    calendar["historical_information_set_certified"] = True
    with pytest.raises(ValueError, match="EXPLICIT_RETROSPECTIVE"):
        select(calendar, store)
    calendar["historical_information_set_certified"] = False
    calendar["policy"] = "STRICT"
    with pytest.raises(ValueError, match="EXPLICIT_RETROSPECTIVE"):
        select(calendar, store)


@pytest.mark.parametrize("mutation", ["hash", "payload", "year", "url", "query_missing", "query_truncated"])
def test_damaged_annual_or_query_keeps_unknown_not_weekday_inference(tmp_path, mutation):
    calendar, store = packet(tmp_path)
    if mutation == "hash": calendar["documents"]["2026"]["document"]["sha256"] = "0"*64
    elif mutation == "payload": calendar["documents"]["2026"]["payload"]["data"].clear()
    elif mutation == "year": calendar["documents"]["2025"] = calendar["documents"].pop("2026")
    elif mutation == "url": calendar["documents"]["2026"]["document"]["url"] = "https://example.test/annual"
    elif mutation == "query_missing": calendar["closure_queries"].clear()
    else: calendar["closure_queries"]["2026"]["payload"]["totalCount"] = 2
    out, audit = select(calendar, store)
    assert out["sessions"] == []
    assert set(out["days"].values()) == {"UNKNOWN"}
    assert audit["excluded"]


@pytest.mark.parametrize("mutation", ["missing", "pdf_hash", "pdf_binding", "detail_binding", "date", "status", "release", "duplicate"])
def test_damaged_or_missing_emergency_proof_blocks_affected_date(tmp_path, mutation):
    calendar, store = packet(tmp_path)
    event = calendar["announcements"][0]
    if mutation == "missing": calendar["announcements"].clear()
    elif mutation == "pdf_hash": event["document"]["sha256"] = "0"*64
    elif mutation == "pdf_binding": event["document"]["url"] = HOST+"/staticFiles/news/news/tsecnews/"+"c"*32+".pdf"
    elif mutation == "detail_binding": event["detail_document"]["url"] += "&fake=1"
    elif mutation == "date": event["date"] = "2026-07-09"
    elif mutation == "status": event["status"] = "OPEN"
    elif mutation == "release": event["declared_published_at"] = "2026-07-09T09:00:00+08:00"
    else: calendar["announcements"].append(deepcopy(event))
    out, audit = select(calendar, store)
    assert out["days"]["2026-07-10"] == "UNKNOWN"
    assert audit["excluded"]


def test_reextracts_raw_pdf_not_mutable_claimed_text(tmp_path):
    calendar, store = packet(tmp_path)
    calendar["announcements"][0]["document"]["extracted_text"] = PDF_TEXT
    out, audit = select_final_calendar(calendar,"2026-08-07T13:30:00+08:00",documents=store,
        start="2026-07-09",end="2026-07-13",evidence_id="d"*64,extract_text=lambda _: "unrelated PDF")
    assert out["days"]["2026-07-10"] == "UNKNOWN"
    assert "SOURCE_TEXT" in audit["excluded"][0]["detail"]


def test_final_packet_does_not_create_strict_publication_proof(tmp_path):
    calendar, _ = packet(tmp_path)
    days, rejected = calendar_at(dict(source="TWSE_OFFICIAL", days={"2026-07-09":"OPEN"},
         documents=calendar["documents"], announcements=[]),
         __import__("datetime").datetime.fromisoformat("2026-08-07T13:30:00+08:00"), strict=True)
    assert days == {"2026-07-09":"UNKNOWN"}
    assert rejected[0]["reason"] == "UNVERIFIED_ANNUAL_CALENDAR"


def test_collector_retains_transport_and_load_verifies_id(tmp_path, monkeypatch):
    monkeypatch.setattr("infrastructure.les_v7_calendar_final._pdf_text", lambda _: PDF_TEXT)
    client = Client()
    path = collect_final_calendars(tmp_path, years=(2026,), client=client, now=lambda:STAMP)
    assert len(client.calls) == 4
    payload, docs = load_final_calendar(path)
    assert len(payload["announcements"]) == 1
    assert payload["announcements"][0]["document"]["retrieved_at"] == STAMP
    assert docs.read(payload["announcements"][0]["document"]["body_ref"]) == b"%PDF-SYNTHETIC"
    completion = json.loads((path.parent/"completion.json").read_text())
    assert completion["status"] == "SOURCE_CAPTURED_RETROSPECTIVE_ONLY"
    assert completion["request_count"] == 4
    record = json.loads(path.read_text()); record["calendar_id"] = "0"*64
    path.write_text(canonical_json(record))
    with pytest.raises(ValueError, match="ID_MISMATCH"):
        load_final_calendar(path)


@pytest.mark.parametrize("error", [AccessBlockedError("blocked"), ValueError("FINAL_CALENDAR_ACCESS_BLOCKED")])
def test_access_blocked_stops_all_later_remote_requests(tmp_path, error):
    client = Client(error=error)
    path = collect_final_calendars(tmp_path, years=(2025,2026), client=client, now=lambda:STAMP)
    assert len(client.calls) == 1
    payload,_ = load_final_calendar(path)
    assert payload["documents"] == {}
    assert payload["closure_queries"] == {}
    assert json.loads((path.parent/"completion.json").read_text())["status"] == "PARTIAL"


def test_request_budget_and_network_failure_do_not_imply_closed(tmp_path):
    client = Client(error=NetworkError("failed"))
    path = collect_final_calendars(tmp_path, years=(2025,2026), client=client,max_requests=2,now=lambda:STAMP)
    assert len(client.calls) == 2
    payload,store = load_final_calendar(path)
    out, audit = select(payload,store)
    assert set(out["days"].values()) == {"UNKNOWN"}
    assert audit["collection_errors"]


def test_partial_repair_reuses_verified_raw_sources_and_preserves_original(tmp_path, monkeypatch):
    monkeypatch.setattr("infrastructure.les_v7_calendar_final._pdf_text", lambda _: PDF_TEXT)
    first_client = Client()
    initial = collect_final_calendars(tmp_path, years=(2026,), client=first_client,
                                      max_requests=2,now=lambda:STAMP)
    initial_bytes = initial.read_bytes()
    second_client = Client()
    repaired = collect_final_calendars(tmp_path, years=(2026,), client=second_client,
        max_requests=2,now=lambda:"2026-10-14T00:00:00+00:00",previous=initial)
    assert len(second_client.calls) == 2
    assert all("holidaySchedule" not in call[1] and call[1] != NEWS_LIST for call in second_client.calls)
    assert repaired != initial
    assert initial.read_bytes() == initial_bytes
    current,store = load_final_calendar(repaired)
    assert current["parent_calendar_id"] == json.loads(initial_bytes)["calendar_id"]
    assert current["documents"]["2026"]["document"]["retrieved_at"] == STAMP
    assert not current["collection_errors"]
    assert select(current,store)[0]["days"]["2026-07-10"] == "CLOSED"
    retry = Client(error=AssertionError("completed collector repair must not fetch"))
    completed = collect_final_calendars(tmp_path, years=(2026,),client=retry,now=lambda:STAMP,previous=repaired)
    assert not retry.calls
    assert completed == repaired
    assert load_final_calendar(completed)[0]["announcements"] == current["announcements"]


def test_range_after_cutoff_rejected(tmp_path):
    calendar,store = packet(tmp_path)
    with pytest.raises(ValueError,match="RANGE_AFTER_CUTOFF"):
        select_final_calendar(calendar,"2026-07-09T13:30:00+08:00",documents=store,
            start="2026-07-09",end="2026-07-10",evidence_id="d"*64)
