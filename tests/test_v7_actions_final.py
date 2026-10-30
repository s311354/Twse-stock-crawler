"""SYNTHETIC transport/contract tests; these do not certify official evidence."""
from copy import deepcopy
import hashlib
import json
from urllib.parse import urlencode

import pytest
from requests.exceptions import ReadTimeout

from domain.errors import AccessBlockedError, NetworkError
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.les_v7_actions_final import (
    CATEGORY_SCOPE, FEEDS, LIMITATIONS, MAX_BYTES, POLICY, REQUIRED_FIELDS, SCHEMA, _packet_id,
    build_final_coordinate_coverages, collect_actions_final, load_actions_final,
    parse_actions_response, verify_packet,
)

START, END = "2026-04-01", "2026-08-07"


def content(feed, rows=()):
    fields = REQUIRED_FIELDS[feed]
    expanded = [list(row) + ["SYNTHETIC_TEST_ONLY"] * (len(fields) - len(row)) if len(row) == 3 else list(row)
                for row in rows]
    payload = dict(stat="OK", fields=fields, data=expanded)
    if feed == "TWTB8U":
        payload["params"] = dict(startDate="20260401", endDate="20260807", controller="change", action=feed)
    else:
        payload.update(strDate="20260401", endDate="20260807")
    return json.dumps(payload, ensure_ascii=False).encode()


def packet(rows=None):
    feeds = []
    for feed, (route, *_) in FEEDS.items():
        body = content(feed, (rows or {}).get(feed, []))
        query = dict(startDate="20260401", endDate="20260807", response="json")
        fetched = "2026-10-13T10:00:00+08:00"
        feeds.append(dict(feed=feed, query=query, source_scope="ALL_TWSE_INSTRUMENTS",
            parsed=parse_actions_response(feed, body, start=START, end=END), fetched_at=fetched,
            historical_available_at=None, document=dict(url="https://www.twse.com.tw/rwd/zh" + route + "?" + urlencode(query),
                retrieved_at=fetched, sha256=hashlib.sha256(body).hexdigest(), body_utf8=body.decode())))
    value = dict(schema=SCHEMA, policy=POLICY, market="TWSE", start=START, end=END, feeds=feeds,
        historical_available_at=None, historical_information_set_certified=False,
        category_scope=CATEGORY_SCOPE, limitations=LIMITATIONS)
    value["evidence_id"] = _packet_id(value)
    return value


@pytest.mark.parametrize("feed", list(FEEDS))
def test_exact_official_empty_table_is_valid_but_not_missing(feed):
    parsed = parse_actions_response(feed, content(feed), start=START, end=END)
    assert parsed["row_count"] == 0 and parsed["actions"] == []
    assert parsed["status"] == "COMPLETE_OFFICIAL_FINAL_TABLE"
    with pytest.raises(ValueError, match="NOT_COMPLETE_OK"):
        parse_actions_response(feed, b'{"stat":"no data"}', start=START, end=END)


@pytest.mark.parametrize("feed", list(FEEDS))
def test_wrong_range_rejected_even_when_empty(feed):
    value = json.loads(content(feed))
    (value["params"] if feed == "TWTB8U" else value)["endDate"] = "20260806"
    with pytest.raises(ValueError, match="RANGE_OR_FEED"):
        parse_actions_response(feed, json.dumps(value).encode(), start=START, end=END)


@pytest.mark.parametrize("body,reason", [(b'{"stat":"OK","stat":"OK"}', "DUPLICATE"),
    (b'{"stat":"OK","data":NaN}', "NON_FINITE"), (b"x" * (MAX_BYTES + 1), "SIZE_LIMIT")])
def test_invalid_or_oversize_originals_rejected(body, reason):
    with pytest.raises(ValueError, match=reason):
        parse_actions_response("TWT49U", body, start=START, end=END)


@pytest.mark.parametrize("row", [["115年03月31日", "3014", "SYNTHETIC_TEST_ONLY"],
                                  ["115年07月01日", "x", "SYNTHETIC_TEST_ONLY"],
                                  ["115年07月01日", "3014"]])
def test_identity_date_and_width_not_silently_filtered(row):
    with pytest.raises(ValueError):
        parse_actions_response("TWT49U", content("TWT49U", [row]), start=START, end=END)


def test_paged_or_partial_tables_rejected():
    value = json.loads(content("TWT49U")); value["total"] = 1
    with pytest.raises(ValueError, match="PARTIAL_OR_PAGED"):
        parse_actions_response("TWT49U", json.dumps(value).encode(), start=START, end=END)


@pytest.mark.parametrize("change", ["bytes", "parsed", "url", "time", "future_claim", "feed", "identity"])
def test_source_binding_and_unknown_historical_time_not_bypassed(change):
    value = packet()
    if change == "bytes": value["feeds"][0]["document"]["body_utf8"] += " "
    if change == "parsed": value["feeds"][0]["parsed"]["actions"] = [dict(symbol="3014", date=START)]
    if change == "url": value["feeds"][0]["document"]["url"] = "https://example.invalid/"
    if change == "time": value["feeds"][0]["fetched_at"] = "2026-10-12T10:00:00+08:00"
    if change == "future_claim": value["historical_available_at"] = "2026-04-01T00:00:00+08:00"
    if change == "feed": value["feeds"] = value["feeds"][:2]
    if change == "identity": value["evidence_id"] = "0" * 64
    with pytest.raises(ValueError): verify_packet(value)


def test_actions_and_unverified_source_markers_split_intervals_without_imputation():
    value = packet({"TWT49U": [["115年07月01日", "3014", "SYNTHETIC_TEST_ONLY"]],
                    "TWTAUU": [["115/07/20", "3014", "SYNTHETIC_TEST_ONLY"]]})
    histories = [dict(symbol="3014", market="TWSE", coordinate_id="raw3014", bars=[
        {"Date": "2026-07-25", "Corporate Action State": "UNKNOWN"}]),
        dict(symbol="6834", market="TWSE", coordinate_id="raw6834", bars=[])]
    before = deepcopy((value, histories))
    result = build_final_coordinate_coverages(histories, value,
        pending_markers=[dict(symbol="3014", date="20260715")])
    segments = [(c["start_session"], c["end_session"]) for c in result if c["symbol"] == "3014"]
    assert segments == [(START, "2026-06-30"), ("2026-07-02", "2026-07-14"),
        ("2026-07-16", "2026-07-19"), ("2026-07-21", "2026-07-24"), ("2026-07-26", END)]
    assert [(c["start_session"], c["end_session"]) for c in result if c["symbol"] == "6834"] == [(START, END)]
    assert (value, histories) == before
    assert all(c["historical_available_at"] is None and c["category_scope"] == CATEGORY_SCOPE for c in result)
    assert all("HALTS_AND_TRADING_RESTRICTIONS_NOT_CERTIFIED" in c["limitations"] for c in result)
    result[0]["evidence_ids"].append("mutated")
    assert "mutated" not in result[1]["evidence_ids"]
    assert "mutated" not in build_final_coordinate_coverages(histories, value)[0]["evidence_ids"]


def test_same_source_with_different_fetched_time_has_stable_business_identity():
    value = packet(); changed = deepcopy(value)
    for row in changed["feeds"]:
        row["fetched_at"] = row["document"]["retrieved_at"] = "2026-10-14T10:00:00+08:00"
    assert _packet_id(changed) == value["evidence_id"]
    verify_packet(changed)


class Response:
    status_code = 200
    headers = {}
    def __init__(self, body): self.content = body


class Session:
    def __init__(self, bodies): self.headers = {}; self.bodies = list(bodies); self.calls = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(self.bodies.pop(0))


def client(bodies):
    session = Session(bodies)
    value = TwseHttpClient(TwseHttpConfig(min_request_interval_seconds=1, max_attempts=1,
        allow_insecure_ssl_fallback=False), session=session, sleeper=lambda _: None)
    return value, session


def test_collector_retains_originals_and_first_receipt_on_identical_retry(tmp_path):
    value, session = client([content(f) for f in FEEDS] * 2)
    path = collect_actions_final(START, END, tmp_path, client=value)
    first = path.read_bytes()
    assert load_actions_final(path)["evidence_id"] == path.parent.name
    assert all(call[1]["verify"] is True and call[1]["allow_redirects"] is False for call in session.calls)
    assert collect_actions_final(START, END, tmp_path, client=value) == path and path.read_bytes() == first
    original = next((tmp_path / "objects").iterdir())
    assert original.is_file()
    original.write_bytes(b"corrupt")
    with pytest.raises(ValueError): load_actions_final(path)


def test_budget_and_blocked_page_stop_before_following_feed_without_manifest(tmp_path):
    value, session = client([content("TWT49U"), b"<html>Access denied</html>", content("TWTB8U")])
    with pytest.raises(AccessBlockedError): collect_actions_final(START, END, tmp_path, client=value)
    assert len(session.calls) == 2 and not list(tmp_path.glob("*/manifest.json"))
    value, session = client([content(f) for f in FEEDS])
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        collect_actions_final(START, END, tmp_path, client=value, max_requests=2)
    assert len(session.calls) == 2


def test_insecure_or_unthrottled_transport_rejected(tmp_path):
    value, _ = client([])
    value.config = TwseHttpConfig(min_request_interval_seconds=0, allow_insecure_ssl_fallback=False)
    with pytest.raises(ValueError, match="SECURE_RATE_LIMITED"):
        collect_actions_final(START, END, tmp_path, client=value)


class RetrySession(Session):
    def close(self):
        self.closed = True

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        value = self.bodies.pop(0)
        if isinstance(value, BaseException):
            raise value
        return Response(value)


def retry_client(bodies):
    session = RetrySession(bodies)
    value = TwseHttpClient(TwseHttpConfig(timeout_seconds=20, min_request_interval_seconds=1,
        max_attempts=1, allow_insecure_ssl_fallback=False), session=session, sleeper=lambda _: None)
    # Sequence fixtures keep their deterministic injected transport. The real
    # session replacement is exercised separately below, without network I/O.
    value.reset_session = lambda: None
    return value, session


def test_preparation_transport_retries_timeout_once_within_actual_request_budget(tmp_path, monkeypatch):
    monkeypatch.setattr('infrastructure.les_v7_actions_final.time.sleep', lambda _: None)
    value, session = retry_client([ReadTimeout('SYNTHETIC timeout'), *[content(feed) for feed in FEEDS]])
    path = collect_actions_final(START, END, tmp_path, client=value, max_requests=4, resume=True)
    assert len(session.calls) == 4
    assert [row['feed'] for row in load_actions_final(path)['feeds']] == list(FEEDS)
    assert len(list(tmp_path.glob('captures/*/*.json'))) == 3
    assert [call[1]['timeout'] for call in session.calls] == [(20, 60), (20, 60), 20, 20]
    assert value.config.timeout_seconds == 20
    assert all(call[1]['verify'] is True for call in session.calls)


def test_failed_later_feed_resumes_only_missing_sources_and_retains_first_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr('infrastructure.les_v7_actions_final.time.sleep', lambda _: None)
    value, session = retry_client([content('TWT49U'), ReadTimeout('SYNTHETIC one'), ReadTimeout('SYNTHETIC two')])
    with pytest.raises(NetworkError):
        collect_actions_final(START, END, tmp_path, client=value, resume=True)
    assert len(session.calls) == 3 and not list(tmp_path.glob('*/manifest.json'))
    receipt = next(tmp_path.glob('captures/*/TWT49U.json'))
    first_bytes = receipt.read_bytes()
    first_record = json.loads(first_bytes)
    value, session = retry_client([content('TWTAUU'), content('TWTB8U')])
    path = collect_actions_final(START, END, tmp_path, client=value, max_requests=2, resume=True)
    assert len(session.calls) == 2 and all('TWT49U' not in call[0] for call in session.calls)
    assert receipt.read_bytes() == first_bytes
    assert load_actions_final(path)['feeds'][0] == first_record


def test_collector_does_not_multiply_transport_retry_attempts(tmp_path):
    session = RetrySession([ReadTimeout('SYNTHETIC one'), ReadTimeout('SYNTHETIC two')])
    value = TwseHttpClient(TwseHttpConfig(min_request_interval_seconds=1, max_attempts=2,
        allow_insecure_ssl_fallback=False), session=session, sleeper=lambda _: None)
    with pytest.raises(NetworkError):
        collect_actions_final(START, END, tmp_path, client=value, resume=True)
    assert len(session.calls) == 2 and not list(tmp_path.glob('*/manifest.json'))


def test_preparation_retry_replaces_stalled_session_without_resetting_global_budget(tmp_path, monkeypatch):
    from infrastructure.les_v7_training_prepare import _Budget
    delays = []
    monkeypatch.setattr('infrastructure.les_v7_actions_final.time.sleep', delays.append)
    first = RetrySession([ReadTimeout('SYNTHETIC stalled pooled session')])
    second = RetrySession([content(feed) for feed in FEEDS])
    monkeypatch.setattr('infrastructure.crawler.twse_http_client.requests.Session', lambda: second)
    transport = TwseHttpClient(TwseHttpConfig(min_request_interval_seconds=3, max_attempts=1,
        allow_insecure_ssl_fallback=False), session=first, sleeper=lambda _: None)
    budget = _Budget(transport, 4)
    path = collect_actions_final(START, END, tmp_path, client=budget, max_requests=4, resume=True)
    assert first.closed and len(first.calls) == 1
    assert transport.session is second and len(second.calls) == 3
    assert second.headers == first.headers
    assert budget.count == 4 and delays == [3]
    assert first.calls[0][1]['timeout'] == second.calls[0][1]['timeout'] == (30, 60)
    assert [call[1]['timeout'] for call in second.calls[1:]] == [30, 30]
    assert transport.config.timeout_seconds == budget.config.timeout_seconds == 30
    assert all(call[1]['verify'] is True and call[1]['allow_redirects'] is False for call in second.calls)
    assert load_actions_final(path)['evidence_id'] == path.parent.name


def test_transient_retries_and_resumed_feeds_share_preparation_global_budget(tmp_path, monkeypatch):
    from infrastructure.les_v7_training_prepare import _Budget
    monkeypatch.setattr('infrastructure.les_v7_actions_final.time.sleep', lambda _: None)
    client, session = retry_client([ReadTimeout('SYNTHETIC timeout'), content('TWT49U'), content('TWTAUU')])
    budget = _Budget(client, 3)
    with pytest.raises(ValueError, match='BUDGET_EXHAUSTED'):
        collect_actions_final(START, END, tmp_path, client=budget, max_requests=3, resume=True)
    assert budget.count == len(session.calls) == 3
    assert len(list(tmp_path.glob('captures/*/*.json'))) == 2
    assert not list(tmp_path.glob('*/manifest.json'))
    client, session = retry_client([content('TWTB8U')])
    budget = _Budget(client, 1)
    path = collect_actions_final(START, END, tmp_path, client=budget, max_requests=1, resume=True)
    assert budget.count == len(session.calls) == 1
    assert load_actions_final(path)['evidence_id'] == path.parent.name


def test_resume_revalidates_original_bytes_before_any_network_request(tmp_path):
    value, _ = client([content(feed) for feed in FEEDS])
    path = collect_actions_final(START, END, tmp_path, client=value, resume=True)
    packet = load_actions_final(path)
    original = packet['feeds'][0]['document']['body_ref']
    # Corruption is never silently replaced by a different network response.
    from infrastructure.research_documents import EvidenceDocuments
    documents = EvidenceDocuments(tmp_path / 'objects')
    original_path = documents.objects.path(original['sha256'])
    original_path.write_bytes(b'SYNTHETIC corrupt')
    value, session = client([])
    with pytest.raises(ValueError):
        collect_actions_final(START, END, tmp_path, client=value, resume=True)
    assert not session.calls


def test_resumable_collector_never_retries_access_blocked_or_invalid_schema(tmp_path):
    value, session = client([b'<html>Access denied</html>'])
    with pytest.raises(AccessBlockedError):
        collect_actions_final(START, END, tmp_path, client=value, resume=True)
    assert len(session.calls) == 1 and not list(tmp_path.glob('captures/*/*.json'))
    value, session = client([b'{"stat":"OK","fields":[],"data":[]}'])
    with pytest.raises(ValueError, match='RANGE_OR_FEED'):
        collect_actions_final(START, END, tmp_path, client=value, resume=True)
    assert len(session.calls) == 1 and not list(tmp_path.glob('captures/*/*.json'))


def test_large_action_query_preserves_explicit_longer_transport_timeout(tmp_path):
    value, session = client([content(feed) for feed in FEEDS])
    value.config = TwseHttpConfig(timeout_seconds=90, min_request_interval_seconds=1,
        max_attempts=1, allow_insecure_ssl_fallback=False)
    collect_actions_final(START, END, tmp_path, client=value)
    assert [call[1]['timeout'] for call in session.calls] == [90, 90, 90]


def test_exhausted_action_timeouts_leave_no_manifest_and_keep_global_budget(tmp_path, monkeypatch):
    from infrastructure.les_v7_training_prepare import _Budget
    monkeypatch.setattr('infrastructure.les_v7_actions_final.time.sleep', lambda _: None)
    value, session = retry_client([ReadTimeout('SYNTHETIC timeout'), ReadTimeout('SYNTHETIC timeout')])
    budget = _Budget(value, 2)
    with pytest.raises(NetworkError, match='network request failed'):
        collect_actions_final(START, END, tmp_path, client=budget, max_requests=2, resume=True)
    assert budget.count == len(session.calls) == 2
    assert all(call[1]['timeout'] == (20, 60) for call in session.calls)
    assert value.config.timeout_seconds == 20
    assert not list(tmp_path.glob('*/manifest.json'))
