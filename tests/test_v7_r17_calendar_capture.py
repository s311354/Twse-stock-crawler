"""SYNTHETIC ONLY: cutoff-bound current observations, no live official claim."""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from application.evidence_context import ReadOnlyEvidence
from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.session_coverage_r16 import verify_window as strict_verify
from domain.les_v7.session_coverage_r17 import verify_window
from domain.trading_calendar import OfficialAnnualCalendar
from infrastructure.calendar.current_observation import capture_current_calendar, prepare_current_calendar
from tests.test_v7_calendar_final import packet, Client, STAMP, PDF_TEXT

CUTOFF = "2026-10-16T23:30:00+08:00"


def current(tmp_path):
    raw, documents = packet(tmp_path)
    return prepare_current_calendar(raw, documents, observed_at=STAMP, extract_text=lambda _: PDF_TEXT)


def sessions(calendar, *, count=121, end="2026-10-12"):
    calendar = json.loads(canonical(calendar))
    adapter = OfficialAnnualCalendar(calendar["documents"]["2026"]["payload"])
    closed = {event["date"] for event in calendar["announcements"]}
    found, cursor = [], date.fromisoformat(end)
    while len(found) < count:
        if adapter.trading_day_status(cursor).value == "open" and cursor.isoformat() not in closed:
            found.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    return list(reversed(found))


def rehash(value):
    value["evidence_id"] = identity({key: item for key, item in value.items() if key != "evidence_id"})
    return value


def run(calendar, *, observed=None, cutoff=CUTOFF, prior=120, market="TWSE"):
    return verify_window(calendar, market=market, cutoff=cutoff,
        observed_sessions=sessions(calendar) if observed is None else observed, prior_sessions=prior)


def test_current_originals_closure_and_full_official_query_define_exact_window_without_backdating(tmp_path):
    calendar = current(tmp_path)
    original = deepcopy(calendar)
    result = run(ReadOnlyEvidence(calendar))
    assert result["state"] == "PASS" and result["expected_open_count"] == 121
    assert result["expected_sessions"] == result["observed_sessions"]
    assert result["calendar_days"]["2026-07-10"] == "CLOSED"
    assert result["source_proof_ids"] and result["selected_publications"][0]["available_at"] == timestamp(STAMP).isoformat()
    assert result["historical_information_set_certified"] is False
    assert result["original_historical_available_at"] is None
    assert calendar == original
    assert run(calendar, cutoff="2026-10-12T23:30:00+08:00")["state"] == "UNKNOWN"
    # The current schema never becomes a strict historical annual-publication proof.
    assert strict_verify(calendar, market="TWSE", cutoff=CUTOFF, observed_sessions=sessions(calendar))["state"] == "UNKNOWN"


def test_r17_wrapper_preserves_existing_strict_publication_packet_behavior():
    from tests.test_v7_r16_session_coverage import calendar_fixture, observed
    calendar = calendar_fixture()
    values = observed(calendar)
    assert verify_window(calendar, market="TWSE", cutoff=CUTOFF, observed_sessions=values) == strict_verify(
        calendar, market="TWSE", cutoff=CUTOFF, observed_sessions=values)


@pytest.mark.parametrize("damage", ["missing_query", "truncated_query", "wrong_market", "future_original",
    "annual_payload_conflict", "bad_document_hash", "missing_closure", "unbound_closure"])
def test_missing_future_conflicting_or_unbound_sources_never_certify_open_days(tmp_path, damage):
    calendar = current(tmp_path)
    observed = sessions(calendar)
    if damage == "missing_query": calendar["closure_queries"] = {}
    elif damage == "truncated_query": calendar["closure_queries"]["2026"]["payload"]["totalCount"] += 1
    elif damage == "wrong_market": calendar["market"] = "TPEX"
    elif damage == "future_original": calendar["documents"]["2026"]["document"]["retrieved_at"] = "2030-01-01T00:00:00Z"
    elif damage == "annual_payload_conflict": calendar["documents"]["2026"]["payload"]["data"] = []
    elif damage == "bad_document_hash": calendar["documents"]["2026"]["document"]["sha256"] = "0" * 64
    elif damage == "missing_closure": calendar["announcements"] = []
    else: calendar["announcements"][0]["news_id"] = "0" * 32
    result = run(rehash(calendar), observed=observed)
    assert result["state"] == "UNKNOWN"
    assert result["unknown_dates"] or result["reasons"]


def test_empty_complete_query_differs_from_missing_query_and_missing_bars_do_not_create_holidays(tmp_path):
    calendar = current(tmp_path)
    calendar["announcements"] = []
    row = calendar["closure_queries"]["2026"]
    row["payload"].update(totalCount=0, data=[])
    body = canonical(row["payload"]).encode()
    row["payload_sha256"] = identity(row["payload"])
    row["document"].pop("body_base64")
    row["document"].update(body_utf8=body.decode(), sha256=hashlib.sha256(body).hexdigest())
    calendar = rehash(calendar)
    assert run(calendar)["state"] == "PASS"
    extended = sessions(calendar, count=122)
    omitted = extended[-40]
    result = run(calendar, observed=[day for day in extended if day != omitted])
    assert result["state"] == "FAIL" and result["missing"] == [omitted]
    calendar["closure_queries"] = {}
    assert run(rehash(calendar))["state"] == "UNKNOWN"


def test_unrecognized_market_wide_closure_headline_cannot_be_silently_skipped(tmp_path):
    calendar = current(tmp_path)
    observed = sessions(calendar)
    calendar["announcements"] = []
    row = calendar["closure_queries"]["2026"]
    row["payload"]["data"][0][1] = "集中交易市場因特殊情況全日休市"
    body = canonical(row["payload"]).encode()
    row["payload_sha256"] = identity(row["payload"])
    row["document"].pop("body_base64")
    row["document"].update(body_utf8=body.decode(), sha256=hashlib.sha256(body).hexdigest())
    result = run(rehash(calendar), observed=observed)
    assert result["state"] == "UNKNOWN"
    assert "R17_UNRECOGNIZED_MARKET_CLOSURE_NEEDS_REVIEW" in str(result["excluded_evidence"])


@pytest.mark.parametrize("field,value,rehash_cache", [("market", "TPEX", False),
    ("years", [2025], True), ("observed_at", "2026-10-12T00:00:00Z", True),
    ("calendar_id", "bad", True), ("manifest", "../other/calendar.json", True)])
def test_same_day_cache_reuses_exact_originals_but_modified_pointer_fails_closed(tmp_path, monkeypatch,
        field, value, rehash_cache):
    import infrastructure.les_v7_calendar_final as final
    import infrastructure.calendar.current_observation as observed
    monkeypatch.setattr(final, "_pdf_text", lambda _: PDF_TEXT)
    monkeypatch.setattr(observed, "_pdf_text", lambda _: PDF_TEXT)
    client = Client()
    clock = lambda: timestamp(STAMP)
    first = capture_current_calendar(tmp_path, market="TWSE", start="2026-04-01", end="2026-10-12", client=client, clock=clock)
    assert len(client.calls) == 4 and run(first)["state"] == "PASS"
    second = capture_current_calendar(tmp_path, market="TWSE", start="2026-04-15", end="2026-10-12", client=client, clock=clock)
    assert first == second and len(client.calls) == 4
    pointer = next((tmp_path / "TWSE" / "daily").glob("*.json"))
    record = json.loads(pointer.read_text())
    record[field] = value
    if rehash_cache:
        record["cache_id"] = identity({key: item for key, item in record.items() if key != "cache_id"})
    pointer.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="POINTER_"):
        capture_current_calendar(tmp_path, market="TWSE", start="2026-04-01", end="2026-10-12", client=client, clock=clock)
    assert len(client.calls) == 4


def test_tpex_never_borrows_twse_or_opens_a_network_connection(tmp_path):
    client = Mock()
    result = capture_current_calendar(tmp_path, market="TPEX", start="2026-04-01", end="2026-10-12",
        client=client, clock=lambda: timestamp(STAMP))
    assert run(result, market="TPEX", observed=["2026-10-09", "2026-10-12"], prior=1)["state"] == "UNKNOWN"
    client.request_bytes.assert_not_called()


@pytest.mark.parametrize("version", ["V7_RULE_BASED_BASELINE_R17", "V7_RULE_BASED_BASELINE_R18"])
def test_native_capture_activates_for_calendar_consumers_and_shares_immutable_calendar(tmp_path, monkeypatch, version):
    import domain.les_v7.rule_baseline as rules
    from tests.test_native_source_runtime import capture_fixture, crawler_stub
    capture, *_ = capture_fixture(tmp_path)
    collector = Mock(return_value=current(tmp_path / "calendar"))
    capture.calendar_collector = collector
    values = sessions(collector.return_value)
    crawler = crawler_stub(dates=values)
    monkeypatch.setattr(rules, "current_source_baseline_version", lambda _: "V7_RULE_BASED_BASELINE_R16")
    capture.collect(crawler, market="TWSE")
    collector.assert_not_called()
    monkeypatch.setattr(rules, "current_source_baseline_version", lambda _: version)
    capture.collect(crawler, market="TWSE")
    collector.assert_called_once()
    assert collector.call_args.kwargs["start"] == values[0]
    selected = capture.selections(crawler, cutoff=CUTOFF)["2330"]["calendar_evidence"]
    assert run(selected)["state"] == "PASS"
    with pytest.raises(TypeError, match="READ_ONLY"):
        selected["documents"]["2026"]["payload"]["stat"] = "bad"
