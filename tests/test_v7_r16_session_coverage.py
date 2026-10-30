"""SYNTHETIC ONLY: exact official-session coverage; no production evidence."""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json

import pytest

from application.evidence_context import ReadOnlyEvidence
from application.replay_contracts import digest
from domain.les_v7.session_coverage_r16 import verify_window
from infrastructure.calendar.twse_calendar import TwseCalendar

CUTOFF = "2026-10-16T18:30:00+08:00"


def publication(binding, *, market="TWSE", published="2025-12-01", precision="DATE"):
    """Synthetic proof with the real reviewed-publication contract."""
    body = "SYNTHETIC TEST ONLY " + json.dumps(binding, sort_keys=True) + " " + published
    host = "www.twse.com.tw" if market == "TWSE" else "www.tpex.org.tw"
    return dict(status="VERIFIED", verified_by="SYNTHETIC_TEST_PROVIDER",
        verification_method="SYNTHETIC EXACT RAW SCHEDULE BINDING", verified_at="2026-10-20T08:00:00+08:00",
        binding=binding, published_at=published, publication_precision=precision, quotes=[body],
        document=dict(url="https://" + host + "/synthetic-test-only", body_utf8=body,
            sha256=hashlib.sha256(body.encode()).hexdigest(), retrieved_at="2026-10-20T08:00:00+08:00"))


def calendar_fixture(*, market="TWSE", years=(2026,)):
    result = dict(source=market + "_OFFICIAL", days={}, documents={}, announcements=[])
    for year in years:
        payload = dict(stat="ok", fields=["日期", "名稱", "說明"],
            data=[[f"{year}-01-01", "SYNTHETIC 元旦休市", "休市"]])
        binding = dict(kind="ANNUAL_CALENDAR", market=market, year=year, content_sha256=digest(payload))
        result["documents"][str(year)] = dict(payload=payload,
            publication_proof=publication(binding, market=market, published=f"{year-1}-12-01"))
    return result


def observed(calendar, *, end="2026-10-15", count=121):
    """Synthetic expected dates come from the real raw annual adapter."""
    end = date.fromisoformat(end)
    adapters = {int(year): TwseCalendar(fetcher=lambda payload=doc["payload"]: payload, annual_archive=False)
                for year, doc in calendar["documents"].items()}
    found = []
    current = end
    while len(found) < count:
        adapter = adapters.get(current.year)
        if adapter is None: raise AssertionError("SYNTHETIC fixture year not provided")
        if adapter.trading_day_status(current).value == "open": found.append(current.isoformat())
        current -= timedelta(days=1)
    return list(reversed(found))


def run(calendar, sessions, *, market="TWSE", prior_sessions=120, cutoff=CUTOFF):
    return verify_window(calendar, market=market, cutoff=cutoff,
                         observed_sessions=sessions, prior_sessions=prior_sessions)


def exception(calendar, *, day, status, published="2026-08-01", precision="DATE"):
    market = calendar["source"].split("_")[0]
    event = dict(date=day, market=market, status=status)
    event["proof"] = publication(dict(kind="CALENDAR_EXCEPTION", **event), market=market,
                                  published=published, precision=precision)
    calendar["announcements"].append(event)
    return event


def test_full_120_prior_sessions_require_exact_official_open_sequence_and_every_calendar_day():
    calendar = calendar_fixture()
    sessions = observed(calendar)
    original = deepcopy(calendar)
    result = run(calendar, sessions)
    assert result["state"] == "PASS" and result["reasons"] == []
    assert result["expected_sessions"] == result["observed_sessions"] == sessions
    assert result["expected_open_count"] == result["observed_count"] == 121
    assert result["missing"] == result["extra"] == result["unknown_dates"] == []
    assert len(result["calendar_days"]) == (date.fromisoformat(sessions[-1]) - date.fromisoformat(sessions[0])).days + 1
    assert set(result["calendar_days"].values()) == {"OPEN", "CLOSED"}
    assert result["source_id"] == digest(calendar) and result["source_proof_ids"]
    assert calendar == original


def test_missing_open_day_cannot_be_replaced_by_an_older_bar_to_keep_count_121():
    calendar = calendar_fixture()
    complete = observed(calendar, count=122)
    omitted = complete[-40]
    sessions = [day for day in complete if day != omitted]
    result = run(calendar, sessions)
    assert len(sessions) == 121
    assert result["state"] == "FAIL" and result["missing"] == [omitted]
    assert "R16_OFFICIAL_OPEN_SESSION_BAR_MISSING" in result["reasons"]


def test_unbroken_nonpivot_high_day_cannot_disappear_from_the_verified_window():
    calendar = calendar_fixture()
    sessions = observed(calendar, count=122)
    # Even if the missing bar would not have been a 2-left/2-right pivot, its
    # official OPEN date is still compulsory. No OHLC-dependent shortcut here.
    removed = sessions[-2]
    result = run(calendar, [day for day in sessions if day != removed])
    assert result["state"] == "FAIL" and result["missing"] == [removed]


def test_weekend_is_not_guessed_open_and_explicit_makeup_trade_is_respected():
    calendar = calendar_fixture()
    sessions = observed(calendar, count=6)
    saturday = "2026-10-10"
    assert saturday not in sessions
    bad = run(calendar, sorted([*sessions[:-1], saturday]), prior_sessions=5)
    assert bad["state"] == "FAIL" and saturday in bad["extra"]
    annual = calendar["documents"]["2026"]
    annual["payload"]["data"].append([saturday, "SYNTHETIC 補行交易", "補行交易"])
    annual["publication_proof"] = publication(dict(kind="ANNUAL_CALENDAR", market="TWSE", year=2026,
        content_sha256=digest(annual["payload"])))
    sessions = observed(calendar, count=6)
    assert saturday in sessions
    assert run(calendar, sessions, prior_sessions=5)["state"] == "PASS"


def test_reviewed_emergency_closure_excludes_bar_and_missing_bar_is_not_closure():
    calendar = calendar_fixture()
    sessions = observed(calendar, count=7)
    closed = sessions[3]
    exception(calendar, day=closed, status="CLOSED")
    selected = [day for day in sessions if day != closed]
    assert run(calendar, selected, prior_sessions=5)["state"] == "PASS"
    assert run(calendar, sessions[-6:], prior_sessions=5)["state"] == "FAIL"
    calendar["announcements"] = []
    assert run(calendar, selected, prior_sessions=5)["state"] == "FAIL"


@pytest.mark.parametrize("damage", ["missing_calendar", "no_documents", "no_proof", "unverified_proof",
    "bad_hash", "bad_binding", "no_quotes", "wrong_host", "body_ref_without_reader", "bad_payload"])
def test_unknown_sources_never_certify_complete_window(damage):
    calendar = calendar_fixture()
    sessions = observed(calendar)
    annual = calendar["documents"]["2026"]
    if damage == "missing_calendar": calendar = None
    elif damage == "no_documents": calendar["documents"] = {}
    elif damage == "no_proof": annual.pop("publication_proof")
    elif damage == "unverified_proof": annual["publication_proof"]["status"] = "PENDING"
    elif damage == "bad_hash": annual["publication_proof"]["document"]["sha256"] = "0" * 64
    elif damage == "bad_binding": annual["publication_proof"]["binding"]["year"] = 2025
    elif damage == "no_quotes": annual["publication_proof"]["quotes"] = []
    elif damage == "wrong_host": annual["publication_proof"]["document"]["url"] = "https://example.com/calendar"
    elif damage == "body_ref_without_reader":
        document = annual["publication_proof"]["document"]
        size = len(document.pop("body_utf8").encode())
        document["body_ref"] = dict(schema="research-document-ref/v1", sha256=document["sha256"], size_bytes=size)
    else: annual["payload"]["data"][0][1] = "UNKNOWN UNSUPPORTED EVENT"
    result = run(calendar, sessions)
    assert result["state"] == "UNKNOWN"
    assert not result["source_proof_ids"] or damage == "bad_payload"


def test_status_flag_mapping_cannot_override_raw_calendar_or_impute_unknown_year():
    calendar = calendar_fixture()
    sessions = observed(calendar)
    calendar["days"] = {day: "CLOSED" for day in sessions}
    calendar["state"] = "PASS"
    assert run(calendar, sessions)["state"] == "PASS"
    calendar["documents"] = {}
    calendar["days"] = {day: "OPEN" for day in sessions}
    assert run(calendar, sessions)["state"] == "UNKNOWN"


def test_future_publication_is_excluded_but_verifiable_late_collection_keeps_true_clocks():
    calendar = calendar_fixture()
    sessions = observed(calendar)
    # Fixture was downloaded/reviewed after cutoff: existing verified history
    # semantics allow this without changing the original publication version.
    assert run(calendar, sessions)["state"] == "PASS"
    annual = calendar["documents"]["2026"]
    annual["publication_proof"] = publication(annual["publication_proof"]["binding"],
        published="2026-10-18T12:00:00+08:00", precision="TIMESTAMP")
    result = run(calendar, sessions)
    assert result["state"] == "UNKNOWN"
    assert any(row["reason"] == "R16_CALENDAR_PUBLICATION_AFTER_CUTOFF" for row in result["excluded_evidence"])


def test_date_only_announcement_not_used_before_next_taipei_midnight():
    calendar = calendar_fixture()
    sessions = observed(calendar, end="2026-10-15", count=6)
    exception(calendar, day="2026-10-15", status="CLOSED", published="2026-10-15")
    before = run(calendar, sessions, prior_sessions=5, cutoff="2026-10-15T13:30:00+08:00")
    assert before["state"] == "PASS" and before["calendar_days"]["2026-10-15"] == "OPEN"
    after = run(calendar, sessions, prior_sessions=5)
    assert after["state"] == "FAIL" and after["extra"] == ["2026-10-15"]


@pytest.mark.parametrize("market,source_market,host_market", [("TPEX", "TWSE", "TWSE"),
    ("TPEX", "TPEX", "TWSE"), ("TWSE", "TPEX", "TPEX"), ("TWSE", "TWSE", "TPEX")])
def test_market_scope_requires_matching_calendar_binding_and_official_market_host(market, source_market, host_market):
    calendar = calendar_fixture(market=source_market)
    sessions = observed(calendar)
    proof = calendar["documents"]["2026"]["publication_proof"]
    proof["document"]["url"] = "https://" + ("www.twse.com.tw" if host_market == "TWSE" else "www.tpex.org.tw") + "/synthetic-test"
    assert run(calendar, sessions, market=market)["state"] == "UNKNOWN"


def test_tpex_calendar_can_pass_only_with_its_own_market_bound_original():
    calendar = calendar_fixture(market="TPEX")
    sessions = observed(calendar)
    assert run(calendar, sessions, market="TPEX")["state"] == "PASS"


def test_unknown_exception_and_conflicting_equal_time_announcements_remain_unknown():
    calendar = calendar_fixture()
    sessions = observed(calendar, count=6)
    day = sessions[2]
    exception(calendar, day=day, status="UNKNOWN")
    result = run(calendar, sessions, prior_sessions=5)
    assert result["state"] == "UNKNOWN" and day in result["unknown_dates"]
    calendar["announcements"] = []
    exception(calendar, day=day, status="OPEN")
    exception(calendar, day=day, status="CLOSED")
    result = run(calendar, sessions, prior_sessions=5)
    assert result["state"] == "UNKNOWN" and day in result["unknown_dates"]


def test_unverified_emergency_notice_does_not_silently_certify_annual_open():
    calendar = calendar_fixture()
    sessions = observed(calendar, count=6)
    day = sessions[2]
    event = exception(calendar, day=day, status="CLOSED")
    event["proof"]["document"]["sha256"] = "0" * 64
    result = run(calendar, sessions, prior_sessions=5)
    assert result["state"] == "UNKNOWN" and result["calendar_days"][day] == "UNKNOWN"
    assert any(row.get("date") == day for row in result["excluded_evidence"])


def test_multiyear_window_requires_each_year_original_and_does_not_guess_missing_year():
    calendar = calendar_fixture(years=(2025, 2026))
    sessions = observed(calendar, end="2026-02-12")
    assert run(calendar, sessions)["state"] == "PASS"
    calendar["documents"].pop("2025")
    result = run(calendar, sessions)
    assert result["state"] == "UNKNOWN"
    assert any(day.startswith("2025-") for day in result["unknown_dates"])


@pytest.mark.parametrize("damage", ["too_short", "duplicate", "reverse", "bad_day", "timestamp", "future", "boolean_window"])
def test_bad_or_incomplete_window_does_not_pass(damage):
    calendar = calendar_fixture()
    sessions = observed(calendar)
    kwargs = {}
    if damage == "too_short": sessions.pop()
    elif damage == "duplicate": sessions[-2] = sessions[-1]
    elif damage == "reverse": sessions.reverse()
    elif damage == "bad_day": sessions[-1] = "not-a-date"
    elif damage == "timestamp": sessions[-1] += "T00:00:00"
    elif damage == "future": kwargs["cutoff"] = sessions[-1] + "T09:00:00+08:00"
    else: kwargs["prior_sessions"] = True
    assert run(calendar, sessions, **kwargs)["state"] == "UNKNOWN"


def test_readonly_evidence_normalizes_only_the_calendar_and_cannot_be_mutated():
    calendar = calendar_fixture()
    sessions = observed(calendar)
    frozen = ReadOnlyEvidence(calendar)
    expected = digest(calendar)
    assert run(frozen, sessions)["state"] == "PASS"
    assert digest(frozen) == expected


def test_verification_never_fetches_network(monkeypatch):
    from infrastructure.crawler.twse_http_client import TwseHttpClient
    def denied(*args, **kwargs): raise AssertionError("NETWORK_NOT_ALLOWED_IN_R16_VERIFICATION")
    monkeypatch.setattr(TwseHttpClient, "get_json", denied)
    calendar = calendar_fixture()
    assert run(calendar, observed(calendar))["state"] == "PASS"
    assert run(None, observed(calendar))["state"] == "UNKNOWN"
