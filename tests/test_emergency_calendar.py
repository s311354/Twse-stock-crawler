"""Original reviewed bytes and controlled cutoffs; no live HTTP in these checks."""
import datetime as dt
import hashlib
import json
import shutil

import pytest

from infrastructure.calendar.emergency_announcements import (
    EVIDENCE_ROOT, MANIFEST_SHA256, ReviewedEmergencyAnnouncements,
)
from infrastructure.calendar.twse_calendar import TradingDayStatus, TwseCalendar
from infrastructure.crawler.twse_http_client import TwseHttpResponse


ANNUAL = dict(stat="ok", date="20240101", fields=["日期", "名稱", "說明"],
              data=[["2024-01-01", "國曆新年", "依規定放假"]])
AFTER = dt.datetime.fromisoformat("2024-10-03T08:30:00+08:00")


def store(**kwargs):
    return ReviewedEmergencyAnnouncements(cutoff=AFTER, **kwargs)


def test_reviewed_originals_close_both_typhoon_sessions_and_preserve_adjacent_day():
    supplement = store()
    calendar = TwseCalendar(fetcher=lambda: ANNUAL, emergency_announcements=supplement)
    for day in (2, 3):
        assert calendar.trading_day_status(dt.date(2024, 10, day)) is TradingDayStatus.CLOSED
    assert calendar.trading_day_status(dt.date(2024, 10, 4)) is TradingDayStatus.OPEN
    evidence = calendar.emergency_evidence()
    assert evidence["manifestSha256"] == MANIFEST_SHA256
    for event in evidence["announcements"]:
        proof = event["proof"]
        assert proof["publication_precision"] == "DATE"
        assert proof["document"]["retrieved_at"] != proof["published_at"]
        assert "body_base64" not in proof["document"]
        assert event["availableAt"][:10] == event["date"]


def test_default_remote_calendar_applies_supplement_without_any_market_http():
    class AnnualOnly:
        calls = 0
        def get_json(self, url, **kwargs):
            self.calls += 1
            assert "holidaySchedule" in url
            return TwseHttpResponse(ANNUAL, 200, 1)
    client = AnnualOnly()
    calendar = TwseCalendar(http_client=client)
    assert not calendar.is_trading_day(dt.date(2024, 10, 2))
    assert not calendar.is_trading_day(dt.date(2024, 10, 3))
    assert client.calls == 0
    assert calendar.is_trading_day(dt.date(2024, 10, 4))
    assert client.calls == 1


def test_verified_closure_survives_annual_api_unavailability():
    from infrastructure.crawler.twse_http_client import TwseNetworkError
    class UnavailableAnnual:
        def get_json(self, *args, **kwargs): raise TwseNetworkError("offline")
    calendar = TwseCalendar(http_client=UnavailableAnnual())
    assert calendar.trading_day_status(dt.date(2024, 10, 3)) is TradingDayStatus.CLOSED
    assert calendar.trading_day_status(dt.date(2024, 10, 4)) is TradingDayStatus.UNKNOWN


def test_failed_annual_request_is_not_repeated_for_each_day_and_can_be_retried():
    from infrastructure.crawler.twse_http_client import TwseNetworkError
    class RecoverableAnnual:
        calls = 0
        offline = True
        def get_json(self, *args, **kwargs):
            self.calls += 1
            if self.offline: raise TwseNetworkError("offline")
            return TwseHttpResponse(ANNUAL, 200, 1)
    client = RecoverableAnnual()
    calendar = TwseCalendar(http_client=client)
    for day in (4, 7, 8):
        assert calendar.trading_day_status(dt.date(2024, 10, day)) is TradingDayStatus.UNKNOWN
    assert client.calls == 1
    assert calendar.trading_day_status(dt.date(2024, 10, 3)) is TradingDayStatus.CLOSED
    client.offline = False
    calendar.retry_year(2024)
    assert client.calls == 2
    assert calendar.trading_day_status(dt.date(2024, 10, 4)) is TradingDayStatus.OPEN


def test_date_only_publication_not_available_before_next_taipei_midnight():
    before = ReviewedEmergencyAnnouncements(cutoff=dt.datetime.fromisoformat("2024-10-02T23:59:59+08:00"))
    assert before.status(dt.date(2024, 10, 3)) is None
    assert store().status(dt.date(2024, 10, 3)) == "CLOSED"


def test_raw_annual_adapter_does_not_import_current_announcements_into_strict_replay():
    calendar = TwseCalendar(fetcher=lambda: ANNUAL)
    assert calendar.is_trading_day(dt.date(2024, 10, 3))
    assert calendar.emergency_evidence() is None


def test_native_date_schedule_never_requests_verified_closed_market_days():
    from stockanalysis import TwseCrawker
    crawler = TwseCrawker(0)
    calendar = TwseCalendar(fetcher=lambda: ANNUAL, emergency_announcements=store())
    crawler.get_date_times(0, 700, calendar=calendar,
                          reference_date=dt.date(2024, 10, 5),
                          start_date=dt.date(2024, 10, 1), end_date=dt.date(2024, 10, 4))
    assert crawler.iso_scheduled_times == ["2024-10-01", "2024-10-04"]


@pytest.mark.parametrize("damage", ["missing", "hash", "quote", "binding", "host", "manifest"])
def test_invalid_announcement_is_unknown_never_a_silent_open_day(tmp_path, damage):
    shutil.copytree(EVIDENCE_ROOT, tmp_path / "evidence")
    root = tmp_path / "evidence"
    path = root / "announcements.json"
    data = json.loads(path.read_text())
    proof = data["announcements"][1]["proof"]
    original = root / "objects" / proof["document"]["sha256"]
    if damage == "missing": original.unlink()
    elif damage == "hash": original.write_bytes(b"<!doctype html>blocked")
    elif damage == "quote": proof["quotes"] = ["not present in official document"]
    elif damage == "binding": proof["binding"]["date"] = "2024-10-04"
    elif damage == "host": proof["document"]["url"] = "https://example.test/fake.pdf"
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    expected = MANIFEST_SHA256 if damage == "manifest" else hashlib.sha256(raw).hexdigest()
    calendar = TwseCalendar(fetcher=lambda: ANNUAL,
                            emergency_announcements=store(root=root, expected_sha256=expected))
    assert calendar.trading_day_status(dt.date(2024, 10, 3)) is TradingDayStatus.UNKNOWN
    with pytest.raises((ValueError, OSError)):
        calendar.emergency_evidence()


def test_conflicting_same_release_remains_unknown(tmp_path):
    shutil.copytree(EVIDENCE_ROOT, tmp_path / "evidence")
    root = tmp_path / "evidence"
    path = root / "announcements.json"
    data = json.loads(path.read_text())
    duplicate = json.loads(json.dumps(data["announcements"][1]))
    duplicate["status"] = duplicate["proof"]["binding"]["status"] = "OPEN"
    data["announcements"].append(duplicate)
    raw = json.dumps(data).encode(); path.write_bytes(raw)
    calendar = TwseCalendar(fetcher=lambda: ANNUAL,
        emergency_announcements=store(root=root, expected_sha256=hashlib.sha256(raw).hexdigest()))
    assert calendar.trading_day_status(dt.date(2024, 10, 3)) is TradingDayStatus.UNKNOWN
