"""Observed annual fallback is separate from strict publication evidence."""
import datetime as dt
import hashlib
import json
import shutil

import pytest

from infrastructure.calendar.annual_observations import ObservedAnnualSchedules, ROOT
from infrastructure.calendar.twse_calendar import (
    TradingDayStatus, TwseCalendar, TwseCalendarUnavailableError,
)
from infrastructure.crawler.twse_http_client import TwseHttpResponse, TwseNetworkError


class OfflineClient:
    def __init__(self):
        self.calls = []

    def get_json(self, url, **kwargs):
        self.calls.append(kwargs["params"]["date"])
        raise TwseNetworkError("UPSTREAM_MAINTENANCE: synthetic source unavailable")


def archive_calendar(archive=None):
    client = OfflineClient()
    return TwseCalendar(http_client=client, annual_archive=archive or ObservedAnnualSchedules()), client


def test_observed_official_years_are_verified_once_and_retain_original_provenance():
    calendar, client = archive_calendar()
    for value, expected in [
        ("2024-10-01", True), ("2024-10-02", False), ("2024-10-03", False),
        ("2024-10-04", True), ("2025-09-29", False), ("2025-10-06", False),
        ("2024-10-31", False),
        ("2026-09-24", True), ("2026-09-25", False), ("2026-09-28", False),
        ("2026-09-29", True), ("2026-10-09", False),
    ]:
        assert calendar.is_trading_day(dt.date.fromisoformat(value)) is expected
    assert client.calls == ["20240101", "20250101", "20260101"]
    evidence = calendar.annual_evidence()
    assert calendar.annual_failures() == {}
    for year, document in evidence.items():
        assert document["source"] == "VERIFIED_OBSERVED_ARCHIVE"
        assert document["availability_policy"] == "ORIGINAL_PUBLICATION_UNVERIFIED"
        assert document["retrieved_at"].startswith("2026-09-28T05:07:03")
        assert document["use_policy"] == "OBSERVED_CURRENT_CALENDAR_ONLY_NOT_PIT"
        assert "UPSTREAM_MAINTENANCE" in document["live_failure"]
        raw = (ROOT / "objects" / document["sha256"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == document["sha256"]
        assert json.loads(raw)["queryYear"] == int(year)
    evidence["2024"]["retrieved_at"] = "modified caller copy"
    assert calendar.annual_evidence()["2024"]["retrieved_at"] != "modified caller copy"


def test_current_live_source_is_preferred_and_wrong_year_never_falls_back():
    payload, _ = ObservedAnnualSchedules().get(2026)
    class LiveClient:
        def get_json(self, url, **kwargs):
            return TwseHttpResponse(payload, 200, 1)
    class UnusedArchive:
        def get(self, year):
            pytest.fail("Archive must not mask live schema/year failures")
    calendar = TwseCalendar(http_client=LiveClient(), annual_archive=UnusedArchive())
    assert calendar.is_trading_day(dt.date(2026, 9, 29))
    assert calendar.annual_evidence()["2026"]["source"] == "LIVE_OFFICIAL_JSON"
    assert calendar.trading_day_status(dt.date(2025, 9, 29)) is TradingDayStatus.UNKNOWN
    assert "2025" not in calendar.annual_evidence()


def test_unobserved_year_and_earlier_cutoff_remain_unknown():
    archive = ObservedAnnualSchedules(cutoff=dt.datetime.fromisoformat("2026-09-27T00:00:00+08:00"))
    calendar, client = archive_calendar(archive)
    assert calendar.trading_day_status(dt.date(2026, 9, 24)) is TradingDayStatus.UNKNOWN
    assert calendar.trading_day_status(dt.date(2026, 9, 29)) is TradingDayStatus.UNKNOWN
    assert client.calls == ["20260101"]
    current, _ = archive_calendar()
    assert current.trading_day_status(dt.date(2027, 1, 4)) is TradingDayStatus.UNKNOWN


def test_raw_replay_fetcher_and_explicit_disable_do_not_import_observed_archive():
    payload, _ = ObservedAnnualSchedules().get(2026)
    raw = TwseCalendar(fetcher=lambda: payload)
    assert raw.trading_day_status(dt.date(2025, 9, 29)) is TradingDayStatus.UNKNOWN
    assert raw.annual_evidence() == {}
    disabled = TwseCalendar(http_client=OfflineClient(), annual_archive=False)
    assert disabled.trading_day_status(dt.date(2026, 9, 29)) is TradingDayStatus.UNKNOWN
    # The live-fetch primitive used by strict preparation never applies fallback.
    current, _ = archive_calendar()
    with pytest.raises(TwseCalendarUnavailableError, match="UPSTREAM_MAINTENANCE"):
        current._fetch_year_json(2026)


@pytest.mark.parametrize("damage", ["manifest", "object", "missing", "year", "source", "policy", "time", "oversized"])
def test_archive_corruption_is_fail_closed(tmp_path, damage):
    shutil.copytree(ROOT, tmp_path / "evidence")
    root = tmp_path / "evidence"
    path = root / "annual-observations.json"
    manifest = json.loads(path.read_bytes())
    document = manifest["documents"]["2026"]
    object_path = root / "objects" / document["sha256"]
    if damage == "manifest":
        path.write_bytes(path.read_bytes() + b" ")
        archive = ObservedAnnualSchedules(root)
    else:
        if damage == "object":
            object_path.write_bytes(b"corrupt")
        elif damage == "missing":
            object_path.unlink()
        elif damage == "year":
            payload = json.loads(object_path.read_bytes())
            payload["queryYear"] = 2025
            raw = json.dumps(payload).encode()
            document["sha256"] = hashlib.sha256(raw).hexdigest()
            document["bytes"] = len(raw)
            (root / "objects" / document["sha256"]).write_bytes(raw)
        elif damage == "source":
            document["source_url"] = "https://example.test/holidaySchedule"
        elif damage == "policy":
            document["availability_policy"] = "VERIFIED_HISTORICAL_PUBLICATION"
        elif damage == "time":
            document["retrieved_at"] = "2026-09-28T05:07:03"
        elif damage == "oversized":
            object_path.write_bytes(b" " * (512 * 1024 + 1))
        raw = json.dumps(manifest).encode()
        path.write_bytes(raw)
        archive = ObservedAnnualSchedules(root, expected_sha256=hashlib.sha256(raw).hexdigest())
    calendar, client = archive_calendar(archive)
    assert calendar.trading_day_status(dt.date(2026, 9, 29)) is TradingDayStatus.UNKNOWN
    assert calendar.trading_day_status(dt.date(2026, 9, 30)) is TradingDayStatus.UNKNOWN
    assert client.calls == ["20260101"]
    assert "archive invalid" in calendar.annual_failures()["2026"]
    assert calendar.annual_evidence() == {}
