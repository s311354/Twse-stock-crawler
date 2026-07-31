import datetime

import pytest

from infrastructure.calendar.twse_calendar import TradingDayStatus
from infrastructure.calendar.twse_calendar import TwseCalendar
from infrastructure.calendar.twse_calendar import TwseCalendarSchemaError
from infrastructure.calendar.twse_calendar import TwseCalendarUnavailableError
from infrastructure.crawler.twse_http_client import TwseNetworkError


CALENDAR_PAYLOAD = [
    {
        "Name": "國曆新年開始交易日",
        "Date": "1150102",
        "Weekday": "五",
        "Description": "國曆新年開始交易。",
    },
    {
        "Name": "農曆春節前最後交易日",
        "Date": "1150211",
        "Weekday": "三",
        "Description": "農曆春節前最後交易。",
    },
    {
        "Name": "市場無交易，僅辦理結算交割作業",
        "Date": "1150212",
        "Weekday": "四",
        "Description": "",
    },
    {
        "Name": "農曆除夕及春節",
        "Date": "1150216",
        "Weekday": "一",
        "Description": "依規定放假1日。",
    },
    {
        "Name": "和平紀念日",
        "Date": "1150227",
        "Weekday": "五",
        "Description": "和平紀念日為2月28日適逢星期六，於2月27日（星期五）補假。",
    },
]


def test_calendar_distinguishes_open_events_holidays_and_weekdays() -> None:
    calendar = TwseCalendar(fetcher=lambda: CALENDAR_PAYLOAD)

    assert calendar.is_trading_day(datetime.date(2026, 1, 2)) is True
    assert calendar.is_trading_day(datetime.date(2026, 2, 11)) is True
    assert calendar.is_trading_day(datetime.date(2026, 2, 12)) is False
    assert calendar.is_trading_day(datetime.date(2026, 2, 16)) is False
    assert calendar.is_trading_day(datetime.date(2026, 2, 27)) is False
    assert calendar.is_trading_day(datetime.date(2026, 7, 20)) is True
    assert calendar.is_trading_day(datetime.date(2026, 7, 19)) is False


def test_calendar_maps_http_network_error_to_unknown() -> None:
    class FailingHttpClient:
        def get_json(self, url: str, *, context: str) -> object:
            raise TwseNetworkError("offline")

    calendar = TwseCalendar(http_client=FailingHttpClient())
    assert calendar.trading_day_status(datetime.date(2026, 7, 20)) is TradingDayStatus.UNKNOWN
    with pytest.raises(TwseCalendarUnavailableError, match="offline"):
        calendar.is_trading_day(datetime.date(2026, 7, 20))


def test_calendar_returns_unknown_for_year_not_covered_by_payload() -> None:
    calendar = TwseCalendar(fetcher=lambda: CALENDAR_PAYLOAD)

    assert calendar.trading_day_status(datetime.date(2025, 7, 21)) is TradingDayStatus.UNKNOWN
    with pytest.raises(TwseCalendarUnavailableError, match="does not cover year 2025"):
        calendar.is_trading_day(datetime.date(2025, 7, 21))


def test_calendar_rejects_ambiguous_schema_instead_of_assuming_closed() -> None:
    calendar = TwseCalendar(
        fetcher=lambda: [
            {
                "Name": "不明事件",
                "Date": "1150720",
                "Weekday": "一",
                "Description": "",
            }
        ]
    )

    with pytest.raises(TwseCalendarSchemaError, match="Unrecognized"):
        calendar.is_trading_day(datetime.date(2026, 7, 20))


def test_market_open_at_uses_taipei_regular_session_boundaries() -> None:
    calendar = TwseCalendar(fetcher=lambda: CALENDAR_PAYLOAD)

    assert calendar.is_market_open_at(datetime.datetime(2026, 7, 20, 9, 0)) is True
    assert calendar.is_market_open_at(datetime.datetime(2026, 7, 20, 13, 29, 59)) is True
    assert calendar.is_market_open_at(datetime.datetime(2026, 7, 20, 13, 30)) is False
    assert calendar.is_market_open_at(datetime.datetime(2026, 2, 16, 10, 0)) is False
