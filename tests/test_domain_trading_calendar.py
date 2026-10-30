"""Pure annual-parser extraction parity; source payloads are test fixtures."""
import datetime as dt

import pytest

import domain.trading_calendar as pure
import infrastructure.calendar.twse_calendar as adapter
from tests.test_twse_calendar import CALENDAR_PAYLOAD, HISTORICAL_2025_PAYLOAD


@pytest.mark.parametrize("name", ["TradingDayStatus", "TwseCalendarError",
    "TwseCalendarUnavailableError", "TwseCalendarSchemaError"])
def test_existing_public_status_and_error_imports_are_identical_aliases(name):
    assert getattr(adapter, name) is getattr(pure, name)


@pytest.mark.parametrize("name", ["_parse_schedule", "_parse_openapi_schedule",
    "_parse_historical_schedule", "_parse_open_state"])
def test_transport_reuses_the_same_parser_not_a_second_implementation(name):
    assert getattr(adapter.TwseCalendar, name).__func__ is getattr(pure.OfficialAnnualCalendar, name).__func__


@pytest.mark.parametrize("payload,year", [(CALENDAR_PAYLOAD, 2026), (HISTORICAL_2025_PAYLOAD, 2025)])
def test_all_year_days_match_existing_injected_transport_adapter(payload, year):
    original = adapter.TwseCalendar(fetcher=lambda: payload, annual_archive=False)
    calendar = pure.OfficialAnnualCalendar(payload)
    day = dt.date(year, 1, 1)
    while day.year == year:
        assert calendar.trading_day_status(day) is original.trading_day_status(day)
        day += dt.timedelta(days=1)
    assert calendar.trading_day_status(dt.date(year - 1, 7, 20)) is pure.TradingDayStatus.UNKNOWN


@pytest.mark.parametrize("payload", [None, [], {}, {"stat": "error"},
    [{"Date": "1150102", "Name": "不明事件"}],
    [{"Date": "1150102", "Name": "開始交易"}, {"Date": "1150102", "Name": "休市"}]])
def test_bad_source_is_lazy_unknown_without_network_fallback(payload):
    calendar = pure.OfficialAnnualCalendar(payload)
    assert calendar.trading_day_status(dt.date(2026, 1, 2)) is pure.TradingDayStatus.UNKNOWN
    assert calendar.trading_day_status(dt.date(2026, 1, 3)) is pure.TradingDayStatus.UNKNOWN
    with pytest.raises(pure.TwseCalendarError):
        calendar.is_trading_day(dt.date(2026, 1, 2))


def test_source_is_parsed_once_and_covered_year_required_before_weekday_default(monkeypatch):
    calls = []
    original = pure.OfficialAnnualCalendar._parse_schedule.__func__

    def observed(cls, payload):
        calls.append(payload)
        return original(cls, payload)

    monkeypatch.setattr(pure.OfficialAnnualCalendar, "_parse_schedule", classmethod(observed))
    calendar = pure.OfficialAnnualCalendar(CALENDAR_PAYLOAD)
    assert calls == []
    assert calendar.trading_day_status(dt.date(2026, 7, 20)) is pure.TradingDayStatus.OPEN
    assert calendar.trading_day_status(dt.date(2026, 7, 19)) is pure.TradingDayStatus.CLOSED
    assert calendar.trading_day_status(dt.date(2025, 7, 21)) is pure.TradingDayStatus.UNKNOWN
    assert len(calls) == 1


# Exact rows from captured official transport bytes, not inferred from missing
# quotes. These fixtures document parser inputs; source authentication remains
# the adapter's responsibility. URLs are the annual query for the stated year.
CAPTURED_ANNUAL_ROWS = {
    2021: {
        "url": "https://www.twse.com.tw/holidaySchedule/holidaySchedule?response=json&date=20210101",
        "sha256": "cc048d412679032846398c796975bbbb14339e1ad657efbb4c97c252495cc876",
        "rows": [
            ["2021-02-05", "農曆春節前最後交易日", "農曆春節前最後交易。"],
            ["2021-02-08", "農曆春節前最後交易日", "2月8日市場無交易，僅辦理結算交割作業。"],
            ["2021-02-09", "農曆春節前最後交易日", "2月9日市場無交易，僅辦理結算交割作業。"],
        ],
    },
    2022: {
        "url": "https://www.twse.com.tw/holidaySchedule/holidaySchedule?response=json&date=20220101",
        "sha256": "c49edf8212319a24ce1a964e4997b86409971c79cc300bb3e614749ccf7bce8d",
        "rows": [
            ["2022-01-26", "農曆春節前最後交易日", "農曆春節前最後交易。"],
            ["2022-01-27", "農曆春節前最後交易日", "1月27日市場無交易，僅辦理結算交割作業。"],
            ["2022-01-28", "農曆春節前最後交易日", "1月28日市場無交易，僅辦理結算交割作業。"],
        ],
    },
    2023: {
        "url": "https://www.twse.com.tw/holidaySchedule/holidaySchedule?response=json&date=20230101",
        "sha256": "57ac0ea5fdb48cc8dccdc2c691cc3e331f62909d2f63424778aa7a391ca26740",
        "rows": [
            ["2023-01-17", "農曆春節前最後交易日",
             r"農曆春節前最後交易。\r\n1月18日及1月19日市場無交易，僅辦理結算交割作業。"],
            ["2023-01-18", "市場無交易，僅辦理結算交割作業", ""],
            ["2023-01-19", "市場無交易，僅辦理結算交割作業", ""],
        ],
    },
}


@pytest.mark.parametrize("format", ["historical", "openapi"])
@pytest.mark.parametrize("year,expected", [
    (2021, [True, False, False]), (2022, [True, False, False]),
    (2023, [True, False, False]),
])
def test_captured_same_day_settlement_overrides_mislabeled_last_trade_only(format, year, expected):
    source = CAPTURED_ANNUAL_ROWS[year]
    assert len(source["sha256"]) == 64 and source["url"].endswith(f"date={year}0101")
    rows = source["rows"]
    if format == "historical":
        payload = {"stat": "ok", "fields": ["日期", "名稱", "說明"], "data": rows}
    else:
        payload = [{"Date": f"{year-1911}{day[5:7]}{day[8:10]}", "Name": name, "Description": description}
                   for day, name, description in rows]
    calendar = pure.OfficialAnnualCalendar(payload)
    for row, is_open in zip(rows, expected, strict=True):
        assert calendar.is_trading_day(dt.date.fromisoformat(row[0])) is is_open
    assert pure.ANNUAL_PARSER_VERSION == "OFFICIAL_ANNUAL_CURRENT_SESSION_R2"


@pytest.mark.parametrize("separator", ["\r\n", r"\r\n", "<br>", "<BR />"])
def test_current_last_trade_is_not_closed_by_future_day_explanations(separator):
    payload = {"stat": "ok", "fields": ["日期", "名稱", "說明"], "data": [[
        "2023-01-17", "農曆春節前最後交易日",
        "農曆春節前最後交易。" + separator + "1月18日及1月19日市場無交易，僅辦理結算交割作業。",
    ]]}
    assert pure.OfficialAnnualCalendar(payload).trading_day_status(dt.date(2023, 1, 17)) is pure.TradingDayStatus.OPEN


@pytest.mark.parametrize("name,description", [
    ("交易安排", "1月17日開始交易。1月17日休市。"),
    ("最後交易日", "1月17日最後交易但同日休市。"),
    ("最後交易日", "市場無交易，僅辦理結算交割作業。"),
    ("休市", "1月17日開始交易。"),
])
def test_synthetic_ambiguous_current_day_conflicts_remain_unknown(name, description):
    payload = {"stat": "ok", "fields": ["日期", "名稱", "說明"],
               "data": [["2023-01-17", name, description]]}
    calendar = pure.OfficialAnnualCalendar(payload)
    assert calendar.trading_day_status(dt.date(2023, 1, 17)) is pure.TradingDayStatus.UNKNOWN
    with pytest.raises(pure.TwseCalendarSchemaError, match="Conflicting"):
        calendar.is_trading_day(dt.date(2023, 1, 17))


def test_dated_holiday_range_covers_interior_day_without_weekday_fallback():
    # Real source phrasing for a duration; the interior is not a range endpoint.
    payload = {"stat": "ok", "fields": ["日期", "名稱", "說明"], "data": [[
        "2023-01-23", "農曆春節", "依規定於1月22日至1月24日放假3日。",
    ]]}
    assert pure.OfficialAnnualCalendar(payload).trading_day_status(dt.date(2023, 1, 23)) is pure.TradingDayStatus.CLOSED
