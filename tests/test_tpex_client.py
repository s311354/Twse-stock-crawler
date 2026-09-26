"""Offline contract tests for the official TPEx daily JSON shape."""
from copy import deepcopy
import datetime as dt
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from requests import exceptions

from domain.errors import NetworkError, NoDataError, SchemaChangedError
from domain.models import TwseColumns
from domain.tpex_market import TPEX_FIELDS
from infrastructure.crawler.tpex_client import TpexClient, parse_tpex_quotes, to_legacy_row
from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.fundamentals.mops_client import MopsClient
from tests.test_twse_http_client import FakeResponse, FakeSession, config


def payload(date="20260924", symbols=("3105", "5347")):
    target = dt.datetime.strptime(date, "%Y%m%d")
    return {
        "date": date, "stat": "ok", "flagField": "張數",
        "tables": [{
            "title": "上櫃股票每日收盤行情(不含定價)",
            "date": f"{target.year - 1911}/{target.month:02d}/{target.day:02d}",
            "category": "半導體類", "totalCount": len(symbols),
            "fields": ["代號", "名稱", "收盤 ", "漲跌", "開盤 ", "最高 ", "最低", "成交股數  ",
                       " 成交金額(元)", " 成交筆數 ", "最後買價", "最後買量<br>(張數)", "最後賣價",
                       "最後賣量<br>(張數)", "發行股數 ", "次日漲停價 ", "次日跌停價"],
            "data": [[symbol, "測試半導體", "502.00", "+5.00", "495.50", "506.00", "482.00",
                      "21,240,000", "10,515,944,500", "16,350", "502.00", "33", "503.00", "212",
                      "423,940,384", "552.00", "452.00"] for symbol in symbols],
        }],
    }


def test_named_mapping_and_units_with_reordered_fields():
    original = payload()
    rearranged = deepcopy(original)
    table = rearranged["tables"][0]
    table["fields"].reverse()
    for row in table["data"]:
        row.reverse()
    assert parse_tpex_quotes(rearranged, "20260924") == parse_tpex_quotes(original, "20260924")
    quote = parse_tpex_quotes(original, "20260924")[0]
    assert (quote.stock_no, quote.open, quote.close) == ("3105", 495.5, 502.)
    assert (quote.volume, quote.amount, quote.trades) == (21_240_000, 10_515_944_500., 16_350)
    assert (quote.bid_lots, quote.ask_lots, quote.issued_shares) == (33, 212, 423_940_384)
    row = to_legacy_row(quote)
    assert row[TwseColumns.PE] == "--"
    assert row[TwseColumns.CHANGE_SIGN:TwseColumns.PRICE_CHANGE + 1] == ["+", "5.0"]
    assert row[16:] == list(quote.raw_cells)


def test_historical_thousand_share_quote_units_match_common_stock_lots():
    data = payload("20250106", symbols=("3105",))
    data["flagField"] = "千股"
    data["tables"][0]["fields"] = [name.replace("張數", "千股") for name in data["tables"][0]["fields"]]
    quote = parse_tpex_quotes(data, "20250106")[0]
    assert quote.bid_lots == 33 and quote.ask_lots == 212
    assert quote.volume == 21_240_000
    assert quote.raw_cells[11] == "33"


def test_unrecognized_quote_quantity_unit_fails_instead_of_guessing():
    data = payload()
    data["tables"][0]["fields"][11] = "最後買量(股)"
    with pytest.raises(SchemaChangedError):
        parse_tpex_quotes(data, "20260924")


@pytest.mark.parametrize("change", ["-5.00", "除息", "0.00"])
def test_change_direction_and_corporate_action_markers(change):
    data = payload(symbols=("3105",))
    data["tables"][0]["data"][0][3] = change
    quote = parse_tpex_quotes(data, "20260924")[0]
    row = to_legacy_row(quote)
    assert quote.change == change
    if change == "除息":
        assert row[TwseColumns.PRICE_CHANGE] == "--"
    else:
        assert float(row[TwseColumns.PRICE_CHANGE]) == abs(float(change))
        assert row[TwseColumns.CHANGE_SIGN] == ("-" if change.startswith("-") else "+")


@pytest.mark.parametrize("mutation", ["date", "table_date", "category", "fields", "width", "duplicate", "count", "identity", "volume", "amount"])
def test_invalid_schema_cannot_be_used_as_stock_data(mutation):
    data = payload()
    table = data["tables"][0]
    if mutation == "date": data["date"] = "20260923"
    elif mutation == "table_date": table["date"] = "115/09/23"
    elif mutation == "category": table["category"] = "所有類股"
    elif mutation == "fields": table["fields"][7] = "成交張數"
    elif mutation == "width": table["data"][0].pop()
    elif mutation == "duplicate": table["data"][1][0] = table["data"][0][0]
    elif mutation == "count": table["totalCount"] = 99
    elif mutation == "identity": table["data"][0][1] = ""
    elif mutation == "volume": table["data"][0][7] = "1.5"
    elif mutation == "amount": table["data"][0][8] = "NaN"
    with pytest.raises(SchemaChangedError):
        parse_tpex_quotes(data, "20260924")


def test_missing_and_invalid_ohlc_are_not_zero_or_tradable():
    data = payload()
    data["tables"][0]["data"][0][2] = "---"
    data["tables"][0]["data"][1][5] = "1.00"  # High below open/close.
    quotes = parse_tpex_quotes(data, "20260924")
    assert all(quote.close is None for quote in quotes)
    assert all(to_legacy_row(quote)[TwseColumns.CLOSE] == "--" for quote in quotes)
    assert quotes[1].raw_cells[5] == "1.00"


def test_absent_bid_ask_is_unknown_not_a_zero_quote():
    data = payload(symbols=("3105",))
    data["tables"][0]["data"][0][10] = "0.00"
    data["tables"][0]["data"][0][12] = "---"
    quote = parse_tpex_quotes(data, "20260924")[0]
    assert quote.bid is quote.ask is None


def test_http_retries_transient_error_and_never_changes_market(caplog):
    session = FakeSession([exceptions.ConnectionError("offline"), FakeResponse(payload())])
    http = TwseHttpClient(config=config(), session=session, sleeper=lambda _: None)
    client = TpexClient(http)
    with caplog.at_level("DEBUG"):
        rows = client.get_daily_stock_rows("20260924", 24)
    assert len(rows) == 2 and len(session.calls) == 2
    assert session.calls[0]["params"]["date"] == "2026/09/24"
    assert session.calls[0]["params"]["type"] == "24"
    assert all("tpex.org.tw" in call["url"] for call in session.calls)
    assert any(getattr(record, "source", None) == "TPEX" for record in caplog.records)
    with pytest.raises(ValueError, match="SEMI"):
        client.get_daily_stock_rows("20260924", "ALLBUT0999")


def test_empty_weekend_is_no_data_not_closed_or_schema_error():
    data = payload("20260920", symbols=())
    with pytest.raises(NoDataError):
        parse_tpex_quotes(data, "20260920")
    session = FakeSession([FakeResponse(data)])
    client = TpexClient(TwseHttpClient(config=config(), session=session))
    assert client.get_daily_stock_rows("20260920", 24) == []
    assert len(session.calls) == 1


def test_network_exhaustion_is_not_swallowed_as_no_data():
    session = FakeSession([exceptions.Timeout("timeout"), exceptions.Timeout("timeout")])
    client = TpexClient(TwseHttpClient(config=config(), session=session, sleeper=lambda _: None))
    with pytest.raises(NetworkError):
        client.get_daily_stock_rows("20260924", 24)


def test_mops_revision_scan_selects_otc_and_does_not_reuse_sii_response():
    client = MopsClient(http_client=Mock())
    client._request_bytes = Mock(return_value=(SimpleNamespace(content=b"<table></table>"), "https://example.test"))
    client._load_corrections((2026, 2))
    assert client._request_bytes.call_args.kwargs["data"]["TYPEK"] == "sii"
    client.select_market("TPEX")
    client._load_corrections((2026, 2))
    assert client._request_bytes.call_args.kwargs["data"]["TYPEK"] == "otc"
    assert client._request_bytes.call_count == 2
    assert client.load_company_sectors() == {}
    client.http_client.get_json.assert_not_called()
    client.select_market("TWSE")
    client._load_corrections((2026, 2))
    assert client._request_bytes.call_count == 3
    assert client._request_bytes.call_args.kwargs["data"]["TYPEK"] == "sii"
