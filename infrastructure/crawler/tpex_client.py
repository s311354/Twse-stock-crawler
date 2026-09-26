"""Historical TPEx semiconductor quotes using the shared HTTP retry policy."""

from __future__ import annotations

import datetime as dt
import html
import logging
import math
import re
import unicodedata

from domain.errors import NoDataError, SchemaChangedError
from domain.models import StockRows
from domain.tpex_market import TpexDailyQuote
from domain.tpex_market import TPEX_FIELDS
from infrastructure.crawler.twse_client import TwseClient
from infrastructure.crawler.twse_http_client import TwseHttpClient


TPEX_DAILY_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc"

def _text(value: object) -> str:
    return unicodedata.normalize("NFKC", html.unescape(str(value))).strip()


def _field(value: object) -> str:
    name = re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", _text(value)))
    # Older TPEx tables label common-stock quote quantities in thousand shares.
    # One common-stock board lot is 1,000 shares, so these units are equivalent.
    return {"最後買量(千股)": "最後買量(張數)", "最後賣量(千股)": "最後賣量(張數)"}.get(name, name)


def _number(value: str, *, optional: bool = False, integer: bool = False) -> float | int | None:
    text = _text(value).replace(",", "")
    if optional and text in {"", "-", "--", "---", "----"}:
        return None
    try:
        number = float(text)
    except ValueError as error:
        raise SchemaChangedError(f"TPEx invalid numeric cell: {value!r}") from error
    if not math.isfinite(number) or number < 0 or (integer and not number.is_integer()):
        raise SchemaChangedError(f"TPEx invalid numeric cell: {value!r}")
    return int(number) if integer else number


def parse_tpex_quotes(payload: object, trade_date: str) -> list[TpexDailyQuote]:
    """Validate date/category/schema before mapping by field names, not indexes.

    Volume/issued shares are shares; amount is TWD; bid/ask quantities are lots.
    Missing OHLC stays missing so the collector can skip it, never synthesize zero.
    """
    target = dt.datetime.strptime(trade_date, "%Y%m%d").date()
    if not isinstance(payload, dict):
        raise SchemaChangedError("TPEx daily response is not an object")
    status = str(payload.get("stat", "")).strip()
    if status.lower() != "ok":
        if any(marker in status for marker in ("查無資料", "沒有符合", "無資料")):
            raise NoDataError(f"TPEx market data at {trade_date}: {status}")
        raise SchemaChangedError(f"TPEx unexpected status at {trade_date}: {status!r}")
    if str(payload.get("date", "")) != trade_date:
        raise SchemaChangedError(f"TPEx response date does not match {trade_date}")
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise SchemaChangedError("TPEx response is missing tables")
    candidates = [table for table in tables if isinstance(table, dict)
                  and "每日收盤行情" in str(table.get("title", ""))]
    if not candidates and status.lower() == "ok" and not tables:
        raise NoDataError(f"TPEx stock table at {trade_date} is empty")
    if len(candidates) != 1:
        raise SchemaChangedError("TPEx response needs exactly one daily stock table")
    table = candidates[0]
    roc_date = f"{target.year - 1911}/{target.month:02d}/{target.day:02d}"
    if table.get("date") != roc_date or table.get("category") != "半導體類":
        raise SchemaChangedError("TPEx table date/category mismatch; expected semiconductor quotes")
    fields, data = table.get("fields"), table.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        raise SchemaChangedError("TPEx stock table requires fields and data arrays")
    names = [_field(value) for value in fields]
    if len(names) != len(set(names)) or not set(TPEX_FIELDS).issubset(names):
        raise SchemaChangedError(f"TPEx stock schema missing/duplicate fields: {names!r}")
    if table.get("totalCount") != len(data):
        raise SchemaChangedError("TPEx stock table totalCount mismatch")
    if not data:
        raise NoDataError(f"TPEx stock table at {trade_date} is empty")
    quotes = []
    seen = set()
    for row in data:
        if not isinstance(row, list) or len(row) != len(fields):
            raise SchemaChangedError("TPEx stock row width mismatch")
        record = dict(zip(names, row))
        symbol, name = _text(record["代號"]), _text(record["名稱"])
        if not re.fullmatch(r"[0-9A-Z]{4,10}", symbol) or not name or symbol in seen:
            raise SchemaChangedError(f"TPEx invalid/duplicate stock identity: {symbol!r}")
        seen.add(symbol)
        def number(key: str, *, integer: bool = False, optional: bool = True):
            return _number(str(record[key]), integer=integer, optional=optional)
        prices = {key: number(key) for key in ("開盤", "最高", "最低", "收盤")}
        # Invalid price ranges are data errors, not tradable prices.
        if any(value is not None and value <= 0 for value in prices.values()) or (
            all(value is not None for value in prices.values())
            and not (prices["最低"] <= min(prices["開盤"], prices["收盤"])
                     <= max(prices["開盤"], prices["收盤"]) <= prices["最高"])
        ):
            logging.debug("TPEx invalid OHLC symbol=%s date=%s; row not tradable", symbol, trade_date,
                          extra={"source": "TPEX", "symbol": symbol, "trade_date": trade_date,
                                 "status": "OHLC_INVALID"})
            prices = dict.fromkeys(prices)
        quotes.append(TpexDailyQuote(
            trade_date, symbol, name, prices["收盤"], _text(record["漲跌"]),
            prices["開盤"], prices["最高"], prices["最低"],
            number("成交股數", integer=True, optional=False),
            number("成交金額(元)", optional=False), number("成交筆數", integer=True, optional=False),
            number("最後買價") or None, number("最後買量(張數)"),
            number("最後賣價") or None, number("最後賣量(張數)"),
            number("發行股數", integer=True), number("次日漲停價"), number("次日跌停價"),
            tuple(str(record[key]) for key in TPEX_FIELDS),
        ))
    return quotes


def to_legacy_row(quote: TpexDailyQuote) -> list[str]:
    """Explicit adapter to the existing OHLCV consumer, with raw TPEx suffix.

    The endpoint has no PE. Leave it unavailable, never map issued shares to PE.
    Split numeric price change for breadth; ex-right/dividend markers stay unknown.
    """
    change = quote.change.replace(",", "")
    numeric_change = bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", change))
    sign = "-" if change.startswith("-") else "+"
    values = [
        quote.stock_no, quote.stock_name, quote.volume, quote.trades, quote.amount,
        quote.open, quote.high, quote.low, quote.close,
        sign if numeric_change else "", abs(float(change)) if numeric_change else None,
        quote.bid, quote.bid_lots, quote.ask, quote.ask_lots, None,
    ]
    return ["--" if value is None else str(value) for value in values] + list(quote.raw_cells)


class TpexClient:
    """TPEx sector 24 stock source; TAIEX remains the explicit TWSE benchmark."""

    def __init__(self, http_client: TwseHttpClient | None = None, *, benchmark: TwseClient | None = None):
        self.http_client = http_client or TwseHttpClient()
        self.benchmark = benchmark or TwseClient(http_client=self.http_client)

    def get_daily_stock_rows(self, date_time: str, stocktype: int | str) -> StockRows:
        if str(stocktype) != "24":
            raise ValueError("TPEx source supports SEMI (24) only; no TWSE/full-market fallback")
        date = dt.datetime.strptime(date_time, "%Y%m%d").date()
        try:
            response = self.http_client.get_json(
                TPEX_DAILY_URL,
                params={"date": date.strftime("%Y/%m/%d"), "type": "24", "id": "",
                        "response": "json", "order": "0", "sort": "asc"},
                context=f"TPEx market data at {date_time}",
                payload_validator=lambda payload: parse_tpex_quotes(payload, date_time),
            )
        except NoDataError:
            # No data is not proof of an unscheduled market closure.
            return []
        return [to_legacy_row(quote) for quote in parse_tpex_quotes(response.payload, date_time)]

    def get_daily_taiex_close(self, date_time: str) -> float | None:
        return self.benchmark.get_daily_taiex_close(date_time)
