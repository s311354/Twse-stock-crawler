#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from __future__ import annotations

from interface.cli import create_parser

r"""
TWSE Real-time Stock Price Analysis

This script is designed to help streamline models, by taking the Taiwan Stock Exchange dataset. This will be used by an application and figuring out the maximum profit that are required to run for those arguments. The resulting is then sending to private email sandbox.
"""

import argparse
import pandas as pd
import enum
import time
import datetime
import logging

from application.analysis_dataset import AnalysisDatasetBuilder
from domain.models import Stocktype
from domain.models import Ohlc
from domain.models import StockSelector
from domain.models import StockRow
from domain.models import StockRows
from domain.models import TWSEStock
from domain.models import TwseColumns
from domain.market_breadth import derive_twse_market_breadth
from domain.trading_days import build_backtrack_dates
from infrastructure.calendar.twse_calendar import TradingDayStatus
from infrastructure.calendar.twse_calendar import TwseCalendar
from infrastructure.crawler.twse_client import TwseClient
from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.legacy_crawler_facade import CsvRepositoryPort
from infrastructure.legacy_crawler_facade import LegacyCrawlerFacade
from twse.parser import clean_cell
from twse.parser import parse_price


class QUERY(enum.Enum):
    _order_ = 'NEOGENE PALEOGENE CRETACEOUS'
    NEOGENE = 12
    PALEOGENE = 15
    CRETACEOUS = 30

class TwseCrawker(LegacyCrawlerFacade):
    _logged_missing_price_warnings: set[tuple[str, str, str, str, str, str]] = set()

    def __init__(
        self,
        stocklistsize: int,
        twse_client: TwseClient | None = None,
        csv_repository: CsvRepositoryPort | None = None,
        http_client: TwseHttpClient | None = None,
        app_settings: object | None = None,
        analysis_dataset_builder: AnalysisDatasetBuilder | None = None,
    ) -> None:
        if twse_client is not None and http_client is not None:
            raise ValueError("Pass either twse_client or http_client, not both")
        self.stocklistsize: int = stocklistsize
        self.twse_client = twse_client or TwseClient(http_client=http_client)
        self._initialize_legacy_facade(
            csv_repository=csv_repository,
            app_settings=app_settings,
            analysis_dataset_builder=analysis_dataset_builder,
        )
        self.daily_stocks: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.stocknumbers: list[str] = ['' for _ in range(self.stocklistsize)]
        self.stocknames: list[str] = ['' for _ in range(self.stocklistsize)]
        self.tracked_stock_numbers: list[str | None] = [None for _ in range(self.stocklistsize)]
        self.missing_tracked_stock_numbers: set[str] = set()
        self.stocksprice: list[float | None] = list()
        self.daily_highs: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_lows: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_closes: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_volumes: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_trade_counts: list[list[float | None]] = [[] for _ in range(self.stocklistsize)]
        self.daily_trade_values: list[list[float | None]] = [[] for _ in range(self.stocklistsize)]
        self.daily_bid_prices: list[list[float | None]] = [[] for _ in range(self.stocklistsize)]
        self.daily_ask_prices: list[list[float | None]] = [[] for _ in range(self.stocklistsize)]
        self.daily_pe_ratios: list[list[float | None]] = [[] for _ in range(self.stocklistsize)]
        self.daily_taiex_closes: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_dates: list[list[str]] = [[] for _ in range(self.stocklistsize)]
        self.market_breadth_snapshots_by_date: dict[
            str,
            dict[str, object],
        ] = {}
        self.annual_fundamentals_by_stock: dict[str, pd.DataFrame] = {}
        self.fundamental_snapshots_by_stock: dict[str, dict[str, object]] = {}
        self.fundamental_status_by_stock: dict[str, str] = {}
        self.fundamental_status_detail_by_stock: dict[str, str] = {}
        self.low_entry_v4_signal_dates_by_stock: dict[str, str] = {}
        self.low_entry_v4_account_equity: float | None = None
        self.iso_scheduled_times: list[str] = list()
        self.transactiondays: int = 0


    def __del__(self) -> None:
        self.daily_stocks = [[] for _ in range(self.stocklistsize)]
        self.iso_scheduled_times = list()
        self.transactiondays = 0
        self.stocksprice = list()
        self.tracked_stock_numbers = [None for _ in range(self.stocklistsize)]
        self.missing_tracked_stock_numbers = set()
        self.daily_highs = [[] for _ in range(self.stocklistsize)]
        self.daily_lows = [[] for _ in range(self.stocklistsize)]
        self.daily_closes = [[] for _ in range(self.stocklistsize)]
        self.daily_volumes = [[] for _ in range(self.stocklistsize)]
        self.daily_trade_counts = [[] for _ in range(self.stocklistsize)]
        self.daily_trade_values = [[] for _ in range(self.stocklistsize)]
        self.daily_bid_prices = [[] for _ in range(self.stocklistsize)]
        self.daily_ask_prices = [[] for _ in range(self.stocklistsize)]
        self.daily_pe_ratios = [[] for _ in range(self.stocklistsize)]
        self.daily_taiex_closes = [[] for _ in range(self.stocklistsize)]
        self.daily_dates = [[] for _ in range(self.stocklistsize)]
        self.market_breadth_snapshots_by_date = {}
        self.annual_fundamentals_by_stock = {}
        self.fundamental_snapshots_by_stock = {}
        self.fundamental_status_by_stock = {}
        self.fundamental_status_detail_by_stock = {}
        self.low_entry_v4_signal_dates_by_stock = {}


    def clean_data(self, row: StockRow) -> StockRow:
        for index, content in enumerate(row):
            row[index] = clean_cell(content)
        return row


    def parse_price(self, price: object) -> float | None:
        return parse_price(price)


    def validate_twse_row_schema(self, stock_row: object, scheduled_time: str, source: str) -> bool:
        if not isinstance(stock_row, list):
            logging.warning(
                "Skipping TWSE row from %s at %s because row is not a list: %s",
                source,
                scheduled_time,
                stock_row,
            )
            return False

        if len(stock_row) < TwseColumns.REQUIRED_WIDTH:
            logging.warning(
                "Skipping TWSE row from %s at %s because row shape is invalid: expected at least %s columns, got %s: %s",
                source,
                scheduled_time,
                TwseColumns.REQUIRED_WIDTH,
                len(stock_row),
                stock_row,
            )
            return False

        stock_no = clean_cell(stock_row[TwseColumns.STOCK_NO])
        stock_name = clean_cell(stock_row[TwseColumns.NAME])
        if not stock_no or not stock_name:
            logging.warning(
                "Skipping TWSE row from %s at %s because stock identity is missing: stock_no=%s stock_name=%s",
                source,
                scheduled_time,
                stock_no,
                stock_name,
            )
            return False

        return True


    def get_stock_row(self, row: StockRows, stock_index: int, scheduled_time: str) -> StockRow | None:
        if stock_index >= len(row):
            logging.warning(
                "Skipping stock index %s at %s because TWSE returned only %s rows.",
                stock_index,
                scheduled_time,
                len(row),
            )
            return None

        stock_row = row[stock_index]
        if not self.validate_twse_row_schema(stock_row, scheduled_time, "index {}".format(stock_index)):
            return None

        return stock_row


    def build_stock_lookup(self, rows: StockRows, scheduled_time: str) -> dict[str, StockRow]:
        stock_lookup: dict[str, StockRow] = {}

        for row_index, stock_row in enumerate(rows):
            if not self.validate_twse_row_schema(stock_row, scheduled_time, "index {}".format(row_index)):
                continue

            stock_no = clean_cell(stock_row[TwseColumns.STOCK_NO])
            if stock_no in stock_lookup:
                logging.warning(
                    "Duplicate TWSE stock number %s at %s; keeping the first row and skipping duplicate index %s.",
                    stock_no,
                    scheduled_time,
                    row_index,
                )
                continue

            stock_lookup[stock_no] = stock_row

        return stock_lookup


    def parse_selector_as_index(self, selector: StockSelector) -> int | None:
        selector_text = str(selector).strip()
        if not selector_text.isdigit():
            return None
        return int(selector_text)


    def resolve_stock_row(
        self,
        rows: StockRows,
        stock_lookup: dict[str, StockRow],
        selector: StockSelector,
        item: int,
        scheduled_time: str,
    ) -> StockRow | None:
        tracked_stock_no = self.tracked_stock_numbers[item]
        if tracked_stock_no:
            stock_row = stock_lookup.get(tracked_stock_no)
            if stock_row is None:
                if tracked_stock_no not in self.missing_tracked_stock_numbers:
                    logging.warning(
                        "Skipping tracked stock %s starting at %s because it is missing from the TWSE response.",
                        tracked_stock_no,
                        scheduled_time,
                    )
                    self.missing_tracked_stock_numbers.add(tracked_stock_no)
                else:
                    logging.debug(
                        "Tracked stock %s remains missing from the TWSE response at %s.",
                        tracked_stock_no,
                        scheduled_time,
                    )
            elif tracked_stock_no in self.missing_tracked_stock_numbers:
                logging.info(
                    "Tracked stock %s returned to the TWSE response at %s.",
                    tracked_stock_no,
                    scheduled_time,
                )
                self.missing_tracked_stock_numbers.remove(tracked_stock_no)
            return stock_row

        selector_text = str(selector).strip()
        stock_row = stock_lookup.get(selector_text)
        if stock_row is not None:
            self.tracked_stock_numbers[item] = selector_text
            logging.info("Tracking stock %s from stock-number selector at %s.", selector_text, scheduled_time)
            return stock_row

        if selector_text.isdigit() and selector_text != str(int(selector_text)):
            logging.warning(
                "Skipping stock-number selector %s at %s because it is missing from the TWSE response.",
                selector_text,
                scheduled_time,
            )
            return None

        stock_index = self.parse_selector_as_index(selector)
        if stock_index is None:
            logging.warning(
                "Skipping selector %s at %s because it is neither a TWSE stock number nor a legacy row index.",
                selector,
                scheduled_time,
            )
            return None

        stock_row = self.get_stock_row(rows, stock_index, scheduled_time)
        if stock_row is None:
            return None

        resolved_stock_no = clean_cell(stock_row[TwseColumns.STOCK_NO])
        self.tracked_stock_numbers[item] = resolved_stock_no
        logging.info(
            "Resolved legacy row index %s to stock %s at %s; future dates will use stock-number lookup.",
            stock_index,
            resolved_stock_no,
            scheduled_time,
        )
        return stock_row


    def build_record_row(self, stock_row: StockRow) -> StockRow:
        normalized_row = list(stock_row[:TwseColumns.REQUIRED_WIDTH]) + [""] * max(0, TwseColumns.REQUIRED_WIDTH - len(stock_row))
        return self.clean_data([
                normalized_row[TwseColumns.STOCK_NO], # Stock number
                normalized_row[TwseColumns.NAME], # Stock name
                normalized_row[TwseColumns.VOLUME], # Trade volume
                normalized_row[TwseColumns.TRADE_COUNT], # Trade count
                normalized_row[TwseColumns.TRADE_VALUE], # Trade value
                normalized_row[TwseColumns.OPEN], # Stock opening price
                normalized_row[TwseColumns.HIGH], # Stock price high
                normalized_row[TwseColumns.LOW], # Stock price low
                normalized_row[TwseColumns.CLOSE], # Stock close price
                normalized_row[TwseColumns.CHANGE_SIGN], # Change sign
                normalized_row[TwseColumns.PRICE_CHANGE], # Price change
                normalized_row[TwseColumns.BID_PRICE], # Bid price
                normalized_row[TwseColumns.BID_VOLUME], # Bid volume
                normalized_row[TwseColumns.ASK_PRICE], # Ask price
                normalized_row[TwseColumns.ASK_VOLUME], # Ask volume
                normalized_row[TwseColumns.PE], # Pe ratio
                ])


    def parse_twse_stock(self, stock_row: StockRow, scheduled_time: str) -> TWSEStock | None:
        if not self.validate_twse_row_schema(stock_row, scheduled_time, "stock row"):
            return None

        stock_no = clean_cell(stock_row[TwseColumns.STOCK_NO])
        stock_name = clean_cell(stock_row[TwseColumns.NAME])
        open_price = self.parse_price(stock_row[TwseColumns.OPEN])
        high_price = self.parse_price(stock_row[TwseColumns.HIGH])
        low_price = self.parse_price(stock_row[TwseColumns.LOW])
        close_price = self.parse_price(stock_row[TwseColumns.CLOSE])
        volume = self.parse_price(stock_row[TwseColumns.VOLUME])
        trades = self.parse_price(stock_row[TwseColumns.TRADE_COUNT])
        amount = self.parse_price(stock_row[TwseColumns.TRADE_VALUE])
        bid = self.parse_price(stock_row[TwseColumns.BID_PRICE])
        ask = self.parse_price(stock_row[TwseColumns.ASK_PRICE])
        pe_ratio = self.parse_price(stock_row[TwseColumns.PE])

        if None in (open_price, high_price, low_price, close_price):
            warning_key = (
                stock_no,
                scheduled_time,
                stock_row[TwseColumns.OPEN],
                stock_row[TwseColumns.HIGH],
                stock_row[TwseColumns.LOW],
                stock_row[TwseColumns.CLOSE],
            )
            if warning_key not in TwseCrawker._logged_missing_price_warnings:
                logging.warning(
                    "Skipping stock %s at %s because OHLC data is unavailable: open=%s high=%s low=%s close=%s",
                    stock_no,
                    scheduled_time,
                    stock_row[TwseColumns.OPEN],
                    stock_row[TwseColumns.HIGH],
                    stock_row[TwseColumns.LOW],
                    stock_row[TwseColumns.CLOSE],
                )
                TwseCrawker._logged_missing_price_warnings.add(warning_key)
            return None

        return TWSEStock(
            stock_no=stock_no,
            stock_name=stock_name,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=int(volume) if volume is not None else 0,
            pe=pe_ratio,
            trades=int(trades) if trades is not None else None,
            amount=amount,
            bid=bid,
            ask=ask,
        )


    def parse_stock_row_ohlc(self, stock_row: StockRow, scheduled_time: str) -> Ohlc | None:
        twse_stock = self.parse_twse_stock(stock_row, scheduled_time)
        if twse_stock is None:
            return None

        return twse_stock.open, twse_stock.high, twse_stock.low, twse_stock.close


    def parse_stock_row_volume(self, stock_row: StockRow) -> float:
        volume = self.parse_price(stock_row[TwseColumns.VOLUME]) if len(stock_row) > TwseColumns.VOLUME else None
        return volume if volume is not None else 0.0


    def parse_stock_row_pe_ratio(self, stock_row: StockRow) -> float | None:
        if len(stock_row) <= TwseColumns.PE:
            return None
        return self.parse_price(stock_row[TwseColumns.PE])


    def update_tracked_stock_identity(self, item: int, twse_stock: TWSEStock, scheduled_time: str) -> bool:
        expected_stock_no = self.tracked_stock_numbers[item]
        if expected_stock_no and twse_stock.stock_no != expected_stock_no:
            logging.warning(
                "Skipping stock identity mismatch at %s for slot %s: expected %s but got %s %s.",
                scheduled_time,
                item,
                expected_stock_no,
                twse_stock.stock_no,
                twse_stock.stock_name,
            )
            return False

        if expected_stock_no is None:
            self.tracked_stock_numbers[item] = twse_stock.stock_no

        if self.stocknumbers[item] and self.stocknumbers[item] != twse_stock.stock_no:
            logging.warning(
                "Skipping stock slot %s at %s because historical stock number %s would be replaced by %s.",
                item,
                scheduled_time,
                self.stocknumbers[item],
                twse_stock.stock_no,
            )
            return False

        if self.stocknames[item] and self.stocknames[item] != twse_stock.stock_name:
            logging.warning(
                "Stock %s name changed at %s from %s to %s; continuing by stock number.",
                twse_stock.stock_no,
                scheduled_time,
                self.stocknames[item],
                twse_stock.stock_name,
            )

        self.stocknumbers[item] = twse_stock.stock_no
        self.stocknames[item] = twse_stock.stock_name
        return True


    def parse_ohlc(self, row: StockRows, stock_index: int, scheduled_time: str) -> Ohlc | None:
        stock_row = self.get_stock_row(row, stock_index, scheduled_time)
        if stock_row is None:
            return None

        return self.parse_stock_row_ohlc(stock_row, scheduled_time)


    def get_date_times(
        self,
        begin_backtrack: int,
        end_backtrack: int,
        calendar: TwseCalendar | None = None,
        reference_date: datetime.date | None = None,
        include_reference: bool = False,
    ) -> None:
        self.iso_scheduled_times = []
        effective_reference_date = reference_date or datetime.datetime.now().date()

        for candidate in build_backtrack_dates(
            reference_date=effective_reference_date,
            begin_backtrack=begin_backtrack,
            end_backtrack=end_backtrack,
            include_reference=include_reference,
        ):
            status = (
                calendar.trading_day_status(candidate)
                if calendar is not None
                else TradingDayStatus.UNKNOWN
            )
            print(f"TWSE backtrack date {candidate.isoformat()}: {status.value.upper()}")

            if status is TradingDayStatus.CLOSED:
                continue
            if status is TradingDayStatus.UNKNOWN and candidate.isoweekday() in (6, 7):
                continue

            self.iso_scheduled_times.append(candidate.isoformat())

        if self.iso_scheduled_times:
            self.transactiondays = self.days_between_isodates(
                self.iso_scheduled_times[0],
                self.iso_scheduled_times[-1],
            )
        else:
            self.transactiondays = 0


    def days_between_isodates(self, date1: str, date2: str) -> int:
        def is_leap_year(year: int) -> bool:
            return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

        def get_days(date: str) -> int:
            y, m, d = map(int, date.split('-'))

            days = d + int(is_leap_year(y) and m < 2)
            days += sum(365 + int(is_leap_year(y)) for y in range(1971, y))
            days += sum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][:m])

            return days

        return abs(get_days(date1) - get_days(date2)) + 1


    def get_twse_daily_stocks(self, file_name: str, stocktype: Stocktype, stocks: list[StockSelector]) -> None:
        valid_iso_scheduled_times: list[str] = []
        previous_taiex_close: float | None = None
    
        # Crawing daily TWSE Stock data
        for iso_scheduled_time in self.iso_scheduled_times:

            scheduled_time = ''.join(iso_scheduled_time.split('-'))              
            # Get TWSE information of stock price
            row = self.twse_client.get_daily_stock_rows(scheduled_time, stocktype.value[0])
            if not row:
                logging.warning("Skipping %s because TWSE returned no stock rows.", scheduled_time)
                continue

            stock_lookup = self.build_stock_lookup(row, scheduled_time)
            if not stock_lookup:
                logging.warning("Skipping %s because no TWSE rows passed schema validation.", scheduled_time)
                continue

            taiex_close = self.twse_client.get_daily_taiex_close(scheduled_time)
            if taiex_close is None:
                logging.warning(
                    "Skipping %s because no valid TAIEX close was available for date alignment.",
                    scheduled_time,
                )
                continue

            breadth = derive_twse_market_breadth(
                row,
                as_of=iso_scheduled_time,
                universe=getattr(stocktype, "name", str(stocktype)),
                source_final=True,
                taiex_return=(
                    taiex_close / previous_taiex_close - 1
                    if previous_taiex_close not in (None, 0)
                    else None
                ),
            )
            previous_taiex_close = taiex_close
            self.market_breadth_snapshots_by_date[
                iso_scheduled_time
            ] = breadth.to_mapping()

            row_data: StockRows = [[] for _ in range(self.stocklistsize)]
    
            # Store stock data structure
            for item in range(self.stocklistsize):
                stock_row = self.resolve_stock_row(
                    rows = row,
                    stock_lookup = stock_lookup,
                    selector = stocks[item],
                    item = item,
                    scheduled_time = scheduled_time,
                )
                if stock_row is None:
                    continue
                
                twse_stock = self.parse_twse_stock(stock_row, scheduled_time)
                if twse_stock is None:
                    continue

                if not self.update_tracked_stock_identity(item, twse_stock, scheduled_time):
                    continue

                row_data[item] = self.build_record_row(stock_row)
                self.daily_stocks[item].append(twse_stock.open * 1000)
                self.daily_highs[item].append(twse_stock.high * 1000)
                self.daily_lows[item].append(twse_stock.low * 1000)
                self.daily_closes[item].append(twse_stock.close * 1000)
                self.daily_volumes[item].append(twse_stock.volume)
                self.daily_trade_counts[item].append(twse_stock.trades)
                self.daily_trade_values[item].append(twse_stock.amount)
                self.daily_bid_prices[item].append(twse_stock.bid * 1000 if twse_stock.bid is not None else None)
                self.daily_ask_prices[item].append(twse_stock.ask * 1000 if twse_stock.ask is not None else None)
                self.daily_pe_ratios[item].append(twse_stock.pe)
                self.daily_taiex_closes[item].append(taiex_close)
                self.daily_dates[item].append(iso_scheduled_time)
            
            # Record TWSE information of stock price
            if any(row_data):
                valid_iso_scheduled_times.append(iso_scheduled_time)
                self.record(file_name = file_name, scheduled_time = scheduled_time, row_data = row_data)
            else:
                logging.warning(
                    "Skipping CSV write for %s because none of the requested stocks had valid TWSE rows.",
                    scheduled_time,
                )

        if not valid_iso_scheduled_times:
            raise RuntimeError("No valid TWSE daily stock rows were collected for the requested date range.")

        self.iso_scheduled_times = valid_iso_scheduled_times
        self.transactiondays = self.days_between_isodates(self.iso_scheduled_times[0], self.iso_scheduled_times[-1])
