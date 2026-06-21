#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from __future__ import annotations

from interface.cli import create_parser

r"""
TWSE Real-time Stock Price Analysis

This script is designed to help streamline models, by taking the Taiwan Stock Exchange dataset. This will be used by an application and figuring out the maximum profit that are required to run for those arguments. The resulting is then sending to private email sandbox.
"""

import argparse
import html
import pandas as pd
import enum
import time
import datetime
import logging
from tabulate import tabulate

from domain.models import Stocktype
from domain.models import Ohlc
from domain.models import ProfitRow
from domain.models import SignalRow
from domain.models import StockSelector
from domain.models import StockRow
from domain.models import StockRows
from domain.models import TWSEStock
from domain.models import TwseColumns
from domain.services import compute_signal_features
from domain.services import max_profit
from domain.services import max_profit_k_transactions
from domain.services import max_profit_unlimited
from domain.services import max_profit_with_fee
from domain.strategy import get_strategy
from domain.strategies.low_entry_score_v3 import LOW_ENTRY_OUTPUT_COLUMNS
from domain.strategies.low_entry_score_v3 import LOW_ENTRY_STRATEGY_NAME
from infrastructure.report.html_renderer import RendererFactory
from infrastructure.crawler.twse_client import TwseClient
from infrastructure.notification.mail import SMTPEmail
from infrastructure.storage.chart_repository import save_profit_ratio_chart
from infrastructure.storage.csv_repository import CsvRepository
from twse.parser import clean_cell
from twse.parser import parse_price


class QUERY(enum.Enum):
    _order_ = 'NEOGENE PALEOGENE CRETACEOUS'
    NEOGENE = 12
    PALEOGENE = 15
    CRETACEOUS = 30

class TwseCrawker():
    _logged_missing_price_warnings: set[tuple[str, str, str, str, str, str]] = set()

    def __init__(
        self,
        stocklistsize: int,
        twse_client: TwseClient | None = None,
        csv_repository: CsvRepository | None = None,
    ) -> None:
        self.stocklistsize: int = stocklistsize
        self.twse_client = twse_client or TwseClient()
        self.csv_repository = csv_repository or CsvRepository()
        self.daily_stocks: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.stocknumbers: list[str] = ['' for _ in range(self.stocklistsize)]
        self.stocknames: list[str] = ['' for _ in range(self.stocklistsize)]
        self.tracked_stock_numbers: list[str | None] = [None for _ in range(self.stocklistsize)]
        self.stocksprice: list[float | None] = list()
        self.daily_highs: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_lows: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_closes: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_volumes: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_pe_ratios: list[list[float | None]] = [[] for _ in range(self.stocklistsize)]
        self.iso_scheduled_times: list[str] = list()
        self.transactiondays: int = 0


    def __del__(self) -> None:
        self.daily_stocks = [[] for _ in range(self.stocklistsize)]
        self.iso_scheduled_times = list()
        self.transactiondays = 0
        self.stocksprice = list()
        self.tracked_stock_numbers = [None for _ in range(self.stocklistsize)]
        self.daily_highs = [[] for _ in range(self.stocklistsize)]
        self.daily_lows = [[] for _ in range(self.stocklistsize)]
        self.daily_closes = [[] for _ in range(self.stocklistsize)]
        self.daily_volumes = [[] for _ in range(self.stocklistsize)]
        self.daily_pe_ratios = [[] for _ in range(self.stocklistsize)]


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
                logging.warning(
                    "Skipping tracked stock %s at %s because it is missing from the TWSE response.",
                    tracked_stock_no,
                    scheduled_time,
                )
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


    def get_date_times(self, start_date: int, backtrack_days: int, holidays: list[str]) -> None:

        now_date_time = datetime.datetime.now()

        for day in range(-backtrack_days, -start_date, 1):
            # ISO 8601 format, YYYY-MM-DD
            iso_date_time = (now_date_time + datetime.timedelta(days=day)).strftime("%Y-%m-%d")

            # Bypass weekend
            if (now_date_time + datetime.timedelta(days=day)).isoweekday() == 6 or (now_date_time + datetime.timedelta(days=day)).isoweekday() == 7:
                continue

            # TWSE Stock market haven't opened
            date_time = ''.join(iso_date_time.split('-'))
            if date_time in holidays:
                continue

            self.iso_scheduled_times.append(iso_date_time)

        self.transactiondays = self.days_between_isodates(self.iso_scheduled_times[0], self.iso_scheduled_times[-1])


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
                self.daily_pe_ratios[item].append(twse_stock.pe)
            
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


    def smtp_email(self, subject: str, ccreceiver: str | None, stocktype: Stocktype, maxprofits: list[ProfitRow]) -> None:

        analysis_dataset = self.build_analysis_dataset(maxprofits)
        stockprofittable = self.record_to_html_tablefmt(analysis_dataset)
        entryanalysis = self.build_entry_signal_analysis(analysis_dataset)

        smtpemail = SMTPEmail(subject, ccreceiver)
        smtpemail.smtpauthentication()

        # Send HTML email with Python
        smtpemail.textstockprofittable(iso_scheduled_times = self.iso_scheduled_times, 
                                     transactiondays = self.transactiondays,
                                     stocktype = stocktype,
                                     stockprofittable = stockprofittable,
                                     entryanalysis = entryanalysis)


    def smtp_img_email(self, subject: str, ccreceiver: str | None, stocktype: Stocktype, backtrack: str) -> None:
        smtpemail = SMTPEmail(subject, ccreceiver)
        smtpemail.smtpauthentication()

        # Send HTML email with Python
        smtpemail.imgstockprofittable(backtrack = backtrack, stocktype = stocktype)


    def draw_linechart(self, duration: int, maxprofitratios: list[list[float]]) -> None:
        save_profit_ratio_chart(
            duration = duration,
            maxprofitratios = maxprofitratios,
            stocknumbers = self.stocknumbers,
            stocksprice = self.stocksprice,
        )


    def cal_max_profit(self) -> list[ProfitRow]:
        maxprofits: list[ProfitRow] = [[] for _ in range(self.stocklistsize)]
    
        # Caculate max profit and stock profit's table
        for item, daily_stock in zip(range(self.stocklistsize), self.daily_stocks):
            if not daily_stock:
                logging.warning("Skipping max profit calculation for stock index %s because no valid prices were collected.", item)
                maxprofits[item].extend([0, 0, 0, 0, None])
                self.stocksprice.append(None)
                continue

            profit_with_fee = max_profit_with_fee(daily_stock, 300)
            maxprofits[item].append(round(max_profit_unlimited(daily_stock), 2))
            maxprofits[item].append(round(max_profit(daily_stock), 2))
            maxprofits[item].append(round(max_profit_k_transactions(5, daily_stock), 2))
            maxprofits[item].append(round(profit_with_fee, 2))

            # stock profit ratio
            maxprofits[item].append(round(profit_with_fee/daily_stock[-1], 2))

            # stock opening price
            self.stocksprice.append(round(daily_stock[-1], 2))

        return maxprofits


    def cal_signal_features(self) -> list[SignalRow]:
        signal_features: list[SignalRow] = [[] for _ in range(self.stocklistsize)]

        for item in range(self.stocklistsize):
            features = compute_signal_features(
                closes = self.daily_closes[item],
                highs = self.daily_highs[item],
                lows = self.daily_lows[item],
            )
            signal_features[item] = [
                features.trend_score,
                features.mom,
                features.cs,
                features.stop_loss,
                features.take_profit,
            ]

        return signal_features


    def build_strategy_history(self, item: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": self.daily_stocks[item],
                "High": self.daily_highs[item],
                "Low": self.daily_lows[item],
                "Close": self.daily_closes[item],
                "Volume": self.daily_volumes[item],
                "PE": self.daily_pe_ratios[item],
            }
        )


    def cal_low_entry_strategy(self) -> pd.DataFrame:
        strategy = get_strategy(LOW_ENTRY_STRATEGY_NAME)
        strategy_rows: list[dict[str, object]] = []

        for item in range(self.stocklistsize):
            result = strategy.run(self.build_strategy_history(item))
            latest = result.iloc[-1].to_dict() if not result.empty else {}
            strategy_rows.append({column: latest.get(column) for column in LOW_ENTRY_OUTPUT_COLUMNS})

        return pd.DataFrame(strategy_rows, columns = LOW_ENTRY_OUTPUT_COLUMNS)


    def build_analysis_dataset(self, maxprofits: list[ProfitRow]) -> pd.DataFrame:
        df_analysis = pd.DataFrame(maxprofits)
        df_analysis.columns = ["無限次交易", "交易一次", "至多五次交易", "無限次交易(手續費$NTD300)", "利潤比"]
        df_signal_features = pd.DataFrame(
            self.cal_signal_features(),
            columns = ["Trend Score", "MOM", "CS", "Stop_Loss", "Take_Profit"],
        )
        df_low_entry = self.cal_low_entry_strategy()
        df_analysis = pd.concat([df_analysis, df_signal_features, df_low_entry], axis = 1)
        df_analysis["日期"] = self.iso_scheduled_times[-1]
        df_analysis["證券代號"] = self.stocknumbers
        df_analysis["證券名稱"] = self.stocknames
        df_analysis["開盤"] = [stocks[-1] if stocks else None for stocks in self.daily_stocks]
        df_analysis["最高"] = [highs[-1] if highs else None for highs in self.daily_highs]
        df_analysis["最低"] = [lows[-1] if lows else None for lows in self.daily_lows]
        df_analysis["收盤"] = [closes[-1] if closes else None for closes in self.daily_closes]

        base_columns = [
            "日期",
            "證券代號",
            "證券名稱",
            "開盤",
            "最高",
            "最低",
            "收盤",
            "無限次交易(手續費$NTD300)",
            "利潤比",
            "Trend Score",
            "MOM",
            "CS",
            "Stop_Loss",
            "Take_Profit",
        ]

        return df_analysis[base_columns + LOW_ENTRY_OUTPUT_COLUMNS]


    def cal_max_profit_ratio_data(self) -> list[list[float]]:
        maxprofitratios: list[list[float]] = [[] for _ in range(self.stocklistsize)]

        # Caculate max profit and stock profit's table
        for item, daily_stock in zip(range(self.stocklistsize), self.daily_stocks):
            if not daily_stock:
                continue
            # stock profit ratio
            maxprofitratios[item].append(round(max_profit_with_fee(daily_stock, 300)/daily_stock[-1], 2))
             # stock opening price       
            self.stocksprice.append(round(daily_stock[-1], 2))

        logging.info('Max Profit Ratio: ', maxprofitratios)

        return maxprofitratios


    def record_to_html_tablefmt(self, analysis_dataset: pd.DataFrame) -> str:
        max_profit_columns = [
            "日期",
            "證券代號",
            "證券名稱",
            "開盤",
            "收盤",
            "無限次交易(手續費$NTD300)",
            "利潤比",
        ]
        stockprofittable = tabulate(
            analysis_dataset[max_profit_columns],
            headers = 'keys',
            tablefmt = 'html',
            showindex = False,
        )

        return stockprofittable


    def format_signal_value(self, value: object, precision: int = 4) -> str:
        if pd.isna(value):
            return "N/A"
        if isinstance(value, str):
            return html.escape(value)
        return str(round(float(value), precision))


    def build_entry_signal_analysis(self, analysis_dataset: pd.DataFrame) -> str:
        renderer = RendererFactory.get_renderer(LOW_ENTRY_STRATEGY_NAME)
        return renderer.render(analysis_dataset)


    def record_analysis_dataset(self, file_name: str, maxprofits: list[ProfitRow]) -> None:
        self.csv_repository.write_analysis_dataset(file_name, self.build_analysis_dataset(maxprofits))


    def record(self, file_name: str, scheduled_time: str, row_data: StockRows) -> None:
        if not any(row_data):
            logging.warning("Skipping CSV write for %s because row_data is empty.", scheduled_time)
            return
        self.csv_repository.write_daily_rows(file_name, scheduled_time, row_data)
