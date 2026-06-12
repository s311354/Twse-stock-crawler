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
from domain.models import StockRow
from domain.models import StockRows
from domain.services import compute_signal_features
from domain.services import max_profit
from domain.services import max_profit_k_transactions
from domain.services import max_profit_unlimited
from domain.services import max_profit_with_fee
from domain.strategy import evaluate_low_entry
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
        self.stocksprice: list[float | None] = list()
        self.daily_highs: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_lows: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.daily_closes: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.iso_scheduled_times: list[str] = list()
        self.transactiondays: int = 0


    def __del__(self) -> None:
        self.daily_stocks = [[] for _ in range(self.stocklistsize)]
        self.iso_scheduled_times = list()
        self.transactiondays = 0
        self.stocksprice = list()
        self.daily_highs = [[] for _ in range(self.stocklistsize)]
        self.daily_lows = [[] for _ in range(self.stocklistsize)]
        self.daily_closes = [[] for _ in range(self.stocklistsize)]


    def clean_data(self, row: StockRow) -> StockRow:
        for index, content in enumerate(row):
            row[index] = clean_cell(content)
        return row


    def parse_price(self, price: object) -> float | None:
        return parse_price(price)


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
        if not isinstance(stock_row, list) or len(stock_row) <= 8:
            logging.warning(
                "Skipping stock index %s at %s because TWSE row shape is invalid: %s",
                stock_index,
                scheduled_time,
                stock_row,
            )
            return None

        return stock_row


    def build_record_row(self, stock_row: StockRow) -> StockRow:
        return self.clean_data([
                stock_row[0], # Stock number
                stock_row[1], # Stock name
                stock_row[5], # Stock opening price
                stock_row[6], # Stock price high
                stock_row[7], # Stock price low
                stock_row[8], # Stock close price
                ])


    def parse_stock_row_ohlc(self, stock_row: StockRow, scheduled_time: str) -> Ohlc | None:
        open_price = self.parse_price(stock_row[5])
        high_price = self.parse_price(stock_row[6])
        low_price = self.parse_price(stock_row[7])
        close_price = self.parse_price(stock_row[8])

        if None in (open_price, high_price, low_price, close_price):
            warning_key = (stock_row[0], scheduled_time, stock_row[5], stock_row[6], stock_row[7], stock_row[8])
            if warning_key not in TwseCrawker._logged_missing_price_warnings:
                logging.warning(
                    "Skipping stock %s at %s because OHLC data is unavailable: open=%s high=%s low=%s close=%s",
                    stock_row[0],
                    scheduled_time,
                    stock_row[5],
                    stock_row[6],
                    stock_row[7],
                    stock_row[8],
                )
                TwseCrawker._logged_missing_price_warnings.add(warning_key)
            return None

        return open_price, high_price, low_price, close_price


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


    def get_twse_daily_stocks(self, file_name: str, stocktype: Stocktype, stocks: list[int]) -> None:
        valid_iso_scheduled_times: list[str] = []
    
        # Crawing daily TWSE Stock data
        for iso_scheduled_time in self.iso_scheduled_times:

            scheduled_time = ''.join(iso_scheduled_time.split('-'))              
            # Get TWSE information of stock price
            row = self.twse_client.get_daily_stock_rows(scheduled_time, stocktype.value[0])
            if not row:
                logging.warning("Skipping %s because TWSE returned no stock rows.", scheduled_time)
                continue

            valid_iso_scheduled_times.append(iso_scheduled_time)
            row_data: StockRows = [[] for _ in range(self.stocklistsize)]
            missing_index_count = sum(1 for stock_index in stocks if stock_index >= len(row))
            if missing_index_count:
                logging.warning(
                    "TWSE returned %s rows at %s; skipping %s requested stock indexes outside the response.",
                    len(row),
                    scheduled_time,
                    missing_index_count,
                )
    
            # Store stock data structure
            for item in range(self.stocklistsize):
                # Transfer to NTD Price
                # sign = '-' if row[stocks[item][9]].find('green') > 0 else ''
                if stocks[item] >= len(row):
                    continue

                stock_row = self.get_stock_row(row, stocks[item], scheduled_time)
                if stock_row is None:
                    continue

                self.stocknumbers[item] = stock_row[0]
                self.stocknames[item] = stock_row[1]
                row_data[item] = self.build_record_row(stock_row)

                ohlc = self.parse_stock_row_ohlc(stock_row, scheduled_time)
                if ohlc is not None:
                    opening_price, high_price, low_price, close_price = ohlc
                    self.daily_stocks[item].append(opening_price * 1000)
                    self.daily_highs[item].append(high_price * 1000)
                    self.daily_lows[item].append(low_price * 1000)
                    self.daily_closes[item].append(close_price * 1000)
            
            # Record TWSE information of stock price
            self.record(file_name = file_name, scheduled_time = scheduled_time, row_data = row_data)

        if not valid_iso_scheduled_times:
            raise RuntimeError("No valid TWSE daily stock rows were collected for the requested date range.")

        self.iso_scheduled_times = valid_iso_scheduled_times
        self.transactiondays = self.days_between_isodates(self.iso_scheduled_times[0], self.iso_scheduled_times[-1])


    def smtp_email(self, subject: str, ccreceiver: str | None, stocktype: Stocktype, maxprofits: list[ProfitRow]) -> None:

        analysis_dataset = self.build_analysis_dataset(maxprofits)
        stockprofittable = self.record_to_html_tablefmt(maxprofits)
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


    def build_analysis_dataset(self, maxprofits: list[ProfitRow]) -> pd.DataFrame:
        df_analysis = pd.DataFrame(maxprofits)
        df_analysis.columns = ["無限次交易", "交易一次", "至多五次交易", "無限次交易(手續費$NTD300)", "利潤比"]
        df_signal_features = pd.DataFrame(
            self.cal_signal_features(),
            columns = ["Trend Score", "MOM", "CS", "Stop_Loss", "Take_Profit"],
        )
        df_analysis = pd.concat([df_analysis, df_signal_features], axis = 1)
        df_analysis["日期"] = self.iso_scheduled_times[-1]
        df_analysis["證券代號"] = self.stocknumbers
        df_analysis["證券名稱"] = self.stocknames
        df_analysis["開盤"] = [stocks[-1] if stocks else None for stocks in self.daily_stocks]
        df_analysis["最高"] = [highs[-1] if highs else None for highs in self.daily_highs]
        df_analysis["最低"] = [lows[-1] if lows else None for lows in self.daily_lows]
        df_analysis["收盤"] = [closes[-1] if closes else None for closes in self.daily_closes]

        return df_analysis[
            [
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
        ]


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


    def record_to_html_tablefmt(self, maxprofits: list[ProfitRow]) -> str:
        df_analysis = self.build_analysis_dataset(maxprofits)
        stockprofittable = tabulate(df_analysis, headers = 'keys', tablefmt = 'html', showindex = False)

        return stockprofittable


    def format_signal_value(self, value: object, precision: int = 4) -> str:
        if pd.isna(value):
            return "N/A"
        if isinstance(value, str):
            return html.escape(value)
        return str(round(float(value), precision))


    def build_entry_signal_analysis(self, analysis_dataset: pd.DataFrame) -> str:
        rows = []

        for _, stock in analysis_dataset.iterrows():
            stock_no = html.escape(str(stock["證券代號"]))
            stock_name = html.escape(str(stock["證券名稱"]))
            current_price = stock["收盤"]
            stop_loss = stock["Stop_Loss"]
            take_profit = stock["Take_Profit"]
            high_price = stock["最高"]
            low_price = stock["最低"]
            mom = stock["MOM"]
            cs = stock["CS"]

            low_entry = evaluate_low_entry(
                current_price = current_price if pd.notna(current_price) else None,
                high_price = high_price if pd.notna(high_price) else None,
                low_price = low_price if pd.notna(low_price) else None,
                stop_loss = stop_loss if pd.notna(stop_loss) else None,
                take_profit = take_profit if pd.notna(take_profit) else None,
                mom = mom if pd.notna(mom) else None,
                cs = cs if pd.notna(cs) else None,
            )

            rows.append([
                "{} {}".format(stock_no, stock_name),
                self.format_signal_value(current_price, 2),
                self.format_signal_value(low_entry.position_score, 2),
                self.format_signal_value(low_entry.rr, 2),
                self.format_signal_value(mom, 2),
                self.format_signal_value(cs, 2),
                low_entry.decision,
                low_entry.reason,
            ])

        df_entry = pd.DataFrame(
            rows,
            columns = [
                "股票",
                "現價",
                "Position Score",
                "RR",
                "MOM",
                "CS",
                "決策",
                "理由",
            ],
        )

        return """
        <h3>低點進場判斷（Low-Entry Model）</h3>
        <p><strong>BUY 條件</strong>：Position Score &lt;= 0.3、MOM &gt; 0、CS &gt;= 0.2、RR &gt;= 1.5。</p>
        <p><strong>Position Score</strong> = (收盤 - 最低) / (最高 - 最低)；<strong>RR</strong> = (Take_Profit - 收盤) / (收盤 - Stop_Loss)。若風險分母接近 0，標記為無效計算並自動 NO BUY。</p>
        {}
        """.format(tabulate(df_entry, headers = 'keys', tablefmt = 'html', showindex = False))


    def record_analysis_dataset(self, file_name: str, maxprofits: list[ProfitRow]) -> None:
        self.csv_repository.write_analysis_dataset(file_name, self.build_analysis_dataset(maxprofits))


    def record(self, file_name: str, scheduled_time: str, row_data: StockRows) -> None:
        self.csv_repository.write_daily_rows(file_name, scheduled_time, row_data)