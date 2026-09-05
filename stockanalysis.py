#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
TWSE Real-time Stock Price Analysis

This script is designed to help streamline models, by taking the Taiwan Stock Exchange dataset. This will be used by an application and figuring out the maximum profit that are required to run for those arguments. The resulting is then sending to private email sandbox.
"""

from __future__ import annotations

import datetime
import logging

import pandas as pd

from application.analysis_dataset import AnalysisDatasetBuilder
from application.market_history_cache import CachedMarketDay
from application.market_history_cache import MarketHistoryCacheError
from application.market_history_cache import MarketHistoryCachePort
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
from infrastructure.crawler.twse_http_client import TwseMarketClosedError
from infrastructure.legacy_crawler_facade import CsvRepositoryPort
from infrastructure.legacy_crawler_facade import LegacyCrawlerFacade
from twse.parser import clean_cell
from twse.parser import parse_price


FULL_MARKET_TYPE = "ALLBUT0999"


class TwseCrawker(LegacyCrawlerFacade):
    def __init__(
        self,
        stocklistsize: int,
        twse_client: TwseClient | None = None,
        csv_repository: CsvRepositoryPort | None = None,
        http_client: TwseHttpClient | None = None,
        app_settings: object | None = None,
        analysis_dataset_builder: AnalysisDatasetBuilder | None = None,
        market_history_cache: MarketHistoryCachePort | None = None,
    ) -> None:
        if twse_client is not None and http_client is not None:
            raise ValueError("Pass either twse_client or http_client, not both")
        self.stocklistsize: int = stocklistsize
        self.twse_client = twse_client or TwseClient(http_client=http_client)
        self.market_history_cache = market_history_cache
        self._initialize_legacy_facade(
            csv_repository=csv_repository,
            app_settings=app_settings,
            analysis_dataset_builder=analysis_dataset_builder,
        )
        self.daily_stocks: list[list[float]] = [[] for _ in range(self.stocklistsize)]
        self.stocknumbers: list[str] = ['' for _ in range(self.stocklistsize)]
        self.stocknames: list[str] = ['' for _ in range(self.stocklistsize)]
        self.tracked_stock_numbers: list[str | None] = [None for _ in range(self.stocklistsize)]
        self.explicit_stock_numbers: set[str] = set()
        self.missing_tracked_stock_numbers: set[str] = set()
        self.missing_tracked_stock_first_dates: dict[str, str] = {}
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
        self._missing_ohlc_records: dict[
            tuple[str, str],
            tuple[object, object, object, object],
        ] = {}
        self.twse_no_data_dates: set[str] = set()
        self.twse_closed_unscheduled_dates: set[str] = set()
        self._market_history_refresh_dates: set[str] = set()


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
                self.missing_tracked_stock_first_dates.setdefault(
                    tracked_stock_no,
                    scheduled_time,
                )
                if tracked_stock_no not in self.missing_tracked_stock_numbers:
                    logging.debug(
                        "Skipping tracked stock %s starting at %s because it is missing from the TWSE response.",
                        tracked_stock_no,
                        scheduled_time,
                        extra={
                            "symbol": tracked_stock_no,
                            "trade_date": scheduled_time,
                            "source": "TWSE",
                            "status": "TRACKED_STOCK_MISSING",
                        },
                    )
                    self.missing_tracked_stock_numbers.add(tracked_stock_no)
                else:
                    logging.debug(
                        "Tracked stock %s remains missing from the TWSE response at %s.",
                        tracked_stock_no,
                        scheduled_time,
                        extra={
                            "symbol": tracked_stock_no,
                            "trade_date": scheduled_time,
                            "source": "TWSE",
                            "status": "TRACKED_STOCK_STILL_MISSING",
                        },
                    )
            elif tracked_stock_no in self.missing_tracked_stock_numbers:
                logging.debug(
                    "Tracked stock %s returned to the TWSE response at %s.",
                    tracked_stock_no,
                    scheduled_time,
                    extra={
                        "symbol": tracked_stock_no,
                        "trade_date": scheduled_time,
                        "source": "TWSE",
                        "status": "TRACKED_STOCK_RECOVERED",
                    },
                )
                self.missing_tracked_stock_numbers.remove(tracked_stock_no)
            return stock_row

        selector_text = str(selector).strip()
        stock_row = stock_lookup.get(selector_text)
        if stock_row is not None:
            if selector_text in self.tracked_stock_numbers:
                logging.debug(
                    "Skipping explicit stock %s at %s because it is already tracked by the selected industry universe.",
                    selector_text,
                    scheduled_time,
                )
                return None
            self.tracked_stock_numbers[item] = selector_text
            logging.debug("Tracking stock %s from stock-number selector at %s.", selector_text, scheduled_time)
            return stock_row

        if selector_text in self.explicit_stock_numbers:
            self.missing_tracked_stock_first_dates.setdefault(
                selector_text,
                scheduled_time,
            )
            logging.debug(
                "Skipping explicit portfolio stock %s at %s because it is missing from the TWSE full-market response.",
                selector_text,
                scheduled_time,
                extra={
                    "symbol": selector_text,
                    "trade_date": scheduled_time,
                    "source": "TWSE",
                    "status": "PORTFOLIO_STOCK_MISSING",
                },
            )
            return None

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
        if resolved_stock_no in self.tracked_stock_numbers:
            logging.warning(
                "Skipping legacy row index %s at %s because stock %s is already tracked by another slot.",
                stock_index,
                scheduled_time,
                resolved_stock_no,
            )
            return None
        self.tracked_stock_numbers[item] = resolved_stock_no
        logging.debug(
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
            self._missing_ohlc_records[(stock_no, scheduled_time)] = (
                stock_row[TwseColumns.OPEN],
                stock_row[TwseColumns.HIGH],
                stock_row[TwseColumns.LOW],
                stock_row[TwseColumns.CLOSE],
            )
            logging.debug(
                "Skipping stock %s at %s because OHLC data is unavailable: "
                "open=%s high=%s low=%s close=%s",
                stock_no,
                scheduled_time,
                stock_row[TwseColumns.OPEN],
                stock_row[TwseColumns.HIGH],
                stock_row[TwseColumns.LOW],
                stock_row[TwseColumns.CLOSE],
                extra={
                    "symbol": stock_no,
                    "trade_date": scheduled_time,
                    "source": "TWSE",
                    "status": "OHLC_MISSING",
                },
            )
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
        self._market_history_refresh_dates = set()
        effective_reference_date = reference_date or datetime.datetime.now().date()

        status_counts = {status: 0 for status in TradingDayStatus}
        candidates = build_backtrack_dates(
            reference_date=effective_reference_date,
            begin_backtrack=begin_backtrack,
            end_backtrack=end_backtrack,
            include_reference=include_reference,
        )
        for candidate in candidates:
            status = (
                calendar.trading_day_status(candidate)
                if calendar is not None
                else TradingDayStatus.UNKNOWN
            )
            status_counts[status] += 1
            logging.debug(
                "TWSE backtrack date %s: %s",
                candidate.isoformat(),
                status.value.upper(),
            )

            if status is TradingDayStatus.CLOSED:
                continue
            if status is TradingDayStatus.UNKNOWN and candidate.isoweekday() in (6, 7):
                continue

            self.iso_scheduled_times.append(candidate.isoformat())

        reference_iso = effective_reference_date.isoformat()
        if include_reference and reference_iso in self.iso_scheduled_times:
            self._market_history_refresh_dates.add(
                effective_reference_date.strftime("%Y%m%d")
            )

        logging.info(
            "TWSE backtrack calendar candidates=%s open=%s closed=%s "
            "unknown=%s selected=%s",
            len(candidates),
            status_counts[TradingDayStatus.OPEN],
            status_counts[TradingDayStatus.CLOSED],
            status_counts[TradingDayStatus.UNKNOWN],
            len(self.iso_scheduled_times),
        )

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
        self._missing_ohlc_records = {}
        self.twse_no_data_dates = set()
        self.twse_closed_unscheduled_dates = set()
        self.missing_tracked_stock_numbers = set()
        self.missing_tracked_stock_first_dates = {}
        market_type = str(stocktype.value[0])
        scheduled_dates = [
            iso_date.replace("-", "") for iso_date in self.iso_scheduled_times
        ]
        cache_days: dict[str, CachedMarketDay] = {}
        cache_usable = self.market_history_cache is not None
        cache_hits = 0
        cache_writes = 0
        cache_invalid = 0
        network_days = 0
        full_market_cache_days: dict[str, CachedMarketDay] = {}
        full_market_cache_usable = self.market_history_cache is not None
        full_market_cache_hits = 0
        full_market_cache_writes = 0
        full_market_network_days = 0
        if self.market_history_cache is not None:
            cache_query_dates = [
                trade_date
                for trade_date in scheduled_dates
                if trade_date not in self._market_history_refresh_dates
            ]
            try:
                cache_days = self.market_history_cache.load_many(
                    cache_query_dates,
                    market_type=market_type,
                )
            except MarketHistoryCacheError as error:
                cache_usable = False
                logging.warning(
                    "TWSE market history cache status=READ_ERROR; "
                    "falling back to HTTP for this run: %s",
                    error,
                    extra={
                        "source": "TWSE_MARKET_CACHE",
                        "status": "READ_ERROR",
                        "cache_hit": False,
                    },
                )
            if self.explicit_stock_numbers:
                try:
                    load_symbols_many = getattr(
                        self.market_history_cache,
                        "load_symbols_many",
                        None,
                    )
                    if callable(load_symbols_many):
                        full_market_cache_days = load_symbols_many(
                            cache_query_dates,
                            market_type=FULL_MARKET_TYPE,
                            symbols=self.explicit_stock_numbers,
                        )
                    else:
                        full_market_cache_days = (
                            self.market_history_cache.load_many(
                                cache_query_dates,
                                market_type=FULL_MARKET_TYPE,
                            )
                        )
                except MarketHistoryCacheError as error:
                    full_market_cache_usable = False
                    logging.warning(
                        "TWSE full-market history cache status=READ_ERROR; "
                        "falling back to HTTP for portfolio stocks: %s",
                        error,
                        extra={
                            "source": "TWSE_MARKET_CACHE",
                            "status": "READ_ERROR",
                            "cache_hit": False,
                        },
                    )
    
        # Crawing daily TWSE Stock data
        for iso_scheduled_time in self.iso_scheduled_times:

            scheduled_time = iso_scheduled_time.replace("-", "")
            cached_day = cache_days.get(scheduled_time)
            row: StockRows
            taiex_close: float
            stock_lookup: dict[str, StockRow]
            if cached_day is not None:
                row = cached_day.stock_rows
                taiex_close = cached_day.taiex_close
                stock_lookup = self.build_stock_lookup(row, scheduled_time)
                if stock_lookup:
                    cache_hits += 1
                else:
                    cached_day = None
                    cache_invalid += 1

            if cached_day is None:
                network_days += 1
                try:
                    row = self.twse_client.get_daily_stock_rows(
                        scheduled_time,
                        stocktype.value[0],
                    )
                except TwseMarketClosedError as error:
                    self.twse_closed_unscheduled_dates.add(iso_scheduled_time)
                    logging.debug(
                        "TWSE status=CLOSED_UNSCHEDULED date=%s detail=%s",
                        scheduled_time,
                        error,
                        extra={
                            "trade_date": scheduled_time,
                            "source": "TWSE",
                            "status": "CLOSED_UNSCHEDULED",
                        },
                    )
                    continue
                if not row:
                    self.twse_no_data_dates.add(iso_scheduled_time)
                    logging.debug(
                        "TWSE status=NO_DATA date=%s dataset=STOCK_TABLE "
                        "retries_exhausted=true; date skipped.",
                        scheduled_time,
                        extra={
                            "trade_date": scheduled_time,
                            "source": "TWSE",
                            "status": "NO_DATA",
                        },
                    )
                    continue

                stock_lookup = self.build_stock_lookup(row, scheduled_time)
                if not stock_lookup:
                    logging.warning(
                        "Skipping %s because no TWSE rows passed schema validation.",
                        scheduled_time,
                    )
                    continue

                loaded_taiex = self.twse_client.get_daily_taiex_close(
                    scheduled_time
                )
                if loaded_taiex is None:
                    self.twse_no_data_dates.add(iso_scheduled_time)
                    logging.debug(
                        "TWSE status=NO_DATA date=%s dataset=TAIEX "
                        "retries_exhausted=true; date skipped.",
                        scheduled_time,
                        extra={
                            "trade_date": scheduled_time,
                            "source": "TWSE",
                            "status": "NO_DATA",
                        },
                    )
                    continue
                taiex_close = loaded_taiex
                if cache_usable and self.market_history_cache is not None:
                    try:
                        self.market_history_cache.upsert_day(
                            CachedMarketDay(
                                trade_date=scheduled_time,
                                stock_rows=row,
                                taiex_close=taiex_close,
                            ),
                            market_type=market_type,
                        )
                        cache_writes += 1
                    except (MarketHistoryCacheError, ValueError) as error:
                        cache_usable = False
                        logging.warning(
                            "TWSE market history cache status=WRITE_ERROR; "
                            "cache writes disabled for this run: %s",
                            error,
                            extra={
                                "source": "TWSE_MARKET_CACHE",
                                "status": "WRITE_ERROR",
                                "cache_hit": False,
                            },
                        )

            missing_portfolio_symbols = (
                self.explicit_stock_numbers.difference(stock_lookup)
            )
            if missing_portfolio_symbols:
                full_market_day = full_market_cache_days.get(scheduled_time)
                full_market_rows: StockRows | None = None
                full_market_lookup: dict[str, StockRow] = {}
                if full_market_day is not None:
                    full_market_rows = full_market_day.stock_rows
                    full_market_lookup = self.build_stock_lookup(
                        full_market_rows,
                        scheduled_time,
                    )
                    if full_market_lookup:
                        full_market_cache_hits += 1
                    else:
                        full_market_day = None

                if full_market_day is None:
                    full_market_network_days += 1
                    full_market_rows = self.twse_client.get_daily_stock_rows(
                        scheduled_time,
                        FULL_MARKET_TYPE,
                    )
                    full_market_lookup = self.build_stock_lookup(
                        full_market_rows,
                        scheduled_time,
                    )
                    if (
                        full_market_cache_usable
                        and self.market_history_cache is not None
                        and full_market_lookup
                    ):
                        try:
                            self.market_history_cache.upsert_day(
                                CachedMarketDay(
                                    trade_date=scheduled_time,
                                    stock_rows=full_market_rows,
                                    taiex_close=taiex_close,
                                ),
                                market_type=FULL_MARKET_TYPE,
                            )
                            full_market_cache_writes += 1
                        except (MarketHistoryCacheError, ValueError) as error:
                            full_market_cache_usable = False
                            logging.warning(
                                "TWSE full-market history cache status=WRITE_ERROR; "
                                "cache writes disabled for this run: %s",
                                error,
                                extra={
                                    "source": "TWSE_MARKET_CACHE",
                                    "status": "WRITE_ERROR",
                                    "cache_hit": False,
                                },
                            )

                for symbol in missing_portfolio_symbols:
                    portfolio_row = full_market_lookup.get(symbol)
                    if portfolio_row is not None:
                        stock_lookup[symbol] = portfolio_row

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
                logging.debug(
                    "Skipping CSV write for %s because none of the requested stocks had valid TWSE rows.",
                    scheduled_time,
                    extra={
                        "trade_date": scheduled_time,
                        "source": "TWSE",
                        "status": "CSV_WRITE_SKIPPED",
                    },
                )

        if self.market_history_cache is not None:
            logging.info(
                "TWSE market history cache requested=%s hits=%s "
                "network_days=%s writes=%s invalid=%s refreshed=%s",
                len(scheduled_dates),
                cache_hits,
                network_days,
                cache_writes,
                cache_invalid,
                len(
                    self._market_history_refresh_dates.intersection(
                        scheduled_dates
                    )
                ),
                extra={
                    "source": "TWSE_MARKET_CACHE",
                    "status": "CACHE_SUMMARY",
                    "cache_hit": network_days == 0,
                },
            )
            if self.explicit_stock_numbers:
                logging.info(
                    "TWSE portfolio full-market cache requested=%s hits=%s "
                    "network_days=%s writes=%s symbols=%s",
                    len(scheduled_dates),
                    full_market_cache_hits,
                    full_market_network_days,
                    full_market_cache_writes,
                    len(self.explicit_stock_numbers),
                    extra={
                        "source": "TWSE_MARKET_CACHE",
                        "status": "PORTFOLIO_CACHE_SUMMARY",
                        "cache_hit": full_market_network_days == 0,
                    },
                )

        self._log_no_data_summary()
        self._log_unscheduled_closed_summary()
        self._log_missing_ohlc_summary()
        if not valid_iso_scheduled_times:
            raise RuntimeError("No valid TWSE daily stock rows were collected for the requested date range.")

        self.iso_scheduled_times = valid_iso_scheduled_times
        self.transactiondays = self.days_between_isodates(self.iso_scheduled_times[0], self.iso_scheduled_times[-1])


    def _log_no_data_summary(self) -> None:
        if not self.twse_no_data_dates:
            return

        dates = sorted(self.twse_no_data_dates)
        logging.warning(
            "TWSE NO_DATA summary dates=%s first_date=%s last_date=%s; "
            "dates were skipped. Use debug mode for per-date details.",
            len(dates),
            dates[0].replace("-", ""),
            dates[-1].replace("-", ""),
            extra={
                "source": "TWSE",
                "status": "NO_DATA_SUMMARY",
            },
        )


    def _log_unscheduled_closed_summary(self) -> None:
        if not self.twse_closed_unscheduled_dates:
            return

        dates = sorted(self.twse_closed_unscheduled_dates)
        logging.info(
            "TWSE CLOSED_UNSCHEDULED summary dates=%s first_date=%s "
            "last_date=%s; full-market probes confirmed no trading rows.",
            len(dates),
            dates[0].replace("-", ""),
            dates[-1].replace("-", ""),
            extra={
                "source": "TWSE",
                "status": "CLOSED_UNSCHEDULED_SUMMARY",
            },
        )


    def _log_missing_ohlc_summary(self) -> None:
        if not self._missing_ohlc_records:
            return

        symbols = {
            symbol for symbol, _ in self._missing_ohlc_records
        }
        dates = sorted(
            trade_date for _, trade_date in self._missing_ohlc_records
        )
        logging.warning(
            "TWSE OHLC unavailable summary records=%s stocks=%s "
            "first_date=%s last_date=%s; invalid rows were skipped. "
            "Use debug mode for per-record details.",
            len(self._missing_ohlc_records),
            len(symbols),
            dates[0],
            dates[-1],
            extra={
                "source": "TWSE",
                "status": "OHLC_MISSING_SUMMARY",
            },
        )
