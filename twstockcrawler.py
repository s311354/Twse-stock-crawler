#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import time
import logging
from collections.abc import Callable
from typing import Any, NamedTuple, TypeAlias
from config.settings import AppSettings
from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.crawler.twse_http_client import TwseMarketClosedError
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseNoDataError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.observability import structured_extra
from textmewhenitsdone import TextMeWhenItsDone
from twse.parser import clean_cell, parse_price

StockRow: TypeAlias = list[str]
StockRows: TypeAlias = list[StockRow]
JsonPayload: TypeAlias = dict[str, Any]


class AuthenticationServer(NamedTuple):
    subject: str
    email: str
    password: str
    receiver: list[str]
    ccreceiver: str


class SMTPEmail(object):

    def __init__(
        self,
        subject: str | None = None,
        ccreceiver: str | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or AppSettings.from_env(
            mops_enabled=False,
            validate_paths=False,
        )
        self.authentication: AuthenticationServer | None = None
        self.subject = subject
        self.email = self.settings.smtp_email
        self.password = self.settings.smtp_password
        self.receiver = list(self.settings.smtp_receivers)
        self.ccreceiver = (
            self.settings.smtp_cc if ccreceiver is None else ccreceiver
        )


    def prompt(self, prompt: str) -> str:
        return input(prompt).strip()


    def is_configured(self) -> bool:
        if not self.authentication:
            return False

        email = self.authentication.email
        password = self.authentication.password

        if "@" not in email or "." not in email.split("@", 1)[1]:
            return False

        return (
            bool(email) and
            bool(password) and
            bool(self.receiver)
        )


    def smtpauthentication(self) -> None:
        if not self.subject and not self.ccreceiver:
            self.authentication = AuthenticationServer(email = self.email,
                                                       password = self.password,
                                                       subject = self.prompt("Subject: ").split()[0],
                                                       receiver = self.receiver,
                                                       ccreceiver = self.prompt("Cc: ").split()[0])
        else:
            self.authentication = AuthenticationServer(email = self.email,
                                                       password = self.password,
                                                       subject = self.subject,
                                                       receiver = self.receiver,
                                                       ccreceiver = self.ccreceiver)


    def imgstockprofittable(self, backtrack: str, stocktype: object) -> None:
        if not self.is_configured():
            logging.warning("Skipping email send because SMTP credentials are not configured.")
            return

        textmewhenitsdone = TextMeWhenItsDone(
            self.authentication.email,
            smtp_host=self.settings.smtp_server_host,
            smtp_port=self.settings.smtp_port,
        )

        textmewhenitsdone.login(self.authentication.email, self.authentication.password)

        textmewhenitsdone.imgstockprofittableme( subject  = self.authentication.subject,
                                                 email    = self.authentication.email,
                                                 password = self.authentication.password,
                                                 stocktype = stocktype,
                                                 receiver = self.authentication.receiver,
                                                 ccreceiver = self.authentication.ccreceiver,
                                                 backtrack = backtrack)

        print(f"Sending Email at {datetime.datetime.now()}")


    def textstockprofittable(
        self,
        iso_scheduled_times: list[str],
        transactiondays: int,
        stocktype: object,
        stockprofittable: str,
        entryanalysis: str = "",
    ) -> None:
        if not self.is_configured():
            logging.warning("Skipping email send because SMTP credentials are not configured.")
            return

        textmewhenitsdone = TextMeWhenItsDone(
            self.authentication.email,
            smtp_host=self.settings.smtp_server_host,
            smtp_port=self.settings.smtp_port,
        )

        textmewhenitsdone.login(self.authentication.email, self.authentication.password)

        textmewhenitsdone.textstockprofittableme(subject  = self.authentication.subject,
                                                 email    = self.authentication.email,
                                                 password = self.authentication.password,
                                                 stocktype = stocktype,
                                                 stockprofittable = stockprofittable,
                                                 receiver = self.authentication.receiver,
                                                 ccreceiver = self.authentication.ccreceiver,
                                                 transactiondays = transactiondays,
                                                 iso_scheduled_times = iso_scheduled_times,
                                                 entryanalysis = entryanalysis)

        print(f"Sending Email at {datetime.datetime.now()}")


class TwStockCrawler(object):
    TAIEX_INDEX_NAME = "發行量加權股價指數"
    FULL_MARKET_TYPE = "ALLBUT0999"
    NO_DATA_MARKERS = ("沒有符合條件", "查無資料", "無資料")
    MARKET_DATA_FALLBACK_URL = (
        "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
    )

    def __init__(self, http_client: TwseHttpClient | None = None) -> None:
        self.url: str = 'https://www.twse.com.tw/exchangeReport/MI_INDEX'
        self.fallback_url: str = self.MARKET_DATA_FALLBACK_URL
        self.http_client = http_client or TwseHttpClient()
        self.timeout = self.http_client.config.timeout_seconds
        self.max_attempts = self.http_client.config.max_attempts


    def _request_stocktype_data(
        self,
        query_params: dict[str, str | int],
        date_time: str,
        payload_validator: Callable[[object], None] | None = None,
    ) -> object:
        endpoints = tuple(dict.fromkeys((self.url, self.fallback_url)))
        for endpoint_index, endpoint in enumerate(endpoints):
            try:
                return self.http_client.get_json(
                    endpoint,
                    params=query_params,
                    context="TWSE market data at {}".format(date_time),
                    payload_validator=payload_validator,
                ).payload
            except TwseNetworkError as error:
                if endpoint_index == len(endpoints) - 1:
                    logging.error(
                        "TWSE status=NETWORK_ERROR date=%s endpoint=%s "
                        "final_failure=true detail=%s",
                        date_time,
                        endpoint,
                        error,
                        extra=structured_extra(
                            trade_date=date_time,
                            source="TWSE",
                            status="NETWORK_ERROR",
                        ),
                    )
                    raise
                logging.debug(
                    "TWSE status=NETWORK_ERROR date=%s endpoint=%s "
                    "failover_endpoint=%s detail=%s",
                    date_time,
                    endpoint,
                    endpoints[endpoint_index + 1],
                    error,
                    extra=structured_extra(
                        trade_date=date_time,
                        source="TWSE",
                        status="ENDPOINT_FAILOVER",
                    ),
                )

        raise AssertionError("TWSE market-data endpoint list cannot be empty")

    def _raise_if_unscheduled_market_closed(self, date_time: str) -> None:
        """Confirm an empty sector table against the full TWSE market."""

        query_params = {
            "date": date_time,
            "response": "json",
            "type": self.FULL_MARKET_TYPE,
            "_": str(round(time.time() * 1000) - 500),
        }

        def validate_full_market(payload: object) -> None:
            if not isinstance(payload, dict):
                raise TwseSchemaChangedError(
                    "TWSE full-market data at {} is not an object".format(
                        date_time
                    )
                )
            status = payload.get("stat")
            if isinstance(status, str) and any(
                marker in status for marker in self.NO_DATA_MARKERS
            ):
                raise TwseMarketClosedError(
                    "TWSE full market at {} explicitly returned no data".format(
                        date_time
                    )
                )
            try:
                self._extract_stock_rows(payload, date_time)
            except TwseNoDataError as error:
                raise TwseMarketClosedError(
                    "TWSE full market at {} contains no trading rows".format(
                        date_time
                    )
                ) from error

        self._request_stocktype_data(
            query_params=query_params,
            date_time=date_time,
            payload_validator=validate_full_market,
        )


    def _extract_stock_rows(self, content: JsonPayload, date_time: str) -> StockRows:
        if 'data1' in content:
            data1 = content['data1']
            if isinstance(data1, list) and data1:
                return data1
            if data1 == []:
                raise TwseNoDataError(
                    "TWSE stock table at {} is empty".format(date_time),
                    retryable=True,
                )
            raise TwseSchemaChangedError(
                "TWSE data1 at {} is not a list".format(date_time)
            )

        found_stock_table = False
        for table in content.get('tables', []):
            data = table.get('data') if isinstance(table, dict) else None
            fields = table.get('fields') if isinstance(table, dict) else None
            if not isinstance(fields, list) or "證券代號" not in fields:
                continue
            found_stock_table = True
            if data == []:
                raise TwseNoDataError(
                    "TWSE stock table at {} is empty".format(date_time),
                    retryable=True,
                )
            if not isinstance(data, list):
                raise TwseSchemaChangedError(
                    "TWSE stock table data at {} is not a list".format(date_time)
                )
            return data

        if found_stock_table:
            raise TwseNoDataError(
                "TWSE response at {} contains no stock rows".format(date_time),
                retryable=True,
            )
        raise TwseSchemaChangedError(
            "TWSE response at {} does not contain a stock table".format(date_time)
        )


    def _extract_taiex_close(self, content: JsonPayload, date_time: str) -> float:
        """Read TAIEX by schema labels instead of table or row positions."""
        found_index_table = False
        for table in content.get("tables", []):
            if not isinstance(table, dict):
                continue

            fields = table.get("fields")
            data = table.get("data")
            if not isinstance(fields, list) or not isinstance(data, list):
                continue

            index_field = next((field for field in ("指數", "指數名稱") if field in fields), None)
            close_field = next((field for field in ("收盤指數", "收盤價") if field in fields), None)
            if index_field is None or close_field is None:
                continue
            found_index_table = True

            index_column = fields.index(index_field)
            close_column = fields.index(close_field)
            required_width = max(index_column, close_column) + 1

            for row in data:
                if not isinstance(row, list) or len(row) < required_width:
                    continue
                if clean_cell(row[index_column]) != self.TAIEX_INDEX_NAME:
                    continue

                taiex_close = parse_price(row[close_column])
                if taiex_close is None or taiex_close <= 0:
                    raise TwseNoDataError(
                        "TWSE TAIEX close at {} is invalid: {}".format(date_time, row[close_column]),
                        retryable=True,
                    )
                return taiex_close

        if found_index_table:
            raise TwseNoDataError(
                "TWSE response at {} contains no {}".format(date_time, self.TAIEX_INDEX_NAME),
                retryable=True,
            )
        raise TwseSchemaChangedError(
            "TWSE response at {} does not contain an index table".format(date_time)
        )


    def _get_content_keys(self, content: object) -> list[str]:
        return sorted(content.keys()) if isinstance(content, dict) else []


    def get_stocktype_data(self, date_time: str, stocktype: int) -> StockRows:
        query_params = {
            'date': date_time,
            'response': 'json',
            'type': stocktype,
            '_': str(round(time.time() * 1000) - 500)
        }

        def validate_stock_payload(payload: object) -> None:
            if not isinstance(payload, dict):
                raise TwseSchemaChangedError(
                    "TWSE market data at {} is not an object".format(date_time)
                )
            self._extract_stock_rows(content=payload, date_time=date_time)

        try:
            content = self._request_stocktype_data(
                query_params=query_params,
                date_time=date_time,
                payload_validator=validate_stock_payload,
            )
        except TwseNoDataError as error:
            if str(stocktype) == self.FULL_MARKET_TYPE:
                raise TwseMarketClosedError(
                    "TWSE full market at {} contains no trading rows".format(
                        date_time
                    )
                ) from error
            self._raise_if_unscheduled_market_closed(date_time)
            logging.debug(
                "TWSE status=NO_DATA date=%s detail=%s",
                date_time,
                error,
                extra={
                    "trade_date": date_time,
                    "source": "TWSE",
                    "status": "NO_DATA",
                },
            )
            return []

        logging.debug("TWSE market data date=%s", date_time)
        if not isinstance(content, dict):
            raise TwseSchemaChangedError(
                "TWSE market data at {} is not an object".format(date_time)
            )
        try:
            row = self._extract_stock_rows(content=content, date_time=date_time)
        except TwseNoDataError as error:
            logging.debug(
                "TWSE status=NO_DATA date=%s keys=%s stat=%s detail=%s",
                date_time,
                self._get_content_keys(content),
                content.get('stat'),
                error,
                extra={
                    "trade_date": date_time,
                    "source": "TWSE",
                    "status": "NO_DATA",
                },
            )
            return []

        logging.debug(
            "TWSE sample row date=%s stock_number=%s open=%s",
            date_time,
            row[0][0],
            row[0][5],
        )
        return row


    def get_taiex_close(self, date_time: str) -> float | None:
        """Fetch one TAIEX close for a trading day from the index-only payload."""
        query_params = {
            "date": date_time,
            "response": "json",
            "type": "IND",
            "_": str(round(time.time() * 1000) - 500),
        }

        def validate_taiex_payload(payload: object) -> None:
            if not isinstance(payload, dict):
                raise TwseSchemaChangedError(
                    "TWSE TAIEX data at {} is not an object".format(date_time)
                )
            self._extract_taiex_close(content=payload, date_time=date_time)

        try:
            content = self._request_stocktype_data(
                query_params=query_params,
                date_time=date_time,
                payload_validator=validate_taiex_payload,
            )
        except TwseNoDataError as error:
            logging.debug(
                "TWSE status=NO_DATA date=%s dataset=TAIEX detail=%s",
                date_time,
                error,
                extra={
                    "trade_date": date_time,
                    "source": "TWSE",
                    "status": "NO_DATA",
                },
            )
            return None

        if not isinstance(content, dict):
            raise TwseSchemaChangedError(
                "TWSE TAIEX data at {} is not an object".format(date_time)
            )

        try:
            return self._extract_taiex_close(content=content, date_time=date_time)
        except TwseNoDataError as error:
            logging.debug(
                "TWSE status=NO_DATA date=%s dataset=TAIEX detail=%s",
                date_time,
                error,
                extra={
                    "trade_date": date_time,
                    "source": "TWSE",
                    "status": "NO_DATA",
                },
            )
            return None


    def maxProfit(self, prices: list[int]) -> int:
        dp_hold, dp_not_hold = -float('inf'), 0
        
        for stock_price in prices:
            dp_not_hold = max(dp_not_hold, dp_hold + stock_price)
            dp_hold = max (dp_hold, - stock_price)

        return dp_not_hold


    def maxProfitII(self, prices: list[int]) -> int:
        total = 0

        for i in range(1, len(prices)):
            profit = prices[i] - prices[i-1]
            if profit > 0:
                total += profit

        return total


    def maxProfitIV(self, k: int, prices: list[int]) -> int:
        n = len(prices)

        if n == 0:
            return 0

        dp = [[0 for _ in range(n)] for _ in range(k+1)]

        for trans_k in range(1, k + 1):
            cur_balance_with_buy = 0 - prices[0]
            for day_d in range(1, n):
                dp[trans_k][day_d] = max(dp[trans_k][day_d - 1], cur_balance_with_buy + prices[day_d])
                cur_balance_with_buy = max(cur_balance_with_buy, dp[trans_k - 1][day_d - 1] - prices[day_d])

        return dp[k][n-1]


    def maxProfitwithfee(self, prices: list[int], fee: int) -> int:
        dp_hold, dp_not_hold = -float('inf'), 0

        for stock_price in prices:
           dp_not_hold = max(dp_not_hold, dp_hold + stock_price)
           dp_hold = max(dp_hold, dp_not_hold - stock_price - fee)

        return dp_not_hold if prices else 0


    def minimumUpLines(self, stockPrices: list[list[int]]) -> int:
        if len(stockPrices) == 1:
            return 0
        if len(stockPrices) == 2:
            return 1

        n = len(stockPrices)
        lines = 0

        for i in range(1, n-1):
            a, b, c = stockPrices[i-1], stockPrices[i], stockPrices[i+1]

            if ((b[0] - a[0]) * (c[1] - b[1]) <= (c[0] - b[0]) * (b[1] - a[1])):
                lines += 1

        return lines
