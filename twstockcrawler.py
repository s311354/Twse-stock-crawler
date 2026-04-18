#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import os
import requests
import pandas as pd
import datetime
import json
import time
import logging
from typing import List, NamedTuple
from requests import Response
from requests import exceptions as requests_exceptions
from textmewhenitsdone import TextMeWhenItsDone

class AuthenticationServer(NamedTuple):
    subject: str
    email: str
    password: str
    receiver: str
    ccreceiver: str


class SMTPEmail(object):

    def __init__(self, subject = None, ccreceiver = None):
        self.authentication = None
        self.subject = subject
        self.email = os.getenv("TWSE_SMTP_EMAIL", "your-email-account@gmail")
        self.password = os.getenv("TWSE_SMTP_PASSWORD", "your-password")
        receivers = os.getenv("TWSE_SMTP_RECEIVERS", "receiver1,receiver2")
        self.receiver = [receiver.strip() for receiver in receivers.split(",") if receiver.strip()]
        self.ccreceiver = ccreceiver or os.getenv("TWSE_SMTP_CC", "ccreceiver")


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
            email != "your-email-account@gmail" and
            password != "your-password" and
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


    def imgstockprofittable(self, backtrack: str, stocktype: int) -> None:
        if not self.is_configured():
            logging.warning("Skipping email send because SMTP credentials are not configured.")
            return

        textmewhenitsdone = TextMeWhenItsDone(self.authentication.email)

        textmewhenitsdone.login(self.authentication.email, self.authentication.password)

        textmewhenitsdone.imgstockprofittableme( subject  = self.authentication.subject,
                                                 email    = self.authentication.email,
                                                 password = self.authentication.password,
                                                 stocktype = stocktype,
                                                 receiver = self.authentication.receiver,
                                                 ccreceiver = self.authentication.ccreceiver,
                                                 backtrack = backtrack)

        print(f"Sending Email at {datetime.datetime.now()}")


    def textstockprofittable(self, iso_scheduled_times: List[int], transactiondays: int, stocktype: int, stockprofittable) -> None:
        if not self.is_configured():
            logging.warning("Skipping email send because SMTP credentials are not configured.")
            return

        textmewhenitsdone = TextMeWhenItsDone(self.authentication.email)

        textmewhenitsdone.login(self.authentication.email, self.authentication.password)

        textmewhenitsdone.textstockprofittableme(subject  = self.authentication.subject,
                                                 email    = self.authentication.email,
                                                 password = self.authentication.password,
                                                 stocktype = stocktype,
                                                 stockprofittable = stockprofittable,
                                                 receiver = self.authentication.receiver,
                                                 ccreceiver = self.authentication.ccreceiver,
                                                 transactiondays = transactiondays,
                                                 iso_scheduled_times = iso_scheduled_times)

        print(f"Sending Email at {datetime.datetime.now()}")


class TwStockCrawler(object):
    _logged_insecure_ssl_fallback = False

    def __init__(self):
        self.url = 'https://www.twse.com.tw/exchangeReport/MI_INDEX'
        self.timeout = 30
        self.max_attempts = 2
        self.allow_insecure_twse_fallback = os.getenv("TWSE_ALLOW_INSECURE_SSL_FALLBACK", "1") != "0"


    def _should_retry_without_ssl_verification(self, error: requests_exceptions.SSLError) -> bool:
        return self.allow_insecure_twse_fallback and "Missing Subject Key Identifier" in str(error)


    def _request_stocktype_data(self, query_params: dict, date_time: str) -> Response:
        try:
            return requests.get(self.url, params = query_params, timeout = self.timeout)
        except requests_exceptions.SSLError as error:
            if not self._should_retry_without_ssl_verification(error):
                raise RuntimeError(
                    "TWSE SSL verification failed at {}. Set TWSE_ALLOW_INSECURE_SSL_FALLBACK=1 "
                    "to allow the compatibility fallback if you trust the network path.".format(date_time)
                ) from error

            if not TwStockCrawler._logged_insecure_ssl_fallback:
                logging.warning(
                    "TWSE SSL certificate verification failed at %s; retrying once without certificate verification.",
                    date_time,
                )
                TwStockCrawler._logged_insecure_ssl_fallback = True
            requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
            return requests.get(self.url, params = query_params, timeout = self.timeout, verify = False)
        except requests_exceptions.RequestException as error:
            raise RuntimeError("Unable to fetch TWSE stock data at {}: {}".format(date_time, error)) from error


    def _extract_stock_rows(self, content: dict, date_time: str) -> List[List[str]]:
        if 'data1' in content and content['data1']:
            return content['data1']

        for table in content.get('tables', []):
            data = table.get('data') if isinstance(table, dict) else None
            fields = table.get('fields') if isinstance(table, dict) else None
            if not data:
                continue
            if fields and "證券代號" not in fields:
                continue
            return data

        raise RuntimeError("TWSE response at {} does not contain stock data".format(date_time))


    def _get_content_keys(self, content: dict) -> List[str]:
        return sorted(content.keys()) if isinstance(content, dict) else []


    def get_stocktype_data(self, date_time: str, stocktype: int) -> List[List[str]]:
        row = list()

        query_params = {
            'date': date_time,
            'response': 'json',
            'type': stocktype,
            '_': str(round(time.time() * 1000) - 500)
        }

        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            # Get json data
            page = self._request_stocktype_data(query_params = query_params, date_time = date_time)

            print(date_time)
            if not page.ok:
                raise RuntimeError("TWSE returned HTTP {} at {}".format(page.status_code, date_time))

            try:
                content = page.json()
            except json.JSONDecodeError as error:
                raise RuntimeError("TWSE returned invalid JSON at {}".format(date_time)) from error

            try:
                for data in self._extract_stock_rows(content = content, date_time = date_time):
                    row.append(data)
                break
            except RuntimeError as error:
                last_error = error
                logging.warning(
                    "TWSE payload at %s had no stock rows on attempt %s/%s; keys=%s stat=%s",
                    date_time,
                    attempt,
                    self.max_attempts,
                    self._get_content_keys(content),
                    content.get('stat') if isinstance(content, dict) else None,
                )
                row = list()
                if attempt < self.max_attempts:
                    time.sleep(1)
                    continue
                raise RuntimeError(
                    "TWSE response at {} does not contain stock data after {} attempts".format(
                        date_time,
                        self.max_attempts,
                    )
                ) from last_error

        if not row:
            raise RuntimeError("TWSE returned an empty stock dataset at {}".format(date_time))
    
        logging.info('Date: {}, Stock number: {}, Stock price: {}'.format(date_time, row[0][0], row[0][5]))
        return row


    def maxProfit(self, prices: List[int]) -> int:
        dp_hold, dp_not_hold = -float('inf'), 0
        
        for stock_price in prices:
            dp_not_hold = max(dp_not_hold, dp_hold + stock_price)
            dp_hold = max (dp_hold, - stock_price)

        return dp_not_hold


    def maxProfitII(self, prices: List[int]) -> int:
        total = 0

        for i in range(1, len(prices)):
            profit = prices[i] - prices[i-1]
            if profit > 0:
                total += profit

        return total


    def maxProfitIV(self, k: int, prices: List[int]) -> int:
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


    def maxProfitwithfee(self, prices: List[int], fee: int) -> int:
        dp_hold, dp_not_hold = -float('inf'), 0

        for stock_price in prices:
           dp_not_hold = max(dp_not_hold, dp_hold + stock_price)
           dp_hold = max(dp_hold, dp_not_hold - stock_price - fee)

        return dp_not_hold if prices else 0


    def minimumUpLines(self, stockPrices: List[List[int]]) -> int:
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
