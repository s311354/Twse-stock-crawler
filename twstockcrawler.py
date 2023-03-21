#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import requests
import pandas as pd
import datetime
import json
import time
import logging
from typing import List, NamedTuple
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
        self.email = "your-email-account@gmail"
        self.receiver = ["receiver1", "receiver2"]
        self.ccreceiver = "ccreceiver"


    def prompt(self, prompt: str) -> str:
        return input(prompt).strip()


    def smtpauthentication(self) -> None:
        if not self.subject and not self.ccreceiver:
            self.authentication = AuthenticationServer(email = self.email,
                                                       password = "your-password",
                                                       subject = self.prompt("Subject: ").split()[0],
                                                       receiver = self.receiver,
                                                       ccreceiver = self.prompt("Cc: ").split()[0])
        else:
            self.authentication = AuthenticationServer(email = self.email,
                                                       password = "your-password",
                                                       subject = self.subject,
                                                       receiver = self.receiver,
                                                       ccreceiver = self.ccreceiver)


    def imgstockprofittable(self, backtrack: str, stocktype: int) -> None:
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

    def __init__(self):
        self.url = 'https://www.twse.com.tw/exchangeReport/MI_INDEX'


    def get_stocktype_data(self, date_time: str, stocktype: int) -> List[List[str]]:
        row = list()

        query_params = {
            'date': date_time,
            'response': 'json',
            'type': stocktype,
            '_': str(round(time.time() * 1000) - 500)
        }

        # Get json data
        page = requests.get(self.url, params = query_params)

        print(date_time)
        if not page.ok:
            logging.error("Can not get TWSE stock data at {}".format(date_time))

        content = page.json()

        for data in content['data1']:
            row.append(data) 
    
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
