#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
TWSE Real-time Stock Price Analysis

This script is designed to help streamline models, by taking the Taiwan Stock Exchange dataset. This will be used by an application and figuring out the maximum profit that are required to run for those arguments. The resulting is then sending to private email sandbox.

An example of command-line usage to analyze max profit table is:
python stockanalysis.py -t ELEC -s I \ 
                        -o shirong \
                        -e 7 -b 0 \
                        -m \
                        ./stocklist_elec ./holidays_2023

An example of command-line usage to analyze max profit ratio is:
python stockanalysis.py -t ELEC  -s I \
                        -o shirong \
                        -e 35 -b 0 -p 7 \
                        -l -m  \
                        ./stocklist_elec ./holidays_2023
"""

import os
import re
import csv
import argparse
import pandas as pd
import enum
import time
import datetime
import logging
import matplotlib.pyplot as plt
from typing import List, NamedTuple
from tabulate import tabulate

from twstockcrawler import SMTPEmail
from twstockcrawler import TwStockCrawler

class Stocktype(enum.Enum):
    VEH  = 12,
    ELEC = 13,
    SEMI = 14,
    AIR  = 15,
    BIO  = 22,
    COMM = 27

    def __str__(self):
        return self.name

class QUERY(enum.Enum):
    _order_ = 'NEOGENE PALEOGENE CRETACEOUS'
    NEOGENE = 12
    PALEOGENE = 15
    CRETACEOUS = 30

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'stocklist',
        metavar = 'stocklist',
        type = str,
        nargs = '+',
        help = 'If a single file format is passed in, then we assume it contains a'
        'semicolon-separated list of stock that we expect this script to stock list. '
        'If multiple stocks formats are passed in, then we assume stocks are listed directly as arguments.')
    parser.add_argument(
        'holidays',
        metavar = 'holidays',
        type = str,
        nargs = '+',
        help = 'Public holidays in Taiwan, comma separated.')
    parser.add_argument(
        '-t', '--type',
        default = 'ELEC',
        type = lambda stocktype: Stocktype[stocktype],
        choices = list(Stocktype),
        help = 'The stock market you want to choose.')
    parser.add_argument(
        '-o', '--output_file_names',
        default = 'SHIRONG',
        type = str,
        choices = ['SHIRONG', 'shirong'],
        help = 'The owner you want to choose output file name.')
    parser.add_argument(
        '-e', '--endbacktrack',
        default = '10',
        type = int,
        help = 'The owner you want to choose the end of backtrack days.')
    parser.add_argument(
        '-b', '--beginbacktrack',
        default = '0',
        type = int,
        help = 'The owner you want to choose the begin of backtrack days.')
    parser.add_argument(
        '-s', '--subject',
        default = 'None',
        type = str,
        help = 'The owner you want to set the email Subject.')
    parser.add_argument(
        '-cc', '--ccreceiver',
        default = None,
        type = str,
        help = 'The owner you want to cc the email to someone apart from the recipient.')
    parser.add_argument(
        '-l', '--linechart',
        action = 'store_true',
        help = 'The owner you want to show a trend of stock profit ratio over time.')
    parser.add_argument(
        '-p', '--period',
        default = 7,
        type = int,
        help = 'the owner you want to show the period of time on the line chart of stocks profit ratio.')
    parser.add_argument(
        '-m', '--mail',
        action = 'store_true',
        help = 'the owner you want to send the email to recipients from your mail address.')

    return parser


class TwseCrawker():
    def __init__(self, stocklistsize):
        self.stocklistsize = len(stocklist)
        self.daily_stocks = [[] for _ in range(self.stocklistsize)]
        self.stocknumbers = ['' for _ in range(self.stocklistsize)]
        self.stocknames = ['' for _ in range(self.stocklistsize)]
        self.stocksprice = list()
        self.iso_scheduled_times = list()
        self.transactiondays = 0


    def __del__(self):
        self.daily_stocks = [[] for _ in range(self.stocklistsize)]
        self.iso_scheduled_times = list()
        self.transactiondays = 0
        self.stocksprice = list()


    def clean_data(self, row: List[List[str]]) -> List[List[str]]:
        for index, content in enumerate(row):
            row[index] = re.sub(",", "", content.strip())
        return row


    def get_date_times(self, start_date: int, backtrack_days: int, holidays: List[str]) -> None:

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


    def get_twse_daily_stocks(self, file_name: str, stocktype: str, stocks: List[str]) -> None:

        twstockcrawler = TwStockCrawler()
    
        row_data = [[] for _ in range(self.stocklistsize)]

        # Crawing daily TWSE Stock data
        for iso_scheduled_time in self.iso_scheduled_times:

            scheduled_time = ''.join(iso_scheduled_time.split('-'))              
            # Get TWSE information of stock price
            row = twstockcrawler.get_stocktype_data(scheduled_time, stocktype.value[0])
    
            # Store stock data structure
            for item in range(self.stocklistsize):
                # Transfer to NTD Price
                # sign = '-' if row[stocks[item][9]].find('green') > 0 else ''
                self.daily_stocks[item].append(float(row[stocks[item]][5]) * 1000)
                self.stocknumbers[item] = row[stocks[item]][0]
                self.stocknames[item] = row[stocks[item]][1]
                row_data[item] = self.clean_data([
                        row[stocks[item]][0], # Stock number
                        row[stocks[item]][1], # Stock name
                        row[stocks[item]][5], # Stock opening price
                        row[stocks[item]][6], # Stock price high
                        row[stocks[item]][7], # Stock price low
                        ])
            
            # Record TWSE information of stock price
            self.record(file_name = file_name, scheduled_time = scheduled_time, row_data = row_data)


    def smtp_email(self, subject: str, ccreceiver: str, stocktype: str, maxprofits: List[List[int]]) -> None:

        stockprofittable = self.record_to_html_tablefmt(maxprofits)

        smtpemail = SMTPEmail(subject, ccreceiver)
        smtpemail.smtpauthentication()

        # Send HTML email with Python
        smtpemail.textstockprofittable(iso_scheduled_times = self.iso_scheduled_times, 
                                     transactiondays = self.transactiondays,
                                     stocktype = stocktype,
                                     stockprofittable = stockprofittable)


    def smtp_img_email(self, subject: str, ccreceiver: str, stocktype: str, backtrack: str) -> None:
        smtpemail = SMTPEmail(subject, ccreceiver)
        smtpemail.smtpauthentication()

        # Send HTML email with Python
        smtpemail.imgstockprofittable(backtrack = backtrack, stocktype = stocktype)


    def draw_linechart(self, duration: int, maxprofitratios: List[List[int]]) -> None:
    
        # Crreat some sample data
        x = [i for i in range(duration)]
    
        # Create a figure and axis object
        fig, ax = plt.subplots()
    
        # Plot the data as a multi-line chart
        for item  in range(self.stocklistsize):
            ax.plot(x, maxprofitratios[item], label='Stock {} ; Opening price {}'.format(self.stocknumbers[item], self.stocksprice[item]))
    
        # Add labels and title
        ax.set_xlabel('Period')
        ax.set_ylabel('Max Profit Ratio')
        ax.set_title('The Trend of Performance Indicators for Taiwan Stock Market ($NTD)')
        ax.legend()
    
        # Save the chart to a file
        plt.savefig('max_profit_analysis_chart.png')
    
        # Close the chart
        plt.close()


    def cal_max_profit(self) -> List[List[int]]:
        twstockcrawler = TwStockCrawler()

        maxprofits = [[] for _ in range(self.stocklistsize)]
    
        # Caculate max profit and stock profit's table
        for item, daily_stock in zip(range(self.stocklistsize), self.daily_stocks):
            maxprofits[item].append(round(twstockcrawler.maxProfitII(daily_stock), 2))
            maxprofits[item].append(round(twstockcrawler.maxProfit(daily_stock), 2))
            maxprofits[item].append(round(twstockcrawler.maxProfitIV(5, daily_stock), 2))
            maxprofits[item].append(round(twstockcrawler.maxProfitwithfee(daily_stock, 300), 2))

            # stock profit ratio
            maxprofits[item].append(round(twstockcrawler.maxProfitwithfee(daily_stock, 300)/daily_stock[-1], 2))

            # stock opening price
            self.stocksprice.append(round(daily_stock[-1], 2))

        return maxprofits


    def cal_max_profit_ratio_data(self) -> List[List[int]]:
        twstockcrawler = TwStockCrawler()
        maxprofitratios = [[] for _ in range(self.stocklistsize)]

        # Caculate max profit and stock profit's table
        for item, daily_stock in zip(range(self.stocklistsize), self.daily_stocks):
            # stock profit ratio
            maxprofitratios[item].append(round(twstockcrawler.maxProfitwithfee(daily_stock, 300)/daily_stock[-1], 2))
             # stock opening price       
            self.stocksprice.append(round(daily_stock[-1], 2))

        logging.info('Max Profit Ratio: ', maxprofitratios)

        return maxprofitratios


    def record_to_html_tablefmt(self, maxprofits: List[List[int]]):
        df_maxprofits = pd.DataFrame(maxprofits)
        df_maxprofits.columns = ["無限次交易", "交易一次", "至多五次交易", "無限次交易(手續費$NTD300)", "利潤比"]
        df_maxprofits["開盤價"] = self.stocksprice
        df_maxprofits['Name'] = self.stocknames
        df_maxprofits.index = self.stocknumbers
        stockprofittable = tabulate(df_maxprofits, headers = 'keys', tablefmt = 'html')

        return stockprofittable


    def record(self, file_name, scheduled_time: int, row_data: List[List[int]]) -> None:
        # Dump to CSV file
        prefix = './twse'
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        output_file = file_name + '_' + scheduled_time

        with open('{}/{}.csv'.format(prefix, output_file), 'w', newline = '') as file:
            writer = csv.writer(file)
            writer.writerows(row_data)


if __name__ == '__main__':
    # Get arguments
    parser = create_parser()
    args = parser.parse_args()

    if len(args.stocklist) == 1:
        # if we only get a single argument, then it must be a file containing list of outputs
        with open(args.stocklist[0]) as output_list_file:
            stocklist = [int(line.strip()) for line in output_list_file.read().split(';')]
    else:
        stocklist = int(args.stocklist)

    if len(args.holidays) == 1:
        # if we only get a single argument, then it must be a file containing list of outputs
        with open(args.holidays[0]) as output_list_file:
            holidays = [line.strip() for line in output_list_file.read().split(',')]
    else:
        holidays = args.holidays

    if args.linechart:
        startofbacktrack = args.endbacktrack - args.period
        now_date_time = datetime.datetime.now()
        backtrack = iso_date_time = (now_date_time + datetime.timedelta(days=-args.endbacktrack)).strftime("%Y-%m-%d")

        maxprofitratios = [[] for _ in range(len(stocklist))]

        for back in range(startofbacktrack): 
            twsecrawler = TwseCrawker(len(stocklist))
            print('Back {}'.format(back))
            twsecrawler.get_date_times(start_date = startofbacktrack - back, backtrack_days = args.endbacktrack - back, holidays = holidays)
            twsecrawler.get_twse_daily_stocks(file_name = args.output_file_names, stocktype = args.type, stocks = stocklist)

            for item, maxprofitratiodata in zip(range(len(stocklist)), twsecrawler.cal_max_profit_ratio_data()):
                maxprofitratios[item].append(maxprofitratiodata)

            if back == startofbacktrack - 1:
                twsecrawler.draw_linechart(duration = startofbacktrack, maxprofitratios = maxprofitratios)
                if args.mail:
                    twsecrawler.smtp_img_email(subject = args.subject, ccreceiver = args.ccreceiver, stocktype = args.type, backtrack = backtrack)
                continue

            time.sleep(QUERY.NEOGENE.value)
        logging.info('The trend of performance indicators for TWSE stock market: ', maxprofitratios)
    else:
        twsecrawler = TwseCrawker(len(stocklist))
        twsecrawler.get_date_times(start_date = args.beginbacktrack, backtrack_days = args.endbacktrack, holidays = holidays)
        twsecrawler.get_twse_daily_stocks(file_name = args.output_file_names, stocktype = args.type, stocks = stocklist)

        if args.mail:
            twsecrawler.smtp_email(subject = args.subject, ccreceiver = args.ccreceiver, stocktype = args.type, maxprofits = twsecrawler.cal_max_profit())
