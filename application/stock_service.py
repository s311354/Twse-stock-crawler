"""Application service that orchestrates TWSE stock analysis workflows."""

from __future__ import annotations

import datetime
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from config.settings import QUERY_WAIT_SECONDS


@dataclass(frozen=True)
class StockAnalysisRequest:
    stocklist: list[int]
    holidays: list[str]
    stocktype: Any
    output_file_names: str
    endbacktrack: int
    beginbacktrack: int = 0
    subject: str = "None"
    ccreceiver: str | None = None
    linechart: bool = False
    period: int = 7
    mail: bool = False


class StockAnalysisService:
    """Coordinates crawler, analysis, storage, chart, and mail boundaries."""

    def __init__(self, crawler_factory: Callable[[int], Any] | None = None) -> None:
        if crawler_factory is None:
            from stockanalysis import TwseCrawker

            self.crawler_factory = TwseCrawker
        else:
            self.crawler_factory = crawler_factory

    def run(self, request: StockAnalysisRequest) -> None:
        if request.linechart:
            self._run_linechart(request)
            return

        twsecrawler = self.crawler_factory(len(request.stocklist))
        twsecrawler.get_date_times(
            start_date=request.beginbacktrack,
            backtrack_days=request.endbacktrack,
            holidays=request.holidays,
        )
        twsecrawler.get_twse_daily_stocks(
            file_name=request.output_file_names,
            stocktype=request.stocktype,
            stocks=request.stocklist,
        )
        maxprofits = twsecrawler.cal_max_profit()
        twsecrawler.record_analysis_dataset(file_name=request.output_file_names, maxprofits=maxprofits)

        if request.mail:
            twsecrawler.smtp_email(
                subject=request.subject,
                ccreceiver=request.ccreceiver,
                stocktype=request.stocktype,
                maxprofits=maxprofits,
            )

    def _run_linechart(self, request: StockAnalysisRequest) -> None:
        startofbacktrack = request.endbacktrack - request.period
        now_date_time = datetime.datetime.now()
        backtrack = (now_date_time + datetime.timedelta(days=-request.endbacktrack)).strftime("%Y-%m-%d")
        maxprofitratios: list[list[float]] = [[] for _ in range(len(request.stocklist))]

        for back in range(startofbacktrack):
            twsecrawler = self.crawler_factory(len(request.stocklist))
            print("Back {}".format(back))
            twsecrawler.get_date_times(
                start_date=startofbacktrack - back,
                backtrack_days=request.endbacktrack - back,
                holidays=request.holidays,
            )
            twsecrawler.get_twse_daily_stocks(
                file_name=request.output_file_names,
                stocktype=request.stocktype,
                stocks=request.stocklist,
            )

            for item, maxprofitratiodata in zip(range(len(request.stocklist)), twsecrawler.cal_max_profit_ratio_data()):
                maxprofitratios[item].append(maxprofitratiodata)

            if back == startofbacktrack - 1:
                twsecrawler.draw_linechart(duration=startofbacktrack, maxprofitratios=maxprofitratios)
                if request.mail:
                    twsecrawler.smtp_img_email(
                        subject=request.subject,
                        ccreceiver=request.ccreceiver,
                        stocktype=request.stocktype,
                        backtrack=backtrack,
                    )
                continue

            time.sleep(QUERY_WAIT_SECONDS)

        logging.info("The trend of performance indicators for TWSE stock market: %s", maxprofitratios)
