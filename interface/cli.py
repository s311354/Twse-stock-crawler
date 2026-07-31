"""Command-line interface for TWSE stock analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from application.stock_service import StockAnalysisRequest
from config.settings import AppSettings
from config.settings import SettingsError
from domain.models import Stocktype
from infrastructure.bootstrap import build_stock_analysis_service
from infrastructure.calendar.twse_calendar import TwseCalendarError
from infrastructure.crawler.twse_http_client import TwseClientError
from infrastructure.observability import configure_structured_logging


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "stocklist",
        metavar="stocklist",
        type=str,
        nargs="+",
        help="Stock numbers directly, or legacy row indexes in a semicolon-separated stocklist file.",
    )
    parser.add_argument(
        "-t",
        "--type",
        default="ELEC",
        type=lambda stocktype: Stocktype[stocktype],
        choices=list(Stocktype),
        help="The stock market you want to choose.",
    )
    parser.add_argument(
        "-o",
        "--output_file_names",
        default="SHIRONG",
        type=str,
        choices=["SHIRONG", "shirong"],
        help="Output file name prefix.",
    )
    parser.add_argument(
        "-e",
        "--endbacktrack",
        default="10",
        type=int,
        help="Older backtrack boundary in calendar days (inclusive).",
    )
    parser.add_argument(
        "-b",
        "--beginbacktrack",
        default="0",
        type=int,
        help="Newer backtrack boundary in calendar days (exclusive).",
    )
    parser.add_argument(
        "-s",
        "--subject",
        default="None",
        type=str,
        help="Email subject.",
    )
    parser.add_argument(
        "-cc",
        "--ccreceiver",
        default=None,
        type=str,
        help="Email CC receiver.",
    )
    parser.add_argument(
        "-l",
        "--linechart",
        action="store_true",
        help="Show a stock profit-ratio line chart.",
    )
    parser.add_argument(
        "-p",
        "--period",
        default=7,
        type=int,
        help="Line-chart period.",
    )
    parser.add_argument(
        "-m",
        "--mail",
        action="store_true",
        help="Send email to recipients.",
    )
    parser.add_argument(
        "--skip-mops",
        action="store_true",
        help="Skip MOPS/iXBRL refresh; LES v4 will remain WATCH_ONLY without cached context.",
    )
    parser.add_argument(
        "--mops-history-years",
        default=None,
        type=int,
        help="Number of annual MOPS filings retained for the LES v4 quality gate (minimum 3).",
    )
    parser.add_argument(
        "--account-equity",
        default=None,
        type=float,
        help="Account equity in NTD used for LES v4 board-lot risk sizing.",
    )

    return parser


def load_stocklist(values: list[str]) -> list[str]:
    if len(values) == 1 and Path(values[0]).exists():
        with open(values[0]) as output_list_file:
            return [line.strip() for line in output_list_file.read().split(";") if line.strip()]

    return [stock.strip() for stock in values if stock.strip()]


def build_run_configuration(
    argv: list[str] | None = None,
) -> tuple[StockAnalysisRequest, AppSettings]:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        settings = AppSettings.from_env(
            mail_enabled=args.mail,
            mops_enabled=not args.skip_mops,
            account_equity=args.account_equity,
            mops_history_years=args.mops_history_years,
        )
    except SettingsError as error:
        parser.error(str(error))

    request = StockAnalysisRequest(
        stocklist=load_stocklist(args.stocklist),
        stocktype=args.type,
        output_file_names=args.output_file_names,
        endbacktrack=args.endbacktrack,
        beginbacktrack=args.beginbacktrack,
        subject=args.subject,
        ccreceiver=args.ccreceiver,
        linechart=args.linechart,
        period=args.period,
        mail=args.mail,
        mops_enabled=not args.skip_mops,
        mops_history_years=settings.mops_history_years,
        account_equity=settings.account_equity,
    )
    return request, settings


def build_request(argv: list[str] | None = None) -> StockAnalysisRequest:
    request, _ = build_run_configuration(argv)
    return request


def main(argv: list[str] | None = None) -> int:
    request, settings = build_run_configuration(argv)
    configure_structured_logging(settings.log_level)
    try:
        build_stock_analysis_service(settings=settings).run(request)
    except TwseCalendarError as error:
        print(f"TWSE market status: UNKNOWN ({error})", file=sys.stderr)
        return 2
    except TwseClientError as error:
        print(
            f"TWSE request status: {error.category.value.upper()} ({error})",
            file=sys.stderr,
        )
        return 3
    return 0
