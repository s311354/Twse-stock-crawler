"""Command-line interface for TWSE stock analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from application.stock_service import StockAnalysisRequest
from application.stock_service import StockAnalysisService
from domain.models import Stocktype


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
        "holidays",
        metavar="holidays",
        type=str,
        nargs="+",
        help="Public holidays in Taiwan, comma separated or a holidays file.",
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
        help="End of backtrack days.",
    )
    parser.add_argument(
        "-b",
        "--beginbacktrack",
        default="0",
        type=int,
        help="Begin of backtrack days.",
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

    return parser


def load_stocklist(values: list[str]) -> list[str]:
    if len(values) == 1 and Path(values[0]).exists():
        with open(values[0]) as output_list_file:
            return [line.strip() for line in output_list_file.read().split(";") if line.strip()]

    return [stock.strip() for stock in values if stock.strip()]


def load_holidays(values: list[str]) -> list[str]:
    if len(values) == 1 and Path(values[0]).exists():
        with open(values[0]) as output_list_file:
            return [line.strip() for line in output_list_file.read().split(",")]

    return values


def build_request(argv: list[str] | None = None) -> StockAnalysisRequest:
    parser = create_parser()
    args = parser.parse_args(argv)

    return StockAnalysisRequest(
        stocklist=load_stocklist(args.stocklist),
        holidays=load_holidays(args.holidays),
        stocktype=args.type,
        output_file_names=args.output_file_names,
        endbacktrack=args.endbacktrack,
        beginbacktrack=args.beginbacktrack,
        subject=args.subject,
        ccreceiver=args.ccreceiver,
        linechart=args.linechart,
        period=args.period,
        mail=args.mail,
    )


def main(argv: list[str] | None = None) -> None:
    request = build_request(argv)
    StockAnalysisService().run(request)
