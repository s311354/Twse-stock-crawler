"""Command-line interface for TWSE stock analysis."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

from application.stock_service import StockAnalysisRequest
from config.settings import AppSettings
from config.settings import SettingsError
from domain.models import PortfolioPosition
from domain.models import Stocktype
from domain.strategy import available_strategies
from domain.strategy import strategy_registry
from infrastructure.bootstrap import build_stock_analysis_service
from infrastructure.calendar.twse_calendar import TwseCalendarError
from infrastructure.crawler.twse_http_client import TwseClientError
from infrastructure.observability import configure_structured_logging
from infrastructure.run_lock import ProcessRunLock
from infrastructure.run_lock import RunAlreadyActiveError


def non_negative_int(value: str) -> int:
    """Validate that a value is a non-negative integer."""
    try:
        int_value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer")
    
    if int_value < 0:
        raise argparse.ArgumentTypeError(
            f"backtrack value must be non-negative, got {int_value}"
        )
    return int_value


def validate_backtrack_order(begin: int, end: int, parser: argparse.ArgumentParser) -> None:
    """Validate that end_backtrack > begin_backtrack."""
    if end <= begin:
        parser.error(
            f"--endbacktrack ({end}) must be greater than --beginbacktrack ({begin}). "
            f"Example: -e 500 -b 0 analyzes the past 500 days excluding today."
        )


def list_strategies() -> None:
    """Print all available strategies with their metadata."""
    print("\nAvailable Trading Strategies:\n")
    print(f"{'Name':<30} {'Version':<10} {'Description'}")
    print("-" * 100)
    
    strategy_names = available_strategies()
    for name in strategy_names:
        try:
            strategy = strategy_registry.get(name)
            version = getattr(strategy, "version", "N/A")
            description = getattr(strategy, "description", "No description available")
            print(f"{name:<30} {version:<10} {description}")
        except Exception as error:
            print(f"{name:<30} {'ERROR':<10} Failed to load: {error}")
    
    print()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List all available trading strategies with descriptions and exit.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help=(
            "Validate runtime settings and inputs, print a safe JSON summary, "
            "and exit without network requests, analysis, or email."
        ),
    )

    parser.add_argument(
        "stocklist",
        metavar="stocklist",
        type=str,
        nargs="*",
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
        default=10,
        type=non_negative_int,
        help="Older backtrack boundary in calendar days (inclusive). Must be greater than --beginbacktrack.",
    )
    parser.add_argument(
        "-b",
        "--beginbacktrack",
        default=0,
        type=non_negative_int,
        help="Newer backtrack boundary in calendar days (exclusive). Must be less than --endbacktrack.",
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
        help=(
            "Skip MOPS/iXBRL refresh; LES v5 Extended Quality becomes "
            "unavailable and formal BUY signals are disabled."
        ),
    )
    parser.add_argument(
        "--mops-history-years",
        default=None,
        type=int,
        help="Number of annual MOPS filings retained for LES v5 Extended Quality (minimum 3).",
    )
    parser.add_argument(
        "--account-equity",
        default=None,
        type=float,
        help="Account equity in NTD used for LES v5 risk-based position sizing.",
    )
    parser.add_argument(
        "--pae-holdings-file",
        type=Path,
        default=None,
        help=(
            "PAE holdings text file. Use comma/semicolon/whitespace-separated "
            "symbols, or symbol:shares for explicit positions."
        ),
    )
    parser.add_argument(
        "--pae-current-cash",
        type=float,
        default=None,
        help="Optional current portfolio cash in NTD for PAE reconciliation.",
    )
    parser.add_argument(
        "--mode",
        choices=("release", "debug"),
        default=None,
        help=(
            "Runtime output mode. release suppresses diagnostic details; "
            "debug enables per-date and per-symbol diagnostics."
        ),
    )

    return parser


def load_stocklist(values: list[str]) -> list[str]:
    if len(values) == 1 and Path(values[0]).exists():
        with open(values[0]) as output_list_file:
            return [line.strip() for line in output_list_file.read().split(";") if line.strip()]

    return [stock.strip() for stock in values if stock.strip()]


def load_pae_holdings_file(path: str | Path) -> tuple[PortfolioPosition, ...]:
    """Parse ``symbol[:shares[:entry_date[:entry_les]]]`` holdings input.

    Empty optional fields are supported, for example
    ``2449::2026-08-05:82`` or ``2449:125::``.  Existing symbol-only and
    ``symbol:shares`` files remain valid.
    """
    holdings_path = Path(path).expanduser()
    try:
        raw = holdings_path.read_text(encoding="utf-8")
    except OSError as error:
        raise SettingsError(
            "PAE holdings file is not readable: {}".format(holdings_path)
        ) from error

    content = "\n".join(line.split("#", 1)[0] for line in raw.splitlines())
    tokens = [
        token
        for token in re.split(r"[\s,;()]+", content.strip())
        if token
    ]
    if not tokens:
        raise SettingsError("PAE holdings file does not contain any symbols")
    if len(tokens) > 4:
        raise SettingsError("PAE holdings file supports at most 4 holdings")

    positions: list[PortfolioPosition] = []
    for token in tokens:
        parts = token.split(":")
        if len(parts) > 4:
            raise SettingsError("Invalid PAE holding token: {}".format(token))
        symbol = parts[0].strip().upper()
        if re.fullmatch(r"[0-9A-Z]{4,10}", symbol) is None:
            raise SettingsError("Invalid PAE holding symbol: {}".format(symbol))
        shares: float | None = None
        if len(parts) >= 2 and parts[1].strip():
            try:
                shares = float(parts[1])
            except ValueError as error:
                raise SettingsError(
                    "Invalid PAE shares for {}: {}".format(symbol, parts[1])
                ) from error
        entry_date = parts[2].strip() if len(parts) >= 3 else ""
        entry_les: float | None = None
        if len(parts) >= 4 and parts[3].strip():
            try:
                entry_les = float(parts[3])
            except ValueError as error:
                raise SettingsError(
                    "Invalid PAE Entry LES for {}: {}".format(symbol, parts[3])
                ) from error
        try:
            positions.append(
                PortfolioPosition(
                    symbol=symbol,
                    shares=shares,
                    entry_date=entry_date or None,
                    entry_les=entry_les,
                )
            )
        except ValueError as error:
            raise SettingsError(str(error)) from error

    symbols = [position.symbol for position in positions]
    if len(symbols) != len(set(symbols)):
        raise SettingsError("PAE holdings file contains duplicate symbols")
    return tuple(positions)


def build_run_configuration(
    argv: list[str] | None = None,
) -> tuple[StockAnalysisRequest, AppSettings]:
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Validate backtrack parameter relationship
    validate_backtrack_order(args.beginbacktrack, args.endbacktrack, parser)
    
    try:
        settings = AppSettings.from_env(
            mail_enabled=args.mail,
            mops_enabled=not args.skip_mops,
            account_equity=args.account_equity,
            mops_history_years=args.mops_history_years,
            runtime_mode=args.mode,
        )
        if args.pae_holdings_file is not None or args.pae_current_cash is not None:
            settings = replace(
                settings,
                pae_positions=(
                    load_pae_holdings_file(args.pae_holdings_file)
                    if args.pae_holdings_file is not None
                    else settings.pae_positions
                ),
                pae_current_cash=(
                    args.pae_current_cash
                    if args.pae_current_cash is not None
                    else settings.pae_current_cash
                ),
            )
            settings.validate(
                mail_enabled=args.mail,
                mops_enabled=not args.skip_mops,
                validate_paths=False,
            )
    except SettingsError as error:
        parser.error(str(error))

    stocklist = load_stocklist(args.stocklist)
    tracked = set(stocklist)
    for position in settings.pae_positions:
        if position.symbol not in tracked:
            stocklist.append(position.symbol)
            tracked.add(position.symbol)

    request = StockAnalysisRequest(
        stocklist=stocklist,
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


def build_preflight_summary(
    request: StockAnalysisRequest,
    settings: AppSettings,
) -> dict[str, object]:
    """Return an operational summary without exposing SMTP credentials."""
    return {
        "account_equity": settings.account_equity,
        "backtrack": {
            "begin": request.beginbacktrack,
            "end": request.endbacktrack,
        },
        "mail_enabled": request.mail,
        "market_history_cache": str(settings.market_history_cache_path),
        "mops_cache": str(settings.mops_cache_dir),
        "mops_enabled": request.mops_enabled,
        "output": request.output_file_names,
        "pae_holding_count": len(settings.pae_positions),
        "pae_holding_symbols": [
            position.symbol for position in settings.pae_positions
        ],
        "pae_persistence": str(settings.pae_persistence_path),
        "runtime_mode": settings.runtime_mode,
        "run_lock": str(settings.run_lock_path),
        "smtp_configured": bool(
            settings.smtp_email
            and settings.smtp_password
            and settings.smtp_receivers
            and settings.smtp_server_host
        ),
        "status": "OK",
        "stock_type": request.stocktype.name,
        "symbol_count": len(request.stocklist),
    }


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Handle --list-strategies early exit
    if args.list_strategies:
        list_strategies()
        return 0
    
    # Validate stocklist is provided for normal operation
    if not args.stocklist:
        parser.error("the following arguments are required: stocklist")
    
    request, settings = build_run_configuration(argv)
    configure_structured_logging(settings.effective_log_level)
    if args.check_config:
        print(
            json.dumps(
                build_preflight_summary(request, settings),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    try:
        with ProcessRunLock(settings.run_lock_path):
            build_stock_analysis_service(settings=settings).run(request)
    except RunAlreadyActiveError as error:
        print(f"TWSE run status: ALREADY_RUNNING ({error})", file=sys.stderr)
        return 4
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
