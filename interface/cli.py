"""Command-line interface for TWSE stock analysis."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import uuid
from dataclasses import replace
from pathlib import Path

from application.stock_service import StockAnalysisRequest
from application.historical_replay import replay_close_cutoff
from config.settings import AppSettings, DATA_DIR
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
from infrastructure.research_series import ResearchSeriesError
from infrastructure.storage.research_portfolio import ResearchStateError


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


def backtrack_boundary(value: str) -> int | dt.date:
    """Keep legacy integer parsing; accept only explicit ISO calendar dates."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return dt.date.fromisoformat(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"Invalid YYYY-MM-DD date: {value}") from error
    return non_negative_int(value)


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
        "--les-v7-research-bundle", type=Path,
        help="Use a READY_TO_FREEZE real offline frozen_bundle.json inside its native archive for current-data v7 research. Verifies OOF/source archives, uses the frozen stock universe, emails a labelled research report by default, and isolates PAE state; never deployment approval. --check-config sends nothing. Default warm-up: 1100 calendar days; actual valid-close requirements remain native gates.",
    )
    parser.add_argument("--research-id",
        help="Optional fixed ID for --les-v7-research-bundle only. Continues isolated cross-session PAE evidence with pinned model/code/policy/universe; per-run reports stay separate.")
    parser.add_argument("--les-v7-strategy-policy", type=Path,
        help="Explicit immutable fixed-rule StrategyPolicy JSON; must match any fitted model's policy blocks.")
    parser.add_argument("--les-v7-diagnostics-only", action="store_true", default=None,
        help="Explicit fixed-policy v7 diagnostics without formal BUY/BUY_SMALL eligibility; no model fallback.")

    parser.add_argument(
        "stocklist",
        metavar="stocklist",
        type=str,
        nargs="*",
        help="Stock numbers or a semicolon-separated file. SEMI uses TPEx; ALL discovers its historical universe (no legacy indexes).",
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
        type=backtrack_boundary,
        help="Integer: older inclusive calendar-day offset. Date: inclusive history end (YYYY-MM-DD), fixed at that day's Taipei 13:30 replay cutoff; pair with a date -b.",
    )
    parser.add_argument(
        "-b",
        "--beginbacktrack",
        default=0,
        type=backtrack_boundary,
        help="Integer: newer exclusive calendar-day offset. Date: inclusive history start (YYYY-MM-DD); pair with a date -e.",
    )
    parser.add_argument(
        "--replay-mail", action="store_true",
        help="Explicitly send a historical research report via existing SMTP; requires a date -b/-e pair. Never authorizes trading.",
    )
    parser.add_argument(
        "--replay-restrictions-file", type=Path, default=None,
        help="Archived per-symbol restriction JSON or restriction-announcements/v1 source bundle. Requires a date -b/-e pair; incomplete coverage remains UNKNOWN.",
    )
    parser.add_argument("--replay-id", help="Isolated strict-replay job ID (date mode).")
    parser.add_argument("--replay-evidence-file", type=Path, help="Pinned replay-evidence/v1 package; no runtime downloads.")
    parser.add_argument("--replay-state-path", type=Path, help="Research-only SQLite state; never use the live PAE database.")
    parser.add_argument("--replay-output-dir", type=Path, help="Research JSON/CSV directory; defaults to data/replays/<id>.")
    parser.add_argument("--replay-parse-cache", type=Path, help="Optional derived SQLite parse/cutoff index for historical replay only; never a live MOPS/PAE database.")
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
        "--no-mail",
        action="store_true",
        help="Explicitly suppress email, including the default delivery for offline v7 research.",
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
        help="Completed fiscal years of quarterly MOPS/iXBRL history to cache, plus current YTD (default 5, minimum 3).",
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
        "--include-symbols-file",
        type=Path,
        default=None,
        help=(
            "Additional symbols to include in the shared market analysis "
            "universe without treating them as portfolio holdings."
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


def load_pae_holdings_file(path: str | Path, *, allow_empty: bool = False) -> tuple[PortfolioPosition, ...]:
    """Parse ``symbol[:shares[:entry_date[:entry_les[:entry_decision]]]]``.

    Empty optional fields are supported, for example
    ``2449::2026-08-05:82:BUY`` or ``2449:125:::``. Existing one- through
    four-field holding tokens remain valid.
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
        if allow_empty:
            return ()
        raise SettingsError("PAE holdings file does not contain any symbols")
    if len(tokens) > 4:
        raise SettingsError("PAE holdings file supports at most 4 holdings")

    positions: list[PortfolioPosition] = []
    for token in tokens:
        parts = token.split(":")
        if len(parts) > 5:
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
        entry_decision = parts[4].strip() if len(parts) >= 5 else ""
        try:
            positions.append(
                PortfolioPosition(
                    symbol=symbol,
                    shares=shares,
                    entry_date=entry_date or None,
                    entry_les=entry_les,
                    entry_decision=entry_decision or None,
                )
            )
        except ValueError as error:
            raise SettingsError(str(error)) from error

    symbols = [position.symbol for position in positions]
    if len(symbols) != len(set(symbols)):
        raise SettingsError("PAE holdings file contains duplicate symbols")
    return tuple(positions)


def load_include_symbols_file(path: str | Path) -> tuple[str, ...]:
    """Load bounded, symbol-only additions to a shared analysis universe."""
    symbols_path = Path(path).expanduser()
    try:
        raw = symbols_path.read_text(encoding="utf-8")
    except OSError as error:
        raise SettingsError(
            "include symbols file is not readable: {}".format(symbols_path)
        ) from error
    content = "\n".join(line.split("#", 1)[0] for line in raw.splitlines())
    tokens = [
        token.strip().upper()
        for token in re.split(r"[\s,;()]+", content.strip())
        if token.strip()
    ]
    if len(tokens) > 40_000:
        raise SettingsError("include symbols file exceeds the 40000 symbol limit")
    invalid = next(
        (symbol for symbol in tokens if re.fullmatch(r"[0-9A-Z]{4,10}", symbol) is None),
        None,
    )
    if invalid is not None:
        raise SettingsError("Invalid included symbol: {}".format(invalid))
    return tuple(dict.fromkeys(tokens))


def build_run_configuration(
    argv: list[str] | None = None,
) -> tuple[StockAnalysisRequest, AppSettings]:
    parser = create_parser()
    args = parser.parse_args(argv)
    research = args.les_v7_research_bundle is not None
    if args.research_id is not None:
        if not research:
            parser.error("--research-id requires --les-v7-research-bundle")
        from infrastructure.research_series import validate_research_id
        try:
            validate_research_id(args.research_id)
        except ValueError as error:
            parser.error(str(error))
    supplied = sys.argv[1:] if argv is None else argv
    if args.no_mail and (args.mail or args.replay_mail):
        parser.error("--no-mail cannot be combined with --mail or --replay-mail")
    if research:
        if args.replay_mail or args.linechart:
            parser.error("LES_V7_RESEARCH_ONLY: replay-mail and line charts are not supported")
        if isinstance(args.beginbacktrack, dt.date) or isinstance(args.endbacktrack, dt.date):
            parser.error("LES_V7_RESEARCH_CURRENT_DATA_ONLY: an offline publication cannot be backdated into strict historical replay")
        if args.type != Stocktype.ELEC:
            parser.error("LES_V7_RESEARCH_FIXED_ELEC_SCOPE_REQUIRED")
        if not any(value in {"-e", "--endbacktrack"} or value.startswith("--endbacktrack=")
                   or (value.startswith("-e") and len(value) > 2)
                   for value in supplied):
            args.endbacktrack = 1100
    
    # Normalize the CLI only: downstream offset consumers retain integer types.
    # Explicit dates are also carried through so a single-day range never
    # includes a fabricated extra warm-up day.
    history_start = history_end = None
    replay_cutoff = None
    if isinstance(args.beginbacktrack, dt.date) or isinstance(args.endbacktrack, dt.date):
        if not (isinstance(args.beginbacktrack, dt.date) and isinstance(args.endbacktrack, dt.date)):
            parser.error("-b/-e must both be non-negative integers or both YYYY-MM-DD dates; mixed boundaries are not supported")
        history_start, history_end = args.beginbacktrack, args.endbacktrack
        if history_start > history_end:
            parser.error("Date range requires -b START <= -e END")
        replay_cutoff = replay_close_cutoff(history_end.isoformat())
        args.beginbacktrack = 0
        args.endbacktrack = max(1, (history_end - history_start).days)
    else:
        validate_backtrack_order(args.beginbacktrack, args.endbacktrack, parser)
        if args.beginbacktrack > 0:
            parser.error("HISTORICAL_RANGE_REQUIRES_DATE_REPLAY: integer -b > 0 excludes recent prices; use -b START_DATE -e END_DATE with --replay-evidence-file, --replay-id and an explicit historical account instead of today's evidence")
    if replay_cutoff is not None:
        if replay_cutoff > dt.datetime.now(dt.timezone.utc):
            parser.error("Replay end-date close cannot be in the future")
        if args.linechart:
            parser.error("Historical replay does not support --linechart")
        if args.mail and not args.replay_mail:
            parser.error("Historical email requires explicit --replay-mail; --mail alone is not sufficient")
        if args.pae_holdings_file is None or args.pae_current_cash is None or args.account_equity is None:
            parser.error("Historical replay requires explicit --pae-holdings-file (empty for no holdings), --pae-current-cash and --account-equity as the historical portfolio snapshot")
        if not math.isfinite(args.pae_current_cash) or not math.isfinite(args.account_equity):
            parser.error("Historical cash and equity must be finite numbers")
    elif args.replay_restrictions_file is not None:
        parser.error("--replay-restrictions-file requires a date -b/-e pair")
    if args.replay_mail and replay_cutoff is None:
        parser.error("--replay-mail requires a date -b/-e pair")
    # Research report delivery is not model deployment or trading authority.
    # Preflight remains read-only and needs no SMTP account; real research runs
    # validate configured SMTP before any analysis or source refresh begins.
    mail_enabled = ((not args.check_config) and not args.no_mail) if research else (args.mail or args.replay_mail) and not args.no_mail
    research_run_id = uuid.uuid4().hex if research else None
    environment = ({key: value for key, value in os.environ.items()
                    if key not in {"TWSE_PAE_HOLDINGS_JSON", "TWSE_PAE_CURRENT_CASH"}}
                   if replay_cutoff is not None else dict(os.environ) if research else None)
    if research:
        if args.research_id is not None:
            from infrastructure.research_series import validate_study_paths
            try:
                validate_study_paths(DATA_DIR / "les-v7-research" / args.research_id, args.research_id)
            except ValueError as error:
                parser.error(str(error))
        environment["TWSE_PAE_PERSISTENCE_PATH"] = str(DATA_DIR / "les-v7-research" / (args.research_id or research_run_id) / "portfolio.sqlite3")
    
    try:
        settings = AppSettings.from_env(
            environ=environment,
            mail_enabled=mail_enabled,
            mops_enabled=not args.skip_mops,
            account_equity=args.account_equity,
            mops_history_years=args.mops_history_years,
            runtime_mode=args.mode,
            les_v7_research_bundle=args.les_v7_research_bundle,
            les_v7_strategy_policy=args.les_v7_strategy_policy,
            les_v7_diagnostics_only=args.les_v7_diagnostics_only,
        )
        if args.pae_holdings_file is not None or args.pae_current_cash is not None:
            settings = replace(
                settings,
                pae_positions=(
                    load_pae_holdings_file(args.pae_holdings_file, allow_empty=replay_cutoff is not None or research)
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
                mail_enabled=mail_enabled,
                mops_enabled=not args.skip_mops,
                validate_paths=False,
            )
    except SettingsError as error:
        parser.error(str(error))

    if replay_cutoff is not None:
        for position in settings.pae_positions:
            if position.shares is None or position.entry_date is None:
                parser.error("Historical holdings require explicit shares and entry_date")
            entry_date = dt.date.fromisoformat(str(position.entry_date))
            if entry_date > replay_cutoff.date():
                parser.error("Historical holdings cannot contain entries after the replay date")
        if args.replay_restrictions_file is not None and not args.replay_restrictions_file.is_file():
            parser.error("Archived restriction evidence file does not exist")

    metadata = json.loads(settings.les_v7_research_metadata_json) if research else None
    stocklist = list(metadata["symbols"]) if research else load_stocklist(args.stocklist)
    if args.type == Stocktype.SEMI:
        stocklist = list(dict.fromkeys(stocklist or ["ALL"]))
        if any(symbol != "ALL" and re.fullmatch(r"\d{4}", symbol) is None for symbol in stocklist):
            parser.error("SEMI uses TPEx stock numbers or ALL; old TWSE stocklist_semi row indexes are not supported")
        if args.linechart and "ALL" in stocklist:
            parser.error("SEMI line charts require explicit TPEx stock numbers, not ALL")
    tracked = set(stocklist)
    included_symbols: tuple[str, ...] = ()
    if args.include_symbols_file is not None:
        if research:
            parser.error("LES_V7_RESEARCH_FIXED_SCOPE: --include-symbols-file cannot expand a frozen model universe")
        included_symbols = load_include_symbols_file(args.include_symbols_file)
        for symbol in included_symbols:
            if symbol not in tracked:
                stocklist.append(symbol)
                tracked.add(symbol)
    for position in settings.pae_positions:
        if position.symbol not in tracked:
            if research:
                parser.error("LES_V7_RESEARCH_FIXED_SCOPE: account holdings outside the frozen universe are not supported")
            stocklist.append(position.symbol)
            tracked.add(position.symbol)

    if research:
        from domain.les_v7.fr2_policy import component as fr2_entry_component
        from domain.les_v7.fr2_execution import component as fr2_execution_component
        from domain.les_v7.valuation_price import policy as eps_price_policy
        included_symbols = tuple(stocklist)  # Symbols, never legacy market-row indexes.
        metadata.update(research_run_id=research_run_id, warmup_calendar_days=args.endbacktrack,
            required_close_count_policy="NATIVE_VALID_CLOSES_NOT_CALENDAR_ESTIMATE",
            admission_component=fr2_entry_component(), execution_component=fr2_execution_component(),
            valuation_policy=eps_price_policy(), prediction_bundle_unchanged=True,
            decision_thresholds=dict(BUY=75., BUY_SMALL=65.),
            return_basis="PRICE_RETURN_ONLY_NOT_TOTAL_RETURN")
        settings = replace(settings, les_v7_research_metadata_json=json.dumps(metadata))

    request = StockAnalysisRequest(
        stocklist=stocklist,
        stocktype=args.type,
        output_file_names=(f"{args.output_file_names}_LESv7_RESEARCH_{metadata['bundle_sha256'][:12]}_{research_run_id}"
                           if research else args.output_file_names + "_tpex_semi"
                           if args.type == Stocktype.SEMI else args.output_file_names),
        endbacktrack=args.endbacktrack,
        beginbacktrack=args.beginbacktrack,
        subject=(f"[LES_V7_RESEARCH｜研究用途｜不可下單] {args.subject}" if research else args.subject),
        ccreceiver=args.ccreceiver,
        linechart=args.linechart,
        period=args.period,
        mail=mail_enabled,
        mops_enabled=not args.skip_mops,
        mops_history_years=settings.mops_history_years,
        account_equity=settings.account_equity,
        portfolio_symbols=tuple(
            position.symbol for position in settings.pae_positions
        ),
        explicit_symbols=tuple(
            dict.fromkeys(
                [
                    *included_symbols,
                    *(position.symbol for position in settings.pae_positions),
                ]
            )
        ),
        as_of=replay_cutoff,
        replay_mail=args.replay_mail,
        replay_restrictions_file=str(args.replay_restrictions_file) if args.replay_restrictions_file else None,
        history_start_date=history_start,
        history_end_date=history_end,
    )
    if args.research_id is not None:
        from infrastructure.research_series import make_research_identity, check_study
        from application.replay_contracts import digest
        metadata["research_id"] = args.research_id
        identity = make_research_identity(metadata, request, settings)
        metadata.update(research_identity=identity, research_identity_id=digest(identity))
        try:
            check_study(settings.pae_persistence_path.parent, identity)
        except ValueError as error:
            parser.error(str(error))
        settings = replace(settings, les_v7_research_metadata_json=json.dumps(metadata))
    return request, settings


def build_request(argv: list[str] | None = None) -> StockAnalysisRequest:
    request, _ = build_run_configuration(argv)
    return request


def build_preflight_summary(
    request: StockAnalysisRequest,
    settings: AppSettings,
) -> dict[str, object]:
    """Return an operational summary without exposing SMTP credentials."""
    from domain.strategies.expected_return_exit_b import DAILY_EREM_MODEL_VERSION
    from domain.strategies.erem_soft_reduction import SoftReductionFormalPolicy
    quality = None
    from application.les_v7_preflight import daily_v7_preflight
    readiness = daily_v7_preflight(settings, checked_at=dt.datetime.now(dt.timezone.utc), market=request.market_source)
    if settings.daily_entry_model == "v7":
        from domain.les_v7.quality_policy import default_quality_policy, is_source_evidence_policy, validate_quality_policy
        from domain.les_v7.rule_baseline import baseline_policy, current_source_baseline_version
        default = (baseline_policy(request.market_source, version=current_source_baseline_version(request.as_of or dt.datetime.now(dt.timezone.utc)))["quality_policy"]
            if settings.les_v7_rule_baseline and settings.les_v7_source_repair else default_quality_policy())
        configured = (json.loads(settings.les_v7_strategy_policy_json)["payload"]["quality_policy"]
                      if settings.les_v7_strategy_policy_json else
                      json.loads(settings.les_v7_bundle_json)["payload"]["quality_policy"]
                      if settings.les_v7_bundle_json else default)
        cutoff = (request.as_of or dt.datetime.now(dt.timezone.utc)).isoformat()
        try:
            policy = validate_quality_policy(configured, cutoff, market=request.market_source)
        except ValueError as error:
            if readiness["state"] != "BLOCKED":
                raise
            policy = dict(configured, period_selection_status="CONFIG_NOT_READY", configuration_error=str(error))
        quality = dict(profile_id=policy.get("profile_id", "QG_V7_BOUNDED_GAPS_R1"),
                       policy_sha256=policy.get("policy_sha256"),
                       period_selection_status=policy.get("period_selection_status", "EXPLICIT_LEGACY_PERIOD"),
                       independent_of_model_bundle=is_source_evidence_policy(policy))
    return {
        "account_equity": settings.account_equity,
        "backtrack": {
            "begin": request.beginbacktrack,
            "end": request.endbacktrack,
        },
        "date_range": ({"start": request.history_start_date.isoformat(),
                        "end": request.history_end_date.isoformat(), "inclusive": True}
                       if request.history_start_date is not None else None),
        "mail_enabled": request.mail,
        "replay_mail_enabled": request.replay_mail,
        "market_history_cache": str(settings.market_history_cache_path),
        "mops_cache": str(settings.mops_cache_dir),
        "mops_enabled": request.mops_enabled,
        "daily_strategy_workers": settings.daily_strategy_workers,
        "daily_entry_model": settings.daily_entry_model,
        "daily_exit_model": DAILY_EREM_MODEL_VERSION,
        "daily_exit_policy": SoftReductionFormalPolicy().record(),
        "les_v7_bundle_configured": settings.les_v7_bundle_json is not None,
        "les_v7_strategy_policy_configured": settings.les_v7_strategy_policy_json is not None,
        "les_v7_rule_baseline_enabled": settings.les_v7_rule_baseline,
        "les_v7_source_repair_enabled": settings.les_v7_source_repair,
        **({"industry_benchmark_sectors": sorted(json.loads(settings.industry_benchmarks_json))}
           if settings.industry_benchmarks_json else {}),
        "les_v7_readiness": readiness,
        "les_v7_quality_policy": quality,
        "output": request.output_file_names,
        "report_output": request.report_file_prefix,
        "as_of": request.as_of.isoformat() if request.as_of else None,
        "evaluation_mode": ("OFFLINE_MODEL_RESEARCH_ON_CURRENT_DATA" if settings.les_v7_research_metadata_json
                            else "HISTORICAL_REPLAY" if request.as_of else "LIVE"),
        "les_v7_research_model": (json.loads(settings.les_v7_research_metadata_json)
                                  if settings.les_v7_research_metadata_json else None),
        "pae_holding_count": len(settings.pae_positions),
        "pae_holding_symbols": [
            position.symbol for position in settings.pae_positions
        ],
        "pae_persistence": "ISOLATED_NO_HISTORY" if request.as_of else str(settings.pae_persistence_path),
        "runtime_mode": settings.runtime_mode,
        "run_lock": str(settings.run_lock_path),
        "smtp_configured": bool(
            settings.smtp_email
            and settings.smtp_password
            and settings.smtp_receivers
            and settings.smtp_server_host
        ),
        "status": "CONFIG_NOT_READY" if readiness["state"] == "BLOCKED" else "OK",
        "stock_type": request.stocktype.name,
        "market_source": request.market_source,
        "universe": ("FROZEN_MODEL_TWSE_ELEC" if settings.les_v7_research_metadata_json else
                     "HISTORICAL_SECTOR_UNION" if "ALL" in request.stocklist else "SELECTED"),
        "symbol_count": None if "ALL" in request.stocklist else len(request.stocklist),
    }


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Handle --list-strategies early exit
    if args.list_strategies:
        list_strategies()
        return 0

    if args.research_id is not None and args.les_v7_research_bundle is None:
        parser.error("--research-id requires --les-v7-research-bundle")
    if isinstance(args.beginbacktrack, dt.date) or isinstance(args.endbacktrack, dt.date):
        if args.les_v7_research_bundle is not None:
            parser.error("LES_V7_RESEARCH_CURRENT_DATA_ONLY: offline models cannot be backdated")
        if args.les_v7_strategy_policy is not None or args.les_v7_diagnostics_only:
            parser.error("LES_V7_DAILY_POLICY_OPTIONS_CURRENT_DATA_ONLY: historical replay retains its existing contract")
        from interface.replay import run_from_args
        try:
            result = run_from_args(args)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(result if args.check_config else {
            "status": result["status"], "replay_id": result["replay_id"],
            "as_of": result["as_of"], "signal_date": result["signal_date"],
            "symbols": result["symbols"], "live_order_authorized": False,
        }, ensure_ascii=False, sort_keys=True))
        return 0
    if any((args.replay_id, args.replay_evidence_file, args.replay_state_path, args.replay_output_dir)):
        parser.error("--replay-* options require a date -b/-e pair")
    
    # Validate stocklist is provided for normal operation
    if not args.stocklist and args.type != Stocktype.SEMI and args.les_v7_research_bundle is None:
        parser.error("the following arguments are required: stocklist")
    
    request, settings = build_run_configuration(argv)
    configure_structured_logging(settings.effective_log_level)
    if args.check_config:
        summary = build_preflight_summary(request, settings)
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2 if summary["status"] == "CONFIG_NOT_READY" else 0
    from application.les_v7_preflight import daily_v7_preflight
    readiness = daily_v7_preflight(settings, checked_at=dt.datetime.now(dt.timezone.utc), market=request.market_source)
    if readiness["state"] == "BLOCKED":
        parser.error("LES_V7_CONFIGURATION_NOT_READY: " + "; ".join(readiness["reason_codes"]))
    try:
        with ProcessRunLock(settings.run_lock_path):
            research_metadata = (json.loads(settings.les_v7_research_metadata_json)
                                 if settings.les_v7_research_metadata_json else None)
            if research_metadata:
                from infrastructure.research_archive import write_once
                archive = settings.pae_persistence_path.parent
                if research_metadata.get("research_id"):
                    from infrastructure.research_series import ensure_study, metadata_identity
                    ensure_study(archive, metadata_identity(research_metadata))
                    archive = archive / "runs" / research_metadata["research_run_id"]
                write_once(archive / "request.json", dict(model_use=research_metadata,
                    report_file_prefix=request.report_file_prefix, mail_enabled=request.mail,
                    research_only=True, approved_for_live=False, execution_action="NO_ACTION"))
                print("LES v7 RESEARCH ONLY: verified native archive; research report email enabled; no broker action or deployment approval.")
            build_stock_analysis_service(settings=settings).run(request)
            if research_metadata:
                if research_metadata.get("research_id"):
                    from infrastructure.research_series import check_versions
                    check_versions(metadata_identity(research_metadata))
                write_once(archive / "completion.json", dict(status="COMPLETE", model_use=research_metadata,
                    report_file_prefix=request.report_file_prefix, mail_enabled=request.mail, research_only=True,
                    approved_for_live=False, execution_action="NO_ACTION"))
                print(json.dumps(dict(status="COMPLETE", report_file_prefix=request.report_file_prefix,
                    research_archive=str(archive), research_only=True, approved_for_live=False,
                    execution_action="NO_ACTION"), ensure_ascii=False, sort_keys=True))
    except RunAlreadyActiveError as error:
        print(f"TWSE run status: ALREADY_RUNNING ({error})", file=sys.stderr)
        return 4
    except (ResearchSeriesError, ResearchStateError) as error:
        print(f"LES v7 research status: REJECTED ({error})", file=sys.stderr)
        return 2
    except TwseCalendarError as error:
        print(f"TWSE market status: UNKNOWN ({error})", file=sys.stderr)
        return 2
    except TwseClientError as error:
        print(
            f"{request.market_source} request status: {error.category.value.upper()} ({error})",
            file=sys.stderr,
        )
        return 3
    return 0
