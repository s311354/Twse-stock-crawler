"""Offline replay entry and separate evidence preparation commands."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from application.historical_replay import replay_close_cutoff
from application.replay_contracts import ReplayAccount, ReplayHolding, ReplayRequest, canonical_json
from infrastructure.replay_evidence import load_evidence, prepare_evidence
from infrastructure.replay_runtime import build_replay_engine, current_versions


def load_holdings(path: Path) -> tuple[ReplayHolding, ...]:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        records = json.loads(raw)
        try:
            return tuple(ReplayHolding(**record) for record in records)
        except TypeError as error:
            raise ValueError("INVALID_HOLDINGS_JSON: require symbol, shares, entry_at; optional entry_les, entry_decision") from error
    # Reuse the daily text syntax, but validate stricter share/date requirements
    # before constructing a replay account. JSON retains exact execution times.
    from interface.cli import load_pae_holdings_file
    return tuple(ReplayHolding(p.symbol, p.shares, p.entry_date.isoformat() if p.entry_date else "",
                               p.entry_les, p.entry_decision)
                 for p in load_pae_holdings_file(path, allow_empty=True))


def request_from_args(args) -> ReplayRequest:
    from interface.cli import load_stocklist
    if not all((args.replay_id, args.replay_evidence_file, args.pae_holdings_file)) or args.pae_current_cash is None or args.account_equity is None:
        raise ValueError("Historical replay requires --replay-id, --replay-evidence-file, --pae-holdings-file (empty allowed), --pae-current-cash and --account-equity")
    if args.mail or args.replay_mail or args.linechart or args.skip_mops:
        raise ValueError("Historical replay is offline: no mail/charts/skip-mops; missing evidence is diagnosed, not bypassed")
    if args.replay_restrictions_file:
        raise ValueError("Import restrictions into the pinned evidence package with interface.replay prepare")
    if args.include_symbols_file:
        raise ValueError("Historical replay symbols must be explicit or ALL from the pinned evidence universe")
    if not isinstance(args.beginbacktrack, dt.date) or not isinstance(args.endbacktrack, dt.date):
        raise ValueError("Historical replay requires a date -b/-e pair")
    evidence_id, payload = load_evidence(args.replay_evidence_file)
    symbols = tuple(load_stocklist(args.stocklist))
    if symbols == ("ALL",):
        symbols = tuple(payload["symbols"])
    if not set(symbols).issubset(payload["symbols"]):
        raise ValueError("REPLAY_UNIVERSE_NOT_IN_EVIDENCE")
    if args.beginbacktrack.isoformat() < payload["start"] or args.endbacktrack.isoformat() > payload["end"]:
        raise ValueError("REPLAY_RANGE_OUTSIDE_EVIDENCE")
    account = ReplayAccount(args.pae_current_cash, args.account_equity, load_holdings(args.pae_holdings_file))
    state = args.replay_state_path or Path("data/replays") / args.replay_id / "state.sqlite3"
    return ReplayRequest(args.replay_id, replay_close_cutoff(args.endbacktrack.isoformat()),
        args.beginbacktrack, args.endbacktrack, symbols, payload["market"], payload["sector"], account,
        current_versions(), evidence_id, state)


def run_from_args(args) -> dict:
    request = request_from_args(args)
    engine, _ = build_replay_engine(args.replay_evidence_file, parse_cache_path=args.replay_parse_cache)
    if args.check_config:
        return dict(status="OK", **request.inputs(), state_path=str(request.state_path), mail_enabled=False)
    result = engine.run(request)
    directory = args.replay_output_dir or Path("data/replays") / request.replay_id
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / request.as_of.strftime("%Y%m%dT%H%M%S%z")
    report_path = prefix.with_suffix(".json")
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(canonical_json(result), encoding="utf-8")
    temporary.replace(report_path)
    csv_path = prefix.with_suffix(".csv")
    columns = list(dict.fromkeys(key for row in result["rows"] for key in row))
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["replay_id", "as_of", "evaluation_mode", *columns])
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(dict(replay_id=request.replay_id, as_of=request.as_of.isoformat(), evaluation_mode="HISTORICAL_REPLAY", **row))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Freeze existing archives; this does not download financials or restrictions")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--market-cache", required=True, type=Path)
    prepare.add_argument("--mops-cache", type=Path)
    prepare.add_argument("--restriction-cache", type=Path)
    prepare.add_argument("--restriction-file", type=Path)
    prepare.add_argument("--calibration-file", type=Path, help="Optional dated empirical Half Kelly calibration records; never inferred from LES")
    prepare.add_argument("--calendar-file", type=Path, help="Prepared official calendar JSON; without it, query existing TWSE calendar client during preparation only")
    prepare.add_argument("--market", choices=("TWSE", "TPEX"), default="TWSE")
    prepare.add_argument("--sector", default="13")
    prepare.add_argument("--start", required=True, type=dt.date.fromisoformat)
    prepare.add_argument("--end", required=True, type=dt.date.fromisoformat)
    prepare.add_argument("symbols", nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.calendar_file:
            calendar = json.loads(args.calendar_file.read_text(encoding="utf-8"))
        else:
            from infrastructure.calendar.twse_calendar import TwseCalendar
            client = TwseCalendar()
            from application.replay_contracts import digest
            documents, calendars = {}, {}
            for year in range(args.start.year, args.end.year + 1):
                raw = client._fetch_year_json(year)
                documents[str(year)] = dict(url="https://www.twse.com.tw/holidaySchedule/holidaySchedule",
                    query=dict(date=f"{year}0101", response="json"), sha256=digest(raw), payload=raw,
                    retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat())
                calendars[year] = TwseCalendar(fetcher=lambda raw=raw: raw)
            calendar = dict(source="TWSE_OFFICIAL", policy="HISTORICAL_EXCHANGE_CALENDAR", documents=documents,
                days={day.isoformat(): "OPEN" if calendars[day.year].is_trading_day(day) else "CLOSED"
                      for i in range((args.end - args.start).days + 1) for day in [args.start + dt.timedelta(days=i)]})
        evidence_id = prepare_evidence(destination=args.output, market_cache=args.market_cache, mops_cache=args.mops_cache,
            restriction_cache=args.restriction_cache, restriction_file=args.restriction_file, calendar=calendar,
            symbols=tuple(args.symbols), market=args.market, sector=args.sector, start=args.start, end=args.end,
            calibration_file=args.calibration_file)
        print(canonical_json(dict(status="PREPARED", evidence_id=evidence_id, path=str(args.output))))
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
