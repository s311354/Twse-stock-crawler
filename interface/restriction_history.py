"""Collect official history, or validate/reconstruct a reviewed evidence bundle."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path

from application.historical_replay import replay_close_cutoff
from application.restriction_history import HistoricalRestrictionLedger
from infrastructure.crawler.restriction_history import HistoricalRestrictionCollector
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig, TwseClientError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="Opt-in TWSE network collection, never marks coverage VERIFIED")
    collect.add_argument("--start", type=dt.date.fromisoformat, required=True)
    collect.add_argument("--end", type=dt.date.fromisoformat, required=True)
    collect.add_argument("--output", type=Path, required=True)
    rebuild = sub.add_parser("reconstruct", help="Offline raw-proof validation and point-in-time reconstruction")
    rebuild.add_argument("--bundle", type=Path, required=True)
    rebuild.add_argument("--market", choices=("TWSE", "TPEX"), required=True)
    rebuild.add_argument("-e", "--end", type=replay_close_cutoff, required=True, metavar="YYYY-MM-DD",
                         help="Replay date, fixed at Taipei 13:30 close; timestamps are not supported")
    rebuild.add_argument("--market-data-date", type=dt.date.fromisoformat, required=True)
    rebuild.add_argument("--symbols", nargs="+", required=True)
    rebuild.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Refuse to overwrite evidence or reports, even before network access.
        if args.output.exists():
            raise ValueError("OUTPUT_ALREADY_EXISTS")
        if args.command == "collect":
            client = TwseHttpClient(TwseHttpConfig(max_attempts=1, min_request_interval_seconds=1.,
                                                  allow_insecure_ssl_fallback=False))
            try:
                result = HistoricalRestrictionCollector(client).collect(start=args.start, end=args.end)
            finally:
                client.session.close()
            summary = {"collection_status": result["collection_status"], "documents": len(result["documents"])}
        else:
            ledger = HistoricalRestrictionLedger.from_file(args.bundle)
            result = ledger.reconstruct(args.symbols, market=args.market, as_of=args.end,
                                        market_data_date=args.market_data_date.isoformat())
            summary = dict(Counter(row["reconstruction_result"]["status"] for row in result.values()))
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as output:
            output.write(text)
        print(json.dumps({"output": str(args.output.resolve()), "summary": summary}, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError, TwseClientError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
