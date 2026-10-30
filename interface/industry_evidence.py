"""Manual official price-index capture/verification; never enables R11 alone."""
import argparse
import json
from pathlib import Path

from infrastructure.crawler.industry_index import capture_index, capture_indices, verify_capture, SECTOR_SCHEMA


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/industry-evidence"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true", help="Explicit bounded official monthly downloads")
    mode.add_argument("--verify", type=Path, help="Offline verification of an immutable capture manifest")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-months", type=int, default=4)
    parser.add_argument("--sectors", nargs="+",
        help="Explicit official sector codes/names; shared monthly downloads, still no membership proof")
    args = parser.parse_args(argv)
    if args.capture:
        if not args.start or not args.end:
            parser.error("--capture requires --start and --end (ISO dates)")
        if args.sectors is not None:
            paths = capture_indices(args.root, sectors=args.sectors, start=args.start, end=args.end,
                max_months=args.max_months)
            captures = []
            for sector, path in paths.items():
                result = verify_capture(path, args.root)
                captures.append(dict(manifest=str(path), capture_id=result["capture_id"],
                    sector=sector, index_name=result["index_name"], schema=result["schema"],
                    status=result["status"], rows=len(result["rows"]),
                    excluded_rows=result["excluded_rows"], gaps=result["gaps"],
                    eligible_for_industry_scoring=result["eligible_for_industry_scoring"]))
            print(json.dumps(dict(captures=captures, requests=len(result["documents"]),
                no_accounts_or_mail=True), ensure_ascii=False))
            return 0
        path = capture_index(args.root, start=args.start, end=args.end, max_months=args.max_months)
    else:
        if args.start or args.end or args.sectors:
            parser.error("--verify uses the retained manifest range/sector; do not supply --start/--end/--sectors")
        path = args.verify
    result = verify_capture(path, args.root)
    print(json.dumps(dict(manifest=str(path), capture_id=result["capture_id"], status=result["status"],
        rows=len(result["rows"]), requests=len(result["documents"]) if args.capture else 0,
        excluded_rows=result["excluded_rows"], gaps=result["gaps"],
        eligible_for_industry_scoring=result["eligible_for_industry_scoring"], no_accounts_or_mail=True,
        **(dict(sector=result["sector"], index_name=result["index_name"], schema=result["schema"])
           if result["schema"] == SECTOR_SCHEMA else {})), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
