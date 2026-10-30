"""Manual bounded current-source capture/audit; no market analysis or mail."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.official_pe import OfficialPeIndex, session_date
from domain.les_v7.source_policy import source_policy
from infrastructure.crawler.official_pe import OfficialPeRepository
from infrastructure.crawler.security_master import load_security_master, prepare_security_master
from infrastructure.research_archive import write_once
from infrastructure.runtime.native_source_capture import NativeSourceCapture


def source_audit(capture, crawler, *, cutoff):
    """Source statuses only, not a Quality/LES verdict or vintage certificate."""
    timestamp(cutoff)
    rows = capture.selections(crawler, cutoff=cutoff)
    summary = dict(security_status_counts=dict(Counter(
        row["security"]["status"] if row["security"] else "UNKNOWN" for row in rows.values())),
        pe_status_counts=dict(Counter(node["status"] for row in rows.values() for node in row["official_pe"])))
    body = dict(schema="native-source-repair-audit/v1", data_kind="OFFICIAL_CURRENT_CAPTURE",
        cutoff=cutoff, market=capture.market, session_dates=list(crawler.iso_scheduled_times),
        symbols=list(crawler.stocknumbers), rows=rows, summary=summary,
        source_policy=source_policy(), source_manifest=capture.master_manifest,
        diagnostics=list(capture.diagnostics), historical_information_set_certified=False,
        no_accounts_or_mail=True, scope="SOURCE_COMPONENTS_NOT_FULL_LES_RECOMPUTATION")
    return dict(body, audit_id=identity(body))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/native-source"))
    parser.add_argument("--market", choices=("TWSE", "TPEX"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--capture", action="store_true", help="Explicitly fetch two master batches and one dated P/E table")
    parser.add_argument("--master-manifest", type=Path, help="Cache-only mode requires an exact retained manifest")
    parser.add_argument("--cutoff", help="Aware cutoff; default is actual clock after capture")
    parser.add_argument("--output", type=Path, required=True, help="New immutable JSON; never an existing report")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("NATIVE_SOURCE_AUDIT_OUTPUT_EXISTS")
    if not 1 <= len(args.symbols) <= 2000 or len(set(args.symbols)) != len(args.symbols):
        raise ValueError("NATIVE_SOURCE_AUDIT_EXPLICIT_UNIQUE_SYMBOLS_REQUIRED")
    if args.output.resolve().is_relative_to((args.root / "security-master").resolve()) or args.output.resolve() == (args.root / "official-pe.sqlite3").resolve():
        raise ValueError("NATIVE_SOURCE_AUDIT_MUST_NOT_OVERWRITE_ORIGINALS")
    day = session_date(args.session)
    crawler = SimpleNamespace(iso_scheduled_times=[day], stocknumbers=args.symbols,
                              daily_dates=[[day] for _ in args.symbols])
    capture = NativeSourceCapture(args.root)
    if args.capture:
        if args.master_manifest:
            raise ValueError("NATIVE_SOURCE_CAPTURE_CANNOT_MIX_EXISTING_MASTER")
        capture.collect(crawler, market=args.market)
    else:
        if args.master_manifest is None:
            raise ValueError("CACHE_ONLY_MASTER_MANIFEST_REQUIRED")
        packet, documents = load_security_master(args.master_manifest)
        if packet["market"] != args.market:
            raise ValueError("NATIVE_SOURCE_AUDIT_MASTER_MARKET_MISMATCH")
        capture.market = args.market
        capture.master = prepare_security_master(packet, document_reader=documents.read, check_sources=documents.check)
        capture.master_manifest = str(args.master_manifest)
        at = args.cutoff or datetime.now(timezone.utc).isoformat()
        capture.pe_index = OfficialPeIndex(OfficialPeRepository(args.root / "official-pe.sqlite3").load_as_of(
            market=args.market, trade_dates=[day], cutoff=at))
    cutoff = timestamp(args.cutoff or datetime.now(timezone.utc).isoformat()).isoformat()
    if timestamp(cutoff) > datetime.now(timezone.utc):
        raise ValueError("NATIVE_SOURCE_AUDIT_FUTURE_CUTOFF_NOT_ALLOWED")
    result = source_audit(capture, crawler, cutoff=cutoff)
    write_once(args.output, result)
    print(json.dumps(dict(output=str(args.output), audit_id=result["audit_id"], cutoff=cutoff,
                          summary=result["summary"], diagnostics=result["diagnostics"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
