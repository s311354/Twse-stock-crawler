"""Offline research archive tools. No mail, trading, scheduler or account ledger."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from application.historical_replay import replay_close_cutoff
from application.replay_contracts import ReplayAccount, ReplayRequest, calendar_window, canonical_json
from application.research_audit import coverage_audit, summarize
from infrastructure.daily_research import retain_research, reproduce_research
from infrastructure.replay_evidence import collect_evidence, load_evidence
from infrastructure.replay_runtime import build_replay_engine, current_versions
from infrastructure.research_archive import create_dataset, write_once, utc_now
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.storage.replay_state import SQLiteReplayState
from interface.replay import load_holdings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    from interface.filing_publication import register_commands, run_command
    register_commands(subs)
    daily = subs.add_parser("daily-audit", help="Freeze and audit the FULL retained daily report; no replay or current-cache substitution")
    daily.add_argument("--analysis-csv", required=True, type=Path)
    daily.add_argument("--reports", required=True, type=Path)
    daily.add_argument("--stocklist", required=True, type=Path)
    daily.add_argument("--stocklist-format", required=True, choices=("symbols", "legacy-indices"))
    daily.add_argument("--output", required=True, type=Path)
    create = subs.add_parser("dataset", help="Freeze raw cache into a new content-addressed dataset (no HTTP)")
    create.add_argument("--store", required=True, type=Path)
    create.add_argument("--market-cache", required=True, type=Path)
    create.add_argument("--mops-cache", type=Path)
    create.add_argument("--restriction-cache", type=Path)
    create.add_argument("--restriction-file", type=Path)
    create.add_argument("--calibration-file", type=Path)
    create.add_argument("--publication-file", type=Path)
    create.add_argument("--lifecycle-file", type=Path)
    create.add_argument("--calendar-file", required=True, type=Path)
    create.add_argument("--parent", type=Path)
    create.add_argument("--start", required=True, type=dt.date.fromisoformat)
    create.add_argument("--end", required=True, type=dt.date.fromisoformat)
    create.add_argument("--market", choices=("TWSE", "TPEX"), default="TWSE")
    create.add_argument("--sector", default="13")
    create.add_argument("symbols", nargs="+")
    for command in ("dataset-migrate", "dataset-verify"):
        sub = subs.add_parser(command, help="Offline source-document reference migration / integrity check")
        sub.add_argument("--evidence", required=True, type=Path)
        if command == "dataset-migrate":
            sub.add_argument("--store", required=True, type=Path)
    for command in ("audit", "capture"):
        sub = subs.add_parser(command)
        sub.add_argument("--evidence", required=True, type=Path)
        sub.add_argument("--date", required=True, type=dt.date.fromisoformat)
        sub.add_argument("--start", type=dt.date.fromisoformat,
                         help="Inclusive history start; omitted means date minus 699 calendar days, no automatic padding")
        sub.add_argument("--holdings", required=True, type=Path)
        sub.add_argument("--cash", required=True, type=float)
        sub.add_argument("--equity", required=True, type=float)
        sub.add_argument("--run-id", required=True)
        sub.add_argument("--output", required=True, type=Path)
        if command == "capture":
            group = sub.add_mutually_exclusive_group(required=True)
            group.add_argument("--predecessor", type=Path)
            group.add_argument("--initial-state", action="store_true", help="Explicit no prior PAE observations; not an account default")
    reproduce = subs.add_parser("reproduce")
    reproduce.add_argument("--archive", required=True, type=Path)
    reproduce.add_argument("--evidence", required=True, type=Path)
    reproduce.add_argument("--output", required=True, type=Path)
    case = subs.add_parser("case-audit", help="Read-only audit of an existing BankEmulator run")
    case.add_argument("--run", required=True, type=Path)
    case.add_argument("--output", required=True, type=Path)
    case.add_argument("symbols", nargs="+")
    for command in ('job-capture', 'job-reproduce', 'job-export'):
        sub = subs.add_parser(command, help='Offline job-scoped retention; no email or fills')
        sub.add_argument('--evidence', type=Path, required=True)
        sub.add_argument('--job', type=Path, required=True)
        sub.add_argument('--objects', type=Path, help='Optional shared immutable object store')
        sub.add_argument('--parse-cache', type=Path, help='Optional shared derived SQLite parse/index cache; never an MOPS or PAE database')
        if command == 'job-capture':
            sub.add_argument('--requests', type=Path, required=True, help='JSON array of complete ReplayRequest.inputs() documents')
            sub.add_argument('--no-cache', action='store_true')
        else:
            sub.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "daily-audit":
            from infrastructure.daily_report_audit import retain_daily_report_audit
            folder = retain_daily_report_audit(args.analysis_csv, args.reports, args.stocklist,
                args.output, stocklist_format=args.stocklist_format)
            print(canonical_json(dict(status="REPORT_ONLY_INPUT_PARITY_UNVERIFIABLE",
                baseline_id=folder.name, output=str(folder))))
        elif args.command.startswith("filing-"):
            run_command(args)
        elif args.command in {"dataset-migrate", "dataset-verify"}:
            eid, payload = load_evidence(args.evidence)
            documents = EvidenceDocuments.for_evidence(args.evidence, eid)
            verified = documents.verify(payload)
            if args.command == "dataset-verify":
                print(canonical_json(dict(evidence_id=eid, **verified)))
            else:
                path = create_dataset(payload, args.store, parent=args.evidence, document_reader=documents.read)
                print(canonical_json(dict(status="MIGRATED", evidence=str(path), dataset_id=path.parent.name,
                                          parent_dataset_id=eid if eid != path.parent.name else None)))
        elif args.command.startswith('job-'):
            from infrastructure.research_job import ResearchJob
            from infrastructure.daily_research import request_from_document
            with ResearchJob(args.evidence, args.job, object_root=args.objects,
                             cache_enabled=not getattr(args, 'no_cache', False), parse_cache_path=args.parse_cache) as job:
                if args.command == 'job-capture':
                    documents = json.loads(args.requests.read_text())
                    if not isinstance(documents, list) or not documents:
                        raise ValueError('EXPLICIT_DAILY_REQUEST_ARRAY_REQUIRED')
                    for document in documents:
                        req = request_from_document(document, args.job/'IGNORED.sqlite3')
                        day = job.run_day(req)
                        print(canonical_json(day.receipt))
                elif args.command == 'job-reproduce':
                    result = job.reproduce()
                    write_once(args.output, result)
                    print(canonical_json(result))
                    return 0 if result['equal'] else 1
                else:
                    job.export_bundle(args.output)
                    print(canonical_json(dict(status='EXPORTED', path=str(args.output))))
        elif args.command == "dataset":
            started = utc_now()
            documents = EvidenceDocuments(args.store / "objects")
            payload = collect_evidence(market_cache=args.market_cache, mops_cache=args.mops_cache,
                restriction_cache=args.restriction_cache, restriction_file=args.restriction_file,
                calendar=json.loads(args.calendar_file.read_text()), symbols=tuple(args.symbols),
                market=args.market, sector=args.sector, start=args.start, end=args.end,
                calibration_file=args.calibration_file, raw=True, publication_file=args.publication_file,
                document_store=documents.store)
            if args.lifecycle_file:
                payload["security_lifecycle"] = json.loads(args.lifecycle_file.read_text())
            path = create_dataset(payload, args.store, parent=args.parent, collected_at=started)
            print(canonical_json(dict(status="FROZEN", evidence=str(path), dataset_id=path.parent.name)))
        elif args.command == "reproduce":
            result = reproduce_research(args.archive, args.evidence)
            write_once(args.output, result)
            print(canonical_json(result))
            return 0 if result["equal"] else 1
        elif args.command == "case-audit":
            from infrastructure.research_case import audit_existing_case
            result = audit_existing_case(args.run, tuple(args.symbols))
            write_once(args.output, result)
            print(canonical_json(dict(status=result["status"], source_run=result["source_run"],
                                      report_count=len(result["days"]), output=str(args.output))))
        else:
            eid, payload = load_evidence(args.evidence)
            start, end = calendar_window(args.date)
            if args.start is not None:
                start = args.start
            account = ReplayAccount(args.cash, args.equity, load_holdings(args.holdings))
            request = ReplayRequest(args.run_id, replay_close_cutoff(args.date.isoformat()), start, end,
                tuple(payload["symbols"]), payload["market"], payload["sector"], account,
                current_versions(), eid, args.output / "unused.sqlite3")
            if args.command == "audit":
                engine, _ = build_replay_engine(args.evidence)
                selected = engine.evidence.select(request)
                result = coverage_audit(request, selected)
                write_once(args.output / "coverage.json", result)
                # Human output is a rendering of the immutable JSON, never input.
                text = summarize(result)
                path = args.output / "coverage.md"
                if path.exists():
                    if path.read_text() != text: raise ValueError("IMMUTABLE_AUDIT_CONFLICT")
                else:
                    with path.open("x") as stream: stream.write(text)
                print(text)
            else:
                predecessor = (json.loads(args.predecessor.read_text()) if args.predecessor
                               else SQLiteReplayState.empty_snapshot(request))
                folder = retain_research(args.evidence, request, args.output, predecessor=predecessor)
                print(canonical_json(dict(status="RETAINED", archive=str(folder))))
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
