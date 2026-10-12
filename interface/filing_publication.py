"""Subcommands for the existing research CLI, not a separate research engine."""
import json
from pathlib import Path

from application.replay_contracts import aware_timestamp, canonical_json, digest
from infrastructure.filing_evidence_collection import collect_sources
from infrastructure.publication_repair import audit_dataset, review_repair, publish_repair, compare_datasets
from infrastructure.research_archive import write_once, write_bytes_once


def register_commands(subs):
    parser = subs.add_parser("filing-pdf-fetch", help="Online bounded/resumable official PDF collection; no auto-proof")
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--filing-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path, help="Separate collection directory")
    parser = subs.add_parser("filing-pdf-compare", help="Offline original PDF/iXBRL field comparison; not publication proof")
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--filing-sha256", required=True)
    parser.add_argument("--pdf-document", required=True, type=Path, help="Official PDF receipt JSON and sibling objects/")
    parser.add_argument("--mapping", type=Path, help="Reviewed page/column/context mappings; omit to inventory unmapped facts")
    parser.add_argument("--output", required=True, type=Path)
    for command in ("filing-audit", "filing-verify", "filing-publish", "filing-pdf-policy"):
        parser = subs.add_parser(command, help="Offline historical publication/version/comparability audit")
        parser.add_argument("--evidence", required=True, type=Path)
        parser.add_argument("--as-of", required=True, type=aware_timestamp)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("symbols", nargs="+")
        parser.add_argument("--collection", type=Path, help="Optional reports/<hash>.json from filing-collect")
        if command == "filing-audit":
            parser.add_argument("--summary-output", type=Path, help="Optional write-once Markdown field-gap summary")
        if command != "filing-audit":
            parser.add_argument("--review", required=True, type=Path)
        if command in {"filing-publish", "filing-pdf-policy"}:
            parser.add_argument("--store", required=True, type=Path)
        if command == "filing-pdf-policy":
            parser.add_argument("--accept-pdf-upload-assumption", action="store_true", required=True,
                                help="Use official PDF upload time as an explicit unverified iXBRL version assumption")
    parser = subs.add_parser("filing-collect", help="Online bounded collection; never automatically verifies publication")
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-requests", default=12, type=int)
    parser.add_argument("--retry-unresolved", action="store_true")
    parser = subs.add_parser("filing-compare", help="Paired offline native research, fixed explicit accounts/models")
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-state", required=True, action="store_true")
    parser.add_argument("--no-cache", action="store_true")


def run_command(args):
    if args.command == "filing-pdf-fetch":
        from infrastructure.filing_pdf_comparison import collect_filing_pdf
        result = collect_filing_pdf(args.evidence, symbol=args.symbol, filing_sha256=args.filing_sha256,
                                   output=args.output)
        print(canonical_json(result))
        return
    elif args.command == "filing-pdf-compare":
        from infrastructure.filing_pdf_comparison import compare_filing_pdf
        result = compare_filing_pdf(args.evidence, symbol=args.symbol, filing_sha256=args.filing_sha256,
                                   pdf_document=args.pdf_document, mapping_file=args.mapping)
    elif args.command == "filing-audit":
        result = audit_dataset(args.evidence, tuple(args.symbols), args.as_of, collection_file=args.collection)
    elif args.command == "filing-collect":
        audit = json.loads(args.audit.read_text())
        result = collect_sources(audit["collection_plan"], args.output,
            max_requests=args.max_requests, retry_unresolved=args.retry_unresolved)
        path = args.output / "reports" / f"{digest(result)}.json"
        print(canonical_json(dict(status=result["status"], report=str(path), counts={
            state: sum(r["status"] == state for r in result["results"])
            for state in sorted({r["status"] for r in result["results"]})})))
        return
    elif args.command == "filing-compare":
        result = compare_datasets(args.before, args.after, json.loads(args.requests.read_text()), args.output,
                                  cache_enabled=not args.no_cache, initial_state=args.initial_state)
        print(canonical_json(dict(report=str(args.output / 'comparison.json'), days=len(result["days"]))))
        return
    else:
        kwargs = dict(as_of=args.as_of, symbols=tuple(args.symbols), collection_file=args.collection)
        if args.command == "filing-pdf-policy":
            _, result = publish_repair(args.evidence, args.review, args.store,
                                       accept_pdf_upload_policy=args.accept_pdf_upload_assumption, **kwargs)
        elif args.command == "filing-publish":
            _, result = publish_repair(args.evidence, args.review, args.store, **kwargs)
        else:
            _, _, result = review_repair(args.evidence, args.review, **kwargs)
    write_once(args.output, result)
    if args.command == "filing-audit" and args.summary_output:
        from application.filing_publication import render_filing_audit
        write_bytes_once(args.summary_output, render_filing_audit(result).encode("utf-8"))
    print(canonical_json(dict(status=result["status"], report=str(args.output),
                              dataset_id=result.get("dataset_id"), difference=result.get("difference"))))
