"""Read-only Bank CSV attribution. No mail, research run, account or order port."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from application.simulation_performance_audit import audit_simulation_report
from infrastructure.research_archive import write_once


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--bank-run", type=Path, help="Optional exact saved result JSON or JSON.gz; read-only.")
    parser.add_argument("--output", type=Path, help="Optional immutable JSON audit; never overwrites changed content.")
    args = parser.parse_args(argv)
    text = args.report.read_text(encoding="utf-8-sig")
    saved = None
    if args.bank_run:
        content = args.bank_run.read_bytes()
        saved = json.loads(gzip.decompress(content) if args.bank_run.suffix == ".gz" else content)
    result = audit_simulation_report(text, bank_run=saved)
    result["source_files"] = {"report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest()}
    if args.bank_run:
        result["source_files"]["bank_run_sha256"] = hashlib.sha256(content).hexdigest()
    if args.output:
        if args.output.resolve() in {args.report.resolve(), args.bank_run.resolve() if args.bank_run else args.report.resolve()}:
            parser.error("audit output must not replace a source file")
        write_once(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
