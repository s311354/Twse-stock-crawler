"""Manual, no-mail recorded-account EREM research comparison; not a backtest."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

from application.erem_research import compare_recorded_erem
from infrastructure.research_archive import write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.resolve() == args.bank_run.resolve():
        parser.error("research output must not replace its source")
    guard = ResearchVersionGuard()
    content = args.bank_run.read_bytes()
    run = json.loads(gzip.decompress(content) if args.bank_run.suffix == ".gz" else content)
    result = compare_recorded_erem(run)
    result["source_file_sha256"] = hashlib.sha256(content).hexdigest()
    result["research_execution_versions"] = asdict(guard.versions)
    guard.check()
    if args.output:
        objects = ResearchObjects(args.output.parent / "objects")
        result["execution_objects"] = guard.retain(objects)
        write_once(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
