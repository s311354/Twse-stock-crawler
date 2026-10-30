"""Read-only four-account readiness audit; never runs research, fills or mail."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

from application.exit_comparison_readiness import audit_exit_comparison
from application.replay_contracts import digest
from infrastructure.research_archive import write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bank-run", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--bank-engine", type=Path, required=True)
    parser.add_argument("--bank-prices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    files = {key: getattr(args, key) for key in ("report", "bank_run", "bank_manifest", "bank_engine", "bank_prices")}
    if args.output.resolve() in {path.resolve() for path in files.values()}:
        parser.error("audit must not replace its inputs")
    guard = ResearchVersionGuard()
    content = {key: path.read_bytes() for key, path in files.items()}
    run = json.loads(gzip.decompress(content["bank_run"]) if args.bank_run.suffix == ".gz" else content["bank_run"])
    result = audit_exit_comparison(content["report"].decode("utf-8-sig"), run,
        json.loads(content["bank_manifest"]), engine_text=content["bank_engine"].decode(),
        prices_text=content["bank_prices"].decode())
    result["input_sha256"] = {key: hashlib.sha256(value).hexdigest() for key, value in content.items()}
    result["audit_execution_versions"] = asdict(guard.versions)
    # Refuse concurrent source changes; the report never mutates any source.
    if any(path.read_bytes() != content[key] for key, path in files.items()):
        raise ValueError("EXIT_COMPARISON_INPUT_CHANGED_DURING_AUDIT")
    guard.check()
    objects = ResearchObjects(args.output.parent / "objects")
    result["audit_execution_objects"] = guard.retain(objects)
    result["audit_id"] = digest({key: value for key, value in result.items() if key != "audit_id"})
    write_once(args.output, result)
    print(json.dumps(dict(status=result["status"], audit_id=result["audit_id"],
        promotion_decision=result["promotion_decision"], arms={row["arm"]: row["status"] for row in result["arms"]},
        gaps=[row["code"] for row in result["gaps"]]), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
