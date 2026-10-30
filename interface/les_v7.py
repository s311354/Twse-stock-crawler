"""Offline LES v7 runner. No network, mail, schedules, PAE writes or orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from application.les_v7 import analyze_request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--bundle", type=Path, help="Hash-bound frozen JSON bundle, not the untrained YAML template")
    parser.add_argument("--output", type=Path, help="Publish to a NEW immutable JSON file; default stdout")
    parser.add_argument("--store", type=Path, help="Optional NEW research signal directory, never formal account/PAE storage")
    args = parser.parse_args(argv)
    try:
        if args.input.stat().st_size > 16 * 1024 * 1024: raise ValueError("BOUNDED_REQUEST_REQUIRED")
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if args.bundle:
            if args.bundle.stat().st_size > 1024 * 1024: raise ValueError("BOUNDED_MODEL_BUNDLE_REQUIRED")
            payload["model_bundle"] = json.loads(args.bundle.read_text(encoding="utf-8"))
        result = analyze_request(payload)
        if args.store:
            from infrastructure.storage.les_v7_repository import LesV7SignalRepository
            repository = LesV7SignalRepository(args.store)
            result["publication_receipts"] = [repository.publish(report) for report in result["reports"]]
        if args.output:
            from infrastructure.research_archive import write_once
            write_once(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
