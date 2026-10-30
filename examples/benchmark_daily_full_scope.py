"""Offline full native daily cold/warm benchmark over a frozen report universe.

This command never invokes the live crawler, notifier, broker or formal PAE
store. The saved source account is an explicit benchmark input, not a statement
about a current account. A fresh work context is used for each measured round.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time

from application.replay_contracts import digest
from infrastructure.research_archive import write_once


SCHEMA = "daily-full-scope-benchmark/v1"


@contextmanager
def offline_only():
    """Fail closed if a mistakenly connected adapter attempts network I/O."""
    connect, connect_ex, create = socket.socket.connect, socket.socket.connect_ex, socket.create_connection

    def denied(*args, **kwargs):
        raise RuntimeError("OFFLINE_BENCHMARK_NETWORK_FORBIDDEN")

    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = denied
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = connect, connect_ex, create


def harness_identity():
    root = Path(__file__).resolve().parent
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                   ("benchmark_daily_full_scope.py", "daily_fixed_fixture.py", "daily_fixed_pipeline.py")})


def compare_measurements(rows, fields):
    if len(rows) < 2:
        raise ValueError("FULL_SCOPE_BENCHMARK_REQUIRES_COLD_AND_WARM")
    for key in fields:
        values = [row.get(key) for row in rows]
        if any(value is None for value in values) or any(value != values[0] for value in values[1:]):
            raise ValueError("FULL_SCOPE_COMPLETE_PARITY_FAILED:" + key)
    if len({row["input_id"] for row in rows}) != 1:
        raise ValueError("FULL_SCOPE_INPUT_ID_CHANGED")


def worker(args):
    import numpy as np
    import pandas as pd
    from examples.daily_fixed_fixture import validate_fixture
    from examples.daily_fixed_pipeline import PARITY_FIELDS, run_round
    from infrastructure.research_objects import ResearchVersionGuard

    began = time.perf_counter()
    code = ResearchVersionGuard()
    harness = harness_identity()
    with offline_only():
        manifest = validate_fixture(args.fixture)
        verified_seconds = time.perf_counter() - began
        rows = []
        for iteration in range(args.rounds):
            code.check()
            if harness_identity() != harness:
                raise ValueError("FULL_SCOPE_HARNESS_CHANGED")
            print(json.dumps(dict(event="round_started", iteration=iteration,
                temperature="cold_empty_derived_cache" if iteration == 0 else "warm_new_work_context",
                symbols=len(manifest["symbols"])), ensure_ascii=False), flush=True)
            def progress(event):
                name, calls = event["name"], event.get("calls", 0)
                outer = name in {"fundamentals", "strategies", "pae", "full_result_hashing",
                                 "report", "complete_result_verification"}
                milestone = name in {"fundamental_historical_derivation", "les_and_valuation"} and calls % 50 == 0
                if outer or event["event"] == "finished" and milestone:
                    value = dict(event="stage_progress", iteration=iteration, name=name,
                        phase=event["event"], calls=calls, seconds=event.get("seconds"),
                        status=event.get("status"), input_id=manifest["input_id"])
                    encoded = json.dumps(value, ensure_ascii=False)
                    # This append-only diagnostic is not a result or evidence
                    # identity. A partial log can never publish complete.json.
                    with (args.run_dir / "stage-events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(encoded + "\n")
                    print(encoded, flush=True)

            row = run_round(fixture=args.fixture, manifest=manifest,
                output=args.run_dir / "reports", workspace=args.workspace / "staging",
                cache_path=args.workspace / "derived-cache.sqlite3", iteration=iteration,
                stage_callback=progress)
            code.check()
            if harness_identity() != harness:
                raise ValueError("FULL_SCOPE_HARNESS_CHANGED")
            rows.append(row)
            write_once(args.run_dir / ("round-%02d.json" % iteration), row)
            print(json.dumps(dict(event="round_completed", iteration=iteration,
                stages=row.get("stages"), counters=row.get("counters")), ensure_ascii=False), flush=True)
        compare_measurements(rows, PARITY_FIELDS)
        report = dict(schema=SCHEMA, scope="FIXED_SOURCE_NATIVE_LES_VALUATION_EREM_PAE_AND_REPORT_STORAGE",
            input_id=manifest["input_id"], symbol_count=len(manifest["symbols"]),
            symbols=manifest["symbols"], fixture=str(args.fixture),
            evaluation_cutoff=manifest["evaluation_cutoff"],
            fixture_verification_seconds=verified_seconds, total_worker_seconds=time.perf_counter() - began,
            source_versions=vars(code.versions), harness_sha256=harness,
            python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
            sqlite=sqlite3.sqlite_version, platform=platform.platform(),
            workspace_device=args.workspace.stat().st_dev, output_device=args.run_dir.stat().st_dev,
            parity_fields=list(PARITY_FIELDS), full_output_equal=True, measurements=rows,
            limitations=[
                "Same current source/model/parameters on both routes; not old/new policy performance.",
                "Cold means empty derived cache, not a flushed operating-system filesystem cache.",
                "Warm starts a new work context with the same bounded disk cache and retained evidence store.",
                "Frozen saved account and empty research prestate; no formal account or multi-day PAE replay.",
                "Counterfactual fixed-source evaluation, not certification of historical deployment.",
                "Source gaps stay UNKNOWN/PARTIAL; trade counts and profit are not acceptance criteria.",
                "Nested stage seconds must not be summed; RSS is process lifetime high-water.",
                "Fixture creation and full initial integrity verification are reported separately.",
                "No email, network, orders, scheduling or formal state mutation."])
        write_once(args.run_dir / "benchmark.json", report)
        write_once(args.run_dir / "complete.json", dict(schema=SCHEMA, input_id=manifest["input_id"],
            benchmark_sha256=digest(report), full_output_equal=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-bundle", type=Path)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--mops-db", type=Path)
    parser.add_argument("--fixture", type=Path, help="Reuse an already frozen, fully verified fixture.")
    parser.add_argument("--evaluation-cutoff", default="2026-10-16T15:16:04+00:00")
    parser.add_argument("--symbols", nargs="+", help="Explicit small smoke-test subset; omit for whole saved universe.")
    parser.add_argument("--run-dir", type=Path, required=True, help="New isolated durable output directory.")
    parser.add_argument("--workspace-parent", type=Path, help="Existing internal SSD directory with sufficient free space.")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if not 2 <= args.rounds <= 5:
        parser.error("Use 2-5 cold/warm rounds")
    if args.worker:
        if not args.fixture or not args.workspace:
            parser.error("Worker requires fixture and workspace")
        worker(args)
        return
    if args.run_dir.exists():
        parser.error("Benchmark outputs are immutable; choose a new run directory")
    if args.fixture and any((args.report_bundle, args.mops_db, args.bundle_sha256, args.symbols)):
        parser.error("A frozen fixture cannot be mixed with new source selections")
    if not args.fixture and not all((args.report_bundle, args.mops_db, args.bundle_sha256)):
        parser.error("Provide --fixture or --report-bundle, --bundle-sha256 and --mops-db")
    args.run_dir.mkdir(parents=True)
    # Intentionally retained for reproducibility, not auto-deleted on failure.
    workspace = Path(tempfile.mkdtemp(prefix="twse-full-daily-", dir=args.workspace_parent)).resolve()
    began = time.perf_counter()
    try:
        if args.fixture:
            fixture = args.fixture.resolve()
        else:
            from examples.daily_fixed_fixture import freeze_fixture
            fixture = workspace / "fixed-inputs"
            with offline_only():
                freeze_fixture(args.report_bundle, args.mops_db, fixture,
                    expected_bundle_sha256=args.bundle_sha256, symbols=args.symbols,
                    evaluation_cutoff=args.evaluation_cutoff)
        write_once(args.run_dir / "command.json", dict(schema=SCHEMA, argv=sys.argv,
            fixture=str(fixture), workspace=str(workspace), freeze_seconds=time.perf_counter() - began,
            harness_sha256=harness_identity()))
        command = [sys.executable, "-m", "examples.benchmark_daily_full_scope", "--worker",
            "--fixture", str(fixture), "--run-dir", str(args.run_dir), "--workspace", str(workspace),
            "--rounds", str(args.rounds)]
        subprocess.run(command, check=True)
        report = json.loads((args.run_dir / "benchmark.json").read_bytes())
        print(json.dumps(dict(complete=True, full_output_equal=report["full_output_equal"],
            symbols=report["symbol_count"], report=str(args.run_dir / "benchmark.json")), ensure_ascii=False))
    except BaseException as error:
        write_once(args.run_dir / "failed.json", dict(schema=SCHEMA,
            error_type=type(error).__name__, error=str(error), workspace=str(workspace),
            elapsed_seconds=time.perf_counter() - began, complete=False))
        raise


if __name__ == "__main__":
    main()
