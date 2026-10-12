"""Offline native serial/process comparison, not official daily research.

Optional original MOPS snapshots are read-only; prices and account are explicitly
synthetic. No HTTP, official cache/report/state writes, mail or trades. Process
startup, packet encoding, IPC, worker shutdown and PAE all count in total time.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance, measure_stage
from application.daily_work import DailyWorkContext
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from examples.benchmark_daily_computation import source_identity
from examples.benchmark_evidence_context import benchmark_market
from infrastructure.daily_parallel import DailyProcessExecutor


def worker(args):
    market, sources = benchmark_market(args)
    measurements = []
    # Both paths use cold per-analysis caches, as native daily jobs do. Cache
    # capacity is equal per process; process mode has additional worker memory.
    for iteration in range(args.rounds):
        probe = DailyPerformance()
        work = DailyWorkContext("parallel-benchmark", max_bytes=16 * 1024 * 1024)
        try:
            with work.activate(), probe.activate(), measure_stage("total"):
                executor = DailyProcessExecutor(args.workers) if args.workers > 1 else None
                decisions = AnalysisDatasetBuilder(stock_executor=executor).build_decisions(
                    market, [[1, 1, 1, 1, .01]] * market.stock_count)
                with measure_stage("pae"):
                    decisions = PortfolioAllocationV2Pipeline().enrich(decisions,
                        PortfolioRuntimeContext(account_equity=1_000_000., current_cash=1_000_000.,
                                                positions=(), default_sector="ELEC"))
                with measure_stage("serialization"):
                    body = json.dumps([asdict(row) for row in decisions], sort_keys=True,
                                      ensure_ascii=False, allow_nan=False, default=str).encode()
            measurements.append(dict(iteration=iteration, temperature="first_process_run" if iteration == 0 else "warm_os_new_pool",
                result_sha256=hashlib.sha256(body).hexdigest(), result_bytes=len(body), **probe.snapshot()))
        finally:
            work.close()
    return dict(workers=args.workers, inputs=sources, measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mops-cache", type=Path)
    parser.add_argument("--symbols", nargs="+", default=["2330", "3019"])
    parser.add_argument("--cutoff", default="2026-10-07T13:30:00+08:00")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--metadata-nodes", type=int, default=2000)
    parser.add_argument("--compare-workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 2 <= len(args.symbols) <= 16 or len(set(args.symbols)) != len(args.symbols):
        parser.error("Use 2-16 distinct symbols, not the full universe")
    if not 1 <= args.rounds <= 3 or args.metadata_nodes < 0 or not 2 <= args.compare_workers <= 4:
        parser.error("Use 1-3 rounds, nonnegative metadata and 2-4 comparison workers")
    if args.output and args.output.exists():
        parser.error("Choose a new output path; results are not overwritten")
    if args.workers is not None:
        print(json.dumps(worker(args), allow_nan=False))
        return
    source = source_identity()
    runs = []
    for count in (1, args.compare_workers):
        argv = [sys.executable, "-m", "examples.benchmark_daily_parallel", "--workers", str(count),
                "--symbols", *args.symbols, "--cutoff", args.cutoff, "--rounds", str(args.rounds),
                "--metadata-nodes", str(args.metadata_nodes), "--compare-workers", str(args.compare_workers)]
        if args.mops_cache:
            argv.extend(["--mops-cache", str(args.mops_cache)])
        completed = subprocess.run(argv, check=True, capture_output=True, text=True)
        runs.append(json.loads(completed.stdout))
    if source_identity() != source:
        raise ValueError("BENCHMARK_SOURCE_CHANGED_DURING_RUN")
    parity = len({row["result_sha256"] for run in runs for row in run["measurements"]}) == 1
    report = dict(schema="daily-parallel-benchmark/v1", scope="NO_NETWORK_MAIL_ORDERS_OR_FORMAL_STATE",
        data_kind="LOCAL_CACHED_ORIGINALS_SYNTHETIC_PRICES_NOT_OFFICIAL_RESEARCH" if args.mops_cache else "SYNTHETIC_ONLY",
        source_sha256=source, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, platform=platform.platform(),
        symbols=args.symbols, cutoff=args.cutoff, full_result_parity=parity,
        input_parity=all(run["inputs"] == runs[0]["inputs"] for run in runs), runs=runs,
        memory_policy="PARENT_AND_MAX_CHILD_RSS_ARE_SEPARATE_NOT_TOTAL_PEAK",
        startup_policy="EACH_ANALYSIS_SPAWNS_NEW_POOL_STARTUP_AND_IPC_INCLUDED")
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)
    if not parity or not report["input_parity"]:
        raise ValueError("BENCHMARK_FULL_RESULT_OR_INPUT_MISMATCH")


if __name__ == "__main__":
    main()
