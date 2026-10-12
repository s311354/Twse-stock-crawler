"""Offline synthetic CPU benchmark. No refresh, email, trades or PAE DB writes.

Run: .venv/bin/python -m examples.benchmark_daily_computation --stocks 8 --bars 500 --rounds 2
Each mode runs in a fresh child process, then repeats in that process. The
first iteration is process-cold (not OS filesystem-cold); later ones are warm.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import gc
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance, measure_stage
from application.leadership_screen import screen_once, watch_result_key
from application.market_history import MarketHistoryInput
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from domain.numeric_history import metadata_free_frame


def synthetic_market(stocks, bars, metadata_nodes=0):
    dates = pd.bdate_range(end="2026-09-30", periods=bars).strftime("%Y-%m-%d").tolist()
    # Deliberately incomplete fundamentals/restrictions/Kelly: benchmark data
    # cannot grant research entry or be mistaken for official market evidence.
    symbols = [str(2300 + i) for i in range(stocks)]
    closes = [100 + np.arange(bars) * .03 + np.sin(np.arange(bars) / 7 + i) for i in range(stocks)]
    scaled = lambda delta: [(close + delta).tolist() for close in closes]
    prices = lambda delta: [[value * 1000 for value in row] for row in scaled(delta)]
    return MarketHistoryInput(stock_count=stocks, stock_numbers=symbols,
        stock_names=["SYNTHETIC_NOT_MARKET_DATA"] * stocks, dates=[dates] * stocks,
        opens=prices(-.4), highs=prices(1), lows=prices(-1), closes=prices(0),
        volumes=[[1_000_000.] * bars for _ in symbols], trade_counts=[[500] * bars for _ in symbols],
        trade_values=[(close * 1_000_000).tolist() for close in closes], bids=prices(-.05), asks=prices(.05),
        pe_ratios=[[20.] * bars for _ in symbols], taiex_closes=[np.linspace(19000, 20000, bars).tolist()] * stocks,
        annual_fundamentals={}, fundamental_snapshots={symbol: {
            "synthetic_metadata": [{"period": i, "source": "SYNTHETIC_NOT_FINANCIAL_EVIDENCE",
                                    "contexts": [{"id": str(i), "facts": [float(j) for j in range(12)],
                                                  "origin": {"version": "synthetic", "hash": "0" * 64}}]}
                                   for i in range(metadata_nodes)]}
            for symbol in symbols} if metadata_nodes else {},
        fundamental_statuses={}, fundamental_status_details={},
        signal_dates={symbol: "2026-09-30T15:00:00+08:00" for symbol in symbols},
        account_equity=1_000_000., default_sector="ELEC", fallback_date=dates[-1],
        les_v6_portfolio={"value": 1_000_000., "available_cash": 1_000_000., "intended_weight": .25, "capacity_weight": .25})


@contextmanager
def observe_gc():
    """Observe collections only; never disable GC or change its thresholds."""
    metrics = {"calls": 0, "seconds": 0., "collected": 0, "uncollectable": 0}
    started = {}
    def observe(phase, info):
        generation = info["generation"]
        if phase == "start":
            started[generation] = time.perf_counter()
        elif phase == "stop":
            begin = started.pop(generation, None)
            metrics["calls"] += 1
            metrics["seconds"] += time.perf_counter() - begin if begin is not None else 0.
            metrics["collected"] += info["collected"]
            metrics["uncollectable"] += info["uncollectable"]
    gc.callbacks.append(observe)
    try:
        yield metrics
    finally:
        gc.callbacks.remove(observe)


def worker(args):
    market = synthetic_market(args.stocks, args.bars, args.metadata_nodes)
    builder = AnalysisDatasetBuilder(optimized_daily=args.mode == "optimized")
    boundary = None
    if args.metadata_nodes:
        # Directly compare the original pandas column/attrs boundary with the
        # new value boundary. Separate from whole-pipeline/reference timings.
        history = builder.history_assembler.build(market, 0)
        class CopyCounter:
            calls = 0
            def __deepcopy__(self, memo):
                self.calls += 1
                return self
        counter = CopyCounter()
        history.attrs["benchmark_copy_counter"] = counter
        began = time.perf_counter()
        with observe_gc() as gc_metrics:
            values = []
            for _ in range(10):
                numeric = metadata_free_frame(history) if args.mode == "optimized" else history
                values.append(numeric["Close"].iloc[-1])
        boundary = dict(operations=10, operation="CONSTRUCT_NUMERIC_INPUT_AND_SELECT_CLOSE",
                        seconds=time.perf_counter() - began, attrs_deepcopy_calls=counter.calls,
                        values_sha256=hashlib.sha256(json.dumps(values).encode()).hexdigest(),
                        gc_observation=gc_metrics)
        del history, numeric, counter
    measurements = []
    for iteration in range(args.rounds):
        probe = DailyPerformance()
        with observe_gc() as gc_metrics, probe.activate(), measure_stage("total"):
            # Simulate the real pre-refresh watch pass, without financial I/O.
            histories = ([builder.history_assembler.build_screen(market, i) for i in range(args.stocks)]
                         if args.mode == "optimized" else builder.history_assembler.build_all(market))
            watches, key = screen_once(histories, session_date=market.fallback_date,
                as_of="2026-09-30T15:00:00+08:00")
            effective = replace(market, les_v6_research_context={watch["symbol"]:
                                {"watch": watch, "watch_input_sha256": key,
                                 "watch_result_sha256": watch_result_key(watch)} for watch in watches})
            del histories
            decisions = builder.build_decisions(effective, [[1, 1, 1, 1, .01]] * args.stocks)
            with measure_stage("pae"):
                decisions = PortfolioAllocationV2Pipeline().enrich(decisions,
                    PortfolioRuntimeContext(account_equity=1_000_000., current_cash=1_000_000.,
                                            positions=(), default_sector="ELEC"))
            with measure_stage("result_serialization"):
                payload = json.dumps([asdict(row) for row in decisions], sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, default=str).encode()
        measurements.append(dict(iteration=iteration, temperature="process_cold" if iteration == 0 else "warm",
                                 result_sha256=hashlib.sha256(payload).hexdigest(), result_bytes=len(payload),
                                 gc_observation=gc_metrics,
                                 **probe.snapshot()))
    return dict(mode=args.mode, metadata_boundary_probe=boundary, measurements=measurements)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = sorted(path for folder in ("application", "domain", "infrastructure", "config")
                   for path in (root / folder).rglob("*.py")) + [Path(__file__).resolve()]
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks", type=int, default=8)
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--metadata-nodes", type=int, default=0,
                        help="Unused synthetic dict/list provenance per stock; never supplies financial evidence")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["baseline", "optimized"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stocks < 1 or args.bars < 1 or args.rounds < 1:
        parser.error("stocks, bars and rounds must be positive")
    if args.metadata_nodes < 0:
        parser.error("metadata-nodes must be nonnegative")
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new benchmark path")
    if args.mode:
        print(json.dumps(worker(args), allow_nan=False))
        return
    source_sha256 = source_identity()
    runs = []
    for mode in ("baseline", "optimized"):
        completed = subprocess.run([sys.executable, "-m", "examples.benchmark_daily_computation",
            "--stocks", str(args.stocks), "--bars", str(args.bars), "--rounds", str(args.rounds),
            "--metadata-nodes", str(args.metadata_nodes), "--mode", mode],
            check=True, capture_output=True, text=True)
        runs.append(json.loads(completed.stdout))
    if source_identity() != source_sha256:
        raise SystemExit("BENCHMARK_SOURCE_CHANGED_DURING_RUN")
    hashes = {row["result_sha256"] for run in runs for row in run["measurements"]}
    boundaries = {run["metadata_boundary_probe"]["values_sha256"] for run in runs
                  if run["metadata_boundary_probe"] is not None}
    report = dict(schema="daily-computation-benchmark/v1", data_kind="SYNTHETIC_NOT_INVESTMENT_PERFORMANCE",
        scope="CPU_ONLY_NO_MOPS_SMTP_OR_FORMAL_STATE", stocks=args.stocks, bars=args.bars,
        metadata_nodes_per_stock=args.metadata_nodes,
        source_sha256=source_sha256, reference="SAME_CODE_UNOPTIMIZED_DAILY_COMPOSITION",
        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, platform=platform.platform(),
        full_result_parity=len(hashes) == 1,
        metadata_boundary_values_parity=len(boundaries) <= 1, runs=runs)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)
    if not report["full_result_parity"]:
        raise SystemExit("DAILY_BENCHMARK_RESULT_MISMATCH")
    if not report["metadata_boundary_values_parity"]:
        raise SystemExit("DAILY_BENCHMARK_METADATA_VALUES_MISMATCH")


if __name__ == "__main__":
    main()
