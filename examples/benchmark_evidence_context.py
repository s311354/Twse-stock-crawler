"""Bounded offline evidence CPU comparison, not official stock research.

Optionally read 1-2 existing MOPS snapshots in read-only mode. Prices, account
and trading calendars are synthetic. Both modes execute the same native LES,
EREM and PAE; the reference only restores the prior capture/plain boundaries.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
from types import MappingProxyType

import numpy as np
import pandas as pd

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance, count_work, measure_stage
from application.daily_work import DailyWorkContext
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from application.replay_contracts import digest
from examples.benchmark_daily_computation import observe_gc, source_identity, synthetic_market


class ReferenceEvidence(Mapping):
    """The former two-wrapper capture, benchmark-only, never a model fork."""
    def __init__(self, values, memo=None):
        memo = {} if memo is None else memo
        self.values = MappingProxyType({k: reference_freeze(v, memo) for k, v in values.items()})

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __deepcopy__(self, memo):
        return self


def reference_freeze(value, memo):
    if isinstance(value, ReferenceEvidence):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        key = id(value)
        if key not in memo:
            memo[key] = (ReferenceEvidence(value, memo) if isinstance(value, Mapping)
                         else tuple(reference_freeze(item, memo) for item in value))
        return memo[key]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    import datetime
    if isinstance(value, (datetime.date, datetime.time)):
        return value
    if hasattr(value, "item"):
        return reference_freeze(value.item(), memo)
    raise TypeError("UNSUPPORTED_MUTABLE_EVIDENCE_VALUE:" + type(value).__name__)


def reference_plain(value):
    if isinstance(value, Mapping):
        return {key: reference_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [reference_plain(item) for item in value]
    return value


@contextmanager
def reference_boundary(enabled):
    from application import evidence_context as context, low_entry_v6_daily as daily
    original = context.ReadOnlyEvidence, context._freeze, daily._plain_evidence, context.StockEvidence.filings_content_key
    def content_key(packet):
        with measure_stage("quality_diagnostics_input_hash"):
            count_work("quality_diagnostics_input_hashes")
            return digest(reference_plain(packet.snapshot.get("les_v6_filings", ())))
    if enabled:
        context.ReadOnlyEvidence, context._freeze = ReferenceEvidence, reference_freeze
        daily._plain_evidence = reference_plain
        context.StockEvidence.filings_content_key = content_key
    try:
        yield
    finally:
        context.ReadOnlyEvidence, context._freeze, daily._plain_evidence, context.StockEvidence.filings_content_key = original


@contextmanager
def financial_graph_boundary(reference):
    """Same model/source: compare independent traversal with scoped reuse."""
    from application.evidence_context import StockEvidence
    from domain.strategies import low_entry_v6_quality as quality, low_entry_v6_valuation as valuation
    original_prepare, original_key = StockEvidence.quality_preprocessing, StockEvidence.diagnostics_content_key
    saved = []
    for module, name, counter in (
        (quality, "_eligible", "quality_selection_calls"),
        (quality, "normalize_quarters_with_diagnostics", "quality_normalization_calls"),
        (valuation, "_eligible", "quality_selection_calls"),
        (valuation, "normalize_quarters_with_diagnostics", "quality_normalization_calls"),
        (quality, "reported_period_inputs", "quality_reported_inputs_calls"),
    ):
        original = getattr(module, name)
        def counted(*args, _original=original, _counter=counter, **kwargs):
            count_work(_counter)
            return _original(*args, **kwargs)
        saved.append((module, name, original))
        setattr(module, name, counted)
    if reference:
        StockEvidence.quality_preprocessing = lambda self, kind, compute, **kwargs: compute()
        StockEvidence.diagnostics_content_key = lambda self, **kwargs: self.filings_content_key()
    try:
        yield
    finally:
        StockEvidence.quality_preprocessing, StockEvidence.diagnostics_content_key = original_prepare, original_key
        for module, name, original in saved:
            setattr(module, name, original)


def benchmark_market(args):
    market = synthetic_market(len(args.symbols), 500, args.metadata_nodes)
    sources, snapshots = [], {}
    if args.mops_cache:
        uri = args.mops_cache.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            for symbol in args.symbols:
                row = connection.execute("""SELECT filing_set_sha256, sector, snapshot_json
                    FROM fundamental_snapshots WHERE symbol=? ORDER BY rowid DESC LIMIT 1""", (symbol,)).fetchone()
                if row is None:
                    raise ValueError("NO_LOCAL_SNAPSHOT:" + symbol)
                snapshot = json.loads(row[2])
                # Mirror the raw MOPS arm only, without a current batch fetch.
                # No claimed official current/historical report is generated.
                snapshot["les_v6_filings"] = snapshot.get("les_v6_mops_filings", [])
                snapshots[symbol] = snapshot
                sources.append(dict(symbol=symbol, sector=row[1], filing_set_sha256=row[0],
                    snapshot_sha256=hashlib.sha256(row[2].encode()).hexdigest(), snapshot_bytes=len(row[2].encode()),
                    selection="LATEST_LOCAL_ROW_RAW_MOPS_ONLY_NOT_HISTORICAL_REPLAY"))
    else:
        snapshots = {symbol: market.fundamental_snapshots.get(str(2300+i), {}) for i, symbol in enumerate(args.symbols)}
    market = replace(market, stock_numbers=args.symbols, fundamental_snapshots=snapshots,
                     signal_dates={symbol: args.cutoff for symbol in args.symbols})
    return market, sources


def worker(args):
    market, sources = benchmark_market(args)
    measurements = []
    work = DailyWorkContext("bounded-evidence-benchmark-only")
    try:
        boundary = (financial_graph_boundary(args.mode == "reference") if args.boundary == "financial-graph"
                    else reference_boundary(args.mode == "reference"))
        with boundary, work.activate():
            for iteration in range(args.rounds):
                probe = DailyPerformance()
                with observe_gc() as gc_metrics, probe.activate(), measure_stage("total"):
                    decisions = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]] * market.stock_count)
                    with measure_stage("pae"):
                        decisions = PortfolioAllocationV2Pipeline().enrich(decisions,
                            PortfolioRuntimeContext(account_equity=1_000_000., current_cash=1_000_000.,
                                                    positions=(), default_sector="ELEC"))
                    with measure_stage("serialization"):
                        body = json.dumps([asdict(row) for row in decisions], sort_keys=True,
                            ensure_ascii=False, allow_nan=False, default=str).encode()
                measurements.append(dict(iteration=iteration, temperature="process_cold" if iteration == 0 else "warm",
                    result_sha256=hashlib.sha256(body).hexdigest(), result_bytes=len(body),
                    gc_observation=dict(gc_metrics), cache=work.stats, **probe.snapshot()))
    finally:
        work.close()
    return dict(mode=args.mode, inputs=sources, measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mops-cache", type=Path)
    parser.add_argument("--symbols", nargs="+", default=["2330", "3019"])
    parser.add_argument("--cutoff", default="2026-10-07T13:30:00+08:00")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--metadata-nodes", type=int, default=2000)
    parser.add_argument("--boundary", choices=["capture", "financial-graph"], default="capture",
                        help="Compare capture wrappers or repeated financial preprocessing")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["reference", "optimized"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= len(args.symbols) <= 2 or len(set(args.symbols)) != len(args.symbols):
        parser.error("Use 1-2 distinct symbols; this is not a whole-universe job")
    if args.rounds < 2 or args.metadata_nodes < 0:
        parser.error("rounds must be >=2 and metadata-nodes nonnegative")
    if args.output and args.output.exists():
        parser.error("Choose a new output path; benchmark results are not overwritten")
    if args.mode:
        print(json.dumps(worker(args), allow_nan=False))
        return
    source = source_identity()
    runs = []
    for mode in ("reference", "optimized"):
        argv = [sys.executable, "-m", "examples.benchmark_evidence_context", "--mode", mode,
                "--symbols", *args.symbols, "--cutoff", args.cutoff, "--rounds", str(args.rounds),
                "--metadata-nodes", str(args.metadata_nodes), "--boundary", args.boundary]
        if args.mops_cache:
            argv.extend(["--mops-cache", str(args.mops_cache)])
        completed = subprocess.run(argv, check=True, capture_output=True, text=True)
        runs.append(json.loads(completed.stdout))
    if source_identity() != source:
        raise ValueError("BENCHMARK_SOURCE_CHANGED_DURING_RUN")
    parity = len({row["result_sha256"] for run in runs for row in run["measurements"]}) == 1
    report = dict(schema="evidence-context-benchmark/v1", scope="NO_NETWORK_MAIL_ORDERS_OR_FORMAL_STATE",
        data_kind="LOCAL_CACHED_ORIGINALS_SYNTHETIC_PRICES_NOT_OFFICIAL_RESEARCH" if args.mops_cache else "SYNTHETIC_ONLY",
        reference=("SAME_MODEL_INDEPENDENT_SELECTION_NORMALIZATION_AND_FULL_RECORD_HASH"
                   if args.boundary == "financial-graph" else
                   "SAME_MODEL_PRIOR_IMMUTABLE_CAPTURE_AND_PLAIN_BOUNDARIES_NOT_OLD_DEPLOYED_MODEL"),
        boundary=args.boundary,
        source_sha256=source, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, platform=platform.platform(),
        cutoff=args.cutoff, full_result_parity=parity, input_parity=runs[0]["inputs"] == runs[1]["inputs"], runs=runs)
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
