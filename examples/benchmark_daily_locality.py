"""Bounded offline benchmark of the 27-minute run's remaining CPU/I/O boundaries.

Reference retains the existing 8 MiB encoding memo, but uses the former report
normalizer and 256-key SHA-order lookup. Optimized uses the same data/engine,
native scalar fast path and 1024-key physical-order lookup. No financial rules,
official caches, mail, network, accounts, or PAE states are changed.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile

import pandas as pd

from application.daily_performance import DailyPerformance, measure_stage
from application.fundamental_service import MopsFundamentalService
from application.replay_contracts import canonical_json, digest
from domain.strategies.low_entry_score_v6 import _clean
from domain.strategies.low_entry_v6_quality import evaluate_quality, number
from examples.benchmark_daily_computation import source_identity
from examples.benchmark_daily_hotspots import storage_sample
from examples.benchmark_fundamental_preflight import seed_copy
from infrastructure.research_archive import write_once
from infrastructure.storage.evidence_graph import EvidenceGraphReader, EvidenceGraphWriter


def reference_clean(value):
    """Frozen pre-upgrade normalizer; not regenerated expected output."""
    if isinstance(value, Mapping):
        return {str(key): reference_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [reference_clean(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and number(value) is None:
        return None
    return value


def worker(args):
    inputs, business, measurements = [], {}, []
    with tempfile.TemporaryDirectory(prefix="twse-locality-benchmark-") as temporary:
        directory = Path(temporary)
        setup = DailyPerformance()
        with setup.activate(), measure_stage("fixture_setup"):
            if args.mops_cache:
                repo, sectors, inputs = seed_copy(args, directory / "isolated-mops")
                service = MopsFundamentalService(object(), repository=repo,
                    filing_documents=object(), filing_parser=object())
                service._company_sectors = sectors
                def prohibited(*args, **kwargs):
                    raise AssertionError("BENCHMARK_MUST_NOT_FETCH_OR_DERIVE")
                repo.load_filed_financials = prohibited
                with repo.unit_of_work() as uow:
                    for symbol in args.symbols:
                        bundle = service._load_cached_bundle(symbol, parsed_signal=pd.Timestamp(args.cutoff),
                            sector=sectors[symbol], uow=uow)[0]
                        quality = asdict(evaluate_quality(bundle.snapshot["les_v6_mops_filings"],
                            symbol=symbol, market="TWSE", as_of=args.cutoff))
                        business[symbol] = dict(snapshot=bundle.snapshot, as_of=bundle.as_of,
                            annual=bundle.annual_fundamentals.to_json(orient="table"), quality=quality)
                reports = [dict(scope="BENCHMARK_ONLY_NOT_RESEARCH", symbol=symbol, cutoff=args.cutoff,
                    proof=record["snapshot"], quality=record["quality"]) for symbol, record in business.items()]
            else:
                reports = [dict(scope="SYNTHETIC_BENCHMARK_ONLY", symbol=symbol,
                    proof=[storage_sample() for _ in range(3)]) for symbol in args.symbols]
        normalize = _clean if args.mode == "optimized" else reference_clean
        for iteration in range(args.rounds):
            probe = DailyPerformance()
            with probe.activate(), measure_stage("total_measured_boundaries"):
                with measure_stage("report_normalization"):
                    normalized = normalize(reports)
                with measure_stage("storage_boundary"), EvidenceGraphWriter(directory / "evidence.sqlite3",
                        encoding_cache_bytes=8 * 1024 * 1024,
                        batch_nodes=1024 if args.mode == "optimized" else 256,
                        ordered_lookup=args.mode == "optimized") as store:
                    refs = [store.retain(report) for report in normalized]
                    cache_bytes = store._encoding.bytes
            # Full restore and native Quality/bundle hashes are unmeasured
            # correctness checks, not an alleged LES/EREM/PAE benchmark.
            with EvidenceGraphReader(directory) as reader:
                restored = [reader.resolve(ref) for ref in refs]
            if canonical_json(restored) != canonical_json(reports):
                raise ValueError("FULL_REPORT_PARITY_FAILED")
            signature = hashlib.sha256()
            with sqlite3.connect(directory / "evidence.sqlite3") as connection:
                for sha, body in connection.execute("SELECT sha256,body FROM report_objects ORDER BY sha256"):
                    signature.update(sha.encode()); signature.update(body)
            measurements.append(dict(iteration=iteration,
                temperature="process_cold_after_fixture_setup" if iteration == 0 else "warm_retry_new_writer",
                full_business_sha256=digest(business), full_reports_sha256=digest(normalized),
                node_bytes_sha256=signature.hexdigest(), references_sha256=digest(refs),
                retained_bytes=(directory / "evidence.sqlite3").stat().st_size,
                encoding_cache_accounted_bytes=cache_bytes, **probe.snapshot()))
    return dict(mode=args.mode, inputs=inputs, setup=setup.snapshot(), measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mops-cache", type=Path)
    parser.add_argument("--symbols", nargs="+", default=["2330", "3019"])
    parser.add_argument("--cutoff", default="2026-10-07T23:59:59+08:00",
                        help="Fixed snapshot selection, not extension of strict replay's cutoff contract")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["reference", "optimized"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= len(args.symbols) <= 2 or len(set(args.symbols)) != len(args.symbols) or not 2 <= args.rounds <= 5:
        parser.error("Use 1-2 distinct symbols and 2-5 rounds; not a full-market job")
    if args.output and args.output.exists():
        parser.error("Choose a new output path; saved benchmarks are immutable")
    if args.mode:
        print(json.dumps(worker(args), ensure_ascii=False, allow_nan=False))
        return
    source = source_identity()
    results = []
    for mode in ("reference", "optimized"):
        command = [sys.executable, "-m", "examples.benchmark_daily_locality", "--mode", mode,
                   "--symbols", *args.symbols, "--cutoff", args.cutoff, "--rounds", str(args.rounds)]
        if args.mops_cache:
            command.extend(["--mops-cache", str(args.mops_cache)])
        results.append(json.loads(subprocess.check_output(command, text=True)))
    if source_identity() != source:
        raise ValueError("BENCHMARK_SOURCE_CHANGED")
    if results[0]["inputs"] != results[1]["inputs"]:
        raise ValueError("BENCHMARK_EVIDENCE_CHANGED")
    for field in ("full_business_sha256", "full_reports_sha256", "node_bytes_sha256", "references_sha256"):
        if len({row[field] for result in results for row in result["measurements"]}) != 1:
            raise ValueError("BENCHMARK_PARITY_FAILED:" + field)
    report = dict(schema="daily-locality-benchmark/v1", source_sha256=source,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="OFFLINE_TEMPORARY_STORAGE_NOT_FULL_DAILY_RUN", full_output_equal=True,
        evidence="LOCAL_MOPS_SNAPSHOTS" if args.mops_cache else "SYNTHETIC_TEST_ONLY",
        cutoff=args.cutoff, python=platform.python_version(), platform=platform.platform(),
        pandas=pd.__version__, sqlite=sqlite3.sqlite_version, results=results,
        limitations=["Reference retains prior encoding memo; not an older model binary.",
            "Process cold after fixture setup; OS storage cache is not controlled.",
            "System temporary volume, not external-volume full-market I/O.",
            "Full native Quality/bundle parity, not LES/EREM/PAE performance.",
            "SQLite physical layout may differ; all node IDs and compressed bytes must match.",
            "Catalog preload correctness is regression-tested, not timed by this harness.",
            "RSS is each child process high water, including setup/verification.",
            "No HTTP, SMTP, official-cache writes, accounts, orders or PAE state."])
    if args.output:
        write_once(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
