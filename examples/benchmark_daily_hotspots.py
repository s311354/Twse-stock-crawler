"""Bounded offline comparison of transaction hydration and graph encoding.

Real MOPS snapshots are copied read-only into temporary SQLite; no originals,
network, accounts, PAE state or email are written. Storage envelopes are marked
BENCHMARK_ONLY, not research outputs. Each path uses the same native engine and
encoding. Process-cold means after setup, not cold hardware/filesystem caches.
"""
from __future__ import annotations

import argparse
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
from domain.strategies.low_entry_v6_quality import evaluate_quality
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from examples.benchmark_daily_computation import source_identity
from examples.benchmark_fundamental_preflight import seed_copy
from infrastructure.research_archive import write_once
from infrastructure.storage.evidence_graph import EvidenceGraphReader, EvidenceGraphWriter


def storage_sample():
    return dict(scope="SYNTHETIC_BENCHMARK_ONLY", fields=[dict(name=f"field-{i}", value=i,
        original="SYNTHETIC_ONLY_" + str(i) + "x" * 1200) for i in range(700)])


def worker(args):
    measurements, inputs = [], []
    with tempfile.TemporaryDirectory(prefix="twse-hotspot-benchmark-") as temporary:
        directory = Path(temporary)
        if args.mops_cache:
            repo, sectors, inputs = seed_copy(args, directory / "isolated-mops")
            service = MopsFundamentalService(object(), repository=repo,
                filing_documents=object(), filing_parser=object())
            service._company_sectors = sectors
            def prohibited(*args, **kwargs):
                raise AssertionError("BENCHMARK_MUST_NOT_FETCH_OR_DERIVE_MISSING_EVIDENCE")
            repo.load_filed_financials = prohibited
            # Capture the actual public query/parameters; reference changes only
            # access path to the former PK index, not fields or cutoff policy.
            statements = []
            with repo.unit_of_work() as uow:
                uow.connection.set_trace_callback(statements.append)
                repo.load_cached_bundle_headers(sectors, as_of=args.cutoff,
                    derivation_version=FUNDAMENTAL_DERIVATION_VERSION, uow=uow)
                uow.connection.set_trace_callback(None)
            header_sql = [sql for sql in statements if "INDEXED BY fundamental_snapshots_header" in sql]
            if len(header_sql) != 1:
                raise ValueError("BENCHMARK_EXPECTS_ONE_BOUNDED_HEADER_QUERY")
            header_sql = header_sql[0]
        reports = None
        expected_nodes = None
        for iteration in range(args.rounds):
            probe = DailyPerformance()
            bundles, qualities = {}, {}
            header_rows = []
            with probe.activate(), measure_stage("header_boundary"):
                if args.mops_cache:
                    sql = (header_sql.replace("INDEXED BY fundamental_snapshots_header",
                        "INDEXED BY sqlite_autoindex_fundamental_snapshots_1")
                        if args.mode == "reference" else header_sql)
                    with repo.unit_of_work() as uow:
                        header_rows = uow.connection.execute(sql).fetchall()
            with probe.activate(), measure_stage("fundamental_boundary"):
                if args.mops_cache:
                    with repo.unit_of_work() as uow:
                        selected = repo.load_filings_many(args.symbols, as_of=args.cutoff, uow=uow)
                        for symbol in args.symbols:
                            first = service._load_cached_bundle(symbol, parsed_signal=pd.Timestamp(args.cutoff),
                                sector=sectors[symbol], uow=uow, filings=selected[symbol])
                            reuse = (service._bundle_selection_key(symbol, selected[symbol], sectors[symbol]), first)
                            # Both boundaries explicitly reselect at the cutoff.
                            second = service._load_cached_bundle(symbol, parsed_signal=pd.Timestamp(args.cutoff),
                                sector=sectors[symbol], uow=uow,
                                local_reuse=reuse if args.mode == "optimized" else None)
                            bundles[symbol] = second[0]
            # Outside hydration timing: complete business equivalence, not a
            # score-only comparison or regenerated expected results.
            for symbol, bundle in bundles.items():
                qualities[symbol] = asdict(evaluate_quality(bundle.snapshot["les_v6_mops_filings"],
                    symbol=symbol, market="TWSE", as_of=args.cutoff))
            business = {symbol: dict(snapshot=b.snapshot, as_of=b.as_of,
                annual=b.annual_fundamentals.to_json(orient="table"), quality=qualities[symbol])
                for symbol, b in bundles.items()}
            if reports is None:
                reports = ([dict(scope="BENCHMARK_ONLY_NOT_RESEARCH", symbol=s,
                    cutoff=args.cutoff, proof=b.snapshot, quality=qualities[s]) for s, b in bundles.items()]
                    if bundles else [dict(scope="SYNTHETIC_BENCHMARK_ONLY", symbol=s,
                        proof=[storage_sample() for _ in range(12)]) for s in args.symbols])
                # JSON contract treats tuple dependencies/reasons as arrays.
                # Preserve every field; normalize only the serialization type.
                reports = json.loads(canonical_json(reports))
            with probe.activate(), measure_stage("storage_boundary"):
                with EvidenceGraphWriter(directory / "evidence.sqlite3",
                    encoding_cache_bytes=8 * 1024 * 1024 if args.mode == "optimized" else 0) as store:
                    refs = [store.retain(report) for report in reports]
                    cache_bytes = store._encoding.bytes
            with measure_stage("unmeasured_verification"), EvidenceGraphReader(directory) as reader:
                restored = [reader.resolve(ref) for ref in refs]
            if restored != reports or digest(restored) != digest(reports):
                raise ValueError("FULL_REPORT_PARITY_FAILED")
            with sqlite3.connect(directory / "evidence.sqlite3") as connection:
                signature = hashlib.sha256()
                for sha, body in connection.execute("SELECT sha256,body FROM report_objects ORDER BY sha256"):
                    signature.update(sha.encode()); signature.update(body)
            node_hash = signature.hexdigest()
            if expected_nodes is not None and expected_nodes != node_hash:
                raise ValueError("RETRY_CHANGED_STORED_BYTES")
            expected_nodes = node_hash
            measurements.append(dict(iteration=iteration, temperature="process_cold_after_setup" if iteration == 0 else "warm_retry",
                full_business_sha256=digest(dict(headers=header_rows, bundles=business)), full_reports_sha256=digest(reports),
                node_bytes_sha256=node_hash, retained_bytes=(directory / "evidence.sqlite3").stat().st_size,
                encoding_cache_accounted_bytes=cache_bytes, **probe.snapshot()))
    return dict(mode=args.mode, inputs=inputs, measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mops-cache", type=Path, help="Read-only compatible active MOPS SQLite snapshots.")
    parser.add_argument("--symbols", nargs="+", default=["2330", "3019"])
    parser.add_argument("--cutoff", default="2026-10-07T23:59:59+08:00",
                        help="Snapshot selection boundary, not strict replay cutoff support.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["reference", "optimized"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= len(args.symbols) <= 2 or len(set(args.symbols)) != len(args.symbols) or not 2 <= args.rounds <= 5:
        parser.error("Use 1-2 distinct symbols and 2-5 rounds; this is not a market job")
    if args.output and args.output.exists():
        parser.error("Choose a new output path; existing results are immutable")
    if args.mode:
        print(json.dumps(worker(args), ensure_ascii=False, allow_nan=False))
        return
    version = source_identity()
    results = []
    for mode in ("reference", "optimized"):
        command = [sys.executable, "-m", "examples.benchmark_daily_hotspots", "--mode", mode,
            "--symbols", *args.symbols, "--cutoff", args.cutoff, "--rounds", str(args.rounds)]
        if args.mops_cache:
            command.extend(["--mops-cache", str(args.mops_cache)])
        results.append(json.loads(subprocess.check_output(command, text=True)))
    if source_identity() != version:
        raise ValueError("BENCHMARK_SOURCE_CHANGED")
    if results[0]["inputs"] != results[1]["inputs"]:
        raise ValueError("BENCHMARK_EVIDENCE_CHANGED")
    for field in ("full_business_sha256", "full_reports_sha256", "node_bytes_sha256", "retained_bytes"):
        if len({row[field] for result in results for row in result["measurements"]}) != 1:
            raise ValueError("BENCHMARK_PARITY_FAILED:" + field)
    report = dict(schema="daily-hotspots-benchmark/v1", scope="OFFLINE_TEMPORARY_STORAGE_NOT_FULL_DAILY_RUN",
        evidence="LOCAL_MOPS_SNAPSHOTS" if args.mops_cache else "SYNTHETIC_TEST_ONLY",
        source_sha256=version, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cutoff=args.cutoff, python=platform.python_version(), platform=platform.platform(),
        pandas=pd.__version__, sqlite=sqlite3.sqlite_version, full_output_equal=True, results=results,
        limitations=["No network, mail, accounts, orders, PAE state or official-cache writes.",
                     "Reference disables reuse in the same code version, not an older model binary.",
                     "Quality is recomputed for parity; LES/EREM/PAE are not measured here.",
                     "Temporary storage is on the system volume; not an external-volume I/O benchmark.",
                     "RSS is each child process high water, not a stage delta.",
                     "Warm storage retries use a new writer; encoding memo never survives transactions."])
    if args.output:
        write_once(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
