"""Read-only, two-issuer MOPS snapshot preflight benchmark (not live research).

Copies selected local rows into an isolated temporary SQLite fixture. Measures
the former eager entry path, header-first path and lazy legacy indexing; no
HTTP, original-file parsing, official-cache writes, accounts, reports or mail.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time

from application.daily_performance import DailyPerformance, measure_stage
from application.fundamental_service import MopsFundamentalService
from application.replay_contracts import digest
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from domain.strategies.low_entry_v6_quality import evaluate_quality
from infrastructure.fundamentals.mops_repository import MopsRepository
from examples.benchmark_daily_computation import source_identity


def seed_copy(args, root):
    repo = MopsRepository(root)
    inputs, sectors = [], {}
    uri = args.mops_cache.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as source, repo.unit_of_work() as uow:
        source.execute("BEGIN")  # Consistent read-only source view, not a file copy.
        for symbol in args.symbols:
            row = source.execute("""SELECT fs.symbol,fs.filing_set_sha256,fs.sector,
                fs.annual_json,fs.snapshot_json,fs.created_at_epoch FROM refresh_state rs
                JOIN fundamental_snapshots fs ON fs.symbol=rs.symbol
                  AND fs.filing_set_sha256=rs.active_snapshot_cache_key
                WHERE rs.symbol=? AND rs.active_derivation_version=?""",
                (symbol, FUNDAMENTAL_DERIVATION_VERSION)).fetchone()
            if row is None:
                raise ValueError("NO_COMPATIBLE_ACTIVE_SNAPSHOT:" + symbol)
            snapshot = json.loads(row[4])
            marker = None if args.mode == "legacy-index" else snapshot.get("les_v6_filing_schema_version")
            has_records = None if args.mode == "legacy-index" else int("les_v6_mops_filings" in snapshot)
            uow.connection.execute("INSERT INTO fundamental_snapshots VALUES (?,?,?,?,?,?,?,?)",
                                   (*row, marker, has_records))
            refresh = source.execute("SELECT * FROM refresh_state WHERE symbol=?", (symbol,)).fetchone()
            columns = [r[1] for r in source.execute("PRAGMA table_info(refresh_state)")]
            uow.connection.execute("INSERT INTO refresh_state ({}) VALUES ({})".format(
                ",".join(columns), ",".join("?" for _ in columns)), refresh)
            columns = [r[1] for r in source.execute("PRAGMA table_info(filings)")]
            # Keep real identities/provenance; the fixture intentionally has no facts/raw files.
            rows = source.execute("SELECT * FROM filings WHERE symbol=?", (symbol,))
            uow.connection.executemany("INSERT INTO filings ({}) VALUES ({})".format(
                ",".join(columns), ",".join("?" for _ in columns)), rows)
            sectors[symbol] = row[2]
            inputs.append(dict(symbol=symbol, sector=row[2], snapshot_cache_key=row[1],
                snapshot_sha256=hashlib.sha256(row[4].encode()).hexdigest(), snapshot_bytes=len(row[4].encode()),
                selection="FIXED_LOCAL_ACTIVE_SNAPSHOT_NOT_HISTORICAL_RESEARCH"))
    # Refuse an inconsistent local active pointer rather than fabricate facts.
    metadata = repo.load_filings_many(args.symbols, as_of=args.cutoff)
    headers = repo.load_cached_bundle_headers(sectors, as_of=args.cutoff,
                                              derivation_version=FUNDAMENTAL_DERIVATION_VERSION)
    if set(headers) != set(args.symbols):
        raise ValueError("SNAPSHOT_NOT_AVAILABLE_AT_BENCHMARK_CUTOFF")
    for symbol, header in headers.items():
        if header.filing_set_sha256 != repo.filing_set_hash(metadata[symbol]):
            raise ValueError("LOCAL_SELECTION_CHANGED:" + symbol)
    return repo, sectors, inputs


def worker(args):
    with tempfile.TemporaryDirectory(prefix="twse-fundamental-preflight-") as temporary:
        started = time.perf_counter()
        repo, sectors, inputs = seed_copy(args, Path(temporary) / "isolated-cache")
        setup_seconds = time.perf_counter() - started
        service = MopsFundamentalService(object(), repository=repo,
                                        filing_documents=object(), filing_parser=object())
        service._company_sectors = sectors
        if args.mode == "reference":
            # Original eager materialization boundary, same formulas and engine.
            repo.load_cached_bundle_headers = None
        def prohibited(*args, **kwargs):
            raise AssertionError("BENCHMARK_MUST_NOT_DERIVE_OR_FETCH_MISSING_EVIDENCE")
        repo.load_filed_financials = prohibited
        service.fundamental_deriver._derive_bundle = prohibited
        measurements = []
        for iteration in range(args.rounds):
            probe = DailyPerformance()
            with probe.activate(), measure_stage("total"):
                with measure_stage("local_migration"):
                    migrated = service.migrate_local_cache(args.symbols, signal_date=args.cutoff, sector="24")
                with measure_stage("bulk_preload"):
                    preloaded = service.load_cached_bundles(args.symbols, signal_date=args.cutoff, sector="24")
                outputs = {}
                with measure_stage("selected_snapshot_load"):
                    import pandas as pd
                    with repo.unit_of_work() as uow:
                        for symbol in args.symbols:
                            bundle = preloaded.get(symbol)
                            if bundle is None:
                                bundle = service._load_cached_bundle(symbol, parsed_signal=pd.Timestamp(args.cutoff),
                                                                     sector=sectors[symbol], uow=uow)[0]
                            outputs[symbol] = bundle
            # Full payload + native Quality parity, outside the preflight timer.
            records = {symbol: dict(snapshot=bundle.snapshot, as_of=bundle.as_of,
                annual_json=bundle.annual_fundamentals.to_json(orient="records", date_format="iso"),
                quality=asdict(evaluate_quality(bundle.snapshot["les_v6_mops_filings"],
                    symbol=symbol, market="TWSE", as_of=args.cutoff)))
                for symbol, bundle in outputs.items()}
            measurements.append(dict(iteration=iteration, temperature="process_cold_after_fixture_setup" if iteration == 0 else "warm_sqlite_pages",
                migrated=migrated, preloaded=len(preloaded), result_sha256=digest(records), **probe.snapshot()))
        return dict(mode=args.mode, inputs=inputs, measurements=measurements,
                    isolated_fixture_setup_seconds=setup_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mops-cache", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", default=["2330", "3019"])
    parser.add_argument("--cutoff", default="2026-10-07T05:30:00+00:00")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["reference", "optimized", "legacy-index"])
    args = parser.parse_args()
    if not 1 <= len(args.symbols) <= 2 or not 1 <= args.rounds <= 5:
        parser.error("bounded benchmark: 1-2 issuers, 1-5 rounds")
    if args.mode:
        print(json.dumps(worker(args), ensure_ascii=False, allow_nan=False))
        return
    results = []
    def fingerprint():
        return digest(dict(source=source_identity(), benchmark=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    source_sha = fingerprint()
    for mode in ("reference", "optimized", "legacy-index"):
        command = [sys.executable, "-m", "examples.benchmark_fundamental_preflight", "--mode", mode,
                   "--mops-cache", str(args.mops_cache), "--cutoff", args.cutoff,
                   "--rounds", str(args.rounds), "--symbols", *args.symbols]
        results.append(json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout))
    if fingerprint() != source_sha:
        raise ValueError("BENCHMARK_SOURCE_CHANGED")
    hashes = {row["result_sha256"] for result in results for row in result["measurements"]}
    if len(hashes) != 1:
        raise ValueError("FUNDAMENTAL_PREFLIGHT_FULL_OUTPUT_PARITY_FAILED")
    if any(row["inputs"] != results[0]["inputs"] for row in results):
        raise ValueError("BENCHMARK_SOURCE_SNAPSHOTS_CHANGED")
    import numpy as np
    import pandas as pd
    report = dict(schema="fundamental-preflight-benchmark/v1", cutoff=args.cutoff,
        platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, sqlite=sqlite3.sqlite_version, source_sha256=source_sha,
        policy="OFFLINE_TEMPORARY_SQLITE_NO_HTTP_NO_OFFICIAL_CACHE_WRITES",
        limitations=["Not full ELEC runtime", "Cold is process cold after fixture setup, not cold storage",
                     "Quality parity is native; no LES/EREM/PAE/accounts/orders evaluated",
                     "Reference restores eager entry loading, not an old model binary"],
        full_output_equal=True, results=results)
    body = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
