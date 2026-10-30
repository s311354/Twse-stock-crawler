"""Bounded offline raw-parse/financial-companion/storage cold/warm comparison.

Freeze at most two issuers' cutoff-valid MOPS metadata and hash-verified originals
into a new temporary fixture. Recompute the complete native financial Quality
result and proof output with/without preprocessing reuse. This is not an account,
LES/EREM/PAE execution or historical deployment benchmark. No HTTP/SMTP is used.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from application.daily_performance import DailyPerformance, measure_stage
from application.les_v7 import prepare_quality_context
from application.replay_contracts import canonical_json, digest
from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_models import FilingCandidate
from domain.les_v7.quality import evaluate_financials
from domain.les_v7.quality_policy import r6_quality_policy
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION, ixbrl_record
from examples.benchmark_daily_computation import source_identity
from infrastructure.daily_work import open_daily_work
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_queries import MopsQueryRepository, _raw_normalized_facts
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.research_archive import write_bytes_once, write_once
from infrastructure.storage.evidence_graph import EvidenceGraphReader, EvidenceGraphWriter


SCHEMA = "daily-preprocessing-benchmark/v1"
MAX_ORIGINALS = 64
MAX_ORIGINAL_BYTES = 16 * 1024 * 1024
PARITY_FIELDS = ("full_business_sha256", "full_proofs_sha256", "quality_gates_sha256",
                 "full_reports_sha256", "references_sha256", "node_bytes_sha256")


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def cache_rates(counters):
    def fraction(prefix):
        hit = counters.get(prefix + "_hits", 0)
        miss = counters.get(prefix + "_misses", 0)
        return dict(hits=hit, misses=miss, hit_fraction=hit / (hit + miss) if hit + miss else None)
    return {kind: fraction(kind) for kind in ("raw_parse_disk", "financial_companion_disk",
                                             "financial_companion_cache")}


def _copy_original(source, destination, expected_hash):
    if not source.is_file() or source.stat().st_size > MAX_ORIGINAL_BYTES:
        raise ValueError("BENCHMARK_ORIGINAL_MISSING_OR_TOO_LARGE:" + str(source))
    body = source.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_hash:
        raise ValueError("BENCHMARK_ORIGINAL_HASH_MISMATCH")
    write_bytes_once(destination, body)
    return len(body)


def freeze_fixture(database, directory, *, symbols, cutoff, period_count):
    """One read-only transaction, bounded rows; never migrate the real cache."""
    database, directory = Path(database).resolve(), Path(directory).resolve()
    repo = MopsRepository(directory)
    at = pd.Timestamp(cutoff)
    if at.tzinfo is None:
        raise ValueError("BENCHMARK_TIMEZONE_REQUIRED")
    original_rows, periods, inputs, missing = [], {}, [], []
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
        source.execute("PRAGMA query_only=1")
        source.execute("BEGIN")
        columns = [r[1] for r in source.execute("PRAGMA table_info(filings)")]
        for symbol in symbols:
            rows = [dict(zip(columns, row)) for row in source.execute(
                "SELECT * FROM filings WHERE symbol=? AND available_at_epoch<=? "
                "ORDER BY fiscal_year,quarter,revision_number,id", (symbol, int(at.timestamp())))]
            # The SQLite header is second-granularity. Preserve the exact cutoff.
            rows = [r for r in rows if pd.Timestamp(r["available_at"]) <= at]
            chosen = sorted({(r["fiscal_year"], r["quarter"]) for r in rows})[-period_count:]
            periods[symbol] = chosen
            original_rows.extend(r for r in rows if (r["fiscal_year"], r["quarter"]) in chosen)
        if not original_rows or len(original_rows) > MAX_ORIGINALS:
            raise ValueError("BENCHMARK_REQUIRES_1_TO_64_SELECTED_ORIGINALS")
        with repo.unit_of_work() as uow:
            fixture_columns = [r[1] for r in uow.connection.execute("PRAGMA table_info(filings)")]
            if set(columns) != set(fixture_columns):
                raise ValueError("BENCHMARK_FILING_SCHEMA_MISMATCH")
            for row in original_rows:
                relative = Path(row["raw_path"])
                original = (database.parent / relative).resolve()
                destination = (directory / relative).resolve()
                if (not original.is_relative_to((database.parent / "raw").resolve())
                        or not destination.is_relative_to((directory / "raw").resolve())):
                    raise ValueError("BENCHMARK_RAW_PATH_OUTSIDE_ROOT")
                available = original.is_file()
                if available:
                    size = _copy_original(original, destination, row["content_sha256"])
                else:
                    size = 0
                    missing.append(dict(symbol=row["symbol"], content_sha256=row["content_sha256"],
                                        reason="LOCAL_ORIGINAL_MISSING_NOT_REPAIRED"))
                uow.connection.execute("INSERT INTO filings ({}) VALUES ({})".format(
                    ",".join(columns), ",".join("?" for _ in columns)), [row[c] for c in columns])
                inputs.append(dict(metadata=row, original_present=available, size_bytes=size))
        # Preserve every existing signed-proof candidate for these originals;
        # the native selector still verifies identities and rejects future ones.
        for row in original_rows:
            root = database.parent / "signed-audit"
            for path in sorted((root / "bindings" / row["content_sha256"]).glob("*.json")):
                packet = json.loads(path.read_bytes())
                write_bytes_once(directory / "signed-audit" / path.relative_to(root), path.read_bytes())
                ref = packet["body_ref"]
                _copy_original(root / "objects" / ref["sha256"],
                    directory / "signed-audit" / "objects" / ref["sha256"], ref["sha256"])
                inputs.append(dict(signed_binding_sha256=file_hash(path),
                                   signed_pdf_sha256=ref["sha256"]))
    record = dict(schema="daily-preprocessing-fixed-inputs/v1", evidence="REAL_LOCAL_ORIGINALS",
                  cutoff=cutoff, symbols=symbols, periods=periods, inputs=inputs, gaps=missing)
    record["input_id"] = digest(record)
    write_once(directory / "fixture.json", record)
    return record


def synthetic_fixture(directory, *, symbols, cutoff):
    """Explicit test-only sparse financials; UNKNOWN must remain UNKNOWN."""
    repo = MopsRepository(directory)
    for symbol in symbols:
        body = ('''<html xmlns:xbrli="http://www.xbrl.org/2003/instance"
          xmlns:currency="http://www.xbrl.org/2003/iso4217"><body>
          <xbrli:context id="YTD"><xbrli:period><xbrli:startDate>2026-01-01</xbrli:startDate>
          <xbrli:endDate>2026-06-30</xbrli:endDate></xbrli:period></xbrli:context>
          <xbrli:unit id="money"><xbrli:measure>currency:TWD</xbrli:measure></xbrli:unit>
          <ix:nonFraction name="ifrs-full:Revenue" contextRef="YTD" unitRef="money" scale="3">5000</ix:nonFraction>
          <!-- SYNTHETIC_TEST_ONLY_SYMBOL_''' + symbol + ''' --></body></html>''').encode()
        candidate = FilingCandidate(symbol, 2026, 2, "C", "https://example.invalid/SYNTHETIC_ONLY",
                                    period_end="2026-06-30")
        repo.save_revision(candidate, body, parse_ixbrl(body), available_at="2026-08-10T05:00:00Z",
                           fetched_at="2026-08-10T05:01:00Z")
    record = dict(schema="daily-preprocessing-fixed-inputs/v1", evidence="SYNTHETIC_TEST_ONLY",
                  cutoff=cutoff, symbols=symbols, periods={s: [(2026, 2)] for s in symbols}, gaps=[])
    record["input_id"] = digest(record)
    write_once(Path(directory) / "fixture.json", record)
    return record


def _financial_business(repo, manifest):
    outputs = {}
    cutoff = manifest["cutoff"]
    for symbol in manifest["symbols"]:
        periods = tuple(tuple(p) for p in manifest["periods"][symbol])
        with measure_stage("native_source_selection"):
            companions = repo.load_v7_financial_sources(symbol, cutoff, periods=periods,
                                                        source_repair=True, market="TWSE")
            filings = repo.load_filings_many([symbol], as_of=cutoff)[symbol]
        records = []
        with measure_stage("canonical_financial_rows"):
            for filing in filings:
                if (filing.fiscal_year, filing.quarter) not in periods or pd.Timestamp(filing.available_at) > pd.Timestamp(cutoff):
                    continue
                path = (repo.root / filing.raw_path).resolve()
                if not path.is_file():
                    continue
                stat = path.stat()
                parsed = _raw_normalized_facts(path, filing.content_sha256,
                    (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size), V6_FILING_SCHEMA_VERSION)
                if parsed is None:
                    raise ValueError("BENCHMARK_FROZEN_ORIGINAL_NO_LONGER_VERIFIES")
                source = FiledFinancials(fiscal_year=filing.fiscal_year, quarter=filing.quarter,
                    available_at=filing.available_at, revision_number=filing.revision_number,
                    facts=parsed[0], correction_description=filing.correction_description,
                    availability_source=filing.availability_source, filing_id=filing.filing_id,
                    content_sha256=filing.content_sha256, source_url=filing.source_url,
                    period_end=filing.period_end, statement_scope=filing.statement_scope,
                    audit_status=filing.audit_status, taxonomy_schema_refs=parsed[1])
                records.append(dict(ixbrl_record(symbol, source), market="TWSE"))
        context = dict(symbol=symbol, market="TWSE", as_of=cutoff, filings=records,
                       quality_policy=r6_quality_policy())
        with measure_stage("native_quality_enrichment"):
            prepare_quality_context(context, **{k: companions.get(k, {}) for k in
                ("financial_audits", "financial_supplements", "financial_repairs_r5", "signed_financial_audits")})
        with measure_stage("native_quality_gate"):
            quality = evaluate_financials(context["filings"], symbol=symbol, market="TWSE",
                                          cutoff=cutoff, policy=context["quality_policy"])
        outputs[symbol] = dict(filings=context["filings"], companions=companions, quality=quality)
    return outputs


def compare_parity(results):
    for field in PARITY_FIELDS:
        if len({row[field] for output in results for row in output["measurements"]}) != 1:
            raise ValueError("BENCHMARK_FULL_PARITY_FAILED:" + field)


def worker(args):
    manifest = json.loads((args.fixture / "fixture.json").read_bytes())
    expected = manifest.pop("input_id")
    if digest(manifest) != expected:
        raise ValueError("BENCHMARK_FIXTURE_MANIFEST_CHANGED")
    manifest["input_id"] = expected
    # The frozen fixture is also read-only, not only the production source.
    # Do not run a repository schema migrator during either measured path.
    class ReadOnlyFixture:
        root = args.fixture
        queries = MopsQueryRepository(
            lambda: sqlite3.connect((args.fixture / "mops.sqlite3").resolve().as_uri() + "?mode=ro", uri=True),
            lambda: None, raw_root=args.fixture / "raw")
        load_v7_financial_sources = queries.load_v7_financial_sources
        load_filings_many = queries.load_filings_many
    repo = ReadOnlyFixture()
    measurements = []
    directory = args.storage / args.mode
    directory.mkdir(parents=True, exist_ok=False)
    cache = args.workspace / args.mode / "raw.sqlite3"
    cache.parent.mkdir(parents=True, exist_ok=True)
    for iteration in range(args.rounds):
        probe = DailyPerformance()
        before_bytes = tree_bytes(directory)
        with probe.activate(), measure_stage("total_measured_boundaries"):
            # A new context tests restart-safe disk reuse, not only memory hits.
            with open_daily_work(enabled=args.mode != "reference", parse_cache_path=cache,
                    parse_cache_compression=args.mode != "uncompressed",
                    parse_cache_max_bytes=getattr(args, "parse_cache_bytes", 512 * 1024 * 1024)) as work:
                with measure_stage("native_financial_research"):
                    business = _financial_business(repo, manifest)
                reports = [dict(scope="BENCHMARK_ONLY_NOT_RESEARCH", symbol=s,
                    cutoff=manifest["cutoff"], input_id=expected, **record) for s, record in business.items()]
                with measure_stage("storage_and_publication"):
                    options = ({"workspace_root": args.workspace / "staging"}
                               if args.mode != "reference" else {})
                    with EvidenceGraphWriter(directory / "evidence.sqlite3", **options) as writer:
                        references = [writer.retain(report) for report in reports]
                work_stats = work.stats
                disk_stats = work.parse_memo.stats if work.parse_memo is not None else None
        with EvidenceGraphReader(directory) as reader:
            restored = [reader.resolve(ref) for ref in references]
        if canonical_json(restored) != canonical_json(reports):
            raise ValueError("BENCHMARK_FULL_REPORT_RESTORE_MISMATCH")
        signature = hashlib.sha256()
        with closing(sqlite3.connect((directory / "evidence.sqlite3").as_uri() + "?mode=ro", uri=True)) as connection:
            if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("BENCHMARK_CLOSED_STORE_INTEGRITY_FAILED")
            for sha, body in connection.execute("SELECT sha256,body FROM report_objects ORDER BY sha256"):
                signature.update(sha.encode()); signature.update(body)
        qualities = {s: r["quality"] for s, r in business.items()}
        gates = {s: {k: q.get(k) for k in ("status", "core_inputs_pass", "mandatory_safety_pass",
            "reason_codes", "quality_score", "data_confidence", "coverage_total")} for s, q in qualities.items()}
        measurements.append(dict(iteration=iteration,
            temperature="process_cold_empty_derived_cache" if iteration == 0 else "warm_disk_new_work_context",
            input_id=expected, full_business_sha256=digest(business), full_proofs_sha256=digest({s:r["companions"] for s,r in business.items()}),
            quality_gates_sha256=digest(gates), quality_gates=gates,
            full_reports_sha256=digest(reports), references_sha256=digest(references),
            node_bytes_sha256=signature.hexdigest(), retained_bytes=tree_bytes(directory),
            added_retained_bytes=tree_bytes(directory)-before_bytes,
            parse_cache_bytes=tree_bytes(cache.parent), parse_cache=disk_stats,
            work_cache=work_stats, work_version=work.version,
            cache_hit_rates=cache_rates(probe.counters), **probe.snapshot()))
    return dict(mode=args.mode, input_id=expected, measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mops-cache", type=Path)
    parser.add_argument("--symbols", nargs="+", default=["3014", "6834"])
    parser.add_argument("--cutoff", default="2026-10-16T23:00:00+08:00")
    parser.add_argument("--periods", type=int, default=9)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-source-codecs", action="store_true",
                        help="Compare uncompressed/compressed source caches at the same byte budget; no model change.")
    parser.add_argument("--parse-cache-bytes", type=int, default=512 * 1024 * 1024,
                        help="Stored-payload budget for each isolated derived cache (default 512 MiB).")
    parser.add_argument("--storage-parent", type=Path,
                        help="Temporary destination on the volume to benchmark; never an existing store.")
    parser.add_argument("--workspace-parent", type=Path,
                        help="Internal SSD temporary workspace parent; default system temporary directory.")
    parser.add_argument("--mode", choices=["reference", "uncompressed", "optimized"], help=argparse.SUPPRESS)
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--storage", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= len(args.symbols) <= 2 or len(set(args.symbols)) != len(args.symbols) or not 2 <= args.rounds <= 5 or not 1 <= args.periods <= 9:
        parser.error("Use 1-2 distinct issuers, 1-9 periods and 2-5 rounds; not a full-market job")
    if not 0 <= args.parse_cache_bytes <= 4 * 1024 ** 3:
        parser.error("Use a derived-cache budget between zero and 4 GiB")
    if args.output and args.output.exists():
        parser.error("Saved benchmark receipts are immutable; choose a new output path")
    if args.mode:
        print(json.dumps(worker(args), ensure_ascii=False, allow_nan=False))
        return
    started = time.perf_counter()
    code = source_identity()
    with (tempfile.TemporaryDirectory(prefix="twse-preprocessing-local-", dir=args.workspace_parent) as local,
          tempfile.TemporaryDirectory(prefix="twse-preprocessing-destination-", dir=args.storage_parent) as storage):
        local, storage = Path(local), Path(storage)
        fixture = local / "fixed-inputs"
        manifest = (freeze_fixture(args.mops_cache, fixture, symbols=args.symbols, cutoff=args.cutoff, period_count=args.periods)
                    if args.mops_cache else synthetic_fixture(fixture, symbols=args.symbols, cutoff=args.cutoff))
        setup_seconds = time.perf_counter() - started
        results = []
        modes = ("uncompressed", "optimized") if args.compare_source_codecs else ("reference", "optimized")
        for mode in modes:
            command = [sys.executable, "-m", "examples.benchmark_daily_preprocessing", "--mode", mode,
                "--fixture", str(fixture), "--storage", str(storage), "--workspace", str(local),
                "--rounds", str(args.rounds), "--parse-cache-bytes", str(args.parse_cache_bytes)]
            results.append(json.loads(subprocess.check_output(command, text=True)))
        if source_identity() != code:
            raise ValueError("BENCHMARK_SOURCE_CHANGED")
        compare_parity(results)
        report = dict(schema=SCHEMA, scope="OFFLINE_TWO_ISSUER_PREPROCESSING_AND_STORAGE_NOT_FULL_DAILY",
            evidence=manifest["evidence"], fixed_inputs=manifest, source_sha256=code,
            harness_sha256=file_hash(__file__), fixture_setup_seconds=setup_seconds,
            comparison="source_cache_codec_only" if args.compare_source_codecs else "preprocessing_and_storage_reuse",
            parse_cache_stored_payload_budget=args.parse_cache_bytes,
            total_harness_seconds=time.perf_counter() - started,
            python=platform.python_version(), platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
            sqlite=sqlite3.sqlite_version, storage_volume_device=storage.stat().st_dev,
            workspace_volume_device=local.stat().st_dev, full_output_equal=True, results=results,
            limitations=["Same source version with cache disabled/enabled, not old/new model binaries.",
                "Process-cold derived cache; operating-system filesystem cache is uncontrolled.",
                "Complete native financial Quality, proof and graph parity; no LES/EREM/PAE/account simulation.",
                "Missing evidence remains UNKNOWN/DATA_INSUFFICIENT; no fabricated coverage or signals.",
                "No HTTP, SMTP, orders, formal accounts, official cache writes or schedules.",
                "Small fixture timings cannot establish full-universe speedup; stages may be nested.",
                "RSS is each child process high-water including verification, not a stage delta."])
    if args.output:
        write_once(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
