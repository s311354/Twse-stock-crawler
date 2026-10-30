"""Small, offline exact-report validation benchmark; never runs the daily job.

Source reports are fully restored and source-bound once, then the identical
encoded inputs cross repeated pure native-validation boundaries. This isolates
successful-validation token reuse, not SMTP, financial refresh or market-wide
model runtime. Originals, evidence SQLite, accounts and PAE state are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from application.daily_performance import DailyPerformance, measure_stage
from application.daily_work import DailyWorkContext
from application.replay_contracts import digest
from infrastructure.research_archive import write_once
from infrastructure.storage.report_evidence import ReportEvidenceReader
from infrastructure.storage.valuation_source_verification import (
    valuation_history_verifier_scope, verify_report_valuation_sources,
)


def _stamp(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _measure_validation(encoded, *, version, enabled, passes):
    from application.v7_report_input import read_v7_report_json
    expected = [digest(json.loads(item)) for item in encoded]
    probe, measurements = DailyPerformance(), []
    work = DailyWorkContext(version, enabled=enabled)
    try:
        with probe.activate(), work.activate():
            for iteration in range(passes):
                started = time.perf_counter()
                with measure_stage("report_validation_pass"):
                    actual = []
                    for content in encoded:
                        report = read_v7_report_json(content)
                        with measure_stage("full_result_parity"):
                            actual.append(digest(report))
                        del report
                if actual != expected:
                    raise ValueError("RUNTIME_BENCHMARK_FULL_RESULT_MISMATCH")
                measurements.append(dict(iteration=iteration,
                    temperature=("cold_new_work" if iteration == 0 else "warm_same_work"),
                    seconds=time.perf_counter() - started, full_report_sha256=actual,
                    metrics=probe.snapshot()))
            cache = work.stats
    finally:
        work.close()
    return dict(enabled=enabled, passes=measurements, cache=cache, metrics=probe.snapshot())


def benchmark_reports(path, *, stocks=2, passes=3, version="BENCHMARK_COMPONENT_ONLY"):
    """One to three fixed archived reports; no synthetic fallback or gate edits."""
    if type(stocks) is not int or not 1 <= stocks <= 3:
        raise ValueError("SMALL_BENCHMARK_ONLY_1_TO_3_STOCKS")
    if type(passes) is not int or not 2 <= passes <= 5:
        raise ValueError("SMALL_BENCHMARK_ONLY_2_TO_5_PASSES")
    path = Path(path).resolve()
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("RUNTIME_BENCHMARK_REFERENCE_BUNDLE_TOO_LARGE")
    content = path.read_bytes()
    metadata = json.loads(content)
    if not isinstance(metadata, dict):
        raise ValueError("RUNTIME_BENCHMARK_REFERENCED_REPORTS_REQUIRED")
    refs = metadata.get("reports")
    if (metadata.get("storage_schema") != "daily-reports/v2" or not isinstance(refs, list)
            or len(refs) < stocks):
        raise ValueError("RUNTIME_BENCHMARK_REFERENCED_REPORTS_REQUIRED")
    refs = refs[:stocks]
    if any(not isinstance(ref, dict) or ref.get("schema") != "daily-report-reference/v1" for ref in refs):
        raise ValueError("RUNTIME_BENCHMARK_REFERENCED_REPORTS_REQUIRED")
    watched = {str(path): _stamp(path)}
    confinement = ReportEvidenceReader(path.parent)
    for ref in refs:
        store = confinement.validate_store(ref.get("store"))
        if not store.is_file():
            raise ValueError("RUNTIME_BENCHMARK_SQLITE_STORE_REQUIRED")
        watched[str(store)] = _stamp(store)
    encoded, identities = [], []
    setup = DailyPerformance()
    with setup.activate(), measure_stage("fixture_restore_and_source_validation"), ReportEvidenceReader(path.parent) as reader:
        with valuation_history_verifier_scope() as verifier:
            for ref in refs:
                report = reader.resolve(ref)
                verify_report_valuation_sources(report, verifier=verifier)
                encoded.append(json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
                native = report.get("diagnostics", {}).get("native_evidence") or {}
                authorization = report.get("pae_authorization")
                identities.append(dict(symbol=report["symbol"], signal_date=report["signal_date"],
                    cutoff_at=report.get("envelope", {}).get("event", {}).get("cutoff_at"),
                    full_report_sha256=digest(report), reference_report_sha256=ref["report_sha256"],
                    source_packet_sha256=digest(native.get("source_packet")),
                    account_authorization_sha256=digest(authorization),
                    encoded_bytes=len(encoded[-1].encode())))
                del report
    reference = _measure_validation(encoded, version=version, enabled=False, passes=passes)
    optimized = _measure_validation(encoded, version=version, enabled=True, passes=passes)
    restarted = _measure_validation(encoded, version=version, enabled=True, passes=2)
    for filename, before in watched.items():
        if _stamp(Path(filename)) != before:
            raise ValueError("RUNTIME_BENCHMARK_SOURCE_CHANGED")
    all_passes = [row["full_report_sha256"] for mode in (reference, optimized, restarted) for row in mode["passes"]]
    if any(row != all_passes[0] for row in all_passes[1:]):
        raise ValueError("RUNTIME_BENCHMARK_FULL_RESULT_MISMATCH")
    return dict(schema="daily-runtime-component-benchmark/v1", status="COMPLETE",
        scope="SMALL_READ_ONLY_NATIVE_VALIDATION_NOT_FULL_DAILY_RUNTIME",
        source_bundle=str(path), source_bundle_sha256=hashlib.sha256(content).hexdigest(),
        source_stamps=watched, source_version=version, stock_count=stocks, identities=identities,
        fixture=setup.snapshot(), reference=reference, optimized=optimized, restarted=restarted,
        full_native_results_identical=True, source_bundle_and_evidence_store_unchanged=True,
        limitations=["Not a whole-market run or an end-to-end speedup claim.",
            "No market fetch, refresh, mail, order, account evolution or PAE state writes.",
            "The same full native reports, including gates and accounts, are compared.",
            "Fixture restore/source-file verification is separate; only pure native validation is memoized.",
            "Unchanged-source stamps cover the report bundle and referenced SQLite, not every upstream original file.",
            "Cold means a new work memo, not cleared operating-system file caches.",
            "RSS is process high-water, not an independent per-mode peak."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--stocks", type=int, default=2)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; benchmark results are immutable")
    from dataclasses import asdict
    from infrastructure.research_objects import ResearchVersionGuard
    started = time.perf_counter()
    runner = Path(__file__).resolve()
    runner_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    guard_started = time.perf_counter()
    guard = ResearchVersionGuard()
    guard_seconds = time.perf_counter() - guard_started
    result = benchmark_reports(args.reports, stocks=args.stocks, passes=args.passes,
                               version=digest(asdict(guard.versions)))
    check_started = time.perf_counter()
    guard.check()
    guard_check_seconds = time.perf_counter() - check_started
    if hashlib.sha256(runner.read_bytes()).hexdigest() != runner_sha:
        raise ValueError("RUNTIME_BENCHMARK_RUNNER_CHANGED")
    result["source_versions"] = asdict(guard.versions)
    result["benchmark_code_sha256"] = runner_sha
    result["environment"] = guard.environment
    result["source_guard_seconds"] = dict(open=guard_seconds, final_check=guard_check_seconds)
    result["command_seconds_excluding_receipt_write"] = time.perf_counter() - started
    result["recorded_at"] = datetime.now(timezone.utc).isoformat()
    write_once(args.output, result)
    print(json.dumps(dict(status=result["status"], stocks=args.stocks,
                         full_native_results_identical=True, output=str(args.output))))


if __name__ == "__main__":
    main()
