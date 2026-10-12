"""Read-only small archived-report storage benchmark, not model recomputation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time

from application.daily_performance import DailyPerformance
from application.replay_contracts import digest
from infrastructure.research_archive import write_once
from infrastructure.storage.report_evidence import (ReportEvidenceWriter, STORE_NAME,
    compact_report, iter_report_bundle, load_report_bundle)


def verify_report_stream(path, *, require_reference=True):
    """Verify a read-only bundle without collecting its restored reports."""
    probe = DailyPerformance()
    signature = hashlib.sha256()
    count = 0
    sample = []
    start = time.perf_counter()
    with probe.activate():
        for report in iter_report_bundle(path, require_reference=require_reference):
            signature.update(bytes.fromhex(digest(report)))
            count += 1
            if len(sample) < 5:
                sample.append(report.get("symbol"))
            del report
    return dict(schema="report-stream-verification/v1", reports_processed=count,
                reports_verified=count if require_reference else None,
                status="COMPLETE_REFERENCE_VALIDATION" if require_reference else "LEGACY_PARSE_COMPLETE_NOT_HASH_VERIFIED",
                sample_symbols=sample, ordered_report_hashes_sha256=signature.hexdigest(),
                integrity_policy="REQUIRE_REFERENCES" if require_reference else "LEGACY_COMPATIBILITY",
                seconds=time.perf_counter() - start, metrics=probe.snapshot(),
                scope="READ_ONLY_STORAGE_VERIFICATION_NOT_MODEL_RECOMPUTATION")


def benchmark(csv_path, *, stocks=3):
    if not 1 <= stocks <= 5:
        raise ValueError("SMALL_BENCHMARK_ONLY_1_TO_5_STOCKS")
    prior = csv.field_size_limit()
    try:
        csv.field_size_limit(64 * 1024 * 1024)
        reports = []
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                value = json.loads(row["LES v6 Report"])
                if "quality" not in value:
                    raise ValueError("BENCHMARK_REQUIRES_LEGACY_EMBEDDED_NATIVE_REPORTS")
                reports.append(value)
                if len(reports) == stocks:
                    break
    finally:
        csv.field_size_limit(prior)
    probe = DailyPerformance()
    measurements = []
    expected = digest(reports)
    with tempfile.TemporaryDirectory(prefix="daily-storage-benchmark-") as temporary:
        directory = Path(temporary)
        start = time.perf_counter()
        baseline = json.dumps(dict(reports=reports), ensure_ascii=False, separators=(",", ":")).encode()
        (directory / "legacy.json").write_bytes(baseline)
        # Former CSV repeats an escaped copy of every embedded native report.
        with (directory / "legacy.csv").open("w", newline="") as stream:
            writer = csv.writer(stream);writer.writerow(["LES v6 Report"])
            writer.writerows([json.dumps(report, ensure_ascii=False)] for report in reports)
        measurements.append(dict(mode="legacy_export", seconds=time.perf_counter() - start,
            bytes=sum((directory / name).stat().st_size for name in ("legacy.json", "legacy.csv"))))
        for mode in ("first_publish", "same_store_retry"):
            start = time.perf_counter()
            with probe.activate(), ReportEvidenceWriter(directory / STORE_NAME) as store:
                refs = [store.retain(report) for report in reports]
            (directory / "reports.json").write_text(json.dumps(dict(storage_schema="daily-reports/v2",
                reports=[compact_report(report, ref) for report, ref in zip(reports, refs, strict=True)]), ensure_ascii=False))
            (directory / "reports.csv").write_text("\n".join(json.dumps(ref) for ref in refs))
            write_seconds = time.perf_counter() - start
            start = time.perf_counter()
            with probe.activate():
                actual = load_report_bundle(directory / "reports.json")["reports"]
            if actual != reports or digest(actual) != expected:
                raise ValueError("NATIVE_REPORT_ROUNDTRIP_MISMATCH")
            measurements.append(dict(mode=mode, seconds=write_seconds, verify_seconds=time.perf_counter() - start,
                bytes=sum((directory / name).stat().st_size for name in ("reports.json", "reports.csv", STORE_NAME)),
                full_native_results_identical=True))
    return dict(schema="daily-storage-benchmark/v1", scope="STORAGE_ONLY_NOT_RESEARCH_REPLAY",
        stock_count=len(reports), symbols=[r["symbol"] for r in reports],
        native_content_sha256=expected, measurements=measurements, metrics=probe.snapshot(),
        limitations=["No model, network, email, account or PAE execution.",
                      "Small sequential sample, not whole-ELEC runtime or investment performance.",
                      "Logical file sizes; SQLite allocation and compression are included.",
                      "OS disk caches are not cleared; first_publish is not a machine cold boot."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path)
    source.add_argument("--reports", type=Path, help="Verify a complete bundle by streaming, read-only.")
    parser.add_argument("--allow-legacy", action="store_true", help="Accept embedded reports lacking independent hashes.")
    parser.add_argument("--stocks", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = (verify_report_stream(args.reports, require_reference=not args.allow_legacy)
              if args.reports else benchmark(args.csv, stocks=args.stocks))
    write_once(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
