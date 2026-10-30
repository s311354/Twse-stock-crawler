"""Read-only two-map/shared-map spool microbenchmark; never an analysis run.

Uses the same fixed in-memory snapshots for both variants. The native spool
owns and cleans its private temporary files; no database, source, report,
account or portfolio state is written. Timings are component measurements,
not estimates of the daily script's end-to-end speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import pandas as pd

from application.daily_performance import DailyPerformance
from infrastructure.fundamentals.mops_queries import MopsQueryRepository
from infrastructure.snapshot_retention import WorkSnapshotRetention


def compare(bundles, *, rounds=3):
    """CI should assert complete equality and retain counts, not wall time."""
    if rounds < 1 or not bundles:
        raise ValueError("BENCHMARK_REQUIRES_POSITIVE_ROUNDS_AND_SNAPSHOTS")
    results = []
    for round_number in range(rounds):
        for shared in (False, True):
            probe = DailyPerformance()
            started = time.perf_counter()
            owner = WorkSnapshotRetention("preload-map-component-benchmark", memory_bytes=0)
            try:
                with probe.activate():
                    issuers = owner.bundles()
                    preloaded = issuers if shared else owner.bundles()
                    for symbol, bundle in bundles.items():
                        preloaded[symbol] = bundle
                    for symbol, expected in bundles.items():
                        actual = issuers.get(symbol)
                        if actual is None:
                            actual = preloaded[symbol]
                            issuers[symbol] = actual
                        if actual.snapshot != expected.snapshot or actual.as_of != expected.as_of:
                            raise AssertionError("SNAPSHOT_CONTENT_CHANGED")
                        pd.testing.assert_frame_equal(actual.annual_fundamentals,
                                                      expected.annual_fundamentals, check_exact=True)
                    stats = owner.stats
            finally:
                owner.close()
            results.append(dict(round=round_number + 1, shared_issuer_map=shared,
                seconds=time.perf_counter() - started, retained_storage=stats,
                metrics=probe.snapshot(), complete_bundle_equality=True))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mops-db", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", default=["3014", "6834"])
    parser.add_argument("--rounds", type=int, default=3)
    options = parser.parse_args(argv)
    database = options.mops_db.expanduser().resolve(strict=True)
    bundles, sources = {}, {}
    # One read transaction freezes all selected rows before either variant.
    # Do not invoke schema migrations or a write-capable repository connection.
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        for symbol in dict.fromkeys(options.symbols):
            row = connection.execute(
                "SELECT annual_json,snapshot_json,filing_set_sha256,sector,created_at_epoch "
                "FROM fundamental_snapshots WHERE symbol=? "
                "ORDER BY created_at_epoch DESC,filing_set_sha256,sector LIMIT 1", (symbol,)).fetchone()
            if row is None:
                raise ValueError("BENCHMARK_SNAPSHOT_MISSING:" + symbol)
            bundles[symbol] = MopsQueryRepository._decode_bundle(row[0], row[1])
            sources[symbol] = dict(annual_json_sha256=hashlib.sha256(row[0].encode()).hexdigest(),
                snapshot_json_sha256=hashlib.sha256(row[1].encode()).hexdigest(),
                filing_set_sha256=row[2], sector=row[3], created_at_epoch=row[4])
    report = dict(schema="preload-retention-benchmark/v1", scope="READ_ONLY_COMPONENT_NOT_DAILY_RUNTIME",
        timing_policy="SAME_PROCESS_NO_OS_CACHE_RESET_NON_ADDITIVE_NESTED_STAGES",
        dataset=dict(database=str(database), sources=sources),
        rounds=compare(bundles, rounds=options.rounds))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
