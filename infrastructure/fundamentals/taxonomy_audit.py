"""Bounded, offline audit of originals; SQLite is opened read-only.

No network, cache migration writes, model orders or availability repair. This
inventory is not a declaration that stored timestamps prove historical release.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
import sqlite3

import pandas as pd

from application.replay_contracts import aware_timestamp, canonical_json, digest
from application.research_evidence import select_filing_versions
from domain.financial_taxonomy import TAXONOMY_MAPPING_VERSION
from domain.strategies.low_entry_v6_filings import ixbrl_record
from domain.strategies.low_entry_v6_quality import normalize_quarters_with_diagnostics
from infrastructure.fundamentals.mops_queries import MopsQueryRepository


def audit_local_cache(root: Path, symbols: tuple[str, ...], as_of: str, *, period_count=5) -> dict:
    cutoff = aware_timestamp(as_of).isoformat()
    if type(period_count) is not int or not 1 <= period_count <= 20:
        raise ValueError("LOCAL_AUDIT_PERIOD_COUNT_MUST_BE_1_TO_20")
    root = Path(root).resolve()
    db = root / "mops.sqlite3"
    if not db.is_file():
        raise ValueError("LOCAL_MOPS_CACHE_NOT_FOUND")
    symbols = tuple(dict.fromkeys(symbols))
    if not symbols:
        raise ValueError("LOCAL_AUDIT_REQUIRES_EXPLICIT_SYMBOLS")
    def connect():
        return sqlite3.connect(db.as_uri() + "?mode=ro")
    queries = MopsQueryRepository(connect, lambda: None, raw_root=root / "raw")
    companies = []
    for symbol in symbols:
        # One shared connection and one facts JOIN for this company's bounded
        # periods. The original query/adapter is the native daily/replay path.
        with closing(connect()) as connection:
            metadata = queries.load_filings(symbol, as_of=cutoff, connection=connection)
            latest = max((pd.Period(year=f.fiscal_year, quarter=f.quarter, freq="Q") for f in metadata), default=None)
            wanted = [latest - i for i in range(period_count)] if latest is not None else []
            filings = queries.load_filed_financials(symbol, cutoff, connection=connection,
                periods=tuple((p.year, p.quarter) for p in wanted))
        indices, conflicts = select_filing_versions([{key: getattr(f, key) for key in
            ("fiscal_year", "quarter", "statement_scope", "available_at", "revision_number", "content_sha256")}
            for f in filings])
        records = [ixbrl_record(symbol, filings[i]) for i in indices]
        periods = {pd.Period(r["period_end"], freq="Q"): r for r in records}
        quarters, methods = normalize_quarters_with_diagnostics(periods)
        metadata_by_id = {f.filing_id: f for f in metadata}
        rows = []
        for i, record in zip(indices, records):
            filing = filings[i]
            period = pd.Period(record["period_end"], freq="Q")
            rows.append(dict(filing_id=filing.filing_id,
                original=asdict(metadata_by_id[filing.filing_id]), record=record,
                normalized_quarter=quarters[period], quarter_derivation=methods[period]))
        companies.append(dict(symbol=symbol, periods=[str(p) for p in wanted],
            missing_periods=[str(p) for p in wanted if p not in periods],
            conflicts=conflicts, filings=rows))
    body = dict(schema="local-taxonomy-audit/v1", mapping_version=TAXONOMY_MAPPING_VERSION,
        as_of=cutoff, symbols=list(symbols), period_count=period_count,
        availability_policy="STORED_CUTOFF_NOT_A_HISTORICAL_PUBLICATION_PROOF",
        no_missing_amount_imputation=True, companies=companies)
    return dict(body, audit_id=digest(body))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--period-count", type=int, default=5)
    parser.add_argument("--output", type=Path, help="New write-once JSON audit; never inside the MOPS cache")
    parser.add_argument("symbols", nargs="+")
    args = parser.parse_args(argv)
    result = audit_local_cache(args.cache, tuple(args.symbols), args.as_of, period_count=args.period_count)
    if args.output:
        if args.output.resolve().is_relative_to(args.cache.resolve()):
            raise ValueError("AUDIT_OUTPUT_MUST_NOT_MODIFY_MOPS_CACHE")
        from infrastructure.research_archive import write_once
        write_once(args.output, result)
        print(canonical_json(dict(audit_id=result["audit_id"], report=str(args.output))))
    else:
        print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
