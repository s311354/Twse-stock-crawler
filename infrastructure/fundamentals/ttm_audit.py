"""Offline native Quality/TTM audit of a fixed universe; no cache writes/mail.

The comparison disables only the new bridge attestation on the same freshly
parsed originals. It is not a reproduction of an old full LES/EREM/PAE run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pandas as pd

from application.replay_contracts import aware_timestamp, canonical_json, digest
from application.research_evidence import select_filing_versions
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION, ixbrl_record
from domain.strategies.low_entry_v6_quality import QUALITY_POLICY_VERSION, DIAGNOSTIC_TTM_FIELDS, evaluate_quality
from domain.strategies.low_entry_v6_ttm import TTM_BRIDGE_POLICY_VERSION
from infrastructure.fundamentals.mops_queries import MopsQueryRepository


def _result(quality):
    return dict(status=quality.status, score=quality.score, confidence=quality.confidence,
                coverage=quality.coverage, components=quality.components, reasons=list(quality.reasons),
                quality_policy=quality.metrics.get("quality_policy_version"),
                profitability_basis=quality.metrics.get("quality_profitability_basis"),
                field_checks=quality.metrics.get("quality_field_checks", {}),
                metric_checks=quality.metrics.get("quality_metric_checks", {}),
                diagnostic_checks=quality.metrics.get("quality_diagnostic_checks", {}),
                scoring_policy=quality.metrics.get("quality_scoring_policy", {}),
                verification=quality.metrics.get("quality_verification", {}),
                dc_inputs=quality.metrics.get("quality_dc_inputs", {}),
                required_missing=quality.metrics.get("quality_required_missing", []),
                reported_basis=quality.metrics.get("quality_reported_basis"),
                reported_inputs=quality.metrics.get("quality_reported_inputs", {}),
                reported={key: value for key, value in quality.metrics.items() if key.endswith("_reported")},
                available_weight=quality.metrics.get("quality_available_weight"),
                raw_score=quality.metrics.get("quality_raw_score"), penalty=quality.metrics.get("quality_penalty"),
                ttm={key: value for key, value in quality.metrics.items() if key.endswith("_ttm")},
                ttm_inputs=quality.metrics.get("quality_ttm_inputs", {}))


def audit_ttm(root: Path, symbols: tuple[str, ...], as_of: str, *, progress=None) -> dict:
    cutoff = aware_timestamp(as_of).isoformat()
    root = Path(root).resolve()
    db = root / "mops.sqlite3"
    if not db.is_file() or not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("EXPLICIT_UNIQUE_UNIVERSE_AND_LOCAL_CACHE_REQUIRED")
    rows = []
    # One read-only connection/snapshot for the audit, never a concurrent live
    # cache migration. Raw documents are read through the native hash checker.
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro")) as connection:
        connection.execute("BEGIN")
        queries = MopsQueryRepository(lambda: connection, lambda: None, raw_root=root / "raw")
        for index, symbol in enumerate(symbols):
            metadata = queries.load_filings(symbol, as_of=cutoff, connection=connection)
            latest = max((pd.Period(year=f.fiscal_year, quarter=f.quarter, freq="Q") for f in metadata), default=None)
            wanted = [latest - i for i in range(9)] if latest is not None else []
            filings = queries.load_filed_financials(symbol, cutoff, connection=connection,
                periods=tuple((p.year, p.quarter) for p in wanted))
            indices, conflicts = select_filing_versions([{k: getattr(f, k) for k in
                ("fiscal_year", "quarter", "statement_scope", "available_at", "revision_number", "content_sha256")}
                for f in filings])
            records = [ixbrl_record(symbol, filings[i]) for i in indices]
            # A shallow top-level copy suffices: the native evaluator never
            # mutates nested source facts. Disable only this new TTM bridge.
            before = evaluate_quality([dict(row, accounting_basis_evidence={}) for row in records],
                symbol=symbol, market="TWSE", as_of=cutoff)
            after = evaluate_quality(records, symbol=symbol, market="TWSE", as_of=cutoff)
            rows.append(dict(symbol=symbol, wanted_periods=[str(p) for p in wanted],
                missing_periods=[str(p) for p in wanted if not any(r["period_end"] == p.end_time.date().isoformat() for r in records)],
                conflicts=conflicts, before=_result(before), after=_result(after),
                originals=[dict(period_end=r["period_end"], **{k: r.get(k) for k in
                    ("content_sha256", "revision_id", "source", "source_url", "available_at", "scope", "currency", "unit")})
                    for r in records]))
            if progress:
                progress(index + 1, len(symbols), symbol)
            # Heavy facts/context lists are bounded to one company plus the
            # existing native parser LRU, not retained in the output report.
            del records, filings
    gap_counts = Counter()
    for row in rows:
        for field in DIAGNOSTIC_TTM_FIELDS:
            trace = row["after"]["ttm_inputs"].get(field + "_ttm", {})
            for gap in trace.get("comparative_bridge_gaps", []):
                gap_counts[gap["reason"]] += 1
    complete = lambda row, side: bool(DIAGNOSTIC_TTM_FIELDS) and all(
        row[side]["ttm"].get(field + "_ttm") is not None for field in DIAGNOSTIC_TTM_FIELDS)
    # Retain legacy summary keys for CLI consumers; these are diagnostic counts,
    # not required-source gates for reported-period Quality scoring.
    summary = dict(symbol_count=len(rows), purpose="TTM_DIAGNOSTIC_ONLY", quality_scoring_required=False,
        before_status_counts=dict(Counter(r["before"]["status"] for r in rows)),
        after_status_counts=dict(Counter(r["after"]["status"] for r in rows)),
        required_ttm_complete_before=sum(complete(r, "before") for r in rows),
        required_ttm_complete_after=sum(complete(r, "after") for r in rows),
        comparative_bridge_gap_field_counts=dict(gap_counts),
        required_ttm_field_available_before={f: sum(r["before"]["ttm"].get(f + "_ttm") is not None for r in rows) for f in DIAGNOSTIC_TTM_FIELDS},
        required_ttm_field_available_after={f: sum(r["after"]["ttm"].get(f + "_ttm") is not None for r in rows) for f in DIAGNOSTIC_TTM_FIELDS})
    body = dict(schema="local-comparative-ttm-audit/v1", as_of=cutoff, symbols=list(symbols),
        filing_schema=V6_FILING_SCHEMA_VERSION, quality_policy=QUALITY_POLICY_VERSION,
        ttm_bridge_policy=TTM_BRIDGE_POLICY_VERSION,
        purpose="TTM_DIAGNOSTIC_ONLY", quality_scoring_required=False,
        availability_policy="STORED_OBSERVATION_CUTOFF_NOT_A_HISTORICAL_PUBLICATION_PROOF",
        comparison="SAME_ORIGINALS_NATIVE_QUALITY_WITH_VS_WITHOUT_COMPARATIVE_BRIDGE_NOT_OLD_RUN_REPRODUCTION",
        no_threshold_changes=True, no_quarter_or_eps_imputation=True,
        not_a_full_les_erem_pae_run=True, summary=summary, rows=rows)
    return dict(body, audit_id=digest(body))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--report-universe", type=Path, help="Both active/stale symbols in an existing retained report")
    parser.add_argument("--output", type=Path, required=True, help="New write-once audit JSON outside the MOPS cache")
    parser.add_argument("symbols", nargs="*")
    args = parser.parse_args(argv)
    if bool(args.symbols) == bool(args.report_universe):
        parser.error("Use explicit symbols OR --report-universe, not both/neither")
    if args.output.resolve().is_relative_to(args.cache.resolve()) or (args.report_universe and args.output.resolve() == args.report_universe.resolve()):
        raise ValueError("AUDIT_MUST_NOT_MODIFY_CACHE_OR_ORIGINAL_REPORT")
    symbols = tuple(args.symbols)
    if args.report_universe:
        with args.report_universe.open() as stream:
            report = json.load(stream)
        symbols = tuple(dict.fromkeys(row["symbol"] for bucket in ("reports", "stale_reports") for row in report.get(bucket, [])))
        del report
    def progress(done, total, symbol):
        if done % 25 == 0 or done == total:
            print(f"TTM audit {done}/{total} ({symbol})", flush=True)
    result = audit_ttm(args.cache, symbols, args.as_of, progress=progress)
    from infrastructure.research_archive import write_once
    write_once(args.output, result)
    print(canonical_json(dict(audit_id=result["audit_id"], output=str(args.output), summary=result["summary"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
