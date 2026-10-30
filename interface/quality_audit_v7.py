"""Read-only, stock-bounded local MOPS Quality v7 audit; never a crawler run."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3

import pandas as pd

from application.daily_work import DailyWorkContext
from application.quality_audit_v7 import QualityAuditStock, audit_quality_v7, quality_audit_summary
from application.replay_contracts import aware_timestamp, canonical_json, digest
from application.research_evidence import select_filing_versions
from domain.strategies.low_entry_v6_filings import ixbrl_record, V6_FILING_SCHEMA_VERSION
from infrastructure.fundamentals.mops_queries import MopsQueryRepository
from infrastructure.research_archive import write_once
from infrastructure.storage.report_evidence import ReportBundleContext, iter_report_bundle


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_audit_universe(stocklist: Path, *, universe_report: Path | None = None) -> tuple[tuple[str, ...], dict]:
    stocklist = stocklist.resolve()
    tokens = tuple(re.split(r"[;,\s]+", stocklist.read_text(encoding="utf-8-sig").strip()))
    if not tokens or tokens == ("",) or len(set(tokens)) != len(tokens):
        raise ValueError("EXPLICIT_UNIQUE_STOCKLIST_REQUIRED")
    stocklist_sha = file_sha(stocklist)
    legacy = all(token.isascii() and token.isdigit() and len(token) < 4 for token in tokens)
    report_symbols, dates = [], set()
    source = dict(stocklist_path=str(stocklist), stocklist_sha256=stocklist_sha,
                  stocklist_format="legacy-indices" if legacy else "symbols")
    if universe_report is not None:
        context = ReportBundleContext()
        original_sha = file_sha(universe_report)
        for report in iter_report_bundle(universe_report, require_reference=True, context=context):
            symbol = report.get("symbol")
            if not isinstance(symbol, str) or not symbol or symbol in report_symbols:
                raise ValueError("UNIVERSE_REPORT_DUPLICATE_OR_MISSING_SYMBOL")
            report_symbols.append(symbol)
            dates.add(report["signal_date"])
            del report
        if not context.complete or context.records != context.referenced_records or not report_symbols:
            raise ValueError("UNIVERSE_REPORT_NOT_COMPLETELY_REFERENCE_VERIFIED")
        if file_sha(universe_report) != original_sha:
            raise ValueError("UNIVERSE_REPORT_CHANGED_DURING_READ")
        source.update(universe_report_path=str(universe_report.resolve()), universe_report_sha256=original_sha,
            referenced_reports=context.referenced_records, report_signal_dates=sorted(dates),
            report_envelope_metadata=dict(context.metadata),
            universe_time_policy="CURRENT_REPORT_UNIVERSE_NOT_HISTORICAL_MEMBERSHIP_PROOF",
            survivorship_bias_status="NOT_ESTABLISHED_BIAS_FREE")
    if legacy:
        if universe_report is None:
            raise ValueError("LEGACY_INDEX_STOCKLIST_REQUIRES_VERIFIED_UNIVERSE_REPORT")
        if tuple(int(token) for token in tokens) != tuple(range(len(report_symbols))):
            raise ValueError("ONLY_COMPLETE_CONTIGUOUS_INDEX_UNIVERSE_CAN_BIND_WITHOUT_GUESSING_MAPPING")
        symbols = tuple(report_symbols)
        source["universe_selection"] = "FULL_INDEX_RANGE_BOUND_TO_COMPLETE_VERIFIED_EXPLICIT_REPORT_UNIVERSE"
    else:
        if any(re.fullmatch(r"[0-9A-Z_]{4,32}", token) is None for token in tokens):
            raise ValueError("INVALID_EXPLICIT_SECURITY_SYMBOL")
        symbols = tokens
        if universe_report is not None and set(symbols) != set(report_symbols):
            raise ValueError("EXPLICIT_STOCKLIST_AND_REPORT_UNIVERSE_MISMATCH")
        source.setdefault("universe_time_policy", "EXPLICIT_FIXED_STOCKLIST_NOT_HISTORICAL_MEMBERSHIP_PROOF")
        source["universe_selection"] = "EXPLICIT_STOCKLIST"
    if file_sha(stocklist) != stocklist_sha: raise ValueError("STOCKLIST_CHANGED_DURING_READ")
    source["universe_sha256"] = digest(list(symbols))
    return symbols, source


def quality_source_versions() -> dict:
    modules = ("application.quality_audit_v7", "interface.quality_audit_v7", "domain.les_v7.quality",
        "domain.les_v7.quality_r2", "domain.les_v7.quality_policy", "domain.les_v7.financial_sources",
        "domain.les_v7.financial_sources_r5",
        "domain.les_v7.signed_audit", "infrastructure.fundamentals.signed_audit",
        "domain.les_v7.quality_reported", "domain.strategies.low_entry_v6_reported_period",
        "domain.strategies.low_entry_v6_filings", "domain.strategies.low_entry_v6_quality",
        "domain.strategies.low_entry_v6_quality_evidence", "domain.strategies.low_entry_v6_ttm",
        "domain.security_master", "infrastructure.crawler.security_master", "infrastructure.research_documents",
        "domain.financial_taxonomy", "domain.fundamental_derivation", "domain.fundamental_models",
        "domain.data_provenance", "domain.text_normalization", "domain.les_v7.contracts",
        "application.replay_contracts", "application.research_evidence",
        "infrastructure.fundamentals.mops_queries", "infrastructure.fundamentals.mops_repository", "infrastructure.fundamentals.mops_sqlite",
        "infrastructure.fundamentals.models", "infrastructure.fundamentals.ixbrl_parser",
        "infrastructure.fundamentals.ixbrl_units")
    result = {}
    for module in modules:
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin: raise ValueError("QUALITY_AUDIT_REQUIRED_IMPLEMENTATION_MISSING:" + module)
        result[module] = file_sha(Path(spec.origin))
    return dict(module_sha256=result, content_sha256=digest(result))


class LocalQualityAuditSource:
    """Reuse native raw verification; no derived snapshot JSON is hydrated."""
    def __init__(self, root: Path, connection, *, max_filings=64, market="TWSE", source_repair_r5=False,
                 signed_reports=False, industry_master_manifest: Path | None = None):
        self.root = root.resolve()
        self.connection = connection
        self.max_filings = max_filings
        self.market = market
        self.source_repair_r5 = source_repair_r5
        self.signed_reports = signed_reports
        self.industry_master_manifest = Path(industry_master_manifest).resolve() if industry_master_manifest else None
        self.industry_master_context = None
        self.industry_documents = None
        if self.industry_master_manifest is not None:
            from infrastructure.crawler.security_master import load_security_master, prepare_security_master
            packet, self.industry_documents = load_security_master(self.industry_master_manifest)
            if packet.get("market") != market:
                raise ValueError("QUALITY_AUDIT_INDUSTRY_MASTER_MARKET_MISMATCH")
            self.industry_master_context = prepare_security_master(packet,
                document_reader=self.industry_documents.read, check_sources=self.industry_documents.check)
        self.queries = MopsQueryRepository(lambda: connection, lambda: None, raw_root=self.root / "raw")

    def load_stock(self, symbol, cutoff, *, period_count):
        from domain.les_v7.financial_sources import ixbrl_record_r2
        cutoff_time = aware_timestamp(cutoff)
        try:
            headers = self.queries.load_filings(symbol, connection=self.connection)
        except sqlite3.DatabaseError as error:
            raise ValueError("LOCAL_CACHE_SCHEMA_OR_READ_ERROR:" + str(error)) from error
        eligible, diagnostics = [], []
        for header in headers:
            try: available = aware_timestamp(header.available_at)
            except (ValueError, TypeError):
                diagnostics.append(dict(category="UNTRUSTED_AVAILABILITY", filing_id=header.filing_id,
                                        reason="INVALID_STORED_AVAILABLE_AT"))
                continue
            if available <= cutoff_time: eligible.append(header)
            else: diagnostics.append(dict(category="EXCLUDED_FUTURE_VERSION", filing_id=header.filing_id,
                period=f"{header.fiscal_year}Q{header.quarter}", available_at=header.available_at))
        latest = max((pd.Period(year=h.fiscal_year, quarter=h.quarter, freq="Q") for h in eligible), default=None)
        wanted = tuple(latest - i for i in range(period_count)) if latest is not None else ()
        periods = tuple((p.year, p.quarter) for p in wanted)
        bounded_headers = [h for h in eligible if (h.fiscal_year, h.quarter) in periods]
        if len(bounded_headers) > self.max_filings:
            raise ValueError("FILING_VERSION_CAPACITY_EXCEEDED:" + str(len(bounded_headers)))
        try:
            filings = self.queries.load_filed_financials(symbol, cutoff, connection=self.connection, periods=periods)
        except sqlite3.DatabaseError as error:
            raise ValueError("LOCAL_CACHE_FACTS_READ_ERROR:" + str(error)) from error
        identities = [{key: getattr(filing, key) for key in
            ("fiscal_year", "quarter", "statement_scope", "available_at", "revision_number", "content_sha256")}
            for filing in filings]
        indices, conflicts = select_filing_versions(identities)
        diagnostics.extend(dict(category="VERSION_CONFLICT", detail=conflict) for conflict in conflicts)
        by_id = {header.filing_id: header for header in headers}
        records, baseline_records, source_baseline_records, originals = [], [], [], []
        for index in indices:
            filing = filings[index]
            header = by_id[filing.filing_id]
            path = (self.root / header.raw_path).resolve()
            native_original = any(f.numeric_value is not None and f.unit and
                f.unit.startswith(("iso4217:", "xbrli:")) for f in filing.facts)
            # Units redacted by the native query cannot certify SQLite-only
            # fallback amounts. Independently bind every R2 record to original
            # bytes, not merely the latest stored audit_status text.
            within_raw = path.is_relative_to((self.root / "raw").resolve())
            actual_sha = file_sha(path) if within_raw and path.is_file() else None
            verified = native_original and actual_sha == filing.content_sha256
            record = ixbrl_record_r2(symbol, filing, market=self.market, original_verified=verified)
            record["fetched_at"] = header.fetched_at
            if self.source_repair_r5:
                source_baseline_records.append(record)
                if verified:
                    from domain.les_v7.financial_sources_r5 import repair_source_record_r5
                    record = repair_source_record_r5(record, filing, symbol=symbol, market=self.market,
                                                     original_verified=True)
            if self.signed_reports and verified and (filing.fiscal_year, filing.quarter) == (latest.year, latest.quarter):
                from infrastructure.fundamentals.signed_audit import SignedAuditStore
                from domain.les_v7.signed_audit import audit_proof_r6
                packet = SignedAuditStore(self.root / "signed-audit").select(record, cutoff=cutoff)
                if packet is not None:
                    record = dict(record, audit_evidence_r6=audit_proof_r6(filing, record, packet,
                        symbol=symbol, market=self.market, cutoff=cutoff))
            records.append(record)
            baseline = ixbrl_record(symbol, filing)
            baseline.update(market=self.market, fetched_at=header.fetched_at)
            baseline_records.append(baseline)
            original = dict(filing_id=filing.filing_id, period=f"{filing.fiscal_year}Q{filing.quarter}",
                source_url=filing.source_url, raw_path=header.raw_path, content_sha256=filing.content_sha256,
                available_at=filing.available_at, fetched_at=header.fetched_at,
                availability_source=filing.availability_source, revision_number=filing.revision_number,
                statement_scope=filing.statement_scope, taxonomy_schema_refs=list(filing.taxonomy_schema_refs),
                original_verified=bool(verified),
                verification="NATIVE_QUERY_HASH_CHECK_AND_ORIGINAL_REPARSE_NOT_DERIVED_SNAPSHOT")
            if not within_raw:
                diagnostics.append(dict(category="UNVERIFIED_ORIGINAL", filing_id=filing.filing_id, reason="RAW_PATH_OUTSIDE_CACHE"))
            elif not path.is_file():
                diagnostics.append(dict(category="MISSING_ORIGINAL", filing_id=filing.filing_id, reason="RAW_DOCUMENT_NOT_FOUND"))
            elif not verified:
                reason = "RAW_HASH_MISMATCH" if actual_sha != filing.content_sha256 else "NO_VERIFIED_PARSEABLE_NUMERIC_FACTS"
                diagnostics.append(dict(category="UNVERIFIED_ORIGINAL", filing_id=filing.filing_id, reason=reason))
            originals.append(original)
            if self.source_repair_r5:
                original["source_repair_id"] = digest(record["source_repair_r5"]) if record.get("source_repair_r5") else None
        if latest is not None and not any(r["period_end"] == latest.end_time.date().isoformat() for r in records):
            # A missing/conflicting latest original is not permission to call
            # an older report "latest". Keep unusable catalog candidates for
            # the native policy resolver, with no invented numeric values.
            for header in bounded_headers:
                if (header.fiscal_year, header.quarter) != (latest.year, latest.quarter): continue
                stub = dict(symbol=symbol, market=self.market, source="MOPS_IXBRL",
                    source_url=header.source_url, scope=header.statement_scope, currency="TWD", unit="thousand",
                    period_end=header.period_end or latest.end_time.date().isoformat(),
                    available_at=header.available_at, fetched_at=header.fetched_at,
                    availability_source=header.availability_source, content_sha256=header.content_sha256,
                    revision_id=header.content_sha256, revision_number=header.revision_number,
                    field_evidence={}, audit_evidence={}, catalog_only=True)
                records.append(stub)
                baseline_records.append(dict(stub))
                if self.source_repair_r5:
                    source_baseline_records.append(dict(stub))
            diagnostics.append(dict(category="LATEST_ORIGINAL_UNAVAILABLE", period=str(latest),
                reason="LATEST_CATALOG_PERIOD_NOT_REPLACED_WITH_OLDER_ORIGINAL"))
        industry_evidence = None
        if self.industry_master_context is not None:
            from infrastructure.crawler.security_master import select_security_master
            industry_evidence = select_security_master(self.industry_master_context, symbol, cutoff)
        return QualityAuditStock(tuple(records), tuple(originals), tuple(diagnostics),
            dict(header_count=len(headers), cutoff_eligible_headers=len(eligible), bounded_headers=len(bounded_headers),
                 latest_eligible_catalog_period=str(latest) if latest is not None else None,
                 raw_join_calls=int(bool(periods)), wanted_periods=[str(p) for p in wanted],
                 missing_periods=[str(p) for p in wanted if not any(r["period_end"] == p.end_time.date().isoformat()
                     and r.get("catalog_only") is not True for r in records)],
                 derived_snapshot_json_reads=0, header_policy="FILING_CATALOG_METADATA_FIRST_NO_DERIVED_SNAPSHOT_CERTIFICATION"),
            baseline_records=tuple(baseline_records),
            source_baseline_records=tuple({k: v for k, v in r.items() if k != "audit_evidence_r6"} for r in records) if self.signed_reports else
                tuple(source_baseline_records) if self.source_repair_r5 else None,
            industry_evidence=industry_evidence)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="Explicit timezone-aware financial evidence cutoff")
    parser.add_argument("--stocklist", type=Path, required=True)
    parser.add_argument("--universe-report", type=Path, help="Complete reference-verified report binding the full legacy index universe")
    parser.add_argument("--symbols", nargs="+", help="Explicit subset; full fixed universe remains in the manifest")
    parser.add_argument("--limit", type=int, help="Prefix limit after explicit subset selection")
    parser.add_argument("--market", choices=("TWSE", "TPEX"), default="TWSE")
    parser.add_argument("--industry-master-manifest", type=Path,
        help="Optional retained official security-master manifest; industry labels are usable only after observed availability")
    parser.add_argument("--policy", type=Path, help="Explicit registered R2/R3 policy JSON; omitted resolves the selected native policy")
    parser.add_argument("--profile", choices=("r2", "r3", "r4", "r5", "r6"), help="Quality policy; default R4; R5 same-original repair; R6 verified signed-report precedence")
    parser.add_argument("--data-kind", choices=("LOCAL_CACHE", "SYNTHETIC_ONLY"), default="LOCAL_CACHE")
    parser.add_argument("--output", type=Path, required=True, help="New immutable audit JSON outside all source cache/report paths")
    args = parser.parse_args(argv)
    output, root = args.output.resolve(), args.cache.resolve()
    if output.exists(): raise ValueError("QUALITY_AUDIT_OUTPUT_EXISTS")
    sources = [args.stocklist.resolve(), *([args.universe_report.resolve()] if args.universe_report else []),
               *([args.industry_master_manifest.resolve()] if args.industry_master_manifest else []),
               *([args.policy.resolve()] if args.policy else [])]
    if (output.is_relative_to(root) or output in sources or args.industry_master_manifest and
            output.is_relative_to(args.industry_master_manifest.resolve().parent.parent)):
        raise ValueError("QUALITY_AUDIT_OUTPUT_MUST_NOT_MODIFY_SOURCE")
    database = root / "mops.sqlite3"
    if not database.is_file(): raise ValueError("LOCAL_MOPS_CACHE_NOT_FOUND")
    versions = quality_source_versions()
    fixed, metadata = load_audit_universe(args.stocklist, universe_report=args.universe_report)
    metadata.update(cache_path=str(root), original_adapter_version=V6_FILING_SCHEMA_VERSION,
        code_versions=versions,
        cache_read_policy="ONE_READ_ONLY_SQLITE_SNAPSHOT_AND_ONE_BOUNDED_NATIVE_FACTS_JOIN_PER_STOCK")
    if args.industry_master_manifest:
        metadata["industry_master_source"] = dict(manifest_path=str(args.industry_master_manifest.resolve()),
            manifest_sha256=file_sha(args.industry_master_manifest),
            time_policy="FIRST_VERIFIED_OBSERVATION_NO_HISTORICAL_BACKDATE")
    policy = json.loads(args.policy.read_text()) if args.policy else None
    from domain.les_v7.quality_policy import (default_quality_policy, r2_quality_policy, r5_quality_policy, r6_quality_policy,
        is_r3_policy, is_r2_policy, is_r4_policy, is_r5_policy, is_r6_policy)
    if policy is not None and (args.profile == "r3" and not is_r3_policy(policy)
                              or args.profile == "r2" and not is_r2_policy(policy)
                              or args.profile == "r4" and not is_r4_policy(policy)
                              or args.profile == "r5" and not is_r5_policy(policy)
                              or args.profile == "r6" and not is_r6_policy(policy)):
        raise ValueError("QUALITY_AUDIT_PROFILE_POLICY_MISMATCH")
    resolver = r6_quality_policy if args.profile == "r6" else r5_quality_policy if args.profile == "r5" else r2_quality_policy if args.profile == "r2" else default_quality_policy
    signed = is_r6_policy(policy) if policy is not None else resolver is r6_quality_policy
    repaired = (is_r5_policy(policy) or is_r6_policy(policy)) if policy is not None else resolver in (r5_quality_policy, r6_quality_policy)
    metadata["requested_quality_profile"] = policy["profile_id"] if policy is not None else resolver()["profile_id"]
    if args.policy: metadata["policy_file_sha256"] = file_sha(args.policy)
    def progress(done, total, symbol):
        print(f"Quality audit {done}/{total} ({symbol})", flush=True)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        work = DailyWorkContext("quality-v7-audit:" + digest(dict(cutoff=args.cutoff, policy=policy,
            source=metadata)), max_bytes=16 * 1024 * 1024)
        try:
            with work.activate():
                report = audit_quality_v7(LocalQualityAuditSource(root, connection, market=args.market,
                    source_repair_r5=repaired, signed_reports=signed,
                    industry_master_manifest=args.industry_master_manifest), fixed, args.cutoff,
                    market=args.market, symbols=args.symbols, limit=args.limit, policy=policy,
                    data_kind=args.data_kind, source_metadata=metadata, progress=progress, policy_resolver=resolver)
        finally:
            work.close()
    if quality_source_versions() != versions: raise ValueError("QUALITY_AUDIT_SOURCE_CHANGED_DURING_JOB")
    write_once(output, report)
    print(quality_audit_summary(report))
    print(canonical_json(dict(audit_id=report["audit_id"], output=str(output), summary=report["summary"])))
    return 0


if __name__ == "__main__": raise SystemExit(main())
