"""Full native R13/R14 counterfactual over one pinned saved source universe.

No network, email, order, account ledger, PAE persistence or refresh is used.
Evaluation happens after R14 release, with an explicitly separate frozen source
snapshot clock. This is not historical deployment or certified historical PIT.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from application.evidence_context import ReadOnlyEvidence
from application.les_v7 import prepare_quality_context
from application.replay_contracts import canonical_json, digest
from domain.les_v7.allocation import account_positions
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.native_evidence import produce_native_receipts
from domain.les_v7.report import validate_report
from domain.security_master import content_id, verify_compact_security_selection
from domain.strategy import get_strategy
from examples.compare_les_r13 import file_hash
from infrastructure.daily_work import open_daily_work
from infrastructure.research_archive import write_bytes_once, write_once
from infrastructure.storage.report_evidence import ReportEvidenceWriter, iter_report_bundle

SCHEMA = "les-r13-r14-fixed-source-native-comparison/v1"
MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024


def local_report_copy(bundle, directory):
    """Consistent SQLite backup onto local SSD, never copy an open DB file."""
    bundle, directory = Path(bundle).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original_store = bundle.parent / "report_evidence.sqlite3"
    target_store = directory / original_store.name
    if target_store.exists():
        raise ValueError("LOCAL_REPORT_COPY_REQUIRES_NEW_DIRECTORY")
    original_stat = original_store.stat()
    before = file_hash(bundle)
    with sqlite3.connect(original_store.as_uri() + "?mode=ro", uri=True) as source:
        source.execute("PRAGMA query_only=1")
        with sqlite3.connect(target_store) as target:
            source.backup(target, pages=1024)
            if target.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("LOCAL_SQLITE_BACKUP_INTEGRITY_FAILED")
    write_bytes_once(directory / bundle.name, bundle.read_bytes())
    if file_hash(bundle) != before or file_hash(directory / bundle.name) != before:
        raise ValueError("LOCAL_REPORT_COPY_BUNDLE_CHANGED")
    final_stat = original_store.stat()
    receipt = dict(schema="les-report-local-sqlite-backup/v1", method="SQLITE_BACKUP_READ_ONLY_SOURCE",
        original_bundle=str(bundle), original_store=str(original_store),
        local_bundle=str(directory / bundle.name), local_store=str(target_store), bundle_sha256=before,
        original_store_before=dict(size=original_stat.st_size, mtime_ns=original_stat.st_mtime_ns,
                                   ctime_ns=original_stat.st_ctime_ns),
        original_store_after=dict(size=final_stat.st_size, mtime_ns=final_stat.st_mtime_ns,
                                  ctime_ns=final_stat.st_ctime_ns),
        local_store_sha256=file_hash(target_store), full_report_validation_still_required=True)
    write_once(directory / "source_copy_receipt.json", receipt)
    return directory / bundle.name, directory / "source_copy_receipt.json"


def prepare_rows(records, *, symbol, market, cutoff, quality_policy, companions):
    """Run the same public native source enrichment, then cutoff selection."""
    rows = [dict(row, market=market) if row.get("market") != market else row for row in records]
    context = dict(symbol=symbol, market=market, as_of=cutoff, filings=rows,
                   quality_policy=quality_policy)
    prepare_quality_context(context, **{key: companions.get(key, {}) for key in
        ("financial_audits", "financial_supplements", "financial_repairs_r5", "signed_financial_audits")})
    return [row for row in context["filings"] if timestamp(row["available_at"]) <= timestamp(cutoff)]


class HashMatchedFilings:
    """One read-only SQLite connection; never use active/latest as authority.

    Immutable snapshot candidates are accepted ONLY after the final source
    companion-enriched filing vector equals the saved engine financials hash.
    Source companions are rederived from existing hash-verified originals with
    the existing local-only repository/service. No missing value is imputed.
    """

    def __init__(self, database, symbols):
        from application.fundamental_service import MopsFundamentalService
        from infrastructure.fundamentals.mops_queries import MopsQueryRepository
        from infrastructure.fundamentals.mops_unit_of_work import SqliteFundamentalUnitOfWork
        self.database = Path(database).resolve()
        self.connection = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)
        self.connection.execute("PRAGMA query_only=1")
        queries = MopsQueryRepository(lambda: self.connection, lambda: None,
                                     raw_root=self.database.parent / "raw")
        connection = self.connection

        class ReadOnlySources:
            @contextmanager
            def unit_of_work(self):
                yield SqliteFundamentalUnitOfWork(connection)
            load_v7_financial_sources = queries.load_v7_financial_sources

        self.service = MopsFundamentalService(object(), repository=ReadOnlySources(),
            filing_documents=object(), filing_parser=object())
        self.headers = {}
        self.active_candidates = {}
        self.loaded_symbols = set()
        self.has_refresh_state = bool(self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='refresh_state'").fetchone())
        self.counts = Counter()
        self._load_headers(symbols)

    def _load_headers(self, symbols):
        """Covering-index hints only; content equality still establishes reuse."""
        ordered = sorted(set(symbols) - self.loaded_symbols)
        self.loaded_symbols.update(ordered)
        for symbol in ordered:
            self.headers.setdefault(symbol, [])
        for offset in range(0, len(ordered), 400):
            chunk = ordered[offset:offset + 400]
            for symbol, key, sector in self.connection.execute(
                # Covering primary-key index ONLY. Even created_at_epoch after
                # a giant JSON column can cause overflow-chain traversal.
                "SELECT symbol,filing_set_sha256,sector "
                "FROM fundamental_snapshots WHERE symbol IN (" + ",".join("?" for _ in chunk) + ") "
                "ORDER BY symbol,filing_set_sha256,sector", chunk):
                self.headers.setdefault(symbol, []).append((key, sector))
            if self.has_refresh_state:
                # A pointer only prioritizes a candidate. It never establishes
                # authority: the original saved financial vector hash must
                # still match after all source companions are rederived.
                self.active_candidates.update(self.connection.execute(
                    "SELECT symbol,active_snapshot_cache_key FROM refresh_state WHERE symbol IN (" +
                    ",".join("?" for _ in chunk) + ")", chunk))
        for symbol in ordered:
            self.headers[symbol].sort(key=lambda row: row[0] != self.active_candidates.get(symbol))

    def close(self):
        self.connection.close()

    def restore(self, report):
        symbol = report["symbol"]
        diagnostics = report["diagnostics"]
        binding = diagnostics["evidence_binding"]
        expected = binding["financials_hash"]
        cutoff, market = binding["cutoff"], binding["market"]
        policy = diagnostics["rule_baseline"]["policy"]["quality_policy"]
        if expected == identity([]):
            return [], dict(method="EXACT_SAVED_EMPTY_FINANCIAL_VECTOR", financials_hash=expected)
        # A saved report can contain a separately identified issuer's original
        # filing vector while the traded instrument remains UNKNOWN/OUT_OF_SCOPE.
        # Read the ORIGINAL packet, never infer a relationship from a suffix or
        # security name. A hint only finds candidates; the whole vector hash
        # below must still equal the original instrument report's binding.
        packet = diagnostics.get("native_evidence", {}).get("source_packet") or {}
        declared = [*packet.get("filings", ()), *packet.get("valuation_filings", ())]
        if any(not isinstance(row, dict) or not isinstance(row.get("symbol"), str)
               or not row["symbol"] for row in declared):
            raise ValueError("SAVED_PACKET_FILING_IDENTITY_UNVERIFIED")
        source_symbols = {row["symbol"] for row in declared}
        if len(source_symbols) > 1:
            raise ValueError("SAVED_PACKET_FILING_IDENTITY_CONFLICT")
        candidates = [symbol, *sorted(source_symbols - {symbol})]
        self._load_headers(candidates)
        attempts = []
        snapshots = ((source_symbol, key, sector) for source_symbol in candidates
                     for key, sector in self.headers.get(source_symbol, []))
        for source_symbol, key, sector in snapshots:
            self.counts["snapshot_candidates"] += 1
            row = self.connection.execute(
                "SELECT CASE WHEN octet_length(snapshot_json)<=? THEN json_extract(snapshot_json,'$.les_v6_mops_filings') END, "
                "CASE WHEN octet_length(snapshot_json)<=? THEN json_extract(snapshot_json,'$.les_v6_filings') END, "
                "octet_length(snapshot_json) FROM fundamental_snapshots "
                "WHERE symbol=? AND filing_set_sha256=? AND sector=?",
                (MAX_SNAPSHOT_BYTES, MAX_SNAPSHOT_BYTES, source_symbol, key, sector)).fetchone()
            if row is None:
                raise ValueError("IMMUTABLE_SNAPSHOT_DISAPPEARED")
            if row[2] > MAX_SNAPSHOT_BYTES:
                attempts.append(dict(source_symbol=source_symbol, snapshot_key=key, reason="SNAPSHOT_CAPACITY_EXCEEDED"))
                continue
            raw = row[0] or row[1] or "[]"
            if len(raw.encode()) > MAX_SNAPSHOT_BYTES:
                raise ValueError("FILING_VECTOR_CAPACITY_EXCEEDED")
            records = json.loads(raw)
            if not isinstance(records, list):
                raise ValueError("INVALID_SNAPSHOT_FILING_VECTOR")
            try:
                companions = self.service.load_v7_financial_sources(source_symbol, as_of=cutoff,
                    records=records, market=market, source_repair=True)
                selected = prepare_rows(records, symbol=source_symbol, market=market, cutoff=cutoff,
                                        quality_policy=policy, companions=companions)
                actual = identity(selected)
            except (KeyError, TypeError, ValueError) as error:
                attempts.append(dict(source_symbol=source_symbol, snapshot_key=key, reason=str(error)))
                continue
            if actual == expected:
                self.counts["exact_matches"] += 1
                return selected, dict(method="IMMUTABLE_SNAPSHOT_PLUS_VERIFIED_LOCAL_ORIGINAL_COMPANIONS",
                    snapshot_key=key, sector=sector, snapshot_byte_count=row[2],
                    snapshot_filing_vector_sha256=digest(records), financials_hash=actual,
                    selected_filing_count=len(selected), companions_sha256=digest(companions),
                    source_selection_cutoff=cutoff, report_symbol=symbol, restored_source_symbol=source_symbol,
                    source_identity_hint_basis="REPORT_SYMBOL" if source_symbol == symbol else "SAVED_PACKET_EXPLICIT_FILING_SYMBOL",
                    security_identity_not_reassigned=True)
            attempts.append(dict(source_symbol=source_symbol, snapshot_key=key, reason="FINANCIAL_VECTOR_HASH_MISMATCH", actual=actual))
        raise ValueError("SAVED_FINANCIAL_VECTOR_NOT_RECOVERED:" + canonical_json(attempts))


def _security(selection, *, source_cutoff, evaluation_cutoff):
    if selection is None:
        return None
    original = verify_compact_security_selection(selection, cutoff=source_cutoff)
    selected = dict(original, cutoff=evaluation_cutoff)
    selected["selection_id"] = content_id({key: value for key, value in selected.items()
                                            if key not in {"master", "selection_id"}})
    return verify_compact_security_selection(selected, cutoff=evaluation_cutoff)


def assert_snapshot_visibility(packet):
    """No new clock may admit a raw operand unavailable in the frozen snapshot.

    Check only known financial/quote/source availability contracts, not arbitrary
    nested keys such as policy release or later verification metadata. Missing
    and malformed evidence remains for native UNKNOWN/failure handling.
    """
    cutoff = timestamp(packet["cutoff"])
    checked = 0

    def check(value, path):
        nonlocal checked
        if value is None:
            return
        try:
            when = timestamp(value)
        except (TypeError, ValueError):
            return
        checked += 1
        if when > cutoff:
            raise ValueError("FROZEN_SNAPSHOT_CONTAINS_POST_CUTOFF_SOURCE:" + path)

    def records(rows, path):
        for index, row in enumerate(rows or ()):
            check(row.get("available_at"), f"{path}[{index}].available_at")

    def market(block, path):
        evidence = block.get("market_evidence") or {}
        check(evidence.get("source_available_at"), path + ".market_evidence.source_available_at")
        check((evidence.get("valuation_history_selection") or {}).get("observed_at"),
              path + ".market_evidence.valuation_history_selection.observed_at")
        for index, row in enumerate(block.get("market_rows") or ()):
            day = str(row.get("Date", ""))[:10]
            if len(day) == 10:
                check(day + "T13:30:00+08:00", f"{path}.market_rows[{index}].close_at")
            for key in ("Available At", "available_at"):
                check(row.get(key), f"{path}.market_rows[{index}].{key}")
        security = block.get("security_evidence") or {}
        check((security.get("master") or {}).get("available_at"), path + ".security.master.available_at")

    records(packet.get("valuation_filings"), "valuation_filings")
    records(packet.get("filings"), "filings")
    for index, selection in enumerate(packet.get("official_pe") or ()):
        prefix = f"official_pe[{index}]"
        check(selection.get("source_available_at"), prefix + ".source_available_at")
        for proof_index, proof in enumerate(selection.get("selection_proofs") or ()):
            check((proof.get("manifest") or {}).get("available_at"),
                  f"{prefix}.selection_proofs[{proof_index}].manifest.available_at")
    market(packet, "source_packet")
    restriction = packet.get("restriction_evidence") or {}
    for key in ("source_available_at", "available_at", "observed_at"):
        check(restriction.get(key), "restriction_evidence." + key)
    comparable = packet.get("r10_valuation_context")
    if comparable:
        for index, block in enumerate([comparable["issuer"], *comparable.get("peers", ())]):
            prefix = f"r10_valuation_context.blocks[{index}]"
            records(block.get("annual_filings"), prefix + ".annual_filings")
            market(block, prefix)
    return dict(policy="RAW_OPERANDS_VISIBLE_BY_SAVED_SOURCE_SNAPSHOT_NO_NEW_CUTOFF_INFORMATION",
                source_snapshot_cutoff=packet["cutoff"], checked_timestamp_count=checked,
                post_cutoff_source_count=0)


def reselect_packet(packet, *, evaluation_cutoff):
    """Change derived selection clocks only; never raw availability/content."""
    evaluation_cutoff = timestamp(evaluation_cutoff).isoformat()
    result = deepcopy(packet)
    source_cutoff = packet["cutoff"]
    if timestamp(evaluation_cutoff) < timestamp(source_cutoff):
        raise ValueError("COMPARISON_CANNOT_PRECEDE_SOURCE_SNAPSHOT")
    result["cutoff"] = evaluation_cutoff
    if "security_evidence" in result:
        result["security_evidence"] = _security(result["security_evidence"],
            source_cutoff=source_cutoff, evaluation_cutoff=evaluation_cutoff)
    comparable = result.get("r10_valuation_context")
    if comparable is not None:
        old_cutoff = comparable["cutoff"]
        comparable["cutoff"] = evaluation_cutoff
        for block in [comparable["issuer"], *comparable.get("peers", [])]:
            block["security_evidence"] = _security(block.get("security_evidence"),
                source_cutoff=old_cutoff, evaluation_cutoff=evaluation_cutoff)
    return result


def _account(saved, evaluation_cutoff):
    if not isinstance(saved, dict) or any(saved.get(key) is None for key in
            ("value", "available_cash", "capacity_weight", "intended_weight", "positions", "as_of")):
        raise ValueError("EXPLICIT_SAVED_ACCOUNT_REQUIRED")
    if account_positions(saved["positions"]) is None:
        raise ValueError("EXPLICIT_SAVED_POSITIONS_REQUIRED")
    # Values are reused as a declared counterfactual input, not a claim that a
    # real historical/current account remained unchanged between these clocks.
    return dict(saved, as_of=evaluation_cutoff)


def recompute(report, filings, *, evaluation_cutoff, policy_version, packet=None):
    from application.low_entry_v7_daily import finalize_daily_report
    evaluation_cutoff = timestamp(evaluation_cutoff).isoformat()
    diagnostics = report["diagnostics"]
    binding = diagnostics["evidence_binding"]
    if identity(filings) != binding["financials_hash"]:
        raise ValueError("SAVED_FINANCIAL_VECTOR_HASH_REQUIRED")
    packet = reselect_packet(diagnostics["native_evidence"]["source_packet"],
        evaluation_cutoff=evaluation_cutoff) if packet is None else packet
    bars = diagnostics["rule_baseline"]["market_rows"]
    if identity(bars) != binding["bars_hash"] or identity(bars) != identity(packet["market_rows"]):
        raise ValueError("SAVED_BAR_VECTOR_HASH_REQUIRED")
    context = dict(symbol=report["symbol"], market=binding["market"], as_of=evaluation_cutoff,
        filings=filings, model_bundle=None, quality_policy=None, receipts={},
        external_evidence_ids=diagnostics["evidence_binding"]["external_evidence_ids"],
        rule_baseline=True, rule_baseline_policy_version=policy_version,
        native_evidence_source=packet)
    result = get_strategy("low_entry_score_v7").evaluate(pd.DataFrame(bars),
        context=ReadOnlyEvidence(context), receipt_producer=produce_native_receipts)
    account = _account(report.get("pae_authorization", {}).get("account"), evaluation_cutoff)
    return finalize_daily_report(result, account=account,
        daily_input_errors=diagnostics.get("daily_input_errors", ()))


def valuation_business(diagnostics):
    """Retain every scenario operand/result without repeating raw source graphs."""
    receipt = diagnostics.get("native_evidence", {}).get("produced_receipts", {}).get("VALUATION")
    raw = receipt.get("valuation_diagnostics") if isinstance(receipt, dict) else None
    if not isinstance(raw, dict):
        return None
    calculation = {key: raw[key] for key in (
        "schema", "policy_version", "policy_id", "status", "reasons", "inputs", "assumptions",
        "selected_method", "method_small_only", "method_position_multiplier", "anchor", "observed_growth",
        "scenarios", "gate_checks", "bear_permanent_loss", "discounted_bear_value_shortfall",
        "bear_shortfall_band", "bear_metric", "multiple_selection", "prerequisite_result_id") if key in raw}
    if isinstance(raw.get("comparables"), dict):
        comp = raw["comparables"]
        summary = {key: comp[key] for key in ("schema", "status", "reasons", "industry", "context_id",
            "industry_source_status", "industry_profile_status", "issuer_month_count", "peer_count",
            "available_at", "peer_exclusions") if key in comp}
        for component in ("issuer_history", "peers"):
            if isinstance(comp.get(component), dict):
                summary[component] = {key: comp[component][key] for key in
                    ("status", "count", "minimum", "median", "reasons") if key in comp[component]}
        calculation["comparables"] = summary
    if isinstance(raw.get("recovery"), dict):
        calculation["recovery"] = {key: raw["recovery"][key] for key in
            ("status", "reasons", "values", "available_at", "source_ids", "source_record_ids") if key in raw["recovery"]}
    return calculation


def business(report):
    diagnostics, envelope = report["diagnostics"], report["envelope"]
    return dict(strategy=envelope["strategy"], score=envelope["score"],
        quality=envelope["quality"], quality_ledger=diagnostics["quality"]["metric_ledger"],
        technical_vetoes=diagnostics["rule_baseline"].get("technical_vetoes"),
        baseline_calculation={key: value for key, value in diagnostics["rule_baseline"].items()
                              if key not in {"market_rows", "policy"}},
        valuation_calculation=valuation_business(diagnostics), structure_plan=diagnostics.get("structure_plan"),
        context=envelope["event"]["context"], factor_ledger=diagnostics["factor_ledger"],
        gates=diagnostics["gate_receipts"], readiness=diagnostics.get("readiness"),
        reference_plan=diagnostics["reference_plan"], entry_admission=diagnostics.get("entry_admission"),
        starter=diagnostics["starter"], confirmed=diagnostics["confirmed"],
        starter_gate=diagnostics["starter_gate"], confirmed_gate=diagnostics["confirmed_gate"],
        daily_input_errors=diagnostics.get("daily_input_errors", []),
        pae_authorization=report["pae_authorization"])


def differences(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            location = path + "." + key if path else key
            if key not in before or key not in after:
                yield dict(path=location, before=before.get(key), after=after.get(key))
            else:
                yield from differences(before[key], after[key], location)
    elif before != after:
        yield dict(path=path, before=before, after=after)


def compare_report(report, filings, *, evaluation_cutoff):
    from domain.les_v7.policy_r13 import BASELINE_VERSION as R13
    from domain.les_v7.policy_r14 import BASELINE_VERSION as R14, AVAILABLE_AT
    evaluation_cutoff = timestamp(evaluation_cutoff).isoformat()
    validate_report(report)
    visibility = assert_snapshot_visibility(report["diagnostics"]["native_evidence"]["source_packet"])
    if timestamp(evaluation_cutoff) < timestamp(AVAILABLE_AT):
        raise ValueError("R14_COMPARISON_BEFORE_POLICY_RELEASE")
    packet = reselect_packet(report["diagnostics"]["native_evidence"]["source_packet"],
                             evaluation_cutoff=evaluation_cutoff)
    old = recompute(report, filings, evaluation_cutoff=evaluation_cutoff, policy_version=R13, packet=packet)
    new = recompute(report, filings, evaluation_cutoff=evaluation_cutoff, policy_version=R14, packet=packet)
    for result in (old, new):
        validate_report(result)
        if result["pae_authorization"]["live_order_authorized"] is not False:
            raise ValueError("COUNTERFACTUAL_CANNOT_AUTHORIZE_ORDERS")
    before, after = business(old), business(new)
    return old, new, dict(symbol=report["symbol"], source_snapshot_cutoff=report["envelope"]["event"]["cutoff_at"],
        evaluation_cutoff=evaluation_cutoff, state="FULL_NATIVE_RECOMPUTED",
        frozen_source_packet_sha256=digest(report["diagnostics"]["native_evidence"]["source_packet"]),
        reselected_source_packet_sha256=digest(packet), financials_sha256=identity(filings),
        bars_sha256=report["diagnostics"]["evidence_binding"]["bars_hash"],
        original_account_sha256=report["pae_authorization"]["account_sha256"],
        snapshot_visibility=visibility,
        evaluation_account_sha256=old["pae_authorization"]["account_sha256"],
        account_inputs_equal=old["pae_authorization"]["account"] == new["pae_authorization"]["account"],
        original_decision=report["envelope"]["strategy"]["strategy_decision"],
        r13=before, r14=after, differences=list(differences(before, after)))


def markdown(result):
    summary = result["summary"]
    decisions = sorted(set(summary["r13_decisions"]) | set(summary["r14_decisions"]))
    lines = ["# R13／R14：固定舊證據的完整原生比較", "",
        f"股票範圍：{summary['total']} 檔；完整重算：{summary['states'].get('FULL_NATIVE_RECOMPUTED', 0)} 檔；"
        f"缺口：{summary['states'].get('SOURCE_OR_RECOMPUTATION_GAP', 0)} 檔。", "",
        f"研究截止：`{result['evaluation_cutoff']}`；原證據截止另行保留，未倒填部署時間。", "",
        "| 原生決策 | R13 | R14 |", "|---|---:|---:|"]
    lines.extend(f"| {decision} | {summary['r13_decisions'].get(decision, 0)} | "
                 f"{summary['r14_decisions'].get(decision, 0)} |" for decision in decisions)
    lines.extend(["", "兩版使用相同原始證據、完整財報雜湊與保存的帳戶數值，重新計算品質、技術分數、"
        "安全閘門、交易計畫、決策及候選限額；完整結果與所有業務差異保存在 JSON／證據圖。", "",
        "這是目前政策對固定舊證據的反事實研究，不是歷史當時部署、獲利或成交證明。"
        "未模擬 EREM、完整 PAE Persistence、帳戶累積、成交或費用帳本；未操作正式帳戶或下單。", "",
        f"來源 bundle SHA256：`{result['source_bundle_sha256']}`", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-bundle", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--fundamental-db", required=True, type=Path)
    parser.add_argument("--evaluation-cutoff", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-copy-receipt", type=Path)
    args = parser.parse_args(argv)
    evaluation_cutoff = timestamp(args.evaluation_cutoff).isoformat()
    if file_hash(args.report_bundle) != args.expected_bundle_sha256:
        raise ValueError("COMPARISON_SOURCE_BUNDLE_HASH_MISMATCH")
    source_copy = None
    if args.source_copy_receipt:
        source_copy = json.loads(args.source_copy_receipt.read_text())
        if (source_copy.get("schema") != "les-report-local-sqlite-backup/v1"
                or source_copy.get("method") != "SQLITE_BACKUP_READ_ONLY_SOURCE"
                or source_copy.get("bundle_sha256") != args.expected_bundle_sha256
                or Path(source_copy["local_bundle"]).resolve() != args.report_bundle.resolve()
                or file_hash(source_copy["local_store"]) != source_copy.get("local_store_sha256")):
            raise ValueError("COMPARISON_SOURCE_COPY_RECEIPT_INVALID")
    # The small reference envelope fixes scope; full report trees are streamed.
    bundle = json.loads(args.report_bundle.read_text())
    symbols = [ref["symbol"] for key in ("reports", "stale_reports") for ref in bundle.get(key, [])]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("COMPARISON_SCOPE_EMPTY_OR_DUPLICATE")
    del bundle
    rows = []
    restorer = HashMatchedFilings(args.fundamental_db, symbols)
    try:
        with open_daily_work(max_bytes=64 * 1024 * 1024) as work, ReportEvidenceWriter(
                args.output_dir / "report_evidence.sqlite3") as writer:
            versions = asdict(work.check_version())
            for saved in iter_report_bundle(args.report_bundle):
                work.check_version()
                try:
                    validate_report(saved)
                    filings, restoration = restorer.restore(saved)
                    old, new, row = compare_report(saved, filings, evaluation_cutoff=evaluation_cutoff)
                    row.update(restoration=restoration, r13_report=writer.retain(old), r14_report=writer.retain(new))
                    del old, new, filings
                except (KeyError, TypeError, ValueError, OSError) as error:
                    row = dict(symbol=saved["symbol"], state="SOURCE_OR_RECOMPUTATION_GAP",
                               error=str(error), original_report_sha256=digest(saved))
                rows.append(row)
                print(json.dumps(dict(event="comparison_stock_completed", symbol=saved["symbol"],
                                      state=row["state"], completed=len(rows), total=len(symbols))), flush=True)
                del saved
            work.check_version()
        if file_hash(args.report_bundle) != args.expected_bundle_sha256:
            raise ValueError("COMPARISON_INPUT_CHANGED_DURING_JOB")
        result = dict(schema=SCHEMA, source_bundle_sha256=args.expected_bundle_sha256,
            source_copy_receipt=source_copy,
            source_copy_receipt_sha256=file_hash(args.source_copy_receipt) if args.source_copy_receipt else None,
            evaluation_cutoff=evaluation_cutoff, source_snapshot_cutoffs=sorted({r["source_snapshot_cutoff"] for r in rows
                                                                              if "source_snapshot_cutoff" in r}),
            universe_symbols=symbols, versions=versions, comparison_program_sha256=file_hash(__file__),
            historical_information_set_certified=False, historical_deployment_claim=False,
            scope="CURRENT_POLICY_COUNTERFACTUAL_FIXED_OLDER_RAW_SOURCE_SNAPSHOT",
            account_policy="EXPLICIT_SAVED_VALUES_REUSED_NOT_CURRENT_ACCOUNT_OR_LEDGER_CLAIM",
            pae_persistence="NOT_PRESENT_IN_PER_STOCK_NATIVE_REPORT_NOT_REBASED_OR_SIMULATED",
            order_authority=False, full_financials_hash_match_required=True,
            summary=dict(total=len(rows), states=dict(Counter(r["state"] for r in rows)),
                r13_decisions=dict(Counter(r["r13"]["strategy"]["strategy_decision"] for r in rows if "r13" in r)),
                r14_decisions=dict(Counter(r["r14"]["strategy"]["strategy_decision"] for r in rows if "r14" in r)),
                restore_counts=dict(restorer.counts)), records=rows)
        result["comparison_id"] = digest(result)
        write_once(args.output_dir / "comparison.json", result)
        write_bytes_once(args.output_dir / "comparison.md", markdown(result).encode())
        write_once(args.output_dir / "complete.json", dict(comparison_id=result["comparison_id"],
            comparison_sha256=file_hash(args.output_dir / "comparison.json"),
            markdown_sha256=file_hash(args.output_dir / "comparison.md"),
            status="COMPLETE" if all(r["state"] == "FULL_NATIVE_RECOMPUTED" for r in rows) else "PARTIAL_GAPS_RETAINED"))
        print(json.dumps(dict(comparison_id=result["comparison_id"], summary=result["summary"]), ensure_ascii=False))
        return 0
    finally:
        restorer.close()


if __name__ == "__main__":
    raise SystemExit(main())
