"""Full native R16/R17 comparison on one immutable, complete source universe.

The existing source restorer, native plugin, receipt producer, public daily
finalizer and report validator are reused. No network, mail, formal account,
PAE persistence or fill ledger is touched. A conditional signal is not a fill.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.replay_contracts import digest
from domain.les_v7.contracts import timestamp
from domain.les_v7.report import validate_report
from examples import compare_les_r14 as source_runtime
from examples import compare_les_r16 as previous
from examples.compare_les_r13 import file_hash
from infrastructure.daily_work import open_daily_work
from infrastructure.research_archive import write_bytes_once, write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.report_evidence import ReportEvidenceWriter, iter_report_bundle

SCHEMA = "les-r16-r17-fixed-source-native-comparison/v1"
PROGRAM_PATHS = (Path(__file__).resolve(), Path(previous.__file__).resolve(),
                 Path(previous.previous.__file__).resolve(), Path(source_runtime.__file__).resolve(),
                 Path(__file__).resolve().with_name("compare_les_r13.py"))


def program_bindings(paths=PROGRAM_PATHS):
    """Bind helpers outside the runtime code guard to their actual bytes."""
    return previous.program_bindings(paths)


def check_program_bindings(expected, paths=PROGRAM_PATHS):
    previous.check_program_bindings(expected, paths)


def differences(before, after, path=""):
    """Lossless field differences, including absent versus explicitly null."""
    yield from previous.differences(before, after, path)


def business(report):
    """Every computational diagnostic survives, including future branches.

    Full-report differences and retained report hashes remain authoritative;
    this projection only avoids repeating the large unchanged evidence graph.
    """
    result = previous.business(report)
    result["r17_route_evaluation"] = report["diagnostics"].get("r17_route_evaluation")
    return result


def compare_report(report, filings, *, evaluation_cutoff):
    from domain.les_v7.policy_r16 import BASELINE_VERSION as R16
    from domain.les_v7.policy_r17 import BASELINE_VERSION as R17, AVAILABLE_AT

    cutoff = timestamp(evaluation_cutoff).isoformat()
    account = source_runtime._account(report.get("pae_authorization", {}).get("account"), cutoff)
    validate_report(report)
    source = report["diagnostics"]["native_evidence"]["source_packet"]
    visibility = source_runtime.assert_snapshot_visibility(source)
    if timestamp(cutoff) < timestamp(AVAILABLE_AT):
        raise ValueError("R17_COMPARISON_BEFORE_POLICY_RELEASE")
    packet = source_runtime.reselect_packet(source, evaluation_cutoff=cutoff)
    old = source_runtime.recompute(report, filings, evaluation_cutoff=cutoff, policy_version=R16, packet=packet)
    new = source_runtime.recompute(report, filings, evaluation_cutoff=cutoff, policy_version=R17, packet=packet)
    saved_binding = report["diagnostics"]["evidence_binding"]
    old_binding = old["diagnostics"]["evidence_binding"]
    new_binding = new["diagnostics"]["evidence_binding"]
    for value in old, new:
        validate_report(value)
        if (timestamp(value["envelope"]["event"]["cutoff_at"]) != timestamp(cutoff)
                or timestamp(value["diagnostics"]["evidence_binding"]["cutoff"]) != timestamp(cutoff)):
            raise ValueError("COUNTERFACTUAL_EVALUATION_CUTOFF_CHANGED")
        if value["pae_authorization"]["live_order_authorized"] is not False:
            raise ValueError("COUNTERFACTUAL_CANNOT_AUTHORIZE_ORDERS")
        if value["pae_authorization"]["account"] != account:
            raise ValueError("COUNTERFACTUAL_ACCOUNT_CHANGED")
    if old["pae_authorization"]["account_sha256"] != new["pae_authorization"]["account_sha256"]:
        raise ValueError("COUNTERFACTUAL_ACCOUNT_HASH_CHANGED")
    for key in ("symbol", "market", "bars_hash", "financials_hash", "external_evidence_ids"):
        if old_binding[key] != new_binding[key] or old_binding[key] != saved_binding[key]:
            raise ValueError("COUNTERFACTUAL_RAW_SOURCE_BINDING_CHANGED:" + key)
    if old_binding["native_source_id"] != new_binding["native_source_id"]:
        raise ValueError("COUNTERFACTUAL_SELECTED_SOURCE_CHANGED")
    raw_identity = dict(symbol=report["symbol"], market=saved_binding["market"],
        source_cutoff=saved_binding["cutoff"], bars_hash=saved_binding["bars_hash"],
        financials_hash=saved_binding["financials_hash"], frozen_source_packet_sha256=digest(source),
        external_evidence_ids=saved_binding["external_evidence_ids"],
        calendar_evidence_sha256=digest(source["calendar_evidence"])
            if source.get("calendar_evidence") is not None else None)
    before, after = business(old), business(new)
    return old, new, dict(symbol=report["symbol"], state="FULL_NATIVE_RECOMPUTED",
        source_snapshot_cutoff=report["envelope"]["event"]["cutoff_at"], evaluation_cutoff=cutoff,
        original_report_sha256=digest(report), raw_source_identity=raw_identity, raw_source_id=digest(raw_identity),
        frozen_source_packet_sha256=raw_identity["frozen_source_packet_sha256"],
        reselected_source_packet_sha256=digest(packet), financials_sha256=saved_binding["financials_hash"],
        bars_sha256=saved_binding["bars_hash"], snapshot_visibility=visibility,
        calendar_source_present=source.get("calendar_evidence") is not None,
        original_account_sha256=report["pae_authorization"]["account_sha256"],
        evaluation_account_sha256=old["pae_authorization"]["account_sha256"], account_inputs_equal=True,
        original_decision=report["envelope"]["strategy"]["strategy_decision"],
        r16=before, r17=after, differences=list(differences(before, after)),
        full_report_differences=list(differences(old, new)))


def markdown(result):
    summary = result["summary"]
    names = sorted(set(summary["r16_decisions"]) | set(summary["r17_decisions"]))
    lines = ["# R16／R17：固定舊證據的完整原生比較", "",
        f"股票範圍：{summary['total']} 檔；完整重算：{summary['states'].get('FULL_NATIVE_RECOMPUTED', 0)} 檔；"
        f"缺口：{summary['states'].get('SOURCE_OR_RECOMPUTATION_GAP', 0)} 檔。", "",
        f"共同評估截止：`{result['evaluation_cutoff']}`；原始來源截止另行保留。", "",
        "| 原生決策 | R16 | R17 |", "|---|---:|---:|"]
    lines.extend(f"| {key} | {summary['r16_decisions'].get(key, 0)} | {summary['r17_decisions'].get(key, 0)} |"
                 for key in names)
    lines.extend(["", "兩版使用同一完整財報向量、行情、原始來源與明確保存的帳戶數值，重新執行原生策略、"
        "receipt producer、公開每日 finalizer 與 report validator。core／tactical、R17 結構條件路徑、"
        "分數、Hard Gates、限倉及完整報告逐欄差異均留存。", "",
        "較早來源的交易限制不會因較晚評估而刷新；過期仍 UNKNOWN。零買訊不代表資料修復失敗，"
        "亦不能用出現買訊證明低點捕捉有效。", "",
        "來源缺少可驗證原始行事曆時，不以現今來源補成歷史已知。可選 CAS seed 只備份舊成果的"
        "內容定址節點以減少重複寫入；兩版研究仍完整重新執行，不重用舊分數、決策或資格。", "",
        "條件式 BUY_SMALL 是尚未成交的研究訊號：僅下一已核實官方交易日開盤位於合法價帶內，"
        "才可進一步檢查成交。日內高低價觸帶不作成交；本次未模擬任何成交。", "",
        "這是目前政策對固定舊證據的反事實研究，不是歷史當時部署、獲利或成交證明。"
        "未模擬 EREM、完整 PAE Persistence、帳戶累積或費用帳本；未抓取新資料、寄信或下單。", "",
        "原件／MOPS snapshots／外部證明仍須保留；比較 JSON、runtime 程式物件與結果 DB 並非自足可攜原始資料集。", "",
        f"來源 bundle SHA256：`{result['source_bundle_sha256']}`", ""])
    return "\n".join(lines)


def _copy_receipt(path, bundle, expected):
    if path is None:
        return None
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "les-report-local-sqlite-backup/v1"
            or receipt.get("method") != "SQLITE_BACKUP_READ_ONLY_SOURCE"
            or receipt.get("bundle_sha256") != expected
            or Path(receipt["local_bundle"]).resolve() != bundle.resolve()
            or file_hash(receipt["local_store"]) != receipt.get("local_store_sha256")):
        raise ValueError("COMPARISON_SOURCE_COPY_RECEIPT_INVALID")
    return receipt


def _store_receipt(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=1")
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("COMPARISON_RESULT_STORE_INTEGRITY_FAILED")
    return dict(name=path.name, sha256=file_hash(path), size_bytes=path.stat().st_size,
                quick_check="ok", checkpoint="CLOSED_WRITER")


def _seed_store(source, expected_sha256, target):
    """Read-only SQLite backup of an immutable CAS, not old model results.

    Native reports are recalculated regardless of seed presence. The existing
    graph writer still validates every reused node against its actual content.
    The completed older store is never opened writable or linked into a mutable
    new job. Nonempty WAL/journal sidecars cannot be omitted from an asserted
    immutable file hash; such seeds must first be closed by their own producer.
    """
    if source is None and expected_sha256 is None:
        return None
    if source is None or expected_sha256 is None:
        raise ValueError("COMPARISON_SEED_PATH_AND_SHA256_REQUIRED")
    source, target = Path(source).resolve(), Path(target).resolve()
    sidecars = [Path(str(source) + suffix) for suffix in ("-wal", "-journal")]
    def immutable_source():
        if any(path.exists() and path.stat().st_size for path in sidecars):
            raise ValueError("COMPARISON_SEED_STORE_NOT_CLOSED")
        if file_hash(source) != expected_sha256:
            raise ValueError("COMPARISON_SEED_STORE_HASH_MISMATCH")
    if target.exists():
        raise ValueError("COMPARISON_SEED_TARGET_ALREADY_EXISTS")
    immutable_source()
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        original.execute("PRAGMA query_only=1")
        tables = {row[0] for row in original.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables != {"report_objects"}:
            raise ValueError("COMPARISON_SEED_REQUIRES_SEPARATE_EVIDENCE_STORE")
        columns = [(row[1], row[2]) for row in original.execute("PRAGMA table_info(report_objects)")]
        if columns != [("sha256", "TEXT"), ("body", "BLOB")]:
            raise ValueError("COMPARISON_SEED_STORE_SCHEMA_INVALID")
        with closing(sqlite3.connect(target)) as copied:
            original.backup(copied, pages=1024)
            if copied.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("COMPARISON_SEED_BACKUP_INTEGRITY_FAILED")
            row_count = copied.execute("SELECT COUNT(*) FROM report_objects").fetchone()[0]
    immutable_source()
    return dict(schema="les-comparison-evidence-cas-seed/v1", source_store=str(source),
        source_sha256=expected_sha256, source_size_bytes=source.stat().st_size,
        copied_sha256=file_hash(target), copied_size_bytes=target.stat().st_size, node_count=row_count,
        method="READ_ONLY_SQLITE_BACKUP_OF_IMMUTABLE_CONTENT_ADDRESSED_NODES",
        native_recomputation_required=True, old_scores_or_decisions_reused=False,
        reused_nodes_verified_by="EXISTING_EVIDENCE_GRAPH_WRITER_CONTENT_CHECK")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-bundle", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--fundamental-db", required=True, type=Path)
    parser.add_argument("--evaluation-cutoff", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-copy-receipt", type=Path)
    parser.add_argument("--seed-evidence-store", type=Path,
        help="Optional completed immutable CAS; copies nodes only, never cached model results")
    parser.add_argument("--expected-seed-sha256")
    args = parser.parse_args(argv)
    if (args.seed_evidence_store is None) != (args.expected_seed_sha256 is None):
        raise ValueError("COMPARISON_SEED_PATH_AND_SHA256_REQUIRED")
    cutoff = timestamp(args.evaluation_cutoff).isoformat()
    from domain.les_v7.policy_r17 import AVAILABLE_AT
    if timestamp(cutoff) < timestamp(AVAILABLE_AT):
        raise ValueError("R17_COMPARISON_BEFORE_POLICY_RELEASE")
    pinned_programs = program_bindings()
    if file_hash(args.report_bundle) != args.expected_bundle_sha256:
        raise ValueError("COMPARISON_SOURCE_BUNDLE_HASH_MISMATCH")
    copied = _copy_receipt(args.source_copy_receipt, args.report_bundle, args.expected_bundle_sha256)
    bundle = json.loads(args.report_bundle.read_text())
    symbols = [row["symbol"] for key in ("reports", "stale_reports") for row in bundle.get(key, [])]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("COMPARISON_SCOPE_EMPTY_OR_DUPLICATE")
    del bundle
    # An interrupted attempt is immutable too: use a fresh output directory.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for entry, path in zip(pinned_programs, PROGRAM_PATHS):
        write_bytes_once(args.output_dir / "programs" / entry["repository_path"], path.read_bytes())
    write_once(args.output_dir / "programs/manifest.json", dict(schema="les-comparison-programs/v1", files=pinned_programs))
    write_once(args.output_dir / "command.json", dict(schema="les-comparison-command/v1", argv=list(argv or sys.argv[1:]),
        report_bundle=str(args.report_bundle.resolve()), fundamental_db=str(args.fundamental_db.resolve()),
        expected_bundle_sha256=args.expected_bundle_sha256, evaluation_cutoff=cutoff,
        seed_evidence_store=str(args.seed_evidence_store.resolve()) if args.seed_evidence_store else None,
        expected_seed_sha256=args.expected_seed_sha256))
    seed = _seed_store(args.seed_evidence_store, args.expected_seed_sha256,
                       args.output_dir / "report_evidence.sqlite3")
    if seed is not None:
        write_once(args.output_dir / "evidence-seed.json", seed)
    guard = ResearchVersionGuard()
    objects = ResearchObjects(args.output_dir / "runtime-objects/objects")
    runtime_references = guard.retain(objects)
    write_once(args.output_dir / "runtime-objects/manifest.json", dict(schema="les-comparison-runtime/v1",
        versions=asdict(guard.versions), references=runtime_references,
        uncommitted_effective_source_included=True, source_scope="EXISTING_RESEARCH_VERSION_GUARD"))
    rows = []
    restorer = source_runtime.HashMatchedFilings(args.fundamental_db, symbols)
    try:
        with open_daily_work(max_bytes=64 * 1024 * 1024) as work, ReportEvidenceWriter(
                args.output_dir / "report_evidence.sqlite3") as writer:
            versions = asdict(work.check_version())
            if versions != asdict(guard.versions):
                raise ValueError("COMPARISON_RUNTIME_CAPTURE_VERSION_MISMATCH")
            for saved in iter_report_bundle(args.report_bundle):
                work.check_version()
                guard.check()
                check_program_bindings(pinned_programs)
                try:
                    validate_report(saved)
                    filings, restoration = restorer.restore(saved)
                    old, new, row = compare_report(saved, filings, evaluation_cutoff=cutoff)
                    row.update(restoration=restoration, r16_report=writer.retain(old), r17_report=writer.retain(new))
                    del old, new, filings
                except (KeyError, TypeError, ValueError, OSError) as error:
                    row = dict(symbol=saved["symbol"], state="SOURCE_OR_RECOMPUTATION_GAP",
                               error=str(error), original_report_sha256=digest(saved))
                rows.append(row)
                print(json.dumps(dict(event="comparison_stock_completed", symbol=saved["symbol"],
                    state=row["state"], completed=len(rows), total=len(symbols))), flush=True)
                del saved
                work.check_version()
                guard.check()
                check_program_bindings(pinned_programs)
            work.check_version()
        if [row["symbol"] for row in rows] != symbols:
            raise ValueError("COMPARISON_COMPLETED_SCOPE_MISMATCH")
        if file_hash(args.report_bundle) != args.expected_bundle_sha256:
            raise ValueError("COMPARISON_INPUT_CHANGED_DURING_JOB")
        check_program_bindings(pinned_programs)
        guard.check()
        store = _store_receipt(args.output_dir / "report_evidence.sqlite3")
        result = dict(schema=SCHEMA, source_bundle_sha256=args.expected_bundle_sha256,
            source_copy_receipt=copied,
            source_copy_receipt_sha256=file_hash(args.source_copy_receipt) if args.source_copy_receipt else None,
            evaluation_cutoff=cutoff, source_snapshot_cutoffs=sorted({row["source_snapshot_cutoff"] for row in rows
                if "source_snapshot_cutoff" in row}), universe_symbols=symbols, versions=versions,
            comparison_programs=pinned_programs, runtime_references=runtime_references, result_store=store,
            evidence_seed=seed, evidence_retention_stats=dict(writer.stats),
            historical_information_set_certified=False, historical_deployment_claim=False,
            scope="CURRENT_POLICY_COUNTERFACTUAL_FIXED_OLDER_RAW_SOURCE_SNAPSHOT",
            account_policy="EXPLICIT_SAVED_VALUES_REUSED_NOT_CURRENT_ACCOUNT_OR_LEDGER_CLAIM",
            pae_persistence="NOT_PRESENT_IN_PER_STOCK_NATIVE_REPORT_NOT_REBASED_OR_SIMULATED",
            order_authority=False, simulated_fill_count=0, full_financials_hash_match_required=True,
            raw_dataset_self_contained=False,
            summary=dict(total=len(rows), states=dict(Counter(row["state"] for row in rows)),
                r16_decisions=dict(Counter(row["r16"]["strategy"]["strategy_decision"] for row in rows if "r16" in row)),
                r17_decisions=dict(Counter(row["r17"]["strategy"]["strategy_decision"] for row in rows if "r17" in row)),
                calendar_source_presence=dict(Counter("PRESENT_NOT_AUTOMATICALLY_VERIFIED" if row["calendar_source_present"]
                    else "MISSING" for row in rows if row["state"] == "FULL_NATIVE_RECOMPUTED")),
                restore_counts=dict(restorer.counts)), records=rows)
        result["comparison_id"] = digest(result)
        write_once(args.output_dir / "comparison.json", result)
        write_bytes_once(args.output_dir / "comparison.md", markdown(result).encode())
        check_program_bindings(pinned_programs)
        guard.check()
        write_once(args.output_dir / "complete.json", dict(comparison_id=result["comparison_id"],
            comparison_sha256=file_hash(args.output_dir / "comparison.json"),
            markdown_sha256=file_hash(args.output_dir / "comparison.md"), result_store=store,
            runtime_manifest_sha256=file_hash(args.output_dir / "runtime-objects/manifest.json"),
            programs_manifest_sha256=file_hash(args.output_dir / "programs/manifest.json"),
            command_sha256=file_hash(args.output_dir / "command.json"),
            evidence_seed_sha256=file_hash(args.output_dir / "evidence-seed.json") if seed else None,
            status="COMPLETE" if all(row["state"] == "FULL_NATIVE_RECOMPUTED" for row in rows) else "PARTIAL_GAPS_RETAINED"))
        print(json.dumps(dict(comparison_id=result["comparison_id"], summary=result["summary"]), ensure_ascii=False))
        return 0
    finally:
        restorer.close()


if __name__ == "__main__":
    raise SystemExit(main())
