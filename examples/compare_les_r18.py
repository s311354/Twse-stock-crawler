"""Resumable full native R17/R18 comparison on one pinned source universe.

Reuse the existing source restorer, native plugin, receipt producer, daily
finalizer and validator. No refresh, network, mail, formal account, PAE state
or fill ledger is used. Conditional BUY_SMALL is not a simulated fill.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sys
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.replay_contracts import digest
from domain.les_v7.contracts import timestamp
from domain.les_v7.report import validate_report
from examples import compare_les_r17 as previous
from examples import compare_les_r14 as source_runtime
from examples.compare_les_r13 import file_hash
from infrastructure.daily_work import open_daily_work
from infrastructure.research_archive import write_bytes_once, write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.report_evidence import ReportEvidenceReader, ReportEvidenceWriter
from infrastructure.storage.valuation_source_verification import (
    valuation_history_verifier_scope, verify_report_valuation_sources,
)

SCHEMA = "les-r17-r18-fixed-source-native-comparison/v1"
PROGRAM_PATHS = (Path(__file__).resolve(), *previous.PROGRAM_PATHS)


def program_bindings(paths=PROGRAM_PATHS):
    return previous.program_bindings(paths)


def check_program_bindings(expected, paths=PROGRAM_PATHS):
    previous.check_program_bindings(expected, paths)


def _verify_program_archive(directory, expected):
    manifest = json.loads((directory / "programs/manifest.json").read_text())
    if manifest != dict(schema="les-comparison-programs/v1", files=expected):
        raise ValueError("COMPARISON_RESUME_PROGRAM_ARCHIVE_MISMATCH")
    for entry in expected:
        relative = Path(entry["repository_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("COMPARISON_RESUME_PROGRAM_ARCHIVE_MISMATCH")
        if file_hash(directory / "programs" / relative) != entry["sha256"]:
            raise ValueError("COMPARISON_RESUME_PROGRAM_ARCHIVE_MISMATCH")


def differences(before, after, path=""):
    yield from previous.differences(before, after, path)


def business(report):
    result = previous.business(report)
    result["r18_route_evaluation"] = report["diagnostics"].get("r18_route_evaluation")
    return result


def _cost_operands(report):
    reference = report["diagnostics"]["rule_baseline"]["policy"]["reference_policy"]
    path = reference["cost_path_policy"]
    # Policy publication clocks differ by version; arithmetic inputs may not.
    # Both complete policy objects, including these clocks, remain in the full
    # report differences. Do not compare merely absent top-level placeholders.
    return dict(reference_shares=reference["reference_shares"],
        **{key: path[key] for key in ("instrument_type", "market", "price_grid", "coordinate_policy", "slippage_bps")},
        cost_policy={key: value for key, value in path["cost_policy"].items() if key != "available_at"})


def compare_report(report, filings, *, evaluation_cutoff):
    from domain.les_v7.policy_r17 import BASELINE_VERSION as R17
    from domain.les_v7.policy_r18 import BASELINE_VERSION as R18, AVAILABLE_AT

    cutoff = timestamp(evaluation_cutoff).isoformat()
    account = source_runtime._account(report.get("pae_authorization", {}).get("account"), cutoff)
    validate_report(report)
    source = report["diagnostics"]["native_evidence"]["source_packet"]
    visibility = source_runtime.assert_snapshot_visibility(source)
    if timestamp(cutoff) < timestamp(AVAILABLE_AT):
        raise ValueError("R18_COMPARISON_BEFORE_POLICY_RELEASE")
    packet = source_runtime.reselect_packet(source, evaluation_cutoff=cutoff)
    old = source_runtime.recompute(report, filings, evaluation_cutoff=cutoff, policy_version=R17, packet=packet)
    new = source_runtime.recompute(report, filings, evaluation_cutoff=cutoff, policy_version=R18, packet=packet)
    saved_binding = report["diagnostics"]["evidence_binding"]
    bindings = [value["diagnostics"]["evidence_binding"] for value in (old, new)]
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
        if bindings[0][key] != bindings[1][key] or bindings[0][key] != saved_binding[key]:
            raise ValueError("COUNTERFACTUAL_RAW_SOURCE_BINDING_CHANGED:" + key)
    if bindings[0]["native_source_id"] != bindings[1]["native_source_id"]:
        raise ValueError("COUNTERFACTUAL_SELECTED_SOURCE_CHANGED")
    cost_inputs = [_cost_operands(value) for value in (old, new)]
    if cost_inputs[0] != cost_inputs[1]:
        raise ValueError("COUNTERFACTUAL_COST_POLICY_CHANGED")
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
        common_cost_inputs=cost_inputs[0], common_cost_inputs_sha256=digest(cost_inputs[0]),
        original_decision=report["envelope"]["strategy"]["strategy_decision"],
        r17=before, r18=after, differences=list(differences(before, after)),
        full_report_differences=list(differences(old, new)))


def markdown(result):
    summary = result["summary"]
    names = sorted(set(summary["r17_decisions"]) | set(summary["r18_decisions"]))
    lines = ["# R17／R18：固定來源的完整原生比較", "",
        f"股票範圍：{summary['total']} 檔；完整重算：{summary['states'].get('FULL_NATIVE_RECOMPUTED', 0)} 檔；"
        f"缺口：{summary['states'].get('SOURCE_OR_RECOMPUTATION_GAP', 0)} 檔。", "",
        f"共同評估截止：`{result['evaluation_cutoff']}`；原始來源截止另行保留。", "",
        "| 原生決策 | R17 | R18 |", "|---|---:|---:|"]
    lines.extend(f"| {key} | {summary['r17_decisions'].get(key, 0)} | {summary['r18_decisions'].get(key, 0)} |"
                 for key in names)
    lines.extend(["", "同一固定股票範圍、原始來源、完整財報向量、行情、成本、截止與保存帳戶，"
        "分別完整執行原生策略、receipt producer、每日 finalizer 與 validator；"
        "分數、Hard Gates、資格、交易計畫、部位上限及所有報告欄位差異保留。", "",
        "條件式 BUY_SMALL 是尚未成交的研究訊號；價格未符合不得成交。"
        "本次未模擬任何成交，不以之後價格或目前來源回填；過期／缺失證據仍 UNKNOWN。", "",
        "這是目前政策對固定來源的反事實研究，不是歷史當時部署、獲利或成交證明。"
        "未模擬 EREM、完整 PAE Persistence、帳戶累積或費用帳本；未抓取新資料、寄信或下單。", "",
        "每檔結果先提交不可變證據節點，再原子發布帶雜湊鏈的 row；重試驗證身分、"
        "完整已提交報告與鏈後續跑，不覆寫既有 row。缺口保留，不強制產生買訊。", "",
        "原件／MOPS snapshots／外部證明仍須保留；比較結果並非自足可攜原始資料集。", "",
        f"來源 bundle SHA256：`{result['source_bundle_sha256']}`", ""])
    return "\n".join(lines)


def _row_receipt(row, *, job_id, index, previous_sha256):
    payload = dict(schema="les-r18-comparison-row/v1", job_id=job_id, index=index,
                   previous_sha256=previous_sha256, row=row)
    return dict(payload=payload, row_sha256=digest(payload))


def _load_rows(directory, *, job_id, references):
    """Only an uninterrupted prefix is resumable; gaps are never skipped."""
    files = sorted((directory / "rows").glob("*.json"))
    rows, chain = [], job_id
    for index, path in enumerate(files):
        if path.name != f"{index:04d}.json" or index >= len(references):
            raise ValueError("COMPARISON_RESUME_ROW_SEQUENCE_INVALID")
        receipt = json.loads(path.read_text())
        payload = receipt["payload"]
        row, ref = payload["row"], references[index]
        if (payload.get("schema") != "les-r18-comparison-row/v1"
                or receipt.get("row_sha256") != digest(payload) or payload.get("job_id") != job_id
                or payload.get("index") != index or payload.get("previous_sha256") != chain
                or row.get("symbol") != ref["symbol"]
                or row.get("original_report_sha256") != ref["report_sha256"]
                or row.get("state") not in {"FULL_NATIVE_RECOMPUTED", "SOURCE_OR_RECOMPUTATION_GAP"}):
            raise ValueError("COMPARISON_RESUME_ROW_IDENTITY_MISMATCH")
        rows.append(row)
        chain = receipt["row_sha256"]
    return rows, chain


def _verify_retained_row(row, reader, verifier, cutoff):
    if row["state"] == "SOURCE_OR_RECOMPUTATION_GAP":
        if any(key in row for key in ("r17", "r18", "r17_report", "r18_report")):
            raise ValueError("COMPARISON_RESUME_GAP_WITH_RESULTS")
        return
    old, new = (reader.resolve(row[key]) for key in ("r17_report", "r18_report"))
    for report, key in ((old, "r17"), (new, "r18")):
        verify_report_valuation_sources(report, verifier=verifier)
        validate_report(report)
        if (report["symbol"] != row["symbol"] or business(report) != row[key]
                or timestamp(report["envelope"]["event"]["cutoff_at"]) != timestamp(cutoff)
                or report["pae_authorization"]["live_order_authorized"] is not False):
            raise ValueError("COMPARISON_RESUME_NATIVE_RESULT_MISMATCH")
    if (row["differences"] != list(differences(business(old), business(new)))
            or row["full_report_differences"] != list(differences(old, new))
            or old["pae_authorization"]["account"] != new["pae_authorization"]["account"]
            or old["pae_authorization"]["account_sha256"] != row["evaluation_account_sha256"]
            or old["pae_authorization"]["account_sha256"] != new["pae_authorization"]["account_sha256"]):
        raise ValueError("COMPARISON_RESUME_NATIVE_RESULT_MISMATCH")


def _assert_sources(args):
    for path, expected, name in (
        (args.report_bundle, args.expected_bundle_sha256, "BUNDLE"),
        (args.report_bundle.parent / "report_evidence.sqlite3", args.expected_report_store_sha256, "STORE"),
        (args.fundamental_db, args.expected_fundamental_sha256, "FUNDAMENTAL"),
    ):
        if any(Path(str(path) + suffix).exists() and Path(str(path) + suffix).stat().st_size
               for suffix in ("-wal", "-journal")):
            raise ValueError("COMPARISON_SOURCE_NOT_CLOSED:" + name)
        if file_hash(path) != expected:
            raise ValueError("COMPARISON_SOURCE_" + name + "_HASH_MISMATCH")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-bundle", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-report-store-sha256", required=True)
    parser.add_argument("--fundamental-db", required=True, type=Path)
    parser.add_argument("--expected-fundamental-sha256", required=True)
    parser.add_argument("--evaluation-cutoff", required=True)
    parser.add_argument("--expected-count", type=int, default=436)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-copy-receipt", type=Path)
    parser.add_argument("--seed-evidence-store", type=Path)
    parser.add_argument("--expected-seed-sha256")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    from domain.les_v7.policy_r18 import AVAILABLE_AT
    cutoff = timestamp(args.evaluation_cutoff).isoformat()
    if timestamp(cutoff) < timestamp(AVAILABLE_AT):
        raise ValueError("R18_COMPARISON_BEFORE_POLICY_RELEASE")
    if (args.seed_evidence_store is None) != (args.expected_seed_sha256 is None):
        raise ValueError("COMPARISON_SEED_PATH_AND_SHA256_REQUIRED")
    if args.expected_count <= 0:
        raise ValueError("COMPARISON_EXPECTED_SCOPE_INVALID")
    _assert_sources(args)
    copied = previous._copy_receipt(args.source_copy_receipt, args.report_bundle, args.expected_bundle_sha256)
    if copied is not None and (
            Path(copied["local_store"]).resolve() != (args.report_bundle.parent / "report_evidence.sqlite3").resolve()
            or copied["local_store_sha256"] != args.expected_report_store_sha256):
        raise ValueError("COMPARISON_SOURCE_COPY_RECEIPT_STORE_MISMATCH")
    bundle = json.loads(args.report_bundle.read_text())
    references = [row.get("_report_reference", row) for key in ("reports", "stale_reports")
                  for row in bundle.get(key, [])]
    symbols = [row["symbol"] for row in references]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("COMPARISON_SCOPE_EMPTY_OR_DUPLICATE")
    if len(symbols) != args.expected_count:
        raise ValueError("COMPARISON_EXPECTED_SCOPE_MISMATCH")
    if "count" in bundle and bundle["count"] != len(symbols):
        raise ValueError("COMPARISON_SOURCE_BUNDLE_COUNT_MISMATCH")
    if any(ref.get("schema") != "daily-report-reference/v1"
           or ref.get("store") != "report_evidence.sqlite3" for ref in references):
        raise ValueError("COMPARISON_SOURCE_REFERENCES_REQUIRED")
    del bundle
    pinned_programs, guard = program_bindings(), ResearchVersionGuard()
    versions = asdict(guard.versions)
    identity = dict(schema=SCHEMA, evaluation_cutoff=cutoff, universe_symbols=symbols,
        source_bundle=str(args.report_bundle.resolve()), source_bundle_sha256=args.expected_bundle_sha256,
        source_report_store_sha256=args.expected_report_store_sha256,
        fundamental_db=str(args.fundamental_db.resolve()), fundamental_db_sha256=args.expected_fundamental_sha256,
        source_copy_receipt_sha256=file_hash(args.source_copy_receipt) if copied else None,
        comparison_programs=pinned_programs, versions=versions,
        seed_evidence_store=str(args.seed_evidence_store.resolve()) if args.seed_evidence_store else None,
        expected_seed_sha256=args.expected_seed_sha256)
    job_id = digest(identity)
    if args.resume:
        expected_job = json.loads((args.output_dir / "job.json").read_text())
        if expected_job != dict(job_id=job_id, identity=identity):
            raise ValueError("COMPARISON_RESUME_JOB_IDENTITY_MISMATCH")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    # Job-local lock never touches or truncates the daily production lock.
    with ExitStack() as stack:
        lock = stack.enter_context((args.output_dir / ".comparison.lock").open("a+b"))
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        objects = ResearchObjects(args.output_dir / "runtime-objects/objects")
        if args.resume:
            _verify_program_archive(args.output_dir, pinned_programs)
            runtime = json.loads((args.output_dir / "runtime-objects/manifest.json").read_text())
            if runtime.get("versions") != versions:
                raise ValueError("COMPARISON_RESUME_RUNTIME_MISMATCH")
            runtime_references = runtime["references"]
            for reference in runtime_references.values():
                objects.get(reference)
            if guard.retain(objects) != runtime_references:
                raise ValueError("COMPARISON_RESUME_RUNTIME_MISMATCH")
            seed_path = args.output_dir / "evidence-seed.json"
            if seed_path.exists() != (args.seed_evidence_store is not None):
                raise ValueError("COMPARISON_RESUME_SEED_RECEIPT_MISMATCH")
            seed = json.loads(seed_path.read_text()) if seed_path.exists() else None
            if seed is not None and (seed.get("source_sha256") != args.expected_seed_sha256
                    or seed.get("source_store") != str(args.seed_evidence_store.resolve())
                    or seed.get("native_recomputation_required") is not True
                    or seed.get("old_scores_or_decisions_reused") is not False):
                raise ValueError("COMPARISON_RESUME_SEED_RECEIPT_MISMATCH")
        else:
            for entry, path in zip(pinned_programs, PROGRAM_PATHS):
                write_bytes_once(args.output_dir / "programs" / entry["repository_path"], path.read_bytes())
            write_once(args.output_dir / "programs/manifest.json", dict(schema="les-comparison-programs/v1", files=pinned_programs))
            runtime_references = guard.retain(objects)
            write_once(args.output_dir / "runtime-objects/manifest.json", dict(schema="les-comparison-runtime/v1",
                versions=versions, references=runtime_references, uncommitted_effective_source_included=True))
            seed = previous._seed_store(args.seed_evidence_store, args.expected_seed_sha256,
                                       args.output_dir / "report_evidence.sqlite3")
            if seed is not None:
                write_once(args.output_dir / "evidence-seed.json", seed)
            write_once(args.output_dir / "job.json", dict(job_id=job_id, identity=identity))
        rows, chain = _load_rows(args.output_dir, job_id=job_id, references=references)
        stats = Counter()
        restorer = source_runtime.HashMatchedFilings(args.fundamental_db, symbols)
        stack.callback(restorer.close)
        work = stack.enter_context(open_daily_work(max_bytes=64 * 1024 * 1024))
        if asdict(work.check_version()) != versions:
            raise ValueError("COMPARISON_RUNTIME_CAPTURE_VERSION_MISMATCH")
        reader = stack.enter_context(ReportEvidenceReader(args.report_bundle.parent))
        verifier = stack.enter_context(valuation_history_verifier_scope())
        with ReportEvidenceReader(args.output_dir) as retained:
            for row in rows:
                _verify_retained_row(row, retained, verifier, cutoff)
        already_complete = args.output_dir / "complete.json"
        if already_complete.exists():
            receipt = json.loads(already_complete.read_text())
            if (receipt.get("job_id") != job_id or receipt.get("row_chain_sha256") != chain
                    or len(rows) != len(symbols)
                    or receipt.get("comparison_sha256") != file_hash(args.output_dir / "comparison.json")
                    or receipt.get("markdown_sha256") != file_hash(args.output_dir / "comparison.md")
                    or receipt.get("runtime_manifest_sha256") != file_hash(args.output_dir / "runtime-objects/manifest.json")
                    or receipt.get("programs_manifest_sha256") != file_hash(args.output_dir / "programs/manifest.json")
                    or receipt.get("job_manifest_sha256") != file_hash(args.output_dir / "job.json")
                    or receipt.get("result_store") != previous._store_receipt(args.output_dir / "report_evidence.sqlite3")):
                raise ValueError("COMPARISON_RESUME_COMPLETION_MISMATCH")
            check_program_bindings(pinned_programs)
            guard.check()
            _assert_sources(args)
            print(json.dumps(dict(event="comparison_already_complete", job_id=job_id, status=receipt["status"])))
            return 0
        for index in range(len(rows), len(references)):
            work.check_version()
            guard.check()
            check_program_bindings(pinned_programs)
            tick = perf_counter()
            saved = reader.resolve(references[index])
            verify_report_valuation_sources(saved, verifier=verifier)
            try:
                validate_report(saved)
                filings, restoration = restorer.restore(saved)
                restored_at = perf_counter()
                old, new, row = compare_report(saved, filings, evaluation_cutoff=cutoff)
                computed_at = perf_counter()
                # Commit CAS before publishing the immutable row receipt.
                with ReportEvidenceWriter(args.output_dir / "report_evidence.sqlite3") as writer:
                    row.update(restoration=restoration, r17_report=writer.retain(old), r18_report=writer.retain(new))
                stats.update(writer.stats)
                row["timing_seconds"] = dict(restore=restored_at - tick,
                    native_comparison=computed_at - restored_at, retention=perf_counter() - computed_at)
                del old, new, filings
            except (KeyError, TypeError, ValueError, OSError) as error:
                row = dict(symbol=saved["symbol"], state="SOURCE_OR_RECOMPUTATION_GAP",
                           error=str(error), original_report_sha256=digest(saved))
            if row["original_report_sha256"] != references[index]["report_sha256"]:
                raise ValueError("COMPARISON_SAVED_REPORT_IDENTITY_MISMATCH")
            work.check_version()
            guard.check()
            check_program_bindings(pinned_programs)
            receipt = _row_receipt(row, job_id=job_id, index=index, previous_sha256=chain)
            write_once(args.output_dir / "rows" / f"{index:04d}.json", receipt)
            chain = receipt["row_sha256"]
            rows.append(row)
            print(json.dumps(dict(event="comparison_stock_completed", symbol=saved["symbol"],
                state=row["state"], completed=len(rows), total=len(symbols))), flush=True)
            del saved
        if [row["symbol"] for row in rows] != symbols:
            raise ValueError("COMPARISON_COMPLETED_SCOPE_MISMATCH")
        # A gap-only run still has an explicit, verified empty result store.
        if not (args.output_dir / "report_evidence.sqlite3").exists():
            with ReportEvidenceWriter(args.output_dir / "report_evidence.sqlite3"):
                pass
        _assert_sources(args)
        check_program_bindings(pinned_programs)
        guard.check()
        store = previous._store_receipt(args.output_dir / "report_evidence.sqlite3")
        result = dict(schema=SCHEMA, job_id=job_id, source_bundle_sha256=args.expected_bundle_sha256,
            source_copy_receipt=copied, source_report_store_sha256=args.expected_report_store_sha256,
            fundamental_db_sha256=args.expected_fundamental_sha256, evaluation_cutoff=cutoff,
            source_snapshot_cutoffs=sorted({row["source_snapshot_cutoff"] for row in rows if "source_snapshot_cutoff" in row}),
            universe_symbols=symbols, versions=versions, comparison_programs=pinned_programs,
            runtime_references=runtime_references, result_store=store, evidence_seed=seed,
            evidence_retention_stats_current_attempt=dict(stats), row_chain_sha256=chain,
            current_attempt_seconds=perf_counter() - started,
            historical_information_set_certified=False, historical_deployment_claim=False,
            scope="CURRENT_POLICY_COUNTERFACTUAL_FIXED_RAW_SOURCE_SNAPSHOT",
            account_policy="EXPLICIT_SAVED_VALUES_REUSED_NOT_CURRENT_ACCOUNT_OR_LEDGER_CLAIM",
            pae_persistence="NOT_PRESENT_IN_PER_STOCK_NATIVE_REPORT_NOT_REBASED_OR_SIMULATED",
            order_authority=False, simulated_fill_count=0, full_financials_hash_match_required=True,
            raw_dataset_self_contained=False,
            summary=dict(total=len(rows), states=dict(Counter(row["state"] for row in rows)),
                r17_decisions=dict(Counter(row["r17"]["strategy"]["strategy_decision"] for row in rows if "r17" in row)),
                r18_decisions=dict(Counter(row["r18"]["strategy"]["strategy_decision"] for row in rows if "r18" in row)),
                calendar_source_presence=dict(Counter("PRESENT_NOT_AUTOMATICALLY_VERIFIED" if row["calendar_source_present"]
                    else "MISSING" for row in rows if row["state"] == "FULL_NATIVE_RECOMPUTED")),
                restore_counts_current_attempt=dict(restorer.counts)), records=rows)
        result["comparison_id"] = digest(result)
        comparison_path = args.output_dir / "comparison.json"
        if comparison_path.exists():
            # Resume after result publication but before the final receipt.
            # Attempt timing/counters are execution metadata, not a reason to
            # overwrite already published business results.
            published = json.loads(comparison_path.read_text())
            if (published.get("job_id") != job_id or published.get("records") != rows
                    or published.get("row_chain_sha256") != chain
                    or published.get("result_store") != store
                    or published.get("comparison_id") != digest({key: value for key, value in published.items()
                                                                  if key != "comparison_id"})):
                raise ValueError("COMPARISON_RESUME_PUBLISHED_RESULT_MISMATCH")
            result = published
        write_once(comparison_path, result)
        write_bytes_once(args.output_dir / "comparison.md", markdown(result).encode())
        check_program_bindings(pinned_programs)
        guard.check()
        write_once(args.output_dir / "complete.json", dict(job_id=job_id, comparison_id=result["comparison_id"],
            comparison_sha256=file_hash(args.output_dir / "comparison.json"),
            markdown_sha256=file_hash(args.output_dir / "comparison.md"), result_store=store,
            row_chain_sha256=chain, runtime_manifest_sha256=file_hash(args.output_dir / "runtime-objects/manifest.json"),
            programs_manifest_sha256=file_hash(args.output_dir / "programs/manifest.json"),
            job_manifest_sha256=file_hash(args.output_dir / "job.json"),
            status="COMPLETE" if all(row["state"] == "FULL_NATIVE_RECOMPUTED" for row in rows) else "PARTIAL_GAPS_RETAINED"))
        print(json.dumps(dict(comparison_id=result["comparison_id"], summary=result["summary"]), ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
