"""Streaming old/native versus FR2 EPS-price admission on one frozen universe.

No refresh, network, mail, execution, formal account or PAE state is opened.
The original predictor/Quality/causal plan remains fixed. Recovered companions
must name the exact original hashes already present in each saved report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from application.replay_contracts import digest
from domain.les_v7.contracts import identity, timestamp
from examples.compare_les_r13 import file_hash
from infrastructure.research_archive import write_bytes_once, write_once

SCHEMA = "les-v7-fr2-eps-fixed-source-comparison/v1"
INPUT_SCHEMA = "les-v7-fr2-eps-inputs/v1"
PRESERVED_DIAGNOSTICS = ("features", "feature_audit", "factor_ledger", "quality", "quality_policy",
    "gate_bindings", "localization", "reference_plan", "strategy_policy",
    "evidence_binding", "native_evidence", "starter", "confirmed", "starter_gate", "confirmed_gate")


def prepare_local_sources(bundle, directory):
    """Use the existing SQLite online-backup boundary, not a live-file copy."""
    from examples.compare_les_r14 import local_report_copy
    bundle, directory = Path(bundle).resolve(), Path(directory).resolve()
    receipt_path = directory / "source_copy_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        local = directory / bundle.name
        if (receipt["original_bundle"] != str(bundle)
                or file_hash(bundle) != receipt["bundle_sha256"]
                or file_hash(local) != receipt["bundle_sha256"]
                or file_hash(directory / "report_evidence.sqlite3") != receipt["local_store_sha256"]):
            raise ValueError("FR2_EPS_LOCAL_SOURCE_COPY_MISMATCH")
        return local
    local, _ = local_report_copy(bundle, directory)
    return local


class BoundAnnualOriginals:
    """One read-only SQLite snapshot; parse only preselected original hashes."""

    def __init__(self, database):
        from infrastructure.fundamentals.mops_queries import MopsQueryRepository
        from infrastructure.fundamentals.mops_unit_of_work import SqliteFundamentalUnitOfWork
        self.database = Path(database).resolve()
        self.connection = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)
        self.connection.execute("PRAGMA query_only=1")
        self.connection.execute("BEGIN")
        self.uow = SqliteFundamentalUnitOfWork(self.connection)
        self.queries = MopsQueryRepository(lambda: self.connection, lambda: None,
                                          raw_root=self.database.parent / "raw")
        self.counts = Counter()

    def close(self):
        self.connection.rollback()
        self.connection.close()

    def valuation_inputs(self, original):
        from domain.fundamental_derivation import FiledFinancials
        from domain.les_v7.financial_sources import ixbrl_record_r2
        from domain.les_v7.valuation_earnings import valuation_earnings
        from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
        from infrastructure.fundamentals.mops_queries import _raw_normalized_facts

        source = original["diagnostics"]["native_evidence"]["source_packet"]
        symbol, market, cutoff = source["symbol"], source["market"], source["cutoff"]
        selected = [row for row in source.get("valuation_filings", ())
                    if timestamp(row["available_at"]) <= timestamp(cutoff)]
        hashes = {row["content_sha256"] for row in selected}
        filings = self.queries.load_filings_many([symbol], as_of=cutoff, uow=self.uow).get(symbol, [])
        matched = {}
        for filing in filings:
            if filing.content_sha256 not in hashes:
                continue
            if filing.quarter != 4 or timestamp(filing.available_at) > timestamp(cutoff):
                continue
            path = (self.database.parent / filing.raw_path).resolve()
            if not path.is_relative_to((self.database.parent / "raw").resolve()):
                raise ValueError("FR2_EPS_RAW_PATH_OUTSIDE_ROOT")
            stat = path.stat()
            parsed = _raw_normalized_facts(path, filing.content_sha256,
                (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size), V6_FILING_SCHEMA_VERSION)
            if parsed is None:
                raise ValueError("FR2_EPS_ORIGINAL_PARSE_OR_HASH_UNVERIFIED")
            facts, refs = parsed[:2]
            raw = FiledFinancials(fiscal_year=filing.fiscal_year, quarter=filing.quarter,
                available_at=filing.available_at, revision_number=filing.revision_number,
                facts=facts, correction_description=filing.correction_description,
                availability_source=filing.availability_source, filing_id=filing.filing_id,
                content_sha256=filing.content_sha256, source_url=filing.source_url,
                period_end=filing.period_end, statement_scope=filing.statement_scope,
                audit_status=filing.audit_status, taxonomy_schema_refs=refs)
            mapped = ixbrl_record_r2(symbol, raw, market=market, original_verified=True)
            mapped["revision_number"] = filing.revision_number
            eps = valuation_earnings(raw, symbol=symbol, market=market,
                                     original_verified=True, as_of=cutoff)
            if filing.content_sha256 in matched:
                if identity(matched[filing.content_sha256]) != identity((mapped, eps)):
                    raise ValueError("FR2_EPS_CONFLICTING_ORIGINAL_METADATA")
            matched[filing.content_sha256] = (mapped, eps)
            self.counts["verified_raw_originals"] += 1
        enriched = []
        for saved in selected:
            pair = matched.get(saved["content_sha256"])
            if pair is None:
                # Keep the missing latest version; never silently fall back.
                enriched.append(deepcopy(saved))
                self.counts["selected_originals_missing"] += 1
                continue
            mapped, earnings = pair
            for key in ("symbol", "market", "source", "period_end", "scope", "currency", "unit",
                        "revision_id", "content_sha256", "revenue", "operating_income"):
                if saved.get(key) != mapped.get(key):
                    raise ValueError("FR2_EPS_RAW_ORIGINAL_SAVED_OPERAND_MISMATCH:" + key)
            if timestamp(saved["available_at"]) != timestamp(mapped["available_at"]):
                raise ValueError("FR2_EPS_ORIGINAL_AVAILABILITY_MISMATCH")
            # Never overwrite saved current values/proofs, Quality or market data.
            row = deepcopy(saved)
            row.update(source_url=mapped["source_url"], revision_number=mapped["revision_number"],
                       earnings_source=earnings)
            for group in ("comparative_ytd_fields", "comparative_ytd_field_evidence"):
                row[group] = {field: mapped.get(group, {}).get(field)
                              for field in ("revenue", "operating_income")}
            enriched.append(row)
        return dict(schema=INPUT_SCHEMA, original_report_id=identity(original), symbol=symbol,
                    market=market, cutoff=cutoff, valuation_filings=enriched)


def differences(before, after, path=""):
    """Lossless JSON-pointer replacements, including absent versus null."""
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            location = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                yield dict(path=location, before_present=key in before, after_present=key in after,
                           before=before.get(key), after=after.get(key))
            else:
                yield from differences(before[key], after[key], location)
    elif type(before) is not type(after) or before != after:
        yield dict(path=path, before_present=True, after_present=True, before=before, after=after)


def restore_delta(original, changes):
    result = deepcopy(original)
    for change in changes:
        tokens = [text.replace("~1", "/").replace("~0", "~") for text in change["path"].split("/")[1:]]
        if not tokens:
            if not change["before_present"] or result != change["before"]:
                raise ValueError("FR2_EPS_DELTA_SOURCE_MISMATCH")
            result = deepcopy(change["after"])
            continue
        node = result
        for token in tokens[:-1]: node = node[token]
        key = tokens[-1]
        if ((key in node) != change["before_present"]
                or key in node and node[key] != change["before"]):
            raise ValueError("FR2_EPS_DELTA_SOURCE_MISMATCH")
        if change["after_present"]: node[key] = deepcopy(change["after"])
        else: node.pop(key)
    return result


def compare_report(saved, bundle, valuation_inputs, *, apply=None, validate=None):
    """Compare complete report identities and every retained original operand."""
    from domain.les_v7.report import validate_report
    from domain.les_v7.fr2_policy import apply_fr2_eps, restore_original
    validate = validate or validate_report
    apply = apply or apply_fr2_eps
    validate(saved)
    original = deepcopy(saved)
    prior_authorization = original.pop("pae_authorization", None)
    if prior_authorization is not None and prior_authorization.get("live_order_authorized") is not False:
        raise ValueError("FR2_EPS_COMPARISON_LIVE_AUTHORIZATION_FORBIDDEN")
    # Admission is deliberately unallocated. Preserve the exact prior account
    # snapshot separately; do not reuse its stale signal authorization.
    valuation_inputs = dict(valuation_inputs, original_report_id=identity(original))
    result = apply(original, bundle, valuation_inputs)
    validate(result)
    restored = restore_original(result)
    if restored != original:
        raise ValueError("FR2_EPS_COMPARISON_ORIGINAL_RESTORATION_MISMATCH")
    for key in PRESERVED_DIAGNOSTICS:
        if result["diagnostics"].get(key) != original["diagnostics"].get(key):
            raise ValueError("FR2_EPS_COMPARISON_IMMUTABLE_OPERAND_CHANGED:" + key)
    for key in ("model", "quality", "localization"):
        if result["envelope"].get(key) != original["envelope"].get(key):
            raise ValueError("FR2_EPS_COMPARISON_FROZEN_MODEL_CHANGED:" + key)
    for gate in original["diagnostics"]["gate_receipts"]:
        if gate["gate_id"] != "VALUATION" and gate not in result["diagnostics"]["gate_receipts"]:
            raise ValueError("FR2_EPS_COMPARISON_NON_VALUATION_GATE_CHANGED")
    overlay = result["diagnostics"]["fr2"]
    row = dict(symbol=original["symbol"], signal_date=original["signal_date"],
        cutoff=original["envelope"]["event"]["cutoff_at"], state="FULL_POLICY_RECOMPUTED",
        saved_report_sha256=identity(saved), unallocated_original_sha256=identity(original),
        result_sha256=identity(result), valuation_inputs_sha256=identity(valuation_inputs),
        original_source_id=original["diagnostics"]["native_evidence"]["source_id"],
        old_decision=original["envelope"]["strategy"]["strategy_decision"],
        new_decision=result["envelope"]["strategy"]["strategy_decision"],
        old_score=original["envelope"]["score"], new_score=result["envelope"]["score"],
        old_gates=original["diagnostics"]["gate_receipts"], new_gates=result["diagnostics"]["gate_receipts"],
        admission=overlay["admission"],
        frozen_operands_equal=True, original_report_restored_exactly=True,
        original_authorization=prior_authorization, new_account_authorization=None,
        original_account_sha256=(prior_authorization or {}).get("account_sha256"),
        # Lossless delta plus verified original reconstructs the exact output;
        # do not duplicate each large unchanged source graph in another store.
        full_report_differences=list(differences(original, result)))
    if restore_delta(original, row["full_report_differences"]) != result:
        raise ValueError("FR2_EPS_LOSSLESS_DELTA_RECONSTRUCTION_MISMATCH")
    return result, row, valuation_inputs


def summarize(rows):
    complete = [r for r in rows if r["state"] == "FULL_POLICY_RECOMPUTED"]
    return dict(total=len(rows), states=dict(Counter(r["state"] for r in rows)),
        old_decisions=dict(Counter(r["old_decision"] for r in complete)),
        new_decisions=dict(Counter(r["new_decision"] for r in complete)),
        valuation_states=dict(Counter(r["admission"]["valuation"]["native_state"] for r in complete)),
        warning_eligible=sum(r["admission"]["valuation"]["warning_eligible"] for r in complete),
        valuation_reasons=dict(Counter(reason for r in complete for reason in
            r["admission"]["valuation"]["reason_codes"])),
        hard_vetoes=dict(Counter(reason for r in complete for reason in r["admission"]["hard_vetoes"])),
        signals=[dict(symbol=r["symbol"], decision=r["new_decision"], score=r["new_score"].get("final_score"))
                 for r in complete if r["new_decision"] in {"BUY", "BUY_SMALL"}],
        validated_count=len(complete),
        full_frozen_operand_consistency=len(complete) == len(rows) and bool(rows)
            and all(r["frozen_operands_equal"] for r in complete),
        account_execution_or_profitability_assessed=False)


def verify_retained_row(row, ref, directory, reader):
    """A resumed prefix is revalidated, not merely trusted because JSON exists."""
    if row["saved_report_sha256"] != ref["report_sha256"] or row["symbol"] != ref["symbol"]:
        raise ValueError("FR2_EPS_RESUMED_SOURCE_BINDING_MISMATCH")
    if row["state"] == "SOURCE_OR_RECOMPUTATION_GAP":
        if any(key in row for key in ("new_score", "admission", "full_report_differences")):
            raise ValueError("FR2_EPS_RESUMED_GAP_WITH_RESULTS")
        return
    if row["state"] != "FULL_POLICY_RECOMPUTED":
        raise ValueError("FR2_EPS_RESUMED_STATE_INVALID")
    from domain.les_v7.report import validate_report
    original = reader.resolve(ref)
    authorization = original.pop("pae_authorization", None)
    if identity(original) != row["unallocated_original_sha256"] or authorization != row["original_authorization"]:
        raise ValueError("FR2_EPS_RESUMED_ORIGINAL_MISMATCH")
    inputs = json.loads((directory / "valuation-inputs" / (row["valuation_inputs_sha256"] + ".json")).read_text())
    result = restore_delta(original, row["full_report_differences"])
    validate_report(result)
    if (identity(inputs) != row["valuation_inputs_sha256"] or identity(result) != row["result_sha256"]
            or result["diagnostics"]["fr2"]["valuation_inputs"] != inputs
            or result["diagnostics"]["fr2"]["admission"] != row["admission"]
            or result["envelope"]["strategy"]["strategy_decision"] != row["new_decision"]
            or result["envelope"]["score"] != row["new_score"]
            or original["envelope"]["score"] != row["old_score"]):
        raise ValueError("FR2_EPS_RESUMED_RESULT_MISMATCH")


def markdown(result):
    s = result["summary"]
    lines = ["# 原 bundle／FR2 核實年度 EPS 價格估值：固定來源比較", "",
        f"完整範圍 {s['total']} 檔；狀態：`{json.dumps(s['states'], ensure_ascii=False)}`。", "",
        "| 決策 | 原政策 | 新研究政策 |", "|---|---:|---:|"]
    for key in sorted(s["old_decisions"].keys() | s["new_decisions"].keys()):
        lines.append(f"| {key} | {s['old_decisions'].get(key, 0)} | {s['new_decisions'].get(key, 0)} |")
    lines += ["", f"核實 EPS 估值狀態：`{json.dumps(s['valuation_states'], ensure_ascii=False)}`。", "",
        "原始 frozen bundle、五頭預測、校準、Quality、因子、型態、觸發及交易計畫保持不變。"
        "新研究準入採 FR2 BUY 75／BUY_SMALL 65；核實年度 EPS 僅計算價格報酬，"
        "不推估缺失股數、不把缺股利補零、不宣稱包含股利的總報酬。", "",
        "逐檔驗證原報告來源圖與雜湊，只從已選中原件相同 content hash／版本／截止重建 EPS 及比較欄位；"
        "沒有使用新報價或截止後新原件。完整來源財報、行情、審計、限制、Quality 與模型缺口仍保留。", "",
        "這是對先前已觀察資料的政策比較，不是樣本外預測／成本後增益證明。"
        "原研究帳戶輸入與授權另行留存，不沿用舊授權、不執行新配置、成交、PAE Persistence 或帳本；"
        "未寄信、抓取遠端資料或操作正式帳戶。", "",
        "每檔完整差異為 lossless delta，配合已核實原報告可重建新報告；"
        "原件與來源 store 必須保留，本報告不是可獨立攜出的完整原始資料集。", ""]
    return "\n".join(lines)


def business_identity(result):
    # Timings/counters and filesystem locations describe execution, not policy
    # outcomes. A completion retry keeps their original first-publication copy.
    return identity({key: result[key] for key in ("schema", "comparison_id", "inputs",
        "row_chain_sha256", "rows", "summary", "comparison_scope")})


def run(args):
    from infrastructure.storage.report_evidence import ReportEvidenceReader
    from infrastructure.storage.valuation_source_verification import (
        valuation_history_verifier_scope, verify_report_valuation_sources)
    bundle_path, model_path, output = Path(args.report_bundle).resolve(), Path(args.model_bundle).resolve(), Path(args.output).resolve()
    bundle_bytes_sha, model_bytes_sha = file_hash(bundle_path), file_hash(model_path)
    bundle = json.loads(bundle_path.read_text())
    refs = bundle.get("reports", [])
    if (len(refs) != args.expected_count or len({r["symbol"] for r in refs}) != len(refs)
            or any(r.get("schema") != "daily-report-reference/v1" for r in refs)):
        raise ValueError("FR2_EPS_COMPLETE_FIXED_UNIVERSE_REQUIRED")
    if args.expected_bundle_sha256 and bundle_bytes_sha != args.expected_bundle_sha256:
        raise ValueError("FR2_EPS_SOURCE_BUNDLE_HASH_MISMATCH")
    model = json.loads(model_path.read_text())
    # One local online SQLite backup amortizes random external-disk reads.
    local = prepare_local_sources(bundle_path, args.workspace)
    if args.prepare_only:
        print(json.dumps(dict(local_bundle=str(local), source_bundle_sha256=bundle_bytes_sha)))
        return
    from domain.les_v7.fr2_policy import _apply_validated
    from domain.les_v7.valuation_price import policy
    from domain.les_v7.fr2_policy import component
    from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
    guard = ResearchVersionGuard()
    versions = asdict(guard.versions)
    program_sha256 = file_hash(Path(__file__))
    copy_receipt = json.loads((local.parent / "source_copy_receipt.json").read_text())
    settings = dict(schema=SCHEMA, source_bundle_sha256=bundle_bytes_sha,
        predictor_file_sha256=model_bytes_sha, fixed_symbols=[r["symbol"] for r in refs],
        entry=component(), valuation=policy(), expected_count=args.expected_count,
        no_live_execution=True, account_state_changed=False, versions=versions,
        environment_sha256=identity(guard.environment), comparison_program_sha256=program_sha256,
        source_report_store_sha256=copy_receipt["local_store_sha256"])
    job_id = identity(settings)
    directory = output / job_id
    write_once(directory / "inputs.json", settings)
    # Job-local only; opening a+b never truncates production lock metadata.
    job_lock = (directory / ".comparison.lock").open("a+b")
    fcntl.flock(job_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_bytes_once(directory / "program.py", Path(__file__).read_bytes())
    write_bytes_once(directory / "frozen_bundle.json", model_path.read_bytes())
    objects = ResearchObjects(output / "runtime-objects/objects")
    references = guard.retain(objects)
    write_once(directory / "runtime.json", dict(versions=versions, references=references,
        uncommitted_effective_source_included=True))
    complete_path = directory / "completion.json"
    if complete_path.exists():
        completion = json.loads(complete_path.read_text())
        result = json.loads((directory / "comparison.json").read_text())
        if (completion.get("comparison_id") != job_id or completion.get("result_sha256") != identity(result)
                or completion.get("business_sha256") != result.get("business_sha256")
                or business_identity(result) != result["business_sha256"]):
            raise ValueError("FR2_EPS_COMPARISON_COMPLETION_MISMATCH")
        guard.check()
        print(json.dumps(dict(comparison=str(directory / "comparison.json"), summary=result["summary"],
                              already_complete=True), ensure_ascii=False))
        job_lock.close()
        return
    rows, chain, started = [], job_id, perf_counter()
    annual = BoundAnnualOriginals(args.fundamental_db)
    try:
        from application.daily_work import DailyWorkContext
        from infrastructure.storage.research_cache import SQLiteRawParseCache
        with ExitStack() as stack:
            parse_cache = stack.enter_context(SQLiteRawParseCache(local.parent / "raw-parse-cache.sqlite3"))
            work = DailyWorkContext(identity(versions), max_bytes=64 * 1024 * 1024,
                parse_memo=parse_cache, check_version=guard.check)
            stack.callback(work.close)
            stack.enter_context(work.activate())
            reader = stack.enter_context(ReportEvidenceReader(local.parent, cache_bytes=32 * 1024 * 1024))
            verifier = stack.enter_context(valuation_history_verifier_scope())
            for index, ref in enumerate(refs):
                path = directory / "rows" / f"{index:04d}.json"
                if path.exists():
                    receipt = json.loads(path.read_text())
                    payload = receipt["payload"]
                    if (receipt["row_sha256"] != identity(payload) or payload["previous_sha256"] != chain
                            or payload["index"] != index or payload["job_id"] != job_id
                            or payload["row"]["symbol"] != ref["symbol"]
                            or payload["row"]["saved_report_sha256"] != ref["report_sha256"]):
                        raise ValueError("FR2_EPS_COMPARISON_RESUME_IDENTITY_MISMATCH")
                    verify_retained_row(payload["row"], ref, directory, reader)
                    rows.append(payload["row"])
                    chain = receipt["row_sha256"]
                    continue
                tick = perf_counter()
                report = reader.resolve(ref)
                verify_report_valuation_sources(report, verifier=verifier)
                original = dict(report)
                original.pop("pae_authorization", None)
                try:
                    inputs = annual.valuation_inputs(original)
                    # compare_report first validates the complete original,
                    # including its old account authorization. Removing only
                    # that authorization does not change native research.
                    # The complete new report is independently revalidated.
                    _, row, inputs = compare_report(report, model, inputs,
                        apply=lambda original, model, inputs: _apply_validated(original, model,
                            valuation_inputs=inputs))
                    write_once(directory / "valuation-inputs" / (identity(inputs) + ".json"), inputs)
                except (KeyError, TypeError, ValueError, OSError) as error:
                    row = dict(symbol=report["symbol"], state="SOURCE_OR_RECOMPUTATION_GAP",
                        saved_report_sha256=identity(report), reason=str(error),
                        original_decision=report["envelope"]["strategy"]["strategy_decision"])
                payload = dict(schema=SCHEMA, job_id=job_id, index=index, previous_sha256=chain, row=row)
                receipt = dict(payload=payload, row_sha256=identity(payload))
                write_once(path, receipt)
                chain = receipt["row_sha256"]
                rows.append(row)
                print(json.dumps(dict(index=index + 1, total=len(refs), symbol=row["symbol"], state=row["state"],
                    seconds=round(perf_counter() - tick, 3)), ensure_ascii=False), flush=True)
                del report, original
    finally:
        counts = dict(annual.counts)
        annual.close()
    guard.check()
    if file_hash(Path(__file__)) != program_sha256:
        raise ValueError("FR2_EPS_COMPARISON_PROGRAM_CHANGED")
    if file_hash(bundle_path) != bundle_bytes_sha or file_hash(model_path) != model_bytes_sha:
        raise ValueError("FR2_EPS_SOURCE_OR_PREDICTOR_CHANGED_DURING_COMPARISON")
    result = dict(schema=SCHEMA, comparison_id=job_id, inputs=settings, source_bundle_path=str(bundle_path),
        predictor_path=str(model_path), row_chain_sha256=chain, rows=rows, summary=summarize(rows),
        raw_original_counts=counts, duration_seconds=round(perf_counter() - started, 3),
        comparison_scope="SAME_FROZEN_PREDICTIONS_QUALITY_PLAN_NOT_OUT_OF_SAMPLE_OR_EXECUTION")
    # Elapsed time is execution metadata, outside stable business identity.
    result["business_sha256"] = business_identity(result)
    result_path = directory / "comparison.json"
    if result_path.exists():
        retained = json.loads(result_path.read_text())
        if (business_identity(retained) != result["business_sha256"]
                or retained.get("business_sha256") != result["business_sha256"]):
            raise ValueError("FR2_EPS_RETAINED_COMPARISON_CONFLICT")
        result = retained
    write_once(directory / "comparison.json", result)
    write_bytes_once(directory / "summary.md", markdown(result).encode())
    write_once(directory / "completion.json", dict(schema=SCHEMA, comparison_id=job_id,
        result_sha256=identity(result), business_sha256=result["business_sha256"],
        complete_census=len(rows) == len(refs), summary=result["summary"],
        source_bundle_sha256=bundle_bytes_sha, predictor_file_sha256=model_bytes_sha))
    job_lock.close()
    print(json.dumps(dict(comparison=str(directory / "comparison.json"), summary=result["summary"]), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-bundle", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--fundamental-db", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=457)
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
