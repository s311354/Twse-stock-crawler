"""Bounded fixed-input scoring audit plus explicit native pilot recomputation.

The universe score audit is NOT a new entry decision: it fixes the old plan,
execution operands and deductions. The native pilot uses the existing complete
source restorer/engine/finalizer/validator, with a separately declared cutoff.
Never sends mail, refreshes sources or mutates account/PAE state.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.replay_contracts import digest
from domain.les_v7.contracts import timestamp
from domain.les_v7.plan_r19 import plan_from_record
from domain.les_v7.policy_r21 import AVAILABLE_AT
from domain.les_v7.routing_r15 import _score
from domain.les_v7.rule_baseline import CONFIG
from domain.les_v7.rule_score_r20 import score_factors_r20, score_policy as previous_policy
from domain.les_v7.rule_score_r21 import score_factors_r21, score_policy
from domain.strategies.low_entry_v6_hybrid import ContextSelection
from examples import compare_les_r14 as sources
from examples.compare_les_r13 import file_hash
from examples.compare_les_r17 import differences
from examples.compare_les_r20 import compare_report
from infrastructure.daily_work import open_daily_work
from infrastructure.research_archive import write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.report_evidence import ReportEvidenceReader, ReportEvidenceWriter
from infrastructure.storage.valuation_source_verification import valuation_history_verifier_scope, verify_report_valuation_sources


def rescore_saved(body):
    """Replay the saved technical ledger before auditing a new interpretation."""
    baseline = body["baseline_calculation"]
    saved = baseline["score_evidence"]
    conditions = {c["id"]: c for c in saved["conditions"]}
    spread = conditions["execution:spread"]
    liquidity = conditions["execution:liquidity"]
    quote = dict(spread=(spread["sources"][0].get("operands") or {}).get("spread"),
        adv=(liquidity["sources"][0].get("operands") or {}).get("adv"),
        score_availability=dict(spread=spread["status"] == "OBSERVED", adv=liquidity["status"] == "OBSERVED"))
    s = baseline["selection"]
    selection = ContextSelection(s["selected_setup"], tuple((name, tuple(checks.items())) for name, checks in s["evidence"].items()))
    plan = plan_from_record(body["reference_plan"])
    old = score_factors_r20(baseline["features"], selection, plan, quote, config=CONFIG, policy=previous_policy())
    if old["score_evidence"] != saved:
        raise ValueError("SAVED_R20_TECHNICAL_LEDGER_NOT_REPRODUCED")
    new = score_factors_r21(baseline["features"], selection, plan, quote, config=CONFIG, policy=score_policy())
    # Only fix the already-declared prior route's deduction, not a new risk,
    # account or routing result. Never label this a native Final/BUY decision.
    fixed = {"SAVED_ROUTE_DEDUCTION": body["score"]["total_penalty"]}
    return dict(old_raw=old["score_evidence"]["formal_score"], new_raw=new["score_evidence"]["formal_score"],
        fixed_deduction=body["score"]["total_penalty"],
        old_after_same_deduction=_score(old["score_evidence"], fixed)["final_score"],
        new_after_same_deduction=_score(new["score_evidence"], fixed)["final_score"],
        scoring_interpretation=new["score_evidence"]["scoring_interpretation"],
        condition_differences=list(differences(saved["conditions"], new["score_evidence"]["conditions"])),
        new_entry_decision="NOT_RECOMPUTED_IN_SCORE_ONLY_AUDIT")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--evaluation-cutoff", required=True)
    parser.add_argument("--expected-count", type=int, default=436)
    parser.add_argument("--fundamental-db", type=Path)
    parser.add_argument("--native-symbols", nargs="+", default=["2303", "2308", "2327", "2345"])
    args = parser.parse_args(argv)
    cutoff = timestamp(args.evaluation_cutoff).isoformat()
    if timestamp(cutoff) < timestamp(AVAILABLE_AT):raise ValueError("R21_AUDIT_BEFORE_RELEASE")
    root = args.comparison_root.resolve()
    index = json.loads((root / "comparison-index.json").read_text())
    complete = json.loads((root / "complete-index.json").read_text())
    if digest(index) != complete["index_sha256"]:raise ValueError("SOURCE_INDEX_HASH_MISMATCH")
    if len(index["records"]) != args.expected_count:raise ValueError("SOURCE_SCOPE_MISMATCH")
    symbols = [entry["symbol"] for entry in index["records"]]
    if len(set(symbols)) != len(symbols) or not set(args.native_symbols) <= set(symbols):raise ValueError("SOURCE_SCOPE_INVALID")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    guard = ResearchVersionGuard()
    objects = ResearchObjects(args.output_dir / "runtime/objects")
    runtime = guard.retain(objects)
    write_once(args.output_dir / "job.json", dict(schema="les-r21-score-audit/v1", source_index_sha256=digest(index),
        source_root=str(root), source_cutoff=index["evaluation_cutoff"], native_evaluation_cutoff=cutoff,
        versions=asdict(guard.versions), runtime=runtime, program_sha256=file_hash(Path(__file__)),
        symbols=symbols, native_symbols=args.native_symbols if args.fundamental_db else [], order_authority=False))
    rows, native_rows = [], []
    with ExitStack() as stack:
        stack.enter_context(open_daily_work(max_bytes=64 * 1024 * 1024))
        verifier = stack.enter_context(valuation_history_verifier_scope())
        restorer = sources.HashMatchedFilings(args.fundamental_db, args.native_symbols) if args.fundamental_db else None
        if restorer:stack.callback(restorer.close)
        for number, entry in enumerate(index["records"]):
            guard.check()
            path = root / entry["row_path"]
            original = json.loads(path.read_text())
            if digest(original) != entry["row_sha256"]:raise ValueError("SOURCE_ROW_HASH_MISMATCH")
            row = dict(symbol=entry["symbol"], source_row_sha256=entry["row_sha256"])
            try:
                row.update(state="SCORE_REPRODUCED_AND_RESCORED", **rescore_saved(original["r20"]))
            except (ValueError, KeyError, TypeError) as error:
                row.update(state="SCORE_REPRODUCTION_GAP", error=str(error))
            write_once(args.output_dir / "scores" / f"{number:04d}.json", row)
            rows.append(row)
            if restorer and entry["symbol"] in args.native_symbols:
                with ReportEvidenceReader(path.parent.parent) as reader:
                    saved = reader.resolve(original["r20_report"])
                verify_report_valuation_sources(saved, verifier=verifier)
                filings, restoration = restorer.restore(saved)
                old, new, result = compare_report(saved, filings, evaluation_cutoff=cutoff,
                    versions=("V7_RULE_BASED_BASELINE_R20", "V7_RULE_BASED_BASELINE_R21"))
                with ReportEvidenceWriter(args.output_dir / "report_evidence.sqlite3") as writer:
                    references = dict(r20=writer.retain(old), r21=writer.retain(new))
                native = dict(symbol=entry["symbol"], source_row_sha256=entry["row_sha256"],
                    source_report_sha256=digest(saved), restoration=restoration, reports=references,
                    r20_score=old["envelope"]["score"], r21_score=new["envelope"]["score"],
                    r20_decision=old["envelope"]["strategy"]["strategy_decision"],
                    r21_decision=new["envelope"]["strategy"]["strategy_decision"],
                    reason_codes=new["envelope"]["strategy"]["reason_codes"],
                    gate_states={g["gate_id"]:g["state"] for g in new["diagnostics"]["gate_receipts"]},
                    account_sha256=new["pae_authorization"]["account_sha256"],
                    business_difference_paths=[d["path"] for d in result["business_differences"]],
                    full_difference_paths=[d["path"] for d in result["full_report_differences"]])
                write_once(args.output_dir / "native" / (entry["symbol"] + ".json"), native)
                native_rows.append(native)
                del saved, filings, old, new, result
            del original
    guard.check()
    reproduced = [r for r in rows if r["state"] == "SCORE_REPRODUCED_AND_RESCORED"]
    summary = dict(total=len(rows), scores_reproduced=len(reproduced), score_gaps=len(rows)-len(reproduced),
        increased=sum(r["new_raw"] > r["old_raw"] for r in reproduced),
        decreased=sum(r["new_raw"] < r["old_raw"] for r in reproduced),
        old_zero_after_same_deduction=sum(r["old_after_same_deduction"] == 0 for r in reproduced),
        new_zero_after_same_deduction=sum(r["new_after_same_deduction"] == 0 for r in reproduced),
        native_recomputed=len(native_rows), new_universe_entry_decisions="NOT_RECOMPUTED_BY_SCORE_ONLY_AUDIT")
    result = dict(summary=summary, records=rows, native=native_rows, runtime=runtime,
        out_of_sample_claim=False, simulated_fill_count=0, order_authority=False)
    write_once(args.output_dir / "audit.json", result)
    write_once(args.output_dir / "complete.json", dict(audit_sha256=digest(result), summary=summary))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":raise SystemExit(main())
