"""Frozen-universe score sensitivity and separately identified native pilots.

No network, mail, fills, PAE persistence or account ledger writes. Missing
source evidence stays missing; this is not a historical deployment or OOS test.
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
from domain.les_v7.policy_r22 import AVAILABLE_AT, BASELINE_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.routing_r15 import _score
from domain.les_v7.rule_baseline import CONFIG
from domain.les_v7.rule_score_r21 import score_factors_r21, score_policy as old_policy
from domain.les_v7.rule_score_r22 import score_factors_r22, score_policy
from domain.strategies.low_entry_v6_hybrid import ContextSelection
from examples.audit_les_r21 import rescore_saved
from examples import compare_les_r14 as sources
from examples.compare_les_r13 import file_hash
from examples.compare_les_r17 import differences
from infrastructure.daily_work import open_daily_work
from infrastructure.research_archive import write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.report_evidence import ReportEvidenceReader, ReportEvidenceWriter
from infrastructure.storage.valuation_source_verification import valuation_history_verifier_scope, verify_report_valuation_sources


def score_pair(body):
    verified = rescore_saved(body)  # Exact R20 replay before using saved operands.
    b = body["baseline_calculation"]; conditions = {c["id"]:c for c in b["score_evidence"]["conditions"]}
    quote = {name:(conditions["execution:"+slot]["sources"][0].get("operands") or {}).get(name)
        for slot, name in (("spread", "spread"), ("liquidity", "adv"))}
    quote["score_availability"] = {name:conditions["execution:"+slot]["status"] == "OBSERVED"
        for slot, name in (("spread", "spread"), ("liquidity", "adv"))}
    selected = b["selection"]
    selection = ContextSelection(selected["selected_setup"], tuple((k, tuple(v.items())) for k,v in selected["evidence"].items()))
    args = (b["features"], selection, plan_from_record(body["reference_plan"]), quote)
    old = score_factors_r21(*args, config=CONFIG, policy=old_policy())
    new = score_factors_r22(*args, config=CONFIG, policy=score_policy())
    if old["score_evidence"]["formal_score"] != verified["new_raw"]:
        raise ValueError("R21_TECHNICAL_REPRODUCTION_MISMATCH")
    penalty = {"FIXED_SAVED_ROUTE_DEDUCTION":body["score"]["total_penalty"]}
    return dict(r21=_score(old["score_evidence"], penalty), r22=_score(new["score_evidence"], penalty),
        condition_differences=list(differences(old["score_evidence"]["conditions"], new["score_evidence"]["conditions"])),
        old_decision=body["strategy"]["strategy_decision"],
        native_decision="NOT_RECOMPUTED_BY_FIXED_INPUT_SCORE_SENSITIVITY")


def compare_native(saved, filings, *, cutoff):
    validate_report(saved)
    sources.assert_snapshot_visibility(saved["diagnostics"]["native_evidence"]["source_packet"])
    packet = sources.reselect_packet(saved["diagnostics"]["native_evidence"]["source_packet"], evaluation_cutoff=cutoff)
    old, new = [sources.recompute(saved, filings, evaluation_cutoff=cutoff,
        policy_version=v, packet=packet) for v in ("V7_RULE_BASED_BASELINE_R21", BASELINE_VERSION)]
    for report in (old, new):
        validate_report(report)
        if report["pae_authorization"]["live_order_authorized"]:
            raise ValueError("R22_AUDIT_MUST_NOT_AUTHORIZE_LIVE_ORDERS")
        for key in ("symbol", "market", "bars_hash", "financials_hash", "external_evidence_ids"):
            if report["diagnostics"]["evidence_binding"][key] != saved["diagnostics"]["evidence_binding"][key]:
                raise ValueError("R22_RAW_SOURCE_BINDING_CHANGED:"+key)
    if old["pae_authorization"]["account_sha256"] != new["pae_authorization"]["account_sha256"]:
        raise ValueError("R22_ACCOUNT_BINDING_CHANGED")
    a,b = old["diagnostics"],new["diagnostics"]
    if a["quality"] != b["quality"]:raise ValueError("R22_QUALITY_CHANGED")
    for gate in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION"):
        state = lambda d:next(g["state"] for g in d["gate_receipts"] if g["gate_id"] == gate)
        if state(a) != state(b):raise ValueError("R22_SOURCE_GATE_CHANGED:"+gate)
    # A newly measurable structural route may create a plan, never rescue or
    # reanchor an existing one to manufacture better RR.
    if a["reference_plan"] and b["reference_plan"]:
        for key in ("entry", "stop", "target", "atr", "band", "cost_up", "cost_stop", "reference_shares"):
            if a["reference_plan"][key] != b["reference_plan"][key]:raise ValueError("R22_EXISTING_PLAN_CHANGED:"+key)
    return old,new,dict(business_differences=list(differences(sources.business(old),sources.business(new))),
        full_report_differences=list(differences(old,new)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--evaluation-cutoff",required=True)
    parser.add_argument("--expected-count",type=int,default=436)
    parser.add_argument("--fundamental-db",type=Path)
    parser.add_argument("--native-symbols",nargs="+",default=["2377","3380","2347","6277","6756"])
    args=parser.parse_args(argv);cutoff=timestamp(args.evaluation_cutoff).isoformat()
    if timestamp(cutoff)<timestamp(AVAILABLE_AT):raise ValueError("R22_AUDIT_BEFORE_RELEASE")
    root=args.comparison_root.resolve();idx=json.loads((root/"comparison-index.json").read_text())
    complete=json.loads((root/"complete-index.json").read_text())
    if digest(idx)!=complete["index_sha256"]:raise ValueError("SOURCE_INDEX_HASH_MISMATCH")
    symbols=[e["symbol"] for e in idx["records"]]
    if len(symbols)!=args.expected_count or len(set(symbols))!=len(symbols) or not set(args.native_symbols)<=set(symbols):
        raise ValueError("SOURCE_SCOPE_MISMATCH")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    guard=ResearchVersionGuard();runtime=guard.retain(ResearchObjects(args.output_dir/"runtime/objects"))
    write_once(args.output_dir/"job.json",dict(schema="les-r22-audit/v1",source_index_sha256=digest(idx),
        source_root=str(root),native_cutoff=cutoff,versions=asdict(guard.versions),runtime=runtime,
        program_sha256=file_hash(Path(__file__)),order_authority=False))
    scores=[];native=[]
    with ExitStack() as stack:
        stack.enter_context(open_daily_work(max_bytes=64*1024*1024))
        verifier=stack.enter_context(valuation_history_verifier_scope())
        restorer=sources.HashMatchedFilings(args.fundamental_db,args.native_symbols) if args.fundamental_db else None
        if restorer:stack.callback(restorer.close)
        for n,e in enumerate(idx["records"]):
            guard.check();path=root/e["row_path"];row=json.loads(path.read_text())
            if digest(row)!=e["row_sha256"]:raise ValueError("SOURCE_ROW_HASH_MISMATCH")
            result=dict(symbol=e["symbol"],source_row_sha256=e["row_sha256"],**score_pair(row["r20"]))
            write_once(args.output_dir/"scores"/f"{n:04d}.json",result);scores.append(result)
            if restorer and e["symbol"] in args.native_symbols:
                with ReportEvidenceReader(path.parent.parent) as reader:saved=reader.resolve(row["r20_report"])
                verify_report_valuation_sources(saved,verifier=verifier)
                filings,restoration=restorer.restore(saved)
                old,new,diff=compare_native(saved,filings,cutoff=cutoff)
                with ReportEvidenceWriter(args.output_dir/"report_evidence.sqlite3") as writer:
                    refs=dict(r21=writer.retain(old),r22=writer.retain(new))
                record=dict(symbol=e["symbol"],reports=refs,restoration=restoration,
                    r21=old["envelope"],r22=new["envelope"],differences=diff,
                    gates={g["gate_id"]:g["state"] for g in new["diagnostics"]["gate_receipts"]},
                    r21_selection=old["diagnostics"]["rule_baseline"]["selection"],
                    r22_selection=new["diagnostics"]["rule_baseline"]["selection"])
                write_once(args.output_dir/"native"/(e["symbol"]+".json"),record);native.append(record)
                del old,new,diff,filings,saved,record
            del row
    guard.check()
    summary=dict(total=len(scores),native_recomputed=len(native),
        raw_increased=sum(r["r22"]["raw_score"]>r["r21"]["raw_score"] for r in scores),
        old_zero=sum(r["r21"]["final_score"]==0 for r in scores),new_zero=sum(r["r22"]["final_score"]==0 for r in scores),
        crosses60=sum(r["r21"]["final_score"]<60<=r["r22"]["final_score"] for r in scores),
        universe_native_decisions="NOT_RECOMPUTED_BY_SCORE_SENSITIVITY",out_of_sample_claim=False,simulated_fills=0)
    write_once(args.output_dir/"audit.json",dict(summary=summary,scores=scores,native=native))
    write_once(args.output_dir/"complete.json",dict(audit_sha256=file_hash(args.output_dir/"audit.json"),summary=summary))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
