"""Reproduce zero-score causes without modifying scores or source evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from math import fsum, isclose
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.replay_contracts import digest
from domain.les_v7.plan_r19 import plan_from_record
from domain.les_v7.rule_baseline import CONFIG
from domain.les_v7.rule_score_r22 import score_factors_r22, score_policy
from domain.strategies.low_entry_v6_hybrid import ContextSelection
from examples.audit_les_r22 import score_pair
from examples.compare_les_r13 import file_hash
from infrastructure.research_archive import write_once


def explain_zero(ledger, score, *, plan, features, gates, quality):
    raw=fsum(c["earned_points"] for c in ledger["conditions"])
    if not isclose(raw,score["raw_score"],rel_tol=0,abs_tol=1e-10):raise ValueError("ZERO_AUDIT_RAW_LEDGER_MISMATCH")
    penalty=score["total_penalty"];signed=raw-penalty
    if score["final_score"]!=max(0.,signed):raise ValueError("ZERO_AUDIT_SCORE_ARITHMETIC_MISMATCH")
    available=fsum(c["available_weight"] for c in ledger["conditions"])
    zero=fsum(c["available_weight"] for c in ledger["conditions"] if c["earned_points"]==0)
    states={g["gate_id"]:g["state"] for g in gates}
    reasons={g["gate_id"]:g["reasons"] for g in gates}
    return dict(raw=raw,penalty=penalty,signed_after_penalty=signed,final=score["final_score"],
        cause="EXPLICIT_PENALTY_FLOOR_NOT_ZERO_RAW" if raw>0 and signed<=0 else "OTHER",
        available_weight=available,known_zero_weight=zero,unavailable_weight=100.-available,
        groups=ledger["components"],missing_fields=ledger["missing_fields"],
        history_ready=features.get("history_ready"),market_ready=features.get("market_ready"),
        support_intact=features.get("support_intact"),no_new_low2=features.get("no_new_low2"),
        plan_present=plan is not None,gate_states=states,gate_reasons=reasons,
        safety_scope="NATIVE_PRICE_TECHNICAL_SAFETY_NOT_FINANCIAL_REJECT_COUNT",
        stress_scope="NO_PLAN_TO_EVALUATE" if plan is None else "BOUND_PLAN",
        quality_status=quality["status"],quality_score=quality.get("quality_score"),
        mathematical_ceiling_if_all_unavailable_full=max(0.,raw+100.-available-penalty),
        ceiling_role="NON_EXECUTABLE_DIAGNOSTIC_NOT_IMPUTED_POINTS",
        decision_recomputed=False,order_authority=False)


def diagnose(body):
    pair=score_pair(body)
    b=body["baseline_calculation"];saved={c["id"]:c for c in b["score_evidence"]["conditions"]}
    quote={name:(saved["execution:"+slot]["sources"][0].get("operands") or {}).get(name)
        for slot,name in (("spread","spread"),("liquidity","adv"))}
    quote["score_availability"]={name:saved["execution:"+slot]["status"]=="OBSERVED"
        for slot,name in (("spread","spread"),("liquidity","adv"))}
    s=b["selection"];selection=ContextSelection(s["selected_setup"],tuple((k,tuple(v.items())) for k,v in s["evidence"].items()))
    plan=plan_from_record(body["reference_plan"])
    calculated=score_factors_r22(b["features"],selection,plan,quote,config=CONFIG,policy=score_policy())
    return explain_zero(calculated["score_evidence"],pair["r22"],plan=plan,
        features=b["features"],gates=body["gates"],quality=body["quality"])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-root",required=True,type=Path)
    p.add_argument("--score-audit-root",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    args=p.parse_args(argv)
    index=json.loads((args.comparison_root/"comparison-index.json").read_text())
    complete=json.loads((args.comparison_root/"complete-index.json").read_text())
    if digest(index)!=complete["index_sha256"]:raise ValueError("SOURCE_INDEX_HASH_MISMATCH")
    audit_path=args.score_audit_root/"audit.json"
    audit=json.loads(audit_path.read_text());done=json.loads((args.score_audit_root/"complete.json").read_text())
    if file_hash(audit_path)!=done["audit_sha256"]:raise ValueError("SCORE_AUDIT_HASH_MISMATCH")
    entries={e["symbol"]:e for e in index["records"]}
    audited_symbols=[r["symbol"] for r in audit["scores"]]
    if (len(entries)!=len(index["records"]) or len(audited_symbols)!=len(entries)
            or len(set(audited_symbols))!=len(audited_symbols) or set(audited_symbols)!=set(entries)):
        raise ValueError("SOURCE_SCOPE_MISMATCH")
    results=[]
    for saved in audit["scores"]:
        if saved["r22"]["final_score"]!=0:continue
        e=entries[saved["symbol"]]
        if e["row_sha256"]!=saved["source_row_sha256"]:raise ValueError("SCORE_SOURCE_BINDING_MISMATCH")
        row=json.loads((args.comparison_root/e["row_path"]).read_text())
        if digest(row)!=e["row_sha256"]:raise ValueError("SOURCE_ROW_HASH_MISMATCH")
        result=diagnose(row["r20"])
        if result["raw"]!=saved["r22"]["raw_score"]:raise ValueError("SAVED_ZERO_SCORE_NOT_REPRODUCED")
        results.append(dict(symbol=saved["symbol"],source_row_sha256=e["row_sha256"],**result))
    summary=dict(zero_count=len(results),raw_zero_count=sum(r["raw"]==0 for r in results),
        history_ready=sum(r["history_ready"] is True for r in results),
        market_ready=sum(r["market_ready"] is True for r in results),
        broken_or_unheld_support=sum(r["support_intact"] is False for r in results),
        plans=sum(r["plan_present"] for r in results),
        raw_min=min((r["raw"] for r in results),default=None),raw_max=max((r["raw"] for r in results),default=None),
        known_zero_weight_mean=fsum(r["known_zero_weight"] for r in results)/len(results) if results else None,
        penalties=dict(Counter(str(r["penalty"]) for r in results)),
        quality=dict(Counter(r["quality_status"] for r in results)),
        gates={g:dict(Counter(r["gate_states"][g] for r in results)) for g in
            ("DATA_INTEGRITY","TRADING_STATUS","SAFETY","VALUATION","STRESS_RISK")})
    result=dict(schema="les-zero-score-audit/v1",source_index_sha256=digest(index),
        score_audit_sha256=file_hash(audit_path),source_cutoff=index["evaluation_cutoff"],
        scoring_policy_id=digest(score_policy()),summary=summary,records=results,
        scope="ALL_ZERO_ROWS_NO_OUTCOME_SELECTION",out_of_sample_claim=False,order_authority=False)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    write_once(args.output_dir/"audit.json",result)
    write_once(args.output_dir/"complete.json",dict(audit_sha256=digest(result),summary=summary))
    print(json.dumps(summary,ensure_ascii=False))
    return 0


if __name__=="__main__":raise SystemExit(main())
