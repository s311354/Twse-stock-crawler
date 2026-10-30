"""Native R22/R23 source-dependency comparison on unchanged saved originals."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in {None,""}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from application.replay_contracts import digest
from domain.les_v7.contracts import timestamp
from domain.les_v7.policy_r23 import AVAILABLE_AT, BASELINE_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy
from domain.les_v7.source_projection import join_official_pe
from examples import compare_les_r14 as sources
from examples.compare_les_r17 import differences
from infrastructure.daily_work import open_daily_work
from infrastructure.research_archive import write_once
from infrastructure.research_objects import ResearchObjects,ResearchVersionGuard
from infrastructure.storage.report_evidence import ReportEvidenceReader,ReportEvidenceWriter
from infrastructure.storage.valuation_source_verification import valuation_history_verifier_scope,verify_report_valuation_sources


def compare_report(saved,filings,*,cutoff):
    validate_report(saved)
    original=saved["diagnostics"]["native_evidence"]["source_packet"]
    sources.assert_snapshot_visibility(original)
    selected=sources.reselect_packet(original,evaluation_cutoff=cutoff)
    reports=[]
    for version in ("V7_RULE_BASED_BASELINE_R22",BASELINE_VERSION):
        policy=baseline_policy(saved["diagnostics"]["evidence_binding"]["market"],version=version)
        packet=dict(selected)
        packet["market_rows"],packet["source_join_diagnostics"]=join_official_pe(
            selected["quote_market_rows"],selected["official_pe"],symbol=saved["symbol"],
            market=policy["market"],cutoff=cutoff,source_policy=policy["source_policy"])
        report=sources.recompute(saved,filings,evaluation_cutoff=cutoff,policy_version=version,packet=packet)
        validate_report(report)
        if report["pae_authorization"]["live_order_authorized"]:raise ValueError("NO_LIVE_ORDER_AUTHORITY")
        for key in ("bars_hash","financials_hash","external_evidence_ids","market","symbol"):
            if report["diagnostics"]["evidence_binding"][key]!=saved["diagnostics"]["evidence_binding"][key]:
                raise ValueError("SOURCE_INPUT_CHANGED:"+key)
        reports.append(report)
    old,new=reports;a,b=old["diagnostics"],new["diagnostics"]
    for key in ("quality","features","reference_plan","factor_ledger"):
        if key=="reference_plan":
            for name in ("entry","stop","target","atr","band","cost_up","cost_stop","structure_id"):
                if (a[key] or {}).get(name)!=(b[key] or {}).get(name):raise ValueError("REFERENCE_OPERAND_CHANGED:"+name)
        elif a[key]!=b[key]:raise ValueError("UNEXPECTED_RESEARCH_DIFFERENCE:"+key)
    if old["envelope"]["score"]!=new["envelope"]["score"]:raise ValueError("TECHNICAL_SCORE_CHANGED")
    if old["pae_authorization"]["account_sha256"]!=new["pae_authorization"]["account_sha256"]:
        raise ValueError("ACCOUNT_CHANGED")
    return old,new,dict(business=list(differences(sources.business(old),sources.business(new))),
        complete=list(differences(old,new)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-root",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--fundamental-db",required=True,type=Path)
    p.add_argument("--evaluation-cutoff",required=True)
    p.add_argument("--symbols",nargs="+",default=["2301","2377"])
    args=p.parse_args(argv);cutoff=timestamp(args.evaluation_cutoff).isoformat()
    if timestamp(cutoff)<timestamp(AVAILABLE_AT):raise ValueError("R23_BEFORE_RELEASE")
    root=args.comparison_root.resolve();index=json.loads((root/"comparison-index.json").read_text())
    done=json.loads((root/"complete-index.json").read_text())
    if digest(index)!=done["index_sha256"]:raise ValueError("SOURCE_INDEX_HASH_MISMATCH")
    entries={e["symbol"]:e for e in index["records"]}
    if len(set(args.symbols))!=len(args.symbols) or not set(args.symbols)<=set(entries):raise ValueError("SOURCE_SCOPE_MISMATCH")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    guard=ResearchVersionGuard();runtime=guard.retain(ResearchObjects(args.output_dir/"runtime/objects"))
    write_once(args.output_dir/"job.json",dict(source_index_sha256=digest(index),versions=asdict(guard.versions),
        runtime=runtime,cutoff=cutoff,symbols=args.symbols,order_authority=False))
    rows=[]
    with ExitStack() as stack:
        stack.enter_context(open_daily_work(max_bytes=64*1024*1024))
        verifier=stack.enter_context(valuation_history_verifier_scope())
        restorer=sources.HashMatchedFilings(args.fundamental_db,args.symbols);stack.callback(restorer.close)
        for symbol in args.symbols:
            guard.check();entry=entries[symbol];path=root/entry["row_path"];row=json.loads(path.read_text())
            if digest(row)!=entry["row_sha256"]:raise ValueError("SOURCE_ROW_HASH_MISMATCH")
            with ReportEvidenceReader(path.parent.parent) as reader:saved=reader.resolve(row["r20_report"])
            verify_report_valuation_sources(saved,verifier=verifier)
            filings,restoration=restorer.restore(saved)
            old,new,diff=compare_report(saved,filings,cutoff=cutoff)
            with ReportEvidenceWriter(args.output_dir/"report_evidence.sqlite3") as writer:
                refs=dict(r22=writer.retain(old),r23=writer.retain(new))
            record=dict(symbol=symbol,source_row_sha256=entry["row_sha256"],reports=refs,restoration=restoration,
                old=old["envelope"],new=new["envelope"],differences=diff,
                gate_states={v:{g["gate_id"]:g["state"] for g in r["diagnostics"]["gate_receipts"]}
                    for v,r in (("r22",old),("r23",new))},
                source_join_diagnostics=new["diagnostics"]["native_evidence"]["source_packet"]["source_join_diagnostics"])
            write_once(args.output_dir/"rows"/(symbol+".json"),record);rows.append(record)
            del saved,filings,old,new,diff,row
    guard.check()
    result=dict(schema="les-r23-native-source-comparison/v1",records=rows,
        score_formulas_changed=False,out_of_sample_claim=False,simulated_fills=0,order_authority=False)
    write_once(args.output_dir/"comparison.json",result)
    write_once(args.output_dir/"complete.json",dict(comparison_sha256=digest(result),count=len(rows)))
    print(json.dumps(dict(count=len(rows),gates={r["symbol"]:r["gate_states"] for r in rows}),ensure_ascii=False))
    return 0


if __name__=="__main__":raise SystemExit(main())
