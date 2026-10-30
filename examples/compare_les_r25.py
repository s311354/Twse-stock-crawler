"""Offline R24/R25 score/routing ablation on verified, fixed R24 reports.

Uses the native score_baseline, not a replacement scoring/entry engine. Original
reports are fully resolved and validated. Unchanged Quality/source/plan receipts
are reused, NOT recollected or claimed available today. A new policy applied to
an earlier cutoff is explicitly counterfactual; no new formal report, PAE
authorization, fill, order, mail or account state is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.replay_contracts import digest
from application.low_entry_v7_daily import daily_input_outcome
from domain.les_v7.contracts import GateReceipt, identity, timestamp
from domain.les_v7.decision import vetoes
from domain.les_v7.plan_r19 import plan_from_record
from domain.les_v7.policy_r25 import BASELINE_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy, score_baseline
from examples.compare_les_r13 import file_hash
from infrastructure.research_archive import write_once, write_bytes_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.report_evidence import ReportEvidenceReader

SCHEMA = "les-r24-r25-fixed-input-score-routing-ablation/v1"
PREVIOUS = "V7_RULE_BASED_BASELINE_R24"


def compare_report(report):
    validate_report(report)
    original_hash = digest(report)
    d = report["diagnostics"]
    saved = d["rule_baseline"]
    if saved["policy"]["policy_version"] != PREVIOUS:
        raise ValueError("R25_REQUIRES_VERIFIED_R24_REPORT")
    policy = baseline_policy(d["evidence_binding"]["market"], version=BASELINE_VERSION)
    for key in ("quality_policy", "source_policy", "source_selection_policy", "reference_policy",
                "valuation_policy", "valuation_input_policy", "entry_admission_policy",
                "native_risk_limits", "tactical_risk_limits"):
        if policy[key] != saved["policy"][key]:
            raise ValueError("R25_UNCHANGED_DEPENDENCY_MISMATCH:" + key)
    plan = plan_from_record(d["reference_plan"]) if d["reference_plan"] else None
    gates = [GateReceipt(**dict(row, reasons=tuple(row["reasons"]),
        proof_record_ids=tuple(row["proof_record_ids"]), input_ids=tuple(row["input_ids"])))
        for row in d["gate_receipts"]]
    execution = d["execution_assessment"]
    risks = list(vetoes(d["features"])) + [g.gate_id + "_FAIL" for g in gates if g.state == "FAIL"]
    if execution["state"] == "FAIL": risks.append("EXECUTION_FAIL")
    common = dict(plan=plan, quality=d["quality"], gates=gates, native_features=d["features"],
        risks=risks, standard=bool(plan and execution["state"] == "PASS"),
        starter_gate=d["starter_gate"], confirmed_gate=d["confirmed_gate"],
        valuation_receipt=d["native_evidence"]["produced_receipts"].get("VALUATION"),
        entry_binding=d["gate_bindings"]["SAFETY"], execution_assessment=execution)
    variants = {"r24": saved["policy"], "score_only": dict(policy,
        unified_technical_policy=dict(policy["unified_technical_policy"], small_minimum=60.)),
        "threshold_only": dict(saved["policy"], unified_technical_policy=dict(
            saved["policy"]["unified_technical_policy"], small_minimum=55.)), "r25": policy}
    outcomes = {}
    for name, fixed in variants.items():
        # Ablations are deliberately not registered formal policies and never
        # enter require_baseline_policy/report/PAE publication. Their exact
        # parameter hashes are retained for deterministic research comparison.
        prepared = dict(saved, policy=fixed, policy_id=identity(fixed))
        value = score_baseline(prepared, **common)
        score, decision = daily_input_outcome(value["score"], value["decision"],
            errors=d.get("daily_input_errors", ()))
        outcomes[name] = dict(decision=decision, score=score, pre_input_error_score=value["score"],
            penalties=value["penalties"], factor_ledger=value["factor_ledger"],
            score_evidence=value["score_evidence"], routing=value["r19_route_evaluation"],
            admission=value["entry_admission"], parameter_sha256=identity(fixed),
            registered_formal_policy=name in {"r24", "r25"})
    if (outcomes["r24"]["decision"] != report["envelope"]["strategy"]["strategy_decision"]
            or outcomes["r24"]["score"]["final_score"] != report["envelope"]["score"]["final_score"]
            or outcomes["r24"]["score_evidence"] != saved["score_evidence"]
            or outcomes["r24"]["routing"] != d["r19_route_evaluation"]):
        raise ValueError("R25_R24_CONTROL_REPRODUCTION_MISMATCH")
    if digest(report) != original_hash:
        raise ValueError("R25_SHARED_INPUT_MUTATED")
    return dict(schema=SCHEMA, symbol=report["symbol"], signal_date=report["signal_date"],
        original_report_sha256=original_hash, source_cutoff=report["envelope"]["event"]["cutoff_at"],
        policy_available_at=policy["available_at"],
        temporal_counterfactual=timestamp(report["envelope"]["event"]["cutoff_at"]) < timestamp(policy["available_at"]),
        evidence_binding=d["evidence_binding"], account_sha256=report["pae_authorization"]["account_sha256"],
        source_receipts_sha256=identity(d["gate_receipts"]), quality_sha256=identity(d["quality"]),
        quality_status=d["quality"]["status"], gates={g.gate_id:g.state for g in gates},
        daily_input_errors=d.get("daily_input_errors", []),
        unchanged_plan=d["reference_plan"], unchanged_execution=execution, outcomes=outcomes,
        full_native_r24_revalidated=True, native_r25_score_and_routes_recomputed=True,
        quality_and_sources_recollected=False, new_formal_report=False, new_pae_authorization=False,
        simulated_fills=0, order_authority=False, account_or_pae_writes=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--symbols", nargs="+")
    args = parser.parse_args(argv)
    root = args.comparison_root.resolve()
    index = json.loads((root / "comparison-index.json").read_text())
    complete = json.loads((root / "complete-index.json").read_text())
    if digest(index) != complete["index_sha256"]:
        raise ValueError("R25_SOURCE_INDEX_HASH_MISMATCH")
    entries = {row["symbol"]: row for row in index["records"]}
    symbols = args.symbols or list(entries)
    if (not symbols or len(symbols) != len(set(symbols)) or len(entries) != len(index["records"])
            or not set(symbols) <= entries.keys()):
        raise ValueError("R25_FIXED_SCOPE_MISMATCH")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    guard = ResearchVersionGuard()
    runtime = guard.retain(ResearchObjects(args.output_dir / "runtime-objects/objects"))
    program = Path(__file__).read_bytes()
    job = dict(schema=SCHEMA, source_index_sha256=digest(index), fixed_symbols=symbols,
        program_sha256=file_hash(__file__), mode="OFFLINE_FIXED_INPUT_NOT_FORMAL_REPORT_OR_OOS",
        r25_policy=baseline_policy(version=BASELINE_VERSION), runtime=runtime,
        mail_enabled=False, order_authority=False)
    write_once(args.output_dir / "job.json", job)
    write_bytes_once(args.output_dir / "programs" / Path(__file__).name, program)
    rows, readers = [], {}
    # Bounded private SSD copies of four immutable stores, not the 17GB source
    # DB. Full report/node/source checks remain enabled; release each report.
    with tempfile.TemporaryDirectory(prefix="les-r25-stores-") as temporary, ExitStack() as stack:
        for ordinal, symbol in enumerate(symbols):
            guard.check()
            entry = entries[symbol]
            path = (root / entry["row_path"]).resolve()
            if not path.is_relative_to(root): raise ValueError("R25_SOURCE_ROW_PATH_ESCAPE")
            row = json.loads(path.read_text())
            if digest(row) != entry["row_sha256"] or row["symbol"] != symbol:
                raise ValueError("R25_SOURCE_ROW_HASH_MISMATCH")
            directory = path.parent.parent
            if directory not in readers:
                local = Path(temporary) / directory.name
                local.mkdir()
                original = directory / "report_evidence.sqlite3"
                copied = local / original.name
                shutil.copyfile(original, copied)
                if file_hash(original) != file_hash(copied):
                    raise ValueError("R25_STORE_COPY_HASH_MISMATCH")
                readers[directory] = stack.enter_context(ReportEvidenceReader(local))
            report = readers[directory].resolve(row["reports"]["r24"])
            result = compare_report(report)
            result.update(source_row_sha256=digest(row), source_report_reference=row["reports"]["r24"])
            target = args.output_dir / "rows" / f"{ordinal:04d}.json"
            write_once(target, result)
            rows.append(dict(symbol=symbol, row_path=str(target.relative_to(args.output_dir)),
                row_sha256=digest(result), quality_status=result["quality_status"],
                outcomes={k:dict(decision=v["decision"], final=v["score"]["final_score"],
                    raw=v["score"]["raw_score"]) for k,v in result["outcomes"].items()}))
            print(json.dumps(dict(completed=ordinal+1, total=len(symbols), **rows[-1]), ensure_ascii=False), flush=True)
            del result, report, row
    guard.check()
    summary = {name: dict(decisions=dict(Counter(r["outcomes"][name]["decision"] for r in rows)),
        final_zero=sum(r["outcomes"][name]["final"] == 0 for r in rows),
        final_missing=sum(r["outcomes"][name]["final"] is None for r in rows))
        for name in ("r24", "score_only", "threshold_only", "r25")}
    result = dict(schema=SCHEMA, job_sha256=digest(job), records=rows, summary=summary,
        total=len(rows), oos_or_performance_claim=False, account_or_pae_writes=False)
    write_once(args.output_dir / "comparison-index.json", result)
    write_once(args.output_dir / "complete-index.json", dict(index_sha256=digest(result), summary=summary))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
