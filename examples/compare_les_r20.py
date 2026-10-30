"""Offline R19/R20 native comparison on hash-matched, saved raw inputs.

Never refresh sources, send mail, simulate fills or mutate account/PAE state.
The evaluation clock is explicit and distinct from the frozen source cutoff.
This is a counterfactual, not retroactive deployment or out-of-sample evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
import subprocess
from time import perf_counter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.replay_contracts import digest
from domain.les_v7.contracts import timestamp
from domain.les_v7.policy_r20 import BASELINE_VERSION as R20, AVAILABLE_AT
from domain.les_v7.report import validate_report
from examples import compare_les_r14 as sources
from examples.compare_les_r13 import file_hash
from examples.compare_les_r17 import differences
from examples.compare_les_r18 import _cost_operands
from infrastructure.daily_work import open_daily_work
from infrastructure.research_archive import write_once, write_bytes_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.report_evidence import ReportEvidenceReader, ReportEvidenceWriter
from infrastructure.storage.valuation_source_verification import (
    valuation_history_verifier_scope, verify_report_valuation_sources,
)

R19 = "V7_RULE_BASED_BASELINE_R19"
SCHEMA = "les-r19-r20-fixed-source-native-comparison/v1"
PROGRAMS = tuple(Path(__file__).resolve().parent / name for name in (
    "compare_les_r20.py", "compare_les_r18.py", "compare_les_r17.py", "compare_les_r14.py", "compare_les_r13.py"))


def program_hashes():
    return {path.name: file_hash(path) for path in PROGRAMS}


def aggregate_shards(directory):
    """Verify one row at a time; retain full reports/diffs only in child stores.

    Compatible with completed pre-index runs. Archived comparison programs are
    verified against their own frozen identities, not silently replaced by the
    newer aggregation implementation.
    """
    directory = Path(directory)
    job = json.loads((directory / "job.json").read_text())
    children, by_symbol = [], {}
    for number in range(job["workers"]):
        child = directory / f"shard-{number:02d}"
        record = json.loads((child / "job.json").read_text())
        receipt = json.loads((child / "complete.json").read_text())
        if (file_hash(child / "comparison.json") != receipt["comparison_sha256"]
                or any(record[key] != job[key] for key in
                       ("programs", "versions", "source_bundle_sha256", "evaluation_cutoff"))):
            raise ValueError("R20_SHARD_IDENTITY_OR_COMPLETION_MISMATCH")
        for name, sha in job["programs"].items():
            if file_hash(child / "programs" / name) != sha:
                raise ValueError("R20_SHARD_PROGRAM_ARCHIVE_MISMATCH")
        row_paths = sorted((child / "rows").glob("*.json"))
        if len(row_paths) != len(record["symbols"]) or len(row_paths) != len(receipt["row_hashes"]):
            raise ValueError("R20_SHARD_COMPLETED_SCOPE_MISMATCH")
        for index, path in enumerate(row_paths):
            row = json.loads(path.read_text())
            if (path.name != f"{index:04d}.json" or digest(row) != receipt["row_hashes"][index]
                    or row["symbol"] != record["symbols"][index] or row["symbol"] in by_symbol):
                raise ValueError("R20_SHARD_ROW_BINDING_MISMATCH")
            selected = {key: row[key] for key in ("symbol", "signal_date", "state")}
            selected.update(row_path=str(path.relative_to(directory)), row_sha256=receipt["row_hashes"][index])
            if row["state"] == "FULL_NATIVE_RECOMPUTED":
                for version in ("r19", "r20"):
                    body = row[version]
                    selected[version] = dict(decision=body["strategy"]["strategy_decision"],
                        score=body["score"], quality=body["quality"],
                        gates={g["gate_id"]: g["state"] for g in body["gates"]})
            else:
                selected["error"] = row["error"]
            by_symbol[row["symbol"]] = selected
        children.append(dict(directory=child.name, complete_sha256=file_hash(child / "complete.json"),
            comparison_sha256=receipt["comparison_sha256"],
            report_store_sha256=file_hash(child / "report_evidence.sqlite3")))
    if set(by_symbol) != set(job["symbols"]):
        raise ValueError("R20_SHARD_COMPLETED_SCOPE_MISMATCH")
    rows = [by_symbol[symbol] for symbol in job["symbols"]]
    summary = dict(total=len(rows), states=dict(Counter(row["state"] for row in rows)),
        **{version + "_decisions": dict(Counter(row[version]["decision"] for row in rows if version in row))
           for version in ("r19", "r20")},
        raw_score_changed=sum(row["r19"]["score"]["raw_score"] != row["r20"]["score"]["raw_score"]
                              for row in rows if "r20" in row))
    result = dict(schema="les-r19-r20-native-comparison-index/v1", job_sha256=file_hash(directory / "job.json"),
        source_bundle_sha256=job["source_bundle_sha256"], evaluation_cutoff=job["evaluation_cutoff"],
        children=children, records=rows, summary=summary, aggregation_programs=program_hashes(),
        full_differences_retained_in_rows=True, simulated_fill_count=0, order_authority=False)
    write_once(directory / "comparison-index.json", result)
    write_once(directory / "complete-index.json", dict(index_sha256=digest(result), summary=summary))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def compare_report(saved, filings, *, evaluation_cutoff, versions=(R19, R20)):
    cutoff = timestamp(evaluation_cutoff).isoformat()
    from domain.les_v7.rule_baseline import baseline_policy
    if versions not in {(R19, R20), (R20, "V7_RULE_BASED_BASELINE_R21")}:
        raise ValueError("NATIVE_COMPARISON_POLICY_PAIR_UNSUPPORTED")
    names = tuple(version.rsplit("_", 1)[-1].lower() for version in versions)
    available = baseline_policy(version=versions[1])["available_at"]
    if timestamp(cutoff) < timestamp(available):
        raise ValueError("R20_COMPARISON_BEFORE_PUBLICATION")
    validate_report(saved)
    source = saved["diagnostics"]["native_evidence"]["source_packet"]
    visibility = sources.assert_snapshot_visibility(source)
    packet = sources.reselect_packet(source, evaluation_cutoff=cutoff)
    reports = [sources.recompute(saved, filings, evaluation_cutoff=cutoff,
        policy_version=version, packet=packet) for version in versions]
    expected_account = sources._account(saved.get("pae_authorization", {}).get("account"), cutoff)
    for report in reports:
        validate_report(report)
        if (report["pae_authorization"]["account"] != expected_account
                or report["pae_authorization"]["live_order_authorized"] is not False
                or timestamp(report["envelope"]["event"]["cutoff_at"]) != timestamp(cutoff)):
            raise ValueError("R20_COMPARISON_ACCOUNT_OR_CUTOFF_CHANGED")
        for key in ("symbol", "market", "bars_hash", "financials_hash", "external_evidence_ids"):
            if report["diagnostics"]["evidence_binding"][key] != saved["diagnostics"]["evidence_binding"][key]:
                raise ValueError("R20_COMPARISON_RAW_INPUT_CHANGED:" + key)
    old, new = reports
    if (old["pae_authorization"]["account_sha256"] != new["pae_authorization"]["account_sha256"]
            or _cost_operands(old) != _cost_operands(new)
            or old["diagnostics"]["evidence_binding"]["native_source_id"] !=
               new["diagnostics"]["evidence_binding"]["native_source_id"]):
        raise ValueError("R20_COMPARISON_BOUND_INPUT_CHANGED")
    for key in ("quality_policy", "valuation_policy", "valuation_input_policy", "native_risk_limits",
                "tactical_risk_limits", "entry_admission_policy", "model_spec"):
        if old["diagnostics"]["rule_baseline"]["policy"][key] != new["diagnostics"]["rule_baseline"]["policy"][key]:
            raise ValueError("R20_NON_TECHNICAL_POLICY_CHANGED:" + key)
    # Identity/publication clocks may change, but stop/target/cost/band arithmetic
    # and every source-bound safety gate state may not change under score-only R20.
    a, b = old["diagnostics"], new["diagnostics"]
    for key in ("entry", "stop", "target", "atr", "band", "cost_up", "cost_stop", "reference_shares"):
        if (a["reference_plan"] or {}).get(key) != (b["reference_plan"] or {}).get(key):
            raise ValueError("R20_PLAN_OPERAND_CHANGED:" + key)
    if ({g["gate_id"]: g["state"] for g in a["gate_receipts"]} !=
            {g["gate_id"]: g["state"] for g in b["gate_receipts"]}):
        raise ValueError("R20_GATE_STATE_CHANGED")
    business = [{**sources.business(report),
                 "r19_route_evaluation": report["diagnostics"].get("r19_route_evaluation")} for report in reports]
    return old, new, dict(symbol=saved["symbol"], signal_date=saved["signal_date"], state="FULL_NATIVE_RECOMPUTED",
        original_report_sha256=digest(saved), source_cutoff=saved["envelope"]["event"]["cutoff_at"],
        evaluation_cutoff=cutoff, snapshot_visibility=visibility,
        frozen_source_packet_sha256=digest(source), selected_source_packet_sha256=digest(packet),
        bars_sha256=a["evidence_binding"]["bars_hash"], financials_sha256=a["evidence_binding"]["financials_hash"],
        account_sha256=old["pae_authorization"]["account_sha256"], common_cost_operands=_cost_operands(old),
        **{names[0]: business[0], names[1]: business[1]},
        business_differences=list(differences(*business)), full_report_differences=list(differences(old, new)))


def run_shards(args, refs, *, cutoff, source_hash):
    """Bounded isolated processes, sharing only read-only source artifacts."""
    if args.local_copy_dir:
        raise ValueError("R20_PARALLEL_REQUIRES_PREPARED_READ_ONLY_LOCAL_COPY")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    programs, guard = program_hashes(), ResearchVersionGuard()
    runtime = guard.retain(ResearchObjects(args.output_dir / "runtime-objects/objects"))
    groups = [refs[index::args.workers] for index in range(args.workers)]
    groups = [group for group in groups if group]
    write_once(args.output_dir / "job.json", dict(schema=SCHEMA, mode="ISOLATED_READ_ONLY_SHARDS",
        workers=len(groups), symbols=[ref["symbol"] for ref in refs], source_bundle_sha256=source_hash,
        evaluation_cutoff=cutoff, versions=asdict(guard.versions), programs=programs, runtime=runtime))

    def run(index_group):
        index, group = index_group
        target = args.output_dir / f"shard-{index:02d}"
        command = [sys.executable, str(Path(__file__).resolve()), "--report-bundle", str(args.report_bundle),
            "--fundamental-db", str(args.fundamental_db), "--evaluation-cutoff", cutoff,
            "--output-dir", str(target), "--symbols", *[ref["symbol"] for ref in group]]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("R20_SHARD_FAILED:" + str(index) + ":" + result.stderr[-4000:])
        receipt = json.loads((target / "complete.json").read_text())
        child_job = json.loads((target / "job.json").read_text())
        if (receipt["comparison_sha256"] != file_hash(target / "comparison.json")
                or child_job["symbols"] != [ref["symbol"] for ref in group]
                or child_job["programs"] != programs
                or child_job["versions"] != asdict(guard.versions)
                or child_job["source_bundle_sha256"] != source_hash
                or child_job["evaluation_cutoff"] != cutoff):
            raise ValueError("R20_SHARD_IDENTITY_OR_COMPLETION_MISMATCH")
        print(json.dumps(dict(shard=index, **receipt["summary"])), flush=True)
        return target

    start = perf_counter()
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        list(pool.map(run, enumerate(groups)))
    guard.check()
    if program_hashes() != programs or file_hash(args.report_bundle) != source_hash:
        raise ValueError("R20_SHARD_PROGRAM_OR_SOURCE_CHANGED")
    return aggregate_shards(args.output_dir)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["aggregate"]:
        parser = argparse.ArgumentParser(description="Verify completed shards into a bounded-memory index")
        parser.add_argument("--directory", required=True, type=Path)
        return aggregate_shards(parser.parse_args(argv[1:]).directory)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-bundle", required=True, type=Path)
    parser.add_argument("--fundamental-db", required=True, type=Path)
    parser.add_argument("--evaluation-cutoff", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--symbols", nargs="+", help="Explicit pilot subset; omission retains the entire saved universe")
    parser.add_argument("--local-copy-dir", type=Path, help="New SSD directory for consistent report SQLite backup")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1,
        help="Independent read-only processes; not daily strategy worker configuration")
    args = parser.parse_args(argv)
    cutoff = timestamp(args.evaluation_cutoff).isoformat()
    if timestamp(cutoff) < timestamp(AVAILABLE_AT):
        raise ValueError("R20_COMPARISON_BEFORE_PUBLICATION")
    original_bundle = args.report_bundle.resolve()
    bundle_bytes = original_bundle.read_bytes()
    source_hash = hashlib.sha256(bundle_bytes).hexdigest()
    bundle = json.loads(bundle_bytes)
    refs = [ref.get("_report_reference", ref) for bucket in ("reports", "stale_reports") for ref in bundle.get(bucket, [])]
    symbols = [ref["symbol"] for ref in refs]
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("R20_COMPARISON_SCOPE_INVALID")
    if args.symbols:
        if len(set(args.symbols)) != len(args.symbols) or not set(args.symbols) <= set(symbols):
            raise ValueError("R20_COMPARISON_SUBSET_INVALID")
        refs = [ref for ref in refs if ref["symbol"] in args.symbols]
    if args.workers > 1:
        return run_shards(args, refs, cutoff=cutoff, source_hash=source_hash)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if file_hash(original_bundle) != source_hash:
        raise ValueError("R20_COMPARISON_SOURCE_CHANGED_BEFORE_START")
    local_bundle, copy_receipt = (sources.local_report_copy(original_bundle, args.local_copy_dir)
        if args.local_copy_dir else (original_bundle, None))
    db = args.fundamental_db.resolve()
    db_stat = db.stat()
    source_db_stat = dict(size=db_stat.st_size, mtime_ns=db_stat.st_mtime_ns, ctime_ns=db_stat.st_ctime_ns)
    guard = ResearchVersionGuard()
    programs = program_hashes()
    for path in PROGRAMS:
        write_bytes_once(args.output_dir / "programs" / path.name, path.read_bytes())
    if program_hashes() != programs:
        raise ValueError("R20_COMPARISON_PROGRAM_CHANGED_DURING_CAPTURE")
    objects = ResearchObjects(args.output_dir / "runtime-objects/objects")
    runtime = guard.retain(objects)
    job = dict(schema=SCHEMA, source_bundle=str(original_bundle), source_bundle_sha256=source_hash,
        source_copy_receipt=str(copy_receipt) if copy_receipt else None,
        fundamental_db=str(db), fundamental_stat=source_db_stat, evaluation_cutoff=cutoff,
        symbols=[ref["symbol"] for ref in refs], versions=asdict(guard.versions), runtime=runtime, programs=programs,
        source_policy="EXACT_SAVED_FINANCIAL_AND_BAR_HASHES_NOT_ACTIVE_LATEST", order_authority=False)
    write_once(args.output_dir / "job.json", job)
    rows, start = [], perf_counter()
    with ExitStack() as stack:
        restorer = sources.HashMatchedFilings(db, job["symbols"])
        stack.callback(restorer.close)
        stack.enter_context(open_daily_work(max_bytes=64 * 1024 * 1024))
        reader = stack.enter_context(ReportEvidenceReader(local_bundle.parent))
        verifier = stack.enter_context(valuation_history_verifier_scope())
        for index, ref in enumerate(refs):
            guard.check()
            if program_hashes() != programs:
                raise ValueError("R20_COMPARISON_PROGRAM_CHANGED_DURING_JOB")
            tick = perf_counter()
            saved = reader.resolve(ref)
            verify_report_valuation_sources(saved, verifier=verifier)
            try:
                filings, restoration = restorer.restore(saved)
                old, new, row = compare_report(saved, filings, evaluation_cutoff=cutoff)
                with ReportEvidenceWriter(args.output_dir / "report_evidence.sqlite3") as writer:
                    row.update(restoration=restoration, r19_report=writer.retain(old), r20_report=writer.retain(new))
                del old, new, filings
            except (ValueError, KeyError, TypeError, OSError) as error:
                row = dict(symbol=ref["symbol"], signal_date=ref["signal_date"],
                    original_report_sha256=ref["report_sha256"], state="SOURCE_OR_RECOMPUTATION_GAP", error=str(error))
            if row["original_report_sha256"] != ref["report_sha256"]:
                raise ValueError("R20_COMPARISON_REPORT_IDENTITY_CHANGED")
            row["seconds"] = perf_counter() - tick
            guard.check()
            write_once(args.output_dir / "rows" / f"{index:04d}.json", row)
            rows.append(row)
            print(json.dumps(dict(completed=len(rows), total=len(refs), symbol=ref["symbol"],
                state=row["state"], error=row.get("error"))), flush=True)
            del saved
    final_stat = db.stat()
    if (file_hash(original_bundle) != source_hash or source_db_stat != dict(size=final_stat.st_size,
            mtime_ns=final_stat.st_mtime_ns, ctime_ns=final_stat.st_ctime_ns)):
        raise ValueError("R20_COMPARISON_SOURCE_CHANGED_DURING_JOB")
    guard.check()
    if program_hashes() != programs:
        raise ValueError("R20_COMPARISON_PROGRAM_CHANGED_DURING_JOB")
    summary = dict(total=len(rows), states=dict(Counter(row["state"] for row in rows)),
        **{version + "_decisions": dict(Counter(row[version]["strategy"]["strategy_decision"]
            for row in rows if version in row)) for version in ("r19", "r20")})
    result = dict(schema=SCHEMA, job=job, summary=summary, seconds=perf_counter() - start, records=rows,
        historical_deployment_claim=False, out_of_sample_claim=False, simulated_fill_count=0,
        account_policy="EXPLICIT_SAVED_VALUES_REUSED_NOT_CURRENT_ACCOUNT_OR_LEDGER_CLAIM",
        pae_persistence="NOT_REBASED_OR_SIMULATED", raw_dataset_self_contained=False)
    write_once(args.output_dir / "comparison.json", result)
    write_once(args.output_dir / "complete.json", dict(schema=SCHEMA, comparison_sha256=digest(result),
        row_hashes=[digest(row) for row in rows], runtime=runtime, summary=summary))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
