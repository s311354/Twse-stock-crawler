"""Bounded offline verification using real pinned evidence and explicit case accounts.

This is NOT a performance or fill simulation. It publishes only new study
artifacts and checks cache/API/CLI parity; missing evidence stays missing.
"""
import argparse
from dataclasses import asdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from application.historical_replay import replay_close_cutoff
from application.replay_contracts import ReplayAccount, ReplayHolding, ReplayRequest, calendar_window, digest
from infrastructure.replay_evidence import load_evidence
from infrastructure.replay_runtime import current_versions
from infrastructure.research_archive import write_once
from infrastructure.research_job import ResearchJob
from domain.strategies.low_entry_v6_research import score_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--case-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dates", nargs="+", type=dt.date.fromisoformat,
                        default=[dt.date(2026, 8, day) for day in (3, 4, 5)])
    parser.add_argument("--score-audit-validation-start", type=dt.date.fromisoformat,
                        help="Optional chronological split for read-only score contribution diagnostics")
    args = parser.parse_args()
    if not 3 <= len(args.dates) <= 5 or args.dates != sorted(set(args.dates)):
        parser.error("Use 3--5 explicit increasing cutoff dates")
    if args.output.exists():
        parser.error("Use a new output directory; existing studies are never overwritten")
    before = hashlib.sha256(args.evidence.read_bytes()).hexdigest()
    evidence_id, payload = load_evidence(args.evidence)
    symbols = ("3014", "6834")
    if not set(symbols).issubset(payload["symbols"]):
        raise ValueError("PILOT_UNIVERSE_NOT_IN_DATASET")
    case = json.loads(args.case_audit.read_text())
    by_day = {}
    for row in case["days"]:
        date = dt.date.fromisoformat(row["as_of"][:10])
        if date in by_day:
            raise ValueError("AMBIGUOUS_DAILY_ACCOUNT")
        by_day[date] = row["account"]
    versions = current_versions()
    requests = []
    for date in args.dates:
        if date not in by_day:
            raise ValueError("EXPLICIT_DAILY_ACCOUNT_REQUIRED")
        account = by_day[date]
        requests.append(ReplayRequest("v6-cutoff-quality-risk-pilot", replay_close_cutoff(str(date)),
            *calendar_window(date), symbols, payload["market"], payload["sector"],
            ReplayAccount(account["cash"], account["account_equity"],
                          tuple(ReplayHolding(**h) for h in account["holdings"])),
            versions, evidence_id, args.output / "UNUSED.sqlite3"))
    outcomes = {}
    for mode, cache_enabled in (("cached", True), ("uncached", False)):
        with ResearchJob(args.evidence, args.output / mode, cache_enabled=cache_enabled) as job:
            outcomes[mode] = []
            for request in requests:
                day = job.run_day(request)
                result = json.loads(json.dumps(day.result))
                coverage = json.loads(json.dumps(day.coverage))
                outcomes[mode].append((result, coverage))
            retry = job.run_day(requests[-1])
            assert outcomes[mode][-1] == (json.loads(json.dumps(retry.result)),
                                          json.loads(json.dumps(retry.coverage)))
    # A separate state runs the same account sequence through the CLI, not
    # merely reading the results saved by the API.
    parity = []
    for request, cached, uncached in zip(requests, outcomes["cached"], outcomes["uncached"]):
        holdings = args.output / "cli" / f"holdings-{request.history_end}.json"
        write_once(holdings, [asdict(h) for h in request.account.holdings])
        output = args.output / "cli" / "results"
        subprocess.run([sys.executable, "main.py", *symbols,
            "-b", str(request.history_start), "-e", str(request.history_end),
            "--replay-id", request.replay_id, "--replay-evidence-file", str(args.evidence),
            "--pae-holdings-file", str(holdings), "--pae-current-cash", str(request.account.cash),
            "--account-equity", str(request.account.account_equity),
            "--replay-state-path", str(args.output / "cli" / "state.sqlite3"),
            "--replay-output-dir", str(output)], check=True, capture_output=True, text=True)
        cli = json.loads((output / (request.as_of.strftime("%Y%m%dT%H%M%S%z") + ".json")).read_text())
        same = cached == uncached and cached[0] == cli
        parity.append(dict(date=str(request.history_end), equal=same,
            result_sha256=digest(cached[0]), coverage_sha256=digest(cached[1]),
            decisions=[dict(symbol=d["symbol"], raw=d["raw"], gates=d["hard_gates"],
                            research_entry_eligible=d["research_entry_eligible"])
                       for d in cached[0]["decisions"]], coverage_status=cached[1].get("status")))
        if not same:
            raise ValueError("FULL_RESULT_OR_COVERAGE_PARITY_FAILED")
    after = hashlib.sha256(args.evidence.read_bytes()).hexdigest()
    if before != after:
        raise ValueError("PINNED_EVIDENCE_CHANGED")
    report = dict(schema="v6-cutoff-quality-risk-verification/v1", evidence_id=evidence_id,
        source_file_sha256=before, source_unchanged=True, versions=asdict(versions),
        account_source=str(args.case_audit), account_policy="EXPLICIT_CASE_ACCOUNTS_NOT_SIMULATED_FILLS",
        full_cache_parity=True, full_cli_api_parity=True, same_day_retry_parity=True, days=parity,
        email_sent=False, live_order_authorized=False, is_performance_comparison=False)
    if args.score_audit_validation_start:
        native_reports = [json.loads(row["LES v6 Report"])
                          for result, _ in outcomes["cached"] for row in result["rows"]]
        audit = score_audit(native_reports, validation_start=str(args.score_audit_validation_start))
        audit.update(evidence_id=evidence_id, source_file_sha256=before, versions=asdict(versions),
                     result_sha256_by_date={row["date"]: row["result_sha256"] for row in parity},
                     source_posture="FIXED_STRICT_REPLAY_RESULTS_NOT_PERFORMANCE_VALIDATION")
        write_once(args.output / "score-audit.json", audit)
    write_once(args.output / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
