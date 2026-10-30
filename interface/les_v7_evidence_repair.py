"""Manual retrospective source repair and preregistration; no fitting/live work."""
import argparse
from copy import deepcopy
from datetime import date
from pathlib import Path

from domain.les_v7.contracts import canonical
from infrastructure.storage.les_v7_training import read_json_bounded, MAX_CONFIG_BYTES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-final-reference-reconstruction", action="store_true",
        help="Explicitly accept FINAL retrospective sources, NOT original historical information sets")
    commands = parser.add_subparsers(dest="command", required=True)
    calendar = commands.add_parser("calendar")
    calendar.add_argument("--output-dir", type=Path, required=True)
    calendar.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    calendar.add_argument("--previous", type=Path)
    calendar.add_argument("--max-requests", type=int, default=20)
    actions = commands.add_parser("actions")
    actions.add_argument("--start", required=True)
    actions.add_argument("--end", required=True)
    actions.add_argument("--output-dir", type=Path, required=True)
    actions.add_argument("--max-requests", type=int, default=24)
    warmup = commands.add_parser("warmup")
    warmup.add_argument("--calendar", type=Path, required=True)
    warmup.add_argument("--start", type=date.fromisoformat, required=True)
    warmup.add_argument("--end", type=date.fromisoformat, required=True)
    warmup.add_argument("--output-dir", type=Path, required=True)
    warmup.add_argument("--max-requests", type=int, default=128)
    warmup.add_argument("--previous", type=Path, help="Verify and reuse earlier raw day captures while expanding the warmup range")
    listing = commands.add_parser("listing")
    listing.add_argument("--output-dir", type=Path, required=True)
    audit = commands.add_parser("warmup-audit")
    audit.add_argument("--freeze", type=Path, required=True)
    audit.add_argument("--listing", type=Path, required=True)
    audit.add_argument("--cutoff", required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    plan = commands.add_parser("freeze-plan")
    plan.add_argument("--seed-build-config", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    repair = commands.add_parser("repair")
    repair.add_argument("--parent", type=Path, required=True)
    repair.add_argument("--calendar", type=Path, required=True)
    repair.add_argument("--actions", type=Path, required=True)
    repair.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.allow_final_reference_reconstruction:
        parser.error("explicit --allow-final-reference-reconstruction required; strict evidence is never changed")
    try:
        if args.command == "calendar":
            from infrastructure.les_v7_calendar_final import collect_final_calendars
            path = collect_final_calendars(args.output_dir, years=tuple(args.years),
                max_requests=args.max_requests, previous=args.previous)
            result = dict(artifact=str(path), status=read_json_bounded(path.parent / "completion.json", MAX_CONFIG_BYTES)["status"])
        elif args.command == "actions":
            from infrastructure.les_v7_actions_final import collect_actions_final
            result = dict(artifact=str(collect_actions_final(args.start, args.end,
                args.output_dir, max_requests=args.max_requests)))
        elif args.command == "warmup":
            from infrastructure.les_v7_market_final import collect_final_market, load_final_market
            result = dict(artifact=str(collect_final_market(args.calendar, start=args.start,
                end=args.end, root=args.output_dir, max_requests=args.max_requests,
                previous=args.previous,
                progress=lambda d, c, n: print(canonical(dict(event="warmup_captured", date=d,
                    requests=c, completed_days=n)), flush=True))))
            body, _ = load_final_market(result["artifact"])
            result.update(status=body["status"], captured_days=len(body["documents"]), missing_days=len(body["gaps"]))
        elif args.command == "listing":
            from infrastructure.les_v7_warmup_audit import collect_listing_final
            result = dict(artifact=str(collect_listing_final(args.output_dir, include_profiles=True, max_requests=2)))
        elif args.command == "warmup-audit":
            from infrastructure.les_v7_warmup_audit import audit_scope_warmup
            result = dict(artifact=str(audit_scope_warmup(args.freeze, args.listing,
                cutoff=args.cutoff, root=args.output_dir)))
            summary = read_json_bounded(result["artifact"], 4 * MAX_CONFIG_BYTES)["body"]
            result.update(ordinary_complete=summary["ordinary_complete"],
                ordinary_incomplete=summary["ordinary_incomplete"], status_counts=summary["status_counts"])
        elif args.command == "freeze-plan":
            from application.les_v7_training_plan import generate_fixed_research_plan
            from infrastructure.les_v7_training_plan import freeze_fixed_research_plan
            from domain.les_v7.reference_path import FINAL_COORDINATE_POLICY
            seed = deepcopy(read_json_bounded(args.seed_build_config, MAX_CONFIG_BYTES)["seed"])
            seed["reference_policy"]["cost_path_policy"]["coordinate_policy"] = FINAL_COORDINATE_POLICY
            result = freeze_fixed_research_plan(generate_fixed_research_plan(seed), args.output_dir)
        else:
            from infrastructure.les_v7_reference_repair import repair_reference_evidence
            result = dict(artifact=str(repair_reference_evidence(args.parent, calendar_path=args.calendar,
                actions_path=args.actions, store=args.output_dir)))
        print(canonical(dict(result=result, research_only=True, historical_information_set_certified=False,
            fitting_performed=False, approved_for_live=False, execution_action="NO_ACTION")))
        return 2 if result.get("status") == "PARTIAL" else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
