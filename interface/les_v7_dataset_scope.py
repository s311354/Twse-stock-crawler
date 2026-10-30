"""Manual full-ELEC reference dataset build; no model fitting, HTTP or accounts."""
import argparse
from datetime import date
from pathlib import Path

from domain.les_v7.contracts import canonical
from infrastructure.les_v7_dataset_scope import freeze_elec_scope, run_elec_scope, verify_elec_scope
from infrastructure.storage.les_v7_training import read_json_bounded, MAX_CONFIG_BYTES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--calendar-evidence", type=Path, help="Original calendar evidence only; does not expand its old stock universe")
    inputs.add_argument("--resume-freeze", type=Path, help="Reuse exactly frozen shards, without rereading today's market cache")
    inputs.add_argument("--verify-job", type=Path)
    parser.add_argument("--market-cache", type=Path)
    parser.add_argument("--market-supplement", type=Path, help="Pinned official FINAL warmup manifest; does not change the market cache")
    parser.add_argument("--history-start", type=date.fromisoformat)
    parser.add_argument("--history-end", type=date.fromisoformat)
    parser.add_argument("--universe-date", type=date.fromisoformat)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-final-daily-reconstruction", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--freeze-only", action="store_true", help="Publish immutable sources only; do not run thousands of candidate computations")
    args = parser.parse_args(argv)
    if args.freeze_only and not args.calendar_evidence:
        parser.error("--freeze-only requires --calendar-evidence (new source freeze)")
    try:
        if args.verify_job:
            receipt = verify_elec_scope(args.verify_job, root=args.output_dir)
        else:
            config = read_json_bounded(args.config, MAX_CONFIG_BYTES)
            freeze = args.resume_freeze
            if args.calendar_evidence:
                if not all((args.market_cache, args.history_start, args.history_end, args.universe_date)):
                    parser.error("new freeze requires --market-cache, --history-start, --history-end and --universe-date")
                freeze = freeze_elec_scope(args.calendar_evidence, market_cache=args.market_cache,
                    history_start=args.history_start, history_end=args.history_end, universe_date=args.universe_date,
                    config=config, root=args.output_dir, batch_size=args.batch_size, allow_synthetic=args.allow_synthetic,
                    **({"market_supplement": args.market_supplement} if args.market_supplement else {}))
            print(canonical(dict(event="elec_scope_frozen", freeze=str(freeze))), flush=True)
            if args.freeze_only:
                print(canonical(dict(status="SOURCES_FROZEN_NOT_TRAINING_READY", freeze=str(freeze),
                    research_only=True, approved_for_live=False, execution_action="NO_ACTION")))
                return 0
            def progress(done, total, receipt):
                print(canonical(dict(event="elec_scope_batch_completed", completed_batches=done,
                    total_batches=total, status=receipt["status"], counts=receipt["counts"])), flush=True)
            receipt = run_elec_scope(freeze, config, root=args.output_dir,
                allow_final_daily_reconstruction=args.allow_final_daily_reconstruction,
                allow_synthetic=args.allow_synthetic, progress=progress)
        print(canonical(receipt))
        return 0 if receipt["status"] == "EVIDENCE_READY" else 2
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
