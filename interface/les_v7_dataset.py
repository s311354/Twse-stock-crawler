"""Build immutable research training data from pinned real daily-price archives."""
import argparse
from datetime import date
from pathlib import Path

from application.les_v7_dataset import validate_build_config
from domain.les_v7.contracts import canonical, timestamp
from infrastructure.research_archive import utc_now
from infrastructure.les_v7_training_source import FINAL_CLOSE, OBSERVED_ONLY, load_training_source
from infrastructure.storage.les_v7_dataset import LesV7DatasetRepository
from infrastructure.storage.les_v7_training import MAX_CONFIG_BYTES, read_json_bounded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="Existing immutable replay-evidence/v2 file")
    parser.add_argument("--config", type=Path, required=True, help="Frozen research reference-path and event policy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+", help="Explicit bounded subset; never winner-selected")
    parser.add_argument("--allow-final-daily-reconstruction", action="store_true",
                        help="Opt in to retrospective final-day close convention, NOT original observed/PIT quotes")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--backfill-market-cache", type=Path,
                        help="Offline append-only repair from the existing market cache; no HTTP")
    parser.add_argument("--history-start", type=date.fromisoformat, help="Earlier raw history start for cache-only backfill")
    parser.add_argument("--source-store", type=Path, help="New immutable evidence store for backfill; old source is never overwritten")
    args = parser.parse_args(argv)
    if any((args.backfill_market_cache, args.history_start, args.source_store)) and not all(
            (args.backfill_market_cache, args.history_start, args.source_store)):
        parser.error("--backfill-market-cache, --history-start and --source-store must be supplied together")
    try:
        config = read_json_bounded(args.config, MAX_CONFIG_BYTES)
        validate_build_config(config)
        if timestamp(config["evaluation_cutoff"]) > timestamp(utc_now()):
            raise ValueError("DATASET_EVALUATION_AFTER_BUILD_CLOCK")
        evidence, repair = args.evidence, None
        if args.backfill_market_cache:
            from infrastructure.les_v7_training_backfill import backfill_training_evidence
            repair = backfill_training_evidence(evidence, market_cache=args.backfill_market_cache,
                history_start=args.history_start, store=args.source_store, allow_synthetic=args.allow_synthetic)
            evidence = Path(repair["evidence"])
        source = load_training_source(evidence, evaluation_cutoff=config["evaluation_cutoff"],
            symbols=args.symbols, availability_policy=FINAL_CLOSE if args.allow_final_daily_reconstruction else OBSERVED_ONLY,
            **({"reference_evidence_policy": config["reference_evidence_policy"]} if config.get("reference_evidence_policy") else {}))
        receipt = LesV7DatasetRepository(args.output_dir).run(source, config, allow_synthetic=args.allow_synthetic)
        summary = {key: receipt[key] for key in ("schema", "job_id", "dataset_id", "dataset_kind",
            "status", "counts", "dataset", "output", "approved_for_live", "execution_action")}
        if repair:
            summary["market_backfill"] = repair
        print(canonical(summary))
        return 0 if receipt["status"] == "EVIDENCE_READY" else 2
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
