"""Offline LES v7 fitting, OOF validation and research-only model publication."""
from __future__ import annotations

import argparse
from pathlib import Path

from application.les_v7_training import train_request
from domain.les_v7.contracts import canonical
from infrastructure.storage.les_v7_training import (
    LesV7TrainingRepository, MAX_CONFIG_BYTES, MAX_DATASET_BYTES, read_json_bounded,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Hash-bound immutable training dataset JSON")
    parser.add_argument("--config", required=True, type=Path, help="Explicit fit cutoff, folds and policies")
    parser.add_argument("--output-dir", required=True, type=Path, help="Research-only content-addressed job root")
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicit isolated SYNTHETIC_TEST_ONLY artifacts")
    args = parser.parse_args(argv)
    try:
        dataset = read_json_bounded(args.dataset, MAX_DATASET_BYTES)
        config = read_json_bounded(args.config, MAX_CONFIG_BYTES)
        receipt = train_request(dataset, config, LesV7TrainingRepository(args.output_dir),
                                allow_synthetic=args.allow_synthetic)
        print(canonical({key: receipt[key] for key in (
            "schema", "job_id", "status", "dataset_kind", "bundle_sha256", "first_published_at",
            "output", "research_only", "approved_for_live", "execution_action")}))
        return 0 if receipt["status"] == "READY_TO_FREEZE" else 2
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
