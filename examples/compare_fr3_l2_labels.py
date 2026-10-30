"""Read-only R1/L2 label census; not prediction or account performance.

Both preparations are independently fully verified. Only small detached row
summaries are joined; raw datasets are released one shard at a time. All
symbols, including zero-row securities, remain in the output.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.les_v7.contracts import CONTEXTS, identity, timestamp
from infrastructure.les_v7_scope_training import verify_sharded_training_dataset, _relative
from infrastructure.storage.les_v7_prepared import inspect_prepared_packet
from infrastructure.storage.les_v7_training import read_json_bounded, MAX_CONFIG_BYTES, MAX_DATASET_BYTES
from infrastructure.research_archive import write_once


def _summary(row):
    return dict(symbol=row["symbol"], context=row["context"], cutoff=row["cutoff"],
        label_available_at=row["label_available_at"], labels=row["labels"], utility=row["utility"])


def compare_labels(parent_root, new_root, output_root):
    roots = [Path(p).resolve() / "native" for p in (parent_root, new_root)]
    manifests, configs = [], []
    for root in roots:
        config = read_json_bounded(root / "scope-training-config.json", MAX_CONFIG_BYTES)
        record = read_json_bounded(root / "scope-training-dataset.json", MAX_DATASET_BYTES)
        manifests.append(verify_sharded_training_dataset(record, root, config=config))
        configs.append(config)
    left, right = manifests
    if any(left[k] != right[k] for k in ("universe", "freeze_id", "scope_id", "source_evidence_ids")):
        raise ValueError("FR3_L2_LABEL_COMPARISON_REQUIRES_IDENTICAL_RAW_SCOPE")
    for key in ("fit_cutoff", "validation_start", "validation_end", "evaluation_cutoff", "folds"):
        if configs[0][key] != configs[1][key]:
            raise ValueError("FR3_L2_LABEL_COMPARISON_REQUIRES_IDENTICAL_SPLITS")
    stocks = {s: dict(symbol=s, market=m, R1=Counter(), L2=Counter(), changes=Counter()) for s, m in left["universe"]}
    totals = dict(R1=Counter(), L2=Counter(), changes=Counter())
    context_counts = {c: dict(R1=Counter(), L2=Counter(), changes=Counter()) for c in CONTEXTS}
    episodes, episode_symbols, episode_months = set(), set(), set()
    fit = timestamp(configs[1]["fit_cutoff"])
    folds = [(timestamp(f["prediction_start"]), timestamp(f["prediction_end"])) for f in configs[1]["folds"]]
    for a, b in zip(left["children"], right["children"]):
        if a["symbols"] != b["symbols"]:
            raise ValueError("FR3_L2_LABEL_COMPARISON_SHARD_ORDER_CHANGED")
        joined = []
        for root, child in zip(roots, (a, b)):
            packet = inspect_prepared_packet(_relative(root, child["prepared_path"], "prepared")).value
            if identity(packet) != child["prepared_sha256"]:
                raise ValueError("FR3_L2_LABEL_COMPARISON_PREPARED_CHANGED")
            joined.append({r["row_id"]: _summary(r) for r in packet["rows"]})
            del packet
        for version, rows in zip(("R1", "L2"), joined):
            for row in rows.values():
                counts = dict(prepared=1, filled=int(row["labels"]["fill"] == 1),
                    known_utility=int(row["utility"] is not None))
                for collection in (totals, stocks[row["symbol"]], context_counts[row["context"]]):
                    collection[version].update(counts)
        for rid in set(joined[0]) | set(joined[1]):
            old, new = joined[0].get(rid), joined[1].get(rid)
            row = new or old
            state = "NEW_PREPARED" if old is None else "NO_LONGER_PREPARED" if new is None else (
                "UNCHANGED_LABEL_AND_UTILITY" if old["labels"] == new["labels"] and old["utility"] == new["utility"]
                else "CHANGED_LABEL_OR_UTILITY")
            for collection in (totals, stocks[row["symbol"]], context_counts[row["context"]]):
                collection["changes"][state] += 1
        # Count entry episodes only where the new native path really filled
        # and its H20 utility is known; candidates and losers are not hidden.
        dataset = read_json_bounded(_relative(roots[1], b["archive_path"], "builds") / "dataset.json", MAX_DATASET_BYTES)
        if identity(dataset["payload"]) != b["dataset_id"]:
            raise ValueError("FR3_L2_LABEL_COMPARISON_DATASET_CHANGED")
        for event in dataset["payload"]["events"]:
            execution = event["execution"]
            if identity(execution) != event["execution_sha256"]:
                raise ValueError("FR3_L2_LABEL_COMPARISON_EXECUTION_CHANGED")
            cutoff = timestamp(event["cutoff"])
            rid = identity(dict(symbol=event["symbol"], market=event["market"], cutoff=cutoff.isoformat()))
            row = joined[1].get(rid)
            if (row is None or row["context"] != "LEADERSHIP_CONTINUATION" or row["utility"] is None
                    or timestamp(row["label_available_at"]) > fit
                    or not any(first <= cutoff <= last for first, last in folds)
                    or execution.get("filled") is not True):
                continue
            filled = timestamp(execution["filled_at"]).date().isoformat()
            episodes.add((event["symbol"], filled, row["context"]))
            episode_symbols.add(event["symbol"])
            episode_months.add(filled[:7])
        del dataset, joined
    report = dict(schema="les-v7-fr3-l2-native-label-comparison/v1", universe=left["universe"],
        fixed_universe_count=len(stocks), parent_config_id=identity(configs[0]), config_id=identity(configs[1]),
        raw_freeze_id=left["freeze_id"], counts=totals, contexts=context_counts, stocks=list(stocks.values()),
        leadership_mature_oof_utility_entry_episodes=len(episodes),
        leadership_episode_definition="UNIQUE_SYMBOL_ACTUAL_ENTRY_DATE_CONTEXT_NOT_INDEPENDENT_TRADES",
        leadership_episode_stock_count=len(episode_symbols), leadership_episode_month_count=len(episode_months),
        valid_calibrated_oof_count=None, forecasts_recomputed=False,
        interpretation="NATIVE_REFERENCE_LABELS_NOT_MODEL_GAIN_OR_ACCOUNT_RETURN",
        research_only=True, approved_for_live=False, execution_action="NO_ACTION")
    folder = Path(output_root) / identity(report)
    write_once(folder / "labels.json", report)
    return folder, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-training-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    folder, report = compare_labels(args.parent_training_root, args.training_root, args.output_dir)
    from domain.les_v7.contracts import canonical
    print(canonical(dict(output=str(folder), fixed_universe_count=report["fixed_universe_count"],
        counts=report["counts"], contexts=report["contexts"], valid_calibrated_oof_count=None)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
