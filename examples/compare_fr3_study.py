r"""Recompute the new FR3 archive's matched holdout and native reference costs.

Run only after the separate full-scope training job has finished::

    .venv-py315/bin/python examples/compare_fr3_study.py \
      --archive /path/to/original/frozen_bundle.json \
      --training-root /path/to/new-fr3-training \
      --source-audit /path/to/source-audit.json \
      --output-dir /path/to/separate-fr3-study

The native archive reconstructor verifies prepared rows and re-executes saved
M0/M1 heads, past-only OOF calibration, holdout metrics and exact fee arithmetic.
Its legacy FR2 admission-arm labels are deliberately NOT used here. Both models
use the NEW FR3 plan/labels; this is a viewed holdout, not fresh confirmation.
One-share reference PnL is not an allocated-account return. Missing native
signals, financial/restriction evidence and four account paths remain gaps.
No network, mail, deployment, account ledger or order port is opened.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import hashlib
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.les_v7.contracts import canonical, identity
from domain.les_v7.fr3_plan import LABEL_VERSION, PATH_VERSION, PLAN_VERSION, require_policy
from infrastructure.les_v7_fr2_archive import reconstruct_archive
from infrastructure.les_v7_fr3_comparison import compare_fr3_research, fr3_reference_versions
from infrastructure.les_v7_fr3_training import verify_fr3_scope_training
from infrastructure.research_archive import write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard, source_paths
from infrastructure.storage.les_v7_training import MAX_DATASET_BYTES, read_json_bounded


SCHEMA = "les-v7-fr3-recomputed-study/v1"
FORECAST_SCHEMA = "les-v7-fr3-recomputed-forecasts/v1"
HOLDOUT_POSTURE = "VIEWED_HOLDOUT_RETROSPECTIVE_NOT_NEW_CONFIRMATION"


def _read(path):
    return read_json_bounded(Path(path), MAX_DATASET_BYTES)


def _new_bundle(outcome):
    """Resolve only the new native model, never a parent-model fallback."""
    native = outcome["native_outcome"]
    relative = Path(native["bundle_path"])
    root = Path(outcome["native_root"]).resolve()
    path = (root / relative).resolve()
    if (relative.is_absolute() or ".." in relative.parts
            or not path.is_relative_to(root / "models")
            or path.name != "frozen_bundle.json"):
        raise ValueError("FR3_STUDY_NATIVE_BUNDLE_PATH_INVALID")
    return path


def _extract_recomputed(report, artifact, outcome, config):
    """Extract only verified numbers; do not rename legacy FR2 arm failures.

    ``reconstruct_archive`` has already required complete native baseline and
    full-model holdout metric/hash equality. These checks bind its two detached
    outputs to the exact NEW model/config/dataset and retain that requirement.
    """
    reference = require_policy(config["reference_policy"])
    versions = fr3_reference_versions(reference)
    bindings = report["source_bindings"]
    native = outcome["native_outcome"]
    if (bindings != artifact.get("source_bindings")
            or bindings.get("dataset_id") != native["dataset_id"]
            or bindings.get("config_id") != native["config_id"]
            or bindings.get("bundle_sha256") != native["bundle_sha256"]):
        raise ValueError("FR3_STUDY_RECONSTRUCTION_SOURCE_BINDING_MISMATCH")
    reproduced = report["forecast_reproduction"]
    if (reproduced.get("baseline_complete_metric_and_hash_equality") is not True
            or reproduced.get("fractal_complete_metric_and_hash_equality") is not True
            or reproduced.get("new_head_fits") != 0):
        raise ValueError("FR3_STUDY_COMPLETE_NATIVE_REPRODUCTION_REQUIRED")
    left, right = artifact["M0"], artifact["M1"]
    if not isinstance(left, list) or not left or not isinstance(right, list) or len(left) != len(right):
        raise ValueError("FR3_STUDY_MATCHED_NONEMPTY_FORECASTS_REQUIRED")
    ids = [row["row_id"] for row in left]
    if len(ids) != len(set(ids)) or report["counts"]["paired_forecast_events"] != len(ids):
        raise ValueError("FR3_STUDY_FORECAST_EVENT_COVERAGE_MISMATCH")
    for baseline, fractal in zip(left, right):
        if ({k: v for k, v in baseline.items() if k != "forecast"}
                != {k: v for k, v in fractal.items() if k != "forecast"}):
            raise ValueError("FR3_STUDY_FORECAST_CONDITIONS_OR_LABELS_DIFFER")
    comparisons = report["forecast_comparisons"]
    if comparisons.get("P0") != comparisons.get("P1") or not comparisons.get("P0"):
        raise ValueError("FR3_STUDY_NATIVE_PAIRED_METRICS_REQUIRED")
    costs, rows = report["reference_path_diagnostic"], artifact["reference_cost_rows"]
    if (costs.get("cost_path_policy") != reference["cost_path_policy"]
            or costs["cost_path_policy"].get("version") != versions["path_version"]
            or costs.get("execution_rows_id") != identity(rows)
            or len(rows) != len(ids) or {row["row_id"] for row in rows} != set(ids)):
        raise ValueError("FR3_STUDY_COST_PLAN_OR_EVENT_BINDING_MISMATCH")
    completed = [row for row in rows if row["filled"]]
    pnls = [Decimal(row["realized_net_pnl"]) for row in completed]
    if (costs["completed_reference_paths"] != len(completed)
            or costs["unfilled_reference_paths"] != len(rows) - len(completed)
            or Decimal(costs["realized_net_pnl_twd_sum"]) != sum(pnls, Decimal(0))):
        raise ValueError("FR3_STUDY_REFERENCE_COST_SUMMARY_MISMATCH")
    shared = dict(**versions,
        holdout_interpretation=HOLDOUT_POSTURE, fresh_confirmatory=False,
        source_bindings=deepcopy(bindings), research_only=True, approved_for_live=False)
    forecast = dict(schema="les-v7-fr3-matched-holdout-recomputation/v1", **shared,
        counts=deepcopy(report["counts"]), sessions=list(report["sessions"]),
        metrics=deepcopy(comparisons["P0"]),
        complete_native_reproduction=deepcopy(reproduced),
        after_cost_account_status="NOT_ASSESSABLE",
        after_cost_account_reason="FOUR_SOURCE_BOUND_EXECUTED_ACCOUNTS_REQUIRED")
    cost_summary = dict(deepcopy(costs), **{k: v for k, v in shared.items() if k not in costs})
    cost_summary.update(scope="NATIVE_FR3_REFERENCE_PATHS_NOT_ALLOCATION_OR_ACCOUNT_PERFORMANCE",
        account_performance_status="NOT_ASSESSABLE")
    forecasts = dict(schema=FORECAST_SCHEMA, **shared, M0=deepcopy(left), M1=deepcopy(right),
        reference_cost_rows=deepcopy(rows),
        interpretation="RECOMPUTED_NEW_FR3_LABELS_NOT_OLD_FR2_ADMISSION_ARMS")
    return forecast, cost_summary, forecasts


def run_fr3_study(original_archive, training_root, source_audit, output_root):
    """Re-execute a completed new archive, then publish completion last.

    Existing files are never overwritten. A retry re-executes native research;
    it does not mistake saved output for a new reproduction. A NOT_READY fit
    retains its actual gaps without calling any prediction reconstruction.
    """
    original, training, output = (Path(value).expanduser().resolve()
                                  for value in (original_archive, training_root, output_root))
    if output.is_relative_to(original.parent) or output.is_relative_to(training):
        raise ValueError("FR3_STUDY_SEPARATE_OUTPUT_REQUIRED")
    # This example can be added while a native training job is running. Its
    # own effective bytes are retained separately from the native source zip.
    runner = Path(__file__).resolve()
    if any(path.resolve() == runner for path in source_paths()):
        raise ValueError("FR3_STUDY_RUNNER_MUST_NOT_CHANGE_NATIVE_SOURCE_IDENTITY")
    runner_bytes = runner.read_bytes()
    guard = ResearchVersionGuard()
    objects = ResearchObjects(output / "objects")
    runtime = dict(versions=asdict(guard.versions), objects=guard.retain(objects))
    runtime["runner"] = dict(name="examples/compare_fr3_study.py",
        sha256=hashlib.sha256(runner_bytes).hexdigest(), object_id=objects.put(runner_bytes),
        native_source_zip_excludes_runner=True)
    audit = _read(source_audit) if isinstance(source_audit, (str, Path)) else deepcopy(source_audit)
    outcome = verify_fr3_scope_training(training)
    request = _read(training / "fr3-training-request.json")
    if request.get("sha256") != outcome["request_id"] or identity(request.get("payload")) != request["sha256"]:
        raise ValueError("FR3_STUDY_TRAINING_REQUEST_MISMATCH")
    config = request["payload"]["training_config"]
    reference = require_policy(config["reference_policy"])
    versions = fr3_reference_versions(reference)
    forecast = costs = forecasts = None
    if outcome["status"] == "READY_TO_FREEZE":
        native = outcome["native_outcome"]
        if not native.get("bundle_path") or not native.get("bundle_sha256"):
            raise ValueError("FR3_STUDY_READY_MODEL_ARTIFACT_MISSING")
        reconstruction, numeric_rows = reconstruct_archive(_new_bundle(outcome))
        forecast, costs, forecasts = _extract_recomputed(reconstruction, numeric_rows, outcome, config)
        # The exact helper output is retained for forensic traceability only;
        # its legacy FR2 admission arms are not promoted to FR3 evidence.
        runtime["objects"]["native_reconstruction_diagnostic"] = objects.put_json(reconstruction)
    elif outcome["native_outcome"].get("bundle_path") or outcome["native_outcome"].get("bundle_sha256"):
        raise ValueError("FR3_STUDY_NOT_READY_MUST_NOT_USE_FALLBACK_BUNDLE")
    comparison = compare_fr3_research(original, training, source_audit=audit,
        forecast_comparison=forecast, reference_costs=costs)
    if (comparison.get("universe") != outcome["symbols"]
            or comparison.get("fixed_universe_count") != len(outcome["symbols"])):
        raise ValueError("FR3_STUDY_FULL_FROZEN_SCOPE_NOT_RETAINED")
    if comparison.get("status") == "PASS":
        raise ValueError("FR3_STUDY_MISSING_FOUR_ACCOUNT_PATHS_CANNOT_PASS")
    runtime["objects"].update(source_audit=objects.put_json(audit),
        training_request=objects.put_json(request), training_completion=objects.put_json(outcome))
    if forecasts is not None:
        runtime["objects"]["recomputed_forecasts"] = objects.put_json(forecasts)
    report = dict(schema=SCHEMA,
        status="NOT_READY" if outcome["status"] != "READY_TO_FREEZE" else "NOT_ASSESSABLE",
        native_training_status=outcome["status"],
        native_training_reason_codes=list(outcome["native_outcome"].get("reason_codes", [])),
        full_frozen_universe=list(outcome["symbols"]), fixed_universe_count=len(outcome["symbols"]),
        **versions,
        holdout_interpretation=HOLDOUT_POSTURE, fresh_confirmatory=False,
        matched_forecast_recomputation=forecast,
        reference_path_cost_diagnostic=costs,
        native_signal_comparison_status="NOT_ASSESSABLE",
        four_account_cost_comparison_status="NOT_ASSESSABLE",
        comparison=comparison, runtime=runtime,
        original_bundle_modified=False, ordinary_daily_modified=False,
        research_only=True, approved_for_live=False, execution_action="NO_ACTION",
        mail_sent=False, order_authority=False)
    report_id = identity(report)
    folder = output / report_id
    guard.check()
    if runner.read_bytes() != runner_bytes:
        raise ValueError("FR3_STUDY_RUNNER_CHANGED_DURING_JOB")
    # Read back every referenced object before publishing success/completion.
    for object_id in runtime["objects"].values():
        objects.get(object_id)
    objects.get(runtime["runner"]["object_id"])
    if forecasts is not None:
        write_once(folder / "recomputed-forecasts.json", forecasts)
    write_once(folder / "study.json", dict(report, study_id=report_id))
    completion = dict(schema="les-v7-fr3-study-completion/v1", study_id=report_id,
        status=report["status"], native_training_status=outcome["status"],
        study_file="study.json", runner_sha256=runtime["runner"]["sha256"],
        research_only=True, approved_for_live=False, mail_sent=False, execution_action="NO_ACTION")
    write_once(folder / "completion.json", completion)
    return folder, completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True, help="Unchanged original FR2 frozen bundle")
    parser.add_argument("--training-root", type=Path, required=True, help="Completed NEW FR3 wrapper root")
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate write-once study directory")
    args = parser.parse_args(argv)
    try:
        folder, completion = run_fr3_study(args.archive, args.training_root, args.source_audit, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(canonical(dict(completion, output=str(folder))), flush=True)
    return 2 if completion["status"] in {"NOT_READY", "NOT_ASSESSABLE"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
