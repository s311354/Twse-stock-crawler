"""Manual FR3-L2 preparation, capacity, native fit, M0 and matched study.

Only existing public research ports calculate labels, forecasts or costs.
Completion means the requested stages ended, never deployment or profitability.
No network, mail, scheduler, real order or official account port is opened.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.les_v7.contracts import canonical, identity
from examples.run_fr3_manual_chain import lock, require_source_audit
from infrastructure.les_v7_fr3_capacity import audit_fr3_prepared_capacity, retain_fr3_capacity
from infrastructure.les_v7_fr3_training import prepare_fr3_scope, train_fr3_scope, verify_fr3_scope_training
from infrastructure.les_v7_fr2_m0 import publish_m0
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.les_v7_training import MAX_CONFIG_BYTES, MAX_DATASET_BYTES, read_json_bounded

SCHEMA = "les-v7-fr3-l2-manual-workflow/v1"


def _file_id(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def run(freeze_path, build_path, config_path, original_archive, source_audit, output_root):
    from domain.les_v7.fr3_l2 import PLAN_VERSION
    from examples.compare_fr3_study import run_fr3_study

    freeze, build_file, config_file, original, audit_file, root = [Path(p).expanduser().resolve()
        for p in (freeze_path, build_path, config_path, original_archive, source_audit, output_root)]
    if any(root == p or root.is_relative_to(p) for p in (original.parent, freeze.parent.parent.parent)):
        raise ValueError("FR3_L2_SEPARATE_OUTPUT_REQUIRED")
    build = read_json_bounded(build_file, MAX_CONFIG_BYTES)
    config = read_json_bounded(config_file, MAX_CONFIG_BYTES)
    if config["reference_policy"].get("version") != PLAN_VERSION:
        raise ValueError("FR3_L2_EXPLICIT_SETTINGS_REQUIRED")
    audit = read_json_bounded(audit_file, MAX_DATASET_BYTES)
    files = {str(p): _file_id(p) for p in (freeze, build_file, config_file, original, audit_file, Path(__file__))}
    guard = ResearchVersionGuard()
    objects = ResearchObjects(root / "objects")
    training = root / "training"
    with lock(root / ".workflow.lock"):
        pinned = guard.retain(objects)
        request = dict(schema=SCHEMA, files=files, build_config_id=identity(build),
            training_config_id=identity(config), source_audit_id=audit.get("audit_id"),
            versions=asdict(guard.versions), pinned_objects=pinned,
            research_only=True, approved_for_live=False, execution_action="NO_ACTION")
        write_once(root / "request.json", request)
        attempts = root / "attempts"
        number = 1 + max([int(p.name) for p in attempts.glob("[0-9]*") if p.is_dir()] or [0])
        attempt = attempts / f"{number:06d}"
        attempt.mkdir(parents=True)
        stages = []

        def check():
            guard.check()
            if any(_file_id(p) != sha for p, sha in files.items()):
                raise ValueError("FR3_L2_BOUND_INPUT_OR_RUNNER_CHANGED")

        def progress(value):
            print(canonical(dict(event="native_progress", detail=value)), file=sys.__stdout__, flush=True)

        def stage(name, compute):
            check()
            started, tick = utc_now(), time.perf_counter()
            print(canonical(dict(event="stage_started", stage=name, at=started)), flush=True)
            stdout, stderr = attempt / (name + ".stdout.log"), attempt / (name + ".stderr.log")
            error = value = None
            with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
                with redirect_stdout(out), redirect_stderr(err):
                    try:
                        value = compute()
                        check()
                    except Exception as caught:
                        traceback.print_exc()
                        error = caught
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            receipt = dict(stage=name, started_at=started, finished_at=utc_now(),
                seconds=time.perf_counter()-tick, exit_code=1 if error else 0,
                exit_semantics="DIRECT_PUBLIC_API_NOT_SUBPROCESS_EXIT",
                stdout_sha256=_file_id(stdout), stderr_sha256=_file_id(stderr),
                process_peak_rss_bytes=peak if sys.platform == "darwin" else peak * 1024,
                memory_scope="PROCESS_HIGH_WATER_MARK_NOT_STAGE_DELTA")
            if error:
                receipt.update(status="ERROR", error_type=type(error).__name__, error=str(error))
            else:
                receipt.update(status="RETURNED", result_id=objects.put_json(value),
                    business_status=value.get("status", value.get("completion", {}).get("status")))
            write_once(attempt / (name + ".json"), receipt)
            stages.append(dict(file=name + ".json", sha256=identity(receipt)))
            print(canonical(dict(event="stage_finished", **receipt)), flush=True)
            if error:
                raise error
            return value

        def finish(status, capacity, outcome=None, m0=None, study=None, m0_gap=None):
            check()
            result = dict(schema=SCHEMA, request_id=identity(request), status=status,
                capacity_id=capacity["capacity_id"], dataset_id=capacity["dataset_id"],
                native_status=outcome.get("status") if outcome else None,
                native_counts=outcome.get("native_outcome", {}).get("counts") if outcome else None,
                m0_model_id=m0.get("model_id") if m0 else None, m0_gap=m0_gap,
                study_id=study["completion"]["study_id"] if study else None,
                after_cost_accounts="NOT_ASSESSABLE", fresh_confirmatory=False,
                original_bundle_modified=False, ordinary_daily_modified=False,
                research_only=True, approved_for_live=False, execution_action="NO_ACTION", mail_sent=False)
            write_once(root / "completion.json", result)
            write_once(attempt / "completion.json", dict(result_id=identity(result), stages=stages))
            return result

        stage("prepare", lambda: prepare_fr3_scope(freeze, build, config, training, progress=progress))
        dataset = read_json_bounded(training / "native/scope-training-dataset.json", MAX_DATASET_BYTES)
        require_source_audit(audit, dataset["payload"], config)

        def capacity_call():
            value = audit_fr3_prepared_capacity(training)
            retain_fr3_capacity(value, root / "capacity")
            return value

        capacity = stage("capacity", capacity_call)
        if not capacity["count_capacity_allows_fit"]:
            return finish("STOPPED_INSUFFICIENT_CAPACITY_NO_FIT", capacity)
        stage("fit", lambda: train_fr3_scope(freeze, build, config, training, progress=progress))
        outcome = stage("verify", lambda: verify_fr3_scope_training(training))
        if outcome["status"] != "READY_TO_FREEZE":
            return finish("STOPPED_NATIVE_NOT_READY", capacity, outcome)
        native = outcome["native_outcome"]
        relative = Path(native["bundle_path"])
        bundle = (training / "native" / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not bundle.is_relative_to(training / "native/models"):
            raise ValueError("FR3_L2_NATIVE_BUNDLE_PATH_INVALID")
        # A genuine M0 readiness gap must not erase the completed M1 or invent
        # substitute weights. Other corruption/control failures still raise.
        m0 = m0_gap = None
        try:
            m0 = stage("m0", lambda: publish_m0(bundle, root / "m0"))
        except ValueError as error:
            if str(error) not in {"FR2_M0_INSUFFICIENT_OWN_PAST_CONTEXT_WEIGHT_ROWS", "FR2_M0_WEIGHT_SOLVER_NOT_CONVERGED"}:
                raise
            m0_gap = str(error)

        def study_call():
            folder, receipt = run_fr3_study(original, training, audit, root / "study")
            return dict(directory=str(folder), completion=receipt)

        study = stage("study", study_call)
        return finish("RESEARCH_COMPLETE_NOT_PORTFOLIO_ACCEPTANCE", capacity, outcome, m0, study, m0_gap)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "build-config", "training-config", "original-archive", "source-audit", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    value = run(args.freeze, args.build_config, args.training_config,
        args.original_archive, args.source_audit, args.output_root)
    print(canonical(value), flush=True)
    return 2  # Research completion is not deployment or economic acceptance.


if __name__ == "__main__":
    raise SystemExit(main())
