"""One manually invoked FR3 continuation; no waiting, scheduling or prepare.

This example requires the parent's retained real prepare exit observation.
It only dispatches existing public APIs; it owns no model or account math.
Stage exit codes describe direct-API outcomes, not subprocess exit codes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.les_v7.contracts import canonical, identity, timestamp
from examples import compare_fr3_study as study
from infrastructure.les_v7_fr2_m0 import publish_m0
from infrastructure.les_v7_fr3_capacity import audit_fr3_prepared_capacity, retain_fr3_capacity
from infrastructure.les_v7_fr3_comparison import AUDIT_SCHEMA
from infrastructure.les_v7_fr3_training import REQUEST_SCHEMA, train_fr3_scope, verify_fr3_scope_training
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard, source_paths
from infrastructure.storage.les_v7_training import MAX_CONFIG_BYTES, MAX_DATASET_BYTES

SCHEMA = "les-v7-fr3-manual-chain/v1"
EXIT_SCHEMA = "les-v7-fr3-prepare-exit-observation/v1"


def file_id(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def lock(path, *, existing=False):
    if not existing:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb" if existing else "a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("FR3_MANUAL_CHAIN_BUSY") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def require_prepare_exit(record, request, dataset, native):
    body = record.get("payload", {})
    if (not isinstance(body, dict) or record.get("sha256") != identity(body) or body.get("schema") != EXIT_SCHEMA
            or body.get("request_id") != request["sha256"]
            or body.get("dataset_id") != dataset["sha256"]
            or body.get("pinned_objects") != dataset["payload"]["pinned_objects"]
            or not isinstance(body.get("source_session_id"), str) or not body["source_session_id"]
            or type(body.get("pid")) is not int or body["pid"] <= 0):
        raise ValueError("FR3_MANUAL_PREPARE_EXIT_BINDING_REQUIRED")
    timestamp(body["observed_at"])
    result = body.get("exit_result", {})
    if not isinstance(result, dict) or type(result.get("exit_code")) is not int or result["exit_code"] != 0:
        raise ValueError("FR3_MANUAL_REAL_PREPARE_EXIT_ZERO_REQUIRED")
    # Native summary may precede /usr/bin/time's trailer, or arrive in a
    # previous poll. Preserve raw tool objects; find exactly one prepare JSON.
    try:
        observations = [result] + ([body["stdout_result"]] if "stdout_result" in body else [])
        summaries = []
        for observation in observations:
            if not isinstance(observation, dict) or not isinstance(observation["output"], str):
                raise TypeError("Original tool output must be text")
            for line in observation["output"].splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("command") == "prepare":
                    summaries.append(item)
        if len(summaries) != 1:
            raise ValueError("Exactly one original prepare summary is required")
        summary = summaries[0]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("FR3_MANUAL_PREPARE_FINAL_SUMMARY_REQUIRED") from error
    if (not isinstance(summary, dict) or summary.get("command") != "prepare" or summary.get("status") != "DATASET_PREPARED"
            or summary.get("content_id") != request["sha256"]):
        raise ValueError("FR3_MANUAL_PREPARE_FINAL_SUMMARY_REQUIRED")
    try:
        os.kill(body["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("FR3_MANUAL_PREPARE_PROCESS_STILL_PRESENT")
    # A manifest can exist before the final native source checks finish.
    # This probe never creates/truncates the original native lock file.
    with lock(native / ".scope-training.lock", existing=True):
        pass


def require_source_audit(audit, dataset, config):
    if (not isinstance(audit, dict) or audit.get("schema") != AUDIT_SCHEMA
            or audit.get("audit_id") != identity({k: v for k, v in audit.items() if k != "audit_id"})):
        raise ValueError("FR3_MANUAL_SOURCE_AUDIT_HASH_OR_SCHEMA_INVALID")
    try:
        matches = (audit.get("sources_verified") is True
            and all(audit[key] == dataset[key] for key in
                    ("freeze_id", "scope_id", "dataset_kind", "source_evidence_ids"))
            and timestamp(audit["evaluation_cutoff"]) == timestamp(config["evaluation_cutoff"])
            and audit["universe"] == [row[0] for row in dataset["universe"]]
            and isinstance(audit["previously_inspected_sessions"], list))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("FR3_MANUAL_SOURCE_AUDIT_NATIVE_BINDING_MISMATCH") from error
    if not matches:
        raise ValueError("FR3_MANUAL_SOURCE_AUDIT_NATIVE_BINDING_MISMATCH")


def read_input(path, limit):
    """Parse and bind the same bounded bytes, never a later file reread."""
    if path.stat().st_size > limit:
        raise ValueError("FR3_MANUAL_INPUT_SIZE_LIMIT")
    with path.open("rb") as stream:
        encoded = stream.read(limit + 1)
    if len(encoded) > limit:
        raise ValueError("FR3_MANUAL_INPUT_SIZE_LIMIT")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError("FR3_MANUAL_INPUT_JSON_OBJECT_REQUIRED")
    identity(value)
    return encoded, value


def run_manual_chain(training_root, original_archive, prepare_exit, source_audit, output_root):
    training, original, observed, audit_path, output = [Path(p).expanduser().resolve() for p in
        (training_root, original_archive, prepare_exit, source_audit, output_root)]
    request_path = training / "fr3-training-request.json"
    dataset_path = training / "native" / "scope-training-dataset.json"
    inputs = {path: read_input(path, limit) for path, limit in (
        (request_path, 2 * MAX_CONFIG_BYTES + 64 * 1024), (dataset_path, MAX_DATASET_BYTES),
        (observed, 2 * MAX_CONFIG_BYTES + 64 * 1024), (audit_path, MAX_DATASET_BYTES))}
    request, dataset, observation, audit = [inputs[p][1] for p in (request_path, dataset_path, observed, audit_path)]
    body = request.get("payload", {})
    if (not isinstance(body, dict) or request.get("sha256") != identity(body) or body.get("schema") != REQUEST_SCHEMA
            or dataset.get("sha256") != identity(dataset.get("payload"))
            or dataset["payload"].get("freeze_id") != body.get("freeze_id")
            or dataset["payload"].get("training_config_id") != identity(body["training_config"])
            or dataset["payload"].get("build_config_id") != identity(body["build_config"])):
        raise ValueError("FR3_MANUAL_NATIVE_INPUT_BINDING_MISMATCH")
    require_source_audit(audit, dataset["payload"], body["training_config"])
    if any(output.is_relative_to(root) for root in
           (training, original.parent, Path(dataset["payload"]["source_root"]).resolve())):
        raise ValueError("FR3_MANUAL_SEPARATE_RESEARCH_OUTPUT_REQUIRED")
    require_prepare_exit(observation, request, dataset, training / "native")
    runner, study_runner = Path(__file__).resolve(), Path(study.__file__).resolve()
    if runner in [p.resolve() for p in source_paths()]:
        raise ValueError("FR3_MANUAL_RUNNER_MUST_NOT_CHANGE_NATIVE_SOURCE_IDENTITY")
    bound_files = {str(p): hashlib.sha256(encoded).hexdigest() for p, (encoded, _) in inputs.items()}
    bound_files.update({str(p): file_id(p) for p in (original, runner, study_runner)})
    guard = ResearchVersionGuard()
    objects = ResearchObjects(output / "objects")
    with lock(output / ".manual-chain.lock"):
        pinned = guard.retain(objects)
        if pinned != dataset["payload"]["pinned_objects"]:
            raise ValueError("FR3_MANUAL_PREPARED_VERSION_DRIFT")
        runtime = dict(versions=asdict(guard.versions), pinned_objects=pinned,
            runner=objects.put(runner.read_bytes()), study_runner=objects.put(study_runner.read_bytes()),
            source_audit=objects.put(inputs[audit_path][0]),
            input_objects={str(p): objects.put(encoded) for p, (encoded, _) in inputs.items()})
        job = dict(schema=SCHEMA, request_id=request["sha256"], dataset_id=dataset["sha256"],
            source_audit_id=audit["audit_id"], source_audit_path=str(audit_path),
            files=bound_files, runtime=runtime, training_root=str(training), original_archive=str(original),
            execution_action="NO_ACTION", research_only=True, approved_for_live=False)
        write_once(output / "request.json", job)
        attempt_root = output / "attempts"
        number = 1 + max([int(p.name) for p in attempt_root.glob("[0-9]*") if p.is_dir()] or [0])
        attempt = attempt_root / f"{number:06d}"
        attempt.mkdir(parents=True)
        stages = []

        def check():
            guard.check()
            if any(file_id(path) != expected for path, expected in bound_files.items()):
                raise ValueError("FR3_MANUAL_BOUND_INPUT_OR_RUNNER_CHANGED")

        def stage(name, compute):
            check()
            started, tick = utc_now(), time.perf_counter()
            stdout, stderr = attempt / (name + ".stdout.log"), attempt / (name + ".stderr.log")
            result, failure = None, None
            with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
                with redirect_stdout(out), redirect_stderr(err):
                    try:
                        result = compute()
                        check()
                    except Exception as error:
                        traceback.print_exc()
                        failure = error
            receipt = dict(stage=name, started_at=started, finished_at=utc_now(),
                seconds=time.perf_counter()-tick, exit_code=1 if failure else 0,
                exit_semantics="DIRECT_PUBLIC_API_OUTCOME_NOT_SUBPROCESS_EXIT",
                stdout_sha256=file_id(stdout), stderr_sha256=file_id(stderr))
            if failure:
                receipt.update(status="ERROR", error_type=type(failure).__name__, error=str(failure))
            else:
                receipt.update(status="RETURNED", result_id=objects.put_json(result),
                    business_status=result.get("status", result.get("completion", {}).get("status")))
            write_once(attempt / (name + ".json"), receipt)
            stages.append(dict(file=name + ".json", sha256=identity(receipt)))
            if failure:
                raise failure
            return result

        def finish(status, capacity, native=None, m0=None, study_result=None):
            check()
            outcome = dict(schema=SCHEMA, workflow_request_id=identity(job), status=status,
                source_audit_id=audit["audit_id"],
                capacity_id=capacity["capacity_id"], native_outcome_id=identity(native) if native else None,
                published_m0_id=m0.get("model_id") if m0 else None,
                study_id=study_result["completion"]["study_id"] if study_result else None,
                after_cost_four_account_status="NOT_ASSESSABLE", fresh_confirmatory=False,
                execution_action="NO_ACTION", mail_sent=False, research_only=True, approved_for_live=False)
            write_once(output / "completion.json", outcome)
            write_once(attempt / "completion.json", dict(workflow_completion_id=identity(outcome), stages=stages))
            return outcome

        def capacity_call():
            value = audit_fr3_prepared_capacity(training)
            retain_fr3_capacity(value, output / "capacity")
            return value
        capacity = stage("capacity", capacity_call)
        if not capacity["count_capacity_allows_fit"]:
            return finish("STOPPED_INSUFFICIENT_CAPACITY_NO_FIT", capacity)
        if not (training / "fr3-training-completion.json").exists():
            # Same public dispatcher as CLI fit-prepared, after its necessary
            # capacity gate. Native persistence owns incomplete-fit recovery.
            stage("fit-prepared", lambda: train_fr3_scope(body["freeze_path"], body["build_config"],
                body["training_config"], training, availability_policy=body["availability_policy"]))
        native = stage("verify", lambda: verify_fr3_scope_training(training))
        if native["status"] != "READY_TO_FREEZE":
            return finish("STOPPED_NATIVE_NOT_READY", capacity, native)
        relative = Path(native["native_outcome"]["bundle_path"])
        bundle = (training / "native" / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not bundle.is_relative_to(training / "native" / "models"):
            raise ValueError("FR3_MANUAL_NATIVE_BUNDLE_PATH_INVALID")
        m0 = stage("m0", lambda: publish_m0(bundle, output / "m0"))
        def study_call():
            folder, receipt = study.run_fr3_study(original, training, audit, output / "study")
            return dict(directory=str(folder), completion=receipt)
        result = stage("study", study_call)
        return finish("CHAIN_FINISHED_RESEARCH_NOT_ASSESSABLE", capacity, native, m0, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("training-root", "original-archive", "prepare-exit", "source-audit", "output-root"):
        parser.add_argument("--"+option, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = run_manual_chain(args.training_root, args.original_archive, args.prepare_exit, args.source_audit, args.output_root)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(canonical(value), flush=True)
    return 2  # Completion is not predictive/account acceptance or deployment.


if __name__ == "__main__":
    raise SystemExit(main())
