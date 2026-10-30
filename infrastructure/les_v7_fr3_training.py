"""Opt-in FR3 native scope training; never relabel an old H20 prepared archive.

This adapter pins the approved R6 quality and source-bound conditional-limit
reference contract, then delegates source building, labels, purged OOF, the five
heads, calibration, same-row M0 ablation and immutable model publication to the
existing native pipeline.  It has no daily configuration, mail or account port.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import resource
import sys
import time

from application.les_v7_dataset import build_training_dataset, validate_build_config
from domain.les_v7.contracts import identity
from domain.les_v7.quality_policy import r6_quality_policy
from domain.les_v7.reference_memo import MAX_REFERENCE_ENTRIES, NativeReferenceMemo
from domain.les_v7.training import prepare_events, reference_config_id, validate_training_config
from infrastructure.les_v7_scope_training import (
    inspect_elec_scope_freeze, prepare_elec_scope_dataset, train_elec_scope,
    verify_elec_scope_training,
)
from infrastructure.les_v7_training_source import FINAL_CLOSE, load_training_source
from infrastructure.research_archive import write_once
from infrastructure.research_objects import ResearchVersionGuard
from infrastructure.storage.les_v7_training import MAX_CONFIG_BYTES, MAX_DATASET_BYTES, read_json_bounded

REQUEST_SCHEMA = "les-v7-fr3-training-request/v1"
COMPLETION_SCHEMA = "les-v7-fr3-training-completion/v1"
INTERPRETATION = "RETROSPECTIVE_FIXED_RULES_NOT_DEPLOYMENT"
HOLDOUT_INTERPRETATION = "VIEWED_HOLDOUT_RETROSPECTIVE"


def validate_fr3_training_configs(build_config, training_config):
    """One exact plan/label/cost identity for build, fit and future inference.

    Financial valuation is a separate research-admission component.  It is not
    an input to the technical predictive head fit and cannot create labels.
    """
    from domain.les_v7.fr3_plan import (LABEL_VERSION, PATH_VERSION, PLAN_VERSION,
        is_fr3_reference, validate_policy)
    validate_build_config(build_config)
    validate_training_config(training_config)
    plan = training_config["reference_policy"]
    validate_policy(plan)
    if plan.get("version") != PLAN_VERSION:
        from domain.les_v7.fr3_l2 import PLAN_VERSION, PATH_VERSION, LABEL_VERSION
    if (not is_fr3_reference(plan) or plan.get("version") != PLAN_VERSION
            or plan.get("cost_path_policy", {}).get("version") != PATH_VERSION
            or plan.get("cost_path_policy", {}).get("label_version") != LABEL_VERSION
            or training_config.get("quality_policy") != r6_quality_policy()
            or build_config["seed"].get("reference_policy") != plan
            or build_config["path_policy"] != plan["cost_path_policy"]
            or reference_config_id(build_config["seed"]) != reference_config_id(training_config)
            or build_config["evaluation_cutoff"] != training_config["evaluation_cutoff"]
            or build_config.get("reference_evidence_policy") != training_config.get("reference_evidence_policy")):
        raise ValueError("FR3_TRAINING_COMPONENT_BINDING_MISMATCH")
    return dict(plan_version=PLAN_VERSION, path_version=PATH_VERSION, label_version=LABEL_VERSION,
        quality_policy_id=training_config["quality_policy"]["policy_id"],
        quality_policy_sha256=identity(training_config["quality_policy"]),
        reference_config_id=reference_config_id(training_config),
        source_interpretation=INTERPRETATION, holdout_interpretation=HOLDOUT_INTERPRETATION)


def _request(freeze_path, build_config, training_config, output_root, *, availability_policy,
             allow_synthetic):
    root = Path(output_root)
    if root.exists() and any(root.iterdir()) and not (root / "fr3-training-request.json").is_file():
        raise ValueError("FR3_NEW_OUTPUT_ROOT_REQUIRED_NO_OLD_ARCHIVE_REBASE")
    components = validate_fr3_training_configs(build_config, training_config)
    scope = inspect_elec_scope_freeze(freeze_path, verify_sources=False)
    if scope["dataset_kind"] == "SYNTHETIC_TEST_ONLY" and allow_synthetic is not True:
        raise ValueError("FR3_SYNTHETIC_SCOPE_REQUIRES_EXPLICIT_OPT_IN")
    body = dict(schema=REQUEST_SCHEMA, freeze_path=str(Path(freeze_path).resolve()),
        freeze_id=scope["freeze_id"], scope_id=scope["scope_id"], symbols=scope["symbols"],
        dataset_kind=scope["dataset_kind"], availability_policy=availability_policy,
        build_config=deepcopy(build_config), training_config=deepcopy(training_config),
        components=components, research_only=True, approved_for_live=False,
        execution_action="NO_ACTION", original_bundle_modified=False,
        prepared_policy="REBUILD_FROM_FROZEN_RAW_SOURCES_NEVER_REUSE_OLD_PLAN_LABELS")
    record = dict(payload=body, sha256=identity(body))
    write_once(root / "fr3-training-request.json", record)
    return record


def prepare_fr3_scope(freeze_path, build_config, training_config, output_root, *,
                      availability_policy=FINAL_CLOSE, allow_synthetic=False, **native_options):
    """Prepare every frozen symbol through the native builder, without fitting."""
    build_config, training_config = deepcopy(build_config), deepcopy(training_config)
    root = Path(output_root).resolve()
    request = _request(freeze_path, build_config, training_config, root,
        availability_policy=availability_policy, allow_synthetic=allow_synthetic)
    native = prepare_elec_scope_dataset(freeze_path, build_config, training_config, root / "native",
        availability_policy=availability_policy, allow_synthetic=allow_synthetic, **native_options)
    return dict(schema="les-v7-fr3-scope-preparation/v1", request_id=request["sha256"],
        native_root=str(root / "native"), native_preparation=native,
        components=deepcopy(request["payload"]["components"]),
        status=native["status"], model_fitted=False, research_only=True, approved_for_live=False)


def train_fr3_scope(freeze_path, build_config, training_config, output_root, *,
                    availability_policy=FINAL_CLOSE, allow_synthetic=False, **native_options):
    """Fit a new write-once native research archive; do not activate it.

    The native engine owns fold maturity, both binary classes, context sample
    minima, OOF-only calibration, numerical acceptance and recovery guards.
    NOT_READY is retained as an auditable result, never replaced by old heads.
    """
    if "prepare_only" in native_options:
        raise ValueError("FR3_TRAINING_PREPARE_ONLY_USE_PUBLIC_PREPARATION_ENTRY")
    build_config, training_config = deepcopy(build_config), deepcopy(training_config)
    root = Path(output_root).resolve()
    request = _request(freeze_path, build_config, training_config, root,
        availability_policy=availability_policy, allow_synthetic=allow_synthetic)
    native = train_elec_scope(freeze_path, build_config, training_config, root / "native",
        availability_policy=availability_policy, allow_synthetic=allow_synthetic, **native_options)
    outcome = dict(schema=COMPLETION_SCHEMA, request_id=request["sha256"],
        native_root=str(root / "native"), native_outcome=native, native_outcome_sha256=identity(native),
        components=deepcopy(request["payload"]["components"]), symbols=list(native["symbols"]),
        status=native["status"], source_interpretation=INTERPRETATION,
        holdout_interpretation=HOLDOUT_INTERPRETATION,
        freeze_meaning="NATIVE_ENGINEERING_ACCEPTANCE_NOT_PREDICTIVE_OR_PORTFOLIO_GAIN",
        predictive_gain_validated=False, portfolio_gain_validated=False,
        research_only=True, approved_for_live=False, execution_action="NO_ACTION",
        original_bundle_modified=False, ordinary_daily_modified=False)
    write_once(root / "fr3-training-completion.json", outcome)
    return outcome


def verify_fr3_scope_training(output_root):
    """Reverify the FR3 request plus all native sources, child labels and model."""
    root = Path(output_root).resolve()
    request = read_json_bounded(root / "fr3-training-request.json", 2 * MAX_CONFIG_BYTES + 64 * 1024)
    outcome = read_json_bounded(root / "fr3-training-completion.json", MAX_DATASET_BYTES)
    body = request.get("payload", {})
    components = validate_fr3_training_configs(body["build_config"], body["training_config"])
    native = outcome.get("native_outcome", {})
    if (request.get("sha256") != identity(body) or body.get("schema") != REQUEST_SCHEMA
            or outcome.get("schema") != COMPLETION_SCHEMA
            or outcome.get("request_id") != request["sha256"]
            or body.get("components") != components or outcome.get("components") != components
            or outcome.get("native_outcome_sha256") != identity(native)
            or outcome.get("native_root") != str(root / "native")
            or outcome.get("symbols") != body.get("symbols") or native.get("symbols") != body.get("symbols")
            or outcome.get("status") != native.get("status")
            or native.get("build_config_id") != identity(body["build_config"])
            or native.get("config_id") != identity(body["training_config"])
            or native.get("freeze_id") != body.get("freeze_id")
            or outcome.get("source_interpretation") != INTERPRETATION
            or outcome.get("holdout_interpretation") != HOLDOUT_INTERPRETATION
            or outcome.get("predictive_gain_validated") is not False
            or outcome.get("portfolio_gain_validated") is not False
            or outcome.get("freeze_meaning") != "NATIVE_ENGINEERING_ACCEPTANCE_NOT_PREDICTIVE_OR_PORTFOLIO_GAIN"
            or outcome.get("research_only") is not True or outcome.get("approved_for_live") is not False
            or outcome.get("execution_action") != "NO_ACTION"
            or outcome.get("original_bundle_modified") is not False
            or outcome.get("ordinary_daily_modified") is not False):
        raise ValueError("FR3_TRAINING_COMPLETION_BINDING_MISMATCH")
    verify_elec_scope_training(root / "native", native)
    return outcome


def preview_fr3_shard(freeze_path, build_config, training_config, *, index=0,
                      availability_policy=FINAL_CLOSE, allow_synthetic=False,
                      guard_factory=ResearchVersionGuard):
    """Read-only native build/prepare smoke; never a fitted model or full fit.

    The preview does not publish or recycle prepared rows.  Full training must
    still rebuild/verify every frozen shard with the final pinned source code.
    Timings and process high-water RSS are diagnostics, not a speed guarantee.
    """
    build_config, training_config = deepcopy(build_config), deepcopy(training_config)
    validate_fr3_training_configs(build_config, training_config)
    guard = guard_factory()
    guard.check()
    scope = inspect_elec_scope_freeze(freeze_path, verify_sources=False)
    if scope["dataset_kind"] == "SYNTHETIC_TEST_ONLY" and allow_synthetic is not True:
        raise ValueError("FR3_SYNTHETIC_SCOPE_REQUIRES_EXPLICIT_OPT_IN")
    freeze = read_json_bounded(freeze_path, MAX_DATASET_BYTES)["body"]
    if type(index) is not int or not 0 <= index < len(freeze["chunks"]):
        raise ValueError("FR3_PREVIEW_SHARD_INDEX_INVALID")
    chunk = freeze["chunks"][index]
    started = time.perf_counter()
    source = load_training_source(Path(scope["source_root"]) / "evidence" / chunk["evidence_id"] / "evidence.json",
        evaluation_cutoff=build_config["evaluation_cutoff"], availability_policy=availability_policy,
        **({"reference_evidence_policy": build_config["reference_evidence_policy"]}
           if build_config.get("reference_evidence_policy") else {}))
    if (source["payload"]["source_evidence_ids"] != [chunk["evidence_id"]]
            or source["payload"]["universe"] != [[symbol, "TWSE"] for symbol in chunk["symbols"]]):
        raise ValueError("FR3_PREVIEW_SOURCE_SCOPE_BINDING_MISMATCH")
    loaded_at = time.perf_counter()
    with NativeReferenceMemo(max_entries=min(build_config["max_events"], MAX_REFERENCE_ENTRIES)):
        built = build_training_dataset(source, build_config)
        built_at = time.perf_counter()
        rows, diagnostics = prepare_events(built["dataset"], training_config)
    guard.check()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(schema="les-v7-fr3-shard-preview/v1", freeze_id=scope["freeze_id"], index=index,
        symbols=list(chunk["symbols"]), source_evidence_id=chunk["evidence_id"],
        build_config_id=identity(build_config), training_config_id=identity(training_config),
        native_counts=deepcopy(built["result"]["counts"]), prepared_count=len(rows), diagnostic_count=len(diagnostics),
        seconds=dict(source_load=loaded_at - started, native_build=built_at - loaded_at,
            prepare=time.perf_counter() - built_at, total=time.perf_counter() - started),
        process_peak_rss_bytes=peak if sys.platform == "darwin" else peak * 1024,
        memory_scope="PROCESS_HIGH_WATER_MARK_NOT_PREVIEW_DELTA",
        rows_reused_for_training=False, model_fitted=False, research_only=True,
        interpretation=INTERPRETATION, approved_for_live=False)
