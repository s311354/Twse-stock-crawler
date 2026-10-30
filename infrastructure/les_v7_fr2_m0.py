"""Publish the missing M0 weights from its own strictly-past native OOF.

The preserved M1 archive is read-only. This is not a refit of its five heads,
not a new history/deployment claim and never a fallback to M1 factor weights.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from domain.les_v7.contracts import CONTEXTS, PRIOR, identity, timestamp
from domain.les_v7.decision import factor_vector
from domain.les_v7.fractals import is_active_fractal_feature
from domain.les_v7.localization import Forecast, FrozenBundle
from domain.les_v7.provenance import archive_provenance, compact_candidate, verify_archived_compaction
from domain.les_v7.research import fit_factor_weights
from domain.les_v7.training import _calibrate, _holdout, _oof
from infrastructure.les_v7_fr2_archive import (_read, _report, _saved_head_provider,
    _verify_baseline_fold_reconstruction)
from infrastructure.les_v7_scope_training import (_unpack_row, _packet_feature_columns,
    MAX_PREPARED_BYTES, MAX_PREPARED_DECODED_BYTES, MAX_PREPARED_ROWS, prepared_decoded_limit)
from infrastructure.storage.les_v7_prepared import inspect_prepared_packet
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.les_v7_training import (LesV7TrainingRepository, MAX_CONFIG_BYTES,
                                                   _job_lock, read_json_bounded)

LEGACY_SCHEMA = "les-v7-fr2-m0-freeze/v1"
SCHEMA = "les-v7-fr2-m0-freeze/v2"


def _original_binding(archive, objects):
    """Keep byte identity distinct from canonical packet and model identities."""
    record = read_json_bounded(archive, MAX_CONFIG_BYTES)
    encoded = Path(archive).read_bytes()
    if len(encoded) > MAX_CONFIG_BYTES or json.loads(encoded) != record:
        raise ValueError("FR2_M0_ORIGINAL_BUNDLE_CHANGED_DURING_READ")
    if record.get("sha256") != identity(record.get("payload")):
        raise ValueError("FR2_M0_ORIGINAL_BUNDLE_HASH_MISMATCH")
    return dict(original_bundle_sha256=record["sha256"], original_bundle_packet_id=identity(record),
        original_bundle_file_sha256=objects.put(encoded))


def weight_rows_from_oof(rows, predictions, config):
    """The original trainer's Stage C, using M0's own calibrated predictions.

    Every row is still checked by ``fit_factor_weights`` for head/calibration
    and label time isolation. Unfilled/unknown-utility events are not silently
    assigned a zero outcome. The full exclusion census is retained.
    """
    fit_at = timestamp(config["fit_cutoff"])
    by_id = {row["row_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("FR2_M0_DUPLICATE_PREPARED_ROW")
    selected, exclusions, seen = [], Counter(), set()
    for prediction in predictions:
        if prediction["row_id"] in seen:
            raise ValueError("FR2_M0_DUPLICATE_OOF_ROW")
        seen.add(prediction["row_id"])
        row = by_id[prediction["row_id"]]
        if timestamp(prediction["label_available_at"]) > fit_at:
            exclusions["LABEL_NOT_MATURE_AT_WEIGHT_FIT"] += 1
            continue
        if prediction["forecast"]["status"] != "VALID":
            exclusions["NO_STRICTLY_PAST_VALID_CALIBRATOR"] += 1
            continue
        if row["utility"] is None:
            exclusions["NO_REALIZED_FILLED_REFERENCE_UTILITY"] += 1
            continue
        f = prediction["forecast"]
        forecast = Forecast("VALID", tuple(f["quantiles"]), tuple(f["timing"]),
            f["p_fill"], f["p_rebound"], f["p_entry"], calibration_version=f["calibration_version"])
        factors, _ = factor_vector(row["context"], row["features"], forecast, row["plan"],
            leadership_rr_min=config["reference_policy"].get("leadership_rr_min"))
        if factors is None:
            exclusions["FACTOR_VECTOR_NOT_READY"] += 1
            continue
        selected.append(dict(factors=list(factors), utility=row["utility"], context=row["context"],
            prediction_kind="STRICTLY_PAST_OOF", cutoff=row["cutoff"], label_available_at=row["label_available_at"],
            head_fit_cutoff=prediction["head_fit_cutoff"],
            head_last_label_available_at=prediction["head_last_label_available_at"],
            calibration_fit_cutoff=prediction["calibration_fit_cutoff"],
            calibration_last_label_available_at=prediction["calibration_last_label_available_at"],
            observation_weight=row["observation_weight"], row_id=row["row_id"],
            prediction_id=prediction["prediction_id"]))
    counts = dict(Counter(row["context"] for row in selected))
    if (len(selected) < config["min_weight_rows"]
            or any(counts.get(c, 0) < config["min_context_weight_rows"] for c in CONTEXTS)):
        raise ValueError("FR2_M0_INSUFFICIENT_OWN_PAST_CONTEXT_WEIGHT_ROWS")
    return selected, dict(exclusions)


def _prepared(folder, config, dataset):
    root = folder.parent.parent
    rows, total, decoded_total = [], 0, 0
    for child in dataset["payload"]["children"]:
        relative = Path(child["prepared_path"])
        source = (root / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts
                or not source.is_relative_to((root / "prepared").resolve())):
            raise ValueError("FR2_M0_PREPARED_PATH_INVALID")
        prepared = inspect_prepared_packet(source)
        total += prepared.encoded_bytes
        decoded_total += prepared.decoded_bytes
        if total > MAX_PREPARED_BYTES or decoded_total > prepared_decoded_limit(config,
                legacy_limit=MAX_PREPARED_DECODED_BYTES):
            raise ValueError("FR2_M0_PREPARED_BYTE_BUDGET_EXCEEDED")
        packet = prepared.value
        if (identity(packet) != child["prepared_sha256"] or packet["dataset_id"] != child["dataset_id"]
                or packet["config_id"] != identity(config)):
            raise ValueError("FR2_M0_PREPARED_SOURCE_CHANGED")
        columns = _packet_feature_columns(packet)
        rows.extend(_unpack_row(row, columns) for row in packet["rows"])
        if len(rows) > MAX_PREPARED_ROWS:
            raise ValueError("FR2_M0_PREPARED_ROW_BUDGET_EXCEEDED")
        del prepared, packet
    return sorted(rows, key=lambda row: (timestamp(row["cutoff"]), row["symbol"], row["market"]))


def reconstruct_m0(archive):
    """Actual native computation, with exact saved heads and new M0 weights.

    Returns a pending model plus complete proof, never publishes in the source
    archive. Numerical head/calibration/holdout identity is checked before the
    newly fitted weight artifact can be frozen.
    """
    archive = Path(archive).expanduser().resolve()
    if archive.name != "frozen_bundle.json":
        raise ValueError("FR2_M0_NATIVE_FROZEN_BUNDLE_REQUIRED")
    folder = archive.parent
    receipt = LesV7TrainingRepository.verify(folder)
    if receipt.get("dataset_kind") != "RESEARCH" or receipt.get("status") != "READY_TO_FREEZE":
        raise ValueError("FR2_M0_REAL_READY_ARCHIVE_REQUIRED")
    original, config, dataset, fitted = (_read(folder / name) for name in
        ("frozen_bundle.json", "config.json", "dataset.json", "fit_result.json"))
    if (identity(original["payload"]) != original["sha256"]
            or original["sha256"] != receipt["bundle_sha256"]
            or identity(fitted["payload"]) != fitted["sha256"]):
        raise ValueError("FR2_M0_NATIVE_INPUT_HASH_MISMATCH")
    if dataset["payload"].get("schema") != "les-v7-sharded-training-dataset/v1":
        raise ValueError("FR2_M0_NATIVE_SHARDED_DATASET_REQUIRED")
    candidate, fit = original["payload"], fitted["payload"]
    rows = _prepared(folder, config, dataset)
    reports = {rid: _report(folder, rid) for rid in fit["report_references"]}
    ablation = _report(folder, candidate["archive_validation_report_ids"]["ablation"])
    names = [n for n in config["feature_names"] if not is_active_fractal_feature(n)]
    bounds = [b for n, b in zip(config["feature_names"], config["feature_bounds"]) if not is_active_fractal_feature(n)]
    if not names or len(names) == len(config["feature_names"]):
        raise ValueError("FR2_M0_NO_FRACTAL_ABLATION_REQUIRED")
    fold_proof = fit.get("baseline_folds")
    predictions, folds, derived_reports = _oof(rows, config, names, bounds,
        head_provider=_saved_head_provider(config, names, bounds, reports,
            baseline_folds=fold_proof, rows=rows))
    _verify_baseline_fold_reconstruction(folds, fold_proof)
    mature = [p for p in predictions if timestamp(p["label_available_at"]) <= timestamp(config["fit_cutoff"])]
    calibration, cal_report = _calibrate(mature, config, config["fit_cutoff"])
    head_report_id = fit["baseline_head_fit_report_id"]
    head_report = _report(folder, head_report_id)
    if (head_report["heads_sha256"] != ablation["baseline"]["heads_sha256"]
            or identity(calibration) != ablation["baseline"]["calibration_sha256"]):
        raise ValueError("FR2_M0_FROZEN_HEAD_OR_CALIBRATION_CHANGED")
    validation = [r for r in rows if timestamp(config["validation_start"]) <= timestamp(r["cutoff"])
                  <= timestamp(config["validation_end"])]
    holdout, forecasts = _holdout(validation, head_report["raw_heads"], calibration, names, bounds)
    if holdout != ablation["baseline"]:
        raise ValueError("FR2_M0_NATIVE_HOLDOUT_REPRODUCTION_MISMATCH")
    selected, exclusions = weight_rows_from_oof(rows, predictions, config)
    weights = fit_factor_weights(selected, fit_cutoff=config["fit_cutoff"], **config.get("weight_solver", {}))
    if weights.get("status") != "CONVERGED":
        raise ValueError("FR2_M0_WEIGHT_SOLVER_NOT_CONVERGED")
    counts = deepcopy(fit["counts"])
    counts.update(oof=len(predictions), weight_rows=len(selected),
        weight_contexts=dict(Counter(row["context"] for row in selected)))
    holdout = dict(holdout, schema="v7-independent-holdout/v1", status="COMPLETED", fit_cutoff=config["fit_cutoff"],
        validation_start=config["validation_start"], validation_end=config["validation_end"],
        last_label_available_at=max(timestamp(r["label_available_at"]) for r in validation).isoformat(),
        acceptance=deepcopy(config["acceptance"]), scope="REFERENCE_H20_PREDICTIONS_NOT_PORTFOLIO_CAGR")
    projection = dict(schema="v7-runtime-projection-validation/v1", status="PASS", row_count=len(forecasts), checks=dict(
        ordered_quantiles=all(list(f.quantiles) == sorted(f.quantiles) for _, f in forecasts),
        known_low_upper_bound=all(f.quantiles[-1] <= r["features"]["low"] + 1e-10 for r, f in forecasts),
        joint_le_rebound=all(f.p_entry <= f.p_rebound for _, f in forecasts),
        timing_sum_one=all(abs(sum(f.timing) - 1.) <= 1e-9 for _, f in forecasts)))
    if not all(projection["checks"].values()):
        raise ValueError("FR2_M0_PROJECTION_CHECK_FAILED")
    null_report = _report(folder, candidate["archive_validation_report_ids"]["fractal_null"])
    native_reports = dict(derived_reports)
    for report in (cal_report, holdout, projection, ablation, null_report, head_report):
        native_reports[identity(report)] = report
    ids = dict(calibration=identity(cal_report), holdout=identity(holdout), projection=identity(projection),
        ablation=identity(ablation), fractal_null=identity(null_report))
    provenance = archive_provenance(config, ids, [r["prediction_id"] for r in selected], weights)
    manifest = dict(dataset_id=dataset["sha256"], config_id=identity(config), dataset_kind="RESEARCH",
        source_authenticity="NOT_INDEPENDENTLY_CERTIFIED", upstream_predictions="STRICTLY_PAST_OOF",
        folds=[f for f in folds if f["status"] == "FITTED"], oof_prediction_ids=[p["prediction_id"] for p in predictions],
        weight_prediction_ids=[r["prediction_id"] for r in selected], weight_fit=weights, prior=list(PRIOR), counts=counts,
        transformations="NONE_RAW_FEATURE_COORDINATES", factor_weight_scope="FILLED_REALIZED_NET_PNL_CONDITIONAL_SAMPLES")
    body = dataset["payload"]
    manifest.update(dataset_schema=body["schema"], freeze_id=body["freeze_id"], scope_id=body["scope_id"],
        full_universe_sha256=identity(body["universe"]), child_dataset_ids=[c["dataset_id"] for c in body["children"]],
        diagnostics_policy=body["diagnostics_policy"])
    m0 = dict(feature_profile=candidate["feature_profile"], mapping_version=candidate["mapping_version"],
        feature_names=names, feature_bounds=bounds, imputation=[None] * len(names), supported_masks=[[True] * len(names)],
        heads=deepcopy(head_report["raw_heads"]), weights=weights["weights"], calibration=calibration,
        context_rules=deepcopy(config["context_rules"]), trigger_rules=deepcopy(config["trigger_rules"]),
        reference_policy=deepcopy(config["reference_policy"]), quality_policy=deepcopy(config["quality_policy"]),
        trained_until=None, fit_cutoff=config["fit_cutoff"], last_label_available_at=head_report["last_label_available_at"],
        validation_available_at=holdout["last_label_available_at"], training_pipeline_schema=candidate["training_pipeline_schema"],
        research_only=True, approved_for_live=False, deployment_status="RESEARCH_ONLY_NOT_APPROVED",
        fractal_mode="V7_NO_FRACTAL_BASELINE", baseline_validation_report_id=ids["ablation"],
        fractal_null_validation_report_id=ids["fractal_null"], validation_report_ids=ids,
        validation_reports={rid: native_reports[rid] for rid in ids.values()}, training_manifest=manifest)
    if "execution_feature_profile" in config:
        m0["execution_feature_profile"] = deepcopy(config["execution_feature_profile"])
        extension = deepcopy(config["fill_only_feature_extension"])
        extension["base_feature_names"] = [name for name in extension["base_feature_names"] if name in names]
        extension["selected_features"] = [name for name in extension["selected_features"] if name in names]
        m0["fill_only_feature_extension"] = extension
    if "fill_calibration_profile" in config:
        m0["fill_calibration_profile"] = deepcopy(config["fill_calibration_profile"])
    m0 = compact_candidate(m0, provenance)
    m0["baseline_validation_report_id"] = m0["validation_report_ids"]["ablation"]
    m0.pop("bundle_id")
    m0["bundle_id"] = identity(m0)
    result = dict(dataset_id=dataset["sha256"], config_id=identity(config), counts=counts, folds=folds,
        oof_predictions=predictions, reports=native_reports, training_provenance=provenance)
    verify_archived_compaction(m0, result, config, dataset)
    proof = dict(schema=SCHEMA, original_bundle_sha256=original["sha256"], original_completion_id=identity(receipt),
        original_archive_completion_id=identity(_read(folder / "completion.json")),
        original_pinned_objects=receipt["pinned_objects"], original_baseline_head_report_id=head_report_id,
        original_heads_sha256=identity(head_report["raw_heads"]), original_calibration_id=identity(calibration),
        holdout_prediction_id=ablation["baseline"]["prediction_sha256"],
        full_native_holdout_equality=True, new_head_fits=0, source_authenticity=receipt["source_authenticity"],
        same_dataset_scope_folds_plan_and_labels=True, own_m0_context_weights=True,
        calibration_fit_converged=cal_report.get("converged"),
        calibration_scope="PAST_OOF_CALIBRATION_SET_NOT_FRESH_HOLDOUT",
        weight_exclusions=exclusions, weight_fit=weights, counts=counts,
        last_weight_label_available_at=max(timestamp(r["label_available_at"]) for r in selected).isoformat(),
        weight_rows_id=identity(selected), ordered_weight_prediction_ids=provenance["weight_prediction_ids"],
        parameter_search_on_holdout=False, holdout_role="PREVIOUSLY_INSPECTED_DEVELOPMENT_ONLY",
        shared_fractal_null_report_role="ORIGINAL_ESTIMATOR_ENGINEERING_REPORT_NOT_M0_ACTIVE_FEATURES",
        historical_information_set_certified=False, research_only=True, approved_for_live=False)
    return m0, dict(proof=proof, native_result=result, weight_rows=dict(rows=selected), config=config, dataset=dataset)


def publish_m0(archive, output_root, *, clock=utc_now):
    """New output only; retry is idempotent, corruption is never overwritten."""
    archive, output_root = Path(archive).expanduser().resolve(), Path(output_root).expanduser().resolve()
    if output_root == archive.parent or output_root.is_relative_to(archive.parent):
        raise ValueError("FR2_M0_OUTPUT_MUST_NOT_MODIFY_ORIGINAL_ARCHIVE")
    guard = ResearchVersionGuard()
    objects = ResearchObjects(output_root / "objects")
    pinned = guard.retain(objects)
    binding = _original_binding(archive, objects)
    request = dict(schema=SCHEMA, **binding,
        calculation_versions=asdict(guard.versions), calculation_objects=pinned,
        calculation_environment_id=identity(guard.environment))
    job = output_root / "jobs" / identity(request)
    with _job_lock(job):
        guard.check()
        if (job / "completion.json").exists():
            model, metadata = _load_completed_job(job, output_root, identity(request))
            return dict(metadata, output=str(output_root / "models" / model["sha256"]))
        write_once(job / "request.json", request)
        pending = job / "reconstruction.json"
        if pending.exists():
            saved = _read(pending)
            if identity(saved["payload"]) != saved["sha256"]:
                raise ValueError("FR2_M0_RECOVERY_RECONSTRUCTION_HASH_MISMATCH")
            m0, details = saved["payload"]["candidate"], saved["payload"]["details"]
        else:
            m0, details = reconstruct_m0(archive)
            saved = dict(candidate=m0, details=details)
            guard.check()
            write_once(pending, dict(payload=saved, sha256=identity(saved)))
        if (details["proof"]["original_bundle_sha256"] != binding["original_bundle_sha256"]
                or _original_binding(archive, objects) != binding):
            raise ValueError("FR2_M0_ORIGINAL_BUNDLE_CHANGED_DURING_FIT")
        # Publication time is not the fit cutoff and is never backdated.
        model_file = job / "publication.json"
        if model_file.exists():
            record = _read(model_file)
            if dict(record["payload"], trained_until=None) != m0:
                raise ValueError("FR2_M0_RECOVERY_PUBLICATION_CONFLICT")
        else:
            record = dict(payload=dict(m0, trained_until=clock()))
            record["sha256"] = identity(record["payload"])
            FrozenBundle.load(record)
            if timestamp(record["payload"]["trained_until"]) < max(timestamp(m0["validation_available_at"]), timestamp(m0["fit_cutoff"])):
                raise ValueError("FR2_M0_PUBLICATION_BEFORE_VALIDATION")
            write_once(model_file, record)
        folder = output_root / "models" / record["sha256"]
        write_once(folder / "frozen_bundle.json", record)
        artifact_ids = {}
        for name, value in details.items():
            file_name = name + ".json"
            artifact_ids[file_name] = identity(value)
            write_once(folder / file_name, value)
        write_once(folder / "request.json", request)
        artifact_ids.update({"frozen_bundle.json": identity(record), "request.json": identity(request)})
        completion = dict(schema=SCHEMA, model_id=record["sha256"], request_id=identity(request),
            original_bundle_sha256=details["proof"]["original_bundle_sha256"], artifacts=artifact_ids,
            calculation_objects=pinned, completed_at=clock(), research_only=True, approved_for_live=False,
            historical_information_set_certified=False, execution_action="NO_ACTION", order_authority=False)
        guard.check()
        model, metadata = _finish_completion(folder, job, completion, check=guard.check)
        return dict(metadata, output=str(folder))


def _load_completed_job(job, output_root, expected_request_id, *, verify=None):
    """A completed job must repeat its authoritative model receipt in full."""
    receipt = _read(job / "completion.json")
    model_id = receipt.get("model_id")
    if (not isinstance(model_id, str) or len(model_id) != 64
            or any(c not in "0123456789abcdef" for c in model_id)):
        raise ValueError("FR2_M0_COMPLETED_JOB_MODEL_ID_INVALID")
    folder = output_root / "models" / model_id
    model, metadata = (verify or load_m0_bundle)(folder)
    authoritative = _read(folder / "completion.json")
    if (receipt != authoritative or receipt.get("request_id") != expected_request_id
            or metadata.get("request_id") != expected_request_id or model.get("sha256") != model_id):
        raise ValueError("FR2_M0_COMPLETED_JOB_RECEIPT_MISMATCH")
    return model, metadata


def _finish_completion(folder, job, proposed, *, verify=None, check=None):
    """Recover completion-last publication without inventing a new receipt time.

    The public caller holds the native nonblocking job lock. Model results are
    validated before the job receipt is exposed; a crash at either boundary
    cannot cause a duplicate fit or replace an existing immutable receipt.
    """
    verify = verify or load_m0_bundle
    path = folder / "completion.json"
    if path.exists():
        receipt = _read(path)
        static = lambda value: {k: v for k, v in value.items() if k != "completed_at"}
        if static(receipt) != static(proposed):
            raise ValueError("FR2_M0_RECOVERY_COMPLETION_BINDING_MISMATCH")
    else:
        receipt = proposed
        write_once(path, receipt)
    model, metadata = verify(folder)
    if not (timestamp(model["payload"]["trained_until"]) <= timestamp(receipt["completed_at"])
            <= timestamp(proposed["completed_at"])):
        raise ValueError("FR2_M0_RECOVERY_COMPLETION_TIME_INVALID")
    if check:
        check()
    write_once(job / "completion.json", receipt)
    return model, metadata


def load_m0_bundle(folder):
    """Verify the retained native OOF/weights and return detached runtime JSON."""
    folder = Path(folder).expanduser().resolve()
    if folder.name == "frozen_bundle.json":
        folder = folder.parent
    receipt = _read(folder / "completion.json")
    required = {"frozen_bundle.json", "request.json", "proof.json", "native_result.json", "weight_rows.json", "config.json", "dataset.json"}
    if (receipt.get("schema") not in {SCHEMA, LEGACY_SCHEMA} or set(receipt.get("artifacts", {})) != required
            or receipt.get("research_only") is not True or receipt.get("approved_for_live") is not False
            or receipt.get("order_authority") is not False or receipt.get("execution_action") != "NO_ACTION"):
        raise ValueError("FR2_M0_COMPLETION_BINDING_MISMATCH")
    loaded = {}
    for name, expected in receipt["artifacts"].items():
        loaded[name] = _read(folder / name)
        if identity(loaded[name]) != expected:
            raise ValueError("FR2_M0_ARTIFACT_HASH_MISMATCH:" + name)
    record, request, proof, result = (loaded[n] for n in
        ("frozen_bundle.json", "request.json", "proof.json", "native_result.json"))
    model = FrozenBundle.load(record)
    body = model.record()
    if (model.sha256 != folder.name or model.sha256 != receipt["model_id"]
            or proof.get("schema") != receipt["schema"]
            or receipt["request_id"] != identity(request)
            or receipt["original_bundle_sha256"] != proof["original_bundle_sha256"]
            or body.get("fractal_mode") != "V7_NO_FRACTAL_BASELINE"
            or proof.get("own_m0_context_weights") is not True
            or proof.get("new_head_fits") != 0 or proof.get("full_native_holdout_equality") is not True
            or identity(body["heads"]) != proof["original_heads_sha256"]
            or body["weights"] != proof["weight_fit"]["weights"]
            or proof["weight_fit"]["status"] != "CONVERGED"
            or identity(loaded["weight_rows.json"]["rows"]) != proof["weight_rows_id"]
            or proof["ordered_weight_prediction_ids"] != [r["prediction_id"] for r in loaded["weight_rows.json"]["rows"]]):
        raise ValueError("FR2_M0_MODEL_PROOF_BINDING_MISMATCH")
    verify_archived_compaction(body, result, loaded["config.json"], loaded["dataset.json"])
    refitted = fit_factor_weights(loaded["weight_rows.json"]["rows"], fit_cutoff=body["fit_cutoff"],
        **loaded["config.json"].get("weight_solver", {}))
    if refitted != proof["weight_fit"]:
        raise ValueError("FR2_M0_NATIVE_WEIGHT_REPRODUCTION_MISMATCH")
    if timestamp(proof["last_weight_label_available_at"]) > timestamp(body["fit_cutoff"]):
        raise ValueError("FR2_M0_FUTURE_WEIGHT_LABEL")
    objects = ResearchObjects(folder.parent.parent / "objects")
    if receipt["calculation_objects"] != request["calculation_objects"]:
        raise ValueError("FR2_M0_SOURCE_OBJECT_BINDING_MISMATCH")
    for ref in receipt["calculation_objects"].values():
        objects.get(ref)
    legacy = receipt["schema"] == LEGACY_SCHEMA
    if not legacy:
        encoded = objects.get(request["original_bundle_file_sha256"])
        if len(encoded) > MAX_CONFIG_BYTES:
            raise ValueError("FR2_M0_ORIGINAL_BUNDLE_SIZE_LIMIT")
        original = json.loads(encoded)
        if (not isinstance(original, dict) or request.get("schema") != SCHEMA
                or identity(original) != request["original_bundle_packet_id"]
                or original.get("sha256") != request["original_bundle_sha256"]
                or identity(original.get("payload")) != request["original_bundle_sha256"]
                or request["original_bundle_sha256"] != proof["original_bundle_sha256"]):
            raise ValueError("FR2_M0_ORIGINAL_BYTE_PACKET_MODEL_BINDING_MISMATCH")
    return record, dict(receipt, source_authenticity=proof["source_authenticity"],
        model_kind="TRAINED_NO_FRACTAL_FIVE_HEAD_ABLATION", publication_time=body["trained_until"],
        own_weight_rows=proof["counts"]["weight_rows"], full_native_holdout_equality=True,
        source_file_hash_verified=not legacy,
        compatibility_warnings=["LEGACY_CANONICAL_PACKET_ID_MISNAMED_AS_FILE_SHA256"] if legacy else [])
