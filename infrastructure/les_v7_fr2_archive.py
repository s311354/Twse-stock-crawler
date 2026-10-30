"""Read-only native archive reconstruction for FR2 development comparisons.

Reuses verified prepared shards and saved fitted heads; native OOF/calibration
and holdout calculations are rerun. It does not fabricate missing historical
financial/safety packets, accounts, fills, source availability or fresh tests.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from decimal import Decimal, localcontext
from math import fsum
from pathlib import Path
from zoneinfo import ZoneInfo

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.fractals import is_active_fractal_feature
from domain.les_v7.fr2_comparison import ARMS, POLICY, REPORT_SCHEMA, _forecast_comparison
from domain.les_v7.training import (BASELINE_FOLD_SCHEMA, _baseline_fold_summary,
    _calibrate, _holdout, _oof)
from domain.les_v7.provenance import id_reference
from infrastructure.les_v7_scope_training import (_unpack_row, _packet_feature_columns,
    MAX_PREPARED_BYTES, MAX_PREPARED_DECODED_BYTES, MAX_PREPARED_ROWS, prepared_decoded_limit)
from infrastructure.storage.les_v7_prepared import inspect_prepared_packet
from infrastructure.storage.les_v7_training import LesV7TrainingRepository, MAX_DATASET_BYTES, read_json_bounded


def _read(path):
    return read_json_bounded(path, MAX_DATASET_BYTES)


def _report(folder, report_id):
    if not isinstance(report_id, str) or len(report_id) != 64 or any(c not in "0123456789abcdef" for c in report_id):
        raise ValueError("FR2_NATIVE_REPORT_ID_REQUIRED")
    result = _read(folder / "reports" / (report_id + ".json"))
    if identity(result) != report_id:
        raise ValueError("FR2_NATIVE_REPORT_HASH_MISMATCH")
    return result


def _checked_baseline_folds(config, names, bounds, proof, rows):
    """Only a native, config-bound FR3 record may replay a skipped head fold.

    The enclosing native repository has already verified its fit-result hash,
    source archive and every prepared row. Recompute *all* ordered train/test
    row bindings here; a checksum or plausible skip string alone is not proof.
    """
    if proof is None:
        return None
    from domain.les_v7.fr3_plan import is_fr3_reference
    body = proof.get("payload", {}) if isinstance(proof, dict) else {}
    if (not isinstance(proof, dict) or not isinstance(body, dict) or proof.get("sha256") != identity(body)
            or set(proof) != {"payload", "sha256"}
            or set(body) != {"schema", "config_id", "feature_names", "feature_bounds", "folds"}
            or body.get("schema") != BASELINE_FOLD_SCHEMA
            or not is_fr3_reference(config.get("reference_policy", {}))
            or body.get("config_id") != identity(config)
            or body.get("feature_names") != names or body.get("feature_bounds") != bounds
            or not isinstance(rows, list) or len(rows) > MAX_PREPARED_ROWS
            or len({row["row_id"] for row in rows}) != len(rows)
            or not isinstance(body.get("folds"), list) or len(body["folds"]) != len(config["folds"])
            or any(not isinstance(fold, dict) for fold in body["folds"])):
        raise ValueError("FR3_NATIVE_BASELINE_FOLD_PROOF_BINDING_MISMATCH")
    for saved, declared in zip(body["folds"], config["folds"]):
        fit, start, end = (timestamp(declared[key]) for key in
            ("fit_cutoff", "prediction_start", "prediction_end"))
        training = [row["row_id"] for row in rows
            if timestamp(row["cutoff"]) < fit and timestamp(row["label_available_at"]) <= fit]
        testing = [row["row_id"] for row in rows if start <= timestamp(row["cutoff"]) <= end]
        if (saved.get("fold_id") != declared["fold_id"]
                or any(timestamp(saved[key]) != expected for key, expected in
                    (("fit_cutoff", fit), ("prediction_at", start), ("prediction_end", end)))
                or saved.get("training_row_ids") != id_reference(training)
                or saved.get("prediction_row_ids") != id_reference(testing)
                or saved.get("status") not in {"FITTED", "NOT_READY"}
                or saved["status"] == "NOT_READY" and not isinstance(saved.get("reason"), str)):
            raise ValueError("FR3_NATIVE_BASELINE_FOLD_TIME_OR_ROWS_MISMATCH")
    return deepcopy(body["folds"])


def _verify_baseline_fold_reconstruction(folds, proof):
    if proof is not None and _baseline_fold_summary(folds) != proof["payload"]["folds"]:
        raise ValueError("FR3_NATIVE_BASELINE_FOLD_REPRODUCTION_MISMATCH")


def _saved_head_provider(config, baseline_names, baseline_bounds, reports, *, baseline_folds=None, rows=None):
    """Internal provider over canonically verified native archive reports only."""
    verified = _checked_baseline_folds(config, baseline_names, baseline_bounds, baseline_folds, rows)
    by_cutoff = {} if verified is not None else None
    for fold in verified or []:
        by_cutoff.setdefault(timestamp(fold["fit_cutoff"]), []).append(fold)
    def saved_heads(training, training_config, cutoff, names, bounds, target_reports):
        if training_config != config or names != baseline_names or bounds != baseline_bounds:
            raise ValueError("FR2_ARCHIVED_HEAD_REQUEST_CHANGED")
        bound = by_cutoff.get(timestamp(cutoff), []) if by_cutoff is not None else []
        if by_cutoff is not None and (not bound or any(fold["training_row_ids"] !=
                id_reference([row["row_id"] for row in training]) for fold in bound)):
            raise ValueError("FR3_NATIVE_BASELINE_FOLD_TRAINING_REQUEST_MISMATCH")
        # Native configs can reuse the same fit for disjoint prediction
        # windows. Those calls have identical training rows and head readiness;
        # only an empty prediction window may differ from a FITTED fold.
        failed = [fold for fold in bound if fold["status"] == "NOT_READY" and fold["reason"].startswith(
            ("INSUFFICIENT_", "BOTH_", "POSITIVE_CONDITIONAL_SAMPLE_WEIGHT_REQUIRED",
             "HEAD_SOLVER_NOT_CONVERGED", "SPARSE_TIMING_CLASSES_REQUIRE_"))]
        if failed and (len(failed) != len(bound) or len({fold["reason"] for fold in failed}) != 1):
            raise ValueError("FR3_NATIVE_BASELINE_SHARED_FIT_READINESS_CONFLICT")
        saved = failed[0] if failed else bound[0] if bound else None
        found = [(rid, r) for rid, r in reports.items()
                 if r.get("schema") == "v7-standardized-convex-head-fit/v2"
                 and timestamp(r["fit_cutoff"]) == timestamp(cutoff) and r["feature_count"] == len(names)]
        skipped = bool(failed)
        if skipped:
            # A head-fit failure never produced FITTED head metadata. A saved
            # converged head cannot be hidden by relabeling its fold as skipped.
            if any(fold.get("heads_sha256") or fold.get("head_fit_report_id") for fold in failed):
                raise ValueError("FR3_NATIVE_BASELINE_SKIPPED_HEAD_CONFLICT")
            if saved["reason"] == "HEAD_SOLVER_NOT_CONVERGED":
                if len(found) != 1:
                    raise ValueError("FR2_EXACT_ARCHIVED_HEAD_NOT_FOUND")
                rid, report = found[0]
                if (identity(report) != rid or report.get("converged") is not False
                        or report["training_row_count"] != len(training)
                        or report["ordered_training_row_ids_sha256"] != identity([r["row_id"] for r in training])
                        or identity(report["raw_heads"]) != report["heads_sha256"]
                        or timestamp(report["last_label_available_at"]) > timestamp(cutoff)):
                    raise ValueError("FR2_ARCHIVED_HEAD_TRAINING_BINDING_MISMATCH")
                target_reports[rid] = report
            elif found:
                raise ValueError("FR3_NATIVE_BASELINE_SKIPPED_HEAD_CONFLICT")
            raise ValueError(saved["reason"])
        if len(found) != 1:
            raise ValueError("FR2_EXACT_ARCHIVED_HEAD_NOT_FOUND")
        rid, report = found[0]
        if (identity(report) != rid or report.get("converged") is not True
                or report["training_row_count"] != len(training)
                or report["ordered_training_row_ids_sha256"] != identity([r["row_id"] for r in training])
                or identity(report["raw_heads"]) != report["heads_sha256"]
                or timestamp(report["last_label_available_at"]) > timestamp(cutoff)):
            raise ValueError("FR2_ARCHIVED_HEAD_TRAINING_BINDING_MISMATCH")
        if any(fold["status"] == "FITTED" and (fold.get("heads_sha256") != report["heads_sha256"]
                or fold.get("head_fit_report_id") != rid) for fold in bound):
            raise ValueError("FR3_NATIVE_BASELINE_FITTED_HEAD_PROOF_MISMATCH")
        target_reports[rid] = report
        return report["raw_heads"]
    return saved_heads


def _reference_costs(root, dataset, config, validation):
    """Verified one-share H20 path totals, explicitly not four account returns."""
    from domain.les_v7.reference_path import _cash, _decimal
    targets = {row["row_id"]: row for row in validation}
    observed, diagnostics = set(), []
    policy = config["reference_policy"]["cost_path_policy"]
    cost = policy["cost_policy"]
    for child in dataset["payload"]["children"]:
        folder = (root / child["archive_path"]).resolve()
        if not folder.is_relative_to((root / "builds").resolve()):
            raise ValueError("FR2_CHILD_DATASET_PATH_INVALID")
        packet = _read(folder / "dataset.json")
        if packet.get("sha256") != child["dataset_id"] or identity(packet.get("payload")) != child["dataset_id"]:
            raise ValueError("FR2_REFERENCE_COST_DATASET_HASH_MISMATCH")
        for event in packet["payload"]["events"]:
            row_id = identity(dict(symbol=event["symbol"], market=event["market"], cutoff=timestamp(event["cutoff"]).isoformat()))
            if row_id not in targets:
                continue
            execution = event["execution"]
            if identity(execution) != event["execution_sha256"] or row_id in observed:
                raise ValueError("FR2_REFERENCE_EXECUTION_HASH_OR_DUPLICATE_INVALID")
            observed.add(row_id)
            row = targets[row_id]
            if execution.get("filled") is not bool(row["labels"]["fill"]):
                raise ValueError("FR2_REFERENCE_FILL_LABEL_MISMATCH")
            if not execution["filled"]:
                diagnostics.append(dict(row_id=row_id, execution_id=identity(execution), filled=False))
                continue
            if (execution.get("path_order_verified") is not True or not execution.get("path_evidence_ids")
                    or execution.get("cost_policy_id") != identity(cost)
                    or execution.get("shares") != row["plan"].reference_shares
                    or execution.get("plan_id") != row["plan"].plan_id
                    or timestamp(execution["path_available_at"]) > timestamp(config["evaluation_cutoff"])):
                raise ValueError("FR2_REFERENCE_COST_PATH_BINDING_MISMATCH")
            with localcontext() as context:
                context.prec = 50
                shares = execution["shares"]
                debit = _cash(_decimal(execution["entry"], "entry"), shares, "buy", cost)
                credit = _cash(_decimal(execution["exit_price"], "exit_price"), shares, "sell", cost)
                pnl = credit - debit
                arithmetic = execution["cost_arithmetic"]
                if (Decimal(arithmetic["buy_debit"]) != debit or Decimal(arithmetic["sell_proceeds"]) != credit
                        or Decimal(arithmetic["realized_net_pnl"]) != pnl
                        or Decimal(str(execution["realized_net_pnl"])) != pnl):
                    raise ValueError("FR2_REFERENCE_NET_PNL_ARITHMETIC_MISMATCH")
                diagnostics.append(dict(row_id=row_id, execution_id=identity(execution), filled=True,
                    shares=shares, realized_net_pnl=str(pnl), buy_debit=str(debit), sell_proceeds=str(credit),
                    filled_at=execution["filled_at"], realized_at=execution["realized_at"],
                    exit_reason=execution["exit_reason"], path_evidence_ids=execution["path_evidence_ids"]))
        del packet
    if observed != targets.keys():
        raise ValueError("FR2_REFERENCE_COST_EVENT_COVERAGE_MISMATCH")
    completed = [r for r in diagnostics if r["filled"]]
    pnls = [Decimal(r["realized_net_pnl"]) for r in completed]
    return dict(scope="ONE_NATIVE_REFERENCE_SHARE_PER_EVENT_NOT_ACCOUNT_OR_ADMISSION_PERFORMANCE",
        completed_reference_paths=len(completed), unfilled_reference_paths=len(diagnostics) - len(completed),
        realized_net_pnl_twd_sum=str(sum(pnls, Decimal(0))),
        positive_net_pnl_paths=sum(p > 0 for p in pnls), negative_net_pnl_paths=sum(p < 0 for p in pnls),
        zero_net_pnl_paths=sum(p == 0 for p in pnls), cost_policy_id=identity(cost),
        execution_rows_id=identity(diagnostics), cost_path_policy=policy,
        warning="IDENTICAL_REFERENCE_LABEL_SET_FOR_M0_AND_M1_NOT_FOUR_ALLOCATED_ACCOUNT_PATHS"), diagnostics


def _detached_results(report, forecasts):
    """Separate outbound JSON ownership without changing canonical content."""
    return deepcopy(report), deepcopy(forecasts)


def reconstruct_archive(path):
    """Actual replay of the preserved holdout, explicitly development-only.

    A frozen five-head ablation is a forecast comparison, NOT four historical
    entry/account backtests. All four requested admission/cost arms are retained
    in the output with their real evidence gap; no placeholder empty account is
    substituted. This function never writes into the archive.
    """
    path = Path(path).expanduser().resolve()
    if path.name != "frozen_bundle.json":
        raise ValueError("FR2_NATIVE_FROZEN_BUNDLE_REQUIRED")
    folder, root = path.parent, path.parent.parent.parent
    receipt = LesV7TrainingRepository.verify(folder)
    if receipt.get("dataset_kind") != "RESEARCH" or receipt.get("status") != "READY_TO_FREEZE":
        raise ValueError("FR2_REAL_READY_ARCHIVE_REQUIRED")
    bundle = _read(path)
    if identity(bundle["payload"]) != bundle.get("sha256") or bundle["sha256"] != receipt["bundle_sha256"]:
        raise ValueError("FR2_NATIVE_BUNDLE_HASH_MISMATCH")
    candidate = bundle["payload"]
    config, dataset = _read(folder / "config.json"), _read(folder / "dataset.json")
    if dataset["payload"].get("schema") != "les-v7-sharded-training-dataset/v1":
        raise ValueError("FR2_NATIVE_PREPARED_SHARDED_ARCHIVE_REQUIRED")
    # Repository.verify above already validates all original children/sources,
    # prepared SHA, scope, units, maturities and pinned input identities. Decode
    # the exact prepared bytes once more for computation, without verifying the
    # full source archive a second time. Per-shard SHA rejects concurrent edits.
    rows, prepared_bytes, prepared_decoded_bytes = [], 0, 0
    for child in dataset["payload"]["children"]:
        relative = Path(child["prepared_path"])
        prepared_path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not prepared_path.is_relative_to((root / "prepared").resolve()):
            raise ValueError("FR2_PREPARED_PATH_INVALID")
        prepared = inspect_prepared_packet(prepared_path)
        prepared_bytes += prepared.encoded_bytes
        prepared_decoded_bytes += prepared.decoded_bytes
        if prepared_bytes > MAX_PREPARED_BYTES or prepared_decoded_bytes > prepared_decoded_limit(config,
                legacy_limit=MAX_PREPARED_DECODED_BYTES):
            raise ValueError("FR2_PREPARED_BYTE_BUDGET_EXCEEDED")
        packet = prepared.value
        if (identity(packet) != child["prepared_sha256"] or packet["dataset_id"] != child["dataset_id"]
                or packet["config_id"] != identity(config)):
            raise ValueError("FR2_PREPARED_CHANGED_AFTER_NATIVE_VERIFICATION")
        columns = _packet_feature_columns(packet)
        rows.extend(_unpack_row(row, columns) for row in packet["rows"])
        if len(rows) > MAX_PREPARED_ROWS:
            raise ValueError("FR2_PREPARED_ROW_BUDGET_EXCEEDED")
        del packet, prepared
    rows.sort(key=lambda row: (timestamp(row["cutoff"]), row["symbol"], row["market"]))
    fit = _read(folder / "fit_result.json")
    if identity(fit["payload"]) != fit["sha256"]:
        raise ValueError("FR2_NATIVE_FIT_RESULT_HASH_MISMATCH")
    original = _report(folder, candidate["archive_validation_report_ids"]["ablation"])
    if original.get("same_row_ids") is not True:
        raise ValueError("FR2_NATIVE_PAIRED_ABLATION_REQUIRED")
    baseline_names = [n for n in config["feature_names"] if not is_active_fractal_feature(n)]
    baseline_bounds = [b for n, b in zip(config["feature_names"], config["feature_bounds"]) if not is_active_fractal_feature(n)]
    reports = {rid: _report(folder, rid) for rid in fit["payload"]["report_references"]}

    # The old publisher preserved baseline head coefficients but not its final
    # calibrator. Rebuild ONLY strictly-past baseline OOF/calibration using saved
    # heads through the original native engine; no refitting or new labels.
    proof = fit["payload"].get("baseline_folds")
    predictions, rebuilt_folds, rebuilt_reports = _oof(rows, config, baseline_names, baseline_bounds,
        head_provider=_saved_head_provider(config, baseline_names, baseline_bounds, reports,
            baseline_folds=proof, rows=rows))
    _verify_baseline_fold_reconstruction(rebuilt_folds, proof)
    past = [p for p in predictions if timestamp(p["label_available_at"]) <= timestamp(config["fit_cutoff"])]
    calibration, calibration_report = _calibrate(past, config, config["fit_cutoff"])
    baseline_report = _report(folder, fit["payload"]["baseline_head_fit_report_id"])
    if (identity(calibration) != original["baseline"]["calibration_sha256"]
            or baseline_report["heads_sha256"] != original["baseline"]["heads_sha256"]):
        raise ValueError("FR2_BASELINE_RECONSTRUCTION_ID_MISMATCH")
    validation = [r for r in rows if timestamp(config["validation_start"]) <= timestamp(r["cutoff"]) <= timestamp(config["validation_end"])]
    baseline, baseline_forecasts = _holdout(validation, baseline_report["raw_heads"], calibration, baseline_names, baseline_bounds)
    # Compacted bundle proof IDs differ from native training proof IDs but not
    # probabilities. Use the source-linked native calibrator report identity to
    # reproduce the original holdout's full prediction/calibration hash.
    full_cal_report_id = candidate["archive_validation_report_ids"]["calibration"]
    full_cal_report = _report(folder, full_cal_report_id)
    full_cal = dict(full_cal_report["fitted_parameters"], projection_validated=True,
                    validation_report_id=full_cal_report_id)
    full, full_forecasts = _holdout(validation, candidate["heads"], full_cal, config["feature_names"], config["feature_bounds"])
    if baseline != original["baseline"]:
        raise ValueError("FR2_BASELINE_HOLDOUT_REPRODUCTION_MISMATCH")
    compared_keys = tuple(baseline)
    if any(full[k] != original["full"][k] for k in compared_keys):
        raise ValueError("FR2_FRACTAL_HOLDOUT_REPRODUCTION_MISMATCH")

    def loss_rows(forecasts):
        # Reuse the common comparison kernel, including explicit conditional
        # labels and exact original normalization (weighted probability metrics,
        # unweighted median/quantile summaries).
        from domain.les_v7.fr2_comparison import _event
        model = dict(fit_cutoff=config["fit_cutoff"])
        events, losses = [], []
        for row, forecast in forecasts:
            event = dict(row_id=row["row_id"], symbol=row["symbol"], market=row["market"],
                cutoff=row["cutoff"], label_available_at=row["label_available_at"],
                bottom=row["bottom"], close=row["features"]["close"], atr=row["plan"].atr,
                labels=row["labels"], observation_weight=row["observation_weight"], forecast=forecast.record())
            events.append(event)
            losses.append(_event(event, model, timestamp(config["evaluation_cutoff"])))
        return events, losses

    baseline_events, baseline_losses = loss_rows(baseline_forecasts)
    full_events, full_losses = loss_rows(full_forecasts)
    days = [timestamp(r["cutoff"]).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat() for r in validation]
    sessions = sorted(set(days))
    actual = _forecast_comparison(baseline_losses, full_losses, days)
    counts = Counter(r["context"] for r in validation)
    path_rows = [r for r in validation if r.get("utility") is not None]
    reference_costs, cost_rows = _reference_costs(root, dataset, config, validation)
    forecast_sources = dict(dataset_id=dataset["sha256"], config_id=identity(config),
        bundle_sha256=bundle["sha256"], bundle_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ablation_report_id=identity(original), archive_completion_id=identity(_read(folder / "completion.json")),
        original_pinned_objects=receipt["pinned_objects"],
        native_baseline_head_report_id=identity(baseline_report),
        reconstructed_baseline_calibration_report_id=identity(calibration_report),
        baseline_calibration_id=identity(calibration), full_calibration_id=identity(full_cal),
        prepared_shard_ids=[c["prepared_sha256"] for c in dataset["payload"]["children"]])
    gaps = ["HOLDOUT_PREVIOUSLY_INSPECTED_DEVELOPMENT_ONLY", "THREE_DISJOINT_60_SESSION_WINDOWS_REQUIRED",
            "INSUFFICIENT_PAIRED_FORECAST_EVENTS", "HISTORICAL_QUALITY_SAFETY_RESTRICTION_PACKETS_NOT_IN_TRAINING_ROWS",
            "NO_FRACTAL_CONTEXT_WEIGHT_ARTIFACT_NOT_PUBLISHED",
            "EXPLICIT_INITIAL_ACCOUNT_AND_PREDECESSOR_STATE_NOT_SUPPLIED",
            "FOUR_SOURCE_BOUND_AFTER_COST_ACCOUNT_PATHS_NOT_AVAILABLE"]
    arms = {name: dict(status="NOT_ASSESSABLE", model_kind="TRAINED_NO_FRACTAL_FIVE_HEAD_ABLATION" if name.startswith("M0") else "FROZEN_FRACTAL",
        entry_policy="ORIGINAL_BUNDLE" if name.endswith("P0") else "FR_ENTRY_R2",
        forecast_status="RECOMPUTED_IDENTICAL_TO_ARCHIVED_NATIVE_HOLDOUT", forecast_id=identity(baseline_events if name.startswith("M0") else full_events),
        admission_status="NOT_ASSESSABLE", after_cost_status="NOT_ASSESSABLE",
        reason_codes=gaps[3:]) for name in ARMS}
    report = dict(schema=REPORT_SCHEMA, mode="NATIVE_ARCHIVE_DEVELOPMENT_RECONSTRUCTION",
        status="NOT_ASSESSABLE", acceptance_policy=POLICY, source_bindings=forecast_sources,
        counts=dict(paired_forecast_events=len(validation), sessions=len(sessions),
                    distinct_symbols=len(set(r["symbol"] for r in validation)), contexts=dict(counts)),
        sessions=sessions, forecast_comparisons=dict(P0=actual, P1=actual), arms=arms,
        forecast_reproduction=dict(baseline_complete_metric_and_hash_equality=True, fractal_complete_metric_and_hash_equality=True,
                                   baseline_oof_rows=len(predictions), baseline_calibration_rows=len(past),
                                   reused_fitted_head_reports=len([r for r in rebuilt_reports.values() if "raw_heads" in r]),
                                   new_head_fits=0,
                                   original_full_calibrator_converged=full_cal_report.get("converged"),
                                   reconstructed_baseline_calibrator_converged=calibration_report.get("converged"),
                                   calibration_scope="PAST_OOF_CALIBRATION_SET_NOT_FRESH_HOLDOUT"),
        reference_path_diagnostic=dict(reference_costs, completed_cost_label_rows=len(path_rows),
            clipped_normalized_utility_mean=fsum(r["utility"] for r in path_rows) / len(path_rows) if path_rows else None),
        reason_codes=gaps, source_authenticity=receipt["source_authenticity"],
        historical_information_set_certified=False, research_only=True, approved_for_live=False,
        execution_action="NO_ACTION", order_authority=False)
    # These detached numeric rows make further tie-out possible without changing
    # the original archive. Admission/account arms still require native evidence.
    forecast_artifact = dict(schema="les-v7-fr2-reconstructed-forecasts/v1", source_bindings=forecast_sources,
        M0=baseline_events, M1=full_events, reference_cost_rows=cost_rows,
        research_only=True, approved_for_live=False)
    # Neither a consumer mutation nor edits to the report's source references
    # may alter the global acceptance policy or the independent forecast file.
    return _detached_results(report, forecast_artifact)
