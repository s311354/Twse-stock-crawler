"""Past-only offline v7 fitting over explicitly prepared, hash-bound evidence.

No HTTP, fills, calendar guesses, account evolution or deployment approval.
Input hashes establish integrity, not authenticity of arbitrary caller JSON.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
from math import fsum, log

import numpy as np
import pandas as pd

from domain.les_v7.contracts import CONTEXTS, FEATURE_VERSION, MAPPING_VERSION, PRIOR, canonical, finite, identity, timestamp
from domain.les_v7.decision import factor_vector, make_plan, rule, select_context
from domain.les_v7.features import compute_features, select_bars
from domain.les_v7.fractals import dfa, higuchi, is_active_fractal_feature
from domain.les_v7.localization import head_outputs, project_outputs
from domain.les_v7.research import fit_factor_weights, fit_heads, label_event
from domain.les_v7.calibration import fit_calibration, score_calibration
from domain.les_v7.quality_policy import is_source_evidence_policy, validate_quality_policy
from domain.les_v7.reference_memo import memoize_reference, _bind_reference_history
from domain.les_v7.provenance import archive_provenance, compact_candidate, id_reference
from domain.les_v7.execution_profile import validate_execution_profile, execution_base_count, execution_feature_provider
from domain.les_v7.fr3_plan import is_fr3_reference

PIPELINE = "LES_V7_OFFLINE_TRAINING_R1"
DATASET_SCHEMA = "les-v7-training-dataset/v1"
RESULT_SCHEMA = "les-v7-training-result/v1"
FINAL_REFERENCE_POLICY = "OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_R1"
BASELINE_FOLD_SCHEMA = "les-v7-fr3-native-baseline-folds/v1"


def _baseline_fold_summary(folds):
    """Bind complete ordered ID arrays without duplicating their large graphs."""
    return [{key: id_reference(value) if key in {"training_row_ids", "prediction_row_ids"}
             else deepcopy(value) for key, value in fold.items()} for fold in folds]


def _baseline_fold_proof(config, names, bounds, folds):
    """Native FR3-only reconstruction proof, not caller permission to skip fit."""
    body = dict(schema=BASELINE_FOLD_SCHEMA, config_id=identity(config),
        feature_names=list(names), feature_bounds=deepcopy(bounds), folds=_baseline_fold_summary(folds))
    return dict(payload=body, sha256=identity(body))


def _validate_reference_evidence_policy(config):
    policy = config.get("reference_evidence_policy")
    coordinate = config.get("reference_policy", {}).get("cost_path_policy", {}).get("coordinate_policy")
    if policy not in {None, FINAL_REFERENCE_POLICY}:
        raise ValueError("INVALID_REFERENCE_EVIDENCE_POLICY")
    if (policy == FINAL_REFERENCE_POLICY) != (coordinate == "OFFICIAL_FINAL_ACTIONS_RECONSTRUCTION_R1"):
        raise ValueError("FINAL_REFERENCE_COORDINATE_POLICY_MISMATCH")


def _count(config, name, lower=1, upper=100000):
    value = config[name]
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("INVALID_TRAINING_COUNT:" + name)
    return value


def _validate_fill_calibration_profile(config):
    if "fill_calibration_profile" not in config:
        return
    from domain.les_v7.fill_calibration import validate_profile
    validate_profile(config["fill_calibration_profile"])


def validate_reference_config(config):
    """One shared feature/context/reference-plan contract; no training partitions."""
    names = config["feature_names"]
    if (not isinstance(names, list) or not 2 <= len(names) <= 64 or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n for n in names)):
        raise ValueError("BOUNDED_UNIQUE_TRAINING_FEATURES_REQUIRED")
    if not any(is_active_fractal_feature(n) for n in names) or not any(not is_active_fractal_feature(n) for n in names):
        raise ValueError("FRACTAL_AND_BASELINE_FEATURES_REQUIRED")
    if any(n.startswith(("lppl", "mfdfa")) for n in names):
        raise ValueError("SHADOW_FEATURES_CANNOT_TRAIN_ACTIVE_MODEL")
    bounds = config["feature_bounds"]
    if len(bounds) != len(names) or any(len(b) != 2 or finite(b[0]) >= finite(b[1]) for b in bounds):
        raise ValueError("INVALID_TRAINING_FEATURE_BOUNDS")
    if set(config["context_rules"]) != set(CONTEXTS):
        raise ValueError("CONTEXT_RULES_INCOMPLETE")
    for node in config["context_rules"].values():
        rule(node, {})  # Validate the bounded causal AST even on an empty dataset.
    if config["reference_policy"].get("policy_id") != "REFERENCE_H20":
        raise ValueError("REFERENCE_H20_REQUIRED")
    if is_fr3_reference(config["reference_policy"]):
        from domain.les_v7.fr3_plan import validate_policy
        validate_policy(config["reference_policy"])
    else:
        make_plan("REVERSAL", dict(close=100., support=90., atr=1., hh20=110.),
                  config["reference_policy"], "CONFIG_SHAPE_VALIDATION_ONLY")
    finite(config["reference_policy"]["leadership_rr_min"], "leadership_rr_min", 0)
    validate_execution_profile(config)


def reference_config_id(config):
    payload = {key: config[key] for key in
               ("feature_names", "feature_bounds", "context_rules", "reference_policy")}
    if validate_execution_profile(config) is not None:
        payload.update({key: config[key] for key in ("execution_feature_profile", "fill_only_feature_extension")})
    return identity(payload)


def validate_training_config(config):
    if config.get("schema") != "les-v7-training-config/v1":
        raise ValueError("V7_TRAINING_CONFIG_SCHEMA_REQUIRED")
    validate_reference_config(config)
    _validate_fill_calibration_profile(config)
    if config.get("head_solver") not in {None, "STANDARDIZED_CONVEX_R2"}:
        raise ValueError("UNSUPPORTED_HEAD_SOLVER")
    if validate_execution_profile(config) is not None and config.get("head_solver") != "STANDARDIZED_CONVEX_R2":
        raise ValueError("FILL_ONLY_EXTENSION_REQUIRES_STANDARDIZED_CONVEX_R2")
    _validate_reference_evidence_policy(config)
    if set(config["trigger_rules"]) != {"near_pullback_support", "tight_base_or_retest"}:
        raise ValueError("TRIGGER_RULES_INCOMPLETE")
    for node in config["trigger_rules"].values():
        rule(node, {})
    # No parameter search or support-bound fitting on the final holdout.
    fit, start, end, evaluation = (timestamp(config[k]) for k in
        ("fit_cutoff", "validation_start", "validation_end", "evaluation_cutoff"))
    if not fit < start <= end < evaluation:
        raise ValueError("INDEPENDENT_VALIDATION_TIME_RANGE_REQUIRED")
    folds = config["folds"]
    if not isinstance(folds, list) or not 2 <= len(folds) <= 64:
        raise ValueError("BOUNDED_TEMPORAL_FOLDS_REQUIRED")
    previous, identifiers = None, set()
    for fold in folds:
        f, s, e = (timestamp(fold[k]) for k in ("fit_cutoff", "prediction_start", "prediction_end"))
        if (not fold.get("fold_id") or fold["fold_id"] in identifiers or not f < s <= e < fit
                or previous is not None and s <= previous):
            raise ValueError("OOF_FOLDS_OVERLAP_OR_TIME_LEAKAGE")
        previous = e
        identifiers.add(fold["fold_id"])
    for key in ("min_head_rows", "min_calibration_rows", "min_weight_rows", "min_context_weight_rows", "min_holdout_rows"):
        _count(config, key)
    _count(config, "head_iterations", upper=10000)
    _count(config, "calibration_iterations", upper=10000)
    _count(config, "null_replicates", lower=16, upper=256)
    _count(config, "null_seed", lower=0, upper=2 ** 32 - 1)
    finite(config["regularization"], "regularization", 0, 100)
    if not is_source_evidence_policy(config["quality_policy"]):
        raise ValueError("REGISTERED_SOURCE_BOUND_QUALITY_POLICY_REQUIRED")
    validate_quality_policy(config["quality_policy"], config["evaluation_cutoff"])
    criteria = config["acceptance"]
    for key in ("max_brier", "max_timing_nll", "max_pinball_loss"):
        finite(criteria[key], key, 0)
    if any(key not in {"delta", "pooling", "prior_strength", "tolerance", "max_iterations"}
           for key in config.get("weight_solver", {})):
        raise ValueError("UNSUPPORTED_WEIGHT_SOLVER_OPTION")


def _vectors(features, plan, context, names, bounds):
    supplied = dict(features, entry=plan.entry, stop=plan.stop, reference_shares=float(plan.reference_shares),
                    entry_stop_atr=plan.stop_atr, **{f"context_{c}": float(c == context) for c in CONTEXTS})
    values = [supplied.get(name) for name in names]
    if any(v is None for v in values):
        return None, "TRAINING_FEATURES_INCOMPLETE"
    values = [finite(v, "training_feature") for v in values]
    if any(not lo <= v <= hi for v, (lo, hi) in zip(values, bounds)):
        return None, "FEATURE_OUTSIDE_FROZEN_SUPPORT"
    return values, None


def prepare_reference(symbol, market, history, cutoff, config, evidence_ids, *, session_evidence=None):
    """Shared causal seed for dataset construction and offline label selection."""
    cutoff = timestamp(cutoff).isoformat()
    bars = select_bars(history, cutoff)
    market_rows = bars.astype(object).where(pd.notna(bars), None).to_dict("records")
    for row in market_rows:
        row["Date"] = pd.Timestamp(row["Date"]).date().isoformat()
    fr3 = is_fr3_reference(config["reference_policy"])
    binding = dict(symbol=symbol, market=market, cutoff=cutoff, bars=market_rows,
                   data_rules=FEATURE_VERSION, external_evidence=evidence_ids)
    if fr3:
        binding["reference_calendar_id"] = identity(session_evidence)
    evidence_id = identity(binding)
    def calculate():
        features, audit = compute_features(bars)
        context, matches = select_context(features, config["context_rules"])
        if fr3:
            from domain.les_v7.fr3_plan import make_plan as fr3_plan
            plan, plan_audit = fr3_plan(context, features, config["reference_policy"], evidence_id,
                bars=bars, cutoff=cutoff, session_evidence=session_evidence)
            audit = dict(audit, fr3_reference=dict(version=config["reference_policy"]["version"],
                state=plan_audit["state"], reasons=list(plan_audit["reasons"]),
                plan_id=plan.plan_id if plan else None,
                calendar_id=plan_audit.get("session_coverage", {}).get("source_id")))
        else:
            plan = make_plan(context, features, config["reference_policy"], evidence_id)
        partition = validate_execution_profile(config)
        if partition is not None:
            provider = execution_feature_provider(config)
            extension, execution_audit = provider._from_selected_bars(bars, base_features=features, plan=plan)
            features = dict(features, **{name: extension[name] for name in partition[1]})
            audit = dict(audit, execution_feature_audit=execution_audit)
        vector, gap = (_vectors(features, plan, context, config["feature_names"], config["feature_bounds"])
                       if plan is not None else (None, "REFERENCE_PLAN_NOT_READY"))
        return dict(features=features, feature_audit=audit, context=context, context_matches=matches,
                    plan=plan, vector=vector, feature_gap=gap, evidence_id=evidence_id)
    return memoize_reference((evidence_id, reference_config_id(config)), calculate)


def prepare_events(dataset_record, config):
    """Recompute causal features/plans and labels through native v7 functions."""
    payload = dataset_record["payload"]
    dataset_id = dataset_record["sha256"]
    _validate_reference_evidence_policy(config)
    if identity(payload) != dataset_id or payload.get("schema") != DATASET_SCHEMA:
        raise ValueError("V7_TRAINING_DATASET_HASH_OR_SCHEMA_MISMATCH")
    if payload.get("reference_evidence_policy") != config.get("reference_evidence_policy"):
        raise ValueError("TRAINING_REFERENCE_EVIDENCE_POLICY_MISMATCH")
    manifest = payload.get("builder_manifest")
    if manifest is not None and manifest.get("reference_config_id") != reference_config_id(config):
        raise ValueError("TRAINING_DATASET_REFERENCE_CONFIG_MISMATCH")
    if payload.get("dataset_kind") not in {"RESEARCH", "SYNTHETIC_TEST_ONLY"}:
        raise ValueError("EXPLICIT_TRAINING_DATASET_KIND_REQUIRED")
    if (payload.get("price_unit") != "TWD" or payload.get("volume_unit") != "SHARES"
            or payload.get("amount_unit") != "TWD" or not payload.get("source_evidence_ids")
            or not payload.get("universe_policy")):
        raise ValueError("EXPLICIT_TRAINING_UNITS_UNIVERSE_AND_SOURCES_REQUIRED")
    histories, calendars = {}, payload["calendars"]
    for item in payload["histories"]:
        key = item["symbol"], item["market"]
        if (not isinstance(key[0], str) or not key[0] or key[0] != key[0].strip()
                or key in histories or key[1] not in {"TWSE", "TPEX"} or not item.get("coordinate_id")):
            raise ValueError("DUPLICATE_OR_INVALID_TRAINING_HISTORY")
        frame = pd.DataFrame(item["bars"])
        if frame.empty:
            frame = frame.reindex(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Available At", "Complete"])
        elif "Available At" not in frame or "Complete" not in frame:
            raise ValueError("BAR_AVAILABILITY_AND_COMPLETENESS_REQUIRED")
        selected = select_bars(frame, config["evaluation_cutoff"])
        source_rows = {pd.Timestamp(row["Date"]).date().isoformat(): row for row in selected.to_dict("records")}
        histories[key] = item, selected, source_rows, _bind_reference_history(
            item['bars'], selection_cutoff=config['evaluation_cutoff'])
    expected_universe = sorted([list(key) for key in histories])
    if sorted(payload["universe"]) != expected_universe:
        raise ValueError("FIXED_TRAINING_UNIVERSE_MISMATCH")
    for market, calendar in calendars.items():
        source = "SYNTHETIC_TEST_ONLY" if payload["dataset_kind"] == "SYNTHETIC_TEST_ONLY" else market + "_OFFICIAL"
        if (calendar["source"] != source or not calendar.get("evidence_ids")
                or calendar["sessions"] != sorted(set(calendar["sessions"]))):
            raise ValueError("EXPLICIT_FROZEN_CALENDAR_REQUIRED")
        timestamp(calendar["available_at"])
    records, diagnostics, seen = [], [], set()
    events = sorted(payload["events"], key=lambda e: (timestamp(e["cutoff"]), e["symbol"], e["market"]))
    for event in events:
        key = event["symbol"], event["market"]
        cutoff = timestamp(event["cutoff"]).isoformat()
        row_id = identity(dict(symbol=key[0], market=key[1], cutoff=cutoff))
        if row_id in seen:
            raise ValueError("DUPLICATE_TRAINING_EVENT")
        seen.add(row_id)
        if key not in histories or key[1] not in calendars or not event.get("evidence_ids"):
            raise ValueError("TRAINING_EVENT_NOT_BOUND_TO_UNIVERSE_OR_EVIDENCE")
        item, history, source_rows, reference = histories[key]
        fr3 = is_fr3_reference(config["reference_policy"])
        options = dict(session_evidence=calendars[key[1]]) if fr3 else {}
        prepared = (reference(key[0], key[1], cutoff, config, event['evidence_ids'], **options) if reference is not None else
                    prepare_reference(key[0], key[1], history, cutoff, config, event['evidence_ids'], **options))
        features, audit = prepared["features"], prepared["feature_audit"]
        context, matches, plan = prepared["context"], prepared["context_matches"], prepared["plan"]
        diagnostic = dict(row_id=row_id, symbol=key[0], market=key[1], cutoff=cutoff, context=context,
                          context_matches=list(matches), feature_audit=audit, plan=asdict(plan) if plan else None)
        diagnostics.append(diagnostic)
        session = pd.Timestamp(cutoff).tz_convert("Asia/Taipei").date().isoformat()
        if audit.get("cutoff_session") != session:
            diagnostic["excluded_reason"] = "NO_COMPLETE_BAR_FOR_EVENT_SESSION"
            continue
        if timestamp(cutoff) >= timestamp(config["evaluation_cutoff"]) or plan is None:
            diagnostic["excluded_reason"] = "REFERENCE_PLAN_NOT_READY_OR_FUTURE_EVENT"
            continue
        vector, gap = prepared["vector"], prepared["feature_gap"]
        if gap:
            diagnostic["excluded_reason"] = gap
            continue
        calendar = calendars[key[1]]
        if session not in calendar["sessions"]:
            diagnostic["excluded_reason"] = "CALENDAR_SESSION_UNKNOWN"
            continue
        index = calendar["sessions"].index(session)
        sessions = calendar["sessions"][index:index + 21]
        if len(sessions) != 21:
            diagnostic["excluded_reason"] = "RIGHT_CENSORED_CALENDAR"
            continue
        l2 = config["reference_policy"].get("version") == "FR3_SOURCE_BOUND_LIMIT_H20_L2"
        if l2:
            # L2's conditional waiting/entry evidence binds the complete raw
            # OHLCV/source window. A low-only projection is not that same
            # evidence and must not be hashed as though it were. R1 retains
            # its exact historical label-input shape.
            from domain.les_v7.reference_path import path_window
            window = path_window(item, sessions)
        else:
            window = [dict(session=s, low=source_rows[s]["Low"], complete=source_rows[s]["Complete"],
                           coordinate_id=item["coordinate_id"], available_at=source_rows[s]["Available At"])
                      for s in sessions if s in source_rows]
        fill = event.get("execution")
        if fill is not None and identity(fill) != event.get("execution_sha256"):
            raise ValueError("EXECUTION_EVIDENCE_HASH_MISMATCH")
        if manifest is not None:
            from domain.les_v7.reference_path import evaluate_reference_path, path_window
            binding = event.get("simulation_reference", {})
            policy = manifest["path_policy"]
            coverages = [c for c in manifest["coordinate_coverages"] if identity(c) == binding.get("coverage_sha256")]
            if len(coverages) != 1 or binding.get("path_policy_id") != identity(policy):
                raise ValueError("TRAINING_SIMULATION_POLICY_OR_COVERAGE_BINDING_MISMATCH")
            raw_window = window if l2 else path_window(item, sessions)
            if fr3:
                from domain.les_v7.fr3_plan import bind_entry_route, validate_band
                if validate_band(plan, config["reference_policy"])["state"] != "PASS":
                    diagnostic["excluded_reason"] = "FR3_FIXED_PLAN_RISK_NOT_PASS"
                    continue
                path_options = dict(entry_route=bind_entry_route(plan, config["reference_policy"]))
            else:
                path_options = {}
            expected = evaluate_reference_path(plan, raw_window, policy, expected_sessions=sessions,
                cutoff=cutoff, coordinate_id=item["coordinate_id"], source_evidence_ids=event["evidence_ids"],
                corporate_action_coverage=coverages[0], **path_options)
            if identity(raw_window) != binding.get("raw_window_sha256") or expected["execution"] != fill:
                raise ValueError("TRAINING_SIMULATED_EXECUTION_RECOMPUTATION_MISMATCH")
        label = label_event(window, cutoff=cutoff, expected_sessions=sessions, plan=plan, fill=fill)
        diagnostic["label_status"] = label["status"]
        if label["status"] not in {"MATURED", "TIMEOUT"} or label.get("fill") not in {0, 1} or label.get("bottom") is None:
            diagnostic["excluded_reason"] = label["status"]
            continue
        if "availability_by_date" in calendar:
            dates = calendar["availability_by_date"]
            if "days" in calendar:
                first, last = date.fromisoformat(sessions[0]), date.fromisoformat(sessions[-1])
                needed = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
                if any(calendar["days"].get(d) not in {"OPEN", "CLOSED"} for d in needed):
                    diagnostic["excluded_reason"] = "CALENDAR_WINDOW_CONTAINS_UNKNOWN_DATES"
                    continue
            else:
                needed = sessions  # Compatibility for external pre-builder datasets.
            if any(not dates.get(s) for s in needed):
                diagnostic["excluded_reason"] = "CALENDAR_PUBLICATION_FOR_LABEL_WINDOW_UNKNOWN"
                continue
            calendar_available = max(timestamp(dates[s]) for s in needed)
        else:
            calendar_available = timestamp(calendar["available_at"])
        available = max(timestamp(label["label_available_at"]), calendar_available)
        if available > timestamp(config["evaluation_cutoff"]):
            diagnostic["excluded_reason"] = "UNMATURED_LABEL"
            continue
        if timestamp(cutoff) >= available:
            raise ValueError("LABEL_MUST_MATURE_AFTER_FEATURE_CUTOFF")
        labels = dict(fill=label["fill"], rebound=label.get("rebound"), joint=label.get("joint"), timing=label["timing"])
        weight = finite(event.get("observation_weight", 1.), "weight", 0)
        if weight <= 0:
            raise ValueError("POSITIVE_EVENT_WEIGHT_REQUIRED")
        records.append(dict(row_id=row_id, cutoff=cutoff, symbol=key[0], market=key[1], features=features,
            vector=vector, plan=plan, context=context, labels=labels, label_available_at=available.isoformat(),
            log_bottom_ratio=log(label["bottom"] / features["close"]), bottom_timing=label["timing"],
            bottom=label["bottom"], utility=label.get("normalized_utility"),
            feature_fit_cutoff=cutoff, observation_weight=weight))
    return records, diagnostics


def _head_rows(rows, names, bounds):
    result = []
    for row in rows:
        vector, gap = _vectors(row["features"], row["plan"], row["context"], names, bounds)
        if gap:
            continue
        result.append(dict(row, features=vector, fill=row["labels"]["fill"],
                           rebound=row["labels"]["rebound"], joint=row["labels"]["joint"]))
    return result


def _fit(rows, config, cutoff, names, bounds, reports=None):
    if len(rows) < config["min_head_rows"]:
        raise ValueError("INSUFFICIENT_HEAD_TRAINING_ROWS")
    options = {"solver": config["head_solver"]} if config.get("head_solver") else {}
    base_count = execution_base_count(config, names)
    if base_count is not None:
        options["fill_only_base_count"] = base_count
    fitted = fit_heads(_head_rows(rows, names, bounds), fit_cutoff=cutoff,
        iterations=config["head_iterations"], regularization=config["regularization"], **options)
    if config.get("head_solver"):
        report = fitted["solver_report"]
        if reports is not None:
            reports[identity(report)] = report
        if report["converged"] is not True:
            raise ValueError("HEAD_SOLVER_NOT_CONVERGED")
    return fitted["heads"]


def _calibrate(rows, config, cutoff):
    if len(rows) < config["min_calibration_rows"]:
        raise ValueError("INSUFFICIENT_PAST_OOF_CALIBRATION_ROWS")
    fitted = fit_calibration(rows, iterations=config["calibration_iterations"], regularization=config["regularization"])
    if "fill_calibration_profile" in config:
        from domain.les_v7.fill_calibration import fit_fill_calibration
        _validate_fill_calibration_profile(config)
        fill_fit = fit_fill_calibration(rows, fit_cutoff=cutoff, calibration_profile=config["fill_calibration_profile"])
        legacy_report = fitted["report"]
        parameters = dict(legacy_report["fitted_parameters"], fill=list(fill_fit["pair"]),
            version=legacy_report["fitted_parameters"]["version"] + "+" + config["fill_calibration_profile"]["version"])
        calibration = dict(parameters, projection_validated=True)
        # Component objectives are different.  Keep each original optimizer
        # receipt intact instead of attributing the R1 objective to this pair.
        report = dict(schema_version="LES_V7_OOF_CALIBRATION_REPORT_R2",
            method="COMPOSITE_R1_PROTECTED_WITH_FILL_IDENTITY_PRIOR_R1", version=parameters["version"],
            fit_status="FITTED_COMPONENT_OBJECTIVES_REPORTED_SEPARATELY",
            evaluation_scope="CALIBRATION_SET_NOT_HOLDOUT", data_sha256=identity(rows),
            row_ids=[row["row_id"] for row in rows], fitted_parameters=parameters,
            fill_calibration_profile=deepcopy(config["fill_calibration_profile"]),
            legacy_calibration_report=legacy_report, fill_calibration_report=fill_fit["report"],
            before=legacy_report["before"], after=score_calibration(rows, calibration),
            projection_policy=legacy_report["projection_policy"], approved_for_live=False)
        fitted = dict(calibration=calibration, report=report)
    report = dict(fitted["report"], fit_cutoff=timestamp(cutoff).isoformat(),
                  last_label_available_at=max(timestamp(r["label_available_at"]) for r in rows).isoformat())
    report_id = identity(report)
    calibration = dict(fitted["calibration"], validation_report_id=report_id)
    return calibration, report


def _head_fit_receipt_id(config, reports, heads, rows, cutoff, names):
    """FR3 binds the actual fit, even if another date has identical coefficients.

    Legacy archives retain their original receipt-selection semantics and
    bytes. This changes no fitted coefficient or predictive calculation.
    """
    head_id = identity(heads)
    if not is_fr3_reference(config.get("reference_policy", {})):
        return next(ref for ref, report in reports.items()
            if report.get("schema") == "v7-standardized-convex-head-fit/v2"
            and report.get("heads_sha256") == head_id)
    ordered_ids = identity([row["row_id"] for row in rows])
    found = [ref for ref, report in reports.items()
        if report.get("schema") == "v7-standardized-convex-head-fit/v2"
        and report.get("heads_sha256") == head_id
        and timestamp(report["fit_cutoff"]) == timestamp(cutoff)
        and report.get("feature_count") == len(names)
        and report.get("training_row_count") == len(rows)
        and report.get("ordered_training_row_ids_sha256") == ordered_ids
        and report.get("converged") is True and identity(report) == ref]
    if len(found) != 1:
        raise ValueError("FR3_EXACT_HEAD_FIT_RECEIPT_REQUIRED")
    return found[0]


def _oof(rows, config, names, bounds, *, head_provider=None):
    """Native past-only folds; a verified offline archive can reuse fitted heads.

    ``head_provider`` is internal to immutable-archive reconstruction. The public
    trainer never accepts caller-supplied heads and uses the unchanged fitter.
    """
    predictions, folds, reports = [], [], {}
    for fold in config["folds"]:
        fit_at, start, end = (timestamp(fold[k]) for k in ("fit_cutoff", "prediction_start", "prediction_end"))
        training = [r for r in rows if timestamp(r["cutoff"]) < fit_at and timestamp(r["label_available_at"]) <= fit_at]
        testing = [r for r in rows if start <= timestamp(r["cutoff"]) <= end]
        info = dict(fold_id=fold["fold_id"], fit_cutoff=fit_at.isoformat(), prediction_at=start.isoformat(),
                    prediction_end=end.isoformat(), training_row_ids=[r["row_id"] for r in training],
                    prediction_row_ids=[r["row_id"] for r in testing])
        folds.append(info)
        try:
            heads = (_fit(training, config, fit_at.isoformat(), names, bounds, reports=reports)
                     if head_provider is None else
                     head_provider(training, config, fit_at.isoformat(), names, bounds, reports))
        except ValueError as error:
            if str(error).startswith(("INSUFFICIENT_", "BOTH_", "POSITIVE_CONDITIONAL_SAMPLE_WEIGHT_REQUIRED", "HEAD_SOLVER_NOT_CONVERGED", "SPARSE_TIMING_CLASSES_REQUIRE_")):
                info.update(status="NOT_READY", reason=str(error))
                continue
            raise
        if not testing:
            info.update(status="NOT_READY", reason="OOF_FOLD_HAS_NO_ELIGIBLE_PREDICTIONS")
            continue
        if config.get("head_solver"):
            info["head_fit_report_id"] = _head_fit_receipt_id(config, reports, heads, training,
                fit_at.isoformat(), names)
        info.update(status="FITTED", heads_sha256=identity(heads),
                    last_training_label_available_at=max(timestamp(r["label_available_at"]) for r in training).isoformat())
        past = [p for p in predictions if timestamp(p["label_available_at"]) <= fit_at]
        calibration, calibration_report = None, None
        try:
            calibration, calibration_report = _calibrate(past, config, fit_at.isoformat())
        except ValueError as error:
            if not (str(error).startswith(("INSUFFICIENT_", "BOTH_", "CALIBRATION_"))
                    or str(error) == "CONDITIONAL_CALIBRATION_ROWS_REQUIRED"):
                raise
            info["calibration_reason"] = str(error)
        if calibration_report:
            cid = identity(calibration_report)
            reports[cid] = calibration_report
            info["calibration_report_id"] = cid
        for row in testing:
            vector, gap = _vectors(row["features"], row["plan"], row["context"], names, bounds)
            if gap:
                continue
            raw = head_outputs(heads, vector)
            item = dict(row_id=row["row_id"], cutoff=row["cutoff"], labels=row["labels"],
                        label_available_at=row["label_available_at"], observation_weight=row["observation_weight"],
                        head_fit_cutoff=fit_at.isoformat(), head_last_label_available_at=info["last_training_label_available_at"],
                        heads_sha256=info["heads_sha256"], fold_id=fold["fold_id"], **raw)
            if calibration is not None:
                forecast = project_outputs(raw, row["features"], calibration)
                item.update(forecast=forecast.record(), calibration_fit_cutoff=fit_at.isoformat(),
                    calibration_last_label_available_at=calibration_report["last_label_available_at"],
                    calibration_report_id=identity(calibration_report))
            else:
                item["forecast"] = dict(status="NOT_READY", reasons=["NO_PAST_CALIBRATOR"])
            item["label_id"] = identity(dict(row_id=row["row_id"], labels=row["labels"],
                                            available_at=row["label_available_at"]))
            item["prediction_id"] = identity({k: v for k, v in item.items()
                                              if k not in {"labels", "label_available_at", "label_id"}})
            predictions.append(item)
    return predictions, folds, reports


def _holdout(rows, heads, calibration, names, bounds):
    predicted, forecasts = [], []
    for row in rows:
        vector, gap = _vectors(row["features"], row["plan"], row["context"], names, bounds)
        if gap:
            continue
        raw = head_outputs(heads, vector)
        forecast = project_outputs(raw, row["features"], calibration)
        if forecast.status != "VALID":
            raise ValueError("INVALID_HOLDOUT_FORECAST")
        predicted.append(dict(row_id=row["row_id"], labels=row["labels"], observation_weight=row["observation_weight"], **raw))
        forecasts.append((row, forecast))
    metrics = score_calibration(predicted, calibration)
    losses, covered, near = [], [], []
    for row, forecast in forecasts:
        for quantile, price in zip((.1, .5, .9), forecast.quantiles):
            error = (row["bottom"] - price) / row["features"]["close"]
            losses.append(max(quantile * error, (quantile - 1) * error))
        covered.append(forecast.quantiles[0] <= row["bottom"] <= forecast.quantiles[2])
        near.append(abs(forecast.quantiles[1] - row["bottom"]) / row["plan"].atr)
    return dict(metrics=metrics, pinball_loss=fsum(losses) / len(losses),
        bottom_interval_coverage=fsum(covered) / len(covered), median_bottom_error_atr=float(np.median(near)),
        price_metric_normalization="UNWEIGHTED_EVENT_SUMMARIES_THREE_EQUAL_QUANTILES",
        heads_sha256=identity(heads), calibration_sha256=identity(calibration), feature_names=list(names),
        label_sha256=identity([dict(row_id=r["row_id"], labels=r["labels"], bottom=r["bottom"],
                                   available_at=r["label_available_at"]) for r, _ in forecasts]),
        prediction_sha256=identity([{k: v for k, v in p.items() if k != "labels"} for p in predicted]),
        row_ids=[row["row_id"] for row, _ in forecasts]), forecasts


def fractal_null_report(config):
    """Seeded estimator controls, separate from stock prediction validation."""
    generator = np.random.default_rng(config["null_seed"])
    values, checks = {}, []
    for n in (512, 1024):
        if not any(name in config["feature_names"] for name in (f"dfa{n}", f"delta_dfa{n}_5")):
            continue
        estimates, changes, sign_errors = [], [], []
        for _ in range(config["null_replicates"]):
            sequence = generator.normal(size=n + 5)
            current, flipped = dfa(sequence[-n:]), dfa(-sequence[-n:])
            previous = dfa(sequence[:-5])
            error = abs(current.value - flipped.value) if current.value is not None and flipped.value is not None else None
            checks.append(error is not None and error <= 1e-12)
            sign_errors.append(error)
            estimates.append(current.value)
            changes.append(current.value - previous.value if current.value is not None and previous.value is not None else None)
        valid = [v for v in estimates if v is not None]
        values[f"dfa{n}"] = dict(count=len(valid), mean=fsum(valid) / len(valid) if valid else None,
                                quantiles=np.quantile(valid, [.05, .5, .95]).tolist() if valid else None)
        checks.append(len(valid) == config["null_replicates"] and .35 <= values[f"dfa{n}"]["mean"] <= .65)
        if f"delta_dfa{n}_5" in config["feature_names"]:
            valid_changes = [v for v in changes if v is not None]
            values[f"delta_dfa{n}_5"] = dict(count=len(valid_changes),
                mean=fsum(valid_changes) / len(valid_changes) if valid_changes else None,
                quantiles=np.quantile(valid_changes, [.05, .5, .95]).tolist() if valid_changes else None)
            checks.append(len(valid_changes) == config["null_replicates"] and abs(values[f"delta_dfa{n}_5"]["mean"]) <= .1)
        values[f"dfa{n}"]["maximum_sign_error"] = max((v for v in sign_errors if v is not None), default=None)
    for n in (256, 512):
        if f"higuchi{n}" not in config["feature_names"]:
            continue
        estimates = [higuchi(np.cumsum(generator.normal(size=n))).value for _ in range(config["null_replicates"])]
        valid = [v for v in estimates if v is not None]
        values[f"higuchi{n}"] = dict(count=len(valid), mean=fsum(valid) / len(valid) if valid else None)
        checks.append(len(valid) == config["null_replicates"] and 1.3 <= values[f"higuchi{n}"]["mean"] <= 1.7)
    return dict(schema="v7-fractal-null-controls/v1", status="PASS" if checks and all(checks) else "FAIL",
        seed=config["null_seed"], replicates=config["null_replicates"], estimates=values,
        criteria=dict(iid_return_dfa_mean=[.35, .65], random_walk_higuchi_mean=[1.3, 1.7],
                      dfa_change_mean_abs_max=.1, sign_invariance_error_max=1e-12),
        scope="SEEDED_ESTIMATOR_CONTROLS_NOT_MARKET_PREDICTIVE_SKILL")


def train_offline(dataset_record, config):
    """Deterministic research-only candidate; publication supplies actual availability."""
    config = deepcopy(config)
    validate_training_config(config)
    rows, diagnostics = prepare_events(dataset_record, config)
    return _train_prepared(dataset_record, config, rows, diagnostics)


def _train_prepared(dataset_record, config, rows, diagnostics):
    """One shared fit over native-prepared rows; orchestration is not a new model.

    Internal only: the public single-dataset port prepares its own rows, and the
    full-scope adapter verifies every native child before assembling this input.
    No caller-supplied prepared-row JSON is a public training request.
    """
    validate_training_config(config)
    result = dict(schema=RESULT_SCHEMA, status="NOT_READY", dataset_id=dataset_record["sha256"], config_id=identity(config),
                  candidate=None, reports={}, oof_predictions=[], diagnostics=diagnostics, reason_codes=[],
                  research_only=True, approved_for_live=False, source_authenticity="NOT_INDEPENDENTLY_CERTIFIED")
    fit_at = timestamp(config["fit_cutoff"])
    training = [r for r in rows if timestamp(r["cutoff"]) < fit_at and timestamp(r["label_available_at"]) <= fit_at]
    validation = [r for r in rows if timestamp(config["validation_start"]) <= timestamp(r["cutoff"]) <= timestamp(config["validation_end"])]
    result["counts"] = dict(requested=len(diagnostics), prepared=len(rows), final_training=len(training), holdout=len(validation),
                            exclusions=dict(Counter(d.get("excluded_reason") for d in diagnostics if d.get("excluded_reason"))))
    names, bounds = config["feature_names"], config["feature_bounds"]
    predictions, folds, reports = _oof(rows, config, names, bounds)
    result.update(oof_predictions=predictions, reports=reports, folds=folds)
    mature = [p for p in predictions if timestamp(p["label_available_at"]) <= fit_at]
    by_id = {r["row_id"]: r for r in rows}
    weight_rows = []
    for p in mature:
        row = by_id[p["row_id"]]
        if p["forecast"]["status"] != "VALID" or row["utility"] is None:
            continue
        # Reconstruct the exact projected, past-calibrated forecast, not a caller score.
        from domain.les_v7.localization import Forecast
        f = p["forecast"]
        forecast = Forecast("VALID", tuple(f["quantiles"]), tuple(f["timing"]), f["p_fill"], f["p_rebound"], f["p_entry"],
                            calibration_version=f["calibration_version"])
        factors, _ = factor_vector(row["context"], row["features"], forecast, row["plan"],
                                   leadership_rr_min=config["reference_policy"].get("leadership_rr_min"))
        if factors is None:
            continue
        weight_rows.append(dict(factors=list(factors), utility=row["utility"], context=row["context"],
            prediction_kind="STRICTLY_PAST_OOF", cutoff=row["cutoff"], label_available_at=row["label_available_at"],
            head_fit_cutoff=p["head_fit_cutoff"], head_last_label_available_at=p["head_last_label_available_at"],
            calibration_fit_cutoff=p["calibration_fit_cutoff"],
            calibration_last_label_available_at=p["calibration_last_label_available_at"],
            observation_weight=row["observation_weight"], row_id=row["row_id"], prediction_id=p["prediction_id"]))
    context_counts = dict(Counter(r["context"] for r in weight_rows))
    result["counts"].update(oof=len(predictions), weight_rows=len(weight_rows), weight_contexts=context_counts)
    if len(validation) < config["min_holdout_rows"]:
        result["reason_codes"].append("INSUFFICIENT_INDEPENDENT_HOLDOUT")
    if len(weight_rows) < config["min_weight_rows"] or any(context_counts.get(c, 0) < config["min_context_weight_rows"] for c in CONTEXTS):
        result["reason_codes"].append("INSUFFICIENT_PAST_CALIBRATED_CONTEXT_WEIGHT_ROWS")
    if result["reason_codes"]:
        return result
    try:
        heads = _fit(training, config, config["fit_cutoff"], names, bounds, reports=reports)
        if config.get("head_solver"):
            result["final_head_fit_report_id"] = _head_fit_receipt_id(config, reports, heads, training,
                config["fit_cutoff"], names)
        calibration, cal_report = _calibrate(mature, config, config["fit_cutoff"])
        weights = fit_factor_weights(weight_rows, fit_cutoff=config["fit_cutoff"], **config.get("weight_solver", {}))
    except ValueError as error:
        if str(error).startswith(("INSUFFICIENT_", "BOTH_", "CALIBRATION_", "HEAD_SOLVER_NOT_CONVERGED", "SPARSE_TIMING_CLASSES_REQUIRE_")):
            result["reason_codes"].append(str(error))
            return result
        raise
    if weights["status"] != "CONVERGED":
        result["reason_codes"].append("WEIGHT_SOLVER_NOT_CONVERGED")
        return result
    try:
        holdout, forecasts = _holdout(validation, heads, calibration, names, bounds)
    except ValueError as error:
        if str(error) == "CONDITIONAL_CALIBRATION_ROWS_REQUIRED":
            result["reason_codes"].append("INDEPENDENT_HOLDOUT_NOT_READY:" + str(error))
            return result
        raise
    holdout.update(schema="v7-independent-holdout/v1", status="COMPLETED", fit_cutoff=config["fit_cutoff"],
                   validation_start=config["validation_start"], validation_end=config["validation_end"],
                   last_label_available_at=max(timestamp(r["label_available_at"]) for r in validation).isoformat(),
                   acceptance=config["acceptance"], scope="REFERENCE_H20_PREDICTIONS_NOT_PORTFOLIO_CAGR")
    # A no-fractal ablation uses the SAME events/folds/support and its own past OOF calibrators.
    baseline_indices = [i for i, name in enumerate(names) if not is_active_fractal_feature(name)]
    baseline_names, baseline_bounds = ([names[i] for i in baseline_indices], [bounds[i] for i in baseline_indices])
    baseline_oof, baseline_folds, baseline_reports = _oof(rows, config, baseline_names, baseline_bounds)
    if is_fr3_reference(config["reference_policy"]):
        result["baseline_folds"] = _baseline_fold_proof(config, baseline_names, baseline_bounds, baseline_folds)
    if config.get("head_solver"):
        reports.update(baseline_reports)
    baseline_mature = [p for p in baseline_oof if timestamp(p["label_available_at"]) <= fit_at]
    try:
        baseline_heads = _fit(training, config, config["fit_cutoff"], baseline_names, baseline_bounds, reports=reports)
        if config.get("head_solver"):
            result["baseline_head_fit_report_id"] = _head_fit_receipt_id(config, reports, baseline_heads, training,
                config["fit_cutoff"], baseline_names)
        baseline_cal, _ = _calibrate(baseline_mature, config, config["fit_cutoff"])
        baseline, _ = _holdout(validation, baseline_heads, baseline_cal, baseline_names, baseline_bounds)
    except ValueError as error:
        result["reason_codes"].append("BASELINE_ABLATION_NOT_READY:" + str(error))
        return result
    null = fractal_null_report(config)
    checks = dict(
        ordered_quantiles=all(list(f.quantiles) == sorted(f.quantiles) for _, f in forecasts),
        known_low_upper_bound=all(f.quantiles[-1] <= r["features"]["low"] + 1e-10 for r, f in forecasts),
        joint_le_rebound=all(f.p_entry <= f.p_rebound for _, f in forecasts),
        timing_sum_one=all(abs(fsum(f.timing) - 1.) <= 1e-9 for _, f in forecasts))
    projection = dict(schema="v7-runtime-projection-validation/v1", status="PASS" if all(checks.values()) else "FAIL",
                      row_count=len(forecasts), checks=checks)
    ablation = dict(schema="v7-fractal-head-ablation/v1", status="COMPLETED", full=holdout, baseline=baseline,
        same_row_ids=holdout["row_ids"] == baseline["row_ids"], scope="FIVE_HEAD_FEATURE_ABLATION_NOT_ALLOCATION_PERFORMANCE")
    for report in (cal_report, holdout, null, projection, ablation):
        reports[identity(report)] = report
    result["reports"] = reports
    # Absolute criteria are declared before fitting; no automatic threshold search.
    if (holdout["pinball_loss"] > config["acceptance"]["max_pinball_loss"]
            or null["status"] != "PASS" or projection["status"] != "PASS" or not ablation["same_row_ids"]):
        result["reason_codes"].append("RESEARCH_FREEZE_VALIDATION_FAILED")
        return result
    metrics = holdout["metrics"]
    if metrics["timing_nll"] > config["acceptance"]["max_timing_nll"] or any(
            value > config["acceptance"]["max_brier"] for value in metrics["brier"].values()):
        result["reason_codes"].append("RESEARCH_FREEZE_ACCEPTANCE_FAILED")
        return result
    ids = dict(calibration=identity(cal_report), holdout=identity(holdout), fractal_null=identity(null),
               projection=identity(projection), ablation=identity(ablation))
    final = dict(feature_profile=FEATURE_VERSION, mapping_version=MAPPING_VERSION,
        feature_names=names, feature_bounds=bounds, imputation=[None] * len(names), supported_masks=[[True] * len(names)],
        heads=heads, weights=weights["weights"], calibration=calibration,
        context_rules=config["context_rules"], trigger_rules=config["trigger_rules"], reference_policy=config["reference_policy"],
        quality_policy=config["quality_policy"], trained_until=None, fit_cutoff=fit_at.isoformat(),
        last_label_available_at=max(timestamp(r["label_available_at"]) for r in training).isoformat(),
        validation_available_at=holdout["last_label_available_at"], training_pipeline_schema=PIPELINE,
        research_only=True, approved_for_live=False, deployment_status="RESEARCH_ONLY_NOT_APPROVED",
        fractal_null_validation_report_id=ids["fractal_null"], validation_report_ids=ids,
        validation_reports={rid: reports[rid] for rid in ids.values()},
        training_manifest=dict(dataset_id=result["dataset_id"], config_id=result["config_id"],
            dataset_kind=dataset_record["payload"]["dataset_kind"], source_authenticity="NOT_INDEPENDENTLY_CERTIFIED",
            upstream_predictions="STRICTLY_PAST_OOF", folds=[f for f in folds if f["status"] == "FITTED"],
            oof_prediction_ids=[p["prediction_id"] for p in predictions], weight_prediction_ids=[r["prediction_id"] for r in weight_rows],
            weight_fit=weights, prior=list(PRIOR), counts=result["counts"], transformations="NONE_RAW_FEATURE_COORDINATES",
            factor_weight_scope="FILLED_REALIZED_NET_PNL_CONDITIONAL_SAMPLES"))
    if validate_execution_profile(config) is not None:
        final.update({key: deepcopy(config[key]) for key in ("execution_feature_profile", "fill_only_feature_extension")})
    if "fill_calibration_profile" in config:
        final["fill_calibration_profile"] = deepcopy(config["fill_calibration_profile"])
    if dataset_record["payload"].get("schema") == "les-v7-sharded-training-dataset/v1":
        manifest = dataset_record["payload"]
        final["training_manifest"].update(dataset_schema=manifest["schema"],
            freeze_id=manifest["freeze_id"], scope_id=manifest["scope_id"],
            full_universe_sha256=identity(manifest["universe"]),
            child_dataset_ids=[child["dataset_id"] for child in manifest["children"]],
            diagnostics_policy=manifest["diagnostics_policy"])
        provenance = archive_provenance(config, ids, [row["prediction_id"] for row in weight_rows], weights)
        result["training_provenance"] = provenance
        final = compact_candidate(final, provenance)
    else:
        final["bundle_id"] = identity(final)
    if len(canonical(dict(payload=final, sha256=identity(final))).encode("utf-8")) > 1024 * 1024:
        result["reason_codes"].append("FROZEN_MODEL_EXCEEDS_RUNTIME_SIZE_LIMIT")
        return result
    result.update(status="READY_TO_FREEZE", candidate=final)
    return result
