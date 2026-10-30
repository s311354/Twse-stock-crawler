"""Expand frozen-fit FR2 diagnostics through the existing native dataset core.

This port does not backdate model publication, refit heads, grant admission or
simulate an account. Official final bars are an explicit retrospective research
convention. Historical safety and actual execution evidence remain separate.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from zoneinfo import ZoneInfo

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.fr2_comparison import ARMS, POLICY, REPORT_SCHEMA, _event, _forecast_comparison
from domain.les_v7.fractals import is_active_fractal_feature
from domain.les_v7.training import _holdout, prepare_events, reference_config_id
from infrastructure.les_v7_dataset_scope import verify_elec_scope
from infrastructure.les_v7_fr2_archive import _read, _reference_costs
from infrastructure.les_v7_fr2_m0 import load_m0_bundle
from infrastructure.storage.les_v7_training import LesV7TrainingRepository


def _assert_pair(config, baseline, fractal):
    names = config["feature_names"]
    plain = [n for n in names if not is_active_fractal_feature(n)]
    plain_bounds = [b for n, b in zip(names, config["feature_bounds"]) if not is_active_fractal_feature(n)]
    if (fractal["feature_names"] != names or baseline["feature_names"] != plain
            or fractal["feature_bounds"] != config["feature_bounds"] or baseline["feature_bounds"] != plain_bounds
            or not plain or plain == names
            or baseline["fractal_mode"] != "V7_NO_FRACTAL_BASELINE"):
        raise ValueError("FR2_EXPANSION_MATCHED_FROZEN_PAIR_REQUIRED")
    fit = timestamp(config["fit_cutoff"])
    for model in (baseline, fractal):
        if (timestamp(model["fit_cutoff"]) != fit
                or timestamp(model["last_label_available_at"]) > fit):
            raise ValueError("FR2_EXPANSION_MODEL_FIT_ISOLATION_INVALID")
    return plain


def compare_prepared(rows, config, baseline, fractal):
    """Same native inference math as training holdout, not a runtime time bypass.

    The real publication clocks are untouched. A frozen fit can be studied after
    the fact, but this result cannot be passed off as a historical deployment.
    Every event must be strictly later than fit and fully mature at evaluation.
    """
    plain = _assert_pair(config, baseline, fractal)
    fit, evaluated = timestamp(config["fit_cutoff"]), timestamp(config["evaluation_cutoff"])
    if (not rows or len(rows) > 50000 or len({r["row_id"] for r in rows}) != len(rows)
            or any(not fit < timestamp(r["cutoff"]) < timestamp(r["label_available_at"]) <= evaluated
                   for r in rows)):
        raise ValueError("FR2_EXPANSION_MATURE_UNIQUE_OOS_ROWS_REQUIRED")
    rows = sorted(rows, key=lambda r: (timestamp(r["cutoff"]), r["symbol"], r["market"]))
    bounds = [b for n, b in zip(config["feature_names"], config["feature_bounds"])
              if not is_active_fractal_feature(n)]
    left, lf = _holdout(rows, baseline["heads"], baseline["calibration"], plain, bounds)
    right, rf = _holdout(rows, fractal["heads"], fractal["calibration"], config["feature_names"], config["feature_bounds"])
    expected = [r["row_id"] for r in rows]
    if left["row_ids"] != expected or right["row_ids"] != expected or left["label_sha256"] != right["label_sha256"]:
        raise ValueError("FR2_EXPANSION_FORECAST_PAIRING_MISMATCH")

    def events(forecasts):
        return [dict(row_id=row["row_id"], symbol=row["symbol"], market=row["market"], cutoff=row["cutoff"],
            label_available_at=row["label_available_at"], bottom=row["bottom"], close=row["features"]["close"],
            atr=row["plan"].atr, labels=row["labels"], observation_weight=row["observation_weight"],
            forecast=forecast.record()) for row, forecast in forecasts]

    le, re = events(lf), events(rf)
    days = [timestamp(r["cutoff"]).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat() for r in rows]
    comparison = _forecast_comparison([_event(r, config, evaluated) for r in le],
        [_event(r, config, evaluated) for r in re], days)
    return deepcopy(dict(comparison=comparison, events=dict(M0=le, M1=re), sessions=sorted(set(days)),
        native_metrics=dict(M0=left, M1=right)))


def expand_archive(archive, m0_archive, scope_job, *, progress=None):
    """Verify three immutable inputs, then prepare one bounded shard at a time."""
    archive, scope_job = Path(archive).expanduser().resolve(), Path(scope_job).expanduser().resolve()
    if archive.name != "frozen_bundle.json":
        raise ValueError("FR2_EXPANSION_ORIGINAL_BUNDLE_REQUIRED")
    original_receipt = LesV7TrainingRepository.verify(archive.parent)
    original = _read(archive)
    baseline, m0_receipt = load_m0_bundle(m0_archive)
    if original["sha256"] != original_receipt["bundle_sha256"]:
        raise ValueError("FR2_EXPANSION_ORIGINAL_BUNDLE_CHANGED")
    # A model of an unrelated fit must never masquerade as this ablation.
    m0_body, m1_body = baseline["payload"], original["payload"]
    if m0_receipt.get("original_bundle_sha256") != original["sha256"]:
        raise ValueError("FR2_EXPANSION_M0_PARENT_MISMATCH")
    config = _read(archive.parent / "config.json")
    receipt = verify_elec_scope(scope_job)
    root = scope_job.parent.parent
    build = _read(scope_job / "config.json")
    freeze = _read(root / "freezes" / receipt["freeze_id"] / "manifest.json")["body"]
    old_dataset = _read(archive.parent / "dataset.json")
    if (receipt["dataset_kind"] != "RESEARCH" or freeze["dataset_kind"] != "RESEARCH"
            or [[s, "TWSE"] for s in freeze["universe"]] != old_dataset["payload"]["universe"]
            or reference_config_id(build["seed"]) != reference_config_id(config)
            or build["path_policy"] != config["reference_policy"]["cost_path_policy"]
            or build["reference_evidence_policy"] != config["reference_evidence_policy"]
            or timestamp(build["event_start"] + "T13:30:00+08:00") <= timestamp(config["fit_cutoff"])):
        raise ValueError("FR2_EXPANSION_FIXED_UNIVERSE_OR_REFERENCE_CHANGED")
    # Evaluation controls label maturity only. The fit, features, rules, costs,
    # heads, calibration and weights are never recomputed or overwritten.
    preparation = dict(config, evaluation_cutoff=build["evaluation_cutoff"])
    _assert_pair(preparation, m0_body, m1_body)
    rows, diagnostics, children = [], [], []
    for index, batch in enumerate(receipt["batches"]):
        folder = root / "builds" / batch["job_id"]
        packet = _read(folder / "dataset.json")
        if packet["sha256"] != batch["dataset_id"] or identity(packet["payload"]) != batch["dataset_id"]:
            raise ValueError("FR2_EXPANSION_DATASET_CHANGED_AFTER_VERIFICATION")
        prepared, rejected = prepare_events(packet, preparation)
        rows.extend(prepared)
        diagnostics.extend({k: r[k] for k in ("row_id", "symbol", "market", "cutoff", "context", "excluded_reason", "label_status")
                            if k in r} for r in rejected)
        children.append(dict(archive_path=str(folder.relative_to(root)), dataset_id=packet["sha256"]))
        if progress:
            progress(index + 1, len(receipt["batches"]), len(prepared))
        del packet, prepared, rejected
    computed = compare_prepared(rows, preparation, m0_body, m1_body)
    costs, executions = _reference_costs(root, dict(payload=dict(children=children)), preparation, rows)
    known_days = {timestamp(config[k]).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
                  for k in ("validation_start", "validation_end")}
    first_known, last_known = sorted(known_days)
    subsets = {}
    for name, selected in (("PREVIOUSLY_INSPECTED_DEVELOPMENT", [r for r in rows if first_known <= timestamp(r["cutoff"]).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat() <= last_known]),
                           ("ADDITIONAL_RETROSPECTIVE_DIAGNOSTIC", [r for r in rows if not first_known <= timestamp(r["cutoff"]).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat() <= last_known])):
        if selected:
            result = compare_prepared(selected, preparation, m0_body, m1_body)
            subsets[name] = dict(sessions=result["sessions"], paired_events=len(selected), comparison=result["comparison"])
    reasons = ["RETROSPECTIVE_FROZEN_FIT_NOT_HISTORICAL_DEPLOYMENT", "DEVELOPMENT_OR_NON_PREDECLARED_HOLDOUT",
        "THREE_DISJOINT_60_SESSION_WINDOWS_REQUIRED", "HISTORICAL_SAFETY_AND_OPEN_QUOTE_PROOFS_REQUIRED",
        "FOUR_EXECUTED_AFTER_COST_ACCOUNTS_NOT_AVAILABLE"]
    source = dict(original_bundle_id=original["sha256"], m0_bundle_id=baseline["sha256"],
        scope_job_id=receipt["job_id"], freeze_id=receipt["freeze_id"], universe_id=identity(freeze["universe"]),
        scope_completion_id=identity(_read(scope_job / "completion.json")), training_config_id=identity(config),
        preparation_config_id=identity(preparation), evaluation_cutoff=preparation["evaluation_cutoff"],
        original_fit_cutoff=config["fit_cutoff"], actual_model_publication=dict(M0=m0_body["trained_until"], M1=m1_body["trained_until"]),
        reference_policy_id=identity(config["reference_policy"]), original_heads_unchanged=True,
        m0_weights_id=identity(m0_body["weights"]), m1_weights_id=identity(m1_body["weights"]))
    arms = {name: dict(status="NOT_ASSESSABLE", model_kind="FROZEN_NO_FRACTAL" if name.startswith("M0") else "FROZEN_FRACTAL",
        entry_policy="ORIGINAL_BUNDLE" if name.endswith("P0") else "FR_ENTRY_R2", forecast_status="RECOMPUTED",
        admission_status="NOT_ASSESSABLE", after_cost_status="NOT_ASSESSABLE", reason_codes=reasons[3:]) for name in ARMS}
    report = dict(schema=REPORT_SCHEMA, mode="EXPANDED_RETROSPECTIVE_FROZEN_FIT_DIAGNOSTIC", status="NOT_ASSESSABLE",
        acceptance_policy=POLICY, source_bindings=source, arms=arms, sessions=computed["sessions"],
        counts=dict(fixed_universe=len(freeze["universe"]), paired_forecast_events=len(rows),
            sessions=len(computed["sessions"]), distinct_symbols=len({r["symbol"] for r in rows}),
            candidate_events=receipt["summary"]["candidate_events"], contexts=dict(Counter(r["context"] for r in rows))),
        candidate_gap_counts=receipt["summary"]["reason_counts"], preparation_gap_counts=dict(Counter(
            r["excluded_reason"] for r in diagnostics if r.get("excluded_reason"))),
        forecast_comparisons=dict(P0=computed["comparison"], P1=computed["comparison"]), subsets=subsets,
        reference_cost_diagnostic=costs, native_metrics=computed["native_metrics"], reason_codes=reasons,
        new_head_fits=0, calibration_refits=0, historical_information_set_certified=False,
        research_only=True, approved_for_live=False, execution_action="NO_ACTION")
    forecasts = dict(source_bindings=source, events=computed["events"], reference_execution_rows=executions,
        preparation_diagnostics=diagnostics)
    return deepcopy(report), deepcopy(forecasts)
