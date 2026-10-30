"""Pure, outcome-blind calendar planning for an independent FR3 expansion.

The caller must verify the official calendar before passing its OPEN sessions.
A calendar content ID establishes binding, not authenticity of arbitrary JSON.
This module does not collect prices, create labels, fit models or certify that
the planned sample will satisfy any native training/comparison requirement.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import date

from application.les_v7_dataset import validate_build_config
from domain.les_v7.contracts import identity
from domain.les_v7.fr2_comparison import POLICY as COMPARISON_POLICY
from domain.les_v7.fr3_plan import LABEL_VERSION, PATH_VERSION, PLAN_VERSION, validate_policy
from domain.les_v7.fractals import fractal_history_requirements
from domain.les_v7.quality_policy import r6_quality_policy
from domain.les_v7.training import reference_config_id, validate_training_config

SCHEMA = "les-v7-fr3-expansion-plan/v1"
VERSION = "FR3_ALL_OFFICIAL_SESSIONS_MONTHLY_OOF_LAST_180_R1"
CANDIDATE_NOT_BEFORE = "2024-01-01"
OOF_NOT_BEFORE = "2024-07-01"  # Fixed six-calendar-month initial training period.
LABEL_MATURITY_SESSIONS = 20
HOLDOUT_SESSIONS = 180
MAX_CALENDAR_SESSIONS = 3660
_BUILD_DATE_KEYS = frozenset({"event_start", "event_end", "evaluation_cutoff"})
_TRAIN_DATE_KEYS = frozenset({"fit_cutoff", "validation_start", "validation_end", "evaluation_cutoff", "folds"})


def _day(value, field):
    if not isinstance(value, str):
        raise ValueError("FR3_EXPANSION_CANONICAL_DATE_REQUIRED:" + field)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("FR3_EXPANSION_CANONICAL_DATE_REQUIRED:" + field) from exc
    if parsed.isoformat() != value:
        raise ValueError("FR3_EXPANSION_CANONICAL_DATE_REQUIRED:" + field)
    return parsed


def _content_id(value, field, *, optional=False):
    if optional and value is None:
        return
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("FR3_EXPANSION_CONTENT_ID_REQUIRED:" + field)


def _close(day):
    return day + "T13:30:00+08:00"


def _non_date_identity(config, date_keys):
    return identity({k: v for k, v in config.items() if k not in date_keys})


def _validate_templates(build, training):
    validate_build_config(build)
    validate_training_config(training)
    validate_policy(training["reference_policy"])
    if (training["reference_policy"].get("version") != PLAN_VERSION
            or build["path_policy"].get("version") != PATH_VERSION
            or build["path_policy"].get("label_version") != LABEL_VERSION
            or training["quality_policy"] != r6_quality_policy()
            or build["seed"]["reference_policy"] != training["reference_policy"]
            or build["path_policy"] != training["reference_policy"]["cost_path_policy"]
            or reference_config_id(build["seed"]) != reference_config_id(training)
            or build["evaluation_cutoff"] != training["evaluation_cutoff"]
            or build.get("reference_evidence_policy") != training.get("reference_evidence_policy")):
        raise ValueError("FR3_EXPANSION_ORIGINAL_COMPONENT_BINDING_REQUIRED")
    # Never lower the engineering thresholds to make an expanded run fit.
    if any(training[k] < floor for k, floor in
           (("min_holdout_rows", 500), ("min_context_weight_rows", 100), ("min_weight_rows", 500))):
        raise ValueError("FR3_EXPANSION_NATIVE_SAMPLE_MINIMA_MUST_NOT_BE_LOWERED")


def plan_fr3_expansion(build_template, training_template, *, sessions,
                       history_start, through, universe_id=None, calendar_id=None):
    """Return a detached deterministic plan before inspecting new outcomes.

    ``sessions`` is the complete, verified, strictly increasing OPEN-date list
    for ``history_start..through``. The official calendar may mark ``through``
    CLOSED; it is not silently changed to an OPEN date. No weekday inference is
    performed. Calendar/source verification remains the infrastructure caller's
    responsibility and must happen before it freezes this planning record.

    Only dates/folds are changed in the supplied configurations. All feature,
    head, cost, context, label, quality, acceptance and solver policies remain
    byte-equivalent as business JSON. The result is retrospective development,
    not a fresh confirmatory holdout or a claim of historical deployment.
    """
    _content_id(calendar_id, "calendar_id")
    _content_id(universe_id, "universe_id", optional=True)
    start, end = _day(history_start, "history_start"), _day(through, "through")
    if start >= end or (end - start).days >= 3660:
        raise ValueError("FR3_EXPANSION_BOUNDED_HISTORY_RANGE_REQUIRED")
    if (not isinstance(sessions, (list, tuple))
            or not 1 <= len(sessions) <= MAX_CALENDAR_SESSIONS):
        raise ValueError("FR3_EXPANSION_BOUNDED_VERIFIED_OPEN_SESSIONS_REQUIRED")
    days = list(sessions)
    for day in days:
        if not start <= _day(day, "session") <= end:
            raise ValueError("FR3_EXPANSION_SESSION_OUTSIDE_SOURCE_RANGE")
    if days != sorted(set(days)):
        raise ValueError("FR3_EXPANSION_SORTED_UNIQUE_OPEN_SESSIONS_REQUIRED")
    build, training = deepcopy(build_template), deepcopy(training_template)
    _validate_templates(build, training)
    cost = build["path_policy"]["cost_policy"]
    cost_start = _day(cost["effective_start"], "cost.effective_start")
    cost_end = _day(cost["effective_end"], "cost.effective_end")
    floor = max(_day(CANDIDATE_NOT_BEFORE, "candidate_policy_floor"), cost_start).isoformat()
    # All future H20 observations also need the unchanged cost policy to apply.
    if end > cost_end:
        raise ValueError("FR3_EXPANSION_COST_POLICY_DOES_NOT_COVER_SOURCE_END")
    mature = days[:-LABEL_MATURITY_SESSIONS]
    candidates = [day for day in mature if day >= floor]
    if len(candidates) <= HOLDOUT_SESSIONS:
        raise ValueError("FR3_EXPANSION_INSUFFICIENT_MATURE_OPEN_SESSIONS")
    holdout = candidates[-HOLDOUT_SESSIONS:]
    fit_day = days[days.index(holdout[0]) - 1]
    by_month = defaultdict(list)
    for day in candidates:
        if OOF_NOT_BEFORE <= day < fit_day:
            by_month[day[:7]].append(day)
    folds = []
    for month, prediction_days in sorted(by_month.items()):
        first, last = prediction_days[0], prediction_days[-1]
        fit = days[days.index(first) - 1]
        folds.append(dict(fold_id="FR3_EXPANSION_MONTH_" + month,
            fit_cutoff=_close(fit), prediction_start=_close(first), prediction_end=_close(last)))
    if not 2 <= len(folds) <= 64:
        raise ValueError("FR3_EXPANSION_INSUFFICIENT_MONTHLY_PAST_ONLY_FOLDS")
    evaluation = _close(through)
    build.update(event_start=candidates[0], event_end=candidates[-1], evaluation_cutoff=evaluation)
    training.update(fit_cutoff=_close(fit_day), validation_start=_close(holdout[0]),
        validation_end=_close(holdout[-1]), evaluation_cutoff=evaluation, folds=folds)
    _validate_templates(build, training)
    unchanged_build = _non_date_identity(build_template, _BUILD_DATE_KEYS)
    unchanged_training = _non_date_identity(training_template, _TRAIN_DATE_KEYS)
    if (_non_date_identity(build, _BUILD_DATE_KEYS) != unchanged_build
            or _non_date_identity(training, _TRAIN_DATE_KEYS) != unchanged_training):
        raise ValueError("FR3_EXPANSION_NON_DATE_POLICY_CHANGED")
    event_calendar_days = (date.fromisoformat(candidates[-1]) - date.fromisoformat(candidates[0])).days + 1
    batch_upper = build["max_events"] // event_calendar_days
    if batch_upper < 1:
        raise ValueError("FR3_EXPANSION_NATIVE_EVENT_BATCH_CAPACITY_EXCEEDED")
    requirements = fractal_history_requirements(0)
    feature_requirements = {name: requirements[name]["required_close_count"]
        for name in training["feature_names"] if name in requirements}
    required_closes = max(feature_requirements.values(), default=0)
    windows = [dict(window_id="FR3_EXPANSION_HOLDOUT_" + str(index + 1),
                    sessions=holdout[index * 60:(index + 1) * 60]) for index in range(3)]
    payload = dict(version=VERSION, history_start=history_start, history_end=through,
        calendar_id=calendar_id, sessions_sha256=identity(days), universe_id=universe_id,
        source_sessions=list(days), candidate_sessions=candidates, build_config=build, training_config=training,
        build_config_id=identity(build), training_config_id=identity(training),
        original_build_config_id=identity(build_template), original_training_config_id=identity(training_template),
        reference_config_id=reference_config_id(training), unchanged_build_policy_id=unchanged_build,
        unchanged_training_policy_id=unchanged_training,
        date_policy=dict(candidate_not_before=CANDIDATE_NOT_BEFORE, effective_candidate_floor=floor,
            initial_training_calendar_months=6, oof_not_before=OOF_NOT_BEFORE,
            oof_sampling="ALL_AVAILABLE_OFFICIAL_OPEN_SESSIONS_MONTHLY_WINDOWS",
            final_holdout_sessions=HOLDOUT_SESSIONS, label_maturity_sessions=LABEL_MATURITY_SESSIONS,
            final_fit_rule="PREVIOUS_OFFICIAL_OPEN_BEFORE_HOLDOUT",
            oof_fit_rule="PREVIOUS_OFFICIAL_OPEN_BEFORE_MONTH_PREDICTION_WINDOW"),
        comparison_windows=windows, comparison_policy_id=identity(COMPARISON_POLICY),
        native_minima={k: training[k] for k in
            ("min_head_rows", "min_calibration_rows", "min_weight_rows", "min_context_weight_rows", "min_holdout_rows")},
        capacity=dict(source_open_sessions=len(days), candidate_open_sessions=len(candidates),
            holdout_open_sessions=len(holdout), monthly_oof_folds=len(folds),
            event_calendar_days=event_calendar_days, batch_size_upper_bound=batch_upper,
            feature_required_close_count=required_closes,
            official_open_capacity_at_first_candidate=days.index(candidates[0]) + 1,
            individual_stock_bar_coverage_verified=False,
            sample_upper_bounds_verified=False),
        status="CALENDAR_PLAN_ONLY_NOT_TRAINING_READY", model_fitted=False,
        source_authenticity="CALLER_MUST_VERIFY_OFFICIAL_CALENDAR_CONTENT_AND_SCOPE",
        holdout_interpretation="RETROSPECTIVE_DEVELOPMENT_NOT_FRESH_CONFIRMATION",
        historical_deployment_claim=False, predeclared_before_expanded_outcome_inspection=True,
        registration_clock_policy="RETAIN_ACTUAL_FREEZE_TIME_SEPARATELY_NEVER_BACKDATE",
        native_prepared_reuse_allowed=False, research_only=True, approved_for_live=False)
    return dict(schema=SCHEMA, plan_id=identity(payload), payload=payload)


def verify_fr3_expansion_plan(record):
    """Recompute the pure planning record, not the external calendar proof."""
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("FR3_EXPANSION_PLAN_SCHEMA_REQUIRED")
    body = record.get("payload", {})
    if record.get("plan_id") != identity(body):
        raise ValueError("FR3_EXPANSION_PLAN_HASH_MISMATCH")
    # Replanning from the new configs is deterministic because no outcome or
    # prior date partition controls the policy. Preserve the true parent IDs.
    expected = plan_fr3_expansion(body["build_config"], body["training_config"],
        sessions=body["source_sessions"], history_start=body["history_start"], through=body["history_end"],
        calendar_id=body["calendar_id"], universe_id=body["universe_id"])["payload"]
    expected["original_build_config_id"] = body["original_build_config_id"]
    expected["original_training_config_id"] = body["original_training_config_id"]
    if body != expected:
        raise ValueError("FR3_EXPANSION_PLAN_RECOMPUTATION_MISMATCH")
    return deepcopy(record)
