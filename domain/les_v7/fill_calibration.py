"""Optional fill-only Platt fit with a fixed, predeclared identity prior.

This is fitted on matured, strictly past OOF predictions. It neither selects
its own prior strength nor changes the conditional heads or native acceptance
thresholds. Calibration-set improvement is not out-of-time evidence.
"""
from __future__ import annotations

from math import isfinite

import numpy as np

from domain.les_v7.contracts import finite, identity, timestamp
from domain.les_v7.research import _lbfgs

METHOD = "WEIGHTED_BINARY_NLL_IDENTITY_PRIOR_PLATT"
VERSION = "V7_FILL_CALIBRATION_IDENTITY_PRIOR_R1"
STRENGTH = 1.
ITERATIONS = 2000
MAX_ROWS = 100000


def profile(alpha=None):
    """Return a fresh exact declaration; none of these fields are tunable."""
    result = dict(schema="les-v7-fill-calibration-profile/v1", method=METHOD, version=VERSION,
        strength=STRENGTH, prior_pair=[1., 0.], iterations=ITERATIONS,
        solver="NATIVE_LBFGS_R2", max_rows=MAX_ROWS,
        objective="WEIGHTED_MEAN_BINARY_NLL_PLUS_STRENGTH_TIMES_SQUARED_IDENTITY_DISTANCE",
        penalty="strength*((slope-1)^2+intercept^2)",
        maturity="HEAD_LAST_LABEL_LE_HEAD_FIT_LT_PREDICTION_LT_LABEL_AVAILABLE_LE_CALIBRATION_FIT",
        other_heads="UNCHANGED_NATIVE_R1", no_identity_fallback=True)
    if alpha is not None:
        alpha = _alpha(alpha)
        label = {.25: "025", .5: "050", 1.: "100"}[alpha]
        result.update(schema="les-v7-fill-calibration-profile/v2",
            version="V7_FILL_CALIBRATION_IDENTITY_PRIOR_SHRINK_R2_ALPHA" + label,
            alpha=alpha, deployment="FIXED_SHRINK_AFTER_ACTUAL_CONVERGED_PRIOR1_FIT",
            deployed_pair="[1+alpha*(optimizer_slope-1),alpha*optimizer_intercept]",
            alpha_selection="SINGLE_PREDECLARED_ALPHA_NO_SELECTION_IN_THIS_FITTER")
    return result


def _alpha(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in {.25, .5, 1.}:
        raise ValueError("FILL_CALIBRATION_SHRINK_ALPHA_INVALID")
    return float(value)


def validate_profile(declared):
    """Accept exact legacy R1 or one predeclared fixed-alpha R2 profile."""
    if isinstance(declared, dict) and declared.get("schema") == "les-v7-fill-calibration-profile/v3":
        from domain.les_v7.fill_slope_calibration import validate_profile as validate_slope
        return validate_slope(declared)
    if declared == profile():
        return profile()
    if isinstance(declared, dict) and "alpha" in declared:
        expected = profile(declared["alpha"])
        if declared == expected:
            return expected
    raise ValueError("FILL_CALIBRATION_PROFILE_MISMATCH")


def deploy_fill_pair(pair, alpha):
    """The shared fixed deployment transform; never fit or select its alpha."""
    alpha = _alpha(alpha)
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError("FILL_CALIBRATION_PAIR_REQUIRED")
    slope, intercept = (_number(value, "fill_pair") for value in pair)
    return [_number(1. + alpha * (slope - 1.), "deployed_fill_slope"),
            _number(alpha * intercept, "deployed_fill_intercept")]


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError("CALIBRATION_FILL_FINITE_NUMBER_REQUIRED:" + name)
    return finite(value, name)


def _time(value):
    try:
        return timestamp(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("CALIBRATION_FILL_AWARE_PROVENANCE_TIME_REQUIRED") from error


def _inputs(rows, fit_cutoff):
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("CALIBRATION_FILL_BOUNDED_OOF_ROWS_REQUIRED")
    fit_at = _time(fit_cutoff)
    seen, logits, labels, weights, provenance = set(), [], [], [], []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("CALIBRATION_FILL_OOF_ROW_REQUIRED")
        key = row.get("row_id")
        if not isinstance(key, str) or not 1 <= len(key) <= 256 or key in seen:
            raise ValueError("CALIBRATION_FILL_UNIQUE_OOF_ROW_IDS_REQUIRED")
        seen.add(key)
        cutoff = _time(row.get("cutoff"))
        available = _time(row.get("label_available_at"))
        head_fit = _time(row.get("head_fit_cutoff"))
        head_last = _time(row.get("head_last_label_available_at"))
        if not head_last <= head_fit < cutoff < available <= fit_at:
            raise ValueError("CALIBRATION_FILL_STRICTLY_PAST_MATURE_OOF_REQUIRED")
        fill = row.get("labels", {}).get("fill")
        if type(fill) is not int or fill not in (0, 1):
            raise ValueError("CALIBRATION_FILL_OBSERVED_BINARY_LABEL_REQUIRED")
        logit = _number(row.get("raw_logits", {}).get("fill"), "fill_logit")
        weight = _number(row.get("observation_weight", 1.), "observation_weight")
        if weight <= 0:
            raise ValueError("CALIBRATION_FILL_POSITIVE_WEIGHT_REQUIRED")
        logits.append(logit)
        labels.append(fill)
        weights.append(weight)
        provenance.append(dict(row_id=key, cutoff=cutoff.isoformat(), label_available_at=available.isoformat(),
            head_fit_cutoff=head_fit.isoformat(), head_last_label_available_at=head_last.isoformat(),
            raw_fill_logit=logit, fill=fill, observation_weight=weight))
    if set(labels) != {0, 1}:
        raise ValueError("BOTH_FILL_CALIBRATION_CLASSES_REQUIRED")
    w = np.asarray(weights, dtype=float)
    w /= w.max()
    w /= w.sum()
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("CALIBRATION_FILL_WEIGHT_DYNAMIC_RANGE_UNREPRESENTABLE")
    return np.asarray(logits), np.asarray(labels, dtype=float), w, provenance


def _metrics(logits, labels, weights, pair):
    z = pair[0] * logits + pair[1]
    p = np.exp(-np.logaddexp(0., -z))
    return dict(fill_brier=float(weights @ ((p - labels) ** 2)),
        fill_nll=float(weights @ (np.logaddexp(0., z) - labels * z)),
        mean_predicted_fill=float(weights @ p), observed_fill_fraction=float(weights @ labels))


def fit_fill_calibration(rows, *, fit_cutoff, calibration_profile=None):
    """Fit a real convex two-parameter model; reject failed fits, never fallback."""
    declared = profile() if calibration_profile is None else validate_profile(calibration_profile)
    if declared.get("schema") == "les-v7-fill-calibration-profile/v3":
        from domain.les_v7.fill_slope_calibration import fit
        return fit(rows, fit_cutoff=fit_cutoff, declared=declared)
    logits, labels, weights, provenance = _inputs(rows, fit_cutoff)
    initial = np.asarray([1., 0.])

    def objective(pair):
        with np.errstate(over="ignore", invalid="ignore"):
            z = pair[0] * logits + pair[1]
            p = np.exp(-np.logaddexp(0., -z))
            residual = weights * (p - labels)
            value = float(weights @ (np.logaddexp(0., z) - labels * z))
            gradient = np.asarray([residual @ logits, residual.sum()])
            distance = pair - initial
            value += STRENGTH * float(distance @ distance)
            gradient += 2 * STRENGTH * distance
        if not isfinite(value) or not np.isfinite(gradient).all():
            raise ValueError("CALIBRATION_FILL_NONFINITE_OBJECTIVE")
        return value, gradient

    pair, solver = _lbfgs(objective, initial, ITERATIONS)
    final_value, final_gradient = objective(pair)
    initial_value, _ = objective(initial)
    if (solver.get("converged") is not True or not np.isfinite(pair).all()
            or float(np.linalg.norm(final_gradient, ord=np.inf)) > 1e-6
            or final_value > initial_value + 1e-12):
        raise ValueError("CALIBRATION_FILL_SOLVER_NOT_CONVERGED")
    fitted = pair.tolist()
    report = dict(schema="les-v7-fill-calibration-fit/v1", method=METHOD, version=VERSION,
        profile=profile(), fit_cutoff=timestamp(fit_cutoff).isoformat(),
        last_label_available_at=max(timestamp(r["label_available_at"]) for r in rows).isoformat(),
        data_sha256=identity(provenance), row_ids=[r["row_id"] for r in rows], row_count=len(rows),
        input_provenance="STRICTLY_PAST_MATURE_HEAD_OOF_PREDICTIONS",
        fitted_pair=fitted, initial_pair=initial.tolist(), solver_report=solver,
        converged=True, initial_objective=initial_value, final_objective=final_value,
        before=_metrics(logits, labels, weights, initial), after=_metrics(logits, labels, weights, pair),
        evaluation_scope="CALIBRATION_SET_NOT_HOLDOUT", fit_status="FITTED_CONVERGED_NO_FALLBACK",
        other_heads="UNCHANGED_NATIVE_R1", research_only=True, approved_for_live=False,
        execution_action="NO_ACTION")
    if declared == profile():
        return dict(pair=fitted, report=report)
    deployed = deploy_fill_pair(fitted, declared["alpha"])
    # The optimizer objective belongs to its actual fitted pair, while the
    # deployed pair has an explicitly declared shrink transform. Retain the
    # complete optimizer receipt instead of relabelling its final objective.
    deployed_report = dict(schema="les-v7-fill-calibration-fit/v2", method=METHOD,
        version=declared["version"], profile=declared, fit_cutoff=report["fit_cutoff"],
        last_label_available_at=report["last_label_available_at"], data_sha256=report["data_sha256"],
        row_ids=list(report["row_ids"]), row_count=report["row_count"],
        input_provenance=report["input_provenance"], fitted_pair=deployed,
        optimizer_fitted_pair=fitted, optimizer_fit_report=report, converged=True,
        before=report["before"], after=_metrics(logits, labels, weights, deployed),
        evaluation_scope="CALIBRATION_SET_NOT_HOLDOUT",
        fit_status="FITTED_CONVERGED_THEN_PREDECLARED_SHRINK_NO_FALLBACK",
        other_heads="UNCHANGED_NATIVE_R1", research_only=True, approved_for_live=False,
        execution_action="NO_ACTION")
    return dict(pair=deployed, report=deployed_report)
