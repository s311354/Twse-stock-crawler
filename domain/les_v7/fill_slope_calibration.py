"""Opt-in past-OOF sigmoid calibration with a free base rate.

The standardized slope is regularized toward zero, allowing weak ranking
evidence to produce modest probability differences. The intercept is fitted
without a penalty. Strength selection belongs to a separately frozen study;
this fitter accepts only one exact declaration and never reads a holdout.
"""
from __future__ import annotations

from math import isfinite

import numpy as np

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.research import _lbfgs

VERSION = "V7_FILL_STANDARDIZED_ZERO_SLOPE_FREE_INTERCEPT_R3"
STRENGTHS = (.001, .01, .1, 1.)
SCALE_FLOOR = 1e-8


def profile(strength):
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or strength not in STRENGTHS:
        raise ValueError("FILL_SLOPE_CALIBRATION_STRENGTH_INVALID")
    return dict(schema="les-v7-fill-calibration-profile/v3", method="STANDARDIZED_WEIGHTED_NLL_SLOPE_RIDGE",
        version=VERSION + "_L" + {.001: "0001", .01: "001", .1: "01", 1.: "1"}[strength],
        strength=float(strength), iterations=2000, solver="NATIVE_LBFGS_R2", max_rows=100000,
        objective="WEIGHTED_MEAN_BINARY_NLL_PLUS_STRENGTH_TIMES_STANDARDIZED_SLOPE_SQUARED",
        standardization="PAST_OOF_WEIGHTED_MEAN_AND_POPULATION_STD", scale_floor=SCALE_FLOOR,
        slope_prior=0., intercept_penalty=0., initial="ZERO_SLOPE_WEIGHTED_PREVALENCE_INTERCEPT",
        maturity="HEAD_LAST_LABEL_LE_HEAD_FIT_LT_PREDICTION_LT_LABEL_AVAILABLE_LE_CALIBRATION_FIT",
        other_heads="UNCHANGED_NATIVE_R1", no_identity_fallback=True,
        selection="SINGLE_PREDECLARED_STRENGTH_NO_SELECTION_IN_THIS_FITTER")


def validate_profile(declared):
    if isinstance(declared, dict) and declared == profile(declared.get("strength")):
        return profile(declared["strength"])
    raise ValueError("FILL_SLOPE_CALIBRATION_PROFILE_MISMATCH")


def fit(rows, *, fit_cutoff, declared):
    from domain.les_v7.fill_calibration import _inputs, _metrics
    declared = validate_profile(declared)
    logits, labels, weights, provenance = _inputs(rows, fit_cutoff)
    center = float(weights @ logits)
    scale = max(SCALE_FLOOR, float(np.sqrt(weights @ ((logits - center) ** 2))))
    x = (logits - center) / scale
    prevalence = float(weights @ labels)
    initial = np.asarray([0., np.log(prevalence) - np.log1p(-prevalence)])
    strength = declared["strength"]

    def objective(pair):
        with np.errstate(over="ignore", invalid="ignore"):
            z = pair[0] * x + pair[1]
            p = np.exp(-np.logaddexp(0., -z))
            residual = weights * (p - labels)
            value = float(weights @ (np.logaddexp(0., z) - labels * z)) + strength * pair[0] ** 2
            gradient = np.asarray([residual @ x + 2 * strength * pair[0], residual.sum()])
        if not isfinite(value) or not np.isfinite(gradient).all():
            raise ValueError("FILL_SLOPE_CALIBRATION_NONFINITE_OBJECTIVE")
        return value, gradient

    standardized, solver = _lbfgs(objective, initial, declared["iterations"])
    final_value, gradient = objective(standardized)
    initial_value, _ = objective(initial)
    if (solver.get("converged") is not True or not np.isfinite(standardized).all()
            or float(np.linalg.norm(gradient, ord=np.inf)) > 1e-6
            or final_value > initial_value + 1e-12):
        raise ValueError("FILL_SLOPE_CALIBRATION_SOLVER_NOT_CONVERGED")
    pair = [float(standardized[0] / scale), float(standardized[1] - standardized[0] * center / scale)]
    if not np.isfinite(pair).all():
        raise ValueError("FILL_SLOPE_CALIBRATION_NONFINITE_DEPLOYED_PAIR")
    report = dict(schema="les-v7-fill-calibration-fit/v3", method=declared["method"],
        version=declared["version"], profile=declared, fit_cutoff=timestamp(fit_cutoff).isoformat(),
        last_label_available_at=max(timestamp(row["label_available_at"]) for row in rows).isoformat(),
        data_sha256=identity(provenance), row_ids=[row["row_id"] for row in rows], row_count=len(rows),
        input_provenance="STRICTLY_PAST_MATURE_HEAD_OOF_PREDICTIONS",
        weighted_logit_center=center, weighted_logit_scale=scale, weighted_fill_prevalence=prevalence,
        standardized_fitted_pair=standardized.tolist(), initial_standardized_pair=initial.tolist(),
        fitted_pair=pair, solver_report=solver, converged=True,
        initial_objective=initial_value, final_objective=final_value,
        gradient_inf_norm=float(np.linalg.norm(gradient, ord=np.inf)),
        before=_metrics(logits, labels, weights, [1., 0.]), after=_metrics(logits, labels, weights, pair),
        evaluation_scope="CALIBRATION_SET_NOT_HOLDOUT", fit_status="FITTED_CONVERGED_NO_FALLBACK",
        other_heads="UNCHANGED_NATIVE_R1", research_only=True, approved_for_live=False,
        execution_action="NO_ACTION")
    return dict(pair=pair, report=report)


def verify_component(report):
    """Check serialized standardized/raw coordinates; native replay verifies fit."""
    declared = validate_profile(report.get("profile"))
    try:
        center, scale = float(report["weighted_logit_center"]), float(report["weighted_logit_scale"])
        slope, intercept = map(float, report["standardized_fitted_pair"])
        pair = [slope / scale, intercept - slope * center / scale]
        if (not np.isfinite([center, scale, slope, intercept, *pair,
                report["gradient_inf_norm"], report["final_objective"], report["initial_objective"]]).all() or scale < SCALE_FLOOR
                or report["fitted_pair"] != pair or report.get("version") != declared["version"]
                or report.get("converged") is not True or report["solver_report"].get("converged") is not True
                or not 0 <= report["gradient_inf_norm"] <= 1e-6
                or report["final_objective"] > report["initial_objective"] + 1e-12):
            raise ValueError("FILL_SLOPE_CALIBRATION_COMPONENT_BINDING_MISMATCH")
    except (KeyError, TypeError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("FILL_SLOPE_CALIBRATION_COMPONENT_BINDING_MISMATCH") from error
