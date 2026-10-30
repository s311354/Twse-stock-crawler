"""Bounded, deterministic calibration of strictly-past OOF logits.

The caller proves OOF and cutoff provenance. These helpers fit probabilities,
not trading permission, and report calibration-set metrics, not future results.
"""
from __future__ import annotations

import math
import numpy as np

from domain.les_v7.contracts import finite, identity

METHOD = "POST_PROJECTION_PLATT_TEMPERATURE_R1"
VERSION = "V7_OOF_CALIBRATION_R1"
_HEADS = ("fill", "rebound", "joint")
_EPS = 1e-15
_MAX_ROWS = 250000


def _inputs(rows, *, require_classes=False):
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= _MAX_ROWS:
        raise ValueError("BOUNDED_CALIBRATION_ROWS_REQUIRED")
    ids, logits, timing, labels, weights = set(), [], [], [], []
    for row in rows:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not 1 <= len(row_id) <= 256 or row_id in ids:
            raise ValueError("UNIQUE_CALIBRATION_ROW_IDS_REQUIRED")
        ids.add(row_id)
        raw, target = row.get("raw_logits", {}), row.get("labels", {})
        logits.append([finite(raw.get(key), key + "_logit") for key in _HEADS])
        values = row.get("timing_logits")
        if not isinstance(values, (list, tuple)) or len(values) != 21:
            raise ValueError("TWENTY_ONE_TIMING_LOGITS_REQUIRED")
        timing.append([finite(value, "timing_logit") for value in values])
        fill = target.get("fill")
        if type(fill) is not int or fill not in {0, 1}:
            raise ValueError("OBSERVED_BINARY_FILL_REQUIRED")
        rebound, joint = target.get("rebound"), target.get("joint")
        if fill == 0 and (rebound is not None or joint is not None):
            raise ValueError("NO_FILL_CONDITIONAL_LABELS_MUST_BE_NULL")
        if fill == 1 and (type(rebound) is not int or type(joint) is not int
                          or rebound not in {0, 1} or joint not in {0, 1} or joint > rebound):
            raise ValueError("INVALID_CONDITIONAL_EVENT_LABELS")
        bucket = target.get("timing")
        if type(bucket) is not int or not 0 <= bucket <= 20:
            raise ValueError("INVALID_TIMING_LABEL")
        labels.append((fill, rebound if fill else 0, joint if fill else 0, bucket))
        weight = finite(row.get("observation_weight", 1.), "observation_weight")
        if weight <= 0:
            raise ValueError("POSITIVE_CALIBRATION_WEIGHT_REQUIRED")
        weights.append(weight)
    x, t, y = np.asarray(logits), np.asarray(timing), np.asarray(labels, dtype=int)
    w = np.asarray(weights)
    # Scaling by the maximum avoids overflow without changing relative weights.
    w /= w.max()
    w /= w.sum()
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("CALIBRATION_WEIGHT_DYNAMIC_RANGE_UNREPRESENTABLE")
    keep = y[:, 0] == 1
    if not keep.any():
        raise ValueError("CONDITIONAL_CALIBRATION_ROWS_REQUIRED")
    conditional = np.where(keep, w, 0.)
    conditional /= conditional.sum()
    if require_classes:
        for i, head in enumerate(_HEADS):
            values = y[:, i] if i == 0 else y[keep, i]
            if set(values.tolist()) != {0, 1}:
                raise ValueError("BOTH_CALIBRATION_CLASSES_REQUIRED:" + head)
    return x, t, y, w, conditional


def _parameters(calibration):
    if not isinstance(calibration, dict):
        raise ValueError("CALIBRATION_PARAMETERS_REQUIRED")
    result = []
    for key in _HEADS:
        pair = calibration.get(key)
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("INVALID_PLATT_CALIBRATION")
        result.extend(finite(value, "calibration") for value in pair)
    temperature = finite(calibration.get("timing_temperature"), "timing_temperature")
    if temperature <= 0:
        raise ValueError("POSITIVE_TIMING_TEMPERATURE_REQUIRED")
    return np.asarray([*result, math.log(temperature)])


def _outputs(x, timing, parameters):
    binary = parameters[:6].reshape(3, 2)
    z = x * binary[:, 0] + binary[:, 1]
    probabilities = np.exp(-np.logaddexp(0., -z))
    projected = probabilities.copy()
    crossed = probabilities[:, 2] > probabilities[:, 1]
    projected[crossed, 1:] = probabilities[crossed, 1:].mean(axis=1, keepdims=True)
    # Subtract before dividing to keep a finite, stable softmax coordinate.
    q = (timing - timing.max(axis=1, keepdims=True)) / math.exp(parameters[6])
    exp = np.exp(q)
    pmf = exp / exp.sum(axis=1, keepdims=True)
    return probabilities, projected, crossed, q, pmf


def _loss_gradient(parameters, inputs, regularization):
    x, timing, y, weights, conditional = inputs
    raw, projected, crossed, q, pmf = _outputs(x, timing, parameters)
    weight_matrix = np.column_stack((weights, conditional, conditional))
    p = np.clip(projected, _EPS, 1 - _EPS)
    terms = -y[:, :3] * np.log(p) - (1 - y[:, :3]) * np.log1p(-p)
    losses = (weight_matrix * terms).sum(axis=0)
    # Differentiation is performed AFTER runtime's J/R isotonic projection.
    dp = weight_matrix * (p - y[:, :3]) / (p * (1 - p))
    dp[(projected <= _EPS) | (projected >= 1 - _EPS)] = 0.
    dp[crossed, 1:] = dp[crossed, 1:].sum(axis=1, keepdims=True) / 2
    dz = dp * raw * (1 - raw)
    gradient = np.zeros(7)
    gradient[:6:2] = (dz * x).sum(axis=0)
    gradient[1:6:2] = dz.sum(axis=0)
    log_partition = np.log(np.exp(q).sum(axis=1))
    timing_nll = float(weights @ (log_partition - q[np.arange(len(y)), y[:, 3]]))
    gradient[6] = weights @ (q[np.arange(len(y)), y[:, 3]] - (pmf * q).sum(axis=1))
    # Average the four heads; intercepts are not penalized.
    objective = (float(losses.sum()) + timing_nll) / 4
    gradient /= 4
    slopes = parameters[:6:2]
    objective += regularization * float(((slopes - 1) ** 2).sum())
    gradient[:6:2] += 2 * regularization * (slopes - 1)
    objective += regularization * float(parameters[6] ** 2)
    gradient[6] += 2 * regularization * parameters[6]
    if not math.isfinite(objective) or not np.isfinite(gradient).all():
        raise ValueError("NONFINITE_CALIBRATION_OPTIMIZATION")
    return objective, gradient


def score_calibration(rows, calibration):
    """Weighted heldout metrics using the same post-Platt J/R projection.

    Unlike fitting, a heldout subset may have only one observed binary class.
    A nonempty filled subset is still needed for conditional-head diagnostics.
    """
    inputs = _inputs(rows)
    x, timing, y, weights, conditional = inputs
    _, p, crossed, q, pmf = _outputs(x, timing, _parameters(calibration))
    clipped = np.clip(p, _EPS, 1 - _EPS)
    brier, nll = {}, {}
    for i, head in enumerate(_HEADS):
        w = weights if i == 0 else conditional
        brier[head] = float(w @ ((p[:, i] - y[:, i]) ** 2))
        nll[head] = float(w @ (-y[:, i] * np.log(clipped[:, i])
                                     - (1 - y[:, i]) * np.log1p(-clipped[:, i])))
    timing_nll = float(weights @ (np.log(np.exp(q).sum(axis=1)) - q[np.arange(len(y)), y[:, 3]]))
    timing_brier = float(weights @ ((pmf - np.eye(21)[y[:, 3]]) ** 2).sum(axis=1))
    if not all(math.isfinite(value) for value in (*brier.values(), *nll.values(), timing_nll, timing_brier)):
        raise ValueError("NONFINITE_CALIBRATION_METRICS")
    return dict(brier=brier, binary_nll=nll, timing_nll=timing_nll, timing_brier=timing_brier,
                projection_count=int(crossed.sum()), row_count=len(rows),
                conditional_row_count=int((y[:, 0] == 1).sum()),
                loss_normalization="WEIGHTED_MEAN_PER_HEAD")


def fit_calibration(rows, *, iterations=500, regularization=.01):
    """Fit signed Platt pairs and a positive timing temperature jointly.

    A finite projected-gradient optimizer uses deterministic backtracking.
    Initial identity parameters are optimized, never returned as a fallback for
    missing classes or failed optimization. This is not a heldout validation.
    """
    if type(iterations) is not int or not 1 <= iterations <= 10000:
        raise ValueError("BOUNDED_CALIBRATION_ITERATIONS_REQUIRED")
    regularization = finite(regularization, "regularization", 0, 100)
    inputs = _inputs(rows, require_classes=True)
    parameters = np.array([1., 0., 1., 0., 1., 0., 0.])
    lower = np.array([-20., -40., -20., -40., -20., -40., math.log(.01)])
    upper = np.array([20., 40., 20., 40., 20., 40., math.log(100.)])
    initial = dict(version=VERSION, timing_temperature=1., **{key: [1., 0.] for key in _HEADS})
    before = score_calibration(rows, initial)
    value, gradient = _loss_gradient(parameters, inputs, regularization)
    initial_objective = value
    accepted, total_backtracks, converged = 0, 0, False
    for step in range(iterations):
        if float(np.linalg.norm(gradient, ord=np.inf)) <= 1e-8:
            converged = True
            break
        rate = 1.
        updated = False
        for backtrack in range(40):
            candidate = np.clip(parameters - rate * gradient, lower, upper)
            direction = candidate - parameters
            if np.array_equal(candidate, parameters):
                converged = True
                break
            candidate_value, candidate_gradient = _loss_gradient(candidate, inputs, regularization)
            if candidate_value <= value + 1e-4 * float(gradient @ direction):
                parameters, value, gradient = candidate, candidate_value, candidate_gradient
                accepted += 1
                total_backtracks += backtrack
                updated = True
                break
            rate *= .5
        if converged:
            break
        if not updated:
            # At a kink the deterministic finite solver may reach stationarity.
            # Preserve an actual fit and expose its status, never claim optimality.
            break
    calibration = dict(method=METHOD, version=VERSION,
                       timing_temperature=math.exp(float(parameters[6])),
                       **{key: parameters[2*i:2*i+2].tolist() for i, key in enumerate(_HEADS)})
    after = score_calibration(rows, calibration)
    report = dict(schema_version="LES_V7_OOF_CALIBRATION_REPORT_R1", method=METHOD,
                  version=VERSION, data_sha256=identity(rows), row_ids=[row["row_id"] for row in rows],
                  iterations_requested=iterations, iterations_accepted=accepted,
                  line_search_backtracks=total_backtracks, converged=converged,
                  fit_status="FITTED_BOUNDED_DETERMINISTIC", evaluation_scope="CALIBRATION_SET_NOT_HOLDOUT",
                  regularization=regularization, objective="POST_PROJECTION_WEIGHTED_NLL_RIDGE",
                  fitted_parameters=dict(calibration),
                  initial_objective=initial_objective, final_objective=value, before=before, after=after,
                  projection_policy="EQUAL_WEIGHT_BOUNDED_ISOTONIC_JOINT_LE_REBOUND_R1",
                  parameter_bounds=dict(slope=[-20, 20], intercept=[-40, 40], timing_temperature=[.01, 100]),
                  approved_for_live=False)
    calibration.update(projection_validated=True, validation_report_id=identity(report))
    return dict(calibration=calibration, report=report)
