"""Matured-label / offline convex helpers, separate from online feature inputs."""
from __future__ import annotations

from math import fsum
from zoneinfo import ZoneInfo
import numpy as np

from domain.les_v7.contracts import PRIOR, ReferencePlan, finite, identity, timestamp


def label_event(window: list[dict], *, cutoff: str, expected_sessions: list[str] | None = None,
                horizon=20, fill: dict | None = None, plan: ReferencePlan | None = None) -> dict:
    """Label an observed reference execution, never simulate a broker/account.

    window is the frozen calendar's complete t..t+h session sequence. Fill and
    first-hit events come from a separate executable-path evaluator (e.g. Bank).
    Missing path evidence is PATH_AMBIGUOUS, not a favorable OHLC assumption.
    """
    if type(horizon) is not int or horizon != 20: raise ValueError("REFERENCE_H20_REQUIRED")
    if expected_sessions is None:
        return dict(status="CALENDAR_NOT_PROVIDED", bottom=None, timing=None, rebound=None, joint=None, outcome_r=None)
    if len(expected_sessions) != horizon + 1 or expected_sessions != sorted(set(expected_sessions)):
        raise ValueError("FROZEN_CALENDAR_SESSION_SEQUENCE_REQUIRED")
    if len(window) != horizon + 1 or any(row.get("complete") is not True for row in window):
        return dict(status="RIGHT_CENSORED", bottom=None, timing=None, rebound=None, joint=None, outcome_r=None)
    sessions = [row["session"] for row in window]
    if sessions != expected_sessions or sessions[0] != timestamp(cutoff).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat():
        raise ValueError("FROZEN_CALENDAR_SESSION_SEQUENCE_REQUIRED")
    lows = [finite(row["low"], "label_low", 0) for row in window]
    if any(v <= 0 for v in lows): raise ValueError("POSITIVE_LABEL_PRICES_REQUIRED")
    availability = [timestamp(row["available_at"]) for row in window]
    if any(a < timestamp(session + "T13:30:00+08:00") for session, a in zip(sessions, availability)):
        raise ValueError("INCOMPLETE_SESSION_LABEL_AVAILABILITY")
    bottom = min(lows)
    result = dict(status="MATURED", bottom=bottom, timing=lows.index(bottom),
                  label_available_at=max(availability).isoformat(), fill=None, rebound=None, joint=None, outcome_r=None)
    if fill is None: return dict(result, status="PATH_NOT_PROVIDED")
    if plan is None: return dict(result, status="REFERENCE_PLAN_NOT_PROVIDED")
    if (fill.get("policy_id") != plan.policy_id or fill.get("policy_version") != plan.policy_version
            or fill.get("plan_id") != plan.plan_id): raise ValueError("LABEL_POLICY_OR_SEED_MISMATCH")
    fr3_l2 = plan.policy_version == "FR3_SOURCE_BOUND_LIMIT_H20_L2"
    fr3 = plan.policy_version in {"FR3_SOURCE_BOUND_LIMIT_H20_R1", "FR3_SOURCE_BOUND_LIMIT_H20_L2"}
    label_version = ("FR3_L2_FIXED_TARGET_NEAR_BOTTOM_GIVEN_FILL_R1" if fr3_l2
        else "FR3_FIXED_TARGET_NEAR_BOTTOM_GIVEN_FILL_R1")
    path_version = ("FR3_H20_OPEN_LIMIT3_FIXED_TARGET_COST_AFTER_L2" if fr3_l2
        else "FR3_H20_OPEN_ONLY_FIXED_TARGET_COST_AFTER_R1")
    if fr3:
        if (fill.get("label_version") != label_version
                or fill.get("path_policy_version") != path_version
                or fill.get("native_plan_target") != plan.target
                or fill.get("target_policy") != "FIXED_CAUSAL_PLAN_TARGET_NO_2R_SUBSTITUTION"):
            raise ValueError("FR3_FIXED_TARGET_LABEL_BINDING_REQUIRED")
        result["label_version"] = label_version
    if fr3_l2:
        if fill.get("raw_window_sha256") != identity(window):
            raise ValueError("FR3_L2_LABEL_WINDOW_BINDING_REQUIRED")
        result.update(label_source_id=identity(dict(label_version=label_version,
            plan_id=plan.plan_id, execution_id=identity(fill), raw_window_id=identity(window), cutoff=cutoff)),
            horizon_anchor="SIGNAL_SESSION", reference_plan_id=plan.plan_id,
            path_policy_id=fill.get("path_policy_id"))
    if not fill.get("path_evidence_ids") or not fill.get("path_available_at"):
        return dict(result, status="PATH_NOT_PROVIDED")
    result["label_available_at"] = max(max(availability), timestamp(fill["path_available_at"])).isoformat()
    if fill.get("coordinate_id") != window[0].get("coordinate_id") or not fill.get("coordinate_id"):
        raise ValueError("PRICE_COORDINATE_MISMATCH")
    if fill.get("filled") is False: return dict(result, fill=0)
    if fill.get("filled") is not True: return dict(result, status="PATH_AMBIGUOUS")
    entry, stop, atr = (finite(fill[k], k) for k in ("entry", "initial_stop", "cutoff_atr"))
    shares = fill["shares"]
    if type(shares) is not int or shares <= 0 or not 0 < stop < entry or atr <= 0:
        raise ValueError("INVALID_FILLED_TRADE_RISK")
    if stop != plan.stop or atr != plan.atr or shares > plan.reference_shares or not plan.band[0] <= entry <= plan.band[1]:
        raise ValueError("EXECUTED_PATH_NOT_BOUND_TO_REFERENCE_SEED")
    if entry < bottom: raise ValueError("ENTRY_BELOW_WINDOW_BOTTOM_DATA_ERROR")
    near = 0 <= (entry - bottom) / atr <= 1.
    target_at, stop_at = fill.get("reference_target_first_hit_at" if fr3 else "net_2r_first_hit_at"), fill.get("initial_stop_first_hit_at")
    if (fill.get("path_order_verified") is not True or target_at and stop_at and timestamp(target_at) == timestamp(stop_at)):
        return dict(result, status="PATH_AMBIGUOUS", fill=1)
    filled_at = timestamp(fill["filled_at"])
    if any(timestamp(t) < filled_at for t in (target_at, stop_at) if t): raise ValueError("PRE_FILL_HIT_IS_NOT_A_TRADE_OUTCOME")
    filled_session = filled_at.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
    if fr3_l2:
        maximum = 3 if plan.context == "LEADERSHIP_CONTINUATION" else 1
        conditional = fill.get("conditional_entry", {})
        if (filled_session not in sessions[1:maximum+1]
                or filled_at != timestamp(filled_session+"T09:00:00+08:00")
                or conditional.get("filled_opportunity") != sessions.index(filled_session)
                or conditional.get("eligible_sessions") != sessions[1:maximum+1]
                or conditional.get("state") != "REFERENCE_FILLED_AT_ELIGIBLE_OPEN"):
            raise ValueError("FR3_L2_FILL_NOT_IN_AUTHORIZED_OPEN_SEQUENCE")
    elif filled_session != sessions[1]:
        raise ValueError("FILL_NOT_IN_NEXT_SESSION")
    end_at = timestamp(sessions[-1] + "T13:30:00+08:00")
    if any(timestamp(t) > end_at for t in (target_at, stop_at) if t):
        raise ValueError("HIT_OUTSIDE_REFERENCE_HORIZON")
    rebound = bool(target_at and timestamp(target_at) <= availability[-1] and
                   (not stop_at or timestamp(target_at) < timestamp(stop_at)))
    pnl = fill.get("realized_net_pnl")
    if pnl is not None:
        if not fill.get("cost_policy_version") or not fill.get("realized_at"):
            raise ValueError("REALIZED_COST_AND_TIME_EVIDENCE_REQUIRED")
        realized_at = timestamp(fill["realized_at"])
        if not filled_at <= realized_at <= end_at or timestamp(fill["path_available_at"]) < realized_at:
            raise ValueError("INVALID_REALIZED_OUTCOME_TIME")
    outcome = finite(pnl, "total_net_pnl") / (shares * (entry - stop)) if pnl is not None else None
    return dict(result, fill=1, rebound=int(rebound), joint=int(near and rebound), outcome_r=outcome,
                normalized_utility=(min(2., max(-1., outcome)) + 1) / 3 if outcome is not None else None,
                status="MATURED" if target_at or stop_at else "TIMEOUT")


def fit_heads(rows: list[dict], *, fit_cutoff: str, iterations=1000, regularization=.01, solver=None,
              fill_only_base_count=None) -> dict:
    """Finite deterministic convex baselines, not calibration/approval/OOF proof.

    Coefficients have signed effects; only LES weights have simplex constraints.
    No fitted forecasts are passed automatically into Stage C.
    """
    if not rows or not 1 <= iterations <= 10000: raise ValueError("BOUNDED_MATURED_TRAINING_ROWS_REQUIRED")
    finite(regularization, "regularization", 0)
    for row in rows:
        if not timestamp(row["cutoff"]) < timestamp(row["label_available_at"]) <= timestamp(fit_cutoff):
            raise ValueError("UNMATURED_OR_FUTURE_LABEL")
        if timestamp(row["feature_fit_cutoff"]) > timestamp(row["cutoff"]): raise ValueError("FUTURE_SCALER_LEAKAGE")
        if type(row.get("fill")) is not int or row["fill"] not in {0, 1}:
            raise ValueError("OBSERVED_BINARY_FILL_REQUIRED")
        if row["fill"] == 0 and any(row.get(name) is not None for name in ("rebound", "joint")):
            raise ValueError("NO_FILL_CONDITIONAL_LABELS_MUST_BE_NULL")
        if row["fill"] == 1 and (any(type(row.get(name)) is not int or row[name] not in {0, 1}
                for name in ("rebound", "joint")) or row["joint"] > row["rebound"]):
            raise ValueError("INVALID_CONDITIONAL_EVENT_LABELS")
    x = np.array([row["features"] for row in rows], dtype=float)
    if x.ndim != 2 or not 1 <= x.shape[1] <= 64 or not np.isfinite(x).all(): raise ValueError("INVALID_TRAINING_FEATURES")
    a = np.column_stack((np.ones(len(x)), x))
    alpha = np.array([finite(row.get("observation_weight", 1.), "alpha", 0) for row in rows])
    if alpha.sum() <= 0: raise ValueError("POSITIVE_OBSERVATION_WEIGHT_SUM_REQUIRED")
    alpha /= alpha.sum()
    m = np.array([finite(row["log_bottom_ratio"], "m") for row in rows])
    if (m > 0).any(): raise ValueError("BOTTOM_LABEL_CANNOT_EXCEED_CUTOFF_CLOSE")
    k = np.array([row["bottom_timing"] for row in rows])
    if any(type(row["bottom_timing"]) is not int for row in rows) or ((k < 0) | (k > 20)).any(): raise ValueError("INVALID_TIMING_LABEL")
    price, timing = np.zeros((3, a.shape[1])), np.zeros((21, a.shape[1]))
    binary = {name: np.zeros(a.shape[1]) for name in ("fill", "rebound", "joint")}
    for name in binary:
        subset = rows if name == "fill" else [row for row in rows if row["fill"] == 1]
        if not subset or {row[name] for row in subset} != {0, 1}:
            raise ValueError("BOTH_BINARY_CLASSES_REQUIRED:" + name)
        if sum(alpha[i] for i, row in enumerate(rows) if name == "fill" or row["fill"] == 1) <= 0:
            raise ValueError("POSITIVE_CONDITIONAL_SAMPLE_WEIGHT_REQUIRED")
    if fill_only_base_count is not None:
        if (type(fill_only_base_count) is not int or not 1 <= fill_only_base_count < x.shape[1]
                or solver != "STANDARDIZED_CONVEX_R2"):
            raise ValueError("FILL_ONLY_BASE_FEATURE_COUNT_OR_SOLVER_INVALID")
        return _fit_fill_only_extension(rows, x, alpha, fit_cutoff, iterations,
                                        regularization, fill_only_base_count)
    if solver is not None:
        if solver != "STANDARDIZED_CONVEX_R2":
            raise ValueError("UNSUPPORTED_HEAD_SOLVER")
        return _fit_standardized_heads(rows, x, alpha, m, k, fit_cutoff, iterations, regularization)
    learning_rate = .2 / max(1., float(np.linalg.norm(a, ord=2) ** 2 / len(a)))
    for step in range(iterations):
        for j, q in enumerate((.1, .5, .9)):
            error = m - a @ price[j]
            grad = -(a.T @ (alpha * (q - (error < 0))))
            grad[1:] += 2 * regularization * price[j, 1:]
            price[j] -= learning_rate / np.sqrt(step + 1) * grad
        logits = a @ timing.T
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(a)), k] -= 1
        grad = (alpha[:, None] * probs).T @ a
        grad[:, 1:] += 2 * regularization * timing[:, 1:]
        timing -= learning_rate * grad
        timing -= timing[-1]  # Fixed reference class, not a missing timing bucket.
        for name, beta in binary.items():
            keep = np.array([name == "fill" or row["fill"] == 1 for row in rows])
            y = np.array([row[name] for row in rows if name == "fill" or row["fill"] == 1])
            aa, weights = a[keep], alpha[keep]
            weights = weights / weights.sum()
            probability = np.exp(-np.logaddexp(0., -(aa @ beta)))
            gradient = aa.T @ (weights * (probability - y))
            gradient[1:] += 2 * regularization * beta[1:]
            beta -= learning_rate * gradient
    return dict(heads=dict(price=price.tolist(), timing=timing.tolist(), **{k: b.tolist() for k, b in binary.items()}),
                fit_status="FIXED_ITERATION_OFFLINE_FIT", calibration_status="NOT_PROVIDED",
                approved_for_live=False, fit_cutoff=timestamp(fit_cutoff).isoformat())


def fit_factor_weights(rows, *, fit_cutoff, **solver_options):
    """Reuse the existing NumPy hierarchical Huber/simplex solver, not its v6 adapter."""
    from domain.strategies.low_entry_v6_weights import fit_convex_weights
    for row in rows:
        if row.get("prediction_kind") != "STRICTLY_PAST_OOF": raise ValueError("STAGE_C_REQUIRES_OOF_PREDICTIONS")
        if not timestamp(row["head_last_label_available_at"]) <= timestamp(row["head_fit_cutoff"]) < timestamp(row["cutoff"]) < timestamp(row["label_available_at"]) <= timestamp(fit_cutoff):
            raise ValueError("STAGE_C_TIME_LEAKAGE")
        if row.get("calibration_fit_cutoff") is not None and not (
                timestamp(row["calibration_last_label_available_at"]) <= timestamp(row["calibration_fit_cutoff"])
                < timestamp(row["cutoff"])):
            raise ValueError("STAGE_C_CALIBRATION_TIME_LEAKAGE")
    if "alpha" in solver_options: raise ValueError("OBSERVATION_WEIGHT_BELONGS_TO_EACH_ROW")
    if "prior" in solver_options: raise ValueError("V7_PRIOR_IS_FROZEN")
    alpha = [finite(r.get("observation_weight", 1.), "alpha", 0) for r in rows]
    total = fsum(alpha)
    if total <= 0: raise ValueError("POSITIVE_OBSERVATION_WEIGHT_SUM_REQUIRED")
    result = fit_convex_weights([r["factors"] for r in rows], [r["utility"] for r in rows],
                              [r["context"] for r in rows], alpha=np.array(alpha) / total, prior=PRIOR, **solver_options)
    return dict(result, loss_normalization="WEIGHTED_MEAN_STANDARD_HUBER")


# Optional R2 is isolated from the legacy fixed-iteration producer above.
# Its stopping rules are optimization diagnostics, never holdout thresholds.
_R2_GRADIENT_TOLERANCE = 1e-6
_R2_ADMM_ABSOLUTE_TOLERANCE = 1e-7
_R2_ADMM_RELATIVE_TOLERANCE = 1e-6


def _weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left")), len(order) - 1)
    return float(values[order[index]])


def _lbfgs(objective, initial, iterations):
    """Bounded deterministic L-BFGS with exact objective/gradient and Armijo."""
    point = np.asarray(initial, dtype=float).copy()
    value, gradient = objective(point)
    initial_value, history, backtracks, accepted = value, [], 0, 0
    status = "ITERATION_LIMIT"
    for _ in range(iterations):
        if float(np.linalg.norm(gradient, ord=np.inf)) <= _R2_GRADIENT_TOLERANCE:
            status = "CONVERGED"
            break
        direction = gradient.copy()
        coefficients = []
        for step, change, reciprocal in reversed(history):
            coefficient = reciprocal * float(step @ direction)
            coefficients.append(coefficient)
            direction -= coefficient * change
        if history:
            step, change, _ = history[-1]
            direction *= float(step @ change) / float(change @ change)
        for (step, change, reciprocal), coefficient in zip(history, reversed(coefficients)):
            direction += step * (coefficient - reciprocal * float(change @ direction))
        direction = -direction
        slope = float(gradient @ direction)
        if not np.isfinite(direction).all() or slope >= 0:
            direction, slope = -gradient, -float(gradient @ gradient)
            history.clear()
        rate, updated = 1., False
        for backtrack in range(40):
            candidate = point + rate * direction
            candidate_value, candidate_gradient = objective(candidate)
            if candidate_value <= value + 1e-4 * rate * slope:
                updated = True
                break
            rate *= .5
        backtracks += backtrack
        if not updated:
            status = "LINE_SEARCH_STALLED"
            break
        step, change = candidate - point, candidate_gradient - gradient
        curvature = float(step @ change)
        if curvature > 1e-12 * float(np.linalg.norm(step) * np.linalg.norm(change)):
            history.append((step, change, 1. / curvature))
            history = history[-10:]
        point, value, gradient = candidate, candidate_value, candidate_gradient
        accepted += 1
    norm = float(np.linalg.norm(gradient, ord=np.inf))
    if norm <= _R2_GRADIENT_TOLERANCE:
        status = "CONVERGED"
    return point, dict(algorithm="DETERMINISTIC_LBFGS_ARMIJO_R2", status=status,
        converged=status == "CONVERGED", iterations_requested=iterations, iterations_accepted=accepted,
        line_search_backtracks=backtracks, initial_objective=initial_value, final_objective=value,
        gradient_inf_norm=norm, gradient_tolerance=_R2_GRADIENT_TOLERANCE)


def _pinball_dual_certificate(design, weights, target, coefficients, regularization, scaled_dual, rho, proof=None):
    """A lower bound for the exact primal, using box/equality-feasible duals.

    For lambda>0: max v'y - ||X_slopes'v||²/(4lambda), with
    sum(v)=0 and w*(q-1)<=v<=w*q. This certifies the original beta objective,
    independently of ADMM's auxiliary residual. No holdout values are used.
    """
    if regularization <= 0:
        return None, None
    from domain.les_v7.contracts import identity
    objectives, gaps, evidence = [], [], dict(projected_dual_sha256=[], box_violation=[], intercept_residual=[])
    for index, quantile in enumerate((.1, .5, .9)):
        lower, upper = weights * (quantile - 1.), weights * quantile
        value = -rho * weights * scaled_dual[:, index]
        left, right = float(np.min(value - upper)), float(np.max(value - lower))
        for _ in range(60):
            offset = (left + right) / 2.
            projected = np.clip(value - offset, lower, upper)
            if float(projected.sum()) > 0:
                left = offset
            else:
                right = offset
        projected = np.clip(value - (left + right) / 2., lower, upper)
        error = fsum(projected.tolist())
        # Correct the tiny bisection rounding error within a feasible box.
        for row in range(len(projected)):
            candidate = projected[row] - error
            if lower[row] <= candidate <= upper[row]:
                projected[row] = candidate
                break
        feasibility = abs(fsum(projected.tolist()))
        box_violation = max(0., float(np.max(lower - projected)), float(np.max(projected - upper)))
        if feasibility > 1e-12 or box_violation > 1e-12:
            return None, None
        evidence['projected_dual_sha256'].append(identity(projected.tolist()))
        evidence['box_violation'].append(box_violation)
        evidence['intercept_residual'].append(feasibility)
        slope_gradient = design[:, 1:].T @ projected
        bound = float(projected @ target) - float(slope_gradient @ slope_gradient) / (4 * regularization)
        beta = coefficients[:, index]
        error = target - design @ beta
        primal = float(weights @ np.maximum(quantile * error, (quantile - 1.) * error)) + regularization * float(beta[1:] @ beta[1:])
        gap = primal - bound
        if not np.isfinite(gap) or gap < -1e-10:
            return None, None
        objectives.append(bound); gaps.append(max(0., gap))
    if proof is not None:
        proof.update(evidence, convention="V_EQUALS_NEGATIVE_CONSTRAINT_LAGRANGE_MULTIPLIER", feasibility_tolerance=1e-12)
    return np.asarray(objectives), np.asarray(gaps)


def _quantile_admm(design, weights, target, regularization, iterations):
    """Weighted pinball + slope ridge; weighted residual constraint ADMM.

    Using W in the augmented norm keeps average-loss conditioning independent
    of sample count. The proximal thresholds are q/rho and (1-q)/rho. The
    objective remains the exact weighted, nonsmoothed pinball loss.
    """
    quantiles = np.array([.1, .5, .9])
    coefficients = np.zeros((design.shape[1], 3))
    coefficients[0] = [_weighted_quantile(target, weights, quantile) for quantile in quantiles]
    def objective(beta):
        error = target[:, None] - design @ beta
        return np.sum(weights[:, None] * np.maximum(quantiles * error, (quantiles - 1.) * error), axis=0) + regularization * np.sum(beta[1:] ** 2, axis=0)
    initial = objective(coefficients)
    residual = target[:, None] - design @ coefficients
    dual = np.zeros_like(residual)
    rho = 1. / max(float(np.sqrt(weights @ (target - weights @ target) ** 2)), .01)
    gram = design.T @ (weights[:, None] * design)
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0.
    converged = np.zeros(3, dtype=bool)
    primal_norm = dual_norm = np.full(3, np.inf)
    accepted = 0
    dual_objective = duality_gap = None
    certificate_proof = {}
    for step in range(iterations):
        system = rho * gram + 2 * regularization * penalty
        right = rho * design.T @ (weights[:, None] * (target[:, None] - residual - dual))
        try:
            coefficients = np.linalg.solve(system, right)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(system, right, rcond=1e-12)[0]
        prediction = design @ coefficients
        previous = residual
        value = target[:, None] - prediction - dual
        residual = np.where(value > quantiles / rho, value - quantiles / rho,
            np.where(value < -(1. - quantiles) / rho, value + (1. - quantiles) / rho, 0.))
        violation = prediction + residual - target[:, None]
        dual += violation
        primal_norm = np.sqrt(np.sum(weights[:, None] * violation ** 2, axis=0))
        dual_norm = np.linalg.norm(rho * design.T @ (weights[:, None] * (residual - previous)), axis=0)
        primal_bound = _R2_ADMM_ABSOLUTE_TOLERANCE + _R2_ADMM_RELATIVE_TOLERANCE * np.maximum.reduce((
            np.sqrt(np.sum(weights[:, None] * prediction ** 2, axis=0)),
            np.sqrt(np.sum(weights[:, None] * residual ** 2, axis=0)),
            np.full(3, float(np.sqrt(weights @ target ** 2)))))
        dual_bound = _R2_ADMM_ABSOLUTE_TOLERANCE + _R2_ADMM_RELATIVE_TOLERANCE * np.linalg.norm(
            rho * design.T @ (weights[:, None] * dual), axis=0)
        converged = (primal_norm <= primal_bound) & (dual_norm <= dual_bound)
        accepted = step + 1
        if accepted % 25 == 0 or converged.all() or accepted == iterations:
            certificate_proof = {}
            dual_objective, duality_gap = _pinball_dual_certificate(design, weights, target,
                coefficients, regularization, dual, rho, proof=certificate_proof)
            if regularization > 0:
                converged = (duality_gap <= _R2_ADMM_ABSOLUTE_TOLERANCE
                             if duality_gap is not None else np.zeros(3, dtype=bool))
        elif regularization > 0:
            converged = np.zeros(3, dtype=bool)
        if converged.all():
            break
        if accepted % 25 == 0:
            before = rho
            largest_primal, largest_dual = float(primal_norm.max()), float(dual_norm.max())
            if largest_primal > 10 * largest_dual:
                rho = min(rho * 2., 1e6)
            elif largest_dual > 10 * largest_primal:
                rho = max(rho / 2., 1e-6)
            dual *= before / rho
    final = objective(coefficients)
    if not np.isfinite(coefficients).all() or not np.isfinite(final).all():
        raise ValueError("NONFINITE_R2_PRICE_OPTIMIZATION")
    converged &= final <= initial + 1e-10
    return coefficients.T, dict(algorithm="WEIGHTED_ADMM_EXACT_PINBALL_SLOPE_RIDGE_R2",
        status="CONVERGED" if converged.all() else "ITERATION_LIMIT", converged=bool(converged.all()),
        quantiles=quantiles.tolist(), iterations_requested=iterations, iterations_accepted=accepted,
        initial_objective=initial.tolist(), final_objective=final.tolist(), primal_residual=primal_norm.tolist(),
        dual_residual=dual_norm.tolist(), absolute_tolerance=_R2_ADMM_ABSOLUTE_TOLERANCE,
        relative_tolerance=_R2_ADMM_RELATIVE_TOLERANCE, final_rho=rho,
        dual_objective=dual_objective.tolist() if dual_objective is not None else None,
        duality_gap=duality_gap.tolist() if duality_gap is not None else None,
        duality_gap_tolerance=_R2_ADMM_ABSOLUTE_TOLERANCE,
        dual_certificate="WEIGHTED_PINBALL_BOX_ZERO_INTERCEPT_SLOPE_RIDGE_R2" if regularization > 0 else None,
        dual_certificate_feasibility=certificate_proof if regularization > 0 else None,
        convergence_criterion="FEASIBLE_DUALITY_GAP" if regularization > 0 else "ADMM_PRIMAL_DUAL_RESIDUALS_NO_RIDGE_CERTIFICATE")


def _fit_standardized_heads(rows, x, alpha, target, timing_labels, fit_cutoff, iterations, regularization):
    from domain.les_v7.contracts import identity
    active = alpha > 0
    x, alpha, target, timing_labels = x[active], alpha[active], target[active], timing_labels[active]
    selected_rows = [row for row, keep in zip(rows, active) if keep]
    alpha /= alpha.sum()
    center = alpha @ x
    scale = np.sqrt(alpha @ ((x - center) ** 2))
    scale = np.where(scale > 1e-12, scale, 1.)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("NONFINITE_R2_TRAINING_SCALER")
    design = np.column_stack((np.ones(len(x)), (x - center) / scale))
    price, price_report = _quantile_admm(design, alpha, target, regularization, iterations)
    binary, binary_reports = {}, {}
    for name in ("fill", "rebound", "joint"):
        keep = np.array([name == "fill" or row["fill"] == 1 for row in selected_rows])
        labels = np.array([row[name] for row in selected_rows if name == "fill" or row["fill"] == 1], dtype=float)
        if set(labels.tolist()) != {0., 1.}:
            raise ValueError("BOTH_POSITIVE_WEIGHT_BINARY_CLASSES_REQUIRED:" + name)
        matrix, weights = design[keep], alpha[keep]
        weights /= weights.sum()
        prior = float(weights @ labels)
        initial = np.zeros(design.shape[1]); initial[0] = np.log(prior / (1. - prior))
        def objective(beta):
            logits = matrix @ beta
            probability = np.exp(-np.logaddexp(0., -logits))
            value = float(weights @ (np.logaddexp(0., logits) - labels * logits)) + regularization * float(beta[1:] @ beta[1:])
            gradient = matrix.T @ (weights * (probability - labels))
            gradient[1:] += 2 * regularization * beta[1:]
            return value, gradient
        beta, report = _lbfgs(objective, initial, iterations)
        binary[name], binary_reports[name] = beta, dict(report, training_class_prior=prior,
            training_row_count=len(labels), objective="WEIGHTED_BINARY_NLL_SLOPE_RIDGE")
    # Fixed reference category 20 is only a coordinate choice. Ridge acts on
    # all 21 centered coefficient blocks (including intercepts), so permuting
    # bucket names/reference choice does not change the regularized objective.
    # This also makes sparse timing classes finite without dropping a bucket.
    frequencies = np.bincount(timing_labels, weights=alpha, minlength=21)
    effective = 1. / float(alpha @ alpha)
    prior = (frequencies + .5 / effective) / (1. + 21 * .5 / effective)
    if regularization == 0 and (frequencies == 0).any():
        raise ValueError("SPARSE_TIMING_CLASSES_REQUIRE_DECLARED_R2_RIDGE")
    initial = np.zeros((20, design.shape[1])); initial[:, 0] = np.log(prior[:20] / prior[20])
    def timing_objective(flat):
        beta = flat.reshape(20, design.shape[1])
        logits = np.column_stack((design @ beta.T, np.zeros(len(design))))
        maximum = logits.max(axis=1, keepdims=True)
        shifted = logits - maximum
        exponent = np.exp(shifted)
        denominator = exponent.sum(axis=1, keepdims=True)
        probabilities = exponent / denominator
        log_probabilities = shifted - np.log(denominator)
        summed = beta.sum(axis=0)
        penalty = float(flat @ flat - (summed @ summed) / 21.)
        value = -float(alpha @ log_probabilities[np.arange(len(design)), timing_labels]) + regularization * penalty
        probabilities[np.arange(len(design)), timing_labels] -= 1.
        gradient = (alpha[:, None] * probabilities[:, :20]).T @ design + 2 * regularization * (beta - summed / 21.)
        return value, gradient.ravel()
    fitted, timing_report = _lbfgs(timing_objective, initial.ravel(), iterations)
    timing = np.vstack((fitted.reshape(20, design.shape[1]), np.zeros(design.shape[1])))
    def raw_coefficients(value):
        value = np.asarray(value, dtype=float).copy()
        value[..., 1:] /= scale
        value[..., 0] -= value[..., 1:] @ center
        return value.tolist()
    heads = dict(price=raw_coefficients(price), timing=raw_coefficients(timing),
                 **{name: raw_coefficients(beta) for name, beta in binary.items()})
    report = dict(schema="v7-standardized-convex-head-fit/v2", solver="STANDARDIZED_CONVEX_R2",
        fit_cutoff=timestamp(fit_cutoff).isoformat(), training_row_count=len(selected_rows), effective_samples=effective,
        last_label_available_at=max(timestamp(row['label_available_at']) for row in selected_rows).isoformat(),
        ordered_training_row_ids_sha256=identity([row.get('row_id') for row in selected_rows]),
        feature_count=x.shape[1], weighted_center=center.tolist(), weighted_scale=scale.tolist(),
        scaler_scope="ONLY_MATURE_POSITIVE_WEIGHT_TRAINING_ROWS", exported_coordinates="RAW_FEATURE_COEFFICIENTS",
        raw_heads=heads, heads_sha256=identity(heads), regularization=regularization,
        price=price_report, binary=binary_reports,
        timing=dict(timing_report, objective="WEIGHTED_SOFTMAX_NLL_CENTERED_ALL_CLASS_COEFFICIENT_RIDGE",
            reference_class=20, training_class_frequencies=frequencies.tolist(),
            initializer="TRAINING_EFFECTIVE_SAMPLE_JEFFREYS_HALF_COUNT_ONLY", training_class_prior=prior.tolist()))
    report['converged'] = bool(price_report['converged'] and timing_report['converged']
                              and all(value['converged'] for value in binary_reports.values()))
    report['status'] = 'CONVERGED' if report['converged'] else 'NOT_CONVERGED'
    return dict(heads=heads, fit_status=report['status'], calibration_status="NOT_PROVIDED", approved_for_live=False,
                fit_cutoff=timestamp(fit_cutoff).isoformat(), solver_report=report)


def _fit_fill_only_extension(rows, x, alpha, fit_cutoff, iterations, regularization, base_count):
    """Fit protected heads in original coordinates and only fill on the extension.

    No labels or rows are added.  Protected coefficients are never estimated on
    the extension then erased; their original fit, scaler, and convergence proof
    are retained independently.  The full scaler is based only on these same
    matured positive-weight training rows.
    """
    from domain.les_v7.contracts import identity
    base_rows = [dict(row, features=list(row["features"][:base_count])) for row in rows]
    base_fit = fit_heads(base_rows, fit_cutoff=fit_cutoff, iterations=iterations,
                         regularization=regularization, solver="STANDARDIZED_CONVEX_R2")
    base_report = base_fit["solver_report"]
    active = alpha > 0
    selected_rows = [row for row, keep in zip(rows, active) if keep]
    matrix, weights = x[active], alpha[active]
    weights /= weights.sum()
    center = weights @ matrix
    scale = np.sqrt(weights @ ((matrix - center) ** 2))
    scale = np.where(scale > 1e-12, scale, 1.)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("NONFINITE_R2_TRAINING_SCALER")
    design = np.column_stack((np.ones(len(matrix)), (matrix - center) / scale))
    labels = np.array([row["fill"] for row in selected_rows], dtype=float)
    if set(labels.tolist()) != {0., 1.}:
        raise ValueError("BOTH_POSITIVE_WEIGHT_BINARY_CLASSES_REQUIRED:fill")
    prior = float(weights @ labels)
    initial = np.zeros(design.shape[1]); initial[0] = np.log(prior / (1. - prior))
    def objective(beta):
        logits = design @ beta
        probability = np.exp(-np.logaddexp(0., -logits))
        value = float(weights @ (np.logaddexp(0., logits) - labels * logits)) + regularization * float(beta[1:] @ beta[1:])
        gradient = design.T @ (weights * (probability - labels))
        gradient[1:] += 2 * regularization * beta[1:]
        return value, gradient
    beta, fill_report = _lbfgs(objective, initial, iterations)
    raw_fill = beta.copy(); raw_fill[1:] /= scale
    raw_fill[0] -= raw_fill[1:] @ center
    extra_count = x.shape[1] - base_count
    def padded(value):
        array = np.asarray(value, dtype=float)
        padding = [(0, 0)] * array.ndim; padding[-1] = (0, extra_count)
        return np.pad(array, padding, mode="constant").tolist()
    heads = {name: padded(base_fit["heads"][name]) for name in ("price", "timing", "rebound", "joint")}
    heads.update(fill=raw_fill.tolist(), fill_only_base_count=base_count)
    binary_reports = dict(base_report["binary"], fill=dict(fill_report, training_class_prior=prior,
        training_row_count=len(labels), objective="WEIGHTED_BINARY_NLL_SLOPE_RIDGE"))
    report = dict(base_report, feature_count=x.shape[1], weighted_center=center.tolist(), weighted_scale=scale.tolist(),
        raw_heads=heads, heads_sha256=identity(heads), binary=binary_reports,
        fill_only_partition=dict(base_feature_count=base_count, fill_feature_count=x.shape[1],
            extension_feature_count=extra_count, protected_heads=["price", "timing", "rebound", "joint"],
            protected_head_coordinates="UNCHANGED_BASE_PREFIX_WITH_ZERO_PADDING"),
        base_report=base_report)
    report["converged"] = bool(base_report["converged"] and fill_report["converged"])
    report["status"] = "CONVERGED" if report["converged"] else "NOT_CONVERGED"
    return dict(heads=heads, fit_status=report["status"], calibration_status="NOT_PROVIDED", approved_for_live=False,
                fit_cutoff=timestamp(fit_cutoff).isoformat(), solver_report=report)
