"""SYNTHETIC_TEST_ONLY: independent head-solver engineering regressions.

Every observation, feature, label, and timestamp below is invented. These
tests never read market datasets, September holdouts, or archived training
jobs, and provide no evidence of investment performance or live approval.
"""
from copy import deepcopy

import numpy as np
import pytest

from domain.les_v7 import research
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.localization import head_outputs
from domain.les_v7.research import fit_heads


SOLVER = "STANDARDIZED_CONVEX_R2"
FIT_CUTOFF = "2025-02-02T00:00:00Z"
REGULARIZATION = .005


def synthetic_rows():
    """Three feature scales, complete timing support, and learnable outcomes."""
    rows = []
    for index in range(252):
        timing = index // 12
        x, z = -2. + timing * .2, -1.1 + (index % 12) * .2
        fill = int(x > -.5)
        rebound = int(z > -.5) if fill else None
        joint = int(rebound and x > .3) if fill else None
        rows.append(dict(
            row_id=f"SYNTHETIC_TEST_ONLY:{index}",
            cutoff="2025-01-01T00:00:00Z",
            feature_fit_cutoff="2025-01-01T00:00:00Z",
            label_available_at="2025-02-01T00:00:00Z",
            features=[1200. + 180. * x, .005 + .001 * z, 7.],
            fill=fill, rebound=rebound, joint=joint,
            log_bottom_ratio=-.12 + .025 * x - .012 * z,
            bottom_timing=timing, observation_weight=1. + index % 3,
        ))
    return rows


def fit(rows, *, iterations=2000, regularization=REGULARIZATION):
    return fit_heads(rows, fit_cutoff=FIT_CUTOFF, iterations=iterations,
                     regularization=regularization, solver=SOLVER)


@pytest.fixture(scope="module")
def predictive_fit():
    rows = synthetic_rows()
    return rows, fit(rows)


def normalized_weights(rows):
    weights = np.array([row["observation_weight"] for row in rows])
    return weights / weights.sum()


def standardized_coefficients(raw, center, scale):
    """Recover coordinates independently to check the declared penalty."""
    raw = np.asarray(raw)
    result = raw.copy()
    result[..., 0] += raw[..., 1:] @ center
    result[..., 1:] *= scale
    return result


def independently_feasible_dual(seed, lower, upper):
    """Project with exact piecewise-linear breakpoints, without bisection.

    The scalar sum of clipped coordinates changes slope only when a bound
    becomes active. Linear interpolation in the interval bracketing zero
    therefore solves the intercept constraint independently of the solver.
    """
    offsets = np.unique(np.concatenate((seed - lower, seed - upper)))
    sums = np.array([np.clip(seed - offset, lower, upper).sum() for offset in offsets])
    right = int(np.flatnonzero(sums <= 0.)[0])
    if sums[right] == 0.:
        offset = offsets[right]
    else:
        left = right - 1
        offset = offsets[left] + (offsets[right] - offsets[left]) * (
            sums[left] / (sums[left] - sums[right]))
    return np.clip(seed - offset, lower, upper)


def test_convex_fit_is_deterministic_and_keeps_runtime_shapes(predictive_fit):
    rows, fitted = predictive_fit
    assert fitted == fit(deepcopy(rows))
    report = fitted["solver_report"]
    assert report["solver"] == SOLVER
    assert report["status"] == fitted["fit_status"] == "CONVERGED"
    assert report["converged"] is True
    assert fitted["approved_for_live"] is False
    assert fitted["calibration_status"] == "NOT_PROVIDED"
    assert report["raw_heads"] == fitted["heads"]
    assert report["heads_sha256"] == identity(fitted["heads"])
    assert report["ordered_training_row_ids_sha256"] == identity([row["row_id"] for row in rows])
    assert timestamp(report["last_label_available_at"]) <= timestamp(report["fit_cutoff"])
    for name, shape in dict(price=(3, 4), timing=(21, 4), fill=(4,), rebound=(4,), joint=(4,)).items():
        coefficients = np.asarray(fitted["heads"][name])
        assert coefficients.shape == shape
        assert np.isfinite(coefficients).all()


def test_scaler_uses_only_mature_positive_weight_training_rows(predictive_fit):
    rows, fitted = predictive_fit
    report = fitted["solver_report"]
    features = np.asarray([row["features"] for row in rows])
    weights = normalized_weights(rows)
    expected_center = weights @ features
    expected_scale = np.sqrt(weights @ ((features - expected_center) ** 2))
    expected_scale[expected_scale <= 1e-12] = 1.
    assert report["weighted_center"] == pytest.approx(expected_center)
    assert report["weighted_scale"] == pytest.approx(expected_scale)
    assert report["weighted_scale"][2] == 1.
    assert report["scaler_scope"] == "ONLY_MATURE_POSITIVE_WEIGHT_TRAINING_ROWS"
    assert report["exported_coordinates"] == "RAW_FEATURE_COEFFICIENTS"
    assert report["effective_samples"] == pytest.approx(1. / (weights @ weights))

    excluded = deepcopy(rows[0])
    excluded.update(row_id="SYNTHETIC_TEST_ONLY:zero-weight-extreme",
                    features=[1e12, -1e8, 1e7], observation_weight=0.,
                    label_available_at="2025-02-02T00:00:00Z")
    assert fit([*rows, excluded]) == fitted


def test_raw_inference_matches_an_independently_standardized_fit(predictive_fit):
    rows, fitted = predictive_fit
    center = np.asarray(fitted["solver_report"]["weighted_center"])
    scale = np.asarray(fitted["solver_report"]["weighted_scale"])
    standardized = deepcopy(rows)
    for row in standardized:
        row["features"] = ((np.asarray(row["features"]) - center) / scale).tolist()
    transformed_fit = fit(standardized)
    assert transformed_fit["solver_report"]["converged"] is True

    # These are inference-only probes; they cannot participate in either fit.
    for values in ([1560., .0048, 7.], [2040., .0082, 9.], [840., .0057, 7.]):
        actual = head_outputs(fitted["heads"], values)
        expected = head_outputs(transformed_fit["heads"],
                                ((np.asarray(values) - center) / scale).tolist())
        assert actual["log_quantiles"] == pytest.approx(expected["log_quantiles"], abs=2e-6)
        assert actual["timing_logits"] == pytest.approx(expected["timing_logits"], abs=2e-5)
        assert actual["raw_logits"] == pytest.approx(expected["raw_logits"], abs=2e-5)


def test_objectives_agree_with_direct_losses_and_beat_intercept_baselines(predictive_fit):
    rows, fitted = predictive_fit
    report, heads = fitted["solver_report"], fitted["heads"]
    weights = normalized_weights(rows)
    design = np.column_stack((np.ones(len(rows)), [row["features"] for row in rows]))
    center, scale = np.asarray(report["weighted_center"]), np.asarray(report["weighted_scale"])
    target = np.array([row["log_bottom_ratio"] for row in rows])
    price_predictions = design @ np.asarray(heads["price"]).T
    price_coefficients = standardized_coefficients(heads["price"], center, scale)
    for index, quantile in enumerate((.1, .5, .9)):
        errors = target - price_predictions[:, index]
        loss = weights @ np.maximum(quantile * errors, (quantile - 1.) * errors)
        # Evaluate all possible constant predictions, independently of the
        # solver's weighted-quantile initialization implementation.
        errors_at_constants = target[:, None] - target[None, :]
        baseline = np.min(weights @ np.maximum(quantile * errors_at_constants,
                                               (quantile - 1.) * errors_at_constants))
        objective = loss + REGULARIZATION * np.sum(price_coefficients[index, 1:] ** 2)
        assert report["price"]["initial_objective"][index] == pytest.approx(baseline, abs=1e-12)
        assert report["price"]["final_objective"][index] == pytest.approx(objective, abs=1e-10)
        assert objective < .5 * baseline

    for name in ("fill", "rebound", "joint"):
        keep = np.array([name == "fill" or row["fill"] == 1 for row in rows])
        selected_weights = weights[keep] / weights[keep].sum()
        labels = np.array([row[name] for row, selected in zip(rows, keep) if selected])
        logits = design[keep] @ np.asarray(heads[name])
        loss = selected_weights @ (np.logaddexp(0., logits) - labels * logits)
        prior = selected_weights @ labels
        baseline = -prior * np.log(prior) - (1. - prior) * np.log1p(-prior)
        coefficients = standardized_coefficients(heads[name], center, scale)
        objective = loss + REGULARIZATION * np.sum(coefficients[1:] ** 2)
        binary_report = report["binary"][name]
        assert binary_report["training_class_prior"] == pytest.approx(prior)
        assert binary_report["initial_objective"] == pytest.approx(baseline, abs=1e-12)
        assert binary_report["final_objective"] == pytest.approx(objective, abs=1e-10)
        assert objective < .75 * baseline
        assert binary_report["gradient_inf_norm"] <= binary_report["gradient_tolerance"]

    timing_labels = np.array([row["bottom_timing"] for row in rows])
    logits = design @ np.asarray(heads["timing"]).T
    log_denominator = np.logaddexp.reduce(logits, axis=1)
    nll = weights @ (log_denominator - logits[np.arange(len(rows)), timing_labels])
    coefficients = standardized_coefficients(heads["timing"], center, scale)
    centered_coefficients = coefficients - coefficients.mean(axis=0)
    objective = nll + REGULARIZATION * np.sum(centered_coefficients ** 2)
    frequencies = np.bincount(timing_labels, weights=weights, minlength=21)
    intercept_baseline = -np.sum(frequencies * np.log(frequencies))
    assert report["timing"]["objective"] == "WEIGHTED_SOFTMAX_NLL_CENTERED_ALL_CLASS_COEFFICIENT_RIDGE"
    assert report["timing"]["final_objective"] == pytest.approx(objective, abs=1e-10)
    assert objective < .9 * intercept_baseline
    assert report["timing"]["gradient_inf_norm"] <= report["timing"]["gradient_tolerance"]


@pytest.mark.parametrize("buckets", [(0, 5, 20), (0, 5, 12)])
def test_sparse_timing_classes_keep_finite_full_probability_support(buckets):
    rows = synthetic_rows()
    for row in rows:
        row["bottom_timing"] = buckets[min(row["bottom_timing"] // 7, 2)]
    fitted = fit(rows)
    report = fitted["solver_report"]
    assert report["converged"] is True
    assert report["timing"]["reference_class"] == 20
    frequencies = report["timing"]["training_class_frequencies"]
    assert len(frequencies) == 21
    assert {index for index, value in enumerate(frequencies) if value > 0} == set(buckets)
    assert min(report["timing"]["training_class_prior"]) > 0.
    assert sum(report["timing"]["training_class_prior"]) == pytest.approx(1.)
    assert np.asarray(fitted["heads"]["timing"]).shape == (21, 4)
    assert fitted["heads"]["timing"][20] == [0.] * 4
    for values in ([840., .0042, 7.], [1560., .0058, 7.]):
        logits = np.asarray(head_outputs(fitted["heads"], values)["timing_logits"])
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        assert np.isfinite(logits).all()
        assert len(probabilities) == 21
        assert np.all(probabilities > 0.)
        assert probabilities.sum() == pytest.approx(1., abs=1e-12)
        assert np.std(probabilities) > .01
    with pytest.raises(ValueError, match="SPARSE_TIMING_CLASSES_REQUIRE_DECLARED_R2_RIDGE"):
        fit(rows, regularization=0.)


def test_low_budget_reports_nonconvergence_without_approval():
    fitted = fit(synthetic_rows(), iterations=1)
    report = fitted["solver_report"]
    assert fitted["fit_status"] == report["status"] == "NOT_CONVERGED"
    assert report["converged"] is False
    assert fitted["approved_for_live"] is False
    assert any(not subreport["converged"] for subreport in
               (report["price"], report["timing"], *report["binary"].values()))


@pytest.mark.parametrize("iterations,noisy", [(1, False), (2000, False), (2000, True)])
def test_price_dual_certificate_is_feasible_and_independently_reproducible(monkeypatch, iterations, noisy):
    rows = synthetic_rows()
    if noisy:
        for index, row in enumerate(rows):
            row["log_bottom_ratio"] += .006 * np.sin(index * 1.7) + .003 * np.cos(index * .31)
    certificates = []
    original = research._pinball_dual_certificate

    def record_certificate(design, weights, target, coefficients, regularization, scaled_dual, rho, proof=None):
        result = original(design, weights, target, coefficients, regularization, scaled_dual, rho, proof=proof)
        certificates.append((design.copy(), weights.copy(), target.copy(), scaled_dual.copy(), rho))
        return result

    monkeypatch.setattr(research, "_pinball_dual_certificate", record_certificate)
    fitted = fit(rows, iterations=iterations)
    report = fitted["solver_report"]
    price = report["price"]
    assert certificates
    design, weights, target, scaled_dual, rho = certificates[-1]
    center, scale = np.asarray(report["weighted_center"]), np.asarray(report["weighted_scale"])
    coefficients = standardized_coefficients(fitted["heads"]["price"], center, scale)
    assert price["duality_gap_tolerance"] == 1e-7
    assert price["dual_certificate"] == "WEIGHTED_PINBALL_BOX_ZERO_INTERCEPT_SLOPE_RIDGE_R2"
    assert price["convergence_criterion"] == "FEASIBLE_DUALITY_GAP"
    proof = price["dual_certificate_feasibility"]
    assert proof["convention"] == "V_EQUALS_NEGATIVE_CONSTRAINT_LAGRANGE_MULTIPLIER"
    assert proof["feasibility_tolerance"] == 1e-12
    assert len(proof["projected_dual_sha256"]) == 3

    gaps = []
    for index, quantile in enumerate((.1, .5, .9)):
        # lambda convention: prediction + residual - target = 0.
        lower, upper = -weights * quantile, weights * (1. - quantile)
        lagrange = independently_feasible_dual(rho * weights * scaled_dual[:, index], lower, upper)
        assert np.isfinite(lagrange).all()
        assert np.all(lagrange >= lower - 1e-13)
        assert np.all(lagrange <= upper + 1e-13)
        assert lagrange.sum() == pytest.approx(0., abs=1e-12)
        assert proof["box_violation"][index] == pytest.approx(0., abs=1e-13)
        assert proof["intercept_residual"][index] <= proof["feasibility_tolerance"]
        slope_gradient = design[:, 1:].T @ lagrange
        dual = -target @ lagrange - (slope_gradient @ slope_gradient) / (4. * REGULARIZATION)
        error = target - design @ coefficients[index]
        primal = weights @ np.maximum(quantile * error, (quantile - 1.) * error)
        primal += REGULARIZATION * np.sum(coefficients[index, 1:] ** 2)
        gap = primal - dual
        assert primal >= dual - 1e-10
        assert price["final_objective"][index] == pytest.approx(primal, abs=1e-10)
        assert price["dual_objective"][index] == pytest.approx(dual, abs=1e-10)
        assert price["duality_gap"][index] == pytest.approx(max(0., gap), abs=1e-10)
        gaps.append(gap)

    if price["converged"]:
        assert max(gaps) <= price["duality_gap_tolerance"] + 1e-10
    if iterations == 1:
        assert max(gaps) > price["duality_gap_tolerance"]
        assert price["converged"] is False
        assert fitted["fit_status"] == "NOT_CONVERGED"
    elif not noisy:
        assert price["converged"] is True
    assert fitted["approved_for_live"] is False


@pytest.mark.parametrize("change,pattern", [
    ("future_label", "UNMATURED_OR_FUTURE_LABEL"),
    ("zero_weight_future_label", "UNMATURED_OR_FUTURE_LABEL"),
    ("label_at_observation", "UNMATURED_OR_FUTURE_LABEL"),
    ("future_transform", "FUTURE_SCALER_LEAKAGE"),
    ("boolean_fill", "OBSERVED_BINARY_FILL_REQUIRED"),
    ("no_fill_conditional", "NO_FILL_CONDITIONAL_LABELS_MUST_BE_NULL"),
    ("joint_exceeds_rebound", "INVALID_CONDITIONAL_EVENT_LABELS"),
    ("timing_boolean", "INVALID_TIMING_LABEL"),
    ("timing_out_of_range", "INVALID_TIMING_LABEL"),
    ("positive_bottom", "BOTTOM_LABEL_CANNOT_EXCEED_CUTOFF_CLOSE"),
    ("nonfinite_feature", "INVALID_TRAINING_FEATURES"),
])
def test_invalid_or_future_training_rows_fail_closed(change, pattern):
    rows = synthetic_rows()
    if change == "future_label": rows[0]["label_available_at"] = "2025-02-03T00:00:00Z"
    elif change == "zero_weight_future_label": rows[0].update(label_available_at="2025-02-03T00:00:00Z", observation_weight=0.)
    elif change == "label_at_observation": rows[0]["label_available_at"] = rows[0]["cutoff"]
    elif change == "future_transform": rows[0]["feature_fit_cutoff"] = "2025-01-02T00:00:00Z"
    elif change == "boolean_fill": rows[0]["fill"] = False
    elif change == "no_fill_conditional": rows[0]["rebound"] = 0
    elif change == "joint_exceeds_rebound": rows[-1].update(rebound=0, joint=1)
    elif change == "timing_boolean": rows[0]["bottom_timing"] = False
    elif change == "timing_out_of_range": rows[0]["bottom_timing"] = 21
    elif change == "positive_bottom": rows[0]["log_bottom_ratio"] = .01
    elif change == "nonfinite_feature": rows[0]["features"][0] = np.inf
    with pytest.raises(ValueError, match=pattern):
        fit(rows)


@pytest.mark.parametrize("name", ["fill", "rebound", "joint"])
def test_binary_class_present_only_at_zero_weight_is_rejected(name):
    rows = synthetic_rows()
    excluded_class = 0 if name == "fill" else 1
    for row in rows:
        if (name == "fill" or row["fill"] == 1) and row[name] == excluded_class:
            row["observation_weight"] = 0.
    with pytest.raises(ValueError, match="BOTH_POSITIVE_WEIGHT_BINARY_CLASSES_REQUIRED:" + name):
        fit(rows)


def test_legacy_default_retains_existing_inference_golden():
    rows = synthetic_rows()
    default = fit_heads(rows, fit_cutoff=FIT_CUTOFF, iterations=5, regularization=REGULARIZATION)
    explicit = fit_heads(rows, fit_cutoff=FIT_CUTOFF, iterations=5,
                         regularization=REGULARIZATION, solver=None)
    assert default == explicit
    assert default["fit_status"] == "FIXED_ITERATION_OFFLINE_FIT"
    assert "solver_report" not in default
    assert default["approved_for_live"] is False
    outputs = head_outputs(default["heads"], [1560., .0048, 7.])
    assert outputs["log_quantiles"] == pytest.approx(
        [-.2578915195183588, -.14037342569149613, -.0758283802333177], abs=1e-12)
    assert outputs["raw_logits"] == pytest.approx(
        dict(fill=.220183353610157, rebound=.25982858011487564,
             joint=.02076633708747575), abs=1e-12)
    assert [outputs["timing_logits"][index] for index in (0, 5, 10, 15, 20)] == pytest.approx(
        [-.0352767667190702, -.02645647528873186, -.017636914644005137,
         -.008818088352414932, 0.], abs=1e-12)
