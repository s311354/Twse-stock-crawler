"""Synthetic optimization and temporal proof checks; no market performance claim."""
from copy import deepcopy

import numpy as np
import pytest

from domain.les_v7 import fill_calibration as calibration

FIT = "2026-05-01T13:30:00+08:00"


def rows():
    result = []
    for index in range(120):
        value = -1.5 + index / 40
        result.append(dict(row_id=str(index), cutoff="2026-03-03T13:30:00+08:00",
            label_available_at="2026-03-31T13:30:00+08:00",
            head_fit_cutoff="2026-03-02T13:30:00+08:00",
            head_last_label_available_at="2026-02-27T13:30:00+08:00",
            raw_logits=dict(fill=value), labels=dict(fill=int(index % 5 < 4)),
            observation_weight=1. + index % 3))
    return result


def test_fixed_profile_is_exact_and_cannot_be_mutated_by_caller():
    profile = calibration.profile()
    assert profile["strength"] == 1.
    assert profile["iterations"] == 2000
    assert profile["prior_pair"] == [1., 0.]
    assert profile["no_identity_fallback"] is True
    profile["prior_pair"][0] = 0
    assert calibration.profile()["prior_pair"] == [1., 0.]


def test_actual_fit_is_deterministic_convex_stationary_and_preserves_inputs():
    sample = rows()
    original = deepcopy(sample)
    first = calibration.fit_fill_calibration(sample, fit_cutoff=FIT)
    second = calibration.fit_fill_calibration(sample, fit_cutoff=FIT)
    assert first == second
    assert sample == original
    report = first["report"]
    assert first["pair"] != [1., 0.]
    assert report["converged"] is True
    assert report["solver_report"]["iterations_accepted"] > 0
    assert report["final_objective"] <= report["initial_objective"]
    assert report["after"]["fill_nll"] < report["before"]["fill_nll"]
    assert report["evaluation_scope"] == "CALIBRATION_SET_NOT_HOLDOUT"
    assert report["other_heads"] == "UNCHANGED_NATIVE_R1"
    assert report["last_label_available_at"] == "2026-03-31T05:30:00+00:00"
    x = np.asarray([r["raw_logits"]["fill"] for r in sample])
    y = np.asarray([r["labels"]["fill"] for r in sample])
    w = np.asarray([r["observation_weight"] for r in sample]); w /= w.sum()
    pair = np.asarray(first["pair"])
    p = np.exp(-np.logaddexp(0., -(pair[0] * x + pair[1])))
    gradient = np.asarray([(w * (p - y)) @ x, (w * (p - y)).sum()]) + 2 * (pair - [1., 0.])
    assert np.max(np.abs(gradient)) <= 1e-6


def test_identity_prior_penalizes_intercept_as_declared_not_legacy_platt():
    sample = rows()
    for row in sample:
        row["raw_logits"]["fill"] = 0.
    fitted = calibration.fit_fill_calibration(sample, fit_cutoff=FIT)
    slope, intercept = fitted["pair"]
    assert slope == 1.
    assert 0 < intercept < .2  # Unpenalized 80% prevalence would need ~1.39.


def test_weight_rescaling_and_exact_replication_do_not_change_objective_fit():
    sample = rows()
    original = calibration.fit_fill_calibration(sample, fit_cutoff=FIT)
    scaled = deepcopy(sample)
    for row in scaled:
        row["observation_weight"] *= 1e300
    rescaled = calibration.fit_fill_calibration(scaled, fit_cutoff=FIT)
    duplicated = sample + [dict(r, row_id="repeat-" + r["row_id"]) for r in deepcopy(sample)]
    repeated = calibration.fit_fill_calibration(duplicated, fit_cutoff=FIT)
    np.testing.assert_allclose(original["pair"], rescaled["pair"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(original["pair"], repeated["pair"], atol=1e-12, rtol=0)


@pytest.mark.parametrize("field,value", [
    ("head_fit_cutoff", "2026-03-03T13:30:00+08:00"),
    ("head_last_label_available_at", "2026-03-03T13:30:00+08:00"),
    ("label_available_at", "2026-05-04T13:30:00+08:00"),
    ("label_available_at", "2026-03-03T13:30:00+08:00"),
])
def test_rejects_future_in_sample_or_unmatured_provenance(field, value):
    sample = rows(); sample[0][field] = value
    with pytest.raises(ValueError, match="STRICTLY_PAST_MATURE_OOF"):
        calibration.fit_fill_calibration(sample, fit_cutoff=FIT)


def test_label_mature_exactly_at_calibration_fit_is_allowed():
    sample = rows(); sample[0]["label_available_at"] = FIT
    assert calibration.fit_fill_calibration(sample, fit_cutoff=FIT)["report"]["converged"] is True


@pytest.mark.parametrize("field", ["cutoff", "head_fit_cutoff", "head_last_label_available_at", "label_available_at"])
def test_missing_or_naive_provenance_rejected(field):
    sample = rows(); del sample[0][field]
    with pytest.raises(ValueError, match="AWARE_PROVENANCE"):
        calibration.fit_fill_calibration(sample, fit_cutoff=FIT)
    sample = rows(); sample[0][field] = "2026-03-02T13:30:00"
    with pytest.raises(ValueError, match="AWARE_PROVENANCE"):
        calibration.fit_fill_calibration(sample, fit_cutoff=FIT)


def test_duplicate_ids_single_class_invalid_labels_and_bad_weights_rejected():
    sample = rows(); sample[1]["row_id"] = sample[0]["row_id"]
    with pytest.raises(ValueError, match="UNIQUE"):
        calibration.fit_fill_calibration(sample, fit_cutoff=FIT)
    sample = rows()
    for row in sample:
        row["labels"]["fill"] = 0
    with pytest.raises(ValueError, match="BOTH_FILL_CALIBRATION_CLASSES"):
        calibration.fit_fill_calibration(sample, fit_cutoff=FIT)
    for label in (True, None, .5):
        sample = rows(); sample[0]["labels"]["fill"] = label
        with pytest.raises(ValueError, match="OBSERVED_BINARY"):
            calibration.fit_fill_calibration(sample, fit_cutoff=FIT)
    for weight in (0, -1, float("nan"), float("inf"), True):
        sample = rows(); sample[0]["observation_weight"] = weight
        with pytest.raises(ValueError):
            calibration.fit_fill_calibration(sample, fit_cutoff=FIT)


def test_nonfinite_logits_and_unrepresentable_weight_ratios_rejected():
    for logit in (float("nan"), float("inf"), True):
        sample = rows(); sample[0]["raw_logits"]["fill"] = logit
        with pytest.raises(ValueError):
            calibration.fit_fill_calibration(sample, fit_cutoff=FIT)
    sample = rows(); sample[0]["observation_weight"] = 1e308; sample[1]["observation_weight"] = 1e-308
    with pytest.raises(ValueError, match="DYNAMIC_RANGE"):
        calibration.fit_fill_calibration(sample, fit_cutoff=FIT)


def test_solver_failure_never_returns_identity_fallback(monkeypatch):
    monkeypatch.setattr(calibration, "_lbfgs", lambda f, x, n: (x, {"converged": False}))
    with pytest.raises(ValueError, match="SOLVER_NOT_CONVERGED"):
        calibration.fit_fill_calibration(rows(), fit_cutoff=FIT)


def test_false_convergence_claim_rejected_by_objective_gradient(monkeypatch):
    monkeypatch.setattr(calibration, "_lbfgs", lambda f, x, n: (x, {"converged": True}))
    with pytest.raises(ValueError, match="SOLVER_NOT_CONVERGED"):
        calibration.fit_fill_calibration(rows(), fit_cutoff=FIT)
