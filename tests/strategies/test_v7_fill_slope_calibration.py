"""Synthetic contract tests, never market acceptance evidence."""
from copy import deepcopy

import numpy as np
import pytest

from domain.les_v7.fill_calibration import fit_fill_calibration, validate_profile
from domain.les_v7.fill_slope_calibration import profile, verify_component
from domain.les_v7.localization import FrozenBundle
from domain.les_v7.training import _calibrate, train_offline
from tests.strategies.test_v7_fill_calibration_integration import matured_oof, settings, FIT_CUTOFF
from tests.strategies.test_v7_offline_training import synthetic_config, synthetic_dataset, synthetic_published_candidate, seal


@pytest.mark.parametrize("strength", [.001, .01, .1, 1.])
def test_free_intercept_matches_weighted_prevalence_and_retains_real_solver(strength):
    rows = matured_oof()
    for i, row in enumerate(rows): row["observation_weight"] = i + 1
    before = deepcopy(rows)
    fitted = fit_fill_calibration(rows, fit_cutoff=FIT_CUTOFF, calibration_profile=profile(strength))
    report = fitted["report"]
    assert rows == before
    assert report["profile"] == validate_profile(profile(strength))
    verify_component(report)
    assert report["after"]["mean_predicted_fill"] == pytest.approx(report["weighted_fill_prevalence"], abs=1e-6)
    assert report["final_objective"] <= report["initial_objective"]
    # The same standardized objective is invariant to a positive affine logit coordinate change.
    changed = deepcopy(rows)
    for row in changed: row["raw_logits"]["fill"] = 3 * row["raw_logits"]["fill"] + 7
    other = fit_fill_calibration(changed, fit_cutoff=FIT_CUTOFF, calibration_profile=profile(strength))
    for a, b in zip(rows, changed):
        def p(fit, row):
            a, b = fit["pair"]
            return 1 / (1 + np.exp(-(a * row["raw_logits"]["fill"] + b)))
        assert p(fitted, a) == pytest.approx(p(other, b), abs=1e-10)


def test_unavailable_labels_bad_declarations_and_failed_solver_are_not_fallbacks(monkeypatch):
    rows = matured_oof(); rows[0]["label_available_at"] = "2030-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="STRICTLY_PAST"):
        fit_fill_calibration(rows, fit_cutoff=FIT_CUTOFF, calibration_profile=profile(.1))
    for value in (True, .02, "0.1", None, float("nan")):
        with pytest.raises(ValueError): profile(value)
    declared = profile(.1); declared["intercept_penalty"] = .01
    with pytest.raises(ValueError, match="PROFILE_MISMATCH"): validate_profile(declared)
    import domain.les_v7.fill_slope_calibration as module
    monkeypatch.setattr(module, "_lbfgs", lambda f, initial, iterations: (initial, {"converged": False}))
    with pytest.raises(ValueError, match="SOLVER_NOT_CONVERGED"):
        fit_fill_calibration(matured_oof(), fit_cutoff=FIT_CUTOFF, calibration_profile=profile(.1))


def test_constant_logit_is_fitted_prevalence_not_identity_fallback():
    rows = matured_oof()
    for row in rows: row["raw_logits"]["fill"] = 7.
    fit = fit_fill_calibration(rows, fit_cutoff=FIT_CUTOFF, calibration_profile=profile(.1))
    assert fit["pair"][0] == 0.
    assert fit["report"]["solver_report"]["converged"] is True
    assert fit["report"]["weighted_logit_scale"] == 1e-8


def test_composite_preserves_other_heads_and_published_coordinates_cannot_be_relabelled():
    config = settings(); original, _ = _calibrate(matured_oof(), config, FIT_CUTOFF)
    config["fill_calibration_profile"] = profile(.1)
    fitted, _ = _calibrate(matured_oof(), config, FIT_CUTOFF)
    for key in ("rebound", "joint", "timing_temperature", "method"): assert fitted[key] == original[key]
    config = synthetic_config(); config.update(head_solver="STANDARDIZED_CONVEX_R2", head_iterations=2000,
        fill_calibration_profile=profile(.1))
    result = train_offline(synthetic_dataset(config), config)
    assert result["status"] == "READY_TO_FREEZE"
    candidate = synthetic_published_candidate(result["candidate"])
    assert FrozenBundle.load(seal(candidate)).record()["fill_calibration_profile"] == profile(.1)
    from domain.les_v7.contracts import identity
    model = deepcopy(candidate); old = model["calibration"]["validation_report_id"]
    report = model["validation_reports"].pop(old)
    report["fill_calibration_report"]["standardized_fitted_pair"][0] += .1
    new = identity(report); model["validation_reports"][new] = report
    model["validation_report_ids"]["calibration"] = new; model["calibration"]["validation_report_id"] = new
    with pytest.raises(ValueError, match="SLOPE_CALIBRATION_COMPONENT_BINDING"):
        FrozenBundle.load(seal(model))


def test_native_archive_publication_and_complete_reload_keep_the_fitted_component(tmp_path):
    import json
    from application.les_v7_training import train_request
    from infrastructure.storage.les_v7_training import LesV7TrainingRepository

    class SyntheticGuard:
        def retain(self, objects):
            return dict(source=objects.put(b"SYNTHETIC_TEST_ONLY source"),
                environment=objects.put_json(dict(kind="SYNTHETIC_TEST_ONLY environment")),
                parameters=objects.put_json(dict(kind="SYNTHETIC_TEST_ONLY parameters")))
        def check(self): pass

    config = synthetic_config(); config.update(head_solver="STANDARDIZED_CONVEX_R2", head_iterations=2000,
        fill_calibration_profile=profile(.1))
    dataset = synthetic_dataset(config)
    fitted = train_offline(dataset, config)
    repository = LesV7TrainingRepository(tmp_path, guard_factory=SyntheticGuard,
        clock=lambda: "2026-10-13T10:00:00+08:00")
    receipt = train_request(dataset, config, repository, allow_synthetic=True, trainer=lambda *_: deepcopy(fitted))
    folder = tmp_path / receipt['job_id']
    bundle = FrozenBundle.load(json.loads((folder / 'frozen_bundle.json').read_text()))
    assert bundle.record()['fill_calibration_profile'] == profile(.1)
    assert repository.verify(folder)['bundle_sha256'] == bundle.sha256
