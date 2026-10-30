"""SYNTHETIC_ONLY fixed deployment shrink; no model-selection evidence."""
from copy import deepcopy

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.fill_calibration import profile, validate_profile, deploy_fill_pair, fit_fill_calibration
from domain.les_v7.localization import FrozenBundle
from domain.les_v7.training import _calibrate, train_offline
from tests.strategies.test_v7_fill_calibration_integration import matured_oof, settings, FIT_CUTOFF
from tests.strategies.test_v7_offline_training import synthetic_config, synthetic_dataset, synthetic_published_candidate, seal


@pytest.mark.parametrize("alpha", [.25, .5, 1.])
def test_fixed_alpha_deploys_real_converged_optimizer_pair_with_separate_receipt(alpha):
    rows = matured_oof(); original = fit_fill_calibration(rows, fit_cutoff=FIT_CUTOFF)
    declaration = profile(alpha)
    assert validate_profile(declaration) == declaration
    fitted = fit_fill_calibration(rows, fit_cutoff=FIT_CUTOFF, calibration_profile=declaration)
    assert fitted["pair"] == deploy_fill_pair(original["pair"], alpha)
    report = fitted["report"]
    assert report["profile"] == declaration
    assert report["optimizer_fit_report"] == original["report"]
    assert report["optimizer_fitted_pair"] == original["pair"]
    assert report["fitted_pair"] == fitted["pair"] and report["converged"] is True
    assert "final_objective" not in report
    assert report["optimizer_fit_report"]["final_objective"] <= report["optimizer_fit_report"]["initial_objective"]
    assert fit_fill_calibration(rows, fit_cutoff=FIT_CUTOFF, calibration_profile=profile()) == original


def test_profiles_have_unique_versions_and_shared_transform_is_exact():
    assert len({profile(alpha)["version"] for alpha in (.25, .5, 1.)}) == 3
    assert deploy_fill_pair([2., -.8], .25) == [1.25, -.2]
    assert "alpha" not in profile() and profile()["schema"].endswith("/v1")


@pytest.mark.parametrize("alpha", [0, .1, .75, 2., True, "0.25"])
def test_unreviewed_or_incomplete_alpha_is_not_an_optimizer_fallback(alpha):
    with pytest.raises(ValueError): profile(alpha)
    with pytest.raises(ValueError): deploy_fill_pair([1., 0.], alpha)


def test_fixed_shrink_does_not_bypass_missing_classes_or_failed_optimizer(monkeypatch):
    rows = matured_oof()
    for row in rows: row["labels"]["fill"] = 1
    with pytest.raises(ValueError, match="BOTH_FILL_CALIBRATION_CLASSES"):
        fit_fill_calibration(rows, fit_cutoff=FIT_CUTOFF, calibration_profile=profile(.25))
    from domain.les_v7 import fill_calibration
    def failed(_objective, initial, _iterations): return initial, dict(converged=False)
    monkeypatch.setattr(fill_calibration, "_lbfgs", failed)
    with pytest.raises(ValueError, match="SOLVER_NOT_CONVERGED"):
        fit_fill_calibration(matured_oof(), fit_cutoff=FIT_CUTOFF, calibration_profile=profile(.25))


def test_composite_wrapper_keeps_nonfill_parameters_and_version_identifies_alpha():
    rows = matured_oof(); original, _ = _calibrate(rows, settings(), FIT_CUTOFF)
    config = settings(); config["fill_calibration_profile"] = profile(.5)
    current, report = _calibrate(rows, config, FIT_CUTOFF)
    for key in ("rebound", "joint", "timing_temperature", "method"):
        assert current[key] == original[key]
    assert current["version"].endswith("+" + profile(.5)["version"])
    assert report["fill_calibration_report"]["profile"] == profile(.5)
    assert report["fitted_parameters"]["fill"] == current["fill"]


@pytest.fixture(scope="module")
def published_candidate():
    config = synthetic_config(); config.update(head_solver="STANDARDIZED_CONVEX_R2", head_iterations=2000,
        fill_calibration_profile=profile(.5))
    result = train_offline(synthetic_dataset(config), config)
    assert result["status"] == "READY_TO_FREEZE"
    return synthetic_published_candidate(result["candidate"])


def test_native_published_model_reload_binds_actual_optimizer_and_fixed_deployment(published_candidate):
    model = FrozenBundle.load(seal(deepcopy(published_candidate))).record()
    assert model["fill_calibration_profile"] == profile(.5)
    assert model["calibration"]["version"].endswith(profile(.5)["version"])


@pytest.mark.parametrize("damage", ["optimizer_pair", "deployed_pair"])
def test_rehashed_component_cannot_claim_a_different_deployed_or_optimizer_pair(published_candidate, damage):
    model = deepcopy(published_candidate); old = model["calibration"]["validation_report_id"]
    report = model["validation_reports"].pop(old); component = report["fill_calibration_report"]
    if damage == "optimizer_pair": component["optimizer_fitted_pair"][0] += .1
    else:
        model["calibration"]["fill"][1] += .1
        component["fitted_pair"] = list(model["calibration"]["fill"])
        report["fitted_parameters"]["fill"] = list(model["calibration"]["fill"])
    new = identity(report); model["validation_reports"][new] = report
    model["validation_report_ids"]["calibration"] = new; model["calibration"]["validation_report_id"] = new
    with pytest.raises(ValueError, match="SHRINK_COMPONENT_BINDING"):
        FrozenBundle.load(seal(model))
