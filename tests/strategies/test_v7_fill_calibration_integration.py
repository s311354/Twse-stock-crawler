"""SYNTHETIC_TEST_ONLY composite-calibration contracts, not market evidence."""
from copy import deepcopy

import pytest

from domain.les_v7.calibration import fit_calibration, score_calibration
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.fill_calibration import fit_fill_calibration, profile
from domain.les_v7.training import _calibrate, train_offline, validate_training_config
from tests.strategies.test_v7_calibration import rows
from tests.strategies.test_v7_offline_training import synthetic_config, synthetic_dataset


FIT_CUTOFF = "2025-02-02T00:00:00Z"


def matured_oof():
    return [dict(row, cutoff="2025-01-03T00:00:00Z", label_available_at="2025-02-01T00:00:00Z",
        head_fit_cutoff="2025-01-02T00:00:00Z", head_last_label_available_at="2025-01-01T00:00:00Z")
        for row in rows()]


def settings():
    return dict(min_calibration_rows=2, calibration_iterations=80, regularization=.01,
                fill_calibration_profile=profile())


def test_composite_changes_only_fill_and_binds_both_actual_component_fits():
    source = matured_oof(); before = deepcopy(source); config = settings()
    original = fit_calibration(source, iterations=80, regularization=.01)
    fill = fit_fill_calibration(source, fit_cutoff=FIT_CUTOFF)
    calibration, report = _calibrate(source, config, FIT_CUTOFF)
    assert source == before
    assert report["schema_version"] == "LES_V7_OOF_CALIBRATION_REPORT_R2"
    assert report["legacy_calibration_report"] == original["report"]
    assert report["fill_calibration_report"] == fill["report"]
    assert calibration["fill"] == fill["pair"]
    for key in ("timing_temperature", "rebound", "joint", "method"):
        assert calibration[key] == original["calibration"][key]
        assert report["fitted_parameters"][key] == original["report"]["fitted_parameters"][key]
    assert calibration["version"] == original["calibration"]["version"] + "+" + profile()["version"]
    assert report["fitted_parameters"]["version"] == calibration["version"]
    assert report["after"] == score_calibration(source, calibration)
    assert report["data_sha256"] == identity(source)
    assert report["row_ids"] == [row["row_id"] for row in source]
    assert timestamp(report["last_label_available_at"]) <= timestamp(report["fit_cutoff"])
    assert calibration["validation_report_id"] == identity(report)
    assert not {"objective", "regularization", "iterations_accepted"} & report.keys()
    assert report["evaluation_scope"] == "CALIBRATION_SET_NOT_HOLDOUT"


def test_absent_profile_retains_exact_original_calibrator_and_report():
    source, config = matured_oof(), settings(); del config["fill_calibration_profile"]
    original = fit_calibration(source, iterations=80, regularization=.01)
    calibration, report = _calibrate(source, config, FIT_CUTOFF)
    expected = dict(original["report"], fit_cutoff=timestamp(FIT_CUTOFF).isoformat(),
        last_label_available_at=max(timestamp(row["label_available_at"]) for row in source).isoformat())
    assert report == expected
    assert calibration == dict(original["calibration"], validation_report_id=identity(expected))


def test_optional_profile_cannot_change_predeclared_identity_prior():
    config = synthetic_config(); config["fill_calibration_profile"] = profile()
    validate_training_config(config)
    config["fill_calibration_profile"]["strength"] = .1
    with pytest.raises(ValueError, match="FILL_CALIBRATION_PROFILE_MISMATCH"):
        validate_training_config(config)


def test_future_or_same_time_head_prediction_cannot_be_used_by_optional_calibrator():
    source = matured_oof(); source[0]["head_fit_cutoff"] = source[0]["cutoff"]
    with pytest.raises(ValueError, match="CALIBRATION_FILL_STRICTLY_PAST"):
        _calibrate(source, settings(), FIT_CUTOFF)


def test_optional_native_training_exports_profile_and_composite_final_receipt():
    config = synthetic_config(); config.update(head_solver="STANDARDIZED_CONVEX_R2", head_iterations=2000,
                                               fill_calibration_profile=profile())
    result = train_offline(synthetic_dataset(config), config)
    assert result["status"] == "READY_TO_FREEZE"
    candidate = result["candidate"]
    assert candidate["fill_calibration_profile"] == profile()
    report = result["reports"][candidate["validation_report_ids"]["calibration"]]
    assert report["schema_version"] == "LES_V7_OOF_CALIBRATION_REPORT_R2"
    assert report["fitted_parameters"]["fill"] == candidate["calibration"]["fill"]
    assert report["fill_calibration_report"]["converged"] is True
    assert candidate["research_only"] is True and candidate["approved_for_live"] is False
