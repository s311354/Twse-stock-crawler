"""Synthetic-only predictor display lineage; not training or trade approval."""
from copy import deepcopy

import pytest

from domain.les_v7.engine import evaluate
from domain.les_v7.report import validate_report
from tests.strategies.test_low_entry_score_v7 import bars, bundle, seal, CUTOFF, POLICY


def report_for(model=None):
    return evaluate(bars(90), dict(symbol="SYNTHETIC_ONLY", market="TWSE", as_of=CUTOFF,
        filings=[], quality_policy=deepcopy(POLICY), model_bundle=model))


@pytest.fixture(scope="module")
def no_fractal_report():
    return report_for(bundle())


def test_profile_preserves_actual_bundle_inputs_not_computed_fractal_diagnostics(no_fractal_report):
    report = no_fractal_report
    assert report["diagnostics"]["model_input_profile"] == dict(
        schema="les-v7-model-input-profile/v1", bundle_sha256=bundle()["sha256"],
        feature_names=["close"], fractal_mode="V7_NO_FRACTAL_BASELINE")
    assert "dfa512" in report["diagnostics"]["features"]
    validate_report(report)


def test_active_fractal_profile_keeps_actual_missing_mode_and_does_not_imply_readiness():
    model = bundle()
    model["payload"].pop("fractal_mode")
    model["payload"]["feature_names"] = ["delta_dfa512_5"]
    model["payload"]["feature_bounds"] = [[-10., 10.]]
    seal(model)
    report = report_for(model)
    profile = report["diagnostics"]["model_input_profile"]
    assert profile["bundle_sha256"] == model["sha256"]
    assert profile["feature_names"] == ["delta_dfa512_5"] and profile["fractal_mode"] is None
    assert report["envelope"]["localization"]["status"] != "VALID"
    validate_report(report)


def test_missing_profile_is_legacy_compatible_and_does_not_change_business_ids(no_fractal_report):
    legacy = deepcopy(no_fractal_report)
    legacy["diagnostics"].pop("model_input_profile")
    validate_report(legacy)
    assert legacy["envelope"] == no_fractal_report["envelope"]
    assert legacy["parameter_version"] == no_fractal_report["parameter_version"]


@pytest.mark.parametrize("change,expected", [
    (dict(bundle_sha256="0" * 64), "BINDING_MISMATCH"),
    (dict(schema="unrecognised"), "BINDING_MISMATCH"),
    (dict(feature_names=["close", "close"]), "PROFILE_INVALID"),
    (dict(feature_names=[]), "PROFILE_INVALID"),
    (dict(feature_names=["x"] * 65), "PROFILE_INVALID"),
    (dict(feature_names=["x" * 81]), "PROFILE_INVALID"),
    (dict(feature_names=[{}]), "PROFILE_INVALID"),
    (dict(fractal_mode={}), "PROFILE_INVALID"),
    (dict(feature_names=["dfa512"]), "FRACTAL_MODE_CONFLICT"),
    (dict(feature_names=["delta_dfa512_5"]), "FRACTAL_MODE_CONFLICT"),
])
def test_profile_tampering_is_rejected(no_fractal_report, change, expected):
    report = deepcopy(no_fractal_report)
    report["diagnostics"]["model_input_profile"].update(change)
    with pytest.raises(ValueError, match=expected):
        validate_report(report)


def test_no_model_does_not_claim_predictor_profile(no_fractal_report):
    report = report_for()
    assert "model_input_profile" not in report["diagnostics"]
    validate_report(report)
    report["diagnostics"]["model_input_profile"] = deepcopy(
        no_fractal_report["diagnostics"]["model_input_profile"])
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        validate_report(report)


def test_fr2_profile_matches_retained_predictor_and_restores_original():
    from tests.test_v7_fr2_policy import native, apply_fr2
    from domain.les_v7.fr2_policy import restore_original
    original = native()
    report = apply_fr2(original)
    validate_report(report)
    assert restore_original(report) == original
    assert report["diagnostics"]["model_input_profile"] == original["diagnostics"]["model_input_profile"]
    report["diagnostics"]["model_input_profile"]["feature_names"] = ["unused_close"]
    with pytest.raises(ValueError, match="PROFILE_PREDICTOR_MISMATCH"):
        validate_report(report)
