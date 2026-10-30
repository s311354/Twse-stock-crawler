"""SYNTHETIC_ONLY opt-in R2 training, inference, reload and evidence checks."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from domain.les_v7 import execution_atom_features as r2, execution_features as r1
from domain.les_v7.contracts import ReferencePlan, identity
from domain.les_v7.execution_profile import execution_feature_provider, validate_execution_profile
from domain.les_v7.localization import FrozenBundle, model_input_profile
from domain.les_v7.native_evidence import validate_native_receipts
from domain.les_v7.training import train_offline
from tests.strategies.test_low_entry_score_v7 import bundle, bars, CUTOFF
from tests.strategies.test_v7_execution_runtime import evaluate_model, native_report
from tests.strategies.test_v7_offline_training import synthetic_config, synthetic_dataset, synthetic_published_candidate, seal


def extended_bundle(*, full=False):
    record = bundle(); model = record["payload"]
    selected = list(r2.FEATURE_NAMES if full else r2.FEATURE_NAMES[:8]); definition = r2.profile()
    model.update(execution_feature_profile=definition,
        fill_only_feature_extension=dict(base_feature_names=["close"], profile_id=identity(definition), selected_features=selected),
        feature_names=["close"] + selected, feature_bounds=[[1., 200.]] + [list(x) for x in r2.FEATURE_BOUNDS[:len(selected)]],
        imputation=[None] * (len(selected) + 1), supported_masks=[[True] * (len(selected) + 1)])
    for name, value in model["heads"].items():
        array = np.asarray(value)
        model["heads"][name] = np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, len(selected))]).tolist()
    model["heads"]["fill_only_base_count"] = 1
    model["heads"]["fill"][model["feature_names"].index("execution_atr_band_logit60") + 1] = .4
    if full:
        model.update(fractal_mode=None, fractal_null_validation_report_id="SYNTHETIC_ONLY")
    record["sha256"] = identity(model)
    return record


@pytest.mark.parametrize("full", [False, True])
def test_registry_dispatch_and_reload_bind_exact_r2_coordinates(full):
    record = extended_bundle(full=full); model = record["payload"]
    assert execution_feature_provider(model) is r2
    assert validate_execution_profile(model)[1] == list(r2.FEATURE_NAMES if full else r2.FEATURE_NAMES[:8])
    assert FrozenBundle.load(record).record() == model
    legacy = bundle()["payload"]
    assert execution_feature_provider(legacy) is None
    from tests.strategies.test_v7_execution_runtime import extended_bundle as old_bundle
    assert execution_feature_provider(old_bundle()["payload"]) is r1


@pytest.mark.parametrize("damage", ["profile", "bounds", "protected", "bare", "reordered"])
def test_r2_metadata_or_protected_head_tampering_is_not_a_legacy_fallback(damage):
    record = extended_bundle(); model = record["payload"]
    if damage == "profile": model["execution_feature_profile"] = r1.profile()
    if damage == "bounds": model["feature_bounds"][-1][1] += .1
    if damage == "protected": model["heads"]["joint"][-1] = 1.
    if damage == "bare":
        del model["execution_feature_profile"]; del model["fill_only_feature_extension"]
    if damage == "reordered": model["feature_names"][-2:] = reversed(model["feature_names"][-2:])
    record["sha256"] = identity(model)
    with pytest.raises(ValueError): FrozenBundle.load(record)


def test_engine_r2_only_changes_fill_input_and_preserves_causal_base_receipts():
    original, extended = evaluate_model(bundle()), evaluate_model(extended_bundle())
    from domain.les_v7.report import validate_report
    validate_report(extended)
    for name in ("features", "feature_audit", "gate_bindings", "reference_plan"):
        assert original["diagnostics"][name] == extended["diagnostics"][name]
    for name in ("quantiles", "timing", "p_rebound", "p_entry"):
        assert original["diagnostics"]["localization"][name] == extended["diagnostics"]["localization"][name]
    assert original["diagnostics"]["localization"]["p_fill"] != extended["diagnostics"]["localization"]["p_fill"]
    assert extended["diagnostics"]["execution_feature_audit"]["profile"] == r2.EXECUTION_FEATURE_PROFILE
    future = bars(600).iloc[[-1]].copy(); future["Date"] = "2026-10-13"
    future[["Open", "High", "Low", "Close"]] = [10000., 20000., .01, 15000.]
    assert evaluate_model(extended_bundle(), pd.concat([bars(600), future], ignore_index=True)) == extended


def test_native_evidence_recomputes_r2_and_rejects_altered_numbers_or_atr_audit():
    original = native_report(); d = original["diagnostics"]
    record = extended_bundle(); model = record["payload"]
    d["model_input_profile"] = model_input_profile(model, record["sha256"])
    plan = ReferencePlan(**dict(d["reference_plan"], band=tuple(d["reference_plan"]["band"])))
    extra, audit = r2.compute_execution_features(pd.DataFrame(d["native_evidence"]["source_packet"]["market_rows"]),
        cutoff=CUTOFF, base_features=d["features"], plan=plan)
    d.update(execution_features={name: extra[name] for name in model["fill_only_feature_extension"]["selected_features"]},
        execution_feature_audit=audit)
    validate_native_receipts(original)
    for target in ("number", "audit"):
        changed = deepcopy(original)
        if target == "number": changed["diagnostics"]["execution_features"]["execution_atom_band_logit60"] += .1
        else: changed["diagnostics"]["execution_feature_audit"]["native_atr"]["last_value"] += .1
        with pytest.raises(ValueError, match="EXECUTION_FEATURE_RECOMPUTATION"):
            validate_native_receipts(changed)


def test_native_r2_training_and_published_reload_preserve_protected_and_m0_dimensions():
    config = synthetic_config(); base = list(config["feature_names"]); definition = r2.profile()
    config.update(feature_names=base + list(r2.FEATURE_NAMES),
        feature_bounds=config["feature_bounds"] + [list(x) for x in r2.FEATURE_BOUNDS],
        execution_feature_profile=definition,
        fill_only_feature_extension=dict(base_feature_names=base, profile_id=identity(definition), selected_features=list(r2.FEATURE_NAMES)),
        head_solver="STANDARDIZED_CONVEX_R2", head_iterations=2000)
    result = train_offline(synthetic_dataset(config), config)
    assert result["status"] == "READY_TO_FREEZE"
    model = FrozenBundle.load(seal(synthetic_published_candidate(result["candidate"]))).record()
    assert model["execution_feature_profile"] == definition
    assert model["heads"]["fill_only_base_count"] == 2
    baseline = result["reports"][result["baseline_head_fit_report_id"]]
    assert baseline["fill_only_partition"]["base_feature_count"] == 1
    assert baseline["fill_only_partition"]["extension_feature_count"] == 8
    assert model["research_only"] is True and model["approved_for_live"] is False


def test_fr2_optional_overlay_recomputes_the_same_r2_predictor():
    from domain.les_v7.fr2_policy import apply_fr2
    from domain.les_v7.report import validate_report
    from tests.test_v7_fr2_optional_compat import extended_request
    from tests.test_v7_fr2_policy import native
    item = extended_request(); old = item["context"]["model_bundle"]["payload"]
    record = extended_bundle()
    record["payload"].update(quality_policy=deepcopy(old["quality_policy"]), weights=deepcopy(old["weights"]))
    record["sha256"] = identity(record["payload"])
    item["context"]["model_bundle"] = record
    original = native(item); result = apply_fr2(original, record)
    validate_report(result)
    assert result["diagnostics"]["features"] == original["diagnostics"]["features"]
    assert result["diagnostics"]["execution_features"] == original["diagnostics"]["execution_features"]
    assert result["diagnostics"]["localization"] == original["diagnostics"]["localization"]
