"""SYNTHETIC_TEST_ONLY engineering checks; no investment performance evidence."""
from copy import deepcopy

import numpy as np
import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.execution_features import FEATURE_NAMES, FEATURE_BOUNDS, profile
from domain.les_v7.execution_profile import validate_execution_profile, execution_base_count
from domain.les_v7.fractals import is_active_fractal_feature
from domain.les_v7.localization import head_outputs
from domain.les_v7.research import fit_heads
from domain.les_v7.training import reference_config_id, train_offline, validate_training_config
from tests.strategies.test_v7_offline_training import synthetic_config, synthetic_dataset


FIT_CUTOFF = "2025-02-02T00:00:00Z"


def declared_config(*, interactions=True):
    base = ["ret5", "dfa256", "higuchi256", "context_REVERSAL"]
    selected = list(FEATURE_NAMES if interactions else FEATURE_NAMES[:13])
    definition = profile()
    return dict(feature_names=base + selected,
        feature_bounds=[[-10., 10.]] * len(base) + [list(FEATURE_BOUNDS[FEATURE_NAMES.index(name)]) for name in selected],
        execution_feature_profile=definition,
        fill_only_feature_extension=dict(base_feature_names=base, profile_id=identity(definition), selected_features=selected))


def test_profile_is_explicit_exact_and_ablation_removes_all_fractal_derived_inputs():
    assert validate_execution_profile({"feature_names": ["old"]}) is None
    for interactions in (False, True):
        config = declared_config(interactions=interactions)
        base, selected = validate_execution_profile(config)
        assert config["feature_names"] == base + selected
        assert execution_base_count(config, config["feature_names"]) == 4
        ablated = [name for name in config["feature_names"] if not is_active_fractal_feature(name)]
        assert execution_base_count(config, ablated) == 2
        base.clear(); selected.clear()
        assert validate_execution_profile(config)[0]


@pytest.mark.parametrize("change", ["missing", "definition", "id", "order", "bounds", "selected", "extra", "base"])
def test_profile_rejects_partial_relabelled_reordered_or_rebounded_features(change):
    config = declared_config()
    if change == "missing": del config["execution_feature_profile"]
    if change == "definition": config["execution_feature_profile"]["volatility_denominator_floor"] *= 2
    if change == "id": config["fill_only_feature_extension"]["profile_id"] = "0" * 64
    if change == "order": config["feature_names"][-2:] = reversed(config["feature_names"][-2:])
    if change == "bounds": config["feature_bounds"][-1][1] += 1
    if change == "selected": config["fill_only_feature_extension"]["selected_features"] = list(FEATURE_NAMES[:12])
    if change == "extra": config["fill_only_feature_extension"]["choose_best_on_holdout"] = True
    if change == "base": config["fill_only_feature_extension"]["base_feature_names"][0] = []
    with pytest.raises(ValueError): validate_execution_profile(config)


def test_active_partition_cannot_move_extension_into_protected_heads():
    config = declared_config()
    with pytest.raises(ValueError, match="ACTIVE_FEATURE_PARTITION"):
        execution_base_count(config, config["feature_names"][4:] + config["feature_names"][:4])
    with pytest.raises(ValueError, match="ACTIVE_FEATURE_PARTITION"):
        execution_base_count(config, config["feature_names"] + ["unknown"])


def test_bare_extension_coordinates_cannot_masquerade_as_a_legacy_model():
    config = declared_config()
    del config["execution_feature_profile"]; del config["fill_only_feature_extension"]
    with pytest.raises(ValueError, match="PROFILE_DECLARATION_REQUIRED"):
        validate_execution_profile(config)
    assert validate_execution_profile(dict(feature_names=["legacy_unknown_feature"])) is None


def invented_rows():
    rows = []
    for index in range(252):
        x = float(np.sin(index * .29)); z = float(np.cos(index * .43))
        execution = (index % 12 - 5.5) / 3.
        fill = int(execution > 0)
        rebound = int(x > 0) if fill else None
        joint = int(rebound and z > 0) if fill else None
        rows.append(dict(row_id=f"SYNTHETIC_FILL_EXTENSION_ONLY:{index}",
            cutoff="2025-01-01T00:00:00Z", feature_fit_cutoff="2025-01-01T00:00:00Z",
            label_available_at="2025-02-01T00:00:00Z", features=[x * 100., z * .01, 7., execution, x * execution],
            fill=fill, rebound=rebound, joint=joint, log_bottom_ratio=-.1 + .025 * x - .01 * z,
            bottom_timing=index % 21, observation_weight=1. + index % 3))
    return rows


def fitted(rows, **options):
    return fit_heads(rows, fit_cutoff=FIT_CUTOFF, iterations=2000,
        regularization=.005, solver="STANDARDIZED_CONVEX_R2", **options)


def test_only_fill_uses_new_coordinates_and_protected_heads_keep_original_coefficients_and_outputs():
    rows = invented_rows(); before = deepcopy(rows)
    base_rows = [dict(row, features=row["features"][:3]) for row in rows]
    base, extended = fitted(base_rows), fitted(rows, fill_only_base_count=3)
    assert rows == before
    assert extended["solver_report"]["base_report"] == base["solver_report"]
    assert extended["heads"]["fill_only_base_count"] == 3
    for name in ("price", "timing", "rebound", "joint"):
        original = np.asarray(base["heads"][name]); actual = np.asarray(extended["heads"][name])
        np.testing.assert_array_equal(actual[..., :4], original)
        np.testing.assert_array_equal(actual[..., 4:], 0.)
    values = rows[27]["features"]
    original = head_outputs(base["heads"], values[:3])
    first = head_outputs(extended["heads"], values)
    second = head_outputs(extended["heads"], values[:3] + [values[3] + 2., values[4] - 3.])
    for key in ("log_quantiles", "timing_logits"):
        assert first[key] == second[key] == original[key]
    for key in ("rebound", "joint"):
        assert first["raw_logits"][key] == second["raw_logits"][key] == original["raw_logits"][key]
    assert first["raw_logits"]["fill"] != second["raw_logits"]["fill"]
    report = extended["solver_report"]
    assert report["heads_sha256"] == identity(extended["heads"])
    assert report["raw_heads"] == extended["heads"]
    assert report["fill_only_partition"]["fill_feature_count"] == 5
    assert report["fill_only_partition"]["protected_heads"] == ["price", "timing", "rebound", "joint"]
    assert len(report["weighted_center"]) == len(report["weighted_scale"]) == 5
    assert report["converged"] is True


def test_fill_only_fit_is_deterministic_and_optional_none_retains_legacy_fit():
    rows = invented_rows()
    assert fitted(rows, fill_only_base_count=3) == fitted(deepcopy(rows), fill_only_base_count=3)
    assert fitted(rows) == fitted(rows, fill_only_base_count=None)


@pytest.mark.parametrize("count", [0, 5, 6, True, 1.5, "3"])
def test_fill_only_prefix_must_be_a_proper_integer_prefix(count):
    with pytest.raises(ValueError, match="BASE_FEATURE_COUNT_OR_SOLVER"):
        fitted(invented_rows(), fill_only_base_count=count)


def test_fill_only_extension_rejects_legacy_solver():
    with pytest.raises(ValueError, match="BASE_FEATURE_COUNT_OR_SOLVER"):
        fit_heads(invented_rows(), fit_cutoff=FIT_CUTOFF, fill_only_base_count=3)


def test_native_training_prepares_extension_and_keeps_ablation_partition_bound():
    config = synthetic_config(); original_reference = reference_config_id(config)
    base = list(config["feature_names"]); definition = profile()
    config.update(feature_names=base + list(FEATURE_NAMES),
        feature_bounds=config["feature_bounds"] + [list(bounds) for bounds in FEATURE_BOUNDS],
        execution_feature_profile=definition,
        fill_only_feature_extension=dict(base_feature_names=base, profile_id=identity(definition),
                                         selected_features=list(FEATURE_NAMES)),
        head_solver="STANDARDIZED_CONVEX_R2", head_iterations=2000)
    validate_training_config(config)
    assert reference_config_id(config) != original_reference
    result = train_offline(synthetic_dataset(config), config)
    assert result["status"] == "READY_TO_FREEZE"
    candidate = result["candidate"]
    assert candidate["execution_feature_profile"] == definition
    assert candidate["fill_only_feature_extension"] == config["fill_only_feature_extension"]
    assert candidate["heads"]["fill_only_base_count"] == 2
    ablation = result["reports"][result["baseline_head_fit_report_id"]]
    assert ablation["fill_only_partition"]["base_feature_count"] == 1
    assert ablation["fill_only_partition"]["extension_feature_count"] == 13
    assert candidate["research_only"] is True and candidate["approved_for_live"] is False
