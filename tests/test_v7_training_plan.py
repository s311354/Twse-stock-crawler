"""Offline configuration engineering checks, not trained market results."""
from copy import deepcopy
import json
from math import log
from pathlib import Path

import pytest

from application.les_v7_dataset import validate_build_config
from application.les_v7_training_plan import generate_fixed_research_plan, R3_FEATURES, R3_BOUNDS
from domain.les_v7.contracts import identity
from domain.les_v7.decision import rule
from domain.les_v7.quality_policy import default_quality_policy, r2_quality_policy
from domain.les_v7.training import reference_config_id, validate_training_config
from infrastructure.les_v7_training_plan import freeze_fixed_research_plan


def seed():
    # Existing declared real reference plan, no test prices or invented labels.
    source = Path(__file__).parents[1] / "examples/les_v7_dataset_config.json"
    return json.loads(source.read_text())["seed"]


def test_matched_native_configs_preserve_reference_seed_and_context_support_dates():
    source = seed()
    before = deepcopy(source)
    plan = generate_fixed_research_plan(source)
    build, train = plan["build_config"], plan["training_config"]
    validate_build_config(build)
    validate_training_config(train)
    assert source == before
    assert build["seed"] == source
    assert reference_config_id(train) == reference_config_id(source)
    assert train["reference_policy"] == source["reference_policy"]
    assert build["path_policy"] == source["reference_policy"]["cost_path_policy"]
    assert plan["history_start"] == "2023-10-01"
    assert build["event_start"] == "2026-01-14" and build["event_end"] == "2026-07-09"
    assert train["fit_cutoff"] == "2026-06-30T13:30:00+08:00"
    assert train["validation_start"] == "2026-07-01T13:30:00+08:00"
    assert train["validation_end"] == "2026-07-09T13:30:00+08:00"
    assert [fold["fold_id"] for fold in train["folds"]] == [
        "past-only-2026-march", "past-only-2026-april", "past-only-2026-may"]
    assert build["evaluation_cutoff"] == train["evaluation_cutoff"] == "2026-08-07T13:30:00+08:00"
    assert build["max_events"] == 4096
    assert "reference_evidence_policy" not in build and "reference_evidence_policy" not in train
    assert train["acceptance"] == dict(max_brier=.25, max_timing_nll=log(21), max_pinball_loss=.05)
    assert not plan["fitting_performed"] and not plan["approved_for_live"]
    assert plan["readiness"]["status"] == "NOT_ASSESSED_SETTINGS_ONLY"


def test_each_plan_is_detached_from_seed_and_other_config():
    source = seed()
    one, two = generate_fixed_research_plan(source), generate_fixed_research_plan(source)
    one["training_config"]["reference_policy"]["entry_band_atr"][0] = -9
    assert two["training_config"]["reference_policy"] == source["reference_policy"]
    assert one["build_config"]["seed"] == source


def test_r3_fixed_profile_keeps_criteria_and_uses_a_new_independent_holdout(tmp_path):
    source = seed()
    source.update(feature_names=deepcopy(R3_FEATURES), feature_bounds=deepcopy(R3_BOUNDS))
    plan = generate_fixed_research_plan(source)
    build, train = plan['build_config'], plan['training_config']
    assert plan['policy_id'] == 'ELEC_H20_CAUSAL_GEOMETRY_CONVEX_R3'
    assert train['head_solver'] == 'STANDARDIZED_CONVEX_R2'
    assert train['feature_names'] == R3_FEATURES
    assert train['acceptance'] == dict(max_brier=.25, max_timing_nll=log(21), max_pinball_loss=.05)
    assert train['min_context_weight_rows'] == 100
    assert train['validation_start'].startswith('2026-09-01')
    assert train['validation_end'].startswith('2026-09-07')
    assert train['fit_cutoff'].startswith('2026-08-28')
    assert build['fixed_universe_date'] == plan['universe_date'] == '2026-08-07'
    assert build['evaluation_cutoff'].startswith('2026-10-12')
    assert build['event_end'] == '2026-09-07'
    assert plan['history_start'] == '2023-07-03'
    assert plan['interpretation']['development_only_ranges'] == ['2026-07-01/2026-07-09']
    assert not plan['fitting_performed']
    assert freeze_fixed_research_plan(plan, tmp_path)['plan_id'] == identity(plan)
    changed = deepcopy(source)
    changed['feature_bounds'][6][1] = 2
    with pytest.raises(ValueError, match='R3_SUPPORT_BOUNDS'):
        generate_fixed_research_plan(changed)


def test_final_reference_reconstruction_is_explicit_and_source_seed_bound(tmp_path):
    source = seed()
    source["reference_policy"]["cost_path_policy"]["coordinate_policy"] = "OFFICIAL_FINAL_ACTIONS_RECONSTRUCTION_R1"
    plan = generate_fixed_research_plan(source)
    assert plan["build_config"]["reference_evidence_policy"] == "OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_R1"
    assert plan["training_config"]["reference_evidence_policy"] == "OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_R1"
    assert plan["build_config"]["seed"] == source
    assert reference_config_id(plan["training_config"]) == reference_config_id(source)
    receipt = freeze_fixed_research_plan(plan, tmp_path)
    assert receipt["plan_id"] == identity(plan)
    assert not plan["approved_for_live"]
    assert "RETROSPECTIVE" in plan["interpretation"]["availability"]


def test_registered_quality_policy_is_reused_not_replaced_or_guessed():
    default = generate_fixed_research_plan(seed())
    assert default["training_config"]["quality_policy"] == default_quality_policy()
    explicit = r2_quality_policy()
    assert generate_fixed_research_plan(seed(), quality_policy=explicit)["training_config"]["quality_policy"] == explicit
    explicit["policy_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="QUALITY_POLICY_HASH"):
        generate_fixed_research_plan(seed(), quality_policy=explicit)
    with pytest.raises(ValueError, match="REGISTERED_SOURCE_BOUND_QUALITY"):
        generate_fixed_research_plan(seed(), quality_policy={"profile_id": "UNKNOWN"})


def test_causal_trigger_conditions_remain_unknown_when_sources_missing():
    rules = generate_fixed_research_plan(seed())["training_config"]["trigger_rules"]
    assert rule(rules["near_pullback_support"], {}) is None
    assert rule(rules["near_pullback_support"], dict(extension=.5, support_intact=True)) is True
    assert rule(rules["near_pullback_support"], dict(extension=2., support_intact=True)) is False
    assert rule(rules["tight_base_or_retest"], dict(atr_expansion=.8, reclaim=False)) is True
    assert rule(rules["tight_base_or_retest"], dict(atr_expansion=1.4, reclaim=False)) is False


@pytest.mark.parametrize("bad", ["full_fit_config", "synthetic", "other_features"])
def test_real_plan_rejects_full_synthetic_or_unreviewed_feature_config(bad):
    value = seed()
    if bad == "full_fit_config":
        value["schema"] = "les-v7-training-config/v1"
    elif bad == "synthetic":
        value["reference_policy"]["version"] = "SYNTHETIC_TEST_ONLY"
    else:
        value["feature_names"] = ["slope5", "dfa1024"]
    with pytest.raises(ValueError):
        generate_fixed_research_plan(value)


def test_freeze_content_identity_stable_and_creation_time_retained(tmp_path):
    plan = generate_fixed_research_plan(seed())
    first = freeze_fixed_research_plan(plan, tmp_path, clock=lambda: "2026-10-13T15:00:00+08:00")
    again = freeze_fixed_research_plan(plan, tmp_path, clock=lambda: "2026-10-14T15:00:00+08:00")
    other = freeze_fixed_research_plan(plan, tmp_path / "other", clock=lambda: "2026-10-14T15:00:00+08:00")
    assert first == again
    assert first["plan_id"] == other["plan_id"] == identity(plan)
    assert first["created_at"] != other["created_at"]
    assert json.loads((Path(first["directory"]) / "training-config.json").read_text()) == plan["training_config"]


@pytest.mark.parametrize("tamper", ["seed", "threshold", "readiness"])
def test_changed_seed_or_policy_rejected_before_freeze(tmp_path, tamper):
    plan = generate_fixed_research_plan(seed())
    if tamper == "seed":
        plan["training_config"]["reference_policy"]["entry_band_atr"][1] = .5
    elif tamper == "threshold":
        plan["training_config"]["acceptance"]["max_brier"] = 1.
    else:
        plan["readiness"]["status"] = "READY"
    with pytest.raises(ValueError, match="MISMATCH"):
        freeze_fixed_research_plan(plan, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("problem", ["corrupt", "missing", "manifest"])
def test_complete_freeze_cannot_silently_repair_corrupt_reference(tmp_path, problem):
    plan = generate_fixed_research_plan(seed())
    receipt = freeze_fixed_research_plan(plan, tmp_path)
    folder = Path(receipt["directory"])
    if problem == "corrupt":
        (folder / "training-config.json").write_text("{}")
    elif problem == "missing":
        (folder / "training-config.json").unlink()
    else:
        (folder / "manifest.json").write_text("{}")
    with pytest.raises((ValueError, OSError)):
        freeze_fixed_research_plan(plan, tmp_path)


def test_interrupted_before_manifest_retries_without_replacing_inputs(tmp_path, monkeypatch):
    import infrastructure.les_v7_training_plan as module
    plan = generate_fixed_research_plan(seed())
    original = module.write_once
    def stop_before_manifest(path, value):
        if path.name == "manifest.json":
            raise OSError("interrupted")
        original(path, value)
    monkeypatch.setattr(module, "write_once", stop_before_manifest)
    with pytest.raises(OSError, match="interrupted"):
        freeze_fixed_research_plan(plan, tmp_path)
    folder = tmp_path / identity(plan)
    before = (folder / "training-config.json").read_bytes()
    assert not (folder / "manifest.json").exists()
    monkeypatch.setattr(module, "write_once", original)
    receipt = freeze_fixed_research_plan(plan, tmp_path)
    assert (folder / "training-config.json").read_bytes() == before
    assert receipt["plan_id"] == identity(plan)


def test_freeze_does_not_pretend_creation_before_evaluation(tmp_path):
    with pytest.raises(ValueError, match="EVALUATION_AFTER_FREEZE_CLOCK"):
        freeze_fixed_research_plan(generate_fixed_research_plan(seed()), tmp_path,
                                   clock=lambda: "2026-08-01T13:30:00+08:00")
    assert not list(tmp_path.iterdir())
