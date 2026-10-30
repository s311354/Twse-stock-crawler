"""SYNTHETIC ONLY: FR3 contracts/recovery, not a real forecasting result."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.quality_policy import r3_quality_policy, r6_quality_policy
from infrastructure import les_v7_fr3_training as fr3
from infrastructure import les_v7_scope_training as native
from tests.strategies.test_v7_offline_training import synthetic_config
from tests.test_v7_dataset_scope import freeze, scope_inputs
from tests.test_v7_training_publication import Guard


NOW = "2026-10-23T14:00:00+08:00"


@pytest.fixture
def inputs(tmp_path):
    from domain.les_v7.fr3_plan import policy
    source = scope_inputs(tmp_path, count=3)
    # A newer algorithm may study these old source fixtures retrospectively;
    # it is not available in August and cannot be backdated as deployed.
    source["config"]["evaluation_cutoff"] = "2026-10-22T13:30:00+08:00"
    manifest = freeze(source, batch_size=2)
    build = deepcopy(source["config"])
    build["seed"]["reference_policy"] = policy(build["seed"]["reference_policy"], available_at=NOW)
    build["path_policy"] = deepcopy(build["seed"]["reference_policy"]["cost_path_policy"])
    config = synthetic_config()
    config.update(deepcopy(build["seed"]))
    config.update(quality_policy=r6_quality_policy(), evaluation_cutoff=build["evaluation_cutoff"],
        fit_cutoff="2026-07-01T13:30:00+08:00", validation_start="2026-08-03T13:30:00+08:00",
        validation_end="2026-08-05T13:30:00+08:00", folds=[
            dict(fold_id="SYNTHETIC-march", fit_cutoff="2026-03-02T13:30:00+08:00",
                prediction_start="2026-03-03T13:30:00+08:00", prediction_end="2026-03-13T13:30:00+08:00"),
            dict(fold_id="SYNTHETIC-april", fit_cutoff="2026-04-20T13:30:00+08:00",
                prediction_start="2026-04-21T13:30:00+08:00", prediction_end="2026-04-30T13:30:00+08:00")])
    return manifest, build, config, tmp_path / "SYNTHETIC-fr3"


def _run(inputs, *, prepare=False, **options):
    entry = fr3.prepare_fr3_scope if prepare else fr3.train_fr3_scope
    return entry(*inputs, allow_synthetic=True, guard_factory=Guard, clock=lambda: NOW, **options)


def test_fr3_exact_r6_plan_label_and_cost_binding_does_not_mutate_configs(inputs):
    from domain.les_v7.fr3_plan import LABEL_VERSION, PATH_VERSION, PLAN_VERSION
    _, build, config, _ = inputs
    before = deepcopy((build, config))
    components = fr3.validate_fr3_training_configs(build, config)
    assert (build, config) == before
    assert components["plan_version"] == PLAN_VERSION
    assert components["path_version"] == PATH_VERSION
    assert components["label_version"] == LABEL_VERSION
    assert components["quality_policy_sha256"] == identity(r6_quality_policy())
    assert components["source_interpretation"] == "RETROSPECTIVE_FIXED_RULES_NOT_DEPLOYMENT"
    assert components["holdout_interpretation"] == "VIEWED_HOLDOUT_RETROSPECTIVE"


@pytest.mark.parametrize("damage", ["quality", "legacy", "cost", "build_seed", "evaluation"])
def test_fr3_refuses_legacy_prepared_basis_or_inconsistent_inputs(inputs, damage):
    _, build, config, _ = inputs
    if damage == "quality":
        config["quality_policy"] = r3_quality_policy()
    elif damage == "legacy":
        config["reference_policy"]["version"] = "DECLARED_COUNTERFACTUAL_H20_RAW_PRICE_R1"
    elif damage == "cost":
        build["path_policy"]["slippage_bps"] = "1"
    elif damage == "build_seed":
        build["seed"]["feature_bounds"][0][0] -= 1
    else:
        config["evaluation_cutoff"] = "2026-10-21T13:30:00+08:00"
    with pytest.raises(ValueError):
        fr3.validate_fr3_training_configs(build, config)


def test_fr3_native_preparation_keeps_every_symbol_and_never_calls_fitter(inputs, monkeypatch):
    monkeypatch.setattr(native, "_train_prepared", lambda *_: pytest.fail("no fit in preparation"))
    result = _run(inputs, prepare=True)
    assert result["status"] == "DATASET_PREPARED"
    assert result["model_fitted"] is False
    assert result["native_root"] == str(inputs[3] / "native")
    assert not (inputs[3] / "fr3-training-completion.json").exists()
    request = json.loads((inputs[3] / "fr3-training-request.json").read_text())
    assert request["sha256"] == result["request_id"] == identity(request["payload"])
    assert request["payload"]["symbols"] == ["1000", "1001", "1002"]
    assert request["payload"]["original_bundle_modified"] is False
    dataset = json.loads((inputs[3] / "native/scope-training-dataset.json").read_text())["payload"]
    assert [s for child in dataset["children"] for s in child["symbols"]] == ["1000", "1001", "1002"]
    assert all(row["prepared_rows"] == 0 for child in dataset["children"] for row in child["coverage"])
    assert _run(inputs, prepare=True) == result


def test_fr3_not_ready_does_not_replace_missing_training_with_old_heads(inputs):
    prepared = _run(inputs, prepare=True)
    outcome = _run(inputs)
    assert outcome["status"] == "NOT_READY"
    assert outcome["native_outcome"]["bundle_sha256"] is None
    assert outcome["native_outcome"]["dataset_id"] == prepared["native_preparation"]["dataset_id"]
    assert not list(inputs[3].rglob("frozen_bundle.json"))
    assert outcome["original_bundle_modified"] is outcome["ordinary_daily_modified"] is False
    assert fr3.verify_fr3_scope_training(inputs[3]) == outcome
    assert _run(inputs) == outcome


@pytest.mark.parametrize("damage", ["symbols", "native", "component", "authority"])
def test_fr3_completion_reverification_rejects_tampering(inputs, damage):
    _run(inputs)
    path = inputs[3] / "fr3-training-completion.json"
    outcome = json.loads(path.read_text())
    if damage == "symbols":
        outcome["symbols"].pop()
    elif damage == "native":
        outcome["native_outcome"]["counts"]["prepared"] = 999
    elif damage == "component":
        outcome["components"]["label_version"] = "OLD_LABELS"
    else:
        outcome["approved_for_live"] = True
    path.write_text(json.dumps(outcome))
    with pytest.raises(ValueError, match="FR3_TRAINING_COMPLETION_BINDING_MISMATCH"):
        fr3.verify_fr3_scope_training(inputs[3])


def test_fr3_retry_refuses_changed_configuration_before_native_work(inputs, monkeypatch):
    _run(inputs, prepare=True)
    inputs[2]["head_iterations"] += 1
    monkeypatch.setattr(fr3, "train_elec_scope", lambda *_a, **_kw: pytest.fail("native work on changed request"))
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        _run(inputs)


def test_fr3_prepare_only_is_a_distinct_entry_not_a_completed_model(inputs):
    with pytest.raises(ValueError, match="USE_PUBLIC_PREPARATION_ENTRY"):
        _run(inputs, prepare_only=True)
    assert not inputs[3].exists()


def test_fr3_does_not_write_into_a_legacy_or_unrelated_output_folder(inputs):
    root = inputs[3]
    root.mkdir()
    old = root / "scope-training-config.json"
    old.write_text('{"immutable_old_archive":true}')
    original = old.read_bytes()
    with pytest.raises(ValueError, match="FR3_NEW_OUTPUT_ROOT_REQUIRED"):
        _run(inputs)
    assert old.read_bytes() == original
    assert list(root.iterdir()) == [old]


def test_fr3_preview_builds_native_new_labels_without_writing_or_fitting(inputs):
    manifest, build, config, root = inputs
    before = Path(manifest).read_bytes()
    result = fr3.preview_fr3_shard(manifest, build, config, index=0,
        allow_synthetic=True, guard_factory=Guard)
    assert result["symbols"] == ["1000", "1001"]
    assert result["model_fitted"] is result["rows_reused_for_training"] is False
    assert result["prepared_count"] == 0
    assert result["native_counts"]["candidate_events"] == 6
    assert result["seconds"]["total"] >= result["seconds"]["source_load"] >= 0
    assert not root.exists()
    assert Path(manifest).read_bytes() == before


def test_fr3_preview_refuses_invalid_or_boolean_chunk_index(inputs):
    for index in (-1, 2, True):
        with pytest.raises(ValueError, match="FR3_PREVIEW_SHARD_INDEX_INVALID"):
            fr3.preview_fr3_shard(*inputs[:3], index=index, allow_synthetic=True, guard_factory=Guard)
