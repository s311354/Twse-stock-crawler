"""SYNTHETIC ONLY: isolated FR3 CLI routing and frozen settings, no real fit."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.fr3_plan import PLAN_VERSION
from domain.les_v7.quality_policy import r6_quality_policy
from interface import les_v7_fr3 as cli
from tests.test_v7_fr3_plan import reference


def test_freeze_configs_preserves_parent_and_rebuilds_parent_predictor_coordinates(tmp_path, monkeypatch):
    from infrastructure import les_v7_fr3_training as port
    from infrastructure.storage.les_v7_training import LesV7TrainingRepository
    model = tmp_path / "parent"; model.mkdir()
    parent = dict(payload={"SYNTHETIC_ONLY": True}, sha256="a"*64)
    old_training = dict(reference_policy=reference(), feature_names=["slope5", "dfa512"],
        feature_bounds=[[-2, 2], [0, 2]], context_rules={"SYNTHETIC_ONLY": True},
        folds=[dict(fold_id="UNCHANGED_SYNTHETIC_FOLD")], min_holdout_rows=500,
        fill_only_feature_extension={"SYNTHETIC": "PARENT_COORDINATES"},
        execution_feature_profile="PARENT_EXECUTION_PROFILE", fit_cutoff="2026-09-14T05:30:00Z")
    old_build = dict(seed=dict(feature_names=["OLD_BUILD_COORDINATES"], reference_policy=reference()),
                     evaluation_cutoff="2026-10-22T05:30:00Z")
    archive, config, build = model / "frozen_bundle.json", model / "config.json", tmp_path / "build.json"
    for path, value in ((archive, parent), (config, old_training), (build, old_build)):
        path.write_text(json.dumps(value))
    before = {p: p.read_bytes() for p in (archive, config, build)}
    monkeypatch.setattr(LesV7TrainingRepository, "verify", lambda _: {"bundle_sha256": parent["sha256"]})
    captured = []
    monkeypatch.setattr(port, "validate_fr3_training_configs", lambda b, t: captured.append((b, t)) or {"SYNTHETIC": True})
    folder = tmp_path / "new-settings"
    result = cli.freeze_fr3_configs(archive, build, folder)
    frozen_build, frozen_training = captured[0]
    assert frozen_build["seed"]["feature_names"] == old_training["feature_names"]
    assert frozen_build["seed"]["execution_feature_profile"] == old_training["execution_feature_profile"]
    assert frozen_training["folds"] == old_training["folds"]
    assert frozen_training["min_holdout_rows"] == 500
    assert frozen_training["reference_policy"]["version"] == PLAN_VERSION
    assert frozen_training["quality_policy"] == r6_quality_policy()
    assert result["payload"]["validation_posture"].startswith("VIEWED_HOLDOUT")
    assert {p: p.read_bytes() for p in before} == before
    assert cli.freeze_fr3_configs(archive, build, folder) == result


def test_fr3_cannot_write_settings_into_original_model_archive(tmp_path):
    folder = tmp_path / "parent"
    with pytest.raises(ValueError, match="NEW_OUTPUT"):
        cli.freeze_fr3_configs(folder / "frozen_bundle.json", tmp_path / "ignored", folder / "child")
    assert not folder.exists()


def test_training_cli_preserves_not_ready_instead_of_claiming_success(tmp_path, monkeypatch, capsys):
    from infrastructure import les_v7_fr3_training as port
    configs = []
    for name in ("build", "training"):
        path = tmp_path / (name + ".json"); path.write_text(json.dumps({"SYNTHETIC_ONLY": name}))
        configs.append(path)
    calls = []
    monkeypatch.setattr(port, "train_fr3_scope", lambda *args, **kwargs: calls.append((args, kwargs)) or {
        "status": "NOT_READY", "request_id": "a"*64})
    code = cli.main(["train", "--freeze", str(tmp_path / "freeze.json"), "--build-config", str(configs[0]),
        "--training-config", str(configs[1]), "--output-dir", str(tmp_path / "isolated")])
    assert code == 2 and len(calls) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "NOT_READY" and summary["mail_sent"] is False
    assert summary["approved_for_live"] is False


def test_old_daily_bundle_loader_refuses_fr3_before_daily_strategy_work(tmp_path, monkeypatch):
    import infrastructure.les_v7_research_bundle as loader
    folder = tmp_path / ("b"*64); folder.mkdir()
    archive = folder / "frozen_bundle.json"
    archive.write_text(json.dumps({"payload": {}, "sha256": "a"*64}))
    monkeypatch.setattr(loader.LesV7TrainingRepository, "verify", lambda _: dict(status="READY_TO_FREEZE",
        dataset_kind="RESEARCH", research_only=True, approved_for_live=False, execution_action="NO_ACTION"))
    monkeypatch.setattr(loader.FrozenBundle, "load", lambda _: SimpleNamespace(record=lambda: {
        "reference_policy": {"version": PLAN_VERSION}}))
    with pytest.raises(ValueError, match="INDEPENDENT_RESEARCH_ENTRY_REQUIRED"):
        loader.load_research_bundle(archive)


def test_analysis_refuses_to_publish_inside_original_archive_before_reading_inputs(tmp_path, capsys):
    parent = tmp_path / "immutable-model"
    with pytest.raises(SystemExit) as error:
        cli.main(["analyze", "--request", str(tmp_path / "does-not-exist.json"),
            "--bundle", str(parent / "frozen_bundle.json"), "--output-dir", str(parent / "reports")])
    assert error.value.code == 2
    assert "FR3_NEW_OUTPUT_REQUIRED_NOT_ORIGINAL_ARCHIVE" in capsys.readouterr().err
    assert not parent.exists()


def test_expansion_settings_cli_uses_verified_native_source_adapter(tmp_path, monkeypatch, capsys):
    from infrastructure import les_v7_fr3_sources as source
    calls = []
    monkeypatch.setattr(source, "freeze_fr3_expansion_settings", lambda *a, **k:
        calls.append((a, k)) or dict(status="CALENDAR_PLAN_ONLY_NOT_TRAINING_READY"))
    assert cli.main(["expand-configs", "--parent-training-root", str(tmp_path / "parent"),
        "--calendar", str(tmp_path / "calendar.json"), "--history-start", "2021-01-04",
        "--through", "2026-10-23", "--output-dir", str(tmp_path / "new")]) == 0
    assert len(calls) == 1 and calls[0][1]["history_start"] == "2021-01-04"
    assert json.loads(capsys.readouterr().out)["status"] == "CALENDAR_PLAN_ONLY_NOT_TRAINING_READY"


def test_fit_prepared_insufficient_capacity_never_calls_model_fit(tmp_path, monkeypatch, capsys):
    from infrastructure import les_v7_fr3_capacity as capacity
    from infrastructure import les_v7_fr3_training as training
    report = dict(status=capacity.INSUFFICIENT, count_capacity_allows_fit=False,
        necessary_counts_only=True, capacity_id="a" * 64)
    retained = []
    monkeypatch.setattr(capacity, "audit_fr3_prepared_capacity", lambda p: report)
    monkeypatch.setattr(capacity, "retain_fr3_capacity", lambda *a: retained.append(a) or (tmp_path / "audit", {}))
    monkeypatch.setattr(training, "train_fr3_scope", lambda *a, **k: pytest.fail("SYNTHETIC futile fit"))
    assert cli.main(["fit-prepared", "--training-root", str(tmp_path / "new"),
        "--audit-output-dir", str(tmp_path / "audit")]) == 2
    assert retained and json.loads(capsys.readouterr().out)["status"] == capacity.INSUFFICIENT


def test_fit_prepared_never_rebases_completed_old_run(tmp_path, capsys):
    old = tmp_path / "old"; old.mkdir()
    completion = old / "fr3-training-completion.json"
    completion.write_text('{"SYNTHETIC_OLD_NOT_READY":true}')
    before = completion.read_bytes()
    with pytest.raises(SystemExit) as error:
        cli.main(["fit-prepared", "--training-root", str(old), "--audit-output-dir", str(tmp_path / "audit")])
    assert error.value.code == 2 and "NO_OLD_REBASE" in capsys.readouterr().err
    assert completion.read_bytes() == before and not (tmp_path / "audit").exists()


def test_fit_prepared_sufficient_necessary_counts_still_uses_native_training_and_can_fail(tmp_path, monkeypatch, capsys):
    from infrastructure import les_v7_fr3_capacity as capacity
    from infrastructure import les_v7_fr3_training as training
    root = tmp_path / "new"; root.mkdir()
    request = dict(payload=dict(freeze_path="SYNTHETIC_NEW_SCOPE", build_config={"SYNTHETIC": "build"},
        training_config={"SYNTHETIC": "training"}))
    (root / "fr3-training-request.json").write_text(json.dumps(request))
    monkeypatch.setattr(capacity, "audit_fr3_prepared_capacity", lambda p:
        dict(status=capacity.SUFFICIENT, count_capacity_allows_fit=True))
    monkeypatch.setattr(capacity, "retain_fr3_capacity", lambda *a: (tmp_path / "audit", {}))
    calls = []
    monkeypatch.setattr(training, "train_fr3_scope", lambda *a, **k: calls.append((a, k)) or dict(status="NOT_READY"))
    assert cli.main(["fit-prepared", "--training-root", str(root),
        "--audit-output-dir", str(tmp_path / "audit")]) == 2
    assert len(calls) == 1 and calls[0][0][0] == "SYNTHETIC_NEW_SCOPE"
    assert json.loads(capsys.readouterr().out)["status"] == "NOT_READY"
