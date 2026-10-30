"""Synthetic write-once builder publication and CLI contracts; no live work."""
from copy import deepcopy
from collections import Counter
import fcntl
import json
import os
from pathlib import Path

import pytest

from application.les_v7_dataset import build_training_dataset
from infrastructure.storage.les_v7_dataset import LesV7DatasetRepository
from tests.test_v7_dataset_builder import builder_inputs
from tests.test_v7_training_publication import Guard

NOW = "2026-10-13T12:00:00+08:00"


def repository(root, **options):
    return LesV7DatasetRepository(root, guard_factory=Guard, clock=lambda: NOW, **options)


def run(root, *, source=None, config=None, builder=build_training_dataset, **options):
    initial, initial_config = builder_inputs()
    return repository(root, **options).run(source or initial, config or initial_config,
                                           builder=builder, allow_synthetic=True)


def test_actual_native_dataset_publication_verified_retry_is_immutable_and_skips_building(tmp_path):
    calls = []
    receipt = run(tmp_path, builder=lambda s, c: calls.append(True) or build_training_dataset(s, c))
    folder = Path(receipt["output"])
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    retried = run(tmp_path, builder=lambda *_: pytest.fail("completed job rebuilt"))
    assert receipt == retried == LesV7DatasetRepository.verify(folder)
    assert len(calls) == 1
    assert receipt["status"] == "EVIDENCE_READY" and receipt["counts"]["eligible_events"] == 1
    assert receipt["completed_at"] == NOW and receipt["dataset_kind"] == "SYNTHETIC_TEST_ONLY"
    assert receipt["research_only"] and not receipt["approved_for_live"]
    assert receipt["execution_action"] == "NO_ACTION"
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()} == before


def test_saved_build_before_completion_recovers_without_rebuilding_or_overwriting(tmp_path):
    def interrupt(*_):
        raise RuntimeError("SYNTHETIC_INTERRUPTION_BEFORE_COMPLETION")
    with pytest.raises(RuntimeError, match="SYNTHETIC_INTERRUPTION"):
        run(tmp_path, after_build=interrupt)
    folder = next(p for p in tmp_path.iterdir() if p.name != "objects")
    assert (folder / "dataset.json").exists() and (folder / "build_result.json").exists()
    assert not (folder / "completion.json").exists()
    data = (folder / "dataset.json").read_bytes()
    receipt = run(tmp_path, builder=lambda *_: pytest.fail("saved result rebuilt"))
    assert receipt["status"] == "EVIDENCE_READY"
    assert (folder / "dataset.json").read_bytes() == data
    assert (folder / "completion.json").exists()


@pytest.mark.parametrize("which", ["source", "config"])
def test_builder_cannot_mutate_bound_source_or_configuration_during_work(tmp_path, which):
    def changing(source, config):
        built = build_training_dataset(source, config)
        if which == "source":
            source["payload"]["diagnostics"].append("SYNTHETIC_MUTATION")
        else:
            config["max_events"] = 2
        return built
    with pytest.raises(ValueError, match="DATASET_BUILD_INPUT_CHANGED"):
        run(tmp_path, builder=changing)
    assert not list(tmp_path.rglob("completion.json"))
    assert not list(tmp_path.rglob("dataset.json"))


@pytest.mark.parametrize("artifact", ["dataset.json", "config.json", "request.json"])
def test_corrupt_saved_artifact_is_rejected_and_never_recomputed_or_overwritten(tmp_path, artifact):
    receipt = run(tmp_path)
    path = Path(receipt["output"]) / artifact
    value = json.loads(path.read_text())
    value["SYNTHETIC_TAMPER"] = True
    path.write_text(json.dumps(value))
    damaged = path.read_bytes()
    with pytest.raises(ValueError):
        run(tmp_path, builder=lambda *_: pytest.fail("corrupt completed job rebuilt"))
    assert path.read_bytes() == damaged


def test_completion_cannot_add_unbound_artifact_reference(tmp_path):
    receipt = run(tmp_path)
    folder = Path(receipt["output"])
    path = folder / "completion.json"
    value = json.loads(path.read_text())
    value["artifacts"]["../SYNTHETIC_UNBOUND_PATH"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="ARTIFACT_REFERENCES_REQUIRED"):
        LesV7DatasetRepository.verify(folder)


def test_referenced_source_object_corruption_is_detected_not_regenerated(tmp_path):
    receipt = run(tmp_path)
    source_object = tmp_path / "objects" / receipt["pinned_objects"]["source"]
    source_object.write_bytes(b"SYNTHETIC_CORRUPT_SOURCE_OBJECT")
    with pytest.raises(ValueError, match="OBJECT_HASH_MISMATCH"):
        LesV7DatasetRepository.verify(receipt["output"])
    assert source_object.read_bytes() == b"SYNTHETIC_CORRUPT_SOURCE_OBJECT"


def test_busy_job_lock_is_nonblocking_and_preserves_metadata(tmp_path):
    receipt = run(tmp_path)
    lock_path = Path(receipt["output"]) / ".builder.lock"
    metadata = b"SYNTHETIC_TEST_ONLY lock metadata must remain intact"
    lock_path.write_bytes(metadata)
    with lock_path.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="DATASET_BUILD_JOB_BUSY"):
            run(tmp_path, builder=lambda *_: pytest.fail("busy job rebuilt"))
        fcntl.flock(stream, fcntl.LOCK_UN)
    assert lock_path.read_bytes() == metadata


def test_synthetic_source_requires_opt_in_before_source_objects_or_results_are_written(tmp_path):
    source, config = builder_inputs()
    with pytest.raises(ValueError, match="SYNTHETIC_DATASET_BUILD_REQUIRES_EXPLICIT_OPT_IN"):
        repository(tmp_path).run(source, config, builder=lambda *_: pytest.fail("fitted"))
    assert not list(tmp_path.iterdir())


def test_cli_partial_exit_two_retains_diagnostics_without_training_or_live_work(tmp_path, monkeypatch, capsys):
    import interface.les_v7_dataset as cli
    from infrastructure.les_v7_training_source import OBSERVED_ONLY

    source, config = builder_inputs()
    config["coordinate_coverages"] = []
    config_path = tmp_path / "SYNTHETIC-config.json"
    config_path.write_text(json.dumps(config))
    source_path = tmp_path / "SYNTHETIC-immutable-source.json"
    recorded = []

    def load(path, **options):
        recorded.append((path, deepcopy(options)))
        return deepcopy(source)

    monkeypatch.setattr(cli, "load_training_source", load)
    monkeypatch.setattr(cli, "LesV7DatasetRepository", repository)
    status = cli.main([str(source_path), "--config", str(config_path),
                       "--output-dir", str(tmp_path / "archive"), "--allow-synthetic"])
    assert status == 2
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PARTIAL" and receipt["execution_action"] == "NO_ACTION"
    assert not receipt["approved_for_live"] and receipt["counts"]["eligible_events"] == 0
    assert recorded == [(source_path, dict(evaluation_cutoff=config["evaluation_cutoff"],
                                          symbols=None, availability_policy=OBSERVED_ONLY))]
    folder = Path(receipt["output"])
    assert (folder / "build_result.json").exists() and (folder / "completion.json").exists()
    assert not list(folder.rglob("frozen_bundle.json"))


def test_cli_synthetic_opt_in_refusal_never_publishes_a_dataset(tmp_path, monkeypatch, capsys):
    import interface.les_v7_dataset as cli

    source, config = builder_inputs()
    config_path = tmp_path / "SYNTHETIC-config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(cli, "load_training_source", lambda *_, **__: source)
    monkeypatch.setattr(cli, "LesV7DatasetRepository", repository)
    with pytest.raises(SystemExit) as error:
        cli.main([str(tmp_path / "SYNTHETIC-source.json"), "--config", str(config_path),
                  "--output-dir", str(tmp_path / "archive")])
    assert error.value.code == 2
    assert "SYNTHETIC_DATASET_BUILD_REQUIRES_EXPLICIT_OPT_IN" in capsys.readouterr().err
    assert not (tmp_path / "archive").exists()


def test_cli_backfill_requires_all_explicit_flags_before_reading_inputs(tmp_path, capsys):
    import interface.les_v7_dataset as cli

    with pytest.raises(SystemExit) as error:
        cli.main([str(tmp_path / "SYNTHETIC-source.json"), "--config", str(tmp_path / "missing.json"),
                  "--output-dir", str(tmp_path / "archive"), "--history-start", "2024-03-08"])
    assert error.value.code == 2
    assert "must be supplied together" in capsys.readouterr().err
    assert not (tmp_path / "archive").exists()


def test_cli_invalid_build_config_never_starts_backfill(tmp_path, monkeypatch):
    import interface.les_v7_dataset as cli
    import infrastructure.les_v7_training_backfill as backfill

    config_path = tmp_path / "SYNTHETIC-invalid.json"
    config_path.write_text(json.dumps({"schema": "INVALID"}))
    monkeypatch.setattr(backfill, "backfill_training_evidence", lambda *_, **__: pytest.fail("invalid config published evidence"))
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "source.json"), "--config", str(config_path), "--output-dir", str(tmp_path / "builds"),
                  "--backfill-market-cache", str(tmp_path / "cache.sqlite3"), "--history-start", "2024-03-08",
                  "--source-store", str(tmp_path / "source-store")])
    assert not (tmp_path / "source-store").exists()


def test_cli_future_evaluation_cutoff_never_starts_backfill(tmp_path, monkeypatch, capsys):
    import interface.les_v7_dataset as cli
    import infrastructure.les_v7_training_backfill as backfill

    _, config = builder_inputs()
    config["evaluation_cutoff"] = "2027-02-10T13:30:00+08:00"
    config_path = tmp_path / "SYNTHETIC-future.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    monkeypatch.setattr(backfill, "backfill_training_evidence", lambda *_, **__: pytest.fail("future cutoff published evidence"))
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "source.json"), "--config", str(config_path), "--output-dir", str(tmp_path / "builds"),
                  "--backfill-market-cache", str(tmp_path / "cache.sqlite3"), "--history-start", "2024-03-08",
                  "--source-store", str(tmp_path / "source-store")])
    assert "DATASET_EVALUATION_AFTER_BUILD_CLOCK" in capsys.readouterr().err
    assert not (tmp_path / "source-store").exists()


def test_cli_backfill_uses_new_source_and_preserves_explicit_policy_flags(tmp_path, monkeypatch, capsys):
    from datetime import date
    import interface.les_v7_dataset as cli
    import infrastructure.les_v7_training_backfill as backfill
    from infrastructure.les_v7_training_source import FINAL_CLOSE

    source, config = builder_inputs()
    config["coordinate_coverages"] = []
    config_path = tmp_path / "SYNTHETIC-config.json"
    config_path.write_text(json.dumps(config))
    previous, repaired = tmp_path / "old.json", tmp_path / "new.json"
    cache, store, recorded = tmp_path / "cache.sqlite3", tmp_path / "evidence", []
    repair = dict(evidence=str(repaired), evidence_id="SYNTHETIC-NEW", parent_evidence_id="SYNTHETIC-OLD", repair_report="SYNTHETIC")

    def repair_source(path, **options):
        recorded.append((path, options))
        return repair

    def load(path, **options):
        assert path == repaired
        assert options["availability_policy"] == FINAL_CLOSE
        return source

    monkeypatch.setattr(backfill, "backfill_training_evidence", repair_source)
    monkeypatch.setattr(cli, "load_training_source", load)
    monkeypatch.setattr(cli, "LesV7DatasetRepository", repository)
    assert cli.main([str(previous), "--config", str(config_path), "--output-dir", str(tmp_path / "builds"),
        "--backfill-market-cache", str(cache), "--history-start", "2024-03-08", "--source-store", str(store),
        "--allow-synthetic", "--allow-final-daily-reconstruction"]) == 2
    assert recorded == [(previous, dict(market_cache=cache, history_start=date(2024, 3, 8), store=store, allow_synthetic=True))]
    assert json.loads(capsys.readouterr().out)["market_backfill"] == repair


def test_source_version_drift_during_build_cannot_publish_complete_receipt(tmp_path):
    guard = Guard()
    guard.drift_after = 2
    repo = LesV7DatasetRepository(tmp_path, guard_factory=lambda: guard, clock=lambda: NOW)
    source, config = builder_inputs()
    with pytest.raises(ValueError, match="VERSION_CHANGED_DURING_JOB"):
        repo.run(source, config, allow_synthetic=True)
    assert not list(tmp_path.rglob("completion.json"))


def test_wrong_native_result_binding_cannot_be_frozen(tmp_path):
    def invalid(source, config):
        built = build_training_dataset(source, config)
        built["result"]["dataset_id"] = "0" * 64
        return built
    with pytest.raises(ValueError, match="DATASET_BUILD_RESULT_BINDING_MISMATCH"):
        run(tmp_path, builder=invalid)
    assert not list(tmp_path.rglob("completion.json"))
    assert not list(tmp_path.rglob("dataset.json"))


def test_one_verification_reads_each_artifact_once_and_next_call_is_fresh(tmp_path, monkeypatch):
    import infrastructure.storage.les_v7_dataset as storage

    expected = run(tmp_path)
    folder = Path(expected['output'])
    read = storage.read_json_bounded
    calls = Counter()

    def counted(path, limit):
        path = Path(path)
        if path.parent == folder:
            calls[path.name] += 1
        return read(path, limit)

    monkeypatch.setattr(storage, 'read_json_bounded', counted)
    receipt, artifacts = LesV7DatasetRepository._verify_with_artifacts(folder)
    names = {'completion.json', 'request.json', 'source.json', 'config.json', 'dataset.json', 'build_result.json'}
    assert receipt == expected and calls == {name: 1 for name in names}
    assert set(artifacts) == names
    assert artifacts['completion.json']['dataset_id'] == artifacts['dataset.json']['sha256']
    artifacts['dataset.json']['payload']['SYNTHETIC_CALLER_MUTATION'] = True
    assert LesV7DatasetRepository.verify(folder) == expected
    assert calls == {name: 2 for name in names}


@pytest.mark.parametrize('name', ['completion.json', 'request.json', 'source.json', 'config.json',
                                 'dataset.json', 'build_result.json'])
def test_same_bytes_replacement_during_object_checks_rejects_detached_artifacts(tmp_path, monkeypatch, name):
    from infrastructure.research_objects import ResearchObjects

    receipt = run(tmp_path)
    folder = Path(receipt['output'])
    path = folder / name
    native_get, replaced = ResearchObjects.get, []

    def replace_original(objects, reference):
        value = native_get(objects, reference)
        if not replaced:
            replacement = path.with_suffix('.replacement')
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
            replaced.append(True)
        return value

    monkeypatch.setattr(ResearchObjects, 'get', replace_original)
    with pytest.raises(ValueError, match='DATASET_BUILD_ARCHIVE_CHANGED:' + name):
        LesV7DatasetRepository._verify_with_artifacts(folder)


def test_original_change_while_reading_is_rejected_before_payload_reuse(tmp_path, monkeypatch):
    import infrastructure.storage.les_v7_dataset as storage

    receipt = run(tmp_path)
    folder = Path(receipt['output'])
    native_read = storage.read_json_bounded

    def replace_after_read(path, limit):
        value = native_read(path, limit)
        path = Path(path)
        if path == folder / 'dataset.json':
            replacement = path.with_suffix('.replacement')
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
        return value

    monkeypatch.setattr(storage, 'read_json_bounded', replace_after_read)
    with pytest.raises(ValueError, match='DATASET_BUILD_ARCHIVE_CHANGED:dataset.json'):
        LesV7DatasetRepository.verify(folder)
