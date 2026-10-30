"""SYNTHETIC_TEST_ONLY offline artifacts, isolated under pytest tmp_path."""
from copy import deepcopy
import fcntl
import io
import json
import zipfile

import pytest

from application.les_v7_training import train_request
from domain.les_v7.contracts import identity
from infrastructure.storage.les_v7_training import (
    LesV7TrainingRepository, read_json_bounded,
)
from tests.strategies.test_low_entry_score_v7 import bundle

FIT = "2026-01-01T00:00:00+00:00"
FIRST = "2026-10-12T10:00:00+00:00"


class Guard:
    def __init__(self):
        self.checks = 0
        self.drift_after = None

    def retain(self, objects):
        return dict(source=objects.put(b"SYNTHETIC_TEST_ONLY source"),
                    environment=objects.put_json(dict(environment="SYNTHETIC_TEST_ONLY")),
                    parameters=objects.put_json(dict(parameters="SYNTHETIC_TEST_ONLY")))

    def check(self):
        self.checks += 1
        if self.drift_after is not None and self.checks >= self.drift_after:
            raise ValueError("REPLAY_VERSION_CHANGED_DURING_JOB")


def inputs(kind="SYNTHETIC_TEST_ONLY"):
    payload = dict(schema="les-v7-training-dataset/v1", dataset_kind=kind,
                   rows=[], universe=["SYNTHETIC_ONLY"], sources=[])
    return dict(payload=payload, sha256=identity(payload)), dict(fit_cutoff=FIT)


def fitted(dataset, config, *, ready=True):
    reports = {identity(dict(kind=name, status="SYNTHETIC_TEST_ONLY")):
               dict(kind=name, status="SYNTHETIC_TEST_ONLY")
               for name in ("calibration", "fractal_null", "holdout", "ablation", "projection")}
    ids = {report["kind"]: key for key, report in reports.items()}
    candidate = bundle()["payload"] if ready else None
    if candidate is not None:
        candidate.update(trained_until=None, training_pipeline_schema="LES_V7_OFFLINE_TRAINING_R1",
            research_only=True, approved_for_live=False, validation_reports=deepcopy(reports),
            validation_report_ids=ids, baseline_validation_report_id=ids["fractal_null"],
            fractal_null_validation_report_id=ids["fractal_null"],
            validation_available_at="2026-10-01T10:00:00+00:00")
        candidate["training_manifest"].update(dataset_id=dataset["sha256"],
                                               config_id=identity(config),
                                               dataset_kind=dataset["payload"]["dataset_kind"])
        candidate["calibration"]["validation_report_id"] = ids["calibration"]
    return dict(schema="les-v7-training-result/v1", status="READY_TO_FREEZE" if ready else "NOT_READY",
        dataset_id=dataset["sha256"], config_id=identity(config), candidate=candidate,
        reports=reports, oof_predictions=[], diagnostics={}, reason_codes=[])


def run(root, *, dataset=None, config=None, trainer=fitted, **options):
    initial, settings = inputs()
    return train_request(dataset or initial, config or settings,
        LesV7TrainingRepository(root, guard_factory=Guard, clock=lambda: FIRST, **options),
        allow_synthetic=True, trainer=trainer)


def test_publication_timestamp_is_actual_not_historical_and_retry_skips_fitting(tmp_path):
    calls = []
    first = run(tmp_path, trainer=lambda d, c: calls.append(True) or fitted(d, c))
    assert first["first_published_at"] == FIRST
    assert first["first_published_at"] != FIT
    assert first["execution_action"] == "NO_ACTION" and not first["approved_for_live"]
    second = run(tmp_path, trainer=lambda *_: pytest.fail("completed retry fitted again"))
    assert second == first and len(calls) == 1
    candidate = json.loads((tmp_path / first["job_id"] / "frozen_bundle.json").read_text())
    assert candidate["payload"]["trained_until"] == FIRST
    assert candidate["sha256"] == first["bundle_sha256"]


def test_synthetic_requires_explicit_opt_in_and_hash_verification_before_fitting(tmp_path):
    dataset, config = inputs()
    repo = LesV7TrainingRepository(tmp_path, guard_factory=Guard)
    with pytest.raises(ValueError, match="EXPLICIT_OPT_IN"):
        train_request(dataset, config, repo, trainer=lambda *_: pytest.fail("fitted"))
    dataset["payload"]["rows"].append("tamper")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        train_request(dataset, config, repo, allow_synthetic=True, trainer=fitted)
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_not_ready_retains_diagnostics_but_never_publishes_bundle(tmp_path):
    receipt = run(tmp_path, trainer=lambda d, c: fitted(d, c, ready=False))
    assert receipt["status"] == "NOT_READY" and receipt["bundle_sha256"] is None
    assert not (tmp_path / receipt["job_id"] / "frozen_bundle.json").exists()
    assert (tmp_path / receipt["job_id"] / "training_result.json").exists()
    assert run(tmp_path, trainer=lambda *_: pytest.fail("not-ready retry fitted")) == receipt


def test_raw_oof_is_saved_once_and_fold_reports_need_not_be_embedded(tmp_path):
    def with_fold(d, c):
        value = fitted(d, c)
        fold_report = dict(kind="PAST_FOLD", source="SYNTHETIC_TEST_ONLY")
        value["reports"][identity(fold_report)] = fold_report
        value["oof_predictions"] = [dict(prediction_id="SYNTHETIC_TEST_ONLY_UNIQUE_OOF", value=.4)]
        return value
    receipt = run(tmp_path, trainer=with_fold)
    folder = tmp_path / receipt["job_id"]
    compact = json.loads((folder / "fit_result.json").read_text())["payload"]
    index = json.loads((folder / "training_result.json").read_text())
    assert "oof_predictions" not in compact and "oof_predictions" not in index
    assert "reports" not in compact and len(compact["report_references"]) == 6
    assert len(compact["candidate"]["validation_reports"]) == 5
    assert sum("SYNTHETIC_TEST_ONLY_UNIQUE_OOF" in path.read_text()
               for path in folder.rglob("*.json")) == 1


def test_validation_not_yet_available_cannot_publish_a_ready_bundle(tmp_path):
    def future_validation(d, c):
        value = fitted(d, c)
        value["candidate"]["validation_available_at"] = "2027-01-01T00:00:00+00:00"
        return value
    with pytest.raises(ValueError, match="PUBLICATION_TIME_INVALID"):
        run(tmp_path, trainer=future_validation)
    assert not list(tmp_path.rglob("frozen_bundle.json"))
    assert not list(tmp_path.rglob("completion.json"))


@pytest.mark.parametrize("change", ["availability", "manifest_config"])
def test_unpublished_candidate_and_manifest_config_binding_required(tmp_path, change):
    def invalid(d, c):
        value = fitted(d, c)
        if change == "availability":
            value["candidate"]["trained_until"] = FIT
        else:
            value["candidate"]["training_manifest"]["config_id"] = "0" * 64
        return value
    with pytest.raises(ValueError, match="TRAINING_CANDIDATE_BINDING_MISMATCH"):
        run(tmp_path, trainer=invalid)
    assert not list(tmp_path.rglob("frozen_bundle.json"))
    assert not list(tmp_path.rglob("completion.json"))


def test_interruption_before_receipt_recovers_first_publication_without_refitting(tmp_path):
    def interrupt(*_):
        raise RuntimeError("SYNTHETIC_TEST_INTERRUPTION")
    with pytest.raises(RuntimeError, match="INTERRUPTION"):
        run(tmp_path, after_publish=interrupt)
    folder = next(p for p in tmp_path.iterdir() if p.name != "objects")
    assert (folder / "frozen_bundle.json").exists()
    assert not (folder / "completion.json").exists()
    before = (folder / "frozen_bundle.json").read_bytes()
    dataset, config = inputs()
    receipt = train_request(dataset, config, LesV7TrainingRepository(tmp_path,
        guard_factory=Guard, clock=lambda: "2026-10-13T10:00:00+00:00"),
        allow_synthetic=True, trainer=lambda *_: pytest.fail("recovery fitted again"))
    assert receipt["first_published_at"] == FIRST
    assert (folder / "frozen_bundle.json").read_bytes() == before


@pytest.mark.parametrize("name", ["training_result.json", "oof_predictions.json",
                                  "frozen_bundle.json", "config.json", "dataset.json"])
def test_completed_artifact_corruption_is_detected_not_rebuilt(tmp_path, name):
    receipt = run(tmp_path)
    path = tmp_path / receipt["job_id"] / name
    # Intentional synthetic corruption; no production artifacts involved.
    path.write_text('{"corrupted":"SYNTHETIC_TEST_ONLY"}')
    with pytest.raises((ValueError, KeyError), match="HASH_MISMATCH|CONTENT_MISMATCH"):
        run(tmp_path, trainer=lambda *_: pytest.fail("corruption silently recomputed"))


def test_pinned_source_object_corruption_is_detected(tmp_path):
    receipt = run(tmp_path)
    path = tmp_path / "objects" / receipt["pinned_objects"]["source"]
    path.write_bytes(b"SYNTHETIC_TEST_ONLY changed")
    with pytest.raises(ValueError, match="IMMUTABLE|HASH_MISMATCH"):
        run(tmp_path)


def test_source_index_cannot_point_to_another_valid_zip(tmp_path, monkeypatch):
    import infrastructure.research_objects as module
    source = tmp_path / "main.py"
    source.write_text("# SYNTHETIC_TEST_ONLY source A\n")
    monkeypatch.setattr(module.runtime, "ROOT", tmp_path)
    monkeypatch.setattr(module, "source_paths", lambda: [source])
    # Isolate only retain's source-content binding, not the environment guard.
    guard = module.ResearchVersionGuard.__new__(module.ResearchVersionGuard)
    guard.parameters, guard.environment = {}, {"kind": "SYNTHETIC_TEST_ONLY"}
    guard.check = lambda: None
    objects = module.ResearchObjects(tmp_path / "research" / "objects")
    first = guard.retain(objects)
    assert guard.retain(objects) == first
    other = io.BytesIO()
    with zipfile.ZipFile(other, "w") as archive:
        archive.writestr("main.py", "# SYNTHETIC_TEST_ONLY source B\n")
    other_ref = objects.put(other.getvalue())  # Self-consistent but wrong source.
    index = next((tmp_path / "research" / "source-index").glob("*.json"))
    index.write_text(json.dumps({"source": other_ref}))
    with pytest.raises(ValueError, match="RESEARCH_SOURCE_INDEX_BINDING_MISMATCH"):
        guard.retain(objects)


def test_report_link_mismatch_rejects_completion_and_bundle(tmp_path):
    def invalid(d, c):
        result = fitted(d, c)
        result["candidate"]["validation_report_ids"]["projection"] = "0" * 64
        return result
    with pytest.raises(ValueError, match="REPORT_LINK_MISMATCH"):
        run(tmp_path, trainer=invalid)
    assert not list(tmp_path.rglob("frozen_bundle.json"))
    assert not list(tmp_path.rglob("completion.json"))


def test_dataset_config_and_dirty_source_identities_isolate_jobs(tmp_path):
    first = run(tmp_path)
    dataset, config = inputs()
    config["policy"] = "SYNTHETIC_CHANGED_POLICY"
    second = run(tmp_path, dataset=dataset, config=config)
    assert first["job_id"] != second["job_id"]
    class ChangedGuard(Guard):
        def retain(self, objects):
            result = super().retain(objects)
            result["source"] = objects.put(b"SYNTHETIC_TEST_ONLY dirty source")
            return result
    third = train_request(dataset, config,
        LesV7TrainingRepository(tmp_path, guard_factory=ChangedGuard, clock=lambda: FIRST),
        allow_synthetic=True, trainer=fitted)
    assert third["job_id"] != second["job_id"]


def test_version_drift_after_fit_refuses_publication(tmp_path):
    guard = Guard()
    dataset, config = inputs()
    def drift(d, c):
        guard.drift_after = guard.checks + 1
        return fitted(d, c)
    with pytest.raises(ValueError, match="VERSION_CHANGED"):
        train_request(dataset, config, LesV7TrainingRepository(tmp_path,
            guard_factory=lambda: guard, clock=lambda: FIRST), allow_synthetic=True, trainer=drift)
    assert not list(tmp_path.rglob("completion.json"))
    assert not list(tmp_path.rglob("frozen_bundle.json"))


def test_input_mutation_during_fit_cannot_be_archived_as_original_request(tmp_path):
    def mutating(d, c):
        value = fitted(d, c)
        c["undeclared_change"] = "SYNTHETIC_TEST_ONLY"
        return value
    with pytest.raises(ValueError, match="INPUT_CHANGED_DURING_FIT"):
        run(tmp_path, trainer=mutating)
    assert not list(tmp_path.rglob("fit_result.json"))
    assert not list(tmp_path.rglob("completion.json"))


def test_busy_job_lock_never_changes_existing_metadata(tmp_path):
    receipt = run(tmp_path)
    path = tmp_path / receipt["job_id"] / ".training.lock"
    path.write_bytes(b"SYNTHETIC_TEST_ONLY lock owner")
    with path.open("rb") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="TRAINING_JOB_BUSY"):
            run(tmp_path)
    assert path.read_bytes() == b"SYNTHETIC_TEST_ONLY lock owner"


def test_bounded_reader_and_timezone_contract(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"{" + b" " * 128)
    with pytest.raises(ValueError, match="SIZE_LIMIT"):
        read_json_bounded(path, 32)
    dataset, config = inputs()
    config["fit_cutoff"] = "2026-01-01"
    with pytest.raises(ValueError, match="TIMEZONE_REQUIRED"):
        run(tmp_path / "jobs", dataset=dataset, config=config)


def test_cli_prints_summary_only_and_has_no_live_side_effect(tmp_path, monkeypatch, capsys):
    from interface.les_v7_training import main
    dataset, config = inputs()
    dataset_path, config_path = tmp_path / "dataset.json", tmp_path / "config.json"
    dataset_path.write_text(json.dumps(dataset))
    config_path.write_text(json.dumps(config))
    received = []
    def request(data, settings, repository, **options):
        received.append((data, settings, options))
        return dict(schema="les-v7-training-completion/v1", job_id="synthetic", status="NOT_READY",
            dataset_kind="SYNTHETIC_TEST_ONLY", bundle_sha256=None, first_published_at=None,
            output=str(tmp_path / "jobs"), research_only=True, approved_for_live=False,
            execution_action="NO_ACTION", bulky_internal_result="DO_NOT_PRINT")
    monkeypatch.setattr("interface.les_v7_training.train_request", request)
    assert main([str(dataset_path), "--config", str(config_path), "--output-dir",
                 str(tmp_path / "jobs"), "--allow-synthetic"]) == 2
    summary = json.loads(capsys.readouterr().out)
    assert "bulky_internal_result" not in summary and summary["execution_action"] == "NO_ACTION"
    assert received[0][2] == {"allow_synthetic": True}


@pytest.mark.parametrize("key,value", [("dataset_id", "0" * 64), ("config_id", "1" * 64),
                                      ("dataset_kind", "RESEARCH")])
def test_completion_explicit_input_binding_is_verified(tmp_path, key, value):
    receipt = run(tmp_path)
    path = tmp_path / receipt["job_id"] / "completion.json"
    damaged = json.loads(path.read_text())
    damaged[key] = value
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="COMPLETION_BINDING_MISMATCH"):
        LesV7TrainingRepository.verify(path.parent)


def test_completion_cannot_precede_first_actual_publication(tmp_path):
    receipt = run(tmp_path)
    path = tmp_path / receipt["job_id"] / "completion.json"
    damaged = json.loads(path.read_text())
    damaged["completed_at"] = "2026-10-11T00:00:00+00:00"
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="PUBLICATION_TIME_OR_MODEL_MISMATCH"):
        LesV7TrainingRepository.verify(path.parent)


def test_actual_offline_trainer_publication_verification_and_retry(tmp_path):
    """Native synthetic fitting proves integration, not market forecasting skill."""
    from tests.strategies.test_v7_offline_training import synthetic_dataset, synthetic_config
    from domain.les_v7.localization import FrozenBundle
    from domain.les_v7.contracts import timestamp
    config = synthetic_config()
    dataset = synthetic_dataset(config)
    repository = LesV7TrainingRepository(tmp_path, guard_factory=Guard, clock=lambda: FIRST)
    # No injected trainer: execute the actual feature/label/fold/calibration/
    # constrained-weight/holdout pipeline through the public application port.
    receipt = train_request(dataset, config, repository, allow_synthetic=True)
    assert receipt["status"] == "READY_TO_FREEZE"
    assert receipt["dataset_id"] == dataset["sha256"]
    assert receipt["config_id"] == identity(config)
    assert receipt["dataset_kind"] == "SYNTHETIC_TEST_ONLY"
    assert receipt["first_published_at"] == FIRST == receipt["completed_at"]
    assert receipt["execution_action"] == "NO_ACTION" and not receipt["approved_for_live"]
    folder = tmp_path / receipt["job_id"]
    published = read_json_bounded(folder / "frozen_bundle.json", 1024 * 1024)
    payload = FrozenBundle.load(published).record()
    assert payload["trained_until"] == FIRST
    assert timestamp(payload["fit_cutoff"]) < timestamp(payload["validation_available_at"]) <= timestamp(FIRST)
    assert payload["training_manifest"]["dataset_kind"] == "SYNTHETIC_TEST_ONLY"
    assert LesV7TrainingRepository.verify(folder) == receipt
    assert train_request(dataset, config, repository, allow_synthetic=True,
                         trainer=lambda *_: pytest.fail("completed native fit repeated")) == receipt
