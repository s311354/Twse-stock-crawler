"""Immutable, restartable LES v7 training publication; research artifacts only."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path

from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.localization import FrozenBundle
from domain.les_v7.provenance import RUNTIME_SCHEMA, verify_archived_compaction
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard

PIPELINE = "LES_V7_OFFLINE_TRAINING_R1"
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024


def read_json_bounded(path: str | Path, limit: int) -> dict:
    """Bound reads even if the file grows between stat and read."""
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError("TRAINING_INPUT_SIZE_LIMIT:" + path.name)
    with path.open("rb") as stream:
        encoded = stream.read(limit + 1)
    if len(encoded) > limit:
        raise ValueError("TRAINING_INPUT_SIZE_LIMIT:" + path.name)
    result = json.loads(encoded)
    if not isinstance(result, dict):
        raise ValueError("TRAINING_JSON_OBJECT_REQUIRED:" + path.name)
    identity(result)
    return result


@contextmanager
def _job_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    # Never truncate or change lock metadata; acquiring it is non-blocking.
    with (folder / ".training.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("TRAINING_JOB_BUSY") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _record(payload):
    return dict(payload=payload, sha256=identity(payload))


def _load_record(path):
    result = read_json_bounded(path, MAX_DATASET_BYTES)
    if not isinstance(result.get("payload"), dict) or result.get("sha256") != identity(result["payload"]):
        raise ValueError("TRAINING_ARTIFACT_HASH_MISMATCH:" + path.name)
    return result["payload"]


def _validate_result(result, request, *, config=None, dataset_record=None):
    if (result.get("schema") != "les-v7-training-result/v1"
            or result.get("status") not in {"READY_TO_FREEZE", "NOT_READY"}
            or result.get("dataset_id") != request["dataset_id"]
            or result.get("config_id") != request["config_id"]):
        raise ValueError("TRAINING_RESULT_INPUT_BINDING_MISMATCH")
    reports = result.get("reports")
    if not isinstance(reports, dict) or any(identity(body) != key for key, body in reports.items()):
        raise ValueError("TRAINING_REPORT_HASH_MISMATCH")
    if not isinstance(result.get("oof_predictions"), list):
        raise ValueError("TRAINING_OOF_ARRAY_REQUIRED")
    candidate = result.get("candidate")
    if result["status"] == "NOT_READY":
        if candidate is not None:
            raise ValueError("NOT_READY_TRAINING_CANNOT_PUBLISH_MODEL")
        return
    if not isinstance(candidate, dict):
        raise ValueError("TRAINING_CANDIDATE_REQUIRED")
    if (candidate.get("training_pipeline_schema") != PIPELINE
            or candidate.get("research_only") is not True
            or candidate.get("approved_for_live") is not False
            or candidate.get("trained_until") is not None
            or candidate.get("training_manifest", {}).get("dataset_id") != request["dataset_id"]
            or candidate.get("training_manifest", {}).get("dataset_kind") != request["dataset_kind"]
            or candidate.get("training_manifest", {}).get("config_id") != request["config_id"]
            or timestamp(candidate["fit_cutoff"]) != timestamp(request["fit_cutoff"])):
        raise ValueError("TRAINING_CANDIDATE_BINDING_MISMATCH")
    runtime_schema = candidate.get("runtime_provenance_schema")
    if runtime_schema not in {None, RUNTIME_SCHEMA}:
        raise ValueError("UNKNOWN_RUNTIME_PROVENANCE_SCHEMA")
    if runtime_schema == RUNTIME_SCHEMA:
        if (config is None or dataset_record is None
                or identity(config) != request["config_id"]
                or identity(dataset_record.get("payload")) != request["dataset_id"]):
            raise ValueError("R2_NATIVE_CONFIG_AND_DATASET_REQUIRED")
        verify_archived_compaction(candidate, result, config, dataset_record)
    else:
        report_ids = candidate.get("validation_report_ids")
        if (not isinstance(report_ids, dict) or not report_ids
                or any(ref not in reports for ref in report_ids.values())):
            raise ValueError("TRAINING_VALIDATION_REPORT_LINK_MISMATCH")
        if candidate.get("validation_reports") != {ref: reports[ref] for ref in report_ids.values()}:
            raise ValueError("TRAINING_EMBEDDED_VALIDATION_REPORT_MISMATCH")
        timestamp(candidate["validation_available_at"])
        for ref in (candidate.get("calibration", {}).get("validation_report_id"),
                    candidate.get("fractal_null_validation_report_id")
                    or candidate.get("baseline_validation_report_id")):
            if ref not in reports:
                raise ValueError("TRAINING_VALIDATION_REPORT_LINK_MISMATCH")
    # Shape/content validation only, not a runtime artifact or historical release.
    # The actual candidate retains null availability until first publication.
    schema_probe = dict(candidate, trained_until=max(timestamp(candidate["fit_cutoff"]),
                        timestamp(candidate["validation_available_at"])).isoformat())
    FrozenBundle.load(_record(schema_probe))


def _semantic_candidate(candidate):
    value = deepcopy(candidate)
    value.pop("trained_until", None)
    return value


def _hydrate_result(compact, folder):
    """Resolve only exact local references; raw OOF/reports are stored once."""
    value = deepcopy(compact)
    oof = value.pop("oof_prediction_reference", None)
    reports = value.pop("report_references", None)
    if (not isinstance(oof, dict) or oof.get("path") != "oof_predictions.json"
            or not isinstance(reports, dict)):
        raise ValueError("TRAINING_RESULT_REFERENCES_REQUIRED")
    rows = read_json_bounded(folder / oof["path"], MAX_DATASET_BYTES)
    if identity(rows) != oof.get("sha256"):
        raise ValueError("TRAINING_OOF_REFERENCE_HASH_MISMATCH")
    value["oof_predictions"] = rows["predictions"]
    value["reports"] = {}
    for report_id, reference in reports.items():
        if (not isinstance(reference, dict) or reference.get("path") != "reports/" + report_id + ".json"
                or reference.get("sha256") != report_id):
            raise ValueError("TRAINING_REPORT_REFERENCE_INVALID")
        body = read_json_bounded(folder / reference["path"], MAX_DATASET_BYTES)
        if identity(body) != report_id:
            raise ValueError("TRAINING_REPORT_REFERENCE_HASH_MISMATCH")
        value["reports"][report_id] = body
    return value


def _result_index(compact):
    return dict(schema="les-v7-training-result-index/v1", status=compact["status"],
        dataset_id=compact["dataset_id"], config_id=compact["config_id"],
        fit_result_reference=dict(path="fit_result.json", sha256=identity(compact)),
        oof_prediction_reference=compact["oof_prediction_reference"],
        report_references=compact["report_references"])


class LesV7TrainingRepository:
    """Source/environment-pinned job with completion-last atomic publication.

    A saved fitted result is hash-bound and revalidated on resume; expensive fit
    is not repeated after an interrupted publication. A completed job verifies
    every referenced byte before returning, never silently rebuilding corruption.
    """
    def __init__(self, root: str | Path, *, guard_factory=ResearchVersionGuard,
                 clock=utc_now, after_publish=None):
        self.root = Path(root)
        self.guard_factory = guard_factory
        self.clock = clock
        self.after_publish = after_publish

    def run(self, dataset_record, config, *, trainer):
        scope_dataset = dataset_record.get('payload', {}).get('schema') == 'les-v7-sharded-training-dataset/v1'
        if scope_dataset:
            from infrastructure.les_v7_scope_training import verify_sharded_training_dataset
            verify_sharded_training_dataset(dataset_record, self.root.parent, config=config)
        guard = self.guard_factory()
        objects = ResearchObjects(self.root / "objects")
        pinned = guard.retain(objects)
        if scope_dataset and dataset_record['payload']['pinned_objects'] != pinned:
            raise ValueError('TRAINING_SHARDED_PINNED_OBJECTS_MISMATCH')
        request = dict(schema="les-v7-training-request/v1", pipeline_schema=PIPELINE,
            dataset_id=dataset_record["sha256"], config_id=identity(config),
            dataset_kind=dataset_record["payload"]["dataset_kind"],
            fit_cutoff=config["fit_cutoff"], pinned_objects=pinned,
            research_only=True, approved_for_live=False,
            source_authenticity="SOURCE_AUTHENTICITY_NOT_CERTIFIED")
        job_id = identity(request)
        folder = self.root / job_id
        with _job_lock(folder):
            guard.check()
            if (folder / "completion.json").exists():
                return self.verify(folder, request=request, objects=objects, guard=guard)
            if timestamp(self.clock()) < timestamp(config["fit_cutoff"]):
                raise ValueError("TRAINING_FIT_CUTOFF_AFTER_PUBLICATION_CLOCK")
            self._write(folder, "request.json", request)
            self._write(folder, "dataset.json", dataset_record)
            self._write(folder, "config.json", config)
            fit_path = folder / "fit_result.json"
            if fit_path.exists():
                compact = _load_record(fit_path)
                fitted = _hydrate_result(compact, folder)
            else:
                fitted = trainer(dataset_record, config)
                if (identity(dataset_record.get("payload")) != request["dataset_id"]
                        or identity(config) != request["config_id"]):
                    raise ValueError("TRAINING_INPUT_CHANGED_DURING_FIT")
                _validate_result(fitted, request, config=config, dataset_record=dataset_record)
                guard.check()
                compact = deepcopy(fitted)
                raw_oof = dict(predictions=compact.pop("oof_predictions"))
                self._write(folder, "oof_predictions.json", raw_oof)
                compact["oof_prediction_reference"] = dict(path="oof_predictions.json", sha256=identity(raw_oof))
                compact["report_references"] = {}
                for report_id, report in compact.pop("reports").items():
                    name = "reports/" + report_id + ".json"
                    self._write(folder, name, report)
                    compact["report_references"][report_id] = dict(path=name, sha256=report_id)
                guard.check()
                self._write(folder, "fit_result.json", _record(compact))
            _validate_result(fitted, request, config=config, dataset_record=dataset_record)
            artifacts = {name: identity(read_json_bounded(folder / name, MAX_DATASET_BYTES))
                         for name in ("request.json", "dataset.json", "config.json", "fit_result.json")}
            artifacts["oof_predictions.json"] = compact["oof_prediction_reference"]["sha256"]
            index = _result_index(compact)
            self._write(folder, "training_result.json", index)
            artifacts["training_result.json"] = identity(index)
            for report_id, report in fitted["reports"].items():
                name = "reports/" + report_id + ".json"
                artifacts[name] = report_id
            published_at = None
            bundle_sha = None
            if fitted["status"] == "READY_TO_FREEZE":
                candidate = deepcopy(fitted["candidate"])
                bundle_path = folder / "frozen_bundle.json"
                if bundle_path.exists():
                    old = _load_record(bundle_path)
                    if identity(_semantic_candidate(old)) != identity(_semantic_candidate(candidate)):
                        raise ValueError("TRAINING_RECOVERY_MODEL_CONFLICT")
                    if timestamp(old["trained_until"]) < timestamp(config["fit_cutoff"]):
                        raise ValueError("TRAINING_PUBLICATION_TIME_INVALID")
                    candidate = old
                else:
                    candidate["trained_until"] = self.clock()
                    if timestamp(candidate["trained_until"]) < max(
                            timestamp(config["fit_cutoff"]), timestamp(candidate["validation_available_at"])):
                        raise ValueError("TRAINING_PUBLICATION_TIME_INVALID")
                    FrozenBundle.load(_record(candidate))
                    self._write(folder, "frozen_bundle.json", _record(candidate))
                if not max(timestamp(config["fit_cutoff"]), timestamp(candidate["validation_available_at"])) <= timestamp(
                        candidate["trained_until"]) <= timestamp(self.clock()):
                    raise ValueError("TRAINING_PUBLICATION_TIME_INVALID")
                published_at = candidate["trained_until"]
                bundle_sha = identity(candidate)
                artifacts["frozen_bundle.json"] = identity(_record(candidate))
            elif (folder / "frozen_bundle.json").exists():
                raise ValueError("NOT_READY_TRAINING_HAS_UNEXPECTED_MODEL")
            guard.check()
            receipt = dict(schema="les-v7-training-completion/v1", job_id=job_id,
                request_sha256=identity(request), dataset_id=request["dataset_id"],
                config_id=request["config_id"], dataset_kind=request["dataset_kind"],
                status=fitted["status"], artifacts=artifacts, pinned_objects=pinned,
                first_published_at=published_at, completed_at=self.clock(), bundle_sha256=bundle_sha,
                research_only=True, approved_for_live=False, execution_action="NO_ACTION",
                source_authenticity="SOURCE_AUTHENTICITY_NOT_CERTIFIED")
            if self.after_publish:
                self.after_publish(folder, receipt)
            guard.check()
            # Receipt is the final durable publication, after all references.
            write_once(folder / "completion.json", receipt)
            return self.verify(folder, request=request, objects=objects, guard=guard)

    @staticmethod
    def _write(folder, name, value):
        limit = MAX_CONFIG_BYTES if name == "frozen_bundle.json" else MAX_DATASET_BYTES
        if len(canonical(value).encode("utf-8")) > limit:
            raise ValueError("TRAINING_ARTIFACT_SIZE_LIMIT:" + name)
        write_once(folder / name, value)

    @staticmethod
    def verify(folder: str | Path, *, request=None, objects=None, guard=None):
        receipt, _ = LesV7TrainingRepository._verify_with_scope(
            folder, request=request, objects=objects, guard=guard)
        return receipt

    @staticmethod
    def _verify_with_scope(folder: str | Path, *, request=None, objects=None, guard=None):
        """Internal complete verification; return its already-verified scope.

        The public verifier keeps its receipt-only contract. No caller can
        supply a scope or skip source verification through this internal port.
        """
        folder = Path(folder)
        receipt = read_json_bounded(folder / "completion.json", MAX_CONFIG_BYTES)
        stored_request = read_json_bounded(folder / "request.json", MAX_CONFIG_BYTES)
        if (receipt.get("schema") != "les-v7-training-completion/v1"
                or receipt.get("job_id") != folder.name
                or identity(stored_request) != receipt.get("request_sha256")
                or identity(stored_request) != folder.name
                or (request is not None and request != stored_request)
                or receipt.get("research_only") is not True
                or receipt.get("approved_for_live") is not False
                or receipt.get("execution_action") != "NO_ACTION"
                or any(receipt.get(key) != stored_request.get(key)
                       for key in ("dataset_id", "config_id", "dataset_kind"))
                or receipt.get("pinned_objects") != stored_request.get("pinned_objects")):
            raise ValueError("TRAINING_COMPLETION_BINDING_MISMATCH")
        if not isinstance(receipt.get("completed_at"), str):
            raise ValueError("TRAINING_COMPLETION_TIME_INVALID")
        completed_at = timestamp(receipt["completed_at"])
        if completed_at < timestamp(stored_request["fit_cutoff"]):
            raise ValueError("TRAINING_COMPLETION_TIME_INVALID")
        artifacts = receipt.get("artifacts", {})
        minimum = {"request.json", "dataset.json", "config.json", "fit_result.json",
                   "oof_predictions.json", "training_result.json"}
        if not isinstance(artifacts, dict) or not minimum <= artifacts.keys():
            raise ValueError("TRAINING_COMPLETION_ARTIFACTS_INCOMPLETE")
        loaded = {}
        for name, expected in artifacts.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("TRAINING_ARTIFACT_PATH_INVALID")
            value = read_json_bounded(folder / relative, MAX_DATASET_BYTES)
            if identity(value) != expected:
                raise ValueError("TRAINING_ARTIFACT_HASH_MISMATCH:" + name)
            loaded[name] = value
        compact = _load_record(folder / "fit_result.json")
        fitted = _hydrate_result(compact, folder)
        _validate_result(fitted, stored_request, config=loaded["config.json"], dataset_record=loaded["dataset.json"])
        if (loaded["training_result.json"] != _result_index(compact)
                or loaded["oof_predictions.json"] != dict(predictions=fitted["oof_predictions"])
                or loaded["dataset.json"].get("sha256") != stored_request["dataset_id"]
                or identity(loaded["dataset.json"].get("payload")) != stored_request["dataset_id"]
                or identity(loaded["config.json"]) != stored_request["config_id"]
                or receipt["status"] != fitted["status"]):
            raise ValueError("TRAINING_COMPLETION_CONTENT_MISMATCH")
        scope_body = None
        if loaded['dataset.json']['payload'].get('schema') == 'les-v7-sharded-training-dataset/v1':
            from infrastructure.les_v7_scope_training import verify_sharded_training_dataset
            scope_body = verify_sharded_training_dataset(loaded['dataset.json'], folder.parent.parent,
                                                        config=loaded['config.json'])
            if scope_body['pinned_objects'] != stored_request['pinned_objects']:
                raise ValueError('TRAINING_SHARDED_PINNED_OBJECTS_MISMATCH')
        for report_id, report in fitted["reports"].items():
            if loaded.get("reports/" + report_id + ".json") != report:
                raise ValueError("TRAINING_REPORT_REFERENCE_MISSING")
        if fitted["status"] == "READY_TO_FREEZE":
            published = loaded.get("frozen_bundle.json")
            if (not isinstance(published, dict) or published.get("sha256") != receipt["bundle_sha256"]
                    or identity(published.get("payload")) != published.get("sha256")):
                raise ValueError("TRAINING_FROZEN_MODEL_REFERENCE_INVALID")
            candidate = published["payload"]
            if (identity(_semantic_candidate(candidate)) != identity(_semantic_candidate(fitted["candidate"]))
                    or candidate["trained_until"] != receipt["first_published_at"]
                    or timestamp(candidate["trained_until"]) < max(timestamp(stored_request["fit_cutoff"]),
                                                                   timestamp(candidate["validation_available_at"]))
                    or timestamp(candidate["trained_until"]) > completed_at):
                raise ValueError("TRAINING_PUBLICATION_TIME_OR_MODEL_MISMATCH")
            FrozenBundle.load(published)
        elif ("frozen_bundle.json" in loaded or receipt.get("bundle_sha256") is not None
              or receipt.get("first_published_at") is not None):
            raise ValueError("NOT_READY_TRAINING_HAS_UNEXPECTED_MODEL")
        objects = objects or ResearchObjects(folder.parent / "objects")
        for ref in stored_request["pinned_objects"].values():
            objects.get(ref)
        if guard:
            guard.check()
        return dict(receipt, output=str(folder)), scope_body
