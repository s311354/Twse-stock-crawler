"""Write-once reference dataset jobs; no Bank/account or live-data port."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from application.les_v7_dataset import BUILDER_VERSION, build_training_dataset, validate_build_config
from domain.les_v7.contracts import canonical, identity, timestamp
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.les_v7_training import MAX_DATASET_BYTES, read_json_bounded


def _artifact_stamp(path):
    def metadata(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    # Bind both the directory entry and its target for legacy linked archives.
    return metadata(path.lstat()), metadata(path.stat())


class _VerifiedArtifacts(dict):
    """One verification call's detached JSON and bounded original-file stamps."""

    def __init__(self, folder):
        super().__init__()
        self.folder, self.stamps = folder, {}

    def read(self, name):
        if name not in self:
            path = self.folder / name
            before = _artifact_stamp(path)
            self[name] = read_json_bounded(path, MAX_DATASET_BYTES)
            self.stamps[name] = before
            self.check(name)
        return self[name]

    def check(self, name=None):
        for key in self.stamps if name is None else (name,):
            try:
                changed = _artifact_stamp(self.folder / key) != self.stamps[key]
            except OSError as error:
                raise ValueError('DATASET_BUILD_ARCHIVE_CHANGED:' + key) from error
            if changed:
                raise ValueError('DATASET_BUILD_ARCHIVE_CHANGED:' + key)


@contextmanager
def _lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".builder.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("DATASET_BUILD_JOB_BUSY") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class LesV7DatasetRepository:
    def __init__(self, root, *, guard_factory=ResearchVersionGuard, clock=utc_now, after_build=None):
        self.root, self.guard_factory, self.clock = Path(root), guard_factory, clock
        self.after_build = after_build

    def run(self, source, config, *, allow_synthetic=False, builder=build_training_dataset):
        validate_build_config(config)
        if identity(source["payload"]) != source["sha256"]:
            raise ValueError("DATASET_SOURCE_HASH_MISMATCH")
        if source["payload"]["dataset_kind"] == "SYNTHETIC_TEST_ONLY" and not allow_synthetic:
            raise ValueError("SYNTHETIC_DATASET_BUILD_REQUIRES_EXPLICIT_OPT_IN")
        guard = self.guard_factory()
        if timestamp(self.clock()) < timestamp(config["evaluation_cutoff"]):
            raise ValueError("DATASET_EVALUATION_AFTER_BUILD_CLOCK")
        objects = ResearchObjects(self.root / "objects")
        pinned = guard.retain(objects)
        request = dict(schema="les-v7-dataset-build-request/v1", version=BUILDER_VERSION,
            source_packet_id=source["sha256"], config_id=identity(config), pinned_objects=pinned,
            dataset_kind=source["payload"]["dataset_kind"], research_only=True, approved_for_live=False)
        job_id, folder = identity(request), self.root / identity(request)
        with _lock(folder):
            guard.check()
            if (folder / "completion.json").exists():
                return self.verify(folder, expected_request=request, objects=objects, guard=guard)
            for name, value in (("request.json", request), ("source.json", source), ("config.json", config)):
                self._write(folder / name, value)
            result_file = folder / "build_result.json"
            dataset_file = folder / "dataset.json"
            if result_file.exists():
                result = read_json_bounded(result_file, MAX_DATASET_BYTES)
                dataset = read_json_bounded(dataset_file, MAX_DATASET_BYTES)
            else:
                computed = builder(source, config)
                dataset, result = computed["dataset"], computed["result"]
                if identity(source["payload"]) != request["source_packet_id"] or identity(config) != request["config_id"]:
                    raise ValueError("DATASET_BUILD_INPUT_CHANGED")
                guard.check()
                self._validate(dataset, result, request)
                # Dataset is published before its dependent diagnostics; retry
                # after this boundary recomputes deterministically, never clobbers.
                self._write(dataset_file, dataset)
                self._write(result_file, result)
            self._validate(dataset, result, request)
            receipt = dict(schema="les-v7-dataset-build-completion/v1", job_id=job_id,
                request_sha256=identity(request), dataset_id=dataset["sha256"],
                source_packet_id=request["source_packet_id"], config_id=request["config_id"],
                dataset_kind=request["dataset_kind"], status=result["status"], counts=result["counts"],
                artifacts={name: identity(read_json_bounded(folder / name, MAX_DATASET_BYTES)) for name in
                    ("request.json", "source.json", "config.json", "dataset.json", "build_result.json")},
                pinned_objects=pinned, completed_at=self.clock(), research_only=True, approved_for_live=False,
                source_authenticity="NOT_INDEPENDENTLY_CERTIFIED", execution_action="NO_ACTION")
            if self.after_build:
                self.after_build(folder, receipt)
            guard.check()
            write_once(folder / "completion.json", receipt)
            return self.verify(folder, expected_request=request, objects=objects, guard=guard)

    @staticmethod
    def _write(path, value):
        if len(canonical(value).encode("utf-8")) > MAX_DATASET_BYTES:
            raise ValueError("DATASET_BUILD_ARTIFACT_SIZE_LIMIT:" + path.name)
        write_once(path, value)

    @staticmethod
    def _validate(dataset, result, request):
        payload = dataset.get("payload", {})
        manifest = payload.get("builder_manifest", {})
        if (identity(payload) != dataset.get("sha256") or payload.get("schema") != "les-v7-training-dataset/v1"
                or payload.get("dataset_kind") != request["dataset_kind"]
                or manifest.get("version") != BUILDER_VERSION
                or manifest.get("source_packet_id") != request["source_packet_id"]
                or manifest.get("build_config_id") != request["config_id"]
                or result.get("schema") != "les-v7-dataset-build-result/v1"
                or result.get("dataset_id") != dataset["sha256"]
                or result.get("status") not in {"PARTIAL", "EVIDENCE_READY"}
                or result.get("research_only") is not True or result.get("approved_for_live") is not False):
            raise ValueError("DATASET_BUILD_RESULT_BINDING_MISMATCH")

    @staticmethod
    def verify(folder, *, expected_request=None, objects=None, guard=None):
        receipt, _ = LesV7DatasetRepository._verify_with_artifacts(
            folder, expected_request=expected_request, objects=objects, guard=guard)
        return receipt

    @staticmethod
    def _verify_with_artifacts(folder, *, expected_request=None, objects=None, guard=None):
        """Complete fresh verification, retaining the already-read child JSON.

        Internal scope orchestration consumes these exact checked payloads and
        releases them at the child boundary. No prior verdict or caller-loaded
        artifact can skip native hashes, bindings or original object checks.
        """
        folder = Path(folder)
        artifacts = _VerifiedArtifacts(folder)
        receipt = artifacts.read("completion.json")
        request = artifacts.read("request.json")
        timestamp(receipt["completed_at"])
        if (receipt.get("schema") != "les-v7-dataset-build-completion/v1" or folder.name != identity(request)
                or receipt.get("job_id") != folder.name or receipt.get("request_sha256") != identity(request)
                or expected_request is not None and expected_request != request
                or receipt.get("execution_action") != "NO_ACTION" or receipt.get("approved_for_live") is not False
                or receipt.get("research_only") is not True or receipt.get("pinned_objects") != request["pinned_objects"]):
            raise ValueError("DATASET_BUILD_COMPLETION_BINDING_MISMATCH")
        required = {"request.json", "source.json", "config.json", "dataset.json", "build_result.json"}
        if set(receipt.get("artifacts", {})) != required:
            raise ValueError("DATASET_BUILD_ARTIFACT_REFERENCES_REQUIRED")
        for name, expected in receipt["artifacts"].items():
            if identity(artifacts.read(name)) != expected:
                raise ValueError("DATASET_BUILD_ARTIFACT_HASH_MISMATCH:" + name)
        source, config = artifacts["source.json"], artifacts["config.json"]
        if (identity(source["payload"]) != request["source_packet_id"] or source["sha256"] != request["source_packet_id"]
                or identity(config) != request["config_id"]
                or any(receipt.get(k) != request[k] for k in ("source_packet_id", "config_id", "dataset_kind"))):
            raise ValueError("DATASET_BUILD_INPUT_BINDING_MISMATCH")
        LesV7DatasetRepository._validate(artifacts["dataset.json"], artifacts["build_result.json"], request)
        if (receipt["dataset_id"] != artifacts["dataset.json"]["sha256"]
                or receipt["status"] != artifacts["build_result.json"]["status"]
                or receipt["counts"] != artifacts["build_result.json"]["counts"]):
            raise ValueError("DATASET_BUILD_COMPLETION_RESULT_MISMATCH")
        objects = objects or ResearchObjects(folder.parent / "objects")
        for reference in request["pinned_objects"].values():
            objects.get(reference)
        if guard:
            guard.check()
        artifacts.check()
        return dict(receipt, output=str(folder), dataset=str(folder / "dataset.json")), artifacts
