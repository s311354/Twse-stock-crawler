"""Manual native retention and real recomputation through the shared engine."""
from __future__ import annotations

from dataclasses import asdict, replace
import datetime as dt
import fcntl
import hashlib
import io
import json
from pathlib import Path
import platform
import tempfile
import zipfile
from importlib.metadata import distributions

from application.replay_contracts import ReplayRequest, ReplayAccount, ReplayHolding, ReplayVersions, digest
from application.research_audit import coverage_audit, compare_results
from infrastructure.replay_runtime import build_replay_engine, current_versions, parameter_manifest, ROOT
from infrastructure.replay_evidence import load_evidence
from infrastructure.research_archive import write_once, write_bytes_once, utc_now
from infrastructure.storage.replay_state import SQLiteReplayState


def request_from_document(document: dict, state_path: Path) -> ReplayRequest:
    account = dict(document["account"])
    account["holdings"] = tuple(ReplayHolding(**h) for h in account["holdings"])
    return ReplayRequest(document["replay_id"], dt.datetime.fromisoformat(document["as_of"]),
        dt.date.fromisoformat(document["history_start"]), dt.date.fromisoformat(document["history_end"]),
        tuple(document["symbols"]), document["market"], document["sector"], ReplayAccount(**account),
        ReplayVersions(**document["versions"]), document["evidence_id"], state_path)


def retain_research(evidence_file: Path, request: ReplayRequest, root: Path, *, predecessor: dict) -> Path:
    """Account/predecessor explicit. Engine state exists only inside this archive.

    Caller state_path is never written. Completion receipt is published last.
    Retry uses the SAME intent/prestate and committed engine result if present.
    """
    eid, payload = load_evidence(evidence_file)
    from application.research_evidence import RAW_EVIDENCE_SCHEMA
    if payload.get("schema") != RAW_EVIDENCE_SCHEMA:
        raise ValueError("DAILY_RETENTION_REQUIRES_RAW_V2_DATASET")
    if eid != request.evidence_id:
        raise ValueError("DAILY_DATASET_MISMATCH")
    if request.versions != current_versions():
        raise ValueError("REPLAY_MODEL_OR_CODE_VERSION_MISMATCH")
    inputs = request.inputs()
    run_id = digest(dict(request=inputs, predecessor_state_id=predecessor["state_id"]))
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_once(folder / "request.json", inputs)
        write_once(folder / "predecessor.json", predecessor)
        if (folder / "complete.json").exists():
            receipt = json.loads((folder / "complete.json").read_text())
            for name, sha in receipt["artifacts"].items():
                if digest(json.loads((folder / name).read_text())) != sha:
                    raise ValueError("DAILY_ARCHIVE_ARTIFACT_HASH_MISMATCH")
            if hashlib.sha256((folder / "source.zip").read_bytes()).hexdigest() != receipt["source_archive_sha256"]:
                raise ValueError("DAILY_SOURCE_ARCHIVE_HASH_MISMATCH")
            return folder
        if not (folder / "intent.json").exists():
            write_once(folder / "intent.json", dict(schema="daily-research/v1", research_run_id=run_id,
                started_at=utc_now(), dataset_id=eid, evidence_file=str(evidence_file.resolve()),
                request_sha256=digest(inputs), predecessor_state_id=predecessor["state_id"],
                environment=dict(python=platform.python_version(), platform=platform.platform()),
                collection_policy="FROZEN_BEFORE_RESEARCH_NO_LIVE_FALLBACK", versions=asdict(request.versions)))
        local = replace(request, state_path=folder / "state.sqlite3")
        store = SQLiteReplayState()
        try:
            if not local.state_path.exists():
                store.restore_snapshot(local, predecessor)
            elif store.export_snapshot(local, before=request.as_of.isoformat()) != predecessor:
                raise ValueError("DAILY_PREDECESSOR_DRIFT")
            engine, _ = build_replay_engine(evidence_file)
            result = engine.run(local)
            selected = engine.evidence.select(local)
            audit = coverage_audit(local, selected, result)
            post = store.export_snapshot(local)
            values = {"result.json": result, "coverage.json": audit, "poststate.json": post,
                      "parameters.json": parameter_manifest(),
                      "environment.json": dict(python=platform.python_version(), platform=platform.platform(),
                          dependencies=sorted({(d.metadata["Name"], d.version) for d in distributions()
                                               if d.metadata["Name"]}))}
            for name, value in values.items(): write_once(folder / name, value)
            # Keep the actual source used, including dirty changes; a version
            # string/HEAD alone is insufficient to recreate old implementations.
            code_file = folder / "source.zip"
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                paths = [ROOT / "main.py", ROOT / "requirements.txt"]
                for group in ("application", "domain", "infrastructure", "interface", "config", "twse"):
                    paths.extend(sorted((ROOT / group).rglob("*.py")))
                for path in paths:
                    if path.is_file():
                        info = zipfile.ZipInfo(str(path.relative_to(ROOT)))
                        info.compress_type = zipfile.ZIP_DEFLATED
                        archive.writestr(info, path.read_bytes())
            write_bytes_once(code_file, buffer.getvalue())
            if request.versions != current_versions():
                raise ValueError("REPLAY_VERSION_CHANGED_DURING_RETENTION")
            artifacts = {name: digest(value) for name, value in values.items()}
            artifacts.update({"request.json": digest(inputs), "predecessor.json": digest(predecessor)})
            write_once(folder / "complete.json", dict(schema="daily-research/v1", research_run_id=run_id,
                dataset_id=eid, completed_at=utc_now(), status="COMPLETED", coverage_status=audit["status"],
                artifacts=artifacts, result_sha256=digest(result), poststate_id=post["state_id"],
                source_archive_sha256=hashlib.sha256(code_file.read_bytes()).hexdigest(),
                reproducibility="SOURCE_RETAINED_CURRENT_ENV_REEXECUTION_REQUIRED", live_order_authorized=False))
        except Exception as error:
            write_once(folder / f"failure-{digest(str(error))}.json", dict(status="INCOMPLETE", error=str(error)))
            raise
    return folder


def reproduce_research(folder: Path, evidence_file: Path) -> dict:
    receipt = json.loads((folder / "complete.json").read_text())
    for name, sha in receipt["artifacts"].items():
        if digest(json.loads((folder / name).read_text())) != sha:
            raise ValueError("DAILY_ARCHIVE_ARTIFACT_HASH_MISMATCH")
    if hashlib.sha256((folder / "source.zip").read_bytes()).hexdigest() != receipt["source_archive_sha256"]:
        raise ValueError("DAILY_SOURCE_ARCHIVE_HASH_MISMATCH")
    with tempfile.TemporaryDirectory(prefix="twse-reproduce-") as directory:
        request = request_from_document(json.loads((folder / "request.json").read_text()), Path(directory) / "state.sqlite3")
        if request.versions != current_versions():
            raise ValueError("OLD_CODE_OR_ENVIRONMENT_REQUIRED: use retained source and compatible dependencies")
        predecessor = json.loads((folder / "predecessor.json").read_text())
        SQLiteReplayState().restore_snapshot(request, predecessor)
        engine, eid = build_replay_engine(evidence_file)
        if eid != receipt["dataset_id"]: raise ValueError("REPRODUCTION_DATASET_MISMATCH")
        actual = engine.run(request)  # Fresh state: cannot return the retained day's cached result.
        expected = json.loads((folder / "result.json").read_text())
        comparison = compare_results(expected, actual)
        comparison.update(reexecuted=True, reproduced_at=utc_now(),
                          predecessor_state_id=predecessor["state_id"], dataset_id=eid)
        return comparison
