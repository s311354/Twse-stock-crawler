"""Content-addressed evidence and write-once artifacts with atomic publication."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import tempfile

from application.replay_contracts import canonical_json, digest
from infrastructure.replay_evidence import load_evidence


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_once(path: Path, value: dict) -> None:
    write_bytes_once(path, canonical_json(value).encode("utf-8"))


def write_bytes_once(path: Path, content: bytes) -> None:
    """Atomic no-clobber publication, including concurrent writers and retries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"IMMUTABLE_ARTIFACT_CONFLICT:{path.name}")
        # Publish directory entries durably before a later completion receipt.
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.unlink(temporary)


def dataset_diff(before: dict, after: dict) -> dict:
    return {key: dict(before_sha256=digest(before.get(key)), after_sha256=digest(after.get(key)))
            for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}


def create_dataset(payload: dict, root: Path, *, parent: Path | None = None,
                   collected_at: str | None = None, purpose: str = "RESEARCH", document_reader=None) -> Path:
    from application.research_evidence import RAW_EVIDENCE_SCHEMA
    from infrastructure.research_documents import EvidenceDocuments
    if payload.get("schema") != RAW_EVIDENCE_SCHEMA:
        raise ValueError("IMMUTABLE_DATASET_REQUIRES_RAW_V2_EVIDENCE")
    payload = EvidenceDocuments(root / "objects").externalize(payload, document_reader=document_reader)
    dataset_id = digest(payload)
    folder = root / dataset_id
    evidence = folder / "evidence.json"
    parent_id, changes = None, {}
    if parent:
        parent_id, previous = load_evidence(parent)
        changes = dataset_diff(previous, payload)
        if parent_id == dataset_id:
            # Relocation of identical content is not a new version or a self-parent.
            parent_id = None
    write_once(evidence, dict(evidence_id=dataset_id, payload=payload))
    manifest = folder / "manifest.json"
    # All non-semantic collection/publication times are OUTSIDE content identity.
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["dataset_id"] != dataset_id or load_evidence(evidence)[0] != dataset_id:
            raise ValueError("DATASET_MANIFEST_MISMATCH")
    else:
        value = dict(schema="research-dataset/v1", dataset_id=dataset_id, evidence_id=dataset_id,
            evidence_schema=payload["schema"], created_at=utc_now(), collected_at=collected_at,
            market=payload["market"], sector=payload["sector"], symbols=payload["symbols"],
            universe_policy=payload.get("universe_policy"), start=payload["start"], end=payload["end"],
            market_policy=payload["market_policy"], purpose=purpose,
            parent_dataset_id=parent_id, repair_changes=changes,
            preparation_gaps=payload.get("preparation_gaps", []),
            completeness="NOT_CERTIFIED_RUN_CUTOFF_AUDIT", model_derived_inputs=False)
        try:
            write_once(manifest, value)
        except ValueError:
            # A concurrent builder may have published an equivalent dataset at
            # a different wall time. Its complete first manifest is retained.
            existing = json.loads(manifest.read_text())
            if existing["dataset_id"] != dataset_id:
                raise
    return evidence
