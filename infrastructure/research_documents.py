"""Offline document references backed by the existing immutable object store.

Datasets share <store>/objects/<sha256>. Flat evidence files and extracted job
bundles use <evidence directory>/objects/<sha256>. No absolute path is serialized.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

from application.research_evidence import (
    DOCUMENT_REF_SCHEMA, DocumentReader, validate_document_ref, verified_document,
)


def document_nodes(value):
    if isinstance(value, dict):
        if "body_base64" in value or "body_ref" in value:
            yield value
        for child in value.values():
            yield from document_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from document_nodes(child)


class EvidenceDocuments:
    def __init__(self, root: Path):
        # Keep the byte store reusable without creating composition-root cycles.
        from infrastructure.research_objects import ResearchObjects
        self.objects = ResearchObjects(root)
        self._stamps: dict[str, tuple] = {}
        self.references: dict[str, dict] = {}

    @classmethod
    def for_evidence(cls, path: str | Path, evidence_id: str) -> EvidenceDocuments:
        folder = Path(path).resolve().parent
        if folder.name == evidence_id:
            folder = folder.parent
        return cls(folder / "objects")

    def store(self, content: bytes) -> dict:
        sha = hashlib.sha256(content).hexdigest()
        if self.objects.path(sha).exists():
            self.objects.get(sha)
        else:
            self.objects.put(content)
        return dict(schema=DOCUMENT_REF_SCHEMA, sha256=sha, size_bytes=len(content))

    def read(self, reference: dict) -> bytes:
        validate_document_ref(reference)
        sha = reference["sha256"]
        try:
            content = self.objects.get(sha)
        except OSError as error:
            raise ValueError(f"SOURCE_DOCUMENT_UNAVAILABLE:{sha}") from error
        if len(content) != reference["size_bytes"]:
            raise ValueError(f"SOURCE_DOCUMENT_SIZE_MISMATCH:{sha}")
        return content

    def externalize(self, value, *, document_reader: DocumentReader | None = None):
        """Copy, verify and replace Base64; never change provenance or the input.

        Existing inline UTF-8 proof/ledger formats remain supported unchanged.
        An optional source reader copies already-referenced objects across stores.
        """
        if isinstance(value, list):
            return [self.externalize(child, document_reader=document_reader) for child in value]
        if not isinstance(value, dict):
            return value
        if "body_base64" in value or "body_ref" in value:
            content = verified_document(value, document_reader=document_reader or self.read)
            result = {key: child for key, child in value.items() if key not in {"body_base64", "body_ref"}}
            result["body_ref"] = (dict(value["body_ref"]) if "body_ref" in value and document_reader is None
                                  else self.store(content))
            return result
        return {key: self.externalize(child, document_reader=document_reader) for key, child in value.items()}

    def _stamp(self, sha: str) -> tuple:
        try:
            stat = self.objects.path(sha).stat()
        except OSError as error:
            raise ValueError(f"SOURCE_DOCUMENT_UNAVAILABLE:{sha}") from error
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def verify(self, payload: dict) -> dict:
        """Hash every referenced byte at open; pin metadata for warm-cache checks."""
        references, stamps = {}, {}
        inline = 0
        for document in document_nodes(payload):
            if "body_ref" not in document:
                verified_document(document)
                inline += 1
                continue
            ref = document["body_ref"]
            validate_document_ref(ref)
            sha = ref["sha256"]
            before = self._stamp(sha)
            verified_document(document, document_reader=self.read)
            if self._stamp(sha) != before:
                raise ValueError(f"SOURCE_DOCUMENT_CHANGED:{sha}")
            references[sha], stamps[sha] = dict(ref), before
        self.references, self._stamps = references, stamps
        return dict(status="VERIFIED", referenced_documents=len(references),
                    referenced_bytes=sum(ref["size_bytes"] for ref in references.values()),
                    legacy_inline_documents=inline)

    def check(self) -> None:
        """A cached parse may never hide a removed or modified original."""
        for sha, before in self._stamps.items():
            if self._stamp(sha) != before:
                raise ValueError(f"SOURCE_DOCUMENT_CHANGED:{sha}")
