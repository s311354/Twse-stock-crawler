"""Publish explicit retrospective reference evidence without touching old PIT."""
from copy import deepcopy
from pathlib import Path

from application.replay_contracts import digest
from application.research_evidence import RAW_EVIDENCE_SCHEMA
from infrastructure.les_v7_training_source import FINAL_REFERENCE, MAX_BYTES
from infrastructure.replay_evidence import load_evidence
from infrastructure.research_archive import create_dataset, utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments


def reference_document_reader(*stores):
    """Exact hash lookup only; corruption is not hidden by another store."""
    def read(reference):
        for store in stores:
            if store.objects.path(reference["sha256"]).exists():
                return store.read(reference)
        raise ValueError("REFERENCE_REPAIR_SOURCE_DOCUMENT_UNAVAILABLE")
    return read


def repair_reference_evidence(parent, *, calendar_path, actions_path, store):
    from infrastructure.les_v7_calendar_final import load_final_calendar
    from infrastructure.les_v7_actions_final import load_actions_final
    parent = Path(parent)
    parent_id, original = load_evidence(parent, max_bytes=MAX_BYTES)
    if original.get("schema") != RAW_EVIDENCE_SCHEMA or original.get("market") != "TWSE":
        raise ValueError("RAW_TWSE_PARENT_REQUIRED")
    if original.get("purpose") == "SYNTHETIC_TEST_ONLY":
        raise ValueError("REAL_REFERENCE_REPAIR_REJECTS_SYNTHETIC_PARENT")
    calendar, calendar_documents = load_final_calendar(calendar_path)
    actions = load_actions_final(actions_path)
    action_documents = EvidenceDocuments(Path(actions_path).parent.parent / "objects")
    original_documents = EvidenceDocuments.for_evidence(parent, parent_id)
    original_documents.verify(original)
    payload = deepcopy(original)
    payload["original_calendar_before_reference_repair"] = deepcopy(original["calendar"])
    payload["calendar"] = calendar
    payload["corporate_actions_final"] = actions
    payload["reference_evidence_policy"] = FINAL_REFERENCE
    payload["reference_repair"] = dict(schema="les-v7-reference-evidence-repair/v1",
        parent_evidence_id=parent_id, final_calendar_sha256=digest(calendar),
        final_actions_packet_id=actions["evidence_id"], policy=FINAL_REFERENCE,
        historical_information_set_certified=False, original_historical_available_at=None,
        raw_financial_restriction_and_kelly_evidence_unchanged=True,
        limitations=actions["limitations"])
    destination = create_dataset(payload, Path(store), parent=parent, collected_at=utc_now(),
        purpose="RETROSPECTIVE_FINAL_REFERENCE_EVIDENCE_NOT_PIT_REPAIR",
        document_reader=reference_document_reader(original_documents, calendar_documents, action_documents))
    if load_evidence(parent, max_bytes=MAX_BYTES)[0] != parent_id:
        raise ValueError("REFERENCE_REPAIR_PARENT_CHANGED")
    write_once(destination.parent / "reference-repair.json", payload["reference_repair"])
    return destination
