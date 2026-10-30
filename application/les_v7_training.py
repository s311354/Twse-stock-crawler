"""Explicit research-only offline fitting entry; no broker or daily-state port."""
from __future__ import annotations

from typing import Protocol

from domain.les_v7.contracts import identity, timestamp


class TrainingRepository(Protocol):
    def run(self, dataset_record: dict, config: dict, *, trainer) -> dict: ...


def train_request(dataset_record: dict, config: dict, repository: TrainingRepository,
                  *, allow_synthetic: bool = False, trainer=None) -> dict:
    """Fit/publish through a write-once repository, preserving source identity.

    This confirms the supplied content hash, not the authenticity of raw market
    sources. Source evidence validation remains the dataset builder's duty.
    """
    if not isinstance(dataset_record, dict) or not isinstance(config, dict):
        raise ValueError("TRAINING_OBJECT_REQUEST_REQUIRED")
    payload = dataset_record.get("payload")
    if not isinstance(payload, dict) or dataset_record.get("sha256") != identity(payload):
        raise ValueError("TRAINING_DATASET_HASH_MISMATCH")
    if payload.get("schema") != "les-v7-training-dataset/v1":
        raise ValueError("TRAINING_DATASET_SCHEMA_UNSUPPORTED")
    kind = payload.get("dataset_kind")
    if kind not in {"RESEARCH", "SYNTHETIC_TEST_ONLY"}:
        raise ValueError("EXPLICIT_TRAINING_DATASET_KIND_REQUIRED")
    if kind == "SYNTHETIC_TEST_ONLY" and allow_synthetic is not True:
        raise ValueError("SYNTHETIC_TRAINING_REQUIRES_EXPLICIT_OPT_IN")
    timestamp(config["fit_cutoff"])
    # Validate JSON finiteness before either fitting or immutable publication.
    identity(config)
    if trainer is None:
        from domain.les_v7.training import train_offline
        trainer = train_offline
    return repository.run(dataset_record, config, trainer=trainer)
