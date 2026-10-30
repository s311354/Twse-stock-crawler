"""Explicit research inference from a completed native offline model archive.

The ordinary daily model setting retains its deployment-review boundary. This
entry verifies native training/data archives and grants research use only.
"""
from __future__ import annotations

from pathlib import Path
import re

from domain.les_v7.fractals import fractal_history_requirements, is_active_fractal_feature
from domain.les_v7.localization import FrozenBundle
from infrastructure.storage.les_v7_dataset import LesV7DatasetRepository
from infrastructure.storage.les_v7_training import (
    LesV7TrainingRepository, MAX_CONFIG_BYTES, MAX_DATASET_BYTES, read_json_bounded,
)


def _verify_build(dataset, root):
    """Resolve the exact native dataset receipt, never substitute caller rows."""
    matches = []
    for parent in (root / "datasets", root / "builds"):
        if not parent.is_dir():
            continue
        folders = sorted(parent.iterdir())
        if len(folders) > 4096:
            raise ValueError("RESEARCH_BUILD_ARCHIVE_INVENTORY_LIMIT")
        for folder in folders:
            if not re.fullmatch(r"[0-9a-f]{64}", folder.name) or not folder.is_dir():
                continue
            completion = folder / "completion.json"
            if not completion.is_file():
                continue
            receipt = read_json_bounded(completion, MAX_CONFIG_BYTES)
            if receipt.get("dataset_id") != dataset["sha256"]:
                continue
            verified = LesV7DatasetRepository.verify(folder)
            original = read_json_bounded(folder / "dataset.json", MAX_DATASET_BYTES)
            if original != dataset or verified.get("dataset_kind") != "RESEARCH":
                raise ValueError("RESEARCH_SOURCE_DATASET_BINDING_MISMATCH")
            matches.append(verified)
    if len(matches) != 1:
        raise ValueError("RESEARCH_NATIVE_SOURCE_ARCHIVE_REQUIRED")
    return matches[0]


def load_research_bundle(path, *, allow_fr3=False):
    """Verify archive bytes/OOF/source lineage; no deployment or source certification."""
    path = Path(path).expanduser()
    if path.is_symlink() or path.name != "frozen_bundle.json":
        raise ValueError("RESEARCH_NATIVE_FROZEN_BUNDLE_PATH_REQUIRED")
    path = path.resolve()
    folder = path.parent
    if not re.fullmatch(r"[0-9a-f]{64}", folder.name):
        raise ValueError("RESEARCH_NATIVE_TRAINING_ARCHIVE_REQUIRED")
    receipt = LesV7TrainingRepository.verify(folder)
    if (receipt.get("status") != "READY_TO_FREEZE" or receipt.get("dataset_kind") != "RESEARCH"
            or receipt.get("research_only") is not True or receipt.get("approved_for_live") is not False
            or receipt.get("execution_action") != "NO_ACTION"):
        raise ValueError("READY_REAL_RESEARCH_MODEL_REQUIRED")
    record = read_json_bounded(path, MAX_CONFIG_BYTES)
    model = FrozenBundle.load(record)
    body = model.record()
    if type(allow_fr3) is not bool:
        raise ValueError("FR3_RESEARCH_ENTRY_BOOLEAN_REQUIRED")
    from domain.les_v7.fr3_plan import is_fr3_reference
    if is_fr3_reference(body.get("reference_policy", {})) and not allow_fr3:
        raise ValueError("FR3_INDEPENDENT_RESEARCH_ENTRY_REQUIRED_USE_INTERFACE_LES_V7_FR3")
    if (body.get("training_pipeline_schema") != "LES_V7_OFFLINE_TRAINING_R1"
            or body.get("training_manifest", {}).get("dataset_kind") != "RESEARCH"
            or body.get("research_only") is not True or body.get("approved_for_live") is not False
            or receipt.get("bundle_sha256") != model.sha256):
        raise ValueError("RESEARCH_MODEL_PUBLICATION_BINDING_MISMATCH")
    names = body["feature_names"]
    if not any(is_active_fractal_feature(name) for name in names):
        raise ValueError("RESEARCH_FRACTAL_MODEL_REQUIRED")
    dataset = read_json_bounded(folder / "dataset.json", MAX_DATASET_BYTES)
    payload = dataset["payload"]
    if payload.get("schema") != "les-v7-sharded-training-dataset/v1":
        _verify_build(dataset, folder.parent.parent)
    # The sharded dataset/source/OOF verification is delegated by the training
    # repository to its native verifier; do not recreate that logic here.
    universe = payload.get("universe")
    if (not isinstance(universe, list) or not universe or len(universe) > 2048
            or any(not isinstance(row, list) or len(row) != 2 or row[1] != "TWSE"
                   or not isinstance(row[0], str) or re.fullmatch(r"[0-9A-Z]{4,10}", row[0]) is None
                   for row in universe)
            or len({row[0] for row in universe}) != len(universe)):
        raise ValueError("RESEARCH_FIXED_TWSE_UNIVERSE_REQUIRED")
    requirements = fractal_history_requirements(0)
    closes = max((requirements[name]["required_close_count"] for name in names if name in requirements), default=0)
    return record, dict(schema="les-v7-research-model-use/v1", native_job_id=receipt["job_id"],
        bundle_sha256=model.sha256, dataset_id=receipt["dataset_id"], config_id=receipt["config_id"],
        archive_path=str(folder), bundle_path=str(path), published_at=receipt["first_published_at"],
        symbols=[row[0] for row in universe], required_valid_close_count=closes,
        historical_information_set_certified=False, research_only=True, approved_for_live=False,
        execution_action="NO_ACTION", order_authority=False,
        source_authenticity=receipt.get("source_authenticity", "SOURCE_AUTHENTICITY_NOT_CERTIFIED"))
