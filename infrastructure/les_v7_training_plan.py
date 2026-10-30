"""Immutable persistence adapter for the pure preregistered research plan.

The persistence location is separate from model fitting and execution. Moving
this adapter does not alter business plan/config content or frozen identities.
"""
from pathlib import Path

from application.les_v7_training_plan import EVALUATION, _validate_plan
from domain.les_v7.contracts import identity, timestamp
from infrastructure.research_archive import utc_now, write_once
from infrastructure.storage.les_v7_training import MAX_CONFIG_BYTES, read_json_bounded

FREEZE_SCHEMA = "les-v7-fixed-research-plan-freeze/v1"


def freeze_fixed_research_plan(plan, root, *, clock=utc_now):
    """Atomically retain immutable configs; wall-clock time is outside plan ID.

    The completion manifest is published last. Retry retains its first actual
    creation time; a complete but corrupt/missing reference fails explicitly.
    This is configuration preregistration, not model publication or training.
    """
    _validate_plan(plan)
    plan_id = identity(plan)
    folder = Path(root) / plan_id
    documents = {"plan.json": plan, "build-config.json": plan["build_config"],
                 "training-config.json": plan["training_config"]}
    references = {name: identity(value) for name, value in documents.items()}
    manifest_path = folder / "manifest.json"

    def verify_existing():
        manifest = read_json_bounded(manifest_path, MAX_CONFIG_BYTES)
        if (manifest.get("schema") != FREEZE_SCHEMA or manifest.get("plan_id") != plan_id
                or manifest.get("references") != references or manifest.get("research_only") is not True
                or manifest.get("approved_for_live") is not False):
            raise ValueError("RESEARCH_PLAN_FREEZE_MANIFEST_MISMATCH")
        if timestamp(manifest["created_at"]) < timestamp(EVALUATION):
            raise ValueError("RESEARCH_PLAN_EVALUATION_AFTER_FREEZE_CLOCK")
        for name, expected in documents.items():
            actual = read_json_bounded(folder / name, MAX_CONFIG_BYTES)
            if actual != expected or identity(actual) != references[name]:
                raise ValueError("RESEARCH_PLAN_FREEZE_REFERENCE_MISMATCH:" + name)
        return manifest

    if manifest_path.exists():
        manifest = verify_existing()
    else:
        created_at = clock()
        if timestamp(created_at) < timestamp(EVALUATION):
            raise ValueError("RESEARCH_PLAN_EVALUATION_AFTER_FREEZE_CLOCK")
        for name, value in documents.items():
            write_once(folder / name, value)
        manifest = dict(schema=FREEZE_SCHEMA, plan_id=plan_id, references=references,
                        created_at=created_at, research_only=True, approved_for_live=False)
        try:
            write_once(manifest_path, manifest)
        except ValueError:
            # A concurrent identical preregistration can have a different wall time.
            manifest = verify_existing()
    return dict(manifest, directory=str(folder), manifest=str(manifest_path))
