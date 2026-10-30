"""Fixed current-data research identities, separate from execution attempts.

This is not historical replay or an account ledger. Evidence and account inputs
may evolve between sessions; model/code/policy and the ordered universe may not.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re

from application.replay_contracts import digest
from infrastructure.research_archive import utc_now, write_once


STUDY_SCHEMA = "les-v7-daily-study/v1"
IDENTITY_SCHEMA = "les-v7-daily-study-identity/v1"
MAX_STUDY_BYTES = 2 * 1024 * 1024


class ResearchSeriesError(ValueError):
    pass


def current_versions():
    # Reuse the actual code/parameter/runtime identity, not Git HEAD or a label.
    from infrastructure.replay_runtime import current_versions as versions
    return versions()


def validate_research_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is None:
        raise ResearchSeriesError("INVALID_RESEARCH_ID: use 1-80 ASCII letters, digits, '_' or '-'")
    return value


def make_research_identity(metadata, request, settings):
    research_id = validate_research_id(metadata["research_id"])
    symbols = list(metadata["symbols"])
    if symbols != list(request.stocklist) or len(set(symbols)) != len(symbols):
        raise ResearchSeriesError("RESEARCH_UNIVERSE_BINDING_MISMATCH")
    return dict(schema=IDENTITY_SCHEMA, research_id=research_id, model_version="7.0.0",
        bundle_sha256=metadata["bundle_sha256"], training_dataset_id=metadata["dataset_id"],
        training_config_id=metadata["config_id"], native_job_id=metadata["native_job_id"],
        market=request.market_source, sector=request.stocktype.name, symbols=symbols,
        versions=asdict(current_versions()),
        components={name: metadata[name] for name in (
            "admission_component", "execution_component", "valuation_policy", "decision_thresholds")},
        effective_inputs=dict(backtrack_begin=request.beginbacktrack, backtrack_end=request.endbacktrack,
            mops_enabled=request.mops_enabled, mops_history_years=request.mops_history_years,
            native_evidence=settings.les_v7_native_evidence, source_repair=settings.les_v7_source_repair,
            strategy_policy_sha256=(digest(json.loads(settings.les_v7_strategy_policy_json))
                                    if settings.les_v7_strategy_policy_json else None)),
        research_only=True, approved_for_live=False, execution_action="NO_ACTION")


def check_versions(identity):
    if identity.get("versions") != asdict(current_versions()):
        raise ResearchSeriesError("RESEARCH_VERSIONS_CHANGED: start a new --research-id")


def _validate_paths(root, identity):
    root = Path(root)
    if root.name != validate_research_id(identity.get("research_id")):
        raise ResearchSeriesError("RESEARCH_STATE_PATH_BINDING_MISMATCH")
    # Do not let a study/DB link silently redirect the research writer into the
    # ordinary daily database. Never follow a link to read/create a manifest.
    for path in (root.parent, root, root / "study.json", root / "portfolio.sqlite3", root / "runs"):
        if path.is_symlink():
            raise ResearchSeriesError("RESEARCH_STATE_SYMLINK_NOT_ALLOWED")
    if root.exists() and not root.is_dir():
        raise ResearchSeriesError("RESEARCH_STATE_ROOT_NOT_DIRECTORY")
    return root


def validate_study_paths(root, research_id):
    """Check confinement before settings probes create any temporary files."""
    return _validate_paths(root, dict(research_id=research_id))


def check_study(root, identity):
    """Read-only compatibility preflight; does not initialize a DB/manifest."""
    root = _validate_paths(root, identity)
    path = root / "study.json"
    if not path.exists():
        attempts = root / "runs"
        if (any((root / name).exists() for name in ("portfolio.sqlite3", "request.json", "completion.json"))
                or (attempts.exists() and (not attempts.is_dir() or any(attempts.iterdir())))):
            raise ResearchSeriesError("RESEARCH_LEGACY_STATE_REQUIRES_NEW_ID")
        return None
    try:
        if path.stat().st_size > MAX_STUDY_BYTES:
            raise ResearchSeriesError("RESEARCH_STUDY_MANIFEST_TOO_LARGE")
        record = json.loads(path.read_bytes())
        if (not isinstance(record, dict) or record.get("schema") != STUDY_SCHEMA
                or record.get("identity_id") != digest(record.get("identity"))):
            raise ResearchSeriesError("RESEARCH_STUDY_MANIFEST_CORRUPT")
        if record["identity"] != identity:
            raise ResearchSeriesError("RESEARCH_IDENTITY_CHANGED: model, versions or universe changed; start a new --research-id")
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise ResearchSeriesError("RESEARCH_STUDY_MANIFEST_UNREADABLE") from error
    if (root / "portfolio.sqlite3").exists():
        from infrastructure.storage.research_portfolio import SQLiteResearchPortfolioPersistence, ResearchStateError
        try:
            SQLiteResearchPortfolioPersistence(root / "portfolio.sqlite3", identity)
        except ResearchStateError as error:
            raise ResearchSeriesError(str(error)) from error
    return record


def ensure_study(root, identity):
    """Atomically publish one immutable study identity, under the run lock."""
    check_versions(identity)
    existing = check_study(root, identity)
    if existing is not None:
        return existing
    root = Path(root)
    record = dict(schema=STUDY_SCHEMA, identity=identity, identity_id=digest(identity), created_at=utc_now())
    try:
        write_once(root / "study.json", record)
    except ValueError:
        # Another same-identity initializer may have won with a different
        # creation clock; retain that immutable original, never overwrite it.
        existing = check_study(root, identity)
        if existing is None:
            raise
        return existing
    return check_study(root, identity)


def metadata_identity(metadata):
    identity = metadata.get("research_identity")
    if (not isinstance(identity, dict) or identity.get("schema") != IDENTITY_SCHEMA
            or identity.get("research_id") != metadata.get("research_id")
            or metadata.get("research_identity_id") != digest(identity)):
        raise ResearchSeriesError("RESEARCH_METADATA_IDENTITY_MISMATCH")
    # A valid identity hash alone is not permission to attach a different
    # bundle/universe to that study through a manually constructed settings API.
    for field, source in (("bundle_sha256", "bundle_sha256"),
                          ("training_dataset_id", "dataset_id"),
                          ("training_config_id", "config_id"),
                          ("native_job_id", "native_job_id"), ("symbols", "symbols")):
        if identity.get(field) != metadata.get(source):
            raise ResearchSeriesError("RESEARCH_METADATA_IDENTITY_MISMATCH")
    if identity.get("components") != {name: metadata.get(name) for name in (
            "admission_component", "execution_component", "valuation_policy", "decision_thresholds")}:
        raise ResearchSeriesError("RESEARCH_METADATA_IDENTITY_MISMATCH")
    return identity
