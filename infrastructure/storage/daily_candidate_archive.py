"""Immutable per-session index of all v7 candidates and linked evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import datetime as dt

import pandas as pd

from application.replay_contracts import canonical_json, digest


SCHEMA = "les-v7-daily-candidates/v1"
REQUIRED_EREM_FIELDS = {"EREM 分數", "EREM 決策訊號", "EREM 理由"}


def _json_value(value):
    """Normalize dataframe scalars without allowing NaN/NaT into artifacts."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_candidate(report: dict, report_reference: dict, erem: dict) -> dict:
    """Project audit/join fields; the immutable native report remains canonical."""
    envelope = report["envelope"]
    diagnostics = report["diagnostics"]
    strategy = envelope["strategy"]
    decision = strategy["strategy_decision"]
    report_sha = report_reference["report_sha256"]
    erem = {key: _json_value(value) for key, value in sorted(erem.items())}
    if not REQUIRED_EREM_FIELDS <= set(erem):
        raise ValueError("V7_CANDIDATE_SNAPSHOT_EREM_FIELDS_REQUIRED")
    identity = {
        "market": diagnostics["gate_bindings"]["DATA_INTEGRITY"]["market"],
        "symbol": report["symbol"],
        "signal_date": report["signal_date"],
        "cutoff_at": envelope["event"]["cutoff_at"],
        "parameter_version": report["parameter_version"],
        "report_sha256": report_sha,
        "erem_sha256": digest(erem),
    }
    candidate_id = digest(identity)
    plan = diagnostics.get("reference_plan")
    structure = diagnostics.get("structure_plan") or {}
    authorization = report.get("pae_authorization") or {}
    if decision == "DATA_INSUFFICIENT":
        candidate_state = "BLOCKED_DATA_INSUFFICIENT"
    elif decision not in {"BUY", "BUY_SMALL"}:
        candidate_state = "NOT_ADMITTED"
    elif authorization.get("authorized") is True:
        candidate_state = "AWAITING_EXTERNAL_FILL_OBSERVATION"
    else:
        candidate_state = "SIGNAL_NOT_ACCOUNT_AUTHORIZED"
    return {
        **identity,
        "candidate_id": candidate_id,
        "candidate_state": candidate_state,
        "decision": decision,
        "decision_reasons": list(strategy.get("reason_codes", [])),
        "score": envelope["score"],
        "quality": diagnostics["quality"],
        "gate_receipts": diagnostics.get("gate_receipts", []),
        "six_factor_ledger": diagnostics.get("factor_ledger", {}),
        "r19": {
            "policy": (diagnostics.get("rule_baseline") or {}).get("policy"),
            "route_evaluation": diagnostics.get("r19_route_evaluation"),
            "entry_route": diagnostics.get("reference_entry_route"),
            "plan": plan,
            "structure_plan": structure,
            "execution_assessment": diagnostics.get("execution_assessment"),
        },
        "account_authorization": {
            "authorized": authorization.get("authorized"),
            "live_order_authorized": authorization.get("live_order_authorized"),
            "indicative_shares": authorization.get("indicative_shares"),
            "reason_codes": authorization.get("reason_codes", []),
        },
        "erem": erem,
        "fill_observation": {
            "status": "NOT_CAPTURED_BY_CRAWLER",
            "owner": "BankEmulator",
            "join_key": candidate_id,
        },
        "report_reference": report_reference,
    }


def publish_snapshot(output_dir: str | Path, prefix: str, candidates: list[dict]) -> Path:
    """Publish a stable content-addressed snapshot; never replace an existing one."""
    ordered = sorted(candidates, key=lambda row: (row["signal_date"], row["symbol"]))
    if len({(row["signal_date"], row["symbol"]) for row in ordered}) != len(ordered):
        raise ValueError("DUPLICATE_DAILY_CANDIDATE_IDENTITY")
    dates = sorted({row["signal_date"] for row in ordered})
    payload = {
        "schema": SCHEMA,
        "model": "low_entry_score_v7",
        "signal_dates": dates,
        "candidate_count": len(ordered),
        "candidates": ordered,
    }
    snapshot_id = digest(payload)
    document = {"snapshot_id": snapshot_id, **payload}
    encoded = canonical_json(document).encode("utf-8")
    if hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest() != snapshot_id:
        raise ValueError("DAILY_CANDIDATE_SNAPSHOT_ID_MISMATCH")
    target = Path(output_dir) / f"{prefix}_les_v7_candidates" / f"{snapshot_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".candidate-publish-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != encoded:
                raise ValueError("IMMUTABLE_DAILY_CANDIDATE_CONFLICT")
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        os.unlink(temporary)
    return target


def load_snapshot(path: str | Path) -> dict:
    """Read a candidate index only after checking content and row identities."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("DAILY_CANDIDATE_SNAPSHOT_SCHEMA_MISMATCH")
    snapshot_id = document.get("snapshot_id")
    payload = {key: value for key, value in document.items() if key != "snapshot_id"}
    if not isinstance(snapshot_id, str) or digest(payload) != snapshot_id:
        raise ValueError("DAILY_CANDIDATE_SNAPSHOT_HASH_MISMATCH")
    candidates = document.get("candidates")
    if (not isinstance(candidates, list) or document.get("candidate_count") != len(candidates)
            or any(not isinstance(row, dict) or not isinstance(row.get("signal_date"), str)
                   or not isinstance(row.get("symbol"), str) for row in candidates)):
        raise ValueError("DAILY_CANDIDATE_SNAPSHOT_MANIFEST_MISMATCH")
    if document.get("signal_dates") != sorted({row["signal_date"] for row in candidates}):
        raise ValueError("DAILY_CANDIDATE_SNAPSHOT_MANIFEST_MISMATCH")
    identities = set()
    for row in candidates:
        key = (row.get("signal_date"), row.get("symbol"))
        if key in identities:
            raise ValueError("DUPLICATE_DAILY_CANDIDATE_IDENTITY")
        identities.add(key)
        erem = row.get("erem")
        ref = row.get("report_reference")
        expected_identity = {name: row.get(name) for name in (
            "market", "symbol", "signal_date", "cutoff_at", "parameter_version", "report_sha256", "erem_sha256")}
        if (not isinstance(erem, dict) or not REQUIRED_EREM_FIELDS <= set(erem)
                or digest(erem) != row.get("erem_sha256")
                or not isinstance(ref, dict) or ref.get("report_sha256") != row.get("report_sha256")
                or digest(expected_identity) != row.get("candidate_id")):
            raise ValueError("DAILY_CANDIDATE_ROW_IDENTITY_MISMATCH")
    return document
