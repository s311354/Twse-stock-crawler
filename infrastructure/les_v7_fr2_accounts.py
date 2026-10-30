"""Write-once FR2 account scaffolds for explicitly blocked research sessions.

This is not a simulator or a PAE implementation. It records the approved four
independent cash-only accounts while required historical evidence is absent.
No forecasts, decisions, fills, observations, costs or returns are invented.
Caller-supplied audit references are integrity bindings, not source certification.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
import fcntl
import json
from pathlib import Path

from application.replay_contracts import aware_timestamp, canonical_json, digest
from infrastructure.research_archive import write_once

ARMS = ("M0_P0", "M0_P1", "M1_P0", "M1_P1")
SCHEMA = "les-v7-fr2-blocked-accounts/v1"
STATUS = "BLOCKED_CASH_ONLY_PATH_INITIALIZED"


def _identifier(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("FR2_ACCOUNT_INVALID_" + name)
    return value


def _ids(values, name):
    if (not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))):
        raise ValueError("FR2_ACCOUNT_INVALID_" + name)
    return sorted(_identifier(value, name) for value in values)


def _money(value):
    if isinstance(value, bool):
        raise ValueError("FR2_ACCOUNT_EXPLICIT_APPROVED_BALANCE_REQUIRED")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("FR2_ACCOUNT_EXPLICIT_APPROVED_BALANCE_REQUIRED") from None
    if not amount.is_finite() or amount != Decimal("1000000"):
        raise ValueError("FR2_ACCOUNT_EXPLICIT_APPROVED_BALANCE_REQUIRED")
    return format(amount.quantize(Decimal(".01")), "f")


def _request(initial_configuration, arms, sessions, evidence_audit_id,
             session_blockers, calendar_evidence_ids):
    """Normalize only the explicit approved initial configuration, not holdings."""
    if not isinstance(initial_configuration, dict):
        raise ValueError("FR2_ACCOUNT_EXPLICIT_EMPTY_RESEARCH_CONFIGURATION_REQUIRED")
    configuration = json.loads(canonical_json(initial_configuration))
    if (configuration.get("currency") != "TWD" or configuration.get("holdings") != []
            or configuration.get("source") != "USER_EXPLICIT_RESEARCH_CONFIGURATION"):
        raise ValueError("FR2_ACCOUNT_EXPLICIT_EMPTY_RESEARCH_CONFIGURATION_REQUIRED")
    allowed = {"currency", "cash", "account_equity", "holdings", "source", "approved_at", "schema", "purpose"}
    if set(configuration) - allowed:
        raise ValueError("FR2_ACCOUNT_UNSUPPORTED_INITIAL_CONFIGURATION")
    if (configuration.get("schema", "les-v7-fr2-initial-account/v1") != "les-v7-fr2-initial-account/v1"
            or configuration.get("purpose", "FOUR_NEW_ISOLATED_RESEARCH_ACCOUNTS_NOT_FORMAL_ACCOUNTS")
                != "FOUR_NEW_ISOLATED_RESEARCH_ACCOUNTS_NOT_FORMAL_ACCOUNTS"):
        raise ValueError("FR2_ACCOUNT_UNSUPPORTED_INITIAL_CONFIGURATION")
    initial = dict(schema="les-v7-fr2-initial-account/v1", currency="TWD",
        cash=_money(configuration.get("cash")), account_equity=_money(configuration.get("account_equity")),
        holdings=[], source=configuration["source"],
        approved_at=aware_timestamp(configuration["approved_at"]).isoformat(),
        purpose="FOUR_NEW_ISOLATED_RESEARCH_ACCOUNTS_NOT_FORMAL_ACCOUNTS")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("FR2_ACCOUNT_EXACT_FOUR_ARMS_REQUIRED")
    bindings = {}
    for name in ARMS:
        arm = arms[name]
        keys = {"model_id", "entry_policy_id", "execution_policy_id"}
        if not isinstance(arm, dict) or set(arm) != keys:
            raise ValueError("FR2_ACCOUNT_ARM_BINDINGS_REQUIRED")
        bindings[name] = {key: _identifier(arm[key], key.upper()) for key in sorted(keys)}
    if (bindings["M0_P0"]["model_id"] != bindings["M0_P1"]["model_id"]
            or bindings["M1_P0"]["model_id"] != bindings["M1_P1"]["model_id"]
            or bindings["M0_P0"]["model_id"] == bindings["M1_P0"]["model_id"]
            or bindings["M0_P0"]["entry_policy_id"] != bindings["M1_P0"]["entry_policy_id"]
            or bindings["M0_P1"]["entry_policy_id"] != bindings["M1_P1"]["entry_policy_id"]
            or bindings["M0_P0"]["entry_policy_id"] == bindings["M0_P1"]["entry_policy_id"]
            or len({arm["execution_policy_id"] for arm in bindings.values()}) != 1):
        raise ValueError("FR2_ACCOUNT_FOUR_ARM_BINDING_MISMATCH")
    if (not isinstance(sessions, list) or not 1 <= len(sessions) <= 1092
            or any(not isinstance(day, str) for day in sessions)
            or sessions != sorted(set(sessions))
            or any(date.fromisoformat(day).isoformat() != day for day in sessions)):
        raise ValueError("FR2_ACCOUNT_SORTED_OFFICIAL_SESSIONS_REQUIRED")
    if not isinstance(session_blockers, dict) or set(session_blockers) != set(sessions):
        raise ValueError("FR2_ACCOUNT_EACH_SESSION_BLOCKER_REQUIRED")
    blockers = {}
    for day in sessions:
        reasons = session_blockers[day]
        if (not isinstance(reasons, list) or not 1 <= len(reasons) <= 128
                or any(not isinstance(reason, str) or not 1 <= len(reason) <= 1024
                    or reason.strip() != reason for reason in reasons)
                or len(reasons) != len(set(reasons))):
            raise ValueError("FR2_ACCOUNT_EXACT_NONEMPTY_BLOCK_REASONS_REQUIRED")
        blockers[day] = sorted(reasons)
    return dict(schema=SCHEMA, initial_configuration=initial, arms=bindings, sessions=list(sessions),
        evidence_audit_id=_identifier(evidence_audit_id, "EVIDENCE_AUDIT_ID"),
        calendar_evidence_ids=_ids(calendar_evidence_ids, "CALENDAR_EVIDENCE_ID"),
        session_blockers=blockers, session_scope="CALLER_PINNED_OFFICIAL_CALENDAR_AUDIT_REQUIRED",
        research_only=True, approved_for_live=False, comparison_status="NOT_ASSESSABLE")


def _seal(payload):
    return dict(payload=payload, sha256=digest(payload))


def _artifacts(request):
    request_id = digest(request)
    initial = request["initial_configuration"]
    initial_id = digest(initial)
    records = {"request.json": _seal(request), "initial-account.json": _seal(initial)}
    accounts = {}
    for name in ARMS:
        account_id = digest(dict(schema="les-v7-fr2-isolated-account/v1", arm=name,
            initial_account_id=initial_id, bindings=request["arms"][name], request_id=request_id))
        common = dict(account_id=account_id, arm=name, initial_account_id=initial_id,
            bindings=request["arms"][name], request_id=request_id, research_only=True,
            formal_account_modified=False)
        account = dict(currency="TWD", cash=initial["cash"], net_equity=initial["account_equity"],
            positions=[], cumulative_fees="0.00", realized_net_pnl="0.00")
        # Zero incurred costs follow from zero fills, not an assumed fee rate.
        prior = dict(schema="les-v7-fr2-account-state/v1", **common, session=None,
            predecessor_state_id=None, account=account, cumulative_fills=0,
            completed_trades=0, status="INITIALIZED_EXPLICIT_EMPTY_ACCOUNT")
        prior_id = digest(prior)
        records[f"{name}/initial-state.json"] = _seal(prior)
        state_ids = []
        for day in request["sessions"]:
            state = dict(schema="les-v7-fr2-account-state/v1", **common, session=day,
                predecessor_state_id=prior_id, account=account, cumulative_fills=0,
                completed_trades=0, status=STATUS, comparison_status="NOT_ASSESSABLE",
                evidence_audit_id=request["evidence_audit_id"],
                calendar_evidence_ids=request["calendar_evidence_ids"],
                blocking_reasons=request["session_blockers"][day],
                action="PRESERVE_CASH_NO_EXECUTION", model_evaluated=False, pae_evaluated=False,
                model_decision=None, pae_observations=[], fills=[], external_cashflows=[])
            prior_id = digest(state)
            state_ids.append(prior_id)
            records[f"{name}/sessions/{day}.json"] = _seal(state)
        path = dict(schema="les-v7-fr2-blocked-account-path/v1", **common, status=STATUS,
            comparison_status="NOT_ASSESSABLE", evidence_audit_id=request["evidence_audit_id"],
            calendar_evidence_ids=request["calendar_evidence_ids"], sessions=request["sessions"],
            state_ids=state_ids, poststate_id=prior_id, net_equity=[initial["account_equity"]] * (len(state_ids) + 1),
            completed_trades=[], execution_evidence_ids=[], net_of_incurred_costs=True,
            costs_scope="NO_FILLS_NO_COSTS_INCURRED_NOT_A_COST_POLICY_CERTIFICATION",
            efficacy_simulation_completed=False, native_pae_observations=0)
        records[f"{name}/account-path.json"] = _seal(path)
        accounts[name] = dict(account_id=account_id, path_id=digest(path), poststate_id=prior_id,
            path=f"{name}/account-path.json", status=STATUS, comparison_status="NOT_ASSESSABLE")
    completion = dict(schema="les-v7-fr2-blocked-accounts-completion/v1", request_id=request_id,
        initial_account_id=initial_id, status=STATUS, comparison_status="NOT_ASSESSABLE",
        efficacy_simulation_completed=False, evidence_audit_id=request["evidence_audit_id"],
        accounts=accounts, artifacts={path: digest(value) for path, value in sorted(records.items())},
        research_only=True, approved_for_live=False, execution_enabled=False,
        source_authenticity="CALLER_BOUND_AUDIT_REFERENCE_NOT_SOURCE_CERTIFICATION")
    return records, _seal(completion)


def _safe_path(root, relative):
    path = root / relative
    if root.is_symlink():
        raise ValueError("FR2_ACCOUNT_SYMLINK_TARGET_FORBIDDEN")
    cursor = root
    for part in Path(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("FR2_ACCOUNT_SYMLINK_TARGET_FORBIDDEN")
    return path


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("FR2_ACCOUNT_ARTIFACT_MISSING_OR_DAMAGED:" + path.name) from error


def verify_blocked_account_paths(output_root):
    """Revalidate the pinned request and every account/state/link before use."""
    root = Path(output_root)
    packet = _read(_safe_path(root, "request.json"))
    if not isinstance(packet, dict) or not isinstance(packet.get("payload"), dict):
        raise ValueError("FR2_ACCOUNT_REQUEST_BINDING_MISMATCH")
    request = packet.get("payload", {})
    if packet.get("sha256") != digest(request):
        raise ValueError("FR2_ACCOUNT_REQUEST_HASH_MISMATCH")
    try:
        normalized = _request(request["initial_configuration"], request["arms"], request["sessions"],
            request["evidence_audit_id"], request["session_blockers"], request["calendar_evidence_ids"])
    except (KeyError, TypeError) as error:
        raise ValueError("FR2_ACCOUNT_REQUEST_BINDING_MISMATCH") from error
    if normalized != request:
        raise ValueError("FR2_ACCOUNT_REQUEST_BINDING_MISMATCH")
    expected, completion = _artifacts(request)
    for relative, value in (*expected.items(), ("completion.json", completion)):
        if _read(_safe_path(root, relative)) != value:
            raise ValueError("FR2_ACCOUNT_ARTIFACT_OR_STATE_CHAIN_MISMATCH:" + relative)
    return dict(completion["payload"], completion_id=completion["sha256"], folder=str(root.resolve()))


def create_blocked_account_paths(output_root, initial_configuration, arms, sessions,
                                evidence_audit_id, *, session_blockers, calendar_evidence_ids):
    """Publish four independent blocked paths, never a simulated-profit result.

    ``output_root`` is one new isolated research directory, not an existing Bank
    account/run directory. The first request pins it permanently. Exact retries
    resume incomplete publication; changed inputs or damaged bytes never overwrite.
    The evidence audit and calendar IDs must refer to separately retained verified
    artifacts; this function does not assert their external authenticity.
    """
    request = _request(initial_configuration, arms, sessions, evidence_audit_id,
        session_blockers, calendar_evidence_ids)
    root = Path(output_root)
    existing = _safe_path(root, "request.json")
    if root.exists() and not root.is_dir():
        raise ValueError("FR2_ACCOUNT_ISOLATED_DIRECTORY_REQUIRED")
    # This lock belongs only to the requested isolated output, never to Bank or
    # formal PAE state. Append mode preserves metadata on exact retry.
    lock_name = ".publication.lock"
    if (root.exists() and not existing.exists() and not (root / lock_name).exists()
            and any(path.name != lock_name for path in root.iterdir())):
        raise ValueError("FR2_ACCOUNT_NEW_ISOLATED_DIRECTORY_REQUIRED")
    root.mkdir(parents=True, exist_ok=True)
    with _safe_path(root, lock_name).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if existing.exists() and _read(existing) != _seal(request):
            raise ValueError("FR2_ACCOUNT_REQUEST_CHANGED")
        if _safe_path(root, "completion.json").exists():
            return verify_blocked_account_paths(root)
        if not existing.exists() and any(path.name != lock_name for path in root.iterdir()):
            raise ValueError("FR2_ACCOUNT_NEW_ISOLATED_DIRECTORY_REQUIRED")
        records, completion = _artifacts(request)
        for relative, value in records.items():
            write_once(_safe_path(root, relative), value)
        # No reader may treat an interrupted sequence as completed before this file.
        write_once(_safe_path(root, "completion.json"), completion)
        return verify_blocked_account_paths(root)
