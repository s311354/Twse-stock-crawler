"""SYNTHETIC isolation/integrity tests, not market evidence or profit tests."""
from copy import deepcopy
import json

import pytest

from application.replay_contracts import digest
from infrastructure import les_v7_fr2_accounts as accounts


def request():
    initial = dict(currency="TWD", cash=1000000, account_equity=1000000,
        holdings=[], source="USER_EXPLICIT_RESEARCH_CONFIGURATION", approved_at="2026-10-18T12:00:00+08:00")
    arms = {arm: dict(model_id=digest(arm[:2]), entry_policy_id=digest(arm[-2:]),
        execution_policy_id=digest("SYNTHETIC_EXECUTION")) for arm in accounts.ARMS}
    sessions = ["2026-09-08", "2026-09-09", "2026-09-10"]
    options = dict(session_blockers={day: ["SYNTHETIC_MISSING_HISTORICAL_SOURCE"] for day in sessions},
        calendar_evidence_ids=[digest("SYNTHETIC_CALENDAR")])
    return initial, arms, sessions, digest("SYNTHETIC_EVIDENCE_AUDIT"), options


def create(root, values=None):
    initial, arms, sessions, audit, options = values or request()
    return accounts.create_blocked_account_paths(root, initial, arms, sessions, audit, **options)


def test_four_independent_accounts_share_only_initial_business_identity(tmp_path):
    before = request()
    source = deepcopy(before)
    result = create(tmp_path / "research", before)
    assert before == source
    assert result["status"] == accounts.STATUS
    assert result["comparison_status"] == "NOT_ASSESSABLE"
    assert result["efficacy_simulation_completed"] is False
    assert len({row["account_id"] for row in result["accounts"].values()}) == 4
    assert len({row["poststate_id"] for row in result["accounts"].values()}) == 4
    for arm in accounts.ARMS:
        path = json.loads((tmp_path / "research" / arm / "account-path.json").read_text())["payload"]
        assert path["initial_account_id"] == result["initial_account_id"]
        assert path["net_equity"] == ["1000000.00"] * 4
        assert not path["completed_trades"] and not path["execution_evidence_ids"]
        assert path["native_pae_observations"] == 0
        initial = json.loads((tmp_path / "research" / arm / "initial-state.json").read_text())
        prior_id = initial["sha256"]
        for day in source[2]:
            state = json.loads((tmp_path / "research" / arm / "sessions" / (day + ".json")).read_text())
            payload = state["payload"]
            assert payload["predecessor_state_id"] == prior_id
            assert digest(payload) == state["sha256"]
            assert payload["account"]["cash"] == payload["account"]["net_equity"]
            assert payload["account"]["positions"] == []
            assert payload["model_evaluated"] is payload["pae_evaluated"] is False
            assert payload["model_decision"] is None and payload["fills"] == []
            prior_id = state["sha256"]


def test_exact_retry_does_not_change_files_and_changed_request_is_rejected(tmp_path):
    root = tmp_path / "research"
    original = create(root)
    before = {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*.json")}
    assert create(root) == original
    assert before == {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*.json")}
    values = request()
    values[-1]["session_blockers"][values[2][0]] = ["OTHER_SYNTHETIC_GAP"]
    with pytest.raises(ValueError, match="REQUEST_CHANGED"):
        create(root, values)


@pytest.mark.parametrize("mutation", ["cash", "predecessor", "arm", "hash", "completion", "missing"])
def test_tampered_state_or_receipt_cannot_be_accepted(tmp_path, mutation):
    root = tmp_path / "research"
    create(root)
    path = root / "M0_P0" / "sessions" / "2026-09-09.json"
    if mutation == "completion":
        path = root / "completion.json"
    packet = json.loads(path.read_text())
    if mutation == "cash": packet["payload"]["account"]["cash"] = "999999.00"
    elif mutation == "predecessor": packet["payload"]["predecessor_state_id"] = "a" * 64
    elif mutation == "arm": packet["payload"]["arm"] = "M1_P1"
    elif mutation == "hash": packet["sha256"] = "b" * 64
    elif mutation == "completion": packet["payload"]["comparison_status"] = "PASS"
    if mutation == "missing": path.unlink()
    else: path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="MISSING_OR_DAMAGED|STATE_CHAIN_MISMATCH"):
        accounts.verify_blocked_account_paths(root)
    with pytest.raises(ValueError, match="MISSING_OR_DAMAGED|STATE_CHAIN_MISMATCH"):
        create(root)


@pytest.mark.parametrize("failure_point", ["middle", "completion"])
def test_interruption_before_completion_is_not_complete_and_retry_resumes(tmp_path, monkeypatch, failure_point):
    root = tmp_path / "research"
    original = accounts.write_once
    calls = 0
    def fail(path, value):
        nonlocal calls
        calls += 1
        if (failure_point == "completion" and path.name == "completion.json"
                or failure_point == "middle" and calls == 5):
            raise RuntimeError("SYNTHETIC_INTERRUPTION")
        return original(path, value)
    monkeypatch.setattr(accounts, "write_once", fail)
    with pytest.raises(RuntimeError, match="INTERRUPTION"):
        create(root)
    assert not (root / "completion.json").exists()
    with pytest.raises(ValueError, match="MISSING_OR_DAMAGED"):
        accounts.verify_blocked_account_paths(root)
    monkeypatch.setattr(accounts, "write_once", original)
    complete = create(root)
    assert accounts.verify_blocked_account_paths(root) == complete


@pytest.mark.parametrize("change", ["balance", "cash_nav", "holdings", "unknown_initial", "duplicate_sessions",
    "missing_blocker", "empty_blocker", "same_model", "wrong_policy", "different_execution", "audit", "calendar"])
def test_invalid_or_unapproved_inputs_fail_before_publication(tmp_path, change):
    values = request()
    initial, arms, sessions, _, options = values
    if change == "balance": initial["cash"] = initial["account_equity"] = 2000000
    elif change == "cash_nav": initial["cash"] = 900000
    elif change == "holdings": initial["holdings"] = [dict(symbol="SYNTHETIC", shares=1)]
    elif change == "unknown_initial": initial.pop("approved_at")
    elif change == "duplicate_sessions": sessions.append(sessions[-1])
    elif change == "missing_blocker": options["session_blockers"].pop(sessions[0])
    elif change == "empty_blocker": options["session_blockers"][sessions[0]] = []
    elif change == "same_model": arms["M0_P0"]["model_id"] = arms["M1_P0"]["model_id"]
    elif change == "wrong_policy": arms["M0_P0"]["entry_policy_id"] = digest("MISMATCH")
    elif change == "different_execution": arms["M0_P0"]["execution_policy_id"] = digest("MISMATCH")
    elif change == "audit": values = (*values[:3], "invalid", options)
    elif change == "calendar": options["calendar_evidence_ids"] = []
    with pytest.raises((ValueError, KeyError)):
        create(tmp_path / "research", values)
    assert not (tmp_path / "research").exists()


def test_never_adopts_existing_account_directory_or_symlink(tmp_path):
    existing = tmp_path / "formal-account"
    existing.mkdir()
    marker = existing / "unrelated.sqlite3"
    marker.write_bytes(b"SYNTHETIC_NOT_A_REAL_ACCOUNT")
    with pytest.raises(ValueError, match="NEW_ISOLATED_DIRECTORY"):
        create(existing)
    assert marker.read_bytes() == b"SYNTHETIC_NOT_A_REAL_ACCOUNT"
    alias = tmp_path / "alias"
    alias.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="SYMLINK"):
        create(alias)


def test_independent_second_archive_does_not_share_mutable_state(tmp_path):
    first = create(tmp_path / "a")
    second = create(tmp_path / "b")
    assert first["completion_id"] == second["completion_id"]
    first_state = tmp_path / "a" / "M0_P0" / "sessions" / "2026-09-08.json"
    second_state = tmp_path / "b" / "M0_P0" / "sessions" / "2026-09-08.json"
    assert first_state.stat().st_ino != second_state.stat().st_ino
    first_state.write_text("SYNTHETIC_CORRUPTION")
    assert accounts.verify_blocked_account_paths(tmp_path / "b") == second


def test_concurrent_exact_publication_is_idempotent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    root = tmp_path / "research"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: create(root), range(2)))
    assert results[0] == results[1] == accounts.verify_blocked_account_paths(root)


def test_resealed_modified_balance_is_not_a_valid_state(tmp_path):
    root = tmp_path / "research"
    create(root)
    path = root / "M0_P0" / "sessions" / "2026-09-08.json"
    packet = json.loads(path.read_text())
    packet["payload"]["account"]["net_equity"] = "1000001.00"
    packet["sha256"] = digest(packet["payload"])
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="STATE_CHAIN_MISMATCH"):
        accounts.verify_blocked_account_paths(root)
