"""SYNTHETIC ONLY: candidate indexing, not fill or strategy efficacy."""
import json

import pytest

from application.replay_contracts import digest
from infrastructure.storage.daily_candidate_archive import build_candidate, load_snapshot, publish_snapshot


def report():
    return {
        "symbol": "SYNTHETIC_ONLY",
        "signal_date": "2026-10-20",
        "parameter_version": "SYNTHETIC_R19",
        "envelope": {
            "event": {"cutoff_at": "2026-10-20T13:30:00+08:00"},
            "strategy": {"strategy_decision": "BUY_SMALL", "reason_codes": []},
            "score": {"final_score": 66.},
            "quality": {"status": "PASS"},
        },
        "diagnostics": {
            "gate_bindings": {"DATA_INTEGRITY": {"market": "TWSE"}},
            "quality": {"status": "PASS", "quality_score": 82.},
            "gate_receipts": [{"gate_id": "TRADING_STATUS", "state": "PASS"}],
            "factor_ledger": {"six_factor_total": 66.},
            "rule_baseline": {"policy": {"policy_version": "R19"}},
            "r19_route_evaluation": {"position_limits_affect_score": False},
            "reference_plan": {"band": [10., 10.5], "entry": 10.5, "stop": 9.5, "target": 12.},
            "structure_plan": {"conditional_entry": {"state": "UNFILLED"}},
            "execution_assessment": {"status": "PASS"},
        },
        "pae_authorization": {"authorized": True, "live_order_authorized": False,
                              "indicative_shares": 10, "reason_codes": []},
    }


def test_candidate_contains_r19_plan_and_does_not_invent_fill(tmp_path):
    native = report()
    reference = {"report_sha256": digest(native), "root": ["ref", "synthetic"]}
    candidate = build_candidate(native, reference, {
        "EREM 分數": 73., "EREM 決策訊號": "HOLD", "EREM 理由": "HOLD"})
    assert candidate["candidate_state"] == "AWAITING_EXTERNAL_FILL_OBSERVATION"
    assert candidate["six_factor_ledger"] == native["diagnostics"]["factor_ledger"]
    assert candidate["r19"]["plan"]["band"] == [10., 10.5]
    assert candidate["r19"]["structure_plan"]["conditional_entry"]["state"] == "UNFILLED"
    assert candidate["erem"] == {
        "EREM 分數": 73., "EREM 決策訊號": "HOLD", "EREM 理由": "HOLD"}
    assert candidate["fill_observation"] == {
        "status": "NOT_CAPTURED_BY_CRAWLER", "owner": "BankEmulator", "join_key": candidate["candidate_id"]}
    first = publish_snapshot(tmp_path, "first", [candidate])
    second = publish_snapshot(tmp_path, "second", [candidate])
    assert json.loads(first.read_text()) == json.loads(second.read_text())
    assert json.loads(first.read_text())["snapshot_id"] == first.stem
    assert first.read_bytes() == second.read_bytes()
    assert load_snapshot(first) == json.loads(first.read_text())
    tampered = json.loads(first.read_text())
    tampered["candidates"][0]["erem"]["EREM 分數"] = 0.
    damaged = tmp_path / "damaged.json"
    damaged.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="DAILY_CANDIDATE_SNAPSHOT_HASH_MISMATCH"):
        load_snapshot(damaged)


def test_candidate_snapshot_rejects_duplicate_security_day(tmp_path):
    candidate = build_candidate(report(), {"report_sha256": "a" * 64}, {
        "EREM 分數": None, "EREM 決策訊號": "EXIT", "EREM 理由": "SYNTHETIC_ONLY"})
    with pytest.raises(ValueError, match="DUPLICATE_DAILY_CANDIDATE_IDENTITY"):
        publish_snapshot(tmp_path, "duplicate", [candidate, candidate])
