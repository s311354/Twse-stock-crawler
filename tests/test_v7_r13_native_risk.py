"""SYNTHETIC ONLY: native R13 risk receipts and counterfactual entry guards."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from domain.les_v7.contracts import GateReceipt, ReferencePlan, identity, timestamp
from domain.les_v7.decision import vetoes
from domain.les_v7.native_evidence import (REQUIRED_GATES, R13_STRESS_PRODUCER_VERSION,
    SOURCE_SCHEMA, produce_native_receipts)
from domain.les_v7.plan_r10 import policy, validate_band
from domain.les_v7.policy_r13 import AVAILABLE_AT, PLAN_VERSION
from domain.les_v7.reference_path import evaluate_reference_path
from tests.strategies.test_v7_plan_r13 import CUTOFF, build, declared_r13, source_bars
from tests.strategies.test_v7_plan_r10 import declared_policy
from tests.test_v7_reference_path import fixture


def native_inputs(*, context="REVERSAL", p=None, target=127.):
    reference = p or declared_r13()
    data = source_bars(target=target)
    plan, _ = build(context=context, data=data, p=reference)
    rows = data.to_dict("records")
    base = dict(symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=timestamp(CUTOFF).isoformat(),
        bars_hash=identity(rows), financials_hash=identity([]))
    features = dict(close=110., atr=4., spread=.003, adv=100_000_000.)
    quality = dict(core_inputs_pass=False)
    bindings = {gate: base for gate in REQUIRED_GATES}
    bindings["STRESS_RISK"] = dict(base, plan=asdict(plan))
    packet = dict(schema=SOURCE_SCHEMA, symbol=base["symbol"], market="TWSE", cutoff=base["cutoff"],
        expected_session="2026-10-16", historical_replay=False, market_rows=rows,
        market_evidence=dict(symbol=base["symbol"], market="TWSE", session_date="2026-10-16", source="TWSE",
            source_available_at=base["cutoff"], source_final=True, price_unit="TWD", volume_unit="shares",
            amount_unit="TWD", bars_sha256=identity(rows), source_record_ids=[identity(rows)],
            evidence_origin="SYNTHETIC_ONLY_NOT_OFFICIAL_CAPTURE"), restriction_evidence={})
    return dict(symbol=base["symbol"], market="TWSE", cutoff=base["cutoff"], bars=data,
        features=features, quality=quality, plan=asdict(plan), risks=vetoes(features), bindings=bindings,
        source_packet=packet, reference_policy=reference)


def stress(item):
    receipt = produce_native_receipts(**item)["STRESS_RISK"]
    return receipt, GateReceipt.bind(receipt, gate_id="STRESS_RISK",
        inputs=item["bindings"]["STRESS_RISK"], cutoff=item["cutoff"])


@pytest.mark.parametrize("context", ["REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION"])
def test_r13_native_receipt_uses_bound_rules_and_exact_whole_band_boundary(context):
    item = native_inputs(context=context)
    before = deepcopy(item)
    receipt, bound = stress(item)
    assert bound.state == "PASS"
    assert receipt["gate_version"] == R13_STRESS_PRODUCER_VERSION
    plan = ReferencePlan(**dict(item["plan"], band=tuple(item["plan"]["band"])))
    assert validate_band(plan, item["reference_policy"])["stop_atr"] == 1.8
    # Full-band float arithmetic would read 1.8000000000000007; no epsilon or
    # falsely stricter legacy 1.5ATR check is used by the R13 receipt producer.
    assert (111. - plan.stop) / plan.atr > 1.8
    source = dict(market_source=item["source_packet"]["market_evidence"], plan=item["plan"],
        spread=.003, adv=100_000_000., reference_policy_id=identity(item["reference_policy"]),
        structural_rules=item["reference_policy"]["structural_rules"])
    assert receipt["input_ids"] == [identity(source)]
    assert item["features"] == before["features"] and item["source_packet"] == before["source_packet"]
    assert item["plan"] == before["plan"] and item["reference_policy"] == before["reference_policy"]
    assert stress(native_inputs(context=context, p=declared_policy()))[1].state == "FAIL"


@pytest.mark.parametrize("key,value,state", [("spread", None, "UNKNOWN"), ("adv", None, "UNKNOWN"),
    ("spread", .006, "FAIL"), ("adv", 1., "FAIL")])
def test_r13_does_not_relax_missing_liquidity_or_known_spread_and_capacity_failures(key, value, state):
    item = native_inputs()
    item["features"][key] = value
    assert stress(item)[1].state == state


def test_r13_future_market_missing_resistance_and_tampered_rules_never_pass():
    item = native_inputs()
    item["source_packet"]["market_evidence"]["source_available_at"] = "2030-01-01T00:00:00Z"
    assert stress(item)[1].state == "UNKNOWN"
    assert stress(native_inputs(target=111.))[1].state == "UNKNOWN"
    item = native_inputs()
    item["reference_policy"]["structural_rules"]["maximum_stop_atr"] = 99.
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        stress(item)


def reference_fixture(*, version=PLAN_VERSION, stop=97.4):
    old, window, path, options = fixture()
    path["available_at"] = AVAILABLE_AT
    bound = policy(dict(policy_id="REFERENCE_H20", cost_path_policy=path), available_at=AVAILABLE_AT,
        version=version)
    plan = ReferencePlan("REFERENCE_H20", version, old.evidence_id, "REVERSAL",
        100., stop, 110., 2., (100., 101.), old.reference_shares, .02, .02)
    window[1].update(open=101., high=101.5, close=101.)
    return plan, window, bound["cost_path_policy"], options


def test_reference_path_reconstructs_r13_not_stricter_legacy_policy_and_reprices_real_open():
    args = reference_fixture()
    before = deepcopy(args)
    result = evaluate_reference_path(args[0], args[1], args[2], **args[3])
    assert result["status"] == "VERIFIED_FILLED"
    assert result["execution"]["policy_version"] == PLAN_VERSION
    assert result["execution"]["entry_risk_check"]["maximum_stop_atr"] == 1.8
    assert result["execution"]["entry_risk_check"]["minimum_net_rr"] == 1.8
    assert result["execution"]["entry_risk_check"]["entry"] == 101.
    assert result["execution"]["historical_deployment_claim"] is False
    assert args == before
    old = reference_fixture(version="V7_SETUP_STRUCTURE_H20_R10")
    assert evaluate_reference_path(old[0], old[1], old[2], **old[3])["execution"]["no_fill_reason"] == "R10_ACTUAL_ENTRY_RISK_NOT_PASS"


def test_reference_path_overwide_entry_and_missing_guard_do_not_fill():
    args = reference_fixture(stop=97.1)
    result = evaluate_reference_path(args[0], args[1], args[2], **args[3])
    assert result["status"] == "VERIFIED_NO_FILL"
    assert result["execution"]["no_fill_reason"] == "R13_ACTUAL_ENTRY_RISK_NOT_PASS"
    assert "FIXED_STRUCTURE_STOP_ATR_EXCEEDED" in result["execution"]["entry_risk_check"]["reasons"]
    args[2].pop("entry_guard_policy")
    result = evaluate_reference_path(args[0], args[1], args[2], **args[3])
    assert result["status"] != "VERIFIED_FILLED"
    assert "R13_ACTUAL_ENTRY_GUARD_MISSING" in result["diagnostic"]["reason_codes"]
