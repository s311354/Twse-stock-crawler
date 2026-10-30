"""SYNTHETIC ONLY: uncapped scores retain non-compensable entry safety."""
from copy import deepcopy
from datetime import timedelta

import pytest

from application.les_v7 import prepare_quality_context
from application.low_entry_v7_daily import daily_report, project_report
from application.market_history import MarketHistoryAssembler
from infrastructure.crawler.security_master import select_security_master
from tests.test_security_master import packet as security_packet
from domain.les_v7.contracts import GateReceipt, ReferencePlan, identity, timestamp
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (PENALTY_BASELINE_POLICY, PENALTY_AVAILABLE_AT,
    SIGNED_BASELINE_POLICY, SIGNED_AVAILABLE_AT, baseline_policy, current_source_baseline_version,
    require_baseline_policy, score_baseline, select_baseline_context)
from domain.les_v7.rule_score_r8 import adjustment_policy, compose_penalty_score
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from infrastructure.report.low_entry_v7_renderer import score_text
from tests.strategies.test_v7_rule_baseline import unit_features, CUTOFF
from tests.test_v7_source_repair_daily import repaired_market


def outcome(*, states=None, quality="PASS", features=None, risks=(), plan_present=True,
            standard=True, starter_gate=True, confirmed_gate=True, version=PENALTY_BASELINE_POLICY):
    f = features or unit_features()
    prepared = dict(policy=baseline_policy(version=version), features=f,
                    selection=select_baseline_context(f, policy_version=version).record())
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL",
        101., 99., 109., 2., (101., 101.5), 1, .1, .1) if plan_present else None
    gates = [GateReceipt(g, "SYNTHETIC_ONLY", (states or {}).get(g, "PASS"), identity({}),
        timestamp(CUTOFF).isoformat(), (), ("SYNTHETIC_ONLY",), ("SYNTHETIC_ONLY",),
        timestamp(CUTOFF).isoformat()) for g in
        ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK")]
    return score_baseline(prepared, plan=plan, quality=dict(status=quality, quality_score=80.),
        gates=gates, native_features=dict(spread=.003, adv=50_000_000.), risks=list(risks),
        standard=standard, starter_gate=starter_gate, confirmed_gate=confirmed_gate)


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK"])
@pytest.mark.parametrize("state,decision", [("FAIL", "NO_BUY"), ("UNKNOWN", "DATA_INSUFFICIENT")])
def test_high_technical_score_does_not_override_failed_or_unknown_gate(gate, state, decision):
    result = outcome(states={gate: state})
    assert result["score"]["raw_score"] == result["score"]["final_score"] == 94.
    assert result["score"]["effective_cap"] is None and result["caps"] == {}
    assert result["decision"] == decision and gate + "_" + state in result["reasons"]


@pytest.mark.parametrize("status,decision", [("REJECT", "NO_BUY"), ("DATA_INSUFFICIENT", "DATA_INSUFFICIENT")])
def test_quality_readiness_is_not_a_zero_technical_score(status, decision):
    result = outcome(quality=status)
    assert result["score"]["final_score"] == 94. and result["decision"] == decision


def test_execution_fail_and_missing_plan_still_cannot_buy():
    assert outcome(risks=["EXECUTION_FAIL"])["decision"] == "NO_BUY"
    missing = outcome(plan_present=False, standard=False, starter_gate=False, confirmed_gate=False)
    assert missing["score"]["final_score"] > 0 and missing["decision"] == "DATA_INSUFFICIENT"


def test_new_score_reuses_exact_old_factor_ledger_without_gate_caps():
    old = outcome(states={"VALUATION": "FAIL"}, version=SIGNED_BASELINE_POLICY)
    new = outcome(states={"VALUATION": "FAIL"})
    for key in ("score_evidence", "factor_ledger", "factors", "weights", "technical_vetoes"):
        assert new[key] == old[key]
    assert old["score"]["final_score"] == 0 and new["score"]["final_score"] == 94.
    assert old["decision"] == new["decision"] == "NO_BUY"


def test_missing_factors_earn_no_points_no_rescaling_or_estimated_odds():
    f = unit_features(); f.update(rsi_change=None, rsi14=None)
    result = outcome(features=f)
    assert result["score"]["raw_score"] == result["score"]["final_score"] == 89.
    assert result["score_evidence"]["total_weight"] == 100
    assert result["score_evidence"]["coverage_percent"] == 89.


def test_soft_penalties_are_explicit_additive_not_caps_and_do_not_duplicate_trigger_state():
    f = unit_features(); f.update(break_previous_high=False, clv=.5, market_regime="RISK_OFF")
    result = outcome(features=f)
    assert result["trigger"]["confirmed"] is False and result["trigger"]["starter"] is True
    assert result["penalties"] == {"RISK_OFF": 5., "BASELINE_NOT_CONFIRMED": 5.}
    s = result["score"]
    assert s["final_score"] == s["adjusted_score"] == max(0, s["raw_score"] - 10.)
    f.update(drawdown60=None, position60=None)
    unclear = outcome(features=f, plan_present=False, standard=False, starter_gate=False, confirmed_gate=False)
    assert unclear["penalties"] == {"RISK_OFF": 5., "BASELINE_NO_CLEAR_CONTEXT": 10.}
    assert unclear["decision"] == "WATCH_ONLY"


def test_penalty_validation_and_zero_floor_reuse_existing_budget_contract():
    ledger = outcome()["score_evidence"]
    assert compose_penalty_score(ledger, [dict(reason="SYNTHETIC_ONLY", points=100)])["final_score"] == 0
    with pytest.raises(ValueError, match="INVALID_SCORE_ADJUSTMENT"):
        compose_penalty_score(ledger, [dict(reason="SYNTHETIC_ONLY", points=float("nan"))])


def test_policy_is_frozen_cutoff_bound_and_does_not_change_old_r7_identity():
    assert identity(baseline_policy(version=SIGNED_BASELINE_POLICY)) == "ca665765f8b66ef8fb9888a6bcde69590461d5e425a7f232b571e0e26288564e"
    assert current_source_baseline_version(SIGNED_AVAILABLE_AT) == SIGNED_BASELINE_POLICY
    assert current_source_baseline_version(PENALTY_AVAILABLE_AT) == PENALTY_BASELINE_POLICY
    policy = baseline_policy(version=PENALTY_BASELINE_POLICY)
    assert policy["score_adjustment_policy"] == adjustment_policy()
    before = (timestamp(PENALTY_AVAILABLE_AT) - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(policy, market="TWSE", cutoff=before)
    policy["score_adjustment_policy"]["deductions"]["unconfirmed_starter"] = 0
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        require_baseline_policy(policy, market="TWSE", cutoff=PENALTY_AVAILABLE_AT)


def native_report():
    history = MarketHistoryAssembler().build(repaired_market(), 0)
    cutoff = (timestamp(PENALTY_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    history.attrs[SIGNAL_DATE_ATTR] = cutoff
    history.attrs["les_v7_native_sources"] = dict(history.attrs["les_v7_native_sources"],
        security=select_security_master(security_packet(), "2330", cutoff))
    return daily_report(history)


def test_current_daily_source_quality_projection_and_authorization_use_new_version():
    report = native_report(); validate_report(report)
    e, d = report["envelope"], report["diagnostics"]
    assert d["rule_baseline"]["policy"]["policy_version"] == PENALTY_BASELINE_POLICY
    assert d["quality"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R6"
    assert d["safety_caps"] == {} and e["score"]["effective_cap"] is None
    assert e["score"]["final_score"] == e["score"]["adjusted_score"]
    assert report["pae_authorization"]["authorized"] is False
    assert report["pae_authorization"]["indicative_shares"] == 0
    row = project_report(report)
    assert row["低點分數"] == e["score"]["final_score"]
    assert row["LES v7 Safety Cap"] is None and not row["LES v7 Execution Eligible"]
    assert "封頂" not in score_text(e["score"]) and "Final" in score_text(e["score"])


def test_rehashed_score_penalties_caps_and_decision_cannot_be_forged():
    original = native_report()
    for field in ("score", "penalties", "caps", "decision"):
        report = deepcopy(original)
        if field == "score": report["envelope"]["score"]["final_score"] = 100.
        elif field == "penalties": report["diagnostics"]["penalties"] = {"INVENTED": 0.}
        elif field == "caps": report["diagnostics"]["safety_caps"] = {"INVENTED": 100.}
        else: report["envelope"]["strategy"]["strategy_decision"] = "BUY"
        with pytest.raises(ValueError): validate_report(report)


def test_public_quality_preparation_uses_same_source_policy_without_external_default():
    context = dict(symbol="2330", market="TWSE", as_of=PENALTY_AVAILABLE_AT, filings=[],
                   rule_baseline=True, rule_baseline_policy_version=PENALTY_BASELINE_POLICY)
    prepare_quality_context(context)
    assert context["quality_policy"] == baseline_policy(version=PENALTY_BASELINE_POLICY)["quality_policy"]
