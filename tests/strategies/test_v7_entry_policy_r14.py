"""SYNTHETIC ONLY: exact R14 cap/admission rules, not source certification."""
from copy import deepcopy

import pytest

import domain.strategies  # noqa: F401 -- initialize the existing plugin graph
from domain.les_v7.allocation import _authorize_validated
from domain.les_v7.contracts import identity
from domain.les_v7.entry_policy_r14 import assess, valuation_grade
from domain.les_v7.policy_r14 import AVAILABLE_AT, BASELINE_VERSION
from domain.les_v7.valuation_recovery_r14 import (COMPONENT_SCHEMA,
    evaluate_annual_price_valuation, evaluate_recovery_price_valuation)
from tests.strategies.test_v7_entry_policy_r13 import gates, quality
from tests.test_v7_baseline_allocation import boundary_report, account


VARIANT = "EARLY_SHORT_BASE_BREAKOUT_R14"


def valuation(*, peer_only=False, gap=.20):
    args = dict(current_eps=10., prior_eps=10., current_revenue=100., prior_revenue=100.,
        current_margin=.2, prior_margin=.2, current_price=100., current_pe=20.,
        recent_pe_median=20., eps_basis="DILUTED")
    comparable = dict(schema=COMPONENT_SCHEMA, status="PARTIAL" if peer_only else "VERIFIED",
        industry="半導體業", reasons=[],
        issuer_history=dict(status="INSUFFICIENT" if peer_only else "VERIFIED",
            count=3 if peer_only else 24, minimum=24, median=20.),
        peers=dict(status="VERIFIED", count=10, minimum=10, median=20.))
    initial = evaluate_recovery_price_valuation(legacy_result=evaluate_annual_price_valuation(**args),
        comparables=comparable)
    per_eps = initial["scenarios"]["BEAR"]["fair_price"] / args["current_eps"]
    eps = args["current_price"] * (1 - gap) / per_eps
    args.update(current_eps=eps, prior_eps=eps)
    result = evaluate_recovery_price_valuation(legacy_result=evaluate_annual_price_valuation(**args),
        comparables=comparable)
    return dict(gate_id="VALUATION", state=result["status"], valuation_diagnostics=result,
        provenance="SYNTHETIC_ONLY")


def admission(*, peer_only=False, gap=.20, variant=VARIANT, score=80., states=None):
    receipt = valuation(peer_only=peer_only, gap=gap)
    return assess(quality(score), gates(receipt, **(states or {})), receipt, symbol="SYNTHETIC",
        cutoff=AVAILABLE_AT, current_price=100., setup_variant=variant)


@pytest.mark.parametrize("score", [65., 74.999, 75., 100.])
def test_short_base_is_small_only_and_caps_before_multipliers(score):
    result = admission(score=score)
    assert not result["buy_allowed"] and result["buy_total_cap"] == 0
    assert result["small_allowed"] and result["restricted_starter"]
    assert result["small_total_cap"] == .025
    assert result["risk_fraction"] == .00125


def test_peer_only_and_bear_warning_each_multiply_once_and_keep_native_fail():
    normal = admission(peer_only=True, variant=None)
    warning = admission(peer_only=True, gap=.4, variant=None)
    assert normal["small_total_cap"] == .0125
    assert warning["small_total_cap"] == .00625
    assert warning["valuation"]["native_state"] == "FAIL"
    assert warning["valuation"]["points"] == 0.
    assert not warning["buy_allowed"] and warning["small_allowed"]
    assert warning == admission(peer_only=True, gap=.4, variant=None)


def test_old_recognized_full_method_keeps_original_quality_cap():
    result = admission(variant=None)
    assert result["buy_allowed"] and result["buy_total_cap"] == .20
    assert result["small_total_cap"] == .05
    assert not result["restricted_starter"]


@pytest.mark.parametrize("score", [60., 64.999])
def test_new_routes_cannot_borrow_the_lower_existing_small_quality_floor(score):
    assert not admission(score=score)["small_allowed"]
    assert not admission(score=score, peer_only=True, variant=None)["small_allowed"]
    assert admission(score=score, variant=None)["small_allowed"]


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "SAFETY", "TRADING_STATUS", "STRESS_RISK"])
@pytest.mark.parametrize("state", ["UNKNOWN", "FAIL"])
def test_required_unknown_and_hard_failures_stay_non_compensable(gate, state):
    result = admission(states={gate: state}, score=100.)
    assert not result["buy_allowed"] and not result["small_allowed"]
    assert result["state"] == ("UNKNOWN" if state == "UNKNOWN" else "BLOCKED")


def test_tampered_method_limits_and_missing_calculation_are_not_trusted():
    receipt = valuation(peer_only=True)
    before = deepcopy(receipt)
    receipt["valuation_diagnostics"]["method_position_multiplier"] = 1.
    assert valuation_grade(receipt, current_price=100.)["state"] == "UNKNOWN"
    assert before != receipt
    assert valuation_grade(None, current_price=100.)["state"] == "UNKNOWN"
    with pytest.raises(ValueError, match="UNKNOWN_SETUP_VARIANT"):
        assess(quality(), gates(before), before, symbol="SYNTHETIC", cutoff=AVAILABLE_AT,
            current_price=100., setup_variant="UNKNOWN_VARIANT")


def test_total_exposure_and_risk_deduct_holdings_before_incremental_quota():
    # Boundary stub deliberately bypasses full report validation; public native
    # report + authorization revalidation is tested by test_v7_r14_pipeline.
    report = boundary_report("BUY_SMALL")
    report["envelope"]["event"]["security_id"] = "SYNTHETIC"
    from domain.les_v7.rule_baseline import baseline_policy
    report["diagnostics"]["rule_baseline"]["policy"] = baseline_policy(version=BASELINE_VERSION)
    report["diagnostics"]["reference_plan"].update(target=110., reference_shares=1)
    report["diagnostics"]["entry_admission"] = admission()
    empty = _authorize_validated(report, account())
    held_account = dict(account(), positions=[dict(symbol="SYNTHETIC", shares=250)])
    held = _authorize_validated(report, held_account)
    assert empty["maximum_total_weight"] == .025
    assert empty["small_risk_fraction"] == .00125
    from domain.les_v7.reference_path import estimate_reference_costs
    p = report["diagnostics"]["reference_plan"]
    costs = estimate_reference_costs(100.5, p["stop"], p["target"], 1,
        report["diagnostics"]["rule_baseline"]["policy"]["reference_policy"]["cost_path_policy"])
    assert empty["bound_plan_loss_per_share"] == pytest.approx(2.5 + costs["cost_stop_per_share"])
    assert empty["policy_version"] == "LES_V7_PAE_SIGNAL_AUTHORIZATION_R3"
    assert held["current_weight"] == .025 and held["incremental_weight_capacity"] == 0
    assert held["indicative_shares"] == 0 and not held["authorized"]
    assert report["diagnostics"]["entry_admission"]["assessment_id"] == identity({
        k: v for k, v in report["diagnostics"]["entry_admission"].items() if k != "assessment_id"})
