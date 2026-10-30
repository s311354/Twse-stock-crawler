"""SYNTHETIC ONLY: allocation-boundary stubs are NOT native research reports.

The sizing stubs start after report validation. The final public-entry cases
use complete synthetic kernel reports with explicitly synthetic upstream ports;
neither form certifies official evidence, real model quality or a live trade.
"""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR

import pytest

from domain.les_v7.allocation import POLICY, _authorize_validated, validate_authorization
from domain.les_v7.contracts import identity


CUTOFF = "2026-10-14T13:30:00+08:00"


def boundary_report(signal="BUY"):
    rule_id = identity(dict(synthetic_only=True, rule_policy="V6_HYBRID_BASELINE"))
    return dict(model_version="7.0.0", parameter_version="SYNTHETIC_ONLY_RULE_BASELINE:" + rule_id,
        envelope=dict(model=dict(model_status="RULE_BASED_BASELINE", bundle_sha256=None,
            trained_artifacts_status="NOT_APPLICABLE"),
            event=dict(cutoff_at=CUTOFF, signal_id=identity(dict(synthetic_only=True, signal=signal)),
                plan_id=identity(dict(synthetic_only=True, entry=100., stop=98.))),
            strategy=dict(strategy_decision=signal),
            localization=dict(status="NOT_APPLICABLE", bottom_price_quantiles=None, timing_pmf=None,
                p_fill=None, p_rebound_given_fill=None, p_entry_given_fill=None, p_bottom_soon=None),
            execution=dict(live_enabled=False)),
        diagnostics=dict(rule_baseline=dict(policy_id=rule_id),
            features=dict(adv=100_000_000., spread=.001),
            reference_plan=dict(entry=100., stop=98., band=[100., 100.5], cost_up=.3, cost_stop=.3),
            gate_receipts=[dict(gate_id=gate, state="PASS", provenance="SYNTHETIC_ONLY_BOUNDARY_STUB")
                for gate in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK")]))


def account():
    return dict(value=1_000_000., available_cash=1_000_000., intended_weight=.25,
        capacity_weight=.25, positions=[], as_of=CUTOFF)


@pytest.mark.parametrize("signal", ["BUY", "BUY_SMALL"])
def test_rule_baseline_retains_same_v7_risk_budget_without_kelly_or_prediction(signal):
    report = boundary_report(signal)
    expected_full = int((Decimal("5000") / Decimal("2.8")).to_integral_value(rounding=ROUND_FLOOR))
    quota = _authorize_validated(report, account())
    assert quota["authorized"] and quota["full_shares"] == expected_full
    assert quota["indicative_shares"] == (expected_full // 4 if signal == "BUY_SMALL" else expected_full)
    assert quota["policy"] == dict(POLICY)
    assert quota["policy"]["full_risk_fraction"] == .005
    assert quota["policy"]["starter_multiplier"] == .25
    assert quota["signal_basis"] == dict(mode="RULE_BASED_BASELINE",
        rule_policy_id=report["diagnostics"]["rule_baseline"]["policy_id"], model_version="7.0.0",
        parameter_version=report["parameter_version"], calibration_status="NOT_APPLICABLE", order_authority=False)
    assert quota["scope"] == "PAE_CANDIDATE" and quota["live_order_authorized"] is False
    assert report["envelope"]["execution"]["live_enabled"] is False
    assert quota["bound_plan_loss_per_share"] == 2.8
    assert "kelly" not in str(quota).lower() and "win_rate" not in str(quota).lower()
    signed = dict(report, pae_authorization=quota)
    validate_authorization(signed)


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK"])
@pytest.mark.parametrize("state", ["UNKNOWN", "FAIL"])
def test_baseline_cannot_authorize_when_any_mandatory_gate_is_not_pass(gate, state):
    report = boundary_report()
    next(node for node in report["diagnostics"]["gate_receipts"] if node["gate_id"] == gate)["state"] = state
    quota = _authorize_validated(report, account())
    assert not quota["authorized"] and quota["indicative_shares"] == 0
    assert "V7_HARD_GATES_NOT_PASS" in quota["reason_codes"]


@pytest.mark.parametrize("key", ["value", "available_cash", "intended_weight", "capacity_weight", "positions"])
def test_missing_account_inputs_are_not_assumed_empty_or_fully_funded(key):
    inputs = account()
    inputs.pop(key)
    quota = _authorize_validated(boundary_report(), inputs)
    assert not quota["authorized"] and quota["indicative_shares"] == 0
    assert "ACCOUNT_INPUTS_MISSING" in quota["reason_codes"] or "ACCOUNT_POSITIONS_UNKNOWN" in quota["reason_codes"]


def test_cash_capacity_positions_and_asof_are_separate_account_boundaries():
    report = boundary_report()
    first = _authorize_validated(report, account())
    low_cash = _authorize_validated(report, dict(account(), available_cash=1_000.))
    assert 0 < low_cash["indicative_shares"] < first["indicative_shares"]
    assert low_cash["account_sha256"] != first["account_sha256"]
    assert low_cash["authorization_id"] != first["authorization_id"]
    empty_cash = _authorize_validated(report, dict(account(), available_cash=0.))
    assert not empty_cash["authorized"] and empty_cash["indicative_shares"] == 0
    unknown_shares = _authorize_validated(report, dict(account(), positions=[dict(symbol="2330")]))
    assert not unknown_shares["authorized"] and "ACCOUNT_POSITIONS_UNKNOWN" in unknown_shares["reason_codes"]
    with pytest.raises(ValueError, match="ACCOUNT_CUTOFF_MISMATCH"):
        _authorize_validated(report, dict(account(), as_of="2026-10-15T13:30:00+08:00"))


def test_liquidity_and_available_capacity_still_cap_baseline_order_size():
    report = boundary_report()
    report["diagnostics"]["features"]["adv"] = 10_000_000.
    quota = _authorize_validated(report, account())
    assert quota["indicative_shares"] == 995  # floor(ADV × 1% / worst entry price).
    assert quota["indicative_shares"] * quota["sizing_entry_limit"] <= 100_000.
    no_capacity = _authorize_validated(report, dict(account(), capacity_weight=0.))
    assert not no_capacity["authorized"] and no_capacity["indicative_shares"] == 0
    report["diagnostics"]["features"]["spread"] = .02
    wide_spread = _authorize_validated(report, account())
    assert not wide_spread["authorized"] and "LIQUIDITY_NOT_PASS" in wide_spread["reason_codes"]


@pytest.mark.parametrize("damage", ["probability", "trained_artifacts", "model_hash", "policy_id"])
def test_baseline_authorization_rejects_fabricated_calibration_or_model_identity(damage):
    report = boundary_report()
    if damage == "probability": report["envelope"]["localization"]["p_entry_given_fill"] = .99
    elif damage == "trained_artifacts": report["envelope"]["model"]["trained_artifacts_status"] = "BOUND_VERIFIED"
    elif damage == "model_hash": report["envelope"]["model"]["bundle_sha256"] = "0" * 64
    else: report["diagnostics"]["rule_baseline"]["policy_id"] = None
    with pytest.raises(ValueError, match="BASELINE_AUTHORIZATION_POLICY_BINDING"):
        _authorize_validated(report, account())


def test_rule_policy_and_parameter_changes_cannot_reuse_old_account_quota():
    report = boundary_report()
    report["pae_authorization"] = _authorize_validated(report, account())
    modified = deepcopy(report)
    modified["diagnostics"]["rule_baseline"]["policy_id"] = identity(dict(synthetic_only=True, different=True))
    with pytest.raises(ValueError, match="AUTHORIZATION_BINDING_MISMATCH"):
        validate_authorization(modified)
    modified = deepcopy(report)
    modified["parameter_version"] += ":DIFFERENT"
    with pytest.raises(ValueError, match="AUTHORIZATION_BINDING_MISMATCH"):
        validate_authorization(modified)


def test_old_fitted_signal_authorization_body_does_not_gain_rule_baseline_fields():
    report = boundary_report()
    report["envelope"]["model"].update(model_status="OFFLINE_FITTED", bundle_sha256="0" * 64,
        trained_artifacts_status="BOUND_VERIFIED")
    report["envelope"]["localization"]["status"] = "VALID"
    report["diagnostics"].pop("rule_baseline")
    quota = _authorize_validated(report, account())
    assert quota["authorized"] and "signal_basis" not in quota


def public_baseline_request():
    """Existing causal synthetic bars, original-shaped financials, test ports."""
    from dataclasses import replace
    import pandas as pd
    from application.market_history import MarketHistoryAssembler
    from domain.les_v7.engine import evaluate
    from domain.les_v7.contracts import timestamp
    from tests.strategies.test_daily_v7_pipeline import executable_synthetic_market
    from tests.strategies.test_v7_reported_quality import source_records
    market = executable_synthetic_market("BUY")
    dates = pd.bdate_range(end="2026-10-14", periods=len(market.dates[0])).strftime("%Y-%m-%d").tolist()
    market = replace(market, stock_numbers=["2330"], dates=[dates])
    raw = MarketHistoryAssembler().build(market, 0)
    raw.attrs.clear()
    for field in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        raw[field] /= 1000.
    context = dict(symbol="2330", market="TWSE", as_of=CUTOFF,
        filings=source_records(), rule_baseline=True, receipts={})
    for gate in ("SAFETY", "DATA_INTEGRITY", "TRADING_STATUS", "VALUATION", "STRESS_RISK"):
        report = evaluate(raw, context)
        context["receipts"][gate] = dict(gate_id=gate, gate_version="SYNTHETIC_ONLY_NOT_OFFICIAL",
            state="PASS", input_hash=identity(report["diagnostics"]["gate_bindings"][gate]),
            source_available_at=timestamp(CUTOFF).isoformat(), checked_at=timestamp(CUTOFF).isoformat(),
            proof_record_ids=["SYNTHETIC_ONLY"], input_ids=["SYNTHETIC_ONLY"], reason_codes=[])
    return raw, context


def test_public_model_free_baseline_validates_then_authorizes_only_account_bound_candidate():
    from domain.les_v7.engine import evaluate
    from domain.les_v7.allocation import authorize_candidate
    from domain.les_v7.report import validate_report
    raw, context = public_baseline_request()
    report = evaluate(raw, context)
    validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] == "BUY_SMALL"
    assert report["envelope"]["score"]["final_score"] >= 62.
    assert report["envelope"]["localization"]["status"] == "NOT_APPLICABLE"
    assert report["envelope"]["localization"]["p_entry_given_fill"] is None
    quota = authorize_candidate(report, account())
    assert quota["authorized"] and quota["indicative_shares"] > 0
    assert quota["indicative_shares"] == quota["full_shares"] // 4
    assert quota["signal_basis"]["rule_policy_id"] == report["diagnostics"]["rule_baseline"]["policy_id"]
    assert quota["live_order_authorized"] is False and report["envelope"]["execution"]["allocated_shares"] == 0
    report["pae_authorization"] = quota
    validate_report(report)


def test_public_baseline_rejects_rehashed_probability_and_blocks_unknown_gate_or_cash():
    from domain.les_v7.engine import evaluate
    from domain.les_v7.allocation import authorize_candidate
    raw, context = public_baseline_request()
    report = evaluate(raw, context)
    forged = deepcopy(report)
    forged["diagnostics"]["localization"]["p_entry"] = .99
    forged["diagnostics"]["localization"]["calibration_version"] = "FABRICATED_SYNTHETIC_CALIBRATION"
    forged["envelope"]["localization"]["p_entry_given_fill"] = .99
    forged["envelope"]["localization"]["forecast_result_id"] = identity(forged["diagnostics"]["localization"])
    with pytest.raises(ValueError, match="BASELINE_CANNOT_FABRICATE"):
        authorize_candidate(forged, account())
    unknown_cash = account(); unknown_cash.pop("available_cash")
    assert authorize_candidate(report, unknown_cash)["indicative_shares"] == 0
    context["receipts"]["VALUATION"]["state"] = "UNKNOWN"
    partial = evaluate(raw, context)
    assert partial["envelope"]["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"
    assert partial["envelope"]["score"]["final_score"] <= 61.
    assert authorize_candidate(partial, account())["indicative_shares"] == 0
