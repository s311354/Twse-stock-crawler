"""SYNTHETIC ONLY: causal setup anchors and exact risk/cost arithmetic."""
from copy import deepcopy
from dataclasses import asdict

import pandas as pd
import pytest

from domain.les_v7.contracts import ReferencePlan, identity
from domain.les_v7.decision import make_plan as old_make_plan
from domain.les_v7.plan_r10 import VERSION, evaluate_entry, make_plan, policy, validate_band
from domain.les_v7.reference_path import (COORDINATE_POLICY, GRID_VERSION, PATH_SCHEMA,
                                         PATH_VERSION, estimate_reference_costs)

CUTOFF = "2026-10-15T13:30:00+08:00"
AVAILABLE = "2026-10-15T00:00:00Z"
EVIDENCE = "SYNTHETIC_ONLY_NO_SOURCE_AUTHENTICITY_OR_FILL_CLAIM"


def legacy_policy():
    cost = dict(version="SYNTHETIC_ONLY", rounding="ROUND_HALF_UP", evidence_ids=[EVIDENCE],
        available_at=AVAILABLE, currency_quantum=".01", buy_rate=.001425, sell_rate=.004425,
        minimum_buy_fee=0., minimum_sell_fee=0., effective_start="2026-01-01", effective_end="2026-12-31")
    path = dict(schema=PATH_SCHEMA, version=PATH_VERSION, instrument_type="ORDINARY_STOCK",
        market="TWSE", entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND",
        exit_rule="FIRST_STOP_OR_NET_2R_ELSE_H20_CLOSE", exit_activation="NEXT_SESSION_AFTER_FILL",
        shares_rule="NATIVE_REFERENCE_SHARES", price_grid=GRID_VERSION, coordinate_policy=COORDINATE_POLICY,
        source_evidence_ids=[EVIDENCE], available_at=AVAILABLE, slippage_bps=0., cost_policy=cost)
    return dict(policy_id="REFERENCE_H20", version="V7_RULE_BASELINE_H20_CLOSE_R1",
        entry_rule="CUTOFF_CLOSE_NEXT_SESSION_BAND", support_rule="STRICT_PIVOT_ELSE_PRIOR_LL20",
        reference_shares=1, entry_band_atr=[0., .25], leadership_rr_min=1.8, cost_path_policy=path)


def declared_policy():
    return policy(legacy_policy(), available_at=AVAILABLE)


def bars():
    n = 60
    return pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-15", periods=n).strftime("%Y-%m-%d"),
        Open=[110.] * n, High=[111.] * n, Low=[109.] * n, Close=[110.] * n,
        Volume=[10000.] * n, Complete=[True] * n))


def set_low(data, i, low):
    data.loc[i, "Low"] = low


def set_high(data, i, high):
    data.loc[i, "High"] = high


def setup_bars():
    data = bars()
    set_low(data, 40, 100.)
    set_high(data, 45, 125.)
    set_high(data, 50, 130.)
    set_low(data, 55, 105.)
    return data


def build(data=None, context="REVERSAL", *, atr=5., p=None, cutoff=CUTOFF):
    data = setup_bars() if data is None else data
    close = float(data.loc[data["Date"] <= cutoff[:10], "Close"].iloc[-1])
    return make_plan(context, dict(close=close, atr=atr), p or declared_policy(), EVIDENCE,
                     bars=data, cutoff=cutoff)


def test_setup_selected_before_anchors_and_no_old_support_operand_can_override_them():
    data = setup_bars()
    plan, sidecar = make_plan("REVERSAL", dict(close=110., atr=5., support=109.9, hh20=999.),
                             declared_policy(), EVIDENCE, bars=data, cutoff=CUTOFF)
    assert plan.stop == 105. - .3 * 5.
    assert plan.target == 125.  # nearest unbroken, not higher 130 for better RR
    assert sidecar["anchors"]["support"]["confirmed_at_session"] == data.iloc[57]["Date"]
    assert sidecar["anchors"]["selection_order"] == "SETUP_ANCHORS_BEFORE_COSTS_OR_RR"
    assert sidecar["evidence_id"] == EVIDENCE
    assert sidecar["bars_id"] and "bars" not in sidecar


def test_pullback_requires_low_after_same_prior_impulse_not_generic_old_pivot():
    data = setup_bars()
    plan, diag = build(data, "TREND_PULLBACK")
    assert plan.stop == 103.5
    assert diag["anchors"]["support"]["impulse_high"]["session"] == data.iloc[50]["Date"]
    assert diag["anchors"]["support_method"] == "SAME_IMPULSE_CONFIRMED_PULLBACK_LOW"
    set_low(data, 55, 109.)
    missing, diag = build(data, "TREND_PULLBACK")
    assert missing is None and diag["reasons"] == ["SAME_PULLBACK_CONFIRMED_LOW_MISSING"]


def test_two_right_complete_bars_required_to_confirm_support():
    data = bars(); set_high(data, 30, 130.); set_low(data, 58, 105.)
    plan, diag = build(data)
    assert plan is None and diag["reasons"] == ["CONFIRMED_SETUP_LOW_MISSING"]
    set_low(data, 57, 104.)
    plan, diag = build(data)
    assert plan.stop == 102.5
    assert diag["anchors"]["support"]["confirmed_at_session"] == data.iloc[59]["Date"]


def test_broken_most_recent_support_is_not_replaced_with_older_low():
    data = setup_bars(); data.loc[59, ["Open", "Close", "Low"]] = [104., 104., 103.]
    plan, diag = build(data)
    assert plan is None
    assert diag["reasons"] == ["SELECTED_STRUCTURE_BROKEN_OR_STOP_INVALID"]
    assert diag["anchors"]["support"]["value"] == 105.


def test_low_below_fixed_stop_cannot_be_hidden_by_a_recovered_cutoff_close():
    data = setup_bars(); data.loc[59, "Low"] = 103.5
    plan, diag = build(data)
    assert plan is None and diag["reasons"] == ["SELECTED_STRUCTURE_STOP_ALREADY_VIOLATED"]
    # A separate low can become a new independent source only when its two
    # right-hand sessions actually close, not by moving the old stop.
    data = setup_bars(); data.loc[57, "Low"] = 103.
    plan, diag = build(data)
    assert plan.stop == 101.5
    assert diag["anchors"]["support"]["session"] == data.iloc[57]["Date"]


def test_support_beyond_bounded_history_is_missing_not_forward_filled():
    data = pd.concat([bars()] * 2, ignore_index=True)
    data["Date"] = pd.bdate_range(end="2026-10-15", periods=120).strftime("%Y-%m-%d")
    set_low(data, 20, 100.); set_high(data, 30, 140.)
    plan, diag = build(data)
    assert plan is None and diag["reasons"] == ["CONFIRMED_SETUP_LOW_MISSING"]


def test_completed_close_above_resistance_removes_it_without_choosing_wick_cross():
    data = setup_bars()
    data.loc[47, ["Open", "Close", "High"]] = [126., 126., 126.5]
    plan, diag = build(data)
    assert plan.target == 126.5  # new nearest independently confirmed high
    assert diag["anchors"]["resistance"]["session"] == data.iloc[47]["Date"]


def test_leadership_confirmed_consolidation_floor_and_missing_resistance_are_separate():
    data = bars(); set_high(data, 20, 130.); set_low(data, 50, 105.)
    plan, diag = build(data, "LEADERSHIP_CONTINUATION")
    assert plan.target == 130. and plan.stop == 103.5
    assert diag["anchors"]["support_method"] == "CONFIRMED_20_SESSION_BASE_FLOOR"
    set_high(data, 20, 111.)
    plan, diag = build(data, "LEADERSHIP_CONTINUATION")
    assert plan.target is None and diag["state"] == "PARTIAL"
    assert diag["reasons"] == ["EVIDENCE_DERIVED_RESISTANCE_MISSING"]
    assert diag["conditional_entry"] is None
    assert evaluate_entry(plan, 110., declared_policy())["state"] == "UNKNOWN"


def test_leadership_independent_confirmed_retest_has_breakout_source_anchor():
    data = bars()
    data.loc[:30, ["Open", "Close"]] = 100.
    data.loc[:30, "High"] = 101.
    data.loc[:30, "Low"] = 99.
    data.loc[31, ["Open", "Close", "High", "Low"]] = [102., 102., 103., 101.5]
    data.loc[32:36, ["Open", "Close", "High", "Low"]] = [103., 103., 104., 102.]
    data.loc[37, ["Open", "Close", "High", "Low"]] = [103., 103., 104., 101.]
    set_high(data, 45, 125.)
    plan, diag = build(data, "LEADERSHIP_CONTINUATION", atr=2.)
    assert plan.stop == 100.4
    assert diag["anchors"]["support_method"] == "INDEPENDENTLY_CONFIRMED_BREAKOUT_RETEST_LOW"
    assert diag["anchors"]["support"]["breakout"]["session"] == data.iloc[31]["Date"]


def test_whole_band_checks_worst_legal_tick_not_only_cutoff_close():
    data = setup_bars(); set_high(data, 45, 127.)
    plan, diag = build(data)
    p = declared_policy()
    assert evaluate_entry(plan, 110., p)["state"] == "PASS"
    assert validate_band(plan, p)["state"] == "FAIL"
    assert diag["band_evaluation"]["entry"] == 111.
    assert "FIXED_RESISTANCE_NET_RR_BELOW_MINIMUM" in diag["band_evaluation"]["reasons"]
    assert plan.entry == 110. and plan.band == (110., 111.25)


def test_actual_quote_costs_repriced_and_grid_or_invalid_stop_never_pass():
    plan, _ = build(); p = declared_policy()
    result = evaluate_entry(plan, 110.5, p)
    assert result["costs"] == estimate_reference_costs(110.5, plan.stop, plan.target, 1, p["cost_path_policy"])
    assert result["costs"]["detail"]["entry"] == "110.5"
    assert result["net_rr"] != plan.net_rr
    assert evaluate_entry(plan, 110.01, p)["state"] == "FAIL"
    assert evaluate_entry(plan, 103., p)["reasons"] == ["ENTRY_NOT_ABOVE_FIXED_STOP"]
    assert evaluate_entry(plan, None, p)["state"] == "UNKNOWN"


def test_conditional_zone_is_unfilled_and_does_not_change_formal_score_prices():
    data = setup_bars(); data.loc[59, ["Open", "Close", "High"]] = [120., 120., 120.5]
    plan, diag = build(data)
    assert plan.entry == 120. and evaluate_entry(plan, 120., declared_policy())["state"] == "FAIL"
    conditional = diag["conditional_entry"]
    assert conditional["state"] == "UNFILLED" and conditional["band"][1] <= 107.5
    assert conditional["qualifies_current_close"] is False and conditional["live_order_authorized"] is False
    assert conditional["worst_entry"]["state"] == "PASS"
    assert plan.stop == 103.5 and plan.target == 125.


def test_future_bars_and_revisions_not_available_at_cutoff_do_not_change_old_anchors():
    data = setup_bars(); original = build(data)
    future = data.iloc[[-1]].copy(); future.loc[:, "Date"] = "2026-10-16"
    future.loc[:, ["Open", "High", "Low", "Close"]] = [200., 500., 1., 200.]
    appended = pd.concat([data, future], ignore_index=True)
    assert build(appended) == original
    data["Available At"] = data["Date"] + "T13:30:00+08:00"
    data.loc[59, "Available At"] = "2026-10-16T13:30:00+08:00"
    # A mismatching supplied current close cannot quietly pass selection.
    with pytest.raises(ValueError, match="REFERENCE_CLOSE_DOES_NOT_MATCH"):
        make_plan("REVERSAL", dict(close=120., atr=5.), declared_policy(), EVIDENCE, bars=data, cutoff=CUTOFF)


def test_incomplete_bar_or_policy_after_cutoff_remains_rejected():
    data = setup_bars(); data.loc[59, "Complete"] = False
    with pytest.raises(ValueError, match="INCOMPLETE_DAILY_BAR"):
        build(data)
    p = policy(legacy_policy(), available_at="2026-10-16T00:00:00Z")
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        build(p=p)


def test_detached_sidecar_and_policy_keep_old_reference_contract_and_hash_unchanged():
    old = legacy_policy()
    before = deepcopy(old)
    p = policy(old, available_at=AVAILABLE)
    legacy = old_make_plan("REVERSAL", dict(close=110., support=105., atr=5., hh20=125.), old, EVIDENCE)
    original = asdict(legacy), legacy.plan_id
    plan, diag = build(p=p)
    assert set(asdict(plan)) == set(asdict(legacy))
    assert plan.policy_version == VERSION and plan.horizon == legacy.horizon == 20
    diag["anchors"]["support"]["value"] = 999.
    assert build(p=p)[1]["anchors"]["support"]["value"] == 105.
    assert old == before
    assert (asdict(old_make_plan("REVERSAL", dict(close=110., support=105., atr=5., hh20=125.), old, EVIDENCE)),
            old_make_plan("REVERSAL", dict(close=110., support=105., atr=5., hh20=125.), old, EVIDENCE).plan_id) == original
    changed = deepcopy(p); changed["structural_rules"]["maximum_stop_atr"] = 99.
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        build(p=changed)


def test_missing_bars_and_no_setup_never_use_features_as_synthetic_source():
    result, diag = make_plan("REVERSAL", dict(close=110., support=105., hh20=125., atr=5.),
                            declared_policy(), EVIDENCE, cutoff=CUTOFF)
    assert result is None and diag["state"] == "UNKNOWN"
    result, diag = build(context="NO_CLEAR_CONTEXT")
    assert result is None and diag["reasons"] == ["NO_SELECTED_SETUP"]
