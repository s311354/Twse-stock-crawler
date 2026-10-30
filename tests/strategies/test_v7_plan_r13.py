"""SYNTHETIC ONLY: versioned R13 eligibility, never a real fill or BUY claim."""
from copy import deepcopy
from dataclasses import asdict

import pandas as pd
import pytest

from domain.les_v7.contracts import CONTEXTS, identity
from domain.les_v7.decision import make_plan as dispatch_plan
from domain.les_v7.plan_r10 import evaluate_entry, make_plan, policy, validate_band
from domain.les_v7.policy_r13 import AVAILABLE_AT, PLAN_VERSION, risk_limits
from domain.les_v7.reference_path import estimate_reference_costs
from tests.strategies.test_v7_plan_r10 import (EVIDENCE, bars, build as build_r10,
                                               declared_policy, legacy_policy)

CUTOFF = "2026-10-16T13:30:00+08:00"


def declared_r13():
    return policy(legacy_policy(), available_at=AVAILABLE_AT, version=PLAN_VERSION)


def source_bars(*, target=127.):
    data = bars()
    data["Date"] = pd.bdate_range(end="2026-10-16", periods=len(data)).strftime("%Y-%m-%d")
    data.loc[20, "High"] = target
    data.loc[50, "Low"] = 105.
    return data


def build(*, context="REVERSAL", atr=4., data=None, p=None, cutoff=CUTOFF):
    data = source_bars() if data is None else data
    return make_plan(context, dict(close=110., atr=atr), p or declared_r13(), EVIDENCE,
                     bars=data, cutoff=cutoff)


def test_old_default_policy_plan_and_sidecar_content_ids_are_exactly_unchanged():
    original = declared_policy()
    plan, sidecar = build_r10(p=original)
    assert identity(original) == "48972ca58966179b37b26c933ff1c6c0d6cb6ef2a1f8fe03a75c498476fe8d68"
    assert identity(asdict(plan)) == "4fe07348eb94452a35948bf36f1573298ccd17f44b3b583b95f1d33f3c9130b6"
    assert identity(sidecar) == "3a6a6f870af71076e7f1b560a141ec52744d7dd666d1e568964bd06d89b51ba9"


def test_r13_changes_only_explicit_risk_rules_not_support_target_band_or_cost_assumptions():
    legacy = legacy_policy()
    before = deepcopy(legacy)
    old = policy(legacy, available_at=AVAILABLE_AT)
    new = policy(legacy, available_at=AVAILABLE_AT, version=PLAN_VERSION)
    assert legacy == before
    old_rules, new_rules = old.pop("structural_rules"), new.pop("structural_rules")
    assert new.pop("version") == PLAN_VERSION
    assert old.pop("version") == "V7_SETUP_STRUCTURE_H20_R10"
    assert old == new
    limits = risk_limits()
    for key in ("maximum_stop_atr", "starter_stop_atr", "minimum_net_rr"):
        assert new_rules.pop(key) == limits[key]
        old_rules.pop(key)
    assert new_rules == old_rules
    # No caller can mutate global R10/R13 rules through a returned policy.
    altered = declared_r13()
    altered["structural_rules"]["maximum_stop_atr"] = 99.
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        build(p=altered)
    assert declared_r13()["structural_rules"]["maximum_stop_atr"] == 1.8
    assert declared_policy()["structural_rules"]["maximum_stop_atr"] == 1.5


@pytest.mark.parametrize("context", CONTEXTS)
def test_r13_full_and_small_use_same_approved_1_8_atr_with_identical_real_anchors(context):
    data = source_bars()
    new, sidecar = build(context=context, data=data)
    old, old_sidecar = build(context=context, data=data, p=declared_policy())
    assert new.stop == old.stop == 103.8
    assert new.target == old.target == 127.
    assert new.entry == old.entry == 110.
    assert new.band == old.band == (110., 111.)
    assert sidecar["anchors"] == old_sidecar["anchors"]
    assert sidecar["bars_id"] == old_sidecar["bars_id"]
    assert new.policy_version == PLAN_VERSION and new.plan_id != old.plan_id
    for starter in (False, True):
        actual = validate_band(new, declared_r13(), starter=starter)
        assert actual["state"] == "PASS"
        assert actual["maximum_stop_atr"] == 1.8
        assert actual["entry"] == 111. and actual["stop_atr"] == pytest.approx(1.8)
        assert validate_band(old, declared_policy(), starter=starter)["state"] == "FAIL"
    assert dispatch_plan(context, dict(close=110., atr=4.), declared_r13(), EVIDENCE,
                         bars=data, cutoff=CUTOFF) == new


@pytest.mark.parametrize("context,target", [("REVERSAL", 127.), ("TREND_PULLBACK", 125.),
                                             ("LEADERSHIP_CONTINUATION", 125.)])
def test_approved_net_rr_is_cost_inclusive_and_does_not_manufacture_a_better_target(context, target):
    data = source_bars(target=target)
    new, _ = build(context=context, atr=5., data=data)
    old, _ = build(context=context, atr=5., data=data, p=declared_policy())
    normal = validate_band(new, declared_r13())
    assert normal["state"] == "PASS"
    assert normal["minimum_net_rr"] == (1.8 if context == "REVERSAL" else 1.6)
    assert normal["net_rr"] < (2. if context == "REVERSAL" else 1.8)
    assert validate_band(old, declared_policy())["state"] == "FAIL"
    assert new.target == old.target == target
    expected_costs = estimate_reference_costs(normal["entry"], new.stop, new.target, 1,
                                                declared_r13()["cost_path_policy"])
    assert normal["costs"] == expected_costs
    assert normal["net_rr"] == pytest.approx((target - normal["entry"] - expected_costs["cost_up_per_share"])
        / (normal["entry"] - new.stop + expected_costs["cost_stop_per_share"]))


@pytest.mark.parametrize("context", CONTEXTS)
def test_close_pass_does_not_hide_a_failed_worst_tick_in_the_full_band(context):
    p = declared_r13()
    plan, diag = build(context=context, data=source_bars(target=124.))
    assert evaluate_entry(plan, 110., p)["state"] == "PASS"
    for starter in (False, True):
        result = validate_band(plan, p, starter=starter)
        assert result["state"] == "FAIL"
        assert result["entry"] == 111.
        assert "FIXED_RESISTANCE_NET_RR_BELOW_MINIMUM" in result["reasons"]
    assert diag["conditional_entry"]["state"] == "UNFILLED"
    assert diag["conditional_entry"]["qualifies_current_close"] is False
    assert plan.entry == 110. and plan.target == 124.


def test_invalid_broken_or_overwide_stops_are_not_relocated_to_qualify():
    p = declared_r13()
    plan, _ = build(atr=3.)
    assert plan.stop == 104.1
    assert "FIXED_STRUCTURE_STOP_ATR_EXCEEDED" in validate_band(plan, p)["reasons"]
    data = source_bars()
    data.loc[59, "Low"] = 103.8
    broken, diag = build(data=data)
    assert broken is None and diag["reasons"] == ["SELECTED_STRUCTURE_STOP_ALREADY_VIOLATED"]
    assert diag["anchors"]["support"]["value"] == 105.
    assert evaluate_entry(plan, 104., p)["reasons"] == ["ENTRY_NOT_ABOVE_FIXED_STOP"]


def test_actual_reprice_grid_unknown_and_policy_version_binding_fail_closed():
    p = declared_r13()
    plan, _ = build()
    changed = evaluate_entry(plan, 111.5, p)
    assert changed["state"] == "FAIL"
    assert changed["costs"] == estimate_reference_costs(111.5, plan.stop, plan.target, 1, p["cost_path_policy"])
    assert changed["net_rr"] != plan.net_rr
    assert evaluate_entry(plan, 110.01, p)["reasons"] == ["ENTRY_OUTSIDE_LEGAL_PRICE_GRID"]
    assert evaluate_entry(plan, None, p)["state"] == "UNKNOWN"
    assert evaluate_entry(plan, plan.target, p)["reasons"] == ["ENTRY_NOT_BELOW_FIXED_RESISTANCE"]
    with pytest.raises(ValueError, match="PLAN_VERSION_REQUIRED"):
        evaluate_entry(plan, 110., declared_policy())
    old, _ = build(p=declared_policy())
    with pytest.raises(ValueError, match="PLAN_VERSION_REQUIRED"):
        validate_band(old, p)


def test_missing_or_after_cutoff_structure_evidence_stays_unknown_without_synthetic_targets():
    p = declared_r13()
    missing, diag = make_plan("REVERSAL", dict(close=110., atr=4., support=105., hh20=999.),
                              p, EVIDENCE, cutoff=CUTOFF)
    assert missing is None and diag["reasons"] == ["SOURCE_BOUND_COMPLETED_BARS_MISSING"]
    data = source_bars(target=111.)
    partial, diag = build(data=data)
    assert partial.target is None and diag["state"] == "PARTIAL"
    assert evaluate_entry(partial, 110., p)["state"] == "UNKNOWN"
    assert diag["conditional_entry"] is None
    data = source_bars()
    data["Available At"] = data["Date"] + "T13:30:00+08:00"
    data.loc[50, "Available At"] = "2026-10-19T13:30:00+08:00"
    missing, diag = build(data=data)
    assert missing is None and diag["reasons"] == ["CONFIRMED_SETUP_LOW_MISSING"]


def test_future_bars_future_policy_and_before_release_cannot_rewrite_cutoff_plan():
    data = source_bars()
    original = build(data=data)
    future = data.iloc[[-1]].copy()
    future.loc[:, "Date"] = "2026-10-19"
    future.loc[:, ["Open", "High", "Low", "Close"]] = [200., 500., 1., 200.]
    assert build(data=pd.concat([data, future], ignore_index=True)) == original
    future_policy = policy(legacy_policy(), available_at="2026-10-17T00:00:00Z", version=PLAN_VERSION)
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        build(p=future_policy)
    with pytest.raises(ValueError, match="BEFORE_RELEASE"):
        policy(legacy_policy(), available_at="2026-10-15T00:00:00Z", version=PLAN_VERSION)
    tampered = declared_r13()
    tampered["available_at"] = "2026-10-15T00:00:00Z"
    with pytest.raises(ValueError, match="BEFORE_RELEASE"):
        build(p=tampered)
    unknown = declared_r13()
    unknown["version"] = "UNDECLARED_NEW_RISK_POLICY"
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        build(p=unknown)
