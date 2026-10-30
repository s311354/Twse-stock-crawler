"""SYNTHETIC ONLY: R14 anchors and exact executable prefixes, not real fills."""
from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal

import pytest

from domain.les_v7.context_r14 import VARIANT, SUPPORT_METHOD, assess_short_base
from domain.les_v7.contracts import ReferencePlan, identity
from domain.les_v7.plan_r10 import (R14_VERSION, _pass_prefix, evaluate_entry, make_plan,
                                   policy, validate_band)
from domain.les_v7.policy_r14 import AVAILABLE_AT
from domain.les_v7.reference_path import _tick, estimate_reference_costs
from tests.strategies.test_v7_context_r14 import CUTOFF, prepared, short_base_bars
from tests.strategies.test_v7_plan_r10 import EVIDENCE, legacy_policy


def declared_r14():
    return policy(legacy_policy(), available_at=AVAILABLE_AT, version=R14_VERSION)


def build(data=None, *, variant=VARIANT, reference=None, context="LEADERSHIP_CONTINUATION"):
    data, f, _ = prepared(data)
    audit = assess_short_base(data, f, cutoff=CUTOFF)
    native = dict(close=float(data.iloc[-1]["Close"]), atr=audit["metrics"]["signal_atr"])
    return make_plan(context, native, reference or declared_r14(), EVIDENCE,
        bars=data, cutoff=CUTOFF, **(dict(setup_variant=variant) if variant is not None else {}))


def test_source_bound_short_floor_is_not_a_strict_pivot_or_the_closest_low():
    data = short_base_bars()
    plan, audit = build(data)
    support = audit["anchors"]["support"]
    assert audit["setup_variant"] == VARIANT
    assert support["value"] == 80. and support["value"] < data.iloc[-2]["Low"]
    assert audit["anchors"]["support_method"] == SUPPORT_METHOD
    assert support["method"] == SUPPORT_METHOD
    assert support["touch_episode_count"] == 2
    assert plan.stop == 80. - .3 * plan.atr
    assert plan.target == 90.
    assert plan.entry == 82.2
    assert plan.expiry == "NEXT_SESSION_CLOSE"
    assert audit["anchors"]["resistance"]["session"] == data.iloc[10]["Date"]
    assert plan.plan_id == audit["plan_id"]


def test_native_producer_without_a_variant_flag_derives_the_same_unique_plan():
    explicit, left = build()
    implicit, right = build(variant=None)
    assert asdict(explicit) == asdict(implicit)
    assert left == right
    with pytest.raises(ValueError, match="CONTEXT_VARIANT_MISMATCH"):
        build(variant="NOT_A_RECOGNIZED_VARIANT")
    with pytest.raises(ValueError, match="CONTEXT_VARIANT_MISMATCH"):
        build(context="REVERSAL")


def test_changed_bar_or_missing_amount_cannot_reuse_a_saved_short_base_pass():
    data = short_base_bars()
    data.loc[len(data) - 5, "Low"] = 80.2
    with pytest.raises(ValueError, match="CONTEXT_VARIANT_MISMATCH"):
        build(data)
    data = short_base_bars().drop(columns="Amount")
    with pytest.raises(ValueError, match="CONTEXT_VARIANT_MISMATCH"):
        build(data)


def test_short_plan_cannot_override_cutoff_native_atr_to_relocate_its_stop():
    data = short_base_bars()
    with pytest.raises(ValueError, match="ATR_SOURCE_MISMATCH"):
        make_plan("LEADERSHIP_CONTINUATION", dict(close=82.2, atr=.1), declared_r14(),
                  EVIDENCE, bars=data, cutoff=CUTOFF, setup_variant=VARIANT)


def test_new_variant_is_rejected_by_old_r13_policy_without_changing_old_plan_behavior():
    from domain.les_v7.policy_r13 import AVAILABLE_AT as OLD_AT, PLAN_VERSION
    old = policy(legacy_policy(), available_at=OLD_AT, version=PLAN_VERSION)
    with pytest.raises(ValueError, match="VARIANT_REQUIRES_R14"):
        build(reference=old)


def test_r14_policy_and_source_mutations_cannot_relax_declared_limits():
    reference = declared_r14()
    before = deepcopy(reference)
    plan, _ = build(reference=reference)
    assert reference == before
    reference["structural_rules"]["early_short_base_policy"]["base_width_atr"][0] = 999.
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        evaluate_entry(plan, plan.entry, reference)
    assert declared_r14()["structural_rules"]["early_short_base_policy"]["base_width_atr"] == [3., 4.]


def direct_plan(*, entry=83., stop=80., target=89.5, atr=2., band_high=83.5):
    reference = declared_r14()
    costs = estimate_reference_costs(entry, stop, target, 1, reference["cost_path_policy"])
    return ReferencePlan("REFERENCE_H20", R14_VERSION, EVIDENCE, "LEADERSHIP_CONTINUATION",
        entry, stop, target, atr, (entry, band_high), 1,
        costs["cost_up_per_share"], costs["cost_stop_per_share"])


def test_prefix_trims_only_the_first_failed_tick_without_moving_anchors_or_reference_close():
    p = declared_r14()
    original = direct_plan()
    assert evaluate_entry(original, 83., p)["state"] == "PASS"
    assert validate_band(original, p)["state"] == "FAIL"
    clipped, diag = _pass_prefix(original, p)
    assert diag["state"] == "PASS" and diag["clipped"]
    assert clipped.band[0] == original.band[0] == original.entry
    assert clipped.band[1] < original.band[1]
    assert replace(clipped, band=original.band) == original
    assert validate_band(clipped, p)["state"] == "PASS"
    price = Decimal(str(clipped.band[0]))
    while price <= Decimal(str(clipped.band[1])):
        assert evaluate_entry(clipped, float(price), p)["state"] == "PASS"
        price += _tick(price)
    assert evaluate_entry(clipped, float(price), p)["state"] == "FAIL"
    assert diag["first_nonpass"]["entry"] == float(price)
    assert diag["checked_legal_ticks"] > 1
    assert clipped.plan_id != original.plan_id


def test_close_must_pass_no_lower_replacement_entry_or_synthetic_target():
    p = declared_r14()
    original = direct_plan(target=85.)
    unchanged, diag = _pass_prefix(original, p)
    assert unchanged == original and diag["state"] == "FAIL"
    assert diag["executable_prefix"] is None and diag["checked_legal_ticks"] == 0
    assert "CLOSE_MUST_PASS" in diag["reason"]
    unknown = direct_plan(target=None)
    unchanged, diag = _pass_prefix(unknown, p)
    assert unchanged == unknown and diag["state"] == "UNKNOWN"
    assert unchanged.target is None


def test_prefix_respects_legal_tick_transition_and_not_a_binary_search_that_skips_failures():
    p = declared_r14()
    original = direct_plan(entry=99.9, stop=98., target=110., atr=2., band_high=100.4)
    clipped, audit = _pass_prefix(original, p)
    assert clipped.band == (99.9, 100.)
    assert audit["checked_legal_ticks"] == 2
    assert audit["first_nonpass"] is None
    assert validate_band(clipped, p)["state"] == "PASS"


def test_short_branch_plan_and_diagnostics_are_detached_and_deterministic():
    first, audit = build()
    saved = identity(audit)
    audit["anchors"]["support"]["value"] = 999.
    second, again = build()
    assert first == second and identity(again) == saved


def test_reference_path_reprices_r14_and_keeps_counterfactual_deployment_disclaimer():
    from tests.test_v7_reference_path import fixture, run
    _, window, reference, kwargs = fixture()
    reference.update(available_at=AVAILABLE_AT, entry_guard_policy="R10_REPRICE_NATIVE_STOP_AND_RESISTANCE")
    costs = estimate_reference_costs(100., 98., 107., 1000, reference)
    plan = ReferencePlan("REFERENCE_H20", R14_VERSION, EVIDENCE, "LEADERSHIP_CONTINUATION",
        100., 98., 107., 2., (100., 100.5), 1000, costs["cost_up_per_share"], costs["cost_stop_per_share"])
    result = run((plan, window, reference, kwargs))
    assert result["status"] == "VERIFIED_FILLED"
    check = result["execution"]["entry_risk_check"]
    assert check["state"] == "PASS" and check["maximum_stop_atr"] == 1.8
    assert check["minimum_net_rr"] == 1.6
    assert result["execution"]["historical_deployment_claim"] is False
    assert result["execution"]["initial_stop"] == plan.stop
    assert result["execution"]["native_resistance_target"] == plan.target


def test_reference_path_does_not_fill_above_clipped_band_or_bypass_r14_reprice_failure():
    from tests.test_v7_reference_path import fixture, run
    _, window, reference, kwargs = fixture()
    reference.update(available_at=AVAILABLE_AT, entry_guard_policy="R10_REPRICE_NATIVE_STOP_AND_RESISTANCE")
    plan = ReferencePlan("REFERENCE_H20", R14_VERSION, EVIDENCE, "LEADERSHIP_CONTINUATION",
        100., 95., 107., 2., (100., 100.), 1000, .5, .5)
    failed = run((plan, window, reference, kwargs))
    assert failed["status"] == "VERIFIED_NO_FILL"
    assert failed["execution"]["no_fill_reason"] == "R14_ACTUAL_ENTRY_RISK_NOT_PASS"
    assert failed["execution"]["entry_risk_check"]["state"] == "FAIL"
    window[1]["open"] = 100.5
    outside = run((plan, window, reference, kwargs))
    assert outside["status"] == "VERIFIED_NO_FILL"
    assert outside["execution"]["no_fill_reason"] == "NEXT_OPEN_OUTSIDE_NATIVE_BAND"
