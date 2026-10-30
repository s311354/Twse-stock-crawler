"""SYNTHETIC ONLY: explicit core/tactical paths, never live order authority."""
from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal

import pytest

from domain.les_v7.context_r15 import VARIANT, SUPPORT_METHOD, assess_short_base, select_context
from domain.les_v7.contracts import CONTEXTS, ReferencePlan, identity
from domain.les_v7.plan_r10 import (R15_VERSION, _pass_prefix, bind_entry_route, evaluate_entry,
                                   make_plan, policy, validate_band)
from domain.les_v7.policy_r15 import AVAILABLE_AT
from domain.les_v7.reference_path import _tick, estimate_reference_costs, evaluate_reference_path
from tests.strategies.test_v7_context_r14 import CUTOFF, prepared, short_base_bars
from tests.strategies.test_v7_context_r15 import source_bars
from tests.strategies.test_v7_plan_r10 import EVIDENCE, legacy_policy


def declared_r15():
    return policy(legacy_policy(), available_at=AVAILABLE_AT, version=R15_VERSION)


def recognized_bars(*, support=79.8):
    data = short_base_bars()
    data.loc[70, ["Open", "High", "Low", "Close"]] = [90., 91., 89., 90.]
    data.loc[80:93, ["Open", "High", "Low", "Close"]] = [82., 83., 81., 82.]
    data.loc[92, "Low"] = support
    return data


def build(data=None, *, variant=None, context=None):
    data, f, original = prepared(source_bars() if data is None else data)
    selected, _ = select_context(original, f, bars=data, cutoff=CUTOFF)
    atr = assess_short_base(data, f, cutoff=CUTOFF)["metrics"]["signal_atr"]
    return make_plan(context or selected.context, dict(close=float(data.iloc[-1]["Close"]), atr=atr),
        declared_r15(), EVIDENCE, bars=data, cutoff=CUTOFF,
        **(dict(setup_variant=variant) if variant is not None else {}))


def direct_plan(*, context="LEADERSHIP_CONTINUATION", entry=83., stop=79.2,
                target=89.5, atr=2., band_high=83.5, reference=None, shares=1):
    reference = declared_r15() if reference is None else reference
    costs = estimate_reference_costs(entry, stop, target, shares, reference["cost_path_policy"])
    return ReferencePlan("REFERENCE_H20", R15_VERSION, EVIDENCE, context, entry, stop, target,
        atr, (entry, band_high), shares, costs["cost_up_per_share"], costs["cost_stop_per_share"])


def test_new_subtype_uses_exact_five_day_floor_and_never_gets_core_route_even_if_core_risk_passes():
    plan, audit = build()
    assert plan is not None and audit["setup_variant"] == VARIANT
    assert audit["anchors"]["support"]["value"] == 80.
    assert audit["anchors"]["support_method"] == SUPPORT_METHOD
    assert plan.stop == 80. - .3 * plan.atr and plan.target == 90.
    assert audit["r15_risk"]["core"]["state"] == "PASS"
    assert audit["r15_risk"]["tactical"]["state"] == "PASS"
    assert audit["r15_risk"]["band_route"] == "TACTICAL"
    assert audit["r15_risk"]["tactical_band"]["state"] == "PASS"
    route = bind_entry_route(plan, declared_r15(), strategy_decision="BUY_SMALL",
                             risk_route="TACTICAL", setup_variant=VARIANT)
    assert route["plan_id"] == plan.plan_id and route["reference_policy_id"] == identity(declared_r15())
    with pytest.raises(ValueError, match="BUY_REQUIRES_CORE"):
        bind_entry_route(plan, declared_r15(), strategy_decision="BUY", risk_route="TACTICAL", setup_variant=VARIANT)


def test_original_recognized_core_close_keeps_core_prefix_and_same_original_pivot():
    plan, audit = build(recognized_bars())
    assert plan.context == "REVERSAL" and audit["setup_variant"] is None
    assert audit["anchors"]["support"]["value"] == 79.8
    assert audit["anchors"]["support_method"] == "CONFIRMED_BOTTOM_LOW"
    assert audit["r15_risk"]["band_route"] == "CORE"
    assert audit["r15_risk"]["core_band"]["state"] == "PASS"
    receipt = bind_entry_route(plan, declared_r15(), strategy_decision="BUY", risk_route="CORE")
    assert receipt["strategy_decision"] == "BUY"


def test_failed_original_core_risk_uses_only_tactical_same_anchors_not_the_new_short_floor():
    plan, audit = build(recognized_bars(support=79.2))
    assert plan.context == "REVERSAL" and audit["setup_variant"] is None
    assert audit["anchors"]["support"]["value"] == 79.2
    assert audit["r15_risk"]["core"]["state"] == "FAIL"
    assert audit["r15_risk"]["tactical"]["state"] == "PASS"
    assert audit["r15_risk"]["band_route"] == "TACTICAL"
    assert audit["r15_risk"]["core_band"]["state"] == "FAIL"
    assert audit["r15_risk"]["tactical_band"]["state"] == "PASS"
    with pytest.raises(ValueError, match="NATIVE_RISK_NOT_PASS"):
        bind_entry_route(plan, declared_r15(), strategy_decision="BUY", risk_route="CORE")
    with pytest.raises(ValueError, match="CONTEXT_VARIANT_MISMATCH"):
        build(recognized_bars(support=79.2), variant=VARIANT)


@pytest.mark.parametrize("context", CONTEXTS)
def test_r15_tactical_is_explicit_not_a_global_change_to_core_limits(context):
    p = declared_r15(); plan = direct_plan(context=context)
    core = evaluate_entry(plan, plan.entry, p)
    tactical = evaluate_entry(plan, plan.entry, p, starter=True)
    assert core["state"] == "FAIL" and core["maximum_stop_atr"] == 1.8
    assert core["minimum_net_rr"] == (1.8 if context == "REVERSAL" else 1.6)
    assert tactical["state"] == "PASS" and tactical["maximum_stop_atr"] == 2.
    assert tactical["minimum_net_rr"] == 1.3
    assert core["costs"] == tactical["costs"]
    assert plan.stop == 79.2 and plan.target == 89.5


def test_each_tactical_prefix_tick_passes_and_the_first_failed_tick_is_not_skipped():
    p = declared_r15(); raw = direct_plan()
    plan, audit = _pass_prefix(raw, p, starter=True)
    assert plan.entry == raw.entry and replace(plan, band=raw.band) == raw
    assert audit["state"] == "PASS" and audit["clipped"]
    tick = Decimal(str(plan.band[0]))
    while tick <= Decimal(str(plan.band[1])):
        assert evaluate_entry(plan, float(tick), p, starter=True)["state"] == "PASS"
        tick += _tick(tick)
    assert evaluate_entry(plan, float(tick), p, starter=True)["state"] == "FAIL"
    assert audit["first_nonpass"]["entry"] == float(tick)
    assert validate_band(plan, p, starter=True)["state"] == "PASS"
    assert validate_band(plan, p)["state"] == "FAIL"


def test_tactical_does_not_repair_invalid_stop_missing_target_or_failed_close():
    p = declared_r15()
    plan = direct_plan(target=85.)
    same, audit = _pass_prefix(plan, p, starter=True)
    assert same == plan and audit["state"] == "FAIL" and audit["executable_prefix"] is None
    assert evaluate_entry(plan, plan.stop, p, starter=True)["state"] == "FAIL"
    missing = direct_plan(target=None)
    assert evaluate_entry(missing, missing.entry, p, starter=True)["state"] == "UNKNOWN"
    with pytest.raises(ValueError, match="NATIVE_RISK_NOT_PASS"):
        bind_entry_route(missing, p, strategy_decision="BUY_SMALL", risk_route="TACTICAL")


def test_route_receipt_is_detached_plan_policy_and_decision_bound_not_live_authority():
    p = declared_r15(); plan, _ = _pass_prefix(direct_plan(), p, starter=True)
    before = deepcopy(p)
    receipt = bind_entry_route(plan, p, strategy_decision="BUY_SMALL", risk_route="TACTICAL")
    assert p == before and receipt["authority"] == "REFERENCE_RESEARCH_ROUTE_NOT_LIVE_ORDER_AUTHORITY"
    receipt["reference_policy"]["structural_rules"]["maximum_stop_atr"] = 999.
    assert p == before
    with pytest.raises(ValueError, match="BUY_REQUIRES_CORE"):
        bind_entry_route(plan, p, strategy_decision="BUY", risk_route="TACTICAL")


def path_case(*, core=False):
    from tests.test_v7_reference_path import fixture
    _, window, path, kwargs = fixture()
    path.update(available_at=AVAILABLE_AT, entry_guard_policy="R10_REPRICE_NATIVE_STOP_AND_RESISTANCE")
    bound = policy(dict(policy_id="REFERENCE_H20", reference_shares=1000,
        entry_band_atr=[0., .25], cost_path_policy=path), available_at=AVAILABLE_AT, version=R15_VERSION)
    plan = direct_plan(entry=100., stop=98. if core else 96.2, target=108., atr=2.,
        band_high=100., reference=bound, shares=1000)
    receipt = bind_entry_route(plan, bound, strategy_decision="BUY" if core else "BUY_SMALL",
                               risk_route="CORE" if core else "TACTICAL")
    return plan, window, path, kwargs, receipt


@pytest.mark.parametrize("core", [False, True])
def test_public_path_uses_explicit_bound_route_and_exact_actual_entry_repricing(core):
    plan, window, path, kwargs, receipt = path_case(core=core)
    result = evaluate_reference_path(plan, window, path, entry_route=receipt, **kwargs)
    assert result["status"] == "VERIFIED_FILLED"
    check = result["execution"]["entry_risk_check"]
    assert check["maximum_stop_atr"] == (1.8 if core else 2.)
    assert check["minimum_net_rr"] == (1.6 if core else 1.3)
    assert result["execution"]["reference_entry_route"] == receipt
    assert result["execution"]["historical_deployment_claim"] is False


def test_public_path_missing_route_or_tactical_borrowed_for_buy_is_a_gap_not_a_fill():
    plan, window, path, kwargs, receipt = path_case()
    missing = evaluate_reference_path(plan, window, path, **kwargs)
    assert missing["status"] == "DATA_GAP" and missing["execution"] is None
    assert missing["diagnostic"]["reason_codes"] == ["R15_ENTRY_ROUTE_BINDING_REQUIRED"]
    receipt["strategy_decision"] = "BUY"
    receipt["route_id"] = identity({key: value for key, value in receipt.items() if key != "route_id"})
    wrong = evaluate_reference_path(plan, window, path, entry_route=receipt, **kwargs)
    assert wrong["status"] == "DATA_GAP" and wrong["execution"] is None


@pytest.mark.parametrize("mutation", ["plan", "policy", "hash", "extra"])
def test_public_path_detects_route_plan_policy_hash_or_extra_field_tampering(mutation):
    plan, window, path, kwargs, receipt = path_case()
    if mutation == "plan": receipt["plan_id"] = "0" * 64
    if mutation == "policy": receipt["reference_policy"]["cost_path_policy"]["available_at"] = "2026-10-20T00:00:00Z"
    if mutation == "hash": receipt["route_id"] = "0" * 64
    if mutation == "extra": receipt["unbound_field"] = True
    result = evaluate_reference_path(plan, window, path, entry_route=receipt, **kwargs)
    assert result["status"] == "DATA_GAP" and result["execution"] is None
