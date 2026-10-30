"""Synthetic R19 causality and unchanged risk limits; not profit evidence."""
from copy import deepcopy
from dataclasses import asdict, replace

import pandas as pd
import pytest

import domain.strategies  # noqa: F401 -- existing plugin registrations
from domain.les_v7 import plan_r18, plan_r19
from domain.les_v7.contracts import identity
from domain.les_v7.policy_r19 import AVAILABLE_AT
from domain.les_v7.risk_r19 import assess_execution, require_assessment
from domain.les_v7.rule_baseline import RELAXED_BASELINE_POLICY, baseline_policy
from tests.test_v7_plan_r16 import synthetic_calendar
from tests.test_v7_plan_r18 import neutral_bars, selected, native_plan as old_native_plan

CUTOFF = "2026-10-22T14:00:00Z"


def reference():
    return plan_r19.policy(baseline_policy(version=RELAXED_BASELINE_POLICY)["reference_policy"],
        available_at=AVAILABLE_AT)


def native_plan(data, *, calendar=True):
    selection, audit = selected(data, calendar=calendar)
    _, atr = plan_r18.previous.r16._atr(plan_r18.previous.r16.select_bars(data, CUTOFF))
    return plan_r19.make_plan(selection.context,
        dict(close=float(data.iloc[-1]["Close"]), atr=atr), reference(), "a" * 64,
        bars=data, cutoff=CUTOFF, setup_variant=audit.get("setup_variant"),
        session_evidence=synthetic_calendar(data) if calendar else None)


def test_extends_neutral_band_without_moving_source_anchors_or_stop_target():
    data = neutral_bars()
    old, _ = old_native_plan(data)
    before = identity(asdict(old))
    new, sidecar = native_plan(data)
    assert isinstance(new, plan_r19.R19ReferencePlan)
    assert new.band[0] == old.band[0] and new.band[1] > old.band[1]
    assert (new.stop, new.target, new.signal_close, new.context, new.setup_variant) == (
        old.stop, old.target, old.signal_close, old.context, old.setup_variant)
    assert new.entry == new.band[1] and new.entry_basis == plan_r19.A_ENTRY_BASIS
    assert sidecar["r19_entry_band_extension"]["extended"] is True
    assert identity(asdict(old_native_plan(data)[0])) == before
    assert plan_r19.plan_from_record(asdict(new)) == new
    assert plan_r19.plan_from_record(asdict(old)) == old


def test_worst_legal_tick_still_passes_unchanged_tactical_risk_and_net_rr():
    plan, sidecar = native_plan(neutral_bars())
    check = plan_r19.validate_band(plan, reference(), starter=True)
    assert check["state"] == "PASS"
    assert check["maximum_stop_atr"] == 2.5 and check["minimum_net_rr"] == 1.2
    extension = sidecar["r19_entry_band_extension"]
    assert plan.band[1] <= extension["planned_upper"]
    assert plan_r19.evaluate_entry(plan, plan.entry, reference())["state"] == "FAIL"
    route = plan_r19.bind_entry_route(plan, reference(), strategy_decision="BUY_SMALL", risk_route="TACTICAL")
    assert route["plan_id"] == plan.plan_id and route["reference_policy_id"] == identity(reference())
    assert route["entry_rule"] == "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND"
    assert sidecar["conditional_entry"]["assumes_fill"] is False
    with pytest.raises(ValueError, match="TACTICAL_SMALL"):
        plan_r19.bind_entry_route(plan, reference(), strategy_decision="BUY", risk_route="CORE")


def test_stops_at_first_cost_net_rr_failure_without_searching_beyond_it():
    plan, _ = native_plan(neutral_bars())
    original = replace(plan, stop=97., atr=4., target=107.5,
        entry=100., band=(99., 100.), signal_close=100.)
    extended, audit = plan_r19._extend(original, reference())
    assert audit["state"] == "PASS" and audit["extended"]
    first = audit["first_nonpass"]
    assert first and "FIXED_RESISTANCE_NET_RR_BELOW_MINIMUM" in first["check"]["reasons"]
    assert extended.band[1] < first["entry"]
    assert plan_r19.validate_band(extended, reference(), starter=True)["state"] == "PASS"
    assert extended.stop == original.stop and extended.target == original.target


def test_atr_limit_is_not_relaxed_to_capture_a_higher_open():
    plan, _ = native_plan(neutral_bars())
    original = replace(plan, stop=95., atr=2., target=120.,
        entry=100., band=(99., 100.), signal_close=100.)
    extended, audit = plan_r19._extend(original, reference())
    assert extended.band == original.band
    assert audit["first_nonpass"]["check"]["reasons"] == ["FIXED_STRUCTURE_STOP_ATR_EXCEEDED"]


def test_a_failed_original_reference_or_band_is_never_rescued():
    data = neutral_bars()
    data.loc[len(data) - 40, "High"] = 101.5
    old, _ = old_native_plan(data)
    assert plan_r18.validate_band(old, plan_r19._as_previous_policy(reference()), starter=True)["state"] == "FAIL"
    plan, sidecar = native_plan(data)
    audit = sidecar["r19_entry_band_extension"]
    assert audit["extended"] is False and audit["checked_legal_ticks"] == 0
    assert plan.band == old.band and plan.stop == old.stop and plan.target == old.target
    with pytest.raises(ValueError, match="RISK_NOT_PASS"):
        plan_r19.bind_entry_route(plan, reference(), strategy_decision="BUY_SMALL", risk_route="TACTICAL")


def test_original_core_route_is_not_downgraded_to_relax_its_risk_limits():
    plan, _ = native_plan(neutral_bars())
    original = replace(plan, context="TREND_PULLBACK", setup_variant=None,
        entry_basis=plan_r19.CLOSE_ENTRY_BASIS, entry=100., band=(100., 100.),
        signal_close=100., stop=99., target=105., atr=2.)
    extended, audit = plan_r19._extend(original, reference())
    assert audit["route"] == "CORE"
    assert plan_r19.validate_band(extended, reference())["state"] == "PASS"
    assert audit["first_nonpass"] is not None
    assert extended.entry == original.signal_close


def test_missing_official_calendar_and_missing_resistance_are_not_repaired():
    plan, sidecar = native_plan(neutral_bars(), calendar=False)
    assert plan is None
    assert sidecar["state"] == "UNKNOWN"
    data = neutral_bars(); data.loc[len(data) - 70, "High"] = 101.
    plan, sidecar = native_plan(data)
    assert plan is None
    assert "R18_OBSERVED_UNBROKEN_RESISTANCE_MISSING" in sidecar["reasons"]


def test_future_prices_do_not_change_the_band_or_extension_diagnostics():
    data = neutral_bars()
    before, diagnostics = native_plan(data)
    future = data.tail(1).copy(); future["Date"] = "2026-10-23"
    future[["Open", "High", "Low", "Close"]] = [999., 1000., 1., 999.]
    more = pd.concat((data, future), ignore_index=True)
    selection, audit = selected(data)
    _, atr = plan_r18.previous.r16._atr(data)
    after, next_diagnostics = plan_r19.make_plan(selection.context,
        dict(close=float(data.iloc[-1]["Close"]), atr=atr), reference(), "a" * 64,
        bars=more, cutoff=CUTOFF, setup_variant=audit["setup_variant"],
        session_evidence=synthetic_calendar(data))
    assert after == before and next_diagnostics == diagnostics


def test_policy_and_plan_tampering_fail_closed():
    with pytest.raises(ValueError, match="BEFORE_RELEASE"):
        plan_r19.policy(baseline_policy(version=RELAXED_BASELINE_POLICY)["reference_policy"],
            available_at="2026-10-22T13:56:52Z")
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        plan_r19.require_policy(reference(), "2026-10-22T13:56:52Z")
    bad = deepcopy(reference()); bad["structural_rules"]["r19"]["upper_extension_atr"] = 99.
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        plan_r19.require_policy(bad)
    plan, _ = native_plan(neutral_bars())
    with pytest.raises(ValueError, match="MUST_NOT_BE_RELABELLED"):
        plan_r19.plan_from_record(dict(asdict(plan), context="TREND_PULLBACK"))


def test_repriced_assessment_binds_new_band_and_uses_its_high_for_liquidity():
    plan, _ = native_plan(neutral_bars())
    features = dict(spread=.001, adv=1000000.)
    check = assess_execution(plan, reference(), features)
    assert check["liquidity"]["participation"] == plan.band[1] / features["adv"]
    assert check["tactical"]["state"] == "PASS"
    assert require_assessment(check, plan, reference(), features) == check
    changed = replace(plan, band=(plan.band[0], plan.band[1] + .5), entry=plan.entry + .5)
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        require_assessment(check, changed, reference(), features)
    unknown = assess_execution(plan, reference(), dict(spread=None, adv=1000000.))
    assert unknown["state"] == "UNKNOWN"
