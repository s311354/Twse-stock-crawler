"""SYNTHETIC R18 source structures only, not market evidence or profit tests."""
from copy import deepcopy
from dataclasses import asdict, replace

import pandas as pd
import pytest

import domain.strategies  # noqa: F401 -- existing plugin registrations
from domain.les_v7 import plan_r17, plan_r18
from domain.les_v7.context_r17 import price_structure_features, select_context
from domain.les_v7.policy_r18 import AVAILABLE_AT
from domain.les_v7.rule_baseline import RELAXED_BASELINE_POLICY, baseline_policy, prepare_baseline
from domain.strategies.low_entry_v6_hybrid import ContextSelection, NO_CLEAR_CONTEXT
from tests.test_v7_plan_r16 import discovery_bars, pullback_bars, synthetic_calendar
from tests.strategies.test_v7_context_r14 import CUTOFF


def neutral_bars():
    n = 150
    data = pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-20", periods=n).strftime("%Y-%m-%d"),
        Open=[100.] * n, High=[101.] * n, Low=[99.] * n, Close=[100.] * n,
        Volume=[10000.] * n, Amount=[1000000.] * n, Trades=[100] * n,
        Bid=[99.9] * n, Ask=[100.] * n, **{"TAIEX Close": [1000.] * n}, Complete=[True] * n))
    data.loc[n - 70, "High"] = 110.
    data.loc[n - 12, "Low"] = 98.
    data.loc[n - 1, ["Open", "High", "Low", "Close", "Bid", "Ask"]] = [100., 101.5, 99.3, 100., 99.9, 100.]
    return data


def features(data):
    selected = plan_r17.r16.select_bars(data, CUTOFF)
    value = prepare_baseline(selected, market="TWSE", cutoff=CUTOFF, policy_version=RELAXED_BASELINE_POLICY)
    return price_structure_features(data, value["features"], cutoff=CUTOFF)


def reference():
    return plan_r18.policy(baseline_policy(version=RELAXED_BASELINE_POLICY)["reference_policy"],
        available_at=AVAILABLE_AT)


def selected(data, *, calendar=True):
    f = features(data)
    initial = select_context(f)
    selection, audit = plan_r18.prepare_selection(initial, f, bars=data, cutoff=CUTOFF,
        market="TWSE", session_evidence=synthetic_calendar(data) if calendar else None)
    return selection, audit


def native_plan(data):
    selection, audit = selected(data)
    _, atr = plan_r17.r16._atr(plan_r17.r16.select_bars(data, CUTOFF))
    return plan_r18.make_plan(selection.context, dict(close=float(data.iloc[-1]["Close"]), atr=atr),
        reference(), "a" * 64, bars=data, cutoff=CUTOFF,
        setup_variant=audit.get("setup_variant"), session_evidence=synthetic_calendar(data))


def test_neutral_context_is_not_relabelled_as_a_pattern_and_plan_is_unfilled_limit():
    data = neutral_bars()
    selection, audit = selected(data)
    assert select_context(features(data)).context == selection.context == NO_CLEAR_CONTEXT
    assert audit["setup_variant"] == plan_r18.NEUTRAL_VARIANT
    plan, sidecar = native_plan(data)
    assert plan.context == NO_CLEAR_CONTEXT and plan.setup_variant == plan_r18.NEUTRAL_VARIANT
    assert plan.entry_basis == plan_r18.A_ENTRY_BASIS and plan.entry == plan.band[1]
    assert plan.entry <= plan.signal_close == float(data.iloc[-1]["Close"])
    assert plan.target == 110. and plan.target_method == plan_r18.OBSERVED_TARGET
    assert plan.stop == pytest.approx(98. - .3 * plan.atr)
    assert plan_r18.plan_from_record(asdict(plan)) == plan
    conditional = sidecar["conditional_entry"]
    assert conditional["state"] == "UNFILLED" and conditional["order_type"] == "LIMIT"
    assert conditional["entry_rule"] == "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND"
    assert conditional["expiry"] == "NEXT_SESSION_CLOSE"
    assert conditional["live_order_authorized"] is conditional["assumes_fill"] is False
    assert sidecar["anchors"]["target_projection"] is None


def test_neutral_route_is_small_only_and_does_not_grant_order_authority():
    plan, _ = native_plan(neutral_bars())
    assert plan_r18.validate_band(plan, reference(), starter=True)["state"] == "PASS"
    assert plan_r18.evaluate_entry(plan, plan.entry, reference())["state"] == "FAIL"
    route = plan_r18.bind_entry_route(plan, reference(), strategy_decision="BUY_SMALL", risk_route="TACTICAL")
    assert route["setup_variant"] == plan_r18.NEUTRAL_VARIANT and route["plan_id"] == plan.plan_id
    assert route["authority"] == "REFERENCE_RESEARCH_ROUTE_NOT_LIVE_ORDER_AUTHORITY"
    with pytest.raises(ValueError, match="TACTICAL_SMALL"):
        plan_r18.bind_entry_route(plan, reference(), strategy_decision="BUY", risk_route="CORE")
    with pytest.raises(ValueError, match="VARIANT_MISMATCH"):
        plan_r18.bind_entry_route(plan, reference(), strategy_decision="BUY_SMALL", risk_route="TACTICAL",
            setup_variant="OTHER")


def test_missing_or_incomplete_calendar_cannot_be_replaced_by_weekdays_or_scores():
    data = neutral_bars()
    _, audit = selected(data, calendar=False)
    assert audit["state"] == "UNKNOWN" and audit["setup_variant"] is None
    f = features(data); initial = select_context(f)
    calendar = synthetic_calendar(data)
    calendar["documents"].clear()
    _, audit = plan_r18.prepare_selection(initial, f, bars=data, cutoff=CUTOFF,
        market="TWSE", session_evidence=calendar)
    assert audit["state"] != "PASS" and audit["setup_variant"] is None


def test_source_pivots_are_confirmed_before_signal_and_latest_support_cannot_be_replaced():
    data = neutral_bars(); f = features(data)
    initial = ContextSelection(NO_CLEAR_CONTEXT, select_context(f).evidence)
    # A lower low requires signal-day confirmation and is not independently
    # available before the signal. It breaks, rather than replaces, prior support.
    data.loc[len(data) - 3, "Low"] = 97.
    _, audit = plan_r18.prepare_selection(initial, features(data), bars=data, cutoff=CUTOFF,
        market="TWSE", session_evidence=synthetic_calendar(data))
    assert audit["state"] == "FAIL" and audit["support"]["value"] == 98.
    assert audit["reasons"] == ["R18_CONFIRMED_SUPPORT_BROKEN"]
    assert audit["support"]["confirmed_at_session"] < data.iloc[-1]["Date"]


def test_signal_break_below_fixed_support_is_failure_not_a_tighter_stop():
    data = neutral_bars(); data.loc[len(data) - 1, "Low"] = 97.9
    _, audit = selected(data)
    assert audit["state"] == "FAIL" and audit["support"]["value"] == 98.
    assert audit["metrics"]["stop"] == pytest.approx(98. - .3 * audit["metrics"]["signal_atr"])


@pytest.mark.parametrize("index", [-1, -12, -11, -10, -70, -69, -68])
def test_zero_trading_anchor_confirmation_or_signal_cannot_form_a_verified_floor(index):
    data = neutral_bars()
    data.loc[len(data) + index, "Volume"] = 0.
    _, audit = selected(data)
    assert audit["state"] == "FAIL"
    assert audit["setup_variant"] is None
    assert audit["reasons"] == ["R18_ANCHOR_OR_CONFIRMATION_DID_NOT_TRADE"]


def test_illegal_signal_close_does_not_become_a_legal_reference_quote():
    data = neutral_bars(); data.loc[len(data) - 1, "Close"] = 100.2
    _, audit = selected(data)
    assert audit["state"] == "FAIL" and audit["setup_variant"] is None
    assert audit["reasons"] == ["R18_SIGNAL_CLOSE_OUTSIDE_LEGAL_PRICE_GRID"]


def test_nearest_observed_resistance_is_kept_even_when_net_rr_fails():
    data = neutral_bars(); data.loc[len(data) - 40, "High"] = 101.5
    plan, sidecar = native_plan(data)
    assert plan.target == 101.5  # No switch to the older 110 resistance.
    assert plan_r18.validate_band(plan, reference(), starter=True)["state"] == "FAIL"
    assert sidecar["anchors"]["resistance"]["value"] == 101.5
    with pytest.raises(ValueError, match="RISK_NOT_PASS"):
        plan_r18.bind_entry_route(plan, reference(), strategy_decision="BUY_SMALL", risk_route="TACTICAL")


def test_missing_resistance_is_unknown_and_never_a_projected_target():
    data = neutral_bars(); data.loc[len(data) - 70, "High"] = 101.
    _, audit = selected(data)
    assert audit["state"] == "UNKNOWN" and audit["setup_variant"] is None
    assert audit["resistance"] is None
    plan, sidecar = native_plan(data)
    assert plan is None and "R18_OBSERVED_UNBROKEN_RESISTANCE_MISSING" in sidecar["reasons"]


def test_legal_limit_band_is_fixed_geometry_capped_at_signal_close_before_rr():
    data = neutral_bars(); f = features(data)
    data.loc[len(data) - 1, ["Open", "High", "Low", "Close"]] = [99.5, 100., 99.4, 99.5]
    f = features(data)
    initial = ContextSelection(NO_CLEAR_CONTEXT, select_context(f).evidence)
    _, audit = plan_r18.prepare_selection(initial, f, bars=data, cutoff=CUTOFF,
        market="TWSE", session_evidence=synthetic_calendar(data))
    assert audit["state"] == "PASS"
    assert audit["band"][1] <= 99.5
    assert audit["band"][0] >= 98. + .5 * f["atr14"]


def test_future_bars_do_not_change_selection_anchor_hash_or_reference_plan():
    data = neutral_bars()
    selection, audit = selected(data)
    before, _ = native_plan(data)
    future = data.tail(1).copy(); future["Date"] = "2026-10-21"
    future[["Open", "High", "Low", "Close"]] = [999., 1000., 1., 999.]
    more = pd.concat((data, future), ignore_index=True)
    next_selection, next_audit = selected(more)
    assert (next_selection, next_audit) == (selection, audit)
    _, atr = plan_r17.r16._atr(data)
    after, _ = plan_r18.make_plan(selection.context, dict(close=float(data.iloc[-1]["Close"]), atr=atr), reference(),
        "a" * 64, bars=more, cutoff=CUTOFF, setup_variant=audit["setup_variant"],
        session_evidence=synthetic_calendar(data))
    assert after == before


@pytest.mark.parametrize("factory", [pullback_bars, discovery_bars])
def test_existing_r17_structure_is_retained_even_if_tactical_rr_fails(factory):
    data = factory()
    from tests.test_v7_plan_r17 import native_plan as old_plan
    old, _ = old_plan(data)
    new, _ = native_plan(data)
    assert new.setup_variant == old.setup_variant != plan_r18.NEUTRAL_VARIANT
    assert (new.stop, new.target, new.entry, new.band, new.signal_close) == (
        old.stop, old.target, old.entry, old.band, old.signal_close)


def test_new_tactical_limits_do_not_retain_r17_two_atr_floor():
    plan, _ = native_plan(neutral_bars())
    # Explicit synthetic arithmetic boundary, not source-verified new anchors.
    plan = replace(plan, stop=95., atr=2., target=107.5)
    result = plan_r18.evaluate_entry(plan, 100., reference(), starter=True)
    assert result["state"] == "PASS" and result["stop_atr"] == 2.5
    assert result["maximum_stop_atr"] == 2.5 and result["minimum_net_rr"] == 1.2
    assert 1.2 <= result["net_rr"] < 1.3
    assert plan_r18.evaluate_entry(replace(plan, stop=94.9), 100., reference(), starter=True)["state"] == "FAIL"
    assert plan_r18.evaluate_entry(replace(plan, target=106.), 100., reference(), starter=True)["state"] == "FAIL"


def test_policy_time_hash_and_context_tampering_fail_closed():
    with pytest.raises(ValueError, match="BEFORE_RELEASE"):
        plan_r18.policy(baseline_policy(version=RELAXED_BASELINE_POLICY)["reference_policy"],
            available_at="2026-10-17T04:02:59Z")
    modified = deepcopy(reference()); modified["structural_rules"]["r18"]["tactical_minimum_net_rr"] = .1
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        plan_r18.require_policy(modified)
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        plan_r18.require_policy(reference(), "2026-10-17T04:02:59Z")
    plan, _ = native_plan(neutral_bars())
    with pytest.raises(ValueError, match="MUST_NOT_BE_RELABELLED"):
        plan_r18.plan_from_record(dict(asdict(plan), context="TREND_PULLBACK"))
    with pytest.raises(ValueError, match="SOURCE_CLOSE_OR_ATR_MISMATCH"):
        plan_r18.make_plan(plan.context, dict(close=100.2, atr=999.), reference(), "a" * 64,
            bars=neutral_bars(), cutoff=CUTOFF, setup_variant=plan.setup_variant,
            session_evidence=synthetic_calendar(neutral_bars()))
