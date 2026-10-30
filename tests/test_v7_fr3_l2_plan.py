"""SYNTHETIC ONLY: FR3-L2 causality/contracts, not execution or efficacy proof."""
from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal

import pandas as pd
import pytest

from domain.les_v7 import fr3_l2, fr3_plan
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.features import compute_features, select_bars
from domain.les_v7.reference_path import _round_price, estimate_reference_costs, evaluate_reference_path
from domain.les_v7.research import label_event
from tests.test_v7_fr3_plan import CUTOFF, native_plan, path_fixture as r1_path_fixture, reference
from tests.test_v7_plan_r16 import synthetic_calendar
from tests.test_v7_plan_r18 import neutral_bars
from tests.test_v7_reference_path import fixture as ordinary_path_fixture

CREATED = "2026-10-24T10:00:00+00:00"


def declared():
    return fr3_plan.policy(reference(), available_at=CREATED, version=fr3_l2.PLAN_VERSION)


def l2_plan(data=None, context=fr3_l2.LEADER, *, calendar=True):
    data = neutral_bars() if data is None else data
    features, _ = compute_features(select_bars(data, CUTOFF))
    return fr3_plan.make_plan(context, features, declared(), "a"*64, bars=data, cutoff=CUTOFF,
        session_evidence=synthetic_calendar(data) if calendar else None)


def path_fixture(*, context=fr3_l2.LEADER):
    plan, _ = l2_plan(context=context)
    _, window, _, options = ordinary_path_fixture()
    sessions = list(pd.bdate_range("2026-10-20", periods=21).strftime("%Y-%m-%d"))
    for row, session in zip(window, sessions):
        row.update(session=session, available_at=session+"T13:30:00+08:00",
            open=101., high=101.5, low=99., close=101.)
    window[0].update(open=100., high=101., low=99., close=100.)
    options.update(cutoff=CUTOFF, expected_sessions=sessions,
        entry_route=fr3_plan.bind_entry_route(plan, declared()))
    options["corporate_action_coverage"].update(start_session=sessions[0], end_session=sessions[-1],
        available_at=sessions[-1]+"T13:30:00+08:00")
    return plan, window, declared()["cost_path_policy"], options


def run(items):
    return evaluate_reference_path(items[0], items[1], items[2], **items[3])


def label(items, execution):
    return label_event(items[1], cutoff=items[3]["cutoff"],
        expected_sessions=items[3]["expected_sessions"], plan=items[0], fill=execution)


def test_explicit_version_and_original_source_hashes_freeze_only_leader_band():
    old, _ = native_plan(context=fr3_l2.LEADER)
    plan, audit = l2_plan()
    assert plan.original_plan() == old
    assert plan.parent_plan_id == old.plan_id
    assert (plan.stop, plan.target, plan.band[0]) == (old.stop, old.target, old.band[0])
    assert old.band[1] < plan.band[1] <= plan.signal_close+.25*plan.atr
    assert plan.expiry == fr3_l2.EXPIRY
    assert audit["original_band_evaluation"]["state"] == "PASS"
    assert audit["policy_created_at"] == CREATED
    assert audit["historical_deployment_claim"] is False
    assert fr3_plan.plan_from_record(asdict(plan)) == plan
    assert fr3_plan.is_fr3_reference(declared())
    assert fr3_plan.policy(reference(), available_at=CREATED)["version"] == fr3_plan.PLAN_VERSION
    for context in ("REVERSAL", "TREND_PULLBACK"):
        new, _ = l2_plan(context=context)
        old, _ = native_plan(context=context)
        assert new.band == old.band and new.expiry == "NEXT_SESSION_CLOSE"
        assert new.original_plan() == old


def test_future_bars_and_missing_or_broken_source_cannot_rescue_a_plan():
    data = neutral_bars()
    original, _ = l2_plan(data)
    future = deepcopy(data.iloc[-1].to_dict())
    future.update(Date="2026-10-21", Open=500., High=501., Low=499., Close=500.)
    assert l2_plan(pd.concat([data, pd.DataFrame([future])], ignore_index=True))[0] == original
    assert l2_plan(calendar=False)[0] is None
    data.loc[len(data)-1, "Low"] = 97.9
    broken, audit = l2_plan(data)
    assert broken is None and audit["state"] == "FAIL"
    assert audit["support"]["value"] == 98.


def test_failed_original_risk_cannot_be_rescued_by_lower_entry_or_alternate_support():
    data = neutral_bars()
    # Fixed declared costs make the original source plan fail. Lower prices
    # or another anchor may not be tried as a rescue for this event.
    expensive = reference()
    expensive["cost_path_policy"]["cost_policy"]["minimum_buy_fee"] = 10000.
    features, _ = compute_features(select_bars(data, CUTOFF))
    args = dict(bars=data, cutoff=CUTOFF, session_evidence=synthetic_calendar(data))
    old, _ = fr3_plan.make_plan(fr3_l2.LEADER, features, expensive, "a"*64, **args)
    assert old is not None and fr3_plan.validate_band(old, expensive)["state"] == "FAIL"
    new, audit = fr3_plan.make_plan(fr3_l2.LEADER, features,
        fr3_plan.policy(expensive, available_at=CREATED, version=fr3_l2.PLAN_VERSION), "a"*64, **args)
    assert new is None and audit["reasons"] == ["FR3_L2_ORIGINAL_PLAN_NOT_PASS"]


def test_maximum_upper_is_the_last_cost_after_rr_tick_not_observed_open():
    data = neutral_bars()
    data.loc[data["High"] == 110., "High"] = 105.
    new, audit = l2_plan(data)
    assert new is not None, audit
    old = new.original_plan()
    upper = fr3_l2.maximum_upper(old, declared())
    assert upper == new.band[1]
    assert fr3_plan.validate_band(new, declared())["state"] == "PASS"
    cap = float(_round_price(min(Decimal(str(new.signal_close))+Decimal(".25")*Decimal(str(new.atr)),
        Decimal(str(new.stop))+Decimal("2.5")*Decimal(str(new.atr))), "down"))
    if upper < cap:
        tick = _round_price(Decimal(str(upper))+Decimal(".01"), "up")
        assert fr3_l2._risk(tick, old, declared())["state"] == "FAIL"
    # Changed fees produce their own frozen bound; no price observation enters.
    alternate = declared()
    alternate["cost_path_policy"]["cost_policy"]["minimum_buy_fee"] = 25.
    assert fr3_l2.maximum_upper(old, alternate) <= upper


@pytest.mark.parametrize("day", [1, 2, 3])
def test_first_eligible_open_fills_and_h20_does_not_move(day):
    items = path_fixture()
    items[1][day].update(open=100., high=101.5, low=99., close=101.)
    before = deepcopy(items)
    result = run(items)
    assert result["status"] == "VERIFIED_FILLED", result
    execution = result["execution"]
    assert execution["filled_at"].startswith(items[3]["expected_sessions"][day])
    assert execution["conditional_entry"]["filled_opportunity"] == day
    assert len(execution["conditional_entry"]["waiting_observations"]) == day
    assert execution["realized_at"].startswith(items[3]["expected_sessions"][20])
    assert execution["exit_reason"] == "H20_CLOSE"
    assert execution["label_version"] == fr3_l2.LABEL_VERSION
    observed = label(items, execution)
    assert observed["fill"] == 1 and observed["label_version"] == fr3_l2.LABEL_VERSION
    assert observed["reference_plan_id"] == items[0].plan_id
    assert observed["horizon_anchor"] == "SIGNAL_SESSION"
    assert execution["capacity_guarantee"] is execution["approved_for_live"] is False
    assert execution["partial_fills_modeled"] is False
    assert items == before


@pytest.mark.parametrize("change,reason", [({"low":97.9}, "FR3_L2_WAIT_SUPPORT_BROKEN"),
    ({"high":110.}, "FR3_L2_WAIT_TARGET_ALREADY_REACHED")])
def test_only_completed_unfilled_prior_session_cancels_later_open(change, reason):
    items = path_fixture()
    items[1][1].update(change)
    items[1][2]["open"] = 100.
    result = run(items)
    assert result["status"] == "VERIFIED_NO_FILL"
    assert result["execution"]["no_fill_reason"] == reason
    assert result["execution"]["conditional_entry"]["cancelled_at"].startswith(items[1][1]["session"])
    observed = label(items, result["execution"])
    assert observed["fill"] == 0 and observed["joint"] is observed["outcome_r"] is None


def test_fill_open_precedes_own_range_and_exit_activation_is_next_actual_session():
    items = path_fixture()
    # At day two opening this day's low/high is unknown. It may not cancel the
    # already filled opening or grant a forbidden same-day favorable exit.
    items[1][2].update(open=100., low=97., high=110., close=101.)
    items[1][3].update(open=111., low=110., high=112., close=111.)
    execution = run(items)["execution"]
    assert execution["filled_at"].startswith(items[1][2]["session"])
    assert execution["exit_reason"] == "FIXED_TARGET"
    assert execution["exit_price"] == 111.
    assert execution["realized_at"].startswith(items[1][3]["session"])
    assert label(items, execution)["rebound"] == 1


def test_intraday_touch_and_fourth_open_never_fill_nonleaders_stay_next_only():
    items = path_fixture()
    items[1][4]["open"] = 100.
    result = run(items)
    assert result["status"] == "VERIFIED_NO_FILL"
    assert len(result["execution"]["conditional_entry"]["waiting_observations"]) == 3
    assert result["execution"]["conditional_entry"]["intraday_touch_authorizes_fill"] is False
    nonleader = path_fixture(context="TREND_PULLBACK")
    nonleader[1][2]["open"] = 100.
    result = run(nonleader)
    assert result["status"] == "VERIFIED_NO_FILL"
    assert len(result["execution"]["conditional_entry"]["waiting_observations"]) == 1


def test_tampered_source_band_route_and_old_labels_do_not_pass():
    plan, _ = l2_plan()
    with pytest.raises(ValueError, match="FIXED_SOURCE"):
        replace(plan, band=list(plan.band))
    with pytest.raises(ValueError, match="SOURCE_BINDING"):
        replace(plan, original_cost_up=plan.original_cost_up+.01)
    with pytest.raises(ValueError, match="SOURCE_BINDING"):
        replace(plan, source_structure_json=plan.source_structure_json.replace('98.0', '98.1'))
    narrowed = replace(plan, band=plan.original_band, entry=plan.original_band[1],
        structure_id=identity(dict(plan.structure_binding(), band=list(plan.original_band))))
    with pytest.raises(ValueError, match="FROZEN_MAXIMAL"):
        fr3_plan.bind_entry_route(narrowed, declared())
    items = path_fixture()
    items[3]["entry_route"]["expiry"] = "UNTIL_FILLED"
    assert run(items)["status"] == "DATA_GAP"
    items = path_fixture(); items[1][2]["open"] = 100.
    execution = run(items)["execution"]
    execution["label_version"] = fr3_plan.LABEL_VERSION
    with pytest.raises(ValueError, match="LABEL_BINDING"):
        label(items, execution)
    with pytest.raises(ValueError, match="PATH_VERSION_MISMATCH"):
        evaluate_reference_path(items[0], items[1], reference()["cost_path_policy"], **items[3])


def test_r1_outputs_are_byte_stable_across_l2_use_and_keep_one_open():
    items = r1_path_fixture()
    before_plan, before_policy = canonical(asdict(items[0])), canonical(reference())
    before = canonical(run(items))
    result = run(path_fixture())
    assert result["status"] == "VERIFIED_NO_FILL"
    assert canonical(run(items)) == before
    assert canonical(asdict(items[0])) == before_plan and canonical(reference()) == before_policy
    assert "label_source_id" not in label(items, run(items)["execution"])
    items[1][1].update(open=100.5, high=101., low=99., close=100.)
    assert run(items)["status"] == "VERIFIED_NO_FILL"
