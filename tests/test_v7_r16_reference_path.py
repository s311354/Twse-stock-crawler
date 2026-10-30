"""SYNTHETIC ONLY: conditional reference observations, not broker fills.

Source-bound native admission and account authorization are separately tested
in the R16 pipeline suite. The H20 reference path does not authenticate source
providers, certify execution depth or grant live order authority.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import date

import pytest

from domain.les_v7.contracts import ReferencePlan, identity
from domain.les_v7.policy_r16 import AVAILABLE_AT, EARLY_PULLBACK, PLAN_VERSION
from domain.les_v7.reference_path import estimate_reference_costs, evaluate_reference_path
from tests.strategies.test_v7_plan_r10 import legacy_policy
from tests.test_v7_reference_path import fixture as old_fixture


def fixture():
    from domain.les_v7.plan_r16 import R16ReferencePlan, bind_entry_route, policy
    _, window, path, options = old_fixture()
    shift = date(2026, 10, 19) - date.fromisoformat(window[0]["session"])
    sessions = [(date.fromisoformat(row["session"]) + shift).isoformat() for row in window]
    for row, session in zip(window, sessions):
        row.update(session=session, available_at=session + "T13:30:00+08:00",
                   open=99.5, high=100., low=99., close=99.5)
    window[0].update(open=102., high=103.5, low=101.5, close=103.)
    options.update(expected_sessions=sessions, cutoff=sessions[0] + "T13:30:00+08:00")
    options["corporate_action_coverage"].update(start_session=sessions[0], end_session=sessions[-1],
        available_at=sessions[-1] + "T13:30:00+08:00")
    declared = legacy_policy()
    declared.update(reference_shares=1000, cost_path_policy=path)
    bound = policy(declared, available_at=AVAILABLE_AT)
    path = bound["cost_path_policy"]
    costs = estimate_reference_costs(100., 97.4, 107., 1000, path)
    plan = R16ReferencePlan("REFERENCE_H20", PLAN_VERSION, "a" * 64, "TREND_PULLBACK",
        100., 97.4, 107., 2., (99., 100.), 1000,
        costs["cost_up_per_share"], costs["cost_stop_per_share"],
        entry_basis="STRUCTURAL_CONDITIONAL_BAND", signal_close=103.,
        setup_variant=EARLY_PULLBACK, target_method="CONFIRMED_UNBROKEN_OVERHEAD_RESISTANCE",
        structure_id="b" * 64)
    route = bind_entry_route(plan, bound, strategy_decision="BUY_SMALL", risk_route="TACTICAL",
        setup_variant=EARLY_PULLBACK)
    options["entry_route"] = route
    return plan, window, path, options


def run(items):
    return evaluate_reference_path(items[0], items[1], items[2], **items[3])


def test_conditional_price_is_not_the_signal_close_and_actual_open_is_repriced():
    items = fixture(); original = deepcopy(items)
    result = run(items); execution = result["execution"]
    assert result["status"] == "VERIFIED_FILLED", result
    assert execution["entry"] == 99.5 != items[0].entry != items[0].signal_close
    assert execution["entry_risk_check"]["state"] == "PASS"
    assert execution["entry_risk_check"]["costs"] == estimate_reference_costs(
        99.5, items[0].stop, items[0].target, items[0].reference_shares, items[2])
    conditional = execution["conditional_entry"]
    assert conditional["signal_close"] == 103.
    assert conditional["entry_basis"] == "STRUCTURAL_CONDITIONAL_BAND"
    assert conditional["entry_rule"] == "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND"
    assert conditional["eligible_session"] == items[3]["expected_sessions"][1]
    assert conditional["expires_at"].startswith(conditional["eligible_session"])
    assert conditional["expiry"] == "NEXT_SESSION_CLOSE"
    assert conditional["price_condition_met"] is True
    assert conditional["intraday_touch_authorizes_fill"] is False
    assert conditional["state"] == "REFERENCE_FILLED_AT_ELIGIBLE_OPEN"
    assert execution["approved_for_live"] is False
    assert execution["capacity_guarantee"] is False
    assert items == original


@pytest.mark.parametrize("opening", [98.9, 100.5])
def test_both_sides_of_band_expire_without_intraday_or_later_session_rescue(opening):
    items = fixture()
    items[1][1].update(open=opening, high=101., low=98., close=99.5)
    # Later range and later opens cross the band but cannot roll this signal.
    result = run(items); execution = result["execution"]
    assert result["status"] == "VERIFIED_NO_FILL"
    assert execution["filled"] is False
    assert execution["no_fill_reason"] == "NEXT_OPEN_OUTSIDE_NATIVE_BAND"
    conditional = execution["conditional_entry"]
    assert conditional["price_condition_met"] is False
    assert conditional["state"] == "EXPIRED_AT_NEXT_SESSION_CLOSE"
    assert conditional["intraday_touch_authorizes_fill"] is False


@pytest.mark.parametrize("opening", [99., 100.])
def test_conditional_legal_band_edges_are_inclusive(opening):
    items = fixture()
    items[1][1].update(open=opening, high=max(opening, 100.), low=min(opening, 99.))
    assert run(items)["execution"]["entry"] == opening


@pytest.mark.parametrize("damage", ["missing", "basis", "rule", "expiry", "subtype", "source", "score_route", "path"])
def test_missing_or_rehashed_conditional_route_is_not_execution_authority(damage):
    items = fixture(); route = items[3]["entry_route"]
    if damage == "missing":
        items[3].pop("entry_route")
    else:
        key, value = {
            "basis": ("entry_basis", "CUTOFF_CLOSE"),
            "rule": ("entry_rule", "INTRADAY_LOW_TOUCH"),
            "expiry": ("expiry", "UNTIL_FILLED"),
            "subtype": ("setup_variant", "UNSUPPORTED_R16"),
            "source": ("structure_id", "c" * 64),
            "score_route": ("risk_route", "CORE"),
            "path": ("reference_policy", dict(route["reference_policy"], cost_path_policy={})),
        }[damage]
        route[key] = value
        route["route_id"] = identity({key: value for key, value in route.items() if key != "route_id"})
    result = run(items)
    assert result["status"] == "DATA_GAP", result
    assert result["execution"] is None


def test_plain_old_plan_cannot_impersonate_new_source_bound_r16_plan():
    items = fixture()
    fields = asdict(items[0])
    for name in ("entry_basis", "signal_close", "setup_variant", "target_method", "structure_id"):
        fields.pop(name)
    items = (ReferencePlan(**fields), *items[1:])
    assert run(items)["diagnostic"]["reason_codes"] == ["R16_SOURCE_BOUND_PLAN_REQUIRED"]


def test_conditional_price_cannot_replace_signal_session_observed_close():
    items = fixture()
    items[1][0].update(open=100., high=100.5, low=99.5, close=100.)
    result = run(items)
    assert result["status"] == "DATA_GAP"
    assert result["diagnostic"]["reason_codes"] == ["R16_SIGNAL_CLOSE_SOURCE_MISMATCH"]


def test_invalid_observed_price_is_not_rounded_into_the_conditional_band():
    items = fixture()
    items[1][1].update(open=99.51)
    assert run(items)["status"] == "DATA_GAP"


@pytest.mark.parametrize("problem", ["unknown_session", "missing_bar", "zero_volume", "unverified_coordinate"])
def test_missing_execution_evidence_never_becomes_verified_no_fill_or_fill(problem):
    items = fixture()
    if problem == "unknown_session": items[3]["expected_sessions"].pop(1)
    elif problem == "missing_bar": items[1][1]["complete"] = False
    elif problem == "zero_volume": items[1][1]["volume"] = 0
    else: items[3]["corporate_action_coverage"]["status"] = "UNKNOWN"
    if problem == "unknown_session":
        with pytest.raises(ValueError, match="OFFICIAL_H20_SESSION_SEQUENCE_REQUIRED"):
            run(items)
    else:
        result = run(items)
        assert result["status"] == "DATA_GAP" and result["execution"] is None


def test_unknown_r16_plan_version_does_not_silently_skip_the_guard():
    items = fixture()
    fields = {key: value for key, value in asdict(items[0]).items()
              if key not in {"entry_basis", "signal_close", "setup_variant", "target_method", "structure_id"}}
    fields["policy_version"] += "_UNSUPPORTED"
    items = (ReferencePlan(**fields), *items[1:])
    assert run(items)["diagnostic"]["reason_codes"] == ["UNSUPPORTED_R16_REFERENCE_PLAN_VERSION"]
