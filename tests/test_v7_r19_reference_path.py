"""Synthetic R19 conditional execution; no profitability or provider claim."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.plan_r19 import R19ReferencePlan, bind_entry_route, policy
from domain.les_v7.policy_r19 import AVAILABLE_AT, PLAN_VERSION
from domain.les_v7.reference_path import evaluate_reference_path
from tests.test_v7_r18_reference_path import fixture as old_fixture


def fixture():
    old, window, _, options = old_fixture()
    bound = policy(options["entry_route"]["reference_policy"], available_at=AVAILABLE_AT)
    plan = R19ReferencePlan(**dict(asdict(old), policy_version=PLAN_VERSION))
    options["entry_route"] = bind_entry_route(plan, bound,
        strategy_decision="BUY_SMALL", risk_route="TACTICAL")
    return plan, window, bound["cost_path_policy"], options


def run(items):
    return evaluate_reference_path(items[0], items[1], items[2], **items[3])


@pytest.mark.parametrize("opening", [99., 99.5, 100.])
def test_r19_replay_revalidates_legal_open_costs_and_native_risk(opening):
    items = fixture()
    items[1][1]["open"] = opening
    before = deepcopy(items)
    result = run(items)
    assert result["status"] == "VERIFIED_FILLED", result
    execution = result["execution"]
    assert execution["entry"] == opening
    assert execution["entry_risk_check"]["state"] == "PASS"
    assert execution["reference_entry_route"]["plan_id"] == items[0].plan_id
    assert execution["conditional_entry"]["intraday_touch_authorizes_fill"] is False
    assert execution["approved_for_live"] is False
    assert items == before


@pytest.mark.parametrize("opening", [98.9, 100.5])
def test_r19_outside_band_expires_despite_intraday_touch(opening):
    items = fixture()
    items[1][1].update(open=opening, high=101., low=98., close=99.5)
    result = run(items)
    assert result["status"] == "VERIFIED_NO_FILL"
    assert result["execution"]["no_fill_reason"] == "NEXT_OPEN_OUTSIDE_NATIVE_BAND"
    assert result["execution"]["conditional_entry"]["state"] == "EXPIRED_AT_NEXT_SESSION_CLOSE"


@pytest.mark.parametrize("damage", ["missing", "expiry", "risk_route", "policy", "order_type"])
def test_r19_rehashed_or_missing_route_cannot_authorize_execution(damage):
    items = fixture()
    if damage == "missing":
        items[3].pop("entry_route")
    else:
        key, value = dict(expiry=("expiry", "UNTIL_FILLED"), risk_route=("risk_route", "CORE"),
            policy=("reference_policy", {}), order_type=("order_type", "MARKET"))[damage]
        route = items[3]["entry_route"]
        route[key] = value
        route["route_id"] = identity({key: value for key, value in route.items() if key != "route_id"})
    result = run(items)
    assert result["status"] == "DATA_GAP"
    assert result["execution"] is None


def test_r19_observed_open_cannot_be_rounded_into_a_valid_tick():
    items = fixture()
    items[1][1]["open"] = 99.51
    assert run(items)["status"] == "DATA_GAP"


def test_retained_r18_replay_remains_versioned_and_unchanged():
    items = old_fixture()
    before = deepcopy(items)
    result = run(items)
    assert result["status"] == "VERIFIED_FILLED"
    assert result["execution"]["policy_version"] == "V7_SUPPORT_RESISTANCE_LIMIT_R18"
    assert items == before
