"""SYNTHETIC ONLY: conditional reference observations, not broker executions."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.plan_r18 import R18ReferencePlan, bind_entry_route, policy
from domain.les_v7.policy_r18 import AVAILABLE_AT, PLAN_VERSION, SUPPORT_RESISTANCE_LIMIT
from domain.les_v7.reference_path import evaluate_reference_path
from tests.test_v7_r17_reference_path import fixture as previous_fixture


def fixture():
    old, window, _, options = previous_fixture()
    bound = policy(options["entry_route"]["reference_policy"], available_at=AVAILABLE_AT)
    plan = R18ReferencePlan(**dict(asdict(old), policy_version=PLAN_VERSION,
        context="NO_CLEAR_CONTEXT", setup_variant=SUPPORT_RESISTANCE_LIMIT))
    options["entry_route"] = bind_entry_route(plan, bound, strategy_decision="BUY_SMALL", risk_route="TACTICAL")
    return plan, window, bound["cost_path_policy"], options


def run(items):
    return evaluate_reference_path(items[0], items[1], items[2], **items[3])


@pytest.mark.parametrize("opening", [99., 99.5, 100.])
def test_neutral_limit_only_fills_at_next_official_open_inside_fixed_band(opening):
    items = fixture(); before = deepcopy(items)
    items[1][1]["open"] = opening
    result = run(items)
    assert result["status"] == "VERIFIED_FILLED"
    execution = result["execution"]
    assert execution["entry"] == opening and execution["entry_risk_check"]["state"] == "PASS"
    assert execution["reference_entry_route"]["strategy_decision"] == "BUY_SMALL"
    assert execution["conditional_entry"]["intraday_touch_authorizes_fill"] is False
    assert execution["approved_for_live"] is False
    assert items[0] == before[0]


@pytest.mark.parametrize("opening", [98.9, 100.5])
def test_outside_band_cannot_fill_on_intraday_touch_or_later_session(opening):
    items = fixture()
    items[1][1].update(open=opening, high=101., low=98., close=99.5)
    result = run(items)
    assert result["status"] == "VERIFIED_NO_FILL"
    assert result["execution"]["filled"] is False
    assert result["execution"]["no_fill_reason"] == "NEXT_OPEN_OUTSIDE_NATIVE_BAND"
    assert result["execution"]["conditional_entry"]["state"] == "EXPIRED_AT_NEXT_SESSION_CLOSE"


@pytest.mark.parametrize("damage", ["missing", "expiry", "core", "policy", "limit"])
def test_missing_or_rehashed_route_does_not_bypass_conditional_rules(damage):
    items = fixture(); route = items[3]["entry_route"]
    if damage == "missing": items[3].pop("entry_route")
    else:
        key, value = dict(expiry=("expiry", "UNTIL_FILLED"), core=("risk_route", "CORE"),
            policy=("reference_policy", {}), limit=("order_type", "MARKET"))[damage]
        route[key] = value
        route["route_id"] = identity({key: value for key, value in route.items() if key != "route_id"})
    result = run(items)
    assert result["status"] == "DATA_GAP" and result["execution"] is None


def test_signal_close_and_observed_open_are_not_replaced_or_rounded():
    items = fixture(); items[1][0].update(open=100., high=100.5, low=99.5, close=100.)
    assert run(items)["diagnostic"]["reason_codes"] == ["R18_SIGNAL_CLOSE_SOURCE_MISMATCH"]
    items = fixture(); items[1][1]["open"] = 99.51
    assert run(items)["status"] == "DATA_GAP"
