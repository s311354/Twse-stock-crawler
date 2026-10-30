"""SYNTHETIC ONLY: FR3 causal source plans and target labels, not efficacy."""
from copy import deepcopy
from dataclasses import asdict, replace

import pandas as pd
import pytest

from domain.les_v7 import fr3_plan
from domain.les_v7.contracts import identity
from domain.les_v7.features import compute_features, select_bars
from domain.les_v7.reference_path import estimate_reference_costs, evaluate_reference_path
from domain.les_v7.research import label_event
from tests.test_v7_plan_r18 import neutral_bars
from tests.test_v7_plan_r16 import synthetic_calendar
from tests.test_v7_r17_reference_path import fixture as previous_fixture

CREATED = "2026-10-23T10:00:00+00:00"
CUTOFF = "2026-10-20T13:30:00+08:00"


def reference():
    from tests.strategies.test_v7_plan_r10 import legacy_policy
    from tests.test_v7_reference_path import fixture
    base = legacy_policy()
    base.update(reference_shares=1000, cost_path_policy=fixture()[2])
    return fr3_plan.policy(base, available_at=CREATED)


def native_plan(data=None, *, context="TREND_PULLBACK", calendar=True):
    data = neutral_bars() if data is None else data
    f, _ = compute_features(select_bars(data, CUTOFF))
    return fr3_plan.make_plan(context, f, reference(), "a"*64,
        bars=data, cutoff=CUTOFF, session_evidence=synthetic_calendar(data) if calendar else None)


def path_fixture():
    old, window, _, options = previous_fixture()
    policy = reference()
    # A fixed source target below the independent net-2R barrier: hitting it
    # must train the new target head, not inherit the old 2R outcome label.
    costs = estimate_reference_costs(old.entry, old.stop, 105., old.reference_shares, policy["cost_path_policy"])
    plan = fr3_plan.FR3ReferencePlan(**dict(asdict(old), policy_version=fr3_plan.PLAN_VERSION,
        target=105., setup_variant=None, cost_up=costs["cost_up_per_share"], cost_stop=costs["cost_stop_per_share"]))
    options["entry_route"] = fr3_plan.bind_entry_route(plan, policy)
    window[1]["open"] = 100.
    window[2].update(open=100., high=105., low=99., close=104.)
    return plan, window, policy["cost_path_policy"], options


def run(items):
    return evaluate_reference_path(items[0], items[1], items[2], **items[3])


@pytest.mark.parametrize("context", ["REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION"])
def test_same_fixed_anchors_precede_rr_for_all_existing_contexts(context):
    plan, audit = native_plan(context=context)
    assert plan is not None, audit
    assert plan.context == context and plan.entry_basis == fr3_plan.ENTRY_BASIS
    assert plan.stop == pytest.approx(98.-.3*plan.atr)
    assert plan.target == 110.
    assert plan.band[1] <= plan.signal_close == 100.
    assert audit["support"]["confirmed_at_session"] < "2026-10-20"
    assert audit["conditional_entry"]["state"] == "UNFILLED"
    assert audit["historical_deployment_claim"] is False
    assert audit["policy_created_at"] == CREATED
    assert fr3_plan.plan_from_record(asdict(plan)) == plan


def test_neutral_without_frozen_context_weights_is_retained_as_exclusion():
    plan, audit = native_plan(context="NO_CLEAR_CONTEXT")
    assert plan is None and audit["reasons"] == ["FR3_NO_TRAINED_CONTEXT_WEIGHTS"]


def test_no_calendar_or_broken_source_support_cannot_be_repaired_by_score():
    plan, audit = native_plan(calendar=False)
    assert plan is None and audit["state"] == "UNKNOWN"
    data = neutral_bars()
    data.loc[len(data)-1, "Low"] = 97.9
    plan, audit = native_plan(data)
    assert plan is None and audit["state"] == "FAIL"
    assert audit["support"]["value"] == 98.


def test_future_bars_cannot_move_the_source_anchors_or_reference_plan():
    data = neutral_bars()
    plan, _ = native_plan(data)
    future = deepcopy(data.iloc[-1:].to_dict("records")[0])
    future.update(Date="2026-10-21", Open=300., High=301., Low=299., Close=300.)
    extended = pd.concat([data, pd.DataFrame([future])], ignore_index=True)
    again, _ = native_plan(extended)
    assert again == plan


def test_forged_atr_or_close_is_not_used_to_shrink_risk():
    data = neutral_bars(); f, _ = compute_features(select_bars(data, CUTOFF))
    f["atr"] *= 2
    plan, audit = fr3_plan.make_plan("TREND_PULLBACK", f, reference(), "a"*64,
        bars=data, cutoff=CUTOFF, session_evidence=synthetic_calendar(data))
    assert plan is None and audit["state"] == "UNKNOWN"


def test_normalized_final_calendar_is_explicitly_retrospective_and_unknowns_block():
    data = neutral_bars(); raw = synthetic_calendar(data)
    days = raw["days"]
    normalized = dict(source="TWSE_OFFICIAL", purpose="SYNTHETIC_TEST_ONLY",
        sessions=sorted(d for d,s in days.items() if s == "OPEN"), days=deepcopy(days),
        evidence_ids=["a"*64, "b"*64],
        availability_by_date={d:d+"T13:30:00+08:00" for d in days},
        availability_policy=fr3_plan.FINAL_CALENDAR_POLICY,
        historical_information_set_certified=False, original_historical_available_at=None,
        source_versions_by_date={d:["c"*64] for d in days})
    observed = list(data.tail(121).Date)
    receipt = fr3_plan.verify_window(normalized, market="TWSE", cutoff=CUTOFF, observed_sessions=observed)
    assert receipt["state"] == "PASS"
    assert receipt["availability_basis"] == "OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_NOT_PIT"
    assert receipt["historical_information_set_certified"] is False
    normalized["days"][observed[10]] = "UNKNOWN"
    assert fr3_plan.verify_window(normalized, market="TWSE", cutoff=CUTOFF,
        observed_sessions=observed)["state"] == "UNKNOWN"


def test_native_preparation_binds_calendar_and_roundtrips_the_new_plan(monkeypatch):
    from domain.les_v7 import training
    from infrastructure.les_v7_scope_training import _pack_row, _unpack_row
    from tests.strategies.test_low_entry_score_v7 import bundle
    data = neutral_bars(); config = bundle()["payload"]
    config["reference_policy"] = reference()
    # Stub only the classifier in this wiring fixture, never source/cost/risk
    # validation; this test does not claim a real eligible market context.
    monkeypatch.setattr(training, "select_context", lambda f,r:("TREND_PULLBACK", ("TREND_PULLBACK",)))
    calendar = synthetic_calendar(data)
    prepared = training.prepare_reference("2330", "TWSE", data, CUTOFF, config, ["a"*64],
        session_evidence=calendar)
    assert isinstance(prepared["plan"], fr3_plan.FR3ReferencePlan), prepared
    assert prepared["feature_audit"]["fr3_reference"]["plan_id"] == prepared["plan"].plan_id
    row = dict(plan=prepared["plan"], features={"close":100., "atr":prepared["features"]["atr"]})
    assert _unpack_row(_pack_row(row)) == row
    missing = training.prepare_reference("2330", "TWSE", data, CUTOFF, config, ["a"*64])
    assert missing["plan"] is None and missing["evidence_id"] != prepared["evidence_id"]


def test_new_reference_memo_is_detached_bounded_and_calendar_bound(monkeypatch):
    from domain.les_v7 import training
    from domain.les_v7.reference_memo import NativeReferenceMemo
    from tests.strategies.test_low_entry_score_v7 import bundle
    data = neutral_bars(); config = bundle()["payload"]
    config["reference_policy"] = reference()
    monkeypatch.setattr(training, "select_context", lambda f,r:("TREND_PULLBACK", ("TREND_PULLBACK",)))
    calendar = synthetic_calendar(data)
    with NativeReferenceMemo(max_entries=2) as memo:
        a = training.prepare_reference("2330", "TWSE", data, CUTOFF, config, ["a"*64], session_evidence=calendar)
        original = deepcopy(a)
        a["features"]["close"] = -123.
        b = training.prepare_reference("2330", "TWSE", data, CUTOFF, config, ["a"*64], session_evidence=calendar)
        assert b == original and memo.stats["hits"] == 1
        missing = training.prepare_reference("2330", "TWSE", data, CUTOFF, config, ["a"*64])
        assert missing["plan"] is None and memo.stats["misses"] == 2
        assert memo.stats["current_entries"] <= 2


def test_small_only_and_exact_risk_thresholds_do_not_rewrite_anchors():
    plan, audit = native_plan()
    before = asdict(plan)
    check = fr3_plan.validate_band(plan, reference())
    assert check["maximum_stop_atr"] == 2.5 and check["minimum_net_rr"] == 1.2
    assert fr3_plan.evaluate_entry(plan, plan.entry, reference(), starter=False)["state"] == "FAIL"
    with pytest.raises(ValueError, match="TACTICAL_SMALL"):
        fr3_plan.bind_entry_route(plan, reference(), strategy_decision="BUY", risk_route="CORE")
    assert asdict(plan) == before


def test_fixed_target_hit_is_not_mislabelled_as_the_old_2r_event():
    items = path_fixture(); before = deepcopy(items)
    result = run(items)
    assert result["status"] == "VERIFIED_FILLED", result
    execution = result["execution"]
    assert execution["exit_reason"] == "FIXED_TARGET" and execution["exit_price"] == 105.
    assert execution["reference_target_first_hit_at"]
    assert execution["net_2r_first_hit_at"] is None
    assert float(execution["executable_net_2r"]) > execution["exit_price"]
    assert execution["label_version"] == fr3_plan.LABEL_VERSION
    label = label_event(items[1], cutoff=items[3]["cutoff"],
        expected_sessions=items[3]["expected_sessions"], plan=items[0], fill=execution)
    assert label["fill"] == label["rebound"] == label["joint"] == 1
    assert label["label_version"] == fr3_plan.LABEL_VERSION
    assert items == before


@pytest.mark.parametrize("opening", [98.9, 100.5])
def test_only_next_open_in_fixed_band_fills_no_intraday_touch_rescue(opening):
    items = path_fixture()
    items[1][1].update(open=opening, high=101., low=98., close=99.5)
    result = run(items)
    assert result["status"] == "VERIFIED_NO_FILL"
    assert result["execution"]["filled"] is False
    assert result["execution"]["conditional_entry"]["intraday_touch_authorizes_fill"] is False
    label = label_event(items[1], cutoff=items[3]["cutoff"],
        expected_sessions=items[3]["expected_sessions"], plan=items[0], fill=result["execution"])
    assert label["fill"] == 0 and label["joint"] is label["rebound"] is None
    assert label["label_version"] == fr3_plan.LABEL_VERSION


def test_opposing_barrier_order_is_unknown_and_old_label_cannot_impersonate_fr3():
    items = path_fixture()
    items[1][2].update(open=100., high=106., low=97., close=100.)
    result = run(items)
    assert result["status"] == "PATH_AMBIGUOUS"
    label = label_event(items[1], cutoff=items[3]["cutoff"],
        expected_sessions=items[3]["expected_sessions"], plan=items[0], fill=result["execution"])
    assert label["status"] == "PATH_AMBIGUOUS" and label["rebound"] is None
    result["execution"].pop("label_version")
    with pytest.raises(ValueError, match="LABEL_BINDING"):
        label_event(items[1], cutoff=items[3]["cutoff"],
            expected_sessions=items[3]["expected_sessions"], plan=items[0], fill=result["execution"])


def test_path_policy_and_rehashed_route_mismatch_remain_blocked():
    items = path_fixture()
    route = items[3]["entry_route"]
    route["expiry"] = "UNTIL_FILLED"
    route["route_id"] = identity({k:v for k,v in route.items() if k != "route_id"})
    assert run(items)["status"] == "DATA_GAP"
    items = path_fixture()
    with pytest.raises(ValueError, match="PATH_VERSION_MISMATCH"):
        evaluate_reference_path(items[0], items[1], previous_fixture()[2], **items[3])
