"""SYNTHETIC causal structures only; not official sources or fitted profits."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

import domain.strategies  # noqa: F401 -- existing plugin registration
from domain.les_v7.contracts import identity
from domain.les_v7.plan_r16 import (A_VARIANT, C_VARIANT, A_ENTRY_BASIS, CLOSE_ENTRY_BASIS,
    PROJECTED_TARGET, OBSERVED_TARGET, R16ReferencePlan, assess_early_pullback,
    assess_price_discovery, bind_entry_route, make_plan, plan_from_record, policy,
    prepare_selection, validate_band)
from domain.les_v7.rule_baseline import (RELAXED_BASELINE_POLICY, baseline_policy,
    prepare_baseline, select_baseline_context)
from tests.strategies.test_v7_context_r14 import source_bars, CUTOFF


def pullback_bars():
    n = 150
    close = np.linspace(85., 102., n)
    data = pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-20", periods=n).strftime("%Y-%m-%d"),
        Open=close, High=close + 1, Low=close - 1, Close=close, Volume=[10000.] * n,
        Amount=close * 10000, Trades=[100] * n, Bid=close - .1, Ask=close,
        **{"TAIEX Close": [1000.] * n}, Complete=[True] * n))
    data.loc[n - 20, "Low"] = 95.
    data.loc[n - 8, ["Open", "High", "Low", "Close"]] = [104., 109., 103., 104.]
    for index in range(n - 6, n - 1):
        data.loc[index, ["Open", "High", "Low", "Close"]] = [101.5, 102., 100., 101.5]
    data.loc[n - 1, ["Open", "High", "Low", "Close", "Volume", "Amount"]] = [102., 103., 100.5, 102.5, 20000., 2050000.]
    return data


def discovery_bars():
    data = source_bars()
    data = pd.concat((data.iloc[:30], data), ignore_index=True)
    data["Date"] = pd.bdate_range(end="2026-10-20", periods=len(data)).strftime("%Y-%m-%d")
    data.loc[data["High"] == 90., "High"] = 81.
    data.loc[len(data) - 40, "Low"] = 78.
    return data


def synthetic_calendar(data):
    """Explicit synthetic original/proof; never real exchange coverage."""
    from tests.test_research_archive import proof
    annual = dict(stat="ok", fields=["日期", "名稱", "說明"],
        data=[["2026-01-01", "元旦休市", "休市"]])
    binding = dict(kind="ANNUAL_CALENDAR", market="TWSE", year=2026, content_sha256=identity(annual))
    days = pd.date_range(start=data.iloc[0]["Date"], end="2026-10-20")
    return dict(source="TWSE_OFFICIAL", purpose="SYNTHETIC_TEST_ONLY",
        days={str(day.date()): "OPEN" if day.weekday() < 5 else "CLOSED" for day in days},
        documents={"2026": dict(payload=annual, publication_proof=proof(binding, publication="2025-12-01"))},
        announcements=[])


def prepared(data):
    value = prepare_baseline(data, market="TWSE", cutoff=CUTOFF, policy_version=RELAXED_BASELINE_POLICY)
    initial = select_baseline_context(value["features"], policy_version=RELAXED_BASELINE_POLICY)
    selected, audit = prepare_selection(initial, value["features"], bars=data, cutoff=CUTOFF,
        session_evidence=synthetic_calendar(data), market="TWSE")
    return value, initial, selected, audit


def reference():
    from domain.les_v7.policy_r16 import AVAILABLE_AT
    return policy(baseline_policy(version=RELAXED_BASELINE_POLICY)["reference_policy"], available_at=AVAILABLE_AT)


def native_plan(data):
    value, initial, selected, audit = prepared(data)
    # Same causal native Wilder ATR, not a price changed to fit RR.
    feature = dict(close=float(data.iloc[-1]["Close"]), atr=audit["metrics"]["signal_atr"])
    return make_plan(selected.context, feature, reference(), "a" * 64, bars=data,
        cutoff=CUTOFF, setup_variant=audit.get("setup_variant"), session_evidence=synthetic_calendar(data))


def test_early_floor_uses_exact_gap_earliest_tie_and_two_prior_consecutive_closes():
    data = pullback_bars()
    _, initial, selected, audit = prepared(data)
    assert initial.context == selected.context == "TREND_PULLBACK"
    assert audit["original_support_missing"] == "SAME_PULLBACK_CONFIRMED_LOW_MISSING"
    assert audit["setup_variant"] == A_VARIANT and audit["state"] == "PASS"
    assert audit["support"]["formation_session"] == data.iloc[-6]["Date"]
    assert audit["support"]["confirmation_sessions"] == data.iloc[-5:-3]["Date"].tolist()
    assert data.iloc[-1]["Date"] not in audit["support"]["confirmation_sessions"]
    assert audit["support"]["formation_session"] > audit["support"]["impulse_high"]["session"]


def test_conditional_plan_keeps_true_close_and_binds_whole_fixed_band():
    plan, sidecar = native_plan(pullback_bars())
    assert isinstance(plan, R16ReferencePlan) and plan.entry_basis == A_ENTRY_BASIS
    assert plan.signal_close == 102.5 and plan.entry == plan.band[1] != plan.signal_close
    assert plan.target_method == OBSERVED_TARGET and plan.target == 109.
    assert sidecar["conditional_entry"]["state"] == "UNFILLED"
    assert plan_from_record(asdict(plan)) == plan
    assert validate_band(plan, reference(), starter=True)["state"] == "PASS"
    route = bind_entry_route(plan, reference(), strategy_decision="BUY_SMALL", risk_route="TACTICAL", setup_variant=A_VARIANT)
    assert route["entry_rule"] == "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND"
    assert route["expiry"] == "NEXT_SESSION_CLOSE" and route["signal_close"] == 102.5
    with pytest.raises(ValueError, match="TACTICAL_SMALL"):
        bind_entry_route(plan, reference(), strategy_decision="BUY", risk_route="CORE")


def test_signal_bar_cannot_provide_second_confirmation_or_replace_prior_atr():
    data = pullback_bars()
    _, _, _, original = prepared(data)
    impulse = original["support"]["impulse_high"]
    # Known first/third confirmations do not form a consecutive pair.
    data.loc[len(data) - 4, ["Open", "Close"]] = 100.1
    data.loc[len(data) - 2, ["Open", "Close"]] = 100.1
    p = prepare_baseline(data, market="TWSE", cutoff=CUTOFF, policy_version=RELAXED_BASELINE_POLICY)
    audit = assess_early_pullback(data, p["features"], cutoff=CUTOFF, impulse_high=impulse)
    assert audit["checks"]["TWO_CONSECUTIVE_PRIOR_CLOSES"] is False
    prior = audit["metrics"]["prior_atr"]
    data.loc[len(data) - 1, "High"] = 1000.
    again = assess_early_pullback(data, p["features"], cutoff=CUTOFF, impulse_high=impulse)
    assert again["metrics"]["prior_atr"] == prior


def test_failed_a_never_falls_through_to_price_discovery():
    data = pullback_bars(); data.loc[len(data) - 1, "Amount"] = 1.
    _, initial, selected, audit = prepared(data)
    assert selected == initial
    assert audit["candidate_variant"] == A_VARIANT and audit["state"] == "FAIL"
    assert audit["setup_variant"] is None
    assert audit["checks"]["AMOUNT_PARTICIPATION"] is False


def test_missing_amount_is_unknown_not_a_volume_replacement():
    data = pullback_bars().drop(columns="Amount")
    _, _, _, audit = prepared(data)
    assert audit["state"] == "UNKNOWN" and audit["checks"]["AMOUNT_PARTICIPATION"] is None


def test_a_missing_official_session_evidence_is_not_complete_by_bar_count():
    data = pullback_bars()
    value, initial, _, _ = prepared(data)
    _, audit = prepare_selection(initial, value["features"], bars=data, cutoff=CUTOFF, market="TWSE")
    assert audit["state"] == "UNKNOWN"
    assert audit["checks"]["FULL_PRIOR_20"] is None
    assert audit["session_coverage"]["reasons"] == ["R16_OFFICIAL_SESSION_EVIDENCE_MISSING"]


@pytest.mark.parametrize("field", ["ema5", "ema5_slope", "close"])
def test_unavailable_operands_do_not_use_stale_values(field):
    data = pullback_bars(); value, initial, _, _ = prepared(data)
    features = deepcopy(value["features"])
    features["score_availability"][field] = False
    _, audit = prepare_selection(initial, features, bars=data, cutoff=CUTOFF)
    assert audit["state"] == "UNKNOWN" and audit["setup_variant"] is None


def test_original_source_support_and_resistance_are_preserved_before_risk():
    data = pullback_bars()
    # This confirmed low is deliberately far from price; no floor replacement.
    data.loc[len(data) - 5, "Low"] = 97.
    _, initial, selected, audit = prepared(data)
    assert initial.context == selected.context == "TREND_PULLBACK"
    assert audit["state"] == "NOT_APPLICABLE"
    assert audit["original_support"]["value"] == 97.
    assert audit["setup_variant"] is None


def test_price_discovery_has_full_prior120_and_explicit_fixed_projection():
    data = discovery_bars()
    _, initial, selected, audit = prepared(data)
    assert initial.context == "NO_CLEAR_CONTEXT" and selected.context == "LEADERSHIP_CONTINUATION"
    assert audit["setup_variant"] == C_VARIANT and audit["state"] == "PASS"
    assert audit["metrics"]["prior120_high"] < float(data.iloc[-1]["Close"])
    assert audit["metrics"]["target"] == pytest.approx(81.8 + .5 * (81.8 - 79.))
    plan, sidecar = native_plan(data)
    assert plan.entry_basis == CLOSE_ENTRY_BASIS and plan.signal_close == plan.entry
    assert plan.target_method == PROJECTED_TARGET
    assert sidecar["anchors"]["resistance"] is None
    assert sidecar["anchors"]["target_projection"]["multiple"] == .5
    # The projection exists independently of whether the fee-adjusted RR works.
    assert validate_band(plan, reference(), starter=True)["state"] == "FAIL"


def test_short_history_cannot_be_called_price_discovery():
    data = discovery_bars().tail(120).reset_index(drop=True)
    value = prepare_baseline(data, market="TWSE", cutoff=CUTOFF, policy_version=RELAXED_BASELINE_POLICY)
    audit = assess_price_discovery(data, value["features"], cutoff=CUTOFF)
    assert audit["state"] == "UNKNOWN"
    assert audit["reasons"] == ["R16_VERIFIED_PRIOR_120_SESSION_HISTORY_MISSING"]


def test_observed_row_count_is_not_official_calendar_completeness():
    data = discovery_bars()
    value, initial, _, _ = prepared(data)
    selected, audit = prepare_selection(initial, value["features"], bars=data, cutoff=CUTOFF,
        session_evidence=None, market="TWSE")
    assert len(data) >= 121 and selected == initial
    assert audit["state"] == "UNKNOWN" and audit["setup_variant"] is None
    assert audit["checks"]["FULL_PRIOR_120"] is None
    assert audit["session_coverage"]["state"] == "UNKNOWN"


def test_discovery_does_not_replace_an_existing_strict_support_with_nearer_floor():
    data = discovery_bars()
    data.loc[len(data) - 15, "Low"] = 78.5
    _, initial, selected, audit = prepared(data)
    assert initial.context == "NO_CLEAR_CONTEXT" and selected.context == "LEADERSHIP_CONTINUATION"
    assert audit["setup_variant"] == C_VARIANT
    assert audit["support"]["value"] == 78.5
    assert audit["support_method"] == "CONFIRMED_20_SESSION_BASE_FLOOR"
    plan, _ = native_plan(data)
    assert plan.stop == pytest.approx(78.5 - .3 * plan.atr)
    assert validate_band(plan, reference(), starter=True)["state"] == "FAIL"


@pytest.mark.parametrize("damage", ["old_high", "amount", "clv", "trend", "rs"])
def test_known_false_discovery_dependency_is_not_pass(damage):
    data = discovery_bars()
    if damage == "old_high": data.loc[len(data) - 110, "High"] = 83.
    if damage == "amount": data.loc[len(data) - 1, "Amount"] = 100.
    if damage == "clv": data.loc[len(data) - 1, "High"] = 83.
    value, _, _, _ = prepared(data)
    features = deepcopy(value["features"])
    if damage == "trend": features["ema20_slope"] = 0.
    if damage == "rs": features["rs20"] = 0.
    audit = assess_price_discovery(data, features, cutoff=CUTOFF,
        session_evidence=synthetic_calendar(data), market="TWSE")
    assert audit["state"] == "FAIL" and audit["setup_variant"] is None


@pytest.mark.parametrize("factory", [pullback_bars, discovery_bars])
def test_future_bars_and_mutable_audits_cannot_contaminate_selection(factory):
    data = factory()
    value, initial, selected, audit = prepared(data)
    future = data.tail(1).copy(); future["Date"] = "2026-10-21"
    future[["Open", "High", "Low", "Close"]] = [999., 1000., 1., 999.]
    more = pd.concat((data, future), ignore_index=True)
    again, future_audit = prepare_selection(initial, value["features"], bars=more, cutoff=CUTOFF,
        session_evidence=synthetic_calendar(data), market="TWSE")
    assert again == selected and future_audit == audit
    saved = deepcopy(audit); audit["support"]["value"] = 99999.
    _, rechecked = prepare_selection(initial, value["features"], bars=data, cutoff=CUTOFF,
        session_evidence=synthetic_calendar(data), market="TWSE")
    assert rechecked == saved


def test_unknown_subtype_and_tampered_entry_basis_rejected():
    plan, _ = native_plan(pullback_bars())
    with pytest.raises(ValueError, match="NOT_RECOGNIZED"):
        replace(plan, setup_variant="CALLER_INVENTED")
    with pytest.raises(ValueError, match="ENTRY_BASIS"):
        replace(plan, entry_basis=CLOSE_ENTRY_BASIS)
    with pytest.raises(ValueError, match="STRUCTURE_ID"):
        replace(plan, structure_id="x" * 64)
    assert plan.structure_id != identity({})
