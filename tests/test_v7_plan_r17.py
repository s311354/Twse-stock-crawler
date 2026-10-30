"""SYNTHETIC R17 structural operands, not official evidence or performance."""
from copy import deepcopy
from dataclasses import asdict

import pandas as pd
import pytest

import domain.strategies  # noqa: F401
from domain.les_v7 import plan_r16
from domain.les_v7.context_r17 import select_context, price_structure_features
from domain.les_v7.plan_r17 import (A_VARIANT, C_VARIANT, assess_early_pullback,
    assess_price_discovery, bind_entry_route, evaluate_entry, make_plan,
    plan_from_record, policy, prepare_selection, validate_band)
from domain.les_v7.policy_r17 import AVAILABLE_AT
from domain.les_v7.rule_baseline import RELAXED_BASELINE_POLICY, baseline_policy, prepare_baseline
from domain.les_v7.support_r17 import assess_floor
from domain.strategies.low_entry_v6_hybrid import select_context as old_context
from tests.test_v7_plan_r16 import pullback_bars, discovery_bars, synthetic_calendar
from tests.strategies.test_v7_context_r14 import CUTOFF


def prepared(data):
    prior = prepare_baseline(data, market="TWSE", cutoff=CUTOFF, policy_version=RELAXED_BASELINE_POLICY)
    initial = select_context(prior["features"])
    selected, audit = prepare_selection(initial, prior["features"], bars=data, cutoff=CUTOFF,
        market="TWSE", session_evidence=synthetic_calendar(data))
    return prior["features"], initial, selected, audit


def reference():
    return policy(baseline_policy(version=RELAXED_BASELINE_POLICY)["reference_policy"], available_at=AVAILABLE_AT)


def native_plan(data):
    features, _, selected, audit = prepared(data)
    _, atr = plan_r16._atr(plan_r16.select_bars(data, CUTOFF))
    return make_plan(selected.context, dict(close=float(data.iloc[-1]["Close"]), atr=atr),
        reference(), "a" * 64, bars=data, cutoff=CUTOFF,
        setup_variant=audit.get("setup_variant"), session_evidence=synthetic_calendar(data))


@pytest.mark.parametrize("soft", ["ema5_slope", "ema5"])
def test_a_known_soft_failure_is_audit_only_without_a_score_bonus(soft):
    data = pullback_bars(); features, _, _, previous = prepared(data)
    impulse = previous["support"]["impulse_high"]
    features = deepcopy(features)
    features[soft] = -.1 if soft == "ema5_slope" else 9999.
    audit = assess_early_pullback(data, features, cutoff=CUTOFF, impulse_high=impulse,
        market="TWSE", session_evidence=synthetic_calendar(data))
    assert audit["state"] == "PASS" and audit["setup_variant"] == A_VARIANT
    assert audit["soft_warnings"]
    assert all(value is True for value in audit["hard_checks"].values())
    assert audit["soft_effect"] == "EXISTING_FACTOR_LEDGER_ONLY_NO_EXTRA_POINTS_OR_BOOLEAN_VETO"
    assert not {"score", "bonus", "earned_points"} & set(audit)


def test_a_missing_required_operand_is_still_unknown():
    data = pullback_bars(); features, _, _, previous = prepared(data)
    features = deepcopy(features); features["ema5_slope"] = None
    audit = assess_early_pullback(data, features, cutoff=CUTOFF,
        impulse_high=previous["support"]["impulse_high"],
        market="TWSE", session_evidence=synthetic_calendar(data))
    assert audit["state"] == "UNKNOWN" and audit["setup_variant"] is None
    assert "POSITIVE_EMA5_SLOPE" in audit["reasons"]


def test_c_floor_does_not_require_r15_ema_rs_volume_pattern_to_confirm_source():
    data = discovery_bars(); features, _, _, _ = prepared(data)
    changed = deepcopy(features)
    changed.update(ema10=0., ema20=9999., ema20_slope=-1., rs20=-1., rs_slope5=-1.)
    # Keep actual market operands known, but well below the old participation
    # thresholds. The price/floor/cutoff facts, target and costs are not altered.
    data.loc[len(data) - 1, ["Volume", "Amount"]] = [5000., 400000.]
    audit = assess_price_discovery(data, changed, cutoff=CUTOFF,
        market="TWSE", session_evidence=synthetic_calendar(data))
    assert audit["state"] == "PASS" and audit["setup_variant"] == C_VARIANT
    assert audit["support_audit"]["state"] == "PASS"
    assert "EMA_UPTREND" in audit["soft_warnings"] and "POSITIVE_RS20" in audit["soft_warnings"]
    assert "VOLUME_PARTICIPATION" in audit["soft_warnings"]
    assert audit["metrics"]["target_multiple"] == .5


def test_known_no_true120_breakout_is_still_structural_failure():
    data = discovery_bars(); data.loc[len(data) - 110, "High"] = 100.
    features, _, _, _ = prepared(data)
    audit = assess_price_discovery(data, features, cutoff=CUTOFF,
        market="TWSE", session_evidence=synthetic_calendar(data))
    assert audit["state"] == "FAIL" and audit["hard_checks"]["ABOVE_PRIOR_120_HIGH"] is False


def test_source_floor_confirmation_does_not_include_the_signal_or_future():
    data = discovery_bars()
    # Only one prior close is above the frozen confirmation threshold.
    for index in (len(data) - 5, len(data) - 4, len(data) - 3):
        data.loc[index, ["Open", "Close"]] = 80.1
        data.loc[index, "Low"] = min(float(data.loc[index, "Low"]), 80.1)
    floor = assess_floor(data, cutoff=CUTOFF, market="TWSE", session_evidence=synthetic_calendar(data))
    assert floor["state"] == "FAIL" and not floor["checks"]["TWO_SUBSEQUENT_PRIOR_CLOSES"]
    future = data.tail(1).copy(); future["Date"] = "2026-10-21"
    future[["Open", "High", "Low", "Close"]] = [999., 999., 1., 999.]
    assert assess_floor(pd.concat((data, future), ignore_index=True), cutoff=CUTOFF,
        market="TWSE", session_evidence=synthetic_calendar(data)) == floor


def test_missing_official_window_cannot_be_covered_by_soft_scores():
    data = discovery_bars(); features, initial, _, _ = prepared(data)
    selected, audit = prepare_selection(initial, features, bars=data, cutoff=CUTOFF, market="TWSE")
    assert audit["state"] == "UNKNOWN" and audit["setup_variant"] is None
    assert audit["hard_checks"]["FULL_PRIOR_120"] is None


def test_actual_breakout_is_not_vetoed_by_already_scored_ema_or_base_strength():
    data = discovery_bars(); features, _, _, _ = prepared(data)
    changed = deepcopy(features)
    changed.update(ema5_slope=-1., ema20_slope=-1., base_compression=False,
        downside_resilience=False, breakout_quality=False)
    assert old_context(changed).context != "LEADERSHIP_CONTINUATION"
    selected = select_context(changed)
    assert selected.context == "LEADERSHIP_CONTINUATION"
    assert dict(dict(selected.evidence)["LEADERSHIP_CONTINUATION"])["ACTUAL_BREAKOUT_OR_VERIFIED_RETEST"] is True
    changed["support_intact"] = False
    assert select_context(changed).context == "NO_CLEAR_CONTEXT"


def test_conditional_plan_uses_original_band_stop_fees_and_risk_math():
    data = pullback_bars()
    plan, sidecar = native_plan(data)
    assert plan.setup_variant == A_VARIANT
    assert plan.signal_close == float(data.iloc[-1]["Close"]) and plan.entry == plan.band[1]
    assert plan_from_record(asdict(plan)) == plan
    observed = evaluate_entry(plan, plan.entry, reference(), starter=True)
    assert observed["maximum_stop_atr"] == 2. and observed["minimum_net_rr"] == 1.3
    assert sidecar["anchors"]["stop_buffer_atr"] == .3
    assert validate_band(plan, reference(), starter=True)["state"] == "PASS"
    route = bind_entry_route(plan, reference(), strategy_decision="BUY_SMALL", risk_route="TACTICAL")
    assert route["setup_variant"] == A_VARIANT and route["plan_id"] == plan.plan_id
    with pytest.raises(ValueError):
        bind_entry_route(plan, reference(), strategy_decision="BUY", risk_route="CORE")


def test_softened_structure_does_not_raise_target_to_repair_failed_rr():
    data = discovery_bars()
    plan, sidecar = native_plan(data)
    assert plan.setup_variant == C_VARIANT and plan.target == pytest.approx(83.2)
    assert sidecar["anchors"]["target_projection"]["multiple"] == .5
    assert validate_band(plan, reference(), starter=True)["state"] == "FAIL"
    with pytest.raises(ValueError, match="RISK_NOT_PASS"):
        bind_entry_route(plan, reference(), strategy_decision="BUY_SMALL", risk_route="TACTICAL")


def test_fixed_ledger_budget_not_augmented_by_soft_confirmation_audit():
    from domain.les_v7.rule_score_r4 import score_factors_r4
    from domain.les_v7.rule_baseline import CONFIG
    data = pullback_bars(); features, _, selected, _ = prepared(data)
    plan, _ = native_plan(data)
    result = score_factors_r4(features, selected, plan,
        dict(spread=.001, adv=1e8), config=CONFIG, policy=baseline_policy(version=RELAXED_BASELINE_POLICY))
    ledger = result["score_evidence"]
    assert sum(row["weight"] for row in ledger["conditions"]) == pytest.approx(100.)
    assert sum(row["earned_points"] for row in ledger["conditions"]) == pytest.approx(ledger["formal_score"])
    assert all("r17" not in row["id"].lower() for row in ledger["conditions"])


def price_retest_bars():
    n = 100
    data = pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-20", periods=n).strftime("%Y-%m-%d"),
        Open=[100.] * n, High=[101.] * n, Low=[99.] * n, Close=[100.] * n, Volume=[10000.] * n))
    data.loc[n - 3, ["Open", "High", "Low", "Close"]] = [101.5, 102.5, 100.5, 102.]
    data.loc[n - 2, ["Open", "High", "Low", "Close"]] = [101.2, 101.5, 100.8, 101.4]
    data.loc[n - 1, ["Open", "High", "Low", "Close"]] = [101.3, 101.7, 100.9, 101.5]
    return data


def price_retest_features():
    return dict(history_ready=True, market_ready=True, support_intact=True,
        support=99., low=100.9, close=101.5, hh20=102.5, position60=.7,
        no_new_low2=True, drawdown20=-.01, drawdown60=-.01,
        breakout_retest_confirmed=False, breakout_retest_support=None,
        breakout_retest_level=None, breakout_retest_age=None,
        ema20_slope=-1., clv=-1., rs20=-1., rs_slope5=-1., volume_ratio=.5)


def test_price_only_retest_is_independent_of_soft_strength_and_preserves_old_scoring_fields():
    data, features = price_retest_bars(), price_retest_features()
    before = deepcopy(features)
    first = price_structure_features(data, features, cutoff=CUTOFF)
    changed = dict(features, ema20_slope=999., clv=1., rs20=999., rs_slope5=999., volume_ratio=999.,
        breakout_retest_confirmed=True)
    second = price_structure_features(data, changed, cutoff=CUTOFF)
    assert first["r17_price_retest_confirmed"] is True
    assert first["r17_price_retest_trace"] == second["r17_price_retest_trace"]
    assert select_context(first).context == "LEADERSHIP_CONTINUATION"
    assert features == before
    assert {key: first[key] for key in before} == before
    assert first["breakout_retest_confirmed"] is False  # old factor source not rewritten
    trace = first["r17_price_retest_trace"]["current"]
    assert trace["breakout_session"] == data.iloc[-3]["Date"] and trace["session"] == data.iloc[-1]["Date"]
    assert trace["age"] == 2 and trace["support"] == 100.8


def test_old_aggregated_retest_true_does_not_supply_missing_r17_evidence():
    features = dict(price_retest_features(), breakout_retest_confirmed=True)
    assert select_context(features).context != "LEADERSHIP_CONTINUATION"
    selected = select_context(features).record()["evidence"]["LEADERSHIP_CONTINUATION"]
    assert selected["ACTUAL_BREAKOUT_OR_VERIFIED_RETEST"] is None


def test_price_retest_future_bar_is_excluded_before_any_state_or_hash():
    data, features = price_retest_bars(), price_retest_features()
    first = price_structure_features(data, features, cutoff=CUTOFF)
    future = data.tail(1).copy(); future["Date"] = "2026-10-21"
    future[["Open", "High", "Low", "Close", "Volume"]] = [999., 1000., 1., 999., 999999.]
    assert price_structure_features(pd.concat((data, future), ignore_index=True), features,
        cutoff=CUTOFF) == first


def test_zero_trade_or_missing_bar_resets_price_event_without_carrying_confirmation():
    data, features = price_retest_bars(), price_retest_features()
    data.loc[len(data) - 2, "Volume"] = 0.
    result = price_structure_features(data, features, cutoff=CUTOFF)
    assert result["r17_price_retest_confirmed"] is None
    assert result["r17_price_retest_trace"]["current"] is None
    assert result["r17_price_retest_trace"]["resets"][-1]["reason"] == "INVALID_BAR_OR_HISTORY_RESET"
    data = price_retest_bars(); data.loc[len(data) - 2, "High"] = float("nan")
    result = price_structure_features(data, features, cutoff=CUTOFF)
    assert result["r17_price_retest_confirmed"] is None
    assert result["r17_price_retest_trace"]["state"] == "UNKNOWN"


def test_price_retest_uses_same_v6_wilder_atr_seed_and_frozen_breakout_band():
    from domain.strategies.low_entry_v6_features import _wilder
    data = price_retest_bars()
    close, high, low = data["Close"], data["High"], data["Low"]
    previous = close.shift(1)
    atr = _wilder(pd.concat((high - low, (high - previous).abs(),
        (low - previous).abs()), axis=1).max(axis=1))
    result = price_structure_features(data, price_retest_features(), cutoff=CUTOFF)
    current = result["r17_price_retest_trace"]["current"]
    assert current["frozen_atr"] == atr.iloc[-3]
    assert current["band"] == [101. - .3 * atr.iloc[-3], 101. + .5 * atr.iloc[-3]]
