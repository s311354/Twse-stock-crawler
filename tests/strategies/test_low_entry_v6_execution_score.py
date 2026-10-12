"""Synthetic common-scale fixtures; never historical proof or execution authority."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from domain.strategies.low_entry_v6_execution_score import (
    DEPENDENCIES, EXECUTION_RULES, execution_conditions, validate_score_budget, POLICY,
)
from domain.strategies.low_entry_v6_decision import decision_from_score
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.strategies.low_entry_v6_signals import (
    SETUPS, LowEntryV6Config, checked_stop_distances, decide, score_setups, trade_plan,
)
from infrastructure.report.low_entry_v6_renderer import setup_score_summary
from tests.strategies.test_low_entry_score_v6 import features
from tests.strategies.test_low_entry_v6_unified_decision import evaluate, unified_report, verified_quality

# Isolate the shared execution scale from the later PIT leadership slot replacements.
CONFIG = LowEntryV6Config(leadership_policy_version=1)


def scored(f, config=CONFIG):
    return score_setups(f, trade_plan(f, config=config), config=config)


def execution_ledger(setup):
    return [c for c in setup["score_evidence"]["conditions"] if c["group"] == "execution"]


def target_for_net_rr(rr, entry=100.4, stop=98.4):
    # Synthetic boundary fixture only, not a resistance-producing policy.
    cost = entry * (1 + CONFIG.buy_fee_rate)
    risk = cost - stop * (1 - CONFIG.sell_cost_rate)
    return (cost + rr * risk) / (1 - CONFIG.sell_cost_rate)


@pytest.mark.parametrize("rr,points", [(1.7, 0), (1.8, 0), (1.99, 0), (2., 3), (2.01, 3)])
def test_all_setups_share_net_rr_threshold_and_ten_point_scale(rr, points):
    f = features(swing_high=target_for_net_rr(rr), hh20=120., hh60=130.)
    result = scored(f)
    ledgers = [execution_ledger(result[name]) for name in SETUPS]
    assert ledgers[0] == ledgers[1] == ledgers[2]
    for setup in result.values():
        execution = setup["execution_score"]
        assert execution["policy_version"] == POLICY
        assert execution["total_weight"] == execution["available_weight"] == 10
        assert execution["earned_points"] == 7 + points
        assert execution["measured"]["net_reward_risk"] == pytest.approx(rr)
        assert execution["limits"]["minimum_net_rr"] == 2
        assert [c["weight"] for c in execution_ledger(setup)] == [4, 3, 1, 2]
        assert setup["score_evidence"]["total_weight"] == 100


def test_costs_can_fail_rr_even_when_gross_rr_is_two_and_cached_flags_lie():
    f = features(swing_high=104.4, hh20=120., hh60=130.)
    plan = trade_plan(f, config=CONFIG)
    assert plan["reward_risk"] == pytest.approx(2.)
    plan.update(net_reward_risk=999, spread_pass=False, amount_pass=False)
    conditions, detail = execution_conditions(f, plan, distances=checked_stop_distances(plan, f["atr14"]), config=CONFIG)
    assert conditions == [(True, 4), (False, 3), (True, 1), (True, 2)]
    assert detail["measured"]["net_reward_risk"] < 2


@pytest.mark.parametrize("updates,expected", [
    ({"support": 97.4}, 10),  # Stop=96.8; exactly 1.8 ATR.
    ({"support": 97.39}, 6),
    ({"atr14": 10., "support": 95.368}, 10),  # Exactly 8% price risk.
    ({"atr14": 10., "support": 95.36}, 6),
    ({"spread": .01}, 10), ({"spread": .01001}, 9), ({"spread": -1}, 9),
    ({"amount_ma20": 30_000_000.}, 10), ({"amount_ma20": 29_999_999.}, 8),
    ({"spread": None}, 9), ({"amount_ma20": None}, 8),
])
def test_shared_stop_spread_and_liquidity_boundaries(updates, expected):
    result = scored(features(swing_high=150., hh20=160., hh60=170., **updates))
    assert [s["execution_score"]["earned_points"] for s in result.values()] == [expected] * 3


def test_masks_and_missing_quotes_remove_coverage_not_manufacture_points():
    result = scored(features(score_availability={"spread": False, "amount_ma20": False}))
    for setup in result.values():
        assert setup["execution_score"]["earned_points"] == 7
        assert setup["execution_score"]["available_weight"] == 7
        assert setup["score_coverage"] == 97
    result = scored(features(spread=None, amount_ma20=None,
                             score_availability={"spread": True, "amount_ma20": True}))
    assert all(s["execution_score"]["available_weight"] == 7 for s in result.values())


def test_net_reward_requires_unmasked_native_resistance_evidence():
    f = features(score_availability={"swing_high": False})
    result = scored(f)
    for setup in result.values():
        rr = execution_ledger(setup)[1]
        assert not rr["observed"] and rr["earned_points"] == 0


def test_cost_exceeds_upside_is_known_failed_reward_not_unknown():
    result = scored(features(swing_high=100.41, hh20=120., hh60=130.))
    for setup in result.values():
        execution = setup["execution_score"]
        assert execution["payoff_status"] == "COST_EXCEEDS_TARGET_UPSIDE"
        assert execution["measured"]["net_reward_risk"] is None
        assert execution_ledger(setup)[1]["observed"]
        assert execution["available_weight"] == 10 and execution["earned_points"] == 7


def test_structure_and_legacy_execution_preserved_no_extra_positive_budget():
    f = features(swing_high=target_for_net_rr(1.9), hh20=120., hh60=130., breakout_extension=.5)
    old = scored(f, replace(CONFIG, decision_policy_version=1))
    new = scored(f)
    assert [old[n]["components"]["execution"] for n in SETUPS] == [7, 10, 10]
    assert [new[n]["components"]["execution"] for n in SETUPS] == [7, 7, 7]
    for name in SETUPS:
        assert {k: v for k, v in old[name]["components"].items() if k != "execution"} == {
            k: v for k, v in new[name]["components"].items() if k != "execution"}
        assert new[name]["score_evidence"]["total_weight"] == 100


def test_each_setup_uses_its_causal_anchor_without_shortening_for_points():
    f = features(support=90., consolidation_support=99., consolidation_confirmed=True,
                 consolidation_sessions=3, breakout_retest_support=98.8,
                 breakout_retest_level=99., breakout_retest_age=2, breakout_retest_confirmed=True)
    original = deepcopy(f)
    result = scored(f)
    for name in SETUPS:
        plan = trade_plan(f, config=CONFIG, setup=name)
        assert plan["stop_source"] == ("CONFIRMED_BREAKOUT_RETEST" if name == SETUPS[2]
                                       else "CONFIRMED_CONSOLIDATION_LOW")
        assert result[name]["execution_score"]["measured"]["stop_distance_atr"] == plan["stop_distance_atr"]
    assert f == original


def test_price_discovery_keeps_null_rr_and_no_reallocated_rr_points():
    result = evaluate(features(swing_high=None, hh20=100.2, hh60=100.3, breakout_extension=.5))
    assert result["setup"] == "LEADERSHIP_CONTINUATION"
    assert result["rr_policy"] == "PRICE_DISCOVERY_EXTENSION_SUPPORT"
    assert result["plan"]["target_price"] is result["plan"]["net_reward_risk"] is None
    assert result["execution_score"]["earned_points"] == 7
    assert result["execution_score"]["available_weight"] == 7
    assert result["raw_score"] == 97  # Missing RR is not renormalized to 100.
    assert "價格發現" in setup_score_summary({"decision": result})
    unknown = evaluate(features(swing_high=None, hh20=None, hh60=None))
    assert unknown["signal"] not in {"BUY", "BUY_SMALL"}
    assert "NET_RR_UNAVAILABLE" in unknown["reasons"]


def test_selection_reveals_lower_scoring_but_risk_prioritized_setup():
    result = evaluate(features(down_volume_contracts=False))
    assert result["setup"] == "TREND_PULLBACK" and result["score"] == 96
    selection = result["selection"]
    assert selection["highest_score_setups"] == ["REVERSAL"]
    assert selection["highest_final_score"] == 100
    assert not selection["selected_is_highest_score"]
    assert selection["selected_setup"] == result["score_setup"]
    rendered = setup_score_summary({"decision": result})
    assert "選定型態：TREND_PULLBACK" in rendered
    assert "最高 Final：REVERSAL 100.00" in rendered
    assert "不一定取最高分" in rendered and "共用執行風控 10.00/10" in rendered
    assert json.loads(json.dumps(result))["selection"] == selection


def test_nonexecutable_selection_discloses_diagnostic_ranking_not_admission():
    result = evaluate(features(restricted=None))
    assert result["selection"]["mode"] == "DIAGNOSTIC_FINAL_SCORE"
    assert result["signal"] == "WATCH"
    assert "診斷型態" in setup_score_summary({"decision": result})


@pytest.mark.parametrize("field", ["selection", "execution_score"])
def test_report_validation_recomputes_selection_and_execution_details(field):
    report = unified_report()
    assert report_failure(report) is None
    report["decision"][field] = {}
    assert entry_gate_checks(report)["score"] == "FAIL"
    assert report_failure(report) == "V6_SCORE_OR_SETUP_MISMATCH"


def test_execution_weights_and_evidence_dependencies_share_an_immutable_contract():
    assert [(r.name, r.weight) for r in EXECUTION_RULES] == [
        ("stop", 4), ("net_rr", 3), ("spread", 1), ("liquidity", 2)]
    assert DEPENDENCIES == tuple(r.dependencies for r in EXECUTION_RULES)
    with pytest.raises(FrozenInstanceError):
        EXECUTION_RULES[0].weight = 5


@pytest.mark.parametrize("leadership_policy", [1, 2])
@pytest.mark.parametrize("overrides", [
    {}, {"buy_maximum_stop_atr": 1.2, "starter_maximum_stop_fraction": .01},
    {"maximum_spread": .001, "minimum_amount_ma20": 100_000_000.},
    {"buy_fee_rate": .002, "sell_cost_rate": .006},
])
def test_current_setups_share_configured_scale_without_changing_structure(leadership_policy, overrides):
    config = replace(LowEntryV6Config(leadership_policy_version=leadership_policy), **overrides)
    f = features(swing_high=target_for_net_rr(2.01), hh20=120., hh60=130.)
    source = deepcopy(f)
    result = scored(f, config)
    details = [s["execution_score"] for s in result.values()]
    ledgers = [execution_ledger(s) for s in result.values()]
    assert details[0] == details[1] == details[2]
    assert ledgers[0] == ledgers[1] == ledgers[2]
    assert details[0]["limits"]["maximum_stop_atr"] == min(
        config.buy_maximum_stop_atr, config.starter_maximum_stop_atr)
    assert details[0]["limits"]["maximum_stop_fraction"] == min(
        config.buy_maximum_stop_fraction, config.starter_maximum_stop_fraction)
    assert details[0]["limits"]["minimum_amount_ma20"] == config.minimum_amount_ma20
    assert details[0]["limits"]["maximum_spread"] == config.maximum_spread
    assert details[0]["buy_cost_rate"] == config.buy_fee_rate
    assert details[0]["sell_cost_rate"] == config.sell_cost_rate
    for setup in result.values():
        evidence = setup["score_evidence"]
        assert evidence["total_weight"] == 100
        assert sum(c["weight"] for c in evidence["conditions"] if c["group"] != "execution") == 90
        assert evidence["formal_score"] == evidence["earned_points"]
    assert f == source


@pytest.mark.parametrize("weights", [(4, 2, 2, 2), (4, 4, 1, 1), (4, 4, 1, 2)])
def test_scale_drift_rejected_even_with_total_100_and_no_observed_evidence(weights):
    groups = {"structure": [(None, 100 - sum(weights))],
              "execution": [(None, weight) for weight in weights]}
    assert sum(weight for tests in groups.values() for _, weight in tests) == 100
    with pytest.raises(ValueError, match="LES_EXECUTION_SCORE_SCALE_MISMATCH"):
        validate_score_budget(groups)


@pytest.mark.parametrize("weight,error", [
    (89, "STRUCTURE_BUDGET_MUST_EQUAL_90"), (91, "STRUCTURE_BUDGET_MUST_EQUAL_90"),
    (0, "INVALID_STRUCTURE_WEIGHT"), (-1, "INVALID_STRUCTURE_WEIGHT"),
    (float("nan"), "INVALID_STRUCTURE_WEIGHT"), (float("inf"), "INVALID_STRUCTURE_WEIGHT"),
    (None, "INVALID_STRUCTURE_WEIGHT"), (True, "INVALID_STRUCTURE_WEIGHT"),
])
def test_invalid_structure_budget_is_not_silently_rescaled(weight, error):
    groups = {"structure": [(True, weight)], "execution": [(True, r.weight) for r in EXECUTION_RULES]}
    with pytest.raises(ValueError, match=error):
        validate_score_budget(groups)


def test_native_scoring_validates_common_scale_before_building_condition_ledger(monkeypatch):
    import domain.strategies.low_entry_v6_signals as signals

    def drifted_scale(*args, **kwargs):
        tests, detail = execution_conditions(*args, **kwargs)
        tests[0] = (tests[0][0], 5)
        tests[3] = (tests[3][0], 1)
        return tests, detail

    def unexpected_ledger(*args, **kwargs):
        pytest.fail("Invalid scale reached score_evidence")

    monkeypatch.setattr(signals, "execution_conditions", drifted_scale)
    monkeypatch.setattr(signals, "score_evidence", unexpected_ledger)
    with pytest.raises(ValueError, match="LES_EXECUTION_SCORE_SCALE_MISMATCH"):
        scored(features(), LowEntryV6Config())


@pytest.mark.parametrize("policy,expected_calls", [(1, 2), (2, 0)])
def test_legacy_reward_scale_only_runs_for_explicit_legacy_policy(monkeypatch, policy, expected_calls):
    import domain.strategies.low_entry_v6_signals as signals

    calls = []
    original = signals._rr_pass

    def track_legacy(rr, minimum):
        calls.append(minimum)
        return original(rr, minimum)

    monkeypatch.setattr(signals, "_rr_pass", track_legacy)
    scored(features(), LowEntryV6Config(decision_policy_version=policy))
    assert len(calls) == expected_calls
    if policy == 1:
        assert calls == [2., 1.8]


@pytest.mark.parametrize("restricted,cap", [(False, 100), (None, 64), (True, 0)])
def test_all_setups_use_raw_minus_penalties_then_minimum_cap_and_single_decision(restricted, cap):
    f = features(restricted=restricted, market_regime="RISK_OFF")
    result = decide(verified_quality(), f, config=LowEntryV6Config())
    assert set(result["setup_decisions"]) == set(SETUPS)
    for name, outcome in result["setup_decisions"].items():
        evidence = result["setups"][name]["score_evidence"]
        assert evidence["total_weight"] == 100
        assert outcome["raw_score"] == evidence["earned_points"]
        assert outcome["penalty"] >= 5
        assert outcome["adjusted_score"] == max(0., outcome["raw_score"] - outcome["penalty"])
        assert outcome["safety_cap"] <= cap
        assert outcome["score"] == outcome["final_score"] == min(
            outcome["adjusted_score"], outcome["safety_cap"])
        assert outcome["signal"] == decision_from_score(outcome["final_score"])
        if restricted is not False:
            assert outcome["signal"] not in {"BUY", "BUY_SMALL"}
