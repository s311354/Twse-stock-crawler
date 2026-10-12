"""Synthetic decision-policy tests, not probabilities, fills or market evidence."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from application.low_entry_v6_daily import project_report, unavailable_report
from domain.strategies.low_entry_score_v6 import LowEntryScoreV6Strategy, attach_low_entry_v6_context
from domain.strategies.low_entry_v6_decision import (
    POLICY, _budget_groups, compose_score, decision_from_score,
)
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.strategies.low_entry_v6_signals import (
    LowEntryV6Config, decide, net_reward_risk, score_setups, trade_plan,
)
from infrastructure.report.low_entry_v6_renderer import setup_score_summary
from interface.les_v6 import analyze_request, main
from tests.strategies.test_low_entry_score_v6 import bars, execution_report, features, financials, quality

# Pin the earlier additive condition set; later source/leadership policies have separate tests.
CONFIG = LowEntryV6Config(correlation_policy_version=1, leadership_policy_version=1)


def verified_quality(**changes):
    return quality(coverage=dict(balance_sheet=100, profitability=75, growth=80, return_on_capital=100, cash_flow=100), **changes)


def evaluate(f=None, q=None, **kwargs):
    return decide(q or verified_quality(), f if f is not None else features(), config=CONFIG, **kwargs)


def ledger(earned=90., observed=True):
    return {"conditions": [dict(id="one", group="test", weight=100., observed=observed,
                                fraction=earned / 100 if observed else None)]}


def test_score_arithmetic_caps_use_minimum_not_sum_and_do_not_mutate():
    evidence = ledger()
    penalties = [dict(reason="RISK_OFF", points=5.)]
    caps = [dict(reason="BUY_RISK_FAIL", maximum=74.), dict(reason="DATA_UNKNOWN", maximum=64.)]
    original = deepcopy((evidence, penalties, caps))
    score = compose_score(evidence, penalties, caps)
    assert score["raw_score"] == 90 and score["penalty"] == 5
    assert score["adjusted_score"] == 85 and score["final_score"] == score["safety_cap"] == 64
    assert (evidence, penalties, caps) == original
    assert compose_score(ledger(2), penalties, [])["final_score"] == 0


def test_no_calculable_evidence_remains_null_not_estimated_zero():
    result = compose_score(ledger(observed=False), [], [dict(reason="HARD_FAIL", maximum=0)])
    assert result["raw_score"] is result["adjusted_score"] is result["final_score"] is None
    measured_zero = compose_score(ledger(0), [], [])
    assert measured_zero["final_score"] == 0
    empty_features = {key: None for key in features()}
    decision = evaluate(empty_features)
    assert decision["score"] is decision["formal_score"] is decision["estimated_score"] is None
    assert decision["signal"] == "WATCH" and decision["risk_fraction"] == 0
    assert decision["evidence_status"] == "UNAVAILABLE"


@pytest.mark.parametrize("value,signal", [(None, "WATCH"), (0., "AVOID"), (54.9999, "AVOID"),
    (55., "WATCH"), (math.nextafter(65., 0.), "WATCH"), (65., "BUY_SMALL"),
    (math.nextafter(75., 0.), "BUY_SMALL"), (75., "BUY"), (100., "BUY")])
def test_single_decision_uses_unrounded_final_score(value, signal):
    assert decision_from_score(value) == signal


@pytest.mark.parametrize("value", [math.nan, math.inf, -1, 101])
def test_nonfinite_or_out_of_bounds_final_score_rejected(value):
    with pytest.raises(ValueError, match="INVALID_FINAL_SCORE"):
        decision_from_score(value)


def test_fixed_100_budget_existing_weights_and_correlated_groups_are_not_added_twice():
    f = features()
    setups = score_setups(f, trade_plan(f, config=CONFIG), config=CONFIG)
    for setup in setups.values():
        evidence = setup["score_evidence"]
        original = compose_score(evidence, [], [])
        assert original["raw_score"] == setup["formal_score"]
        assert evidence["total_weight"] == 100
        groups = _budget_groups(evidence)
        assert sum(g["budget"] for g in groups.values()) == 100
        assert sum(g["earned"] for g in groups.values()) == original["raw_score"]
        identifiers = [c for g in groups.values() for c in g["condition_ids"]]
        assert len(identifiers) == len(set(identifiers)) == len(evidence["conditions"])
        assert all(g["earned"] <= g["budget"] for g in groups.values())
    bad = ledger(); bad["conditions"][0]["weight"] = 101
    with pytest.raises(ValueError, match="BUDGET_MUST_EQUAL_100"):
        compose_score(bad, [], [])
    with pytest.raises(ValueError, match="INVALID_SCORE_ADJUSTMENT"):
        compose_score(ledger(), [dict(reason="FAKE", points=-1)], [])


def test_risk_off_is_one_explicit_penalty_not_also_a_threshold_increment():
    result = evaluate(features(market_regime="RISK_OFF"))
    assert result["penalties"] == [dict(reason="RISK_OFF", points=5.)]
    assert result["adjusted_score"] == max(0., result["raw_score"] - 5.)
    assert result["entry_checks"]["BUY"]["required_score"] == 75
    assert result["entry_checks"]["BUY_SMALL"]["required_score"] == 65


@pytest.mark.parametrize("f,q,reason,maximum", [
    ({}, {"status": "REJECT"}, "QUALITY_REJECT_OR_INTEGRITY_FAIL", 0),
    ({}, {"status": "DATA_INSUFFICIENT"}, "QUALITY_NOT_VERIFIED_FOR_ENTRY", 64),
    ({"market_ready": False}, {}, "INCOMPLETE_ALIGNED_HISTORY", 64),
    ({"restricted": None}, {}, "RESTRICTION_UNKNOWN", 64),
    ({"restricted": True}, {}, "TRADING_RESTRICTED", 0),
    ({"spread": .02}, {}, "LIQUIDITY_OR_RESTRICTION_FAIL", 64),
    ({"support": 80.}, {}, "STARTER_RISK_FAIL", 64),
    ({"ema5_slope": -1, "ema20_slope": -1, "atr_expansion": 2, "r5": -.2}, {}, "ACCELERATING_DECLINE", 0),
])
def test_high_raw_score_never_compensates_for_safety_failures(f, q, reason, maximum):
    result = evaluate(features(**f), verified_quality(**q))
    assert result["signal"] not in {"BUY", "BUY_SMALL"}
    assert result["safety_cap"] <= maximum
    assert reason in [cap["reason"] for cap in result["safety_caps"]]
    assert result["risk_fraction"] == 0


def test_missing_benchmark_cannot_award_points_even_with_stale_true_mask():
    f = features(market_ready=False, score_availability={"rs20": True, "rs60": True, "rs_improving": True})
    result = evaluate(f)
    for setup in result["setups"].values():
        for condition in setup["score_evidence"]["conditions"]:
            if any(k.startswith(("rs", "resilience")) and not k.startswith("rsi") for k in condition["dependencies"]):
                assert not condition["observed"] and condition["earned_points"] == 0


def test_rsi_oversold_alone_earns_no_recovery_points():
    f = features(rsi14=10., rsi_cross40=False, rsi_pullback=False)
    result = evaluate(f)
    for setup in result["setups"].values():
        for condition in setup["score_evidence"]["conditions"]:
            if any(k.startswith("rsi") for k in condition["dependencies"]):
                assert condition["earned_points"] == 0


def test_soft_setup_and_trigger_flags_are_not_second_admission_gates(monkeypatch):
    import domain.strategies.low_entry_v6_decision as module
    original = module.score_setups
    def diagnostic_only(*args, **kwargs):
        setups = original(*args, **kwargs)
        for value in setups.values():
            value.update(valid=False, strict_valid=False, recognized=False, entry_tier="NONE")
        return setups
    monkeypatch.setattr(module, "score_setups", diagnostic_only)
    monkeypatch.setattr(module, "triggers", lambda *args: (False, False, 0))
    result = evaluate()
    assert result["signal"] == "BUY"
    assert result["raw_score"] == 100
    assert result["starter_trigger"] is result["confirmed_trigger"] is False
    assert result["entry_checks"]["BUY"]["blockers"] == []
    assert "NO_ADDITIONAL_POINTS_OR_GATES" in result["soft_conditions"]["role"]


def test_net_rr_uses_costs_and_missing_resistance_is_not_price_discovery():
    plan = dict(entry_price=100., stop_price=95., target_price=110.)
    expected = (110 * (1 - CONFIG.sell_cost_rate) - 100 * (1 + CONFIG.buy_fee_rate)) / (
        100 * (1 + CONFIG.buy_fee_rate) - 95 * (1 - CONFIG.sell_cost_rate))
    assert net_reward_risk(plan, CONFIG) == pytest.approx(expected)
    assert net_reward_risk(dict(plan, target_price=None), CONFIG) is None
    assert net_reward_risk(dict(plan, stop_price=None), CONFIG) is None
    result = evaluate(features(swing_high=None, hh20=None, hh60=None))
    assert result["signal"] not in {"BUY", "BUY_SMALL"}
    assert result["plan"]["net_reward_risk"] is None
    assert "NET_RR_UNAVAILABLE" in result["reasons"]


def test_stop_is_not_shortened_to_improve_final_score():
    f = features(support=80.)
    before = trade_plan(f, config=CONFIG)
    result = evaluate(f)
    assert result["plan"]["stop_price"] == before["stop_price"]
    assert result["plan"]["stop_source"] == "CONFIRMED_SWING_LOW"
    assert result["signal"] not in {"BUY", "BUY_SMALL"}


def unified_report():
    report = execution_report()
    result = evaluate(report["technical"])
    report.update(parameters=vars(CONFIG), decision=result)
    report["quality"].update(score=85., coverage=verified_quality().coverage)
    report["entry"] = dict(result["plan"], trigger_price=report["technical"]["high"],
        starter_trigger=result["starter_trigger"], confirmed_trigger=result["confirmed_trigger"],
        rule="NEXT_SESSION_BREAKS_SIGNAL_HIGH", is_fill=False)
    report["liquidity"] = {"passed": result["plan"]["liquidity_pass"]}
    report["sizing"] = {"indicative_shares": 0, "portfolio_data_pass": False}
    return json.loads(json.dumps(report))


@pytest.mark.parametrize("key,value", [("raw_score", 0), ("penalty", 1), ("adjusted_score", 0),
    ("final_score", 0), ("safety_cap", 0), ("safety_caps", []), ("budget_groups", {}),
    ("score_evidence", {}), ("soft_conditions", {}), ("setup_decisions", {}), ("comparison", {})])
def test_renderer_and_execution_recompute_ledger_not_trust_labels(key, value):
    report = unified_report()
    assert report_failure(report) is None
    if report["decision"][key] == value:
        value = [dict(reason="FAKE", maximum=100)]
    report["decision"][key] = value
    assert entry_gate_checks(report)["score"] == "FAIL"
    assert report_failure(report) == "V6_SCORE_OR_SETUP_MISMATCH"


def test_projection_and_renderer_publish_final_score_and_adjustment_reasons():
    report = unified_report()
    values = project_report(report)
    for label, field in (("Raw Score", "raw_score"), ("Penalty", "penalty"), ("Adjusted Score", "adjusted_score"),
                         ("Safety Cap", "safety_cap"), ("Final Score", "final_score")):
        assert values["LES v6 " + label] == report["decision"][field]
    assert values["低點分數"] == values["LES v6 Final Score"]
    rendered = setup_score_summary(report)
    assert all(label in rendered for label in ("Raw", "Penalty", "Adjusted", "Safety Cap", "Final"))
    empty = unavailable_report("2330", "TEST", "2026-09-11", "2026-09-11T13:30+08:00", "MISSING")
    assert empty["decision"]["signal"] == "WATCH" and empty["decision"]["score"] is None
    assert "非實際 0 分" in setup_score_summary(empty)


def test_shadow_mode_is_a_cap_before_the_single_decision():
    result = evaluate(industry_mode="SHADOW")
    assert result["signal"] == "WATCH" and result["score"] == 64
    assert "INDUSTRY_PROFILE_SHADOW_MODE" in result["reasons"]


def test_native_plugin_cutoff_and_future_filings_are_still_causal():
    data = attach_low_entry_v6_context(bars(65), symbol="2330", sector="ELEC",
                                      as_of="2026-09-11T13:30+08:00", filings=financials())
    original = LowEntryScoreV6Strategy().run_latest(data).iloc[0]["LES v6 Report"]
    assert original["decision"]["policy_version"] == POLICY
    future = dict(financials()[-1], revision_id="future", available_at="2026-09-12T08:00+08:00", total_equity=-1)
    changed = attach_low_entry_v6_context(bars(65), symbol="2330", sector="ELEC",
        as_of="2026-09-11T13:30+08:00", filings=financials() + [future])
    assert LowEntryScoreV6Strategy().run_latest(changed).iloc[0]["LES v6 Report"] == original


def test_offline_cli_and_native_api_use_new_policy_by_default(tmp_path, capsys):
    frame = bars(61); frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    request = dict(bars=frame.to_dict("records"), context=dict(symbol="2330", sector="ELEC",
        as_of="2026-09-11T13:30+08:00", filings=financials()))
    path = tmp_path/"synthetic.json"; path.write_text(json.dumps(request))
    expected = analyze_request(request)
    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert expected["reports"][0]["parameters"]["decision_policy_version"] == 2
    assert expected["reports"][0]["decision"]["signal"] in {"BUY", "BUY_SMALL", "WATCH", "AVOID"}


def test_legacy_policy_must_be_explicit_and_unsafe_versions_rejected():
    assert CONFIG.decision_policy_version == 2
    legacy = replace(CONFIG, decision_policy_version=1)
    assert decide(quality(status="REJECT"), features(), config=legacy)["signal"] == "NO_BUY"
    for version in (0, 5, True, "2"):
        with pytest.raises(ValueError):
            LowEntryV6Config(decision_policy_version=version)
