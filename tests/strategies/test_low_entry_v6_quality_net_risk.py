"""Generic quality/risk policy: no symbol exception or future-price calibration."""
from copy import deepcopy

import pytest

from domain.strategies import low_entry_v6_signals as signals
from domain.strategies.low_entry_score_v6 import assess_execution
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from tests.strategies.test_low_entry_score_v6 import (
    ACCOUNT, CONFIG, execution_report, quality, raw_execution_report, setup_features,
)


@pytest.mark.parametrize("setup", signals.SETUPS)
@pytest.mark.parametrize("qs", [70., 74.999])
def test_quality_pass_is_sufficient_for_small_in_model_report_and_execution(setup, qs):
    f = setup_features(setup, break_previous_high=False, reclaim10=False, reclaim20=False)
    if setup == "LEADERSHIP_CONTINUATION":
        f["hh20"] = 100.2
    decision = signals.decide(quality(score=qs), f, config=CONFIG)
    assert decision["signal"] == "BUY_SMALL"
    report = deepcopy(execution_report())
    report.update(technical=f, decision=decision)
    report["quality"]["score"] = qs
    report["entry"].update(decision["plan"])
    assert entry_gate_checks(report)["quality"] == "PASS"
    assert report_failure(report) is None
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00",
                 price=100.5, bid=100.4, ask=100.5, restricted=False)
    assert assess_execution(report, quote, now=quote["timestamp"],
                            trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)["action"] == "ELIGIBLE"


@pytest.mark.parametrize("field,value", [("confidence", None), ("confidence", 69.999),
                                        ("coverage", {}), ("status", "WATCH_QUALITY")])
def test_lower_starter_qs_does_not_waive_data_confidence_coverage_or_classification(field, value):
    report = execution_report()
    report["quality"]["score"] = 70.
    report["decision"].update(signal="BUY_SMALL", required_score=65.)
    report["data" if field == "confidence" else "quality"][field] = value
    assert entry_gate_checks(report)["quality"] != "PASS"
    assert report_failure(report) is not None


@pytest.mark.parametrize("setup", signals.SETUPS)
def test_real_arithmetic_buy_atr_boundary_is_inclusive_but_not_unbounded(setup):
    f = setup_features(setup, high=100., atr14=3., support=95.5,
                       swing_high=130., hh20=140., hh60=150.)
    plan = signals.trade_plan(f, config=CONFIG, setup=setup)
    assert plan["stop_distance_atr"] == pytest.approx(1.8)
    assert signals.risk_pass(setup, "BUY", f, plan, "NEUTRAL", CONFIG)[0]
    wider = signals.trade_plan(dict(f, support=95.49999), config=CONFIG, setup=setup)
    assert not signals.risk_pass(setup, "BUY", f, wider, "NEUTRAL", CONFIG)[0]


@pytest.mark.parametrize("fraction,passed", [(.08, True), (.080001, False)])
def test_buy_also_needs_percentage_limit_even_when_atr_is_small(fraction, passed):
    f = setup_features("REVERSAL", high=100., atr14=5., support=100 * (1 - fraction) + 1.5,
                       swing_high=130., hh20=140., hh60=150.)
    plan = signals.trade_plan(f, config=CONFIG)
    assert plan["stop_distance_atr"] < 1.8
    assert signals.risk_pass("REVERSAL", "BUY", f, plan, "NEUTRAL", CONFIG)[0] is passed


@pytest.mark.parametrize("signal,cap", [("BUY", 1.275), ("BUY_SMALL", 1.53)])
def test_riskoff_retains_original_atr_caps(signal, cap):
    f = setup_features("REVERSAL", high=100., support=100. + .6 - cap * 2.,
                       market_regime="RISK_OFF", swing_high=130., hh20=140., hh60=150.)
    plan = signals.trade_plan(f, config=CONFIG)
    passed, limit, rr = signals.risk_pass("REVERSAL", signal, f, plan, "NEUTRAL", CONFIG)
    assert passed and limit == pytest.approx(cap) and rr == 2.
    wider = signals.trade_plan(dict(f, support=f["support"] - .00001), config=CONFIG)
    assert not signals.risk_pass("REVERSAL", signal, f, wider, "NEUTRAL", CONFIG)[0]


def test_wider_buy_needs_confirmed_setup_and_preserves_structural_stop():
    f = setup_features("REVERSAL", support=97.6, swing_high=115.)
    result = signals.decide(quality(score=70), f, config=CONFIG)
    assert result["signal"] == "BUY" and result["confirmed_trigger"]
    assert result["plan"]["stop_distance_atr"] == pytest.approx(1.7)
    assert result["plan"]["stop_price"] == pytest.approx(f["support"] - .3 * f["atr14"])
    assert result["risk_fraction"] == .0075
    assert signals.decide(quality(score=70), dict(f, break_previous_high=False), config=CONFIG)["signal"] == "BUY_SMALL"


def test_small_reversal_can_use_lower_rr_without_widening_stop_or_inventing_target():
    f = setup_features("REVERSAL", high=100., support=92.9, atr14=3.,
                       swing_high=115.5, hh20=120., hh60=125.)
    result = signals.decide(quality(score=70), f, config=CONFIG)
    assert result["signal"] == "BUY_SMALL"
    plan = result["plan"]
    assert plan["stop_price"] == pytest.approx(92.) and plan["target_price"] == 115.5
    assert plan["reward_risk"] < 2 and plan["net_reward_risk"] >= 1.7
    assert result["entry_checks"]["BUY_SMALL"]["rr_basis"] == "NET_OF_CONFIGURED_COSTS"
    assert result["risk_fraction"] == .0025


def test_cached_rr_cannot_override_price_derived_net_rr():
    f = setup_features("REVERSAL", high=100., support=95.6)
    plan = signals.trade_plan(f, config=CONFIG)
    plan.update(target_price=110., reward_risk=999., net_reward_risk=999.)
    assert signals.net_reward_risk(plan, CONFIG) < 1.7
    assert not signals.risk_pass("REVERSAL", "BUY_SMALL", f, plan, "NEUTRAL", CONFIG)[0]
    plan["target_price"] = 110.1
    assert signals.risk_pass("REVERSAL", "BUY_SMALL", f, plan, "NEUTRAL", CONFIG)[0]


@pytest.mark.parametrize("field,value", [("net_reward_risk", 999.), ("net_reward_risk", None),
                                        ("rr_basis", "GROSS")])
def test_report_rejects_forged_or_missing_net_rr_evidence(field, value):
    report = execution_report()
    report["entry"][field] = value
    assert entry_gate_checks(report)["risk"] == "FAIL"
    assert report_failure(report) is not None


def test_pre_cost_only_raw_fixture_is_not_silently_grandfathered():
    report = raw_execution_report(cost_headroom=False)
    assert report["entry"]["reward_risk"] > 2
    assert report["entry"]["net_reward_risk"] < 1.7
    assert report["decision"]["signal"] == "ARMED"
    assert "RR_BELOW_MINIMUM" in report["decision"]["entry_checks"]["BUY_SMALL"]["blockers"]


@pytest.mark.parametrize("field,value", [
    ("buy_maximum_stop_atr", 1.800001), ("buy_maximum_stop_fraction", .080001),
    ("starter_minimum_net_rr", 1.69999), ("starter_minimum_net_rr", float("nan")),
])
def test_new_configuration_bounds_cannot_disable_risk(field, value):
    with pytest.raises(ValueError):
        signals.LowEntryV6Config(**{field: value})
