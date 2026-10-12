"""SYNTHETIC_TEST_ONLY: versioned score policy, not performance or real fills."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from application.low_entry_v6_daily import DAILY_V6_CONFIG, project_report
from domain.strategies.low_entry_score_v6 import assess_execution, parameter_version
from domain.strategies.low_entry_v6_decision import decision_from_score, safety_limits
from domain.strategies.low_entry_v6_hybrid import (
    FACTOR_BUDGETS, RELAXED_FACTOR_BUDGETS, RELAXED_PARAMETER_VERSION, RELAXED_POLICY,
    model_spec,
)
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_config, report_failure
from domain.strategies.low_entry_v6_signals import LowEntryV6Config, decide, relaxed_hybrid_config
from domain.strategy import get_strategy
from infrastructure.report.low_entry_v6_renderer import setup_score_summary
from tests.strategies.test_low_entry_score_v6 import ACCOUNT
from tests.strategies.test_low_entry_v6_hybrid import CONFIG as BASELINE, report_for, strong
from tests.strategies.test_low_entry_v6_unified_decision import verified_quality

CONFIG = relaxed_hybrid_config()


def evaluate(f=None, q=None, config=CONFIG):
    return decide(q or verified_quality(), strong() if f is None else f, valuation="FAIR", config=config)


def native_report(f=None):
    report = report_for(f)
    f = report["technical"]
    d = evaluate(f)
    report.update(parameters=vars(CONFIG), parameter_version=parameter_version(CONFIG),
        decision={key: value for key, value in d.items() if key not in {"setups", "plan"}},
        setups=dict(selection=d["selection"], selected_setup=d["setup"]))
    report["entry"] = dict(d["plan"], trigger_price=f["high"], starter_trigger=d["starter_trigger"],
        confirmed_trigger=d["confirmed_trigger"], rule="NEXT_SESSION_BREAKS_SIGNAL_HIGH", is_fill=False)
    return json.loads(json.dumps(report, allow_nan=False))


def test_explicit_daily_policy_and_legacy_defaults_are_distinct():
    assert DAILY_V6_CONFIG == get_strategy("low_entry_score_v6_hybrid").config == CONFIG
    assert CONFIG.decision_policy_version == 4
    assert (CONFIG.buy_score_minimum, CONFIG.reversal_starter_score_minimum,
            CONFIG.pullback_starter_score_minimum, CONFIG.leadership_starter_score_minimum) == (72, 62, 62, 62)
    assert dict(RELAXED_FACTOR_BUDGETS) == dict(structure=30, trend=10, relative_strength=20,
                                              demand=25, momentum=5, execution=10)
    assert get_strategy("low_entry_score_v6_hybrid_legacy").config == BASELINE
    assert LowEntryV6Config().buy_score_minimum == BASELINE.buy_score_minimum == 75
    assert model_spec()["factor_budgets"] == dict(FACTOR_BUDGETS)
    assert model_spec(CONFIG)["parameter_version"] == RELAXED_PARAMETER_VERSION
    assert safety_limits(CONFIG) == (61, 71) and safety_limits(BASELINE) == (64, 74)
    for name, value in vars(CONFIG).items():
        if name not in {"decision_policy_version", "buy_score_minimum", "reversal_starter_score_minimum",
                        "pullback_starter_score_minimum", "leadership_starter_score_minimum"}:
            assert value == getattr(BASELINE, name)  # No relaxed costs, RR, risk or liquidity.
    with pytest.raises(ValueError, match="thresholds"):
        LowEntryV6Config(decision_policy_version=3, buy_score_minimum=72)
    with pytest.raises(ValueError, match="thresholds"):
        relaxed_hybrid_config(reversal_starter_score_minimum=61)


@pytest.mark.parametrize("score,signal", [(None, "WATCH"), (0., "AVOID"), (55., "WATCH"),
    (math.nextafter(62., 0.), "WATCH"), (62., "BUY_SMALL"),
    (math.nextafter(72., 0.), "BUY_SMALL"), (72., "BUY"), (100., "BUY")])
def test_unrounded_decision_boundaries(score, signal):
    assert decision_from_score(score, buy_minimum=72, small_minimum=62, decision_policy_version=4) == signal


@pytest.mark.parametrize("context", ["REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION", "NO_CLEAR_CONTEXT"])
def test_only_factor_budgets_change_not_context_attainment_or_trade_plan(context):
    f = strong(context, rsi_change=2., amount_ratio=1.1, clv=.1)
    before = deepcopy(f)
    old, new = evaluate(f, config=BASELINE), evaluate(f)
    assert f == before
    assert new["selection"] == old["selection"] and new["plan"] == old["plan"]
    assert new["execution_score"] == old["execution_score"]
    assert new["penalties"] == old["penalties"]
    assert math.fsum(c["weight"] for c in new["score_evidence"]["conditions"]) == pytest.approx(100.)
    expected = 0.
    for prior, current in zip(old["score_evidence"]["conditions"], new["score_evidence"]["conditions"], strict=True):
        group = current["group"]
        assert {k: v for k, v in current.items() if k not in {"weight", "earned_points", "available_weight", "rule"}} == {
            k: v for k, v in prior.items() if k not in {"weight", "earned_points", "available_weight", "rule"}}
        scale = dict(RELAXED_FACTOR_BUDGETS)[group] / dict(FACTOR_BUDGETS)[group]
        assert current["weight"] == pytest.approx(prior["weight"] * scale)
        expected += prior["earned_points"] * scale
    assert new["raw_score"] == pytest.approx(expected)


def test_native_near_miss_uses_new_threshold_without_changing_risk():
    f = strong(ema20_slope=.04, clv=0., cmf20=0., amount_ratio=1., obv_slope5=0., rsi_change=0.)
    old, new = evaluate(f, config=BASELINE), evaluate(f)
    assert old["raw_score"] == pytest.approx(64.5) and old["signal"] == "WATCH"
    assert new["raw_score"] == pytest.approx(66 + 1 / 3) and new["signal"] == "BUY_SMALL"
    assert new["plan"] == old["plan"]
    assert new["risk_gate"] == "PASS" and new["required_score"] == 62
    assert report_failure(native_report(f)) is None


@pytest.mark.parametrize("changes", [dict(restricted=None), dict(market_ready=False), dict(history_ready=False),
    dict(clv=None), dict(close=None), dict(support=None), dict(spread=None), dict(spread=.02),
    dict(amount_ma20=None), dict(amount_ma20=1.), dict(support=80.),
    dict(swing_high=None, hh20=None, hh60=None), dict(swing_high=100.45, hh20=100.45, hh60=100.45)])
def test_missing_evidence_or_small_risk_failure_cannot_become_buy(changes):
    result = evaluate(strong(**changes))
    assert result["safety_cap"] == 61
    assert result["signal"] not in {"BUY", "BUY_SMALL"}
    assert result["risk_fraction"] == 0


@pytest.mark.parametrize("status,cap", [("DATA_INSUFFICIENT", 61), ("PROVISIONAL", 61),
                                         ("WATCH_QUALITY", 61), ("REJECT", 0)])
def test_quality_still_non_compensable(status, cap):
    result = evaluate(q=verified_quality(status=status))
    assert result["raw_score"] == 100 and result["safety_cap"] == cap
    assert result["signal"] not in {"BUY", "BUY_SMALL"}


@pytest.mark.parametrize("changes", [dict(restricted=True), dict(new_lows3=2., clv=-.7)])
def test_known_restriction_and_hard_veto_remain_zero_cap(changes):
    result = evaluate(strong(**changes))
    assert result["final_score"] == result["safety_cap"] == 0
    assert result["signal"] == "AVOID"


def test_buy_risk_failure_only_allows_native_starter_and_caps_track_stricter_thresholds():
    f = strong(support=96., swing_high=120., hh20=130., hh60=140.)
    # 2.5 ATR stop: above BUY limit, within small limit; independently valid resistance/RR.
    result = evaluate(f)
    assert result["plan"]["stop_distance_atr"] == pytest.approx(2.5)
    assert result["signal"] == "BUY_SMALL" and result["safety_cap"] == 71
    assert result["entry_checks"]["BUY"]["risk_pass"] is False
    assert result["entry_checks"]["BUY_SMALL"]["risk_pass"] is True
    stricter = replace(CONFIG, buy_score_minimum=76., reversal_starter_score_minimum=66.,
                       pullback_starter_score_minimum=67., leadership_starter_score_minimum=68.)
    assert safety_limits(stricter) == (65, 75)
    assert evaluate(strong(restricted=None), config=stricter)["signal"] == "WATCH"


def test_missing_optional_pool_source_never_increases_score_or_coverage():
    before, missing = evaluate(), evaluate(strong(cmf20=None))
    assert before["raw_score"] == 100
    assert missing["raw_score"] == missing["score_coverage"] == 92.5
    assert missing["estimated_score"] is None
    assert missing["evidence_status"] == "PARTIAL"
    assert before["budget_groups"]["demand"]["budget"] == missing["budget_groups"]["demand"]["budget"] == 25


def test_native_report_renderer_projection_execution_and_config_validation():
    report = native_report()
    assert report_config(report) == CONFIG and report_failure(report) is None
    assert set(entry_gate_checks(report).values()) == {"PASS"}
    assert report["decision"]["policy_version"] == RELAXED_POLICY
    assert project_report(report)["低點分數"] == 100
    assert "Hybrid" in setup_score_summary(report) and "唯一 Final" in setup_score_summary(report)
    assert report_failure(report_for()) is None  # Retained policy 3 report stays valid.
    args = dict(now="2026-09-14T09:10:30+08:00", trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00", price=100.5, bid=100.4, ask=100.5, restricted=False)
    original = deepcopy(report)
    assert assess_execution(report, quote, **args)["action"] == "ELIGIBLE"
    for bad_quote, portfolio in ((dict(quote, timestamp="2026-09-14T09:00+08:00"), ACCOUNT),
                                 (dict(quote, price=110.), ACCOUNT), (quote, dict(ACCOUNT, available_cash=0.))):
        assert assess_execution(report, bad_quote, **dict(args, portfolio=portfolio))["action"] == "SKIP"
    assert report == original
    report["decision"]["score_model"]["factor_budgets"]["structure"] = 25
    assert entry_gate_checks(report)["score"] == "FAIL" and report_failure(report) is not None
    report = native_report(); report["parameters"]["buy_score_minimum"] = 70
    assert report_failure(report) == "EXECUTION_CONFIG_MISMATCH"
    with pytest.raises(ValueError, match="BASELINE_HYBRID"):
        parameter_version(CONFIG, {})
