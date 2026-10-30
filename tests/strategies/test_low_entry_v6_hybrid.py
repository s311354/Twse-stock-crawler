"""Synthetic research-candidate evidence; not verified filings, fills or returns."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import datetime
import json
import math

import pytest

from application.low_entry_v6_daily import project_report
from domain.strategies.low_entry_score_v6 import (
    LowEntryScoreV6HybridStrategy, LowEntryScoreV6Strategy, assess_execution,
    attach_low_entry_v6_context,
)
from domain.strategies.low_entry_v6_hybrid import (
    CURVES, FACTOR_BUDGETS, NO_CLEAR_CONTEXT, PARAMETER_VERSION, POLICY,
    RELAXED_PARAMETER_VERSION, RELAXED_POLICY,
    model_spec, score_factors, select_context,
)
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.strategies.low_entry_v6_signals import LowEntryV6Config, decide, score_setups, trade_plan
from domain.strategy import available_strategies, get_strategy
from domain.features import SharedFeaturePipeline
from domain.model_contracts import DecisionField, FieldSource, FieldUnit, StrategyDecision
from infrastructure.report.low_entry_v6_renderer import setup_score_summary
from infrastructure.report.low_entry_v6_renderer import LowEntryV6Renderer
from infrastructure.report.html_renderer import RendererFactory
from interface.les_v6 import analyze_request, main
from tests.strategies.test_low_entry_score_v6 import ACCOUNT, bars, execution_report, features, financials
from tests.strategies.test_low_entry_v6_unified_decision import verified_quality

CONFIG = LowEntryV6Config(decision_policy_version=3)


def strong(context="TREND_PULLBACK", **changes):
    f = features(rsi_change=5., previous_high=99., volume_ma20_strict=1_000_000.,
        obv_slope5=200_000., amount_ratio=1.5, industry_status="PASS", industry_rs20=.05,
        industry_rs60=.10, market_correction20=-.04, downside_capture20=.2,
        correction_support=98., atr_expansion=.8, ema20_slope=.15)
    if context == "REVERSAL":
        f.update(ema10=98., ema20=101., ema20_slope=-.1)
    elif context == "LEADERSHIP_CONTINUATION":
        f.update(position60=.9, drawdown60=-.005, drawdown20=-.005,
            base_compression=True, downside_resilience=True, breakout_quality=True,
            base_support_tests=4., base_range_atr10=1., base_range_ratio5_10=.5,
            base_volume_ratio5_20=.5, breakout_extension=.1)
    elif context == NO_CLEAR_CONTEXT:
        f.update(position60=.5, drawdown60=-.005, drawdown20=-.005)
    return dict(f, **changes)


def evaluate(f=None, q=None, **kwargs):
    return decide(q or verified_quality(), strong() if f is None else f, config=CONFIG, **kwargs)


def report_for(f=None):
    report = execution_report()
    f = strong() if f is None else f
    d = evaluate(f)
    report.update(name="SYNTHETIC", parameter_version=PARAMETER_VERSION, parameters=vars(CONFIG),
        technical=f, decision={key: value for key, value in d.items() if key not in {"setups", "plan"}},
        setups=dict(selection=d["selection"], selected_setup=d["setup"]),
        sizing=dict(indicative_shares=0, reason="SYNTHETIC_NOT_A_FILL"),
        liquidity=dict(passed=d["plan"]["liquidity_pass"]))
    report["quality"].update(coverage=verified_quality().coverage)
    report["entry"] = dict(d["plan"], trigger_price=f["high"], starter_trigger=d["starter_trigger"],
        confirmed_trigger=d["confirmed_trigger"], rule="NEXT_SESSION_BREAKS_SIGNAL_HIGH", is_fill=False)
    return json.loads(json.dumps(report, allow_nan=False))


@pytest.mark.parametrize("context", ["REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION", NO_CLEAR_CONTEXT])
def test_context_before_scoring_and_shared_100_budget(context):
    f = strong(context)
    original = deepcopy(f)
    selected = select_context(f)
    assert selected.context == context
    with pytest.raises(FrozenInstanceError):
        selected.context = "REVERSAL"
    d = evaluate(f)
    assert d["selection"] == selected.record()
    assert not d["selection"]["score_or_buy_used_for_selection"]
    assert "setup_decisions" not in d and "highest_final_score" not in d["selection"]
    assert d["setups"] == {}
    assert {key: value["budget"] for key, value in d["budget_groups"].items()} == dict(FACTOR_BUDGETS)
    assert sum(c["weight"] for c in d["score_evidence"]["conditions"]) == 100
    assert d["raw_score"] == 100 and d["score_coverage"] == 100
    assert d["execution_score"]["earned_points"] == 10
    assert d["signal"] == "BUY" and d["final_score"] == 100
    assert d["plan"] == trade_plan(f, config=CONFIG, setup=None if context == NO_CLEAR_CONTEXT else context)
    assert f == original


def test_ambiguous_structure_has_fixed_priority_never_risk_or_score_competition():
    f = strong()
    assert select_context(f).record()["matches"] == ["TREND_PULLBACK", "REVERSAL"]
    baseline = select_context(f)
    changed = dict(f, spread=.99, amount_ma20=0., swing_high=None, hh20=None, hh60=None,
                   restricted=True, score=100., account_equity=1.)
    assert select_context(changed) == baseline
    assert evaluate(changed, verified_quality(status="REJECT"))["setup"] == baseline.context
    assert baseline.record()["status"] == "MULTIPLE_MATCHES_FIXED_PRIORITY"


def test_one_plan_and_ledger_no_three_scores_or_second_correlation_penalty(monkeypatch):
    import domain.strategies.low_entry_v6_signals as signals
    import domain.strategies.low_entry_v6_decision as decisions
    calls = []
    native = signals.trade_plan
    def one_plan(*args, **kwargs):
        calls.append(kwargs.get("setup"))
        return native(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Hybrid must not invoke competing scores or deduplicate twice")
    monkeypatch.setattr(signals, "trade_plan", one_plan)
    monkeypatch.setattr(signals, "score_setups", forbidden)
    monkeypatch.setattr(decisions, "correlation_adjustment", forbidden)
    d = evaluate()
    assert calls == ["TREND_PULLBACK"]
    assert d["correlation_adjustment"]["removed_points"] == 0
    assert d["penalties"] == []


def test_fixed_curves_continuous_and_correlated_sources_share_budget():
    with pytest.raises(TypeError):
        CURVES["clv"] = (0., 1.)
    d = evaluate(strong(ema20_slope=.05, clv=.4, cmf20=.1, rsi_change=2.5))
    groups = d["budget_groups"]
    assert groups["trend"]["earned"] == pytest.approx(10.)
    assert groups["demand"]["earned"] == pytest.approx(6. + 8.)
    assert groups["momentum"]["earned"] == pytest.approx(5.)
    boosted = evaluate(strong(macd_improving=True, macd_strict=999., macd_signal_strict=-999.,
                              ema10=1000., break_previous_high=True))
    assert boosted["raw_score"] == 100
    assert boosted["budget_groups"]["trend"]["earned"] == 15
    assert evaluate(strong(rsi14=10., rsi_change=0.))["budget_groups"]["momentum"]["earned"] == 0
    assert model_spec()["calibration_status"] == "UNCALIBRATED_RESEARCH_CANDIDATE"


@pytest.mark.parametrize("field", ["clv", "cmf20", "amount_ratio", "obv_slope5", "volume_ma20_strict",
    "rsi_change", "industry_rs20", "rs60", "previous_high", "spread", "amount_ma20"])
def test_missing_slots_never_raise_fixed_context_score_or_redistribute(field):
    f = strong()
    selected = select_context(f)
    plan = trade_plan(f, config=CONFIG, setup=selected.context)
    full = score_factors(f, selected, plan, config=CONFIG)
    partial = score_factors(dict(f, **{field: None}), selected, plan, config=CONFIG)
    assert partial["score_evidence"]["formal_score"] < full["score_evidence"]["formal_score"]
    assert partial["score_evidence"]["coverage_percent"] < 100
    assert partial["score_evidence"]["estimated_score"] is None
    assert partial["score_evidence"]["total_weight"] == 100
    assert sum(g["budget"] for g in partial["factor_budgets"].values()) == 100


def test_pool_loses_coverage_instead_of_awarding_full_budget_from_one_source():
    d = evaluate(strong(cmf20=None))
    slot = next(c for c in d["score_evidence"]["conditions"] if c["id"] == "demand:money_flow")
    assert slot["fraction"] == .5 and slot["available_weight"] == slot["earned_points"] == 6
    assert slot["status"] == "PARTIAL" and slot["sources"][1]["status"] == "UNKNOWN"


def test_no_calculable_evidence_null_but_observed_failure_is_real_zero():
    d = evaluate({key: None for key in strong()})
    assert d["raw_score"] is d["final_score"] is d["estimated_score"] is None
    assert d["signal"] == "WATCH" and d["score_coverage"] == 0
    zero = evaluate(dict.fromkeys(("open", "high", "low", "close", "volume"), 1.) | {"rsi_change": 0.})
    assert zero["raw_score"] == 0 and zero["score_coverage"] > 0


@pytest.mark.parametrize("f,reason,cap", [
    ({"restricted": None}, "RESTRICTION_UNKNOWN", 64),
    ({"restricted": True}, "TRADING_RESTRICTED", 0),
    ({"market_ready": False}, "INCOMPLETE_ALIGNED_HISTORY", 64),
    ({"spread": .02}, "LIQUIDITY_OR_RESTRICTION_FAIL", 64),
    ({"support": 80.}, "STARTER_RISK_FAIL", 64),
    ({"ema5_slope": -1., "ema20_slope": -1., "atr_expansion": 2., "r5": -.2}, "ACCELERATING_DECLINE", 0),
])
def test_high_score_cannot_overcome_caps(f, reason, cap):
    d = evaluate(strong(**f))
    assert d["safety_cap"] <= cap and d["signal"] not in {"BUY", "BUY_SMALL"}
    assert reason in d["reasons"] and d["risk_fraction"] == 0
    assert d["safety_cap"] == min(c["maximum"] for c in d["safety_caps"])


def test_quality_hard_reject_and_unknown_and_risk_off_explicit_penalty():
    assert evaluate(q=verified_quality(status="REJECT"))["final_score"] == 0
    assert evaluate(q=verified_quality(status="DATA_INSUFFICIENT"))["final_score"] == 64
    d = evaluate(strong(market_regime="RISK_OFF"))
    assert d["penalties"] == [dict(reason="RISK_OFF", points=5.)]
    assert d["adjusted_score"] == d["raw_score"] - 5.
    assert d["entry_checks"]["BUY"]["required_score"] == 75


@pytest.mark.parametrize("flow_points,expected", [(4.9999, "BUY_SMALL"), (5., "BUY"), (5.0001, "BUY")])
def test_unrounded_final_is_only_decision_input(flow_points, expected):
    f = strong(clv=flow_points / 12 * .8, cmf20=0., amount_ratio=1., obv_slope5=0., rsi_change=0.)
    d = evaluate(f)
    assert d["final_score"] == pytest.approx(70 + flow_points)
    assert d["signal"] == expected
    assert d["penalty"] == 0 and d["safety_cap"] == 100


def test_only_small_risk_pass_caps_at_74_without_shortening_anchor():
    d = evaluate(strong(support=97., swing_high=110., hh20=112.))
    assert d["raw_score"] > 75
    assert d["plan"]["stop_price"] == 96.4
    assert d["safety_cap"] == d["final_score"] == 74
    assert d["signal"] == "BUY_SMALL"
    assert d["entry_checks"]["BUY_SMALL"]["risk_pass"]
    assert not d["entry_checks"]["BUY"]["risk_pass"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_nonfinite_or_boolean_features_are_unknown_not_maximum_points(value):
    d = evaluate(strong(industry_rs20=value, clv=value, rsi_change=value))
    assert d["budget_groups"]["momentum"]["earned"] == 0
    assert d["budget_groups"]["relative_strength"]["earned"] == 16
    assert d["score_coverage"] < 100
    json.dumps(d, allow_nan=False)


def test_unknown_benchmark_and_stale_masks_do_not_grant_points_or_risk_pass():
    d = evaluate(strong(market_ready=False, industry_status="UNKNOWN"))
    assert d["budget_groups"]["relative_strength"]["earned"] == 0
    for key in ("spread", "support", "hh20", "hh60", "swing_high"):
        masks = {key: False}
        if key in {"hh20", "hh60", "swing_high"}:
            masks = dict.fromkeys(("hh20", "hh60", "swing_high"), False)
        masked = evaluate(strong(score_availability=masks))
        assert masked["signal"] not in {"BUY", "BUY_SMALL"}
    assert evaluate(strong(score_availability={"close": False}))["raw_score"] is None


def test_native_net_cost_rr_no_fake_resistance_or_stop_shortening():
    d = evaluate()
    p = d["plan"]
    expected = (p["target_price"] * (1 - CONFIG.sell_cost_rate) - p["entry_price"] * (1 + CONFIG.buy_fee_rate)) / (
        p["entry_price"] * (1 + CONFIG.buy_fee_rate) - p["stop_price"] * (1 - CONFIG.sell_cost_rate))
    assert p["net_reward_risk"] == pytest.approx(expected)
    far = strong(support=80.)
    assert evaluate(far)["plan"]["stop_price"] == trade_plan(far, config=CONFIG, setup="TREND_PULLBACK")["stop_price"]


def correction_slot(result):
    return next(c for c in result["score_evidence"]["conditions"]
                if c["id"] == "relative_strength:correction_resilience")


def test_no_qualifying_market_correction_is_observed_zero_not_unknown():
    f = strong(market_ready=True, market_correction20=-.029,
               downside_capture20=None, correction_support=None)
    slot = correction_slot(evaluate(f))
    assert slot["status"] == "OBSERVED"
    assert slot["fraction"] == 0.
    assert slot["earned_points"] == 0.
    assert slot["available_weight"] == slot["weight"] == 4
    assert slot["dependencies"] == ["market_ready", "market_correction20"]


@pytest.mark.parametrize("changes", [
    {"market_ready": False, "market_correction20": -.01},
    {"market_ready": True, "market_correction20": None},
])
def test_missing_benchmark_history_or_correction_stays_unknown(changes):
    slot = correction_slot(evaluate(strong(**changes)))
    assert slot["status"] == "UNKNOWN"
    assert slot["fraction"] is None
    assert slot["earned_points"] == 0.
    assert slot["available_weight"] == 0.


def test_active_correction_with_failed_support_is_known_zero_without_capture():
    slot = correction_slot(evaluate(strong(market_ready=True, market_correction20=-.03,
        correction_support=95., low=90., downside_capture20=None)))
    assert slot["status"] == "OBSERVED"
    assert slot["fraction"] == 0.
    assert slot["available_weight"] == slot["weight"] == 4


def test_active_correction_requires_capture_only_after_support_holds():
    unknown = correction_slot(evaluate(strong(market_ready=True, market_correction20=-.03,
        correction_support=98., low=98., downside_capture20=None)))
    observed = correction_slot(evaluate(strong(market_ready=True, market_correction20=-.03,
        correction_support=98., low=98., downside_capture20=.2)))
    assert unknown["status"] == "UNKNOWN" and unknown["fraction"] is None
    assert observed["status"] == "OBSERVED" and observed["fraction"] == 1.
    missing = evaluate(strong(swing_high=None, hh20=None, hh60=None))
    assert missing["plan"]["target_price"] is missing["plan"]["net_reward_risk"] is None
    assert "NET_RR_UNAVAILABLE" in missing["reasons"]
    discovery = evaluate(strong("LEADERSHIP_CONTINUATION", swing_high=99.9, hh20=99.8, hh60=100.2))
    assert discovery["rr_policy"] == "PRICE_DISCOVERY_EXTENSION_SUPPORT"
    assert discovery["plan"]["target_price"] is discovery["plan"]["net_reward_risk"] is None
    assert discovery["execution_score"]["earned_points"] == 7
    assert discovery["raw_score"] == 97 and discovery["score_coverage"] == 97


@pytest.mark.parametrize("context", ["TREND_PULLBACK", NO_CLEAR_CONTEXT])
def test_report_single_ledger_validation_projection_renderer_and_execution(context):
    report = report_for(strong(context))
    assert report_failure(report) is None
    assert set(entry_gate_checks(report).values()) == {"PASS"}
    assert project_report(report)["低點分數"] == report["decision"]["final_score"]
    rendered = setup_score_summary(report)
    assert "唯一 Final" in rendered and "不比例估分" in rendered and "未校準" in rendered
    assert "最高 Final：" not in rendered
    before = deepcopy(report)
    args = dict(now="2026-09-14T09:10:30+08:00", trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00", price=100.5, bid=100.4, ask=100.5, restricted=False)
    assert assess_execution(report, quote, **args)["action"] == "ELIGIBLE"
    assert assess_execution(report, dict(quote, price=110.), **args)["action"] == "SKIP"
    assert report == before


@pytest.mark.parametrize("key", ["selection", "score_model", "score_evidence", "budget_groups", "safety_caps", "final_score"])
def test_tampered_hybrid_identity_ledger_or_context_rejected(key):
    report = report_for()
    report["decision"][key] = 0 if key == "final_score" else {}
    assert entry_gate_checks(report)["score"] == "FAIL"
    assert report_failure(report) is not None


def test_extra_competing_scores_or_parameter_migration_rejected():
    report = report_for()
    report["decision"]["setup_decisions"] = {"REVERSAL": {"final_score": 100}}
    assert entry_gate_checks(report)["score"] == "FAIL"
    report = report_for(); report["setups"]["reversal_score"] = 100
    assert report_failure(report) is not None
    report = report_for(); report["parameter_version"] = "future"
    assert report_failure(report) == "EXECUTION_CONFIG_MISMATCH"


def test_plugin_registration_shared_pipeline_and_future_cutoff_parity():
    assert "low_entry_score_v6_hybrid" in available_strategies()
    assert isinstance(get_strategy("low_entry_score_v6_hybrid"), LowEntryScoreV6HybridStrategy)
    assert LowEntryScoreV6Strategy().config.decision_policy_version == 2
    with pytest.raises(ValueError, match="HYBRID_PLUGIN_REQUIRES_POLICY_3"):
        LowEntryScoreV6HybridStrategy(LowEntryV6Config())
    with pytest.raises(ValueError, match="CONTEXT_FIRST"):
        score_setups(strong(), trade_plan(strong(), config=CONFIG), config=CONFIG)
    data = attach_low_entry_v6_context(bars(65).iloc[:-1], symbol="2330", sector="ELEC",
        as_of="2026-09-10T13:30+08:00", filings=financials())
    original = LowEntryScoreV6HybridStrategy().run_latest(data).iloc[0]["LES v6 Report"]
    assert original["signal_date"] == "2026-09-10"
    late = dict(financials()[-1], revision_id="future", available_at="2026-09-11T08:00+08:00", total_equity=-1.)
    changed = attach_low_entry_v6_context(bars(65).iloc[:-1], symbol="2330", sector="ELEC",
        as_of="2026-09-10T13:30+08:00", filings=financials() + [late])
    assert LowEntryScoreV6HybridStrategy().run_latest(changed).iloc[0]["LES v6 Report"] == original
    future_bars = attach_low_entry_v6_context(bars(65), symbol="2330", sector="ELEC",
        as_of="2026-09-10T13:30+08:00", filings=financials())
    with pytest.raises(ValueError, match="INCOMPLETE_OR_FUTURE_DAILY_BAR"):
        LowEntryScoreV6HybridStrategy().run_latest(future_bars)
    future_bars.attrs["les_v6_context"]["as_of"] = "2026-09-11T13:30+08:00"
    future_bars.loc[future_bars.index[-1], "Close"] = 1_000_000.
    assert LowEntryScoreV6HybridStrategy().run(future_bars).iloc[-2]["LES v6 Report"] == original


def test_offline_cli_api_parity_hybrid_opt_in_and_default_unchanged(tmp_path, capsys):
    frame = bars(61); frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    request = dict(bars=frame.to_dict("records"), context=dict(symbol="2330", sector="ELEC",
        as_of="2026-09-11T13:30+08:00", filings=financials()))
    path = tmp_path / "synthetic.json"; path.write_text(json.dumps(request))
    expected = analyze_request(dict(request, score_policy="hybrid"))
    assert main([str(path), "--score-policy", "hybrid"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert expected["reports"][0]["parameter_version"] == RELAXED_PARAMETER_VERSION
    assert expected["reports"][0]["decision"]["policy_version"] == RELAXED_POLICY
    assert expected["reports"][0]["parameters"]["buy_score_minimum"] == 72
    baseline = analyze_request(dict(request, score_policy="hybrid-legacy"))
    assert main([str(path), "--score-policy", "hybrid-legacy"]) == 0
    assert json.loads(capsys.readouterr().out) == baseline
    assert baseline["reports"][0]["parameter_version"] == PARAMETER_VERSION
    assert baseline["reports"][0]["decision"]["policy_version"] == POLICY
    assert baseline["reports"][0]["parameters"]["buy_score_minimum"] == 75
    assert analyze_request(request)["reports"][0]["parameters"]["decision_policy_version"] == 2
    with pytest.raises(ValueError, match="UNKNOWN_V6_SCORE_POLICY"):
        analyze_request(dict(request, score_policy="unknown"))


def test_reuses_prepared_features_without_rebuilding_and_native_report_renderer():
    data = attach_low_entry_v6_context(bars(65), symbol="2330", sector="ELEC",
        as_of="2026-09-11T13:30+08:00", filings=financials())
    shared = SharedFeaturePipeline().build(data)
    plugin = LowEntryScoreV6HybridStrategy()
    result = plugin.run_with_features(data, shared, latest_only=True)
    assert result.iloc[0]["LES v6 Report"] == plugin.run_latest(data).iloc[0]["LES v6 Report"]
    report = report_for()
    timestamp = datetime.fromisoformat(report["decision_time"])
    decision = StrategyDecision(strategy_name="low_entry_score_v6_hybrid", evaluated_at=timestamp,
        fields=tuple(DecisionField(name=key, value=value, source=FieldSource.LES_V6,
            available_at=timestamp, confidence=100., unit=FieldUnit.TEXT, model_version="6.0.0")
            for key, value in project_report(report).items()))
    assert isinstance(RendererFactory.get_renderer("low_entry_score_v6_hybrid"), LowEntryV6Renderer)
    output = LowEntryV6Renderer().render([decision])
    assert "Hybrid" in output and "研究候選" in output and "唯一 Final" in output
    assert "型態／收盤 Trigger 已列入" not in setup_score_summary(report)
