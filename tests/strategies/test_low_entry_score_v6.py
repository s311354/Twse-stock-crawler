from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from application.strategy_execution import StrategyExecutionService
from domain.features import SharedFeatureConfig, SharedFeaturePipeline
from domain.strategies.low_entry_score_v6 import (
    PARAMETER_VERSION, V6_OUTPUT_COLUMNS, LowEntryScoreV6Strategy,
    assess_execution, attach_low_entry_v6_context, gate_funnel,
)
from domain.strategies.low_entry_v6_features import _wilder, build_features
from domain.strategies.low_entry_v6_quality import (
    QualityResult, evaluate_quality, growth, normalize_quarters,
)
from domain.strategies.low_entry_v6_signals import (
    EXECUTABLE_SIGNALS, LowEntryV6Config, decide, evaluate_exit, hard_vetoes,
    score_setups, size_position, trade_plan,
)
from domain.strategy import available_strategies, get_strategy
from interface.les_v6 import analyze_request, main


# Pin the pre-unification contract tests; v2 has separate decision-policy tests.
CONFIG = LowEntryV6Config(decision_policy_version=1)
ACCOUNT = {"value": 1_000_000., "available_cash": 500_000., "intended_weight": .2, "capacity_weight": .2}


def financials():
    """Synthetic complete quarterly evidence; never production financials."""
    records = []
    for i, period in enumerate(pd.period_range("2023Q1", "2026Q2", freq="Q")):
        revenue = 5000 * 1.06 ** i
        records.append({
            "symbol": "2330", "market": "TWSE", "currency": "TWD", "unit": "thousand",
            "scope": "consolidated", "source": "TWSE_OPENAPI", "revision_id": str(i),
            "period_end": period.end_time.date().isoformat(), "flow_basis": "QUARTER",
            "available_at": (period.end_time.normalize() + pd.Timedelta(days=40)).isoformat() + "+08:00",
            "revenue": revenue, "cost_of_revenue": revenue * .5, "operating_income": revenue * .3,
            "net_income": revenue * .25, "basic_eps": 2 * 1.06 ** i,
            "cfo": revenue * .3, "capex": revenue * .1,
            "pretax_income": revenue * .3125, "income_tax": revenue * .0625,
            "cash": 1000., "interest_bearing_debt": 1000.,
            "current_assets": 6000., "non_current_assets": 4000., "total_assets": 10000.,
            "current_liabilities": 2000., "total_liabilities": 3000., "total_equity": 7000.,
        })
    return records


def quality(**changes):
    fields = dict(status="PASS", score=85., confidence=90., integrity="PASS",
                  components={}, coverage={}, metrics={})
    return QualityResult(**dict(fields, **changes))


def bars(count=100):
    dates = pd.bdate_range(end="2026-09-11", periods=count)
    close = 100 + np.arange(count) * .1 + 2 * np.sin(np.arange(count) / 3)
    return pd.DataFrame({
        "Date": dates, "Open": close - .4, "High": close + .2, "Low": close - 1,
        "Close": close, "Volume": 1_000_000., "Amount": close * 1_000_000.,
        "Trades": 500, "Bid": close - .05, "Ask": close + .05, "PE": 20.,
        "TAIEX Close": 20000 + np.arange(count) * 10., "Restricted": False,
    })


def features(**changes):
    # Controlled unit evidence; indicator arithmetic is tested separately on real bars.
    f = dict(
        open=99.5, close=100., high=100.4, low=99.5, volume=1_000_000.,
        ema5=99.8, ema10=99.7, ema20=99.3,
        ema5_slope=.2, ema10_slope=.1, ema20_slope=.1,
        atr14=2., atr_percent=.02, atr_expansion=.95, atr_stable=True,
        clv=.8, cmf20=.15, cmf_improving=True, obv_slope5=1000.,
        up_volume_ratio=1.2, volume_ratio=1.3, amount_ratio=1.3,
        amount_ma20=100_000_000., spread=.001, restricted=False,
        vwap20=99.8, position60=.2, drawdown60=-.2, drawdown20=-.06,
        rs20=.05, rs60=.1, rs_slope5=.002, rs_improving=True,
        resilience=.03, resilience_improving=True, rs_near_high=True,
        no_new_low2=True, new_lows3=0., new_low=False, higher_low=True, higher_high=True,
        support=99., support_intact=True, support_break=False, swing_high=108., hh20=110., hh60=120.,
        undercut_reclaim=False, reclaim10=True, reclaim20=True,
        ema20_extension=.35, breakout_extension=-5., break_previous_high=True,
        rsi14=45., rsi_cross40=True, rsi_pullback=True, macd_improving=True,
        macd_strict=-.1, macd_signal_strict=-.2, down_volume_contracts=True,
        down_volume_expands=False, gap=0., r5=.03,
        market_regime="RISK_ON", history_ready=True, market_ready=True,
    )
    f.update(changes)
    return f


def setup_features(setup, **changes):
    if setup == "REVERSAL":
        f = features(ema10=98., ema20=101., ema20_slope=-.1)
    elif setup == "TREND_PULLBACK":
        f = features(position60=.5, drawdown60=-.06, rs20=-.001, rs60=.1,
                     rs_slope5=.005, rs_improving=True)
    else:
        f = features(position60=.9, drawdown60=-.005, drawdown20=-.005,
                     hh20=99.8, hh60=100.2, swing_high=99.9, breakout_extension=.1)
    f.update(changes)
    return f


def test_quality_core_without_cash_flow_is_insufficient_without_invented_extensions():
    records = financials()
    for row in records:
        for field in ("cfo", "capex", "pretax_income", "income_tax", "cash", "interest_bearing_debt"):
            row.pop(field)
    q = evaluate_quality(records, symbol="2330", market="TWSE", as_of="2026-09-11T13:30+08:00")
    assert q.status == "DATA_INSUFFICIENT"
    assert q.score == pytest.approx(sum(q.components.values()) - q.metrics["quality_penalty"])
    assert q.confidence >= 70
    assert q.optional["cfo"] == q.optional["auditor_opinion"] == "NOT_AVAILABLE"
    assert q.coverage["cash_flow"] == 0 and q.coverage["return_on_capital"] == 100
    assert q.coverage["profitability"] == 100
    assert q.metrics["roe_reported"] == q.metrics["roe"] == pytest.approx(records[-1]["net_income"] / 7000)
    assert q.metrics["roe_ttm"] == pytest.approx(q.metrics["net_income_ttm"] / 7000)


def test_future_restatement_cannot_change_past_quality():
    records = financials()
    reference = evaluate_quality(records, symbol="2330", market="TWSE", as_of="2026-09-11")
    future = dict(records[-1], available_at="2026-09-12", revision_id="future", total_equity=-1)
    assert evaluate_quality(records + [future], symbol="2330", market="TWSE", as_of="2026-09-11") == reference
    assert evaluate_quality(records + [future], symbol="2330", market="TWSE", as_of="2026-09-13").status == "REJECT"


@pytest.mark.parametrize("field,value,reason", [
    ("total_equity", 0, "INVALID_BALANCE"), ("total_assets", 20000, "IDENTITY_FAIL"),
    ("auditor_opinion", "ADVERSE", "HARD_RISK"), ("going_concern", True, "HARD_RISK"),
])
def test_financial_hard_rejects(field, value, reason):
    records = financials()
    records[-1][field] = value
    result = evaluate_quality(records, symbol="2330", market="TWSE", as_of="2026-09-11")
    assert result.status == "REJECT"
    assert reason in result.reasons[0]


@pytest.mark.parametrize("field,value,reason", [
    ("current_assets", None, "UNRESOLVABLE"), ("symbol", "other", "IDENTITY_AMBIGUOUS"),
    ("scope", "standalone", "UNVERIFIED_FILING_SCOPE"), ("available_at", None, "MISSING_AVAILABLE"),
])
def test_unresolved_evidence_is_not_a_confirmed_financial_reject(field, value, reason):
    records = financials()
    records[-1][field] = value
    result = evaluate_quality(records, symbol="2330", market="TWSE", as_of="2026-09-11")
    assert result.status == "DATA_INSUFFICIENT" and result.integrity == "UNKNOWN"
    assert reason in result.reasons[0]


def test_missing_growth_coverage_never_becomes_small_buy():
    q = evaluate_quality(financials()[-4:], symbol="2330", market="TWSE", as_of="2026-09-11")
    assert q.status == "PROVISIONAL"
    assert decide(q, features(), config=CONFIG)["signal"] == "WATCH_ONLY"


def test_no_filings_is_insufficient_not_zero_profit():
    q = evaluate_quality([], symbol="2330", market="TWSE", as_of="2026-09-11")
    assert q.status == "DATA_INSUFFICIENT" and q.score is None
    assert "cfo" not in q.metrics


def test_ytd_conversion_never_subtracts_balances_or_bridges_missing_quarter():
    periods = {pd.Period(f"2025Q{i}"): dict(flow_basis="YTD", revenue=100*i, total_assets=1000+i) for i in (1, 2, 3, 4)}
    normalized = normalize_quarters(periods)
    assert [row["revenue"] for row in normalized.values()] == [100] * 4
    assert "total_assets" not in normalized[pd.Period("2025Q4")]
    del periods[pd.Period("2025Q2")]
    assert normalize_quarters(periods)[pd.Period("2025Q3")]["revenue"] is None


@pytest.mark.parametrize("current,previous,state", [(1, -1, "LOSS_TO_PROFIT"), (-1, -2, "IMPROVING_LOSS"), (-2, -1, "DETERIORATING_LOSS")])
def test_nonpositive_growth_denominator(current, previous, state):
    assert growth(current, previous) == (None, state)


def test_wilder_uses_sma_seed():
    values = pd.Series(np.arange(1, 17, dtype=float))
    output = _wilder(values)
    assert pd.isna(output.iloc[12])
    assert output.iloc[13] == 7.5
    assert output.iloc[14] == pytest.approx((7.5 * 13 + 15) / 14)


@pytest.mark.parametrize("count,available", [(60, False), (61, True)])
def test_rs60_requires_61_consecutive_aligned_closes(count, available):
    data = bars(count)
    f = build_features(data, SharedFeaturePipeline().build(data))
    assert pd.notna(f.iloc[-1]["rs60"]) == available
    data.loc[30, "TAIEX Close"] = np.nan
    f = build_features(data, SharedFeaturePipeline().build(data))
    assert pd.isna(f.iloc[-1]["rs60"])


def test_taiex_join_is_by_date_not_row_position():
    data = bars(65)
    benchmark = data[["Date", "TAIEX Close"]].rename(columns={"TAIEX Close": "Close"}).iloc[::-1]
    prepared = attach_low_entry_v6_context(data.drop(columns="TAIEX Close"), symbol="2330", sector="ELEC", as_of="2026-09-11T15:00+08:00", filings=[], taiex=benchmark)
    pd.testing.assert_series_equal(prepared["TAIEX Close"], data["TAIEX Close"], check_names=False)
    with pytest.raises(ValueError, match="Duplicate"):
        attach_low_entry_v6_context(data, symbol="2330", sector="ELEC", as_of="2026-09-11", filings=[], taiex=pd.concat([benchmark, benchmark]))


def test_pivot_and_hh_are_causal_and_confirmed_two_sessions_later():
    data = bars(70)
    data.loc[65, ["Low", "Open", "Close"]] = [80., 100., 100.]
    f = build_features(data, SharedFeaturePipeline().build(data))
    assert f.iloc[65]["swing_low"] != 80
    assert f.iloc[66]["swing_low"] != 80
    assert f.iloc[67]["swing_low"] == 80
    assert f.iloc[-1]["hh20"] == data["High"].iloc[-21:-1].max()
    prefix = data.iloc[:-1].copy()
    earlier = build_features(prefix, SharedFeaturePipeline().build(prefix))
    pd.testing.assert_frame_equal(f.iloc[:-1], earlier)


def test_shared_feature_window_mismatch_rejected():
    data = bars()
    shared = SharedFeaturePipeline(SharedFeatureConfig(vwap_short_window=60)).build(data)
    with pytest.raises(ValueError, match="20-session"):
        build_features(data, shared)


def test_confirmed_undercut_uses_event_low_and_cannot_bypass_acceleration():
    f = setup_features("REVERSAL", no_new_low2=False, undercut_reclaim=True,
                       support=99.5, stop_support=99.)
    result = decide(quality(), f, config=CONFIG)
    assert result["signal"] == "BUY"
    assert result["plan"]["stop_price"] == 98.4
    f.update(ema5_slope=-1., atr_expansion=2., r5=-.2)
    assert decide(quality(), f, config=CONFIG)["signal"] == "NO_BUY"


def test_zero_range_and_invalid_bid_do_not_create_pass():
    data = bars()
    data.loc[len(data)-1, ["Open", "High", "Low", "Close"]] = 100.
    data.loc[len(data)-1, "Bid"] = 0
    f = build_features(data, SharedFeaturePipeline().build(data))
    assert f.iloc[-1]["clv"] == 0
    assert pd.isna(f.iloc[-1]["spread"])


@pytest.mark.parametrize("setup", ["REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION"])
def test_three_confirmed_setup_buy_paths(setup):
    result = decide(quality(), setup_features(setup), config=CONFIG)
    assert result["setup"] == setup
    assert result["signal"] == "BUY", result
    assert 80 <= result["score"] <= 100
    assert result["risk_gate"] == "PASS"


@pytest.mark.parametrize("setup", ["REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION"])
def test_three_starter_paths_require_quality_and_do_not_use_early_buy(setup):
    f = setup_features(setup, break_previous_high=False, reclaim10=False, reclaim20=False)
    if setup == "LEADERSHIP_CONTINUATION":
        f["hh20"] = 100.2
    result = decide(quality(), f, config=CONFIG)
    assert result["signal"] == "BUY_SMALL", result
    assert result["starter_trigger"] and not result["confirmed_trigger"]
    assert decide(quality(score=70), f, config=CONFIG)["signal"] == "BUY_SMALL"
    assert decide(quality(score=65), f, config=CONFIG)["signal"] == "BUY_SMALL"
    assert decide(quality(score=64.99), f, config=CONFIG)["signal"] not in EXECUTABLE_SIGNALS


@pytest.mark.parametrize("status,expected", [("REJECT", "NO_BUY"), ("PROVISIONAL", "WATCH_ONLY"), ("WATCH_QUALITY", "WATCH_ONLY"), ("DATA_INSUFFICIENT", "DATA_INSUFFICIENT")])
def test_high_score_never_overrides_quality(status, expected):
    assert decide(quality(status=status), features(), config=CONFIG)["signal"] == expected


def test_best_setup_is_max_not_sum_and_missing_rs60_removes_weight():
    f = features(rs60=None)
    scores = score_setups(f, trade_plan(f, config=CONFIG), config=CONFIG)
    assert scores["TREND_PULLBACK"]["available_weight"] == 95
    assert scores["LEADERSHIP_CONTINUATION"]["available_weight"] == 93
    result = decide(quality(), f, config=CONFIG)
    assert result["score"] == max(s["score"] for s in scores.values() if s["valid"])
    assert result["score"] <= 100


def test_reversal_below_ema20_is_allowed_but_accelerating_decline_is_not():
    f = setup_features("REVERSAL", ema20=102.)
    assert decide(quality(), f, config=CONFIG)["signal"] == "BUY"
    f.update(ema5_slope=-.1, atr_expansion=1.4, r5=-.1)
    result = decide(quality(), f, config=CONFIG)
    assert result["signal"] == "NO_BUY"
    assert "ACCELERATING_DECLINE" in result["hard_vetoes"]


def test_undercut_exception_is_narrow_and_does_not_override_acceleration():
    f = features(new_lows3=2, clv=-.2)
    assert "PERSISTENT_NEW_LOW" in hard_vetoes(f)
    f["undercut_reclaim"] = True
    assert "PERSISTENT_NEW_LOW" not in hard_vetoes(f)
    f.update(ema5_slope=-1, ema20_slope=-1, atr_expansion=2, r5=-.3)
    assert "ACCELERATING_DECLINE" in hard_vetoes(f)


def test_weak_volume_requires_support_break_for_hard_veto():
    f = features(open=101., clv=-.8, volume_ratio=2.)
    assert "HIGH_VOLUME_WEAK_SUPPORT_BREAK" not in hard_vetoes(f)
    f["support_break"] = True
    assert "HIGH_VOLUME_WEAK_SUPPORT_BREAK" in hard_vetoes(f)


def test_leadership_no_invented_target_and_extended_breakout_waits():
    result = decide(quality(), setup_features("LEADERSHIP_CONTINUATION"), config=CONFIG)
    assert result["signal"] == "BUY"
    assert result["plan"]["target_price"] is result["plan"]["reward_risk"] is None
    extended = decide(quality(), setup_features("LEADERSHIP_CONTINUATION", breakout_extension=1.6), config=CONFIG)
    assert extended["signal"] == "WAIT_RETEST"


def test_riskoff_is_conditional_not_automatic_veto():
    result = decide(quality(), setup_features("REVERSAL", market_regime="RISK_OFF"), config=CONFIG)
    assert result["signal"] == "BUY"
    assert result["required_score"] == 80
    assert result["position_multiplier"] == .5
    assert result["maximum_stop_atr"] < 1.5
    weak = decide(quality(), setup_features("REVERSAL", market_regime="RISK_OFF", rs20=-.1), config=CONFIG)
    assert weak["signal"] == "ARMED"


@pytest.mark.parametrize("updates", [{"support": None}, {"support": 80.}, {"spread": None}, {"restricted": None}, {"amount_ma20": 10.}])
def test_missing_risk_or_liquidity_never_produces_entry(updates):
    assert decide(quality(), setup_features("REVERSAL", **updates), config=CONFIG)["signal"] not in EXECUTABLE_SIGNALS


def test_starter_size_caps_not_fixed_half_and_missing_capacity_zero():
    result = decide(quality(), setup_features("REVERSAL", break_previous_high=False), config=CONFIG)
    args = dict(portfolio_value=ACCOUNT["value"], available_cash=ACCOUNT["available_cash"], intended_weight=.2,
                portfolio_capacity_weight=.2, amount_ma20=100_000_000., config=CONFIG)
    shares = size_position(result, **args)
    assert shares > 0
    assert shares * result["plan"]["entry_price"] <= .2 * .3 * ACCOUNT["value"]
    assert size_position(result, **dict(args, portfolio_capacity_weight=None)) == 0
    assert size_position(result, **dict(args, available_cash=0)) == 0


def test_plugin_shared_pipeline_and_prefix_outputs_are_immutable_and_deterministic():
    original = bars(80)
    data = attach_low_entry_v6_context(original, symbol="2330", sector="ELEC", as_of="2026-09-11T15:00+08:00", filings=financials(), portfolio=ACCOUNT)
    before = data.copy(deep=True)
    strategy = get_strategy("low_entry_score_v6")
    result = strategy.run(data)
    shared = strategy.run_with_features(data, SharedFeaturePipeline().build(data))
    pd.testing.assert_frame_equal(result, shared)
    pd.testing.assert_frame_equal(data, before)
    prefix = strategy.run(data.iloc[:-1].copy())
    pd.testing.assert_frame_equal(result.iloc[:-1], prefix)
    assert set(result["LES v6 Decision"]) <= {"BUY", "BUY_SMALL", "AVOID", "WATCH", "ARMED", "NO_BUY", "DATA_INSUFFICIENT", "WATCH_ONLY", "SKIP", "WAIT_RETEST"}
    json.dumps(result.iloc[-1]["LES v6 Report"], allow_nan=False)
    assert "low_entry_score_v5" in available_strategies()
    service = StrategyExecutionService()
    latest = service.evaluate("low_entry_score_v6", V6_OUTPUT_COLUMNS, service.prepare([data]))
    assert latest.iloc[0]["LES v6 Score"] == result.iloc[-1]["LES v6 Score"]


def test_incomplete_daily_bar_rejected_and_sector_unsupported():
    data = attach_low_entry_v6_context(bars(), symbol="2330", sector="BANK", as_of="2026-09-11T13:00+08:00", filings=financials())
    with pytest.raises(ValueError, match="INCOMPLETE"):
        LowEntryScoreV6Strategy().run(data)
    data.attrs["les_v6_context"]["as_of"] = "2026-09-11T15:00+08:00"
    report = LowEntryScoreV6Strategy().run(data).iloc[-1]["LES v6 Report"]
    assert report["decision"]["signal"] == "WATCH"
    assert "SECTOR_UNSUPPORTED" in report["decision"]["reasons"]
    assert report["decision"]["safety_cap"] <= 64


def execution_report():
    f = setup_features("REVERSAL")
    result = decide(quality(), f, config=CONFIG)
    return {"symbol": "2330", "signal_date": "2026-09-11", "parameter_version": PARAMETER_VERSION,
            "industry_profile": {"name": "SEMI_PROFILE", "mode": "ACTIVE", "version": "test-only-active"},
            "decision_time": "2026-09-11T13:30:00+08:00",
            "model_version": "6.0.0",
            "data": {"integrity": "PASS", "completed_history": True, "market_aligned": True, "confidence": 90},
            "parameters": vars(CONFIG),
            "quality": {"status": "PASS", "score": 85, "coverage": dict.fromkeys(
                ("balance_sheet", "profitability", "growth", "return_on_capital", "cash_flow"), 100)},
            "downtrend_gate": {"hard_veto": False},
            "decision": result, "technical": f, "valuation": {"interpretation": "NEUTRAL"},
            "entry": dict(result["plan"], trigger_price=100.4, rule="NEXT_SESSION_BREAKS_SIGNAL_HIGH", is_fill=False)}


def test_execution_requires_next_session_fresh_quote_and_reprices_gap_risk():
    report = execution_report()
    quote = {"symbol": "2330", "timestamp": "2026-09-14T09:10+08:00", "price": 100.5, "bid": 100.4, "ask": 100.5, "restricted": False}
    args = dict(now="2026-09-14T09:10:30+08:00", trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    result = assess_execution(report, quote, **args)
    assert result["action"] == "ELIGIBLE" and result["shares"] > 0 and not result["is_fill"]
    assert assess_execution(report, dict(quote, price=110.), **args)["reason"] == "REPRICED_RISK_OR_LIQUIDITY_FAIL"
    assert assess_execution(report, dict(quote, ask=110.), **args)["reason"] == "REPRICED_RISK_OR_LIQUIDITY_FAIL"
    assert assess_execution(report, dict(quote, price=100.), **args)["reason"] == "TRIGGER_NOT_REACHED"
    assert assess_execution(report, dict(quote, timestamp="2026-09-14T09:08+08:00"), **args)["reason"] == "STALE_OR_MISSING_QUOTE"
    assert assess_execution(report, quote, **dict(args, portfolio={})) ["shares"] == 0
    assert assess_execution(report, quote, **dict(args, trading_sessions=[]))["reason"] == "NO_VERIFIED_NEXT_SESSION"
    assert assess_execution(report, quote, hard_risk=True, **args)["action"] == "SKIP"
    assert assess_execution(report, quote, config=LowEntryV6Config(maximum_spread=.02), **args)["reason"] == "EXECUTION_CONFIG_MISMATCH"
    assert assess_execution(report, dict(quote, timestamp="2026-09-11T13:30+08:00"), **dict(args, now="2026-09-11T13:30+08:00"))["reason"] == "OUTSIDE_NEXT_SESSION"
    after_close = "2026-09-14T13:30:01+08:00"
    assert assess_execution(report, dict(quote, timestamp=after_close), **dict(args, now=after_close))["reason"] == "OUTSIDE_NEXT_SESSION"


def raw_execution_report(*, cost_headroom=True):
    """Synthetic raw-data fixture shared with the cross-repository order test."""
    data = bars(100)
    close = np.r_[np.linspace(115., 120., 60), np.linspace(120., 95., 30),
                  [95., 94.8, 95.1, 94.9, 95., 95.3, 95.6, 95.9, 96.2, 96.7]]
    for key, value in {"Close": close, "Open": close - .4, "High": close + .1,
                       "Low": close - 3., "Bid": close - .05, "Ask": close + .05,
                       "Amount": close * 1_000_000., "TAIEX Close": np.linspace(20000., 20200., 100)}.items():
        data[key] = value
    data.loc[95:99, "Low"] = [94.5, 94.8, 94.4, 95.7, 96.1]
    if cost_headroom:
        # Earlier observed wick gives this synthetic positive path enough net
        # reward. cost_headroom=False preserves the former gross-only fixture.
        data.loc[80, "High"] += 3.
    data.loc[99, "Volume"] = 1_200_000
    prepared = attach_low_entry_v6_context(data, symbol="2330", sector="ELEC",
                                          as_of="2026-09-11T15:00+08:00", filings=financials(), portfolio=ACCOUNT)
    report = LowEntryScoreV6Strategy().run(prepared).iloc[-1]["LES v6 Report"]
    return report


@pytest.mark.usefixtures("active_industry_profiles")
def test_raw_ohlcv_to_quality_setup_signal_and_next_session_execution():
    """Synthetic causal example, not a market-performance claim or injected LES score."""
    report = raw_execution_report()
    assert report["quality"]["status"] == "PASS"
    assert report["decision"]["signal"] == "BUY_SMALL"
    assert report["decision"]["setup"] == "REVERSAL"
    assert report["decision"]["raw_score"] == 89
    assert report["decision"]["penalty"] == 18
    assert report["setups"]["les_v6"] == report["decision"]["score"] == 71
    assert report["entry"]["is_fill"] is False
    quote = {"symbol": "2330", "timestamp": "2026-09-14T09:10+08:00", "price": 96.9,
             "bid": 96.8, "ask": 96.9, "restricted": False}
    result = assess_execution(report, quote, now=quote["timestamp"],
                              trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    assert result["action"] == "ELIGIBLE" and result["shares"] > 0
    assert result["is_fill"] is False


def test_exit_hard_risk_and_trailing_stop_only_move_up():
    f = features()
    assert evaluate_exit(f, initial_stop=90, hard_risk=True)["signal"] == "EXIT"
    result = evaluate_exit(f, initial_stop=90, previous_trailing_stop=98, highest_close_since_entry=102)
    assert result["trailing_stop"] == 98
    assert evaluate_exit(features(close=89), initial_stop=90)["signal"] == "EXIT"
    assert evaluate_exit(features(close=None), initial_stop=90)["signal"] == "NO_ACTION"


def test_cli_json_entrypoint_and_gate_funnel(tmp_path, capsys):
    data = bars(65)
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    payload = {"bars": data.to_dict("records"), "context": {"symbol": "2330", "sector": "ELEC", "as_of": "2026-09-11T15:00+08:00", "filings": financials(), "portfolio": ACCOUNT}}
    expected = analyze_request(payload)
    path = tmp_path / "v6.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert expected["funnel"]["universe_count"] == 1
    with pytest.raises(ValueError, match="one report"):
        gate_funnel(expected["reports"] * 2)


def test_config_rejects_nan_or_unsafe_fractions():
    with pytest.raises(ValueError):
        LowEntryV6Config(maximum_spread=float("nan"))
    with pytest.raises(ValueError):
        LowEntryV6Config(starter_fraction=.8)
