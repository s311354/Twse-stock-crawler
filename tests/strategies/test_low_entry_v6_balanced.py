"""Balanced starters broaden setup eligibility, never quality or execution gates."""
from copy import deepcopy

import pytest

from domain.strategies import low_entry_v6_signals as signals
from domain.strategies.low_entry_score_v6 import assess_execution, PARAMETER_VERSION
from domain.strategies.low_entry_v6_report import report_failure
from tests.strategies.test_low_entry_score_v6 import (
    ACCOUNT, CONFIG, execution_report, quality, setup_features,
)


RECOVERY = {
    "REVERSAL": dict(atr_stable=False, atr_expansion=1.1),
    "TREND_PULLBACK": dict(ema20_slope=-.02, rs_near_high=False, drawdown60=-.05),
    "LEADERSHIP_CONTINUATION": dict(ema20_slope=-.02),
}


@pytest.mark.parametrize("setup", signals.SETUPS)
def test_recovery_paths_are_starter_only_even_at_high_score(setup):
    result = signals.decide(quality(), setup_features(setup, **RECOVERY[setup]), config=CONFIG)
    assert result["setup"] == setup
    assert result["score"] > CONFIG.buy_score_minimum
    assert result["signal"] == "BUY_SMALL"
    assert result["entry_tier"] == "STARTER_ONLY"
    assert result["setups"][setup]["strict_valid"] is False
    assert result["risk_gate"] == "PASS" and result["risk_fraction"] == CONFIG.small_risk_fraction
    assert result["setup_decisions"][setup]["signal"] == "BUY_SMALL"


@pytest.mark.parametrize("setup,updates,reason", [
    ("REVERSAL", dict(atr_expansion=1.250001), "CONTROLLED_VOLATILITY"),
    ("REVERSAL", dict(atr_expansion=None), "CONTROLLED_VOLATILITY"),
    ("REVERSAL", dict(drawdown60=-.059, position60=.401), "REVERSAL_LOCATION"),
    ("REVERSAL", dict(no_new_low2=False), "STABLE_LOW"),
    ("REVERSAL", dict(ema5_slope=0), "RECOVERY_MOMENTUM"),
    ("REVERSAL", dict(close=98.9), "SUPPORT_RECOVERED"),
    ("TREND_PULLBACK", dict(ema20_slope=-.050001), "TREND_OR_RECOVERY"),
    ("TREND_PULLBACK", dict(ema20_slope=None), "TREND_OR_RECOVERY"),
    ("TREND_PULLBACK", dict(support_intact=False), "INTACT_SUPPORT"),
    ("TREND_PULLBACK", dict(drawdown20=-.151), "PULLBACK_DEPTH"),
    ("TREND_PULLBACK", dict(drawdown20=-.009), "PULLBACK_DEPTH"),
    ("LEADERSHIP_CONTINUATION", dict(rs20=-.010001), "RELATIVE_LEADERSHIP"),
    ("LEADERSHIP_CONTINUATION", dict(rs20=-.005, rs60=None), "RELATIVE_LEADERSHIP"),
    ("LEADERSHIP_CONTINUATION", dict(rs_slope5=0), "RELATIVE_IMPROVEMENT"),
    ("LEADERSHIP_CONTINUATION", dict(ema20_extension=1.01), "NOT_EXTENDED"),
])
def test_recovery_bounds_are_not_blanket_relaxations(setup, updates, reason):
    f = setup_features(setup, **dict(RECOVERY[setup], **updates))
    observed = signals.score_setups(f, signals.trade_plan(f, config=CONFIG), config=CONFIG)[setup]
    assert not observed["valid"]
    assert reason in observed["blocking_conditions"]


@pytest.mark.parametrize("setup,updates", [
    ("REVERSAL", dict(drawdown60=-.06, position60=.5, atr_expansion=1.25)),
    ("REVERSAL", dict(drawdown60=-.05, position60=.4)),
    ("TREND_PULLBACK", dict(ema20_slope=-.05, drawdown20=-.02)),
    ("TREND_PULLBACK", dict(drawdown20=-.15)),
    ("LEADERSHIP_CONTINUATION", dict(rs20=-.01, rs60=.1, rs_near_high=True)),
])
def test_recovery_boundary_is_inclusive(setup, updates):
    f = setup_features(setup, **dict(RECOVERY[setup], **updates))
    observed = signals.score_setups(f, signals.trade_plan(f, config=CONFIG), config=CONFIG)[setup]
    assert observed["valid"] and observed["entry_tier"] == "STARTER_ONLY"


@pytest.mark.parametrize("setup,threshold", [
    ("REVERSAL", 65), ("TREND_PULLBACK", 65), ("LEADERSHIP_CONTINUATION", 65),
])
def test_starter_threshold_is_exact_and_quality_matches_pass(monkeypatch, setup, threshold):
    f = setup_features(setup, break_previous_high=False, reclaim10=False, reclaim20=False)
    if setup == "LEADERSHIP_CONTINUATION":
        f["hh20"] = 100.2
    candidate = signals.score_setups(f, signals.trade_plan(f, config=CONFIG))[setup]
    # Isolate the numeric decision boundary; arithmetic has independent tests.
    monkeypatch.setattr(signals, "score_setups", lambda *args, **kwargs: {setup: candidate})
    candidate["score"] = threshold - .001
    assert signals.decide(quality(), f, config=CONFIG)["signal"] == ("WATCH" if threshold == 65 else "ARMED")
    candidate["score"] = threshold
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["signal"] == "BUY_SMALL" and result["required_score"] == threshold
    assert signals.decide(quality(score=70), f, config=CONFIG)["signal"] == "BUY_SMALL"
    assert signals.decide(quality(score=65), f, config=CONFIG)["signal"] == "BUY_SMALL"
    assert signals.decide(quality(score=64.99), f, config=CONFIG)["signal"] not in signals.EXECUTABLE_SIGNALS


def test_confirmed_buy_threshold_and_riskoff_increment(monkeypatch):
    f = setup_features("REVERSAL")
    candidate = signals.score_setups(f, signals.trade_plan(f, config=CONFIG))["REVERSAL"]
    monkeypatch.setattr(signals, "score_setups", lambda *args, **kwargs: {"REVERSAL": candidate})
    candidate["score"] = 74.99
    assert signals.decide(quality(), f, config=CONFIG)["signal"] == "BUY_SMALL"
    candidate["score"] = 75
    assert signals.decide(quality(), f, config=CONFIG)["signal"] == "BUY"
    f["market_regime"] = "RISK_OFF"
    assert signals.decide(quality(), f, config=CONFIG)["signal"] == "BUY_SMALL"
    candidate["score"] = 80
    assert signals.decide(quality(), f, config=CONFIG)["signal"] == "BUY"


def test_higher_scoring_starter_does_not_hide_confirmed_buy():
    f = setup_features("TREND_PULLBACK")  # Relaxed reversal now also eligible.
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["setup"] == "TREND_PULLBACK" and result["signal"] == "BUY"


def test_failed_highest_mode_does_not_hide_other_executable_mode(monkeypatch):
    f = setup_features("REVERSAL", support=98.3, swing_high=107.1)
    # Net RR is between 1.8 and 2: pullback BUY outranks reversal BUY_SMALL.
    scored = signals.score_setups(f, signals.trade_plan(f, config=CONFIG))
    rev = dict(scored["REVERSAL"], score=99, strict_valid=False, entry_tier="STARTER_ONLY")
    pull = dict(scored["TREND_PULLBACK"], score=85, valid=True, strict_valid=True, entry_tier="CONFIRMED")
    monkeypatch.setattr(signals, "score_setups", lambda *args, **kwargs: {"REVERSAL": rev, "TREND_PULLBACK": pull})
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["setup"] == "TREND_PULLBACK" and result["signal"] == "BUY"
    assert result["setup_decisions"]["REVERSAL"]["signal"] == "BUY_SMALL"


@pytest.mark.parametrize("updates,veto", [
    (dict(new_lows3=2, clv=-.2), "PERSISTENT_NEW_LOW"),
    (dict(ema5_slope=-.1, ema20_slope=-.1, atr_expansion=1.5, r5=-.1), "ACCELERATING_DECLINE"),
    (dict(open=101, clv=-.8, volume_ratio=2, support_break=True), "HIGH_VOLUME_WEAK_SUPPORT_BREAK"),
    (dict(open=98, close=98, gap=-.02), "UNRECOVERED_GAP"),
    (dict(market_regime="RISK_OFF", rs20=-.1, rs_slope5=-.01, resilience=-.05, new_low=True), "MARKET_RELATIVE_BREAKDOWN"),
])
def test_all_hard_vetoes_still_block_recovery(updates, veto):
    f = setup_features("REVERSAL", **dict(RECOVERY["REVERSAL"], **updates))
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["signal"] == "NO_BUY" and veto in result["hard_vetoes"]


@pytest.mark.parametrize("setup", signals.SETUPS)
@pytest.mark.parametrize("updates", [dict(restricted=None), dict(spread=None), dict(amount_ma20=1),
                                     dict(support=None), dict(support=80), dict(history_ready=False)])
def test_data_liquidity_and_risk_still_block_recovery(setup, updates):
    f = setup_features(setup, **dict(RECOVERY[setup], **updates))
    assert signals.decide(quality(), f, config=CONFIG)["signal"] not in signals.EXECUTABLE_SIGNALS


@pytest.mark.parametrize("status", ["REJECT", "PROVISIONAL", "WATCH_QUALITY", "DATA_INSUFFICIENT"])
def test_no_quality_exception_for_recovery(status):
    f = setup_features("REVERSAL", **RECOVERY["REVERSAL"])
    assert signals.decide(quality(status=status), f, config=CONFIG)["signal"] not in signals.EXECUTABLE_SIGNALS


@pytest.mark.parametrize("old_version", ["2026-09-12-v6-initial", "2026-09-20-v6-balanced-starters",
                                        "2026-09-20-v6-restriction-asof", "2026-09-20-v6-risk-budgeted-starters",
                                        "2026-09-21-v6-confirmed-retest", "2026-09-25-v6-five-gate-contract",
                                        "2026-09-25-v6-industry-shadow", "2026-09-25-v6-held-floor-stabilization",
                                        "2026-09-25-v6-active-adaptive-recovery", "2026-09-25-v6-structure-before-trigger",
                                        "2026-09-25-v6-independent-les-score", "2026-09-25-v6-market-bound-evidence",
                                        "2026-09-25-v6-balanced-quality-net-risk",
                                        "2026-09-25-v6-downside-quality-cost-sized-risk",
                                        "2026-09-26-v6-pit-eps-structural-net-risk",
                                        "2026-09-26-v6-archived-restrictions-price-derived-risk",
                                        "2026-09-26-v6-consistent-risk-boundaries",
                                        "2026-10-02-v6-scoped-quality-causal-structure",
                                        "2026-10-02-v6-unified-score-ledger-risk-comparison",
                                        "2026-10-02-v6-penalties-safety-caps-single-decision",
                                        "2026-10-02-v6-common-execution-risk-score",
                                        "2026-10-02-v6-source-family-dedup",
                                        "2026-10-02-v6-pit-leadership",
                                        "2026-10-04-v6-pit-taxonomy-period-debt",
                                        "2026-10-04-v6-pit-bounded-quality-penalty-v3",
                                        "2026-10-04-v6-pit-comparative-quality-ttm-v5",
                                        "2026-10-04-v6-pit-core-financial-quality-v7"])
def test_old_parameter_reports_cannot_execute_under_new_rules(old_version):
    report = execution_report()
    old = deepcopy(report)
    old["parameter_version"] = old_version
    assert PARAMETER_VERSION == "2026-10-04-v6-pit-reported-period-quality-v8"
    assert report_failure(old) == "EXECUTION_CONFIG_MISMATCH"
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00", price=100.5, bid=100.4, ask=100.5, restricted=False)
    args = dict(now=quote["timestamp"], trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    assert assess_execution(old, quote, **args)["reason"] == "PARAMETER_VERSION_MISMATCH"
    assert assess_execution(report, quote, **args)["action"] == "ELIGIBLE"


def test_unqualified_score_is_not_displayed_as_zero_and_reasons_are_escaped():
    from infrastructure.report.low_entry_v6_renderer import setup_score_summary
    report = {"decision": {"setup": None, "score": 0}, "setups": {
        "TREND_PULLBACK": {"score": 62., "valid": False, "blocking_conditions": ['<script>"']},
    }}
    html = setup_score_summary(report)
    assert html.startswith("62.00<br>TREND_PULLBACK（參考評分；無合格型態）")
    assert "TREND_PULLBACK 62.00（未成立）" in html
    assert "0.00" not in html and "<script>" not in html
    assert "&lt;script&gt;&quot;" in html
    report["decision"].update(setup="TREND_PULLBACK", score=70., entry_tier="STARTER_ONLY")
    report["setups"]["TREND_PULLBACK"].update(valid=True, entry_tier="STARTER_ONLY")
    assert "修復型：僅限 BUY_SMALL" in setup_score_summary(report)


@pytest.mark.parametrize("tier", [None, "STARTER_ONLY", "UNKNOWN"])
def test_execution_cannot_upgrade_starter_only_or_unknown_tier_to_buy(tier):
    report = execution_report()
    report["decision"].update(signal="BUY", score=100, entry_tier=tier)
    assert report_failure(report) == "INVALID_SETUP_ENTRY_TIER"
    assert assess_execution(report, {}, now="2026-09-14", trading_sessions=[], portfolio=ACCOUNT)["reason"] == "INVALID_SETUP_ENTRY_TIER"


@pytest.mark.parametrize("updates", [dict(buy_score_minimum=64), dict(reversal_starter_score_minimum=60),
                                     dict(recovery_ema20_slope_tolerance=.1), dict(leadership_rs20_tolerance=.1),
                                     dict(pullback_maximum_drawdown=.3), dict(recovery_maximum_atr_expansion=2)])
def test_unsafe_balanced_configuration_rejected(updates):
    with pytest.raises(ValueError):
        signals.LowEntryV6Config(**updates)
