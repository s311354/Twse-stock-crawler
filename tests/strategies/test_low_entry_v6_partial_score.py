"""Partial evidence is measurable, causal, and never an entry authorization."""
from copy import deepcopy
import json

import numpy as np
import pytest

from domain.features import SharedFeaturePipeline
from domain.strategies.low_entry_v6_features import build_features
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.strategies.low_entry_v6_signals import SETUPS, decide, score_setups, trade_plan, size_position
from domain.strategies.low_entry_score_v6 import LowEntryScoreV6Strategy, attach_low_entry_v6_context
from application.low_entry_v6_daily import project_report
from infrastructure.report.low_entry_v6_renderer import setup_score_summary
from tests.strategies.test_low_entry_score_v6 import CONFIG, bars, features, quality, financials, execution_report


def computed(data):
    return build_features(data, SharedFeaturePipeline().build(data))


def evidence(f, setup="REVERSAL"):
    return score_setups(f, trade_plan(f, config=CONFIG), config=CONFIG)[setup]["score_evidence"]


@pytest.mark.parametrize("setup", SETUPS)
def test_full_observation_ties_to_original_score_and_one_hundred_points(setup):
    f = features()
    scored = score_setups(f, trade_plan(f, config=CONFIG), config=CONFIG)[setup]
    e = scored["score_evidence"]
    assert e["total_weight"] == e["available_weight"] == e["coverage_percent"] == 100
    assert e["estimated_score"] == pytest.approx(scored["score"])
    assert e["earned_points"] == sum(scored["components"].values())
    assert e["status"] == "COMPLETE" and e["missing_fields"] == []


def test_missing_numeric_and_unknown_boolean_do_not_equal_known_false():
    f = features(cmf_improving=False, obv_slope5=-1.)
    known = evidence(f)
    f.update(cmf_improving=None, obv_slope5=np.nan)
    missing = evidence(f)
    assert missing["earned_points"] == known["earned_points"]
    assert missing["available_weight"] == known["available_weight"] - 10
    assert missing["estimated_score"] > known["estimated_score"]
    assert missing["status"] == "PARTIAL"
    assert {"cmf_improving", "obv_slope5"} <= set(missing["missing_fields"])


@pytest.mark.parametrize("length", [1, 5, 15, 25, 40, 59])
def test_short_history_has_finite_estimates_and_measured_coverage_but_no_entry(length):
    f = computed(bars(length)).iloc[-1].to_dict()
    result = decide(quality(), f, config=CONFIG)
    assert result["score_basis"] == "PARTIAL_DATA" and 0 <= result["score"] <= 100
    assert result["signal"] == "DATA_INSUFFICIENT"
    assert result["score_evidence"]["available_weight"] > 0
    assert result["risk_fraction"] == 0 and result["risk_gate"] == "NOT_EVALUATED"
    assert f["stock_history_observations"] == length
    for setup in SETUPS:
        e = result["setups"][setup]["score_evidence"]
        assert 0 <= e["estimated_score"] <= 100 and e["coverage_percent"] < 100
        assert e["estimated_score"] == pytest.approx(100 * e["earned_points"] / e["available_weight"])
    assert size_position(result, portfolio_value=1e6, available_cash=5e5,
                         intended_weight=.2, portfolio_capacity_weight=.2,
                         amount_ma20=1e8, config=CONFIG) == 0


def test_missing_benchmark_does_not_erase_stock_evidence_or_award_rs_points():
    data = bars().drop(columns="TAIEX Close")
    f = computed(data).iloc[-1].to_dict()
    result = decide(quality(), f, config=CONFIG)
    assert result["score"] is not None and result["signal"] == "DATA_INSUFFICIENT"
    assert f["stock_history_observations"] == 60 and f["market_history_observations"] == 0
    for setup in SETUPS:
        e = result["setups"][setup]["score_evidence"]
        assert e["components"]["relative_strength"]["available_weight"] == 0
        assert e["components"]["relative_strength"]["earned_points"] == 0


def test_boolean_warmup_and_hole_are_unknown_not_failed_conditions():
    short = computed(bars(10)).iloc[-1].to_dict()
    assert not short["macd_improving"] and not short["score_availability"]["macd_improving"]
    assert not short["score_availability"]["no_new_low2"]
    e = evidence(short)
    assert e["components"]["momentum"]["available_weight"] == 0
    data = bars()
    data.loc[len(data) - 3, "Volume"] = np.nan
    f = computed(data).iloc[-1].to_dict()
    assert not f["history_ready"] and f["stock_history_observations"] == 59
    assert not f["score_availability"]["obv_slope5"]
    assert evidence(f)["components"]["demand"]["available_weight"] < 25


@pytest.mark.parametrize("changes", [{"close": None}, {"low": 0.}, {"volume": 0.}, {"high": 1.}])
def test_invalid_current_bar_has_no_estimate_even_if_other_features_are_present(changes):
    f = features(**changes)
    result = decide(quality(), f, config=CONFIG)
    assert result["score"] is None and result["score_basis"] == "INSUFFICIENT_HISTORY"
    assert result["signal"] == "DATA_INSUFFICIENT"
    assert result["score_evidence"]["coverage_percent"] == 0


def test_empty_feature_input_cannot_score_restriction_or_execution_metadata():
    result = decide(quality(), {"restricted": False}, config=CONFIG)
    assert result["score"] is None and result["risk_fraction"] == 0


def test_new_observability_and_estimates_are_prefix_invariant():
    data = bars()
    full = computed(data)
    prefix = computed(data.iloc[:40].copy())
    for key in ("score_availability", "stock_history_observations", "market_history_observations"):
        assert full.iloc[39][key] == prefix.iloc[-1][key]
    first = decide(quality(), full.iloc[39].to_dict(), config=CONFIG)
    second = decide(quality(), prefix.iloc[-1].to_dict(), config=CONFIG)
    assert first["score_evidence"] == second["score_evidence"]


def test_high_partial_score_does_not_override_quality_or_veto():
    f = features(market_ready=False)
    result = decide(quality(status="REJECT"), f, config=CONFIG)
    assert result["score"] >= 75 and result["signal"] == "NO_BUY"
    f.update(ema5_slope=-1., ema20_slope=-1., atr_expansion=2., r5=-.2)
    result = decide(quality(), f, config=CONFIG)
    assert "ACCELERATING_DECLINE" in result["hard_vetoes"]
    assert result["signal"] not in {"BUY", "BUY_SMALL"} and result["risk_fraction"] == 0


def test_partial_metadata_cannot_be_relabelled_as_an_executable_score():
    report = deepcopy(execution_report())
    report["decision"]["score_basis"] = "PARTIAL_DATA"
    assert entry_gate_checks(report)["score"] == "FAIL"
    assert report_failure(report) is not None


def test_native_partial_pipeline_serializes_counts_scores_and_zero_position():
    data = attach_low_entry_v6_context(bars(40), symbol="2330", sector="24",
                                      as_of="2026-09-11T15:00+08:00", filings=financials())
    report = LowEntryScoreV6Strategy().run_latest(data).iloc[-1]["LES v6 Report"]
    decoded = json.loads(json.dumps(report, allow_nan=False))
    projected = project_report(decoded)
    assert projected["LES v6 Score"] == report["decision"]["score"] > 0
    assert projected["建議股數"] == 0 and report["decision"]["signal"] == "DATA_INSUFFICIENT"
    assert report["data"]["stock_history_observations"] == 40
    assert report["data"]["market_history_observations"] == 21
    html = setup_score_summary(decoded)
    assert "日線 40/60；大盤 21/21" in html
    assert "部分資料估算；不可進場" in html and "評分覆蓋率" in html


def test_missing_technical_input_exposes_estimate_without_inflating_admission_score():
    f = features(macd_improving=None, macd_strict=None, macd_signal_strict=None)
    result = decide(quality(), f, config=CONFIG)
    measured = result["setups"][result["score_setup"]]
    assert result["score"] == measured["score"]
    e = result["score_evidence"]
    assert e["coverage_percent"] < 100 and e["estimated_score"] > result["score"]
    report = {"decision": result, "setups": result["setups"]}
    assert "可用資料估算" in setup_score_summary(report)
    assert "不替代進場分數" in setup_score_summary(report)
