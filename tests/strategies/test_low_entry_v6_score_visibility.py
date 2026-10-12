"""Numeric LES evidence must not disappear just because entry is ineligible."""
import json
from copy import deepcopy

import pytest

from application.low_entry_v6_daily import project_report, unavailable_report
from domain.strategies import low_entry_v6_signals as signals
from domain.strategies.low_entry_score_v6 import assess_execution
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from infrastructure.report.low_entry_v6_renderer import setup_score_summary
from tests.strategies.test_low_entry_score_v6 import ACCOUNT, CONFIG, features, quality
from tests.strategies.test_low_entry_v6_setup_recognition import forming, make_report


def no_setup(**changes):
    f = features(no_new_low2=False, higher_low=False, ema5_slope=-.02,
                 ema10=95., ema20_slope=-.02, rs20=-.1, rs_slope5=-.001,
                 rs_improving=False, resilience=-.1, drawdown20=-.2)
    f.update(changes)
    return f


def test_no_setup_preserves_highest_measured_score_without_selecting_an_entry():
    result = signals.decide(quality(), no_setup(), config=CONFIG)
    best = max(result["setups"], key=lambda name: result["setups"][name]["score"])
    assert result["setup"] is None and result["entry_tier"] == "NONE"
    assert result["score"] == result["setups"][best]["score"] > 0
    assert result["score_setup"] == best and result["score_basis"] == "UNQUALIFIED_SETUP"
    assert result["signal"] == "AVOID" and result["reasons"] == ["NO_VALID_SETUP"]
    assert not result["starter_trigger"] and not result["confirmed_trigger"]
    assert result["risk_fraction"] == 0 and result["risk_gate"] == "NOT_EVALUATED"
    assert signals.size_position(result, portfolio_value=1e6, available_cash=5e5,
                                 intended_weight=.2, portfolio_capacity_weight=.2,
                                 amount_ma20=1e8, config=CONFIG) == 0


@pytest.mark.parametrize("setup", signals.SETUPS)
def test_forming_score_keeps_its_source(setup):
    result = signals.decide(quality(), forming(setup), config=CONFIG)
    assert result["score_setup"] == result["setup"] == setup
    assert result["score_basis"] == "FORMING_SETUP"
    assert result["score"] == result["setups"][setup]["score"]
    assert result["signal"] not in signals.EXECUTABLE_SIGNALS


@pytest.mark.parametrize("status,signal", [("REJECT", "NO_BUY"), ("DATA_INSUFFICIENT", "DATA_INSUFFICIENT"),
                                         ("WATCH_QUALITY", "WATCH_ONLY"), ("PROVISIONAL", "WATCH_ONLY")])
def test_quality_failure_does_not_erase_computable_technical_score(status, signal):
    result = signals.decide(quality(status=status), no_setup(), config=CONFIG)
    assert result["signal"] == signal and result["score"] > 0
    assert result["score_basis"] == "UNQUALIFIED_SETUP"
    assert result["risk_fraction"] == 0


@pytest.mark.parametrize("changes", [{"history_ready": False}, {"market_ready": False},
                                     {"history_ready": None}, {"market_ready": None}])
def test_incomplete_history_uses_observable_score_without_granting_entry(changes):
    result = signals.decide(quality(), no_setup(**changes), config=CONFIG)
    assert result["score"] == result["score_evidence"]["estimated_score"]
    assert result["score"] is not None and result["score_setup"] in signals.SETUPS
    assert result["score_basis"] == "PARTIAL_DATA"
    assert result["signal"] == "DATA_INSUFFICIENT" and result["risk_fraction"] == 0


@pytest.mark.parametrize("score", [0., 100.])
def test_real_zero_is_retained_and_high_score_does_not_grant_setup_eligibility(score, monkeypatch):
    f = no_setup()
    setups = signals.score_setups(f, signals.trade_plan(f, config=CONFIG), config=CONFIG)
    for value in setups.values():
        value["score"] = score
    monkeypatch.setattr(signals, "score_setups", lambda *a, **kw: setups)
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["score"] == score and result["score_setup"] == "REVERSAL"  # Stable tie order.
    assert result["signal"] == "AVOID" and result["setup"] is None


def test_executable_lower_score_keeps_its_own_score_source(monkeypatch):
    f = features()
    setups = signals.score_setups(f, signals.trade_plan(f, config=CONFIG), config=CONFIG)
    # A rejected 100-point shape must not supply the selected candidate's score.
    setups["REVERSAL"]["score"] = 70.
    for name in ("TREND_PULLBACK", "LEADERSHIP_CONTINUATION"):
        setups[name].update(score=100., valid=False, recognized=False, entry_tier="NONE")
    monkeypatch.setattr(signals, "score_setups", lambda *a, **kw: setups)
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["signal"] == "BUY_SMALL" and result["score"] == 70
    assert result["score_setup"] == result["setup"] == "REVERSAL"
    assert result["score_basis"] == "SELECTED_SETUP"


def test_plugin_json_projection_and_html_share_the_measured_score(monkeypatch):
    report = make_report(no_setup(), monkeypatch)
    projected = project_report(report)
    decoded = json.loads(projected["LES v6 Report"])
    score = report["decision"]["score"]
    assert score > 0 and report["decision"]["setup"] is None
    assert score == report["setups"]["les_v6"] == decoded["decision"]["score"]
    assert score == projected["LES v6 Score"] == projected["低點分數"]
    assert projected["建議股數"] == projected["Position Fraction"] == 0
    assert entry_gate_checks(report)["score"] == "NOT_EVALUATED"
    html = setup_score_summary(report)
    assert html.startswith(f'{score:.2f}<br>{report["decision"]["score_setup"]}')
    assert "參考評分；無合格型態" in html and "N/A" not in html
    assert report_failure(report) is not None
    forged = deepcopy(report)
    forged["decision"].update(signal="BUY_SMALL", required_score=65., risk_gate="PASS", starter_trigger=True)
    assert report_failure(forged) is not None
    assert assess_execution(forged, {}, now="2026-09-14", trading_sessions=[], portfolio=ACCOUNT)["action"] == "SKIP"


def test_missing_history_estimate_survives_report_projection_but_not_execution(monkeypatch):
    report = make_report(no_setup(history_ready=False), monkeypatch)
    projected = project_report(report)
    score = report["decision"]["score"]
    assert score is not None
    assert projected["LES v6 Score"] == projected["低點分數"] == report["setups"]["les_v6"] == score
    html = setup_score_summary(report)
    assert html.startswith(f"{score:.2f}<br>")
    assert "部分資料估算；不可進場" in html and "評分覆蓋率" in html
    assert "REVERSAL N/A" not in html and "TREND_PULLBACK N/A" not in html
    assert report_failure(report) is not None
    assert projected["建議股數"] == 0


def test_unavailable_report_does_not_invent_a_score():
    report = unavailable_report("2330", "", "2026-09-11", "2026-09-11T15:00+08:00", "NO_HISTORY")
    rendered = setup_score_summary(report)
    assert "Final N/A" in rendered and "非實際 0 分" in rendered
    assert project_report(report)["LES v6 Score"] is None


@pytest.mark.parametrize("score", [None, float("nan"), float("inf")])
def test_nonfinite_legacy_scores_are_not_displayed_as_numbers(score):
    report = {"decision": {"setup": None, "score": 0},
              "setups": {"REVERSAL": {"score": score, "valid": False}}}
    html = setup_score_summary(report)
    assert html.startswith("N/A（技術資料不足）")
    assert "nan" not in html and "inf" not in html and "0.00" not in html
