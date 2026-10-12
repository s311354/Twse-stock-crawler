"""Structure may be recognizable before a tradable trigger exists."""
from copy import deepcopy

import pandas as pd
import pytest

import domain.strategies.low_entry_score_v6 as strategy_module
from application.low_entry_v6_daily import project_report
from domain.strategies import low_entry_v6_signals as signals
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from infrastructure.report.low_entry_v6_renderer import LowEntryV6Renderer, setup_score_summary
from tests.strategies.test_daily_v6_pipeline import build
from tests.strategies.test_low_entry_score_v6 import ACCOUNT, CONFIG, bars, quality, setup_features
from tests.strategies.test_low_entry_v6_gate_contract import changed


def forming(setup, **changes):
    f = setup_features(setup, ema5_slope=-.02, rs_slope5=-.002, rs_improving=False)
    if setup == "TREND_PULLBACK":
        f.update(drawdown60=-.04, drawdown20=-.015)
    f.update(changes)
    return f


def classify(setup, **changes):
    f = forming(setup, **changes)
    return signals.score_setups(f, signals.trade_plan(f, config=CONFIG), config=CONFIG)[setup]


@pytest.mark.parametrize("setup", signals.SETUPS)
def test_held_shapes_are_recognized_before_entry_confirmation(setup):
    observed = classify(setup)
    assert observed["recognized"] and all(observed["recognition_checks"].values())
    assert not observed["valid"] and not observed["strict_valid"]
    assert observed["entry_tier"] == "FORMING"
    result = signals.decide(quality(), forming(setup), config=CONFIG)
    assert result["setup"] == setup and result["score"] == observed["score"]
    assert result["signal"] == "WATCH"
    assert result["reasons"] == ["SETUP_FORMING_WAIT_CONFIRMATION"]
    assert result["risk_gate"] == "NOT_EVALUATED" and result["risk_fraction"] == 0
    assert not result["starter_trigger"] and not result["confirmed_trigger"]
    assert signals.size_position(result, portfolio_value=1e6, available_cash=5e5,
                                 intended_weight=.2, portfolio_capacity_weight=.2,
                                 amount_ma20=1e8, config=CONFIG) == 0


@pytest.mark.parametrize("setup", signals.SETUPS)
@pytest.mark.parametrize("changes", [
    {"history_ready": False}, {"market_ready": False}, {"no_new_low2": False},
    {"support": None}, {"support": 101.}, {"support_intact": False},
    {"atr_expansion": None}, {"atr_expansion": 1.251}, {"atr14": None},
    {"volume": 0.}, {"low": float("nan")}, {"new_lows3": 2.},
])
def test_forming_is_not_a_fallback_for_missing_data_or_broken_support(setup, changes):
    observed = classify(setup, **changes)
    assert not observed["recognized"] and observed["entry_tier"] == "NONE"


@pytest.mark.parametrize("setup,changes", [
    ("REVERSAL", {"drawdown60": -.01, "position60": .8}),
    ("REVERSAL", {"ema5": 103.}),
    ("REVERSAL", {"higher_low": False, "macd_improving": False, "down_volume_contracts": False}),
    ("TREND_PULLBACK", {"ema10": 98., "ema50": 102.}),
    ("TREND_PULLBACK", {"drawdown20": -.151}),
    ("TREND_PULLBACK", {"drawdown20": -.009}),
    ("TREND_PULLBACK", {"ema20_extension": -.501}),
    ("LEADERSHIP_CONTINUATION", {"rs60": None}),
    ("LEADERSHIP_CONTINUATION", {"rs20": 0.}),
    ("LEADERSHIP_CONTINUATION", {"rs_near_high": False}),
    ("LEADERSHIP_CONTINUATION", {"ema20_extension": 1.001}),
])
def test_each_shape_requires_its_own_price_and_trend_structure(setup, changes):
    assert not classify(setup, **changes)["recognized"]


@pytest.mark.parametrize("status,expected", [("REJECT", "NO_BUY"), ("DATA_INSUFFICIENT", "DATA_INSUFFICIENT"),
                                            ("PROVISIONAL", "WATCH_ONLY"), ("WATCH_QUALITY", "WATCH_ONLY")])
def test_forming_never_overrides_quality_status(status, expected):
    result = signals.decide(quality(status=status), forming("REVERSAL"), config=CONFIG)
    assert result["signal"] == expected and result["risk_fraction"] == 0


def test_forming_never_overrides_veto_or_incomplete_history():
    f = forming("REVERSAL", ema20_slope=-.1, atr_expansion=1.5, r5=-.2)
    assert signals.decide(quality(), f, config=CONFIG)["signal"] == "NO_BUY"
    assert signals.decide(quality(), forming("REVERSAL", history_ready=False), config=CONFIG)["signal"] == "DATA_INSUFFICIENT"


def test_high_score_forming_cannot_hide_lower_score_valid_setup(monkeypatch):
    f = setup_features("REVERSAL")
    scored = signals.score_setups(f, signals.trade_plan(f, config=CONFIG), config=CONFIG)
    scored["REVERSAL"]["score"] = 70.
    scored["TREND_PULLBACK"] = dict(classify("TREND_PULLBACK"), score=100.)
    scored.pop("LEADERSHIP_CONTINUATION")
    monkeypatch.setattr(signals, "score_setups", lambda *a, **kw: scored)
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["setup"] == "REVERSAL" and result["signal"] == "BUY_SMALL"
    scored.pop("REVERSAL")
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["score"] == 100 and result["signal"] == "WATCH"
    assert result["risk_fraction"] == 0 and result["position_multiplier"] == 0


def rebound(**changes):
    return setup_features("REVERSAL", **dict(
        higher_low=False, vwap20=101., rs_slope5=-.002, rs_improving=False,
        rs20=-.1, rs60=-.2, consolidation_confirmed=False, **changes))


def test_local_rebound_can_precede_relative_strength_and_vwap_reclaim():
    f = rebound()
    result = signals.decide(quality(), f, config=CONFIG)
    assert result["signal"] == "BUY_SMALL" and result["entry_tier"] == "STARTER_ONLY"
    assert result["setup"] == "REVERSAL"
    setup = result["setups"]["REVERSAL"]
    assert setup["components"]["relative_strength"] == 3.  # Resilience only, no RS points.
    assert setup["components"]["demand"] == 20.  # No VWAP points.
    assert not setup["strict_valid"] and result["risk_gate"] == "PASS"


@pytest.mark.parametrize("changes", [
    {"no_new_low2": False}, {"support_intact": False}, {"r5": 0.},
    {"ema5_slope": 0.}, {"macd_improving": False}, {"clv": .4},
    {"cmf_improving": False, "obv_slope5": 0.}, {"atr_expansion": 1.000001},
    {"market_regime": "RISK_OFF"},
])
def test_local_rebound_requires_all_independent_confirmation_evidence(changes):
    f = rebound(**changes)
    result = signals.decide(quality(), f, config=CONFIG)
    assert not result["setups"]["REVERSAL"]["valid"]
    assert result["signal"] not in signals.EXECUTABLE_SIGNALS


def make_report(f, monkeypatch):
    f = dict(f, pe=None)
    monkeypatch.setattr(strategy_module, "build_features", lambda data, shared: pd.DataFrame([f]))
    complete_quality = quality(coverage=dict.fromkeys(
        ("balance_sheet", "profitability", "growth", "return_on_capital", "cash_flow"), 100.))
    monkeypatch.setattr(strategy_module, "evaluate_quality", lambda *a, **kw: complete_quality)
    data = strategy_module.attach_low_entry_v6_context(bars(1), symbol="2330", sector="24",
            as_of="2026-09-11T15:00+08:00", filings=[], portfolio=ACCOUNT)
    # These regressions pin the former forming/setup admission policy explicitly.
    return strategy_module.LowEntryScoreV6Strategy(config=CONFIG).run_latest(data).iloc[-1]["LES v6 Report"]


@pytest.mark.parametrize("setup", signals.SETUPS)
def test_forming_serialization_projection_and_report_do_not_authorize_entry(setup, monkeypatch):
    template = build()[1][0]
    report = make_report(forming(setup), monkeypatch)
    projected = project_report(report)
    assert projected["LES v6 Setup"] == setup and projected["LES v6 Score"] > 0
    assert projected["建議股數"] == projected["Position Fraction"] == 0
    gates = entry_gate_checks(report)
    assert all(gates[key] == "NOT_EVALUATED" for key in ("score", "trigger", "risk"))
    assert report_failure(report) is not None
    renderer = LowEntryV6Renderer()
    html = renderer.render((changed(template, report),))
    assert "型態形成中，尚未確認進場" in html and "N/A（無合格型態）" not in html
    assert "本次沒有 BUY / BUY_SMALL 訊號" in html
    assert 'data-section="technical-observations"' not in html
    assert 'data-section="leadership-watch"' not in html


@pytest.mark.parametrize("tier", ["FORMING", "STARTER_ONLY"])
def test_forged_forming_report_cannot_pass_score_or_execution(tier, monkeypatch):
    report = make_report(forming("REVERSAL"), monkeypatch)
    report["decision"].update(signal="BUY_SMALL", entry_tier=tier, required_score=65.,
                              risk_gate="PASS", starter_trigger=True, risk_fraction=.0025)
    report["entry"]["starter_trigger"] = True
    report["setups"]["REVERSAL"].update(valid=True, entry_tier="STARTER_ONLY")
    assert entry_gate_checks(report)["score"] == "FAIL"
    assert report_failure(report) is not None
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10+08:00", price=100.5,
                 bid=100.4, ask=100.5, restricted=False)
    result = strategy_module.assess_execution(report, quote, now=quote["timestamp"],
                     trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    assert result["action"] == "SKIP" and result["shares"] == 0


def test_forming_label_preserves_score_and_escapes_metadata():
    report = {"decision": {"setup": "REVERSAL", "score": 64., "entry_tier": "FORMING"},
              "setups": {"REVERSAL": dict(classify("REVERSAL"), blocking_conditions=['<script>'])}}
    html = setup_score_summary(deepcopy(report))
    assert "64.00" in html and "形成中" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
