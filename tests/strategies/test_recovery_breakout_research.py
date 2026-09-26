"""Generic research hypotheses, causal replay, and production isolation."""

from copy import deepcopy
from dataclasses import replace
import json

import pandas as pd
import pytest

from domain.features import SharedFeaturePipeline
from domain.strategy import available_strategies
from domain.strategies.low_entry_score_v6 import (
    LowEntryScoreV6Strategy, assess_execution, attach_low_entry_v6_context,
)
from domain.strategies.low_entry_v6_features import build_features
from domain.strategies.low_entry_v6_report import report_failure
from domain.strategies.recovery_breakout_research import (
    RESEARCH_COLUMN, RecoveryBreakoutConfig, RecoveryBreakoutResearchStrategy,
    build_research_features, evaluate_research,
)
from interface.les_v6_research import analyze_request, main
from tests.strategies.test_low_entry_score_v6 import bars, financials
from tests.strategies.test_low_entry_v6_3016_causal import raw_bars


CONFIG = RecoveryBreakoutConfig()


def history(symbol="2330", sector="ELEC"):
    data = bars()
    data["Close"] = ([120.] * 30 + [120. - i for i in range(20)]
                     + [80. + i for i in range(20)]
                     + [100. + (.2 if i % 2 else -.2) for i in range(29)] + [102.])
    data["Open"], data["High"], data["Low"] = data.Close - .5, data.Close + .3, data.Close - .8
    data["TAIEX Close"] = 20_000.
    data.loc[data.index[-1], "Volume"] = 2_000_000.
    data["Amount"] = data.Close * data.Volume
    records = [dict(row, symbol=symbol) for row in financials()]
    return attach_low_entry_v6_context(data, symbol=symbol, sector=sector,
                                       as_of="2026-09-11T15:00:00+08:00", filings=records)


@pytest.fixture
def evidence():
    data = history()
    shared = SharedFeaturePipeline().build(data)
    f = build_research_features(build_features(data, shared), CONFIG).iloc[-1].to_dict()
    production = LowEntryScoreV6Strategy().run_latest(data)["LES v6 Report"].iloc[0]
    return f, production


def test_both_hypotheses_can_confirm_without_producing_an_order(evidence):
    f, production = evidence
    report = evaluate_research(f, production, CONFIG)
    assert report["research_signal"] == "RESEARCH_CONFIRMED"
    assert {p["state"] for p in report["patterns"].values()} == {"CONFIRMED"}
    assert report["scenario"]["floor"] == 99.
    assert report["scenario"]["breakout_level"] == 100.5
    assert report["research_only"] and not report["executable"]
    assert report["sizing"] == {"indicative_shares": 0, "target_weight": 0.}
    assert report["calibration_status"] == "UNVALIDATED"
    assert report["market"] == "TWSE"
    assert report["production_reference"]["parameter_version"] == production["parameter_version"]
    assert report["payoff_scenarios"]["expected_return"] is None
    assert report["payoff_scenarios"]["estimated_win_rate"] is None
    assert "decision" not in report and "LES v6 Report" not in report


@pytest.mark.parametrize("symbol,sector", [("3016", "ELEC"), ("2610", "AIR"), ("9999", "NEW_SECTOR")])
def test_identical_observations_get_identical_patterns_for_any_symbol_sector(symbol, sector):
    plugin = RecoveryBreakoutResearchStrategy()
    actual = plugin.run_latest(history(symbol, sector))[RESEARCH_COLUMN].iloc[0]
    reference = plugin.run_latest(history())[RESEARCH_COLUMN].iloc[0]
    assert actual["patterns"] == reference["patterns"]
    assert actual["research_signal"] == reference["research_signal"]
    assert actual["parameters"] == reference["parameters"]


def test_range_is_frozen_before_signal_and_cannot_expand_with_a_gap():
    data = history()
    before = build_research_features(build_features(data, SharedFeaturePipeline().build(data)), CONFIG)
    data.loc[data.index[-1], ["Open", "High", "Low", "Close"]] = [149., 151., 148., 150.]
    after = build_research_features(build_features(data, SharedFeaturePipeline().build(data)), CONFIG)
    for key in ("base_floor", "base_ceiling", "base_atr", "base_width_atr", "prior_max_drawdown", "prior_atr"):
        assert after[key].iloc[-1] == before[key].iloc[-1]
    report = RecoveryBreakoutResearchStrategy().run_latest(data)[RESEARCH_COLUMN].iloc[0]
    assert report["research_signal"] != "RESEARCH_CONFIRMED"


@pytest.mark.parametrize("key,value", [("clv", -.5), ("research_volume_ratio", .5), ("rs20", -.1)])
def test_weak_close_volume_or_relative_strength_cannot_confirm(evidence, key, value):
    f, production = evidence
    report = evaluate_research(dict(f, **{key: value}), production, CONFIG)
    assert report["research_signal"] != "RESEARCH_CONFIRMED"


@pytest.mark.parametrize("change", ["veto", "quality_reject", "restricted"])
def test_hard_blocks_override_a_confirmed_technical_pattern(evidence, change):
    f, production = deepcopy(evidence)
    if change == "veto":
        production["downtrend_gate"]["reasons"] = ["ACCELERATING_DECLINE"]
    elif change == "quality_reject":
        production["quality"]["status"] = "REJECT"
    else:
        f["restricted"] = True
    report = evaluate_research(f, production, CONFIG)
    assert report["patterns"]["BASE_BREAKOUT"]["state"] == "CONFIRMED"
    assert report["research_signal"] == "RESEARCH_BLOCKED"
    assert report["hard_blocks"] and not report["executable"]


def test_watch_quality_is_visible_for_research_without_changing_production(evidence):
    f, production = deepcopy(evidence)
    production["quality"].update(status="WATCH_QUALITY", score=68.)
    production["decision"]["signal"] = "WATCH_ONLY"
    original = deepcopy(production)
    report = evaluate_research(f, production, CONFIG)
    assert report["research_signal"] == "RESEARCH_CONFIRMED"
    assert report["production_reference"]["signal"] == "WATCH_ONLY"
    assert report["quality"]["score"] == 68.
    assert production == original


@pytest.mark.parametrize("change", ["filings", "benchmark", "infinite_benchmark", "volume", "restriction", "oldest_bar", "short_history"])
def test_missing_evidence_never_becomes_confirmation(change):
    data = history()
    if change == "filings":
        data.attrs["les_v6_context"]["filings"] = []
    elif change == "benchmark":
        data.loc[data.index[-2], "TAIEX Close"] = None
    elif change == "infinite_benchmark":
        data.loc[data.index[-2], "TAIEX Close"] = float("inf")
    elif change == "volume":
        data.loc[data.index[-1], "Volume"] = 0
    elif change == "restriction":
        data = data.drop(columns="Restricted")
    elif change == "oldest_bar":
        data.loc[data.index[-61], "Low"] = -1
    else:
        data = data.tail(40)
    report = RecoveryBreakoutResearchStrategy().run_latest(data)[RESEARCH_COLUMN].iloc[0]
    assert report["research_signal"] == "RESEARCH_DATA_INSUFFICIENT"
    assert report["missing_evidence"] and report["sizing"]["indicative_shares"] == 0


@pytest.mark.parametrize("config", [replace(CONFIG, correction_window=80), replace(CONFIG, stabilization_window=100)])
def test_custom_windows_require_complete_ohlcv_coverage(config):
    data = history()
    data.loc[data.index[-81], "High"] = -1
    report = RecoveryBreakoutResearchStrategy(config).run_latest(data)[RESEARCH_COLUMN].iloc[0]
    assert "COMPLETE_RESEARCH_HISTORY" in report["missing_evidence"]
    assert report["research_signal"] == "RESEARCH_DATA_INSUFFICIENT"


def test_research_does_not_invent_a_target_when_known_resistance_is_absent(evidence):
    f, production = evidence
    report = evaluate_research(dict(f, swing_high=None, hh20=None, hh60=None), production, CONFIG)
    assert report["scenario"]["target_price"] is None
    assert report["payoff_scenarios"]["status"] == "NO_VALID_UPSIDE_TARGET"
    assert report["payoff_scenarios"]["net_reward_risk"] is None


def test_later_rally_does_not_rewrite_3016_research_or_production():
    def attached(include_future):
        data = raw_bars(include_future=include_future)
        return attach_low_entry_v6_context(data, symbol="3016", sector="ELEC",
            as_of=data.Date.iloc[-1].strftime("%Y-%m-%dT15:00:00+08:00"), filings=[])
    past, future = attached(False), attached(True)
    plugin = RecoveryBreakoutResearchStrategy()
    before, after = plugin.run(past), plugin.run(future)
    assert before[RESEARCH_COLUMN].tolist() == after[RESEARCH_COLUMN].iloc[:len(past)].tolist()
    last = before[RESEARCH_COLUMN].iloc[-1]
    assert last["signal_date"] == "2026-09-18"
    assert not last["executable"]
    assert last["production_reference"]["signal"] not in {"BUY", "BUY_SMALL"}
    assert last["scenario"] is None or last["scenario"]["target_price"] != 160.


def test_future_filing_cannot_repair_historical_research_quality():
    data = history()
    baseline = RecoveryBreakoutResearchStrategy().run(data)
    data.attrs["les_v6_context"]["filings"] += [dict(row, available_at="2026-09-12T00:00:00+08:00",
                                                    revision_id="future", net_income=999999.)
                                               for row in financials()]
    changed = RecoveryBreakoutResearchStrategy().run(data)
    assert changed[RESEARCH_COLUMN].tolist() == baseline[RESEARCH_COLUMN].tolist()


def test_formal_output_is_unchanged_and_research_is_rejected_by_execution():
    data = history()
    original = data.copy(deep=True)
    production = LowEntryScoreV6Strategy().run_latest(data)
    research = RecoveryBreakoutResearchStrategy().run_latest(data)
    pd.testing.assert_frame_equal(data, original)
    assert "LES v6 Report" not in research
    assert "recovery_breakout_research" not in available_strategies()
    pd.testing.assert_frame_equal(production, LowEntryScoreV6Strategy().run_latest(data))
    report = research[RESEARCH_COLUMN].iloc[0]
    assert report_failure(report) is not None
    result = assess_execution(report, {}, now="2026-09-14T09:05:00+08:00", trading_sessions=[], portfolio={})
    assert result["action"] == "SKIP" and result["shares"] == 0


def test_enriched_input_does_not_leak_old_actionable_columns():
    data = history()
    data["LES v6 Report"] = [{"decision": {"signal": "BUY"}}] * len(data)
    data["低點決策"] = "BUY"
    data["建議股數"] = 1000
    result = RecoveryBreakoutResearchStrategy().run_latest(data)
    assert not {"LES v6 Report", "低點決策", "建議股數"}.intersection(result.columns)


@pytest.mark.parametrize("change", ["incomplete", "future", "duplicate"])
def test_invalid_daily_time_evidence_is_rejected(change):
    data = history()
    if change == "incomplete":
        data["Complete"] = False
    elif change == "future":
        data.attrs["les_v6_context"]["as_of"] = "2026-09-11T12:00:00+08:00"
    else:
        data.loc[data.index[-1], "Date"] = data.Date.iloc[-2]
    with pytest.raises(ValueError):
        RecoveryBreakoutResearchStrategy().run_latest(data)


@pytest.mark.parametrize("changes", [{"base_window": 0}, {"correction_window": 1.5},
                                     {"min_correction": 1}, {"max_base_width_atr": float("nan")},
                                     {"base_window": 80}])
def test_invalid_research_parameters_fail_fast(changes):
    with pytest.raises(ValueError):
        replace(CONFIG, **changes)


def test_offline_cli_supports_latest_universe_and_causal_history(tmp_path, capsys):
    data = history()
    data["Date"] = data.Date.dt.strftime("%Y-%m-%d")
    request = {"bars": data.to_dict("records"), "context": data.attrs["les_v6_context"]}
    payload = {"securities": [request]}
    path = tmp_path / "research.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert main([str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["reports"]) == 1
    assert result["summary"] == {"RESEARCH_CONFIRMED": 1}
    assert result["evaluation_mode"] == "AS_OF_SNAPSHOT"
    replay = analyze_request(payload, history=True)
    assert replay["evaluation_mode"] == "CLOSE_TIME_REPLAY"
    assert len(replay["reports"]) == len(data)
    assert replay["reports"][0]["research_signal"] == "RESEARCH_DATA_INSUFFICIENT"
    assert main([str(path), "--history"]) == 0
    assert len(json.loads(capsys.readouterr().out)["reports"]) == len(data)


def test_empty_input_produces_no_candidates():
    assert analyze_request({"securities": []})["reports"] == []
