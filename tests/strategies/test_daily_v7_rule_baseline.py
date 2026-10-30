"""SYNTHETIC ONLY: model-free daily wiring; not real market eligibility."""
from copy import deepcopy
from dataclasses import replace
import json

import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.market_history import MarketHistoryAssembler
from application.low_entry_v7_daily import daily_report, project_report
from domain.les_v7.contracts import ReferencePlan, canonical, timestamp
from domain.les_v7.report import validate_report
from domain.trading_restrictions import CURRENT_RESTRICTION_SCOPE, RESTRICTION_ENDPOINTS
from infrastructure.daily_parallel import DailyProcessExecutor
from infrastructure.report.html_renderer import RendererFactory
from infrastructure.storage.csv_repository import CsvRepository
from infrastructure.storage.report_evidence import iter_report_bundle
from tests.strategies.test_daily_v6_pipeline import daily_market
from tests.strategies.test_v7_reported_quality import source_records


CUTOFF = "2026-10-14T15:00:00+08:00"


def rule_market():
    market = daily_market()
    records = source_records()
    dates = pd.bdate_range(end="2026-10-14", periods=len(market.dates[0])).strftime("%Y-%m-%d").tolist()
    restriction = dict(source="TWSE_OPENAPI", session_date=dates[-1], observation_date=dates[-1],
        scope=CURRENT_RESTRICTION_SCOPE, available_at=CUTOFF, restricted=False,
        endpoints=list(RESTRICTION_ENDPOINTS))
    return replace(market, dates=[dates], entry_model="v7", les_v7_native_evidence=True,
        les_v7_rule_baseline=True, les_v7_bundle_json=None, signal_dates={"2330": CUTOFF},
        fallback_date=dates[-1], trading_restrictions={"2330": restriction},
        market_observed_at=CUTOFF,
        market_breadth_snapshots={dates[-1]: dict(source_final=True, as_of=dates[-1])},
        fundamental_snapshots={"2330": {"sector": "24", "les_v6_filings": records,
            "les_v6_mops_filings": records}})


def build(market=None, *, parallel=False):
    builder = AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)) if parallel else AnalysisDatasetBuilder()
    return builder, builder.build_decisions(market or rule_market(), [[1, 1, 1, 1, .01]])


def test_native_daily_rules_compute_real_factor_score_without_fake_forecast():
    market = rule_market()
    before = deepcopy(market.fundamental_snapshots)
    report = daily_report(MarketHistoryAssembler().build(market, 0))
    envelope, diagnostics = report["envelope"], report["diagnostics"]
    assert envelope["model"]["model_status"] == "RULE_BASED_BASELINE"
    assert envelope["model"]["bundle_sha256"] is None
    assert envelope["quality"]["status"] == "PASS"
    assert envelope["score"]["raw_score"] > 0
    assert envelope["score"]["final_score"] is not None
    assert envelope["localization"]["status"] == "NOT_APPLICABLE"
    assert envelope["localization"]["p_entry_given_fill"] is None
    assert diagnostics["localization"]["calibration_version"] is None
    gates = {row["gate_id"]: row["state"] for row in diagnostics["gate_receipts"]}
    assert gates["DATA_INTEGRITY"] == gates["TRADING_STATUS"] == "PASS"
    assert gates["VALUATION"] == "UNKNOWN"  # Missing genuine annual EPS source companion.
    assert envelope["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert report["pae_authorization"]["authorized"] is False
    assert "MODEL_ARTIFACTS_NOT_PROVIDED" not in envelope["strategy"]["reason_codes"]
    assert diagnostics["reference_plan"]["entry"] == pytest.approx(market.closes[0][-1] / 1000)
    assert market.fundamental_snapshots == before
    validate_report(report)


def test_default_rule_pipeline_csv_and_renderer_share_same_native_report(tmp_path):
    builder, decisions = build()
    decision = decisions[0]
    report = json.loads(decision.value("LES v7 Report"))
    assert decision.value("LES Model Version") == "7.0.0"
    assert decision.value("LES v6 Report") is None
    assert decision.value("LES v7 Raw Score") == report["envelope"]["score"]["raw_score"]
    assert decision.value("低點分數") == report["envelope"]["score"]["final_score"]
    assert decision.value("LES v7 Required Score") in {62, 72}
    assert decision.value("Position Fraction") == 0
    CsvRepository(tmp_path).write_analysis_dataset("synthetic_rules", builder.project_decisions(decisions))
    assert list(iter_report_bundle(tmp_path / "synthetic_rules_les_v7_reports.json", require_reference=True)) == [report]
    html = RendererFactory.get_renderer("daily_strategy_email").render(decisions)
    assert '<strong>V7 模型</strong>' in html
    assert "使用 v6 型態規則與固定配分" in html
    assert "固定配分 · 未使用機率預測模型" in html
    assert "預測／校準：不適用" not in html
    assert "不適用：v6" not in html
    assert "BUY 72／BUY_SMALL 62" in html
    assert "校準 p_entry N/A" not in html
    assert canonical(json.loads(decision.value("LES v7 Report"))) == canonical(report)


def test_daily_baseline_flag_survives_spawn_with_identical_result():
    _, serial = build()
    _, parallel = build(parallel=True)
    assert parallel == serial


@pytest.mark.parametrize("change", ["missing_capture", "stale_session", "future_capture", "future_restriction"])
def test_rule_score_never_turns_missing_native_safety_into_entry(change):
    market = rule_market()
    if change == "missing_capture":
        market = replace(market, market_observed_at=None)
    elif change == "stale_session":
        market = replace(market, fallback_date="2026-10-15")
    elif change == "future_capture":
        market = replace(market, market_observed_at="2026-10-15T00:00:00Z")
    else:
        restrictions = deepcopy(market.trading_restrictions)
        restrictions["2330"]["available_at"] = "2026-10-15T00:00:00Z"
        market = replace(market, trading_restrictions=restrictions)
    report = daily_report(MarketHistoryAssembler().build(market, 0))
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert report["pae_authorization"]["authorized"] is False
    validate_report(report)


def test_daily_baseline_is_not_silently_used_by_legacy_manual_or_replay_callers():
    market = rule_market()
    legacy = replace(market, les_v7_rule_baseline=False, les_v7_native_evidence=False)
    report = daily_report(MarketHistoryAssembler().build(legacy, 0))
    assert "rule_baseline" not in report["diagnostics"]
    assert report["envelope"]["score"]["final_score"] is None
    assert "MODEL_ARTIFACTS_NOT_PROVIDED" in report["envelope"]["strategy"]["reason_codes"]
    history = MarketHistoryAssembler().build(market, 0)
    history.attrs["replay_as_of"] = timestamp(CUTOFF).isoformat()
    with pytest.raises(ValueError, match="CURRENT_NATIVE_DAILY_REQUIRED"):
        daily_report(history)


def test_partial_rule_result_projects_exact_capped_score_not_probability():
    report = daily_report(MarketHistoryAssembler().build(rule_market(), 0))
    projected = project_report(report)
    assert projected["低點分數"] == report["envelope"]["score"]["final_score"]
    assert projected["RewardRisk"] == ReferencePlan(**report["diagnostics"]["reference_plan"]).net_rr
    assert projected["LES v7 Execution Eligible"] is False
    assert projected["建議股數"] == 0


def test_flat_r4_projection_keeps_execution_rejection_separate_from_falling_knife():
    report = daily_report(MarketHistoryAssembler().build(rule_market(), 0))
    # The synthetic H20 stop/RR fails native risk, while rules retain the
    # measured raw score. R4 rejects it without falsely calling it Falling Knife.
    assert report["diagnostics"]["safety_caps"]["BASELINE_EXECUTION_FAIL"] == 0
    assert report["diagnostics"]["rule_baseline"]["technical_vetoes"] == []
    assert report["envelope"]["strategy"]["strategy_decision"] == "NO_BUY"
    projected = project_report(report)
    assert projected["Falling Knife Veto"] is False
    assert projected["Falling Knife State"] == "CLEAR"
    assert projected["No Veto"] is True
    assert projected["Risk Gate"] == "FAIL"
    assert projected["低點決策"] == "NO_BUY"
    assert projected["建議股數"] == 0
    assert projected["LES v7 Execution Eligible"] is False
