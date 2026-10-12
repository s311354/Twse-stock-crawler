"""Five completed-day entry gates agree across projection, renderer and execution."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from application.low_entry_v6_daily import project_report, unavailable_report
from domain.model_contracts import FieldSource, FieldUnit
from application.strategy_contracts import StrategyDecisionSchema, StrategyContractError
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from infrastructure.report.html_renderer import RendererFactory
from infrastructure.report.low_entry_v6_renderer import LowEntryV6Renderer
from aistock_twse_integration.les_v6_execution import execute_request
from tests.strategies.test_daily_v6_pipeline import build
from tests.strategies.test_les_v6_execution_bridge import request


@pytest.fixture
def daily_decision(active_industry_profiles):
    return build()[1][0]


def changed(decision, report):
    values = project_report(report)
    return replace(decision, fields=tuple(replace(field, value=values[field.name])
                                         if field.name in values else field for field in decision.fields))


def test_raw_twse_mops_pipeline_projects_actual_v6_gate_sources(daily_decision):
    report = json.loads(daily_decision.value("LES v6 Report"))
    original = deepcopy(report)
    gates = entry_gate_checks(report)
    assert gates == dict.fromkeys(("quality", "no_veto", "score", "trigger", "risk"), "PASS")
    assert report == original  # Inspection must not change frozen evidence.
    for column in ("Quality Gate", "No Veto", "LES v6 Score Gate", "LES v6 Trigger Gate", "Risk Gate"):
        field = daily_decision.field(column)
        assert field.source == FieldSource.LES_V6
        assert field.model_version == "6.0.0"
        assert field.value is True if column == "No Veto" else field.value == "PASS"
        assert field.unit == (FieldUnit.BOOLEAN if column == "No Veto" else FieldUnit.TEXT)
    assert daily_decision.value("LES v5 Final Quality Status") == "DATA_INSUFFICIENT"
    assert daily_decision.value("低點決策") == "BUY_SMALL"


@pytest.mark.parametrize("column", ["Quality Gate", "No Veto", "LES v6 Score Gate", "LES v6 Trigger Gate", "Risk Gate"])
def test_flat_gate_fields_cannot_disagree_with_frozen_v6_report(daily_decision, column):
    damaged = replace(daily_decision, fields=tuple(replace(field, value=False if column == "No Veto" else "FAIL")
                                                  if field.name == column else field for field in daily_decision.fields))
    with pytest.raises(StrategyContractError, match="v6 projection mismatch"):
        StrategyDecisionSchema().validate(damaged)


@pytest.mark.parametrize("path,value,gate", [
    (("quality", "status"), "REJECT", "quality"),
    (("data", "confidence"), None, "quality"),
    (("quality", "coverage"), {}, "quality"),
    (("downtrend_gate", "hard_veto"), True, "no_veto"),
    (("technical", "support"), None, "no_veto"),
    (("decision", "confirmed_trigger"), False, "trigger"),
    (("decision", "confirmed_trigger"), None, "trigger"),
    (("entry", "confirmed_trigger"), False, "trigger"),
    (("technical", "break_previous_high"), False, "trigger"),
    (("decision", "risk_gate"), "FAIL", "risk"),
    (("entry", "reward_risk"), 999, "risk"),
    (("entry", "target_price"), 999, "risk"),
    (("entry", "stop_price"), 100, "risk"),
    (("entry", "trigger_price"), 1, "risk"),
    (("technical", "spread"), .02, "risk"),
])
def test_high_score_cannot_replace_any_gate_or_fabricate_a_plan(path, value, gate):
    payload = request()
    report = payload["report"]
    report["decision"]["score"] = 100
    target = report
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    assert entry_gate_checks(report)[gate] != "PASS"
    assert report_failure(report) is not None
    assert execute_request(payload)["action"] == "SKIP"


@pytest.mark.parametrize("signal,score,claimed_floor", [("BUY", 74, 60), ("BUY_SMALL", 64, 60)])
def test_caller_cannot_lower_its_own_score_requirement(signal, score, claimed_floor):
    payload = request()
    payload["report"]["decision"].update(signal=signal, score=score, required_score=claimed_floor)
    assert entry_gate_checks(payload["report"])["score"] == "FAIL"
    assert report_failure(payload["report"]) == "V6_SCORE_BELOW_REQUIRED"
    assert execute_request(payload)["action"] == "SKIP"


def test_risk_off_uses_the_existing_extra_score_requirement():
    report = request()["report"]
    report["technical"]["market_regime"] = "RISK_OFF"
    report["decision"].update(score=79, required_score=75)
    assert entry_gate_checks(report)["score"] == "FAIL"
    report["decision"].update(score=request()["report"]["decision"]["score"], required_score=80)
    assert entry_gate_checks(report)["score"] == "PASS"


def test_published_score_is_actual_computed_score_not_an_inflated_label():
    payload = request()
    assert payload["report"]["decision"]["score"] == 100
    # Remove earned momentum points while retaining the published 100 label.
    payload["report"]["technical"]["rsi_cross40"] = False
    assert report_failure(payload["report"]) == "V6_SCORE_OR_SETUP_MISMATCH"
    assert execute_request(payload)["action"] == "SKIP"


def test_risk_off_requires_extra_confirmation_even_with_published_trigger():
    report = request()["report"]
    report["technical"].update(market_regime="RISK_OFF", resilience=0)
    report["decision"]["required_score"] = 80
    assert report["decision"]["confirmed_trigger"] is True
    assert entry_gate_checks(report)["trigger"] == "FAIL"


def test_direct_plugin_keeps_explicit_matching_config_support():
    from domain.strategies.low_entry_score_v6 import assess_execution
    from domain.strategies.low_entry_v6_signals import LowEntryV6Config

    payload = request()
    config = LowEntryV6Config(decision_policy_version=1, quote_max_age_seconds=45)
    payload["report"]["parameters"] = vars(config)
    assert report_failure(payload["report"]) == "EXECUTION_CONFIG_MISMATCH"
    assert report_failure(payload["report"], config=config) is None
    assert assess_execution(**payload, config=config)["action"] == "ELIGIBLE"


def test_recomputed_veto_cannot_be_cleared_by_a_false_label():
    payload = request()
    payload["report"]["technical"].update(new_lows3=3, clv=-.8, undercut_reclaim=False)
    assert payload["report"]["downtrend_gate"]["hard_veto"] is False
    assert entry_gate_checks(payload["report"])["no_veto"] == "FAIL"
    assert execute_request(payload)["action"] == "SKIP"


def test_unavailable_data_does_not_display_no_veto_as_pass():
    report = unavailable_report("2330", "測試", "2026-09-11", "2026-09-11T15:00:00+08:00", "MISSING")
    assert entry_gate_checks(report) == dict(quality="UNKNOWN", no_veto="UNKNOWN", score="UNKNOWN",
                                           trigger="UNKNOWN", risk="UNKNOWN")
    assert project_report(report)["No Veto"] is None
    assert project_report(report)["Quality Gate"] == "UNKNOWN"


def test_buy_small_uses_starter_not_confirmed_trigger():
    report = request()["report"]
    report["decision"].update(signal="BUY_SMALL", required_score=65, confirmed_trigger=False)
    assert entry_gate_checks(report)["trigger"] == "PASS"
    assert report_failure(report) is None
    report["decision"]["starter_trigger"] = False
    assert report_failure(report) == "V6_TRIGGER_NOT_CONFIRMED"


def test_entry_summary_shows_five_gates_without_claiming_a_fill(daily_decision, tmp_path):
    rendered = LowEntryV6Renderer().render((daily_decision,))
    for gate in ("quality", "no_veto", "score", "trigger", "risk"):
        assert f'data-entry-gate="{gate}" data-status="PASS"' in rendered
    assert "收盤條件明細一致性（非額外資格）" in rendered
    assert "下一交易日確認，尚未成交" in rendered
    assert "毛 RR " in rendered and "淨 RR " in rendered
    assert "低點進場判斷（Low-Entry Model v6）" in rendered
    assert 'data-section="technical-observations"' not in rendered
    assert 'data-section="leadership-watch"' not in rendered
    # A self-contained local inspection artifact; no SMTP, credentials or orders.
    daily_html = RendererFactory.get_renderer("daily_strategy_email").render((daily_decision,))
    assert 'class="twse-daily-report"' in daily_html
    (tmp_path / "v6-five-gates.html").write_text(
        '<!doctype html><html lang="zh-Hant"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>LES v6 synthetic gate check</title><body>' + daily_html + '</body></html>', encoding="utf-8")


def test_invalid_published_buy_is_not_a_formal_renderer_candidate(daily_decision):
    report = json.loads(daily_decision.value("LES v6 Report"))
    report["decision"]["confirmed_trigger"] = False
    item = changed(daily_decision, report)
    renderer = LowEntryV6Renderer()
    assert "本次沒有 BUY / BUY_SMALL 訊號" in renderer.render_entry_summary((item,))
    analysis = renderer.render_analysis((item,))
    assert "訊號無效（BUY_SMALL）" in analysis
    assert "V6_TRIGGER_NOT_CONFIRMED" in analysis
    assert 'data-entry-gate="trigger" data-status="FAIL"' in analysis
