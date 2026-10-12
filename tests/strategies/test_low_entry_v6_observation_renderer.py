"""Internal observation/research metadata must not add report sections or reasons."""
from copy import deepcopy
from dataclasses import replace
import json
import re

import pytest

from infrastructure.report.low_entry_v6_renderer import LowEntryV6Renderer
from domain.strategies.low_entry_v6_quality import QualityResult
from domain.strategies.low_entry_v6_signals import LowEntryV6Config, decide
from domain.strategies.low_entry_score_v6 import parameter_version
from tests.strategies.test_daily_v6_pipeline import build


@pytest.fixture
def daily_decision(active_industry_profiles):
    return build()[1][0]


def with_report(decision, report):
    values = {"LES v6 Report": json.dumps(report, ensure_ascii=False), "低點決策": report["decision"]["signal"]}
    return replace(decision, fields=tuple(replace(field, value=values[field.name]) if field.name in values else field
                                         for field in decision.fields))


def observation_report(decision, *, signal="NO_BUY", quality="REJECT"):
    report = json.loads(decision.value("LES v6 Report"))
    # This legacy rendering contract permits manually selected lower tiers;
    # v2 validates its sole native decision and is covered by unified tests.
    config = LowEntryV6Config(decision_policy_version=1)
    q, data = report["quality"], report["data"]
    original = QualityResult(q["status"], q["score"], data["confidence"], data["integrity"],
                             q["components"], q["coverage"], q["metrics"], reasons=tuple(q["reasons"]))
    report["parameters"] = vars(config)
    report["parameter_version"] = parameter_version(config)
    report["decision"] = decide(original, report["technical"], config=config,
                                valuation=report["valuation"]["interpretation"])
    # Keep the explicit legacy fixture internally consistent after daily
    # scoring switched to Hybrid. No production report is relabelled here.
    report["entry"].update(report["decision"]["plan"],
                          starter_trigger=report["decision"]["starter_trigger"],
                          confirmed_trigger=report["decision"]["confirmed_trigger"])
    report["quality"]["status"] = quality
    report["decision"].update(signal=signal, reasons=["ORIGINAL_DECISION_REASON"], technical_observation={
        "patterns": ["RELATIVE_STRENGTH", "LOW_ZONE_RECOVERY"], "is_entry_signal": False,
        "blocking_reasons": ["OBSERVATION_ONLY_BLOCKER"],
    })
    return report


def section(rendered, name):
    match = re.search(rf'<section data-section="{name}"[^>]*>(.*?)</section>', rendered, re.DOTALL)
    return match.group(1) if match else ""


@pytest.mark.parametrize("signal,quality", [
    ("NO_BUY", "REJECT"), ("WATCH_ONLY", "WATCH_QUALITY"), ("AVOID", "PASS"),
    ("BUY", "PASS"), ("BUY_SMALL", "PASS"),
])
@pytest.mark.parametrize("include_entry_summary", [True, False])
def test_legacy_observation_is_ignored_without_changing_actual_decision(daily_decision, signal, quality,
                                                                     include_entry_summary):
    report = observation_report(daily_decision, signal=signal, quality=quality)
    renderer = LowEntryV6Renderer()
    rendered = renderer.render_analysis((with_report(daily_decision, report),),
                                        include_entry_summary=include_entry_summary)
    report["decision"].pop("technical_observation")
    baseline = renderer.render_analysis((with_report(daily_decision, report),),
                                        include_entry_summary=include_entry_summary)

    assert rendered == baseline
    assert 'data-section="technical-observations"' not in rendered
    assert "技術觀察（未通過進場條件）" not in rendered
    assert "OBSERVATION_ONLY_BLOCKER" not in rendered
    analysis = section(rendered, "v6-analysis")
    assert f'>{signal}</td>' in analysis and f'{quality} / ' in analysis
    assert "ORIGINAL_DECISION_REASON" in analysis
    assert f'{report["symbol"]} {report["name"]}' in analysis
    entry = section(rendered, "entry-summary")
    if not include_entry_summary:
        assert not entry
    elif signal in {"BUY", "BUY_SMALL"}:
        assert f'>{signal}</td>' in entry
    else:
        assert "本次沒有 BUY / BUY_SMALL 訊號" in entry


@pytest.mark.parametrize("metadata", [None, {}, {"patterns": [], "is_entry_signal": False},
                                     {"patterns": ["<script>unsafe</script>"], "is_entry_signal": True}])
def test_non_observation_metadata_also_has_no_rendered_effect(daily_decision, metadata):
    report = observation_report(daily_decision)
    report["decision"].pop("technical_observation")
    renderer = LowEntryV6Renderer()
    baseline = renderer.render((with_report(daily_decision, report),))
    report["decision"]["technical_observation"] = metadata
    assert renderer.render((with_report(daily_decision, report),)) == baseline


def test_internal_research_and_observation_metadata_are_not_rendered(daily_decision):
    report = observation_report(daily_decision)
    report["decision"].pop("technical_observation")
    report.pop("research", None)
    renderer = LowEntryV6Renderer()
    baseline = renderer.render((with_report(daily_decision, report),))
    report["research"] = {
        "watch": {"status": "LEADERSHIP_WATCH", "patterns": ["EXISTING_RESEARCH_PATTERN"],
                  "metrics": {"rs20": .05}, "reasons": ["EXISTING_RESEARCH_REASON"]},
        "verification": {"missing_periods": ["2026Q2"], "unresolved_fields": []},
        "blocking_reasons": [], "next_action": "VERIFY", "refresh_status": "PENDING",
    }
    with_metadata = deepcopy(report)
    with_metadata["decision"]["technical_observation"] = observation_report(daily_decision)["decision"]["technical_observation"]
    rendered = renderer.render((with_report(daily_decision, with_metadata),))

    assert rendered == baseline
    assert not section(rendered, "leadership-watch")
    assert "強勢候選優先驗證（非進場訊號）" not in rendered
    assert "EXISTING_RESEARCH_PATTERN" not in rendered and "EXISTING_RESEARCH_REASON" not in rendered
    assert "OBSERVATION_ONLY_BLOCKER" not in rendered
