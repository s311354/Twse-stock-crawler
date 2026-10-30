"""Presentation-only synthetic checks; no model readiness or real email claim."""
from copy import deepcopy
from dataclasses import replace
from email import message_from_string
from email.header import decode_header, make_header
import json
from unittest.mock import Mock

import pytest

from infrastructure.report.low_entry_v7_renderer import (
    LowEntryV7Renderer, architecture_comparison, render_reports, report_mode, rendered_model_label,
)
from tests.strategies.test_daily_v7_rule_baseline import build
from tests.strategies.test_daily_v7_pipeline import build as learned_build, prepared_synthetic_market


def test_baseline_label_once_and_not_applicable_only_describes_predictions():
    _, decisions = build()
    before = decisions[0].value("LES v7 Report")
    html = LowEntryV7Renderer().render(decisions * 2)
    assert html.count('data-report-mode="RULE_BASELINE"') == 1
    assert "使用 v6 型態規則與固定配分" in html
    assert '<strong>V7 模型</strong>' in html
    assert "固定配分 · 未使用機率預測模型" in html
    assert rendered_model_label(html) == "V7 模型"
    assert "預測／校準：不適用" not in html and "不適用：v6" not in html
    assert "LES 技術分數" in html and "技術分數不等於進場許可" in html
    assert "v7 模型路徑" not in html
    assert "white-space:normal" in html and "min-width" not in html
    assert '<td data-label="LES 技術分數"><div>' in html
    assert html.count('data-section="model-architecture-comparison"') == 1
    assert '<th>項目</th><th>V7 碎形模型（研究 bundle）</th><th>V7 模型（一般 --profile elec）</th>' in html
    assert "五預測頭與校準" in html and "固定六因子配分" in html
    assert "UNKNOWN、重大財務風險、交易限制與失效停損仍阻擋" in html
    assert decisions[0].value("LES v7 Report") == before


def test_loaded_model_uses_original_path_label_without_claiming_valid_predictions():
    _, decisions = learned_build(prepared_synthetic_market())
    report = json.loads(decisions[0].value("LES v7 Report"))
    before = deepcopy(report)
    html = LowEntryV7Renderer().render(decisions)
    assert 'data-report-mode="MODEL"' in html
    assert rendered_model_label(html) == "V7 已載入模型"
    assert 'data-model-label="V7 碎形模型"' not in html
    assert "已載入模型，保留原生 v7 預測與計分路徑" in html
    assert "v6 規則式基線" not in html and "預測／校準：不適用" not in html
    assert report == before
    assert "原 bundle 進場政策 · BUY ≥80／BUY_SMALL ≥72" in html
    assert "不自動取代正式每日流程" in html
    assert "底部≤3交易日" in html
    assert "成交後近底反彈" in html
    assert "不是無條件勝率" in html
    assert "／進場 " not in html
    # Mode is model presence, not a fabricated VALID/calibration verdict.
    report["envelope"]["localization"]["status"] = "NOT_READY"
    assert report_mode(report) == "MODEL"


def test_diagnostic_and_missing_model_are_not_mislabelled_as_loaded_models():
    report = dict(envelope=dict(model=dict(bundle_sha256=None)), diagnostics={})
    assert report_mode(report) == "NO_MODEL"
    report["diagnostics"]["run_scope"] = None
    assert report_mode(report) == "NO_MODEL"
    report["diagnostics"]["run_scope"] = dict(diagnostics_only=True)
    assert report_mode(report) == "DIAGNOSTIC"
    report["envelope"]["model"]["bundle_sha256"] = "synthetic_only"
    assert report_mode(report) == "DIAGNOSTIC"


def test_mixed_report_labels_each_row_and_empty_report_has_no_false_mode():
    _, baseline = build()
    _, learned = learned_build(prepared_synthetic_market())
    html = LowEntryV7Renderer().render(baseline + learned)
    assert "混合模式" in html
    assert html.count('data-report-mode="RULE_BASELINE"') == 1
    assert html.count('data-report-mode="MODEL"') == 1
    empty = LowEntryV7Renderer().render([])
    assert "尚無可顯示的研究報告" in empty
    assert "data-report-mode=" not in empty
    assert 'data-section="model-architecture-comparison"' not in empty


def test_fr2_architecture_lists_three_bound_versions_without_altering_forecasts():
    """Pure presentation input, not evidence of a trained/approved model."""
    components = {key: dict(version=version, component_id=key + "_<unsafe>" + "0" * 64)
        for key, version in (("prediction", "FROZEN_FIVE_HEAD_V1"),
                             ("entry", "FR_ENTRY_R2"), ("execution", "FR_EXEC_R1"))}
    before = deepcopy(components)
    html = architecture_comparison(modes={"MODEL"}, models={"f" * 64}, policies=set(),
                                   components=[components, components])
    assert components == before
    assert "本次採用 · FR2" in html and "FR_ENTRY_R2 · BUY ≥75／BUY_SMALL ≥65" in html
    assert "已核實估值警訊扣5分，僅小倉" in html
    assert html.count("預測 FROZEN_FIVE_HEAD_V1") == 1
    assert html.count("進場 FR_ENTRY_R2") == 1 and html.count("執行 FR_EXEC_R1") == 1
    assert "&lt;unsafe&gt;" in html and "<unsafe>" not in html
    assert "部位上限不回頭改 LES Score" in html
    assert 'data-label="V7 已載入模型（研究 bundle）"><div>' in html
    assert 'data-label="V7 模型（一般 --profile elec）"><div>' in html
    assert "已訓練不代表已證明增益" in html


def test_architecture_uses_actual_daily_policy_thresholds_not_research_thresholds():
    html = architecture_comparison(modes={"RULE_BASELINE"}, models=set(),
        policies={("V7_RULE_BASED_BASELINE_R18", 70., 60.)}, components=[])
    assert "V7_RULE_BASED_BASELINE_R18 · BUY ≥70／BUY_SMALL ≥60" in html
    assert "FR_ENTRY_R2" not in html
    assert 'data-section="research-component-identities"' not in html


def test_public_native_report_renderer_preserves_reports_and_does_not_fabricate_erem_or_accounts():
    _, decisions = learned_build(prepared_synthetic_market())
    report = json.loads(decisions[0].value("LES v7 Report"))
    before = deepcopy(report)
    html = render_reports([report])
    assert report == before
    assert html.startswith('<!doctype html><html lang="zh-Hant">')
    assert '<meta name="viewport" content="width=device-width,initial-scale=1">' in html
    assert "@media only screen and (max-width: 720px)" in html
    assert 'data-report-mode="MODEL"' in html
    assert '<td data-label="EREM v3"><div>N/A<br>N/A</div></td>' in html
    assert "未建立帳戶或成交，不寄信、不下單" in html
    close = report["diagnostics"]["features"]["close"]
    assert f"現價 NTD {close * 1000:,.2f}" in html


def test_public_native_report_renderer_rejects_tampering_instead_of_displaying_it():
    import pytest
    _, decisions = learned_build(prepared_synthetic_market())
    report = json.loads(decisions[0].value("LES v7 Report"))
    strategy = report["envelope"]["strategy"]
    strategy["strategy_decision"] = "NO_BUY" if strategy["strategy_decision"] == "BUY" else "BUY"
    with pytest.raises(ValueError):
        render_reports([report])


def test_fr2_native_admission_is_not_labelled_original_bundle_decision():
    from tests.test_v7_fr2_policy import native, apply_fr2
    original = native()
    report = apply_fr2(original)
    before = deepcopy(report)
    html = render_reports([report])
    assert '<th>FR2 研究決策</th>' in html
    assert '<td data-label="FR2 研究決策"><div>' in html
    assert "v7 原始決策" not in html
    assert report == before
    original_html = render_reports([original])
    assert '<th>v7 原始決策</th>' in original_html
    mixed_html = render_reports([original, report])
    assert '<th>LES 決策（配置前）</th>' in mixed_html
    assert '<td data-label="LES 決策（配置前）"><div>' in mixed_html


def fractal_decisions():
    """Synthetic active-fractal declaration, NOT a trained/approved predictor."""
    from tests.strategies.test_daily_v7_pipeline import synthetic_market
    from tests.strategies.test_low_entry_score_v7 import bundle, seal
    model = bundle()
    model["payload"].update(fractal_mode="V7_FRACTAL_RESEARCH", feature_names=["dfa512"],
        feature_bounds=[[0., 2.]], fractal_null_validation_report_id="SYNTHETIC_ONLY")
    market = replace(synthetic_market(), les_v7_bundle_json=json.dumps(seal(model)))
    return learned_build(prepared_synthetic_market(market))[1]


def test_fractal_label_comes_from_actual_bundle_features_even_when_forecast_not_ready():
    decisions = fractal_decisions()
    before = decisions[0].value("LES v7 Report")
    report = json.loads(before)
    assert report["diagnostics"]["model_input_profile"]["feature_names"] == ["dfa512"]
    assert report_mode(report) == "FRACTAL_MODEL"
    html = LowEntryV7Renderer().render(decisions)
    assert rendered_model_label(html) == "V7 碎形模型"
    assert 'data-report-mode="FRACTAL_MODEL"' in html
    assert "五預測頭、校準與凍結權重" in html
    assert decisions[0].value("LES v7 Report") == before


def test_legacy_report_does_not_infer_fractal_model_from_computed_dfa_or_bundle_hash():
    report = json.loads(fractal_decisions()[0].value("LES v7 Report"))
    report["diagnostics"].pop("model_input_profile")
    # All paths calculate these features; that does not prove predictor use.
    report["diagnostics"]["features"]["dfa512"] = 1.
    assert report_mode(report) == "MODEL"
    assert rendered_model_label(render_reports([report])) == "V7 已載入模型"


def test_mixed_fractal_and_daily_rows_have_explicit_labels_without_mutation():
    _, baseline = build()
    learned = fractal_decisions()
    before = [item.value("LES v7 Report") for item in baseline + learned]
    html = LowEntryV7Renderer().render(baseline + learned)
    assert rendered_model_label(html) == "V7 混合模式"
    assert 'data-report-mode="RULE_BASELINE"' in html
    assert 'data-report-mode="FRACTAL_MODEL"' in html
    assert [item.value("LES v7 Report") for item in baseline + learned] == before


@pytest.mark.parametrize("fractal,label", [(False, "V7 模型"), (True, "V7 碎形模型")])
def test_resender_subject_and_body_share_model_label_without_sending(fractal, label):
    from infrastructure.report.html_renderer import RendererFactory
    from textmewhenitsdone import TextMeWhenItsDone
    decisions = fractal_decisions() if fractal else build()[1]
    markup = RendererFactory.get_renderer("daily_strategy_email").render(decisions)
    assert f"今日決策卡｜{label}</h2>" in markup
    transport = TextMeWhenItsDone.__new__(TextMeWhenItsDone)
    transport.server = Mock()
    transport.textstockprofittableme(subject="synthetic", email="sender@example.invalid",
        password="unused", iso_scheduled_times=["2024-11-15", "2026-10-15"], transactiondays=700,
        stocktype="ELEC", stockprofittable="", entryanalysis=markup,
        receiver="recipient@example.invalid", les_model_version="7.0.0")
    transport.server.sendmail.assert_called_once()
    message = message_from_string(transport.server.sendmail.call_args.kwargs["msg"])
    assert len(message.get_all("Subject")) == 1
    subject = str(make_header(decode_header(message["Subject"])))
    assert subject.startswith(f"[{label}]") and "LES v7 / EREM v3" in subject
    assert rendered_model_label(message.get_payload()[0].get_payload(decode=True).decode("utf-8")) == label


def test_subject_label_accepts_only_unambiguous_renderer_metadata():
    assert rendered_model_label('<div>V7 碎形模型</div>') is None
    assert rendered_model_label('<div data-model-label="invented"></div>') is None
    assert rendered_model_label('<div data-model-label="V7 模型"></div><div data-model-label="V7 碎形模型"></div>') is None
