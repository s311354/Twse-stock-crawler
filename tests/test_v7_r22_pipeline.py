"""Native R22 source/plan/score reconstruction, synthetic data only."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy
from tests.test_v7_r19_pipeline import native, current_market
from tests.test_v7_r21_pipeline import fixture as previous_fixture

R22 = "V7_RULE_BASED_BASELINE_R22"


def fixture(kind="neutral"):
    with patch("tests.test_v7_r19_pipeline.AT", "2026-10-28T06:35:00Z"):
        return current_market(previous_fixture(kind))


@pytest.mark.parametrize("kind", ["neutral", "core", "pullback", "discovery"])
def test_full_native_recomputes_same_safety_and_risk_economics(kind):
    value = fixture(kind)
    old, new = native(value, "V7_RULE_BASED_BASELINE_R21"), native(value, R22)
    validate_report(new)
    a, b = old["diagnostics"], new["diagnostics"]
    assert old["pae_authorization"]["account_sha256"] == new["pae_authorization"]["account_sha256"]
    assert a["quality"] == b["quality"]
    assert {g["gate_id"]:g["state"] for g in a["gate_receipts"]} == {g["gate_id"]:g["state"] for g in b["gate_receipts"]}
    for key in ("entry", "stop", "target", "atr", "band", "cost_up", "cost_stop"):
        assert (a["reference_plan"] or {}).get(key) == (b["reference_plan"] or {}).get(key)
    assert b["rule_baseline"]["policy"]["source_selection_policy"] == "V7_STRUCTURE_SOURCE_SEMANTICS_R22"
    assert b["r19_route_evaluation"]["position_limits_affect_score"] is False
    assert not new["pae_authorization"]["live_order_authorized"]


def test_retained_policy_ids_and_non_technical_parameters_unchanged():
    old, new = baseline_policy(version="V7_RULE_BASED_BASELINE_R21"), baseline_policy(version=R22)
    assert identity(old) == "bb6bc7d94bd68edc0c05f2039f57c21fc6de5407cb1f30abcbc854bfb3f179a4"
    for key in ("quality_policy", "valuation_policy", "valuation_input_policy", "model_spec",
                "native_risk_limits", "tactical_risk_limits", "entry_admission_policy"):
        assert old[key] == new[key]
    assert new["unified_technical_policy"]["buy_minimum"] == 70
    assert new["unified_technical_policy"]["small_minimum"] == 60


def test_modified_score_or_selection_cannot_bypass_reconstruction():
    report = native(fixture(), R22)
    damaged = deepcopy(report)
    damaged["diagnostics"]["rule_baseline"]["policy"]["source_selection_policy"] = "PASS_ALL"
    with pytest.raises(ValueError): validate_report(damaged)
    damaged = deepcopy(report)
    damaged["diagnostics"]["rule_baseline"]["score_evidence"]["conditions"][0]["earned_points"] += .5
    with pytest.raises(ValueError): validate_report(damaged)


def test_renderer_reports_current_thresholds_and_no_probability_claim():
    from infrastructure.report.low_entry_v7_renderer import render_reports
    html = render_reports([native(fixture(), R22)])
    assert 'data-entry-policy="R22"' in html
    assert 'BUY 70／BUY_SMALL 60' in html
    assert '必要來源與交易風控不變' in html
    assert '未使用機率預測模型' in html
