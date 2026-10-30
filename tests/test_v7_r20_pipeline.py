"""Synthetic full native R20/retained R19 comparison, never performance evidence."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.policy_r20 import BASELINE_VERSION as R20, AVAILABLE_AT
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy, current_source_baseline_version
from tests.test_v7_r19_pipeline import current_market, native
from tests.test_v7_r18_pipeline import neutral_market, core_market
from tests.test_v7_r16_pipeline import market, pullback_bars, discovery_bars


def value_for(kind):
    source = {"neutral": neutral_market, "core": core_market,
        "pullback": lambda: market(data=pullback_bars()),
        "discovery": lambda: market(data=discovery_bars())}[kind]()
    with patch("tests.test_v7_r19_pipeline.AT", "2026-10-28T02:20:00Z"):
        return current_market(source)


@pytest.mark.parametrize("kind", ["neutral", "core", "pullback", "discovery"])
def test_native_recomputation_keeps_all_r19_safety_economics_and_account(kind):
    value = value_for(kind)
    old, new = native(value), native(value, R20)
    validate_report(new)
    assert old["diagnostics"]["quality"] == new["diagnostics"]["quality"]
    a, b = old["diagnostics"], new["diagnostics"]
    for key in ("entry", "stop", "target", "atr", "band", "cost_up", "cost_stop", "reference_shares"):
        assert a["reference_plan"][key] == b["reference_plan"][key]
    assert {g["gate_id"]: g["state"] for g in a["gate_receipts"]} == {
        g["gate_id"]: g["state"] for g in b["gate_receipts"]}
    assert a["rule_baseline"]["technical_vetoes"] == b["rule_baseline"]["technical_vetoes"]
    assert old["pae_authorization"]["account"] == new["pae_authorization"]["account"]
    assert old["pae_authorization"]["account_sha256"] == new["pae_authorization"]["account_sha256"]
    assert not new["pae_authorization"]["live_order_authorized"]
    assert new["envelope"]["score"]["raw_score"] >= old["envelope"]["score"]["raw_score"]
    assert b["r19_route_evaluation"]["position_limits_affect_score"] is False


@pytest.mark.parametrize("missing", ["calendar_evidence", "security", "restriction"])
def test_points_do_not_replace_missing_safety_sources(missing):
    value = value_for("neutral")
    if missing == "restriction":
        value = replace(value, trading_restrictions={})
    else:
        sources = deepcopy(value.native_source_selections)
        sources["2330"][missing] = None
        value = replace(value, native_source_selections=sources)
    report = native(value, R20)
    validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert report["pae_authorization"]["authorized"] is False


def test_rehashed_score_is_rejected_and_old_policy_identity_unchanged():
    assert identity(baseline_policy(version="V7_RULE_BASED_BASELINE_R19")) == "4723ab8f53a4c9ac2db05e123ba970db5242a759ff25107ce0fc173ccfa70637"
    old, new = baseline_policy(version="V7_RULE_BASED_BASELINE_R19"), baseline_policy(version=R20)
    for key in ("quality_policy", "valuation_policy", "valuation_input_policy", "native_risk_limits",
                "tactical_risk_limits", "entry_admission_policy", "model_spec"):
        assert old[key] == new[key]
    report = native(value_for("neutral"), R20)
    report["envelope"]["score"]["raw_score"] += 1.
    with pytest.raises(ValueError):
        validate_report(report)


def test_no_retroactive_r20_on_october_27_and_unactivated_candidate_is_not_default():
    assert current_source_baseline_version("2026-10-27T13:30:00+08:00") == "V7_RULE_BASED_BASELINE_R19"
    from domain.les_v7 import policy_r20
    from domain.les_v7 import policy_r21
    with patch.object(policy_r20, "DAILY_ACTIVATED_AT", None), patch.object(policy_r21, "DAILY_ACTIVATED_AT", None), \
            patch("domain.les_v7.policy_r22.DAILY_ACTIVATED_AT", None), \
            patch("domain.les_v7.policy_r23.DAILY_ACTIVATED_AT", None), \
            patch("domain.les_v7.policy_r24.DAILY_ACTIVATED_AT", None), \
            patch("domain.les_v7.policy_r25.DAILY_ACTIVATED_AT", None):
        assert current_source_baseline_version("2030-01-01T00:00:00Z") == "V7_RULE_BASED_BASELINE_R19"


def test_account_cap_cannot_feedback_into_les_score_or_authorize_a_cashless_order():
    from application.low_entry_v7_daily import finalize_daily_report
    report = native(value_for("neutral"), R20)
    original_score = deepcopy(report["envelope"]["score"])
    account = dict(report["pae_authorization"]["account"], available_cash=0.)
    result = finalize_daily_report(deepcopy(report), account=account,
        daily_input_errors=report["diagnostics"].get("daily_input_errors", []))
    assert result["envelope"]["score"] == original_score
    assert result["pae_authorization"]["indicative_shares"] == 0
    assert not result["pae_authorization"]["authorized"]
    assert not result["pae_authorization"]["live_order_authorized"]


def test_renderer_names_r20_and_actual_thresholds_without_prediction_or_order_claim():
    from infrastructure.report.low_entry_v7_renderer import render_reports
    report = native(value_for("neutral"), R20)
    original = deepcopy(report)
    html = render_reports([report])
    assert report == original
    assert 'data-entry-policy="R20"' in html
    assert "BUY 70／BUY_SMALL 60" in html
    assert "早期止跌部分計分／停損連續配分" in html
    assert "沿用 R19 估值與風控" in html
    assert "未使用機率預測模型" in html
    assert "72／BUY_SMALL 62" not in html
