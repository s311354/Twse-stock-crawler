"""Synthetic R21 native source/score/authorization regression, no return claim."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.rule_baseline import baseline_policy
from domain.les_v7.report import validate_report
from tests.test_v7_r19_pipeline import current_market, native
from tests.test_v7_r18_pipeline import neutral_market, core_market
from tests.test_v7_r16_pipeline import market, pullback_bars, discovery_bars

R21 = "V7_RULE_BASED_BASELINE_R21"


def fixture(kind="neutral"):
    value = {"neutral": neutral_market, "core": core_market,
        "pullback": lambda: market(data=pullback_bars()),
        "discovery": lambda: market(data=discovery_bars())}[kind]()
    with patch("tests.test_v7_r19_pipeline.AT", "2026-10-28T06:10:00Z"):
        return current_market(value)


@pytest.mark.parametrize("kind", ["neutral", "core", "pullback", "discovery"])
def test_native_source_gates_risk_account_and_formal_setup_stay_bound(kind):
    value = fixture(kind)
    old, new = native(value, "V7_RULE_BASED_BASELINE_R20"), native(value, R21)
    validate_report(new)
    a, b = old["diagnostics"], new["diagnostics"]
    assert old["pae_authorization"]["account_sha256"] == new["pae_authorization"]["account_sha256"]
    assert a["quality"] == b["quality"]
    assert old["envelope"]["event"]["context"] == new["envelope"]["event"]["context"]
    assert {g["gate_id"]:g["state"] for g in a["gate_receipts"]} == {g["gate_id"]:g["state"] for g in b["gate_receipts"]}
    for key in ("entry", "stop", "target", "atr", "band", "cost_up", "cost_stop"):
        assert (a["reference_plan"] or {}).get(key) == (b["reference_plan"] or {}).get(key)
    assert not new["pae_authorization"]["live_order_authorized"]
    assert b["rule_baseline"]["score_evidence"]["scoring_interpretation"]["entry_permission_granted"] is False


@pytest.mark.parametrize("missing", ["calendar_evidence", "security", "restriction"])
def test_recovery_interpretation_never_supplies_missing_safety_sources(missing):
    value = fixture()
    if missing == "restriction":
        value = replace(value, trading_restrictions={})
    else:
        sources = deepcopy(value.native_source_selections);sources["2330"][missing] = None
        value = replace(value, native_source_selections=sources)
    report = native(value, R21)
    validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not report["pae_authorization"]["authorized"]


def test_retained_policy_ids_and_non_technical_economics_unchanged():
    old, new = baseline_policy(version="V7_RULE_BASED_BASELINE_R20"), baseline_policy(version=R21)
    assert identity(old) == "b5b37baac74dfb08f9e3dc6a9f015463286aae95c5eecaed206b54ef7dc7afff"
    for key in ("quality_policy", "valuation_policy", "valuation_input_policy", "model_spec",
                "native_risk_limits", "tactical_risk_limits", "entry_admission_policy"):
        assert old[key] == new[key]


def test_tampered_interpretation_or_score_cannot_pass_native_reconstruction():
    report = native(fixture(), R21)
    for key in ("scoring_context", "entry_permission_granted"):
        damaged = deepcopy(report)
        damaged["diagnostics"]["rule_baseline"]["score_evidence"]["scoring_interpretation"][key] = "TAMPERED"
        with pytest.raises(ValueError):validate_report(damaged)


def test_zero_display_explains_arithmetic_without_mislabeling_as_cap():
    from infrastructure.report.low_entry_v7_renderer import score_text, render_reports
    value = dict(raw_score=4.59, total_penalty=14., adjusted_score=0., final_score=0., effective_cap=None)
    text = score_text(value, show_arithmetic=True)
    assert "扣分後 -9.41" in text and "分數下限 0" in text
    assert "Final 0.00" in text and "封頂" not in text
    html = render_reports([native(fixture(), R21)])
    assert 'data-entry-policy="R21"' in html
    assert "BUY 70／BUY_SMALL 60" in html
    assert "非進場資格" in html
