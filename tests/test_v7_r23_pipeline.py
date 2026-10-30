"""Native source-policy isolation; synthetic reports are not official repairs."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy
from tests.test_v7_r19_pipeline import native,current_market
from tests.test_v7_r22_pipeline import fixture as previous_fixture

R23="V7_RULE_BASED_BASELINE_R23"


def fixture(kind="neutral"):
    with patch("tests.test_v7_r19_pipeline.AT","2026-10-28T07:15:00Z"):
        return current_market(previous_fixture(kind))


@pytest.mark.parametrize("kind",["neutral","core","pullback","discovery"])
def test_no_conflict_has_exact_same_scores_gates_plans_and_account(kind):
    value=fixture(kind)
    old,new=native(value,"V7_RULE_BASED_BASELINE_R22"),native(value,R23)
    validate_report(new)
    assert old["envelope"]["score"]==new["envelope"]["score"]
    decision=new["envelope"]["strategy"]["strategy_decision"]
    assert old["envelope"]["strategy"]["strategy_decision"]==decision
    if kind=="core":assert decision=="BUY"
    if kind in {"neutral","discovery"}:assert decision=="BUY_SMALL"
    a,b=old["diagnostics"],new["diagnostics"]
    assert a["quality"]==b["quality"]
    for key in ("entry","stop","target","atr","band","cost_up","cost_stop","setup_variant","structure_id"):
        assert (a["reference_plan"] or {}).get(key)==(b["reference_plan"] or {}).get(key)
    assert {g["gate_id"]:g["state"] for g in a["gate_receipts"]}=={g["gate_id"]:g["state"] for g in b["gate_receipts"]}
    assert old["pae_authorization"]["account_sha256"]==new["pae_authorization"]["account_sha256"]
    assert not new["pae_authorization"]["live_order_authorized"]


def test_policy_preserves_r22_content_and_all_scoring_risk_parameters():
    old,new=baseline_policy(version="V7_RULE_BASED_BASELINE_R22"),baseline_policy(version=R23)
    assert identity(old)=="8f1cc12ad81530a5a1c22819f84f5e8ba6ff9a8381cfc88b10bc07183cb6646a"
    for key in ("quality_policy","valuation_policy","valuation_input_policy","model_spec","score_policy",
        "native_risk_limits","tactical_risk_limits","entry_admission_policy","reference_policy"):
        assert old[key]==new[key]
    assert new["predecessor_policy_id"]==identity(old)


def test_mixed_policy_header_cannot_fall_back_to_old_72_62():
    from infrastructure.report.low_entry_v7_renderer import render_reports
    v=fixture();reports=[native(v,"V7_RULE_BASED_BASELINE_R22"),native(v,R23)];before=deepcopy(reports)
    html=render_reports(reports)
    assert reports==before
    assert 'data-entry-policy="MIXED"' in html
    assert 'data-rule-policy="V7_RULE_BASED_BASELINE_R22"' in html
    assert 'data-rule-policy="V7_RULE_BASED_BASELINE_R23"' in html
    assert "R22 · BUY 70／BUY_SMALL 60" in html and "R23 · BUY 70／BUY_SMALL 60" in html
    assert "BUY 72／BUY_SMALL 62" not in html
