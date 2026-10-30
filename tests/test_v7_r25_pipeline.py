"""Native R25 checks use clearly synthetic data, not missing official evidence."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pandas as pd
import pytest

from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.les_v7.contracts import identity
from domain.les_v7.financial_audit_r7 import audit_source_r7
from domain.les_v7.policy_r25 import AVAILABLE_AT, BASELINE_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy, require_baseline_policy
from tests.test_v7_r19_pipeline import current_market, native
from tests.test_v7_r23_pipeline import fixture

AT = "2026-10-29T02:00:00Z"


def bound_market(kind="neutral"):
    with patch("tests.test_v7_r19_pipeline.AT", AT):
        value = current_market(fixture(kind))
    snapshots = deepcopy(value.fundamental_snapshots)
    row = snapshots["2330"]["les_v6_mops_filings"][-1]
    period, proof = pd.Period(row["period_end"], freq="Q"), row["audit_evidence_r5"]
    facts = tuple(FiledFact(**dict(f, dimensions=tuple(map(tuple, f.get("dimensions", ())))))
                  for f in proof["contexts"])
    filing = FiledFinancials(period.year, period.quarter, row["available_at"], row["revision_number"], facts,
        content_sha256=row["content_sha256"], source_url=row["source_url"],
        statement_scope=row["scope"], period_end=row["period_end"], audit_status=proof.get("audit_status"))
    snapshots["2330"]["les_v7_context"]["financial_audits_r7"] = {
        row["content_sha256"]: audit_source_r7(filing, symbol="2330", original_verified=True)}
    return replace(value, fundamental_snapshots=snapshots)


def test_r24_identity_preserved_and_only_explicit_score_threshold_policy_changes():
    old, new = (baseline_policy(version=v) for v in ("V7_RULE_BASED_BASELINE_R24", BASELINE_VERSION))
    assert identity(old) == "659a496973b6de40543e92bedec5d3dc1b122ea69998e43e7aa1998609ab32bc"
    assert new["predecessor_policy_id"] == identity(old)
    for key in ("source_policy", "source_selection_policy", "quality_policy", "valuation_policy",
                "valuation_input_policy", "entry_admission_policy", "reference_policy",
                "native_risk_limits", "tactical_risk_limits"):
        assert new[key] == old[key]
    assert new["model_spec"]["factor_budgets"] == old["model_spec"]["factor_budgets"]
    assert new["model_spec"]["curves"]["amount_ratio"] == [.5, 1.5]
    assert new["unified_technical_policy"]["buy_minimum"] == 70
    assert new["unified_technical_policy"]["small_minimum"] == 55
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(new, market="TWSE", cutoff="2026-10-28T13:00:00Z")


@pytest.mark.parametrize("kind", ["neutral", "core", "pullback", "discovery"])
def test_native_one_plan_same_quality_gates_costs_and_account(kind):
    value = bound_market(kind)
    before = deepcopy(value.fundamental_snapshots)
    old, new = native(value, "V7_RULE_BASED_BASELINE_R24"), native(value, BASELINE_VERSION)
    validate_report(new)
    a, b = old["diagnostics"], new["diagnostics"]
    assert a["quality"] == b["quality"]
    assert {g["gate_id"]: g["state"] for g in a["gate_receipts"]} == {g["gate_id"]: g["state"] for g in b["gate_receipts"]}
    for key in ("entry", "stop", "target", "band", "atr", "cost_up", "cost_stop"):
        assert (a["reference_plan"] or {}).get(key) == (b["reference_plan"] or {}).get(key)
    assert new["envelope"]["score"]["raw_score"] >= old["envelope"]["score"]["raw_score"]
    assert new["pae_authorization"]["account_sha256"] == old["pae_authorization"]["account_sha256"]
    assert not new["pae_authorization"]["live_order_authorized"]
    from application.low_entry_v7_daily import project_report
    projected = project_report(new, include_report=False)
    assert projected["LES v7 Required Score"] == (70 if new["envelope"]["strategy"]["strategy_decision"] == "BUY" else 55)
    assert b["r19_route_evaluation"]["position_limits_affect_score"] is False
    assert value.fundamental_snapshots == before


@pytest.mark.parametrize("missing", ["restriction", "identity", "calendar", "audit"])
def test_lower_score_threshold_cannot_replace_missing_necessary_evidence(missing):
    value = bound_market()
    if missing == "restriction": value = replace(value, trading_restrictions={})
    elif missing == "audit":
        snapshots = deepcopy(value.fundamental_snapshots)
        snapshots["2330"]["les_v7_context"]["financial_audits_r7"] = {}
        for row in snapshots["2330"]["les_v6_mops_filings"]:
            row.pop("audit_evidence_r7", None)
        value = replace(value, fundamental_snapshots=snapshots)
    else:
        sources = deepcopy(value.native_source_selections)
        sources["2330"]["security" if missing == "identity" else "calendar_evidence"] = None
        value = replace(value, native_source_selections=sources)
    report = native(value, BASELINE_VERSION)
    validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not report["pae_authorization"]["authorized"]


def test_rehashed_score_or_policy_cannot_override_native_recalculation():
    report = native(bound_market(), BASELINE_VERSION)
    damaged = deepcopy(report)
    damaged["envelope"]["score"]["final_score"] += 1
    with pytest.raises(ValueError): validate_report(damaged)
    damaged = deepcopy(report)
    damaged["diagnostics"]["rule_baseline"]["policy"]["unified_technical_policy"]["small_minimum"] = 0
    with pytest.raises(ValueError): validate_report(damaged)


def test_renderer_discloses_native_thresholds_and_conditional_execution():
    from infrastructure.report.low_entry_v7_renderer import render_reports
    html = render_reports([native(bound_market(), BASELINE_VERSION)])
    assert 'data-entry-policy="R25"' in html
    assert 'BUY 70／BUY_SMALL 55' in html and '條件未達不得成交' in html
    assert 'BUY 72／BUY_SMALL 62' not in html


@pytest.mark.parametrize("decision,expected", [("BUY_SMALL", "DATA_INSUFFICIENT"),
    ("WATCH_ONLY", "DATA_INSUFFICIENT"), ("NO_BUY", "NO_BUY")])
def test_public_daily_input_override_keeps_raw_score_but_never_manufactures_a_final(decision, expected):
    from application.low_entry_v7_daily import daily_input_outcome
    score = dict(raw_score=58., final_score=58.)
    outcome, actual = daily_input_outcome(score, decision, errors=["SYNTHETIC_INPUT_ERROR"])
    assert outcome == dict(raw_score=58., final_score=None)
    assert actual == expected and score["final_score"] == 58.
