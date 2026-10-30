"""Synthetic native R24 checks; no hardcoded issuer exception or forced buy."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pandas as pd
import pytest

from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.contracts import identity
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.les_v7.financial_audit_r7 import audit_source_r7
from domain.les_v7.policy_r24 import BASELINE_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy, baseline_decision_policy
from tests.test_v7_r19_pipeline import current_market, native
from tests.test_v7_r23_pipeline import fixture


def test_registered_r24_changes_source_interpretation_not_score_or_plan_parameters():
    old = baseline_policy(version="V7_RULE_BASED_BASELINE_R23")
    new = baseline_policy(version=BASELINE_VERSION)
    assert identity(old) == "571cbf6eec65df7ed7849df4583fda0911609fbc480ef0b76fa12e1558987142"
    assert new["predecessor_policy_id"] == identity(old)
    for key in ("valuation_policy", "valuation_input_policy", "model_spec", "score_policy",
                "native_risk_limits", "tactical_risk_limits", "entry_admission_policy", "reference_policy",
                "unified_technical_policy"):
        assert old[key] == new[key]
    assert new["quality_policy"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R7"
    assert baseline_decision_policy(new) == "RULE_BASED_SOURCE_BOUND_AUDIT_R24"


def test_native_r24_with_missing_audit_never_grants_entry_or_zeroes_technical_score():
    with patch("tests.test_v7_r19_pipeline.AT", "2026-10-28T13:00:00Z"):
        value = current_market(fixture("core"))
    before = deepcopy(value.fundamental_snapshots)
    report = native(value, BASELINE_VERSION)
    validate_report(report)
    assert report["diagnostics"]["quality"]["status"] == "DATA_INSUFFICIENT"
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert report["envelope"]["score"]["final_score"] > 0
    assert not report["pae_authorization"]["authorized"]
    assert value.fundamental_snapshots == before


def test_renderer_labels_r24_without_legacy_72_62_or_probability_claim():
    from infrastructure.report.low_entry_v7_renderer import render_reports
    with patch("tests.test_v7_r19_pipeline.AT", "2026-10-28T13:00:00Z"):
        value = current_market(fixture())
    html = render_reports([native(value, BASELINE_VERSION)])
    assert 'data-entry-policy="R24"' in html
    assert "BUY 70／BUY_SMALL 60" in html and "Quality R7" in html
    assert "未使用機率預測模型" in html
    assert "BUY 72／BUY_SMALL 62" not in html


@pytest.mark.parametrize("kind", ["core", "neutral", "discovery"])
def test_fully_bound_r7_daily_input_retains_qualified_native_entry_and_score(kind):
    with patch("tests.test_v7_r19_pipeline.AT", "2026-10-28T13:00:00Z"):
        value = current_market(fixture(kind))
    snapshots = deepcopy(value.fundamental_snapshots)
    row = snapshots["2330"]["les_v6_mops_filings"][-1]
    period = pd.Period(row["period_end"], freq="Q")
    proof = row["audit_evidence_r5"]
    facts = tuple(FiledFact(**dict(f, dimensions=tuple(map(tuple, f.get("dimensions", ())))))
                  for f in proof["contexts"])
    filing = FiledFinancials(period.year, period.quarter, row["available_at"], row["revision_number"], facts,
        content_sha256=row["content_sha256"], source_url=row["source_url"],
        statement_scope=row["scope"], period_end=row["period_end"], audit_status=proof.get("audit_status"))
    snapshots["2330"]["les_v7_context"]["financial_audits_r7"] = {
        row["content_sha256"]: audit_source_r7(filing, symbol="2330", original_verified=True)}
    value = replace(value, fundamental_snapshots=snapshots)
    old, new = native(value, "V7_RULE_BASED_BASELINE_R23"), native(value, BASELINE_VERSION)
    validate_report(new)
    assert old["envelope"]["strategy"]["strategy_decision"] == ("BUY" if kind == "core" else "BUY_SMALL")
    assert old["envelope"]["strategy"]["strategy_decision"] == new["envelope"]["strategy"]["strategy_decision"]
    assert old["envelope"]["score"] == new["envelope"]["score"]
    assert new["diagnostics"]["quality"]["status"] == "PASS"
    assert new["pae_authorization"]["account_sha256"] == old["pae_authorization"]["account_sha256"]
    assert not new["pae_authorization"]["live_order_authorized"]
