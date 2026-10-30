"""Synthetic native R19 signals cross PAE without expanding their share quota."""
import json
from unittest.mock import patch

import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioAllocationInputError, PortfolioRuntimeContext
from domain.les_v7.contracts import timestamp
from domain.les_v7.entry_policy_r9 import R9PortfolioSecurityV2, pae_admission
from domain.les_v7.policy_r18 import ADMISSION_VERSION
from domain.les_v7.policy_r19 import BASELINE_VERSION
from domain.strategies.portfolio_allocation_v2_contracts import PortfolioAllocationRequestV2, PortfolioAllocationV2Config
from domain.strategies.portfolio_allocation_v2_evaluator import PortfolioSecurityEvaluatorV2
from domain.strategies.portfolio_allocation_v2_weight_allocator import PortfolioWeightAllocatorV2
from infrastructure.report.low_entry_v7_renderer import LowEntryV7Renderer
from tests.test_v7_r18_pipeline import core_market, neutral_market
from tests.test_v7_r19_pipeline import current_market


@pytest.fixture(scope="module", params=[(core_market, "BUY"), (neutral_market, "BUY_SMALL")],
                ids=["core", "neutral"])
def native_decision(request):
    factory, signal = request.param
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
        decision = AnalysisDatasetBuilder().build_decisions(current_market(factory()), [[1, 1, 1, 1, .01]])[0]
    return decision, signal


def test_native_r19_admission_and_worst_price_account_quota_reach_existing_pae(native_decision):
    decision, signal = native_decision
    encoded = decision.value("LES v7 Report")
    report = json.loads(encoded)
    assert report["envelope"]["strategy"]["strategy_decision"] == signal
    authorization = report["pae_authorization"]
    assert authorization["authorized"] and not authorization["live_order_authorized"]
    assert authorization["policy_version"] == "LES_V7_PAE_SIGNAL_AUTHORIZATION_R8"
    account = authorization["account"]
    context = PortfolioRuntimeContext(account_equity=account["value"],
        current_cash=account["available_cash"], positions=(), default_sector="ELEC")
    pipeline = PortfolioAllocationV2Pipeline()
    pipeline._validate_v7_account(decision, account["value"], account["available_cash"], ())
    security = pipeline._security(decision, context, position=None)
    assert isinstance(security, R9PortfolioSecurityV2)
    admission = pae_admission(security)
    assert admission["policy_version"] == ADMISSION_VERSION
    assert admission["assessment_id"] == report["diagnostics"]["entry_admission"]["assessment_id"]
    assert PortfolioSecurityEvaluatorV2(PortfolioAllocationV2Config()).direct_selection_executable(security)
    assert security.les_position_cap == decision.value("Position Fraction")
    assert security.les_position_cap <= authorization["maximum_total_weight"]
    request = PortfolioAllocationRequestV2("SYNTHETIC_R19_NO_LEDGER",
        timestamp(account["as_of"]), account["value"], account["available_cash"], candidates=(security,))
    weight, _ = PortfolioWeightAllocatorV2(PortfolioAllocationV2Config()).target_weight(request, security, .9, None)
    assert 0 <= weight <= security.les_position_cap
    with pytest.raises(PortfolioAllocationInputError, match="ACCOUNT_BINDING_MISMATCH"):
        pipeline._validate_v7_account(decision, account["value"], account["available_cash"] / 2, ())
    assert decision.value("LES v7 Report") == encoded


def test_r19_report_names_risk_bounded_band_and_keeps_model_mode_distinct(native_decision):
    decision, _ = native_decision
    markup = LowEntryV7Renderer().render([decision])
    assert 'data-entry-policy="R19"' in markup
    assert 'data-report-mode="RULE_BASELINE"' in markup
    assert "進場價帶依波動擴展" in markup
    assert "成本後風險報酬限制" in markup
