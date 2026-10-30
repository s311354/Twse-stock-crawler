"""SYNTHETIC ONLY: explicit research EPS/FR2 wiring, never forecast efficacy."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from application.low_entry_v7_daily import (
    _native_source_packet, daily_report, finalize_daily_report, research_valuation_inputs,
)
from application.market_history import MarketHistoryAssembler
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.fr2_policy import restore_original
from domain.les_v7.quality_policy import r3_quality_policy
from domain.les_v7.report import validate_report
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from infrastructure.report.low_entry_v7_renderer import render_reports
from tests.strategies.test_low_entry_score_v7 import bundle, seal
from tests.test_v7_eps_price_daily import packet_context, rule_market_with_earnings
from tests.strategies.test_v7_rule_baseline import legal_bars


def research_market(*, explicit=True, account=None):
    model = bundle()
    model["payload"]["quality_policy"] = r3_quality_policy()
    seal(model)
    market = replace(rule_market_with_earnings(), les_v7_rule_baseline=False,
        les_v7_bundle_json=json.dumps(model), les_v6_portfolio={} if account is None else account)
    if explicit:
        metadata = dict(schema="les-v7-research-model-use/v1", bundle_sha256=model["sha256"],
            symbols=["2330"], research_only=True, approved_for_live=False,
            execution_action="NO_ACTION", order_authority=False)
        market = replace(market, les_v7_research_metadata_json=json.dumps(metadata))
    return market


def research_history(*, explicit=True, account=None):
    return MarketHistoryAssembler().build(research_market(explicit=explicit, account=account), 0)


def test_eps_research_operands_are_separate_from_the_unchanged_original_source_packet():
    context = packet_context(); context.pop("rule_baseline")
    history = legal_bars(); history.attrs["les_v6_analysis_session"] = str(history.Date.iloc[-1])[:10]
    original = dict(SYNTHETIC_ONLY=True)
    packet = _native_source_packet(history, history, context)
    old = canonical(packet)
    inputs = research_valuation_inputs(original, context)
    row = inputs["valuation_filings"][0]
    assert inputs["schema"] == "les-v7-fr2-eps-inputs/v1"
    assert inputs["original_report_id"] == identity(original)
    assert row["source_url"] == context["filings"][0]["source_url"]
    assert row["revision_number"] == context["filings"][0]["revision_number"]
    assert row["earnings_source"] == context["valuation_earnings"][row["content_sha256"]]
    assert row["comparative_ytd_fields"]["revenue"] == 100000.
    assert row["comparative_ytd_field_evidence"]["operating_income"]
    assert "earnings_source" not in packet["valuation_filings"][0]
    assert canonical(_native_source_packet(history, history, context)) == old
    row["earnings_source"]["fields"]["diluted_eps"] = -999.
    assert context["valuation_earnings"][row["content_sha256"]]["fields"]["diluted_eps"] == 12.


def test_eps_research_operands_do_not_include_future_filings_or_impute_missing_companions():
    context = packet_context(); row = deepcopy(context["filings"][0])
    row["available_at"] = "2099-01-01T00:00:00Z"
    context["filings"].append(row)
    context["valuation_earnings"] = {}
    inputs = research_valuation_inputs(dict(SYNTHETIC_ONLY=True), context)
    assert len(inputs["valuation_filings"]) == 1
    assert inputs["valuation_filings"][0]["earnings_source"] is None
    assert inputs["valuation_filings"][0].get("dividends") is None
    assert inputs["valuation_filings"][0].get("diluted_shares") is None


def test_explicit_research_uses_fr2_eps_without_changing_the_frozen_prediction_or_plan():
    history = research_history()
    before = deepcopy(history.attrs)
    report = daily_report(history)
    validate_report(report)
    original = restore_original(report)
    assert report["diagnostics"]["fr2"]["schema"] == "les-v7-fr2-eps-overlay/v1"
    assert report["diagnostics"]["research_components"]["entry"]["version"] == "FR_ENTRY_R2"
    assert report["diagnostics"]["research_components"]["valuation"]
    for key in ("model", "quality", "localization"):
        assert report["envelope"][key] == original["envelope"][key]
    for key in ("features", "reference_plan", "localization", "factor_ledger"):
        assert report["diagnostics"][key] == original["diagnostics"][key]
    assert "pae_authorization" not in report
    assert "pae_authorization" not in original
    assert report["fr2_execution"]["live_order_authorized"] is False
    assert report["fr2_execution"]["filled_shares"] == 0
    assert "ACCOUNT_INPUTS_UNKNOWN" in report["fr2_execution"]["reason_codes"]
    assert history.attrs == before


def test_generic_configured_model_does_not_silently_opt_into_research_admission():
    report = daily_report(research_history(explicit=False))
    validate_report(report)
    assert "fr2" not in report["diagnostics"]
    assert "pae_authorization" in report
    assert "fr2_execution" not in report


def test_research_missing_eps_companion_stays_unknown_instead_of_fabricating_operands():
    from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_SNAPSHOT_ATTR
    history = research_history()
    history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR]["les_v7_context"]["valuation_earnings"] = {}
    report = daily_report(history)
    validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    states = {gate["gate_id"]: gate["state"] for gate in report["diagnostics"]["gate_receipts"]}
    assert states["VALUATION"] == "UNKNOWN"
    assert report["diagnostics"]["fr2"]["admission"]["valuation_warning"] is False


def test_complete_same_original_eps_revision_and_official_price_inputs_produce_positive_valuation_pass():
    history = research_history()
    # A declared synthetic valuation scenario, not an edit to real prices or
    # evidence: use a lower price path and current official PE below its prior
    # history. The model and all reported EPS operands remain unchanged.
    for column in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        history[column] *= .1
    history.loc[history.index[-1], "PE"] = 6.
    report = daily_report(history)
    validate_report(report)
    valuation = report["diagnostics"]["fr2"]["admission"]["valuation"]
    assert valuation["native_state"] == "PASS"
    assert valuation["reason_codes"] == []
    assert valuation["snapshot"]["scenarios"] is not None
    assert valuation["snapshot"]["assumptions"]["dividends"] == "EXCLUDED_NOT_IMPUTED"
    selected = report["diagnostics"]["fr2"]["valuation_inputs"]["valuation_filings"]
    assert selected and all(type(row["revision_number"]) is int for row in selected)
    assert "pae_authorization" not in report and not report["fr2_execution"]["live_order_authorized"]


def test_daily_input_errors_are_bound_before_research_overlay_and_survive_restoration():
    history = research_history()
    history.attrs["les_v6_analysis_session"] = "2026-10-15"
    report = daily_report(history)
    validate_report(report)
    original = restore_original(report)
    assert original["diagnostics"]["daily_input_errors"] == ["STALE_MARKET_DATA"]
    assert report["diagnostics"]["daily_input_errors"] == ["STALE_MARKET_DATA"]
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not report["fr2_execution"]["allocation_authorized"]


@pytest.mark.parametrize("damage", ["bundle", "scope", "live", "order", "replay", "baseline", "native"])
def test_explicit_research_marker_must_match_the_real_daily_model_and_scope(damage):
    history = research_history()
    metadata = json.loads(history.attrs["les_v7_research_metadata_json"])
    if damage == "bundle": metadata["bundle_sha256"] = "0" * 64
    elif damage == "scope": metadata["symbols"] = ["OTHER_SYNTHETIC"]
    elif damage == "live": metadata["approved_for_live"] = True
    elif damage == "order": metadata["order_authority"] = True
    elif damage == "replay": history.attrs["replay_as_of"] = history.attrs[SIGNAL_DATE_ATTR]
    elif damage == "baseline": history.attrs["les_v7_rule_baseline"] = True
    else: history.attrs["les_v7_native_evidence"] = False
    history.attrs["les_v7_research_metadata_json"] = json.dumps(metadata)
    with pytest.raises(ValueError, match="V7_RESEARCH_BUNDLE_DAILY_BINDING_MISMATCH"):
        daily_report(history)


def test_research_exec_checks_an_explicit_account_without_granting_legacy_pae_authority():
    history = research_history()
    cutoff = history.attrs[SIGNAL_DATE_ATTR]
    account = dict(value=1_000_000., available_cash=1_000_000., intended_weight=.25,
                   capacity_weight=.25, positions=[], as_of=cutoff)
    history.attrs["les_v6_portfolio"] = account
    report = daily_report(history)
    validate_report(report)
    assert report["fr2_execution"]["supplied_account"] == account
    assert "ACCOUNT_INPUTS_UNKNOWN" not in report["fr2_execution"]["reason_codes"]
    assert not report["fr2_execution"]["live_order_authorized"]
    assert "pae_authorization" not in report
    before_score = deepcopy(report["envelope"]["score"])
    bad_account = dict(account, available_cash=900_000.)
    checked = finalize_daily_report(report, account=bad_account)
    assert checked["envelope"]["score"] == before_score
    assert any(reason.startswith("ACCOUNT_INVALID:") for reason in checked["fr2_execution"]["reason_codes"])
    assert not checked["fr2_execution"]["allocation_authorized"]


def test_eps_fr2_renderer_explains_price_only_return_without_claiming_total_return():
    report = daily_report(research_history())
    before = canonical(report)
    html = render_reports([report])
    assert "FR2｜核實年度EPS價格估值｜僅價格報酬／不含股利" in html
    assert "BUY 75／BUY_SMALL 65" in html
    assert "不推估缺失股數" in html
    assert canonical(report) == before


def test_cli_preflight_metadata_lists_actual_research_admission_and_price_only_policy(monkeypatch, tmp_path):
    from interface import cli
    from tests.test_v7_research_cli import synthetic_verified_model
    args = synthetic_verified_model(monkeypatch, tmp_path)
    request, settings = cli.build_run_configuration([*args, "--skip-mops"])
    summary = cli.build_preflight_summary(request, settings)
    metadata = summary["les_v7_research_model"]
    assert metadata["decision_thresholds"] == dict(BUY=75., BUY_SMALL=65.)
    assert metadata["admission_component"]["version"] == "FR_ENTRY_R2"
    assert metadata["execution_component"]["version"] == "FR_EXEC_R1"
    assert metadata["valuation_policy"]["dividends"] == "EXCLUDED_NOT_IMPUTED"
    assert metadata["prediction_bundle_unchanged"] is True
    assert metadata["return_basis"] == "PRICE_RETURN_ONLY_NOT_TOTAL_RETURN"
    assert metadata["execution_action"] == "NO_ACTION" and metadata["order_authority"] is False
    assert summary["les_v7_readiness"]["state"] == "READY"
    assert summary["les_v7_readiness"]["mode"] == "FR2_EPS_RESEARCH_ONLY"
    assert summary["les_v7_readiness"]["formal_entry_enabled"] is False
    assert summary["les_v7_readiness"]["research_entry_enabled"] is True
    assert summary["les_v7_readiness"]["order_authority"] is False


def test_preflight_generic_model_keeps_its_policy_and_invalid_research_marker_is_blocked(monkeypatch, tmp_path):
    from application.les_v7_preflight import daily_v7_preflight
    from interface import cli
    from tests.test_v7_research_cli import synthetic_verified_model
    args = synthetic_verified_model(monkeypatch, tmp_path)
    _, settings = cli.build_run_configuration([*args, "--skip-mops"])
    at = "2026-10-18T16:00:00+08:00"
    generic = daily_v7_preflight(replace(settings, les_v7_research_metadata_json=None), checked_at=at)
    assert generic["state"] == "READY" and generic["mode"] == "FULL_V7"
    assert "admission_component" not in generic
    metadata = json.loads(settings.les_v7_research_metadata_json)
    metadata["bundle_sha256"] = "0" * 64
    invalid = daily_v7_preflight(replace(settings, les_v7_research_metadata_json=json.dumps(metadata)), checked_at=at)
    assert invalid["state"] == "BLOCKED" and invalid["formal_entry_enabled"] is False
    assert invalid["reason_codes"] == ["V7_RESEARCH_BUNDLE_PREFLIGHT_BINDING_MISMATCH"]


def test_research_daily_pipeline_csv_and_pae_keep_raw_policy_and_do_not_authorize_orders(tmp_path):
    from application.analysis_dataset import AnalysisDatasetBuilder
    from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
    from application.portfolio_contracts import PortfolioRuntimeContext
    from application.strategy_contracts import StrategyDecisionSchema
    from infrastructure.storage.csv_repository import CsvRepository
    from infrastructure.storage.report_evidence import iter_report_bundle
    builder = AnalysisDatasetBuilder()
    decisions = builder.build_decisions(research_market(), [[1, 1, 1, 1, .01]])
    item = decisions[0]
    StrategyDecisionSchema().validate(item)
    report = json.loads(item.value("LES v7 Report"))
    assert item.value("LES v7 Required Score") in {65., 75.}
    assert item.value("低點決策") == report["envelope"]["strategy"]["strategy_decision"]
    assert item.value("LES v7 Execution Eligible") is False
    assert item.value("Position Fraction") == item.value("建議股數") == 0
    pipeline = PortfolioAllocationV2Pipeline()  # Native null persistence, no account/state writes.
    enriched = pipeline.enrich(decisions, PortfolioRuntimeContext(
        account_equity=1_000_000., current_cash=1_000_000., positions=(), default_sector="ELEC"))
    assert enriched[0].value("LES v7 Report") == item.value("LES v7 Report")
    assert enriched[0].value("低點決策") == item.value("低點決策")
    CsvRepository(tmp_path).write_analysis_dataset("SYNTHETIC_RESEARCH_ONLY", builder.project_decisions(enriched))
    saved = list(iter_report_bundle(tmp_path / "SYNTHETIC_RESEARCH_ONLY_les_v7_reports.json", require_reference=True))
    assert saved == [report]
