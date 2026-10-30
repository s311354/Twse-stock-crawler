"""SYNTHETIC ONLY: formal R14 wiring, not real evidence or performance."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.policy_r14 import AVAILABLE_AT, BASELINE_VERSION, VALUATION_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (RELAXED_BASELINE_POLICY, baseline_policy,
    current_source_baseline_version, require_baseline_policy)
from tests.test_v7_r13_pipeline import market as old_market


def market():
    from infrastructure.crawler.security_master import select_security_master
    from tests.test_security_master import packet
    original = old_market()
    at = (timestamp(AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    sources = deepcopy(original.native_source_selections)
    sources["2330"]["security"] = select_security_master(packet(), "2330", at)
    return replace(original, signal_dates={"2330": at}, native_source_selections=sources,
        les_v6_portfolio=dict(original.les_v6_portfolio, as_of=at))


def native(value=None):
    return daily_report(MarketHistoryAssembler().build(capture_comparison_contexts(value or market()), 0))


def test_r13_identity_and_release_boundary_are_preserved():
    assert identity(baseline_policy(version=RELAXED_BASELINE_POLICY)) == (
        "e282c9a245886afdfa8ddb11046cd36d3835ba09995a65e72da573b3bedff148")
    before = timestamp(AVAILABLE_AT) - timedelta(microseconds=1)
    assert current_source_baseline_version(before) == RELAXED_BASELINE_POLICY
    assert current_source_baseline_version(AVAILABLE_AT) == BASELINE_VERSION
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=BASELINE_VERSION), market="TWSE", cutoff=before)


def test_formal_daily_flow_uses_r14_without_model_or_probability_defaults():
    report = native()
    validate_report(report)
    d = report["diagnostics"]
    assert d["rule_baseline"]["policy"]["policy_version"] == BASELINE_VERSION
    assert d["native_evidence"]["produced_receipts"]["VALUATION"]["valuation_diagnostics"]["policy_version"] == VALUATION_VERSION
    assert report["envelope"]["localization"]["status"] == "NOT_APPLICABLE"
    assert not report["pae_authorization"]["live_order_authorized"]
    assert d["rule_baseline"]["policy"]["model_spec"]["factor_budgets"] == baseline_policy(version=RELAXED_BASELINE_POLICY)["model_spec"]["factor_budgets"]


def test_same_sources_cutoff_account_quality_and_features_compare_both_policies(monkeypatch):
    value = market()
    with monkeypatch.context() as old:
        old.setattr("domain.les_v7.rule_baseline.current_source_baseline_version", lambda cutoff: RELAXED_BASELINE_POLICY)
        previous = native(value)
    current = native(value)
    for result in (previous, current): validate_report(result)
    for key in ("quality", "features", "feature_audit"):
        assert previous["diagnostics"][key] == current["diagnostics"][key]
    assert previous["diagnostics"]["native_evidence"]["source_packet"] == current["diagnostics"]["native_evidence"]["source_packet"]
    assert previous["pae_authorization"]["account_sha256"] == current["pae_authorization"]["account_sha256"]
    assert previous["envelope"]["event"]["cutoff_at"] == current["envelope"]["event"]["cutoff_at"]


def test_cash_is_sizing_only_and_future_revision_does_not_change_sources():
    value = market()
    normal = native(value)
    empty = native(replace(value, les_v6_portfolio=dict(value.les_v6_portfolio, available_cash=0.)))
    assert normal["envelope"] == empty["envelope"]
    assert normal["diagnostics"] == empty["diagnostics"]
    snapshots = deepcopy(value.fundamental_snapshots)
    for key in ("les_v6_mops_filings", "les_v6_filings"):
        future = deepcopy(snapshots["2330"][key][-1])
        future.update(available_at="2030-01-01T00:00:00Z", revision_id="SYNTHETIC_FUTURE")
        snapshots["2330"][key].append(future)
    assert native(replace(value, fundamental_snapshots=snapshots)) == normal


@pytest.mark.parametrize("damage", ["valuation", "score", "cap", "subtype", "context", "risk"])
def test_report_recalculation_rejects_tampering(damage):
    report = native(); d = report["diagnostics"]
    if damage == "valuation":
        d["native_evidence"]["produced_receipts"]["VALUATION"]["valuation_diagnostics"]["policy"]["required_return"] = .01
    elif damage == "score": report["envelope"]["score"]["final_score"] += 1
    elif damage == "cap": d["entry_admission"]["small_total_cap"] += .01
    elif damage == "subtype": d["rule_baseline"]["selection"]["setup_variant"] = "EARLY_SHORT_BASE_BREAKOUT_R14"
    elif damage == "context": d["rule_baseline"]["selection"]["short_base_audit"]["state"] = "SYNTHETIC_TAMPER"
    else: d["rule_baseline"]["policy"]["reference_policy"]["structural_rules"]["maximum_stop_atr"] = 99.
    with pytest.raises(ValueError): validate_report(report)


def test_renderer_keeps_one_clean_version_label_without_modifying_reports():
    from infrastructure.report.low_entry_v7_renderer import LowEntryV7Renderer
    decisions = AnalysisDatasetBuilder().build_decisions(market(), [[1, 1, 1, 1, .01]])
    before = decisions[0].value("LES v7 Report")
    html = LowEntryV7Renderer().render(decisions * 2)
    assert html.count('data-entry-policy="R14"') == 1
    assert decisions[0].value("LES v7 Report") == before


def test_public_finalization_retains_stale_input_errors_and_explicit_account():
    from application.low_entry_v7_daily import finalize_daily_report
    result = native()
    account = result.pop("pae_authorization")["account"]
    final = finalize_daily_report(result, account=account, daily_input_errors=["STALE_MARKET_DATA"])
    validate_report(final)
    assert final["envelope"]["score"]["final_score"] is None
    assert final["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert final["diagnostics"]["daily_input_errors"] == ["STALE_MARKET_DATA"]
    assert final["pae_authorization"]["account_sha256"] == identity(account)
    assert not final["pae_authorization"]["authorized"]
