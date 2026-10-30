"""SYNTHETIC ONLY: R13 native integration, not real sources or BUY outcomes."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.features import CLOSE_SUPPORT_FEATURE_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (
    CLOSE_BASELINE_POLICY, RELAXED_AVAILABLE_AT, RELAXED_BASELINE_POLICY,
    baseline_policy, current_source_baseline_version, require_baseline_policy,
)
from tests.test_v7_r12_pipeline import market as close_market


def market():
    original = close_market()
    at = (timestamp(RELAXED_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    from infrastructure.crawler.security_master import select_security_master
    from tests.test_security_master import packet
    sources = deepcopy(original.native_source_selections)
    sources["2330"]["security"] = select_security_master(packet(), "2330", at)
    return replace(original, signal_dates={"2330": at}, native_source_selections=sources,
        les_v6_portfolio=dict(original.les_v6_portfolio, as_of=at))


def native(value=None):
    return daily_report(MarketHistoryAssembler().build(capture_comparison_contexts(value or market()), 0))


def test_r12_identity_and_release_boundary_remain_immutable():
    assert identity(baseline_policy(version=CLOSE_BASELINE_POLICY)) == (
        "e23bbb11610b2d6c4d67563a27def9058d2722153334000aa646824fb52707aa")
    before = timestamp(RELAXED_AVAILABLE_AT) - timedelta(microseconds=1)
    assert current_source_baseline_version(before) == CLOSE_BASELINE_POLICY
    assert current_source_baseline_version(RELAXED_AVAILABLE_AT) == RELAXED_BASELINE_POLICY
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=RELAXED_BASELINE_POLICY), market="TWSE", cutoff=before)


def test_policy_changes_only_approved_economics_risk_and_version_bindings():
    old, new = (baseline_policy(version=version) for version in (CLOSE_BASELINE_POLICY, RELAXED_BASELINE_POLICY))
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    assert changed == {"policy_version", "available_at", "mapping_version", "valuation_policy",
        "valuation_input_policy", "entry_admission_policy", "native_risk_limits", "reference_policy"}
    assert new["native_risk_limits"]["maximum_stop_atr"] == 1.8
    assert set(new["native_risk_limits"]["starter_stop_atr"].values()) == {1.8}
    assert new["native_risk_limits"]["minimum_net_rr"] == dict(
        REVERSAL=1.8, TREND_PULLBACK=1.6, LEADERSHIP_CONTINUATION=1.6)
    for key in ("maximum_spread", "maximum_reference_order_adv", "source_bound_hard_gates_required",
                "buy_quality_minimum", "small_quality_minimum"):
        assert old["native_risk_limits"][key] == new["native_risk_limits"][key]
    old_rules = old["reference_policy"]["structural_rules"]
    new_rules = new["reference_policy"]["structural_rules"]
    assert {key for key in old_rules if old_rules[key] != new_rules[key]} == {
        "maximum_stop_atr", "starter_stop_atr", "minimum_net_rr"}
    assert new_rules["stop_buffer_atr"] == .3
    assert new["valuation_policy"]["required_return"] == .08
    assert new["valuation_policy"]["maximum_bear_discounted_shortfall"] == .45
    assert new["entry_admission_policy"]["bear_warning_small_only"] is True
    assert new["entry_admission_policy"]["bear_warning_position_multiplier"] == .5
    for key in ("quality_tiers", "valuation_bands", "quality_buy_minimum", "quality_small_minimum"):
        assert new["entry_admission_policy"][key] == old["entry_admission_policy"][key]
    for key in ("recovery", "industry_profiles", "base", "bear", "bull", "basic_eps_haircut",
                "maximum_annual_age_days", "return_basis", "dividends", "dilution"):
        assert new["valuation_policy"][key] == old["valuation_policy"][key]


def test_default_native_daily_report_binds_r13_and_preserves_missing_evidence():
    report = native()
    validate_report(report)
    diagnostic = report["diagnostics"]
    assert diagnostic["rule_baseline"]["policy"]["policy_version"] == RELAXED_BASELINE_POLICY
    assert diagnostic["feature_audit"]["profile"] == CLOSE_SUPPORT_FEATURE_VERSION
    receipt = diagnostic["native_evidence"]["produced_receipts"]["VALUATION"]
    assert receipt["valuation_diagnostics"]["policy_version"] == "V7_ANNUAL_EPS_GRADED_PRICE_RETURN_R13"
    assert receipt["valuation_diagnostics"]["selected_method"] == "R13_ANNUAL_FALLBACK"
    assert receipt["valuation_diagnostics"]["comparables"]["status"] == "INSUFFICIENT"
    assert report["pae_authorization"]["live_order_authorized"] is False


def test_cash_changes_authorization_only_not_score_sources_or_gates():
    value = market()
    full = native(value)
    empty = native(replace(value, les_v6_portfolio=dict(value.les_v6_portfolio, available_cash=0.)))
    assert full["envelope"] == empty["envelope"]
    assert full["diagnostics"] == empty["diagnostics"]
    assert empty["pae_authorization"]["authorized"] is False


def test_future_financial_revision_does_not_change_selected_result():
    value = market()
    before = native(value)
    snapshots = deepcopy(value.fundamental_snapshots)
    for key in ("les_v6_mops_filings", "les_v6_filings"):
        future = deepcopy(snapshots["2330"][key][-1])
        future["available_at"] = "2030-01-01T00:00:00Z"
        future["revision_id"] = "SYNTHETIC_FUTURE_REVISION"
        snapshots["2330"][key].append(future)
    assert native(replace(value, fundamental_snapshots=snapshots)) == before


@pytest.mark.parametrize("change", ["valuation", "risk_policy", "total_cap", "score"])
def test_altered_r13_economics_caps_or_score_are_rejected(change):
    report = native()
    diagnostic = report["diagnostics"]
    if change == "valuation":
        diagnostic["native_evidence"]["produced_receipts"]["VALUATION"]["valuation_diagnostics"]["policy"]["required_return"] = .01
    elif change == "risk_policy":
        diagnostic["rule_baseline"]["policy"]["reference_policy"]["structural_rules"]["maximum_stop_atr"] = 999.
    elif change == "total_cap":
        diagnostic["entry_admission"]["small_total_cap"] += .1
    else:
        report["envelope"]["score"]["final_score"] += .1
    with pytest.raises(ValueError):
        validate_report(report)


def test_renderer_has_one_r13_label_and_does_not_modify_report():
    from infrastructure.report.low_entry_v7_renderer import LowEntryV7Renderer
    decisions = AnalysisDatasetBuilder().build_decisions(market(), [[1, 1, 1, 1, .01]])
    before = decisions[0].value("LES v7 Report")
    html = LowEntryV7Renderer().render(decisions * 2)
    assert html.count('data-entry-policy="R13"') == 1
    assert "年度 EPS 價格估值 · 分級限倉 · 型態一致風控" in html
    assert decisions[0].value("LES v7 Report") == before


def _differences(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            child = path + "." + key if path else key
            if key not in before or key not in after:
                yield dict(path=child, before=before.get(key), after=after.get(key))
            else:
                yield from _differences(before[key], after[key], child)
    elif before != after:
        yield dict(path=path, before=before, after=after)


def test_same_cutoff_r12_r13_sources_and_technical_ledger_match_with_all_differences_retained(monkeypatch, record_property):
    value = market()
    original_sources = deepcopy(value.native_source_selections)
    with monkeypatch.context() as legacy:
        legacy.setattr("domain.les_v7.rule_baseline.current_source_baseline_version", lambda cutoff: CLOSE_BASELINE_POLICY)
        old = native(value)
    new = native(value)
    validate_report(old)
    validate_report(new)
    assert value.native_source_selections == original_sources
    old_d, new_d = old["diagnostics"], new["diagnostics"]
    assert old_d["native_evidence"]["source_packet"] == new_d["native_evidence"]["source_packet"]
    for key in ("features", "feature_audit", "quality", "quality_policy", "context_matches",
                "starter", "confirmed", "evidence_families", "factor_ledger"):
        assert old_d[key] == new_d[key], key
    assert old["envelope"]["score"]["raw_score"] == new["envelope"]["score"]["raw_score"]
    assert old["envelope"]["score"]["factor_values"] == new["envelope"]["score"]["factor_values"]
    assert old["envelope"]["score"]["applicable_weights"] == new["envelope"]["score"]["applicable_weights"]
    for gate in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY"):
        assert old_d["native_evidence"]["produced_receipts"][gate] == new_d["native_evidence"]["produced_receipts"][gate]
    if old_d["reference_plan"] is not None:
        for key in ("entry", "stop", "target", "atr", "band", "reference_shares", "cost_up", "cost_stop"):
            assert old_d["reference_plan"][key] == new_d["reference_plan"][key], key
    assert old["pae_authorization"]["live_order_authorized"] is False
    assert new["pae_authorization"]["live_order_authorized"] is False
    changes = list(_differences(old, new))
    assert changes and len({row["path"] for row in changes}) == len(changes)
    # All changed values are retained, not erased/ignored to make equality pass.
    # Native validators above separately recompute policy, gates, score, caps
    # and eligibility for each version. This comparison is not deployed R12 at
    # the historical cutoff, nor proof of any real-stock BUY or profitability.
    for row in changes:
        row["reason"] = ("VERSION_BOUND_IDENTITY_AND_AUTHORIZATION_REBINDING" if
            row["path"].startswith(("parameter_version", "envelope.event.", "envelope.localization.", "pae_authorization."))
            else "APPROVED_POLICY_RECOMPUTATION_FULL_OLD_AND_NEW_VALUES_RETAINED")
    record_property("same_cutoff_r12_r13_all_differences", canonical(changes))
    assert old["envelope"]["event"]["cutoff_at"] == new["envelope"]["event"]["cutoff_at"]
