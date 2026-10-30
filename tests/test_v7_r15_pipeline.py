"""SYNTHETIC ONLY: R15 formal wiring, not a real investment dataset."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.low_entry_v7_daily import daily_report, finalize_daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.policy_r15 import AVAILABLE_AT, BASELINE_VERSION, VALUATION_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy, current_source_baseline_version, require_baseline_policy
from tests.test_v7_r14_pipeline import market as old_market


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
    # Retained R15 regression must not silently become the latest daily policy
    # when a synthetic fixture happens to be dated after a later release.
    from unittest.mock import patch
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
        return daily_report(MarketHistoryAssembler().build(capture_comparison_contexts(value or market()), 0))


def test_r14_identity_and_new_release_boundary_preserved():
    from domain.les_v7.policy_r14 import BASELINE_VERSION as previous
    assert identity(baseline_policy(version=previous)) == "ec1dc441e672fcd5ff4fb91dba609baa130090c5319c73a583a44e6f3bc33109"
    before = timestamp(AVAILABLE_AT) - timedelta(microseconds=1)
    assert current_source_baseline_version(before) == previous
    assert current_source_baseline_version(AVAILABLE_AT) == BASELINE_VERSION
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=BASELINE_VERSION), market="TWSE", cutoff=before)


def test_formal_daily_r15_retains_unknown_and_both_score_routes():
    report = native(); validate_report(report)
    d = report["diagnostics"]
    assert d["rule_baseline"]["policy"]["policy_version"] == BASELINE_VERSION
    assert d["native_evidence"]["produced_receipts"]["VALUATION"]["valuation_diagnostics"]["policy_version"] == VALUATION_VERSION
    routes = d["r15_route_evaluation"]
    assert routes["position_limits_affect_score"] is False
    assert report["envelope"]["score"]["final_score"] == routes[routes["selected_route"].lower()]["score"]["final_score"]
    assert report["envelope"]["localization"]["status"] == "NOT_APPLICABLE"
    assert not report["pae_authorization"]["live_order_authorized"]


def fresh_market():
    value = market()
    restrictions = deepcopy(value.trading_restrictions)
    # New synthetic observation, not a change to any real retained evidence.
    at = value.signal_dates["2330"]
    day = timestamp(at).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
    restrictions["2330"].update(available_at=at, session_date=day, observation_date=day, scope="CURRENT_AS_OF")
    return replace(value, trading_restrictions=restrictions)


def test_complete_native_core_buy_binds_route_and_rejects_route_tampering():
    report = native(fresh_market()); validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] == "BUY"
    assert report["diagnostics"]["r15_route_evaluation"]["selected_route"] == "CORE"
    route = report["diagnostics"]["reference_entry_route"]
    assert route["risk_route"] == "CORE" and route["strategy_decision"] == "BUY"
    assert report["pae_authorization"]["reference_entry_route_id"] == route["route_id"]
    altered = deepcopy(report)
    altered["diagnostics"]["reference_entry_route"]["risk_route"] = "TACTICAL"
    with pytest.raises(ValueError, match="ROUTE"):
        validate_report(altered)


@pytest.mark.parametrize("damage", ["route_score", "route_name", "source_state", "admission_cap", "entry_receipt"])
def test_report_rejects_rehashed_routing_or_cap_tampering(damage):
    report = native(); d = report["diagnostics"]
    if damage == "route_score": d["r15_route_evaluation"]["tactical"]["score"]["final_score"] += 1
    elif damage == "route_name": d["r15_route_evaluation"]["selected_route"] = "CORE"
    elif damage == "source_state": d["r15_route_evaluation"]["source_state"] = "TAMPERED"
    elif damage == "admission_cap": d["entry_admission"]["tactical_total_cap"] += .1
    else: d["reference_entry_route"] = dict(route_id="FORGED")
    with pytest.raises(ValueError): validate_report(report)


def test_cash_caps_do_not_change_native_scores_or_routes():
    value = market()
    normal = native(value)
    empty = native(replace(value, les_v6_portfolio=dict(value.les_v6_portfolio, available_cash=0.)))
    assert normal["envelope"] == empty["envelope"]
    assert normal["diagnostics"] == empty["diagnostics"]


def test_native_early_tactical_is_small_only_and_downgrade_removes_route():
    from tests.test_v7_r14_early_native import early_market
    report = native(early_market()); validate_report(report)
    d = report["diagnostics"]
    assert report["envelope"]["strategy"]["strategy_decision"] == "BUY_SMALL"
    assert d["reference_entry_route"]["risk_route"] == "TACTICAL"
    assert d["reference_entry_route"]["setup_variant"] == "EARLY_SHORT_BASE_BREAKOUT_R15"
    assert report["pae_authorization"]["small_risk_fraction"] == .0005
    account = report.pop("pae_authorization")["account"]
    final = finalize_daily_report(report, account=account, daily_input_errors=["STALE_MARKET_DATA"])
    validate_report(final)
    assert final["envelope"]["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"
    assert "reference_entry_route" not in final["diagnostics"]
    assert not final["pae_authorization"]["authorized"]


def test_same_fixed_sources_account_and_features_between_r14_r15(monkeypatch):
    from domain.les_v7.policy_r14 import BASELINE_VERSION as previous
    value = market()
    with monkeypatch.context() as old:
        old.setattr("domain.les_v7.rule_baseline.current_source_baseline_version", lambda cutoff: previous)
        r14 = native(value)
    r15 = native(value)
    for row in (r14, r15): validate_report(row)
    for key in ("quality", "features", "feature_audit"):
        assert r14["diagnostics"][key] == r15["diagnostics"][key]
    assert r14["diagnostics"]["native_evidence"]["source_packet"] == r15["diagnostics"]["native_evidence"]["source_packet"]
    assert r14["pae_authorization"]["account_sha256"] == r15["pae_authorization"]["account_sha256"]


def test_public_finalization_keeps_stale_errors_and_explicit_account():
    report = native(fresh_market()); account = report.pop("pae_authorization")["account"]
    assert report["envelope"]["strategy"]["strategy_decision"] == "BUY"
    assert "reference_entry_route" in report["diagnostics"]
    final = finalize_daily_report(report, account=account, daily_input_errors=["STALE_MARKET_DATA"])
    validate_report(final)
    assert final["envelope"]["score"]["final_score"] is None
    assert final["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert final["pae_authorization"]["account_sha256"] == identity(account)
    assert not final["pae_authorization"]["authorized"]
    assert "reference_entry_route" not in final["diagnostics"]


def test_clean_r15_renderer_label_and_no_report_mutation():
    from infrastructure.report.low_entry_v7_renderer import LowEntryV7Renderer
    decisions = AnalysisDatasetBuilder().build_decisions(market(), [[1, 1, 1, 1, .01]])
    before = decisions[0].value("LES v7 Report")
    html = LowEntryV7Renderer().render(decisions * 2)
    assert html.count('data-entry-policy="R15"') == 1
    assert "核心 BUY／戰術 BUY_SMALL" in html
    assert decisions[0].value("LES v7 Report") == before
