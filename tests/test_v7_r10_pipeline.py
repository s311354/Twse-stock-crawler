"""SYNTHETIC ONLY: deployed R10 wiring, not sources, fills or performance."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (RECOVERY_BASELINE_POLICY, RECOVERY_AVAILABLE_AT,
    GRADED_BASELINE_POLICY, current_source_baseline_version, baseline_policy)
from tests.strategies.test_v7_entry_policy_r9 import graded_market


def market():
    old = graded_market()
    at = (timestamp(RECOVERY_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    from infrastructure.crawler.security_master import select_security_master
    from tests.test_security_master import packet
    sources = deepcopy(old.native_source_selections)
    sources["2330"]["security"] = select_security_master(packet(), "2330", at)
    return replace(old, signal_dates={"2330": at}, native_source_selections=sources,
                   les_v6_portfolio=dict(old.les_v6_portfolio, as_of=at))


def native(m=None):
    return daily_report(MarketHistoryAssembler().build(capture_comparison_contexts(m or market()), 0))


def test_release_boundary_does_not_borrow_new_policy_in_old_cutoff():
    prior = timestamp(RECOVERY_AVAILABLE_AT) - timedelta(microseconds=1)
    assert current_source_baseline_version(prior) == GRADED_BASELINE_POLICY
    assert current_source_baseline_version(RECOVERY_AVAILABLE_AT) == RECOVERY_BASELINE_POLICY
    from domain.les_v7.rule_baseline import require_baseline_policy
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=RECOVERY_BASELINE_POLICY), market="TWSE", cutoff=prior)


def test_daily_r10_uses_native_sources_and_explicit_comparable_fallback():
    report = native()
    validate_report(report)
    assert report["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == RECOVERY_BASELINE_POLICY
    assert report["diagnostics"]["quality"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R6"
    receipt = report["diagnostics"]["native_evidence"]["produced_receipts"]["VALUATION"]
    assert receipt["valuation_diagnostics"]["selected_method"] == "ORIGINAL_R1_FALLBACK"
    assert receipt["valuation_diagnostics"]["comparables"]["status"] == "INSUFFICIENT"
    assert "structure_plan" in report["diagnostics"]
    assert report["pae_authorization"]["live_order_authorized"] is False


def test_optional_capture_does_not_mutate_caller_or_legacy_versions():
    original = graded_market(); before = deepcopy(original.native_source_selections)
    assert capture_comparison_contexts(original) is original
    assert original.native_source_selections == before
    m = market(); before = deepcopy(m.native_source_selections)
    changed = capture_comparison_contexts(m)
    assert changed is not m and m.native_source_selections == before
    context = changed.native_source_selections["2330"]["r10_valuation_context"]
    assert context["universe_symbols"] == ("2330",)
    with pytest.raises(TypeError, match="READ_ONLY"):
        context["cutoff"] = "2030-01-01T00:00:00Z"


def test_account_and_zero_cash_never_change_score_or_native_decision():
    m = market(); first = native(m)
    other = native(replace(m, les_v6_portfolio=dict(m.les_v6_portfolio, available_cash=0.)))
    assert first["envelope"] == other["envelope"]
    assert first["diagnostics"] == other["diagnostics"]
    assert other["pae_authorization"]["authorized"] is False


def test_future_filing_does_not_change_cutoff_selected_inputs_or_result():
    m = market(); first = native(m)
    snapshots = deepcopy(m.fundamental_snapshots)
    for key in ("les_v6_mops_filings", "les_v6_filings"):
        future = deepcopy(snapshots["2330"][key][-1])
        future["available_at"] = "2030-01-01T00:00:00Z"
        future["revision_id"] = "SYNTHETIC_FUTURE_REVISION"
        snapshots["2330"][key].append(future)
    assert native(replace(m, fundamental_snapshots=snapshots)) == first


def test_rehashed_conditional_diagnostic_cannot_replace_verified_plan():
    report = native()
    report["diagnostics"]["structure_plan"]["reasons"].append("SYNTHETIC_UNAPPROVED")
    with pytest.raises(ValueError, match="STRUCTURE_RECOMPUTATION"):
        validate_report(report)


def test_serial_full_history_and_spawn_use_same_fixed_universe_capture():
    from infrastructure.daily_parallel import DailyProcessExecutor
    m = market(); profits = [[1, 1, 1, 1, .01]]
    serial = AnalysisDatasetBuilder().build_decisions(m, profits)
    assert serial == AnalysisDatasetBuilder(optimized_daily=False).build_decisions(m, profits)
    assert serial == AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)).build_decisions(m, profits)


def test_r9_policy_retains_original_valuation_and_plan():
    from domain.les_v7.valuation_price import policy
    r9 = baseline_policy(version=GRADED_BASELINE_POLICY)
    assert r9["valuation_policy"] == policy()
    assert r9["reference_policy"]["support_rule"] == "STRICT_PIVOT_ELSE_PRIOR_LL20"
    assert r9["reference_policy"]["version"] == "V7_RULE_BASELINE_H20_CLOSE_R1"
    assert r9["valuation_policy"]["bear"]["years"] == 3
    assert "entry_guard_policy" not in r9["reference_policy"]["cost_path_policy"]


def test_r10_observed_open_inside_band_still_rechecks_rr_and_atr():
    from domain.les_v7.contracts import ReferencePlan
    from domain.les_v7.plan_r10 import VERSION, policy as structure_policy
    from domain.les_v7.reference_path import evaluate_reference_path
    from tests.test_v7_reference_path import fixture
    old, window, path, options = fixture()
    bound = structure_policy(dict(policy_id="REFERENCE_H20", cost_path_policy=path),
                             available_at=path["available_at"])
    plan = ReferencePlan("REFERENCE_H20", VERSION, old.evidence_id, "REVERSAL",
        100., 97.1, 110., 2., (100., 101.), old.reference_shares, .02, .02)
    window[1].update(open=101., high=101.5, close=101.)
    result = evaluate_reference_path(plan, window, bound["cost_path_policy"], **options)
    assert result["status"] == "VERIFIED_NO_FILL"
    assert result["execution"]["filled"] is False
    assert result["execution"]["entry_risk_check"]["stop_atr"] > 1.5
    assert result["execution"]["no_fill_reason"] == "R10_ACTUAL_ENTRY_RISK_NOT_PASS"


def test_below_close_conditional_zone_cannot_authorize_a_signal():
    from domain.les_v7.plan_r10 import make_plan
    from tests.strategies.test_v7_plan_r10 import setup_bars, declared_policy, CUTOFF
    plan, evidence = make_plan("REVERSAL", dict(close=110., atr=5.), declared_policy(),
        "SYNTHETIC_ONLY", bars=setup_bars(), cutoff=CUTOFF)
    conditional = evidence["conditional_entry"]
    assert conditional["state"] == "UNFILLED"
    assert conditional["live_order_authorized"] is False
    assert plan.entry == 110. and conditional["band"][1] < plan.entry


@pytest.mark.parametrize("gate,state", [("DATA_INTEGRITY", "UNKNOWN"), ("TRADING_STATUS", "UNKNOWN"),
    ("TRADING_STATUS", "FAIL"), ("SAFETY", "UNKNOWN"), ("SAFETY", "FAIL"), ("STRESS_RISK", "FAIL")])
def test_r10_keeps_noncompensable_gates_even_with_high_technical_score(gate, state):
    from domain.les_v7.rule_baseline import score_baseline, select_baseline_context
    from tests.strategies.test_v7_entry_policy_r9 import outcome, unit_features, valuation, quality, gates
    from domain.les_v7.contracts import ReferencePlan
    features = unit_features(); v = valuation()
    p = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL",
        101., 99., 109., 2., (101., 101.5), 1, .1, .1)
    prepared = dict(policy=baseline_policy(version=RECOVERY_BASELINE_POLICY), features=features,
        selection=select_baseline_context(features, policy_version=RECOVERY_BASELINE_POLICY).record())
    result = score_baseline(prepared, plan=p, quality=quality(), gates=gates(v, **{gate: state}),
        native_features=dict(close=101., spread=.003, adv=100_000_000.), risks=[], standard=True,
        starter_gate=True, confirmed_gate=True, valuation_receipt=v,
        entry_binding=dict(symbol="2330", cutoff=RECOVERY_AVAILABLE_AT))
    assert result["score"]["raw_score"] >= 72.
    assert result["decision"] in {"NO_BUY", "DATA_INSUFFICIENT"}


def test_r10_report_has_one_concise_policy_label_without_changing_scores():
    from infrastructure.report.low_entry_v7_renderer import LowEntryV7Renderer
    m = market(); decisions = AnalysisDatasetBuilder().build_decisions(m, [[1, 1, 1, 1, .01]])
    before = decisions[0].value("LES v7 Report")
    html = LowEntryV7Renderer().render(decisions * 2)
    assert html.count('data-entry-policy="R10"') == 1
    assert "近期復甦／產業估值 · 型態一致交易計畫" in html
    assert decisions[0].value("LES v7 Report") == before


def test_source_verified_preferred_share_is_not_a_common_share_peer_or_global_unknown():
    from application.daily_stock import PER_STOCK_SEQUENCES
    from infrastructure.crawler.security_master import capture_security_master, select_security_master
    from tests.test_security_master import synthetic_instruments, synthetic_issuers
    m = market(); cutoff = m.signal_dates["2330"]
    master = capture_security_master(market="TWSE", instrument_content=synthetic_instruments(rows=[
        ("股票", ["2330　合成普通股", "TW0002330008", "1994/09/05", "上市", "半導體業", "ESVUFR", ""]),
        ("特別股", ["8112A　合成甲特", "TW0008112A08", "2022/03/07", "上市", "半導體業", "EPRNAR", ""])]),
        issuer_content=synthetic_issuers(), fetched_at="2026-10-14T06:00:00Z")
    values = {name: tuple(getattr(m, name)) * 2 for name in PER_STOCK_SEQUENCES}
    values.update(stock_count=2, stock_numbers=("2330", "8112A"), stock_names=("合成普通股", "合成甲特"),
        signal_dates={"2330": cutoff, "8112A": cutoff}, native_source_selections={
            symbol: dict(security=select_security_master(master, symbol, cutoff), official_pe=[])
            for symbol in ("2330", "8112A")})
    captured = capture_comparison_contexts(replace(m, **values))
    comparison = captured.native_source_selections["2330"]["r10_valuation_context"]
    assert comparison["peers"] == ()
    assert comparison["exclusions"][0]["reason"] == "VERIFIED_NON_COMMON_SECURITY_OUT_OF_SCOPE"
    assert comparison["exclusions"][0]["source_id"]
    report = daily_report(MarketHistoryAssembler().build(captured, 0))
    result = report["diagnostics"]["native_evidence"]["produced_receipts"]["VALUATION"]["valuation_diagnostics"]
    assert result["comparables"]["status"] == "INSUFFICIENT"  # Not false clearance or global UNKNOWN.
