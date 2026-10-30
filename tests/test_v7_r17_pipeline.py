"""SYNTHETIC ONLY: native R17 source, score, plan and account integration.

Prices, financial originals, calendars and observations are explicit test
fixtures, not official historical coverage or evidence of profitable trades.
Only the deployment-version selector is fixed for explicit-vintage tests;
no score, gate, context, plan, decision or account permission is patched.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import json
from math import fsum
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance
from application.daily_work import DailyWorkContext
from application.low_entry_v7_daily import daily_report, finalize_daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.allocation import authorize_candidate
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.policy_r16 import BASELINE_VERSION as R16
from domain.les_v7.policy_r17 import AVAILABLE_AT, BASELINE_VERSION, EARLY_PULLBACK, PRICE_DISCOVERY
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy, current_source_baseline_version, require_baseline_policy
from tests.test_v7_r16_pipeline import discovery_bars, market, pullback_bars


def native(value=None):
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
        return daily_report(MarketHistoryAssembler().build(capture_comparison_contexts(value or market()), 0))


def core_market():
    """An ordinary synthetic pullback with authentic source-bound anchors.

    A low-volume declining penultimate session is an observed input, making
    down-volume contraction measurable rather than setting a technical flag.
    This fixture does not move the existing support, target or risk limits.
    """
    from infrastructure.crawler.security_master import select_security_master
    from tests.test_security_master import packet
    from tests.test_v7_r15_pipeline import fresh_market
    original = fresh_market()
    at = (timestamp(AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    sources = deepcopy(original.native_source_selections)
    sources["2330"]["security"] = select_security_master(packet(), "2330", at)
    restrictions = deepcopy(original.trading_restrictions)
    day = timestamp(at).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
    restrictions["2330"].update(available_at=at, session_date=day, observation_date=day, scope="CURRENT_AS_OF")
    closes, volumes, amounts = (deepcopy(getattr(original, name)) for name in ("closes", "volumes", "trade_values"))
    closes[0][-2] = 95.8 * 1000
    volumes[0][-2] = 100000.
    amounts[0][-2] = 95.8 * volumes[0][-2]
    return replace(original, signal_dates={"2330": at}, native_source_selections=sources,
        trading_restrictions=restrictions, les_v6_portfolio=dict(original.les_v6_portfolio, as_of=at),
        closes=closes, volumes=volumes, trade_values=amounts)


@pytest.fixture(scope="module")
def early_report():
    return native()


@pytest.fixture(scope="module")
def discovery_report():
    return native(market(data=discovery_bars()))


@pytest.fixture(scope="module")
def core_report():
    return native(core_market())


def gate_states(report):
    return {row["gate_id"]: row["state"] for row in report["diagnostics"]["gate_receipts"]}


def test_r16_identity_release_boundary_and_fixed_score_budget_are_preserved():
    assert identity(baseline_policy(version=R16)) == "52dd3204a8bf611abeb428fa2ee988b296918f69c3a62c0d54c73dfe7febd1de"
    before = timestamp(AVAILABLE_AT) - timedelta(microseconds=1)
    assert current_source_baseline_version(before) == R16
    assert current_source_baseline_version(AVAILABLE_AT) == BASELINE_VERSION
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=BASELINE_VERSION), market="TWSE", cutoff=before)
    old, new = (baseline_policy(version=version) for version in (R16, BASELINE_VERSION))
    for key in ("model_spec", "native_risk_limits", "valuation_policy", "valuation_input_policy", "quality_policy"):
        assert old[key] == new[key], key
    assert new["model_spec"]["factor_budgets"] == dict(
        structure=30, trend=10, relative_strength=20, demand=25, momentum=5, execution=10)


def test_complete_native_core_buy_does_not_require_a_second_technical_confirmation(core_report):
    report = core_report; validate_report(report)
    d, e = report["diagnostics"], report["envelope"]
    assert d["rule_baseline"]["policy"]["policy_version"] == BASELINE_VERSION
    assert e["strategy"]["strategy_decision"] == "BUY"
    assert e["score"]["final_score"] >= 72.
    assert all(state == "PASS" for state in gate_states(report).values())
    assert d["reference_plan"]["setup_variant"] is None
    assert d["reference_plan"]["stop"] < d["reference_plan"]["signal_close"] < d["reference_plan"]["target"]
    assert d["r17_route_evaluation"]["core"]["buy_eligible"] is True
    assert d["reference_entry_route"]["strategy_decision"] == "BUY"
    assert d["reference_entry_route"]["risk_route"] == "CORE"
    assert report["pae_authorization"]["authorized"]
    assert not report["pae_authorization"]["live_order_authorized"]
    assert e["localization"]["status"] == "NOT_APPLICABLE"


@pytest.mark.parametrize("kind", ["A", "C"])
def test_complete_native_small_routes_keep_source_anchors_and_unchanged_risk(kind, early_report, discovery_report):
    report = early_report if kind == "A" else discovery_report
    validate_report(report)
    d, e = report["diagnostics"], report["envelope"]
    variant = EARLY_PULLBACK if kind == "A" else PRICE_DISCOVERY
    assert e["strategy"]["strategy_decision"] == "BUY_SMALL"
    assert e["score"]["final_score"] >= 62.
    assert all(state == "PASS" for state in gate_states(report).values())
    assert d["rule_baseline"]["selection"]["setup_variant"] == variant
    assert d["reference_plan"]["setup_variant"] == variant
    assert d["reference_entry_route"]["risk_route"] == "TACTICAL"
    assert d["reference_entry_route"]["entry_rule"] == "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND"
    assert d["reference_entry_route"]["expiry"] == "NEXT_SESSION_CLOSE"
    assert d["r17_route_evaluation"]["core"]["buy_eligible"] is False
    assert d["r17_route_evaluation"]["tactical"]["small_eligible"] is True
    for check in ("close_risk", "band_risk"):
        risk = d["r17_route_evaluation"]["tactical"][check]
        assert risk["state"] == "PASS" and risk["maximum_stop_atr"] == 2.
        assert risk["minimum_net_rr"] == 1.3
    if kind == "A":
        assert d["reference_plan"]["entry_basis"] == "STRUCTURAL_CONDITIONAL_BAND"
        assert d["reference_plan"]["signal_close"] == 82.1 != d["reference_plan"]["entry"]
    else:
        assert d["rule_baseline"]["selection"]["short_base_audit"]["session_coverage"]["state"] == "PASS"
        assert d["reference_plan"]["target_method"] == "MODEL_PROJECTED_TARGET"
        assert d["structure_plan"]["anchors"]["resistance"] is None
        assert d["structure_plan"]["anchors"]["target_projection"]["multiple"] == .5
    assert not report["pae_authorization"]["live_order_authorized"]


@pytest.mark.parametrize("kind", ["A", "C", "CORE"])
def test_one_fixed_ledger_sums_to_the_score_without_duplicate_technical_deductions(
        kind, early_report, discovery_report, core_report):
    report = {"A": early_report, "C": discovery_report, "CORE": core_report}[kind]
    d, score = report["diagnostics"], report["envelope"]["score"]
    ledger = d["rule_baseline"]["score_evidence"]
    conditions = ledger["conditions"]
    assert len({row["id"] for row in conditions}) == len(conditions)
    assert fsum(row["weight"] for row in conditions) == pytest.approx(100.)
    assert fsum(row["earned_points"] for row in conditions) == pytest.approx(score["raw_score"])
    for name, weight in zip(score["factor_order"], (30, 10, 20, 25, 5, 10)):
        assert fsum(row["weight"] for row in conditions if row["group"] == name) == pytest.approx(weight)
    assert score["applicable_weights"] == [.3, .1, .2, .25, .05, .1]
    assert not {"BASELINE_NO_CLEAR_CONTEXT", "BASELINE_TRIGGER_NOT_READY", "BASELINE_NOT_CONFIRMED"} & set(d["penalties"])
    assert d["safety_caps"] == {} and score["effective_cap"] is None
    assert score["final_score"] == pytest.approx(max(0., score["raw_score"] - fsum(d["penalties"].values())))
    routes = d["r17_route_evaluation"]
    assert routes["technical_confirmation"] == "EXISTING_FIXED_FACTOR_LEDGER_ONLY"
    assert routes["position_limits_affect_score"] is False


@pytest.mark.parametrize("missing", ["restriction", "identity", "market", "financials", "valuation"])
def test_real_native_required_source_gaps_are_not_compensated_by_high_technical_scores(missing):
    value = market()
    if missing == "restriction": value = replace(value, trading_restrictions={})
    elif missing == "identity":
        selections = deepcopy(value.native_source_selections)
        selections["2330"]["security"] = None
        value = replace(value, native_source_selections=selections)
    elif missing == "market": value = replace(value, market_observed_at=None)
    elif missing == "valuation": value = replace(value, pe_ratios=[[None] * len(value.dates[0])])
    else:
        snapshots = deepcopy(value.fundamental_snapshots)
        snapshots["2330"].update(les_v6_filings=[], les_v6_mops_filings=[], les_v7_context={})
        value = replace(value, fundamental_snapshots=snapshots)
    report = native(value); validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not report["pae_authorization"]["authorized"]
    assert "reference_entry_route" not in report["diagnostics"]
    gates = gate_states(report)
    if missing == "restriction": assert gates["TRADING_STATUS"] == "UNKNOWN"
    elif missing in {"identity", "market"}: assert gates["DATA_INTEGRITY"] == "UNKNOWN"
    elif missing == "valuation": assert gates["VALUATION"] == "UNKNOWN"
    else: assert report["diagnostics"]["quality"]["status"] in {"UNKNOWN", "DATA_INSUFFICIENT"}


@pytest.mark.parametrize("kind", ["A", "C"])
def test_missing_official_sessions_is_unknown_not_an_assumed_weekday_window(kind):
    report = native(market(data=discovery_bars() if kind == "C" else pullback_bars(), calendar=False))
    validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert report["diagnostics"]["r17_route_evaluation"]["source_state"] == "UNKNOWN"
    assert "OFFICIAL_SESSION_COVERAGE" in report["diagnostics"]["r17_route_evaluation"]["source_unknowns"]
    assert not report["pae_authorization"]["authorized"]


def test_known_trading_restriction_still_fails_native_gate():
    value = market(); restrictions = deepcopy(value.trading_restrictions)
    restrictions["2330"]["restricted"] = True
    report = native(replace(value, trading_restrictions=restrictions)); validate_report(report)
    assert gate_states(report)["TRADING_STATUS"] == "FAIL"
    assert report["envelope"]["strategy"]["strategy_decision"] == "NO_BUY"
    assert not report["pae_authorization"]["authorized"]


def test_invalid_source_floor_still_refuses_and_does_not_change_to_price_discovery():
    data = pullback_bars(); data.loc[len(data) - 1, "Low"] = 79.2
    report = native(market(data=data)); validate_report(report)
    assert report["diagnostics"]["rule_baseline"]["selection"].get("setup_variant") != PRICE_DISCOVERY
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not report["pae_authorization"]["authorized"]


def test_known_failed_rr_keeps_its_native_projection_instead_of_a_favorable_target():
    from tests.test_v7_plan_r16 import discovery_bars as insufficient_rr_bars
    report = native(market(data=insufficient_rr_bars())); validate_report(report)
    d = report["diagnostics"]
    assert d["structure_plan"]["anchors"]["target_projection"]["multiple"] == .5
    assert d["r17_route_evaluation"]["tactical"]["band_risk"]["state"] == "FAIL"
    assert gate_states(report)["STRESS_RISK"] == "FAIL"
    assert report["envelope"]["strategy"]["strategy_decision"] == "NO_BUY"
    assert not report["pae_authorization"]["authorized"]


@pytest.mark.parametrize("kind,cap,budget", [("A", .005, 500.), ("C", .0025, 250.)])
def test_zero_cash_held_exposure_and_plan_loss_reduce_size_not_score(
        kind, cap, budget, early_report, discovery_report):
    report = early_report if kind == "A" else discovery_report
    before = deepcopy(report)
    account = dict(report["pae_authorization"]["account"], value=1_000_000., available_cash=1_000_000.,
        intended_weight=1., capacity_weight=1., positions=[])
    allowed = authorize_candidate(report, account)
    assert allowed["authorized"] and allowed["maximum_total_weight"] == cap
    assert Decimal(str(allowed["bound_plan_loss_per_share"])) * allowed["indicative_shares"] <= Decimal(str(budget))
    empty = authorize_candidate(report, dict(account, available_cash=0.))
    assert not empty["authorized"] and empty["indicative_shares"] == 0
    close = report["diagnostics"]["reference_plan"]["signal_close"]
    held_count = int(np.ceil(cap * account["value"] / close))
    held = authorize_candidate(report, dict(account, positions=[dict(symbol="2330", shares=held_count)]))
    assert not held["authorized"] and held["incremental_weight_capacity"] == 0.
    assert report == before


def test_account_does_not_reenter_native_les_score_or_sources(early_report):
    value = market()
    empty = native(replace(value, les_v6_portfolio=dict(value.les_v6_portfolio, available_cash=0.)))
    assert empty["envelope"] == early_report["envelope"]
    assert empty["diagnostics"] == early_report["diagnostics"]
    assert empty["pae_authorization"]["indicative_shares"] == 0


@pytest.mark.parametrize("damage", ["score", "floor", "route", "cap", "source_state"])
def test_public_recomputation_rejects_rehashed_business_tampering(early_report, damage):
    report = deepcopy(early_report); d = report["diagnostics"]
    if damage == "score": report["envelope"]["score"]["final_score"] += 1.
    elif damage == "floor": d["structure_plan"]["anchors"]["support"]["value"] += .1
    elif damage == "route":
        route = d["reference_entry_route"]
        route["risk_route"] = "CORE"
        route["route_id"] = identity({k: v for k, v in route.items() if k != "route_id"})
    elif damage == "cap": d["entry_admission"]["small_total_cap"] += .1
    else: d["r17_route_evaluation"]["source_state"] = "VERIFIED_BY_HIGH_SCORE"
    with pytest.raises(ValueError): validate_report(report)
    with pytest.raises(ValueError): authorize_candidate(report, report["pae_authorization"]["account"])


def test_stale_finalization_strips_conditional_route_and_score(early_report):
    report = deepcopy(early_report); account = report.pop("pae_authorization")["account"]
    result = finalize_daily_report(report, account=account, daily_input_errors=["STALE_MARKET_DATA"])
    validate_report(result)
    assert result["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert result["envelope"]["score"]["final_score"] is None
    assert "reference_entry_route" not in result["diagnostics"]
    assert not result["pae_authorization"]["authorized"]


def reference_execution(report, *, opening=None):
    """Explicit synthetic H20 observations, not certified broker executions."""
    import pandas as pd
    from domain.les_v7.plan_r17 import plan_from_record
    from domain.les_v7.reference_path import evaluate_reference_path
    from tests.test_v7_reference_path import fixture
    _, window, _, options = fixture()
    d = report["diagnostics"]; plan = plan_from_record(d["reference_plan"])
    sessions = pd.bdate_range(start=report["signal_date"], periods=21).strftime("%Y-%m-%d").tolist()
    price = plan.band[0]
    for row, session in zip(window, sessions):
        row.update(session=session, open=price, high=price, low=price, close=price,
            available_at=session + "T13:30:00+08:00")
    raw = d["rule_baseline"]["market_rows"][-1]
    window[0].update({target: raw[source] for target, source in
                     (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"))})
    if opening is not None:
        window[1].update(open=opening, high=max(opening, price), low=min(opening, price))
    options.update(expected_sessions=sessions, cutoff=report["envelope"]["event"]["cutoff_at"],
        entry_route=d["reference_entry_route"])
    options["corporate_action_coverage"].update(start_session=sessions[0], end_session=sessions[-1],
        available_at=sessions[-1] + "T13:30:00+08:00")
    return evaluate_reference_path(plan, window,
        d["rule_baseline"]["policy"]["reference_policy"]["cost_path_policy"], **options)


def test_authentic_native_conditional_route_requires_next_observed_open_not_intraday_touch(early_report):
    from domain.les_v7.reference_path import _tick
    inside = reference_execution(early_report)
    assert inside["status"] == "VERIFIED_FILLED", inside
    execution = inside["execution"]
    assert execution["entry"] == early_report["diagnostics"]["reference_plan"]["band"][0]
    assert execution["entry_risk_check"]["state"] == "PASS"
    assert execution["conditional_entry"]["signal_close"] == 82.1
    assert execution["conditional_entry"]["intraday_touch_authorizes_fill"] is False
    assert execution["approved_for_live"] is False
    upper = Decimal(str(early_report["diagnostics"]["reference_plan"]["band"][1]))
    outside = reference_execution(early_report, opening=float(upper + _tick(upper)))
    assert outside["status"] == "VERIFIED_NO_FILL"
    assert outside["execution"]["conditional_entry"]["state"] == "EXPIRED_AT_NEXT_SESSION_CLOSE"
    assert outside["execution"]["filled"] is False


def test_native_projected_target_remains_a_model_scenario_in_reference_execution(discovery_report):
    result = reference_execution(discovery_report)
    assert result["status"] == "VERIFIED_FILLED", result
    assert result["execution"]["target_method"] == "MODEL_PROJECTED_TARGET"
    assert result["execution"]["native_resistance_target"] is None
    assert result["execution"]["native_plan_target"] == discovery_report["diagnostics"]["reference_plan"]["target"]


def _initialize_explicit_r17(*args):
    """Pin the retained vintage in children, not a score or source verdict."""
    from infrastructure.daily_parallel import _initialize
    _initialize(*args)
    patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION).start()


@pytest.mark.parametrize("kind", ["A", "C"])
def test_explicit_r17_serial_unoptimized_and_spawn_preserve_complete_native_contracts(kind):
    from infrastructure.daily_parallel import DailyProcessExecutor
    value = market(data=discovery_bars() if kind == "C" else pullback_bars())
    before = deepcopy(value); profits = [[1, 1, 1, 1, .01]]
    work = DailyWorkContext("SYNTHETIC_R17_PARITY", enabled=True)
    probe = DailyPerformance()
    try:
        with work.activate(), probe.activate(), \
                patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
            serial = AnalysisDatasetBuilder().build_decisions(value, profits)
            full = AnalysisDatasetBuilder(optimized_daily=False).build_decisions(value, profits)
            with patch("infrastructure.daily_parallel._initialize", _initialize_explicit_r17):
                spawned = AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)).build_decisions(value, profits)
    finally:
        work.close()
    assert serial == full == spawned
    assert value == before
    report = json.loads(serial[0].value("LES v7 Report")); validate_report(report)
    assert report["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == BASELINE_VERSION
    assert report["envelope"]["strategy"]["strategy_decision"] == "BUY_SMALL"
    assert probe.counters["stock_packets_submitted"] == 1
    assert probe.calls["worker_sum.stock_compute"] == 1


def test_clean_explicit_r17_renderer_keeps_one_version_and_unfilled_conditional_label():
    from infrastructure.report.low_entry_v7_renderer import LowEntryV7Renderer
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
        decisions = AnalysisDatasetBuilder().build_decisions(market(), [[1, 1, 1, 1, .01]])
    before = decisions[0].value("LES v7 Report")
    html = LowEntryV7Renderer().render(decisions * 2)
    assert html.count('data-entry-policy="R17"') == 1
    assert "條件式／尚未成交" in html
    assert "未使用機率預測模型" in html
    assert decisions[0].value("LES v7 Report") == before
