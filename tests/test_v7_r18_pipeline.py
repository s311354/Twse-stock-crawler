"""SYNTHETIC ONLY: native R18 source, score, conditional price and account flow.

Only the policy-vintage selector is fixed for the explicit R18 candidate; no
score, context, source verdict, gate, plan or decision is replaced by a fixture.
This does not certify official historical coverage or profitable executions.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import json
from math import fsum
from types import SimpleNamespace
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
from domain.les_v7.policy_r18 import (AVAILABLE_AT, BASELINE_VERSION, EARLY_PULLBACK,
    PRICE_DISCOVERY, SUPPORT_RESISTANCE_LIMIT)
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy, require_baseline_policy
from tests.test_v7_plan_r18 import neutral_bars
from tests.test_v7_r16_pipeline import discovery_bars, market, pullback_bars
from tests.test_v7_r17_pipeline import core_market as previous_core_market


def core_market():
    """Same released R17 observations, evaluated after candidate publication."""
    from infrastructure.crawler.security_master import select_security_master
    from tests.test_security_master import packet
    value = previous_core_market()
    at = (timestamp(AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    sources = deepcopy(value.native_source_selections)
    sources["2330"]["security"] = select_security_master(packet(), "2330", at)
    restrictions = deepcopy(value.trading_restrictions)
    day = timestamp(at).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
    restrictions["2330"].update(available_at=at, session_date=day, observation_date=day, scope="CURRENT_AS_OF")
    return replace(value, signal_dates={"2330": at}, native_source_selections=sources,
        trading_restrictions=restrictions, les_v6_portfolio=dict(value.les_v6_portfolio, as_of=at))


def neutral_market():
    data = neutral_bars()
    # Legal observed prices and a high-close, higher-volume session provide
    # measurable demand. They do not assert a pattern, score or buy decision.
    data.loc[len(data) - 2, ["Close", "Bid", "Ask", "Amount"]] = [99.9, 99.8, 99.9, 999000.]
    data.loc[len(data) - 1, ["Open", "High", "Low", "Close", "Bid", "Ask", "Volume"]] = [99.9, 100., 99.3, 100., 99.9, 100., 20000.]
    data.loc[len(data) - 1, "Amount"] = data.iloc[-1]["Volume"] * 100.
    return market(data=data)


def native(value=None):
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
        return daily_report(MarketHistoryAssembler().build(capture_comparison_contexts(value or neutral_market()), 0))


@pytest.fixture(scope="module")
def neutral_report():
    return native()


@pytest.fixture(scope="module")
def core_report():
    return native(core_market())


def gate_states(report):
    return {row["gate_id"]: row["state"] for row in report["diagnostics"]["gate_receipts"]}


def test_explicit_candidate_has_new_thresholds_without_mutating_r17_or_core_economics():
    from domain.les_v7.policy_r17 import BASELINE_VERSION as old
    original, candidate = (baseline_policy(version=v) for v in (old, BASELINE_VERSION))
    assert candidate["unified_technical_policy"]["buy_minimum"] == 70.
    assert candidate["unified_technical_policy"]["small_minimum"] == 60.
    assert original["unified_technical_policy"]["buy_minimum"] == 72.
    assert original["unified_technical_policy"]["small_minimum"] == 62.
    for key in ("model_spec", "native_risk_limits", "valuation_policy", "valuation_input_policy", "quality_policy"):
        assert original[key] == candidate[key], key
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(candidate, market="TWSE", cutoff=timestamp(AVAILABLE_AT) - timedelta(microseconds=1))


def test_native_core_buy_uses_70_without_a_second_trigger_or_size_score(core_report):
    report = core_report; validate_report(report)
    d, e = report["diagnostics"], report["envelope"]
    assert e["strategy"]["strategy_decision"] == "BUY"
    assert e["score"]["final_score"] >= 70.
    assert all(state == "PASS" for state in gate_states(report).values())
    assert d["reference_plan"]["setup_variant"] is None
    assert d["r18_route_evaluation"]["core"]["buy_eligible"] is True
    assert d["reference_entry_route"]["risk_route"] == "CORE"
    assert report["pae_authorization"]["authorized"]
    assert not report["pae_authorization"]["live_order_authorized"]


def test_neutral_native_context_stays_neutral_and_is_small_only_unfilled(neutral_report):
    report = neutral_report; validate_report(report)
    d, e = report["diagnostics"], report["envelope"]
    assert e["event"]["context"] == "NO_CLEAR_CONTEXT"
    assert d["rule_baseline"]["selection"]["selected_setup"] == "NO_CLEAR_CONTEXT"
    assert d["reference_plan"]["setup_variant"] == SUPPORT_RESISTANCE_LIMIT
    assert e["strategy"]["strategy_decision"] == "BUY_SMALL"
    assert e["score"]["final_score"] >= 60.
    assert all(state == "PASS" for state in gate_states(report).values())
    routes = d["r18_route_evaluation"]
    assert routes["core"]["buy_eligible"] is False and routes["tactical"]["small_eligible"] is True
    assert routes["technical_confirmation"] == "EXISTING_FIXED_FACTOR_LEDGER_ONLY"
    assert d["reference_entry_route"]["entry_rule"] == "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND"
    assert d["reference_entry_route"]["expiry"] == "NEXT_SESSION_CLOSE"
    assert d["structure_plan"]["conditional_entry"]["state"] == "UNFILLED"
    assert not d["structure_plan"]["conditional_entry"]["assumes_fill"]
    assert not report["pae_authorization"]["live_order_authorized"]
    assert d["reference_plan"]["target_method"] == "CONFIRMED_UNBROKEN_OVERHEAD_RESISTANCE"
    assert d["reference_plan"]["target"] == 110.


@pytest.mark.parametrize("kind", ["A", "C"])
def test_native_existing_conditional_routes_keep_anchors_and_use_new_risk_once(kind):
    report = native(market(data=discovery_bars() if kind == "C" else pullback_bars()))
    validate_report(report)
    d, e = report["diagnostics"], report["envelope"]
    assert e["strategy"]["strategy_decision"] == "BUY_SMALL"
    assert d["reference_plan"]["setup_variant"] == (EARLY_PULLBACK if kind == "A" else PRICE_DISCOVERY)
    for name in ("close_risk", "band_risk"):
        value = d["r18_route_evaluation"]["tactical"][name]
        assert value["state"] == "PASS" and value["maximum_stop_atr"] == 2.5
        assert value["minimum_net_rr"] == 1.2
    assert d["reference_plan"]["stop"] < d["reference_plan"]["entry"] < d["reference_plan"]["target"]
    assert not report["pae_authorization"]["live_order_authorized"]


def test_fixed_100_point_ledger_has_no_pattern_deduction_or_sizing_feedback(neutral_report):
    d, score = neutral_report["diagnostics"], neutral_report["envelope"]["score"]
    rows = d["rule_baseline"]["score_evidence"]["conditions"]
    assert len({row["id"] for row in rows}) == len(rows)
    assert fsum(row["weight"] for row in rows) == pytest.approx(100.)
    assert fsum(row["earned_points"] for row in rows) == pytest.approx(score["raw_score"])
    assert not {"BASELINE_NO_CLEAR_CONTEXT", "BASELINE_TRIGGER_NOT_READY", "BASELINE_NOT_CONFIRMED"} & set(d["penalties"])
    assert d["safety_caps"] == {} and score["effective_cap"] is None
    assert score["final_score"] == pytest.approx(max(0., score["raw_score"] - fsum(d["penalties"].values())))
    assert d["r18_route_evaluation"]["position_limits_affect_score"] is False


@pytest.mark.parametrize("missing", ["restriction", "identity", "market", "financials", "valuation"])
def test_high_native_technical_score_does_not_compensate_unknown_evidence(missing):
    value = neutral_market()
    if missing == "restriction": value = replace(value, trading_restrictions={})
    elif missing == "identity":
        sources = deepcopy(value.native_source_selections); sources["2330"]["security"] = None
        value = replace(value, native_source_selections=sources)
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
    states = gate_states(report)
    if missing == "restriction": assert states["TRADING_STATUS"] == "UNKNOWN"
    elif missing in {"identity", "market"}: assert states["DATA_INTEGRITY"] == "UNKNOWN"
    elif missing == "valuation": assert states["VALUATION"] == "UNKNOWN"
    else: assert report["diagnostics"]["quality"]["status"] in {"UNKNOWN", "DATA_INSUFFICIENT"}


def test_known_restriction_and_missing_official_calendar_still_block():
    value = neutral_market(); restrictions = deepcopy(value.trading_restrictions)
    restrictions["2330"]["restricted"] = True
    restricted = native(replace(value, trading_restrictions=restrictions)); validate_report(restricted)
    assert gate_states(restricted)["TRADING_STATUS"] == "FAIL"
    assert restricted["envelope"]["strategy"]["strategy_decision"] == "NO_BUY"
    sources = deepcopy(value.native_source_selections); sources["2330"].pop("calendar_evidence")
    unknown = native(replace(value, native_source_selections=sources)); validate_report(unknown)
    assert unknown["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not unknown["pae_authorization"]["authorized"]


def test_known_adverse_financial_opinion_stays_reject_not_a_tactical_warning():
    import pandas as pd
    from domain.fundamental_derivation import FiledFact, FiledFinancials
    from domain.les_v7.financial_sources_r5 import audit_proof_r5
    from tests.strategies.test_v7_reported_quality import source_records
    rows = source_records(opinion="AdverseOpinionAbstract")
    for row in rows:
        period, proof = pd.Period(row["period_end"], freq="Q"), row["audit_evidence"]
        facts = tuple(FiledFact(**dict(fact, dimensions=tuple(tuple(d) for d in fact.get("dimensions", ()))))
                      for fact in proof["contexts"])
        filing = FiledFinancials(period.year, period.quarter, row["available_at"], row["revision_number"], facts,
            content_sha256=row["content_sha256"], source_url=row["source_url"], statement_scope=row["scope"],
            period_end=row["period_end"], audit_status=proof.get("audit_status"))
        row["audit_evidence_r5"] = audit_proof_r5(filing, symbol="2330", original_verified=True)
    value = neutral_market(); snapshots = deepcopy(value.fundamental_snapshots)
    snapshots["2330"].update(les_v6_filings=deepcopy(rows), les_v6_mops_filings=rows, les_v7_context={})
    report = native(replace(value, fundamental_snapshots=snapshots)); validate_report(report)
    assert report["diagnostics"]["quality"]["status"] == "REJECT"
    assert report["envelope"]["strategy"]["strategy_decision"] == "NO_BUY"
    assert not report["pae_authorization"]["authorized"]


@pytest.mark.parametrize("failure", ["spread", "stop", "rr"])
def test_bad_liquidity_invalid_support_or_bad_rr_is_not_relaxed_into_pass(failure):
    value = neutral_market()
    if failure == "spread":
        bids = deepcopy(value.bids); bids[0][-1] = 99. * 1000.
        value = replace(value, bids=bids)
    else:
        data = neutral_bars()
        data.loc[len(data) - 1, ["Open", "High", "Low", "Close", "Bid", "Ask"]] = [100., 101.5, 99.3, 100., 99.9, 100.]
        if failure == "stop": data.loc[len(data) - 1, "Low"] = 97.9
        else: data.loc[len(data) - 40, "High"] = 101.5
        value = market(data=data)
    report = native(value); validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not report["pae_authorization"]["authorized"]
    if failure in {"spread", "rr"}: assert gate_states(report)["STRESS_RISK"] == "FAIL"
    if failure == "rr": assert report["diagnostics"]["reference_plan"]["target"] == 101.5


def test_one_complete_execution_assessment_per_native_production(neutral_report):
    """Capture adapter operands, then run the pure engine without finalization.

    Public validation intentionally regenerates proof; it is not part of this
    production-call count and may never be skipped by actual daily callers.
    """
    from domain.les_v7.engine import evaluate
    from domain.les_v7.risk_r18 import assess_execution
    captured = {}
    def capture(data, *, context, features=None, **options):
        captured.update(data=data, context=context, shared=features, options=options)
        return {}
    # Build the source fixture before replacing the adapter port; its legacy
    # fixture constructors independently evaluate their own source receipts.
    history = MarketHistoryAssembler().build(capture_comparison_contexts(neutral_market()), 0)
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION), \
            patch("application.low_entry_v7_daily.get_strategy", return_value=SimpleNamespace(evaluate=capture)), \
            patch("application.low_entry_v7_daily.finalize_daily_report", side_effect=lambda report, **kwargs: report):
        daily_report(history)
    with patch("domain.les_v7.risk_r18.assess_execution", wraps=assess_execution) as check:
        report = evaluate(captured["data"], captured["context"], shared=captured["shared"], **captured["options"])
    assert check.call_count == 1
    assert report["envelope"] == neutral_report["envelope"]
    d = report["diagnostics"]; execution = d["execution_assessment"]
    assert execution == d["native_evidence"]["produced_receipts"]["STRESS_RISK"]["execution_assessment"]
    assert d["r18_route_evaluation"]["execution_assessment_id"] == execution["assessment_id"]
    with patch("domain.les_v7.risk_r18.assess_execution", wraps=assess_execution) as independently:
        validate_report(report)
    assert independently.call_count >= 1


def test_cash_held_exposure_and_loss_budget_change_size_never_score(neutral_report):
    report = neutral_report; before = deepcopy(report)
    account = dict(report["pae_authorization"]["account"], value=1_000_000., available_cash=1_000_000.,
        intended_weight=1., capacity_weight=1., positions=[])
    allowed = authorize_candidate(report, account)
    assert allowed["authorized"] and allowed["maximum_total_weight"] <= .01
    assert Decimal(str(allowed["bound_plan_loss_per_share"])) * allowed["indicative_shares"] <= Decimal("500")
    empty = authorize_candidate(report, dict(account, available_cash=0.))
    assert not empty["authorized"] and empty["indicative_shares"] == 0
    close = report["diagnostics"]["reference_plan"]["signal_close"]
    held = int(np.ceil(allowed["maximum_total_weight"] * account["value"] / close))
    exhausted = authorize_candidate(report, dict(account, positions=[dict(symbol="2330", shares=held)]))
    assert not exhausted["authorized"] and exhausted["incremental_weight_capacity"] == 0.
    assert report == before
    value = neutral_market()
    zero = native(replace(value, les_v6_portfolio=dict(value.les_v6_portfolio, available_cash=0.)))
    assert zero["envelope"] == report["envelope"] and zero["diagnostics"] == report["diagnostics"]


def test_wider_native_stop_never_increases_shares_under_the_same_absolute_budget(neutral_report):
    # Change a real operand of this synthetic path, not a retained plan or
    # source receipt. Both independently confirmed support levels are selected
    # by the same native engine before score/RR; observed resistance stays 120.
    data = neutral_bars(); count = len(data)
    data["High"], data["Low"] = 104., 96.
    data.loc[count - 70, "High"] = 120.
    data.loc[count - 12, "Low"] = 95.
    data.loc[count - 2, ["Close", "Bid", "Ask", "Amount"]] = [99.9, 99.8, 99.9, 999000.]
    data.loc[count - 1, ["Open", "High", "Low", "Close", "Bid", "Ask", "Volume", "Amount"]] = [
        99.9, 100., 99.3, 100., 99.9, 100., 20000., 2000000.]
    narrow = native(market(data=data)); validate_report(narrow)
    data.loc[count - 12, "Low"] = 94.
    wide = native(market(data=data)); validate_report(wide)
    for report in (narrow, wide):
        assert report["envelope"]["strategy"]["strategy_decision"] == "BUY_SMALL"
        assert report["diagnostics"]["reference_plan"]["target"] == 120.
        assert report["diagnostics"]["reference_plan"]["entry"] == 100.
    assert wide["diagnostics"]["reference_plan"]["stop"] < narrow["diagnostics"]["reference_plan"]["stop"]
    account = dict(neutral_report["pae_authorization"]["account"], value=1_000_000., available_cash=1_000_000.,
        intended_weight=1., capacity_weight=1., positions=[])
    narrow_quota, wide_quota = (authorize_candidate(report, account) for report in (narrow, wide))
    assert narrow_quota["authorized"] and wide_quota["authorized"]
    assert wide_quota["bound_plan_loss_per_share"] > narrow_quota["bound_plan_loss_per_share"]
    assert wide_quota["indicative_shares"] < narrow_quota["indicative_shares"]
    for quota in (narrow_quota, wide_quota):
        assert quota["small_risk_fraction"] == .0005 and quota["maximum_total_weight"] == .01
        assert Decimal(str(quota["bound_plan_loss_per_share"])) * quota["indicative_shares"] <= Decimal("500")
    # Here declared raw volatility makes the unchanged loss budget bind before
    # the 1% exposure ceiling, so a strict share decrease has an actual basis.
    assert all(quota["indicative_shares"] < int(10000. / quota["exposure_price"])
               for quota in (narrow_quota, wide_quota))


@pytest.mark.parametrize("damage", ["score", "support", "route", "cap", "source_state", "execution"])
def test_public_native_recomputation_rejects_rehashed_business_changes(neutral_report, damage):
    report = deepcopy(neutral_report); d = report["diagnostics"]
    if damage == "score": report["envelope"]["score"]["final_score"] += 1.
    elif damage == "support": d["structure_plan"]["anchors"]["support"]["value"] += .1
    elif damage == "route":
        route = d["reference_entry_route"]; route["risk_route"] = "CORE"
        route["route_id"] = identity({k: v for k, v in route.items() if k != "route_id"})
    elif damage == "cap": d["entry_admission"]["small_total_cap"] += .1
    elif damage == "source_state": d["r18_route_evaluation"]["source_state"] = "VERIFIED_BY_HIGH_SCORE"
    else:
        assessment = d["execution_assessment"]; assessment["tactical"]["state"] = "FAIL"
        assessment["assessment_id"] = identity({k: v for k, v in assessment.items() if k != "assessment_id"})
    with pytest.raises(ValueError): validate_report(report)
    with pytest.raises(ValueError): authorize_candidate(report, report["pae_authorization"]["account"])


def test_stale_daily_finalization_removes_route_and_does_not_keep_buy(neutral_report):
    report = deepcopy(neutral_report); account = report.pop("pae_authorization")["account"]
    result = finalize_daily_report(report, account=account, daily_input_errors=["STALE_MARKET_DATA"])
    validate_report(result)
    assert result["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert result["envelope"]["score"]["final_score"] is None
    assert "reference_entry_route" not in result["diagnostics"]
    assert not result["pae_authorization"]["authorized"]


def reference_execution(report, *, opening=None):
    """Synthetic observed opens prove contract mechanics, not official fills."""
    import pandas as pd
    from domain.les_v7.plan_r18 import plan_from_record
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


def test_native_limit_requires_next_observed_open_not_later_intraday_touch(neutral_report):
    from domain.les_v7.reference_path import _tick
    inside = reference_execution(neutral_report)
    assert inside["status"] == "VERIFIED_FILLED", inside
    assert inside["execution"]["conditional_entry"]["intraday_touch_authorizes_fill"] is False
    assert not inside["execution"]["approved_for_live"]
    upper = Decimal(str(neutral_report["diagnostics"]["reference_plan"]["band"][1]))
    outside = reference_execution(neutral_report, opening=float(upper + _tick(upper)))
    assert outside["status"] == "VERIFIED_NO_FILL"
    assert outside["execution"]["filled"] is False


def _initialize_explicit_r18(*args):
    """Spawn workers select only the candidate vintage, never a model outcome."""
    from infrastructure.daily_parallel import _initialize
    _initialize(*args)
    patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION).start()


def test_serial_unoptimized_and_spawn_match_all_native_business_fields():
    from infrastructure.daily_parallel import DailyProcessExecutor
    value = neutral_market(); before = deepcopy(value)
    work = DailyWorkContext("SYNTHETIC_R18_PARITY", enabled=True); probe = DailyPerformance()
    try:
        with work.activate(), probe.activate(), \
                patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
            serial = AnalysisDatasetBuilder().build_decisions(value, [[1, 1, 1, 1, .01]])
            full = AnalysisDatasetBuilder(optimized_daily=False).build_decisions(value, [[1, 1, 1, 1, .01]])
            with patch("infrastructure.daily_parallel._initialize", _initialize_explicit_r18):
                spawned = AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)).build_decisions(value, [[1, 1, 1, 1, .01]])
    finally:
        work.close()
    assert serial == full == spawned
    assert value == before
    report = json.loads(serial[0].value("LES v7 Report")); validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] == "BUY_SMALL"
    assert report["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == BASELINE_VERSION
    assert probe.counters["stock_packets_submitted"] == probe.calls["worker_sum.stock_compute"] == 1


def test_clean_renderer_has_correct_r18_threshold_and_unfilled_limit_label():
    from infrastructure.report.low_entry_v7_renderer import LowEntryV7Renderer
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
        decisions = AnalysisDatasetBuilder().build_decisions(neutral_market(), [[1, 1, 1, 1, .01]])
    before = decisions[0].value("LES v7 Report")
    html = LowEntryV7Renderer().render(decisions * 2)
    assert html.count('data-entry-policy="R18"') == 1
    assert "BUY 70／BUY_SMALL 60" in html
    assert "條件式／尚未成交" in html and "未使用機率預測模型" in html
    assert decisions[0].value("LES v7 Report") == before
