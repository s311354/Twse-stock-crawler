"""SYNTHETIC ONLY: R16 native source/score/plan/account contracts.

No official source repair, deployment, trading or performance claim. The
actual daily plugin, source producers and public validators are used; no gate,
score, selected context or decision is patched into a positive report.
"""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, ROUND_FLOOR

import numpy as np
import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.low_entry_v7_daily import daily_report, finalize_daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.allocation import authorize_candidate
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.policy_r16 import AVAILABLE_AT, BASELINE_VERSION, EARLY_PULLBACK, PRICE_DISCOVERY
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy, current_source_baseline_version, require_baseline_policy
from tests.test_v7_r14_early_native import early_market


def pullback_bars():
    """Causal impulse, unconfirmed tied floor and two subsequent prior closes."""
    n = 130
    data = pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-20", periods=n).strftime("%Y-%m-%d"),
        Open=[80.] * n, High=[81.] * n, Low=[79.] * n, Close=[80.] * n,
        Volume=[10000.] * n, Amount=[800000.] * n, Trades=[100] * n,
        Bid=[79.9] * n, Ask=[80.] * n, Complete=[True] * n))
    data.loc[10, "High"] = 90.
    data.loc[n - 12, ["Open", "High", "Low", "Close"]] = [83., 90., 82., 84.]
    for index, (low, high, close) in enumerate(
        [(80., 81.4, 81.3), (80., 81.4, 81.3), (80.8, 82., 81.7),
         (81., 82., 81.8), (81.1, 82., 81.9)], start=n - 6):
        data.loc[index, ["Open", "Low", "High", "Close", "Bid", "Ask"]] = [close, low, high, close, close - .1, close]
    data.loc[n - 1, ["Open", "Low", "High", "Close", "Volume", "Amount", "Bid", "Ask"]] = [
        82., 81.8, 82.2, 82.1, 20000., 1642000., 82., 82.1]
    return data


def native(value=None):
    # Retained R16 regression selects its own released policy; only the
    # deployment-version selector is fixed, never a score, gate or decision.
    from unittest.mock import patch
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
        return daily_report(MarketHistoryAssembler().build(capture_comparison_contexts(value or market()), 0))


def discovery_bars():
    """Volatile old range, subsequently confirmed short floor, genuine high.

    This deliberately unusual engineering fixture tests a fully qualifying
    C route without changing its 0.5 projection or RR policy. It is not a
    claim that real data contain this shape or that its target will be met.
    """
    n = 130
    data = pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-20", periods=n).strftime("%Y-%m-%d"),
        Open=[80.] * n, High=[84.] * n, Low=[76.] * n, Close=[80.] * n,
        Volume=[10000.] * n, Amount=[800000.] * n, Trades=[100] * n,
        Bid=[79.9] * n, Ask=[80.] * n, Complete=[True] * n))
    data.loc[n - 15, "Low"] = 55.
    for index in range(n - 6, n - 1):
        data.loc[index, ["Open", "High", "Low", "Close", "Bid", "Ask"]] = [84., 84., 80., 84., 83.9, 84.]
    data.loc[n - 1, ["Open", "High", "Low", "Close", "Volume", "Amount", "Bid", "Ask"]] = [
        84., 84.1, 83.9, 84.1, 20000., 1682000., 84., 84.1]
    return data


def market(*, data=None, calendar=True):
    data = pullback_bars() if data is None else data
    result = early_market(bars=data)
    if calendar:
        from tests.test_v7_plan_r16 import synthetic_calendar
        selections = deepcopy(result.native_source_selections)
        selections["2330"]["calendar_evidence"] = synthetic_calendar(data)
        result = replace(result, native_source_selections=selections)
    return result


@pytest.fixture(scope="module")
def early_report():
    return native()


@pytest.fixture(scope="module")
def discovery_report():
    return native(market(data=discovery_bars(), calendar=True))


def test_r15_identity_and_new_release_boundary_remain_exact():
    from domain.les_v7.policy_r15 import BASELINE_VERSION as old
    assert identity(baseline_policy(version=old)) == "30ff3530e27fd55f2aee4e6b40cd6568d86e513aeb1d356c657ad98d90e53086"
    before = timestamp(AVAILABLE_AT) - pd.Timedelta(microseconds=1)
    assert current_source_baseline_version(before) == old
    assert current_source_baseline_version(AVAILABLE_AT) == BASELINE_VERSION
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=BASELINE_VERSION), market="TWSE", cutoff=before)


def test_complete_native_early_route_is_small_only_with_a_distinct_true_close(early_report):
    report = early_report; validate_report(report)
    d, e = report["diagnostics"], report["envelope"]
    assert d["rule_baseline"]["policy"]["policy_version"] == BASELINE_VERSION
    assert d["rule_baseline"]["selection"]["selected_setup"] == "TREND_PULLBACK"
    assert d["rule_baseline"]["selection"]["setup_variant"] == EARLY_PULLBACK
    assert all(row["state"] == "PASS" for row in d["gate_receipts"])
    assert e["strategy"]["strategy_decision"] == "BUY_SMALL"
    assert e["score"]["final_score"] >= 62.
    assert d["starter"] is True and d["confirmed"] is False
    assert "BASELINE_NOT_CONFIRMED" not in d["penalties"]
    plan = d["reference_plan"]
    assert plan["entry_basis"] == "STRUCTURAL_CONDITIONAL_BAND"
    assert plan["signal_close"] == 82.1 and plan["entry"] == plan["band"][1] != 82.1
    assert d["rule_baseline"]["features"]["close"] == plan["signal_close"]
    valuation = d["native_evidence"]["produced_receipts"]["VALUATION"]["valuation_diagnostics"]
    assert valuation["inputs"]["current_price"] == plan["signal_close"]
    route = d["reference_entry_route"]
    assert route["strategy_decision"] == "BUY_SMALL" and route["risk_route"] == "TACTICAL"
    assert route["entry_rule"] == "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND"
    assert route["expiry"] == "NEXT_SESSION_CLOSE"
    assert route["structure_id"] == plan["structure_id"]
    assert report["pae_authorization"]["live_order_authorized"] is False
    assert report["pae_authorization"]["reference_entry_route_id"] == route["route_id"]


def test_complete_native_discovery_is_a_projection_not_observed_resistance(discovery_report):
    report = discovery_report; validate_report(report)
    d, e = report["diagnostics"], report["envelope"]
    assert e["strategy"]["strategy_decision"] == "BUY_SMALL"
    assert all(row["state"] == "PASS" for row in d["gate_receipts"])
    assert d["rule_baseline"]["selection"]["setup_variant"] == PRICE_DISCOVERY
    audit = d["rule_baseline"]["selection"]["short_base_audit"]
    assert audit["session_coverage"]["state"] == "PASS"
    assert audit["metrics"]["prior120_high"] < d["reference_plan"]["signal_close"]
    plan = d["reference_plan"]
    assert plan["entry_basis"] == "CUTOFF_CLOSE" and plan["entry"] == plan["signal_close"]
    assert plan["target_method"] == "MODEL_PROJECTED_TARGET"
    assert d["structure_plan"]["anchors"]["resistance"] is None
    target = d["structure_plan"]["anchors"]["target_projection"]
    assert target["multiple"] == .5 and target["value"] == plan["target"]
    assert plan["target"] == 84. + .5 * (84. - 55.)
    assert d["starter"] is True and d["confirmed"] is False
    assert "BASELINE_NOT_CONFIRMED" not in d["penalties"]
    authorization = report["pae_authorization"]
    assert authorization["authorized"] and authorization["maximum_total_weight"] == .0025
    assert authorization["small_risk_fraction"] == .00025
    assert authorization["indicative_shares"] * authorization["bound_plan_loss_per_share"] <= 250.
    assert authorization["live_order_authorized"] is False


def test_discovery_cannot_assume_120_valid_sessions_without_official_calendar():
    result = native(market(data=discovery_bars(), calendar=False)); validate_report(result)
    selection = result["diagnostics"]["rule_baseline"]["selection"]
    assert selection.get("setup_variant") is None
    audit = selection["short_base_audit"]
    assert audit["state"] == "UNKNOWN" and audit["session_coverage"]["state"] == "UNKNOWN"
    assert result["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not result["pae_authorization"]["authorized"]


def test_early_floor_cannot_assume_20_complete_prior_sessions_without_calendar():
    result = native(market(calendar=False)); validate_report(result)
    selection = result["diagnostics"]["rule_baseline"]["selection"]
    assert selection.get("setup_variant") is None
    audit = selection["short_base_audit"]
    assert audit["state"] == "UNKNOWN" and audit["session_coverage"]["state"] == "UNKNOWN"
    assert result["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not result["pae_authorization"]["authorized"]


def test_discovery_missing_rr_is_not_rescued_by_inflating_the_fixed_projection():
    from tests.test_v7_plan_r16 import discovery_bars as insufficient_rr_bars
    report = native(market(data=insufficient_rr_bars(), calendar=True)); validate_report(report)
    d = report["diagnostics"]
    assert d["rule_baseline"]["selection"]["setup_variant"] == PRICE_DISCOVERY
    assert d["structure_plan"]["starter_band_evaluation"]["state"] == "FAIL"
    assert d["structure_plan"]["anchors"]["target_projection"]["multiple"] == .5
    assert next(row["state"] for row in d["gate_receipts"] if row["gate_id"] == "STRESS_RISK") == "FAIL"
    assert report["envelope"]["strategy"]["strategy_decision"] == "NO_BUY"
    assert "reference_entry_route" not in d


def test_known_financial_audit_hard_risk_cannot_use_the_new_floor():
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
    value = market(); snapshots = deepcopy(value.fundamental_snapshots)
    snapshots["2330"].update(les_v6_filings=deepcopy(rows), les_v6_mops_filings=rows, les_v7_context={})
    report = native(replace(value, fundamental_snapshots=snapshots)); validate_report(report)
    assert report["diagnostics"]["quality"]["status"] == "REJECT"
    assert report["envelope"]["strategy"]["strategy_decision"] == "NO_BUY"
    assert not report["pae_authorization"]["authorized"]


def test_broken_signal_floor_cannot_switch_to_another_route_or_move_the_stop():
    data = pullback_bars(); data.loc[len(data) - 1, "Low"] = 79.2
    report = native(market(data=data)); validate_report(report)
    assert report["diagnostics"]["rule_baseline"]["selection"].get("setup_variant") != PRICE_DISCOVERY
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not report["pae_authorization"]["authorized"]


@pytest.mark.parametrize("missing", ["restriction", "identity", "market", "financials"])
def test_new_floor_never_overrides_unknown_necessary_sources(missing):
    value = market()
    if missing == "restriction": value = replace(value, trading_restrictions={})
    elif missing == "identity":
        selections = deepcopy(value.native_source_selections)
        selections["2330"]["security"] = None
        value = replace(value, native_source_selections=selections)
    elif missing == "market": value = replace(value, market_observed_at=None)
    else:
        snapshots = deepcopy(value.fundamental_snapshots)
        snapshots["2330"].update(les_v6_filings=[], les_v6_mops_filings=[], les_v7_context={})
        value = replace(value, fundamental_snapshots=snapshots)
    result = native(value); validate_report(result)
    assert result["diagnostics"]["rule_baseline"]["selection"]["setup_variant"] == EARLY_PULLBACK
    assert result["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not result["pae_authorization"]["authorized"]
    assert "reference_entry_route" not in result["diagnostics"]


def test_known_trading_restriction_cannot_be_compensated_by_score():
    value = market(); restrictions = deepcopy(value.trading_restrictions)
    restrictions["2330"]["restricted"] = True
    report = native(replace(value, trading_restrictions=restrictions)); validate_report(report)
    assert next(row["state"] for row in report["diagnostics"]["gate_receipts"]
                if row["gate_id"] == "TRADING_STATUS") == "FAIL"
    assert report["envelope"]["strategy"]["strategy_decision"] == "NO_BUY"
    assert not report["pae_authorization"]["authorized"]


@pytest.mark.parametrize("damage", ["score", "source_floor", "context_subtype", "route_subtype", "basis", "price", "stop", "cap"])
def test_public_recomputation_denies_rehashed_new_route_tampering(early_report, damage):
    report = deepcopy(early_report); d = report["diagnostics"]
    if damage == "score": report["envelope"]["score"]["final_score"] += 1.
    elif damage == "source_floor": d["structure_plan"]["anchors"]["support"]["value"] += .1
    elif damage == "context_subtype": d["rule_baseline"]["selection"]["setup_variant"] = PRICE_DISCOVERY
    elif damage == "route_subtype":
        route = d["reference_entry_route"]; route["setup_variant"] = PRICE_DISCOVERY
        route["route_id"] = identity({key: value for key, value in route.items() if key != "route_id"})
    elif damage == "basis": d["reference_plan"]["entry_basis"] = "CUTOFF_CLOSE"
    elif damage == "price": d["reference_plan"]["signal_close"] = d["reference_plan"]["entry"]
    elif damage == "stop": d["reference_plan"]["stop"] += .1
    else: d["entry_admission"]["small_total_cap"] += .01
    with pytest.raises(ValueError): validate_report(report)
    with pytest.raises(ValueError): authorize_candidate(report, report["pae_authorization"]["account"])


def test_zero_cash_and_existing_exposure_reduce_quota_not_score(early_report):
    report = deepcopy(early_report); original = deepcopy(report)
    account = dict(report["pae_authorization"]["account"], value=1_000_000., available_cash=1_000_000.,
                   intended_weight=1., capacity_weight=1., positions=[])
    allowed = authorize_candidate(report, account)
    assert allowed["authorized"] and allowed["maximum_total_weight"] == .005
    assert allowed["small_risk_fraction"] == .0005
    assert allowed["indicative_shares"] * allowed["bound_plan_loss_per_share"] <= 500.
    empty = authorize_candidate(report, dict(account, available_cash=0.))
    assert not empty["authorized"] and empty["indicative_shares"] == 0
    held = int(np.ceil(.005 * account["value"] / report["diagnostics"]["reference_plan"]["signal_close"]))
    occupied = authorize_candidate(report, dict(account, positions=[dict(symbol="2330", shares=held)]))
    assert not occupied["authorized"] and occupied["incremental_weight_capacity"] == 0.
    assert report == original


def test_account_cash_does_not_reenter_native_les_calculation(early_report):
    value = market()
    no_cash = native(replace(value, les_v6_portfolio=dict(value.les_v6_portfolio, available_cash=0.)))
    assert no_cash["envelope"] == early_report["envelope"]
    assert no_cash["diagnostics"] == early_report["diagnostics"]
    assert no_cash["pae_authorization"]["indicative_shares"] == 0


def test_existing_holding_consumes_incremental_plan_loss_once(early_report):
    account = dict(early_report["pae_authorization"]["account"],
        value=1_000_000., available_cash=1_000_000., intended_weight=1., capacity_weight=1., positions=[])
    empty = authorize_candidate(early_report, account)
    held_count = 10
    held = authorize_candidate(early_report, dict(account, positions=[dict(symbol="2330", shares=held_count)]))
    number = lambda value: Decimal(str(value))
    loss = number(empty["bound_plan_loss_per_share"])
    risk_quota = int(((Decimal("500") - held_count * loss) / loss).to_integral_value(rounding=ROUND_FLOOR))
    assert held["indicative_shares"] <= risk_quota
    assert (held["indicative_shares"] + held_count) * loss <= Decimal("500")
    assert held["current_weight"] == pytest.approx(held_count * 82.1 / 1_000_000.)


def test_risk_quota_subtracts_held_risk_when_it_binds_before_weight(early_report):
    """Already-validated arithmetic boundary double, not a native signal.

    The altered plan is explicitly rejected by the public validator. Only the
    private quota primitive is isolated here to make the loss budget bind.
    """
    from domain.les_v7.allocation import _authorize_validated
    report = deepcopy(early_report)
    report["diagnostics"]["reference_plan"]["stop"] = 50.
    account = dict(report["pae_authorization"]["account"],
        value=1_000_000., available_cash=1_000_000., intended_weight=1., capacity_weight=1., positions=[])
    with pytest.raises(ValueError): authorize_candidate(report, account)
    empty = _authorize_validated(report, account)
    held = _authorize_validated(report, dict(account, positions=[dict(symbol="2330", shares=5)]))
    loss = Decimal(str(empty["bound_plan_loss_per_share"]))
    assert empty["indicative_shares"] == int((Decimal("500") / loss).to_integral_value(rounding=ROUND_FLOOR))
    assert held["indicative_shares"] == empty["indicative_shares"] - 5
    assert (held["indicative_shares"] + 5) * loss <= Decimal("500")


def test_public_stale_downgrade_removes_conditional_route(early_report):
    report = deepcopy(early_report); account = report.pop("pae_authorization")["account"]
    final = finalize_daily_report(report, account=account, daily_input_errors=["STALE_MARKET_DATA"])
    validate_report(final)
    assert final["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert final["envelope"]["score"]["final_score"] is None
    assert "reference_entry_route" not in final["diagnostics"]
    assert final["pae_authorization"]["authorized"] is False


def reference_execution(report, *, opening=None):
    """Explicit synthetic H20 observations: reference-only, not real fills."""
    from domain.les_v7.plan_r16 import plan_from_record
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


def test_native_conditional_report_only_fills_next_observed_open_in_band(early_report):
    from domain.les_v7.reference_path import _tick
    result = reference_execution(early_report)
    assert result["status"] == "VERIFIED_FILLED", result
    execution = result["execution"]
    assert execution["entry"] == early_report["diagnostics"]["reference_plan"]["band"][0]
    assert execution["entry_risk_check"]["state"] == "PASS"
    assert execution["conditional_entry"]["signal_close"] == 82.1
    assert execution["conditional_entry"]["state"] == "REFERENCE_FILLED_AT_ELIGIBLE_OPEN"
    assert execution["approved_for_live"] is False
    high = Decimal(str(early_report["diagnostics"]["reference_plan"]["band"][1]))
    outside = reference_execution(early_report, opening=float(high + _tick(high)))
    assert outside["status"] == "VERIFIED_NO_FILL"
    assert outside["execution"]["filled"] is False
    assert outside["execution"]["conditional_entry"]["state"] == "EXPIRED_AT_NEXT_SESSION_CLOSE"


def test_native_projected_reference_target_is_never_labeled_observed_resistance(discovery_report):
    result = reference_execution(discovery_report)
    assert result["status"] == "VERIFIED_FILLED", result
    execution = result["execution"]
    assert execution["native_plan_target"] == discovery_report["diagnostics"]["reference_plan"]["target"]
    assert execution["target_method"] == "MODEL_PROJECTED_TARGET"
    assert execution["native_resistance_target"] is None
    assert execution["entry_risk_check"]["state"] == "PASS"


def test_clean_renderer_distinguishes_unfilled_conditional_signal():
    from infrastructure.report.low_entry_v7_renderer import LowEntryV7Renderer
    from unittest.mock import patch
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
        decisions = AnalysisDatasetBuilder().build_decisions(market(), [[1, 1, 1, 1, .01]])
    before = decisions[0].value("LES v7 Report")
    html = LowEntryV7Renderer().render(decisions * 2)
    assert html.count('data-entry-policy="R16"') == 1
    assert "條件式／尚未成交" in html
    assert "僅下一官方交易日開盤入帶有效；收盤到期" in html
    assert decisions[0].value("LES v7 Report") == before
