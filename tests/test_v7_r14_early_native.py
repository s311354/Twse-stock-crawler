"""SYNTHETIC ONLY: complete EARLY native contracts, never official BUY evidence.

All market paths and source documents come from explicit test fixtures. The
normal daily adapter, source producers, score ledger, validator and account
authorization are exercised without changing a score, gate or decision.
"""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from application.industry_benchmark import prepare_industry_benchmark
from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.allocation import authorize_candidate
from domain.les_v7.context_r14 import VARIANT
from domain.les_v7.contracts import ReferencePlan
from domain.les_v7.plan_r10 import evaluate_entry, validate_band
from domain.les_v7.reference_path import COORDINATE_POLICY, _tick, evaluate_reference_path
from domain.les_v7.report import validate_report
from infrastructure.crawler.security_master import select_security_master
from tests.strategies.test_v7_context_r14 import CUTOFF, source_bars
from tests.test_industry_benchmark import packet as industry_packet
from tests.test_security_master import packet as security_packet
from tests.test_v7_r14_pipeline import market as existing_market


def early_market(*, bars=None):
    """Rebind all explicitly synthetic evidence to one fixed completed session."""
    original = existing_market()
    data = source_bars() if bars is None else bars.copy(deep=True)
    # A genuine earlier completed pivot supplies the unchanged scoring/Veto
    # features. It is not the new five-day plan anchor, which remains 80.
    data.loc[len(data) - 40, "Low"] = 78.
    dates = data["Date"].tolist()
    # The benchmark declines while this synthetic stock holds its five-day
    # base. These are input observations, not fabricated RS or score values.
    benchmark = np.linspace(1300., 1000., len(data)).tolist()
    sources = deepcopy(original.native_source_selections)
    sources["2330"] = dict(security=select_security_master(security_packet(), "2330", CUTOFF),
                           official_pe=[])
    restriction = deepcopy(original.trading_restrictions["2330"])
    restriction.update(session_date=dates[-1], observation_date=dates[-1], available_at=CUTOFF)
    raw_index = industry_packet(pd.to_datetime(dates), np.linspace(120., 100., len(data)))
    for document in raw_index["documents"].values():
        document["retrieved_at"] = CUTOFF
    for row in [*raw_index["memberships"], *raw_index["bars"]]:
        row["publication_proof"]["verified_at"] = CUTOFF
        row["publication_proof"]["document"]["retrieved_at"] = CUTOFF
    scaled = lambda name: [(data[name].to_numpy(dtype=float) * 1000.).tolist()]
    return replace(original, dates=[dates], opens=scaled("Open"), highs=scaled("High"),
        lows=scaled("Low"), closes=scaled("Close"), bids=scaled("Bid"), asks=scaled("Ask"),
        volumes=[data["Volume"].tolist()], trade_counts=[data["Trades"].tolist()],
        trade_values=[data["Amount"].tolist()], pe_ratios=[[20.] * len(data)],
        taiex_closes=[benchmark], signal_dates={"2330": CUTOFF}, fallback_date=dates[-1],
        market_observed_at=CUTOFF,
        market_breadth_snapshots={dates[-1]: dict(source_final=True, as_of=dates[-1])},
        trading_restrictions={"2330": restriction}, native_source_selections=sources,
        industry_benchmarks={"24": prepare_industry_benchmark(raw_index)},
        les_v6_portfolio=dict(original.les_v6_portfolio, as_of=CUTOFF))


def native(value=None):
    # This retained regression owns R14, even though its synthetic bar date is
    # after later releases. Do not make the formal daily vintage go backwards.
    from unittest.mock import patch
    from domain.les_v7.policy_r14 import BASELINE_VERSION
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=BASELINE_VERSION):
        selected = capture_comparison_contexts(value or early_market())
        return daily_report(MarketHistoryAssembler().build(selected, 0))


@pytest.fixture(scope="module")
def early_report():
    return native()


def test_complete_native_early_source_plan_report_and_account_contract():
    value = early_market()
    before = deepcopy(value)
    report = native(value)
    validate_report(report)
    d, e = report["diagnostics"], report["envelope"]
    selection = d["rule_baseline"]["selection"]
    assert selection["setup_variant"] == VARIANT
    assert selection["short_base_audit"]["original_selection"]["selected_setup"] == "NO_CLEAR_CONTEXT"
    assert e["event"]["context"] == "LEADERSHIP_CONTINUATION"
    assert d["structure_plan"]["setup_variant"] == VARIANT
    assert d["structure_plan"]["anchors"]["support"]["value"] == 80.
    assert d["starter"] is True and d["confirmed"] is False
    assert d["entry_admission"]["restricted_starter"] is True
    assert d["entry_admission"]["buy_allowed"] is False
    plan = ReferencePlan(**dict(d["reference_plan"], band=tuple(d["reference_plan"]["band"])))
    bound = d["rule_baseline"]["policy"]["reference_policy"]
    assert evaluate_entry(plan, plan.entry, bound)["state"] == "PASS"
    assert validate_band(plan, bound, starter=True)["state"] == "PASS"
    assert report["pae_authorization"] == authorize_candidate(report, report["pae_authorization"]["account"])
    assert report["pae_authorization"]["live_order_authorized"] is False
    assert value == before
    assert e["strategy"]["strategy_decision"] == "BUY_SMALL", (e["score"], e["strategy"], d["entry_admission"])
    assert report["pae_authorization"]["authorized"] is True


@pytest.mark.parametrize("missing", ["restriction", "identity", "market", "financials"])
def test_complete_short_pattern_does_not_override_unknown_required_sources(missing):
    value = early_market()
    if missing == "restriction":
        value = replace(value, trading_restrictions={})
    elif missing == "identity":
        selections = deepcopy(value.native_source_selections)
        selections["2330"]["security"] = None
        value = replace(value, native_source_selections=selections)
    elif missing == "market":
        value = replace(value, market_observed_at=None)
    else:
        snapshots = deepcopy(value.fundamental_snapshots)
        snapshots["2330"].update(les_v6_filings=[], les_v6_mops_filings=[], les_v7_context={})
        value = replace(value, fundamental_snapshots=snapshots)
    report = native(value)
    validate_report(report)
    d = report["diagnostics"]
    # Price-only recognition can remain valid, but cannot certify absent
    # identity, restrictions, final source capture or financial evidence.
    assert d["rule_baseline"]["selection"]["setup_variant"] == VARIANT
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert d["entry_admission"]["state"] in {"UNKNOWN", "BLOCKED"}
    assert not report["pae_authorization"]["authorized"]
    assert not authorize_candidate(report, report["pae_authorization"]["account"])["authorized"]
    states = {row["gate_id"]: row["state"] for row in d["gate_receipts"]}
    if missing == "restriction": assert states["TRADING_STATUS"] == "UNKNOWN"
    if missing in {"identity", "market"}: assert states["DATA_INTEGRITY"] == "UNKNOWN"
    if missing == "financials": assert d["quality"]["status"] in {"UNKNOWN", "DATA_INSUFFICIENT"}


def test_known_insufficient_rr_is_not_replaced_by_an_older_favorable_target():
    data = source_bars()
    data.loc[10, "High"] = 85.  # The same genuine overhead pivot, less upside.
    report = native(early_market(bars=data))
    validate_report(report)
    d = report["diagnostics"]
    assert d["rule_baseline"]["selection"]["setup_variant"] == VARIANT
    assert d["reference_plan"]["target"] == 85.
    assert d["structure_plan"]["entry_band_selection"]["state"] == "FAIL"
    states = {row["gate_id"]: row["state"] for row in d["gate_receipts"]}
    assert states["STRESS_RISK"] == "FAIL"
    assert report["envelope"]["strategy"]["strategy_decision"] == "NO_BUY"
    assert not report["pae_authorization"]["authorized"]


def test_early_total_cap_and_existing_exposure_change_quota_not_les_score(early_report):
    report = deepcopy(early_report)
    before = deepcopy(report)
    d = report["diagnostics"]
    admission = d["entry_admission"]
    expected_cap = min(admission["quality"]["small_cap"], .025) * admission["valuation"]["multiplier"]
    assert admission["small_total_cap"] == pytest.approx(expected_cap)
    assert admission["buy_total_cap"] == 0.
    account = dict(report["pae_authorization"]["account"], value=1_000_000., available_cash=1_000_000.,
        intended_weight=1., capacity_weight=1., positions=[])
    permitted = authorize_candidate(report, account)
    assert permitted["authorized"] and permitted["indicative_shares"] > 0
    assert permitted["maximum_total_weight"] == pytest.approx(expected_cap)
    assert permitted["small_risk_fraction"] == .00125
    assert permitted["position_fraction"] <= expected_cap
    assert permitted["indicative_shares"] * permitted["bound_plan_loss_per_share"] <= 1250.
    empty = authorize_candidate(report, dict(account, available_cash=0.))
    assert not empty["authorized"] and empty["indicative_shares"] == 0
    # Existing shares consume the total allowance, not a separate new 2.5%.
    close = d["reference_plan"]["entry"]
    occupied = int(np.ceil(expected_cap * account["value"] / close))
    held = authorize_candidate(report, dict(account, positions=[dict(symbol="2330", shares=occupied)]))
    assert not held["authorized"] and held["indicative_shares"] == 0
    assert held["incremental_weight_capacity"] == 0.
    assert report == before


def reference_execution(report, *, outside=False):
    """Explicit test-only future path; never an official calendar/fill claim."""
    d = report["diagnostics"]
    plan = ReferencePlan(**dict(d["reference_plan"], band=tuple(d["reference_plan"]["band"])))
    policy = d["rule_baseline"]["policy"]["reference_policy"]["cost_path_policy"]
    # A fixed synthetic weekday sequence is valid as engineering evidence only.
    sessions = pd.bdate_range(start=report["signal_date"], periods=21).strftime("%Y-%m-%d").tolist()
    coordinate = "SYNTHETIC_TEST_ONLY:EARLY_R14_RAW_TWD:2330"
    price = plan.entry
    offset = lambda value, increment: float(Decimal(str(value)) + Decimal(increment))
    window = [dict(session=day, open=price, high=offset(price, ".1"), low=offset(price, "-.1"), close=price,
        volume=1_000_000, complete=True, available_at=day + "T13:30:00+08:00",
        coordinate_id=coordinate, evidence_ids=["SYNTHETIC_TEST_ONLY:EARLY_BAR:" + day],
        corporate_action_state="NO_ACTION") for day in sessions]
    if outside:
        upper = Decimal(str(plan.band[1]))
        wrong = float(upper + _tick(upper))
        window[1].update(open=wrong, high=offset(wrong, ".1"), low=offset(wrong, "-.1"), close=wrong)
    coverage = dict(schema="les-v7-coordinate-coverage/v1", policy=COORDINATE_POLICY,
        coordinate_id=coordinate, market="TWSE", start_session=sessions[0], end_session=sessions[-1],
        status="VERIFIED_NO_ACTIONS", evidence_ids=["SYNTHETIC_TEST_ONLY:NO_ACTION_COVERAGE"],
        available_at=sessions[-1] + "T13:30:00+08:00")
    return evaluate_reference_path(plan, window, policy, expected_sessions=sessions,
        cutoff=CUTOFF, coordinate_id=coordinate, source_evidence_ids=["SYNTHETIC_TEST_ONLY:EARLY_PATH"],
        corporate_action_coverage=coverage)


def test_authentic_native_early_plan_reprices_next_open_and_never_fills_outside_band(early_report):
    saved = deepcopy(early_report)
    inside = reference_execution(early_report)
    assert inside["status"] == "VERIFIED_FILLED", inside
    assert inside["execution"]["entry_risk_check"]["state"] == "PASS"
    assert inside["execution"]["initial_stop"] == early_report["diagnostics"]["reference_plan"]["stop"]
    assert inside["execution"]["native_resistance_target"] == early_report["diagnostics"]["reference_plan"]["target"]
    assert inside["execution"]["historical_deployment_claim"] is False
    outside = reference_execution(early_report, outside=True)
    assert outside["status"] == "VERIFIED_NO_FILL"
    assert outside["execution"]["no_fill_reason"] == "NEXT_OPEN_OUTSIDE_NATIVE_BAND"
    assert early_report == saved


@pytest.mark.parametrize("damage", ["score", "cap", "floor", "band", "subtype"])
def test_early_native_report_cannot_authorize_tampered_business_fields(early_report, damage):
    report = deepcopy(early_report)
    d = report["diagnostics"]
    if damage == "score": report["envelope"]["score"]["final_score"] += 1.
    elif damage == "cap": d["entry_admission"]["small_total_cap"] += .01
    elif damage == "floor": d["structure_plan"]["anchors"]["support"]["value"] += .1
    elif damage == "band": d["reference_plan"]["band"][1] += .1
    else: d["rule_baseline"]["selection"]["setup_variant"] = None
    with pytest.raises(ValueError): validate_report(report)
    with pytest.raises(ValueError): authorize_candidate(report, report["pae_authorization"]["account"])

