"""Synthetic source-to-account R19 flow; no historical performance claim."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from application.low_entry_v7_daily import daily_report, project_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.allocation import authorize_candidate
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.policy_r19 import BASELINE_VERSION, PLAN_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy
from infrastructure.crawler.security_master import select_security_master
from tests.test_security_master import packet
from tests.test_v7_r16_pipeline import discovery_bars, market, pullback_bars
from tests.test_v7_r18_pipeline import core_market, neutral_market

AT = "2026-10-22T14:00:00Z"


def current_market(value):
    """Explicit synthetic observations after publication, not source repair."""
    sources = deepcopy(value.native_source_selections)
    sources["2330"]["security"] = select_security_master(packet(), "2330", AT)
    restrictions = deepcopy(value.trading_restrictions)
    day = timestamp(AT).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
    restrictions["2330"].update(available_at=AT, observation_date=day,
        session_date=day, scope="CURRENT_AS_OF")
    return replace(value, signal_dates={"2330": AT}, market_observed_at=AT,
        native_source_selections=sources, trading_restrictions=restrictions,
        les_v6_portfolio=dict(value.les_v6_portfolio, as_of=AT))


def native(value, version=BASELINE_VERSION):
    with patch("domain.les_v7.rule_baseline.current_source_baseline_version", return_value=version):
        return daily_report(MarketHistoryAssembler().build(capture_comparison_contexts(value), 0))


@pytest.mark.parametrize("kind", ["neutral", "core", "pullback", "discovery"])
def test_native_source_to_report_projection_and_account_revalidate(kind):
    value = {"neutral": neutral_market, "core": core_market,
        "pullback": lambda: market(data=pullback_bars()),
        "discovery": lambda: market(data=discovery_bars())}[kind]()
    value = current_market(value)
    old = native(value, "V7_RULE_BASED_BASELINE_R18")
    report = native(value)
    validate_report(report)
    d = report["diagnostics"]
    plan, previous = d["reference_plan"], old["diagnostics"]["reference_plan"]
    assert d["rule_baseline"]["policy"]["policy_version"] == BASELINE_VERSION
    assert plan["policy_version"] == PLAN_VERSION
    assert (plan["stop"], plan["target"], plan["signal_close"]) == (
        previous["stop"], previous["target"], previous["signal_close"])
    assert plan["band"][0] == previous["band"][0]
    assert plan["band"][1] >= previous["band"][1]
    assert d["r19_route_evaluation"]["position_limits_affect_score"] is False
    assert d["execution_assessment"]["schema"] == "les-v7-r19-execution-assessment/v1"
    assert d["structure_plan"]["r19_entry_band_extension"]["state"] == "PASS"
    assert not report["pae_authorization"]["live_order_authorized"]
    decision = report["envelope"]["strategy"]["strategy_decision"]
    if kind in {"neutral", "core", "discovery"}:
        assert decision in {"BUY", "BUY_SMALL"}
    else:
        # A higher structural reference entry is honestly re-scored; widening
        # cannot force a marginal old signal to retain its candidate status.
        assert decision == "WATCH" and report["envelope"]["score"]["final_score"] < 60.
        assert report["pae_authorization"]["authorized"] is False
    account = report["pae_authorization"]["account"]
    assert authorize_candidate(report, account) == report["pae_authorization"]
    projected = project_report(report, price_scale=1., include_report=False)
    assert projected["Entry Trigger Price"] == plan["band"][1]
    if report["pae_authorization"]["authorized"]:
        assert report["pae_authorization"]["sizing_entry_limit"] == max(plan["entry"], plan["band"][1])
    assert report["pae_authorization"]["indicative_shares"] <= old["pae_authorization"]["indicative_shares"]


@pytest.mark.parametrize("missing", ["calendar", "restriction", "identity"])
def test_wider_band_cannot_replace_unknown_sources(missing):
    value = current_market(neutral_market())
    if missing == "restriction":
        value = replace(value, trading_restrictions={})
    else:
        sources = deepcopy(value.native_source_selections)
        sources["2330"]["calendar_evidence" if missing == "calendar" else "security"] = None
        value = replace(value, native_source_selections=sources)
    report = native(value)
    validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert report["pae_authorization"]["authorized"] is False


def test_rehashed_band_cannot_pass_native_source_reconstruction():
    report = native(current_market(neutral_market()))
    forged = deepcopy(report)
    forged["diagnostics"]["reference_plan"]["band"][1] += .1
    with pytest.raises(ValueError):
        validate_report(forged)


def test_retained_r18_policy_identity_and_economics_are_unchanged():
    old = baseline_policy(version="V7_RULE_BASED_BASELINE_R18")
    assert identity(old) == "b43f1e233211df05d168bef0cf6fd0abe9f3f6937b6b8e1b3e9537b9fb13efd5"
    new = baseline_policy(version=BASELINE_VERSION)
    for key in ("quality_policy", "valuation_policy", "valuation_input_policy",
                "native_risk_limits", "tactical_risk_limits", "entry_admission_policy"):
        assert old[key] == new[key]
