"""Generic formula regressions; no stock-specific admission or future prices."""
from dataclasses import replace
from math import floor

import numpy as np
import pandas as pd
import pytest

from domain.features import SharedFeaturePipeline
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.strategies.low_entry_v6_features import build_features
from domain.strategies.low_entry_v6_filings import ixbrl_record
from domain.strategies.low_entry_v6_payoff import payoff_scenarios
from domain.strategies.low_entry_v6_quality import evaluate_quality, margin_drawdown, normalize_quarters, ratio
from domain.strategies.low_entry_v6_signals import size_position
from domain.strategies.low_entry_score_v6 import valuation_evidence
from tests.strategies.test_low_entry_score_v6 import CONFIG, bars, financials


def evaluate(records):
    return evaluate_quality(records, symbol="2330", market="TWSE", as_of="2026-09-18")


@pytest.mark.parametrize("values,expected", [
    ([.05, .10, .15, .20], 0.),
    ([.10, .10, .10, .10], 0.),
    ([.10, .25, .18, .20], .07),
    ([.10, .12, .07, .09], .05),
    ([-.20, -.10, -.05, .02], 0.),
    ([.10, None, .12, .14], None),
    ([.10, float("inf"), .12, .14], None),
])
def test_margin_drawdown_is_chronological_and_missing_is_not_stable(values, expected):
    result = margin_drawdown(values)
    assert result == pytest.approx(expected) if expected is not None else result is None


def test_improving_margins_earn_stability_without_removing_other_quality_checks():
    improving, falling = financials(), financials()
    for records, margins in ((improving, [.05, .10, .15, .20]),
                             (falling, [.05, .30, .15, .20])):
        for row, margin in zip(records[-4:], margins):
            row["cost_of_revenue"] = row["revenue"] * (1 - margin)
    good, bad = evaluate(improving), evaluate(falling)
    assert good.metrics["gross_margin_drawdown_4q"] == 0
    assert bad.metrics["gross_margin_drawdown_4q"] == pytest.approx(.15)
    assert good.components["profitability"] > bad.components["profitability"]
    improving[-1]["auditor_opinion"] = "ADVERSE"
    assert evaluate(improving).status == "REJECT"


def test_missing_margin_reduces_coverage_not_zero_drawdown():
    records = financials()
    before = evaluate(records)
    records[-2]["cost_of_revenue"] = None
    after = evaluate(records)
    assert after.metrics["gross_margin_drawdown_4q"] is None
    assert after.coverage["profitability"] < before.coverage["profitability"]


def test_ytd_eps_is_not_additive_when_weighted_shares_change():
    # Q1 income100/shares100 =>1.0; Q2 income100/shares400 =>.25;
    # H1 income200/weighted shares250 =>.8, NOT Q2 EPS=-.2.
    periods = {
        pd.Period("2026Q1"): dict(flow_basis="YTD", net_income=100., basic_eps=1.),
        pd.Period("2026Q2"): dict(flow_basis="YTD", net_income=200., basic_eps=.8,
                                  basic_eps_quarter=.25),
    }
    q = normalize_quarters(periods)
    assert q[pd.Period("2026Q2")]["net_income"] == 100
    assert q[pd.Period("2026Q2")]["basic_eps"] == .25
    del periods[pd.Period("2026Q2")]["basic_eps_quarter"]
    assert normalize_quarters(periods)[pd.Period("2026Q2")]["basic_eps"] is None


def test_eps_missing_never_invents_positive_quarters_or_ttm():
    records = financials()
    records[-1].update(flow_basis="YTD", basic_eps=8., basic_eps_quarter=None)
    result = evaluate(records)
    assert result.metrics["positive_eps_quarters"] is None
    assert result.metrics["basic_eps_ttm"] is None
    assert result.metrics["basic_eps_ttm_basis"] == "UNAVAILABLE"
    assert "QUARTER_EPS_EVIDENCE_INCOMPLETE" in result.warnings


def test_reported_full_year_eps_is_used_without_reconstructing_q4():
    records = financials()[:-2]
    records[-1].update(flow_basis="YTD", basic_eps=8.)
    result = evaluate(records)
    assert result.metrics["basic_eps_ttm"] == 8.
    assert result.metrics["basic_eps_ttm_basis"] == "REPORTED_FULL_YEAR"
    assert result.metrics["positive_eps_quarters"] is None


def test_ixbrl_adapter_separates_quarter_and_ytd_eps_contexts():
    fact = FiledFact("ifrs-full:BasicEarningsLossPerShare", .8, None,
                     "2026-01-01", "2026-06-30", None, unit="iso4217:TWD/xbrli:shares")
    quarter = replace(fact, start_date="2026-04-01", numeric_value=.25)
    # Comparative and dimensioned contexts cannot masquerade as current EPS.
    other = (replace(quarter, end_date="2025-06-30", numeric_value=999.),
             replace(quarter, dimensions=(("segment", "x"),), numeric_value=999.))
    filing = FiledFinancials(2026, 2, "2026-08-10T00:00:00Z", 1, (fact, quarter, *other))
    record = ixbrl_record("2330", filing)
    assert record["basic_eps"] == .8 and record["basic_eps_quarter"] == .25
    conflicting = replace(filing, facts=filing.facts + (replace(quarter, numeric_value=.3),))
    assert ixbrl_record("2330", conflicting)["basic_eps_quarter"] is None


@pytest.mark.parametrize("signal,risk", [("BUY", .0075), ("BUY_SMALL", .0025)])
@pytest.mark.parametrize("target", [120., None])
def test_risk_budget_includes_entry_and_stop_sale_costs(signal, risk, target):
    decision = dict(signal=signal, risk_fraction=risk, position_multiplier=1.,
                    plan=dict(entry_price=100., stop_price=95., target_price=target,
                              net_risk_per_share=.000001))  # cached evidence is untrusted
    shares = size_position(decision, portfolio_value=1_000_000., available_cash=1_000_000.,
                           intended_weight=1., portfolio_capacity_weight=1.,
                           amount_ma20=1_000_000_000., config=CONFIG)
    cost = 100 * (1 + CONFIG.buy_fee_rate)
    loss = cost - 95 * (1 - CONFIG.sell_cost_rate)
    assert shares == floor(1_000_000 * risk / loss)
    assert shares * loss <= 1_000_000 * risk < (shares + 1) * loss
    assert shares < floor(1_000_000 * risk / 5.)
    payoff = payoff_scenarios(decision["plan"], buy_cost_rate=CONFIG.buy_fee_rate,
                             sell_cost_rate=CONFIG.sell_cost_rate)
    assert payoff["net_risk_per_share"] == pytest.approx(loss)
    if target is not None:
        assert payoff["net_reward_risk"] == pytest.approx(payoff["net_reward_per_share"] / loss)


@pytest.mark.parametrize("invalid", [float("inf"), -float("inf"), float("nan"), 0., -1.])
def test_market_readiness_requires_consecutive_finite_positive_benchmark(invalid):
    data = bars(100)
    data.loc[75, "TAIEX Close"] = invalid
    features = build_features(data, SharedFeaturePipeline().build(data))
    assert not features.loc[75:95, "market_ready"].any()
    assert features.loc[96, "market_ready"]
    assert features.loc[75:95, "rs20"].isna().all()
    assert features.loc[75:80, "rs_slope5"].isna().all()
    assert pd.isna(features.loc[75, "rs_line"])


def test_finite_inputs_overflow_cannot_be_a_valid_quality_ratio():
    assert ratio(1e308, 1e-308) is None


@pytest.mark.parametrize("pe,expected", [(30., "EXPENSIVE"), (10., "NOT_AVAILABLE")])
def test_missing_quarter_eps_does_not_erase_expensive_evidence_or_award_cheapness(pe, expected):
    result = valuation_evidence(pe, pd.Series(np.linspace(10., 30., 60)), None)
    assert result["interpretation"] == expected
    assert result["pe"] is None and result["pe_percentile_60"] is None
    assert result["eps_status"] == "UNKNOWN"
    if expected == "EXPENSIVE":
        assert result["expensive_warning_basis"] == "REPORTED_PE_EPS_UNVERIFIED"


@pytest.mark.parametrize("eps", [0., -1.])
def test_nonpositive_eps_cannot_be_cheap_according_to_reported_pe(eps):
    assert valuation_evidence(10., pd.Series(np.linspace(10., 30., 60)), eps)["interpretation"] == "NOT_AVAILABLE"


def test_infinite_pe_history_is_not_a_complete_valuation_window():
    history = pd.Series(np.linspace(10., 30., 60))
    history.iloc[0] = float("inf")
    assert valuation_evidence(30., history, 1.)["interpretation"] == "NOT_AVAILABLE"
