"""Independent ELEC research scoring, PIT cycles and daily shadow evidence."""
from copy import deepcopy
from dataclasses import replace
import json

import pandas as pd
import pytest

from domain.elec_quality_metrics import ELEC_CYCLE_METRICS
from domain.fundamentals import GateStatus, _score_quality_metrics, evaluate_quality_gate
from domain.industry_profiles import (
    ELEC_PROFILE, ELEC_QUALITY_PROFILE_VERSION, ELECTRONICS_PROFILE,
    IndustryProfileMode, IndustryProfileRegistry, SEMI_PROFILE,
)
from domain.strategies.low_entry_score_v4 import attach_low_entry_v4_context
from domain.strategies.low_entry_score_v4_1 import LowEntryScoreV41Strategy
from infrastructure.report.html_renderer import RendererFactory
from tests.strategies.test_daily_v6_pipeline import build, daily_market
from tests.strategies.test_low_entry_score_v4 import (
    OFFICIAL_FILING_TIMESTAMP, buy_candidate_history, fundamental_snapshot,
)


CUTOFF = "2026-07-21T05:30:00Z"


def annual_history() -> pd.DataFrame:
    return pd.DataFrame({
        "fiscal_year": [2020, 2021, 2022, 2023, 2024],
        "available_at": [f"{year}-03-01T00:00:00Z" for year in range(2021, 2026)],
        "availability_source": [OFFICIAL_FILING_TIMESTAMP] * 5,
        "revenue": [100., 110., 121., 133.1, 146.41],
        "gross_margin": [.20, .22, .24, .23, .22],
        "operating_margin": [.09, .11, .13, .15, .17],
        "roic": [.08, .10, .12, .14, .16],
        "operating_cash_flow": [15., 16., 17., 18., 19.],
        "capex": [5.] * 5,
        "net_income": [10., 11., 12., 13., 14.],
        "diluted_shares": [10.] * 5,
    })


def snapshot(**overrides):
    return fundamental_snapshot(**dict({"sector": "13", "gross_margin_ttm": .23}, **overrides))


def evaluate(annual=None, snap=None):
    return evaluate_quality_gate(
        annual_history() if annual is None else annual,
        snapshot() if snap is None else snap, CUTOFF,
    )


def test_elec_rules_are_independent_and_do_not_change_subindustries():
    assert IndustryProfileRegistry.resolve("13") is ELEC_PROFILE
    assert IndustryProfileRegistry.resolve("ELEC") is ELEC_PROFILE
    assert IndustryProfileRegistry.resolve("24") is SEMI_PROFILE
    assert ELEC_PROFILE.metric_rules != ELECTRONICS_PROFILE.metric_rules
    assert SEMI_PROFILE.metric_rules == ELECTRONICS_PROFILE.metric_rules
    assert ELEC_PROFILE.version == ELEC_QUALITY_PROFILE_VERSION
    assert ELEC_PROFILE.mode is IndustryProfileMode.ACTIVE
    assert ELEC_PROFILE.allow_buy and ELEC_PROFILE.allow_buy_small
    assert {category: sum(r.maximum for r in ELEC_PROFILE.metric_rules if r.category == category)
            for category in "ABCDE"} == {"A": 25, "B": 25, "C": 25, "D": 15, "E": 10}


def test_cycle_formulas_are_reproducible_and_use_five_observations_four_intervals():
    annual, snap = annual_history(), snapshot()
    before_annual, before_snap = annual.copy(deep=True), deepcopy(snap)
    result = evaluate(annual, snap)
    expected = {
        "elec_roic_5y_median": .12, "elec_positive_cfo_year_ratio": 1.,
        "elec_positive_fcf_year_ratio": 1., "elec_capex_to_cfo_5y": 25 / 85,
        "elec_gross_margin_drawdown_5y": -.01, "elec_operating_margin_trend_3y": .02,
        "elec_revenue_cagr_5y": .1,
    }
    for name, value in expected.items():
        assert result.metrics[name] == pytest.approx(value)
    assert result.score == 100 and result.available_metric_weight == 100
    assert result.status is GateStatus.PASS and result.industry_profile_mode == "ACTIVE"
    assert result == evaluate(annual, snap)
    pd.testing.assert_frame_equal(annual, before_annual)
    assert snap == before_snap


@pytest.mark.parametrize("name,value,points", [
    ("elec_roic_5y_median", .12, 10), ("elec_roic_5y_median", .08, 7),
    ("elec_roic_5y_median", .05, 3), ("elec_roic_5y_median", -.01, 0),
    ("elec_positive_cfo_year_ratio", 1., 7), ("elec_positive_cfo_year_ratio", .8, 5),
    ("elec_positive_cfo_year_ratio", .6, 2), ("elec_positive_cfo_year_ratio", .4, 0),
    ("elec_positive_fcf_year_ratio", .8, 5), ("elec_positive_fcf_year_ratio", .6, 3),
    ("elec_positive_fcf_year_ratio", .4, 1), ("elec_positive_fcf_year_ratio", .2, 0),
    ("elec_capex_to_cfo_5y", .6, 5), ("elec_capex_to_cfo_5y", 1., 3),
    ("elec_capex_to_cfo_5y", 1.2, 1), ("elec_capex_to_cfo_5y", 1.21, 0),
    ("elec_gross_margin_drawdown_5y", -.02, 5), ("elec_gross_margin_drawdown_5y", -.05, 3),
    ("elec_gross_margin_drawdown_5y", -.08, 1), ("elec_gross_margin_drawdown_5y", -.081, 0),
    ("elec_operating_margin_trend_3y", .01, 5), ("elec_operating_margin_trend_3y", 0., 3),
    ("elec_operating_margin_trend_3y", -.01, 1), ("elec_operating_margin_trend_3y", -.011, 0),
    ("elec_revenue_cagr_5y", .08, 5), ("elec_revenue_cagr_5y", 0., 3),
    ("elec_revenue_cagr_5y", -.05, 1), ("elec_revenue_cagr_5y", -.051, 0),
    ("debt_ratio", .55, 3), ("debt_ratio", .551, 0),
])
def test_profile_scoring_thresholds(name, value, points):
    rule = next(r for r in _score_quality_metrics({name: value}, ELEC_PROFILE) if r.name == name)
    assert rule.available and rule.earned == points


@pytest.mark.parametrize("case", ["short", "gap", "untrusted", "fractional_year"])
def test_incomplete_five_year_windows_never_backfill(case):
    annual = annual_history()
    if case == "short":
        annual = annual.iloc[1:]
    elif case == "gap":
        annual.loc[0, "fiscal_year"] = 2019
    elif case == "untrusted":
        annual.loc[0, "availability_source"] = "INFERRED"
    else:
        annual["fiscal_year"] += .5
    result = evaluate(annual)
    for name in ELEC_CYCLE_METRICS:
        if name != "elec_operating_margin_trend_3y":
            assert result.metrics[name] is None
    assert result.available_metric_weight < 100
    assert any(flag.startswith("ELEC_CYCLE_MISSING=") for flag in result.qa_flags)


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), "--"])
def test_missing_roic_reduces_only_its_weight_not_its_value_to_zero(invalid):
    annual = annual_history().astype({"roic": object})
    annual.loc[4, "roic"] = invalid
    result = evaluate(annual)
    assert result.metrics["elec_roic_5y_median"] is None
    assert result.available_metric_weight == 90
    assert not next(r for r in result.metric_scores if r.name == "elec_roic_5y_median").available
    assert not result.hard_red_flags


def test_derived_decimal_thresholds_do_not_drop_a_tier_due_to_float_noise():
    annual = annual_history()
    annual["gross_margin"] = [.32] * 5
    annual["operating_margin"] = [.02, .02, .02, .03, .04]
    result = evaluate(annual, snapshot(gross_margin_ttm=.30))
    assert result.metrics["elec_gross_margin_drawdown_5y"] == -.02
    assert result.metrics["elec_operating_margin_trend_3y"] == .01
    scored = {metric.name: metric.earned for metric in result.metric_scores}
    assert scored["elec_gross_margin_drawdown_5y"] == 5
    assert scored["elec_operating_margin_trend_3y"] == 5


@pytest.mark.parametrize("flows", [[0.] * 5, [-1.] * 5, [1., 1., 1., 1., -4. + 1e-13]])
def test_zero_negative_or_near_zero_cumulative_cfo_is_not_a_valid_capex_ratio(flows):
    annual = annual_history()
    annual["operating_cash_flow"] = flows
    assert evaluate(annual).metrics["elec_capex_to_cfo_5y"] is None


def test_negative_capex_is_not_silently_converted_to_investment_outflow():
    annual = annual_history()
    annual.loc[2, "capex"] = -5
    metrics = evaluate(annual).metrics
    assert metrics["elec_positive_fcf_year_ratio"] is None
    assert metrics["elec_capex_to_cfo_5y"] is None
    assert metrics["elec_positive_cfo_year_ratio"] == 1


def test_future_revision_and_snapshot_never_enter_cycle_metrics():
    annual = annual_history()
    revision = annual.iloc[[-1]].copy()
    revision["available_at"] = "2027-03-01T00:00:00Z"
    revision["roic"] = -1.
    future = evaluate(pd.concat([annual, revision]))
    assert future.metrics == evaluate().metrics
    revision["available_at"] = "2026-03-01T00:00:00Z"
    current = evaluate(pd.concat([annual, revision]))
    assert current.metrics["elec_roic_5y_median"] == .10
    for changed in (snapshot(available_at="2027-01-01T00:00:00Z"), snapshot(availability_source="INFERRED")):
        result = evaluate(snap=changed)
        assert result.metrics["elec_gross_margin_drawdown_5y"] is None
        assert result.status is GateStatus.UNKNOWN


def test_peak_reversal_and_structural_deterioration_score_below_resilient_cycle():
    stable = evaluate()
    annual = annual_history()
    annual["operating_margin"] = [.17, .17, .16, .10, .04]
    annual["operating_cash_flow"] = [15., -16., -17., -18., -19.]
    annual["roic"] = [-.02, -.01, 0., .01, .02]
    weak = evaluate(annual, snapshot(gross_margin_ttm=.10, debt_ratio=.7))
    assert weak.score < stable.score
    assert weak.metrics["elec_operating_margin_trend_3y"] < 0
    assert weak.metrics["elec_positive_cfo_year_ratio"] == .2
    assert "NON_POSITIVE_THREE_YEAR_CFO" in weak.qa_flags
    debt = evaluate(snap=snapshot(cash=10, total_debt=100, ebitda_ttm=-1))
    assert "NET_DEBT_TO_EBITDA_NOT_MEANINGFUL" in debt.qa_flags
    rejected = evaluate(snap=snapshot(going_concern=True))
    assert rejected.status is GateStatus.REJECT


def test_v41_active_elec_keeps_native_entry_checks():
    data = attach_low_entry_v4_context(
        buy_candidate_history(), annual_fundamentals=annual_history(),
        fundamental_snapshot=snapshot(), signal_date=CUTOFF,
    )
    row = LowEntryScoreV41Strategy().run(data).iloc[-1]
    assert row["Industry Profile Version"] == ELEC_QUALITY_PROFILE_VERSION
    assert row["低點決策"] == "BUY" and row["建議股數"] > 0


def test_daily_elec_active_evidence_is_rendered_without_overwriting_v6_quality():
    market = daily_market()
    snap = snapshot()
    snap["les_v6_filings"] = market.fundamental_snapshots["2330"]["les_v6_filings"]
    market = replace(market, annual_fundamentals={"2330": annual_history()},
                     fundamental_snapshots={"2330": snap})
    _, decisions = build(market)
    decision = decisions[0]
    report = json.loads(decision.value("LES v6 Report"))
    industry = report["industry_profile"]["quality_evaluation"]
    assert industry["score"] == 100 and industry["available_weight"] == 100
    assert industry["metrics"]["elec_roic_5y_median"]["value"] == .12
    assert report["quality"]["score"] == decision.value("Quality Score")
    assert report["decision"]["signal"] == "BUY_SMALL" and report["sizing"]["indicative_shares"] > 0
    html = RendererFactory.get_renderer("daily_strategy_email").render(decisions)
    assert "產業 QS 100.00" in html and ELEC_QUALITY_PROFILE_VERSION in html
    assert "ACTIVE" in html and "進場已啟用" in html and "SHADOW" not in html
    _, missing = build(replace(market, annual_fundamentals={}))
    missing_report = json.loads(missing[0].value("LES v6 Report"))
    assert missing_report["industry_profile"]["quality_evaluation"]["available_weight"] < 100
    assert missing_report["quality"] == report["quality"]
    assert "缺失指標" in RendererFactory.get_renderer("daily_strategy_email").render(missing)
