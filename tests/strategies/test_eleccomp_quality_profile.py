"""Component-specific cycle policy, PIT evidence and daily shadow admission."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pandas as pd
import pytest

from domain.eleccomp_quality_metrics import ELECCOMP_CYCLE_METRICS
from domain.fundamentals import GateStatus, _score_quality_metrics, evaluate_quality_gate
from domain.industry_profiles import (
    ELECCOMP_PROFILE, ELECCOMP_QUALITY_PROFILE_VERSION, ELEC_PROFILE,
    ELECTRONICS_PROFILE, IndustryProfileMode, IndustryProfileRegistry, SEMI_PROFILE,
)
from domain.models import Stocktype
from domain.strategies.low_entry_score_v4 import attach_low_entry_v4_context
from domain.strategies.low_entry_score_v4_1 import LowEntryScoreV41Config, LowEntryScoreV41Strategy
from domain.strategies.low_entry_v6_signals import LowEntryV6Config
from infrastructure.report.html_renderer import RendererFactory
from infrastructure.storage.csv_repository import CsvRepository
from tests.strategies.test_daily_v6_pipeline import build, daily_market
from tests.strategies.test_elec_quality_profile import annual_history as electronics_history
from tests.strategies.test_low_entry_score_v4 import buy_candidate_history, fundamental_snapshot


CUTOFF = "2026-07-21T05:30:00Z"


def annual_history() -> pd.DataFrame:
    return electronics_history().assign(total_assets=100.)


def snapshot(**overrides) -> dict:
    return fundamental_snapshot(**dict({"sector": "28", "gross_margin_ttm": .23}, **overrides))


def evaluate(annual=None, snap=None):
    return evaluate_quality_gate(
        annual_history() if annual is None else annual,
        snapshot() if snap is None else snap, CUTOFF,
    )


def test_component_profile_is_immutable_independent_and_exactly_one_sector():
    for key in (28, "28", "ELECCOMP", Stocktype.ELECCOMP, "Stocktype.ELECCOMP"):
        assert IndustryProfileRegistry.resolve(key) is ELECCOMP_PROFILE
    assert ELECCOMP_PROFILE.sector_codes == frozenset({"28"})
    assert ELECCOMP_PROFILE.version == ELECCOMP_QUALITY_PROFILE_VERSION
    assert ELECCOMP_PROFILE.metric_rules not in (ELEC_PROFILE.metric_rules, ELECTRONICS_PROFILE.metric_rules)
    assert ELECCOMP_PROFILE.calibration_status == "UNCALIBRATED"
    assert ELECCOMP_PROFILE.mode is IndustryProfileMode.ACTIVE
    assert ELECCOMP_PROFILE.allow_buy and ELECCOMP_PROFILE.allow_buy_small
    assert {category: sum(rule.maximum for rule in ELECCOMP_PROFILE.metric_rules if rule.category == category)
            for category in "ABCDE"} == {"A": 25, "B": 25, "C": 25, "D": 15, "E": 10}
    with pytest.raises(FrozenInstanceError):
        ELECCOMP_PROFILE.pass_score = 0
    assert IndustryProfileRegistry.resolve("13") is ELEC_PROFILE
    assert IndustryProfileRegistry.resolve("24") is SEMI_PROFILE


def test_derived_cycle_values_are_deterministic_with_no_input_or_risk_mutation():
    annual, snap = annual_history(), snapshot()
    before, original = annual.copy(deep=True), deepcopy(snap)
    risk41, risk6 = LowEntryScoreV41Config(), LowEntryV6Config()
    result = evaluate(annual, snap)
    expected = {
        "eleccomp_roic_5y_p25": .10, "eleccomp_gross_margin_5y_p25": .22,
        "eleccomp_positive_cfo_year_ratio": 1., "eleccomp_positive_fcf_year_ratio": 1.,
        "eleccomp_capex_to_cfo_5y": 25 / 85,
        "eleccomp_gross_margin_drawdown_5y": -.01,
        "eleccomp_operating_margin_trend_3y": .02,
        "eleccomp_asset_turnover_trend_3y": (1.4641 - 1.21) / 2,
    }
    for name, value in expected.items():
        assert result.metrics[name] == pytest.approx(value)
    assert result.score == 100 and result.available_metric_weight == 100
    assert result.status is GateStatus.PASS and result.industry_profile_mode == "ACTIVE"
    assert evaluate(annual, snap) == result
    pd.testing.assert_frame_equal(annual, before)
    assert snap == original
    assert LowEntryScoreV41Config() == risk41 and LowEntryV6Config() == risk6


@pytest.mark.parametrize("name,value,points", [
    ("eleccomp_roic_5y_p25", .08, 10), ("eleccomp_roic_5y_p25", .05, 6),
    ("eleccomp_roic_5y_p25", .02, 2), ("eleccomp_roic_5y_p25", .019, 0),
    ("eleccomp_gross_margin_5y_p25", .20, 5), ("eleccomp_gross_margin_5y_p25", .15, 3),
    ("eleccomp_gross_margin_5y_p25", .10, 1), ("eleccomp_gross_margin_5y_p25", .099, 0),
    ("eleccomp_positive_cfo_year_ratio", 1., 7), ("eleccomp_positive_cfo_year_ratio", .8, 5),
    ("eleccomp_positive_cfo_year_ratio", .6, 2), ("eleccomp_positive_cfo_year_ratio", .4, 0),
    ("eleccomp_positive_fcf_year_ratio", .8, 5), ("eleccomp_positive_fcf_year_ratio", .6, 3),
    ("eleccomp_positive_fcf_year_ratio", .4, 1), ("eleccomp_positive_fcf_year_ratio", .2, 0),
    ("eleccomp_capex_to_cfo_5y", .8, 5), ("eleccomp_capex_to_cfo_5y", 1., 3),
    ("eleccomp_capex_to_cfo_5y", 1.2, 1), ("eleccomp_capex_to_cfo_5y", 1.21, 0),
    ("eleccomp_gross_margin_drawdown_5y", -.03, 5), ("eleccomp_gross_margin_drawdown_5y", -.06, 3),
    ("eleccomp_gross_margin_drawdown_5y", -.10, 1), ("eleccomp_gross_margin_drawdown_5y", -.101, 0),
    ("eleccomp_operating_margin_trend_3y", .005, 5), ("eleccomp_operating_margin_trend_3y", 0., 3),
    ("eleccomp_operating_margin_trend_3y", -.015, 1), ("eleccomp_operating_margin_trend_3y", -.016, 0),
    ("eleccomp_asset_turnover_trend_3y", .03, 5), ("eleccomp_asset_turnover_trend_3y", 0., 3),
    ("eleccomp_asset_turnover_trend_3y", -.05, 1), ("eleccomp_asset_turnover_trend_3y", -.051, 0),
    ("cash_conversion", .9, 8), ("cash_conversion", .7, 5), ("cash_conversion", .5, 2),
    ("debt_ratio", .45, 6), ("debt_ratio", .60, 3), ("debt_ratio", .601, 0),
    ("net_debt_to_equity", .35, 3), ("net_debt_to_equity", .65, 1),
    ("cash_to_current_liabilities", .4, 4), ("cash_to_current_liabilities", .25, 2),
])
def test_threshold_boundaries(name, value, points):
    scored = next(rule for rule in _score_quality_metrics({name: value}, ELECCOMP_PROFILE) if rule.name == name)
    assert scored.available and scored.earned == points


@pytest.mark.parametrize("case", ["short", "gap", "untrusted", "fractional_year"])
def test_five_year_metrics_require_complete_trusted_consecutive_years(case):
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
    for name in ELECCOMP_CYCLE_METRICS:
        if not name.endswith("_3y"):
            assert result.metrics[name] is None
    assert result.available_metric_weight < 100
    assert any(flag.startswith("ELECCOMP_CYCLE_MISSING=") for flag in result.qa_flags)
    assert not result.hard_red_flags


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), "--", 0., -1.])
def test_invalid_asset_denominator_removes_only_turnover_weight(invalid):
    annual = annual_history().astype({"total_assets": object})
    annual.loc[4, "total_assets"] = invalid
    result = evaluate(annual)
    assert result.metrics["eleccomp_asset_turnover_trend_3y"] is None
    assert result.available_metric_weight == 95
    assert not next(rule for rule in result.metric_scores if rule.name == "eleccomp_asset_turnover_trend_3y").available
    assert not result.hard_red_flags


@pytest.mark.parametrize("flows", [[0.] * 5, [-1.] * 5, [1., 1., 1., 1., -4. + 1e-13]])
def test_cashflow_denominator_cannot_fabricate_capex_coverage(flows):
    annual = annual_history().assign(operating_cash_flow=flows)
    assert evaluate(annual).metrics["eleccomp_capex_to_cfo_5y"] is None


def test_negative_capex_is_missing_not_a_positive_free_cash_flow_boost():
    annual = annual_history()
    annual.loc[2, "capex"] = -5
    result = evaluate(annual)
    assert result.metrics["eleccomp_positive_fcf_year_ratio"] is None
    assert result.metrics["eleccomp_capex_to_cfo_5y"] is None
    assert result.metrics["eleccomp_positive_cfo_year_ratio"] == 1


def test_scale_invariance_and_decimal_boundary_comparisons():
    annual = annual_history()
    annual["gross_margin"] = .32
    annual["operating_margin"] = [.02, .02, .02, .025, .03]
    annual["revenue"] = [100., 100., 100., 103., 106.]
    result = evaluate(annual, snapshot(gross_margin_ttm=.29))
    scored = {metric.name: metric.earned for metric in result.metric_scores}
    assert scored["eleccomp_gross_margin_drawdown_5y"] == 5
    assert scored["eleccomp_operating_margin_trend_3y"] == 5
    assert scored["eleccomp_asset_turnover_trend_3y"] == 5
    annual[["revenue", "total_assets", "operating_cash_flow", "capex"]] *= 1000
    scaled = evaluate(annual, snapshot(gross_margin_ttm=.29))
    for name in ELECCOMP_CYCLE_METRICS:
        assert scaled.metrics[name] == result.metrics[name]


def test_future_and_current_revisions_are_selected_at_signal_cutoff():
    annual = annual_history()
    revision = annual.iloc[[-1]].copy()
    revision["available_at"] = "2027-03-01T00:00:00Z"
    revision["roic"] = -1
    assert evaluate(pd.concat([annual, revision])).metrics == evaluate().metrics
    revision["available_at"] = "2026-03-01T00:00:00Z"
    current = evaluate(pd.concat([annual, revision]))
    assert current.metrics["eleccomp_roic_5y_p25"] == .08
    for changed in (snapshot(available_at="2027-01-01T00:00:00Z"), snapshot(availability_source="INFERRED")):
        result = evaluate(snap=changed)
        assert result.metrics["eleccomp_gross_margin_drawdown_5y"] is None
        assert result.status is GateStatus.UNKNOWN


def test_peak_earnings_do_not_erase_troughs_or_structural_deterioration():
    annual = annual_history()
    annual["roic"] = [-.04, -.02, .03, .10, .30]
    annual["gross_margin"] = [.06, .08, .10, .18, .40]
    annual["operating_margin"] = [.1, .1, .1, .06, .02]
    annual["total_assets"] = [100., 110., 130., 160., 220.]
    annual["operating_cash_flow"] = [15., 16., -10., -10., -10.]
    weak = evaluate(annual, snapshot(gross_margin_ttm=.15))
    assert weak.metrics["eleccomp_roic_5y_p25"] == -.02
    assert weak.metrics["eleccomp_gross_margin_5y_p25"] == .08
    assert weak.metrics["eleccomp_asset_turnover_trend_3y"] < 0
    assert weak.score < evaluate().score
    assert "NON_POSITIVE_THREE_YEAR_CFO" in weak.qa_flags
    annual.loc[4, "operating_margin"] = .13
    recovery = evaluate(annual, snapshot(gross_margin_ttm=.38))
    assert recovery.score > weak.score
    assert recovery.industry_profile_mode == "ACTIVE"


def test_negative_ebitda_and_hard_rejects_are_not_relaxed():
    debt = evaluate(snap=snapshot(cash=10, total_debt=100, ebitda_ttm=-1, interest_coverage=2))
    assert {"NET_DEBT_TO_EBITDA_NOT_MEANINGFUL", "INTEREST_COVERAGE_BELOW_3"}.issubset(debt.qa_flags)
    for changes in ({"going_concern": True}, {"audit_opinion": "ADVERSE"}, {"equity": -1}):
        assert evaluate(snap=snapshot(**changes)).status is GateStatus.REJECT


def test_v41_active_component_profile_keeps_native_entry_checks():
    data = attach_low_entry_v4_context(
        buy_candidate_history(), annual_fundamentals=annual_history(),
        fundamental_snapshot=snapshot(), signal_date=CUTOFF,
    )
    row = LowEntryScoreV41Strategy().run(data).iloc[-1]
    assert row["Industry Profile Version"] == ELECCOMP_QUALITY_PROFILE_VERSION
    assert row["低點決策"] == "BUY" and row["建議股數"] > 0


def test_daily_component_evidence_is_published_without_changing_native_v6_quality(tmp_path):
    market = daily_market()
    snap = snapshot()
    snap["les_v6_filings"] = market.fundamental_snapshots["2330"]["les_v6_filings"]
    market = replace(market, annual_fundamentals={"2330": annual_history()},
                     fundamental_snapshots={"2330": snap})
    builder, decisions = build(market)
    decision = decisions[0]
    report = json.loads(decision.value("LES v6 Report"))
    profile = report["industry_profile"]
    evidence = profile["quality_evaluation"]
    assert market.default_sector == "ELEC" and profile["name"] == "ELECCOMP_PROFILE"
    assert profile["version"] == ELECCOMP_QUALITY_PROFILE_VERSION
    assert evidence["score"] == 100 and evidence["available_weight"] == 100
    assert evidence["metrics"]["eleccomp_roic_5y_p25"]["value"] == .10
    assert report["quality"]["score"] == decision.value("Quality Score")
    assert report["decision"]["signal"] == "BUY_SMALL" and report["sizing"]["indicative_shares"] > 0
    html = RendererFactory.get_renderer("daily_strategy_email").render(decisions)
    assert "產業 QS 100.00" in html and ELECCOMP_QUALITY_PROFILE_VERSION in html
    assert "ACTIVE" in html and "進場已啟用" in html and "SHADOW" not in html
    CsvRepository(tmp_path).write_analysis_dataset("eleccomp", builder.project_decisions(decisions))
    restored = pd.read_csv(tmp_path / "eleccomp_analysis_dataset.csv")
    from infrastructure.storage.report_evidence import resolve_csv_report
    assert resolve_csv_report(restored.iloc[0]["LES v6 Report"], tmp_path)["industry_profile"] == profile
    _, partial = build(replace(market, annual_fundamentals={}))
    partial_report = json.loads(partial[0].value("LES v6 Report"))
    assert partial_report["industry_profile"]["quality_evaluation"]["available_weight"] < 100
    assert partial_report["quality"] == report["quality"]
    assert "缺失指標" in RendererFactory.get_renderer("daily_strategy_email").render(partial)
