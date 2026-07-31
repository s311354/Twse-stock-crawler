import numpy as np
import pandas as pd
import pytest

from domain.data_provenance import OFFICIAL_FILING_TIMESTAMP
from domain.data_provenance import MOPS_PUBLIC_ANNOUNCEMENT
from domain.fundamentals import GateStatus
from domain.fundamentals import MetricScore
from domain.fundamentals import QualityGateResult
from domain.fundamentals import evaluate_quality_gate
from domain.industry_profiles import CHEMICAL_PROFILE
from domain.industry_profiles import ELECTRONICS_PROFILE
from domain.industry_profiles import IndustryProfileMode
from domain.industry_profiles import IndustryProfileRegistry
from domain.strategies.low_entry_score_v4 import LowEntryScoreV4Strategy
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_STATUS_ATTR
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_STATUS_DETAIL_ATTR
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from domain.strategies.low_entry_score_v4 import attach_low_entry_v4_context
from domain.strategies.low_entry_score_v4_1 import LowEntryScoreV41Config
from domain.strategies.low_entry_score_v4_1 import LowEntryScoreV41Strategy
from domain.strategy import available_strategies
from domain.strategy import get_strategy


def annual_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fiscal_year": [2022, 2023, 2024],
            "available_at": [
                "2023-03-01T00:00:00Z",
                "2024-03-01T00:00:00Z",
                "2025-03-01T00:00:00Z",
            ],
            "availability_source": [OFFICIAL_FILING_TIMESTAMP] * 3,
            "revenue": [100.0, 110.0, 125.0],
            "net_income": [10.0, 12.0, 14.0],
            "operating_cash_flow": [15.0, 16.0, 18.0],
            "capex": [3.0, 3.0, 4.0],
            "roic": [0.16, 0.17, 0.18],
            "operating_margin": [0.15, 0.16, 0.17],
            "diluted_shares": [10.0, 10.0, 10.0],
        }
    )


def fundamental_snapshot(**overrides: object) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "available_at": "2025-05-01T00:00:00Z",
        "availability_source": OFFICIAL_FILING_TIMESTAMP,
        "ttm_eps": 5.0,
        "cash": 100.0,
        "total_debt": 20.0,
        "ebitda_ttm": 50.0,
        "ebit_ttm": 40.0,
        "interest_expense_ttm": 2.0,
        "going_concern": False,
        "material_accounting_issue": False,
        "sector": "24",
        "filing_id": 233020251,
        "revision_number": 1,
        "content_sha256": "a" * 64,
        "source_url": "https://mops.example/2330-2025Q1.html",
        "correction_announced_on": None,
        "derivation_version": "2026-07-21-qs-dc-v2",
        "period_end": "2026-03-31",
        "statement_scope": "consolidated",
        "audit_status": "reviewed",
        "audit_opinion": "REVIEWED_UNMODIFIED",
        "audit_evidence_available": True,
        "revenue_ttm": 125.0,
        "net_income_ttm": 14.0,
        "operating_cash_flow_ttm": 18.0,
        "capex_ttm": 4.0,
        "free_cash_flow_ttm": 14.0,
        "gross_profit_ttm": 50.0,
        "total_assets": 500.0,
        "total_liabilities": 200.0,
        "equity": 300.0,
        "total_equity": 300.0,
        "current_assets": 220.0,
        "current_liabilities": 100.0,
        "roe_ttm": 0.16,
        "roa_ttm": 0.09,
        "roic_ttm": 0.18,
        "cash_conversion": 1.2,
        "accrual_ratio": -0.01,
        "debt_ratio": 0.4,
        "net_debt_to_equity": -0.2,
        "interest_coverage": 20.0,
        "current_ratio": 2.2,
        "cash_to_current_liabilities": 1.0,
        "revenue_growth_ttm_yoy": 0.1,
        "operating_income_growth_ttm_yoy": 0.12,
        "positive_operating_quarters_8": 8,
        "positive_eps_quarters_8": 8,
        "gross_margin_change": 0.0,
        "share_dilution_yoy": 0.0,
        "distribution_coverage": 0.5,
        "bvps_trend": 0.05,
    }
    snapshot.update(overrides)
    return snapshot


def chemical_annual_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fiscal_year": [2020, 2021, 2022, 2023, 2024],
            "available_at": [
                "2021-03-01T00:00:00Z",
                "2022-03-01T00:00:00Z",
                "2023-03-01T00:00:00Z",
                "2024-03-01T00:00:00Z",
                "2025-03-01T00:00:00Z",
            ],
            "availability_source": [OFFICIAL_FILING_TIMESTAMP] * 5,
            "revenue": [120.0, 115.0, 100.0, 102.0, 105.0],
            "gross_profit": [26.4, 23.0, 16.0, 17.34, 18.9],
            "net_income": [10.0, 8.0, 4.0, 5.0, 6.0],
            "operating_cash_flow": [14.0, 12.0, 8.0, 9.0, 10.0],
            "capex": [5.0, 5.0, 4.0, 4.0, 4.0],
            "free_cash_flow": [9.0, 7.0, 4.0, 5.0, 6.0],
            "roic": [0.08, 0.07, 0.05, 0.06, 0.06],
            "operating_margin": [0.10, 0.07, 0.03, 0.035, 0.04],
            "diluted_shares": [10.0] * 5,
        }
    )


def chemical_snapshot(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "sector": "21",
        "revenue_ttm": 108.0,
        "gross_profit_ttm": 19.44,
        "gross_margin_ttm": 0.18,
        "operating_income_growth_ttm_yoy": 0.06,
        "revenue_growth_ttm_yoy": 0.03,
        "gross_margin_change": 0.01,
        "roic_ttm": 0.06,
    }
    values.update(overrides)
    return fundamental_snapshot(**values)


def buy_candidate_history() -> pd.DataFrame:
    size = 100
    close = np.concatenate(
        [
            np.linspace(95, 110, 40),
            np.linspace(82, 88, 40),
            np.linspace(88.5, 90, 20),
        ]
    )
    open_price = close - 0.1
    high = close + 0.5
    low = close - 0.5

    close[40], low[40] = 82.0, 80.0
    open_price[41], close[41], high[41], low[41] = 109.5, 110.0, 111.0, 109.0
    close[80:] = np.linspace(88.5, 90.4, 20)
    open_price[80:] = close[80:] - 0.2
    high[80:] = close[80:] + 0.8
    low[80:] = close[80:] - 0.8

    # Two confirmed swing lows create a higher-low structure without look-ahead.
    low[88:93] = [88.5, 88.4, 88.0, 88.5, 88.6]
    low[95:100] = [89.6, 89.7, 89.5, 89.6, 89.6]
    close[97:100] = [90.0, 90.1, 90.8]
    open_price[97:100] = [89.7, 89.8, 89.9]
    high[97:100] = [90.3, 90.4, 90.9]

    volume = np.full(size, 1_000_000.0)
    volume[-1] = 1_500_000.0
    amount = np.full(size, 100_000_000.0)
    trades = np.full(size, 5_000.0)
    pe = np.full(size, 15.0)
    pe[-1] = 10.0
    taiex = np.linspace(20_000, 20_100, size)

    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=size, freq="B"),
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Amount": amount,
            "Trades": trades,
            "Bid": close - 0.05,
            "Ask": close + 0.05,
            "PE": pe,
            "TAIEX Close": taiex,
        }
    )


def with_quality_context(data: pd.DataFrame, snapshot: dict[str, object] | None = None) -> pd.DataFrame:
    return attach_low_entry_v4_context(
        data,
        annual_fundamentals=annual_fundamentals(),
        fundamental_snapshot=snapshot or fundamental_snapshot(),
        signal_date="2026-07-21T05:30:00Z",
    )


def test_low_entry_score_v4_is_discoverable() -> None:
    assert "low_entry_score_v4" in available_strategies()
    assert get_strategy("low_entry_score_v4").name == "low_entry_score_v4"


def test_unavailable_mops_context_has_typed_reason_instead_of_missing_columns() -> None:
    data = buy_candidate_history()
    data.attrs[SIGNAL_DATE_ATTR] = "2026-07-21T07:00:00+08:00"
    data.attrs[FUNDAMENTAL_STATUS_ATTR] = "CIRCUIT_OPEN"
    data.attrs[FUNDAMENTAL_STATUS_DETAIL_ATTR] = "MOPS schema failure circuit is open"

    latest = LowEntryScoreV4Strategy().run(data).iloc[-1]

    assert latest["Quality Gate"] == "UNKNOWN"
    assert latest["低點決策"] == "DATA_INSUFFICIENT"
    assert "MOPS_STATUS=CIRCUIT_OPEN" in latest["低點理由"]
    assert "missing annual columns" not in latest["低點理由"]


def test_quality_gate_is_point_in_time_and_uses_latest_available_revision() -> None:
    annual = pd.concat(
        [
            annual_fundamentals(),
            pd.DataFrame(
                {
                    "fiscal_year": [2024],
                    "available_at": ["2026-08-01T00:00:00Z"],
                    "availability_source": [OFFICIAL_FILING_TIMESTAMP],
                    "revenue": [1.0],
                    "net_income": [-100.0],
                    "operating_cash_flow": [-100.0],
                    "capex": [100.0],
                    "roic": [-1.0],
                    "operating_margin": [-1.0],
                    "diluted_shares": [100.0],
                }
            ),
        ],
        ignore_index=True,
    )

    result = evaluate_quality_gate(
        annual,
        fundamental_snapshot(),
        "2026-07-21T05:30:00Z",
    )

    assert result.status is GateStatus.PASS
    assert result.score == 100.0
    assert result.data_confidence == 100.0
    assert result.available_metric_weight == 100.0
    assert result.metrics["cumulative_cfo_3y"] == 49.0


def test_non_positive_eps_is_watch_quality_not_audit_hard_reject() -> None:
    result = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(ttm_eps=0),
        "2026-07-21T05:30:00Z",
    )

    assert result.status is GateStatus.WATCH_QUALITY
    assert result.score is not None
    assert result.hard_red_flags == []
    assert "NON_POSITIVE_TTM_EPS" in result.qa_flags


def test_leverage_and_interest_coverage_are_quality_warnings_not_audit_rejects() -> None:
    result = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(
            cash=0,
            total_debt=400,
            ebitda_ttm=50,
            net_debt_to_equity=1.33,
            interest_coverage=2,
        ),
        "2026-07-21T05:30:00Z",
    )

    assert result.status is not GateStatus.REJECT
    assert result.hard_red_flags == []
    assert "NET_DEBT_TO_EBITDA_ABOVE_3" in result.qa_flags
    assert "INTEREST_COVERAGE_BELOW_3" in result.qa_flags


def test_quality_gate_does_not_treat_unknown_audit_flag_as_false() -> None:
    result = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(material_accounting_issue=None),
        "2026-07-21T05:30:00Z",
    )

    assert result.status is GateStatus.PASS
    assert result.score == 100.0
    assert result.data_confidence == 95.0
    assert "material_accounting_issue" in "；".join(result.qa_flags)


def test_quality_gate_scope_limitation_reduces_dc_without_hard_reject() -> None:
    result = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(
            audit_opinion="QUALIFIED_SCOPE_LIMITATION",
            accounting_concern_classification="QUALIFIED_SCOPE_LIMITATION",
            material_accounting_issue=False,
        ),
        "2026-07-21T05:30:00Z",
    )

    assert result.status is GateStatus.PASS
    assert result.data_confidence == 90.0
    assert result.hard_red_flags == []
    assert "QUALIFIED_SCOPE_LIMITATION" in result.qa_flags


def test_quality_gate_material_misstatement_remains_hard_reject() -> None:
    result = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(
            audit_opinion="QUALIFIED_MATERIAL_MISSTATEMENT",
            accounting_concern_classification="QUALIFIED_MATERIAL_MISSTATEMENT",
            material_accounting_issue=True,
        ),
        "2026-07-21T05:30:00Z",
    )

    assert result.status is GateStatus.REJECT
    assert "MATERIAL_ACCOUNTING_OR_FILING_CONCERN" in result.hard_red_flags


def test_quality_gate_adverse_and_disclaimer_remain_hard_rejects() -> None:
    expected_flags = {
        "ADVERSE": "ADVERSE_AUDIT_OPINION",
        "DISCLAIMER": "DISCLAIMER_OF_OPINION",
    }
    for opinion, expected_flag in expected_flags.items():
        result = evaluate_quality_gate(
            annual_fundamentals(),
            fundamental_snapshot(
                audit_opinion=opinion,
                material_accounting_issue=False,
            ),
            "2026-07-21T05:30:00Z",
        )

        assert result.status is GateStatus.REJECT
        assert expected_flag in result.hard_red_flags


def test_prior_period_modified_opinion_is_qa_not_current_hard_reject() -> None:
    result = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(
            audit_opinion="REVIEWED_UNMODIFIED",
            current_audit_opinion="REVIEWED_UNMODIFIED",
            comparative_audit_opinion="ADVERSE",
            material_accounting_issue=False,
        ),
        "2026-07-21T05:30:00Z",
    )

    assert result.status is GateStatus.PASS
    assert result.hard_red_flags == []
    assert "PRIOR_PERIOD_AUDIT_OPINION=ADVERSE" in result.qa_flags


@pytest.mark.parametrize(
    "comparative_opinion",
    [None, np.nan, "", "NONE", "N/A", "UNKNOWN"],
)
def test_missing_prior_period_opinion_does_not_create_qa_warning(
    comparative_opinion: object,
) -> None:
    result = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(
            comparative_audit_opinion=comparative_opinion,
        ),
        "2026-07-21T05:30:00Z",
    )

    assert not any(
        flag.startswith("PRIOR_PERIOD_AUDIT_OPINION=")
        for flag in result.qa_flags
    )


def test_quality_gate_requires_known_sector() -> None:
    result = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(sector=None),
        "2026-07-21T05:30:00Z",
    )

    assert result.status is GateStatus.UNKNOWN
    assert result.score == 100.0


def test_quality_gate_rejects_snapshot_not_public_at_signal_time() -> None:
    result = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(available_at="2026-07-21T06:00:00Z"),
        "2026-07-21T05:30:00Z",
    )

    assert result.status is GateStatus.UNKNOWN
    assert "SNAPSHOT_NOT_POINT_IN_TIME" in result.qa_flags


def test_low_entry_score_v4_missing_mops_context_is_watch_only() -> None:
    latest = LowEntryScoreV4Strategy().run(buy_candidate_history()).iloc[-1]

    assert latest["Quality Gate"] == "UNKNOWN"
    assert latest["低點決策"] == "DATA_INSUFFICIENT"
    assert latest["低點分數"] == 100
    assert latest["LES v4 Trigger"] == 1


def test_low_entry_score_v4_requires_all_entry_gates_for_buy() -> None:
    latest = LowEntryScoreV4Strategy().run(with_quality_context(buy_candidate_history())).iloc[-1]

    assert latest["Quality Gate"] == "PASS"
    assert latest["No Veto"] == 1
    assert latest["低點分數"] == 100
    assert latest["LES v4 Trigger"] == 1
    assert latest["Risk Gate"] == "PASS"
    assert latest["RewardRisk"] >= 2
    assert latest["低點決策"] == "BUY"
    assert latest["建議股數"] > 0


def test_low_entry_score_v4_1_end_to_end_buy_is_auditable() -> None:
    latest = LowEntryScoreV41Strategy().run(
        with_quality_context(buy_candidate_history())
    ).iloc[-1]

    assert latest["Quality Gate"] == "PASS"
    assert latest["低點分數"] == 88
    assert latest["No Veto"] == 1
    assert latest["LES v4.1 Technical Trigger"] == 1
    assert latest["LES v4.1 Trigger"] == 1
    assert latest["Liquidity Gate"] == "PASS"
    assert latest["Risk Gate"] == "PASS"
    assert latest["低點決策"] == "BUY"
    assert latest["LES Model Version"] == "4.1.0"


def test_low_entry_score_v4_1_balanced_provisional_uses_small_position(
    monkeypatch,
) -> None:
    strategy = LowEntryScoreV41Strategy()
    provisional = QualityGateResult(
        status=GateStatus.PROVISIONAL,
        score=80.0,
        data_confidence=65.0,
        metric_scores=[
            MetricScore("profit", "A", 20, 25, True, "available"),
            MetricScore("balance", "C", 20, 25, True, "available"),
        ],
        metrics={"cfo_ttm": None, "cumulative_cfo_3y": 10.0},
    )
    monkeypatch.setattr(
        strategy,
        "_evaluate_quality",
        lambda data, context: provisional,
    )

    latest = strategy.run(with_quality_context(buy_candidate_history())).iloc[-1]

    assert latest["Quality Gate"] == "PROVISIONAL"
    assert latest["Provisional Buy Eligible"] == 1
    assert latest["LES v4.1 Strict Trigger"] == 1
    assert latest["Strict Risk Gate"] == "PASS"
    assert latest["低點決策"] == "BUY_SMALL"
    assert latest["Position Fraction"] == pytest.approx(1 / 6, abs=0.0001)
    assert latest["Effective Risk Percent"] == pytest.approx(
        0.005 / 6,
        abs=0.000001,
    )
    assert latest["建議股數"] > 0


def test_low_entry_score_v4_cannot_buy_without_aligned_taiex_history() -> None:
    data = buy_candidate_history().drop(columns=["TAIEX Close"])
    latest = LowEntryScoreV4Strategy().run(with_quality_context(data)).iloc[-1]

    assert latest["Risk Gate"] == "UNKNOWN"
    assert "MARKET_HISTORY_INSUFFICIENT" in latest["Risk Gate Reasons"]
    assert latest["低點決策"] != "BUY"


def test_low_entry_score_v4_veto_precedes_high_technical_score() -> None:
    data = buy_candidate_history()
    data.loc[data.index[-1], ["Open", "High", "Low", "Close"]] = [79.0, 79.5, 77.0, 77.5]
    latest = LowEntryScoreV4Strategy().run(with_quality_context(data)).iloc[-1]

    assert latest["No Veto"] == 0
    assert "創20日新低" in latest["Falling Knife Veto"]
    assert latest["低點決策"] == "NO_BUY"


def test_low_entry_score_v4_quality_fail_has_highest_priority() -> None:
    latest = LowEntryScoreV4Strategy().run(
        with_quality_context(
            buy_candidate_history(),
            snapshot=fundamental_snapshot(going_concern=True),
        )
    ).iloc[-1]

    assert latest["Quality Gate"] == "REJECT"
    assert "GOING_CONCERN_MATERIAL_UNCERTAINTY" in latest["Quality Hard Failures"]
    assert latest["低點決策"] == "REJECT"


def test_low_entry_score_v4_rejects_incomplete_same_day_bar() -> None:
    data = buy_candidate_history()
    signal_date = pd.Timestamp(data["Date"].iloc[-1], tz="Asia/Taipei") + pd.Timedelta(
        hours=13,
        minutes=29,
    )
    contextual = attach_low_entry_v4_context(
        data,
        annual_fundamentals=annual_fundamentals(),
        fundamental_snapshot=fundamental_snapshot(),
        signal_date=signal_date,
    )

    latest = LowEntryScoreV4Strategy().run(contextual).iloc[-1]

    assert latest["Market Data Gate"] == "FAIL"
    assert "LATEST_BAR_NOT_COMPLETED" in latest["Market Data Reasons"]
    assert latest["Risk Gate"] == "UNKNOWN"
    assert latest["低點決策"] == "NO_BUY"


def test_low_entry_score_v4_emits_spec_audit_contract() -> None:
    latest = LowEntryScoreV4Strategy().run(
        with_quality_context(buy_candidate_history())
    ).iloc[-1]

    assert latest["Fundamental Available At"] == "2025-05-01T00:00:00Z"
    assert latest["Fundamental Availability Source"] == OFFICIAL_FILING_TIMESTAMP
    assert latest["Fundamental Filing ID"] == 233020251
    assert latest["Fundamental Revision"] == 1
    assert latest["Fundamental SHA256"] == "a" * 64
    assert latest["Fundamental Source URL"].endswith("2330-2025Q1.html")
    assert pd.isna(latest["Fundamental Correction Announced On"])
    assert latest["Quality Hard Failures"] == "NONE"
    assert latest["Market Data Gate"] == "PASS"
    assert latest["LES Model Version"] == "4.1.0"
    assert latest["LES Parameter Version"] == "2026-07-21-qs-dc-v2"
    assert latest["Data Confidence"] == 100.0
    assert latest["Available Metric Weight"] == 100.0
    assert latest["Fundamental Derivation Version"] == "2026-07-21-qs-dc-v2"
    assert latest["Risk Percent"] == 0.005
    assert latest["Maximum Stop ATR"] == 1.5
    assert latest["Minimum RewardRisk"] == 2.0
    veto_columns = [column for column in latest.index if column.startswith("Veto ")]
    assert len(veto_columns) == 9
    assert set(latest[veto_columns].astype(int)) <= {0, 1}


def test_quality_gate_rejects_non_electronic_sector_and_untrusted_time() -> None:
    invalid_sector = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(sector="20"),
        "2026-07-21T05:30:00Z",
    )
    untrusted_time = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(availability_source="LEGACY_UNVERIFIED"),
        "2026-07-21T05:30:00Z",
    )
    announcement_only = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(availability_source=MOPS_PUBLIC_ANNOUNCEMENT),
        "2026-07-21T05:30:00Z",
    )

    assert invalid_sector.status is GateStatus.SECTOR_UNSUPPORTED
    assert "SECTOR_UNSUPPORTED" in invalid_sector.qa_flags
    assert untrusted_time.status is GateStatus.UNKNOWN
    assert "UNTRUSTED_SNAPSHOT_SOURCE" in untrusted_time.qa_flags
    assert announcement_only.status is GateStatus.UNKNOWN


def test_industry_profile_registry_separates_chemical_from_electronics() -> None:
    assert IndustryProfileRegistry.resolve("21") is CHEMICAL_PROFILE
    assert IndustryProfileRegistry.resolve("24") is ELECTRONICS_PROFILE
    assert CHEMICAL_PROFILE.mode is IndustryProfileMode.ACTIVE
    assert ELECTRONICS_PROFILE.sector_codes == frozenset(
        str(code) for code in range(24, 32)
    )
    assert CHEMICAL_PROFILE.sector_codes == frozenset({"21"})
    assert (
        ELECTRONICS_PROFILE.version
        == "electronics-quality-2026-07-21-v2"
    )
    assert CHEMICAL_PROFILE.version == "chemical-quality-2026-07-27-v2"
    assert sum(rule.maximum for rule in CHEMICAL_PROFILE.metric_rules) == 100
    assert CHEMICAL_PROFILE.pass_score == 65
    assert CHEMICAL_PROFILE.fail_score == 50
    assert CHEMICAL_PROFILE.buy_small_quality_score == 60


def test_chemical_profile_scores_cycle_trough_with_recovery_evidence() -> None:
    result = evaluate_quality_gate(
        chemical_annual_fundamentals(),
        chemical_snapshot(),
        "2026-07-21T05:30:00Z",
    )

    assert result.industry_profile == "CHEMICAL_PROFILE"
    assert result.industry_profile_mode == "ACTIVE"
    assert result.status is GateStatus.PASS
    assert result.metrics["roic_5y_median"] == pytest.approx(0.06)
    assert result.metrics["positive_cfo_year_ratio"] == 1.0
    assert result.metrics["gross_margin_5y_percentile"] == pytest.approx(0.6)
    assert result.metrics["operating_margin_trend"] > 0


def test_chemical_profile_penalizes_structural_deterioration() -> None:
    annual = chemical_annual_fundamentals()
    annual["roic"] = [0.04, 0.02, 0.0, -0.02, -0.04]
    annual["operating_cash_flow"] = [5.0, 2.0, -2.0, -4.0, -6.0]
    annual["operating_margin"] = [0.08, 0.05, 0.02, -0.01, -0.03]
    result = evaluate_quality_gate(
        annual,
        chemical_snapshot(
            ttm_eps=-1.0,
            operating_cash_flow_ttm=-6.0,
            free_cash_flow_ttm=-10.0,
            cash=0.0,
            total_debt=400.0,
            ebitda_ttm=40.0,
            debt_ratio=0.75,
            net_debt_to_equity=1.2,
            interest_coverage=1.5,
            current_ratio=0.8,
            operating_income_growth_ttm_yoy=-0.3,
        ),
        "2026-07-21T05:30:00Z",
    )

    assert result.score is not None and result.score < 55
    assert result.status is GateStatus.REJECT
    assert result.hard_red_flags == []
    assert "NON_POSITIVE_THREE_YEAR_CFO" in result.qa_flags


def test_chemical_profile_penalizes_cycle_peak_reversal() -> None:
    stable = evaluate_quality_gate(
        chemical_annual_fundamentals(),
        chemical_snapshot(),
        "2026-07-21T05:30:00Z",
    )
    reversal = evaluate_quality_gate(
        chemical_annual_fundamentals(),
        chemical_snapshot(
            gross_margin_ttm=0.25,
            gross_profit_ttm=27.0,
            gross_margin_change=-0.08,
            operating_income_growth_ttm_yoy=-0.2,
            revenue_growth_ttm_yoy=-0.08,
        ),
        "2026-07-21T05:30:00Z",
    )

    assert reversal.metrics["gross_margin_5y_percentile"] == 1.0
    assert reversal.score is not None
    assert stable.score is not None
    assert reversal.score < stable.score


def test_chemical_profile_marks_negative_ebitda_ratio_not_meaningful() -> None:
    result = evaluate_quality_gate(
        chemical_annual_fundamentals(),
        chemical_snapshot(
            cash=0.0,
            total_debt=100.0,
            ebitda_ttm=-1.0,
        ),
        "2026-07-21T05:30:00Z",
    )

    assert result.metrics["net_debt_to_ebitda"] is None
    assert "NET_DEBT_TO_EBITDA_NOT_MEANINGFUL" in result.qa_flags
    assert result.hard_red_flags == []


def test_chemical_profile_missing_cycle_metrics_are_not_scored_as_zero() -> None:
    annual = chemical_annual_fundamentals().drop(
        columns=["gross_profit", "roic", "operating_margin"]
    )
    result = evaluate_quality_gate(
        annual,
        chemical_snapshot(
            gross_margin_ttm=None,
            gross_profit_ttm=None,
            roic_ttm=None,
        ),
        "2026-07-21T05:30:00Z",
    )
    cycle_scores = {
        metric.name: metric
        for metric in result.metric_scores
        if metric.name
        in {
            "roic_5y_median",
            "gross_margin_5y_percentile",
            "operating_margin_trend",
        }
    }

    assert all(not metric.available for metric in cycle_scores.values())
    assert result.available_metric_weight < 100
    assert result.score is not None


def test_chemical_profile_preserves_cross_industry_hard_rejects() -> None:
    result = evaluate_quality_gate(
        chemical_annual_fundamentals(),
        chemical_snapshot(going_concern=True),
        "2026-07-21T05:30:00Z",
    )

    assert result.status is GateStatus.REJECT
    assert "GOING_CONCERN_MATERIAL_UNCERTAINTY" in result.hard_red_flags


def test_active_chemical_profile_can_generate_standard_buy() -> None:
    contextual = attach_low_entry_v4_context(
        buy_candidate_history(),
        annual_fundamentals=chemical_annual_fundamentals(),
        fundamental_snapshot=chemical_snapshot(),
        signal_date="2026-07-21T05:30:00Z",
    )

    latest = LowEntryScoreV41Strategy().run(contextual).iloc[-1]

    assert latest["Quality Gate"] == "PASS"
    assert latest["Industry Quality Profile"] == "CHEMICAL_PROFILE"
    assert latest["Industry Profile Mode"] == "ACTIVE"
    assert latest["LES v4.1 Trigger"] == 1
    assert latest["低點決策"] == "BUY"
    assert latest["建議股數"] > 0


def test_active_chemical_profile_can_generate_strict_buy_small(
    monkeypatch,
) -> None:
    strategy = LowEntryScoreV41Strategy()
    chemical_watch = QualityGateResult(
        status=GateStatus.WATCH_QUALITY,
        score=62.0,
        data_confidence=80.0,
        industry_profile="CHEMICAL_PROFILE",
        industry_profile_version=CHEMICAL_PROFILE.version,
        industry_profile_mode="ACTIVE",
    )
    monkeypatch.setattr(
        strategy,
        "_evaluate_quality",
        lambda data, context: chemical_watch,
    )

    latest = strategy.run(with_quality_context(buy_candidate_history())).iloc[-1]

    assert latest["LES v4.1 Strict Trigger"] == 1
    assert latest["Strict Risk Gate"] == "PASS"
    assert latest["Veto 資料或基本面禁止"] == 0
    assert latest["低點決策"] == "BUY_SMALL"
    assert latest["Position Fraction"] == pytest.approx(1 / 8)
    assert latest["建議股數"] > 0


def test_chemical_buy_small_does_not_relax_confidence_or_strict_risk(
    monkeypatch,
) -> None:
    low_confidence = QualityGateResult(
        status=GateStatus.WATCH_QUALITY,
        score=62.0,
        data_confidence=74.0,
        industry_profile="CHEMICAL_PROFILE",
        industry_profile_version=CHEMICAL_PROFILE.version,
        industry_profile_mode="ACTIVE",
    )
    strategy = LowEntryScoreV41Strategy()
    monkeypatch.setattr(
        strategy,
        "_evaluate_quality",
        lambda data, context: low_confidence,
    )
    confidence_result = strategy.run(
        with_quality_context(buy_candidate_history())
    ).iloc[-1]

    strict_strategy = LowEntryScoreV41Strategy(
        LowEntryScoreV41Config(maximum_stop_atr=0.01)
    )
    eligible = QualityGateResult(
        status=GateStatus.WATCH_QUALITY,
        score=62.0,
        data_confidence=80.0,
        industry_profile="CHEMICAL_PROFILE",
        industry_profile_version=CHEMICAL_PROFILE.version,
        industry_profile_mode="ACTIVE",
    )
    monkeypatch.setattr(
        strict_strategy,
        "_evaluate_quality",
        lambda data, context: eligible,
    )
    strict_result = strict_strategy.run(
        with_quality_context(buy_candidate_history())
    ).iloc[-1]

    assert confidence_result["低點決策"] == "WATCH_ONLY"
    assert confidence_result["Veto 資料或基本面禁止"] == 1
    assert strict_result["Strict Risk Gate"] == "FAIL"
    assert strict_result["低點決策"] == "WATCH_ONLY"


def test_missing_optional_metrics_reduce_confidence_without_becoming_zero_quality() -> None:
    complete = evaluate_quality_gate(
        annual_fundamentals(),
        fundamental_snapshot(),
        "2026-07-21T05:30:00Z",
    )
    partial_snapshot = fundamental_snapshot(
        gross_profit_ttm=None,
        current_assets=None,
        current_liabilities=None,
        current_ratio=None,
        cash_to_current_liabilities=None,
        distribution_coverage=None,
        bvps_trend=None,
    )
    partial = evaluate_quality_gate(
        annual_fundamentals(),
        partial_snapshot,
        "2026-07-21T05:30:00Z",
    )

    assert complete.score == 100.0
    assert partial.score == 100.0
    assert partial.available_metric_weight < complete.available_metric_weight
    assert partial.data_confidence < complete.data_confidence
