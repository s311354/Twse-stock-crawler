from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from domain.data_provenance import OFFICIAL_FILING_TIMESTAMP
from domain.quality_v5 import BaseQualityStatus
from domain.quality_v5 import ExtendedQualityStatus
from domain.quality_v5 import FinalQualityStatus
from domain.quality_v5 import evaluate_v5_quality
from domain.strategies.low_entry_score_v4 import attach_low_entry_v4_context
from domain.strategies.low_entry_score_v4 import ACCOUNT_EQUITY_ATTR
from domain.strategies.low_entry_score_v4 import LowEntryScoreV4Strategy
from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
from domain.strategies.low_entry_score_v5 import FallingKnifeState
from domain.strategies.low_entry_score_v5 import LowEntryScoreV5Config
from domain.strategies.low_entry_score_v5 import LowEntryScoreV5Strategy
from domain.strategies.low_entry_score_v5 import ValuationStatus
from domain.strategy import available_strategies
from domain.strategy import get_strategy


SIGNAL_DATE = "2026-07-21T06:00:00Z"


def _annual_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fiscal_year": [2022, 2023, 2024],
            "available_at": [
                "2023-03-01T00:00:00Z",
                "2024-03-01T00:00:00Z",
                "2025-03-01T00:00:00Z",
            ],
            "revenue": [800.0, 880.0, 968.0],
            "net_income": [80.0, 92.0, 106.0],
            "operating_margin": [0.12, 0.125, 0.13],
            "diluted_shares": [10.0, 10.1, 10.2],
            "dividends_paid": [30.0, 34.0, 38.0],
            "operating_cash_flow": [10.0, 12.0, 14.0],
            "free_cash_flow": [7.0, 8.0, 9.0],
        }
    )


def _quality_snapshot(**overrides: object) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "sector": "24",
        "available_at": "2026-07-20T00:00:00Z",
        "availability_source": OFFICIAL_FILING_TIMESTAMP,
        "twse_financial_observed_at": "2026-07-20T00:00:00Z",
        "twse_financial_source": "TWSE_OPENAPI_V1",
        "twse_parent_equity": 300.0,
        "twse_total_assets": 500.0,
        "twse_total_liabilities": 200.0,
        "twse_total_equity": 300.0,
        "twse_debt_ratio": 0.4,
        "twse_current_ratio": 2.0,
        "debt_to_equity_below_industry_median": True,
        "equity_growth_yoy": 0.1,
        "share_dilution_yoy": 0.0,
        "revenue_growth_ttm_yoy": 0.1,
        "positive_operating_quarters_8": 8,
        "gross_margin_change": 0.0,
        "twse_operating_margin_ytd": 0.15,
        "operating_margin_change": 0.0,
        "net_margin_above_industry_median": True,
        "twse_roe_ytd_end_balance": 0.15,
        "twse_roa_ytd_end_balance": 0.08,
        "roe_change_4q": 0.0,
        "asset_growth_supported": True,
        "revenue_growth_above_industry_median": True,
        "eps_growth_ttm_yoy": 0.1,
        "asset_growth_yoy": 0.05,
        "operating_cash_flow_ttm": 18.0,
        "capex_ttm": 4.0,
        "free_cash_flow_ttm": 14.0,
        "cash_conversion": 1.2,
        "accrual_ratio": -0.01,
        "net_debt_to_equity": -0.2,
        "interest_coverage": 20.0,
        "cash_to_short_term_debt": 2.0,
        "interest_bearing_debt_growth_yoy": 0.02,
        "interest_expense_ttm": 2.0,
        "receivable_growth_supported": True,
        "inventory_growth_supported": True,
        "receivable_days_change": -1.0,
        "inventory_days_change": -1.0,
        "fixed_asset_turnover_change": 0.1,
        "construction_growth_supported": True,
        "distribution_coverage": 0.5,
        "roic_change_yoy": 0.1,
        "audit_opinion": "REVIEWED_UNMODIFIED",
        "material_accounting_issue": False,
        "going_concern": False,
        "cash": 100.0,
        "total_debt": 20.0,
        "statement_scope": "consolidated",
        "revision_number": 1,
        "derivation_version": "les-v5-test",
        "filing_id": 1,
        "ttm_eps": 11.0,
        "operating_margin_ttm": 0.13,
    }
    snapshot.update(overrides)
    return snapshot


def _panic_history() -> pd.DataFrame:
    size = 80
    close = np.linspace(110.0, 100.0, size)
    open_price = close + 0.2
    high = close + 1.0
    low = close - 1.0
    volume = np.full(size, 1_000_000.0)

    open_price[-1] = 95.0
    high[-1] = 96.0
    low[-1] = 88.0
    close[-1] = 89.0
    volume[-1] = 3_000_000.0

    return pd.DataFrame(
        {
            "Date": pd.bdate_range(end="2026-07-21", periods=size),
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Amount": np.full(size, 100_000_000.0),
            "Trades": np.full(size, 5_000.0),
            "Bid": close - 0.05,
            "Ask": close + 0.05,
            "PE": np.linspace(15.0, 10.0, size),
            "TAIEX Close": np.linspace(20_000.0, 20_500.0, size),
        }
    )


def test_v5_quality_scores_only_available_metrics_and_confirms_full_evidence() -> None:
    full = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    missing_one = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(debt_to_equity_below_industry_median=None),
        SIGNAL_DATE,
    )

    assert full.final_status is FinalQualityStatus.CONFIRMED
    assert full.base_score == 100
    assert full.extended_score == 100
    assert sum(metric.maximum for metric in full.base_metrics) == 100
    assert sum(metric.maximum for metric in full.extended_metrics) == 100
    assert missing_one.base_score == 100
    assert missing_one.base_available_weight == 95
    assert missing_one.base_data_confidence < full.base_data_confidence


def test_v5_quality_separates_twse_and_mops_point_in_time_clocks() -> None:
    future_mops = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(available_at="2026-07-22T00:00:00Z"),
        SIGNAL_DATE,
    )
    future_twse = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(
            twse_financial_observed_at="2026-07-22T00:00:00Z",
        ),
        SIGNAL_DATE,
    )
    adverse = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(audit_opinion="ADVERSE"),
        SIGNAL_DATE,
    )

    assert future_mops.base_status is BaseQualityStatus.PASS
    assert future_mops.extended_score is None
    assert future_mops.final_status is FinalQualityStatus.BASE_ONLY
    assert "MOPS_EXTENDED_NOT_POINT_IN_TIME" in future_mops.warnings
    assert future_twse.base_status is BaseQualityStatus.PROVISIONAL
    assert future_twse.final_status is FinalQualityStatus.WATCH
    assert "TWSE_BASE_OBSERVED_AFTER_SIGNAL" in future_twse.warnings
    assert "MOPS_POINT_IN_TIME_BASE_FALLBACK" in future_twse.warnings
    assert adverse.final_status is FinalQualityStatus.HARD_NO_BUY
    assert "ADVERSE_AUDIT_OPINION" in adverse.hard_rejects


def test_base_provisional_remains_watch_only_even_with_extended_pass() -> None:
    provisional_snapshot = _quality_snapshot(
        twse_financial_observed_at="2024-01-01T00:00:00Z",
        twse_financial_source="UNKNOWN",
        twse_total_assets=None,
        twse_total_liabilities=None,
        twse_total_equity=None,
    )
    result = evaluate_v5_quality(
        _annual_fundamentals(),
        provisional_snapshot,
        SIGNAL_DATE,
    )

    assert result.base_status is BaseQualityStatus.PROVISIONAL
    assert result.final_status is FinalQualityStatus.WATCH


def test_production_shape_partial_base_coverage_is_provisional_not_unknown() -> None:
    snapshot = _quality_snapshot()
    production_missing = (
        "debt_to_equity_below_industry_median",
        "equity_growth_yoy",
        "positive_operating_quarters_8",
        "operating_margin_change",
        "net_margin_above_industry_median",
        "roe_change_4q",
        "asset_growth_supported",
        "revenue_growth_above_industry_median",
        "eps_growth_ttm_yoy",
        "asset_growth_yoy",
    )
    for key in production_missing:
        snapshot.pop(key)

    result = evaluate_v5_quality(
        _annual_fundamentals(),
        snapshot,
        SIGNAL_DATE,
    )

    assert result.base_available_weight == 53
    assert result.base_status is BaseQualityStatus.PROVISIONAL
    assert result.final_status is FinalQualityStatus.WATCH
    assert "BASE_PARTIAL_COVERAGE=53/70" in result.warnings


def test_negative_cfo_marks_dependent_ratios_not_applicable() -> None:
    result = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(
            operating_cash_flow_ttm=-10.0,
            free_cash_flow_ttm=-12.0,
            net_income_ttm=-8.0,
        ),
        SIGNAL_DATE,
    )

    warnings = ";".join(result.warnings)
    assert "EXTENDED_FAILED=positive_cfo,positive_fcf" in warnings
    assert "fcf_to_cfo(BLOCKED_BY_NON_POSITIVE_CFO)" in warnings
    assert "capex_to_cfo(BLOCKED_BY_NON_POSITIVE_CFO)" in warnings
    assert "cash_conversion(BLOCKED_BY_NON_POSITIVE_TTM_NET_INCOME)" in warnings
    extended_missing = next(
        (item for item in result.warnings if item.startswith("EXTENDED_MISSING=")),
        "",
    )
    assert "fcf_to_cfo" not in extended_missing
    assert "capex_to_cfo" not in extended_missing


def test_low_entry_v5_is_discoverable_and_blocks_high_volume_weak_close() -> None:
    assert "low_entry_score_v5" in available_strategies()
    assert get_strategy("low_entry_score_v5").name == "low_entry_score_v5"

    data = attach_low_entry_v4_context(
        _panic_history(),
        annual_fundamentals=_annual_fundamentals(),
        fundamental_snapshot=_quality_snapshot(),
        signal_date=SIGNAL_DATE,
    )
    latest = LowEntryScoreV5Strategy().run(data).iloc[-1]

    assert latest["LES v5 Final Quality Status"] == "QUALITY_CONFIRMED"
    assert latest["Market Data Gate"] == "PASS"
    assert latest["Valuation Model Status"] == "COMPLETE"
    assert latest["Valuation Gate"] != "UNKNOWN"
    assert latest["Base Case Fair Value"] > 0
    assert latest["Valuation Model Version"].startswith("les-v5-valuation-")
    assert latest["Falling Knife State"] == FallingKnifeState.PANIC_WATCH.value
    assert "HIGH_VOLUME_WEAK_CLOSE" in latest["Temporary Falling Knife Blocks"]
    assert latest["低點決策"] == "PANIC_WATCH"


def test_v5_reports_no_model_for_a_preferred_security() -> None:
    data = attach_low_entry_v4_context(
        _panic_history(),
        annual_fundamentals=_annual_fundamentals(),
        fundamental_snapshot=_quality_snapshot(
            market_symbol="3036A",
            issuer_symbol="3036",
            security_type="PREFERRED",
        ),
        signal_date=SIGNAL_DATE,
    )

    latest = LowEntryScoreV5Strategy().run(data).iloc[-1]

    assert latest["Security Type"] == "PREFERRED"
    assert latest["Issuer Symbol"] == "3036"
    assert latest["低點決策"] == "NO_MODEL"
    assert latest["Position Fraction"] == 0
    assert latest["建議股數"] == 0
    assert "MODEL_UNSUPPORTED_FOR_SECURITY_TYPE" in latest["低點理由"]


def test_v5_strategy_uses_composition_instead_of_version_inheritance() -> None:
    assert not issubclass(LowEntryScoreV5Strategy, LowEntryScoreV4Strategy)
    strategy = LowEntryScoreV5Strategy()
    components = strategy.pipeline.components
    assert len(
        {
            id(components.market),
            id(components.fundamentals),
            id(components.panic),
            id(components.scoring),
            id(components.decision),
            id(components.report),
        }
    ) == 6


def test_v5_strategy_can_replace_the_pipeline_without_subclassing() -> None:
    class StubPipeline:
        def run(self, data: pd.DataFrame) -> pd.DataFrame:
            return data.assign(**{"低點決策": "CUSTOM_PIPELINE"})

        def run_with_features(self, data, features):
            del features
            return self.run(data)

    strategy = LowEntryScoreV5Strategy(
        pipeline=StubPipeline(),  # type: ignore[arg-type]
    )

    result = strategy.run(pd.DataFrame({"Close": [100.0]}))

    assert result.iloc[-1]["低點決策"] == "CUSTOM_PIPELINE"


def test_next_day_confirmation_requires_price_risk_and_positive_clv() -> None:
    strategy = LowEntryScoreV5Strategy()
    index = pd.RangeIndex(2)
    indicators = pd.DataFrame(
        {
            "high": [100.0, 101.0],
            "low": [90.0, 90.0],
            "clv": [0.6, 0.2],
        },
        index=index,
    )
    plan = pd.DataFrame({"reward_risk": [2.1, 2.2]}, index=index)
    state = pd.Series(
        [FallingKnifeState.REVERSAL_WATCH.value, FallingKnifeState.CLEAR.value],
        index=index,
    )

    confirmed = strategy._next_day_confirmation(indicators, plan, state)

    assert confirmed.tolist() == [False, True]


def test_v5_decision_component_can_be_replaced_independently() -> None:
    default = LowEntryScoreV5Strategy()

    class NoConfirmationDecision:
        def __init__(self, delegate) -> None:
            self.delegate = delegate

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def next_day_confirmation(self, indicators, plan, falling_state):
            del plan, falling_state
            return pd.Series(False, index=indicators.index)

    replacement = NoConfirmationDecision(
        default.pipeline.components.decision
    )
    components = replace(
        default.pipeline.components,
        decision=replacement,  # type: ignore[arg-type]
    )
    strategy = LowEntryScoreV5Strategy(components=components)
    indicators = pd.DataFrame(
        {
            "high": [100.0, 101.0],
            "low": [90.0, 91.0],
            "clv": [0.6, 0.5],
        }
    )
    plan = pd.DataFrame({"reward_risk": [2.1, 2.2]})
    state = pd.Series(
        [FallingKnifeState.REVERSAL_WATCH.value, FallingKnifeState.CLEAR.value]
    )

    confirmed = strategy._next_day_confirmation(indicators, plan, state)

    assert confirmed.tolist() == [False, False]


def test_v5_valuation_normalizes_crawler_price_scale() -> None:
    unscaled = attach_low_entry_v4_context(
        _panic_history(),
        annual_fundamentals=_annual_fundamentals(),
        fundamental_snapshot=_quality_snapshot(),
        signal_date=SIGNAL_DATE,
    )
    scaled_history = _panic_history()
    for column in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        scaled_history[column] *= 1000.0
    scaled = attach_low_entry_v4_context(
        scaled_history,
        annual_fundamentals=_annual_fundamentals(),
        fundamental_snapshot=_quality_snapshot(),
        signal_date=SIGNAL_DATE,
        price_scale=1000.0,
    )

    unscaled_latest = LowEntryScoreV5Strategy().run(unscaled).iloc[-1]
    scaled_latest = LowEntryScoreV5Strategy().run(scaled).iloc[-1]

    assert scaled_latest["Valuation Gate"] == unscaled_latest["Valuation Gate"]
    assert scaled_latest["Base Case Fair Value"] == pytest.approx(
        unscaled_latest["Base Case Fair Value"]
    )


def test_v5_uses_odd_lots_and_reprices_breadth_entry_above_signal_high() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.config.lot_size == 1
    assert strategy.rule_engine is not None
    index = pd.RangeIndex(1)
    indicators = pd.DataFrame(
        {
            "high": [78_000.0],
            "low": [74_000.0],
            "atr14": [5_006.564528],
            "amount_ma20": [50_000_000.0],
            "spread": [0.005],
            "taiex": [24_000.0],
            "taiex_ema20": [25_000.0],
            "taiex_r5": [-0.04],
            "rs20": [-0.10],
            "price_position60": [0.10],
        },
        index=index,
    )
    plan = pd.DataFrame(
        {
            "entry": [78_000.0],
            "stop": [68_598.030642],
            "target": [118_000.0],
            "risk": [9_401.969358],
            "reward_risk": [4.254428],
            "shares": [531.0],
            "order_amount": [41_418.0],
        },
        index=index,
    )
    context = {PRICE_SCALE_ATTR: 1000.0, ACCOUNT_EQUITY_ATTR: 1_000_000.0}

    repriced = strategy.rule_engine.build_broad_rally_plan(
        indicators,
        plan,
        context,
    )
    risk_gate, reasons = strategy.rule_engine.evaluate_broad_rally_risk_gate(
        indicators,
        repriced,
        pd.Series("PASS", index=index),
    )
    fraction = pd.Series(
        strategy.config.breadth_rally_position_fraction,
        index=index,
    )
    shares = strategy.rule_engine.cap_position_shares(
        repriced,
        fraction,
        context,
    )

    assert repriced.loc[0, "entry"] == pytest.approx(78_100.0)
    assert repriced.loc[0, "stop"] == pytest.approx(
        74_000.0 - 0.3 * 5_006.564528
    )
    assert repriced.loc[0, "reward_risk"] > 7.0
    assert risk_gate.loc[0] == "PASS"
    assert reasons.loc[0] == "PASS"
    assert shares.loc[0] == 111


def test_v5_breadth_liquidity_uses_largest_authorized_fraction() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    index = pd.RangeIndex(1)
    indicators = pd.DataFrame(
        {
            "atr14": [5.0],
            "amount_ma20": [30_000_000.0],
            "spread": [0.005],
            "taiex": [24_000.0],
            "taiex_ema20": [23_500.0],
            "taiex_r5": [0.01],
            "rs20": [0.02],
            "price_position60": [0.30],
            "drawdown60": [-0.10],
        },
        index=index,
    )
    # Full risk-sized order is above the 5% liquidity limit (1.5m), while the
    # largest breadth position (1/4) is executable.  The gate must evaluate
    # the order that can actually be authorized, not a never-traded full size.
    plan = pd.DataFrame(
        {
            "entry": [100.0],
            "stop": [95.0],
            "target": [110.0],
            "risk": [5.0],
            "reward_risk": [2.0],
            "shares": [20_000.0],
            "order_amount": [2_000_000.0],
        },
        index=index,
    )

    gate, reasons = strategy.rule_engine.evaluate_broad_rally_risk_gate(
        indicators,
        plan,
        pd.Series("PASS", index=index),
    )

    assert gate.loc[0] == "PASS"
    assert reasons.loc[0] == "PASS"


def test_v5_breadth_starter_uses_guarded_25m_turnover_floor() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    index = pd.RangeIndex(1)
    indicators = pd.DataFrame(
        {
            "atr14": [5.0],
            "amount_ma20": [27_000_000.0],
            "spread": [0.005],
            "taiex": [24_000.0],
            "taiex_ema20": [23_500.0],
            "taiex_r5": [0.01],
            "rs20": [0.02],
            "price_position60": [0.30],
            "drawdown60": [-0.10],
        },
        index=index,
    )
    plan = pd.DataFrame(
        {
            "entry": [100.0],
            "stop": [95.0],
            "target": [110.0],
            "risk": [5.0],
            "reward_risk": [2.0],
            "shares": [10_000.0],
            "order_amount": [1_000_000.0],
        },
        index=index,
    )
    market_gate = pd.Series("PASS", index=index)

    broad_gate, broad_reasons = (
        strategy.rule_engine.evaluate_broad_rally_risk_gate(
            indicators,
            plan,
            market_gate,
        )
    )
    normal_gate, normal_reasons = strategy.rule_engine.evaluate_risk_gate(
        indicators,
        plan,
        market_gate,
    )

    assert broad_gate.loc[0] == "PASS"
    assert broad_reasons.loc[0] == "PASS"
    assert normal_gate.loc[0] == "FAIL"
    assert "LIQUIDITY_FAILED" in normal_reasons.loc[0]


def test_v5_breadth_starter_accepts_1_6_rr_but_normal_gate_stays_1_7() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    index = pd.RangeIndex(1)
    indicators = pd.DataFrame(
        {
            "atr14": [5.0],
            "amount_ma20": [50_000_000.0],
            "spread": [0.005],
            "taiex": [24_000.0],
            "taiex_ema20": [23_500.0],
            "taiex_r5": [0.01],
            "rs20": [0.02],
            "price_position60": [0.30],
            "drawdown60": [-0.10],
        },
        index=index,
    )
    plan = pd.DataFrame(
        {
            "entry": [100.0],
            "stop": [95.0],
            "target": [108.3],
            "risk": [5.0],
            "reward_risk": [1.66],
            "shares": [1_000.0],
            "order_amount": [100_000.0],
        },
        index=index,
    )
    market_gate = pd.Series("PASS", index=index)

    broad_gate, _ = strategy.rule_engine.evaluate_broad_rally_risk_gate(
        indicators,
        plan,
        market_gate,
    )
    normal_gate, _ = strategy.rule_engine.evaluate_risk_gate(
        indicators,
        plan,
        market_gate,
    )

    assert strategy.config.breadth_rally_minimum_reward_risk == pytest.approx(1.6)
    assert strategy.config.minimum_reward_risk == pytest.approx(1.7)
    assert broad_gate.loc[0] == "PASS"
    assert normal_gate.loc[0] == "FAIL"


def test_v5_broad_rally_regime_requires_final_date_aligned_breadth() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    data = pd.DataFrame(
        {"Date": pd.to_datetime(["2026-07-30", "2026-07-31"])}
    )
    indicators = pd.DataFrame({"taiex_r1": [0.01, 0.0798]})
    snapshot = {
        "as_of": "2026-07-31",
        "universe": "ELEC",
        "coverage": 1.0,
        "valid_return_count": 456,
        "advancer_ratio": 0.94956,
        "median_return": 0.0658,
        "source_final": True,
        "taiex_return": 0.0798,
    }

    regime = strategy.rule_engine._evaluate_breadth_rally_regime(
        data,
        indicators,
        breadth_snapshot=snapshot,
        market_data_through="2026-07-31",
        market_session_final=True,
    )
    stale = strategy.rule_engine._evaluate_breadth_rally_regime(
        data,
        indicators,
        breadth_snapshot=snapshot,
        market_data_through="2026-07-30",
        market_session_final=True,
    )

    assert regime.tolist() == [False, True]
    assert stale.tolist() == [False, False]


def test_v5_elec_breadth_accepts_three_quarters_participation() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    data = pd.DataFrame({"Date": pd.to_datetime(["2026-08-04"])})
    indicators = pd.DataFrame({"taiex_r1": [0.01]})

    def snapshot(advancers: float) -> dict[str, object]:
        return {
            "as_of": "2026-08-04",
            "universe": "ELEC",
            "coverage": 0.99,
            "valid_return_count": 438,
            "advancer_ratio": advancers,
            "median_return": 0.018,
            "source_final": True,
            "taiex_return": 0.01,
        }

    accepted = strategy.rule_engine._evaluate_breadth_rally_regime(
        data,
        indicators,
        breadth_snapshot=snapshot(0.751),
        market_data_through="2026-08-04",
        market_session_final=True,
    )
    rejected = strategy.rule_engine._evaluate_breadth_rally_regime(
        data,
        indicators,
        breadth_snapshot=snapshot(0.749),
        market_data_through="2026-08-04",
        market_session_final=True,
    )

    assert accepted.iloc[-1]
    assert not rejected.iloc[-1]


def test_v5_recent_breadth_memory_separates_rally_and_selloff() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    data = pd.DataFrame({"Date": pd.to_datetime(["2026-08-07"])})
    indicators = pd.DataFrame(index=data.index)

    def snapshot(
        date: str,
        advancers: float,
        median: float,
        taiex: float,
    ) -> dict[str, object]:
        return {
            "as_of": date,
            "universe": "SEMI",
            "coverage": 1.0,
            "valid_return_count": 50,
            "advancer_ratio": advancers,
            "median_return": median,
            "source_final": True,
            "taiex_return": taiex,
        }

    rally_history = {
        "2026-08-04": snapshot("2026-08-04", 0.80, 0.02, 0.01),
        "2026-08-05": snapshot("2026-08-05", 0.78, 0.018, 0.008),
        "2026-08-06": snapshot("2026-08-06", 0.52, 0.002, -0.001),
    }
    rally, rally_count = strategy.rule_engine._evaluate_recent_breadth_regime(
        data,
        indicators,
        breadth_history=rally_history,
        market_data_through="2026-08-07",
        direction="rally",
    )
    selloff, selloff_count = (
        strategy.rule_engine._evaluate_recent_breadth_regime(
            data,
            indicators,
            breadth_history=rally_history,
            market_data_through="2026-08-07",
            direction="selloff",
        )
    )

    assert rally.tolist() == [True]
    assert rally_count.tolist() == [2]
    assert selloff.tolist() == [False]
    assert selloff_count.tolist() == [0]


def test_v5_recent_breadth_memory_accepts_breadth_led_recovery() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    data = pd.DataFrame({"Date": pd.to_datetime(["2026-08-07"])})
    indicators = pd.DataFrame(index=data.index)

    def snapshot(
        date: str,
        advancers: float,
        median: float,
        taiex: float,
    ) -> dict[str, object]:
        return {
            "as_of": date,
            "universe": "SEMI",
            "coverage": 1.0,
            "valid_return_count": 50,
            "advancer_ratio": advancers,
            "median_return": median,
            "source_final": True,
            "taiex_return": taiex,
        }

    history = {
        "2026-08-03": snapshot("2026-08-03", 0.86, 0.026, -0.012),
        "2026-08-04": snapshot("2026-08-04", 0.85, 0.020, -0.007),
        "2026-08-05": snapshot("2026-08-05", 0.78, 0.018, -0.003),
        "2026-08-06": snapshot("2026-08-06", 0.55, 0.003, -0.002),
    }

    regime, count = strategy.rule_engine._evaluate_recent_breadth_regime(
        data,
        indicators,
        breadth_history=history,
        market_data_through="2026-08-07",
        direction="rally",
    )

    assert regime.tolist() == [True]
    assert count.tolist() == [3]


def test_v5_signal_day_cannot_create_its_own_recent_regime() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    data = pd.DataFrame({"Date": pd.to_datetime(["2026-08-07"])})
    indicators = pd.DataFrame(index=data.index)
    signal_day_only = {
        "2026-08-07": {
            "as_of": "2026-08-07",
            "universe": "SEMI",
            "coverage": 1.0,
            "valid_return_count": 50,
            "advancer_ratio": 0.95,
            "median_return": 0.07,
            "source_final": True,
            "taiex_return": 0.05,
        }
    }

    regime, count = strategy.rule_engine._evaluate_recent_breadth_regime(
        data,
        indicators,
        breadth_history=signal_day_only,
        market_data_through="2026-08-07",
        direction="rally",
    )

    assert regime.tolist() == [False]
    assert count.tolist() == [0]


def test_v5_breadth_trigger_cannot_offset_temporary_falling_knife_block() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    quality = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    base = {
        "pe": 10.0,
        "pe_median60": 12.0,
        "drawdown60": -0.20,
        "price_position60": 0.10,
        "close": 77.6,
        "previous_high": 76.5,
        "previous_low20": 75.0,
        "clv": 0.636,
        "low": 75.8,
        "open": 77.0,
        "new_low_count3": 0.0,
        "volume_ratio": 0.45,
        "weak_clv3": False,
        "ema5_slope": -0.1,
        "r5": -0.05,
        "atr_percent": 0.0645,
        "drawdown20": -0.16,
        "true_range": 2.2,
        "atr14": 5.0,
        "previous_midpoint": 76.0,
        "lower_shadow_ratio": 0.55,
        "broad_rally_regime": True,
        "cmf20": -0.21,
    }
    previous = dict(base)
    previous.update(
        {
            "close": 70.0,
            "open": 70.5,
            "broad_rally_regime": False,
            "drawdown20": -0.10,
            "price_position60": 0.25,
        }
    )
    indicators = pd.DataFrame([previous, base])
    valuation = pd.Series(
        [ValuationStatus.PASS.value, ValuationStatus.PASS.value]
    )

    clear = strategy.rule_engine.evaluate_panic(
        indicators,
        quality,
        valuation,
    )
    blocked_indicators = indicators.assign(weak_clv3=True)
    blocked = strategy.rule_engine.evaluate_panic(
        blocked_indicators,
        quality,
        valuation,
    )

    assert bool(clear["broad_rally_trigger"].iloc[-1]) is True
    assert clear["block_text"].iloc[-1] == "NONE"
    assert bool(blocked["broad_rally_trigger"].iloc[-1]) is False
    assert "THREE_WEAK_CLOSES" in blocked["block_text"].iloc[-1]


def test_v5_broad_rally_path_emits_only_conditional_buy_small() -> None:
    strategy = LowEntryScoreV5Strategy(LowEntryScoreV5Config())
    assert strategy.rule_engine is not None
    quality = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    index = pd.RangeIndex(1)

    decision = strategy.rule_engine.resolve_decision(
        score=pd.Series([30.0], index=index),
        quality=quality,
        valuation=pd.Series([ValuationStatus.PASS.value], index=index),
        falling_state=pd.Series(
            [FallingKnifeState.REVERSAL_WATCH.value],
            index=index,
        ),
        next_day_confirmation=pd.Series([False], index=index),
        trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["FAIL"], index=index),
        market_gate=pd.Series(["PASS"], index=index),
        required_score=pd.Series([85.0], index=index),
        broad_rally_trigger=pd.Series([True], index=index),
        broad_rally_risk_gate=pd.Series(["PASS"], index=index),
        raw_score=pd.Series([40.0], index=index),
    )
    fraction = strategy.rule_engine.position_fraction(
        decision,
        quality,
        pd.Series([True], index=index),
    )

    assert decision.loc[0] == "BUY_SMALL"
    assert fraction.loc[0] == pytest.approx(1 / 8)


def test_v5_strong_breadth_reversal_can_emit_conditional_buy() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    confirmed = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    fallback = replace(
        confirmed,
        base_status=BaseQualityStatus.PROVISIONAL,
        final_status=FinalQualityStatus.WATCH,
        final_score=80.86,
        base_data_confidence=79.7,
        base_available_weight=53.0,
        extended_data_confidence=83.8,
        critical_field_coverage=100.0,
        warnings=("MOPS_POINT_IN_TIME_BASE_FALLBACK",),
    )
    index = pd.RangeIndex(1)
    common = {
        "score": pd.Series([37.0], index=index),
        "quality": fallback,
        "valuation": pd.Series([ValuationStatus.PASS.value], index=index),
        "falling_state": pd.Series(
            [FallingKnifeState.CLEAR.value],
            index=index,
        ),
        "next_day_confirmation": pd.Series([False], index=index),
        "trigger": pd.Series([False], index=index),
        "market_gate": pd.Series(["PASS"], index=index),
        "required_score": pd.Series([80.0], index=index),
        "broad_rally_trigger": pd.Series([True], index=index),
        "broad_rally_risk_gate": pd.Series(["PASS"], index=index),
        "raw_score": pd.Series([47.0], index=index),
        "broad_rally_full_buy_trigger": pd.Series([True], index=index),
    }

    decision = strategy.rule_engine.resolve_decision(
        **common,
        risk_gate=pd.Series(["PASS"], index=index),
    )
    fraction = strategy.rule_engine.position_fraction(
        decision,
        fallback,
        pd.Series([False], index=index),
        pd.Series([True], index=index),
    )
    risk_failed = strategy.rule_engine.resolve_decision(
        **common,
        risk_gate=pd.Series(["FAIL"], index=index),
    )

    assert decision.loc[0] == "BUY"
    assert fraction.loc[0] == pytest.approx(1 / 4)
    assert risk_failed.loc[0] == "BUY_SMALL"


def test_v5_balanced_profile_uses_75_point_entry_and_1_7_reward_risk() -> None:
    config = LowEntryScoreV5Config()

    assert config.confirmed_buy_score == 75
    assert config.provisional_buy_score == 80
    assert config.maximum_stop_atr == pytest.approx(1.8)
    assert config.minimum_reward_risk == pytest.approx(1.7)
    assert config.breadth_rally_minimum_fallback_base_weight == 50
    assert config.breadth_rally_strong_close_minimum_volume_ratio == pytest.approx(
        0.30
    )
    assert config.breadth_rally_full_buy_score_threshold == 35
    assert config.breadth_rally_full_buy_minimum_raw_score == 45
    assert config.breadth_rally_full_buy_minimum_quality_score == 80
    assert config.breadth_rally_full_buy_position_fraction == pytest.approx(1 / 4)
    assert config.breadth_absorption_score_threshold == 25
    assert config.breadth_absorption_minimum_raw_score == 35
    assert config.breadth_absorption_minimum_cmf20 == pytest.approx(-0.45)
    assert config.breadth_absorption_minimum_volume_ratio == pytest.approx(0.40)


def test_v5_balanced_trigger_accepts_price_and_obv_confirmation() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    indicators = pd.DataFrame(
        {
            "close": [99.0, 101.0],
            "previous_high": [100.0, 102.0],
            "ema5": [100.0, 100.0],
            "vwap20": [100.0, 100.5],
            "higher_low": [False, False],
            "ema5_slope": [-0.1, 0.1],
            "clv": [-0.1, 0.3],
            "cmf20_improving": [False, False],
            "normalized_obv_slope5": [-0.1, 0.2],
            "volume_ratio": [0.5, 0.5],
        }
    )

    trigger = strategy.rule_engine.evaluate_trigger(indicators)

    assert trigger.tolist() == [False, True]


def test_v5_broad_rally_accepts_depressed_recovery_without_panic_shock() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    quality = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    previous = {
        "pe": 11.0,
        "pe_median60": 12.0,
        "drawdown60": -0.08,
        "price_position60": 0.35,
        "close": 76.0,
        "previous_high": 76.5,
        "previous_low20": 74.0,
        "clv": 0.1,
        "low": 75.0,
        "open": 76.2,
        "new_low_count3": 0.0,
        "volume_ratio": 0.7,
        "weak_clv3": False,
        "ema5_slope": 0.05,
        "r5": 0.01,
        "atr_percent": 0.06,
        "drawdown20": -0.05,
        "true_range": 3.0,
        "atr14": 5.0,
        "previous_midpoint": 75.5,
        "lower_shadow_ratio": 0.2,
        "broad_rally_regime": False,
        "cmf20": -0.10,
    }
    recovery = dict(previous)
    recovery.update(
        {
            "close": 77.6,
            "open": 76.8,
            "low": 76.0,
            "clv": 0.65,
            "volume_ratio": 0.31,
            "price_position60": 0.40,
            "drawdown60": -0.06,
            "previous_high": 77.0,
            "previous_midpoint": 76.5,
            "lower_shadow_ratio": 0.3,
            "broad_rally_regime": True,
        }
    )
    indicators = pd.DataFrame([previous, recovery])
    valuation = pd.Series(
        [ValuationStatus.PASS.value, ValuationStatus.PASS.value]
    )

    result = strategy.rule_engine.evaluate_panic(
        indicators,
        quality,
        valuation,
    )

    assert result["panic_count"].iloc[-1] == 0
    assert bool(result["broad_reversal_setup"].iloc[-1]) is True
    assert bool(result["broad_rally_trigger"].iloc[-1]) is True


def test_v5_absorption_trigger_accepts_deep_reversal_before_cmf_recovers() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    quality = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    previous = {
        "pe": 11.0,
        "pe_median60": 12.0,
        "drawdown60": -0.25,
        "price_position60": 0.12,
        "close": 90.0,
        "previous_high": 91.0,
        "previous_low20": 88.0,
        "clv": 0.1,
        "low": 89.0,
        "open": 90.2,
        "new_low_count3": 0.0,
        "volume_ratio": 0.6,
        "weak_clv3": False,
        "ema5_slope": 0.05,
        "r5": -0.05,
        "atr_percent": 0.05,
        "drawdown20": -0.16,
        "true_range": 4.0,
        "atr14": 5.0,
        "previous_midpoint": 89.5,
        "lower_shadow_ratio": 0.2,
        "broad_rally_regime": False,
        "cmf20": -0.40,
    }
    absorption = dict(previous)
    absorption.update(
        {
            "close": 91.0,
            "open": 90.5,
            "low": 89.5,
            "clv": 0.40,
            "volume_ratio": 0.50,
            "price_position60": 0.10,
            "drawdown60": -0.30,
            "previous_high": 92.0,
            "previous_midpoint": 90.5,
            "drawdown20": -0.20,
            "broad_rally_regime": True,
        }
    )
    indicators = pd.DataFrame([previous, absorption])
    valuation = pd.Series(
        [ValuationStatus.PASS.value, ValuationStatus.PASS.value]
    )

    result = strategy.rule_engine.evaluate_panic(
        indicators,
        quality,
        valuation,
    )

    assert result["panic_count"].iloc[-1] >= 2
    assert result["reversal_count"].iloc[-1] >= 3
    assert bool(result["broad_rally_trigger"].iloc[-1]) is False
    assert bool(result["broad_rally_absorption_trigger"].iloc[-1]) is True


def test_v5_absorption_path_is_buy_small_and_never_bypasses_valuation() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    confirmed = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    fallback = replace(
        confirmed,
        base_status=BaseQualityStatus.PROVISIONAL,
        final_status=FinalQualityStatus.WATCH,
        final_score=90.0,
        base_data_confidence=79.7,
        base_available_weight=53.0,
        extended_data_confidence=83.8,
        critical_field_coverage=100.0,
        warnings=("MOPS_POINT_IN_TIME_BASE_FALLBACK",),
    )
    index = pd.RangeIndex(1)
    common = {
        "score": pd.Series([27.0], index=index),
        "quality": fallback,
        "falling_state": pd.Series(
            [FallingKnifeState.CLEAR.value],
            index=index,
        ),
        "next_day_confirmation": pd.Series([False], index=index),
        "trigger": pd.Series([False], index=index),
        "risk_gate": pd.Series(["PASS"], index=index),
        "market_gate": pd.Series(["PASS"], index=index),
        "required_score": pd.Series([80.0], index=index),
        "broad_rally_trigger": pd.Series([False], index=index),
        "broad_rally_risk_gate": pd.Series(["PASS"], index=index),
        "raw_score": pd.Series([37.0], index=index),
        "broad_rally_full_buy_trigger": pd.Series([False], index=index),
        "broad_rally_absorption_trigger": pd.Series([True], index=index),
    }

    decision = strategy.rule_engine.resolve_decision(
        **common,
        valuation=pd.Series([ValuationStatus.PASS.value], index=index),
    )
    valuation_failed = strategy.rule_engine.resolve_decision(
        **common,
        valuation=pd.Series([ValuationStatus.FAIL.value], index=index),
    )

    assert decision.loc[0] == "BUY_SMALL"
    assert valuation_failed.loc[0] == "WATCH_ONLY"


def test_v5_recent_broad_rally_retest_can_emit_conditional_buy_small() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    quality = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    previous = {
        "pe": 11.0,
        "pe_median60": 12.0,
        "drawdown60": -0.10,
        "price_position60": 0.50,
        "close": 100.0,
        "previous_high": 101.0,
        "previous_low20": 90.0,
        "clv": 0.1,
        "low": 98.0,
        "open": 99.0,
        "new_low_count3": 0.0,
        "volume_ratio": 0.7,
        "weak_clv3": False,
        "ema5_slope": 0.1,
        "r5": 0.03,
        "atr_percent": 0.05,
        "drawdown20": -0.08,
        "true_range": 4.0,
        "atr14": 5.0,
        "previous_midpoint": 99.0,
        "lower_shadow_ratio": 0.2,
        "broad_rally_regime": False,
        "recent_broad_selloff_regime": False,
        "recent_broad_rally_regime": False,
        "cmf20": -0.10,
    }
    retest = dict(previous)
    retest.update(
        {
            "close": 99.0,
            "open": 98.5,
            "low": 97.5,
            "clv": 0.40,
            "volume_ratio": 0.80,
            "previous_high": 102.0,
            "previous_midpoint": 99.0,
            "lower_shadow_ratio": 0.50,
            "recent_broad_rally_regime": True,
        }
    )
    indicators = pd.DataFrame([previous, retest])
    valuation = pd.Series(
        [ValuationStatus.PASS.value, ValuationStatus.PASS.value]
    )

    panic = strategy.rule_engine.evaluate_panic(
        indicators,
        quality,
        valuation,
    )

    assert panic["block_text"].iloc[-1] == "NONE"
    assert bool(panic["recent_rally_retest_trigger"].iloc[-1]) is True

    index = pd.RangeIndex(1)
    decision = strategy.rule_engine.resolve_decision(
        score=pd.Series([30.0], index=index),
        quality=quality,
        valuation=pd.Series([ValuationStatus.PASS.value], index=index),
        falling_state=pd.Series([FallingKnifeState.CLEAR.value], index=index),
        next_day_confirmation=pd.Series([False], index=index),
        trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["FAIL"], index=index),
        market_gate=pd.Series(["PASS"], index=index),
        required_score=pd.Series([75.0], index=index),
        broad_rally_trigger=pd.Series([False], index=index),
        broad_rally_risk_gate=pd.Series(["PASS"], index=index),
        raw_score=pd.Series([40.0], index=index),
        recent_rally_retest_trigger=pd.Series([True], index=index),
    )

    assert decision.loc[0] == "BUY_SMALL"


def test_v5_recent_rally_retest_accepts_only_controlled_negative_clv() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    quality = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    previous = {
        "pe": 11.0,
        "pe_median60": 12.0,
        "drawdown60": -0.10,
        "price_position60": 0.50,
        "close": 100.0,
        "previous_high": 101.0,
        "previous_low20": 90.0,
        "clv": 0.1,
        "low": 98.0,
        "open": 99.0,
        "new_low_count3": 0.0,
        "volume_ratio": 0.7,
        "weak_clv3": False,
        "ema5_slope": 0.1,
        "r5": 0.03,
        "atr_percent": 0.05,
        "drawdown20": -0.08,
        "true_range": 4.0,
        "atr14": 5.0,
        "previous_midpoint": 99.0,
        "lower_shadow_ratio": 0.2,
        "broad_rally_regime": False,
        "recent_broad_selloff_regime": False,
        "recent_broad_rally_regime": False,
        "cmf20": -0.10,
    }
    controlled = dict(previous)
    controlled.update(
        {
            "close": 99.0,
            "open": 98.5,
            "low": 97.5,
            "clv": -0.29,
            "volume_ratio": 0.80,
            "previous_high": 102.0,
            "previous_midpoint": 99.0,
            "lower_shadow_ratio": 0.50,
            "recent_broad_rally_regime": True,
        }
    )
    weak = dict(controlled, clv=-0.31)
    valuation = pd.Series(
        [ValuationStatus.PASS.value, ValuationStatus.PASS.value]
    )

    controlled_result = strategy.rule_engine.evaluate_panic(
        pd.DataFrame([previous, controlled]),
        quality,
        valuation,
    )
    weak_result = strategy.rule_engine.evaluate_panic(
        pd.DataFrame([previous, weak]),
        quality,
        valuation,
    )

    assert bool(
        controlled_result["recent_rally_retest_trigger"].iloc[-1]
    ) is True
    assert bool(weak_result["recent_rally_retest_trigger"].iloc[-1]) is False


def test_v5_recent_rally_retest_accepts_guarded_quality_fallback_starter() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    confirmed = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    fallback = replace(
        confirmed,
        base_status=BaseQualityStatus.PROVISIONAL,
        final_status=FinalQualityStatus.WATCH,
        final_score=77.0,
        base_data_confidence=80.0,
        base_available_weight=60.0,
        extended_status=ExtendedQualityStatus.PASS,
        extended_data_confidence=83.0,
        critical_field_coverage=100.0,
        warnings=("MOPS_POINT_IN_TIME_BASE_FALLBACK",),
        hard_rejects=(),
    )
    index = pd.RangeIndex(1)

    decision = strategy.rule_engine.resolve_decision(
        score=pd.Series([30.0], index=index),
        quality=fallback,
        valuation=pd.Series([ValuationStatus.PASS.value], index=index),
        falling_state=pd.Series([FallingKnifeState.CLEAR.value], index=index),
        next_day_confirmation=pd.Series([False], index=index),
        trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["FAIL"], index=index),
        market_gate=pd.Series(["PASS"], index=index),
        required_score=pd.Series([80.0], index=index),
        broad_rally_trigger=pd.Series([False], index=index),
        broad_rally_risk_gate=pd.Series(["PASS"], index=index),
        raw_score=pd.Series([40.0], index=index),
        recent_rally_retest_trigger=pd.Series([True], index=index),
    )

    assert strategy.rule_engine._broad_rally_quality_eligible(fallback) is True
    assert (
        strategy.rule_engine._broad_rally_full_buy_quality_eligible(fallback)
        is False
    )
    assert decision.loc[0] == "BUY_SMALL"


def test_v5_recent_rally_retest_cannot_bypass_valuation_or_veto() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    quality = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    index = pd.RangeIndex(1)
    common = {
        "score": pd.Series([30.0], index=index),
        "quality": quality,
        "next_day_confirmation": pd.Series([False], index=index),
        "trigger": pd.Series([False], index=index),
        "risk_gate": pd.Series(["FAIL"], index=index),
        "market_gate": pd.Series(["PASS"], index=index),
        "required_score": pd.Series([75.0], index=index),
        "broad_rally_trigger": pd.Series([False], index=index),
        "broad_rally_risk_gate": pd.Series(["PASS"], index=index),
        "raw_score": pd.Series([40.0], index=index),
        "recent_rally_retest_trigger": pd.Series([True], index=index),
    }

    valuation_failed = strategy.rule_engine.resolve_decision(
        **common,
        valuation=pd.Series([ValuationStatus.FAIL.value], index=index),
        falling_state=pd.Series([FallingKnifeState.CLEAR.value], index=index),
    )
    vetoed_common = dict(common)
    vetoed_common["recent_rally_retest_trigger"] = pd.Series(
        [False], index=index
    )
    vetoed = strategy.rule_engine.resolve_decision(
        **vetoed_common,
        valuation=pd.Series([ValuationStatus.PASS.value], index=index),
        falling_state=pd.Series(
            [FallingKnifeState.PANIC_WATCH.value], index=index
        ),
    )

    assert valuation_failed.loc[0] != "BUY_SMALL"
    assert vetoed.loc[0] == "PANIC_WATCH"


def test_v5_recent_rally_reclaim_arms_weak_close_only_after_confirmation() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    quality = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    previous = {
        "pe": 11.0,
        "pe_median60": 12.0,
        "drawdown60": -0.20,
        "price_position60": 0.30,
        "close": 100.0,
        "previous_high": 101.0,
        "previous_low20": 90.0,
        "clv": 0.1,
        "low": 98.0,
        "open": 99.0,
        "new_low_count3": 0.0,
        "volume_ratio": 0.7,
        "weak_clv3": False,
        "ema5_slope": 0.1,
        "r5": 0.03,
        "atr_percent": 0.05,
        "drawdown20": -0.08,
        "true_range": 4.0,
        "atr14": 10.0,
        "previous_midpoint": 99.0,
        "lower_shadow_ratio": 0.2,
        "broad_rally_regime": False,
        "recent_broad_selloff_regime": False,
        "recent_broad_rally_regime": False,
        "cmf20": -0.10,
    }
    reclaim_setup = dict(previous)
    reclaim_setup.update(
        {
            "close": 94.0,
            "open": 99.0,
            "low": 92.0,
            "clv": -0.65,
            "cmf20": 0.05,
            "volume_ratio": 0.70,
            "true_range": 8.0,
            "recent_broad_rally_regime": True,
        }
    )
    valuation = pd.Series(
        [ValuationStatus.PASS.value, ValuationStatus.PASS.value]
    )

    panic = strategy.rule_engine.evaluate_panic(
        pd.DataFrame([previous, reclaim_setup]),
        quality,
        valuation,
    )

    assert panic["block_text"].iloc[-1] == "NONE"
    assert bool(panic["recent_rally_retest_trigger"].iloc[-1]) is False
    assert bool(panic["recent_rally_reclaim_trigger"].iloc[-1]) is True

    index = pd.RangeIndex(1)
    decision = strategy.rule_engine.resolve_decision(
        score=pd.Series([30.0], index=index),
        quality=quality,
        valuation=pd.Series([ValuationStatus.PASS.value], index=index),
        falling_state=pd.Series([FallingKnifeState.CLEAR.value], index=index),
        next_day_confirmation=pd.Series([False], index=index),
        trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["FAIL"], index=index),
        market_gate=pd.Series(["PASS"], index=index),
        required_score=pd.Series([75.0], index=index),
        broad_rally_trigger=pd.Series([False], index=index),
        broad_rally_risk_gate=pd.Series(["PASS"], index=index),
        raw_score=pd.Series([40.0], index=index),
        recent_rally_reclaim_trigger=pd.Series([True], index=index),
    )

    assert decision.loc[0] == "BUY_SMALL"


def test_v5_recent_rally_reclaim_requires_demand_seed_and_all_hard_gates() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    quality = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    previous = {
        "pe": 11.0,
        "pe_median60": 12.0,
        "drawdown60": -0.20,
        "price_position60": 0.30,
        "close": 100.0,
        "previous_high": 101.0,
        "previous_low20": 90.0,
        "clv": 0.1,
        "low": 98.0,
        "open": 99.0,
        "new_low_count3": 0.0,
        "volume_ratio": 0.7,
        "weak_clv3": False,
        "ema5_slope": 0.1,
        "r5": 0.03,
        "atr_percent": 0.05,
        "drawdown20": -0.08,
        "true_range": 4.0,
        "atr14": 10.0,
        "previous_midpoint": 99.0,
        "lower_shadow_ratio": 0.2,
        "broad_rally_regime": False,
        "recent_broad_selloff_regime": False,
        "recent_broad_rally_regime": False,
        "cmf20": -0.10,
    }
    no_demand = dict(previous)
    no_demand.update(
        {
            "close": 94.0,
            "open": 99.0,
            "low": 92.0,
            "clv": -0.65,
            "cmf20": -0.10,
            "volume_ratio": 0.70,
            "true_range": 8.0,
            "recent_broad_rally_regime": True,
        }
    )
    valuation = pd.Series(
        [ValuationStatus.PASS.value, ValuationStatus.PASS.value]
    )
    panic = strategy.rule_engine.evaluate_panic(
        pd.DataFrame([previous, no_demand]),
        quality,
        valuation,
    )

    assert bool(panic["recent_rally_reclaim_trigger"].iloc[-1]) is False

    index = pd.RangeIndex(1)
    common = {
        "score": pd.Series([30.0], index=index),
        "quality": quality,
        "next_day_confirmation": pd.Series([False], index=index),
        "trigger": pd.Series([False], index=index),
        "risk_gate": pd.Series(["FAIL"], index=index),
        "market_gate": pd.Series(["PASS"], index=index),
        "required_score": pd.Series([75.0], index=index),
        "broad_rally_trigger": pd.Series([False], index=index),
        "broad_rally_risk_gate": pd.Series(["PASS"], index=index),
        "raw_score": pd.Series([40.0], index=index),
        "recent_rally_reclaim_trigger": pd.Series([True], index=index),
    }
    valuation_failed = strategy.rule_engine.resolve_decision(
        **common,
        valuation=pd.Series([ValuationStatus.FAIL.value], index=index),
        falling_state=pd.Series([FallingKnifeState.CLEAR.value], index=index),
    )
    vetoed_common = dict(common)
    vetoed_common["recent_rally_reclaim_trigger"] = pd.Series(
        [False], index=index
    )
    vetoed = strategy.rule_engine.resolve_decision(
        **vetoed_common,
        valuation=pd.Series([ValuationStatus.PASS.value], index=index),
        falling_state=pd.Series(
            [FallingKnifeState.PANIC_WATCH.value], index=index
        ),
    )

    assert valuation_failed.loc[0] != "BUY_SMALL"
    assert vetoed.loc[0] == "PANIC_WATCH"


def test_v5_provisional_quality_can_only_use_conditional_breadth_starter() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    confirmed = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    provisional = replace(
        confirmed,
        final_status=FinalQualityStatus.PROVISIONAL,
        final_score=78.0,
        base_data_confidence=80.0,
        extended_data_confidence=72.0,
        critical_field_coverage=75.0,
    )
    index = pd.RangeIndex(1)

    decision = strategy.rule_engine.resolve_decision(
        score=pd.Series([35.0], index=index),
        quality=provisional,
        valuation=pd.Series([ValuationStatus.PASS.value], index=index),
        falling_state=pd.Series(
            [FallingKnifeState.REVERSAL_WATCH.value],
            index=index,
        ),
        next_day_confirmation=pd.Series([False], index=index),
        trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["FAIL"], index=index),
        market_gate=pd.Series(["PASS"], index=index),
        required_score=pd.Series([80.0], index=index),
        broad_rally_trigger=pd.Series([True], index=index),
        broad_rally_risk_gate=pd.Series(["PASS"], index=index),
        raw_score=pd.Series([40.0], index=index),
    )

    assert decision.loc[0] == "BUY_SMALL"


def test_v5_trusted_mops_base_fallback_can_use_conditional_starter() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    confirmed = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    fallback = replace(
        confirmed,
        base_status=BaseQualityStatus.PROVISIONAL,
        final_status=FinalQualityStatus.WATCH,
        final_score=90.0,
        base_data_confidence=79.7,
        base_available_weight=53.0,
        extended_data_confidence=83.8,
        critical_field_coverage=100.0,
        warnings=("MOPS_POINT_IN_TIME_BASE_FALLBACK",),
    )
    index = pd.RangeIndex(1)

    decision = strategy.rule_engine.resolve_decision(
        score=pd.Series([35.0], index=index),
        quality=fallback,
        valuation=pd.Series([ValuationStatus.PASS.value], index=index),
        falling_state=pd.Series(
            [FallingKnifeState.REVERSAL_WATCH.value],
            index=index,
        ),
        next_day_confirmation=pd.Series([False], index=index),
        trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["FAIL"], index=index),
        market_gate=pd.Series(["PASS"], index=index),
        required_score=pd.Series([80.0], index=index),
        broad_rally_trigger=pd.Series([True], index=index),
        broad_rally_risk_gate=pd.Series(["PASS"], index=index),
        raw_score=pd.Series([40.0], index=index),
    )

    assert decision.loc[0] == "BUY_SMALL"


def test_v5_trusted_mops_fallback_rejects_confirmed_negative_cfo() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    confirmed = evaluate_v5_quality(
        _annual_fundamentals(),
        _quality_snapshot(),
        SIGNAL_DATE,
    )
    extended_metrics = tuple(
        replace(metric, value=-1.0, passed=False)
        if metric.name == "positive_cfo"
        else metric
        for metric in confirmed.extended_metrics
    )
    fallback = replace(
        confirmed,
        base_status=BaseQualityStatus.PROVISIONAL,
        final_status=FinalQualityStatus.WATCH,
        final_score=90.0,
        base_data_confidence=79.7,
        base_available_weight=53.0,
        extended_data_confidence=83.8,
        critical_field_coverage=100.0,
        extended_metrics=extended_metrics,
        warnings=("MOPS_POINT_IN_TIME_BASE_FALLBACK",),
    )
    index = pd.RangeIndex(1)

    decision = strategy.rule_engine.resolve_decision(
        score=pd.Series([35.0], index=index),
        quality=fallback,
        valuation=pd.Series([ValuationStatus.PASS.value], index=index),
        falling_state=pd.Series(
            [FallingKnifeState.REVERSAL_WATCH.value],
            index=index,
        ),
        next_day_confirmation=pd.Series([False], index=index),
        trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["FAIL"], index=index),
        market_gate=pd.Series(["PASS"], index=index),
        required_score=pd.Series([80.0], index=index),
        broad_rally_trigger=pd.Series([True], index=index),
        broad_rally_risk_gate=pd.Series(["PASS"], index=index),
        raw_score=pd.Series([40.0], index=index),
    )

    assert decision.loc[0] == "WATCH_ONLY"
