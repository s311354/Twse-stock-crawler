import numpy as np
import pandas as pd

from domain.fundamentals import GateStatus
from domain.fundamentals import MetricScore
from domain.fundamentals import QualityGateResult
from domain.industry_profiles import CHEMICAL_PROFILE
from domain.industry_profiles import ELECTRONICS_PROFILE
from domain.industry_profiles import IndustryProfileMode
from domain.strategies.low_entry_score_v4_1 import LowEntryProfile
from domain.strategies.low_entry_score_v4_1 import LowEntryScoreV41Config
from domain.strategies.low_entry_score_v4_1 import LowEntryScoreV41Strategy
from domain.strategy import available_strategies
from domain.strategy import get_strategy


def _complete_component_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.RangeIndex(3)
    indicators = pd.DataFrame(
        {
            "taiex": [100.0, 100.5, 101.0],
            "taiex_ema20": [99.0, 99.5, 100.0],
            "taiex_ema20_slope": [0.1, 0.1, 0.1],
            "taiex_r5": [0.01, 0.01, 0.01],
            "r1": [0.0, 0.005, 0.02],
            "rs20": [0.01, 0.02, 0.03],
            "restriction_known": [True, True, True],
            "restriction_pass": [True, True, True],
            "price_position60": [0.2, 0.2, 0.2],
            "drawdown60": [-0.1, -0.1, -0.1],
            "recent_confirmed_swing_low": [98.0, 98.0, 98.0],
            "atr14": [2.0, 2.0, 2.0],
            "pe": [10.0, 10.0, 10.0],
            "pe_lower_bound60": [12.0, 12.0, 12.0],
            "new_20_day_low": [False, False, False],
            "higher_low": [True, True, True],
            "open": [98.0, 98.6, 99.2],
            "close": [98.5, 99.0, 100.0],
            "ema5": [98.0, 98.5, 99.0],
            "ema5_slope": [0.1, 0.1, 0.1],
            "ema20_slope": [0.1, 0.1, 0.1],
            "high": [98.8, 99.5, 100.5],
            "atr_percent": [0.02, 0.02, 0.02],
            "atr_percent_ma20": [0.03, 0.03, 0.03],
            "clv": [0.5, 0.6, 0.8],
            "cmf20": [0.1, 0.1, 0.1],
            "cmf20_improving": [True, True, True],
            "normalized_obv_slope5": [0.1, 0.1, 0.1],
            "volume_ratio": [1.1, 1.1, 1.2],
            "vwap20": [98.0, 98.5, 99.0],
            "rsi40_reclaim": [False, False, True],
            "macd_histogram_improving": [False, False, True],
            "macd_bullish_cross": [False, False, True],
            "amount_ma20": [100_000_000.0] * 3,
            "spread": [0.001] * 3,
        },
        index=index,
    )
    plan = pd.DataFrame(
        {
            "risk": [2.0] * 3,
            "reward_risk": [3.0] * 3,
            "order_amount": [1_000_000.0] * 3,
        },
        index=index,
    )
    return indicators, plan


def _market_history(size: int = 100) -> pd.DataFrame:
    close = np.linspace(110.0, 90.0, size)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=size, freq="B"),
            "Open": close + 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(size, 1_000_000.0),
            "Amount": np.full(size, 100_000_000.0),
            "Trades": np.full(size, 5_000.0),
            "Bid": close - 0.05,
            "Ask": close + 0.05,
            "PE": np.full(size, 12.0),
            "TAIEX Close": np.linspace(20_000.0, 20_100.0, size),
            "Trading Status": ["NORMAL"] * size,
        }
    )


def test_low_entry_score_v4_1_is_discoverable() -> None:
    assert "low_entry_score_v4_1" in available_strategies()
    assert get_strategy("low_entry_score_v4_1").name == "low_entry_score_v4_1"
    assert LowEntryScoreV41Strategy().profile is LowEntryProfile.BALANCED
    assert LowEntryScoreV41Strategy()._standard_risk_thresholds == (1.8, 1.7)
    assert LowEntryScoreV41Strategy().config.lot_size == 1
    assert (
        LowEntryScoreV41Strategy()
        .config.breadth_rally_initial_position_fraction
        == 1 / 8
    )


def test_v4_1_component_weights_total_exactly_100() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, plan = _complete_component_inputs()

    latest = strategy._score_v41_components(indicators, plan).iloc[-1]

    assert latest.to_dict() == {
        "market_regime": 15,
        "price_position": 15,
        "stabilization": 30,
        "demand_confirmation": 20,
        "momentum": 10,
        "execution_risk": 10,
    }
    assert latest.sum() == 100


def test_v4_1_oversold_rsi_alone_does_not_add_momentum_score() -> None:
    strategy = LowEntryScoreV41Strategy()
    history = _market_history()
    indicators = strategy._calculate_indicators(history)
    plan = strategy._build_trade_plan(indicators, {})

    latest_indicators = indicators.iloc[-1]
    latest_score = strategy._score_v41_components(indicators, plan).iloc[-1]

    assert latest_indicators["rsi14"] < 40
    assert not latest_indicators["rsi40_reclaim"]
    expected_macd_score = (
        3 * int(latest_indicators["macd_histogram_improving"])
        + 3 * int(latest_indicators["macd_bullish_cross"])
    )
    assert latest_score["momentum"] == expected_macd_score


def test_v4_1_quality_unknown_is_non_offsettable_veto() -> None:
    strategy = LowEntryScoreV41Strategy()

    latest = strategy.run(_market_history()).iloc[-1]

    assert latest["Quality Gate"] == "UNKNOWN"
    assert latest["Veto 資料或基本面禁止"] == 1
    assert latest["No Veto"] == 0
    assert latest["LES v4.1 Trigger"] == 0
    assert latest["低點決策"] == "DATA_INSUFFICIENT"


def test_v4_1_buy_requires_every_final_gate() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=80.0,
        data_confidence=90.0,
    )
    vetoes = pd.DataFrame(
        {name: [False] for name in (
            "breaks_previous_low",
            "high_volume_weak_close",
            "accelerating_downtrend",
            "repeated_breakdown",
            "gap_below_support",
            "market_relative_deterioration",
            "quality_prohibition",
        )},
        index=index,
    )

    buy = strategy._resolve_v41_state(
        quality=quality,
        provisional_eligible=False,
        vetoes=vetoes,
        score=pd.Series([80], index=index),
        trigger=pd.Series([True], index=index),
        strict_trigger=pd.Series([True], index=index),
        risk_gate=pd.Series(["PASS"], index=index),
        strict_risk_gate=pd.Series(["PASS"], index=index),
        market_data_gate=pd.Series(["PASS"], index=index),
    )
    armed = strategy._resolve_v41_state(
        quality=quality,
        provisional_eligible=False,
        vetoes=vetoes,
        score=pd.Series([80], index=index),
        trigger=pd.Series([False], index=index),
        strict_trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["PASS"], index=index),
        strict_risk_gate=pd.Series(["PASS"], index=index),
        market_data_gate=pd.Series(["PASS"], index=index),
    )

    assert buy.iloc[0] == "BUY"
    assert armed.iloc[0] == "ARMED"


def test_balanced_profile_uses_55_65_75_thresholds() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(3)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=80.0,
        data_confidence=90.0,
    )
    vetoes = pd.DataFrame(
        {
            name: [False] * 3
            for name in (
                "breaks_previous_low",
                "high_volume_weak_close",
                "accelerating_downtrend",
                "repeated_breakdown",
                "gap_below_support",
                "market_relative_deterioration",
                "quality_prohibition",
            )
        },
        index=index,
    )

    states = strategy._resolve_v41_state(
        quality=quality,
        provisional_eligible=False,
        vetoes=vetoes,
        score=pd.Series([55, 65, 75], index=index),
        trigger=pd.Series([False, False, True], index=index),
        strict_trigger=pd.Series([False, False, True], index=index),
        risk_gate=pd.Series(["PASS"] * 3, index=index),
        strict_risk_gate=pd.Series(["PASS"] * 3, index=index),
        market_data_gate=pd.Series(["PASS"] * 3, index=index),
    )

    assert states.tolist() == ["WATCH", "ARMED", "BUY"]


def test_balanced_pass_has_no_generic_early_buy_small_path() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=80.0,
        data_confidence=90.0,
    )
    vetoes = pd.DataFrame(
        {
            name: [False]
            for name in (
                "breaks_previous_low",
                "high_volume_weak_close",
                "accelerating_downtrend",
                "repeated_breakdown",
                "gap_below_support",
                "market_relative_deterioration",
                "quality_prohibition",
            )
        },
        index=index,
    )

    state = strategy._resolve_v41_state(
        quality=quality,
        provisional_eligible=False,
        vetoes=vetoes,
        score=pd.Series([60], index=index),
        trigger=pd.Series([True], index=index),
        strict_trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["PASS"], index=index),
        strict_risk_gate=pd.Series(["FAIL"], index=index),
        market_data_gate=pd.Series(["PASS"], index=index),
    )

    assert state.iloc[0] == "WATCH"


def test_strict_profile_retains_60_70_80_thresholds() -> None:
    strategy = LowEntryScoreV41Strategy(
        LowEntryScoreV41Config(profile=LowEntryProfile.STRICT)
    )
    index = pd.RangeIndex(1)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=80.0,
        data_confidence=90.0,
    )
    vetoes = pd.DataFrame(
        {
            name: [False]
            for name in (
                "breaks_previous_low",
                "high_volume_weak_close",
                "accelerating_downtrend",
                "repeated_breakdown",
                "gap_below_support",
                "market_relative_deterioration",
                "quality_prohibition",
            )
        },
        index=index,
    )

    state = strategy._resolve_v41_state(
        quality=quality,
        provisional_eligible=False,
        vetoes=vetoes,
        score=pd.Series([75], index=index),
        trigger=pd.Series([True], index=index),
        strict_trigger=pd.Series([True], index=index),
        risk_gate=pd.Series(["PASS"], index=index),
        strict_risk_gate=pd.Series(["PASS"], index=index),
        market_data_gate=pd.Series(["PASS"], index=index),
    )

    assert state.iloc[0] == "ARMED"
    assert strategy._standard_risk_thresholds == (1.5, 2.0)


def test_balanced_trigger_accepts_reclaim_without_prior_high_breakout() -> None:
    balanced = LowEntryScoreV41Strategy()
    strict = LowEntryScoreV41Strategy(
        LowEntryScoreV41Config(profile=LowEntryProfile.STRICT)
    )
    indicators, _ = _complete_component_inputs()
    indicators.loc[indicators.index[-1], "close"] = 99.4

    balanced_trigger = balanced._evaluate_balanced_technical_trigger(indicators)
    strict_trigger = strict._evaluate_strict_technical_trigger(indicators)

    assert balanced_trigger.iloc[-1]
    assert not strict_trigger.iloc[-1]


def test_balanced_liquidity_uses_conservative_estimate_when_spread_is_missing() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, plan = _complete_component_inputs()
    indicators["spread"] = np.nan
    indicators["amount_ma20"] = 25_000_000.0
    plan["order_amount"] = 200_000.0

    status, passed = strategy._evaluate_liquidity_gate(indicators, plan)

    assert status.iloc[-1] == "PASS_ESTIMATED"
    assert bool(passed.iloc[-1])


def test_chemical_probe_accepts_strong_close_with_dry_volume() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, plan = _complete_component_inputs()
    indicators.loc[indicators.index[-1], "clv"] = 1.0
    indicators.loc[indicators.index[-1], "volume_ratio"] = 0.15
    indicators.loc[indicators.index[-1], "amount_ma20"] = 4_800_000.0
    indicators.loc[indicators.index[-1], "spread"] = 0.004
    plan.loc[plan.index[-1], "order_amount"] = 460_000.0

    standard_status, standard_pass = strategy._evaluate_liquidity_gate(
        indicators,
        plan,
    )
    probe_status, probe_pass = (
        strategy._evaluate_chemical_probe_liquidity_gate(indicators, plan)
    )
    probe_trigger = strategy._evaluate_chemical_probe_technical_trigger(
        indicators
    )

    assert standard_status.iloc[-1] == "FAIL"
    assert not bool(standard_pass.iloc[-1])
    assert probe_status.iloc[-1] == "PASS"
    assert bool(probe_pass.iloc[-1])
    assert bool(probe_trigger.iloc[-1])


def test_chemical_probe_only_creates_small_entry_after_quality_and_veto_gates() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=85.0,
        data_confidence=100.0,
        industry_profile=CHEMICAL_PROFILE.name,
        industry_profile_version=CHEMICAL_PROFILE.version,
        industry_profile_mode=IndustryProfileMode.ACTIVE.value,
    )
    vetoes = pd.DataFrame(
        {
            name: [False]
            for name in (
                "breaks_previous_low",
                "high_volume_weak_close",
                "accelerating_downtrend",
                "repeated_breakdown",
                "gap_below_support",
                "market_relative_deterioration",
                "quality_prohibition",
            )
        },
        index=index,
    )
    common = {
        "quality": quality,
        "provisional_eligible": False,
        "score": pd.Series([65], index=index),
        "trigger": pd.Series([False], index=index),
        "strict_trigger": pd.Series([False], index=index),
        "risk_gate": pd.Series(["FAIL"], index=index),
        "strict_risk_gate": pd.Series(["FAIL"], index=index),
        "market_data_gate": pd.Series(["PASS"], index=index),
        "chemical_probe_trigger": pd.Series([True], index=index),
        "chemical_probe_risk_gate": pd.Series(["PASS"], index=index),
    }

    state = strategy._resolve_v41_state(vetoes=vetoes, **common)
    vetoes.loc[0, "breaks_previous_low"] = True
    vetoed_state = strategy._resolve_v41_state(vetoes=vetoes, **common)

    assert state.iloc[0] == "BUY_SMALL"
    assert vetoed_state.iloc[0] == "NO_BUY"


def test_chemical_probe_does_not_relax_electronics_profile() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=85.0,
        data_confidence=100.0,
        industry_profile=ELECTRONICS_PROFILE.name,
        industry_profile_version=ELECTRONICS_PROFILE.version,
        industry_profile_mode=IndustryProfileMode.ACTIVE.value,
    )
    vetoes = pd.DataFrame(
        {
            name: [False]
            for name in (
                "breaks_previous_low",
                "high_volume_weak_close",
                "accelerating_downtrend",
                "repeated_breakdown",
                "gap_below_support",
                "market_relative_deterioration",
                "quality_prohibition",
            )
        },
        index=index,
    )

    state = strategy._resolve_v41_state(
        quality=quality,
        provisional_eligible=False,
        vetoes=vetoes,
        score=pd.Series([65], index=index),
        trigger=pd.Series([False], index=index),
        strict_trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["FAIL"], index=index),
        strict_risk_gate=pd.Series(["FAIL"], index=index),
        market_data_gate=pd.Series(["PASS"], index=index),
        chemical_probe_trigger=pd.Series([True], index=index),
        chemical_probe_risk_gate=pd.Series(["PASS"], index=index),
    )

    assert state.iloc[0] == "ARMED"


def test_electronics_probe_is_limited_to_non_semiconductor_electronics() -> None:
    strategy = LowEntryScoreV41Strategy()
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=80.0,
        data_confidence=90.0,
        industry_profile=ELECTRONICS_PROFILE.name,
        industry_profile_version=ELECTRONICS_PROFILE.version,
        industry_profile_mode=IndustryProfileMode.ACTIVE.value,
    )

    assert strategy._electronics_probe_quality_eligible(
        quality,
        sector_code="25",
    )
    assert strategy._electronics_probe_quality_eligible(
        quality,
        sector_code="31",
    )
    assert not strategy._electronics_probe_quality_eligible(
        quality,
        sector_code="24",
    )
    assert not strategy._electronics_probe_quality_eligible(
        quality,
        sector_code="32",
    )

    quality.score = 79.99
    assert not strategy._electronics_probe_quality_eligible(
        quality,
        sector_code="25",
    )

    quality.score = 80.0
    quality.data_confidence = 79.99
    assert not strategy._electronics_probe_quality_eligible(
        quality,
        sector_code="25",
    )


def test_electronics_probe_requires_higher_low_reclaim_and_strong_close() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, _ = _complete_component_inputs()

    trigger = strategy._evaluate_electronics_probe_technical_trigger(
        indicators
    )
    assert bool(trigger.iloc[-1])

    indicators.loc[indicators.index[-1], "volume_ratio"] = 0.79
    weak_volume = strategy._evaluate_electronics_probe_technical_trigger(
        indicators
    )
    assert not bool(weak_volume.iloc[-1])

    indicators.loc[indicators.index[-1], "volume_ratio"] = 1.0
    indicators.loc[indicators.index[-1], "close"] = 98.9
    below_ema5 = strategy._evaluate_electronics_probe_technical_trigger(
        indicators
    )
    assert not bool(below_ema5.iloc[-1])


def test_electronics_probe_requires_every_demand_conjunct() -> None:
    strategy = LowEntryScoreV41Strategy()
    latest = -1
    failures = {
        "higher_low": False,
        "ema5_slope": -0.20,
        "open": 100.0,
        "clv": 0.5999,
        "cmf20": 0.0,
        "volume_ratio": 0.7999,
    }

    for column, value in failures.items():
        indicators, _ = _complete_component_inputs()
        indicators.loc[indicators.index[latest], column] = value

        trigger = strategy._evaluate_electronics_probe_technical_trigger(
            indicators
        )

        assert not bool(trigger.iloc[latest]), column


def test_electronics_probe_uses_one_sixteenth_liquidity_and_exact_risk_bounds() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, plan = _complete_component_inputs()
    indicators["amount_ma20"] = 1_000_000.0
    indicators["spread"] = 0.015
    plan["order_amount"] = 320_000.0
    plan["entry"] = 100.0
    plan["stop"] = 96.4
    plan["target"] = 107.2
    plan["risk"] = 3.6
    plan["reward_risk"] = 2.0
    plan["shares"] = 100.0
    market_data_gate = pd.Series("PASS", index=indicators.index)

    liquidity_gate, liquidity_pass = (
        strategy._evaluate_electronics_probe_liquidity_gate(
            indicators,
            plan,
        )
    )
    risk_gate, _ = strategy._evaluate_profile_risk_gate(
        indicators,
        plan,
        market_data_gate,
        maximum_stop_atr=(
            strategy.config.electronics_probe_maximum_stop_atr
        ),
        minimum_reward_risk=(
            strategy.config.electronics_probe_minimum_reward_risk
        ),
        liquidity_gate=liquidity_gate,
        liquidity_pass=liquidity_pass,
    )

    assert liquidity_gate.iloc[-1] == "PASS"
    assert bool(liquidity_pass.iloc[-1])
    assert risk_gate.iloc[-1] == "PASS"

    plan["order_amount"] = 320_001.0
    over_order_gate, over_order_pass = (
        strategy._evaluate_electronics_probe_liquidity_gate(
            indicators,
            plan,
        )
    )
    assert over_order_gate.iloc[-1] == "FAIL"
    assert not bool(over_order_pass.iloc[-1])

    plan["order_amount"] = 320_000.0
    plan["reward_risk"] = 1.9999
    low_rr_gate, _ = strategy._evaluate_profile_risk_gate(
        indicators,
        plan,
        market_data_gate,
        maximum_stop_atr=(
            strategy.config.electronics_probe_maximum_stop_atr
        ),
        minimum_reward_risk=(
            strategy.config.electronics_probe_minimum_reward_risk
        ),
        liquidity_gate=liquidity_gate,
        liquidity_pass=liquidity_pass,
    )
    assert low_rr_gate.iloc[-1] == "FAIL"

    plan["reward_risk"] = 2.0
    plan["risk"] = 3.6001
    far_stop_gate, _ = strategy._evaluate_profile_risk_gate(
        indicators,
        plan,
        market_data_gate,
        maximum_stop_atr=(
            strategy.config.electronics_probe_maximum_stop_atr
        ),
        minimum_reward_risk=(
            strategy.config.electronics_probe_minimum_reward_risk
        ),
        liquidity_gate=liquidity_gate,
        liquidity_pass=liquidity_pass,
    )
    assert far_stop_gate.iloc[-1] == "FAIL"


def test_electronics_probe_only_creates_one_sixteenth_buy_small_without_veto() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=80.0,
        data_confidence=90.0,
        industry_profile=ELECTRONICS_PROFILE.name,
        industry_profile_version=ELECTRONICS_PROFILE.version,
        industry_profile_mode=IndustryProfileMode.ACTIVE.value,
    )
    vetoes = pd.DataFrame(
        {
            name: [False]
            for name in (
                "breaks_previous_low",
                "high_volume_weak_close",
                "accelerating_downtrend",
                "repeated_breakdown",
                "gap_below_support",
                "market_relative_deterioration",
                "quality_prohibition",
            )
        },
        index=index,
    )
    common = {
        "quality": quality,
        "provisional_eligible": False,
        "score": pd.Series([51], index=index),
        "trigger": pd.Series([False], index=index),
        "strict_trigger": pd.Series([False], index=index),
        "risk_gate": pd.Series(["PASS"], index=index),
        "strict_risk_gate": pd.Series(["FAIL"], index=index),
        "market_data_gate": pd.Series(["PASS"], index=index),
        "electronics_probe_eligible": True,
        "electronics_probe_trigger": pd.Series([True], index=index),
        "electronics_probe_risk_gate": pd.Series(["PASS"], index=index),
    }

    state = strategy._resolve_v41_state(vetoes=vetoes, **common)
    vetoes.loc[0, "breaks_previous_low"] = True
    vetoed_state = strategy._resolve_v41_state(vetoes=vetoes, **common)

    assert state.iloc[0] == "BUY_SMALL"
    assert vetoed_state.iloc[0] == "NO_BUY"
    assert strategy.config.electronics_buy_small_initial_position_fraction == (
        1 / 16
    )


def test_semiconductor_probe_is_limited_to_sector_24_quality_pass() -> None:
    strategy = LowEntryScoreV41Strategy()
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=85.0,
        data_confidence=90.0,
        industry_profile=ELECTRONICS_PROFILE.name,
        industry_profile_version=ELECTRONICS_PROFILE.version,
        industry_profile_mode=IndustryProfileMode.ACTIVE.value,
    )

    assert strategy._semiconductor_probe_quality_eligible(
        quality,
        sector_code="24",
    )
    assert not strategy._semiconductor_probe_quality_eligible(
        quality,
        sector_code="25",
    )

    quality.score = 74.99
    assert not strategy._semiconductor_probe_quality_eligible(
        quality,
        sector_code="24",
    )


def test_semiconductor_probe_requires_confirmed_rebound_evidence() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, _ = _complete_component_inputs()

    trigger = strategy._evaluate_semiconductor_probe_technical_trigger(
        indicators
    )
    assert bool(trigger.iloc[-1])

    indicators.loc[indicators.index[-1], "macd_histogram_improving"] = False
    no_momentum_trigger = (
        strategy._evaluate_semiconductor_probe_technical_trigger(indicators)
    )
    assert not bool(no_momentum_trigger.iloc[-1])


def test_semiconductor_probe_only_creates_one_eighth_buy_small_without_veto() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=85.0,
        data_confidence=90.0,
        industry_profile=ELECTRONICS_PROFILE.name,
        industry_profile_version=ELECTRONICS_PROFILE.version,
        industry_profile_mode=IndustryProfileMode.ACTIVE.value,
    )
    vetoes = pd.DataFrame(
        {
            name: [False]
            for name in (
                "breaks_previous_low",
                "high_volume_weak_close",
                "accelerating_downtrend",
                "repeated_breakdown",
                "gap_below_support",
                "market_relative_deterioration",
                "quality_prohibition",
            )
        },
        index=index,
    )
    common = {
        "quality": quality,
        "provisional_eligible": False,
        "score": pd.Series([55], index=index),
        "trigger": pd.Series([True], index=index),
        "strict_trigger": pd.Series([False], index=index),
        "risk_gate": pd.Series(["FAIL"], index=index),
        "strict_risk_gate": pd.Series(["FAIL"], index=index),
        "market_data_gate": pd.Series(["PASS"], index=index),
        "semiconductor_probe_eligible": True,
        "semiconductor_probe_trigger": pd.Series([True], index=index),
        "semiconductor_probe_risk_gate": pd.Series(["PASS"], index=index),
    }

    state = strategy._resolve_v41_state(vetoes=vetoes, **common)
    vetoes.loc[0, "breaks_previous_low"] = True
    vetoed_state = strategy._resolve_v41_state(vetoes=vetoes, **common)

    assert state.iloc[0] == "BUY_SMALL"
    assert vetoed_state.iloc[0] == "NO_BUY"
    assert strategy.config.semiconductor_buy_small_initial_position_fraction == (
        1 / 8
    )


def test_semiconductor_probe_accepts_wider_three_atr_structural_stop() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, plan = _complete_component_inputs()
    indicators["taiex_r5"] = 0.01
    plan["entry"] = 100.0
    plan["stop"] = 94.4
    plan["target"] = 110.0
    plan["risk"] = 5.6
    plan["reward_risk"] = 10.0 / 5.6
    plan["shares"] = 100.0
    market_data_gate = pd.Series("PASS", index=indicators.index)
    liquidity_gate, liquidity_pass = (
        strategy._evaluate_semiconductor_probe_liquidity_gate(
            indicators,
            plan,
        )
    )

    standard_gate, _ = strategy._evaluate_profile_risk_gate(
        indicators,
        plan,
        market_data_gate,
        maximum_stop_atr=2.0,
        minimum_reward_risk=1.5,
        liquidity_gate=liquidity_gate,
        liquidity_pass=liquidity_pass,
    )
    probe_gate, _ = strategy._evaluate_profile_risk_gate(
        indicators,
        plan,
        market_data_gate,
        maximum_stop_atr=3.0,
        minimum_reward_risk=1.5,
        liquidity_gate=liquidity_gate,
        liquidity_pass=liquidity_pass,
    )

    assert standard_gate.iloc[-1] == "FAIL"
    assert probe_gate.iloc[-1] == "PASS"


def test_balanced_trade_plan_supports_twse_odd_lots() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(30)
    indicators = pd.DataFrame(
        {
            "high": np.full(30, 100.0),
            "recent_confirmed_swing_low": np.full(30, 98.0),
            "atr14": np.full(30, 1.0),
        },
        index=index,
    )

    latest = strategy._build_trade_plan(indicators, {}).iloc[-1]

    assert latest["shares"] > 0
    assert latest["shares"] % 1000 != 0


def test_balanced_provisional_requires_domain_coverage_and_nonnegative_cfo() -> None:
    strategy = LowEntryScoreV41Strategy()
    metric_scores = [
        MetricScore("profit", "A", 20, 25, True, "available"),
        MetricScore("balance", "C", 20, 25, True, "available"),
    ]
    quality = QualityGateResult(
        status=GateStatus.PROVISIONAL,
        score=80.0,
        data_confidence=65.0,
        metric_scores=metric_scores,
        metrics={"cfo_ttm": None, "cumulative_cfo_3y": 10.0},
    )

    coverage = strategy._quality_domain_coverage(quality)

    assert coverage["profitability"] == 100.0
    assert coverage["balance_sheet_safety"] == 100.0
    assert coverage["no_confirmed_negative_cfo"] is True
    assert strategy._provisional_buy_eligible(quality, coverage) is True

    quality.metrics["cfo_ttm"] = -1.0
    negative_cfo_coverage = strategy._quality_domain_coverage(quality)
    assert negative_cfo_coverage["no_confirmed_negative_cfo"] is False
    assert (
        strategy._provisional_buy_eligible(
            quality,
            negative_cfo_coverage,
        )
        is False
    )


def test_breadth_rally_requires_date_aligned_completed_electronics_session() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, _ = _complete_component_inputs()
    indicators["taiex_r1"] = [0.0, 0.0, 0.005]
    data = pd.DataFrame(
        {
            "Date": [
                "2026-07-29",
                "2026-07-30",
                "2026-07-31",
            ]
        },
        index=indicators.index,
    )
    snapshot = {
        "as_of": "2026-07-31",
        "universe": "ELEC",
        "coverage": 0.80,
        "valid_return_count": 100,
        "advancer_ratio": 0.80,
        "median_return": 0.015,
        "source_final": True,
    }

    regime = strategy._evaluate_breadth_rally_regime(
        data,
        indicators,
        breadth_snapshot=snapshot,
        market_data_through="2026-07-31",
        market_session_final=True,
    )
    incomplete = strategy._evaluate_breadth_rally_regime(
        data,
        indicators,
        breadth_snapshot=snapshot,
        market_data_through="2026-07-31",
        market_session_final=False,
    )
    stale = strategy._evaluate_breadth_rally_regime(
        data,
        indicators,
        breadth_snapshot=snapshot,
        market_data_through="2026-07-30",
        market_session_final=True,
    )

    assert regime.tolist() == [False, False, True]
    assert not incomplete.any()
    assert not stale.any()


def test_semiconductor_breadth_rally_uses_stricter_sector_thresholds() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, _ = _complete_component_inputs()
    indicators["taiex_r1"] = [0.0, 0.0, 0.005]
    data = pd.DataFrame(
        {"Date": ["2026-07-29", "2026-07-30", "2026-07-31"]},
        index=indicators.index,
    )
    snapshot = {
        "as_of": "2026-07-31",
        "universe": "SEMI",
        "coverage": 0.80,
        "valid_return_count": 40,
        "advancer_ratio": 0.85,
        "median_return": 0.020,
        "source_final": True,
    }

    regime = strategy._evaluate_breadth_rally_regime(
        data,
        indicators,
        breadth_snapshot=snapshot,
        market_data_through="2026-07-31",
        market_session_final=True,
    )
    snapshot["advancer_ratio"] = 0.8499
    below_threshold = strategy._evaluate_breadth_rally_regime(
        data,
        indicators,
        breadth_snapshot=snapshot,
        market_data_through="2026-07-31",
        market_session_final=True,
    )

    assert bool(regime.iloc[-1])
    assert not below_threshold.any()


def test_completed_snapshot_taiex_return_overrides_stock_history_gap() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, _ = _complete_component_inputs()
    fallback = pd.Series(
        [0.0, 0.0, -0.04],
        index=indicators.index,
    )
    data = pd.DataFrame(
        {"Date": ["2026-07-29", "2026-07-30", "2026-07-31"]},
        index=indicators.index,
    )

    resolved = strategy._date_aligned_market_return(
        data,
        fallback,
        {
            "as_of": "2026-07-31",
            "taiex_return": 0.006,
        },
    )

    assert resolved.tolist() == [0.0, 0.0, 0.006]


def test_recovery_plan_reprices_risk_at_higher_of_high_and_ema5() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    plan = pd.DataFrame(
        {
            "entry": [100.0],
            "stop": [90.0],
            "target": [120.0],
            "risk": [10.0],
            "reward_risk": [2.0],
            "shares": [1000.0],
            "order_amount": [100_000.0],
        },
        index=index,
    )

    repriced = strategy._reprice_trade_plan(
        plan,
        entry=pd.Series([105.0], index=index),
        context={},
    )

    assert repriced.loc[0, "entry"] == 105.0
    assert repriced.loc[0, "risk"] == 15.0
    assert repriced.loc[0, "reward_risk"] == 1.0


def test_recovery_plan_rounds_trigger_up_to_valid_twse_stock_tick() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    plan = pd.DataFrame(
        {
            "entry": [68_700.0],
            "stop": [65_000.0],
            "target": [84_000.0],
            "risk": [3_700.0],
            "reward_risk": [4.0],
            "shares": [1000.0],
            "order_amount": [68_700.0],
        },
        index=index,
    )

    repriced = strategy._reprice_trade_plan(
        plan,
        entry=pd.Series([68_721.03], index=index),
        context={"les_v4_price_scale": 1000.0},
    )

    assert repriced.loc[0, "entry"] == 68_800.0
    assert repriced.loc[0, "risk"] == 3_800.0
    assert repriced.loc[0, "reward_risk"] == 4.0


def test_electronics_breadth_rally_trigger_requires_stock_reversal() -> None:
    strategy = LowEntryScoreV41Strategy()
    indicators, _ = _complete_component_inputs()
    regime = pd.Series(
        [False, False, True],
        index=indicators.index,
    )

    trigger = strategy._evaluate_electronics_breadth_rally_trigger(
        indicators,
        regime,
    )
    indicators.loc[indicators.index[-1], "close"] = 98.9
    weak_stock = strategy._evaluate_electronics_breadth_rally_trigger(
        indicators,
        regime,
    )

    assert bool(trigger.iloc[-1])
    assert not bool(weak_stock.iloc[-1])


def test_electronics_breadth_rally_buy_small_never_overrides_veto() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=85.0,
        data_confidence=90.0,
        industry_profile=ELECTRONICS_PROFILE.name,
        industry_profile_version=ELECTRONICS_PROFILE.version,
        industry_profile_mode=IndustryProfileMode.ACTIVE.value,
    )
    vetoes = pd.DataFrame(
        {
            name: [False]
            for name in (
                "breaks_previous_low",
                "high_volume_weak_close",
                "accelerating_downtrend",
                "repeated_breakdown",
                "gap_below_support",
                "market_relative_deterioration",
                "quality_prohibition",
            )
        },
        index=index,
    )
    common = {
        "quality": quality,
        "provisional_eligible": False,
        "score": pd.Series([45], index=index),
        "trigger": pd.Series([False], index=index),
        "strict_trigger": pd.Series([False], index=index),
        "risk_gate": pd.Series(["FAIL"], index=index),
        "strict_risk_gate": pd.Series(["FAIL"], index=index),
        "market_data_gate": pd.Series(["PASS"], index=index),
        "electronics_rally_eligible": True,
        "electronics_rally_trigger": pd.Series([True], index=index),
        "electronics_rally_risk_gate": pd.Series(["PASS"], index=index),
    }

    state = strategy._resolve_v41_state(vetoes=vetoes, **common)
    vetoes.loc[0, "gap_below_support"] = True
    vetoed = strategy._resolve_v41_state(vetoes=vetoes, **common)

    assert state.iloc[0] == "BUY_SMALL"
    assert vetoed.iloc[0] == "NO_BUY"


def test_electronics_breadth_rally_below_relaxed_floor_does_not_buy() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=85.0,
        data_confidence=90.0,
        industry_profile=ELECTRONICS_PROFILE.name,
        industry_profile_version=ELECTRONICS_PROFILE.version,
        industry_profile_mode=IndustryProfileMode.ACTIVE.value,
    )
    vetoes = pd.DataFrame(
        {
            name: [False]
            for name in (
                "breaks_previous_low",
                "high_volume_weak_close",
                "accelerating_downtrend",
                "repeated_breakdown",
                "gap_below_support",
                "market_relative_deterioration",
                "quality_prohibition",
            )
        },
        index=index,
    )

    state = strategy._resolve_v41_state(
        quality=quality,
        provisional_eligible=False,
        score=pd.Series([44.99], index=index),
        trigger=pd.Series([False], index=index),
        strict_trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["FAIL"], index=index),
        strict_risk_gate=pd.Series(["FAIL"], index=index),
        market_data_gate=pd.Series(["PASS"], index=index),
        vetoes=vetoes,
        electronics_rally_eligible=True,
        electronics_rally_trigger=pd.Series([True], index=index),
        electronics_rally_risk_gate=pd.Series(["PASS"], index=index),
    )

    assert state.iloc[0] == "AVOID"


def test_zero_share_buy_is_downgraded_to_armed() -> None:
    state = pd.Series(["BUY", "BUY_SMALL", "WATCH"])
    suggested_shares = pd.Series([0.0, np.nan, 0.0])

    executable = LowEntryScoreV41Strategy._enforce_executable_position(
        state,
        suggested_shares,
    )

    assert executable.tolist() == ["ARMED", "ARMED", "WATCH"]


def test_electronics_recovery_state_expands_watchlist_without_buying() -> None:
    strategy = LowEntryScoreV41Strategy()
    index = pd.RangeIndex(1)
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=76.0,
        data_confidence=90.0,
        industry_profile=ELECTRONICS_PROFILE.name,
        industry_profile_version=ELECTRONICS_PROFILE.version,
        industry_profile_mode=IndustryProfileMode.ACTIVE.value,
    )
    vetoes = pd.DataFrame(
        {
            name: [False]
            for name in (
                "breaks_previous_low",
                "high_volume_weak_close",
                "accelerating_downtrend",
                "repeated_breakdown",
                "gap_below_support",
                "market_relative_deterioration",
                "quality_prohibition",
            )
        },
        index=index,
    )

    state = strategy._resolve_v41_state(
        quality=quality,
        provisional_eligible=False,
        vetoes=vetoes,
        score=pd.Series([45], index=index),
        trigger=pd.Series([False], index=index),
        strict_trigger=pd.Series([False], index=index),
        risk_gate=pd.Series(["FAIL"], index=index),
        strict_risk_gate=pd.Series(["FAIL"], index=index),
        market_data_gate=pd.Series(["PASS"], index=index),
        electronics_recovery_armed=pd.Series([True], index=index),
    )

    assert state.iloc[0] == "ARMED"


def _veto_inputs() -> tuple[
    LowEntryScoreV41Strategy,
    pd.DataFrame,
    QualityGateResult,
]:
    strategy = LowEntryScoreV41Strategy()
    indicators = pd.DataFrame(
        {
            "new_20_day_low": [False] * 5,
            "recent_confirmed_swing_low": [98.0] * 5,
            "clv": [0.1] * 5,
            "taiex": [101.0] * 5,
            "taiex_ema20": [100.0] * 5,
            "taiex_ema20_slope": [0.1] * 5,
            "rs20_deteriorating": [False] * 5,
            "close": [99.0, 99.0, 99.0, 100.0, 96.0],
            "open": [99.0, 99.0, 99.0, 100.0, 97.0],
            "volume_ratio": [1.0] * 5,
            "ema20_slope": [0.0] * 5,
            "ema5_slope": [0.0] * 5,
            "r5": [0.0] * 5,
            "atr_percent": [0.02] * 5,
            "normalized_gap": [0.0, 0.0, 0.0, 0.0, -0.6],
        }
    )
    quality = QualityGateResult(
        status=GateStatus.PASS,
        score=80.0,
        data_confidence=90.0,
    )
    return strategy, indicators, quality


def test_gap_veto_requires_a_new_gap_from_above_prior_support() -> None:
    strategy, indicators, quality = _veto_inputs()

    vetoes = strategy._evaluate_v41_vetoes(
        indicators,
        pd.DataFrame(index=indicators.index),
        quality,
        provisional_eligible=False,
    )
    indicators.loc[3, "close"] = 96.0
    already_below = strategy._evaluate_v41_vetoes(
        indicators,
        pd.DataFrame(index=indicators.index),
        quality,
        provisional_eligible=False,
    )

    assert bool(vetoes.iloc[-1]["gap_below_support"])
    assert not bool(already_below.iloc[-1]["gap_below_support"])


def test_acceleration_and_repeated_breakdown_require_material_evidence() -> None:
    strategy, indicators, quality = _veto_inputs()
    indicators.loc[4, ["ema20_slope", "ema5_slope", "r5"]] = [
        -0.05,
        -0.1,
        -0.10,
    ]
    indicators.loc[3:4, "new_20_day_low"] = True
    indicators.loc[2:4, "clv"] = 0.1

    weak = strategy._evaluate_v41_vetoes(
        indicators,
        pd.DataFrame(index=indicators.index),
        quality,
        provisional_eligible=False,
    )
    indicators.loc[4, "ema20_slope"] = -0.11
    indicators.loc[2:4, "clv"] = -0.1
    material = strategy._evaluate_v41_vetoes(
        indicators,
        pd.DataFrame(index=indicators.index),
        quality,
        provisional_eligible=False,
    )

    assert not bool(weak.iloc[-1]["accelerating_downtrend"])
    assert not bool(weak.iloc[-1]["repeated_breakdown"])
    assert bool(material.iloc[-1]["accelerating_downtrend"])
    assert bool(material.iloc[-1]["repeated_breakdown"])
