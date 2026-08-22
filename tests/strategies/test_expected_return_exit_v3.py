import math

import pandas as pd

from domain.strategies.expected_return_exit_v3 import EREM_V3_MAX_RAW_SCORE
from domain.strategies.expected_return_exit_v3 import ExpectedReturnExitV3Strategy
from domain.strategy import available_strategies
from domain.strategy import get_strategy


def make_history(include_optional_inputs: bool = True) -> pd.DataFrame:
    count = 240
    closes = [100 + index * 0.12 + math.sin(index / 4) * 0.8 for index in range(count)]
    closes[-1] = closes[-2] + 1
    data: dict[str, list[float]] = {
        "Open": [close - 0.5 for close in closes],
        "High": [close + 3 for close in closes],
        "Low": [close - 8 for close in closes],
        "Close": closes,
        "Volume": [1000.0] * (count - 1) + [2500.0],
    }
    if include_optional_inputs:
        data.update(
            {
                "TAIEX Close": [100 + index * 0.03 for index in range(count)],
                "Entry LES": [math.nan] * (count - 30) + [80.0] + [math.nan] * 29,
                "Holding Days": [math.nan] * (count - 30) + list(range(30)),
                "Turnover": [1.0] * (count - 1) + [2.5],
            }
        )
    return pd.DataFrame(data)


def decision_for(score: float) -> str:
    if score >= 85:
        return "STRONG HOLD"
    if score >= 70:
        return "HOLD"
    if score >= 55:
        return "HOLD"
    if score >= 40:
        return "REDUCE 25%"
    return "EXIT"


def test_expected_return_exit_v3_strategy_is_discoverable() -> None:
    assert "expected_return_exit_v3" in available_strategies()
    assert get_strategy("expected_return_exit_v3").name == "expected_return_exit_v3"


def test_expected_return_exit_v3_decision_boundaries_avoid_overtrading() -> None:
    scores = pd.Series([85, 84.99, 70, 69.99, 55, 54.99, 40, 39.99])

    assert ExpectedReturnExitV3Strategy()._decision(scores).tolist() == [
        "STRONG HOLD",
        "HOLD",
        "HOLD",
        "HOLD",
        "HOLD",
        "REDUCE 25%",
        "REDUCE 25%",
        "EXIT",
    ]


def test_expected_return_exit_v3_counts_consecutive_exit_warnings() -> None:
    streak = ExpectedReturnExitV3Strategy._consecutive_true(
        pd.Series([False, True, True, False, True])
    )

    assert streak.tolist() == [0, 1, 2, 0, 1]


def test_expected_return_exit_v3_decays_entry_les_weight_with_holding_days() -> None:
    history = make_history()
    result = ExpectedReturnExitV3Strategy().run(history)
    latest = result.iloc[-1]

    expected_score = round(latest["EREM Raw Score"] * 100 / EREM_V3_MAX_RAW_SCORE, 2)
    expected_weight = max(0.10, 0.40 * math.exp(-29 / 20))
    expected_confidence = round(
        ((1 - expected_weight) * expected_score) + (expected_weight * 80),
        2,
    )

    assert 0 <= latest["EREM Raw Score"] <= EREM_V3_MAX_RAW_SCORE
    assert latest["EREM 分數"] == expected_score
    assert latest["EREM Holding Confidence"] == expected_confidence
    assert result.iloc[-30]["EREM Entry Weight"] == 0.4
    assert result.iloc[-10]["EREM Entry Weight"] == round(
        max(0.10, 0.40 * math.exp(-20 / 20)),
        4,
    )
    assert latest["EREM Entry Weight"] == 0.1
    assert latest["EREM Holding Days"] == 29
    assert latest["EREM 決策訊號"] == decision_for(expected_confidence)
    assert latest["EREM Relative Strength Score"] == 10
    expected_rs60 = (
        history["Close"].iloc[-1] / history["Close"].iloc[-61] - 1
        - (history["TAIEX Close"].iloc[-1] / history["TAIEX Close"].iloc[-61] - 1)
    )
    assert latest["EREM Relative Strength60"] == round(expected_rs60, 4)
    assert latest["EREM Structure Score"] == 5
    assert latest["EREM Entry Quality Score"] == 5
    assert latest["EREM Entry LES"] == 80
    assert latest["EREM Exit Streak Count"] >= 0
    assert pd.notna(latest["EREM EMA20 Slope"])
    assert latest["EREM Turnover Score"] == 5
    assert latest["EREM Turnover Source"] == "Turnover"


def test_expected_return_exit_v3_uses_floor_when_holding_days_are_missing() -> None:
    history = make_history().drop(columns=["Holding Days"])
    latest = ExpectedReturnExitV3Strategy().run(history).iloc[-1]
    expected = round((0.9 * latest["EREM 分數"]) + (0.1 * 80), 2)

    assert latest["EREM Entry Weight"] == 0.1
    assert pd.isna(latest["EREM Holding Days"])
    assert latest["EREM Holding Confidence"] == expected
    assert "Holding Days" in latest["EREM 理由"]


def test_expected_return_exit_v3_scores_vwap_support_and_recovery() -> None:
    close = pd.Series([90.0, 92.0, 103.0, 88.0, 87.0])
    vwap60 = pd.Series([100.0, 99.0, 100.0, 101.0, 89.0])

    score, state = ExpectedReturnExitV3Strategy._vwap_assessment(
        close,
        vwap60,
    )

    assert score.tolist() == [0.0, 2.0, 5.0, 0.0, 0.0]
    assert state.tolist() == [
        "BELOW_NOT_RECOVERING",
        "RECOVERING",
        "ABOVE_OR_EQUAL",
        "BELOW_NOT_RECOVERING",
        "BELOW_SYNCHRONOUS_DECLINE",
    ]


def test_expected_return_exit_v3_handles_missing_taiex_and_entry_les() -> None:
    latest = ExpectedReturnExitV3Strategy().run(make_history(include_optional_inputs=False)).iloc[-1]

    assert latest["EREM Relative Strength Score"] == 0
    assert latest["EREM Entry Quality Score"] == 0
    assert pd.isna(latest["EREM Relative Strength60"])
    assert pd.isna(latest["EREM Entry LES"])
    assert latest["EREM Holding Confidence"] == latest["EREM 分數"]
    assert latest["EREM Turnover Source"] == "Volume/VolumeMA20 proxy"
    assert "缺少=TAIEX,Entry LES" in latest["EREM 理由"]


def test_expected_return_exit_v3_handles_zero_reward_risk_denominator() -> None:
    result = ExpectedReturnExitV3Strategy().run(
        pd.DataFrame(
            {
                "High": [100.0] * 121,
                "Low": [100.0] * 121,
                "Close": [100.0] * 121,
                "Volume": [0.0] * 121,
            }
        )
    )
    latest = result.iloc[-1]

    assert pd.isna(latest["EREM RewardRisk"])
    assert pd.isna(latest["EREM VWAP60"])
    assert latest["EREM Remaining Return Score"] == 0
    assert latest["EREM Turnover Score"] == 0
