import math

import pandas as pd

from domain.strategies.expected_return_exit import ExpectedReturnExitStrategy
from domain.strategy import available_strategies
from domain.strategy import get_strategy


def test_expected_return_exit_strategy_is_discoverable() -> None:
    assert "expected_return_exit" in available_strategies()
    assert get_strategy("expected_return_exit").name == "expected_return_exit"


def test_expected_return_exit_scores_strong_hold_path() -> None:
    closes = [100 + index * 0.03 + math.sin(index / 2) for index in range(220)]
    closes[-2] = closes[-3] - 0.5
    closes[-1] = closes[-2] + 0.3
    highs = [close + 3 for close in closes]
    lows = [close - 10 for close in closes]
    highs[-30] = 170
    volumes = [1000] * 219 + [3000]

    strategy = ExpectedReturnExitStrategy()
    result = strategy.run(
        pd.DataFrame(
            {
                "Open": [close - 1 for close in closes],
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            }
        )
    )
    latest = result.iloc[-1]

    assert latest["EREM 決策訊號"] == "Strong Hold"
    assert latest["EREM Trend Score"] == 25
    assert latest["EREM Momentum Score"] == 15
    assert latest["EREM RewardRisk"] > 2
    assert latest["EREM 分數"] >= 85


def test_expected_return_exit_penalizes_exit_risk() -> None:
    closes = [160 - index * 0.2 for index in range(220)]
    highs = [close + 1 for close in closes]
    lows = [close - 1 for close in closes]
    volumes = [1000] * 219 + [5000]
    closes[-1] = closes[-2]
    highs[-1] = highs[-2]
    lows[-1] = lows[-2]

    strategy = ExpectedReturnExitStrategy()
    result = strategy.run(
        pd.DataFrame(
            {
                "Open": [close for close in closes],
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            }
        )
    )
    latest = result.iloc[-1]

    assert latest["EREM Trend Score"] < 0
    assert latest["EREM Distribution Score"] <= -10
    assert latest["EREM 分數"] < 40
    assert latest["EREM 決策訊號"] == "Exit"
