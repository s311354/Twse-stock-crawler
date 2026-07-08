import math

import pandas as pd

from domain.strategies.expected_return_exit_v2 import ExpectedReturnExitV2Strategy
from domain.strategy import available_strategies
from domain.strategy import get_strategy


def test_expected_return_exit_v2_strategy_is_discoverable() -> None:
    assert "expected_return_exit_v2" in available_strategies()
    assert get_strategy("expected_return_exit_v2").name == "expected_return_exit_v2"


def test_expected_return_exit_v2_scores_strong_hold_with_120_day_upside() -> None:
    closes = [100 + index * 0.02 + math.sin(index / 2) * 0.5 for index in range(260)]
    closes[-3] = closes[-4] - 0.5
    closes[-2] = closes[-3] - 0.2
    closes[-1] = closes[-2] + 0.6
    highs = [close + 3 for close in closes]
    lows = [close - 10 for close in closes]
    highs[-80] = 170
    volumes = [1000] * 259 + [3000]

    result = ExpectedReturnExitV2Strategy().run(
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
    assert latest["EREM Momentum Score"] == 20
    assert latest["EREM MACD Histogram Rising"] == 1
    assert latest["EREM Highest120"] == 170
    assert latest["EREM RewardRisk"] > 2
    assert latest["EREM 分數"] >= 85


def test_expected_return_exit_v2_penalizes_exit_risk() -> None:
    closes = [170 - index * 0.2 for index in range(260)]
    highs = [close + 1 for close in closes]
    lows = [close - 1 for close in closes]
    volumes = [1000] * 259 + [5000]
    closes[-1] = closes[-2]
    highs[-1] = highs[-2]
    lows[-1] = lows[-2]

    result = ExpectedReturnExitV2Strategy().run(
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
