import pandas as pd

from domain.strategies.low_entry_score import LowEntryScoreStrategy
from domain.strategies.low_entry_score_v2 import LowEntryScoreV2Strategy
from domain.strategies.low_entry_score_v3 import LowEntryScoreV3Strategy
from domain.strategy import Strategy
from domain.strategy import StrategyRegistry
from domain.strategy import available_strategies
from domain.strategy import get_strategy


class EchoStrategy(Strategy):
    name = "echo"

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.assign(ran=True)


def test_strategy_registry_accepts_external_plugin() -> None:
    registry = StrategyRegistry()
    registry.register(EchoStrategy())

    result = registry.get("echo").run(pd.DataFrame({"Close": [1]}))

    assert registry.names() == ["echo"]
    assert result["ran"].tolist() == [True]


def test_builtin_low_entry_strategy_is_discoverable() -> None:
    assert "low_entry_score" in available_strategies()
    assert "low_entry_score_v2" in available_strategies()
    assert "low_entry_score_v3" in available_strategies()
    assert get_strategy("low_entry_score").name == "low_entry_score"
    assert get_strategy("low_entry_score_v2").name == "low_entry_score_v2"
    assert get_strategy("low_entry_score_v3").name == "low_entry_score_v3"


def test_low_entry_score_strategy_buy_path_is_deterministic() -> None:
    closes = [90 + (20 * index / 69) for index in range(70)]
    highs = [close + 2 for close in closes]
    lows = [close - 2 for close in closes]
    opens = [close - 1 for close in closes]
    volumes = [1000] * 69 + [5000]
    pe_values = [10] * 69 + [7]

    highs[10] = 220
    lows[-1] = 100
    opens[-1] = 112
    closes[-1] = 110
    highs[-1] = 114

    strategy = LowEntryScoreStrategy()
    result = strategy.run(
        pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
                "PE": pe_values,
            }
        )
    )
    latest = result.iloc[-1]

    assert latest["低點決策"] == "BUY"
    assert latest["低點分數"] >= 75
    assert latest["60日位置"] < 0.2
    assert latest["策略停損"] < latest["Close"]
    assert latest["策略停利"] > latest["Close"]


def test_low_entry_score_strategy_handles_flat_price_range() -> None:
    strategy = LowEntryScoreStrategy()
    result = strategy.run(
        pd.DataFrame(
            {
                "Open": [100, 100],
                "High": [100, 100],
                "Low": [100, 100],
                "Close": [100, 100],
                "Volume": [1000, 1000],
                "PE": [10, 10],
            }
        )
    )
    latest = result.iloc[-1]

    assert pd.isna(latest["60日位置"])
    assert latest["低點決策"] == "WAIT"


def test_low_entry_score_v2_applies_high_position_penalty() -> None:
    strategy = LowEntryScoreV2Strategy()
    result = strategy.run(
        pd.DataFrame(
            {
                "Open": [90] * 60,
                "High": [100] * 60,
                "Low": [0] + [90] * 59,
                "Close": [90] * 60,
                "Volume": [1000] * 60,
                "PE": [None] * 60,
            }
        )
    )
    latest = result.iloc[-1]

    assert latest["60日位置"] == 0.9
    assert latest["低點分數"] == -20
    assert latest["低點決策"] == "WAIT"
    assert "價格位置-20" in latest["低點理由"]


def test_low_entry_score_v2_detects_only_fresh_ma20_reversal() -> None:
    strategy = LowEntryScoreV2Strategy()
    result = strategy.run(
        pd.DataFrame(
            {
                "Open": [100] * 19 + [90, 110],
                "High": [102] * 19 + [92, 112],
                "Low": [98] * 19 + [88, 108],
                "Close": [100] * 19 + [90, 110],
                "Volume": [1000] * 21,
                "PE": [None] * 21,
            }
        )
    )

    assert result.iloc[-2]["Trend Reversal"] == 0
    assert result.iloc[-1]["Trend Reversal"] == 1
    assert result.iloc[-1]["Close Above MA20"] == 1
    assert "趨勢+15" in result.iloc[-1]["低點理由"]


def test_low_entry_score_v2_volume_spike_requires_down_day() -> None:
    strategy = LowEntryScoreV2Strategy()
    base = {
        "Open": [100] * 20,
        "High": [102] * 20,
        "Low": [98] * 20,
        "Close": [100] * 20,
        "Volume": [1000] * 19 + [5000],
        "PE": [None] * 20,
    }

    down_day = pd.DataFrame({**base, "Close": [100] * 19 + [95]})
    up_day = pd.DataFrame({**base, "Close": [100] * 19 + [105]})

    down_result = strategy.run(down_day).iloc[-1]
    up_result = strategy.run(up_day).iloc[-1]

    assert down_result["成交量Z"] > 2
    assert down_result["Volume Spike Down"] == 1
    assert up_result["成交量Z"] > 2
    assert up_result["Volume Spike Down"] == 0


def test_low_entry_score_v2_volume_dry_adds_score() -> None:
    strategy = LowEntryScoreV2Strategy()
    result = strategy.run(
        pd.DataFrame(
            {
                "Open": [100] * 20,
                "High": [102] * 20,
                "Low": [98] * 20,
                "Close": [100] * 20,
                "Volume": [2000] * 19 + [100],
                "PE": [None] * 20,
            }
        )
    )
    latest = result.iloc[-1]

    assert latest["成交量Z"] < -1
    assert latest["Volume Dry"] == 1
    assert "成交量+10" in latest["低點理由"]


def test_low_entry_score_v3_scores_advanced_components() -> None:
    closes = [100 + index * 0.2 for index in range(220)]
    highs = [close + 2 for close in closes]
    lows = [close - 2 for close in closes]
    volumes = [1000] * 219 + [3000]

    strategy = LowEntryScoreV3Strategy()
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

    assert latest["EMA Alignment"] == 1
    assert latest["Price Above EMA20"] == 1
    assert latest["MACD Bullish"] == 1
    assert latest["Volume Above MA20"] == 1
    assert latest["OBV Rising"] == 1
    assert latest["Support Holding"] == 1
    assert latest["Higher Low"] == 1
    assert latest["低點分數"] >= 70


def test_low_entry_score_v3_atr_contraction_and_near_lower_band() -> None:
    closes = [100] * 25
    highs = [115] * 15 + [102] * 10
    lows = [85] * 15 + [98] * 9 + [97]
    volumes = [1000] * 25

    strategy = LowEntryScoreV3Strategy()
    result = strategy.run(
        pd.DataFrame(
            {
                "Open": [100] * 25,
                "High": highs,
                "Low": lows,
                "Close": closes[:-1] + [98],
                "Volume": volumes,
            }
        )
    )
    latest = result.iloc[-1]

    assert latest["Near Lower Band"] == 1
    assert latest["ATR Contraction"] == 1
