"""Refined Low Entry Score v2 strategy plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from domain.strategy import Strategy

LOW_ENTRY_STRATEGY_NAME = "low_entry_score_v2"

LOW_ENTRY_OUTPUT_COLUMNS = [
    "低點分數",
    "低點決策",
    "60日位置",
    "MA20",
    "MA60",
    "Trend Reversal",
    "Close Above MA20",
    "RSI14",
    "布林下緣",
    "成交量Z",
    "Volume Spike Down",
    "Volume Dry",
    "Hammer",
    "MACD",
    "MACD Signal",
    "ATR14",
    "策略停損",
    "策略停利",
    "PE Ratio",
    "低點理由",
]


@dataclass(frozen=True)
class LowEntryScoreV2Config:
    position_window: int = 60
    ma_short_window: int = 20
    ma_long_window: int = 60
    rsi_window: int = 14
    bollinger_window: int = 20
    volume_window: int = 20
    atr_window: int = 14
    epsilon: float = 1e-9


class LowEntryScoreV2Strategy(Strategy):
    """Score refined low-entry setups from daily OHLCV data.

    Score format:
    - Numeric score in "低點分數"; v2 range is -20 to 100.
    - BUY when score >= 75, WATCH when 60 <= score < 75, otherwise WAIT.
    """

    name = LOW_ENTRY_STRATEGY_NAME

    def __init__(self, config: LowEntryScoreV2Config | None = None) -> None:
        self.config = config or LowEntryScoreV2Config()

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame([self._empty_result()])

        df = data.copy()
        open_price = self._numeric_column(df, ("Open", "開盤"))
        high = self._numeric_column(df, ("High", "最高"))
        low = self._numeric_column(df, ("Low", "最低"))
        close = self._numeric_column(df, ("Close", "收盤"))
        volume = self._numeric_column(df, ("Volume", "成交量")).fillna(0)
        pe = self._numeric_column(df, ("PE", "本益比"))

        high60 = high.rolling(self.config.position_window, min_periods=1).max()
        low60 = low.rolling(self.config.position_window, min_periods=1).min()
        position_range = high60 - low60
        position60 = ((close - low60) / position_range).where(position_range.abs() > self.config.epsilon)
        price_score = pd.Series(np.select([position60 < 0.35, position60 > 0.8], [15, -20], default=0), index=df.index)

        ma20 = close.rolling(self.config.ma_short_window, min_periods=1).mean()
        ma60 = close.rolling(self.config.ma_long_window, min_periods=1).mean()
        trend_reversal = (close > ma20) & (close.shift(1) < ma20.shift(1))
        close_above_ma20 = close > ma20
        trend_score = pd.Series(
            np.select([trend_reversal, close_above_ma20], [15, 10], default=0),
            index=df.index,
        )

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(self.config.rsi_window, min_periods=1).mean()
        avg_loss = loss.rolling(self.config.rsi_window, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100, 50)).fillna(50)
        rsi_score = pd.Series(
            np.select([rsi < 30, (rsi >= 30) & (rsi < 35)], [15, 10], default=0),
            index=df.index,
        ) + pd.Series(np.where(rsi > rsi.shift(1), 5, 0), index=df.index)

        std20 = close.rolling(self.config.bollinger_window, min_periods=2).std(ddof=0)
        lower_band = ma20 - (2 * std20)
        bollinger_score = pd.Series(
            np.where(lower_band.notna() & (close < lower_band), 15, 0),
            index=df.index,
        )

        volume_ma20 = volume.rolling(self.config.volume_window, min_periods=1).mean()
        volume_std20 = volume.rolling(self.config.volume_window, min_periods=2).std(ddof=0)
        volume_z = ((volume - volume_ma20) / volume_std20).replace([np.inf, -np.inf], np.nan).fillna(0)
        volume_spike_down = (volume_z > 2) & (close < close.shift(1))
        volume_dry = volume_z < -1
        volume_score = pd.Series(
            np.select([volume_spike_down, volume_dry], [15, 10], default=0),
            index=df.index,
        )

        body = (close - open_price).abs()
        lower_shadow = pd.Series(np.minimum(open_price, close), index=df.index) - low
        hammer = lower_shadow > (2 * body)
        candle_score = pd.Series(np.where(hammer, 10, 0), index=df.index)

        ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
        macd_score = pd.Series(np.where(macd > macd_signal, 10, 0), index=df.index)

        pe_avg = pe.rolling(self.config.position_window, min_periods=1).mean()
        pe_ratio = (pe / pe_avg).where(pe_avg.abs() > self.config.epsilon)
        pe_score = pd.Series(
            np.select([pe_ratio < 0.8, (pe_ratio >= 0.8) & (pe_ratio <= 1.2)], [10, 5], default=0),
            index=df.index,
        ).where(pe_ratio.notna(), 0)

        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = true_range.rolling(self.config.atr_window, min_periods=1).mean()
        strategy_stop_loss = close - (2 * atr14)
        strategy_take_profit = close + (3 * atr14)

        total_score = (
            price_score
            + trend_score
            + rsi_score
            + bollinger_score
            + volume_score
            + candle_score
            + macd_score
            + pe_score
        ).clip(lower=-20, upper=100)
        decision = pd.Series(
            np.select([total_score >= 75, total_score >= 60], ["BUY", "WATCH"], default="WAIT"),
            index=df.index,
        )

        result = pd.DataFrame(
            {
                "低點分數": total_score.round(2),
                "低點決策": decision,
                "60日位置": position60.round(4),
                "MA20": ma20.round(2),
                "MA60": ma60.round(2),
                "Trend Reversal": trend_reversal.astype(int),
                "Close Above MA20": close_above_ma20.astype(int),
                "RSI14": rsi.round(2),
                "布林下緣": lower_band.round(2),
                "成交量Z": volume_z.round(2),
                "Volume Spike Down": volume_spike_down.astype(int),
                "Volume Dry": volume_dry.astype(int),
                "Hammer": hammer.astype(int),
                "MACD": macd.round(2),
                "MACD Signal": macd_signal.round(2),
                "ATR14": atr14.round(2),
                "策略停損": strategy_stop_loss.round(2),
                "策略停利": strategy_take_profit.round(2),
                "PE Ratio": pe_ratio.round(4),
                "低點理由": self._reasons(
                    price_score=price_score,
                    trend_score=trend_score,
                    rsi_score=rsi_score,
                    bollinger_score=bollinger_score,
                    volume_score=volume_score,
                    candle_score=candle_score,
                    macd_score=macd_score,
                    pe_score=pe_score,
                    decision=decision,
                ),
            },
            index=df.index,
        )

        return pd.concat([df, result], axis=1)

    def _numeric_column(self, data: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in data.columns:
                return pd.to_numeric(data[name], errors="coerce")
        return pd.Series(np.nan, index=data.index, dtype="float64")

    def _reasons(
        self,
        price_score: pd.Series,
        trend_score: pd.Series,
        rsi_score: pd.Series,
        bollinger_score: pd.Series,
        volume_score: pd.Series,
        candle_score: pd.Series,
        macd_score: pd.Series,
        pe_score: pd.Series,
        decision: pd.Series,
    ) -> pd.Series:
        score_frame = pd.DataFrame(
            {
                "價格位置": price_score,
                "趨勢": trend_score,
                "RSI": rsi_score,
                "布林": bollinger_score,
                "成交量": volume_score,
                "K線": candle_score,
                "MACD": macd_score,
                "PE": pe_score,
                "決策": decision,
            }
        )
        return score_frame.apply(
            lambda row: (
                "價格位置{價格位置:+.0f}；趨勢+{趨勢:.0f}；RSI+{RSI:.0f}；"
                "布林+{布林:.0f}；成交量+{成交量:.0f}；K線+{K線:.0f}；"
                "MACD+{MACD:.0f}；PE+{PE:.0f}；決策={決策}"
            ).format(**row.to_dict()),
            axis=1,
        )

    def _empty_result(self) -> dict[str, object]:
        return {
            "低點分數": 0,
            "低點決策": "WAIT",
            "60日位置": np.nan,
            "MA20": np.nan,
            "MA60": np.nan,
            "Trend Reversal": 0,
            "Close Above MA20": 0,
            "RSI14": np.nan,
            "布林下緣": np.nan,
            "成交量Z": np.nan,
            "Volume Spike Down": 0,
            "Volume Dry": 0,
            "Hammer": 0,
            "MACD": np.nan,
            "MACD Signal": np.nan,
            "ATR14": np.nan,
            "策略停損": np.nan,
            "策略停利": np.nan,
            "PE Ratio": np.nan,
            "低點理由": "資料不足；決策=WAIT",
        }
