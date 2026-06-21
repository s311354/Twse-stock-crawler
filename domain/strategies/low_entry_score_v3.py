"""Advanced Low Entry Score v3 strategy plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from domain.strategy import Strategy

LOW_ENTRY_STRATEGY_NAME = "low_entry_score_v3"

LOW_ENTRY_OUTPUT_COLUMNS = [
    "低點分數",
    "低點決策",
    "EMA20",
    "EMA50",
    "EMA200",
    "EMA Alignment",
    "Price Above EMA20",
    "RSI14",
    "MACD",
    "MACD Signal",
    "MACD Histogram",
    "MACD Bullish",
    "布林下緣",
    "Near Lower Band",
    "ATR14",
    "ATR20MA",
    "ATR Contraction",
    "成交量MA20",
    "Volume Above MA20",
    "OBV",
    "OBV Rising",
    "Support Holding",
    "Higher Low",
    "策略停損",
    "策略停利",
    "低點理由",
]


@dataclass(frozen=True)
class LowEntryScoreV3Config:
    ema_short_window: int = 20
    ema_medium_window: int = 50
    ema_long_window: int = 200
    rsi_window: int = 14
    bollinger_window: int = 20
    atr_window: int = 14
    atr_ma_window: int = 20
    volume_window: int = 20
    support_window: int = 20
    higher_low_window: int = 5
    epsilon: float = 1e-9


class LowEntryScoreV3Strategy(Strategy):
    """Advanced LES model combining trend, momentum, volatility, volume, and structure."""

    name = LOW_ENTRY_STRATEGY_NAME

    def __init__(self, config: LowEntryScoreV3Config | None = None) -> None:
        self.config = config or LowEntryScoreV3Config()

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame([self._empty_result()])

        df = data.copy()
        #open_price = self._numeric_column(df, ("Open", "開盤"))
        high = self._numeric_column(df, ("High", "最高"))
        low = self._numeric_column(df, ("Low", "最低"))
        close = self._numeric_column(df, ("Close", "收盤"))
        volume = self._numeric_column(df, ("Volume", "成交量")).fillna(0)

        ema20 = close.ewm(span=self.config.ema_short_window, adjust=False, min_periods=1).mean()
        ema50 = close.ewm(span=self.config.ema_medium_window, adjust=False, min_periods=1).mean()
        ema200 = close.ewm(span=self.config.ema_long_window, adjust=False, min_periods=1).mean()
        ema_alignment = (ema20 > ema50) & (ema50 > ema200)
        price_above_ema20 = close > ema20
        trend_score = pd.Series(
            np.where(ema_alignment, 15, 0) + np.where(price_above_ema20, 10, 0),
            index=df.index,
        )

        rsi = self._rsi(close)
        rsi_score = pd.Series(
            np.select([(rsi > 40) & (rsi < 60), rsi < 40], [10, 5], default=0),
            index=df.index,
        )
        ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
        macd_histogram = macd - macd_signal
        macd_bullish = macd > macd_signal
        momentum_score = rsi_score + pd.Series(np.where(macd_bullish, 10, 0), index=df.index)

        ma20 = close.rolling(self.config.bollinger_window, min_periods=1).mean()
        std20 = close.rolling(self.config.bollinger_window, min_periods=2).std(ddof=0)
        upper_band = ma20 + (2 * std20)
        lower_band = ma20 - (2 * std20)
        band_width = upper_band - lower_band
        lower_band_position = ((close - lower_band) / band_width).where(band_width.abs() > self.config.epsilon)
        near_lower_band = lower_band.notna() & (lower_band_position <= 0.2)

        atr14 = self._atr(high=high, low=low, close=close)
        atr20ma = atr14.rolling(self.config.atr_ma_window, min_periods=1).mean()
        atr_contraction = atr14 < atr20ma
        volatility_score = pd.Series(
            np.where(near_lower_band, 10, 0) + np.where(atr_contraction, 10, 0),
            index=df.index,
        )

        volume_ma20 = volume.rolling(self.config.volume_window, min_periods=1).mean()
        volume_above_ma20 = volume > volume_ma20
        obv = self._obv(close=close, volume=volume)
        obv_rising = obv > obv.shift(1)
        volume_score = pd.Series(
            np.where(volume_above_ma20, 10, 0) + np.where(obv_rising, 10, 0),
            index=df.index,
        )

        previous_swing_low = low.shift(1).rolling(self.config.support_window, min_periods=1).min()
        support_holding = low >= previous_swing_low
        current_low_window = low.rolling(self.config.higher_low_window, min_periods=1).min()
        previous_low_window = current_low_window.shift(self.config.higher_low_window)
        higher_low = current_low_window > previous_low_window
        structure_score = pd.Series(
            np.where(support_holding, 10, 0) + np.where(higher_low, 5, 0),
            index=df.index,
        )

        total_score = (trend_score + momentum_score + volatility_score + volume_score + structure_score).clip(lower=0, upper=100)
        decision = pd.Series(
            np.select([total_score >= 75, total_score >= 60], ["BUY", "WATCH"], default="WAIT"),
            index=df.index,
        )
        strategy_stop_loss = close - (2 * atr14)
        strategy_take_profit = close + (3 * atr14)

        result = pd.DataFrame(
            {
                "低點分數": total_score.round(2),
                "低點決策": decision,
                "EMA20": ema20.round(2),
                "EMA50": ema50.round(2),
                "EMA200": ema200.round(2),
                "EMA Alignment": ema_alignment.astype(int),
                "Price Above EMA20": price_above_ema20.astype(int),
                "RSI14": rsi.round(2),
                "MACD": macd.round(2),
                "MACD Signal": macd_signal.round(2),
                "MACD Histogram": macd_histogram.round(2),
                "MACD Bullish": macd_bullish.astype(int),
                "布林下緣": lower_band.round(2),
                "Near Lower Band": near_lower_band.astype(int),
                "ATR14": atr14.round(2),
                "ATR20MA": atr20ma.round(2),
                "ATR Contraction": atr_contraction.astype(int),
                "成交量MA20": volume_ma20.round(2),
                "Volume Above MA20": volume_above_ma20.astype(int),
                "OBV": obv.round(2),
                "OBV Rising": obv_rising.astype(int),
                "Support Holding": support_holding.astype(int),
                "Higher Low": higher_low.astype(int),
                "策略停損": strategy_stop_loss.round(2),
                "策略停利": strategy_take_profit.round(2),
                "低點理由": self._reasons(
                    trend_score=trend_score,
                    momentum_score=momentum_score,
                    volatility_score=volatility_score,
                    volume_score=volume_score,
                    structure_score=structure_score,
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

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(self.config.rsi_window, min_periods=1).mean()
        avg_loss = loss.rolling(self.config.rsi_window, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100, 50)).fillna(50)

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(self.config.atr_window, min_periods=1).mean()

    def _obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.select([close > close.shift(1), close < close.shift(1)], [1, -1], default=0)
        return pd.Series(direction * volume, index=close.index).cumsum()

    def _reasons(
        self,
        trend_score: pd.Series,
        momentum_score: pd.Series,
        volatility_score: pd.Series,
        volume_score: pd.Series,
        structure_score: pd.Series,
        decision: pd.Series,
    ) -> pd.Series:
        score_frame = pd.DataFrame(
            {
                "趨勢": trend_score,
                "動能": momentum_score,
                "波動": volatility_score,
                "量能": volume_score,
                "結構": structure_score,
                "決策": decision,
            }
        )
        return score_frame.apply(
            lambda row: (
                "趨勢+{趨勢:.0f}；動能+{動能:.0f}；波動+{波動:.0f}；"
                "量能+{量能:.0f}；結構+{結構:.0f}；決策={決策}"
            ).format(**row.to_dict()),
            axis=1,
        )

    def _empty_result(self) -> dict[str, object]:
        return {
            "低點分數": 0,
            "低點決策": "WAIT",
            "EMA20": np.nan,
            "EMA50": np.nan,
            "EMA200": np.nan,
            "EMA Alignment": 0,
            "Price Above EMA20": 0,
            "RSI14": np.nan,
            "MACD": np.nan,
            "MACD Signal": np.nan,
            "MACD Histogram": np.nan,
            "MACD Bullish": 0,
            "布林下緣": np.nan,
            "Near Lower Band": 0,
            "ATR14": np.nan,
            "ATR20MA": np.nan,
            "ATR Contraction": 0,
            "成交量MA20": np.nan,
            "Volume Above MA20": 0,
            "OBV": np.nan,
            "OBV Rising": 0,
            "Support Holding": 0,
            "Higher Low": 0,
            "策略停損": np.nan,
            "策略停利": np.nan,
            "低點理由": "資料不足；決策=WAIT",
        }
