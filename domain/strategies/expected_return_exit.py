"""Expected Return Exit Model strategy plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from domain.strategy import Strategy

EXPECTED_RETURN_EXIT_STRATEGY_NAME = "expected_return_exit"

EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS = [
    "EREM 分數",
    "EREM 決策訊號",
    "EREM Trend Score",
    "EREM Momentum Score",
    "EREM Volatility Score",
    "EREM Distribution Score",
    "EREM Remaining Return Score",
    "EREM RSI14",
    "EREM MACD",
    "EREM MACD Signal",
    "EREM MACD Histogram",
    "EREM MACD Histogram Declining",
    "EREM Bearish Divergence",
    "EREM ATR14",
    "EREM ATR20MA",
    "EREM OBV",
    "EREM OBV Rising",
    "EREM OBV Falling",
    "EREM Highest60",
    "EREM Lowest60",
    "EREM Mean60",
    "EREM StdDev60",
    "EREM UpperBound",
    "EREM ExpectedReturn",
    "EREM RewardRisk",
    "EREM 理由",
]


@dataclass(frozen=True)
class ExpectedReturnExitConfig:
    ema_short_window: int = 20
    ema_medium_window: int = 50
    ema_long_window: int = 200
    rsi_window: int = 14
    bollinger_window: int = 20
    atr_window: int = 14
    atr_ma_window: int = 20
    volume_window: int = 20
    range_window: int = 60
    divergence_window: int = 20
    flat_price_threshold: float = 0.005
    epsilon: float = 1e-9


class ExpectedReturnExitStrategy(Strategy):
    """Evaluate remaining expected return and exit timing from OHLCV data."""

    name = EXPECTED_RETURN_EXIT_STRATEGY_NAME

    def __init__(self, config: ExpectedReturnExitConfig | None = None) -> None:
        self.config = config or ExpectedReturnExitConfig()

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame([self._empty_result()])

        df = data.copy()
        high = self._numeric_column(df, ("High", "最高"))
        low = self._numeric_column(df, ("Low", "最低"))
        close = self._numeric_column(df, ("Close", "收盤"))
        volume = self._numeric_column(df, ("Volume", "成交量")).fillna(0)

        ema20 = close.ewm(span=self.config.ema_short_window, adjust=False, min_periods=1).mean()
        ema50 = close.ewm(span=self.config.ema_medium_window, adjust=False, min_periods=1).mean()
        ema200 = close.ewm(span=self.config.ema_long_window, adjust=False, min_periods=1).mean()
        ema_alignment = (ema20 > ema50) & (ema50 > ema200)
        price_above_ema20 = close > ema20
        price_below_ema20 = close < ema20
        ema20_below_ema50 = ema20 < ema50
        trend_score = pd.Series(
            np.where(ema_alignment, 15, 0)
            + np.where(price_above_ema20, 10, 0)
            + np.where(price_below_ema20, -10, 0)
            + np.where(ema20_below_ema50, -15, 0),
            index=df.index,
        )

        rsi = self._rsi(close)
        ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
        macd_histogram = macd - macd_signal
        macd_histogram_declining = macd_histogram < macd_histogram.shift(1)
        price_higher_high = high > high.shift(1).rolling(self.config.divergence_window, min_periods=1).max()
        histogram_lower_high = macd_histogram < macd_histogram.shift(1).rolling(
            self.config.divergence_window,
            min_periods=1,
        ).max()
        bearish_divergence = price_higher_high & histogram_lower_high
        momentum_score = pd.Series(
            np.where((rsi > 45) & (rsi < 70), 10, 0)
            + np.where(rsi > 75, -5, 0)
            + np.where(macd > macd_signal, 5, 0)
            + np.where(macd_histogram_declining, -10, 0)
            + np.where(bearish_divergence, -5, 0),
            index=df.index,
        )

        ma20 = close.rolling(self.config.bollinger_window, min_periods=1).mean()
        std20 = close.rolling(self.config.bollinger_window, min_periods=2).std(ddof=0)
        upper_band = ma20 + (2 * std20)
        price_inside_upper_band = upper_band.notna() & (close < upper_band)
        price_touching_upper_band = upper_band.notna() & (close >= upper_band)
        atr14 = self._atr(high=high, low=low, close=close)
        atr20ma = atr14.rolling(self.config.atr_ma_window, min_periods=1).mean()
        atr_contracting = atr14 < atr20ma
        atr_expanding = atr14 > atr20ma
        volatility_score = pd.Series(
            np.where(price_inside_upper_band, 5, 0)
            + np.where(price_touching_upper_band, -5, 0)
            + np.where(atr_contracting, 10, 0)
            + np.where(atr_expanding, -10, 0),
            index=df.index,
        )

        volume_ma20 = volume.rolling(self.config.volume_window, min_periods=1).mean()
        volume_std20 = volume.rolling(self.config.volume_window, min_periods=2).std(ddof=0)
        volume_z = ((volume - volume_ma20) / volume_std20).replace([np.inf, -np.inf], np.nan).fillna(0)
        obv = self._obv(close=close, volume=volume)
        obv_rising = obv > obv.shift(1)
        obv_falling = obv < obv.shift(1)
        price_continues_higher = close > close.shift(1)
        volume_confirms_higher = (volume > volume_ma20) & price_continues_higher
        price_change_abs = close.pct_change().abs().fillna(0)
        volume_spike_price_flat = (volume_z > 2) & (price_change_abs <= self.config.flat_price_threshold)
        distribution_score = pd.Series(
            np.where(obv_rising, 10, 0)
            + np.where(volume_confirms_higher, 10, 0)
            + np.where(obv_falling, -10, 0)
            + np.where(volume_spike_price_flat, -10, 0),
            index=df.index,
        )

        highest60 = high.rolling(self.config.range_window, min_periods=1).max()
        lowest60 = low.rolling(self.config.range_window, min_periods=1).min()
        mean60 = close.rolling(self.config.range_window, min_periods=1).mean()
        std60 = close.rolling(self.config.range_window, min_periods=2).std(ddof=0)
        upper_bound = mean60 + (2 * std60)
        expected_return = ((upper_bound - close) / close).where(close.abs() > self.config.epsilon)
        reward = highest60 - close
        risk = close - lowest60
        reward_risk = (reward / risk).where(risk > self.config.epsilon)
        remaining_return_score = pd.Series(
            np.select([reward_risk > 2, (reward_risk >= 1) & (reward_risk <= 2), reward_risk < 1], [10, 5, -10], default=0)
            + np.select([close < upper_bound, close > upper_bound], [10, -10], default=0),
            index=df.index,
        )

        total_score = (
            trend_score
            + momentum_score
            + volatility_score
            + distribution_score
            + remaining_return_score
        ).clip(lower=0, upper=100)
        decision = self._decision(total_score)

        result = pd.DataFrame(
            {
                "EREM 分數": total_score.round(2),
                "EREM 決策訊號": decision,
                "EREM Trend Score": trend_score.round(2),
                "EREM Momentum Score": momentum_score.round(2),
                "EREM Volatility Score": volatility_score.round(2),
                "EREM Distribution Score": distribution_score.round(2),
                "EREM Remaining Return Score": remaining_return_score.round(2),
                "EREM RSI14": rsi.round(2),
                "EREM MACD": macd.round(2),
                "EREM MACD Signal": macd_signal.round(2),
                "EREM MACD Histogram": macd_histogram.round(2),
                "EREM MACD Histogram Declining": macd_histogram_declining.astype(int),
                "EREM Bearish Divergence": bearish_divergence.astype(int),
                "EREM ATR14": atr14.round(2),
                "EREM ATR20MA": atr20ma.round(2),
                "EREM OBV": obv.round(2),
                "EREM OBV Rising": obv_rising.astype(int),
                "EREM OBV Falling": obv_falling.astype(int),
                "EREM Highest60": highest60.round(2),
                "EREM Lowest60": lowest60.round(2),
                "EREM Mean60": mean60.round(2),
                "EREM StdDev60": std60.round(2),
                "EREM UpperBound": upper_bound.round(2),
                "EREM ExpectedReturn": expected_return.round(4),
                "EREM RewardRisk": reward_risk.round(4),
                "EREM 理由": self._reasons(
                    trend_score=trend_score,
                    momentum_score=momentum_score,
                    volatility_score=volatility_score,
                    distribution_score=distribution_score,
                    remaining_return_score=remaining_return_score,
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

    def _decision(self, total_score: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [total_score >= 85, total_score >= 70, total_score >= 55, total_score >= 40],
                ["Strong Hold", "Hold", "Reduce Position", "Take Profit"],
                default="Exit",
            ),
            index=total_score.index,
        )

    def _reasons(
        self,
        trend_score: pd.Series,
        momentum_score: pd.Series,
        volatility_score: pd.Series,
        distribution_score: pd.Series,
        remaining_return_score: pd.Series,
        decision: pd.Series,
    ) -> pd.Series:
        score_frame = pd.DataFrame(
            {
                "趨勢延續": trend_score,
                "動能": momentum_score,
                "波動": volatility_score,
                "籌碼": distribution_score,
                "剩餘報酬": remaining_return_score,
                "決策": decision,
            }
        )
        return score_frame.apply(
            lambda row: (
                "趨勢延續{趨勢延續:+.0f}；動能{動能:+.0f}；波動{波動:+.0f}；"
                "籌碼{籌碼:+.0f}；剩餘報酬{剩餘報酬:+.0f}；決策={決策}"
            ).format(**row.to_dict()),
            axis=1,
        )

    def _empty_result(self) -> dict[str, object]:
        return {
            "EREM 分數": 0,
            "EREM 決策訊號": "Exit",
            "EREM Trend Score": 0,
            "EREM Momentum Score": 0,
            "EREM Volatility Score": 0,
            "EREM Distribution Score": 0,
            "EREM Remaining Return Score": 0,
            "EREM RSI14": np.nan,
            "EREM MACD": np.nan,
            "EREM MACD Signal": np.nan,
            "EREM MACD Histogram": np.nan,
            "EREM MACD Histogram Declining": 0,
            "EREM Bearish Divergence": 0,
            "EREM ATR14": np.nan,
            "EREM ATR20MA": np.nan,
            "EREM OBV": np.nan,
            "EREM OBV Rising": 0,
            "EREM OBV Falling": 0,
            "EREM Highest60": np.nan,
            "EREM Lowest60": np.nan,
            "EREM Mean60": np.nan,
            "EREM StdDev60": np.nan,
            "EREM UpperBound": np.nan,
            "EREM ExpectedReturn": np.nan,
            "EREM RewardRisk": np.nan,
            "EREM 理由": "資料不足；決策=Exit",
        }
