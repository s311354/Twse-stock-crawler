"""Expected Return Exit Model v3 strategy plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from domain.strategy import Strategy

EXPECTED_RETURN_EXIT_STRATEGY_NAME = "expected_return_exit_v3"
EREM_V3_MAX_RAW_SCORE = 105.0

EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS = [
    "EREM 分數",
    "EREM Raw Score",
    "EREM Holding Confidence",
    "EREM Entry LES",
    "EREM 決策訊號",
    "EREM Trend Score",
    "EREM Momentum Score",
    "EREM Volatility Score",
    "EREM Distribution Score",
    "EREM Remaining Return Score",
    "EREM Relative Strength Score",
    "EREM Structure Score",
    "EREM Entry Quality Score",
    "EREM Turnover Score",
    "EREM VWAP Score",
    "EREM EMA20",
    "EREM EMA50",
    "EREM EMA200",
    "EREM RSI14",
    "EREM MACD",
    "EREM MACD Signal",
    "EREM MACD Histogram",
    "EREM MACD Histogram Rising",
    "EREM MACD Histogram Falling 2D",
    "EREM Bearish Divergence",
    "EREM ATR14",
    "EREM ATR20MA",
    "EREM Bollinger Upper",
    "EREM OBV",
    "EREM OBV Rising",
    "EREM Distribution Warning",
    "EREM Volume MA20",
    "EREM Highest120",
    "EREM Lowest120",
    "EREM Mean120",
    "EREM StdDev120",
    "EREM UpperBound",
    "EREM ExpectedReturn",
    "EREM RewardRisk",
    "EREM Stock Return60",
    "EREM TAIEX Return60",
    "EREM Relative Strength60",
    "EREM Higher High",
    "EREM Higher Low",
    "EREM Turnover Strength",
    "EREM Turnover Source",
    "EREM VWAP60",
    "EREM 理由",
]


@dataclass(frozen=True)
class ExpectedReturnExitV3Config:
    ema_short_window: int = 20
    ema_medium_window: int = 50
    ema_long_window: int = 200
    rsi_window: int = 14
    bollinger_window: int = 20
    atr_window: int = 14
    atr_ma_window: int = 20
    volume_window: int = 20
    range_window: int = 120
    relative_return_window: int = 60
    structure_window: int = 20
    divergence_window: int = 20
    obv_falling_window: int = 5
    vwap_window: int = 60
    turnover_strength_threshold: float = 1.2
    excessive_atr_ratio: float = 1.2
    epsilon: float = 1e-9


class ExpectedReturnExitV3Strategy(Strategy):
    """Score exit risk using OHLCV plus optional TAIEX and entry LES history.

    The requested component weights total 105 even though the model specifies a
    0-100 result. Component scores retain their stated weights and the raw score
    is normalized by 100 / 105 before applying the decision thresholds.
    """

    name = EXPECTED_RETURN_EXIT_STRATEGY_NAME

    def __init__(self, config: ExpectedReturnExitV3Config | None = None) -> None:
        self.config = config or ExpectedReturnExitV3Config()

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
        # The v2 15:10 trend ratio is scaled to the v3 component maximum of 20.
        trend_score = pd.Series(
            np.where(ema_alignment, 12, 0) + np.where(price_above_ema20, 8, 0),
            index=df.index,
            dtype="float64",
        )

        rsi = self._rsi(close)
        ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
        macd_histogram = macd - macd_signal
        macd_histogram_rising = macd_histogram > macd_histogram.shift(1)
        histogram_falling_two_days = (
            macd_histogram.diff().lt(0).rolling(2, min_periods=2).sum().eq(2)
        )
        previous_price_high = high.shift(1).rolling(
            self.config.divergence_window,
            min_periods=2,
        ).max()
        previous_histogram_high = macd_histogram.shift(1).rolling(
            self.config.divergence_window,
            min_periods=2,
        ).max()
        price_higher_high = high > previous_price_high
        bearish_divergence = price_higher_high & (macd_histogram < previous_histogram_high)
        momentum_exhaustion = histogram_falling_two_days | bearish_divergence
        momentum_score = pd.Series(
            np.where((rsi >= 45) & (rsi <= 70), 5, 0)
            + np.where(macd > macd_signal, 5, 0)
            + np.where(macd_histogram_rising, 5, 0)
            - np.where(momentum_exhaustion, 5, 0),
            index=df.index,
            dtype="float64",
        ).clip(lower=0, upper=15)

        middle_band = close.rolling(self.config.bollinger_window, min_periods=1).mean()
        std20 = close.rolling(self.config.bollinger_window, min_periods=2).std(ddof=0)
        upper_band = middle_band + (2 * std20)
        atr14 = self._atr(high=high, low=low, close=close)
        atr20ma = atr14.rolling(self.config.atr_ma_window, min_periods=1).mean()
        atr_stable = atr14 <= atr20ma
        excessive_atr = atr14 > (atr20ma * self.config.excessive_atr_ratio)
        below_upper_band = upper_band.notna() & (close < upper_band)
        upper_band_exhaustion = upper_band.notna() & (close >= upper_band) & momentum_exhaustion
        volatility_score = pd.Series(
            np.where(atr_stable, 5, 0)
            + np.where(below_upper_band, 5, 0)
            - np.where(excessive_atr, 5, 0)
            - np.where(upper_band_exhaustion, 5, 0),
            index=df.index,
            dtype="float64",
        ).clip(lower=0, upper=10)

        volume_ma20 = volume.rolling(self.config.volume_window, min_periods=1).mean()
        obv = self._obv(close=close, volume=volume)
        obv_rising = obv > obv.shift(1)
        obv_falling_five_days = (
            obv.diff().lt(0).rolling(
                self.config.obv_falling_window,
                min_periods=self.config.obv_falling_window,
            ).sum().eq(self.config.obv_falling_window)
        )
        price_rising = close > close.shift(1)
        price_falling = close < close.shift(1)
        volume_above_ma20 = volume > volume_ma20
        volume_confirms_price = volume_above_ma20 & price_rising
        volume_price_weakness = volume_above_ma20 & price_falling
        turnover, turnover_source = self._turnover_input(df=df, volume=volume)
        turnover_ma20 = turnover.rolling(self.config.volume_window, min_periods=1).mean()
        turnover_strength = (turnover / turnover_ma20).where(
            turnover_ma20.abs() > self.config.epsilon
        )
        turnover_price_weakness = (
            (turnover_strength >= self.config.turnover_strength_threshold) & price_falling
        )
        previous_obv_high = obv.shift(1).rolling(
            self.config.divergence_window,
            min_periods=2,
        ).max()
        obv_divergence = price_higher_high & (obv < previous_obv_high)
        distribution_warning = (
            volume_price_weakness
            | turnover_price_weakness
            | obv_falling_five_days
            | obv_divergence
        )
        distribution_score = pd.Series(
            np.where(obv_rising, 5, 0)
            + np.where(volume_confirms_price, 5, 0)
            + np.where(~distribution_warning, 5, 0),
            index=df.index,
            dtype="float64",
        ).clip(lower=0, upper=15)

        highest120 = high.rolling(self.config.range_window, min_periods=1).max()
        lowest120 = low.rolling(self.config.range_window, min_periods=1).min()
        mean120 = close.rolling(self.config.range_window, min_periods=1).mean()
        std120 = close.rolling(self.config.range_window, min_periods=2).std(ddof=0)
        upper_bound = mean120 + (2 * std120)
        expected_return = ((upper_bound - close) / close).where(close.abs() > self.config.epsilon)
        reward = highest120 - close
        risk = close - lowest120
        reward_risk = (reward / risk).where(risk > self.config.epsilon)
        remaining_return_score = pd.Series(
            np.select(
                [reward_risk > 2, (reward_risk >= 1) & (reward_risk <= 2)],
                [10, 5],
                default=0,
            )
            + np.where(upper_bound.notna() & (close < upper_bound), 5, 0),
            index=df.index,
            dtype="float64",
        ).clip(lower=0, upper=15)

        taiex_close = self._numeric_column(
            df,
            ("TAIEX Close", "TAIEX收盤", "大盤收盤", "Benchmark Close"),
        )
        stock_return60 = close.pct_change(self.config.relative_return_window, fill_method=None)
        taiex_return60 = taiex_close.pct_change(self.config.relative_return_window, fill_method=None)
        relative_strength60 = stock_return60 - taiex_return60
        relative_strength_available = relative_strength60.notna()
        relative_strength_score = pd.Series(
            np.where(relative_strength_available & (relative_strength60 > 0), 10, 0),
            index=df.index,
            dtype="float64",
        )

        current_swing_high = high.rolling(
            self.config.structure_window,
            min_periods=self.config.structure_window,
        ).max()
        previous_swing_high = high.shift(self.config.structure_window).rolling(
            self.config.structure_window,
            min_periods=self.config.structure_window,
        ).max()
        current_swing_low = low.rolling(
            self.config.structure_window,
            min_periods=self.config.structure_window,
        ).min()
        previous_swing_low = low.shift(self.config.structure_window).rolling(
            self.config.structure_window,
            min_periods=self.config.structure_window,
        ).min()
        higher_high = current_swing_high > previous_swing_high
        higher_low = current_swing_low > previous_swing_low
        structure_score = pd.Series(
            np.where(higher_high & higher_low, 5, 0),
            index=df.index,
            dtype="float64",
        )

        entry_les = self._numeric_column(
            df,
            ("Entry LES", "Entry_LES", "進場 LES", "進場LES", "低點分數"),
        ).ffill().clip(lower=0, upper=100)
        entry_les_available = entry_les.notna()
        entry_quality_score = pd.Series(
            np.select([entry_les >= 75, entry_les >= 60], [5, 3], default=0),
            index=df.index,
            dtype="float64",
        )

        healthy_turnover = (
            (turnover_strength >= self.config.turnover_strength_threshold)
            & (close >= close.shift(1))
        )
        turnover_score = pd.Series(
            np.where(healthy_turnover, 5, 0),
            index=df.index,
            dtype="float64",
        )

        rolling_price_volume = (close * volume).rolling(self.config.vwap_window, min_periods=1).sum()
        rolling_volume = volume.rolling(self.config.vwap_window, min_periods=1).sum()
        vwap60 = (rolling_price_volume / rolling_volume).where(rolling_volume > self.config.epsilon)
        vwap_safety = vwap60.notna() & (close < vwap60)
        vwap_score = pd.Series(
            np.where(vwap_safety, 5, 0),
            index=df.index,
            dtype="float64",
        )

        raw_score = (
            trend_score
            + momentum_score
            + volatility_score
            + distribution_score
            + remaining_return_score
            + relative_strength_score
            + structure_score
            + entry_quality_score
            + turnover_score
            + vwap_score
        ).clip(lower=0, upper=EREM_V3_MAX_RAW_SCORE)
        current_score = (raw_score * 100 / EREM_V3_MAX_RAW_SCORE).clip(lower=0, upper=100)
        holding_confidence = pd.Series(
            np.where(
                entry_les_available,
                (0.6 * current_score) + (0.4 * entry_les),
                current_score,
            ),
            index=df.index,
            dtype="float64",
        ).clip(lower=0, upper=100)
        decision = self._decision(holding_confidence)

        result = pd.DataFrame(
            {
                "EREM 分數": current_score.round(2),
                "EREM Raw Score": raw_score.round(2),
                "EREM Holding Confidence": holding_confidence.round(2),
                "EREM Entry LES": entry_les.round(2),
                "EREM 決策訊號": decision,
                "EREM Trend Score": trend_score.round(2),
                "EREM Momentum Score": momentum_score.round(2),
                "EREM Volatility Score": volatility_score.round(2),
                "EREM Distribution Score": distribution_score.round(2),
                "EREM Remaining Return Score": remaining_return_score.round(2),
                "EREM Relative Strength Score": relative_strength_score.round(2),
                "EREM Structure Score": structure_score.round(2),
                "EREM Entry Quality Score": entry_quality_score.round(2),
                "EREM Turnover Score": turnover_score.round(2),
                "EREM VWAP Score": vwap_score.round(2),
                "EREM EMA20": ema20.round(2),
                "EREM EMA50": ema50.round(2),
                "EREM EMA200": ema200.round(2),
                "EREM RSI14": rsi.round(2),
                "EREM MACD": macd.round(4),
                "EREM MACD Signal": macd_signal.round(4),
                "EREM MACD Histogram": macd_histogram.round(4),
                "EREM MACD Histogram Rising": macd_histogram_rising.astype(int),
                "EREM MACD Histogram Falling 2D": histogram_falling_two_days.astype(int),
                "EREM Bearish Divergence": bearish_divergence.astype(int),
                "EREM ATR14": atr14.round(2),
                "EREM ATR20MA": atr20ma.round(2),
                "EREM Bollinger Upper": upper_band.round(2),
                "EREM OBV": obv.round(2),
                "EREM OBV Rising": obv_rising.astype(int),
                "EREM Distribution Warning": distribution_warning.astype(int),
                "EREM Volume MA20": volume_ma20.round(2),
                "EREM Highest120": highest120.round(2),
                "EREM Lowest120": lowest120.round(2),
                "EREM Mean120": mean120.round(2),
                "EREM StdDev120": std120.round(2),
                "EREM UpperBound": upper_bound.round(2),
                "EREM ExpectedReturn": expected_return.round(4),
                "EREM RewardRisk": reward_risk.round(4),
                "EREM Stock Return60": stock_return60.round(4),
                "EREM TAIEX Return60": taiex_return60.round(4),
                "EREM Relative Strength60": relative_strength60.round(4),
                "EREM Higher High": higher_high.astype(int),
                "EREM Higher Low": higher_low.astype(int),
                "EREM Turnover Strength": turnover_strength.round(4),
                "EREM Turnover Source": turnover_source,
                "EREM VWAP60": vwap60.round(2),
                "EREM 理由": self._reasons(
                    raw_score=raw_score,
                    current_score=current_score,
                    holding_confidence=holding_confidence,
                    entry_les_available=entry_les_available,
                    relative_strength_available=relative_strength_available,
                    trend_score=trend_score,
                    momentum_score=momentum_score,
                    volatility_score=volatility_score,
                    distribution_score=distribution_score,
                    remaining_return_score=remaining_return_score,
                    relative_strength_score=relative_strength_score,
                    structure_score=structure_score,
                    entry_quality_score=entry_quality_score,
                    turnover_score=turnover_score,
                    vwap_score=vwap_score,
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

    def _turnover_input(self, df: pd.DataFrame, volume: pd.Series) -> tuple[pd.Series, str]:
        for name in ("Turnover", "Turnover Rate", "週轉率"):
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce"), name
        return volume, "Volume/VolumeMA20 proxy"

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
        direction = np.select(
            [close > close.shift(1), close < close.shift(1)],
            [1, -1],
            default=0,
        )
        return pd.Series(direction * volume, index=close.index).cumsum()

    def _decision(self, score: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [score >= 85, score >= 70, score >= 55, score >= 40],
                ["STRONG HOLD", "HOLD", "HOLD", "REDUCE 25%"],
                default="EXIT",
            ),
            index=score.index,
        )

    def _reasons(
        self,
        raw_score: pd.Series,
        current_score: pd.Series,
        holding_confidence: pd.Series,
        entry_les_available: pd.Series,
        relative_strength_available: pd.Series,
        trend_score: pd.Series,
        momentum_score: pd.Series,
        volatility_score: pd.Series,
        distribution_score: pd.Series,
        remaining_return_score: pd.Series,
        relative_strength_score: pd.Series,
        structure_score: pd.Series,
        entry_quality_score: pd.Series,
        turnover_score: pd.Series,
        vwap_score: pd.Series,
        decision: pd.Series,
    ) -> pd.Series:
        score_frame = pd.DataFrame(
            {
                "raw": raw_score,
                "current": current_score,
                "holding": holding_confidence,
                "entry_available": entry_les_available,
                "rs_available": relative_strength_available,
                "trend": trend_score,
                "momentum": momentum_score,
                "volatility": volatility_score,
                "distribution": distribution_score,
                "remaining": remaining_return_score,
                "relative": relative_strength_score,
                "structure": structure_score,
                "entry": entry_quality_score,
                "turnover": turnover_score,
                "vwap": vwap_score,
                "decision": decision,
            }
        )

        def format_reason(row: pd.Series) -> str:
            missing = []
            if not bool(row["rs_available"]):
                missing.append("TAIEX")
            if not bool(row["entry_available"]):
                missing.append("Entry LES")
            missing_note = "；缺少=" + ",".join(missing) if missing else ""
            values = row.to_dict()
            values["missing_note"] = missing_note
            return (
                "原始{raw:.0f}/105→EREM{current:.2f}；趨勢{trend:+.0f}；動能{momentum:+.0f}；"
                "波動{volatility:+.0f}；籌碼{distribution:+.0f}；剩餘報酬{remaining:+.0f}；"
                "相對強弱{relative:+.0f}；結構{structure:+.0f}；進場品質{entry:+.0f}；"
                "週轉{turnover:+.0f}；VWAP{vwap:+.0f}；持有信心{holding:.2f}；決策={decision}{missing_note}"
            ).format(**values)

        return score_frame.apply(format_reason, axis=1)

    def _empty_result(self) -> dict[str, object]:
        result: dict[str, object] = {column: np.nan for column in EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS}
        result.update(
            {
                "EREM 分數": 0,
                "EREM Raw Score": 0,
                "EREM Holding Confidence": 0,
                "EREM 決策訊號": "EXIT",
                "EREM Trend Score": 0,
                "EREM Momentum Score": 0,
                "EREM Volatility Score": 0,
                "EREM Distribution Score": 0,
                "EREM Remaining Return Score": 0,
                "EREM Relative Strength Score": 0,
                "EREM Structure Score": 0,
                "EREM Entry Quality Score": 0,
                "EREM Turnover Score": 0,
                "EREM VWAP Score": 0,
                "EREM MACD Histogram Rising": 0,
                "EREM MACD Histogram Falling 2D": 0,
                "EREM Bearish Divergence": 0,
                "EREM OBV Rising": 0,
                "EREM Distribution Warning": 0,
                "EREM Higher High": 0,
                "EREM Higher Low": 0,
                "EREM Turnover Source": "Volume/VolumeMA20 proxy",
                "EREM 理由": "資料不足；決策=EXIT",
            }
        )
        return result
