"""Shared, immutable technical features for strategy plugins."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
import json

import numpy as np
import pandas as pd
from domain.numeric_history import metadata_free_frame


FEATURE_PIPELINE_VERSION = "1.0.0"


def feature_cache_identity(data, config):
    """Account-free key including numerical units, index and exact date window."""
    frame = metadata_free_frame(data)
    payload = dict(pipeline=FEATURE_PIPELINE_VERSION, config=asdict(config),
                   source=_source_fingerprint(frame), index_dtype=str(frame.index.dtype),
                   dates=[str(value) for value in frame.get("Date", ())])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SharedFeatureConfig:
    """Versioned windows used by the production LES and EREM strategies."""

    ema_short_window: int = 20
    ema_medium_window: int = 50
    ema_long_window: int = 200
    rsi_window: int = 14
    atr_window: int = 14
    atr_ma_window: int = 20
    volume_window: int = 20
    bollinger_window: int = 20
    vwap_short_window: int = 20
    vwap_long_window: int = 60
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        for name, value in (
            ("ema_short_window", self.ema_short_window),
            ("ema_medium_window", self.ema_medium_window),
            ("ema_long_window", self.ema_long_window),
            ("rsi_window", self.rsi_window),
            ("atr_window", self.atr_window),
            ("atr_ma_window", self.atr_ma_window),
            ("volume_window", self.volume_window),
            ("bollinger_window", self.bollinger_window),
            ("vwap_short_window", self.vwap_short_window),
            ("vwap_long_window", self.vwap_long_window),
        ):
            if value < 1:
                raise ValueError("{} must be at least one".format(name))
        if self.epsilon <= 0:
            raise ValueError("epsilon must be greater than zero")


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """Immutable feature values with copy-on-read pandas interoperability."""

    config: SharedFeatureConfig
    pipeline_version: str
    source_fingerprint: str
    _index: tuple[object, ...]
    _columns: tuple[str, ...]
    _values: tuple[tuple[object, ...], ...]

    @classmethod
    def from_pandas(
        cls,
        frame: pd.DataFrame,
        *,
        config: SharedFeatureConfig,
        source_data: pd.DataFrame,
    ) -> "FeatureFrame":
        columns = tuple(str(column) for column in frame.columns)
        values = tuple(
            tuple(frame[column].tolist())
            for column in frame.columns
        )
        return cls(
            config=config,
            pipeline_version=FEATURE_PIPELINE_VERSION,
            source_fingerprint=_source_fingerprint(source_data),
            _index=tuple(frame.index.tolist()),
            _columns=columns,
            _values=values,
        )

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    def __len__(self) -> int:
        return len(self._index)

    def assert_compatible(self, data: pd.DataFrame) -> None:
        if len(data) != len(self) or tuple(data.index.tolist()) != self._index:
            raise ValueError(
                "FeatureFrame index does not match strategy input"
            )
        if _source_fingerprint(data) != self.source_fingerprint:
            raise ValueError(
                "FeatureFrame source data does not match strategy input"
            )

    def series(self, name: str) -> pd.Series:
        try:
            position = self._columns.index(name)
        except ValueError as error:
            raise KeyError("Unknown shared feature '{}'".format(name)) from error
        return pd.Series(
            list(self._values[position]),
            index=pd.Index(self._index),
            name=name,
        )

    def to_pandas(self) -> pd.DataFrame:
        """Return a mutable copy without exposing the stored feature values."""

        return pd.DataFrame(
            {
                column: list(values)
                for column, values in zip(self._columns, self._values)
            },
            index=pd.Index(self._index),
        )


class SharedFeaturePipeline:
    """Calculate common LES/EREM rolling indicators once per stock history."""

    def __init__(self, config: SharedFeatureConfig | None = None) -> None:
        self.config = config or SharedFeatureConfig()

    def build(self, data: pd.DataFrame) -> FeatureFrame:
        frame = self._calculate(data)
        return FeatureFrame.from_pandas(
            frame,
            config=self.config,
            source_data=data,
        )

    def build_price_pair(self, data: pd.DataFrame, divisor: float = 1000.0) -> tuple[FeatureFrame, FeatureFrame]:
        """Batch native/scaled prices without rescaling computed indicators.

        Each price column retains its original arithmetic and epsilon policy.
        Volume, P/E and benchmark rolling calculations are shared. Returned
        frames have separate source fingerprints and copy-on-read values.
        """
        if not np.isfinite(divisor) or divisor <= 0:
            raise ValueError("Price divisor must be positive and finite")
        data = metadata_free_frame(data)
        normalized = metadata_free_frame(data)
        for name in ("Open", "High", "Low", "Close", "Bid", "Ask"):
            if name not in data:
                raise ValueError("Price pair requires canonical price columns")
            normalized[name] = pd.to_numeric(data[name], errors="coerce") / divisor
        frames = self._calculate_units(data, (1.0, divisor))
        return tuple(FeatureFrame.from_pandas(frame, config=self.config, source_data=source)
                     for frame, source in zip(frames, (data, normalized), strict=True))

    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._calculate_units(data, (1.0,))[0]

    def _calculate_units(self, data: pd.DataFrame, divisors: tuple[float, ...]) -> tuple[pd.DataFrame, ...]:
        data = metadata_free_frame(data)
        config = self.config
        index = data.index
        def prices(names):
            series = _numeric_column(data, names)
            if len(divisors) == 1:
                return series
            return pd.DataFrame({i: series if divisor == 1 else series / divisor
                                 for i, divisor in enumerate(divisors)}, index=index)
        open_price = prices(("Open", "開盤"))
        high = prices(("High", "最高"))
        low = prices(("Low", "最低"))
        close = prices(("Close", "收盤"))
        volume = _numeric_column(data, ("Volume", "成交量")).where(
            lambda values: values >= 0
        )
        volume_filled = volume.fillna(0)
        amount = _numeric_column(data, ("Amount", "成交金額")).where(
            lambda values: values > 0
        )
        trades = _numeric_column(data, ("Trades", "成交筆數")).where(
            lambda values: values > 0
        )
        bid = prices(("Bid", "最後揭示買價")).where(
            lambda values: values > 0
        )
        ask = prices(("Ask", "最後揭示賣價")).where(
            lambda values: values > 0
        )
        pe = _numeric_column(data, ("PE", "本益比"))
        taiex = _numeric_column(
            data,
            ("TAIEX Close", "TAIEX收盤", "大盤收盤", "Benchmark Close", "TAIEX"),
        )

        ema_windows = {
            5,
            10,
            12,
            20,
            26,
            50,
            200,
            config.ema_short_window,
            config.ema_medium_window,
            config.ema_long_window,
        }
        ema = {
            window: close.ewm(
                span=window,
                adjust=False,
                min_periods=1,
            ).mean()
            for window in sorted(ema_windows)
        }
        taiex_ema20 = taiex.ewm(
            span=20,
            adjust=False,
            min_periods=1,
        ).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        rsi14_sma = _rsi_from_averages(
            gain.rolling(config.rsi_window, min_periods=1).mean(),
            loss.rolling(config.rsi_window, min_periods=1).mean(),
        )
        rsi14_wilder = _rsi_from_averages(
            gain.ewm(
                alpha=1 / config.rsi_window,
                adjust=False,
                min_periods=config.rsi_window,
            ).mean(),
            loss.ewm(
                alpha=1 / config.rsi_window,
                adjust=False,
                min_periods=config.rsi_window,
            ).mean(),
            fill_initial=False,
        )

        macd = ema[12] - ema[26]
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
        macd_strict = ema[12].where(_ready(index, 12)) - ema[26].where(
            _ready(index, 26)
        )
        macd_signal_strict = macd_strict.ewm(
            span=9,
            adjust=False,
            min_periods=9,
        ).mean()

        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1) if len(divisors) == 1 else pd.DataFrame({
            i: pd.concat([high[i] - low[i], (high[i] - previous_close[i]).abs(),
                          (low[i] - previous_close[i]).abs()], axis=1).max(axis=1)
            for i in range(len(divisors))
        }, index=index)
        atr14_sma = true_range.rolling(
            config.atr_window,
            min_periods=1,
        ).mean()
        atr14_wilder = true_range.ewm(
            alpha=1 / config.atr_window,
            adjust=False,
            min_periods=config.atr_window,
        ).mean()
        atr14_sma_ma20 = atr14_sma.rolling(
            config.atr_ma_window,
            min_periods=1,
        ).mean()
        atr_percent_wilder = _safe_divide(
            atr14_wilder,
            close,
            epsilon=config.epsilon,
        )
        atr_percent_wilder_ma20 = atr_percent_wilder.rolling(
            config.atr_ma_window,
            min_periods=config.atr_ma_window,
        ).mean()

        middle_band = close.rolling(
            config.bollinger_window,
            min_periods=1,
        ).mean()
        std20 = close.rolling(
            config.bollinger_window,
            min_periods=2,
        ).std(ddof=0)

        volume_ma20 = volume.rolling(
            config.volume_window,
            min_periods=1,
        ).mean()
        volume_ma20_strict = volume_ma20.where(
            _ready(index, config.volume_window)
        )
        volume_sum20_strict = volume.rolling(
            config.volume_window,
            min_periods=config.volume_window,
        ).sum()
        direction = np.select(
            [close > previous_close, close < previous_close],
            [1, -1],
            default=0,
        )
        obv = pd.Series(
            direction * volume_filled,
            index=index,
            dtype="float64",
        ).cumsum() if len(divisors) == 1 else pd.DataFrame(direction, index=index).mul(volume_filled, axis=0).cumsum()

        typical_price = (high + low + close) / 3
        vwap20_typical = _safe_divide(
            _multiply(typical_price, volume).rolling(
                config.vwap_short_window,
                min_periods=config.vwap_short_window,
            ).sum(),
            volume.rolling(
                config.vwap_short_window,
                min_periods=config.vwap_short_window,
            ).sum(),
            epsilon=config.epsilon,
        )
        vwap60_close = _safe_divide(
            _multiply(close, volume_filled).rolling(
                config.vwap_long_window,
                min_periods=1,
            ).sum(),
            volume_filled.rolling(
                config.vwap_long_window,
                min_periods=1,
            ).sum(),
            epsilon=config.epsilon,
        )

        values = {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "volume_filled": volume_filled,
                "amount": amount,
                "trades": trades,
                "bid": bid,
                "ask": ask,
                "pe": pe,
                "taiex": taiex,
                "ema5": ema[5],
                "ema10": ema[10],
                "ema20": ema[20],
                "ema50": ema[50],
                "ema200": ema[200],
                "ema_short": ema[config.ema_short_window],
                "ema_medium": ema[config.ema_medium_window],
                "ema_long": ema[config.ema_long_window],
                "ema5_strict": ema[5].where(_ready(index, 5)),
                "ema10_strict": ema[10].where(_ready(index, 10)),
                "ema20_strict": ema[20].where(_ready(index, 20)),
                "ema50_strict": ema[50].where(_ready(index, 50)),
                "taiex_ema20_strict": taiex_ema20.where(_ready(index, 20)),
                "r1": close.pct_change(1, fill_method=None),
                "r5": close.pct_change(5, fill_method=None),
                "r20": close.pct_change(20, fill_method=None),
                "taiex_r5": taiex.pct_change(5, fill_method=None),
                "taiex_r20": taiex.pct_change(20, fill_method=None),
                "rsi14_sma": rsi14_sma,
                "rsi14_wilder": rsi14_wilder,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_histogram": macd - macd_signal,
                "macd_strict": macd_strict,
                "macd_signal_strict": macd_signal_strict,
                "macd_histogram_strict": (
                    macd_strict - macd_signal_strict
                ),
                "true_range": true_range,
                "atr14_sma": atr14_sma,
                "atr14_wilder": atr14_wilder,
                "atr14_sma_ma20": atr14_sma_ma20,
                "atr_percent_wilder": atr_percent_wilder,
                "atr_percent_wilder_ma20": atr_percent_wilder_ma20,
                "bollinger_middle": middle_band,
                "bollinger_std20": std20,
                "bollinger_upper": middle_band + (2 * std20),
                "bollinger_lower": middle_band - (2 * std20),
                "volume_ma20": volume_ma20,
                "volume_ma20_strict": volume_ma20_strict,
                "volume_sum20_strict": volume_sum20_strict,
                "obv": obv,
                "vwap20_typical": vwap20_typical,
                "vwap60_close": vwap60_close,
            }
        return tuple(pd.DataFrame({name: value[i] if isinstance(value, pd.DataFrame) else value
                                   for name, value in values.items()}, index=index)
                     for i in range(len(divisors)))


def _numeric_column(
    data: pd.DataFrame,
    names: tuple[str, ...],
) -> pd.Series:
    for name in names:
        if name in data.columns:
            return pd.to_numeric(data[name], errors="coerce")
    return pd.Series(np.nan, index=data.index, dtype="float64")


def _source_fingerprint(data: pd.DataFrame) -> str:
    data = metadata_free_frame(data)
    source = pd.DataFrame(
        {
            "open": _numeric_column(data, ("Open", "開盤")),
            "high": _numeric_column(data, ("High", "最高")),
            "low": _numeric_column(data, ("Low", "最低")),
            "close": _numeric_column(data, ("Close", "收盤")),
            "volume": _numeric_column(data, ("Volume", "成交量")),
            "amount": _numeric_column(data, ("Amount", "成交金額")),
            "trades": _numeric_column(data, ("Trades", "成交筆數")),
            "bid": _numeric_column(data, ("Bid", "最後揭示買價")),
            "ask": _numeric_column(data, ("Ask", "最後揭示賣價")),
            "pe": _numeric_column(data, ("PE", "本益比")),
            "taiex": _numeric_column(
                data,
                (
                    "TAIEX Close",
                    "TAIEX收盤",
                    "大盤收盤",
                    "Benchmark Close",
                    "TAIEX",
                ),
            ),
        },
        index=data.index,
    )
    hashed = pd.util.hash_pandas_object(source, index=True)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def _rsi_from_averages(
    average_gain: pd.Series,
    average_loss: pd.Series,
    *,
    fill_initial: bool = True,
) -> pd.Series:
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(
        average_loss != 0,
        np.where(average_gain > 0, 100, 50),
    )
    return rsi.fillna(50) if fill_initial else rsi


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    epsilon: float,
) -> pd.Series:
    quotient = (numerator.div(denominator, axis=0)
                if isinstance(numerator, pd.DataFrame) and isinstance(denominator, pd.Series)
                else numerator / denominator)
    return quotient.where(denominator.abs() > epsilon)


def _multiply(values, weights):
    return values.mul(weights, axis=0) if isinstance(values, pd.DataFrame) else values * weights


def _ready(index: pd.Index, window: int) -> pd.Series:
    return pd.Series(
        np.arange(len(index)) >= window - 1,
        index=index,
    )


__all__ = [
    "FEATURE_PIPELINE_VERSION",
    "FeatureFrame",
    "SharedFeatureConfig",
    "SharedFeaturePipeline",
]
