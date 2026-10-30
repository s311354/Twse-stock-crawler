"""Expected Return Exit Model v3 strategy plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from domain.numeric_history import metadata_free_frame

from domain.features import FeatureFrame
from domain.features import SharedFeatureConfig
from domain.features import SharedFeaturePipeline
from domain.strategy import Strategy

EXPECTED_RETURN_EXIT_STRATEGY_NAME = "expected_return_exit_v3"
EREM_V3_MODEL_VERSION = "3.2.4"
EREM_V3_MAX_RAW_SCORE = 105.0

EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS = [
    "EREM 分數",
    "EREM Raw Score",
    "EREM Holding Confidence",
    "EREM Entry LES",
    "EREM Entry Decision",
    "EREM Grace Profile",
    "EREM Holding Days",
    "EREM Entry Weight",
    "EREM Exit Streak Count",
    "EREM 決策訊號",
    "EREM 建議減倉比例",
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
    "EREM VWAP State",
    "EREM EMA20",
    "EREM EMA20 Slope",
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
    entry_weight_initial: float = 0.40
    entry_weight_floor: float = 0.10
    entry_weight_decay_days: float = 20.0
    exit_confidence_threshold: float = 40.0
    exit_confirmation_sessions: int = 2
    buy_grace_sessions: int = 3
    buy_exit_confirmation_sessions: int = 3
    early_buy_grace_sessions: int = 2
    buy_small_grace_sessions: int = 2
    buy_small_exit_confirmation_sessions: int = 2
    buy_entry_les_threshold: float = 75.0
    ema_slope_lookback: int = 5
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if not 0 <= self.entry_weight_floor <= self.entry_weight_initial <= 1:
            raise ValueError(
                "entry weights must satisfy 0 <= floor <= initial <= 1"
            )
        if self.entry_weight_decay_days <= 0:
            raise ValueError("entry_weight_decay_days must be positive")
        if self.exit_confirmation_sessions <= 0:
            raise ValueError("exit_confirmation_sessions must be positive")
        for name, value in (
            ("buy_grace_sessions", self.buy_grace_sessions),
            ("early_buy_grace_sessions", self.early_buy_grace_sessions),
            ("buy_small_grace_sessions", self.buy_small_grace_sessions),
        ):
            if value < 0:
                raise ValueError("{} must be non-negative".format(name))
        for name, value in (
            (
                "buy_exit_confirmation_sessions",
                self.buy_exit_confirmation_sessions,
            ),
            (
                "buy_small_exit_confirmation_sessions",
                self.buy_small_exit_confirmation_sessions,
            ),
        ):
            if value <= 0:
                raise ValueError("{} must be positive".format(name))


class ExpectedReturnExitV3Strategy(Strategy):
    """Score exit risk using OHLCV plus optional TAIEX and entry LES history.

    The requested component weights total 105 even though the model specifies a
    0-100 result. Component scores retain their stated weights and the raw score
    is normalized by 100 / 105 before applying the decision thresholds.
    """

    name = EXPECTED_RETURN_EXIT_STRATEGY_NAME
    description = (
        "Holding confidence with decaying Entry LES memory, three-state VWAP, "
        "immediate low-confidence exits, new-position grace, trend, momentum, volatility, "
        "distribution, and relative strength"
    )
    version = EREM_V3_MODEL_VERSION

    def __init__(self, config: ExpectedReturnExitV3Config | None = None, *, soft_reduction_policy=None) -> None:
        self.config = config or ExpectedReturnExitV3Config()
        self.soft_reduction_policy = soft_reduction_policy
        if soft_reduction_policy is not None:
            from domain.strategies.erem_soft_reduction import SoftReductionResearchPolicy, SoftReductionFormalPolicy
            if type(soft_reduction_policy) not in (SoftReductionResearchPolicy, SoftReductionFormalPolicy):
                raise ValueError("EXPLICIT_EREM_RESEARCH_POLICY_REQUIRED")
            self.name = EXPECTED_RETURN_EXIT_STRATEGY_NAME + (
                "_soft_reduction_research_r1" if soft_reduction_policy.research_only else "_b1")
            self.version = EREM_V3_MODEL_VERSION + ":" + soft_reduction_policy.version

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        features = SharedFeaturePipeline(
            self._shared_feature_config()
        ).build(data)
        return self.run_with_features(data, features)

    def run_with_features(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
        *,
        latest_only: bool = False,
    ) -> pd.DataFrame:
        if data.empty:
            result = pd.DataFrame([self._empty_result()])
            if self.soft_reduction_policy is not None:
                result["EREM Native Decision"] = result["EREM 決策訊號"]
                result["EREM Research Policy" if self.soft_reduction_policy.research_only
                       else "EREM Exit Policy Version"] = self.soft_reduction_policy.version
                result["EREM Reduction Status"] = "UNKNOWN_EMPTY_INPUT"
                result["EREM Reduction Cooldown"] = 0
            return result

        features.assert_compatible(data)
        if features.config != self._shared_feature_config():
            raise ValueError(
                "FeatureFrame config does not match EREM strategy"
            )
        df = metadata_free_frame(data)
        feature_data = features.to_pandas()
        high = feature_data["high"]
        low = feature_data["low"]
        close = feature_data["close"]
        volume = feature_data["volume_filled"]

        ema20 = feature_data["ema_short"]
        ema50 = feature_data["ema_medium"]
        ema200 = feature_data["ema_long"]
        ema_alignment = (ema20 > ema50) & (ema50 > ema200)
        price_above_ema20 = close > ema20
        # The v2 15:10 trend ratio is scaled to the v3 component maximum of 20.
        trend_score = pd.Series(
            np.where(ema_alignment, 12, 0) + np.where(price_above_ema20, 8, 0),
            index=df.index,
            dtype="float64",
        )

        rsi = feature_data["rsi14_sma"]
        macd = feature_data["macd"]
        macd_signal = feature_data["macd_signal"]
        macd_histogram = feature_data["macd_histogram"]
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

        upper_band = feature_data["bollinger_upper"]
        atr14 = feature_data["atr14_sma"]
        atr20ma = feature_data["atr14_sma_ma20"]
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

        volume_ma20 = feature_data["volume_ma20"]
        obv = feature_data["obv"]
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

        taiex_close = feature_data["taiex"]
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
        holding_days = self._numeric_column(
            df,
            ("Holding Days", "Holding_Days", "持有天數"),
        ).clip(lower=0)
        holding_days_available = holding_days.notna()
        entry_decision = (
            self._text_column(
                df,
                (
                    "Entry Decision",
                    "Entry_Decision",
                    "進場決策",
                    "進場訊號",
                ),
            )
            .ffill()
            .str.strip()
            .str.upper()
            .str.replace("-", "_", regex=False)
            .str.replace(" ", "_", regex=False)
        )
        entry_decision = entry_decision.where(
            entry_decision.isin({"BUY", "EARLY_BUY", "BUY_SMALL"})
        )
        entry_decision_available = entry_decision.notna()
        decayed_entry_weight = self.config.entry_weight_initial * np.exp(
            -holding_days / self.config.entry_weight_decay_days
        )
        entry_weight = pd.Series(
            np.where(
                entry_les_available,
                np.where(
                    holding_days_available,
                    np.maximum(
                        self.config.entry_weight_floor,
                        decayed_entry_weight,
                    ),
                    self.config.entry_weight_floor,
                ),
                0.0,
            ),
            index=df.index,
            dtype="float64",
        ).clip(
            lower=0,
            upper=self.config.entry_weight_initial,
        )
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

        vwap60 = feature_data["vwap60_close"]
        vwap_score, vwap_state = self._vwap_assessment(close, vwap60)

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
        entry_anchor = entry_les.where(entry_les_available, current_score)
        holding_confidence = (
            (1 - entry_weight) * current_score
            + entry_weight * entry_anchor
        ).clip(lower=0, upper=100)
        exit_streak, exit_confirmed, entry_grace, grace_profile = (
            self._exit_confirmation(
                holding_confidence=holding_confidence,
                entry_les=entry_les,
                holding_days=holding_days,
                entry_decision=entry_decision,
            )
        )
        ema20_slope = (
            (ema20 - ema20.shift(self.config.ema_slope_lookback))
            / (self.config.ema_slope_lookback * atr14)
        ).where(atr14.abs() > self.config.epsilon)
        proposed_reduction_percentage = self._recommended_reduction_percentage(
            holding_confidence
        )
        decision = self._decision(
            holding_confidence,
            proposed_reduction_percentage,
            exit_confirmed=exit_confirmed,
            entry_grace=entry_grace,
        )
        soft_reduction_audit = None
        native_decision = decision
        if self.soft_reduction_policy is not None:
            from domain.strategies.erem_soft_reduction import apply_soft_reduction_policy
            soft_reduction_audit = apply_soft_reduction_policy(decision, entry_les=entry_les,
                holding_days=holding_days, entry_decision=entry_decision, entry_grace=entry_grace,
                policy=self.soft_reduction_policy)
            decision = soft_reduction_audit["decision"]
        reduction_percentage = proposed_reduction_percentage.where(
            decision.eq("EXIT") | decision.str.startswith("REDUCE"),
            0.0,
        )

        result = pd.DataFrame(
            {
                "EREM 分數": current_score.round(2),
                "EREM Raw Score": raw_score.round(2),
                "EREM Holding Confidence": holding_confidence.round(2),
                "EREM Entry LES": entry_les.round(2),
                "EREM Entry Decision": entry_decision,
                "EREM Grace Profile": grace_profile,
                "EREM Holding Days": holding_days.round(0),
                "EREM Entry Weight": entry_weight.round(4),
                "EREM Exit Streak Count": exit_streak,
                "EREM 決策訊號": decision,
                "EREM 建議減倉比例": reduction_percentage,
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
                "EREM VWAP State": vwap_state,
                "EREM EMA20": ema20.round(2),
                "EREM EMA20 Slope": ema20_slope.round(4),
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
                    exit_streak=exit_streak,
                    exit_confirmed=exit_confirmed,
                    entry_grace=entry_grace,
                    grace_profile=grace_profile,
                    ema20_slope=ema20_slope,
                    entry_les_available=entry_les_available,
                    entry_decision_available=entry_decision_available,
                    holding_days=holding_days,
                    holding_days_available=holding_days_available,
                    entry_weight=entry_weight,
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
                    vwap_state=vwap_state,
                    decision=decision,
                    latest_only=latest_only,
                ),
            },
            index=df.tail(1).index if latest_only else df.index,
        )

        if soft_reduction_audit is not None:
            result["EREM Native Decision"] = native_decision.reindex(result.index)
            result["EREM Research Policy" if self.soft_reduction_policy.research_only
                   else "EREM Exit Policy Version"] = self.soft_reduction_policy.version
            result["EREM Reduction Status"] = soft_reduction_audit["status"].reindex(result.index)
            result["EREM Reduction Cooldown"] = soft_reduction_audit["cooldown_remaining"].reindex(result.index)
        return pd.concat([df.tail(1) if latest_only else df, result], axis=1)

    def run_latest_with_features(self, data: pd.DataFrame, features: FeatureFrame) -> pd.DataFrame:
        """Keep full warm-up/exit sequence; materialize only the daily result."""
        return self.run_with_features(data, features, latest_only=True)

    def _shared_feature_config(self) -> SharedFeatureConfig:
        return SharedFeatureConfig(
            ema_short_window=self.config.ema_short_window,
            ema_medium_window=self.config.ema_medium_window,
            ema_long_window=self.config.ema_long_window,
            rsi_window=self.config.rsi_window,
            atr_window=self.config.atr_window,
            atr_ma_window=self.config.atr_ma_window,
            volume_window=self.config.volume_window,
            bollinger_window=self.config.bollinger_window,
            vwap_long_window=self.config.vwap_window,
            epsilon=self.config.epsilon,
        )

    def _numeric_column(self, data: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in data.columns:
                return pd.to_numeric(data[name], errors="coerce")
        return pd.Series(np.nan, index=data.index, dtype="float64")

    def _text_column(self, data: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
        for name in names:
            if name in data.columns:
                return data[name].astype("string")
        return pd.Series(pd.NA, index=data.index, dtype="string")

    def _turnover_input(self, df: pd.DataFrame, volume: pd.Series) -> tuple[pd.Series, str]:
        for name in ("Turnover", "Turnover Rate", "週轉率"):
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce"), name
        return volume, "Volume/VolumeMA20 proxy"

    @staticmethod
    def _consecutive_true(condition: pd.Series) -> pd.Series:
        """Count consecutive true sessions without carrying across a reset."""
        mask = condition.fillna(False).astype(bool)
        groups = (~mask).cumsum()
        return mask.groupby(groups).cumsum().astype("int64")

    def _exit_confirmation(
        self,
        holding_confidence: pd.Series,
        entry_les: pd.Series,
        holding_days: pd.Series,
        entry_decision: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Track exit warnings and entry grace without pre-entry carryover.

        Hard stops and confirmed fundamental risks remain outside EREM and may
        still exit immediately. The current native <40 EXIT is immediate too;
        exit_confirmed is diagnostic, not an execution delay. Entry grace only
        suppresses an ordinary REDUCE. The optional, separately versioned
        research policy confirms reductions, not these EXIT warnings.
        """
        exit_warning = (
            holding_confidence < self.config.exit_confidence_threshold
        )
        position_context = entry_les.notna() & holding_days.notna()

        # When position metadata exists, pre-entry rows must reset the streak.
        # Diagnostic runs without position metadata retain ordinary consecutive
        # scoring, but still require more than one warning before EXIT.
        streak_input = exit_warning
        if bool(position_context.any()):
            streak_input = exit_warning & position_context
        exit_streak = self._consecutive_true(streak_input)

        eligible_context = position_context & entry_decision.isin(
            {"BUY", "EARLY_BUY", "BUY_SMALL"}
        )
        holding_session_index = position_context.astype("int64").cumsum().sub(1)
        buy_window = (
            eligible_context
            & entry_decision.eq("BUY")
            & entry_les.ge(self.config.buy_entry_les_threshold)
            & holding_session_index.lt(self.config.buy_grace_sessions)
        )
        buy_small_window = (
            eligible_context
            & entry_decision.eq("BUY_SMALL")
            & holding_session_index.lt(self.config.buy_small_grace_sessions)
        )
        early_buy_window = (
            eligible_context
            & entry_decision.eq("EARLY_BUY")
            & holding_session_index.lt(self.config.early_buy_grace_sessions)
        )
        grace_profile = pd.Series(
            "NONE",
            index=holding_confidence.index,
            dtype="object",
        )
        grace_profile.loc[buy_window] = "BUY_3_SESSION"
        grace_profile.loc[early_buy_window] = "EARLY_BUY_2_SESSION"
        grace_profile.loc[buy_small_window] = "BUY_SMALL_2_SESSION"
        entry_grace = buy_window | early_buy_window | buy_small_window
        required_sessions = pd.Series(
            float(self.config.exit_confirmation_sessions),
            index=holding_confidence.index,
            dtype="float64",
        )
        required_sessions.loc[buy_window] = float(
            self.config.buy_exit_confirmation_sessions
        )
        required_sessions.loc[buy_small_window] = float(
            self.config.buy_small_exit_confirmation_sessions
        )
        exit_confirmed = exit_warning & exit_streak.ge(required_sessions)
        return exit_streak, exit_confirmed, entry_grace, grace_profile

    @staticmethod
    def _vwap_assessment(
        close: pd.Series,
        vwap60: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """Score VWAP support, recovery, or continued weakness."""
        available = close.notna() & vwap60.notna()
        above_or_equal = available & (close >= vwap60)
        below = available & (close < vwap60)
        gap = vwap60 - close
        recovering = (
            below
            & (close > close.shift(1))
            & (gap < gap.shift(1))
        )
        synchronous_decline = (
            below
            & (close < close.shift(1))
            & (vwap60 < vwap60.shift(1))
        )
        score = pd.Series(
            np.select(
                [above_or_equal, recovering],
                [5.0, 2.0],
                default=0.0,
            ),
            index=close.index,
            dtype="float64",
        )
        state = pd.Series(
            np.select(
                [
                    ~available,
                    above_or_equal,
                    recovering,
                    synchronous_decline,
                ],
                [
                    "UNAVAILABLE",
                    "ABOVE_OR_EQUAL",
                    "RECOVERING",
                    "BELOW_SYNCHRONOUS_DECLINE",
                ],
                default="BELOW_NOT_RECOVERING",
            ),
            index=close.index,
            dtype="object",
        )
        return score, state

    @staticmethod
    def _recommended_reduction_percentage(score: pd.Series) -> pd.Series:
        """Return an actionable reduction percentage from Holding Confidence.

        The 40-54 band scales linearly from 50% down to 25%, quantized to
        practical 5% steps. Scores below 40 remain full EXIT signals, while
        HOLD states recommend no reduction.
        """
        continuous = 25.0 + ((54.0 - score) / 14.0 * 25.0)
        stepped = (np.floor((continuous + 2.5) / 5.0) * 5.0).clip(25, 50)
        reduction = pd.Series(
            np.select(
                [score < 40, (score >= 40) & (score < 55)],
                [100.0, stepped],
                default=0.0,
            ),
            index=score.index,
            dtype="float64",
        )
        return reduction.where(score.notna())

    def _decision(
        self,
        score: pd.Series,
        reduction_percentage: pd.Series | None = None,
        *,
        exit_confirmed: pd.Series | None = None,
        entry_grace: pd.Series | None = None,
    ) -> pd.Series:
        reduction = (
            reduction_percentage
            if reduction_percentage is not None
            else self._recommended_reduction_percentage(score)
        )
        grace = (
            entry_grace.fillna(False).astype(bool)
            if entry_grace is not None
            else pd.Series(False, index=score.index, dtype="bool")
        )
        decision = pd.Series("HOLD_REVIEW", index=score.index, dtype="object")
        decision.loc[score >= 55] = "HOLD"
        decision.loc[score >= 85] = "STRONG HOLD"
        reduce_mask = (score >= 40) & (score < 55)
        decision.loc[reduce_mask] = (
            "REDUCE "
            + reduction.loc[reduce_mask].astype(int).astype(str)
            + "%"
        )
        decision.loc[score < self.config.exit_confidence_threshold] = "EXIT"
        # Entry grace only suppresses an ordinary 40-54 REDUCE. It must never
        # delay EXIT (<40), hard stops, or fundamental hard-risk exits.
        decision.loc[grace & reduce_mask] = "HOLD_REVIEW"
        return decision

    def _reasons(
        self,
        raw_score: pd.Series,
        current_score: pd.Series,
        holding_confidence: pd.Series,
        exit_streak: pd.Series,
        exit_confirmed: pd.Series,
        entry_grace: pd.Series,
        grace_profile: pd.Series,
        ema20_slope: pd.Series,
        entry_les_available: pd.Series,
        entry_decision_available: pd.Series,
        holding_days: pd.Series,
        holding_days_available: pd.Series,
        entry_weight: pd.Series,
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
        vwap_state: pd.Series,
        decision: pd.Series,
        latest_only: bool = False,
    ) -> pd.Series:
        score_frame = pd.DataFrame(
            {
                "raw": raw_score,
                "current": current_score,
                "holding": holding_confidence,
                "exit_streak": exit_streak,
                "exit_confirmed": exit_confirmed,
                "entry_grace": entry_grace,
                "grace_profile": grace_profile,
                "ema20_slope": ema20_slope,
                "entry_available": entry_les_available,
                "entry_decision_available": entry_decision_available,
                "holding_days": holding_days,
                "holding_days_available": holding_days_available,
                "entry_weight": entry_weight,
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
                "vwap_state": vwap_state,
                "decision": decision,
            }
        )

        def format_reason(row: pd.Series) -> str:
            missing = []
            if not bool(row["rs_available"]):
                missing.append("TAIEX")
            if not bool(row["entry_available"]):
                missing.append("Entry LES")
            if not bool(row["entry_decision_available"]):
                missing.append("Entry Decision")
            if bool(row["entry_available"]) and not bool(
                row["holding_days_available"]
            ):
                missing.append("Holding Days")
            missing_note = "；缺少=" + ",".join(missing) if missing else ""
            values = row.to_dict()
            values["missing_note"] = missing_note
            values["holding_days_text"] = (
                "{:.0f}".format(row["holding_days"])
                if bool(row["holding_days_available"])
                else "N/A"
            )
            return (
                "原始{raw:.0f}/105→EREM{current:.2f}；趨勢{trend:+.0f}；動能{momentum:+.0f}；"
                "波動{volatility:+.0f}；籌碼{distribution:+.0f}；剩餘報酬{remaining:+.0f}；"
                "相對強弱{relative:+.0f}；結構{structure:+.0f}；進場品質{entry:+.0f}；"
                "週轉{turnover:+.0f}；VWAP{vwap:+.0f}({vwap_state})；"
                "HoldingDays={holding_days_text}；EntryWeight={entry_weight:.4f}；"
                "持有信心{holding:.2f}；"
                "ExitStreak={exit_streak:.0f}；ExitConfirmed={exit_confirmed}；"
                "GraceProfile={grace_profile}；"
                "EMA20Slope={ema20_slope:.4f}；"
                "決策={decision}{missing_note}"
            ).format(**values)

        if latest_only:
            score_frame = score_frame.tail(1)
        return score_frame.apply(format_reason, axis=1)

    def _empty_result(self) -> dict[str, object]:
        result: dict[str, object] = {column: np.nan for column in EXPECTED_RETURN_EXIT_OUTPUT_COLUMNS}
        result.update(
            {
                "EREM 分數": 0,
                "EREM Raw Score": 0,
                "EREM Holding Confidence": 0,
                "EREM Entry Decision": np.nan,
                "EREM Grace Profile": "NONE",
                "EREM Holding Days": np.nan,
                "EREM Entry Weight": 0,
                "EREM Exit Streak Count": 0,
                "EREM 決策訊號": "EXIT",
                "EREM 建議減倉比例": np.nan,
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
                "EREM VWAP State": "UNAVAILABLE",
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
