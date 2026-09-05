"""Technical scoring component for Low Entry Model v5."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from domain.quality_v5 import FinalQualityStatus
from domain.quality_v5 import LowEntryV5QualityResult


class LowEntryV5ScoringConfig(Protocol):
    maximum_stop_atr: float
    minimum_reward_risk: float
    minimum_amount_ma20: float
    maximum_spread: float
    confirmed_buy_score: float
    provisional_buy_score: float
    risk_off_score_increment: float


class LowEntryV5ScoringEvaluator:
    """Calculate additive technical scores and market-risk penalties."""

    def __init__(self, config: LowEntryV5ScoringConfig) -> None:
        self.config = config

    def score_components(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        valuation: pd.Series,
    ) -> pd.DataFrame:
        market_volatility_normal = (
            indicators["taiex_volatility20"].notna()
            & indicators["taiex_volatility_ma20"].notna()
            & (
                indicators["taiex_volatility20"]
                <= 1.5 * indicators["taiex_volatility_ma20"]
            )
        )
        confirmed_breakdown = (
            (indicators["close"] < indicators["previous_low20"])
            & (indicators["clv"] < 0)
        )
        no_new_low3 = indicators["new_low_count3"].eq(0)
        ema5_turn = (
            (indicators["ema5_slope"] > 0)
            & (indicators["ema5_slope"].shift(1) <= 0)
        )
        ema5_follow_through = (
            (indicators["ema5_slope"] > 0) & ~ema5_turn
        ).fillna(False)
        macd_histogram_rising = (
            indicators["macd_histogram"]
            > indicators["macd_histogram"].shift(1)
        ).fillna(False)
        macd_histogram_improving = (
            (
                indicators["macd_histogram"]
                > indicators["macd_histogram"].shift(1)
            )
            & (
                indicators["macd_histogram"].shift(1)
                > indicators["macd_histogram"].shift(2)
            )
        )
        macd_cross = (
            (indicators["macd"] > indicators["macd_signal"])
            & (
                indicators["macd"].shift(1)
                <= indicators["macd_signal"].shift(1)
            )
        )
        rsi_recovery_cross = (
            (indicators["rsi14"] >= 40)
            & (indicators["rsi14"].shift(1) < 40)
        ).fillna(False)
        rsi_recovery_follow_through = (
            indicators["rsi14"].between(40, 55, inclusive="both")
            & (indicators["rsi14"] > indicators["rsi14"].shift(1))
            & ~rsi_recovery_cross
        ).fillna(False)

        market_regime = (
            4 * (indicators["taiex"] > indicators["taiex_ema20"]).astype(int)
            + 3 * (indicators["taiex_ema20_slope"] >= 0).astype(int)
            + 4 * (indicators["rs20"] > 0).astype(int)
            + 4 * market_volatility_normal.astype(int)
        )
        price_position_points = pd.Series(
            np.select(
                [
                    indicators["price_position60"].between(
                        0.05, 0.25, inclusive="both"
                    ),
                    indicators["price_position60"].between(
                        0.00, 0.35, inclusive="both"
                    ),
                ],
                [5, 3],
                default=0,
            ),
            index=indicators.index,
        )
        drawdown_points = pd.Series(
            np.select(
                [
                    indicators["drawdown60"].between(
                        -0.30, -0.10, inclusive="both"
                    ),
                    indicators["drawdown60"].between(
                        -0.40, -0.30, inclusive="left"
                    ),
                ],
                [4, 2],
                default=0,
            ),
            index=indicators.index,
        )
        price_position = (
            price_position_points
            + drawdown_points
            + 3 * (~confirmed_breakdown.fillna(True)).astype(int)
            + 3 * (valuation == "PASS").astype(int)
            + 2 * (valuation == "SUPPORTIVE").astype(int)
        )
        ema5_structure_points = pd.Series(
            np.select(
                [ema5_turn, ema5_follow_through],
                [4, 2],
                default=0,
            ),
            index=indicators.index,
        )
        stabilization = (
            6 * no_new_low3.astype(int)
            + 7 * indicators["higher_low"].astype(int)
            + 4 * (indicators["close"] > indicators["ema5"]).astype(int)
            + ema5_structure_points
            + 5 * (indicators["close"] > indicators["previous_high"]).astype(int)
            + 4
            * (
                indicators["atr14"].notna()
                & (indicators["atr14"] <= indicators["atr14"].shift(1))
            ).astype(int)
        )
        clv_points = pd.Series(
            np.select(
                [indicators["clv"] > 0.4, indicators["clv"] >= 0.1],
                [5, 2],
                default=0,
            ),
            index=indicators.index,
        )
        demand = (
            clv_points
            + 5 * indicators["cmf20_improving"].astype(int)
            + 4 * (indicators["normalized_obv_slope5"] > 0).astype(int)
            + 3
            * (
                (indicators["close"] > indicators["close"].shift(1))
                & indicators["volume_ratio"].between(
                    0.9,
                    1.8,
                    inclusive="both",
                )
                & (indicators["clv"] >= 0)
            ).astype(int)
            + 3 * (indicators["close"] > indicators["vwap20"]).astype(int)
        )
        rsi_points = pd.Series(
            np.select(
                [rsi_recovery_cross, rsi_recovery_follow_through],
                [4, 2],
                default=0,
            ),
            index=indicators.index,
        )
        macd_histogram_points = pd.Series(
            np.select(
                [macd_histogram_improving, macd_histogram_rising],
                [3, 1],
                default=0,
            ),
            index=indicators.index,
        )
        momentum = (
            rsi_points
            + macd_histogram_points
            + 3 * macd_cross.astype(int)
        )
        execution = (
            3
            * (
                (plan["risk"] > 0)
                & (
                    plan["risk"]
                    <= self.config.maximum_stop_atr * indicators["atr14"]
                )
            ).astype(int)
            + 3
            * (
                plan["reward_risk"] >= self.config.minimum_reward_risk
            ).astype(int)
            + 2
            * (
                indicators["amount_ma20"]
                >= self.config.minimum_amount_ma20
            ).astype(int)
            + 1 * (indicators["spread"] <= self.config.maximum_spread).astype(int)
            + 1 * (indicators["normalized_gap"] <= 0.5).astype(int)
        )
        return pd.DataFrame(
            {
                "market_regime": market_regime,
                "price_position": price_position,
                "stabilization": stabilization,
                "demand_confirmation": demand,
                "momentum_confirmation": momentum,
                "execution_risk": execution,
            },
            index=indicators.index,
        )

    def falling_knife_penalty(
        self,
        indicators: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        ema_penalty = pd.Series(
            np.where(indicators["ema20_slope"] < 0, -5, 0),
            index=indicators.index,
        )
        rs_penalty = pd.Series(
            np.select(
                [
                    (indicators["rs20"] < 0)
                    & (indicators["rs20_slope"] < 0),
                    indicators["rs20"] < 0,
                ],
                [-5, -3],
                default=0,
            ),
            index=indicators.index,
        )
        risk_off = (
            (indicators["taiex"] < indicators["taiex_ema20"])
            & (indicators["taiex_ema20_slope"] < 0)
        ).fillna(False)
        return ema_penalty + rs_penalty, risk_off

    def required_score(
        self,
        quality: LowEntryV5QualityResult,
        risk_off: pd.Series,
        index: pd.Index,
    ) -> pd.Series:
        base = (
            self.config.confirmed_buy_score
            if quality.final_status is FinalQualityStatus.CONFIRMED
            else self.config.provisional_buy_score
        )
        return pd.Series(base, index=index) + (
            risk_off.astype(int) * self.config.risk_off_score_increment
        )


__all__ = [
    "LowEntryV5ScoringConfig",
    "LowEntryV5ScoringEvaluator",
]
