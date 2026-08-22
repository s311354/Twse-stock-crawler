"""Falling-knife, panic-reversal and quality eligibility rules for LES v5."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from domain.quality_v5 import BaseQualityStatus
from domain.quality_v5 import ExtendedQualityStatus
from domain.quality_v5 import FinalQualityStatus
from domain.quality_v5 import LowEntryV5QualityResult
from domain.strategies.low_entry_v5_contracts import BroadRallyTier
from domain.strategies.low_entry_v5_contracts import FallingKnifeState
from domain.strategies.low_entry_v5_contracts import ValuationStatus


class LowEntryV5QualityEligibility:
    """Centralize the quality predicates shared by panic and decision rules."""

    def __init__(self, config: Any) -> None:
        self.config = config

    @staticmethod
    def quality_metric_passed(
        quality: LowEntryV5QualityResult,
        metric_name: str,
    ) -> bool:
        return any(
            metric.name == metric_name and metric.passed is True
            for metric in quality.extended_metrics
        )

    def conditional_quality_starter_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        return bool(
            quality.final_status is FinalQualityStatus.WATCH
            and quality.base_status is BaseQualityStatus.PROVISIONAL
            and "MOPS_POINT_IN_TIME_BASE_FALLBACK" in quality.warnings
            and quality.final_score is not None
            and quality.final_score
            >= self.config.breadth_rally_minimum_quality_score
            and quality.base_data_confidence
            >= self.config.breadth_rally_minimum_base_confidence
            and quality.base_available_weight
            >= self.config.breadth_rally_minimum_fallback_base_weight
            and quality.extended_status is ExtendedQualityStatus.PASS
            and quality.extended_data_confidence
            >= self.config.breadth_rally_minimum_extended_confidence
            and quality.critical_field_coverage
            >= self.config.breadth_rally_minimum_critical_coverage
            and not quality.hard_rejects
            and self.quality_metric_passed(quality, "positive_cfo")
            and self.quality_metric_passed(quality, "no_material_issue")
            and self.quality_metric_passed(quality, "no_going_concern")
            and self.quality_metric_passed(quality, "filing_traceable")
        )

    def broad_rally_quality_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        regular_quality = quality.final_status in {
            FinalQualityStatus.CONFIRMED,
            FinalQualityStatus.PROVISIONAL,
        }
        return bool(
            (
                regular_quality
                or self.conditional_quality_starter_eligible(quality)
            )
            and quality.final_score is not None
            and quality.final_score
            >= self.config.breadth_rally_minimum_quality_score
            and quality.base_data_confidence
            >= self.config.breadth_rally_minimum_base_confidence
            and quality.extended_data_confidence
            >= self.config.breadth_rally_minimum_extended_confidence
            and quality.critical_field_coverage
            >= self.config.breadth_rally_minimum_critical_coverage
            and not quality.hard_rejects
        )

    def broad_rally_full_buy_quality_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        return bool(
            self.broad_rally_quality_eligible(quality)
            and quality.final_score is not None
            and quality.final_score
            >= self.config.breadth_rally_full_buy_minimum_quality_score
            and quality.extended_status is ExtendedQualityStatus.PASS
            and not quality.hard_rejects
            and self.quality_metric_passed(quality, "positive_cfo")
            and self.quality_metric_passed(quality, "no_material_issue")
            and self.quality_metric_passed(quality, "no_going_concern")
            and self.quality_metric_passed(quality, "filing_traceable")
        )

    def supportive_non_broad_quality_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        return bool(
            quality.final_status is FinalQualityStatus.CONFIRMED
            and quality.final_score is not None
            and quality.final_score
            >= self.config.supportive_non_broad_minimum_quality_score
            and not quality.hard_rejects
        )

    def quality_pullback_starter_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        return bool(
            quality.final_status is FinalQualityStatus.CONFIRMED
            and quality.final_score is not None
            and quality.final_score
            >= self.config.quality_pullback_minimum_quality_score
            and not quality.hard_rejects
        )


class LowEntryV5PanicEvaluator:
    """Evaluate panic state and all reversal-entry trigger variants."""

    def __init__(
        self,
        config: Any,
        quality_policy: LowEntryV5QualityEligibility | None = None,
    ) -> None:
        self.config = config
        self.quality_policy = quality_policy or LowEntryV5QualityEligibility(config)

    def evaluate_panic(
        self,
        indicators: pd.DataFrame,
        quality: LowEntryV5QualityResult,
        valuation: pd.Series,
    ) -> dict[str, pd.Series]:
        quality_permits_entry = quality.final_status in {
            FinalQualityStatus.CONFIRMED,
            FinalQualityStatus.PROVISIONAL,
        }
        quality_series = pd.Series(quality_permits_entry, index=indicators.index)
        valuation_supports_starter = valuation.isin(
            [ValuationStatus.PASS.value, ValuationStatus.SUPPORTIVE.value]
        )
        legacy_broad_rally = indicators.get(
            "broad_rally_regime",
            pd.Series(False, index=indicators.index),
        ).fillna(False)
        broad_rally_tier = indicators.get("broad_rally_tier")
        if broad_rally_tier is None:
            broad_rally_tier = pd.Series(
                np.where(
                    legacy_broad_rally,
                    BroadRallyTier.STRONG.value,
                    BroadRallyTier.NONE.value,
                ),
                index=indicators.index,
            )
        broad_rally_active = broad_rally_tier.isin(
            [BroadRallyTier.MODERATE.value, BroadRallyTier.STRONG.value]
        )
        strong_broad_rally = broad_rally_tier.eq(BroadRallyTier.STRONG.value)
        pe_below_median = (
            indicators["pe"].notna()
            & indicators["pe_median60"].notna()
            & (indicators["pe"] < indicators["pe_median60"])
        )
        opportunity = (
            indicators["drawdown60"].between(-0.35, -0.10, inclusive="both")
            & indicators["price_position60"].between(0.05, 0.35, inclusive="both")
            & pe_below_median
            & quality_series
        ).fillna(False)

        reclaimed_support = (
            (indicators["close"] > indicators["previous_low20"])
            & (indicators["clv"] > 0.4)
        ).fillna(False)
        confirmed_breakdown = (
            (indicators["close"] < indicators["previous_low20"])
            & (indicators["clv"] < 0)
        ).fillna(False)
        undercut_reclaim = (
            (indicators["low"] < indicators["previous_low20"])
            & (indicators["close"] > indicators["previous_low20"])
            & (indicators["clv"] > 0.4)
        ).fillna(False)
        blocks = pd.DataFrame(
            {
                "REPEATED_NEW_LOW": (
                    (indicators["new_low_count3"] >= 2) & ~reclaimed_support
                ).fillna(False),
                "HIGH_VOLUME_WEAK_CLOSE": (
                    (indicators["close"] < indicators["open"])
                    & (indicators["clv"] < -0.5)
                    & (indicators["volume_ratio"] > 1.5)
                ).fillna(False),
                "CONFIRMED_SUPPORT_BREAKDOWN": confirmed_breakdown,
                "THREE_WEAK_CLOSES": indicators["weak_clv3"].fillna(False),
                "ACCELERATING_DECLINE": (
                    (indicators["ema5_slope"] < 0)
                    & (indicators["r5"] < -1.5 * indicators["atr_percent"])
                    & (indicators["clv"] < 0)
                ).fillna(False),
            },
            index=indicators.index,
        )
        block_text = blocks.apply(
            lambda row: "；".join(row.index[row].tolist()) if row.any() else "NONE",
            axis=1,
        )

        panic_conditions = pd.DataFrame(
            {
                "DRAWDOWN20": indicators["drawdown20"] <= -0.15,
                "R5_ATR_SHOCK": indicators["r5"] <= -2 * indicators["atr_percent"],
                "VOLUME_SPIKE": indicators["volume_ratio"] >= 2,
                "TRUE_RANGE_EXPANSION": (
                    indicators["true_range"] >= 1.5 * indicators["atr14"]
                ),
                "LOW_POSITION60": indicators["price_position60"] <= 0.15,
            },
            index=indicators.index,
        ).fillna(False)
        reversal_conditions = pd.DataFrame(
            {
                "STRONG_CLV": indicators["clv"] >= 0.5,
                "BULLISH_BAR": indicators["close"] > indicators["open"],
                "CLOSE_UP": indicators["close"] > indicators["close"].shift(1),
                "UNDERCUT_RECLAIM": undercut_reclaim,
                "ABOVE_PREVIOUS_MIDPOINT": (
                    indicators["close"] > indicators["previous_midpoint"]
                ),
                "LONG_LOWER_SHADOW": indicators["lower_shadow_ratio"] >= 0.4,
            },
            index=indicators.index,
        ).fillna(False)
        panic_count = panic_conditions.astype(int).sum(axis=1)
        reversal_count = reversal_conditions.astype(int).sum(axis=1)
        capitulation = ((panic_count >= 1) & (reversal_count >= 3)).fillna(False)
        broad_quality = self.quality_policy.broad_rally_quality_eligible(quality)
        panic_reversal_setup = (
            (panic_count >= 1) & (reversal_count >= 2)
        ).fillna(False)
        depressed_recovery_setup = (
            (
                indicators["price_position60"]
                <= self.config.breadth_rally_maximum_price_position
            )
            & (
                indicators["drawdown60"]
                <= self.config.breadth_rally_maximum_drawdown
            )
            & (reversal_count >= 3)
            & (indicators["close"] > indicators["previous_high"])
        ).fillna(False)
        broad_reversal_setup = panic_reversal_setup | depressed_recovery_setup
        volume_confirmation = (
            (
                indicators["volume_ratio"]
                >= self.config.breadth_rally_minimum_volume_ratio
            )
            | (
                (
                    indicators["volume_ratio"]
                    >= self.config.breadth_rally_strong_close_minimum_volume_ratio
                )
                & (
                    indicators["clv"]
                    >= self.config.breadth_rally_strong_close_minimum_clv
                )
                & (
                    reversal_count
                    >= self.config.breadth_rally_strong_close_minimum_reversals
                )
            )
        ).fillna(False)
        broad_rally_trigger = (
            broad_rally_active
            & (block_text == "NONE")
            & broad_reversal_setup
            & (indicators["close"] > indicators["close"].shift(1))
            & (
                indicators["close"]
                >= indicators["open"] - (0.15 * indicators["atr14"])
            )
            & (indicators["clv"] >= self.config.breadth_rally_minimum_clv)
            & volume_confirmation
            & (indicators["cmf20"] >= self.config.breadth_rally_minimum_cmf20)
            & broad_quality
            & valuation_supports_starter
        ).fillna(False)
        broad_rally_full_buy_trigger = (
            broad_rally_trigger
            & strong_broad_rally
            & (valuation == ValuationStatus.PASS.value)
            & (panic_count >= self.config.breadth_rally_full_buy_minimum_panics)
            & (
                reversal_count
                >= self.config.breadth_rally_full_buy_minimum_reversals
            )
            & (indicators["clv"] >= self.config.breadth_rally_full_buy_minimum_clv)
            & (indicators["cmf20"] >= self.config.breadth_rally_full_buy_minimum_cmf20)
            & (
                indicators["volume_ratio"]
                >= self.config.breadth_rally_strong_close_minimum_volume_ratio
            )
            & self.quality_policy.broad_rally_full_buy_quality_eligible(quality)
        ).fillna(False)
        broad_rally_absorption_trigger = (
            broad_rally_active
            & (block_text == "NONE")
            & (panic_count >= self.config.breadth_absorption_minimum_panics)
            & (reversal_count >= self.config.breadth_absorption_minimum_reversals)
            & (indicators["close"] > indicators["close"].shift(1))
            & (
                indicators["close"]
                >= indicators["open"] - (0.15 * indicators["atr14"])
            )
            & (indicators["clv"] >= self.config.breadth_absorption_minimum_clv)
            & (indicators["cmf20"] >= self.config.breadth_absorption_minimum_cmf20)
            & (
                indicators["volume_ratio"]
                >= self.config.breadth_absorption_minimum_volume_ratio
            )
            & (
                indicators["price_position60"]
                <= self.config.breadth_absorption_maximum_price_position
            )
            & (
                indicators["drawdown60"]
                <= self.config.breadth_absorption_maximum_drawdown
            )
            & self.quality_policy.broad_rally_full_buy_quality_eligible(quality)
            & valuation_supports_starter
        ).fillna(False)
        recent_selloff_regime = indicators.get(
            "recent_broad_selloff_regime",
            pd.Series(False, index=indicators.index),
        ).fillna(False)
        multiday_panic_reversal_trigger = (
            recent_selloff_regime
            & (block_text == "NONE")
            & (reversal_count >= self.config.multiday_panic_minimum_reversals)
            & (indicators["close"] > indicators["close"].shift(1))
            & ((indicators["close"] > indicators["previous_high"]) | undercut_reclaim)
            & (indicators["clv"] >= self.config.multiday_panic_minimum_clv)
            & (indicators["cmf20"] >= self.config.multiday_panic_minimum_cmf20)
            & (
                indicators["volume_ratio"]
                >= self.config.multiday_panic_minimum_volume_ratio
            )
            & (
                indicators["price_position60"]
                <= self.config.multiday_panic_maximum_price_position
            )
            & (
                indicators["drawdown60"]
                <= self.config.multiday_panic_maximum_drawdown
            )
            & self.quality_policy.broad_rally_full_buy_quality_eligible(quality)
            & valuation_supports_starter
        ).fillna(False)

        recent_rally_regime = indicators.get(
            "recent_broad_rally_regime",
            pd.Series(False, index=indicators.index),
        ).fillna(False)
        orderly_pullback = (
            indicators["close"]
            >= (
                indicators["close"].shift(1)
                - self.config.rally_retest_maximum_pullback_atr
                * indicators["atr14"]
            )
        ).fillna(False)
        support_holding = (
            indicators["close"]
            >= indicators["previous_midpoint"] - (0.25 * indicators["atr14"])
        ).fillna(False)
        retest_volume = (
            (
                indicators["volume_ratio"]
                >= self.config.rally_retest_minimum_volume_ratio
            )
            & (
                (
                    indicators["volume_ratio"]
                    <= self.config.rally_retest_maximum_volume_ratio
                )
                | (indicators["clv"] >= self.config.breadth_rally_minimum_clv)
            )
        ).fillna(False)
        recent_rally_retest_trigger = (
            recent_rally_regime
            & (block_text == "NONE")
            & orderly_pullback
            & support_holding
            & (reversal_count >= self.config.rally_retest_minimum_reversals)
            & (indicators["clv"] >= self.config.rally_retest_minimum_clv)
            & (indicators["cmf20"] >= self.config.rally_retest_minimum_cmf20)
            & retest_volume
            & (
                indicators["price_position60"]
                <= self.config.rally_retest_maximum_price_position
            )
            & (
                indicators["drawdown60"]
                <= self.config.rally_retest_maximum_drawdown
            )
            & self.quality_policy.broad_rally_quality_eligible(quality)
            & valuation_supports_starter
        ).fillna(False)

        reclaim_orderly_pullback = (
            indicators["close"]
            >= (
                indicators["close"].shift(1)
                - self.config.rally_reclaim_maximum_pullback_atr
                * indicators["atr14"]
            )
        ).fillna(False)
        reclaim_support_holding = (
            indicators["close"]
            >= indicators["previous_midpoint"] - (0.75 * indicators["atr14"])
        ).fillna(False)
        reclaim_range_controlled = (
            indicators["true_range"]
            <= self.config.rally_reclaim_maximum_true_range_atr
            * indicators["atr14"]
        ).fillna(False)
        reclaim_demand_seed = (
            (reversal_count >= self.config.rally_reclaim_minimum_reversals)
            | (indicators["cmf20"] >= 0)
        ).fillna(False)
        recent_rally_reclaim_trigger = (
            recent_rally_regime
            & (block_text == "NONE")
            & reclaim_orderly_pullback
            & reclaim_support_holding
            & reclaim_range_controlled
            & reclaim_demand_seed
            & (indicators["clv"] >= self.config.rally_reclaim_minimum_clv)
            & (indicators["cmf20"] >= self.config.rally_reclaim_minimum_cmf20)
            & (
                indicators["volume_ratio"]
                >= self.config.rally_reclaim_minimum_volume_ratio
            )
            & (
                indicators["volume_ratio"]
                <= self.config.rally_reclaim_maximum_volume_ratio
            )
            & (
                indicators["price_position60"]
                <= self.config.rally_reclaim_maximum_price_position
            )
            & (
                indicators["drawdown60"]
                <= self.config.rally_reclaim_maximum_drawdown
            )
            & self.quality_policy.broad_rally_full_buy_quality_eligible(quality)
            & valuation_supports_starter
        ).fillna(False)

        pullback_demand_seed = (
            indicators.get(
                "cmf20_improving",
                pd.Series(False, index=indicators.index),
            ).fillna(False)
            | (
                indicators.get(
                    "normalized_obv_slope5",
                    pd.Series(np.nan, index=indicators.index),
                )
                > 0
            )
            | (
                (indicators["close"] > indicators["close"].shift(1))
                & (
                    indicators["volume_ratio"]
                    >= self.config.quality_pullback_minimum_volume_ratio
                )
            )
        ).fillna(False)
        pullback_price_confirmation = (
            (
                indicators["close"]
                > indicators.get(
                    "ema5",
                    pd.Series(np.nan, index=indicators.index),
                )
            )
            | (
                indicators.get(
                    "higher_low",
                    pd.Series(False, index=indicators.index),
                ).fillna(False)
                & (indicators["ema5_slope"] >= 0)
            )
            | (
                (indicators["close"] > indicators["close"].shift(1))
                & (indicators["close"] > indicators["previous_midpoint"])
            )
        ).fillna(False)
        pullback_confirmed = (
            pullback_price_confirmation & pullback_demand_seed
        ).fillna(False)
        pullback_stop_entry_setup = (
            (reversal_count >= 1)
            & (indicators["close"] > indicators["previous_low20"])
        ).fillna(False)
        quality_pullback_trigger = (
            (block_text == "NONE")
            & (indicators["new_low_count3"] == 0)
            & (pullback_confirmed | pullback_stop_entry_setup)
            & (indicators["clv"] >= self.config.quality_pullback_minimum_clv)
            & (
                indicators["volume_ratio"]
                >= self.config.quality_pullback_minimum_volume_ratio
            )
            & (
                indicators["price_position60"]
                <= self.config.quality_pullback_maximum_price_position
            )
            & (
                indicators["drawdown60"]
                <= self.config.quality_pullback_maximum_drawdown
            )
            & self.quality_policy.quality_pullback_starter_eligible(quality)
            & valuation_supports_starter
        ).fillna(False)

        state = pd.Series(FallingKnifeState.CLEAR.value, index=indicators.index)
        state.loc[blocks.any(axis=1)] = FallingKnifeState.PANIC_WATCH.value
        state.loc[undercut_reclaim] = FallingKnifeState.REVERSAL_WATCH.value
        state.loc[capitulation] = FallingKnifeState.REVERSAL_WATCH.value

        return {
            "opportunity": opportunity,
            "state": state,
            "block_text": block_text,
            "panic_count": panic_count,
            "reversal_count": reversal_count,
            "capitulation": capitulation,
            "broad_reversal_setup": broad_reversal_setup,
            "broad_rally_trigger": broad_rally_trigger,
            "broad_rally_active": broad_rally_active,
            "broad_rally_full_buy_trigger": broad_rally_full_buy_trigger,
            "broad_rally_absorption_trigger": broad_rally_absorption_trigger,
            "multiday_panic_reversal_trigger": multiday_panic_reversal_trigger,
            "recent_rally_retest_trigger": recent_rally_retest_trigger,
            "recent_rally_reclaim_trigger": recent_rally_reclaim_trigger,
            "quality_pullback_trigger": quality_pullback_trigger,
        }


__all__ = ["LowEntryV5PanicEvaluator", "LowEntryV5QualityEligibility"]
