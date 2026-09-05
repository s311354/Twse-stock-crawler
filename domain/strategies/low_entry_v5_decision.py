"""Decision resolution and position-sizing rules for Low Entry Model v5."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from domain.quality_v5 import FinalQualityStatus
from domain.quality_v5 import LowEntryV5QualityResult
from domain.strategies.low_entry_score_v4 import ACCOUNT_EQUITY_ATTR
from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
from domain.strategies.low_entry_v5_contracts import FallingKnifeState
from domain.strategies.low_entry_v5_contracts import ValuationStatus
from domain.strategies.low_entry_v5_panic import LowEntryV5QualityEligibility


class PositiveFloatAdapter(Protocol):
    def positive_float(self, value: Any, fallback: float) -> float: ...


class LowEntryV5DecisionEvaluator:
    """Resolve model state into executable BUY/EARLY_BUY/BUY_SMALL decisions."""

    def __init__(
        self,
        config: Any,
        market_adapter: PositiveFloatAdapter,
        quality_policy: LowEntryV5QualityEligibility,
    ) -> None:
        self.config = config
        self.market_adapter = market_adapter
        self.quality_policy = quality_policy

    def evaluate_trigger(self, indicators: pd.DataFrame) -> pd.Series:
        price_confirmation = (
            (indicators["close"] > indicators["previous_high"])
            | (
                (indicators["close"] > indicators["ema5"])
                & (indicators["close"] > indicators["vwap20"])
            )
            | (indicators["higher_low"] & (indicators["ema5_slope"] > 0))
        )
        demand_confirmation = (
            indicators["cmf20_improving"]
            | (indicators["normalized_obv_slope5"] > 0)
            | (
                (indicators["close"] > indicators["close"].shift(1))
                & (indicators["volume_ratio"] >= 0.6)
            )
        )
        return (
            price_confirmation
            & (indicators["clv"] > 0.2)
            & demand_confirmation
        ).fillna(False)

    def next_day_confirmation(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        falling_state: pd.Series,
    ) -> pd.Series:
        return (
            (falling_state.shift(1) == FallingKnifeState.REVERSAL_WATCH.value)
            & (indicators["high"] > indicators["high"].shift(1))
            & (indicators["low"] >= indicators["low"].shift(1))
            & (indicators["clv"] > 0)
            & (plan["reward_risk"] >= self.config.minimum_reward_risk)
        ).fillna(False)

    def resolve_decision(
        self,
        *,
        score: pd.Series,
        quality: LowEntryV5QualityResult,
        valuation: pd.Series,
        falling_state: pd.Series,
        next_day_confirmation: pd.Series,
        trigger: pd.Series,
        risk_gate: pd.Series,
        market_gate: pd.Series,
        required_score: pd.Series,
        broad_rally_trigger: pd.Series,
        broad_rally_risk_gate: pd.Series,
        raw_score: pd.Series,
        broad_rally_full_buy_trigger: pd.Series | None = None,
        broad_rally_absorption_trigger: pd.Series | None = None,
        multiday_panic_reversal_trigger: pd.Series | None = None,
        recent_rally_retest_trigger: pd.Series | None = None,
        recent_rally_reclaim_trigger: pd.Series | None = None,
        quality_pullback_trigger: pd.Series | None = None,
        broad_rally_active: pd.Series | None = None,
        elec_trend_pullback_trigger: pd.Series | None = None,
        elec_repriced_risk_gate: pd.Series | None = None,
    ) -> pd.Series:
        full_buy_trigger = self._optional_trigger(
            broad_rally_full_buy_trigger,
            score.index,
        )
        absorption_trigger = self._optional_trigger(
            broad_rally_absorption_trigger,
            score.index,
        )
        multiday_trigger = self._optional_trigger(
            multiday_panic_reversal_trigger,
            score.index,
        )
        retest_trigger = self._optional_trigger(
            recent_rally_retest_trigger,
            score.index,
        )
        reclaim_trigger = self._optional_trigger(
            recent_rally_reclaim_trigger,
            score.index,
        )
        pullback_trigger = self._optional_trigger(
            quality_pullback_trigger,
            score.index,
        )
        broad_active = self._optional_trigger(
            broad_rally_active,
            score.index,
        ).fillna(False)
        decision = pd.Series("ARMED", index=score.index)
        decision.loc[score < 55] = "AVOID"
        decision.loc[score.between(55, 64.999, inclusive="both")] = "WATCH"

        eligible = quality.final_status in {
            FinalQualityStatus.CONFIRMED,
            FinalQualityStatus.PROVISIONAL,
        }
        normal_buy = (
            eligible
            & (score >= required_score)
            & trigger
            & (risk_gate == "PASS")
            & (market_gate == "PASS")
            & (valuation == ValuationStatus.PASS.value)
            & (falling_state == FallingKnifeState.CLEAR.value)
        )
        early_buy = (
            eligible
            & next_day_confirmation
            & (risk_gate == "PASS")
            & (market_gate == "PASS")
            & (valuation == ValuationStatus.PASS.value)
            & (falling_state == FallingKnifeState.CLEAR.value)
            & (
                (quality.final_status is FinalQualityStatus.CONFIRMED)
                | (score >= self.config.provisional_buy_score)
            )
        )
        decision.loc[early_buy] = "EARLY_BUY"
        decision.loc[normal_buy] = "BUY"
        decision.loc[
            falling_state == FallingKnifeState.PANIC_WATCH.value
        ] = "PANIC_WATCH"
        decision.loc[
            falling_state == FallingKnifeState.REVERSAL_WATCH.value
        ] = "REVERSAL_WATCH"
        starter_valuation = valuation.isin(
            [ValuationStatus.PASS.value, ValuationStatus.SUPPORTIVE.value]
        )

        supportive_non_broad_buy_small = (
            self.quality_policy.supportive_non_broad_quality_eligible(quality)
            & ~broad_active
            & (score >= self.config.supportive_non_broad_score_threshold)
            & (score >= required_score)
            & trigger
            & (risk_gate == "PASS")
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & (valuation == ValuationStatus.SUPPORTIVE.value)
            & (falling_state == FallingKnifeState.CLEAR.value)
        ).fillna(False)
        quality_pullback_buy_small = (
            self.quality_policy.quality_pullback_starter_eligible(quality)
            & (score >= self.config.quality_pullback_score_threshold)
            & (raw_score >= self.config.quality_pullback_minimum_raw_score)
            & pullback_trigger
            & (risk_gate == "PASS")
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & starter_valuation
            & (falling_state == FallingKnifeState.CLEAR.value)
        ).fillna(False)
        air_cycle_buy_small = (
            self.quality_policy.air_cycle_starter_eligible(quality)
            & (score >= self.config.air_cycle_score_threshold)
            & (raw_score >= self.config.air_cycle_minimum_raw_score)
            & next_day_confirmation
            & (risk_gate == "PASS")
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & (valuation == ValuationStatus.SUPPORTIVE.value)
            & (falling_state == FallingKnifeState.CLEAR.value)
        ).fillna(False)
        # A different, explicitly priced trade plan is not a risk waiver.
        # Missing repriced risk evidence remains UNKNOWN, never PASS.
        elec_risk = (
            elec_repriced_risk_gate.reindex(score.index)
            if elec_repriced_risk_gate is not None
            else pd.Series("UNKNOWN", index=score.index)
        )
        elec_trend_pullback_buy_small = (
            self.config.elec_trend_pullback_enabled
            & self.quality_policy.elec_trend_pullback_quality_eligible(quality)
            & self._optional_trigger(elec_trend_pullback_trigger, score.index)
            & (score >= self.config.elec_trend_pullback_score_threshold)
            & (raw_score >= self.config.elec_trend_pullback_minimum_raw_score)
            & (elec_risk == "PASS")
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & starter_valuation
            & (falling_state == FallingKnifeState.CLEAR.value)
            & ~(normal_buy | early_buy)
        ).fillna(False)
        broad_rally_buy_small = (
            self.quality_policy.broad_rally_quality_eligible(quality)
            & (score >= self.config.breadth_rally_score_threshold)
            & (raw_score >= self.config.breadth_rally_minimum_raw_score)
            & broad_rally_trigger
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & starter_valuation
        )
        absorption_buy_small = (
            self.quality_policy.broad_rally_full_buy_quality_eligible(quality)
            & (score >= self.config.breadth_absorption_score_threshold)
            & (raw_score >= self.config.breadth_absorption_minimum_raw_score)
            & absorption_trigger
            & (risk_gate == "PASS")
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & starter_valuation
        )
        multiday_panic_buy_small = (
            self.quality_policy.broad_rally_full_buy_quality_eligible(quality)
            & (score >= self.config.multiday_panic_score_threshold)
            & (raw_score >= self.config.multiday_panic_minimum_raw_score)
            & multiday_trigger
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & starter_valuation
        )
        rally_retest_buy_small = (
            self.quality_policy.broad_rally_quality_eligible(quality)
            & (score >= self.config.rally_retest_score_threshold)
            & (raw_score >= self.config.rally_retest_minimum_raw_score)
            & retest_trigger
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & starter_valuation
        )
        rally_reclaim_buy_small = (
            self.quality_policy.broad_rally_full_buy_quality_eligible(quality)
            & (score >= self.config.rally_reclaim_score_threshold)
            & (raw_score >= self.config.rally_reclaim_minimum_raw_score)
            & reclaim_trigger
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & starter_valuation
        )
        broad_rally_buy_small = (
            broad_rally_buy_small
            | absorption_buy_small
            | multiday_panic_buy_small
            | rally_retest_buy_small
            | rally_reclaim_buy_small
            | supportive_non_broad_buy_small
            | quality_pullback_buy_small
            | air_cycle_buy_small
            | elec_trend_pullback_buy_small
        )
        decision.loc[broad_rally_buy_small] = "BUY_SMALL"

        broad_rally_full_buy = (
            self.quality_policy.broad_rally_full_buy_quality_eligible(quality)
            # BUY is the formal full-entry decision and always retains the
            # model-wide 75-point floor.  Strong breadth may unlock a
            # conditional BUY_SMALL below that floor, but it cannot promote a
            # low technical score to BUY.
            & (score >= self.config.confirmed_buy_score)
            & (score >= self.config.breadth_rally_full_buy_score_threshold)
            & (raw_score >= self.config.breadth_rally_full_buy_minimum_raw_score)
            & full_buy_trigger
            & (risk_gate == "PASS")
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & (valuation == ValuationStatus.PASS.value)
        )
        decision.loc[broad_rally_full_buy] = "BUY"

        if quality.final_status in {
            FinalQualityStatus.BASE_ONLY,
            FinalQualityStatus.WATCH,
            FinalQualityStatus.SECTOR_UNSUPPORTED,
        }:
            conditional_breadth_entry = broad_rally_buy_small | broad_rally_full_buy
            decision.loc[~conditional_breadth_entry] = "WATCH_ONLY"
        elif quality.final_status is FinalQualityStatus.DATA_INSUFFICIENT:
            decision[:] = "DATA_INSUFFICIENT"
        elif quality.final_status is FinalQualityStatus.HARD_NO_BUY:
            decision[:] = "NO_BUY"
        elif (market_gate != "PASS").any():
            decision[:] = "DATA_INSUFFICIENT"
        return decision

    def position_fraction(
        self,
        decision: pd.Series,
        quality: LowEntryV5QualityResult,
        risk_off: pd.Series,
        broad_rally_full_buy: pd.Series | None = None,
    ) -> pd.Series:
        fraction = pd.Series(0.0, index=decision.index)
        normal = (
            self.config.confirmed_position_fraction
            if quality.final_status is FinalQualityStatus.CONFIRMED
            else self.config.provisional_position_fraction
        )
        fraction.loc[decision == "BUY"] = normal
        fraction.loc[decision == "EARLY_BUY"] = normal
        fraction.loc[decision == "BUY_SMALL"] = (
            self.config.breadth_rally_position_fraction
        )
        if broad_rally_full_buy is not None:
            fraction.loc[(decision == "BUY") & broad_rally_full_buy] = (
                self.config.breadth_rally_full_buy_position_fraction
            )
        fraction.loc[
            risk_off & fraction.gt(self.config.provisional_position_fraction)
        ] = self.config.provisional_position_fraction
        return fraction

    def cap_position_shares(
        self,
        plan: pd.DataFrame,
        position_fraction: pd.Series,
        context: dict[str, Any],
    ) -> pd.Series:
        equity = self.market_adapter.positive_float(
            context.get(ACCOUNT_EQUITY_ATTR),
            self.config.account_equity,
        )
        price_scale = self.market_adapter.positive_float(
            context.get(PRICE_SCALE_ATTR),
            1.0,
        )
        scaled_risk_shares = np.floor(
            plan["shares"] * position_fraction / self.config.lot_size
        ) * self.config.lot_size
        max_shares = equity * position_fraction / (plan["entry"] / price_scale)
        lots = np.floor(max_shares / self.config.lot_size) * self.config.lot_size
        capped = pd.concat([scaled_risk_shares, lots], axis=1).min(axis=1)
        return capped.where(position_fraction > 0, 0)

    @staticmethod
    def enforce_executable_position(
        decision: pd.Series,
        recommended_shares: pd.Series,
    ) -> pd.Series:
        buy_states = decision.isin(["BUY", "EARLY_BUY", "BUY_SMALL"])
        non_executable = buy_states & recommended_shares.fillna(0).le(0)
        downgraded = decision.copy()
        downgraded.loc[non_executable] = "ARMED"
        return downgraded

    @staticmethod
    def _optional_trigger(
        trigger: pd.Series | None,
        index: pd.Index,
    ) -> pd.Series:
        return trigger if trigger is not None else pd.Series(False, index=index)


__all__ = ["LowEntryV5DecisionEvaluator"]
