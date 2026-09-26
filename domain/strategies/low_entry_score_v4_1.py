"""Low Entry Model v4.1 quality-first stabilized entry strategy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from domain.features import FeatureFrame
from domain.features import SharedFeaturePipeline
from domain.fundamentals import GateStatus
from domain.fundamentals import QualityGateResult
from domain.industry_profiles import ELECTRONICS_SECTOR_CODES
from domain.industry_profiles import IndustryProfileMode
from domain.industry_profiles import IndustryProfileRegistry
from domain.market_breadth import MARKET_BREADTH_ATTR
from domain.market_breadth import MARKET_DATA_THROUGH_ATTR
from domain.market_breadth import MARKET_SESSION_FINAL_ATTR
from domain.market_breadth import normalize_market_breadth_snapshot
from domain.strategies.low_entry_score_v4 import ACCOUNT_EQUITY_ATTR
from domain.strategies.low_entry_score_v4 import ANNUAL_FUNDAMENTALS_ATTR
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_SNAPSHOT_ATTR
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_STATUS_ATTR
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_STATUS_DETAIL_ATTR
from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from domain.strategies.low_entry_score_v4 import LowEntryScoreV4Config
from domain.strategies.low_entry_score_v4 import LowEntryScoreV4Strategy
from domain.strategies.low_entry_score_v4 import attach_low_entry_v4_context


LOW_ENTRY_STRATEGY_NAME = "low_entry_score_v4_1"
LES_V41_MODEL_VERSION = "4.1.0"
LES_V41_PARAMETER_VERSION = "2026-07-31-balanced-spec-elec-breadth-v6"


class LowEntryProfile(str, Enum):
    STRICT = "STRICT_PROFILE"
    BALANCED = "BALANCED_PROFILE"


@dataclass(frozen=True)
class LowEntryScoreV41Config(LowEntryScoreV4Config):
    """Versioned execution profile thresholds from the v4.1 specification."""

    profile: LowEntryProfile = LowEntryProfile.BALANCED
    lot_size: int = 1
    balanced_avoid_threshold: float = 55.0
    balanced_armed_threshold: float = 65.0
    balanced_buy_threshold: float = 75.0
    balanced_early_buy_threshold: float = 50.0
    balanced_maximum_stop_atr: float = 1.8
    balanced_minimum_reward_risk: float = 1.7
    balanced_clv_threshold: float = 0.2
    balanced_volume_ratio_threshold: float = 0.6
    balanced_minimum_amount_ma20: float = 10_000_000.0
    balanced_maximum_spread: float = 0.015
    balanced_estimated_liquidity_amount_multiplier: float = 2.0
    balanced_estimated_liquidity_order_ratio: float = 0.01
    provisional_buy_threshold: float = 80.0
    provisional_minimum_quality_score: float = 75.0
    provisional_minimum_data_confidence: float = 60.0
    required_domain_coverage: float = 70.0
    standard_initial_position_fraction: float = 1 / 3
    provisional_initial_position_fraction: float = 1 / 6
    chemical_buy_small_initial_position_fraction: float = 1 / 8
    electronics_buy_small_initial_position_fraction: float = 1 / 16
    breadth_rally_initial_position_fraction: float = 1 / 8
    semiconductor_buy_small_initial_position_fraction: float = 1 / 8
    chemical_probe_score_threshold: float = 60.0
    chemical_probe_minimum_quality_score: float = 75.0
    chemical_probe_clv_threshold: float = 0.8
    chemical_probe_minimum_volume_ratio: float = 0.1
    chemical_probe_minimum_amount_ma20: float = 4_000_000.0
    chemical_probe_maximum_spread: float = 0.015
    chemical_probe_maximum_order_amount_ratio: float = 0.02
    chemical_probe_estimated_amount_multiplier: float = 2.0
    chemical_probe_estimated_order_amount_ratio: float = 0.01
    chemical_probe_maximum_stop_atr: float = 2.0
    chemical_probe_minimum_reward_risk: float = 1.5
    electronics_probe_score_threshold: float = 50.0
    electronics_probe_minimum_quality_score: float = 80.0
    electronics_probe_minimum_data_confidence: float = 80.0
    electronics_probe_clv_threshold: float = 0.6
    electronics_probe_minimum_volume_ratio: float = 0.8
    electronics_probe_minimum_ema5_slope: float = -0.20
    electronics_probe_minimum_amount_ma20: float = 1_000_000.0
    electronics_probe_maximum_spread: float = 0.015
    electronics_probe_maximum_order_amount_ratio: float = 0.02
    electronics_probe_estimated_amount_multiplier: float = 2.0
    electronics_probe_estimated_order_amount_ratio: float = 0.01
    electronics_probe_maximum_stop_atr: float = 1.8
    electronics_probe_minimum_reward_risk: float = 2.0
    electronics_recovery_score_threshold: float = 40.0
    electronics_recovery_minimum_quality_score: float = 75.0
    electronics_recovery_minimum_data_confidence: float = 80.0
    electronics_recovery_minimum_price_position_score: float = 7.0
    electronics_recovery_minimum_stabilization_score: float = 6.0
    electronics_recovery_minimum_demand_score: float = 4.0
    # A completed, broad, date-aligned market rally can make lagging trend and
    # momentum components understate a genuine same-session reversal.  This
    # lower threshold is intentionally confined to the 1/16 BUY_SMALL path;
    # quality, every veto, the reversal trigger, risk/reward and liquidity
    # gates remain mandatory.
    breadth_rally_score_threshold: float = 45.0
    breadth_rally_minimum_quality_score: float = 80.0
    breadth_rally_minimum_data_confidence: float = 80.0
    breadth_rally_minimum_coverage: float = 0.80
    electronics_breadth_minimum_members: int = 100
    electronics_breadth_minimum_advancer_ratio: float = 0.80
    electronics_breadth_minimum_median_return: float = 0.015
    semiconductor_breadth_minimum_members: int = 40
    semiconductor_breadth_minimum_advancer_ratio: float = 0.85
    semiconductor_breadth_minimum_median_return: float = 0.020
    breadth_rally_minimum_taiex_return: float = 0.005
    breadth_rally_minimum_ema5_slope: float = -0.25
    breadth_rally_minimum_clv: float = 0.50
    breadth_rally_minimum_move_atr: float = 0.75
    breadth_rally_minimum_volume_ratio: float = 0.80
    semiconductor_probe_score_threshold: float = 50.0
    semiconductor_probe_minimum_quality_score: float = 75.0
    semiconductor_probe_minimum_data_confidence: float = 70.0
    semiconductor_probe_clv_threshold: float = 0.4
    semiconductor_probe_minimum_amount_ma20: float = 10_000_000.0
    semiconductor_probe_maximum_spread: float = 0.015
    semiconductor_probe_maximum_order_amount_ratio: float = 0.02
    semiconductor_probe_estimated_amount_multiplier: float = 2.0
    semiconductor_probe_estimated_order_amount_ratio: float = 0.01
    semiconductor_probe_maximum_stop_atr: float = 3.0
    semiconductor_probe_minimum_reward_risk: float = 1.5

_VETO_OUTPUT_COLUMNS = {
    "breaks_previous_low": "Veto 跌破前低",
    "high_volume_weak_close": "Veto 放量弱收",
    "accelerating_downtrend": "Veto 下降趨勢加速",
    "repeated_breakdown": "Veto 連續破底",
    "gap_below_support": "Veto 跳空破壞支撐",
    "market_relative_deterioration": "Veto 市場相對弱化",
    "quality_prohibition": "Veto 資料或基本面禁止",
}

_VETO_LABELS = {
    "breaks_previous_low": "跌破前低",
    "high_volume_weak_close": "放量弱收",
    "accelerating_downtrend": "下降趨勢加速",
    "repeated_breakdown": "連續破底",
    "gap_below_support": "跳空破壞支撐",
    "market_relative_deterioration": "市場與相對強弱惡化",
    "quality_prohibition": "資料或基本面禁止",
}

LOW_ENTRY_OUTPUT_COLUMNS = [
    "低點分數",
    "低點決策",
    "Quality Gate",
    "Quality Score",
    "Industry Quality Profile",
    "Industry Profile Version",
    "Industry Profile Mode",
    "Data Confidence",
    "Available Metric Weight",
    "DC Coverage",
    "DC Freshness",
    "DC Validation",
    "DC Source Reliability",
    "Quality QA Flags",
    "Quality Reasons",
    "Quality Hard Failures",
    "ROIC 5Y Median",
    "Positive CFO Year Ratio",
    "Gross Margin 5Y Percentile",
    "Operating Margin Trend",
    "Audit Opinion",
    "Audit Opinion Source",
    "Current Audit Opinion",
    "Comparative Audit Opinion",
    "Comparative Audit Opinion Source",
    "Accounting Concern Classification",
    "Fundamental Available At",
    "Fundamental Availability Source",
    "Fundamental Filing ID",
    "Fundamental Revision",
    "Fundamental SHA256",
    "Fundamental Source URL",
    "Fundamental Correction Announced On",
    "Fundamental Derivation Version",
    "Market Data Gate",
    "Market Data Reasons",
    "Market Data Through",
    "Market Session Final",
    "Breadth As Of",
    "Breadth Universe",
    "Breadth Coverage",
    "Breadth Valid Members",
    "Sector Advancer Ratio",
    "Sector Median R1",
    "TAIEX R1",
    "Broad Rally Regime",
    "No Veto",
    "Falling Knife Veto",
    *_VETO_OUTPUT_COLUMNS.values(),
    "LES v4.1 Technical Trigger",
    "LES v4.1 Strict Trigger",
    "LES v4.1 Trigger",
    "CHEMI Probe Trigger",
    "ELEC Probe Trigger",
    "ELEC Recovery Armed",
    "ELEC Rally Trigger",
    "SEMI Probe Trigger",
    "LES Profile",
    "Provisional Buy Eligible",
    "Profitability Coverage",
    "Balance Sheet Safety Coverage",
    "Required Domain Coverage Passed",
    "No Confirmed Negative CFO Evidence",
    "Liquidity Gate",
    "CHEMI Probe Liquidity Gate",
    "ELEC Probe Liquidity Gate",
    "ELEC Rally Liquidity Gate",
    "SEMI Probe Liquidity Gate",
    "Risk Gate",
    "Strict Risk Gate",
    "CHEMI Probe Risk Gate",
    "ELEC Probe Risk Gate",
    "ELEC Rally Risk Gate",
    "SEMI Probe Risk Gate",
    "Risk Gate Reasons",
    "CHEMI Probe Risk Reasons",
    "ELEC Probe Risk Reasons",
    "ELEC Rally Risk Reasons",
    "SEMI Probe Risk Reasons",
    "市場環境分數",
    "價格位置分數",
    "止跌結構分數",
    "買盤確認分數",
    "動能分數",
    "執行風控分數",
    "EMA5",
    "EMA20",
    "EMA5 Slope",
    "EMA20 Slope",
    "TAIEX EMA20 Slope",
    "RSI14",
    "RSI40 Reclaim",
    "MACD",
    "MACD Signal",
    "MACD Histogram",
    "MACD Histogram Improving",
    "MACD Bullish Cross",
    "ATR14",
    "CLV",
    "CMF20",
    "Volume Ratio",
    "Amount MA20",
    "Spread",
    "VWAP20",
    "RS20",
    "RS20 Deteriorating",
    "Trading Restriction Known",
    "Trading Restriction Pass",
    "60日位置",
    "Drawdown60",
    "RewardRisk",
    "Entry Candidate",
    "Entry Rule",
    "策略停損",
    "策略停利",
    "建議股數",
    "Position Fraction",
    "Effective Risk Percent",
    "訊號有效交易日",
    "Signal Date",
    "LES Model Version",
    "LES Parameter Version",
    "Risk Percent",
    "Maximum Stop ATR",
    "Minimum RewardRisk",
    "低點理由",
]


class LowEntryScoreV41Strategy(LowEntryScoreV4Strategy):
    """Quality-first LES v4.1 with explicit stabilization confirmation."""

    name = LOW_ENTRY_STRATEGY_NAME
    description = "Balanced entry profile with Quality Gate, Market Data Gate, component scoring, trigger confirmation, and breadth-rally recovery paths"
    version = LES_V41_MODEL_VERSION

    def __init__(self, config: LowEntryScoreV41Config | None = None) -> None:
        super().__init__(config or LowEntryScoreV41Config())
        self.config: LowEntryScoreV41Config

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        features = SharedFeaturePipeline(
            self._shared_feature_config()
        ).build(data)
        return self.run_with_features(data, features)

    def run_with_features(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame([self._empty_v41_result()])

        context = dict(data.attrs)
        snapshot = context.get(FUNDAMENTAL_SNAPSHOT_ATTR)
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        breadth_snapshot = normalize_market_breadth_snapshot(
            context.get(MARKET_BREADTH_ATTR)
        )
        market_data_through = context.get(MARKET_DATA_THROUGH_ATTR)
        market_session_final = bool(
            context.get(MARKET_SESSION_FINAL_ATTR, False)
        )
        sector_code = str(snapshot.get("sector") or "").strip()
        df = data.copy()
        indicators = self._calculate_indicators(df, features)
        indicators["taiex_r1"] = self._date_aligned_market_return(
            df,
            indicators["taiex_r1"],
            breadth_snapshot,
        )
        broad_rally_regime = self._evaluate_breadth_rally_regime(
            df,
            indicators,
            breadth_snapshot=breadth_snapshot,
            market_data_through=market_data_through,
            market_session_final=market_session_final,
        )
        market_data_gate, market_data_reasons = self._evaluate_market_data(
            df,
            indicators,
            context,
        )
        plan = self._build_trade_plan(indicators, context)
        electronics_conditional_plan = self._reprice_trade_plan(
            plan,
            entry=pd.concat(
                [indicators["high"], indicators["ema5"]],
                axis=1,
            ).max(axis=1),
            context=context,
        )
        quality = self._evaluate_quality(df, context)
        domain_coverage = self._quality_domain_coverage(quality)
        provisional_eligible = self._provisional_buy_eligible(
            quality,
            domain_coverage,
        )
        vetoes = self._evaluate_v41_vetoes(
            indicators,
            plan,
            quality,
            provisional_eligible=provisional_eligible,
        )
        no_veto = ~vetoes.any(axis=1)
        components = self._score_v41_components(indicators, plan)
        score = components.sum(axis=1).clip(lower=0, upper=100)
        strict_technical_trigger = self._evaluate_strict_technical_trigger(
            indicators
        )
        balanced_technical_trigger = self._evaluate_balanced_technical_trigger(
            indicators
        )
        chemical_probe_eligible = self._chemical_probe_quality_eligible(quality)
        chemical_probe_trigger = (
            self._evaluate_chemical_probe_technical_trigger(indicators)
            if chemical_probe_eligible
            else pd.Series(False, index=df.index)
        )
        electronics_probe_eligible = (
            self._electronics_probe_quality_eligible(
                quality,
                sector_code=sector_code,
            )
        )
        electronics_probe_trigger = (
            self._evaluate_electronics_probe_technical_trigger(indicators)
            if electronics_probe_eligible
            else pd.Series(False, index=df.index)
        )
        electronics_recovery_eligible = (
            self._electronics_recovery_quality_eligible(
                quality,
                sector_code=sector_code,
            )
        )
        electronics_rally_eligible = self._electronics_rally_quality_eligible(
            quality,
            sector_code=sector_code,
        )
        electronics_rally_trigger = (
            self._evaluate_electronics_breadth_rally_trigger(
                indicators,
                broad_rally_regime,
            )
            if electronics_rally_eligible
            else pd.Series(False, index=df.index)
        )
        semiconductor_probe_eligible = (
            self._semiconductor_probe_quality_eligible(
                quality,
                sector_code=sector_code,
            )
        )
        semiconductor_probe_trigger = (
            self._evaluate_semiconductor_probe_technical_trigger(indicators)
            if semiconductor_probe_eligible
            else pd.Series(False, index=df.index)
        )
        technical_trigger = (
            balanced_technical_trigger
            if self._is_balanced_profile
            else strict_technical_trigger
        )
        liquidity_gate, liquidity_pass = self._evaluate_liquidity_gate(
            indicators,
            plan,
        )
        maximum_stop_atr, minimum_reward_risk = self._standard_risk_thresholds
        trigger = (
            technical_trigger
            & no_veto
            & (plan["reward_risk"] >= minimum_reward_risk)
            & liquidity_pass
        ).fillna(False)
        strict_trigger = (
            strict_technical_trigger
            & no_veto
            & (plan["reward_risk"] >= self.config.minimum_reward_risk)
            & liquidity_pass
        ).fillna(False)
        risk_gate, risk_reasons = self._evaluate_profile_risk_gate(
            indicators,
            plan,
            market_data_gate,
            maximum_stop_atr=maximum_stop_atr,
            minimum_reward_risk=minimum_reward_risk,
        )
        strict_risk_gate, _ = self._evaluate_profile_risk_gate(
            indicators,
            plan,
            market_data_gate,
            maximum_stop_atr=self.config.maximum_stop_atr,
            minimum_reward_risk=self.config.minimum_reward_risk,
        )
        if chemical_probe_eligible:
            (
                chemical_probe_liquidity_gate,
                chemical_probe_liquidity_pass,
            ) = self._evaluate_chemical_probe_liquidity_gate(
                indicators,
                plan,
            )
            (
                chemical_probe_risk_gate,
                chemical_probe_risk_reasons,
            ) = self._evaluate_profile_risk_gate(
                indicators,
                plan,
                market_data_gate,
                maximum_stop_atr=self.config.chemical_probe_maximum_stop_atr,
                minimum_reward_risk=(
                    self.config.chemical_probe_minimum_reward_risk
                ),
                liquidity_gate=chemical_probe_liquidity_gate,
                liquidity_pass=chemical_probe_liquidity_pass,
            )
        else:
            chemical_probe_liquidity_gate = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
            chemical_probe_risk_gate = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
            chemical_probe_risk_reasons = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
        if electronics_probe_eligible:
            (
                electronics_probe_liquidity_gate,
                electronics_probe_liquidity_pass,
            ) = self._evaluate_electronics_probe_liquidity_gate(
                indicators,
                electronics_conditional_plan,
            )
            (
                electronics_probe_risk_gate,
                electronics_probe_risk_reasons,
            ) = self._evaluate_profile_risk_gate(
                indicators,
                electronics_conditional_plan,
                market_data_gate,
                maximum_stop_atr=(
                    self.config.electronics_probe_maximum_stop_atr
                ),
                minimum_reward_risk=(
                    self.config.electronics_probe_minimum_reward_risk
                ),
                liquidity_gate=electronics_probe_liquidity_gate,
                liquidity_pass=electronics_probe_liquidity_pass,
            )
        else:
            electronics_probe_liquidity_gate = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
            electronics_probe_risk_gate = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
            electronics_probe_risk_reasons = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
        if electronics_rally_eligible or electronics_recovery_eligible:
            (
                electronics_rally_liquidity_gate,
                electronics_rally_liquidity_pass,
            ) = self._evaluate_electronics_probe_liquidity_gate(
                indicators,
                plan,
                position_fraction=(
                    self.config.breadth_rally_initial_position_fraction
                ),
            )
            (
                electronics_rally_risk_gate,
                electronics_rally_risk_reasons,
            ) = self._evaluate_profile_risk_gate(
                indicators,
                plan,
                market_data_gate,
                maximum_stop_atr=(
                    self.config.electronics_probe_maximum_stop_atr
                ),
                minimum_reward_risk=(
                    self.config.electronics_probe_minimum_reward_risk
                ),
                liquidity_gate=electronics_rally_liquidity_gate,
                liquidity_pass=electronics_rally_liquidity_pass,
            )
        else:
            electronics_rally_liquidity_gate = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
            electronics_rally_risk_gate = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
            electronics_rally_risk_reasons = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
        if semiconductor_probe_eligible:
            (
                semiconductor_probe_liquidity_gate,
                semiconductor_probe_liquidity_pass,
            ) = self._evaluate_semiconductor_probe_liquidity_gate(
                indicators,
                plan,
            )
            (
                semiconductor_probe_risk_gate,
                semiconductor_probe_risk_reasons,
            ) = self._evaluate_profile_risk_gate(
                indicators,
                plan,
                market_data_gate,
                maximum_stop_atr=(
                    self.config.semiconductor_probe_maximum_stop_atr
                ),
                minimum_reward_risk=(
                    self.config.semiconductor_probe_minimum_reward_risk
                ),
                liquidity_gate=semiconductor_probe_liquidity_gate,
                liquidity_pass=semiconductor_probe_liquidity_pass,
            )
        else:
            semiconductor_probe_liquidity_gate = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
            semiconductor_probe_risk_gate = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
            semiconductor_probe_risk_reasons = pd.Series(
                "NOT_APPLICABLE",
                index=df.index,
            )
        electronics_recovery_armed = (
            pd.Series(electronics_recovery_eligible, index=df.index)
            & (score >= self.config.electronics_recovery_score_threshold)
            & (
                components["price_position"]
                >= self.config.electronics_recovery_minimum_price_position_score
            )
            & (
                components["stabilization"]
                >= self.config.electronics_recovery_minimum_stabilization_score
            )
            & (
                components["demand_confirmation"]
                >= self.config.electronics_recovery_minimum_demand_score
            )
            & (indicators["clv"] >= 0)
            & no_veto
            & (electronics_rally_risk_gate == "PASS")
            & (market_data_gate == "PASS")
        ).fillna(False)
        state = self._resolve_v41_state(
            quality=quality,
            provisional_eligible=provisional_eligible,
            vetoes=vetoes,
            score=score,
            trigger=trigger,
            strict_trigger=strict_trigger,
            risk_gate=risk_gate,
            strict_risk_gate=strict_risk_gate,
            market_data_gate=market_data_gate,
            chemical_probe_trigger=chemical_probe_trigger,
            chemical_probe_risk_gate=chemical_probe_risk_gate,
            electronics_probe_eligible=electronics_probe_eligible,
            electronics_probe_trigger=electronics_probe_trigger,
            electronics_probe_risk_gate=electronics_probe_risk_gate,
            electronics_recovery_armed=electronics_recovery_armed,
            electronics_rally_eligible=electronics_rally_eligible,
            electronics_rally_trigger=electronics_rally_trigger,
            electronics_rally_risk_gate=electronics_rally_risk_gate,
            semiconductor_probe_eligible=semiconductor_probe_eligible,
            semiconductor_probe_trigger=semiconductor_probe_trigger,
            semiconductor_probe_risk_gate=semiconductor_probe_risk_gate,
        )
        buy_small_position_fraction = self._buy_small_position_fraction(quality)
        electronics_probe_selected = (
            electronics_probe_eligible
            & (state == "BUY_SMALL")
            & electronics_probe_trigger
            & (electronics_probe_risk_gate == "PASS")
            & no_veto
        )
        electronics_rally_selected = (
            electronics_rally_eligible
            & (state == "BUY_SMALL")
            & electronics_rally_trigger
            & (electronics_rally_risk_gate == "PASS")
            & no_veto
        )
        semiconductor_probe_selected = (
            semiconductor_probe_eligible
            & (state == "BUY_SMALL")
            & semiconductor_probe_trigger
            & (semiconductor_probe_risk_gate == "PASS")
            & no_veto
        )
        electronics_conditional_selected = (
            electronics_rally_selected
            | (
                electronics_recovery_armed
                & (state == "ARMED")
            )
        )
        output_plan = plan.copy()
        for column in output_plan.columns:
            output_plan[column] = output_plan[column].where(
                ~electronics_conditional_selected,
                electronics_conditional_plan[column],
            )
        position_fraction = pd.Series(
            np.select(
                [
                    electronics_probe_selected,
                    electronics_rally_selected,
                    semiconductor_probe_selected,
                    state == "BUY_SMALL",
                    state == "BUY",
                ],
                [
                    self.config.electronics_buy_small_initial_position_fraction,
                    self.config.breadth_rally_initial_position_fraction,
                    self.config.semiconductor_buy_small_initial_position_fraction,
                    buy_small_position_fraction,
                    self.config.standard_initial_position_fraction,
                ],
                default=0.0,
            ),
            index=df.index,
        )
        suggested_shares = np.floor(
            output_plan["shares"] * position_fraction
        ).where(
            output_plan["shares"].notna()
        )
        state = self._enforce_executable_position(state, suggested_shares)
        non_executable = ~state.isin(["BUY", "BUY_SMALL"])
        position_fraction = position_fraction.mask(non_executable, 0.0)
        suggested_shares = suggested_shares.mask(non_executable, 0.0)

        veto_text = vetoes.apply(self._format_v41_vetoes, axis=1)
        quality_reasons = "；".join(quality.reasons + quality.warnings)
        veto_outputs = {
            output_column: vetoes[veto_name].astype(int)
            for veto_name, output_column in _VETO_OUTPUT_COLUMNS.items()
        }
        result = pd.DataFrame(
            {
                "低點分數": score.round(2),
                "低點決策": state,
                "Quality Gate": quality.status.value,
                "Quality Score": quality.score,
                "Industry Quality Profile": quality.industry_profile,
                "Industry Profile Version": quality.industry_profile_version,
                "Industry Profile Mode": quality.industry_profile_mode,
                "Data Confidence": quality.data_confidence,
                "Available Metric Weight": quality.available_metric_weight,
                "DC Coverage": quality.confidence_components.get("coverage"),
                "DC Freshness": quality.confidence_components.get("freshness"),
                "DC Validation": quality.confidence_components.get("validation"),
                "DC Source Reliability": quality.confidence_components.get(
                    "source_reliability"
                ),
                "Quality QA Flags": "；".join(quality.qa_flags) or "NONE",
                "Quality Reasons": quality_reasons,
                "Quality Hard Failures": (
                    "；".join(quality.hard_red_flags) or "NONE"
                ),
                "ROIC 5Y Median": quality.metrics.get("roic_5y_median"),
                "Positive CFO Year Ratio": quality.metrics.get(
                    "positive_cfo_year_ratio"
                ),
                "Gross Margin 5Y Percentile": quality.metrics.get(
                    "gross_margin_5y_percentile"
                ),
                "Operating Margin Trend": quality.metrics.get(
                    "operating_margin_trend"
                ),
                "Audit Opinion": snapshot.get("audit_opinion"),
                "Audit Opinion Source": snapshot.get("audit_opinion_source"),
                "Current Audit Opinion": snapshot.get(
                    "current_audit_opinion"
                ),
                "Comparative Audit Opinion": snapshot.get(
                    "comparative_audit_opinion"
                ),
                "Comparative Audit Opinion Source": snapshot.get(
                    "comparative_audit_opinion_source"
                ),
                "Accounting Concern Classification": snapshot.get(
                    "accounting_concern_classification"
                ),
                "Fundamental Available At": snapshot.get("available_at"),
                "Fundamental Availability Source": snapshot.get(
                    "availability_source"
                ),
                "Fundamental Filing ID": snapshot.get("filing_id"),
                "Fundamental Revision": snapshot.get("revision_number"),
                "Fundamental SHA256": snapshot.get("content_sha256"),
                "Fundamental Source URL": snapshot.get("source_url"),
                "Fundamental Correction Announced On": snapshot.get(
                    "correction_announced_on"
                ),
                "Fundamental Derivation Version": snapshot.get(
                    "derivation_version"
                ),
                "Market Data Gate": market_data_gate,
                "Market Data Reasons": market_data_reasons,
                "Market Data Through": market_data_through,
                "Market Session Final": int(market_session_final),
                "Breadth As Of": breadth_snapshot.get("as_of"),
                "Breadth Universe": breadth_snapshot.get("universe"),
                "Breadth Coverage": breadth_snapshot.get("coverage"),
                "Breadth Valid Members": breadth_snapshot.get(
                    "valid_return_count"
                ),
                "Sector Advancer Ratio": breadth_snapshot.get(
                    "advancer_ratio"
                ),
                "Sector Median R1": breadth_snapshot.get("median_return"),
                "TAIEX R1": indicators["taiex_r1"].round(4),
                "Broad Rally Regime": broad_rally_regime.astype(int),
                "No Veto": no_veto.astype(int),
                "Falling Knife Veto": veto_text,
                **veto_outputs,
                "LES v4.1 Technical Trigger": technical_trigger.astype(int),
                "LES v4.1 Strict Trigger": strict_trigger.astype(int),
                "LES v4.1 Trigger": trigger.astype(int),
                "CHEMI Probe Trigger": chemical_probe_trigger.astype(int),
                "ELEC Probe Trigger": electronics_probe_trigger.astype(int),
                "ELEC Recovery Armed": electronics_recovery_armed.astype(int),
                "ELEC Rally Trigger": electronics_rally_trigger.astype(int),
                "SEMI Probe Trigger": semiconductor_probe_trigger.astype(int),
                "LES Profile": self.profile.value,
                "Provisional Buy Eligible": int(provisional_eligible),
                "Profitability Coverage": domain_coverage["profitability"],
                "Balance Sheet Safety Coverage": domain_coverage[
                    "balance_sheet_safety"
                ],
                "Required Domain Coverage Passed": int(
                    domain_coverage["required_passed"]
                ),
                "No Confirmed Negative CFO Evidence": int(
                    domain_coverage["no_confirmed_negative_cfo"]
                ),
                "Liquidity Gate": liquidity_gate,
                "CHEMI Probe Liquidity Gate": (
                    chemical_probe_liquidity_gate
                ),
                "ELEC Probe Liquidity Gate": (
                    electronics_probe_liquidity_gate
                ),
                "ELEC Rally Liquidity Gate": (
                    electronics_rally_liquidity_gate
                ),
                "SEMI Probe Liquidity Gate": (
                    semiconductor_probe_liquidity_gate
                ),
                "Risk Gate": risk_gate,
                "Strict Risk Gate": strict_risk_gate,
                "CHEMI Probe Risk Gate": chemical_probe_risk_gate,
                "ELEC Probe Risk Gate": electronics_probe_risk_gate,
                "ELEC Rally Risk Gate": electronics_rally_risk_gate,
                "SEMI Probe Risk Gate": semiconductor_probe_risk_gate,
                "Risk Gate Reasons": risk_reasons,
                "CHEMI Probe Risk Reasons": chemical_probe_risk_reasons,
                "ELEC Probe Risk Reasons": electronics_probe_risk_reasons,
                "ELEC Rally Risk Reasons": electronics_rally_risk_reasons,
                "SEMI Probe Risk Reasons": semiconductor_probe_risk_reasons,
                "市場環境分數": components["market_regime"],
                "價格位置分數": components["price_position"],
                "止跌結構分數": components["stabilization"],
                "買盤確認分數": components["demand_confirmation"],
                "動能分數": components["momentum"],
                "執行風控分數": components["execution_risk"],
                "EMA5": indicators["ema5"].round(2),
                "EMA20": indicators["ema20"].round(2),
                "EMA5 Slope": indicators["ema5_slope"].round(4),
                "EMA20 Slope": indicators["ema20_slope"].round(4),
                "TAIEX EMA20 Slope": indicators["taiex_ema20_slope"].round(4),
                "RSI14": indicators["rsi14"].round(2),
                "RSI40 Reclaim": indicators["rsi40_reclaim"].astype(int),
                "MACD": indicators["macd"].round(4),
                "MACD Signal": indicators["macd_signal"].round(4),
                "MACD Histogram": indicators["macd_histogram"].round(4),
                "MACD Histogram Improving": indicators[
                    "macd_histogram_improving"
                ].astype(int),
                "MACD Bullish Cross": indicators["macd_bullish_cross"].astype(
                    int
                ),
                "ATR14": indicators["atr14"].round(2),
                "CLV": indicators["clv"].round(4),
                "CMF20": indicators["cmf20"].round(4),
                "Volume Ratio": indicators["volume_ratio"].round(4),
                "Amount MA20": indicators["amount_ma20"].round(2),
                "Spread": indicators["spread"].round(4),
                "VWAP20": indicators["vwap20"].round(2),
                "RS20": indicators["rs20"].round(4),
                "RS20 Deteriorating": indicators[
                    "rs20_deteriorating"
                ].astype(int),
                "Trading Restriction Known": indicators[
                    "restriction_known"
                ].astype(int),
                "Trading Restriction Pass": indicators[
                    "restriction_pass"
                ].astype(int),
                "60日位置": indicators["price_position60"].round(4),
                "Drawdown60": indicators["drawdown60"].round(4),
                "RewardRisk": output_plan["reward_risk"].round(4),
                "Entry Candidate": output_plan["entry"].round(2),
                "Entry Rule": np.where(
                    electronics_conditional_selected,
                    "NEXT_SESSION_BREAKS_RECOVERY_LEVEL",
                    "NEXT_SESSION_BREAKS_SIGNAL_HIGH",
                ),
                "策略停損": output_plan["stop"].round(2),
                "策略停利": output_plan["target"].round(2),
                "建議股數": suggested_shares.astype("Int64"),
                "Position Fraction": position_fraction.round(4),
                "Effective Risk Percent": (
                    position_fraction * self.config.risk_percent
                ).round(6),
                "訊號有效交易日": np.where(
                    electronics_recovery_armed,
                    1,
                    np.where(
                        state == "ARMED",
                        self.config.armed_sessions,
                        np.where(
                            state.isin(["BUY", "BUY_SMALL"]),
                            1,
                            0,
                        ),
                    ),
                ),
                "Signal Date": context.get(SIGNAL_DATE_ATTR),
                "LES Model Version": LES_V41_MODEL_VERSION,
                "LES Parameter Version": LES_V41_PARAMETER_VERSION,
                "Risk Percent": self.config.risk_percent,
                "Maximum Stop ATR": maximum_stop_atr,
                "Minimum RewardRisk": minimum_reward_risk,
                "低點理由": self._build_v41_reasons(
                    quality=quality,
                    provisional_eligible=provisional_eligible,
                    components=components,
                    veto_text=veto_text,
                    technical_trigger=technical_trigger,
                    strict_trigger=strict_trigger,
                    trigger=trigger,
                    liquidity_gate=liquidity_gate,
                    risk_gate=risk_gate,
                    strict_risk_gate=strict_risk_gate,
                    risk_reasons=risk_reasons,
                    chemical_probe_trigger=chemical_probe_trigger,
                    chemical_probe_liquidity_gate=(
                        chemical_probe_liquidity_gate
                    ),
                    chemical_probe_risk_gate=chemical_probe_risk_gate,
                    chemical_probe_risk_reasons=chemical_probe_risk_reasons,
                    electronics_probe_eligible=electronics_probe_eligible,
                    electronics_probe_trigger=electronics_probe_trigger,
                    electronics_probe_liquidity_gate=(
                        electronics_probe_liquidity_gate
                    ),
                    electronics_probe_risk_gate=electronics_probe_risk_gate,
                    electronics_probe_risk_reasons=(
                        electronics_probe_risk_reasons
                    ),
                    electronics_recovery_armed=electronics_recovery_armed,
                    electronics_rally_trigger=electronics_rally_trigger,
                    electronics_rally_liquidity_gate=(
                        electronics_rally_liquidity_gate
                    ),
                    electronics_rally_risk_gate=electronics_rally_risk_gate,
                    electronics_rally_risk_reasons=(
                        electronics_rally_risk_reasons
                    ),
                    broad_rally_regime=broad_rally_regime,
                    semiconductor_probe_eligible=(
                        semiconductor_probe_eligible
                    ),
                    semiconductor_probe_trigger=semiconductor_probe_trigger,
                    semiconductor_probe_liquidity_gate=(
                        semiconductor_probe_liquidity_gate
                    ),
                    semiconductor_probe_risk_gate=(
                        semiconductor_probe_risk_gate
                    ),
                    semiconductor_probe_risk_reasons=(
                        semiconductor_probe_risk_reasons
                    ),
                    state=state,
                ),
            },
            index=df.index,
        )
        return pd.concat([df, result], axis=1)

    def _calculate_indicators(
        self,
        data: pd.DataFrame,
        features: FeatureFrame | None = None,
    ) -> pd.DataFrame:
        shared = features or SharedFeaturePipeline(
            self._shared_feature_config()
        ).build(data)
        indicators = super()._calculate_indicators(data, shared)
        feature_data = shared.to_pandas()
        close = indicators["close"]
        ema5 = indicators["ema5"]
        atr14 = indicators["atr14"]
        taiex_ema20 = indicators["taiex_ema20"]

        rsi14 = feature_data["rsi14_wilder"]
        macd = feature_data["macd_strict"]
        macd_signal = feature_data["macd_signal_strict"]
        macd_histogram = feature_data["macd_histogram_strict"]

        restriction_known, restriction_pass = self._restriction_status(data)
        indicators["ema5_slope"] = self._safe_series_div(
            ema5 - ema5.shift(3),
            3 * atr14,
        )
        indicators["taiex_ema20_slope"] = (
            taiex_ema20 - taiex_ema20.shift(5)
        ) / 5
        indicators["taiex_r1"] = self._safe_series_div(
            indicators["taiex"],
            indicators["taiex"].shift(1),
        ) - 1
        indicators["rsi14"] = rsi14
        indicators["rsi40_reclaim"] = (
            (rsi14 > 40) & (rsi14.shift(1) <= 40)
        ).fillna(False)
        indicators["macd"] = macd
        indicators["macd_signal"] = macd_signal
        indicators["macd_histogram"] = macd_histogram
        indicators["macd_histogram_improving"] = (
            (macd_histogram > macd_histogram.shift(1))
            & (macd_histogram.shift(1) > macd_histogram.shift(2))
        ).fillna(False)
        indicators["macd_bullish_cross"] = (
            (macd > macd_signal)
            & (macd.shift(1) <= macd_signal.shift(1))
        ).fillna(False)
        indicators["rs20_deteriorating"] = (
            (indicators["rs20"] < indicators["rs20"].shift(1))
            & (
                indicators["rs20"].shift(1)
                < indicators["rs20"].shift(2)
            )
        ).fillna(False)
        indicators["restriction_known"] = restriction_known
        indicators["restriction_pass"] = restriction_pass
        return indicators

    def _reprice_trade_plan(
        self,
        plan: pd.DataFrame,
        *,
        entry: pd.Series,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        """Recompute risk and size at a conditional recovery trigger.

        Recovery candidates that remain below EMA5 must clear both the
        signal-day high and EMA5. Reusing a plan priced only at the signal-day
        high would overstate reward/risk and position size.
        """

        price_scale = self._positive_float(
            context.get(PRICE_SCALE_ATTR),
            1.0,
        )
        repriced = plan.copy()
        repriced["entry"] = self._round_up_to_twse_stock_tick(
            entry,
            price_scale=price_scale,
        )
        repriced["risk"] = repriced["entry"] - repriced["stop"]
        repriced["reward_risk"] = self._safe_series_div(
            repriced["target"] - repriced["entry"],
            repriced["risk"],
        ).where(repriced["risk"] > 0)

        account_equity = self._positive_float(
            context.get(ACCOUNT_EQUITY_ATTR),
            self.config.account_equity,
        )
        risk_per_share = repriced["risk"] / price_scale
        raw_shares = (
            account_equity * self.config.risk_percent
        ) / risk_per_share.where(risk_per_share > 0)
        lots = np.floor(raw_shares / self.config.lot_size)
        repriced["shares"] = pd.Series(
            lots * self.config.lot_size,
            index=plan.index,
        ).where(raw_shares.notna())
        repriced["order_amount"] = (
            repriced["shares"] * (repriced["entry"] / price_scale)
        )
        return repriced

    @staticmethod
    def _round_up_to_twse_stock_tick(
        prices: pd.Series,
        *,
        price_scale: float,
    ) -> pd.Series:
        """Round a conditional stock trigger up to a valid TWSE price tick."""

        actual = pd.to_numeric(prices, errors="coerce") / price_scale
        tick = pd.Series(
            np.select(
                [
                    actual < 10,
                    actual < 50,
                    actual < 100,
                    actual < 500,
                    actual < 1000,
                ],
                [0.01, 0.05, 0.10, 0.50, 1.00],
                default=5.00,
            ),
            index=prices.index,
            dtype=float,
        )
        rounded = (
            np.ceil((actual / tick) - 1e-12)
            * tick
            * price_scale
        )
        return rounded.where(actual.notna())

    def _evaluate_v41_vetoes(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        quality: QualityGateResult,
        *,
        provisional_eligible: bool,
    ) -> pd.DataFrame:
        del plan
        new_low_count3 = (
            indicators["new_20_day_low"]
            .astype(int)
            .rolling(3, min_periods=3)
            .sum()
        )
        prior_support = indicators["recent_confirmed_swing_low"].shift(1)
        average_clv3 = indicators["clv"].rolling(3, min_periods=3).mean()
        taiex_risk_off = (
            (indicators["taiex"] < indicators["taiex_ema20"])
            & (indicators["taiex_ema20_slope"] < 0)
        )
        quality_prohibited = quality.status != GateStatus.PASS and not (
            self._is_balanced_profile
            and quality.status == GateStatus.PROVISIONAL
            and provisional_eligible
        ) and not self._industry_buy_small_quality_eligible(quality)
        return pd.DataFrame(
            {
                "breaks_previous_low": indicators[
                    "new_20_day_low"
                ].fillna(False),
                "high_volume_weak_close": (
                    (indicators["close"] < indicators["open"])
                    & (indicators["clv"] < -0.5)
                    & (indicators["volume_ratio"] > 1.5)
                ).fillna(False),
                "accelerating_downtrend": (
                    (indicators["ema20_slope"] < -0.10)
                    & (indicators["ema5_slope"] < 0)
                    & (
                        indicators["r5"]
                        < -1.5 * indicators["atr_percent"]
                    )
                ).fillna(False),
                "repeated_breakdown": (
                    (new_low_count3 >= 2) & (average_clv3 < 0)
                ).fillna(False),
                "gap_below_support": (
                    prior_support.notna()
                    & (
                        indicators["close"].shift(1)
                        >= prior_support
                    )
                    & (indicators["open"] < prior_support)
                    & (indicators["close"] < prior_support)
                    & (indicators["normalized_gap"] < -0.5)
                ).fillna(False),
                "market_relative_deterioration": (
                    indicators["rs20_deteriorating"] & taiex_risk_off
                ).fillna(False),
                "quality_prohibition": pd.Series(
                    quality_prohibited,
                    index=indicators.index,
                ),
            },
            index=indicators.index,
        )

    def _score_v41_components(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
    ) -> pd.DataFrame:
        minimum_amount_ma20, maximum_spread = self._liquidity_thresholds
        maximum_stop_atr, minimum_reward_risk = self._standard_risk_thresholds
        support_distance = self._safe_series_div(
            indicators["close"] - indicators["recent_confirmed_swing_low"],
            indicators["atr14"],
        )
        no_new_low3 = (
            indicators["new_20_day_low"]
            .astype(int)
            .rolling(3, min_periods=3)
            .sum()
            == 0
        )
        atr_not_expanding = (
            indicators["atr_percent"] <= indicators["atr_percent_ma20"]
        )
        reasonable_volume = (
            (indicators["close"] > indicators["close"].shift(1))
            & indicators["volume_ratio"].between(
                0.9,
                1.8,
                inclusive="both",
            )
            & (indicators["clv"] >= 0)
        )
        valuation_reasonable = (
            (indicators["pe"] > 0)
            & indicators["pe_lower_bound60"].notna()
            & (indicators["pe"] <= indicators["pe_lower_bound60"])
        )
        average_value_sufficient = (
            (indicators["amount_ma20"] >= minimum_amount_ma20)
            & plan["order_amount"].notna()
            & (
                plan["order_amount"]
                <= self.config.maximum_order_amount_ratio
                * indicators["amount_ma20"]
            )
        )

        market_regime = (
            4 * (indicators["taiex"] > indicators["taiex_ema20"]).astype(int)
            + 3 * (indicators["taiex_ema20_slope"] >= 0).astype(int)
            + 4 * (indicators["rs20"] > 0).astype(int)
            + 4
            * (
                indicators["restriction_known"]
                & indicators["restriction_pass"]
            ).astype(int)
            + 2 * (~indicators["restriction_known"]).astype(int)
        )
        price_position = (
            5
            * indicators["price_position60"]
            .between(0.10, 0.35, inclusive="both")
            .astype(int)
            + 4
            * indicators["drawdown60"]
            .between(-0.25, -0.08, inclusive="both")
            .astype(int)
            + 3
            * (
                indicators["recent_confirmed_swing_low"].notna()
                & (support_distance >= 0)
                & (support_distance <= 1.0)
            ).astype(int)
            + 3 * valuation_reasonable.astype(int)
        )
        stabilization = (
            6 * no_new_low3.astype(int)
            + 7 * indicators["higher_low"].astype(int)
            + 4 * (indicators["close"] > indicators["ema5"]).astype(int)
            + 4 * (indicators["ema5_slope"] > 0).astype(int)
            + 5
            * (
                indicators["close"] > indicators["high"].shift(1)
            ).astype(int)
            + 4 * atr_not_expanding.astype(int)
        )
        demand_confirmation = (
            5 * (indicators["clv"] > 0.4).astype(int)
            + 5 * indicators["cmf20_improving"].astype(int)
            + 4 * (indicators["normalized_obv_slope5"] > 0).astype(int)
            + 3 * reasonable_volume.astype(int)
            + 3 * (indicators["close"] > indicators["vwap20"]).astype(int)
        )
        momentum = (
            4 * indicators["rsi40_reclaim"].astype(int)
            + 3 * indicators["macd_histogram_improving"].astype(int)
            + 3 * indicators["macd_bullish_cross"].astype(int)
        )
        execution_risk = (
            3
            * (
                (plan["risk"] > 0)
                & (
                    plan["risk"]
                    <= maximum_stop_atr * indicators["atr14"]
                )
            ).astype(int)
            + 3
            * (
                plan["reward_risk"] >= minimum_reward_risk
            ).astype(int)
            + 2 * average_value_sufficient.astype(int)
            + 2
            * (
                indicators["spread"].notna()
                & (indicators["spread"] <= maximum_spread)
            ).astype(int)
        )
        return pd.DataFrame(
            {
                "market_regime": market_regime,
                "price_position": price_position,
                "stabilization": stabilization,
                "demand_confirmation": demand_confirmation,
                "momentum": momentum,
                "execution_risk": execution_risk,
            },
            index=indicators.index,
        )

    def _evaluate_strict_technical_trigger(
        self,
        indicators: pd.DataFrame,
    ) -> pd.Series:
        return (
            (indicators["close"] > indicators["high"].shift(1))
            & (indicators["clv"] > 0.4)
            & indicators["cmf20_improving"]
        ).fillna(False)

    def _evaluate_balanced_technical_trigger(
        self,
        indicators: pd.DataFrame,
    ) -> pd.Series:
        price_trigger = (
            (
                (indicators["close"] > indicators["ema5"])
                & (indicators["close"] > indicators["vwap20"])
            )
            | (indicators["close"] > indicators["high"].shift(1))
            | (
                indicators["higher_low"]
                & (indicators["close"] > indicators["ema5"])
                & (indicators["ema5_slope"] > 0)
            )
        )
        demand_evidence = (
            indicators["cmf20_improving"]
            | (indicators["normalized_obv_slope5"] > 0)
            | (
                (indicators["close"] > indicators["close"].shift(1))
                & (
                    indicators["volume_ratio"]
                    >= self.config.balanced_volume_ratio_threshold
                )
            )
        )
        return (
            price_trigger
            & (indicators["clv"] > self.config.balanced_clv_threshold)
            & demand_evidence
        ).fillna(False)

    def _evaluate_chemical_probe_technical_trigger(
        self,
        indicators: pd.DataFrame,
    ) -> pd.Series:
        """Confirm a slow-turnover CHEMI reversal without requiring volume expansion.

        The actual entry still requires the next session to break the signal-day
        high. This path is intentionally limited to a small initial position and
        cannot override a Falling-Knife Veto.
        """

        return (
            indicators["higher_low"]
            & (indicators["close"] > indicators["ema5"])
            & (indicators["ema5_slope"] > 0)
            & (indicators["close"] > indicators["open"])
            & (
                indicators["clv"]
                >= self.config.chemical_probe_clv_threshold
            )
            & (
                indicators["volume_ratio"]
                >= self.config.chemical_probe_minimum_volume_ratio
            )
        ).fillna(False)

    def _evaluate_electronics_probe_technical_trigger(
        self,
        indicators: pd.DataFrame,
    ) -> pd.Series:
        """Arm a small ELEC entry after a higher low and strong demand close.

        This path is limited to non-semiconductor electronics sectors.  It does
        not call a falling price a reversal: price must form a higher low,
        reclaim EMA5, close strongly with positive CMF on adequate volume, and
        the next session must still break the signal-day high before execution.
        """

        return (
            indicators["higher_low"]
            & (indicators["close"] > indicators["ema5"])
            & (
                indicators["ema5_slope"]
                > self.config.electronics_probe_minimum_ema5_slope
            )
            & (indicators["close"] > indicators["open"])
            & (
                indicators["clv"]
                >= self.config.electronics_probe_clv_threshold
            )
            & (indicators["cmf20"] > 0)
            & (
                indicators["volume_ratio"]
                >= self.config.electronics_probe_minimum_volume_ratio
            )
        ).fillna(False)

    def _evaluate_breadth_rally_regime(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        *,
        breadth_snapshot: dict[str, object],
        market_data_through: object,
        market_session_final: bool,
    ) -> pd.Series:
        """Recognize a broad rally only from a date-aligned completed session."""

        false = pd.Series(False, index=indicators.index)
        if not breadth_snapshot or "Date" not in data:
            return false

        universe = str(breadth_snapshot.get("universe") or "").upper()
        if universe == "SEMI":
            minimum_members = (
                self.config.semiconductor_breadth_minimum_members
            )
            minimum_advancer_ratio = (
                self.config.semiconductor_breadth_minimum_advancer_ratio
            )
            minimum_median_return = (
                self.config.semiconductor_breadth_minimum_median_return
            )
        elif universe == "ELEC":
            minimum_members = self.config.electronics_breadth_minimum_members
            minimum_advancer_ratio = (
                self.config.electronics_breadth_minimum_advancer_ratio
            )
            minimum_median_return = (
                self.config.electronics_breadth_minimum_median_return
            )
        else:
            return false

        coverage = pd.to_numeric(
            breadth_snapshot.get("coverage"),
            errors="coerce",
        )
        valid_members = pd.to_numeric(
            breadth_snapshot.get("valid_return_count"),
            errors="coerce",
        )
        advancer_ratio = pd.to_numeric(
            breadth_snapshot.get("advancer_ratio"),
            errors="coerce",
        )
        median_return = pd.to_numeric(
            breadth_snapshot.get("median_return"),
            errors="coerce",
        )
        breadth_as_of = str(breadth_snapshot.get("as_of") or "")
        market_through = str(market_data_through or "")
        session_is_final = bool(
            market_session_final
            and breadth_snapshot.get("source_final") is True
        )

        scalar_gate = bool(
            session_is_final
            and breadth_as_of
            and breadth_as_of == market_through
            and pd.notna(coverage)
            and coverage >= self.config.breadth_rally_minimum_coverage
            and pd.notna(valid_members)
            and valid_members >= minimum_members
            and pd.notna(advancer_ratio)
            and advancer_ratio >= minimum_advancer_ratio
            and pd.notna(median_return)
            and median_return >= minimum_median_return
        )
        if not scalar_gate:
            return false

        row_dates = pd.to_datetime(data["Date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        return (
            (row_dates == breadth_as_of)
            & (
                indicators["taiex_r1"]
                >= self.config.breadth_rally_minimum_taiex_return
            )
        ).fillna(False)

    @staticmethod
    def _date_aligned_market_return(
        data: pd.DataFrame,
        fallback: pd.Series,
        breadth_snapshot: dict[str, object],
    ) -> pd.Series:
        """Use the completed market session return for every tracked stock.

        A suspended or intermittently traded stock may omit sessions from its
        own history. Computing TAIEX R1 from that shortened frame would then
        compare non-adjacent market closes. The cross-sectional snapshot is
        built once per completed market session, so its TAIEX return is the
        canonical date-aligned value.
        """

        resolved = fallback.copy()
        if "Date" not in data:
            return resolved

        as_of = str(breadth_snapshot.get("as_of") or "")
        market_return = pd.to_numeric(
            breadth_snapshot.get("taiex_return"),
            errors="coerce",
        )
        if not as_of or pd.isna(market_return):
            return resolved

        row_dates = pd.to_datetime(
            data["Date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
        resolved.loc[row_dates == as_of] = float(market_return)
        return resolved

    def _evaluate_electronics_breadth_rally_trigger(
        self,
        indicators: pd.DataFrame,
        broad_rally_regime: pd.Series,
    ) -> pd.Series:
        """Allow a small entry on a completed broad reversal session.

        The broad-rally path relaxes only the Higher-Low and already-positive
        CMF requirements. It still requires a strong individual reversal,
        every Falling-Knife Veto to clear, and the existing small-position risk
        and liquidity gates.
        """

        price_expansion = (
            (indicators["close"] > indicators["high"].shift(1))
            | (
                indicators["r1"]
                >= (
                    self.config.breadth_rally_minimum_move_atr
                    * indicators["atr_percent"]
                )
            )
        )
        demand_evidence = (
            indicators["cmf20_improving"]
            | (indicators["normalized_obv_slope5"] > 0)
            | (
                (indicators["close"] > indicators["close"].shift(1))
                & (
                    indicators["volume_ratio"]
                    >= self.config.breadth_rally_minimum_volume_ratio
                )
            )
        )
        return (
            broad_rally_regime
            & (indicators["close"] > indicators["open"])
            & (indicators["close"] > indicators["close"].shift(1))
            & (indicators["close"] > indicators["ema5"])
            & (
                indicators["ema5_slope"]
                > self.config.breadth_rally_minimum_ema5_slope
            )
            & (indicators["clv"] >= self.config.breadth_rally_minimum_clv)
            & price_expansion
            & demand_evidence
        ).fillna(False)

    def _evaluate_semiconductor_probe_technical_trigger(
        self,
        indicators: pd.DataFrame,
    ) -> pd.Series:
        """Confirm a SEMI rebound while retaining positive demand evidence.

        Semiconductor shares often form a valid reversal with a wider
        structural stop than the general-industry two-ATR limit.  The probe
        therefore relaxes only the risk-distance ceiling; it still requires a
        balanced price trigger, a strong close, positive short-term slope, and
        improving MACD histogram.  Execution remains conditional on the next
        session breaking the signal-day high.
        """

        return (
            self._evaluate_balanced_technical_trigger(indicators)
            & (indicators["close"] > indicators["ema5"])
            & (indicators["ema5_slope"] > 0)
            & (
                indicators["clv"]
                >= self.config.semiconductor_probe_clv_threshold
            )
            & indicators["macd_histogram_improving"]
        ).fillna(False)

    def _evaluate_liquidity_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        minimum_amount_ma20, maximum_spread = self._liquidity_thresholds
        return self._evaluate_liquidity_gate_with_thresholds(
            indicators,
            plan,
            minimum_amount_ma20=minimum_amount_ma20,
            maximum_spread=maximum_spread,
            maximum_order_amount_ratio=(
                self.config.maximum_order_amount_ratio
            ),
            allow_estimate=self._is_balanced_profile,
            estimated_amount_multiplier=(
                self.config.balanced_estimated_liquidity_amount_multiplier
            ),
            estimated_order_amount_ratio=(
                self.config.balanced_estimated_liquidity_order_ratio
            ),
        )

    def _evaluate_chemical_probe_liquidity_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        probe_plan = plan.copy()
        probe_plan["order_amount"] = (
            probe_plan["order_amount"]
            * self.config.chemical_buy_small_initial_position_fraction
        )
        return self._evaluate_liquidity_gate_with_thresholds(
            indicators,
            probe_plan,
            minimum_amount_ma20=(
                self.config.chemical_probe_minimum_amount_ma20
            ),
            maximum_spread=self.config.chemical_probe_maximum_spread,
            maximum_order_amount_ratio=(
                self.config.chemical_probe_maximum_order_amount_ratio
            ),
            allow_estimate=True,
            estimated_amount_multiplier=(
                self.config.chemical_probe_estimated_amount_multiplier
            ),
            estimated_order_amount_ratio=(
                self.config.chemical_probe_estimated_order_amount_ratio
            ),
        )

    def _evaluate_electronics_probe_liquidity_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        *,
        position_fraction: float | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        probe_plan = plan.copy()
        if position_fraction is None:
            position_fraction = (
                self.config.electronics_buy_small_initial_position_fraction
            )
        probe_plan["order_amount"] = (
            probe_plan["order_amount"]
            * position_fraction
        )
        return self._evaluate_liquidity_gate_with_thresholds(
            indicators,
            probe_plan,
            minimum_amount_ma20=(
                self.config.electronics_probe_minimum_amount_ma20
            ),
            maximum_spread=self.config.electronics_probe_maximum_spread,
            maximum_order_amount_ratio=(
                self.config.electronics_probe_maximum_order_amount_ratio
            ),
            allow_estimate=True,
            estimated_amount_multiplier=(
                self.config.electronics_probe_estimated_amount_multiplier
            ),
            estimated_order_amount_ratio=(
                self.config.electronics_probe_estimated_order_amount_ratio
            ),
        )

    def _evaluate_semiconductor_probe_liquidity_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        probe_plan = plan.copy()
        probe_plan["order_amount"] = (
            probe_plan["order_amount"]
            * self.config.semiconductor_buy_small_initial_position_fraction
        )
        return self._evaluate_liquidity_gate_with_thresholds(
            indicators,
            probe_plan,
            minimum_amount_ma20=(
                self.config.semiconductor_probe_minimum_amount_ma20
            ),
            maximum_spread=self.config.semiconductor_probe_maximum_spread,
            maximum_order_amount_ratio=(
                self.config.semiconductor_probe_maximum_order_amount_ratio
            ),
            allow_estimate=True,
            estimated_amount_multiplier=(
                self.config.semiconductor_probe_estimated_amount_multiplier
            ),
            estimated_order_amount_ratio=(
                self.config.semiconductor_probe_estimated_order_amount_ratio
            ),
        )

    @staticmethod
    def _evaluate_liquidity_gate_with_thresholds(
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        *,
        minimum_amount_ma20: float,
        maximum_spread: float,
        maximum_order_amount_ratio: float,
        allow_estimate: bool,
        estimated_amount_multiplier: float,
        estimated_order_amount_ratio: float,
    ) -> tuple[pd.Series, pd.Series]:
        amount_and_order_known = (
            indicators["amount_ma20"].notna()
            & plan["order_amount"].notna()
        )
        exact_pass = (
            amount_and_order_known
            & indicators["spread"].notna()
            & (indicators["amount_ma20"] >= minimum_amount_ma20)
            & (indicators["spread"] <= maximum_spread)
            & (
                plan["order_amount"]
                <= maximum_order_amount_ratio * indicators["amount_ma20"]
            )
        )
        estimated_pass = (
            allow_estimate
            & amount_and_order_known
            & indicators["spread"].isna()
            & (
                indicators["amount_ma20"]
                >= (
                    estimated_amount_multiplier * minimum_amount_ma20
                )
            )
            & (
                plan["order_amount"]
                <= (
                    estimated_order_amount_ratio * indicators["amount_ma20"]
                )
            )
        )
        passed = exact_pass | estimated_pass
        known_failure = amount_and_order_known & ~passed & (
            indicators["spread"].notna()
            | (indicators["amount_ma20"] < minimum_amount_ma20)
        )
        status = pd.Series(
            np.select(
                [exact_pass, estimated_pass, known_failure],
                ["PASS", "PASS_ESTIMATED", "FAIL"],
                default="UNKNOWN",
            ),
            index=indicators.index,
        )
        return status, passed

    def _evaluate_profile_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_data_gate: pd.Series,
        *,
        maximum_stop_atr: float,
        minimum_reward_risk: float,
        liquidity_gate: pd.Series | None = None,
        liquidity_pass: pd.Series | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        if liquidity_gate is None or liquidity_pass is None:
            liquidity_gate, liquidity_pass = self._evaluate_liquidity_gate(
                indicators,
                plan,
            )
        market_history_known = indicators[
            [
                "taiex",
                "taiex_ema20",
                "taiex_r5",
                "rs20",
                "price_position60",
                "drawdown60",
            ]
        ].notna().all(axis=1)
        plan_known = (
            plan[
                ["entry", "stop", "target", "risk", "reward_risk", "shares"]
            ].notna().all(axis=1)
            & indicators["atr14"].notna()
            & (liquidity_gate != "UNKNOWN")
            & market_history_known
            & (market_data_gate == "PASS")
        )
        risk_pass = (
            plan_known
            & (plan["risk"] > 0)
            & (plan["risk"] <= maximum_stop_atr * indicators["atr14"])
            & (plan["reward_risk"] >= minimum_reward_risk)
            & (plan["shares"] > 0)
            & liquidity_pass
        )
        known_failure = plan_known & ~risk_pass
        status = pd.Series(
            np.select(
                [risk_pass, known_failure],
                ["PASS", "FAIL"],
                default="UNKNOWN",
            ),
            index=indicators.index,
        )
        reason_frame = pd.DataFrame(
            {
                "NO_CONFIRMED_SWING_LOW": plan["stop"].isna(),
                "INVALID_STOP": plan["risk"].notna() & (plan["risk"] <= 0),
                "STOP_TOO_FAR": plan["risk"].notna()
                & (
                    plan["risk"]
                    > maximum_stop_atr * indicators["atr14"]
                ),
                "INVALID_TARGET": plan["target"].isna()
                | (plan["target"] <= plan["entry"]),
                "INSUFFICIENT_REWARD_RISK": plan["reward_risk"].notna()
                & (plan["reward_risk"] < minimum_reward_risk),
                "LIQUIDITY_UNKNOWN": liquidity_gate == "UNKNOWN",
                "LIQUIDITY_FAILED": liquidity_gate == "FAIL",
                "MARKET_HISTORY_INSUFFICIENT": ~market_history_known,
                "MARKET_DATA_INVALID": market_data_gate != "PASS",
                "POSITION_NOT_EXECUTABLE": plan["shares"].notna()
                & (plan["shares"] <= 0),
            },
            index=indicators.index,
        )
        reasons = reason_frame.apply(
            lambda row: (
                "；".join(row.index[row].tolist()) if row.any() else "PASS"
            ),
            axis=1,
        )
        return status, reasons

    def _resolve_v41_state(
        self,
        *,
        quality: QualityGateResult,
        provisional_eligible: bool,
        vetoes: pd.DataFrame,
        score: pd.Series,
        trigger: pd.Series,
        strict_trigger: pd.Series,
        risk_gate: pd.Series,
        strict_risk_gate: pd.Series,
        market_data_gate: pd.Series,
        chemical_probe_trigger: pd.Series | None = None,
        chemical_probe_risk_gate: pd.Series | None = None,
        electronics_probe_eligible: bool = False,
        electronics_probe_trigger: pd.Series | None = None,
        electronics_probe_risk_gate: pd.Series | None = None,
        electronics_recovery_armed: pd.Series | None = None,
        electronics_rally_eligible: bool = False,
        electronics_rally_trigger: pd.Series | None = None,
        electronics_rally_risk_gate: pd.Series | None = None,
        semiconductor_probe_eligible: bool = False,
        semiconductor_probe_trigger: pd.Series | None = None,
        semiconductor_probe_risk_gate: pd.Series | None = None,
    ) -> pd.Series:
        if quality.status == GateStatus.REJECT:
            return pd.Series("NO_BUY", index=score.index)
        if quality.status == GateStatus.SECTOR_UNSUPPORTED:
            return pd.Series("NO_MODEL", index=score.index)
        if quality.status == GateStatus.UNKNOWN:
            return pd.Series("DATA_INSUFFICIENT", index=score.index)
        if quality.industry_profile_mode == IndustryProfileMode.SHADOW.value:
            blocked = vetoes.any(axis=1) | (market_data_gate != "PASS")
            return pd.Series(np.where(blocked, "NO_BUY", "WATCH_ONLY"), index=score.index)
        industry_profile = IndustryProfileRegistry.get(quality.industry_profile)
        vetoed = vetoes.any(axis=1)
        if quality.status == GateStatus.WATCH_QUALITY:
            if not self._industry_buy_small_quality_eligible(quality):
                return pd.Series("WATCH_ONLY", index=score.index)
            buy_small = (
                (score >= self.config.balanced_early_buy_threshold)
                & strict_trigger
                & (strict_risk_gate == "PASS")
                & (market_data_gate == "PASS")
                & ~vetoed
            )
            return pd.Series(
                np.where(buy_small, "BUY_SMALL", "WATCH_ONLY"),
                index=score.index,
            )
        if industry_profile is not None and not industry_profile.allow_buy:
            return pd.Series("WATCH_ONLY", index=score.index)

        if quality.status == GateStatus.PROVISIONAL:
            if not self._is_balanced_profile or not provisional_eligible:
                return pd.Series("WATCH_ONLY", index=score.index)
            buy_small = (
                (score >= self.config.provisional_buy_threshold)
                & strict_trigger
                & (strict_risk_gate == "PASS")
                & (market_data_gate == "PASS")
            )
            return pd.Series(
                np.select(
                    [market_data_gate != "PASS", vetoed, buy_small],
                    ["NO_BUY", "NO_BUY", "BUY_SMALL"],
                    default="WATCH_ONLY",
                ),
                index=score.index,
            )

        if self._is_balanced_profile:
            avoid_threshold = self.config.balanced_avoid_threshold
            armed_threshold = self.config.balanced_armed_threshold
            buy_threshold = self.config.balanced_buy_threshold
        else:
            avoid_threshold = 60.0
            armed_threshold = 70.0
            buy_threshold = 80.0
        if chemical_probe_trigger is None:
            chemical_probe_trigger = pd.Series(False, index=score.index)
        if chemical_probe_risk_gate is None:
            chemical_probe_risk_gate = pd.Series(
                "NOT_APPLICABLE",
                index=score.index,
            )
        if electronics_probe_trigger is None:
            electronics_probe_trigger = pd.Series(
                False,
                index=score.index,
            )
        if electronics_probe_risk_gate is None:
            electronics_probe_risk_gate = pd.Series(
                "NOT_APPLICABLE",
                index=score.index,
            )
        if electronics_recovery_armed is None:
            electronics_recovery_armed = pd.Series(
                False,
                index=score.index,
            )
        if electronics_rally_trigger is None:
            electronics_rally_trigger = pd.Series(
                False,
                index=score.index,
            )
        if electronics_rally_risk_gate is None:
            electronics_rally_risk_gate = pd.Series(
                "NOT_APPLICABLE",
                index=score.index,
            )
        if semiconductor_probe_trigger is None:
            semiconductor_probe_trigger = pd.Series(
                False,
                index=score.index,
            )
        if semiconductor_probe_risk_gate is None:
            semiconductor_probe_risk_gate = pd.Series(
                "NOT_APPLICABLE",
                index=score.index,
            )
        buy = (
            (score >= buy_threshold)
            & trigger
            & (risk_gate == "PASS")
            & (market_data_gate == "PASS")
        )
        chemical_probe_buy_small = (
            self._chemical_probe_quality_eligible(quality)
            & (
                score
                >= self.config.chemical_probe_score_threshold
            )
            & chemical_probe_trigger
            & (chemical_probe_risk_gate == "PASS")
            & (market_data_gate == "PASS")
            & ~vetoed
        )
        electronics_probe_buy_small = (
            electronics_probe_eligible
            & (
                score
                >= self.config.electronics_probe_score_threshold
            )
            & electronics_probe_trigger
            & (electronics_probe_risk_gate == "PASS")
            & (market_data_gate == "PASS")
            & ~vetoed
        )
        electronics_rally_buy_small = (
            electronics_rally_eligible
            & (score >= self.config.breadth_rally_score_threshold)
            & electronics_rally_trigger
            & (electronics_rally_risk_gate == "PASS")
            & (market_data_gate == "PASS")
            & ~vetoed
        )
        semiconductor_probe_buy_small = (
            semiconductor_probe_eligible
            & (
                score
                >= self.config.semiconductor_probe_score_threshold
            )
            & semiconductor_probe_trigger
            & (semiconductor_probe_risk_gate == "PASS")
            & (market_data_gate == "PASS")
            & ~vetoed
        )
        return pd.Series(
            np.select(
                [
                    market_data_gate != "PASS",
                    vetoed,
                    buy,
                    chemical_probe_buy_small,
                    electronics_probe_buy_small,
                    electronics_rally_buy_small,
                    semiconductor_probe_buy_small,
                    electronics_recovery_armed,
                    score < avoid_threshold,
                    score < armed_threshold,
                ],
                [
                    "NO_BUY",
                    "NO_BUY",
                    "BUY",
                    "BUY_SMALL",
                    "BUY_SMALL",
                    "BUY_SMALL",
                    "BUY_SMALL",
                    "ARMED",
                    "AVOID",
                    "WATCH",
                ],
                default="ARMED",
            ),
            index=score.index,
        )

    def _build_v41_reasons(
        self,
        *,
        quality: QualityGateResult,
        provisional_eligible: bool,
        components: pd.DataFrame,
        veto_text: pd.Series,
        technical_trigger: pd.Series,
        strict_trigger: pd.Series,
        trigger: pd.Series,
        liquidity_gate: pd.Series,
        risk_gate: pd.Series,
        strict_risk_gate: pd.Series,
        risk_reasons: pd.Series,
        chemical_probe_trigger: pd.Series,
        chemical_probe_liquidity_gate: pd.Series,
        chemical_probe_risk_gate: pd.Series,
        chemical_probe_risk_reasons: pd.Series,
        electronics_probe_eligible: bool,
        electronics_probe_trigger: pd.Series,
        electronics_probe_liquidity_gate: pd.Series,
        electronics_probe_risk_gate: pd.Series,
        electronics_probe_risk_reasons: pd.Series,
        electronics_recovery_armed: pd.Series,
        electronics_rally_trigger: pd.Series,
        electronics_rally_liquidity_gate: pd.Series,
        electronics_rally_risk_gate: pd.Series,
        electronics_rally_risk_reasons: pd.Series,
        broad_rally_regime: pd.Series,
        semiconductor_probe_eligible: bool,
        semiconductor_probe_trigger: pd.Series,
        semiconductor_probe_liquidity_gate: pd.Series,
        semiconductor_probe_risk_gate: pd.Series,
        semiconductor_probe_risk_reasons: pd.Series,
        state: pd.Series,
    ) -> pd.Series:
        evidence = "；".join(quality.reasons + quality.warnings)
        quality_text = "Quality={}(QS={}；DC={}；{})".format(
            quality.status.value,
            quality.score if quality.score is not None else "N/A",
            quality.data_confidence,
            evidence,
        )
        detail = pd.DataFrame(
            {
                **components,
                "veto": veto_text,
                "technical_trigger": technical_trigger,
                "strict_trigger": strict_trigger,
                "trigger": trigger,
                "liquidity": liquidity_gate,
                "risk": risk_gate,
                "strict_risk": strict_risk_gate,
                "risk_reasons": risk_reasons,
                "chemical_probe_trigger": chemical_probe_trigger,
                "chemical_probe_liquidity": chemical_probe_liquidity_gate,
                "chemical_probe_risk": chemical_probe_risk_gate,
                "chemical_probe_risk_reasons": chemical_probe_risk_reasons,
                "electronics_probe_trigger": electronics_probe_trigger,
                "electronics_probe_liquidity": (
                    electronics_probe_liquidity_gate
                ),
                "electronics_probe_risk": electronics_probe_risk_gate,
                "electronics_probe_risk_reasons": (
                    electronics_probe_risk_reasons
                ),
                "electronics_recovery_armed": electronics_recovery_armed,
                "broad_rally_regime": broad_rally_regime,
                "electronics_rally_trigger": electronics_rally_trigger,
                "electronics_rally_liquidity": (
                    electronics_rally_liquidity_gate
                ),
                "electronics_rally_risk": electronics_rally_risk_gate,
                "electronics_rally_risk_reasons": (
                    electronics_rally_risk_reasons
                ),
                "semiconductor_probe_trigger": semiconductor_probe_trigger,
                "semiconductor_probe_liquidity": (
                    semiconductor_probe_liquidity_gate
                ),
                "semiconductor_probe_risk": semiconductor_probe_risk_gate,
                "semiconductor_probe_risk_reasons": (
                    semiconductor_probe_risk_reasons
                ),
                "state": state,
            },
            index=components.index,
        )
        return detail.apply(
            lambda row: (
                "{}；分數={:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}；"
                "Profile={}；ProvisionalEligible={}；Veto={}；"
                "TechnicalTrigger={}；StrictTrigger={}；Trigger={}；"
                "Liquidity={}；Risk={}/{}({})；"
                "CHEMIProbe={}/{}/{}({})；"
                "ELECProbe={}/{}/{}/{}({})；"
                "ELECRecovery={}；BroadRally={}；"
                "ELECRally={}/{}/{}({})；"
                "SEMIProbe={}/{}/{}/{}({})；決策={}"
            ).format(
                quality_text,
                row["market_regime"],
                row["price_position"],
                row["stabilization"],
                row["demand_confirmation"],
                row["momentum"],
                row["execution_risk"],
                self.profile.value,
                "YES" if provisional_eligible else "NO",
                row["veto"],
                "YES" if row["technical_trigger"] else "NO",
                "YES" if row["strict_trigger"] else "NO",
                "YES" if row["trigger"] else "NO",
                row["liquidity"],
                row["risk"],
                row["strict_risk"],
                row["risk_reasons"],
                "YES" if row["chemical_probe_trigger"] else "NO",
                row["chemical_probe_liquidity"],
                row["chemical_probe_risk"],
                row["chemical_probe_risk_reasons"],
                "ELIGIBLE" if electronics_probe_eligible else "N/A",
                "YES" if row["electronics_probe_trigger"] else "NO",
                row["electronics_probe_liquidity"],
                row["electronics_probe_risk"],
                row["electronics_probe_risk_reasons"],
                "YES" if row["electronics_recovery_armed"] else "NO",
                "YES" if row["broad_rally_regime"] else "NO",
                "YES" if row["electronics_rally_trigger"] else "NO",
                row["electronics_rally_liquidity"],
                row["electronics_rally_risk"],
                row["electronics_rally_risk_reasons"],
                "ELIGIBLE" if semiconductor_probe_eligible else "N/A",
                "YES" if row["semiconductor_probe_trigger"] else "NO",
                row["semiconductor_probe_liquidity"],
                row["semiconductor_probe_risk"],
                row["semiconductor_probe_risk_reasons"],
                row["state"],
            ),
            axis=1,
        )

    @staticmethod
    def _enforce_executable_position(
        state: pd.Series,
        suggested_shares: pd.Series,
    ) -> pd.Series:
        """Do not publish a buy state when the sized odd-lot is zero shares."""

        non_executable_buy = (
            state.isin(["BUY", "BUY_SMALL"])
            & suggested_shares.fillna(0).lt(1)
        )
        return state.mask(non_executable_buy, "ARMED")

    @property
    def profile(self) -> LowEntryProfile:
        return LowEntryProfile(self.config.profile)

    @property
    def _is_balanced_profile(self) -> bool:
        return self.profile == LowEntryProfile.BALANCED

    @property
    def _standard_risk_thresholds(self) -> tuple[float, float]:
        if self._is_balanced_profile:
            return (
                self.config.balanced_maximum_stop_atr,
                self.config.balanced_minimum_reward_risk,
            )
        return (
            self.config.maximum_stop_atr,
            self.config.minimum_reward_risk,
        )

    def _buy_small_position_fraction(
        self,
        quality: QualityGateResult,
    ) -> float:
        profile = IndustryProfileRegistry.get(quality.industry_profile)
        if profile is not None and "21" in profile.sector_codes:
            return self.config.chemical_buy_small_initial_position_fraction
        return self.config.provisional_initial_position_fraction

    def _chemical_probe_quality_eligible(
        self,
        quality: QualityGateResult,
    ) -> bool:
        profile = IndustryProfileRegistry.get(quality.industry_profile)
        return bool(
            self._is_balanced_profile
            and profile is not None
            and "21" in profile.sector_codes
            and profile.mode is IndustryProfileMode.ACTIVE
            and quality.status == GateStatus.PASS
            and quality.score is not None
            and quality.score
            >= self.config.chemical_probe_minimum_quality_score
            and quality.data_confidence >= 70
            and not quality.hard_red_flags
        )

    def _semiconductor_probe_quality_eligible(
        self,
        quality: QualityGateResult,
        *,
        sector_code: str,
    ) -> bool:
        """Limit the relaxed volatility probe to TWSE sector code 24."""

        profile = IndustryProfileRegistry.get(quality.industry_profile)
        return bool(
            self._is_balanced_profile
            and sector_code == "24"
            and profile is not None
            and sector_code in profile.sector_codes
            and profile.mode is IndustryProfileMode.ACTIVE
            and quality.status == GateStatus.PASS
            and quality.score is not None
            and quality.score
            >= self.config.semiconductor_probe_minimum_quality_score
            and quality.data_confidence
            >= self.config.semiconductor_probe_minimum_data_confidence
            and not quality.hard_red_flags
        )

    def _electronics_probe_quality_eligible(
        self,
        quality: QualityGateResult,
        *,
        sector_code: str,
    ) -> bool:
        """Limit the stabilization probe to non-SEMI electronics sectors."""

        profile = IndustryProfileRegistry.get(quality.industry_profile)
        return bool(
            self._is_balanced_profile
            and sector_code in (ELECTRONICS_SECTOR_CODES - {"24"})
            and profile is not None
            and sector_code in profile.sector_codes
            and profile.mode is IndustryProfileMode.ACTIVE
            and quality.status == GateStatus.PASS
            and quality.score is not None
            and quality.score
            >= self.config.electronics_probe_minimum_quality_score
            and quality.data_confidence
            >= self.config.electronics_probe_minimum_data_confidence
            and not quality.hard_red_flags
        )

    def _electronics_recovery_quality_eligible(
        self,
        quality: QualityGateResult,
        *,
        sector_code: str,
    ) -> bool:
        """Expose a pre-trigger ARMED state without authorizing an order."""

        profile = IndustryProfileRegistry.get(quality.industry_profile)
        return bool(
            self._is_balanced_profile
            and sector_code in (ELECTRONICS_SECTOR_CODES - {"24"})
            and profile is not None
            and sector_code in profile.sector_codes
            and profile.mode is IndustryProfileMode.ACTIVE
            and quality.status == GateStatus.PASS
            and quality.score is not None
            and quality.score
            >= self.config.electronics_recovery_minimum_quality_score
            and quality.data_confidence
            >= self.config.electronics_recovery_minimum_data_confidence
            and not quality.hard_red_flags
        )

    def _electronics_rally_quality_eligible(
        self,
        quality: QualityGateResult,
        *,
        sector_code: str,
    ) -> bool:
        """Require full Quality PASS before applying the breadth-rally probe."""

        profile = IndustryProfileRegistry.get(quality.industry_profile)
        if sector_code == "24":
            minimum_quality = (
                self.config.semiconductor_probe_minimum_quality_score
            )
            minimum_confidence = (
                self.config.semiconductor_probe_minimum_data_confidence
            )
        else:
            minimum_quality = self.config.breadth_rally_minimum_quality_score
            minimum_confidence = (
                self.config.breadth_rally_minimum_data_confidence
            )
        return bool(
            self._is_balanced_profile
            and sector_code in ELECTRONICS_SECTOR_CODES
            and profile is not None
            and sector_code in profile.sector_codes
            and profile.mode is IndustryProfileMode.ACTIVE
            and quality.status == GateStatus.PASS
            and quality.score is not None
            and quality.score >= minimum_quality
            and quality.data_confidence >= minimum_confidence
            and not quality.hard_red_flags
        )

    @staticmethod
    def _industry_buy_small_quality_eligible(
        quality: QualityGateResult,
    ) -> bool:
        profile = IndustryProfileRegistry.get(quality.industry_profile)
        return bool(
            profile is not None
            and "21" in profile.sector_codes
            and profile.mode is IndustryProfileMode.ACTIVE
            and profile.allow_buy_small
            and quality.status == GateStatus.WATCH_QUALITY
            and quality.score is not None
            and quality.score >= profile.buy_small_quality_score
            and quality.data_confidence
            >= profile.buy_small_minimum_data_confidence
            and not quality.hard_red_flags
        )

    @property
    def _liquidity_thresholds(self) -> tuple[float, float]:
        if self._is_balanced_profile:
            return (
                self.config.balanced_minimum_amount_ma20,
                self.config.balanced_maximum_spread,
            )
        return (
            self.config.minimum_amount_ma20,
            self.config.maximum_spread,
        )

    def _quality_domain_coverage(
        self,
        quality: QualityGateResult,
    ) -> dict[str, float | bool]:
        profitability = self._metric_category_coverage(quality, "A")
        balance_sheet_safety = self._metric_category_coverage(quality, "C")
        no_negative_cfo = self._no_confirmed_negative_cfo(quality)
        return {
            "profitability": profitability,
            "balance_sheet_safety": balance_sheet_safety,
            "required_passed": (
                profitability >= self.config.required_domain_coverage
                and balance_sheet_safety
                >= self.config.required_domain_coverage
            ),
            "no_confirmed_negative_cfo": no_negative_cfo,
        }

    @staticmethod
    def _metric_category_coverage(
        quality: QualityGateResult,
        category: str,
    ) -> float:
        metrics = [
            metric
            for metric in quality.metric_scores
            if metric.category == category
        ]
        total_weight = sum(metric.maximum for metric in metrics)
        if total_weight <= 0:
            return 0.0
        available_weight = sum(
            metric.maximum for metric in metrics if metric.available
        )
        return round(100.0 * available_weight / total_weight, 2)

    @staticmethod
    def _no_confirmed_negative_cfo(quality: QualityGateResult) -> bool:
        values = [
            quality.metrics.get("cfo_ttm"),
            quality.metrics.get("cumulative_cfo_3y"),
        ]
        known_values: list[float] = []
        for value in values:
            parsed = pd.to_numeric(value, errors="coerce")
            if pd.notna(parsed):
                known_values.append(float(parsed))
        return not any(value <= 0 for value in known_values)

    def _provisional_buy_eligible(
        self,
        quality: QualityGateResult,
        domain_coverage: dict[str, float | bool],
    ) -> bool:
        return (
            self._is_balanced_profile
            and quality.status == GateStatus.PROVISIONAL
            and quality.score is not None
            and quality.score
            >= self.config.provisional_minimum_quality_score
            and quality.data_confidence
            >= self.config.provisional_minimum_data_confidence
            and bool(domain_coverage["required_passed"])
            and bool(domain_coverage["no_confirmed_negative_cfo"])
            and not quality.hard_red_flags
        )

    def _restriction_status(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        names = (
            "Trading Status",
            "交易狀態",
            "Market Restriction",
            "處置狀態",
        )
        column = next((name for name in names if name in data.columns), None)
        if column is None:
            unknown = pd.Series(False, index=data.index)
            return unknown, unknown

        status = data[column].astype("string").fillna("").str.strip().str.upper()
        known = ~status.isin({"", "N/A", "NA", "NONE", "UNKNOWN", "<NA>"})
        restricted = status.str.contains(
            "處置|變更交易|停止交易|HALT|SUSPEND|DISPOSITION|ALTERED",
            regex=True,
            na=False,
        )
        return known, known & ~restricted

    def _format_v41_vetoes(self, row: pd.Series) -> str:
        labels = [_VETO_LABELS[name] for name in row.index[row.astype(bool)]]
        return "；".join(labels) if labels else "NONE"

    def _empty_v41_result(self) -> dict[str, object]:
        maximum_stop_atr, minimum_reward_risk = self._standard_risk_thresholds
        result = {column: np.nan for column in LOW_ENTRY_OUTPUT_COLUMNS}
        result.update(
            {
                "低點分數": 0,
                "低點決策": "DATA_INSUFFICIENT",
                "Quality Gate": GateStatus.UNKNOWN.value,
                "Industry Quality Profile": "UNSUPPORTED",
                "Industry Profile Version": "N/A",
                "Industry Profile Mode": "UNSUPPORTED",
                "Data Confidence": 0,
                "Available Metric Weight": 0,
                "Quality Reasons": "基本面與日線資料不足",
                "Market Data Gate": "FAIL",
                "Market Data Reasons": "DATA_INSUFFICIENT",
                "Market Session Final": 0,
                "Broad Rally Regime": 0,
                "No Veto": 0,
                "Falling Knife Veto": "N/A",
                "LES v4.1 Technical Trigger": 0,
                "LES v4.1 Strict Trigger": 0,
                "LES v4.1 Trigger": 0,
                "CHEMI Probe Trigger": 0,
                "ELEC Probe Trigger": 0,
                "ELEC Recovery Armed": 0,
                "ELEC Rally Trigger": 0,
                "SEMI Probe Trigger": 0,
                "LES Profile": self.profile.value,
                "Provisional Buy Eligible": 0,
                "Required Domain Coverage Passed": 0,
                "No Confirmed Negative CFO Evidence": 0,
                "Liquidity Gate": "UNKNOWN",
                "CHEMI Probe Liquidity Gate": "NOT_APPLICABLE",
                "ELEC Probe Liquidity Gate": "NOT_APPLICABLE",
                "ELEC Rally Liquidity Gate": "NOT_APPLICABLE",
                "SEMI Probe Liquidity Gate": "NOT_APPLICABLE",
                "Risk Gate": "UNKNOWN",
                "Strict Risk Gate": "UNKNOWN",
                "CHEMI Probe Risk Gate": "NOT_APPLICABLE",
                "ELEC Probe Risk Gate": "NOT_APPLICABLE",
                "ELEC Rally Risk Gate": "NOT_APPLICABLE",
                "SEMI Probe Risk Gate": "NOT_APPLICABLE",
                "Risk Gate Reasons": "DATA_INSUFFICIENT",
                "CHEMI Probe Risk Reasons": "NOT_APPLICABLE",
                "ELEC Probe Risk Reasons": "NOT_APPLICABLE",
                "ELEC Rally Risk Reasons": "NOT_APPLICABLE",
                "SEMI Probe Risk Reasons": "NOT_APPLICABLE",
                "Position Fraction": 0,
                "Effective Risk Percent": 0,
                "訊號有效交易日": 0,
                "LES Model Version": LES_V41_MODEL_VERSION,
                "LES Parameter Version": LES_V41_PARAMETER_VERSION,
                "Risk Percent": self.config.risk_percent,
                "Maximum Stop ATR": maximum_stop_atr,
                "Minimum RewardRisk": minimum_reward_risk,
                "低點理由": "資料不足；決策=DATA_INSUFFICIENT",
            }
        )
        return result


__all__ = [
    "ACCOUNT_EQUITY_ATTR",
    "ANNUAL_FUNDAMENTALS_ATTR",
    "FUNDAMENTAL_SNAPSHOT_ATTR",
    "FUNDAMENTAL_STATUS_ATTR",
    "FUNDAMENTAL_STATUS_DETAIL_ATTR",
    "LES_V41_MODEL_VERSION",
    "LES_V41_PARAMETER_VERSION",
    "LOW_ENTRY_OUTPUT_COLUMNS",
    "LOW_ENTRY_STRATEGY_NAME",
    "LowEntryProfile",
    "LowEntryScoreV4Config",
    "LowEntryScoreV41Config",
    "LowEntryScoreV41Strategy",
    "PRICE_SCALE_ATTR",
    "SIGNAL_DATE_ATTR",
    "attach_low_entry_v4_context",
]
