"""Result-frame and reason rendering for Low Entry Model v5."""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.quality_v5 import FinalQualityStatus
from domain.quality_v5 import LowEntryV5QualityResult
from domain.strategies.low_entry_v5_contracts import BroadRallyTier
from domain.strategies.low_entry_v5_contracts import FallingKnifeState
from domain.strategies.low_entry_v5_contracts import LES_V5_MODEL_VERSION
from domain.strategies.low_entry_v5_contracts import LES_V5_PARAMETER_VERSION
from domain.strategies.low_entry_v5_contracts import ValuationStatus
from domain.valuation_v5 import LowEntryV5ValuationResult
from domain.valuation_v5 import ValuationProductionStatus


class LowEntryV5ReportBuilder:
    """Map typed LES v5 decisions to the stable legacy DataFrame contract."""

    def __init__(self, output_columns: list[str]) -> None:
        self.output_columns = tuple(output_columns)

    def assemble_result(
        self,
        *,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        market_gate: pd.Series,
        market_reasons: pd.Series,
        plan: pd.DataFrame,
        quality: LowEntryV5QualityResult,
        valuation: pd.Series,
        valuation_result: LowEntryV5ValuationResult,
        panic: dict[str, pd.Series],
        components: pd.DataFrame,
        penalty: pd.Series,
        risk_off: pd.Series,
        required_score: pd.Series,
        trigger: pd.Series,
        risk_gate: pd.Series,
        risk_reasons: pd.Series,
        broad_rally_risk_gate: pd.Series,
        broad_rally_risk_reasons: pd.Series,
        next_day_confirmation: pd.Series,
        decision: pd.Series,
        position_fraction: pd.Series,
        recommended_shares: pd.Series,
    ) -> pd.DataFrame:
        frame = data.copy()
        score = (components.sum(axis=1) + penalty).clip(0, 100)
        hard_rejects = "；".join(quality.hard_rejects) or "NONE"
        quality_warnings = "；".join(quality.warnings) or "NONE"
        valuation_confidence = self.valuation_confidence(quality, valuation_result)
        cycle_review = valuation_result.cycle_review
        result = pd.DataFrame(
            {
                "低點分數": score.round(2),
                "低點決策": decision,
                "LES v5 Final Quality Status": quality.final_status.value,
                "LES v5 Final Quality Score": quality.final_score,
                "Base Quality Status": quality.base_status.value,
                "Base Quality Score": quality.base_score,
                "Base Data Confidence": quality.base_data_confidence,
                "Base Available Weight": quality.base_available_weight,
                "Extended Quality Status": quality.extended_status.value,
                "Extended Quality Score": quality.extended_score,
                "Extended Data Confidence": quality.extended_data_confidence,
                "Extended Available Weight": quality.extended_available_weight,
                "Critical Field Coverage": quality.critical_field_coverage,
                "Effective Extended Weight": quality.effective_extended_weight,
                "Quality Hard Rejects": hard_rejects,
                "Quality Warnings": quality_warnings,
                "Valuation Gate": valuation,
                "Valuation Model Status": valuation_result.status.value,
                "Valuation Model Version": valuation_result.model_version,
                "Valuation Reasons": "；".join(valuation_result.reasons) or "NONE",
                "Valuation Confidence": valuation_confidence,
                "Valuation Normalized EPS": self.scenario_value(
                    valuation_result, "base", "normalized_eps"
                ),
                "Valuation TTM EPS": self.cycle_review_value(
                    cycle_review, "ttm_eps"
                ),
                "Normalized EPS Divergence Ratio": self.cycle_review_value(
                    cycle_review, "eps_divergence_ratio"
                ),
                "Cycle Valuation Review": (
                    cycle_review.status.value
                    if cycle_review is not None
                    else "UNKNOWN"
                ),
                "Cycle Valuation Reasons": (
                    ("；".join(cycle_review.reasons) or "NONE")
                    if cycle_review is not None
                    else "NONE"
                ),
                "Cycle Mid-Cycle Operating Margin": self.cycle_review_value(
                    cycle_review, "mid_cycle_operating_margin"
                ),
                "Cycle Normalized ROE": self.cycle_review_value(
                    cycle_review, "normalized_roe"
                ),
                "Cycle Book Value Per Share": self.cycle_review_value(
                    cycle_review, "book_value_per_share"
                ),
                "Cycle Margin Fair Value": self.cycle_review_value(
                    cycle_review, "margin_fair_value"
                ),
                "Cycle PB Fair Value": self.cycle_review_value(
                    cycle_review, "pb_fair_value"
                ),
                "Cycle TTM EPS Fair Value": self.cycle_review_value(
                    cycle_review, "ttm_eps_fair_value"
                ),
                "Cycle Composite Fair Value": self.cycle_review_value(
                    cycle_review, "composite_fair_value"
                ),
                "Cycle Available Anchors": self.cycle_review_value(
                    cycle_review, "available_anchor_count"
                ),
                "Cycle Supporting Anchors": self.cycle_review_value(
                    cycle_review, "supporting_anchor_count"
                ),
                "Passive Benchmark Expected Return": (
                    valuation_result.passive_benchmark_expected_return
                ),
                "Required Valuation Return": valuation_result.required_return,
                "Base Case Revenue Growth": self.scenario_value(
                    valuation_result, "base", "revenue_growth"
                ),
                "Base Case Operating Margin": self.scenario_value(
                    valuation_result, "base", "operating_margin"
                ),
                "Base Case Exit PE": self.scenario_value(
                    valuation_result, "base", "exit_pe"
                ),
                "Base Case Expected CAGR": self.scenario_value(
                    valuation_result, "base", "expected_cagr"
                ),
                "Base Case Fair Value": self.scenario_value(
                    valuation_result, "base", "fair_value"
                ),
                "Bear Case Expected CAGR": self.scenario_value(
                    valuation_result, "bear", "expected_cagr"
                ),
                "Bear Case Fair Value": self.scenario_value(
                    valuation_result, "bear", "fair_value"
                ),
                "Bear Case Permanent Loss": valuation_result.bear_permanent_loss,
                "Bear Case Permanent Loss Acceptable": (
                    valuation_result.bear_permanent_loss_acceptable
                ),
                "Bull Case Expected CAGR": self.scenario_value(
                    valuation_result, "bull", "expected_cagr"
                ),
                "Bull Case Fair Value": self.scenario_value(
                    valuation_result, "bull", "fair_value"
                ),
                "Panic Opportunity Zone": panic["opportunity"].astype(int),
                "Falling Knife State": panic["state"],
                "Temporary Falling Knife Blocks": panic["block_text"],
                "Panic Condition Count": panic["panic_count"],
                "Reversal Confirmation Count": panic["reversal_count"],
                "Capitulation Reversal": panic["capitulation"].astype(int),
                "Breadth As Of": indicators["breadth_as_of"],
                "Breadth Universe": indicators["breadth_universe"],
                "Breadth Coverage": indicators["breadth_coverage"],
                "Breadth Valid Members": indicators["breadth_valid_members"],
                "Sector Advancer Ratio": indicators["sector_advancer_ratio"],
                "Sector Median R1": indicators["sector_median_r1"],
                "TAIEX R1": indicators["taiex_r1"],
                "Broad Rally Tier": indicators["broad_rally_tier"],
                "Broad Rally Regime": indicators["broad_rally_regime"].astype(int),
                "Broad Rally Trigger": panic["broad_rally_trigger"].astype(int),
                "Broad Rally Full BUY Trigger": panic[
                    "broad_rally_full_buy_trigger"
                ].astype(int),
                "Broad Rally Absorption Trigger": panic[
                    "broad_rally_absorption_trigger"
                ].astype(int),
                "Supportive Non-Broad Trigger": (
                    (decision == "BUY_SMALL")
                    & ~panic["broad_rally_active"].fillna(False)
                    & (valuation == ValuationStatus.SUPPORTIVE.value)
                ).astype(int),
                "Quality Pullback Starter Trigger": panic[
                    "quality_pullback_trigger"
                ].astype(int),
                "Recent Selloff Sessions": indicators["recent_selloff_sessions"],
                "Recent Broad Selloff Regime": indicators[
                    "recent_broad_selloff_regime"
                ].astype(int),
                "Multi-Day Panic Reversal Trigger": panic[
                    "multiday_panic_reversal_trigger"
                ].astype(int),
                "Recent Rally Sessions": indicators["recent_rally_sessions"],
                "Recent Broad Rally Regime": indicators[
                    "recent_broad_rally_regime"
                ].astype(int),
                "Recent Rally Retest Trigger": panic[
                    "recent_rally_retest_trigger"
                ].astype(int),
                "Recent Rally Reclaim Trigger": panic[
                    "recent_rally_reclaim_trigger"
                ].astype(int),
                "Broad Rally Risk Gate": broad_rally_risk_gate,
                "Broad Rally Risk Reasons": broad_rally_risk_reasons,
                "Next-Day Confirmation": next_day_confirmation.astype(int),
                "Market Data Gate": market_gate,
                "Market Data Reasons": market_reasons,
                "市場環境分數": components["market_regime"],
                "價格位置分數": components["price_position"],
                "止跌結構分數": components["stabilization"],
                "買盤確認分數": components["demand_confirmation"],
                "動能分數": components["momentum_confirmation"],
                "執行風控分數": components["execution_risk"],
                "Falling Knife Penalty": penalty,
                "Required Technical Score": required_score,
                "TAIEX Risk-Off": risk_off.astype(int),
                "LES v5 Trigger": trigger.astype(int),
                "Risk Gate": risk_gate,
                "Risk Gate Reasons": risk_reasons,
                "Position Fraction": position_fraction,
                "60日位置": indicators["price_position60"],
                "Drawdown60": indicators["drawdown60"],
                "Drawdown20": indicators["drawdown20"],
                "RS20": indicators["rs20"],
                "RS20 Slope": indicators["rs20_slope"],
                "EMA20": indicators["ema20"],
                "EMA20 Slope": indicators["ema20_slope"],
                "ATR14": indicators["atr14"],
                "ATR Percent": indicators["atr_percent"],
                "CLV": indicators["clv"],
                "CMF20": indicators["cmf20"],
                "Volume Ratio": indicators["volume_ratio"],
                "RSI14": indicators["rsi14"],
                "MACD Histogram": indicators["macd_histogram"],
                "RewardRisk": plan["reward_risk"],
                "Entry Candidate": plan["entry"],
                "Entry Rule": pd.Series(
                    np.select(
                        [
                            (decision == "BUY")
                            & panic["broad_rally_full_buy_trigger"],
                            decision == "EARLY_BUY",
                            decision == "BUY_SMALL",
                            decision == "BUY",
                        ],
                        [
                            "NEXT_SESSION_BREAKS_SIGNAL_HIGH",
                            "NEXT_DAY_REVERSAL_CONFIRMATION",
                            "NEXT_SESSION_BREAKS_SIGNAL_HIGH",
                            "CLOSE_CONFIRMED",
                        ],
                        default="NONE",
                    ),
                    index=frame.index,
                ),
                "策略停損": plan["stop"],
                "策略停利": plan["target"],
                "建議股數": recommended_shares,
                "LES Model Version": LES_V5_MODEL_VERSION,
                "LES Parameter Version": LES_V5_PARAMETER_VERSION,
                "低點理由": self.build_reasons(
                    quality,
                    valuation,
                    components,
                    penalty,
                    panic,
                    trigger,
                    risk_gate,
                    risk_reasons,
                    broad_rally_risk_gate,
                    required_score,
                    decision,
                    valuation_result,
                ),
            },
            index=frame.index,
        )
        return pd.concat([frame, result], axis=1)

    @staticmethod
    def scenario_value(
        result: LowEntryV5ValuationResult,
        scenario_name: str,
        field_name: str,
    ) -> object:
        scenario = getattr(result, scenario_name)
        return getattr(scenario, field_name) if scenario is not None else np.nan

    @staticmethod
    def cycle_review_value(review: object, field_name: str) -> object:
        return getattr(review, field_name) if review is not None else np.nan

    @staticmethod
    def valuation_confidence(
        quality: LowEntryV5QualityResult,
        result: LowEntryV5ValuationResult,
    ) -> float:
        if result.status is not ValuationProductionStatus.COMPLETE:
            return 0.0
        return float(
            min(quality.base_data_confidence, quality.extended_data_confidence)
        )

    def build_reasons(
        self,
        quality: LowEntryV5QualityResult,
        valuation: pd.Series,
        components: pd.DataFrame,
        penalty: pd.Series,
        panic: dict[str, pd.Series],
        trigger: pd.Series,
        risk_gate: pd.Series,
        risk_reasons: pd.Series,
        broad_rally_risk_gate: pd.Series,
        required_score: pd.Series,
        decision: pd.Series,
        valuation_result: LowEntryV5ValuationResult,
    ) -> pd.Series:
        detail = components.copy()
        detail["penalty"] = penalty
        detail["valuation"] = valuation
        detail["falling"] = panic["state"]
        detail["blocks"] = panic["block_text"]
        detail["panic"] = panic["panic_count"]
        detail["reversal"] = panic["reversal_count"]
        detail["trigger"] = trigger
        detail["risk"] = risk_gate
        detail["risk_reasons"] = risk_reasons
        detail["broad_risk"] = broad_rally_risk_gate
        detail["broad_full"] = panic["broad_rally_full_buy_trigger"]
        detail["absorption"] = panic["broad_rally_absorption_trigger"]
        detail["multiday"] = panic["multiday_panic_reversal_trigger"]
        detail["retest"] = panic["recent_rally_retest_trigger"]
        detail["reclaim"] = panic["recent_rally_reclaim_trigger"]
        detail["quality_pullback"] = panic["quality_pullback_trigger"]
        detail["high_quality_reversal"] = panic[
            "high_quality_reversal_trigger"
        ]
        detail["air_cycle_watch"] = panic["air_cycle_watch"]
        detail["supportive_non_broad"] = (
            (decision == "BUY_SMALL")
            & ~panic["broad_rally_active"].fillna(False)
            & (valuation == ValuationStatus.SUPPORTIVE.value)
        )
        detail["required"] = required_score
        detail["decision"] = decision
        quality_text = (
            "Quality={}(Final={}；Base={}/{}；Extended={}/{}；Critical={})"
        ).format(
            quality.final_status.value,
            quality.final_score if quality.final_score is not None else "N/A",
            quality.base_score if quality.base_score is not None else "N/A",
            quality.base_data_confidence,
            quality.extended_score if quality.extended_score is not None else "N/A",
            quality.extended_data_confidence,
            quality.critical_field_coverage,
        )
        valuation_reasons = "、".join(valuation_result.reasons) or "NONE"
        reasons = detail.apply(
            lambda row: (
                "{}；Valuation={}({})；分數={:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}"
                "；Penalty={:.0f}；Falling={}({})；Panic/Reversal={:.0f}/{:.0f}"
                "；Trigger={}；Risk={}({})；BroadRally={}/{}/Full={}/Absorb={}/MultiDay={}/Retest={}/Reclaim={}"
                "；SupportiveNonBroad={}；QualityPullback={}；HighQualityReversal={}；AirCycleWatch={}；Required={:.0f}；決策={}"
            ).format(
                quality_text,
                row["valuation"],
                valuation_reasons,
                row["market_regime"],
                row["price_position"],
                row["stabilization"],
                row["demand_confirmation"],
                row["momentum_confirmation"],
                row["execution_risk"],
                row["penalty"],
                row["falling"],
                row["blocks"],
                row["panic"],
                row["reversal"],
                "YES" if row["trigger"] else "NO",
                row["risk"],
                row["risk_reasons"],
                "YES" if panic["broad_rally_trigger"].loc[row.name] else "NO",
                row["broad_risk"],
                "YES" if row["broad_full"] else "NO",
                "YES" if row["absorption"] else "NO",
                "YES" if row["multiday"] else "NO",
                "YES" if row["retest"] else "NO",
                "YES" if row["reclaim"] else "NO",
                "YES" if row["supportive_non_broad"] else "NO",
                "YES" if row["quality_pullback"] else "NO",
                "YES" if row["high_quality_reversal"] else "NO",
                "YES" if row["air_cycle_watch"] else "NO",
                row["required"],
                row["decision"],
            ),
            axis=1,
        )
        elec_trigger = panic.get(
            "elec_trend_pullback_trigger", pd.Series(False, index=detail.index)
        ).fillna(False)
        if not elec_trigger.any():
            return reasons
        elec_risk = panic.get(
            "elec_repriced_risk_gate", pd.Series("UNKNOWN", index=detail.index)
        )
        reasons.loc[elec_trigger] += (
            "；ELECTrendPullback=YES；ELECRepricedRisk=" + elec_risk.loc[elec_trigger]
        )
        for key, label in (
            ("elec_starter_required_score", "ELECStarterRequired"),
            ("elec_starter_required_raw_score", "ELECStarterRawRequired"),
        ):
            if key in panic:
                reasons.loc[elec_trigger] += (
                    "；" + label + "="
                    + panic[key].loc[elec_trigger].map(lambda value: f"{value:g}")
                )
        return reasons

    def empty_result(self) -> dict[str, object]:
        result = {column: np.nan for column in self.output_columns}
        result.update(
            {
                "低點分數": 0,
                "低點決策": "DATA_INSUFFICIENT",
                "LES v5 Final Quality Status": (
                    FinalQualityStatus.DATA_INSUFFICIENT.value
                ),
                "Base Quality Status": "BASE_UNKNOWN",
                "Base Data Confidence": 0,
                "Extended Quality Status": "EXTENDED_UNKNOWN",
                "Extended Data Confidence": 0,
                "Broad Rally Tier": BroadRallyTier.NONE.value,
                "Falling Knife State": FallingKnifeState.DATA_INSUFFICIENT.value,
                "Market Data Gate": "FAIL",
                "Market Data Reasons": "DATA_INSUFFICIENT",
                "LES Model Version": LES_V5_MODEL_VERSION,
                "LES Parameter Version": LES_V5_PARAMETER_VERSION,
                "低點理由": "資料不足；決策=DATA_INSUFFICIENT",
            }
        )
        return result


__all__ = ["LowEntryV5ReportBuilder"]
