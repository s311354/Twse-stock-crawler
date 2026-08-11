"""Low Entry Model v5 quality-first panic reversal strategy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from domain.features import FeatureFrame
from domain.features import SharedFeatureConfig
from domain.market_breadth import MARKET_BREADTH_ATTR
from domain.market_breadth import MARKET_BREADTH_HISTORY_ATTR
from domain.market_breadth import MARKET_DATA_THROUGH_ATTR
from domain.market_breadth import MARKET_SESSION_FINAL_ATTR
from domain.market_breadth import normalize_market_breadth_snapshot
from domain.quality_v5 import BaseQualityStatus
from domain.quality_v5 import ExtendedQualityStatus
from domain.quality_v5 import FinalQualityStatus
from domain.quality_v5 import LowEntryV5QualityConfig
from domain.quality_v5 import LowEntryV5QualityResult
from domain.quality_v5 import evaluate_v5_quality
from domain.strategy import Strategy
from domain.strategies.low_entry_score_v4 import ACCOUNT_EQUITY_ATTR
from domain.strategies.low_entry_score_v4 import ANNUAL_FUNDAMENTALS_ATTR
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_SNAPSHOT_ATTR
from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from domain.strategies.low_entry_score_v4 import LowEntryScoreV4Config
from domain.strategies.low_entry_score_v4 import LowEntryScoreV4Strategy
from domain.strategies.low_entry_v5_pipeline import LowEntryV5DecisionPipeline
from domain.strategies.low_entry_v5_pipeline import LowEntryV5PipelineComponents
from domain.valuation_v5 import LowEntryV5ValuationConfig
from domain.valuation_v5 import LowEntryV5ValuationProducer
from domain.valuation_v5 import LowEntryV5ValuationResult
from domain.valuation_v5 import ValuationProductionStatus


LOW_ENTRY_STRATEGY_NAME = "low_entry_score_v5"
LES_V5_MODEL_VERSION = "5.0.0"
LES_V5_PARAMETER_VERSION = (
    "quality-first-panic-reversal-2026-08-09-v16-elec-confirmed-reclaim"
)


class FallingKnifeState(str, Enum):
    HARD_NO_BUY = "HARD_NO_BUY"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    PANIC_WATCH = "PANIC_WATCH"
    REVERSAL_WATCH = "REVERSAL_WATCH"
    CLEAR = "CLEAR"


class ValuationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LowEntryScoreV5Config(LowEntryScoreV4Config):
    # TWSE supports odd-lot execution.  Requiring one full 1,000-share lot made
    # risk-sized v5 positions appear non-executable even when the account could
    # safely buy a smaller number of shares.
    lot_size: int = 1
    # Balanced profile: a formal entry still requires every non-score gate,
    # but the score and execution limits no longer duplicate the strict
    # profile inherited from v4.
    maximum_stop_atr: float = 1.8
    minimum_reward_risk: float = 1.7
    confirmed_buy_score: float = 75.0
    provisional_buy_score: float = 80.0
    risk_off_score_increment: float = 5.0
    confirmed_position_fraction: float = 1 / 3
    provisional_position_fraction: float = 1 / 4
    early_position_fraction: float = 1 / 4
    breadth_rally_position_fraction: float = 1 / 8
    breadth_rally_score_threshold: float = 30.0
    breadth_rally_minimum_raw_score: float = 40.0
    # A completed breadth reversal may graduate from the tiny starter to a
    # conditional BUY when both the adjusted and pre-penalty scores show more
    # than a one-bar bounce. This path is capped at 1/4 and still requires a
    # next-session break of the signal-day high.
    breadth_rally_full_buy_position_fraction: float = 1 / 4
    breadth_rally_full_buy_score_threshold: float = 35.0
    breadth_rally_full_buy_minimum_raw_score: float = 45.0
    breadth_rally_full_buy_minimum_quality_score: float = 80.0
    breadth_rally_full_buy_minimum_clv: float = 0.60
    breadth_rally_full_buy_minimum_cmf20: float = -0.20
    breadth_rally_full_buy_minimum_panics: int = 1
    breadth_rally_full_buy_minimum_reversals: int = 4
    # Deeply depressed stocks can show a valid reversal bar before the
    # backward-looking CMF20 turns positive. This separate absorption starter
    # is intentionally tiny and requires two panic observations, three
    # reversal confirmations, both risk gates and next-session confirmation.
    breadth_absorption_score_threshold: float = 25.0
    breadth_absorption_minimum_raw_score: float = 35.0
    breadth_absorption_minimum_quality_score: float = 80.0
    breadth_absorption_minimum_panics: int = 2
    breadth_absorption_minimum_reversals: int = 3
    breadth_absorption_minimum_clv: float = 0.30
    breadth_absorption_minimum_cmf20: float = -0.45
    breadth_absorption_minimum_volume_ratio: float = 0.40
    breadth_absorption_maximum_price_position: float = 0.45
    breadth_absorption_maximum_drawdown: float = -0.15
    breadth_rally_minimum_quality_score: float = 75.0
    breadth_rally_minimum_base_confidence: float = 75.0
    # MOPS has enough direct balance-sheet fields to provide 53 points of Base
    # coverage even when the same-quarter TWSE batch snapshot is unavailable.
    # This floor deliberately stays below the normal 70-point Base PASS floor;
    # by itself it authorizes only a conditional 1/8 BUY_SMALL. Graduation to
    # breadth BUY requires the separate stronger quality and reversal gates.
    breadth_rally_minimum_fallback_base_weight: float = 50.0
    breadth_rally_minimum_extended_confidence: float = 70.0
    breadth_rally_minimum_critical_coverage: float = 70.0
    breadth_rally_maximum_stop_atr: float = 2.0
    # Only the conditional 1/8 breadth starter may use the 1.6 floor.  Normal
    # BUY and breadth full-BUY still pass the ordinary Risk Gate at RR >= 1.7.
    # This admits borderline but still asymmetric recoveries such as the
    # 2026-08-03 ELEC setup with repriced RR 1.66.
    breadth_rally_minimum_reward_risk: float = 1.6
    # A breadth entry is a tactical starter whose thesis is invalidated when
    # the signal/retest bar fails.  Using only the older confirmed swing low
    # made the stop unnecessarily wide after a fast ELEC recovery.  The
    # breadth-only plan therefore uses the tighter of the structural stop and
    # the signal-bar low less this ATR buffer.  Ordinary BUYs keep the original
    # structural stop.
    breadth_rally_signal_stop_buffer_atr: float = 0.30
    # The conditional breadth starter is capped at 1/8 and can therefore use
    # a slightly lower absolute turnover floor than an ordinary/full BUY.
    # The order-size ratio and spread limits are unchanged, so a smaller stock
    # must still be executable without dominating its normal daily turnover.
    breadth_rally_minimum_amount_ma20: float = 25_000_000.0
    breadth_rally_minimum_coverage: float = 0.80
    electronics_breadth_minimum_members: int = 100
    # ELEC contains hundreds of constituents.  Requiring 80% advancers missed
    # the 2026-08-03 through 08-05 recovery (roughly 75%-78% advancers) even
    # though the median member gained 1.7%-2.2%.  A 75% participation floor,
    # paired with the unchanged 1.5% median-return floor, still represents a
    # genuine broad move rather than a few cap-weighted leaders.
    electronics_breadth_minimum_advancer_ratio: float = 0.75
    electronics_breadth_minimum_median_return: float = 0.015
    semiconductor_breadth_minimum_members: int = 40
    semiconductor_breadth_minimum_advancer_ratio: float = 0.85
    semiconductor_breadth_minimum_median_return: float = 0.020
    breadth_rally_minimum_taiex_return: float = 0.005
    breadth_rally_minimum_clv: float = 0.35
    breadth_rally_minimum_volume_ratio: float = 0.35
    breadth_rally_strong_close_minimum_clv: float = 0.60
    breadth_rally_strong_close_minimum_volume_ratio: float = 0.30
    breadth_rally_strong_close_minimum_reversals: int = 4
    breadth_rally_minimum_cmf20: float = -0.25
    breadth_rally_maximum_price_position: float = 0.45
    breadth_rally_maximum_drawdown: float = -0.05
    # Multi-session regimes use completed snapshots before the signal session.
    # A regime only unlocks a small conditional path; it never bypasses
    # Quality, Valuation, Falling-Knife or execution-risk gates.
    # Retain the first session of a three-day breadth surge while the stock is
    # completing an orderly retest after that surge.
    breadth_memory_lookback_sessions: int = 4
    breadth_memory_minimum_sessions: int = 2
    panic_selloff_maximum_advancer_ratio: float = 0.20
    panic_selloff_maximum_median_return: float = -0.02
    panic_selloff_maximum_taiex_return: float = -0.005
    recent_rally_minimum_advancer_ratio: float = 0.70
    recent_rally_minimum_median_return: float = 0.01
    # A cap-weighted TAIEX can stay slightly negative while most SEMI/ELEC
    # members advance. The stronger breadth-led override below prevents a few
    # heavyweight constituents from hiding broad participation.
    recent_rally_minimum_taiex_return: float = -0.005
    recent_rally_lead_minimum_advancer_ratio: float = 0.80
    recent_rally_lead_minimum_median_return: float = 0.015
    multiday_panic_score_threshold: float = 25.0
    multiday_panic_minimum_raw_score: float = 35.0
    multiday_panic_minimum_reversals: int = 3
    multiday_panic_minimum_clv: float = 0.35
    multiday_panic_minimum_cmf20: float = -0.45
    multiday_panic_minimum_volume_ratio: float = 0.40
    multiday_panic_maximum_price_position: float = 0.45
    multiday_panic_maximum_drawdown: float = -0.08
    rally_retest_score_threshold: float = 25.0
    rally_retest_minimum_raw_score: float = 35.0
    rally_retest_minimum_reversals: int = 1
    # A green or support-holding retest may close modestly below the midpoint
    # of its intraday range after a broad ELEC advance.  -0.30 admits that
    # controlled give-back without accepting a weak close below the lower 35%
    # of the bar; all Falling-Knife, valuation and risk gates still apply.
    rally_retest_minimum_clv: float = -0.30
    rally_retest_minimum_cmf20: float = -0.35
    rally_retest_minimum_volume_ratio: float = 0.25
    rally_retest_maximum_volume_ratio: float = 1.25
    rally_retest_maximum_price_position: float = 0.70
    rally_retest_maximum_drawdown: float = -0.05
    rally_retest_maximum_pullback_atr: float = 0.75
    # A weak signal-day close is not itself an entry.  This secondary path
    # merely arms a 1/8 stop-entry one tick above the entire signal bar.  It
    # therefore requires stronger quality, a deep discount, orderly range,
    # usable demand evidence and every ordinary breadth veto/risk check.
    rally_reclaim_score_threshold: float = 30.0
    rally_reclaim_minimum_raw_score: float = 40.0
    rally_reclaim_minimum_clv: float = -0.70
    rally_reclaim_minimum_cmf20: float = -0.20
    rally_reclaim_minimum_volume_ratio: float = 0.50
    rally_reclaim_maximum_volume_ratio: float = 1.25
    rally_reclaim_maximum_price_position: float = 0.45
    rally_reclaim_maximum_drawdown: float = -0.15
    rally_reclaim_maximum_pullback_atr: float = 1.25
    rally_reclaim_maximum_true_range_atr: float = 1.25
    rally_reclaim_minimum_reversals: int = 2
    v5_quality: LowEntryV5QualityConfig = LowEntryV5QualityConfig()
    valuation: LowEntryV5ValuationConfig = LowEntryV5ValuationConfig()


LOW_ENTRY_OUTPUT_COLUMNS = [
    "低點分數",
    "低點決策",
    "Security Type",
    "Issuer Symbol",
    "LES v5 Final Quality Status",
    "LES v5 Final Quality Score",
    "Base Quality Status",
    "Base Quality Score",
    "Base Data Confidence",
    "Base Available Weight",
    "Extended Quality Status",
    "Extended Quality Score",
    "Extended Data Confidence",
    "Extended Available Weight",
    "Critical Field Coverage",
    "Effective Extended Weight",
    "Quality Hard Rejects",
    "Quality Warnings",
    "Valuation Gate",
    "Valuation Model Status",
    "Valuation Model Version",
    "Valuation Reasons",
    "Valuation Normalized EPS",
    "Passive Benchmark Expected Return",
    "Required Valuation Return",
    "Base Case Revenue Growth",
    "Base Case Operating Margin",
    "Base Case Exit PE",
    "Base Case Expected CAGR",
    "Base Case Fair Value",
    "Bear Case Fair Value",
    "Bear Case Permanent Loss",
    "Bear Case Permanent Loss Acceptable",
    "Bull Case Fair Value",
    "Panic Opportunity Zone",
    "Falling Knife State",
    "Temporary Falling Knife Blocks",
    "Panic Condition Count",
    "Reversal Confirmation Count",
    "Capitulation Reversal",
    "Breadth As Of",
    "Breadth Universe",
    "Breadth Coverage",
    "Breadth Valid Members",
    "Sector Advancer Ratio",
    "Sector Median R1",
    "TAIEX R1",
    "Broad Rally Regime",
    "Broad Rally Trigger",
    "Broad Rally Full BUY Trigger",
    "Broad Rally Absorption Trigger",
    "Recent Selloff Sessions",
    "Recent Broad Selloff Regime",
    "Multi-Day Panic Reversal Trigger",
    "Recent Rally Sessions",
    "Recent Broad Rally Regime",
    "Recent Rally Retest Trigger",
    "Recent Rally Reclaim Trigger",
    "Broad Rally Risk Gate",
    "Broad Rally Risk Reasons",
    "Next-Day Confirmation",
    "Market Data Gate",
    "Market Data Reasons",
    "市場環境分數",
    "價格位置分數",
    "止跌結構分數",
    "買盤確認分數",
    "動能分數",
    "執行風控分數",
    "Falling Knife Penalty",
    "Required Technical Score",
    "TAIEX Risk-Off",
    "LES v5 Trigger",
    "Risk Gate",
    "Risk Gate Reasons",
    "Position Fraction",
    "60日位置",
    "Drawdown60",
    "Drawdown20",
    "RS20",
    "RS20 Slope",
    "EMA20",
    "EMA20 Slope",
    "ATR14",
    "ATR Percent",
    "CLV",
    "CMF20",
    "Volume Ratio",
    "RSI14",
    "MACD Histogram",
    "RewardRisk",
    "Entry Candidate",
    "Entry Rule",
    "策略停損",
    "策略停利",
    "建議股數",
    "LES Model Version",
    "LES Parameter Version",
    "低點理由",
]


class LowEntryV4MarketAdapter:
    """Expose only the stable market/risk capabilities reused by v5."""

    def __init__(self, config: LowEntryScoreV5Config) -> None:
        self._model = LowEntryScoreV4Strategy(config)

    def shared_feature_config(self) -> SharedFeatureConfig:
        return self._model._shared_feature_config()

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame:
        return self._model._calculate_indicators(data, features)

    def evaluate_market_data(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series]:
        return self._model._evaluate_market_data(data, indicators, context)

    def build_trade_plan(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        return self._model._build_trade_plan(indicators, context)

    def evaluate_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        return self._model._evaluate_risk_gate(
            indicators,
            plan,
            market_gate,
        )

    def safe_series_div(
        self,
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:
        return self._model._safe_series_div(numerator, denominator)

    def positive_float(self, value: Any, fallback: float) -> float:
        return self._model._positive_float(value, fallback)


class LowEntryV5RuleEngine:
    """Default v5 rules implementing replaceable pipeline component ports."""

    @staticmethod
    def _quality_metric_passed(
        quality: LowEntryV5QualityResult,
        metric_name: str,
    ) -> bool:
        return any(
            metric.name == metric_name and metric.passed is True
            for metric in quality.extended_metrics
        )

    def _conditional_quality_starter_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        """Permit a tiny starter when only the TWSE Base snapshot is absent.

        A point-in-time MOPS fallback is less complete than the normal TWSE
        Base snapshot, so this helper admits only the completed broad-rally
        path. A separate, stronger predicate decides whether that path remains
        a 1/8 starter or graduates to a conditional 1/4 BUY.
        """

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
            and self._quality_metric_passed(quality, "positive_cfo")
            and self._quality_metric_passed(quality, "no_material_issue")
            and self._quality_metric_passed(quality, "no_going_concern")
            and self._quality_metric_passed(quality, "filing_traceable")
        )

    def _broad_rally_quality_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        regular_quality = quality.final_status in {
            FinalQualityStatus.CONFIRMED,
            FinalQualityStatus.PROVISIONAL,
        }
        return bool(
            (regular_quality or self._conditional_quality_starter_eligible(quality))
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

    def _broad_rally_full_buy_quality_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        """Require directly verified quality for a conditional breadth BUY.

        This is deliberately stricter than the 1/8 starter. UNKNOWN cash-flow,
        accounting-safety, going-concern or filing evidence cannot graduate to
        the normal BUY state.
        """

        return bool(
            self._broad_rally_quality_eligible(quality)
            and quality.final_score is not None
            and quality.final_score
            >= self.config.breadth_rally_full_buy_minimum_quality_score
            and quality.extended_status is ExtendedQualityStatus.PASS
            and not quality.hard_rejects
            and self._quality_metric_passed(quality, "positive_cfo")
            and self._quality_metric_passed(quality, "no_material_issue")
            and self._quality_metric_passed(quality, "no_going_concern")
            and self._quality_metric_passed(quality, "filing_traceable")
        )

    def __init__(
        self,
        config: LowEntryScoreV5Config | None = None,
        valuation_producer: LowEntryV5ValuationProducer | None = None,
        market_adapter: LowEntryV4MarketAdapter | None = None,
    ) -> None:
        resolved_config = config or LowEntryScoreV5Config()
        self.config = resolved_config
        self.market_adapter = market_adapter or LowEntryV4MarketAdapter(
            resolved_config
        )
        self.valuation_producer = valuation_producer or LowEntryV5ValuationProducer(
            resolved_config.valuation
        )

    def shared_feature_config(self) -> SharedFeatureConfig:
        return self.market_adapter.shared_feature_config()

    def evaluate_market_data(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series]:
        gate, reasons = self.market_adapter.evaluate_market_data(
            data,
            indicators,
            context,
        )
        snapshot = normalize_market_breadth_snapshot(
            context.get(MARKET_BREADTH_ATTR)
        )
        breadth_history = context.get(MARKET_BREADTH_HISTORY_ATTR)
        breadth_history = (
            dict(breadth_history)
            if isinstance(breadth_history, dict)
            else {}
        )
        indicators["taiex_r1"] = indicators["taiex"].pct_change(
            fill_method=None
        )
        self._attach_breadth_facts(indicators, snapshot)
        indicators["broad_rally_regime"] = self._evaluate_breadth_rally_regime(
            data,
            indicators,
            breadth_snapshot=snapshot,
            market_data_through=context.get(MARKET_DATA_THROUGH_ATTR),
            market_session_final=bool(
                context.get(MARKET_SESSION_FINAL_ATTR, False)
            ),
        )
        selloff_regime, selloff_count = self._evaluate_recent_breadth_regime(
            data,
            indicators,
            breadth_history=breadth_history,
            market_data_through=context.get(MARKET_DATA_THROUGH_ATTR),
            direction="selloff",
        )
        rally_regime, rally_count = self._evaluate_recent_breadth_regime(
            data,
            indicators,
            breadth_history=breadth_history,
            market_data_through=context.get(MARKET_DATA_THROUGH_ATTR),
            direction="rally",
        )
        indicators["recent_broad_selloff_regime"] = selloff_regime
        indicators["recent_selloff_sessions"] = selloff_count
        indicators["recent_broad_rally_regime"] = rally_regime
        indicators["recent_rally_sessions"] = rally_count
        return gate, reasons

    def build_trade_plan(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        return self.market_adapter.build_trade_plan(indicators, context)

    def build_broad_rally_plan(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        """Reprice a conditional order one valid TWSE tick above signal high."""

        price_scale = self.market_adapter.positive_float(
            context.get(PRICE_SCALE_ATTR),
            1.0,
        )
        entry = self._next_twse_stock_tick(
            indicators["high"],
            price_scale=price_scale,
        )
        repriced = plan.copy()
        repriced["entry"] = entry
        signal_bar_stop = (
            indicators["low"]
            - self.config.breadth_rally_signal_stop_buffer_atr
            * indicators["atr14"]
        )
        # The higher stop is the nearer invalidation point.  It is used only
        # by conditional breadth entries, all of which require a next-session
        # break of the signal-day high before execution.
        repriced["stop"] = pd.concat(
            [repriced["stop"], signal_bar_stop],
            axis=1,
        ).max(axis=1)
        repriced["risk"] = entry - repriced["stop"]
        repriced["reward_risk"] = self.market_adapter.safe_series_div(
            repriced["target"] - entry,
            repriced["risk"],
        ).where(repriced["risk"] > 0)

        account_equity = self.market_adapter.positive_float(
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

    def evaluate_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        return self.market_adapter.evaluate_risk_gate(
            indicators,
            plan,
            market_gate,
        )

    def evaluate_broad_rally_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        liquidity_known = (
            indicators["amount_ma20"].notna()
            & indicators["spread"].notna()
            & plan["order_amount"].notna()
        )
        # ``plan[order_amount]`` is the full risk-sized position, whereas a
        # breadth trade is capped at 1/8 and can graduate only to 1/4.  Testing
        # the unscaled order incorrectly rejected executable odd-lot starters.
        # Use the largest position this breadth path can actually authorize.
        executable_order_amount = (
            plan["order_amount"]
            * self.config.breadth_rally_full_buy_position_fraction
        )
        liquidity_pass = (
            liquidity_known
            & (
                indicators["amount_ma20"]
                >= self.config.breadth_rally_minimum_amount_ma20
            )
            & (indicators["spread"] <= self.config.maximum_spread)
            & (
                executable_order_amount
                <= self.config.maximum_order_amount_ratio
                * indicators["amount_ma20"]
            )
        )
        market_history_known = indicators[
            ["taiex", "taiex_ema20", "taiex_r5", "rs20", "price_position60"]
        ].notna().all(axis=1)
        plan_known = (
            plan[
                ["entry", "stop", "target", "risk", "reward_risk", "shares"]
            ].notna().all(axis=1)
            & indicators["atr14"].notna()
            & liquidity_known
            & market_history_known
            & (market_gate == "PASS")
        )
        passed = (
            plan_known
            & (plan["risk"] > 0)
            & (
                plan["risk"]
                <= self.config.breadth_rally_maximum_stop_atr
                * indicators["atr14"]
            )
            & (
                plan["reward_risk"]
                >= self.config.breadth_rally_minimum_reward_risk
            )
            & (plan["shares"] > 0)
            & liquidity_pass
        )
        known_failure = plan_known & ~passed
        status = pd.Series(
            np.select(
                [passed, known_failure],
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
                    > self.config.breadth_rally_maximum_stop_atr
                    * indicators["atr14"]
                ),
                "INVALID_TARGET": plan["target"].isna()
                | (plan["target"] <= plan["entry"]),
                "INSUFFICIENT_REWARD_RISK": plan["reward_risk"].notna()
                & (
                    plan["reward_risk"]
                    < self.config.breadth_rally_minimum_reward_risk
                ),
                "LIQUIDITY_UNKNOWN": ~liquidity_known,
                "LIQUIDITY_FAILED": liquidity_known & ~liquidity_pass,
                "MARKET_HISTORY_INSUFFICIENT": ~market_history_known,
                "MARKET_DATA_INVALID": market_gate != "PASS",
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

    @staticmethod
    def _next_twse_stock_tick(
        prices: pd.Series,
        *,
        price_scale: float,
    ) -> pd.Series:
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
        next_price = (
            (np.floor((actual / tick) + 1e-9) + 1)
            * tick
            * price_scale
        )
        return next_price.where(actual.notna())

    @staticmethod
    def _attach_breadth_facts(
        indicators: pd.DataFrame,
        snapshot: dict[str, object],
    ) -> None:
        indicators["breadth_as_of"] = str(snapshot.get("as_of") or "")
        indicators["breadth_universe"] = str(
            snapshot.get("universe") or "UNKNOWN"
        ).upper()
        for output, source in (
            ("breadth_coverage", "coverage"),
            ("breadth_valid_members", "valid_return_count"),
            ("sector_advancer_ratio", "advancer_ratio"),
            ("sector_median_r1", "median_return"),
        ):
            indicators[output] = pd.to_numeric(
                snapshot.get(source),
                errors="coerce",
            )

    def _evaluate_breadth_rally_regime(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        *,
        breadth_snapshot: dict[str, object],
        market_data_through: object,
        market_session_final: bool,
    ) -> pd.Series:
        false = pd.Series(False, index=indicators.index)
        if not breadth_snapshot or "Date" not in data:
            return false

        universe = str(breadth_snapshot.get("universe") or "").upper()
        if universe == "SEMI":
            minimum_members = self.config.semiconductor_breadth_minimum_members
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

        coverage = pd.to_numeric(breadth_snapshot.get("coverage"), errors="coerce")
        members = pd.to_numeric(
            breadth_snapshot.get("valid_return_count"),
            errors="coerce",
        )
        advancers = pd.to_numeric(
            breadth_snapshot.get("advancer_ratio"),
            errors="coerce",
        )
        median_return = pd.to_numeric(
            breadth_snapshot.get("median_return"),
            errors="coerce",
        )
        breadth_as_of = str(breadth_snapshot.get("as_of") or "")
        market_through = str(market_data_through or "")
        scalar_gate = bool(
            market_session_final
            and breadth_snapshot.get("source_final") is True
            and breadth_as_of
            and breadth_as_of == market_through
            and pd.notna(coverage)
            and coverage >= self.config.breadth_rally_minimum_coverage
            and pd.notna(members)
            and members >= minimum_members
            and pd.notna(advancers)
            and advancers >= minimum_advancer_ratio
            and pd.notna(median_return)
            and median_return >= minimum_median_return
        )
        if not scalar_gate:
            return false

        taiex_return = pd.to_numeric(
            breadth_snapshot.get("taiex_return"),
            errors="coerce",
        )
        row_dates = pd.to_datetime(data["Date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        if pd.notna(taiex_return):
            indicators.loc[row_dates == breadth_as_of, "taiex_r1"] = float(
                taiex_return
            )
        return (
            (row_dates == breadth_as_of)
            & (
                indicators["taiex_r1"]
                >= self.config.breadth_rally_minimum_taiex_return
            )
        ).fillna(False)

    def _evaluate_recent_breadth_regime(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        *,
        breadth_history: dict[str, object],
        market_data_through: object,
        direction: str,
    ) -> tuple[pd.Series, pd.Series]:
        """Evaluate point-in-time breadth before the signal session.

        The signal day is excluded so an entry bar cannot manufacture the
        market regime that authorizes it. This also keeps a broad rally alive
        briefly while a stock performs an orderly retest instead of requiring
        every confirmation to occur on the same close.
        """

        false = pd.Series(False, index=indicators.index)
        zero = pd.Series(0, index=indicators.index, dtype=int)
        if direction not in {"selloff", "rally"}:
            raise ValueError("direction must be selloff or rally")
        if not breadth_history or "Date" not in data:
            return false, zero

        signal_date = pd.to_datetime(market_data_through, errors="coerce")
        if pd.isna(signal_date):
            return false, zero

        usable: list[tuple[pd.Timestamp, dict[str, object]]] = []
        for key, raw_snapshot in breadth_history.items():
            snapshot = normalize_market_breadth_snapshot(raw_snapshot)
            snapshot_date = pd.to_datetime(
                snapshot.get("as_of") or key,
                errors="coerce",
            )
            if pd.isna(snapshot_date) or snapshot_date >= signal_date:
                continue
            if snapshot.get("source_final") is not True:
                continue
            usable.append((snapshot_date.normalize(), snapshot))

        usable.sort(key=lambda item: item[0])
        recent = usable[-self.config.breadth_memory_lookback_sessions :]
        if len(recent) < self.config.breadth_memory_minimum_sessions:
            return false, zero

        matching_sessions = 0
        for _, snapshot in recent:
            universe = str(snapshot.get("universe") or "").upper()
            if universe == "SEMI":
                minimum_members = self.config.semiconductor_breadth_minimum_members
            elif universe == "ELEC":
                minimum_members = self.config.electronics_breadth_minimum_members
            else:
                continue
            coverage = pd.to_numeric(snapshot.get("coverage"), errors="coerce")
            members = pd.to_numeric(
                snapshot.get("valid_return_count"), errors="coerce"
            )
            advancers = pd.to_numeric(
                snapshot.get("advancer_ratio"), errors="coerce"
            )
            median_return = pd.to_numeric(
                snapshot.get("median_return"), errors="coerce"
            )
            taiex_return = pd.to_numeric(
                snapshot.get("taiex_return"), errors="coerce"
            )
            data_valid = bool(
                pd.notna(coverage)
                and coverage >= self.config.breadth_rally_minimum_coverage
                and pd.notna(members)
                and members >= minimum_members
                and pd.notna(advancers)
                and pd.notna(median_return)
                and pd.notna(taiex_return)
            )
            if not data_valid:
                continue
            if direction == "selloff":
                matches = bool(
                    advancers
                    <= self.config.panic_selloff_maximum_advancer_ratio
                    and median_return
                    <= self.config.panic_selloff_maximum_median_return
                    and taiex_return
                    <= self.config.panic_selloff_maximum_taiex_return
                )
            else:
                breadth_led_recovery = bool(
                    advancers
                    >= self.config.recent_rally_lead_minimum_advancer_ratio
                    and median_return
                    >= self.config.recent_rally_lead_minimum_median_return
                )
                matches = bool(
                    advancers >= self.config.recent_rally_minimum_advancer_ratio
                    and median_return
                    >= self.config.recent_rally_minimum_median_return
                    and (
                        taiex_return
                        >= self.config.recent_rally_minimum_taiex_return
                        or breadth_led_recovery
                    )
                )
            matching_sessions += int(matches)

        row_dates = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
        current_row = row_dates == signal_date.normalize()
        regime = current_row & (
            matching_sessions >= self.config.breadth_memory_minimum_sessions
        )
        count = zero.copy()
        count.loc[current_row] = matching_sessions
        return regime.fillna(False), count

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
                "Valuation Reasons": (
                    "；".join(valuation_result.reasons) or "NONE"
                ),
                "Valuation Normalized EPS": self.scenario_value(
                    valuation_result,
                    "base",
                    "normalized_eps",
                ),
                "Passive Benchmark Expected Return": (
                    valuation_result.passive_benchmark_expected_return
                ),
                "Required Valuation Return": valuation_result.required_return,
                "Base Case Revenue Growth": self.scenario_value(
                    valuation_result,
                    "base",
                    "revenue_growth",
                ),
                "Base Case Operating Margin": self.scenario_value(
                    valuation_result,
                    "base",
                    "operating_margin",
                ),
                "Base Case Exit PE": self.scenario_value(
                    valuation_result,
                    "base",
                    "exit_pe",
                ),
                "Base Case Expected CAGR": self.scenario_value(
                    valuation_result,
                    "base",
                    "expected_cagr",
                ),
                "Base Case Fair Value": self.scenario_value(
                    valuation_result,
                    "base",
                    "fair_value",
                ),
                "Bear Case Fair Value": self.scenario_value(
                    valuation_result,
                    "bear",
                    "fair_value",
                ),
                "Bear Case Permanent Loss": (
                    valuation_result.bear_permanent_loss
                ),
                "Bear Case Permanent Loss Acceptable": (
                    valuation_result.bear_permanent_loss_acceptable
                ),
                "Bull Case Fair Value": self.scenario_value(
                    valuation_result,
                    "bull",
                    "fair_value",
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
                "Broad Rally Regime": indicators[
                    "broad_rally_regime"
                ].astype(int),
                "Broad Rally Trigger": panic[
                    "broad_rally_trigger"
                ].astype(int),
                "Broad Rally Full BUY Trigger": panic[
                    "broad_rally_full_buy_trigger"
                ].astype(int),
                "Broad Rally Absorption Trigger": panic[
                    "broad_rally_absorption_trigger"
                ].astype(int),
                "Recent Selloff Sessions": indicators[
                    "recent_selloff_sessions"
                ],
                "Recent Broad Selloff Regime": indicators[
                    "recent_broad_selloff_regime"
                ].astype(int),
                "Multi-Day Panic Reversal Trigger": panic[
                    "multiday_panic_reversal_trigger"
                ].astype(int),
                "Recent Rally Sessions": indicators[
                    "recent_rally_sessions"
                ],
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
                            decision == "BUY_SMALL",
                            decision == "EARLY_BUY",
                            decision == "BUY",
                        ],
                        [
                            "NEXT_SESSION_BREAKS_SIGNAL_HIGH",
                            "NEXT_SESSION_BREAKS_SIGNAL_HIGH",
                            "NEXT_DAY_CONFIRMATION",
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

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame:
        indicators = self.market_adapter.calculate_indicators(data, features)
        shared = features.to_pandas()
        close = indicators["close"]
        low = indicators["low"]
        high = indicators["high"]
        open_price = indicators["open"]
        true_range = shared["true_range"]
        lower_shadow = open_price.combine(close, min) - low

        indicators["rsi14"] = shared["rsi14_wilder"]
        indicators["macd"] = shared["macd_strict"]
        indicators["macd_signal"] = shared["macd_signal_strict"]
        indicators["macd_histogram"] = shared["macd_histogram_strict"]
        indicators["true_range"] = true_range
        indicators["drawdown20"] = (
            self.market_adapter.safe_series_div(
                close,
                close.rolling(20, min_periods=20).max(),
            )
            - 1
        )
        indicators["pe_median60"] = (
            indicators["pe"]
            .where(indicators["pe"] > 0)
            .rolling(self.config.range_window, min_periods=20)
            .median()
        )
        indicators["ema5_slope"] = self.market_adapter.safe_series_div(
            indicators["ema5"] - indicators["ema5"].shift(1),
            indicators["atr14"],
        )
        indicators["rs20_slope"] = indicators["rs20"] - indicators["rs20"].shift(5)
        indicators["taiex_ema20_slope"] = (
            indicators["taiex_ema20"] - indicators["taiex_ema20"].shift(5)
        )
        taiex_return = indicators["taiex"].pct_change(fill_method=None)
        indicators["taiex_volatility20"] = taiex_return.rolling(
            20,
            min_periods=20,
        ).std(ddof=0)
        indicators["taiex_volatility_ma20"] = indicators[
            "taiex_volatility20"
        ].rolling(20, min_periods=20).mean()
        indicators["previous_high"] = high.shift(1)
        indicators["previous_midpoint"] = (high.shift(1) + low.shift(1)) / 2
        indicators["lower_shadow_ratio"] = self.market_adapter.safe_series_div(
            lower_shadow,
            true_range,
        )
        indicators["new_low_count3"] = (
            indicators["new_20_day_low"]
            .fillna(False)
            .astype(int)
            .rolling(3, min_periods=3)
            .sum()
        )
        indicators["weak_clv3"] = (
            (indicators["clv"] < 0)
            .astype(int)
            .rolling(3, min_periods=3)
            .sum()
            >= 3
        )
        return indicators

    def evaluate_quality(
        self,
        data: pd.DataFrame,
        context: dict[str, Any],
    ) -> LowEntryV5QualityResult:
        annual = context.get(ANNUAL_FUNDAMENTALS_ATTR)
        annual = annual if isinstance(annual, pd.DataFrame) else pd.DataFrame()
        snapshot = context.get(FUNDAMENTAL_SNAPSHOT_ATTR)
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        signal_date = context.get(SIGNAL_DATE_ATTR)
        if signal_date is None and "Date" in data and not data.empty:
            signal_date = data["Date"].iloc[-1]
        return evaluate_v5_quality(
            annual,
            snapshot,
            signal_date,
            self.config.v5_quality,
        )

    def evaluate_valuation(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, LowEntryV5ValuationResult]:
        annual = context.get(ANNUAL_FUNDAMENTALS_ATTR)
        annual = annual if isinstance(annual, pd.DataFrame) else pd.DataFrame()
        snapshot = context.get(FUNDAMENTAL_SNAPSHOT_ATTR)
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        price_scale = self.market_adapter.positive_float(
            context.get(PRICE_SCALE_ATTR),
            1.0,
        )
        current_price = indicators["close"].iloc[-1] / price_scale
        recent_pe = indicators["pe_median60"].iloc[-1]
        if pd.isna(recent_pe):
            recent_pe = indicators["pe"].iloc[-1]
        result = self.valuation_producer.produce(
            annual,
            snapshot,
            signal_date=context.get(SIGNAL_DATE_ATTR),
            current_price=current_price,
            recent_pe_median=recent_pe,
        )
        if result.status is ValuationProductionStatus.UNKNOWN:
            status = ValuationStatus.UNKNOWN
        elif result.gate_passed:
            status = ValuationStatus.PASS
        else:
            status = ValuationStatus.FAIL
        return (
            pd.Series(status.value, index=indicators.index),
            result,
        )

    @staticmethod
    def scenario_value(
        result: LowEntryV5ValuationResult,
        scenario_name: str,
        field_name: str,
    ) -> object:
        scenario = getattr(result, scenario_name)
        return getattr(scenario, field_name) if scenario is not None else np.nan

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
                    & (
                        indicators["r5"]
                        < -1.5 * indicators["atr_percent"]
                    )
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
                "R5_ATR_SHOCK": (
                    indicators["r5"] <= -2 * indicators["atr_percent"]
                ),
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
        capitulation = (
            (panic_count >= 1)
            & (reversal_count >= 3)
            & quality_series
            & (valuation == ValuationStatus.PASS.value)
            & (quality.extended_status is not ExtendedQualityStatus.REJECT)
        )
        broad_quality = self._broad_rally_quality_eligible(quality)
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
            indicators["broad_rally_regime"]
            & (block_text == "NONE")
            & broad_reversal_setup
            & (indicators["close"] > indicators["close"].shift(1))
            & (
                indicators["close"]
                >= indicators["open"] - (0.15 * indicators["atr14"])
            )
            & (indicators["clv"] >= self.config.breadth_rally_minimum_clv)
            & volume_confirmation
            & (
                indicators["cmf20"]
                >= self.config.breadth_rally_minimum_cmf20
            )
            & broad_quality
            & (valuation == ValuationStatus.PASS.value)
        ).fillna(False)
        broad_rally_full_buy_trigger = (
            broad_rally_trigger
            & (
                panic_count
                >= self.config.breadth_rally_full_buy_minimum_panics
            )
            & (
                reversal_count
                >= self.config.breadth_rally_full_buy_minimum_reversals
            )
            & (
                indicators["clv"]
                >= self.config.breadth_rally_full_buy_minimum_clv
            )
            & (
                indicators["cmf20"]
                >= self.config.breadth_rally_full_buy_minimum_cmf20
            )
            & (
                indicators["volume_ratio"]
                >= self.config.breadth_rally_strong_close_minimum_volume_ratio
            )
            & self._broad_rally_full_buy_quality_eligible(quality)
        ).fillna(False)
        broad_rally_absorption_trigger = (
            indicators["broad_rally_regime"]
            & (block_text == "NONE")
            & (
                panic_count
                >= self.config.breadth_absorption_minimum_panics
            )
            & (
                reversal_count
                >= self.config.breadth_absorption_minimum_reversals
            )
            & (indicators["close"] > indicators["close"].shift(1))
            & (
                indicators["close"]
                >= indicators["open"] - (0.15 * indicators["atr14"])
            )
            & (
                indicators["clv"]
                >= self.config.breadth_absorption_minimum_clv
            )
            & (
                indicators["cmf20"]
                >= self.config.breadth_absorption_minimum_cmf20
            )
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
            & self._broad_rally_full_buy_quality_eligible(quality)
            & (valuation == ValuationStatus.PASS.value)
        ).fillna(False)
        recent_selloff_regime = indicators.get(
            "recent_broad_selloff_regime",
            pd.Series(False, index=indicators.index),
        ).fillna(False)
        multiday_panic_reversal_trigger = (
            recent_selloff_regime
            & (block_text == "NONE")
            & (
                reversal_count
                >= self.config.multiday_panic_minimum_reversals
            )
            & (indicators["close"] > indicators["close"].shift(1))
            & (
                (indicators["close"] > indicators["previous_high"])
                | undercut_reclaim
            )
            & (indicators["clv"] >= self.config.multiday_panic_minimum_clv)
            & (
                indicators["cmf20"]
                >= self.config.multiday_panic_minimum_cmf20
            )
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
            & self._broad_rally_full_buy_quality_eligible(quality)
            & (valuation == ValuationStatus.PASS.value)
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
                (indicators["volume_ratio"]
                 <= self.config.rally_retest_maximum_volume_ratio)
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
            # This path creates only a conditional 1/8 starter.  Requiring the
            # stronger full-BUY quality predicate made it stricter than the
            # other breadth starters and discarded audited MOPS fallback cases
            # with adequate coverage.  The ordinary starter predicate still
            # requires QS/DC floors, Extended confidence, no hard reject and
            # explicit valuation PASS.
            & self._broad_rally_quality_eligible(quality)
            & (valuation == ValuationStatus.PASS.value)
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
            (
                reversal_count
                >= self.config.rally_reclaim_minimum_reversals
            )
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
            & (
                indicators["cmf20"]
                >= self.config.rally_reclaim_minimum_cmf20
            )
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
            & self._broad_rally_full_buy_quality_eligible(quality)
            & (valuation == ValuationStatus.PASS.value)
        ).fillna(False)

        state = pd.Series(FallingKnifeState.CLEAR.value, index=indicators.index)
        state.loc[blocks.any(axis=1)] = FallingKnifeState.PANIC_WATCH.value
        state.loc[undercut_reclaim] = FallingKnifeState.REVERSAL_WATCH.value
        state.loc[capitulation] = FallingKnifeState.REVERSAL_WATCH.value
        if quality.final_status is FinalQualityStatus.HARD_NO_BUY:
            state[:] = FallingKnifeState.HARD_NO_BUY.value
        elif quality.final_status is FinalQualityStatus.DATA_INSUFFICIENT:
            state[:] = FallingKnifeState.DATA_INSUFFICIENT.value

        return {
            "opportunity": opportunity,
            "state": state,
            "block_text": block_text,
            "panic_count": panic_count,
            "reversal_count": reversal_count,
            "capitulation": capitulation,
            "broad_reversal_setup": broad_reversal_setup,
            "broad_rally_trigger": broad_rally_trigger,
            "broad_rally_full_buy_trigger": broad_rally_full_buy_trigger,
            "broad_rally_absorption_trigger": broad_rally_absorption_trigger,
            "multiday_panic_reversal_trigger": (
                multiday_panic_reversal_trigger
            ),
            "recent_rally_retest_trigger": recent_rally_retest_trigger,
            "recent_rally_reclaim_trigger": recent_rally_reclaim_trigger,
        }

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
        macd_histogram_improving = (
            (indicators["macd_histogram"] > indicators["macd_histogram"].shift(1))
            & (
                indicators["macd_histogram"].shift(1)
                > indicators["macd_histogram"].shift(2)
            )
        )
        macd_cross = (
            (indicators["macd"] > indicators["macd_signal"])
            & (indicators["macd"].shift(1) <= indicators["macd_signal"].shift(1))
        )

        market_regime = (
            4 * (indicators["taiex"] > indicators["taiex_ema20"]).astype(int)
            + 3 * (indicators["taiex_ema20_slope"] >= 0).astype(int)
            + 4 * (indicators["rs20"] > 0).astype(int)
            + 4 * market_volatility_normal.astype(int)
        )
        price_position = (
            5
            * indicators["price_position60"]
            .between(0.05, 0.25, inclusive="both")
            .astype(int)
            + 4
            * indicators["drawdown60"]
            .between(-0.30, -0.10, inclusive="both")
            .astype(int)
            + 3 * (~confirmed_breakdown.fillna(True)).astype(int)
            + 3 * (valuation == ValuationStatus.PASS.value).astype(int)
        )
        stabilization = (
            6 * no_new_low3.astype(int)
            + 7 * indicators["higher_low"].astype(int)
            + 4 * (indicators["close"] > indicators["ema5"]).astype(int)
            + 4 * ema5_turn.astype(int)
            + 5 * (indicators["close"] > indicators["previous_high"]).astype(int)
            + 4
            * (
                indicators["atr14"].notna()
                & (indicators["atr14"] <= indicators["atr14"].shift(1))
            ).astype(int)
        )
        demand = (
            5 * (indicators["clv"] > 0.4).astype(int)
            + 5 * indicators["cmf20_improving"].astype(int)
            + 4 * (indicators["normalized_obv_slope5"] > 0).astype(int)
            + 3
            * (
                (indicators["close"] > indicators["close"].shift(1))
                & indicators["volume_ratio"].between(0.9, 1.8, inclusive="both")
                & (indicators["clv"] >= 0)
            ).astype(int)
            + 3 * (indicators["close"] > indicators["vwap20"]).astype(int)
        )
        momentum = (
            4
            * (
                (indicators["rsi14"] >= 40)
                & (indicators["rsi14"].shift(1) < 40)
            ).astype(int)
            + 3 * macd_histogram_improving.astype(int)
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
                indicators["amount_ma20"] >= self.config.minimum_amount_ma20
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
                    (indicators["rs20"] < 0) & (indicators["rs20_slope"] < 0),
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

    def evaluate_trigger(self, indicators: pd.DataFrame) -> pd.Series:
        price_confirmation = (
            (indicators["close"] > indicators["previous_high"])
            | (
                (indicators["close"] > indicators["ema5"])
                & (indicators["close"] > indicators["vwap20"])
            )
            | (
                indicators["higher_low"]
                & (indicators["ema5_slope"] > 0)
            )
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
    ) -> pd.Series:
        full_buy_trigger = (
            broad_rally_full_buy_trigger
            if broad_rally_full_buy_trigger is not None
            else pd.Series(False, index=score.index)
        )
        absorption_trigger = (
            broad_rally_absorption_trigger
            if broad_rally_absorption_trigger is not None
            else pd.Series(False, index=score.index)
        )
        multiday_trigger = (
            multiday_panic_reversal_trigger
            if multiday_panic_reversal_trigger is not None
            else pd.Series(False, index=score.index)
        )
        retest_trigger = (
            recent_rally_retest_trigger
            if recent_rally_retest_trigger is not None
            else pd.Series(False, index=score.index)
        )
        reclaim_trigger = (
            recent_rally_reclaim_trigger
            if recent_rally_reclaim_trigger is not None
            else pd.Series(False, index=score.index)
        )
        decision = pd.Series("ARMED", index=score.index)
        decision.loc[score < 60] = "AVOID"
        decision.loc[score.between(60, 69.999, inclusive="both")] = "WATCH"

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
            & (
                (quality.final_status is FinalQualityStatus.CONFIRMED)
                | (score >= self.config.provisional_buy_score)
            )
        )
        decision.loc[normal_buy] = "BUY"
        decision.loc[early_buy] = "EARLY_BUY"
        decision.loc[
            falling_state == FallingKnifeState.PANIC_WATCH.value
        ] = "PANIC_WATCH"
        decision.loc[
            falling_state == FallingKnifeState.REVERSAL_WATCH.value
        ] = "REVERSAL_WATCH"

        # A completed cross-sectional rally can expose either a genuine panic
        # reversal or a depressed-stock recovery before lagging trend
        # indicators recover. The base path authorizes a conditional 1/8
        # BUY_SMALL and cannot offset any temporary Falling-Knife block because
        # the trigger itself requires NONE.
        broad_rally_buy_small = (
            self._broad_rally_quality_eligible(quality)
            & (score >= self.config.breadth_rally_score_threshold)
            & (raw_score >= self.config.breadth_rally_minimum_raw_score)
            & broad_rally_trigger
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & (valuation == ValuationStatus.PASS.value)
        )
        absorption_buy_small = (
            self._broad_rally_full_buy_quality_eligible(quality)
            & (
                score
                >= self.config.breadth_absorption_score_threshold
            )
            & (
                raw_score
                >= self.config.breadth_absorption_minimum_raw_score
            )
            & absorption_trigger
            & (risk_gate == "PASS")
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & (valuation == ValuationStatus.PASS.value)
        )
        multiday_panic_buy_small = (
            self._broad_rally_full_buy_quality_eligible(quality)
            & (score >= self.config.multiday_panic_score_threshold)
            & (raw_score >= self.config.multiday_panic_minimum_raw_score)
            & multiday_trigger
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & (valuation == ValuationStatus.PASS.value)
        )
        rally_retest_buy_small = (
            self._broad_rally_quality_eligible(quality)
            & (score >= self.config.rally_retest_score_threshold)
            & (raw_score >= self.config.rally_retest_minimum_raw_score)
            & retest_trigger
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & (valuation == ValuationStatus.PASS.value)
        )
        rally_reclaim_buy_small = (
            self._broad_rally_full_buy_quality_eligible(quality)
            & (score >= self.config.rally_reclaim_score_threshold)
            & (raw_score >= self.config.rally_reclaim_minimum_raw_score)
            & reclaim_trigger
            & (broad_rally_risk_gate == "PASS")
            & (market_gate == "PASS")
            & (valuation == ValuationStatus.PASS.value)
        )
        broad_rally_buy_small = (
            broad_rally_buy_small
            | absorption_buy_small
            | multiday_panic_buy_small
            | rally_retest_buy_small
            | rally_reclaim_buy_small
        )
        decision.loc[broad_rally_buy_small] = "BUY_SMALL"

        # The stronger breadth path is a conditional normal BUY. Standard
        # Risk Gate is required in addition to the repriced broad-rally gate,
        # so the relaxed score cannot widen the stop, lower RR, reduce
        # liquidity, or offset a Falling-Knife block.
        broad_rally_full_buy = (
            self._broad_rally_full_buy_quality_eligible(quality)
            & (
                score
                >= self.config.breadth_rally_full_buy_score_threshold
            )
            & (
                raw_score
                >= self.config.breadth_rally_full_buy_minimum_raw_score
            )
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
            # A high-confidence point-in-time MOPS fallback may keep only the
            # explicitly qualified breadth starter or breadth BUY assigned
            # above. All ordinary BUY/EARLY_BUY rows remain forbidden.
            conditional_breadth_entry = (
                broad_rally_buy_small | broad_rally_full_buy
            )
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
        fraction.loc[decision == "EARLY_BUY"] = self.config.early_position_fraction
        fraction.loc[decision == "BUY_SMALL"] = (
            self.config.breadth_rally_position_fraction
        )
        if broad_rally_full_buy is not None:
            fraction.loc[(decision == "BUY") & broad_rally_full_buy] = (
                self.config.breadth_rally_full_buy_position_fraction
            )
        fraction.loc[risk_off & fraction.gt(self.config.provisional_position_fraction)] = (
            self.config.provisional_position_fraction
        )
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
        max_shares = (
            equity * position_fraction / (plan["entry"] / price_scale)
        )
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
        return detail.apply(
            lambda row: (
                "{}；Valuation={}({})；分數={:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}"
                "；Penalty={:.0f}；Falling={}({})；Panic/Reversal={:.0f}/{:.0f}"
                "；Trigger={}；Risk={}({})；BroadRally={}/{}/Full={}/Absorb={}/MultiDay={}/Retest={}/Reclaim={}；Required={:.0f}；決策={}"
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
                row["required"],
                row["decision"],
            ),
            axis=1,
        )

    def empty_result(self) -> dict[str, object]:
        result = {column: np.nan for column in LOW_ENTRY_OUTPUT_COLUMNS}
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
                "Falling Knife State": FallingKnifeState.DATA_INSUFFICIENT.value,
                "Market Data Gate": "FAIL",
                "Market Data Reasons": "DATA_INSUFFICIENT",
                "LES Model Version": LES_V5_MODEL_VERSION,
                "LES Parameter Version": LES_V5_PARAMETER_VERSION,
                "低點理由": "資料不足；決策=DATA_INSUFFICIENT",
            }
        )
        return result


@dataclass(frozen=True)
class LowEntryV5MarketRules:
    rules: LowEntryV5RuleEngine

    def shared_feature_config(self) -> SharedFeatureConfig:
        return self.rules.shared_feature_config()

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame:
        return self.rules.calculate_indicators(data, features)

    def evaluate_market_data(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series]:
        return self.rules.evaluate_market_data(data, indicators, context)

    def build_trade_plan(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        return self.rules.build_trade_plan(indicators, context)

    def build_broad_rally_plan(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        return self.rules.build_broad_rally_plan(indicators, plan, context)

    def evaluate_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        return self.rules.evaluate_risk_gate(indicators, plan, market_gate)

    def evaluate_broad_rally_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        return self.rules.evaluate_broad_rally_risk_gate(
            indicators,
            plan,
            market_gate,
        )


@dataclass(frozen=True)
class LowEntryV5FundamentalRules:
    rules: LowEntryV5RuleEngine

    def evaluate_quality(
        self,
        data: pd.DataFrame,
        context: dict[str, Any],
    ) -> LowEntryV5QualityResult:
        return self.rules.evaluate_quality(data, context)

    def evaluate_valuation(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, LowEntryV5ValuationResult]:
        return self.rules.evaluate_valuation(indicators, context)


@dataclass(frozen=True)
class LowEntryV5PanicRules:
    rules: LowEntryV5RuleEngine

    def evaluate_panic(
        self,
        indicators: pd.DataFrame,
        quality: LowEntryV5QualityResult,
        valuation: pd.Series,
    ) -> dict[str, pd.Series]:
        return self.rules.evaluate_panic(indicators, quality, valuation)


@dataclass(frozen=True)
class LowEntryV5ScoringRules:
    rules: LowEntryV5RuleEngine

    def score_components(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        valuation: pd.Series,
    ) -> pd.DataFrame:
        return self.rules.score_components(indicators, plan, valuation)

    def falling_knife_penalty(
        self,
        indicators: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        return self.rules.falling_knife_penalty(indicators)

    def required_score(
        self,
        quality: LowEntryV5QualityResult,
        risk_off: pd.Series,
        index: pd.Index,
    ) -> pd.Series:
        return self.rules.required_score(quality, risk_off, index)


@dataclass(frozen=True)
class LowEntryV5DecisionRules:
    rules: LowEntryV5RuleEngine

    def evaluate_trigger(self, indicators: pd.DataFrame) -> pd.Series:
        return self.rules.evaluate_trigger(indicators)

    def next_day_confirmation(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        falling_state: pd.Series,
    ) -> pd.Series:
        return self.rules.next_day_confirmation(
            indicators,
            plan,
            falling_state,
        )

    def resolve_decision(self, **kwargs: Any) -> pd.Series:
        return self.rules.resolve_decision(**kwargs)

    def position_fraction(
        self,
        decision: pd.Series,
        quality: LowEntryV5QualityResult,
        risk_off: pd.Series,
        broad_rally_full_buy: pd.Series | None = None,
    ) -> pd.Series:
        return self.rules.position_fraction(
            decision,
            quality,
            risk_off,
            broad_rally_full_buy,
        )

    def cap_position_shares(
        self,
        plan: pd.DataFrame,
        position_fraction: pd.Series,
        context: dict[str, Any],
    ) -> pd.Series:
        return self.rules.cap_position_shares(
            plan,
            position_fraction,
            context,
        )

    def enforce_executable_position(
        self,
        decision: pd.Series,
        recommended_shares: pd.Series,
    ) -> pd.Series:
        return self.rules.enforce_executable_position(
            decision,
            recommended_shares,
        )


@dataclass(frozen=True)
class LowEntryV5ReportRules:
    rules: LowEntryV5RuleEngine

    def empty_result(self) -> dict[str, object]:
        return self.rules.empty_result()

    def assemble_result(self, **kwargs: Any) -> pd.DataFrame:
        return self.rules.assemble_result(**kwargs)


class LowEntryScoreV5Strategy(Strategy):
    """Plugin facade that delegates all model behavior to a decision pipeline."""

    name = LOW_ENTRY_STRATEGY_NAME
    description = "Quality-first panic reversal entry strategy with Base/Extended Quality gates, valuation DCF, and breadth-recovery exceptions"
    version = LES_V5_MODEL_VERSION

    def __init__(
        self,
        config: LowEntryScoreV5Config | None = None,
        valuation_producer: LowEntryV5ValuationProducer | None = None,
        *,
        components: LowEntryV5PipelineComponents | None = None,
        pipeline: LowEntryV5DecisionPipeline | None = None,
    ) -> None:
        if components is not None and pipeline is not None:
            raise ValueError("Provide either components or pipeline, not both")
        self.config = config or LowEntryScoreV5Config()
        self.valuation_producer = (
            valuation_producer
            or LowEntryV5ValuationProducer(self.config.valuation)
        )
        self.rule_engine: LowEntryV5RuleEngine | None = None
        if pipeline is not None:
            self.pipeline = pipeline
            return
        if components is None:
            self.rule_engine = LowEntryV5RuleEngine(
                self.config,
                self.valuation_producer,
            )
            components = LowEntryV5PipelineComponents(
                market=LowEntryV5MarketRules(self.rule_engine),
                fundamentals=LowEntryV5FundamentalRules(self.rule_engine),
                panic=LowEntryV5PanicRules(self.rule_engine),
                scoring=LowEntryV5ScoringRules(self.rule_engine),
                decision=LowEntryV5DecisionRules(self.rule_engine),
                report=LowEntryV5ReportRules(self.rule_engine),
            )
        self.pipeline = LowEntryV5DecisionPipeline(components)

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline.run(data)

    def run_with_features(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame:
        return self.pipeline.run_with_features(data, features)

    def _next_day_confirmation(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        falling_state: pd.Series,
    ) -> pd.Series:
        """Compatibility delegate for callers of the former private helper."""

        return self.pipeline.components.decision.next_day_confirmation(
            indicators,
            plan,
            falling_state,
        )


__all__ = [
    "FallingKnifeState",
    "LES_V5_MODEL_VERSION",
    "LES_V5_PARAMETER_VERSION",
    "LOW_ENTRY_OUTPUT_COLUMNS",
    "LOW_ENTRY_STRATEGY_NAME",
    "LowEntryV4MarketAdapter",
    "LowEntryV5DecisionPipeline",
    "LowEntryV5DecisionRules",
    "LowEntryV5FundamentalRules",
    "LowEntryV5MarketRules",
    "LowEntryV5PanicRules",
    "LowEntryV5PipelineComponents",
    "LowEntryV5ReportRules",
    "LowEntryV5RuleEngine",
    "LowEntryV5ScoringRules",
    "LowEntryScoreV5Config",
    "LowEntryScoreV5Strategy",
    "ValuationStatus",
]
