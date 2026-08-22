"""Low Entry Model v5 quality-first panic reversal strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from domain.features import FeatureFrame
from domain.features import SharedFeatureConfig
from domain.quality_v5 import LowEntryV5QualityConfig
from domain.quality_v5 import LowEntryV5QualityResult
from domain.strategy import Strategy
from domain.strategies.low_entry_score_v4 import LowEntryScoreV4Config
from domain.strategies.low_entry_score_v4 import LowEntryScoreV4Strategy
from domain.strategies.low_entry_v5_contracts import BroadRallyTier
from domain.strategies.low_entry_v5_contracts import FallingKnifeState
from domain.strategies.low_entry_v5_contracts import LES_V5_MODEL_VERSION
from domain.strategies.low_entry_v5_contracts import LES_V5_PARAMETER_VERSION
from domain.strategies.low_entry_v5_contracts import LOW_ENTRY_STRATEGY_NAME
from domain.strategies.low_entry_v5_contracts import ValuationStatus
from domain.strategies.low_entry_v5_decision import LowEntryV5DecisionEvaluator
from domain.strategies.low_entry_v5_fundamentals import (
    LowEntryV5FundamentalEvaluator,
)
from domain.strategies.low_entry_v5_market import LowEntryV5MarketEvaluator
from domain.strategies.low_entry_v5_panic import LowEntryV5PanicEvaluator
from domain.strategies.low_entry_v5_panic import LowEntryV5QualityEligibility
from domain.strategies.low_entry_v5_pipeline import LowEntryV5DecisionPipeline
from domain.strategies.low_entry_v5_pipeline import LowEntryV5PipelineComponents
from domain.strategies.low_entry_v5_report import LowEntryV5ReportBuilder
from domain.strategies.low_entry_v5_scoring import LowEntryV5ScoringEvaluator
from domain.valuation_v5 import LowEntryV5ValuationConfig
from domain.valuation_v5 import LowEntryV5ValuationProducer
from domain.valuation_v5 import LowEntryV5ValuationResult

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
    breadth_rally_position_fraction: float = 1 / 8
    # Outside a Broad Rally, SUPPORTIVE valuation may authorize only a
    # conditional 1/8 starter for a fully confirmed high-quality company.
    # Score, Trigger, Falling-Knife and both execution-risk gates remain hard.
    supportive_non_broad_minimum_quality_score: float = 80.0
    supportive_non_broad_score_threshold: float = 75.0
    # A fully confirmed company may arm a conditional 1/8 starter before the
    # ordinary 75-point BUY threshold. This is a narrowly relaxed path for an
    # orderly high-quality pullback, not a score-only override: valuation must
    # be at least SUPPORTIVE, the Falling-Knife state must be CLEAR, both risk
    # gates and market data must pass, and the next session must still trade
    # above the signal-day high before execution.
    quality_pullback_minimum_quality_score: float = 75.0
    quality_pullback_score_threshold: float = 40.0
    quality_pullback_minimum_raw_score: float = 50.0
    quality_pullback_minimum_clv: float = 0.20
    quality_pullback_minimum_volume_ratio: float = 0.50
    quality_pullback_maximum_price_position: float = 0.50
    quality_pullback_maximum_drawdown: float = -0.05
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
    electronics_moderate_breadth_minimum_advancer_ratio: float = 0.65
    electronics_moderate_breadth_minimum_median_return: float = 0.0075
    semiconductor_breadth_minimum_members: int = 40
    semiconductor_breadth_minimum_advancer_ratio: float = 0.85
    semiconductor_breadth_minimum_median_return: float = 0.020
    semiconductor_moderate_breadth_minimum_advancer_ratio: float = 0.75
    semiconductor_moderate_breadth_minimum_median_return: float = 0.0125
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
    "Valuation Confidence",
    "Valuation Normalized EPS",
    "Valuation TTM EPS",
    "Normalized EPS Divergence Ratio",
    "Cycle Valuation Review",
    "Cycle Valuation Reasons",
    "Cycle Mid-Cycle Operating Margin",
    "Cycle Normalized ROE",
    "Cycle Book Value Per Share",
    "Cycle Margin Fair Value",
    "Cycle PB Fair Value",
    "Cycle TTM EPS Fair Value",
    "Cycle Composite Fair Value",
    "Cycle Available Anchors",
    "Cycle Supporting Anchors",
    "Passive Benchmark Expected Return",
    "Required Valuation Return",
    "Base Case Revenue Growth",
    "Base Case Operating Margin",
    "Base Case Exit PE",
    "Base Case Expected CAGR",
    "Base Case Fair Value",
    "Bear Case Expected CAGR",
    "Bear Case Fair Value",
    "Bear Case Permanent Loss",
    "Bear Case Permanent Loss Acceptable",
    "Bull Case Expected CAGR",
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
    "Broad Rally Tier",
    "Broad Rally Regime",
    "Broad Rally Trigger",
    "Broad Rally Full BUY Trigger",
    "Broad Rally Absorption Trigger",
    "Supportive Non-Broad Trigger",
    "Quality Pullback Starter Trigger",
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
    """Compatibility facade over the composable LES v5 decision components."""

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
        self.quality_policy = LowEntryV5QualityEligibility(resolved_config)
        self.market_evaluator = LowEntryV5MarketEvaluator(
            resolved_config,
            self.market_adapter,
        )
        self.fundamental_evaluator = LowEntryV5FundamentalEvaluator(
            resolved_config.v5_quality,
            self.valuation_producer,
            self.market_adapter.positive_float,
        )
        self.panic_evaluator = LowEntryV5PanicEvaluator(
            resolved_config,
            self.quality_policy,
        )
        self.scoring_evaluator = LowEntryV5ScoringEvaluator(resolved_config)
        self.decision_evaluator = LowEntryV5DecisionEvaluator(
            resolved_config,
            self.market_adapter,
            self.quality_policy,
        )
        self.report_builder = LowEntryV5ReportBuilder(LOW_ENTRY_OUTPUT_COLUMNS)

    @staticmethod
    def _quality_metric_passed(
        quality: LowEntryV5QualityResult,
        metric_name: str,
    ) -> bool:
        return LowEntryV5QualityEligibility.quality_metric_passed(
            quality,
            metric_name,
        )

    def _conditional_quality_starter_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        return self.quality_policy.conditional_quality_starter_eligible(quality)

    def _broad_rally_quality_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        return self.quality_policy.broad_rally_quality_eligible(quality)

    def _broad_rally_full_buy_quality_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        return self.quality_policy.broad_rally_full_buy_quality_eligible(quality)

    def _supportive_non_broad_quality_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        return self.quality_policy.supportive_non_broad_quality_eligible(quality)

    def _quality_pullback_starter_eligible(
        self,
        quality: LowEntryV5QualityResult,
    ) -> bool:
        return self.quality_policy.quality_pullback_starter_eligible(quality)

    def shared_feature_config(self) -> SharedFeatureConfig:
        return self.market_evaluator.shared_feature_config()

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame:
        return self.market_evaluator.calculate_indicators(data, features)

    def evaluate_market_data(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, pd.Series]:
        return self.market_evaluator.evaluate_market_data(
            data,
            indicators,
            context,
        )

    def build_trade_plan(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        return self.market_evaluator.build_trade_plan(indicators, context)

    def build_broad_rally_plan(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        return self.market_evaluator.build_broad_rally_plan(
            indicators,
            plan,
            context,
        )

    def evaluate_risk_gate(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        market_gate: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        return self.market_evaluator.evaluate_risk_gate(
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
        return self.market_evaluator.evaluate_broad_rally_risk_gate(
            indicators,
            plan,
            market_gate,
        )

    @staticmethod
    def _next_twse_stock_tick(
        prices: pd.Series,
        *,
        price_scale: float,
    ) -> pd.Series:
        return LowEntryV5MarketEvaluator._next_twse_stock_tick(
            prices,
            price_scale=price_scale,
        )

    @staticmethod
    def _attach_breadth_facts(
        indicators: pd.DataFrame,
        snapshot: dict[str, object],
    ) -> None:
        LowEntryV5MarketEvaluator._attach_breadth_facts(indicators, snapshot)

    def _evaluate_breadth_rally_regime(self, *args: Any, **kwargs: Any) -> pd.Series:
        return self.market_evaluator._evaluate_breadth_rally_regime(
            *args,
            **kwargs,
        )

    def _evaluate_breadth_rally_tier(self, *args: Any, **kwargs: Any) -> pd.Series:
        return self.market_evaluator._evaluate_breadth_rally_tier(
            *args,
            **kwargs,
        )

    def _evaluate_recent_breadth_regime(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[pd.Series, pd.Series]:
        return self.market_evaluator._evaluate_recent_breadth_regime(
            *args,
            **kwargs,
        )

    def evaluate_quality(
        self,
        data: pd.DataFrame,
        context: dict[str, Any],
    ) -> LowEntryV5QualityResult:
        return self.fundamental_evaluator.evaluate_quality(data, context)

    def evaluate_valuation(
        self,
        indicators: pd.DataFrame,
        context: dict[str, Any],
    ) -> tuple[pd.Series, LowEntryV5ValuationResult]:
        return self.fundamental_evaluator.evaluate_valuation(
            indicators,
            context,
        )

    def evaluate_panic(
        self,
        indicators: pd.DataFrame,
        quality: LowEntryV5QualityResult,
        valuation: pd.Series,
    ) -> dict[str, pd.Series]:
        return self.panic_evaluator.evaluate_panic(
            indicators,
            quality,
            valuation,
        )

    def score_components(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        valuation: pd.Series,
    ) -> pd.DataFrame:
        return self.scoring_evaluator.score_components(
            indicators,
            plan,
            valuation,
        )

    def falling_knife_penalty(
        self,
        indicators: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        return self.scoring_evaluator.falling_knife_penalty(indicators)

    def required_score(
        self,
        quality: LowEntryV5QualityResult,
        risk_off: pd.Series,
        index: pd.Index,
    ) -> pd.Series:
        return self.scoring_evaluator.required_score(
            quality,
            risk_off,
            index,
        )

    def evaluate_trigger(self, indicators: pd.DataFrame) -> pd.Series:
        return self.decision_evaluator.evaluate_trigger(indicators)

    def next_day_confirmation(
        self,
        indicators: pd.DataFrame,
        plan: pd.DataFrame,
        falling_state: pd.Series,
    ) -> pd.Series:
        return self.decision_evaluator.next_day_confirmation(
            indicators,
            plan,
            falling_state,
        )

    def resolve_decision(self, **kwargs: Any) -> pd.Series:
        return self.decision_evaluator.resolve_decision(**kwargs)

    def position_fraction(
        self,
        decision: pd.Series,
        quality: LowEntryV5QualityResult,
        risk_off: pd.Series,
        broad_rally_full_buy: pd.Series | None = None,
    ) -> pd.Series:
        return self.decision_evaluator.position_fraction(
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
        return self.decision_evaluator.cap_position_shares(
            plan,
            position_fraction,
            context,
        )

    def enforce_executable_position(
        self,
        decision: pd.Series,
        recommended_shares: pd.Series,
    ) -> pd.Series:
        return self.decision_evaluator.enforce_executable_position(
            decision,
            recommended_shares,
        )

    @staticmethod
    def scenario_value(
        result: LowEntryV5ValuationResult,
        scenario_name: str,
        field_name: str,
    ) -> object:
        return LowEntryV5ReportBuilder.scenario_value(
            result,
            scenario_name,
            field_name,
        )

    @staticmethod
    def cycle_review_value(review: object, field_name: str) -> object:
        return LowEntryV5ReportBuilder.cycle_review_value(review, field_name)

    @staticmethod
    def valuation_confidence(
        quality: LowEntryV5QualityResult,
        result: LowEntryV5ValuationResult,
    ) -> float:
        return LowEntryV5ReportBuilder.valuation_confidence(quality, result)

    def build_reasons(self, *args: Any, **kwargs: Any) -> pd.Series:
        return self.report_builder.build_reasons(*args, **kwargs)

    def assemble_result(self, **kwargs: Any) -> pd.DataFrame:
        return self.report_builder.assemble_result(**kwargs)

    def empty_result(self) -> dict[str, object]:
        return self.report_builder.empty_result()

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
                market=self.rule_engine.market_evaluator,
                fundamentals=self.rule_engine.fundamental_evaluator,
                panic=self.rule_engine.panic_evaluator,
                scoring=self.rule_engine.scoring_evaluator,
                decision=self.rule_engine.decision_evaluator,
                report=self.rule_engine.report_builder,
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
    "BroadRallyTier",
    "FallingKnifeState",
    "LES_V5_MODEL_VERSION",
    "LES_V5_PARAMETER_VERSION",
    "LOW_ENTRY_OUTPUT_COLUMNS",
    "LOW_ENTRY_STRATEGY_NAME",
    "LowEntryV4MarketAdapter",
    "LowEntryV5DecisionPipeline",
    "LowEntryV5DecisionEvaluator",
    "LowEntryV5DecisionRules",
    "LowEntryV5FundamentalEvaluator",
    "LowEntryV5FundamentalRules",
    "LowEntryV5MarketEvaluator",
    "LowEntryV5MarketRules",
    "LowEntryV5PanicEvaluator",
    "LowEntryV5PanicRules",
    "LowEntryV5PipelineComponents",
    "LowEntryV5QualityEligibility",
    "LowEntryV5ReportBuilder",
    "LowEntryV5ReportRules",
    "LowEntryV5RuleEngine",
    "LowEntryV5ScoringEvaluator",
    "LowEntryV5ScoringRules",
    "LowEntryScoreV5Config",
    "LowEntryScoreV5Strategy",
    "ValuationStatus",
]
