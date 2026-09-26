"""Low Entry Model v5 quality-first panic reversal strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from domain.features import FeatureFrame
from domain.features import SharedFeatureConfig
from domain.quality_v5 import LowEntryV5QualityConfig
from domain.strategy import Strategy
from domain.strategies.low_entry_score_v4 import LowEntryScoreV4Config
from domain.strategies.low_entry_score_v4 import LowEntryScoreV4Strategy
from domain.strategies.low_entry_v5_contracts import BroadRallyTier
from domain.strategies.low_entry_v5_contracts import FallingKnifeState
from domain.strategies.low_entry_v5_contracts import LES_V5_MODEL_VERSION
from domain.strategies.low_entry_v5_contracts import LES_V5_PARAMETER_VERSION
from domain.strategies.low_entry_v5_contracts import LOW_ENTRY_STRATEGY_NAME
from domain.strategies.low_entry_v5_contracts import ValuationStatus
from domain.strategies.low_entry_v5_compat import (
    LowEntryV5CompatibilityFacade,
)
from domain.strategies.low_entry_v5_compat import LowEntryV5DecisionRules
from domain.strategies.low_entry_v5_compat import LowEntryV5FundamentalRules
from domain.strategies.low_entry_v5_compat import LowEntryV5MarketRules
from domain.strategies.low_entry_v5_compat import LowEntryV5PanicRules
from domain.strategies.low_entry_v5_compat import LowEntryV5ReportRules
from domain.strategies.low_entry_v5_compat import LowEntryV5ScoringRules
from domain.strategies.low_entry_v5_composition import LowEntryV5CompositionRoot
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
    # A second, stricter quality tier may arm the same conditional 1/8 starter
    # on a quiet absorption bar.  The relaxed CLV/volume floors are not an
    # immediate entry: the order remains a next-session stop above the entire
    # signal bar, and every valuation, Falling-Knife and execution-risk gate
    # remains binding.
    high_quality_reversal_minimum_quality_score: float = 85.0
    high_quality_reversal_minimum_clv: float = -0.10
    high_quality_reversal_minimum_volume_ratio: float = 0.25
    high_quality_reversal_maximum_price_position: float = 0.40
    high_quality_reversal_maximum_drawdown: float = -0.08
    high_quality_reversal_minimum_reversals: int = 2
    # Electronics can make a useful local pullback while remaining above the
    # lower half of their 60-day range. This path requires a healthy medium
    # trend and an actual reclaim, not just a cheaper price or broad rally.
    # Only a conditional 1/8 starter is possible; the signal-bar plan is
    # rechecked against the ordinary RR/ATR/liquidity limits before selection.
    elec_trend_pullback_enabled: bool = True
    elec_trend_pullback_minimum_quality_score: float = 80.0
    elec_trend_pullback_score_threshold: float = 55.0
    elec_trend_pullback_minimum_raw_score: float = 60.0
    elec_trend_pullback_drawdown_floor: float = -0.15
    elec_trend_pullback_drawdown_ceiling: float = -0.03
    elec_trend_pullback_minimum_ema20_atr: float = -1.0
    elec_trend_pullback_maximum_ema20_atr: float = 1.0
    elec_trend_pullback_minimum_clv: float = 0.20
    elec_trend_pullback_minimum_volume_ratio: float = 0.50
    # Airlines can turn before backward-looking demand and momentum recover.
    # A weak low-zone session only arms a watch; a 1/8 BUY_SMALL still needs
    # next-session confirmation and every ordinary valuation/risk hard gate.
    air_cycle_minimum_quality_score: float = 85.0
    air_cycle_score_threshold: float = 35.0
    air_cycle_minimum_raw_score: float = 45.0
    air_cycle_maximum_price_position: float = 0.35
    air_cycle_maximum_drawdown: float = -0.05
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
    "Industry Quality Profile",
    "Industry Profile Version",
    "Industry Profile Mode",
    "Industry Shadow Decision",
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


class LowEntryV5RuleEngine(LowEntryV5CompatibilityFacade):
    """Compatibility facade; production runs use components directly."""

    def __init__(
        self,
        config: LowEntryScoreV5Config | None = None,
        valuation_producer: LowEntryV5ValuationProducer | None = None,
        market_adapter: LowEntryV4MarketAdapter | None = None,
        *,
        components: LowEntryV5PipelineComponents | None = None,
    ) -> None:
        resolved_config = config or LowEntryScoreV5Config()
        resolved_valuation = (
            valuation_producer
            or LowEntryV5ValuationProducer(resolved_config.valuation)
        )
        resolved_market = market_adapter or LowEntryV4MarketAdapter(
            resolved_config
        )
        resolved_components = components or LowEntryV5CompositionRoot(
            config=resolved_config,
            valuation_producer=resolved_valuation,
            market_adapter=resolved_market,
            output_columns=LOW_ENTRY_OUTPUT_COLUMNS,
        ).build()
        super().__init__(
            resolved_components,
            config=resolved_config,
            valuation_producer=resolved_valuation,
            market_adapter=resolved_market,
        )


class LowEntryScoreV5Strategy(Strategy):
    """Plugin facade that delegates all behavior to composed components."""

    name = LOW_ENTRY_STRATEGY_NAME
    description = (
        "Quality-first panic reversal entry strategy with Base/Extended "
        "Quality gates, valuation DCF, and breadth-recovery exceptions"
    )
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
        self._rule_engine: LowEntryV5RuleEngine | None = None
        self._market_adapter: LowEntryV4MarketAdapter | None = None
        if pipeline is not None:
            self.pipeline = pipeline
            return
        if components is None:
            self._market_adapter = LowEntryV4MarketAdapter(self.config)
            components = LowEntryV5CompositionRoot(
                config=self.config,
                valuation_producer=self.valuation_producer,
                market_adapter=self._market_adapter,
                output_columns=LOW_ENTRY_OUTPUT_COLUMNS,
            ).build()
        self.pipeline = LowEntryV5DecisionPipeline(components)

    @property
    def rule_engine(self) -> LowEntryV5RuleEngine:
        """Build the legacy facade only when a compatibility caller needs it."""
        if self._rule_engine is None:
            market_adapter = self._market_adapter or LowEntryV4MarketAdapter(
                self.config
            )
            self._rule_engine = LowEntryV5RuleEngine(
                self.config,
                self.valuation_producer,
                market_adapter,
                components=self.pipeline.components,
            )
        return self._rule_engine

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
