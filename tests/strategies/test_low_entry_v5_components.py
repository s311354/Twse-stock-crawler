"""Focused contracts for decomposed Low Entry v5 components."""

import pandas as pd

from domain.quality_v5 import FinalQualityStatus
from domain.strategies.low_entry_score_v5 import LowEntryV5MarketRules
from domain.strategies.low_entry_score_v5 import LowEntryV5RuleEngine
from domain.strategies.low_entry_score_v5 import LowEntryScoreV5Strategy
from domain.strategies.low_entry_v5_decision import LowEntryV5DecisionEvaluator
from domain.strategies.low_entry_v5_fundamentals import (
    LowEntryV5FundamentalEvaluator,
)
from domain.strategies.low_entry_v5_market import LowEntryV5MarketEvaluator
from domain.strategies.low_entry_v5_panic import LowEntryV5PanicEvaluator
from domain.strategies.low_entry_v5_report import LowEntryV5ReportBuilder
from domain.strategies.low_entry_v5_scoring import LowEntryV5ScoringEvaluator


def test_default_pipeline_uses_independent_composable_components(
) -> None:
    strategy = LowEntryScoreV5Strategy()

    assert strategy._rule_engine is None
    strategy.run(pd.DataFrame())
    assert strategy._rule_engine is None
    assert isinstance(
        strategy.pipeline.components.market,
        LowEntryV5MarketEvaluator,
    )
    assert isinstance(
        strategy.pipeline.components.fundamentals,
        LowEntryV5FundamentalEvaluator,
    )
    assert isinstance(
        strategy.pipeline.components.panic,
        LowEntryV5PanicEvaluator,
    )
    assert isinstance(
        strategy.pipeline.components.scoring,
        LowEntryV5ScoringEvaluator,
    )
    assert isinstance(
        strategy.pipeline.components.decision,
        LowEntryV5DecisionEvaluator,
    )
    assert isinstance(
        strategy.pipeline.components.report,
        LowEntryV5ReportBuilder,
    )
    compatibility = strategy.rule_engine
    assert compatibility is not None
    assert strategy.pipeline.components.market is compatibility.market_evaluator
    assert (
        strategy.pipeline.components.fundamentals
        is compatibility.fundamental_evaluator
    )
    assert strategy.pipeline.components.panic is compatibility.panic_evaluator
    assert (
        strategy.pipeline.components.scoring
        is compatibility.scoring_evaluator
    )
    assert (
        strategy.pipeline.components.decision
        is compatibility.decision_evaluator
    )
    assert strategy.pipeline.components.report is compatibility.report_builder


def test_rule_engine_preserves_fundamental_compatibility_delegate() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    data = pd.DataFrame({"Date": ["2026-08-22"]})

    direct = strategy.rule_engine.fundamental_evaluator.evaluate_quality(
        data,
        {},
    )
    compatibility = strategy.rule_engine.evaluate_quality(data, {})

    assert compatibility == direct
    assert compatibility.final_status is FinalQualityStatus.DATA_INSUFFICIENT


def test_rule_engine_preserves_scoring_compatibility_delegate() -> None:
    strategy = LowEntryScoreV5Strategy()
    assert strategy.rule_engine is not None
    indicators = pd.DataFrame(
        {
            "ema20_slope": [-0.1, 0.1],
            "rs20": [-0.2, 0.1],
            "rs20_slope": [-0.1, 0.1],
            "taiex": [90.0, 110.0],
            "taiex_ema20": [100.0, 100.0],
            "taiex_ema20_slope": [-1.0, 1.0],
        }
    )

    direct_penalty, direct_risk_off = (
        strategy.rule_engine.scoring_evaluator.falling_knife_penalty(
            indicators
        )
    )
    compatibility_penalty, compatibility_risk_off = (
        strategy.rule_engine.falling_knife_penalty(indicators)
    )

    pd.testing.assert_series_equal(compatibility_penalty, direct_penalty)
    pd.testing.assert_series_equal(compatibility_risk_off, direct_risk_off)
    assert compatibility_penalty.tolist() == [-10, 0]
    assert compatibility_risk_off.tolist() == [True, False]


def test_legacy_rule_engine_and_rules_wrapper_remain_import_compatible() -> None:
    compatibility = LowEntryV5RuleEngine()
    market_rules = LowEntryV5MarketRules(compatibility)

    assert market_rules.shared_feature_config() == (
        compatibility.market_evaluator.shared_feature_config()
    )
    assert compatibility.empty_result()["低點決策"] == "DATA_INSUFFICIENT"
