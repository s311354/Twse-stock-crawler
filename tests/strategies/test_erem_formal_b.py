"""Synthetic adoption/equivalence tests; not evidence of future profitability."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.strategy_contracts import StrategyDecisionAssembler
from domain.features import SharedFeaturePipeline
from domain.model_contracts import FieldSource
from domain.strategies.erem_soft_reduction import (
    FORMAL_VERSION, SoftReductionFormalPolicy, SoftReductionResearchPolicy, apply_soft_reduction_policy,
)
from domain.strategies.expected_return_exit_b import (
    DAILY_EREM_MODEL_VERSION, DAILY_EREM_OUTPUT_COLUMNS, DAILY_EREM_STRATEGY_NAME,
    ExpectedReturnExitBStrategy,
)
from domain.strategies.expected_return_exit_v3 import ExpectedReturnExitV3Strategy
from domain.strategy import get_strategy
from infrastructure.replay_runtime import parameter_manifest, current_versions
from tests.strategies.test_expected_return_exit_v3 import make_history


@pytest.mark.parametrize('commands', [
    ['REDUCE 25%'] * 8, ['REDUCE 50%', 'EXIT', 'HOLD', 'REDUCE 30%', 'REDUCE 30%'],
    ['UNKNOWN', 'HOLD_REVIEW', 'EXIT', 'REDUCE 25%'],
])
def test_b_arithmetic_is_exactly_r1_and_exit_remains_immediate(commands):
    n = len(commands)
    inputs = dict(decisions=pd.Series(commands), entry_les=pd.Series([70.]*n),
        holding_days=pd.Series(list(map(float, range(n)))), entry_decision=pd.Series(['BUY_SMALL']*n),
        entry_grace=pd.Series([False]*n))
    b = apply_soft_reduction_policy(**inputs, policy=SoftReductionFormalPolicy())
    r1 = apply_soft_reduction_policy(**inputs, policy=SoftReductionResearchPolicy())
    pd.testing.assert_frame_equal(b, r1, check_exact=True)
    for i, command in enumerate(commands):
        if command == 'EXIT': assert b.decision.iloc[i] == 'EXIT'


def test_formal_plugin_keeps_native_score_and_exact_research_business_outputs(monkeypatch):
    def commands(self, score, *args, **kwargs):
        result = pd.Series('HOLD', index=score.index)
        result.iloc[-8:] = ['REDUCE 25%'] * 7 + ['EXIT']
        return result
    monkeypatch.setattr(ExpectedReturnExitV3Strategy, '_decision', commands)
    history = make_history()
    formal = ExpectedReturnExitBStrategy()
    research = ExpectedReturnExitV3Strategy(soft_reduction_policy=SoftReductionResearchPolicy()).run(history)
    result = formal.run(history)
    for column in research:
        if column != 'EREM Research Policy':
            pd.testing.assert_series_equal(result[column], research[column], check_exact=True)
    assert result['EREM 決策訊號'].iloc[-1] == 'EXIT'
    assert 'EREM Research Policy' not in result
    record = json.loads(result['EREM Exit Policy'].iloc[-1])
    assert record['research_only'] is False and record['performance_validated'] is False
    assert record['residual_full_exit'] is True
    assert set(DAILY_EREM_OUTPUT_COLUMNS) <= set(result)
    features = formal.run(history)  # A repeated run has no retained warning/cooldown state.
    pd.testing.assert_frame_equal(result, features)
    shared = SharedFeaturePipeline(formal._shared_feature_config()).build(history)
    pd.testing.assert_frame_equal(result.tail(1), formal.run_latest_with_features(history, shared))


def test_daily_entrypoint_and_field_versions_use_b_but_native_plugin_is_preserved():
    assert get_strategy(DAILY_EREM_STRATEGY_NAME).version == DAILY_EREM_MODEL_VERSION
    assert get_strategy('expected_return_exit_v3').version == '3.2.4'
    evaluate = Mock(return_value=pd.DataFrame())
    builder = AnalysisDatasetBuilder(strategy_execution=SimpleNamespace(evaluate=evaluate))
    builder.evaluate_expected_return_exit(None)
    assert evaluate.call_args.args[:2] == (DAILY_EREM_STRATEGY_NAME, DAILY_EREM_OUTPUT_COLUMNS)
    assert StrategyDecisionAssembler._model_version_for('EREM 分數',
        {'EREM Model Version': DAILY_EREM_MODEL_VERSION}, FieldSource.EREM_V3) == DAILY_EREM_MODEL_VERSION
    assert parameter_manifest()['erem_exit_policy'] == SoftReductionFormalPolicy().record()
    assert DAILY_EREM_MODEL_VERSION in current_versions().model


def test_parameters_are_frozen_and_empty_history_cannot_look_complete():
    policy = SoftReductionFormalPolicy()
    for changes in ({'confirmation_observations': 1}, {'cooldown_observations': 0}, {'research_only': True}):
        with pytest.raises(ValueError): replace(policy, **changes)
    empty = ExpectedReturnExitBStrategy().run(pd.DataFrame()).iloc[0]
    assert empty['EREM Exit Policy Version'] == FORMAL_VERSION
    assert empty['EREM Reduction Status'] == 'UNKNOWN_EMPTY_INPUT'
