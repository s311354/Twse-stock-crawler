"""Owned batch-history reuse keeps the public causal selection boundaries."""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from domain.les_v7 import reference_memo, training
from domain.les_v7.features import select_bars
from domain.les_v7.reference_memo import NativeReferenceMemo, _bind_reference_history
from tests.test_v7_dataset_builder import builder_inputs


@pytest.fixture
def inputs():
    source, build = builder_inputs()
    return source['payload']['histories'][0]['bars'], build['seed'], ['SYNTHETIC_SOURCE_NOT_OFFICIAL']


def test_fast_reuse_is_exact_and_returns_detached_results(inputs, monkeypatch):
    bars, config, ids = inputs
    cutoff = '2026-01-05T13:30:00+08:00'
    original = training.prepare_reference('3014', 'TWSE', pd.DataFrame(bars), cutoff, config, ids)
    select = training.select_bars
    calls = []
    def counted(*args):
        calls.append(True)
        return select(*args)
    monkeypatch.setattr(training, 'select_bars', counted)
    with NativeReferenceMemo(max_entries=8) as memo:
        reference = _bind_reference_history(bars)
        assert reference('3014', 'TWSE', cutoff, config, ids) == original
        selected = _bind_reference_history(bars, selection_cutoff='2026-02-10T13:30:00+08:00')
        first = selected('3014', 'TWSE', cutoff, config, ids)
        first['features']['close'] = 999.
        first['feature_audit'].clear()
        assert selected('3014', 'TWSE', cutoff, config, ids) == original
        assert len(calls) == 1 and memo.stats['fast_hits'] == 2
        with pytest.raises(FrozenInstanceError):
            reference._source_json = 'changed'
    assert memo.stats['history_bytes'] == 0


def test_source_snapshot_is_owned_and_a_new_source_never_reuses_old_alias(inputs):
    bars, config, ids = inputs
    cutoff = '2026-01-05T13:30:00+08:00'
    with NativeReferenceMemo(max_entries=8) as memo:
        reference = _bind_reference_history(bars)
        original = reference('3014', 'TWSE', cutoff, config, ids)
        bars[539]['High'] += .1
        assert reference('3014', 'TWSE', cutoff, config, ids) == original
        changed = _bind_reference_history(bars)('3014', 'TWSE', cutoff, config, ids)
        assert changed['evidence_id'] != original['evidence_id']
        bars[539]['High'] = 1.
        with pytest.raises(ValueError, match='INVALID_OHLCV'):
            _bind_reference_history(bars)('3014', 'TWSE', cutoff, config, ids)
        assert memo.stats['fast_hits'] == 1


def test_late_invalid_bar_and_future_availability_are_not_pulled_into_prefix(inputs):
    bars, config, ids = inputs
    bars[540]['High'] = 1.
    bars[540]['Available At'] = '2026-01-08T13:30:00+08:00'
    with NativeReferenceMemo(max_entries=8):
        reference = _bind_reference_history(bars)
        for cutoff in ('2026-01-05T13:30:00+08:00', '2026-01-06T13:30:00+08:00'):
            expected = training.prepare_reference('3014', 'TWSE', pd.DataFrame(bars), cutoff, config, ids)
            assert reference('3014', 'TWSE', cutoff, config, ids) == expected
        with pytest.raises(ValueError, match='INVALID_OHLCV'):
            reference('3014', 'TWSE', '2026-01-08T13:30:00+08:00', config, ids)


def test_evaluation_ceiling_remains_distinct_for_post_evaluation_event(inputs):
    bars, config, ids = inputs
    upper = '2026-01-05T13:30:00+08:00'
    future = '2026-01-06T13:30:00+08:00'
    with NativeReferenceMemo(max_entries=8):
        raw = _bind_reference_history(bars)
        limited = _bind_reference_history(bars, selection_cutoff=upper)
        later = raw('3014', 'TWSE', future, config, ids)
        expected = training.prepare_reference('3014', 'TWSE', select_bars(pd.DataFrame(bars), upper), future, config, ids)
        actual = limited('3014', 'TWSE', future, config, ids)
        assert actual == expected and actual['evidence_id'] != later['evidence_id']
        assert actual['feature_audit']['cutoff_session'] == '2026-01-05'


def test_all_reference_key_inputs_remain_bound(inputs):
    bars, config, ids = inputs
    cutoff = '2026-01-05T13:30:00+08:00'
    with NativeReferenceMemo(max_entries=8):
        reference = _bind_reference_history(bars)
        base = reference('3014', 'TWSE', cutoff, config, ids)
        altered = deepcopy(config)
        altered['reference_policy']['version'] += ':REVISION'
        for symbol, market, when, selected, evidence in (
            ('5471', 'TWSE', cutoff, config, ids),
            ('3014', 'TPEX', cutoff, config, ids),
            ('3014', 'TWSE', '2026-01-05T14:30:00+08:00', config, ids),
            ('3014', 'TWSE', cutoff, config, ['DIFFERENT_SYNTHETIC_SOURCE']),
            ('3014', 'TWSE', cutoff, altered, ids),
        ):
            expected = training.prepare_reference(symbol, market, pd.DataFrame(bars), when, selected, evidence)
            assert reference(symbol, market, when, selected, evidence) == expected
            assert expected != base


def test_bounds_fall_back_without_new_training_rejection_and_lru_is_bounded(inputs, monkeypatch):
    bars, config, ids = inputs
    monkeypatch.setattr(reference_memo, 'MAX_REFERENCE_HISTORY_BYTES', 1)
    with NativeReferenceMemo(max_entries=1) as memo:
        assert _bind_reference_history(bars) is None
        assert memo.stats['history_bytes'] == 0
    monkeypatch.setattr(reference_memo, 'MAX_REFERENCE_HISTORY_BYTES', 32 * 1024 * 1024)
    with NativeReferenceMemo(max_entries=1) as memo:
        reference = _bind_reference_history(bars)
        first = reference('3014', 'TWSE', '2026-01-05T13:30:00+08:00', config, ids)
        reference('3014', 'TWSE', '2026-01-06T13:30:00+08:00', config, ids)
        assert len(memo._aliases) <= 1 and memo.stats['current_entries'] == 1
        assert reference('3014', 'TWSE', '2026-01-05T13:30:00+08:00', config, ids) == first
    assert reference('3014', 'TWSE', '2026-01-05T13:30:00+08:00', config, ids) == first
    assert memo.stats['current_entries'] == 0 and memo.stats['history_bytes'] == 0


def test_numeric_string_dtype_coercion_never_uses_a_prefix_alias(inputs):
    bars, config, ids = inputs
    for row in bars:
        row['Bid'] = '70'
    bars[-1]['Bid'] = '70.5'
    with NativeReferenceMemo(max_entries=8):
        assert _bind_reference_history(bars) is None
        assert _bind_reference_history(bars, selection_cutoff='2026-02-10T13:30:00+08:00') is None
        # Existing public adapters remain responsible for these valid inputs.
        result = training.prepare_reference('3014', 'TWSE', pd.DataFrame(bars),
            '2026-01-05T13:30:00+08:00', config, ids)
        assert result['evidence_id']
