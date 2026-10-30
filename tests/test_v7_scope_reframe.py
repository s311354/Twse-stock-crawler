"""Offline synthetic evidence tests; no market or model certification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from domain.les_v7.contracts import identity
from infrastructure.les_v7_dataset_scope import _load_bound_scope_shard, _verify_freeze
from infrastructure.les_v7_scope_reframe import reframe_elec_scope
from tests.test_v7_dataset_scope import scope_inputs, freeze


def test_same_history_reframe_exact_union_and_idempotent_without_rewriting_parent(tmp_path):
    inputs = scope_inputs(tmp_path, count=7)
    seed = freeze(inputs, batch_size=3)
    original = {str(p): p.read_bytes() for p in inputs['root'].rglob('*') if p.is_file()}
    build = dict(inputs['config'], event_start='2026-07-30', max_events=14)
    root = tmp_path / 'reframed'
    path = reframe_elec_scope(seed, build, root, allow_synthetic=True)
    record = _verify_freeze(path, root, verify_sources=True)
    body = record['body']
    parent = json.loads(seed.read_bytes())['body']
    assert body['batch_size'] == 2
    for field in ('universe', 'universe_record', 'calendar_sha256', 'history_start', 'history_end'):
        assert body[field] == parent[field]
    assert body['reframe_parent']['build_config_id'] == identity(build)
    assert [c['symbols'] for c in body['chunks']] == [['1000', '1001'], ['1002', '1003'], ['1004', '1005'], ['1006']]
    for chunk in body['chunks']:
        _, _, payload = _load_bound_scope_shard(body, root, chunk)
        assert set(payload['filings']) == set(chunk['symbols'])
        assert sorted(s for p in payload['training_extension_parents'] for s in p['symbols']) == chunk['symbols']
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert reframe_elec_scope(seed, build, root, allow_synthetic=True) == path
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert all(Path(name).read_bytes() == value for name, value in original.items())


def test_invalid_window_or_synthetic_opt_in_cannot_publish(tmp_path):
    inputs = scope_inputs(tmp_path, count=2)
    seed = freeze(inputs)
    root = tmp_path / 'reframed'
    with pytest.raises(ValueError, match='EXPLICIT_OPT_IN'):
        reframe_elec_scope(seed, inputs['config'], root)
    build = deepcopy(inputs['config']); build['event_start'] = '2026-07-29'
    with pytest.raises(ValueError, match='OUTSIDE_HISTORY'):
        reframe_elec_scope(seed, build, root, allow_synthetic=True)
    assert not root.exists()


def test_corrupt_parent_source_is_rejected_before_publication(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    seed = freeze(inputs)
    body = json.loads(seed.read_bytes())['body']
    path = inputs['root'] / 'evidence' / body['chunks'][0]['evidence_id'] / 'evidence.json'
    path.write_text('{}')
    root = tmp_path / 'reframed'
    with pytest.raises((ValueError, KeyError)):
        reframe_elec_scope(seed, inputs['config'], root, allow_synthetic=True)
    assert not root.exists()
