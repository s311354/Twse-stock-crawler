"""Synthetic composition/causality checks, not forecasting evidence."""
from copy import deepcopy

import pandas as pd
import pytest

from domain.les_v7 import execution_combined_features as combined
from domain.les_v7 import execution_features as gaps, execution_atom_features as atoms
from domain.les_v7.contracts import identity
from domain.les_v7.execution_profile import validate_execution_profile, execution_feature_provider, execution_base_count
from domain.les_v7.fractals import is_active_fractal_feature
from tests.strategies.test_v7_execution_atom_features import fixture


def test_composition_matches_both_original_estimators_exactly_and_excludes_future_bars():
    data, base, plan, cutoff = fixture(); before = data.copy(deep=True), deepcopy(base)
    values, audit = combined.compute_execution_features(data, cutoff=cutoff, base_features=base, plan=plan)
    one = gaps.compute_execution_features(data, cutoff=cutoff, base_features=base, plan=plan)[0]
    two = atoms.compute_execution_features(data, cutoff=cutoff, base_features=base, plan=plan)[0]
    assert values == {name: dict(one, **two)[name] for name in combined.FEATURE_NAMES}
    assert audit['status'] == 'COMPUTED' and len(values) == 27
    assert {name for name in values if is_active_fractal_feature(name)} == set(combined.FRACTAL_DERIVED_NAMES)
    future = data.iloc[[-1]].copy(); future['Date'] = '2026-09-15'; future['Available At'] = '2026-09-15T13:30:00+08:00'
    future[['Open', 'High', 'Low', 'Close']] = [500., 1000., .01, 700.]
    assert combined.compute_execution_features(pd.concat([data, future], ignore_index=True), cutoff=cutoff,
        base_features=base, plan=plan) == (values, audit)
    pd.testing.assert_frame_equal(data, before[0]); assert base == before[1]


@pytest.mark.parametrize('interactions', [False, True])
def test_exact_registered_partition_and_ablation_protect_base_heads(interactions):
    names = list(combined.FEATURE_NAMES) if interactions else [n for n in combined.FEATURE_NAMES if n not in combined.FRACTAL_DERIVED_NAMES]
    base = ['return20', 'dfa512']
    bounds = dict(zip(combined.FEATURE_NAMES, combined.FEATURE_BOUNDS))
    config = dict(feature_names=base + names, feature_bounds=[[-10, 10], [0, 3]] + [list(bounds[n]) for n in names],
        execution_feature_profile=combined.profile(), fill_only_feature_extension=dict(
            base_feature_names=base, profile_id=identity(combined.profile()), selected_features=names))
    assert validate_execution_profile(config) == (base, names)
    assert execution_feature_provider(config) is combined
    assert execution_base_count(config, config['feature_names']) == 2
    assert execution_base_count(config, [n for n in config['feature_names'] if not is_active_fractal_feature(n)]) == 1
    changed = deepcopy(config); changed['execution_feature_profile']['component_profiles'][1]['jeffreys_alpha'] = 1.
    with pytest.raises(ValueError, match='PROFILE_MISMATCH'): validate_execution_profile(changed)


def test_missing_history_and_missing_plan_do_not_invent_a_combined_feature():
    data, base, plan, cutoff = fixture()
    _, audit = combined.compute_execution_features(data.iloc[-120:], cutoff=cutoff, base_features=base, plan=plan)
    assert audit['status'] == 'NOT_READY'
    values, audit = combined.compute_execution_features(data, cutoff=cutoff, base_features=base, plan=None)
    assert audit['status'] == 'NOT_READY' and all(value is None for value in values.values())
