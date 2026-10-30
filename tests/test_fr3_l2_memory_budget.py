"""Bounded new source-proof transport; never a larger market/sample budget."""
from copy import deepcopy

import pytest

from infrastructure import les_v7_scope_training as scope
from infrastructure.les_v7_fr3_training import prepare_fr3_scope
from tests.test_v7_fr3_training import inputs as r1_inputs
from tests.test_v7_fr3_l2_training import l2_configs
from tests.test_v7_training_publication import Guard


def test_old_limits_and_overrides_are_unchanged(r1_inputs, monkeypatch):
    config = r1_inputs[2]
    assert scope.MAX_PREPARED_DECODED_BYTES == scope.prepared_decoded_limit(config) == 256 * 1024**2
    assert scope.prepared_decoded_limit(config, legacy_limit=1) == 1
    monkeypatch.setattr(scope, "MAX_PREPARED_DECODED_BYTES", 2)
    assert scope.prepared_decoded_limit(config) == 2


def test_only_exact_verified_l2_has_a_distinct_finite_proof_budget(r1_inputs):
    _, config = l2_configs(r1_inputs[1], r1_inputs[2])
    assert scope.prepared_decoded_limit(config) == 384 * 1024**2
    assert scope.MAX_PREPARED_BYTES == 128 * 1024**2
    assert scope.MAX_PREPARED_ROWS == 100000
    before = deepcopy(config)
    broken = deepcopy(config)
    broken["reference_policy"]["structural_rules"]["maximum_stop_atr"] = 999.
    with pytest.raises(ValueError): scope.prepared_decoded_limit(broken)
    assert config == before


def test_l2_preparation_still_enforces_new_aggregate_bound(r1_inputs, tmp_path, monkeypatch):
    freeze, old_build, old_config, _ = r1_inputs
    build, config = l2_configs(old_build, old_config)
    monkeypatch.setattr(scope, "MAX_L2_PREPARED_DECODED_BYTES", 1)
    with pytest.raises(ValueError, match="PREPARED_BUDGET_EXCEEDED"):
        prepare_fr3_scope(freeze, build, config, tmp_path / "SYNTHETIC-L2-small-budget",
            allow_synthetic=True, guard_factory=Guard, clock=lambda: "2026-10-24T12:00:00Z")
