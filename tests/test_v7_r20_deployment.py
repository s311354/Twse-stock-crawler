"""Deployment is forward-only; candidate comparison is not historical rollout."""
from datetime import timedelta

import pytest

from application.les_v7_preflight import daily_v7_preflight
from domain.les_v7 import policy_r20
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.rule_baseline import (CURRENT_SOURCE_BASELINE_POLICY,
    baseline_policy, current_source_baseline_version, require_baseline_policy)
from tests.test_v7_default_baseline import configured


def test_daily_activation_preserves_earlier_vintage_and_frozen_policy_identity(tmp_path):
    at = timestamp(policy_r20.DAILY_ACTIVATED_AT)
    assert at > timestamp(policy_r20.AVAILABLE_AT)
    from domain.les_v7.policy_r25 import BASELINE_VERSION as LATEST
    assert CURRENT_SOURCE_BASELINE_POLICY == LATEST
    assert current_source_baseline_version(at - timedelta(microseconds=1)) == "V7_RULE_BASED_BASELINE_R19"
    assert current_source_baseline_version(at) == policy_r20.BASELINE_VERSION
    assert current_source_baseline_version("2026-10-27T13:30:00+08:00") == "V7_RULE_BASED_BASELINE_R19"
    fixed = baseline_policy(version=policy_r20.BASELINE_VERSION)
    assert identity(fixed) == "b5b37baac74dfb08f9e3dc6a9f015463286aae95c5eecaed206b54ef7dc7afff"
    response = daily_v7_preflight(configured(tmp_path), checked_at=at)
    assert response["state"] == "READY" and response["formal_entry_enabled"]
    assert response["rule_baseline_policy_id"] == identity(fixed)
    assert response["prediction_status"] == "NOT_APPLICABLE"
    assert not response["order_authority"]


def test_explicit_candidate_before_parameter_publication_is_rejected():
    fixed = baseline_policy(version=policy_r20.BASELINE_VERSION)
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(fixed, market="TWSE", cutoff=timestamp(policy_r20.AVAILABLE_AT)-timedelta(microseconds=1))
