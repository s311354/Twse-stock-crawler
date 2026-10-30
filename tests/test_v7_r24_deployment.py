"""Forward-only Quality-source deployment; historical policy IDs stay fixed."""
from datetime import timedelta
from unittest.mock import patch

import pytest

from application.les_v7_preflight import daily_v7_preflight
from domain.les_v7 import policy_r24
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.rule_baseline import (CURRENT_SOURCE_BASELINE_POLICY, baseline_policy,
    current_source_baseline_version, require_baseline_policy)
from tests.test_v7_default_baseline import configured


def test_candidate_not_active_does_not_change_the_deployed_source_path():
    with patch.object(policy_r24, "DAILY_ACTIVATED_AT", None), patch("domain.les_v7.policy_r25.DAILY_ACTIVATED_AT", None):
        assert current_source_baseline_version("2030-01-01T00:00:00Z") == "V7_RULE_BASED_BASELINE_R23"


def test_shipped_forward_activation_and_preflight_are_source_bound(tmp_path):
    at = timestamp(policy_r24.DAILY_ACTIVATED_AT)
    assert at > timestamp(policy_r24.AVAILABLE_AT)
    from domain.les_v7.policy_r25 import BASELINE_VERSION as LATEST
    assert CURRENT_SOURCE_BASELINE_POLICY == LATEST
    assert current_source_baseline_version(at - timedelta(microseconds=1)) == "V7_RULE_BASED_BASELINE_R23"
    assert current_source_baseline_version(at) == policy_r24.BASELINE_VERSION
    assert current_source_baseline_version("2026-10-27T13:30:00+08:00") == "V7_RULE_BASED_BASELINE_R19"
    policy = baseline_policy(version=policy_r24.BASELINE_VERSION)
    assert identity(policy) == "659a496973b6de40543e92bedec5d3dc1b122ea69998e43e7aa1998609ab32bc"
    result = daily_v7_preflight(configured(tmp_path), checked_at=at)
    assert result["state"] == "READY" and result["rule_baseline_policy_id"] == identity(policy)
    assert result["rule_baseline_policy"]["quality_policy"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R7"
    assert result["prediction_status"] == "NOT_APPLICABLE" and not result["order_authority"]
    assert identity(baseline_policy(version="V7_RULE_BASED_BASELINE_R23")) == "571cbf6eec65df7ed7849df4583fda0911609fbc480ef0b76fa12e1558987142"


def test_new_quality_parser_cannot_be_presented_as_an_earlier_cutoff_policy():
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=policy_r24.BASELINE_VERSION), market="TWSE",
            cutoff=timestamp(policy_r24.AVAILABLE_AT) - timedelta(microseconds=1))
