"""Forward-only deployment; historical cutoffs and frozen identities retained."""
from datetime import timedelta
from unittest.mock import patch

from application.les_v7_preflight import daily_v7_preflight
from domain.les_v7 import policy_r25
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.rule_baseline import (CURRENT_SOURCE_BASELINE_POLICY, baseline_policy,
    current_source_baseline_version)
from tests.test_v7_default_baseline import configured


def test_explicitly_disabled_r25_falls_back_to_deployed_r24():
    with patch.object(policy_r25, "DAILY_ACTIVATED_AT", None):
        assert current_source_baseline_version("2030-01-01T00:00:00Z") == "V7_RULE_BASED_BASELINE_R24"


def test_forward_activation_and_config_check_do_not_retroactively_change_old_cutoffs(tmp_path):
    at = timestamp(policy_r25.DAILY_ACTIVATED_AT)
    assert at > timestamp(policy_r25.AVAILABLE_AT)
    assert CURRENT_SOURCE_BASELINE_POLICY == policy_r25.BASELINE_VERSION
    assert current_source_baseline_version(at - timedelta(microseconds=1)) == "V7_RULE_BASED_BASELINE_R24"
    assert current_source_baseline_version(at) == policy_r25.BASELINE_VERSION
    assert current_source_baseline_version("2026-10-27T13:30:00+08:00") == "V7_RULE_BASED_BASELINE_R19"
    fixed = baseline_policy(version=policy_r25.BASELINE_VERSION)
    assert identity(fixed) == "60505422cd352df274fa795c3ca1333bcd455064dbbace228bf8b6b93aeb0ac5"
    result = daily_v7_preflight(configured(tmp_path), checked_at=at)
    assert result["state"] == "READY" and result["rule_baseline_policy_id"] == identity(fixed)
    assert result["rule_baseline_policy"]["unified_technical_policy"]["small_minimum"] == 55
    assert result["prediction_status"] == "NOT_APPLICABLE" and not result["order_authority"]
    assert identity(baseline_policy(version="V7_RULE_BASED_BASELINE_R24")) == "659a496973b6de40543e92bedec5d3dc1b122ea69998e43e7aa1998609ab32bc"
