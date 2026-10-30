"""Forward-only native default; previous vintages and bundle paths are retained."""
from datetime import timedelta
from unittest.mock import patch

from application.les_v7_preflight import daily_v7_preflight
from domain.les_v7 import policy_r21
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.rule_baseline import baseline_policy, current_source_baseline_version, CURRENT_SOURCE_BASELINE_POLICY
from tests.test_v7_default_baseline import configured


def test_candidate_without_activation_does_not_change_daily_default():
    with patch.object(policy_r21, "DAILY_ACTIVATED_AT", None), patch("domain.les_v7.policy_r22.DAILY_ACTIVATED_AT", None), patch("domain.les_v7.policy_r23.DAILY_ACTIVATED_AT", None), patch("domain.les_v7.policy_r24.DAILY_ACTIVATED_AT", None), patch("domain.les_v7.policy_r25.DAILY_ACTIVATED_AT", None):
        assert current_source_baseline_version("2030-01-01T00:00:00Z") == "V7_RULE_BASED_BASELINE_R20"


def test_deployment_clock_and_retained_hashes_not_backdated(tmp_path):
    at = timestamp(policy_r21.DAILY_ACTIVATED_AT)
    assert at > timestamp(policy_r21.AVAILABLE_AT)
    from domain.les_v7.policy_r25 import BASELINE_VERSION as LATEST
    assert CURRENT_SOURCE_BASELINE_POLICY == LATEST
    assert current_source_baseline_version(at - timedelta(microseconds=1)) == "V7_RULE_BASED_BASELINE_R20"
    assert current_source_baseline_version(at) == policy_r21.BASELINE_VERSION
    assert current_source_baseline_version("2026-10-27T13:30:00+08:00") == "V7_RULE_BASED_BASELINE_R19"
    expected = {
        "V7_RULE_BASED_BASELINE_R20": "b5b37baac74dfb08f9e3dc6a9f015463286aae95c5eecaed206b54ef7dc7afff",
        "V7_RULE_BASED_BASELINE_R21": "bb6bc7d94bd68edc0c05f2039f57c21fc6de5407cb1f30abcbc854bfb3f179a4"}
    for version, value in expected.items():assert identity(baseline_policy(version=version)) == value
    result = daily_v7_preflight(configured(tmp_path), checked_at=at)
    assert result["state"] == "READY"
    assert result["rule_baseline_policy_id"] == expected[policy_r21.BASELINE_VERSION]
    assert result["prediction_status"] == "NOT_APPLICABLE" and not result["order_authority"]
