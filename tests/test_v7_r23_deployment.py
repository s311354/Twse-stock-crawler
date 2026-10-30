"""Forward-only dependency fix; old policy identities remain readable."""
from datetime import timedelta
from unittest.mock import patch

import pytest

from application.les_v7_preflight import daily_v7_preflight
from domain.les_v7 import policy_r23
from domain.les_v7.contracts import identity,timestamp
from domain.les_v7.rule_baseline import baseline_policy,current_source_baseline_version,CURRENT_SOURCE_BASELINE_POLICY,require_baseline_policy
from tests.test_v7_default_baseline import configured


def test_published_candidate_and_forward_deployment_are_separate(tmp_path):
    at=timestamp(policy_r23.DAILY_ACTIVATED_AT)
    assert at>timestamp(policy_r23.AVAILABLE_AT)
    from domain.les_v7.policy_r25 import BASELINE_VERSION as LATEST
    assert CURRENT_SOURCE_BASELINE_POLICY==LATEST
    assert current_source_baseline_version(at-timedelta(microseconds=1))=="V7_RULE_BASED_BASELINE_R22"
    assert current_source_baseline_version(at)==policy_r23.BASELINE_VERSION
    assert current_source_baseline_version("2026-10-27T13:30:00+08:00")=="V7_RULE_BASED_BASELINE_R19"
    with patch.object(policy_r23,"DAILY_ACTIVATED_AT",None), patch("domain.les_v7.policy_r24.DAILY_ACTIVATED_AT",None), patch("domain.les_v7.policy_r25.DAILY_ACTIVATED_AT",None):
        assert current_source_baseline_version("2030-01-01T00:00:00Z")=="V7_RULE_BASED_BASELINE_R22"
    expected="571cbf6eec65df7ed7849df4583fda0911609fbc480ef0b76fa12e1558987142"
    assert identity(baseline_policy(version=policy_r23.BASELINE_VERSION))==expected
    assert identity(baseline_policy(version="V7_RULE_BASED_BASELINE_R22"))=="8f1cc12ad81530a5a1c22819f84f5e8ba6ff9a8381cfc88b10bc07183cb6646a"
    result=daily_v7_preflight(configured(tmp_path),checked_at=at)
    assert result["state"]=="READY" and result["rule_baseline_policy_id"]==expected
    assert result["prediction_status"]=="NOT_APPLICABLE" and not result["order_authority"]


def test_new_source_policy_cannot_be_backdated_to_old_signal():
    with pytest.raises(ValueError,match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=policy_r23.BASELINE_VERSION),market="TWSE",
            cutoff=timestamp(policy_r23.AVAILABLE_AT)-timedelta(microseconds=1))
