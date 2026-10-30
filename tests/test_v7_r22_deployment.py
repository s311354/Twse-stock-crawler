"""Published source/score vintage; no retroactive deployment or bundle switch."""
from datetime import timedelta
from unittest.mock import patch

import pytest

from application.les_v7_preflight import daily_v7_preflight
from domain.les_v7 import policy_r22
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.rule_baseline import baseline_policy, current_source_baseline_version, require_baseline_policy, CURRENT_SOURCE_BASELINE_POLICY
from tests.test_v7_default_baseline import configured


def test_forward_activation_and_retained_policy_ids(tmp_path):
    at=timestamp(policy_r22.DAILY_ACTIVATED_AT)
    assert at>timestamp(policy_r22.AVAILABLE_AT)
    from domain.les_v7.policy_r25 import BASELINE_VERSION as LATEST
    assert CURRENT_SOURCE_BASELINE_POLICY==LATEST
    assert current_source_baseline_version(at-timedelta(microseconds=1))=="V7_RULE_BASED_BASELINE_R21"
    assert current_source_baseline_version(at)==policy_r22.BASELINE_VERSION
    assert current_source_baseline_version("2026-10-27T13:30:00+08:00")=="V7_RULE_BASED_BASELINE_R19"
    with patch.object(policy_r22,"DAILY_ACTIVATED_AT",None), patch("domain.les_v7.policy_r23.DAILY_ACTIVATED_AT",None), patch("domain.les_v7.policy_r24.DAILY_ACTIVATED_AT",None), patch("domain.les_v7.policy_r25.DAILY_ACTIVATED_AT",None):
        assert current_source_baseline_version("2030-01-01T00:00:00Z")=="V7_RULE_BASED_BASELINE_R21"
    expected={
        "V7_RULE_BASED_BASELINE_R19":"4723ab8f53a4c9ac2db05e123ba970db5242a759ff25107ce0fc173ccfa70637",
        "V7_RULE_BASED_BASELINE_R20":"b5b37baac74dfb08f9e3dc6a9f015463286aae95c5eecaed206b54ef7dc7afff",
        "V7_RULE_BASED_BASELINE_R21":"bb6bc7d94bd68edc0c05f2039f57c21fc6de5407cb1f30abcbc854bfb3f179a4",
        policy_r22.BASELINE_VERSION:"8f1cc12ad81530a5a1c22819f84f5e8ba6ff9a8381cfc88b10bc07183cb6646a"}
    for version,sha in expected.items():assert identity(baseline_policy(version=version))==sha
    result=daily_v7_preflight(configured(tmp_path),checked_at=at)
    assert result["state"]=="READY" and result["rule_baseline_policy_id"]==expected[policy_r22.BASELINE_VERSION]
    assert result["prediction_status"]=="NOT_APPLICABLE" and not result["order_authority"]


def test_candidate_cannot_borrow_a_later_publication():
    with pytest.raises(ValueError,match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=policy_r22.BASELINE_VERSION),market="TWSE",
            cutoff=timestamp(policy_r22.AVAILABLE_AT)-timedelta(microseconds=1))
