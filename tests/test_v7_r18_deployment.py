"""Synthetic deployment clocks; candidate publication is not deployment."""
from datetime import timedelta

from domain.les_v7 import policy_r18
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.policy_r17 import BASELINE_VERSION as R17
from domain.les_v7.rule_baseline import (
    baseline_policy, current_source_baseline_version, require_baseline_policy,
)


def test_candidate_publication_without_deployment_does_not_change_daily_vintage(monkeypatch):
    monkeypatch.setattr(policy_r18, "DAILY_ACTIVATED_AT", None)
    assert current_source_baseline_version(policy_r18.AVAILABLE_AT) == R17
    assert current_source_baseline_version("2026-10-18T00:00:00Z") == R17
    policy = baseline_policy(version=policy_r18.BASELINE_VERSION)
    assert require_baseline_policy(policy, market="TWSE", cutoff=policy_r18.AVAILABLE_AT) == policy


def test_daily_switch_uses_deployment_instant_not_candidate_publication(monkeypatch):
    deployment = "2026-10-18T01:02:03Z"
    monkeypatch.setattr(policy_r18, "DAILY_ACTIVATED_AT", deployment)
    before = timestamp(deployment) - timedelta(microseconds=1)
    assert current_source_baseline_version(policy_r18.AVAILABLE_AT) == R17
    assert current_source_baseline_version(before.isoformat()) == R17
    assert current_source_baseline_version(deployment) == policy_r18.BASELINE_VERSION
    assert current_source_baseline_version("2026-10-18T09:02:03+08:00") == policy_r18.BASELINE_VERSION


def test_activation_does_not_rewrite_any_retained_policy_content(monkeypatch):
    versions = (R17, policy_r18.BASELINE_VERSION, "V7_RULE_BASED_BASELINE_R16")
    before = {version: baseline_policy(version=version) for version in versions}
    monkeypatch.setattr(policy_r18, "DAILY_ACTIVATED_AT", "2026-10-18T01:02:03Z")
    assert {version: baseline_policy(version=version) for version in versions} == before


def test_activation_keeps_earlier_deployed_vintages(monkeypatch):
    cutoffs = ("2026-10-14T15:30:00+08:00", "2026-10-15T16:00:00Z", "2026-10-16T12:00:00Z")
    monkeypatch.setattr(policy_r18, "DAILY_ACTIVATED_AT", None)
    before = [current_source_baseline_version(cutoff) for cutoff in cutoffs]
    monkeypatch.setattr(policy_r18, "DAILY_ACTIVATED_AT", "2026-10-18T01:02:03Z")
    assert [current_source_baseline_version(cutoff) for cutoff in cutoffs] == before


def test_frozen_candidate_and_predecessor_policy_content_ids_do_not_drift():
    # Pinned before daily activation, not regenerated to accept a changed
    # investment assumption. This also covers the TPEx cost-path binding.
    expected = {
        "TWSE": {
            R17: "108ee5d15da10a6602ccf37ef9d6a5aac8424c596bcf461a5517fd090dacc32d",
            policy_r18.BASELINE_VERSION: "b43f1e233211df05d168bef0cf6fd0abe9f3f6937b6b8e1b3e9537b9fb13efd5",
        },
        "TPEX": {
            R17: "900d01e90db9ac77ed2d10586cb5c56dcd8cc9068ef05e2f66a2732737563462",
            policy_r18.BASELINE_VERSION: "ac15c6ae41c815af1340e8f20917ca11b68a192f53fe4f893ed124fd7a2c6c93",
        },
    }
    for market, versions in expected.items():
        for version, content_id in versions.items():
            assert identity(baseline_policy(market, version=version)) == content_id


def test_shipped_daily_selector_activates_r18_only_after_verified_deployment():
    from domain.les_v7.rule_baseline import CURRENT_SOURCE_BASELINE_POLICY
    from domain.les_v7.policy_r24 import BASELINE_VERSION as LATEST
    assert policy_r18.DAILY_ACTIVATED_AT is not None
    deployment = timestamp(policy_r18.DAILY_ACTIVATED_AT)
    assert deployment > timestamp(policy_r18.AVAILABLE_AT)
    assert CURRENT_SOURCE_BASELINE_POLICY == LATEST
    assert current_source_baseline_version((deployment - timedelta(microseconds=1)).isoformat()) == R17
    assert current_source_baseline_version(deployment.isoformat()) == policy_r18.BASELINE_VERSION


def test_public_daily_preflight_selects_activated_r18_without_a_model_bundle(tmp_path):
    from application.les_v7_preflight import daily_v7_preflight
    from tests.test_v7_default_baseline import configured
    result = daily_v7_preflight(configured(tmp_path), checked_at=policy_r18.DAILY_ACTIVATED_AT)
    assert result["state"] == "READY" and result["formal_entry_enabled"]
    assert result["rule_baseline_policy"]["policy_version"] == policy_r18.BASELINE_VERSION
    assert result["rule_baseline_policy_id"] == "b43f1e233211df05d168bef0cf6fd0abe9f3f6937b6b8e1b3e9537b9fb13efd5"
    assert result["prediction_status"] == result["calibration_status"] == "NOT_APPLICABLE"
    assert not result["order_authority"]
