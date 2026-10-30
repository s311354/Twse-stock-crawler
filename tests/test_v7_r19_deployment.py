"""Deployment-vintage isolation; no retroactive R19 use by older requests."""
from datetime import timedelta
import pytest

from application.les_v7_preflight import daily_v7_preflight
from domain.les_v7 import policy_r19
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.policy_r18 import BASELINE_VERSION as R18
from domain.les_v7.rule_baseline import baseline_policy, current_source_baseline_version
from tests.test_v7_default_baseline import configured


@pytest.fixture(autouse=True)
def isolate_r19_deployment_contract(monkeypatch):
    # A later rollout must not change R19's own activation/deactivation tests.
    from domain.les_v7 import policy_r20
    from domain.les_v7 import policy_r21
    from domain.les_v7 import policy_r22
    from domain.les_v7 import policy_r23
    from domain.les_v7 import policy_r24
    from domain.les_v7 import policy_r25
    monkeypatch.setattr(policy_r20, 'DAILY_ACTIVATED_AT', None)
    monkeypatch.setattr(policy_r21, 'DAILY_ACTIVATED_AT', None)
    monkeypatch.setattr(policy_r22, 'DAILY_ACTIVATED_AT', None)
    monkeypatch.setattr(policy_r23, 'DAILY_ACTIVATED_AT', None)
    monkeypatch.setattr(policy_r24, 'DAILY_ACTIVATED_AT', None)
    monkeypatch.setattr(policy_r25, 'DAILY_ACTIVATED_AT', None)


def test_published_candidate_without_activation_keeps_r18(monkeypatch):
    monkeypatch.setattr(policy_r19, 'DAILY_ACTIVATED_AT', None)
    assert current_source_baseline_version(policy_r19.AVAILABLE_AT) == R18
    assert current_source_baseline_version('2030-01-01T00:00:00Z') == R18


def test_new_policy_only_starts_at_its_deployment_clock(monkeypatch):
    activation = timestamp(policy_r19.AVAILABLE_AT) + timedelta(seconds=10)
    monkeypatch.setattr(policy_r19, 'DAILY_ACTIVATED_AT', activation.isoformat())
    assert current_source_baseline_version(activation-timedelta(microseconds=1)) == R18
    assert current_source_baseline_version(activation) == policy_r19.BASELINE_VERSION
    assert current_source_baseline_version('2026-10-22T04:14:42Z') == R18


def test_shipped_daily_preflight_uses_r19_and_preserves_old_policy(tmp_path):
    from domain.les_v7.rule_baseline import CURRENT_SOURCE_BASELINE_POLICY
    assert policy_r19.DAILY_ACTIVATED_AT is not None
    assert timestamp(policy_r19.DAILY_ACTIVATED_AT) > timestamp(policy_r19.AVAILABLE_AT)
    from domain.les_v7.policy_r25 import BASELINE_VERSION as LATEST
    assert CURRENT_SOURCE_BASELINE_POLICY == LATEST
    response = daily_v7_preflight(configured(tmp_path), checked_at=policy_r19.DAILY_ACTIVATED_AT)
    assert response['state'] == 'READY' and response['formal_entry_enabled']
    assert response['rule_baseline_policy']['policy_version'] == policy_r19.BASELINE_VERSION
    assert response['rule_baseline_policy_id'] == identity(baseline_policy(version=policy_r19.BASELINE_VERSION))
    assert response['prediction_status'] == 'NOT_APPLICABLE'
    assert not response['order_authority']
    assert identity(baseline_policy(version=R18)) == 'b43f1e233211df05d168bef0cf6fd0abe9f3f6937b6b8e1b3e9537b9fb13efd5'
