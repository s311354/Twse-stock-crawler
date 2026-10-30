"""Avoid a repeated full input pass; retain both independent boundaries."""
from copy import deepcopy

import pytest

from application import low_entry_v7_daily as daily
from domain.les_v7 import allocation
from domain.les_v7 import report as reports
from domain.les_v7.engine import evaluate
from tests.test_v7_baseline_allocation import public_baseline_request, account
from tests.test_v7_r19_pipeline import current_market, native, neutral_market


@pytest.fixture(scope='module', params=['legacy', 'r19'])
def source(request):
    if request.param == 'legacy':
        raw, context = public_baseline_request()
        return evaluate(raw, context), account()
    finalized = native(current_market(neutral_market()))
    configuration = deepcopy(finalized['pae_authorization']['account'])
    finalized.pop('pae_authorization')
    return finalized, configuration


def original_finalization(report, configuration, errors=()):
    """Pre-optimization public sequence, using unchanged native functions."""
    daily._finalize_daily_inputs(report, list(errors))
    report['pae_authorization'] = allocation.authorize_candidate(report, configuration)
    reports.validate_report(report)
    return report


@pytest.mark.parametrize('errors', [(), ('STALE_MARKET_DATA',)])
def test_result_is_identical_to_original_full_validation_sequence(source, errors):
    report, configuration = source
    original = original_finalization(deepcopy(report), configuration, errors)
    result = daily.finalize_daily_report(deepcopy(report), account=configuration, daily_input_errors=errors)
    assert result == original
    reports.validate_report(result)
    assert result['pae_authorization']['live_order_authorized'] is False


def test_two_full_checks_remain_and_public_authorization_still_validates(source, monkeypatch):
    source_report, configuration = source
    check = reports.validate_report
    calls = []
    def counted(value):
        calls.append(value.get('pae_authorization') is not None)
        return check(value)
    monkeypatch.setattr(daily, 'validate_report', counted)
    monkeypatch.setattr(reports, 'validate_report', counted)
    result = daily.finalize_daily_report(deepcopy(source_report), account=configuration)
    assert calls == [False, True]
    calls.clear()
    assert allocation.authorize_candidate(result, configuration) == result['pae_authorization']
    assert calls == [False]
    calls.clear()
    original_finalization(deepcopy(source_report), configuration)
    assert calls == [False, False, True]


def test_tampered_source_cannot_reach_account_allocation(source, monkeypatch):
    source_report, configuration = source
    damaged = deepcopy(source_report)
    damaged['diagnostics']['reference_plan']['stop'] += 1.
    with pytest.raises(ValueError):
        allocation.authorize_candidate(damaged, configuration)
    monkeypatch.setattr(allocation, '_authorize_validated', lambda *args: pytest.fail('Invalid source reached account allocation'))
    with pytest.raises(ValueError):
        daily.finalize_daily_report(damaged, account=configuration)


def test_independent_post_check_detects_a_body_change_during_allocation(source, monkeypatch):
    source_report, configuration = source
    allocate = allocation._authorize_validated
    def damaged_allocation(value, selected_account):
        quota = allocate(value, selected_account)
        value['envelope']['score']['final_score'] = 100.
        return quota
    monkeypatch.setattr(allocation, '_authorize_validated', damaged_allocation)
    with pytest.raises(ValueError):
        daily.finalize_daily_report(deepcopy(source_report), account=configuration)
