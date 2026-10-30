"""Native synthetic training integration; no market readiness claim."""
from copy import deepcopy
import pytest
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.training import train_offline, validate_training_config
from tests.strategies.test_v7_offline_training import synthetic_config, synthetic_dataset


@pytest.fixture(scope='module')
def native_case():
    config = synthetic_config()
    config.update(head_solver='STANDARDIZED_CONVEX_R2', head_iterations=2000)
    return synthetic_dataset(config), config


def test_optional_convex_solver_records_native_reports_and_keeps_public_gates(native_case):
    dataset, config = native_case
    result = train_offline(dataset, config)
    assert result['status'] == 'READY_TO_FREEZE'
    reports = [report for report in result['reports'].values() if report.get('schema') == 'v7-standardized-convex-head-fit/v2']
    assert len(reports) >= 6
    for report in reports:
        assert report['converged'] is True
        assert timestamp(report['last_label_available_at']) <= timestamp(report['fit_cutoff'])
        assert report['heads_sha256'] == identity(report['raw_heads'])
    assert result['candidate']['research_only'] is True
    assert result['candidate']['approved_for_live'] is False
    assert result['reports'][result['final_head_fit_report_id']]['raw_heads'] == result['candidate']['heads']
    assert result['baseline_head_fit_report_id'] in result['reports']
    assert all(fold['head_fit_report_id'] in result['reports'] for fold in result['folds'] if fold['status'] == 'FITTED')


def test_unconverged_solver_never_freezes_a_model(native_case):
    dataset, config = native_case
    config = dict(config, head_iterations=1)
    result = train_offline(dataset, config)
    assert result['status'] == 'NOT_READY' and result['candidate'] is None
    assert any(fold.get('reason') == 'HEAD_SOLVER_NOT_CONVERGED' for fold in result['folds'])
    reports = [report for report in result['reports'].values() if report.get('schema') == 'v7-standardized-convex-head-fit/v2']
    assert reports and all(report['converged'] is False for report in reports)


def test_unknown_solver_is_not_a_legacy_fallback(native_case):
    config = deepcopy(native_case[1]); config['head_solver'] = 'UNREVIEWED'
    with pytest.raises(ValueError, match='UNSUPPORTED_HEAD_SOLVER'):
        validate_training_config(config)
