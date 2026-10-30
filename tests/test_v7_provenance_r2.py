"""Synthetic engineering checks; no real-market model readiness assertion."""
from copy import deepcopy
import json

import pytest

from application.evidence_context import ReadOnlyEvidence
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.localization import FrozenBundle, predict
from domain.les_v7.provenance import (RUNTIME_SCHEMA, archive_provenance, compact_candidate,
                                     id_reference, verify_archived_compaction)
from domain.les_v7.training import _train_prepared, prepare_events, train_offline
from infrastructure.storage.les_v7_training import LesV7TrainingRepository, _validate_result
from tests.strategies.test_v7_offline_training import synthetic_config, synthetic_dataset, synthetic_published_candidate
from tests.test_v7_training_publication import Guard


@pytest.fixture(scope='module')
def fitted():
    config = synthetic_config()
    original = synthetic_dataset(config)
    rows, diagnostics = prepare_events(original, config)
    mono = _train_prepared(original, config, rows, diagnostics)
    # Internal native-core fixture: the public scope source/build/coverage
    # contract is separately exercised in test_v7_scope_training.py.
    payload = dict(schema='les-v7-sharded-training-dataset/v1', dataset_kind='SYNTHETIC_TEST_ONLY',
        freeze_id=identity('SYNTHETIC-freeze'), scope_id=identity('SYNTHETIC-scope'),
        universe=original['payload']['universe'], children=[dict(dataset_id=identity('SYNTHETIC-child-1')),
        dict(dataset_id=identity('SYNTHETIC-child-2'))], diagnostics_policy='SYNTHETIC_SLIM_DIAGNOSTICS')
    scoped = dict(payload=payload, sha256=identity(payload))
    result = _train_prepared(scoped, config, rows, diagnostics)
    assert mono['status'] == result['status'] == 'READY_TO_FREEZE'
    request = dict(dataset_id=scoped['sha256'], config_id=identity(config), dataset_kind='SYNTHETIC_TEST_ONLY',
                   fit_cutoff=config['fit_cutoff'])
    return scoped, config, result, mono, rows, request


def test_exact_native_global_numerics_and_oof_unchanged(fitted):
    dataset, config, result, mono, rows, request = fitted
    candidate = result['candidate']
    assert candidate['runtime_provenance_schema'] == RUNTIME_SCHEMA
    for key in ('heads', 'weights', 'feature_names', 'feature_bounds', 'fit_cutoff', 'last_label_available_at', 'validation_available_at'):
        assert candidate[key] == mono['candidate'][key]
    assert result['oof_predictions'] == mono['oof_predictions']
    assert result['folds'] == mono['folds'] and result['reports'] == mono['reports']
    verify_archived_compaction(candidate, result, config, dataset)
    _validate_result(result, request, config=config, dataset_record=dataset)
    old = FrozenBundle.load(dict(payload=synthetic_published_candidate(mono['candidate']),
                                 sha256=identity(synthetic_published_candidate(mono['candidate']))))
    new = FrozenBundle.load(dict(payload=synthetic_published_candidate(candidate),
                                 sha256=identity(synthetic_published_candidate(candidate))))
    for row in rows:
        cutoff = '2026-10-14T13:30:00+08:00'
        assert predict(old, row['features'], row['plan'], cutoff=cutoff) == predict(new, row['features'], row['plan'], cutoff=cutoff)


def test_canonical_id_refs_preserve_order_and_count():
    values = ['SYNTHETIC-a', 'SYNTHETIC-b', 'SYNTHETIC-a']
    ref = id_reference(values)
    assert ref['count'] == 3 and ref['sha256'] == identity(values)
    assert id_reference(list(reversed(values[:-1]))) != id_reference(values[:-1])


@pytest.mark.parametrize('target', ['compact_metric', 'digest', 'count', 'raw_report', 'raw_weight', 'raw_fold', 'raw_oof'])
def test_any_raw_or_compact_provenance_tamper_rejected(fitted, target):
    dataset, config, original, _, _, _ = fitted
    result = deepcopy(original)
    candidate = result['candidate']
    if target == 'compact_metric':
        candidate['validation_reports'][candidate['validation_report_ids']['holdout']]['report']['pinball_loss'] += .001
    elif target in {'digest', 'count'}:
        ref = candidate['training_manifest']['oof_prediction_ids']
        ref['sha256' if target == 'digest' else 'count'] = identity('SYNTHETIC-tamper') if target == 'digest' else ref['count'] + 1
    elif target == 'raw_report':
        raw_id = result['training_provenance']['archive_validation_report_ids']['holdout']
        result['reports'][raw_id]['pinball_loss'] += .001
    elif target == 'raw_weight':
        result['training_provenance']['weight_prediction_ids'].reverse()
    elif target == 'raw_fold':
        result['folds'][0]['training_row_ids'].reverse()
    else:
        result['oof_predictions'][0]['head_fit_cutoff'] = '2027-01-01T13:30:00+08:00'
    with pytest.raises(ValueError, match='R2_'):
        verify_archived_compaction(candidate, result, config, dataset)


@pytest.mark.parametrize('target', ['count_bool', 'zero_count', 'time', 'gate', 'source_hash', 'calibration_parameters', 'schema'])
def test_standalone_r2_loader_checks_all_runtime_contracts(fitted, target):
    candidate = synthetic_published_candidate(fitted[2]['candidate'])
    if target == 'count_bool':
        candidate['training_manifest']['counts']['oof'] = True
    elif target == 'zero_count':
        candidate['training_manifest']['oof_prediction_ids']['count'] = 0
    elif target == 'time':
        candidate['training_manifest']['folds'][0]['last_training_label_available_at'] = '2027-01-01T13:30:00+08:00'
    elif target == 'calibration_parameters':
        candidate['calibration']['fill'][0] += .001
    elif target in {'gate', 'source_hash'}:
        old_id = candidate['validation_report_ids']['projection']
        wrapper = candidate['validation_reports'].pop(old_id)
        if target == 'gate':
            wrapper['report']['status'] = 'FAIL'
        else:
            wrapper['source_report_sha256'] = identity('SYNTHETIC-wrong-source')
        new_id = identity(wrapper)
        candidate['validation_reports'][new_id] = wrapper
        candidate['validation_report_ids']['projection'] = new_id
    else:
        candidate['runtime_provenance_schema'] = 'UNRECOGNIZED'
    with pytest.raises(ValueError, match='R2_|UNKNOWN_RUNTIME_PROVENANCE'):
        FrozenBundle.load(dict(payload=candidate, sha256=identity(candidate)))


def test_large_full_scope_metadata_compacts_below_unchanged_1mib(fitted):
    """Metadata-only bounds check; not a 100k-row numerical training fixture."""
    raw = deepcopy(fitted[3]['candidate'])
    identifiers = [identity(dict(synthetic_metadata_row=index)) for index in range(100000)]
    manifest = raw['training_manifest']
    for fold in manifest['folds']:
        fold['training_row_ids'] = identifiers
        fold['prediction_row_ids'] = identifiers
    manifest.update(dataset_schema='les-v7-sharded-training-dataset/v1', freeze_id=identity('SYNTHETIC-freeze'),
        scope_id=identity('SYNTHETIC-scope'), full_universe_sha256=identity('SYNTHETIC-universe'),
        child_dataset_ids=identifiers[:64], oof_prediction_ids=identifiers, weight_prediction_ids=identifiers)
    new_reports, ids = {}, {}
    for name, report_id in raw['validation_report_ids'].items():
        report = deepcopy(raw['validation_reports'][report_id])
        if 'row_ids' in report:
            report['row_ids'] = identifiers
        if name == 'ablation':
            report['full']['row_ids'] = report['baseline']['row_ids'] = identifiers
        rid = identity(report); new_reports[rid], ids[name] = report, rid
    raw.update(validation_reports=new_reports, validation_report_ids=ids,
               fractal_null_validation_report_id=ids['fractal_null'])
    raw['calibration']['validation_report_id'] = ids['calibration']
    provenance = archive_provenance(fitted[1], ids, identifiers, manifest['weight_fit'])
    compact = compact_candidate(raw, provenance)
    raw_size = len(canonical(dict(payload=raw, sha256=identity(raw))).encode())
    compact_size = len(canonical(dict(payload=compact, sha256=identity(compact))).encode())
    assert raw_size > 1024 * 1024
    assert compact_size < 1024 * 1024
    assert compact_size < 100000
    # Numerical and acceptance contents all remain identical.
    assert compact['heads'] == raw['heads'] and compact['weights'] == raw['weights']
    for name, rid in compact['validation_report_ids'].items():
        report = compact['validation_reports'][rid]
        assert report['source_report_sha256'] == ids[name]
        for key in ('schema', 'schema_version', 'status', 'acceptance', 'metrics', 'fit_cutoff', 'last_label_available_at'):
            if key in new_reports[ids[name]]:
                assert report['report'][key] == new_reports[ids[name]][key]


def test_native_archive_publication_and_resume_keep_raw_evidence(fitted, tmp_path, monkeypatch):
    dataset, config, result, _, _, _ = fitted
    # Source coverage is not claimed by this internal engineering fixture;
    # the independently tested native sharded verifier is replaced only here.
    import infrastructure.les_v7_scope_training as scope
    monkeypatch.setattr(scope, 'verify_sharded_training_dataset', lambda record, *_args, **_kw: record['payload'])
    dataset = deepcopy(dataset)
    dataset['payload']['pinned_objects'] = dict(source=identity('SYNTHETIC-source'), environment=identity('SYNTHETIC-environment'), parameters=identity('SYNTHETIC-parameters'))
    # Guard's canonical synthetic pinned values, rather than invented archive IDs.
    guard = Guard(); from infrastructure.research_objects import ResearchObjects
    dataset['payload']['pinned_objects'] = guard.retain(ResearchObjects(tmp_path / 'objects'))
    dataset['sha256'] = identity(dataset['payload'])
    result = deepcopy(result)
    result['dataset_id'] = dataset['sha256']; result['candidate']['training_manifest']['dataset_id'] = dataset['sha256']
    result['candidate']['bundle_id'] = identity({k: v for k, v in result['candidate'].items() if k != 'bundle_id'})
    repo = LesV7TrainingRepository(tmp_path / 'models', guard_factory=Guard, clock=lambda: '2026-10-13T10:00:00+08:00')
    receipt = repo.run(dataset, config, trainer=lambda *_args: deepcopy(result))
    folder = receipt['output']
    assert receipt['status'] == 'READY_TO_FREEZE'
    assert receipt['research_only'] is True and receipt['approved_for_live'] is False
    assert repo.run(dataset, config, trainer=lambda *_args: pytest.fail('native fit repeated')) == receipt
    raw_fit = json.loads((__import__('pathlib').Path(folder) / 'fit_result.json').read_text())['payload']
    assert isinstance(raw_fit['training_provenance']['weight_prediction_ids'], list)
    assert isinstance(raw_fit['folds'][0]['training_row_ids'], list)
    assert list((__import__('pathlib').Path(folder) / 'reports').glob('*.json'))
    bundle = json.loads((__import__('pathlib').Path(folder) / 'frozen_bundle.json').read_text())
    FrozenBundle.load(bundle)
    assert len(canonical(bundle).encode()) < 1024 * 1024


def test_sparse_holdout_maturity_before_declared_end_retains_original_readiness(fitted):
    """Native H20 labels can mature before a sparse declared holdout end."""
    dataset, config, _, _, rows, _ = fitted
    config = deepcopy(config)
    config['validation_end'] = '2025-07-17T13:30:00+08:00'
    # These are the exact native-prepared synthetic rows, with no later events.
    diagnostics = fitted[2]['diagnostics']
    mono_payload = dict(schema='les-v7-training-dataset/v1', dataset_kind='SYNTHETIC_TEST_ONLY')
    mono_dataset = dict(payload=mono_payload, sha256=identity(mono_payload))
    old = _train_prepared(mono_dataset, config, rows, diagnostics)
    new = _train_prepared(dataset, config, rows, diagnostics)
    assert old['status'] == new['status'] == 'READY_TO_FREEZE'
    assert new['candidate']['validation_available_at'] < config['validation_end']
    assert new['candidate']['heads'] == old['candidate']['heads']
    assert new['candidate']['weights'] == old['candidate']['weights']
    assert new['oof_predictions'] == old['oof_predictions'] and new['reports'] == old['reports']
    verify_archived_compaction(new['candidate'], new, config, dataset)
    published = synthetic_published_candidate(new['candidate'])
    FrozenBundle.load(dict(payload=published, sha256=identity(published)))


def published_r2_record(fitted):
    """Real native synthetic fit; never a market/deployment-readiness fixture."""
    payload = synthetic_published_candidate(fitted[2]['candidate'])
    return dict(payload=payload, sha256=identity(payload))


def test_r2_readonly_bundle_matches_native_json_without_mutation(fitted):
    record = published_r2_record(fitted)
    before = canonical(record)
    frozen = ReadOnlyEvidence(record)
    assert isinstance(record['payload']['training_manifest']['folds'], list)
    assert isinstance(frozen['payload']['training_manifest']['folds'], tuple)
    assert canonical(frozen) == before
    native = FrozenBundle.load(record)
    immutable = FrozenBundle.load(frozen)
    assert immutable == native
    assert immutable.record() == record['payload']
    assert canonical(record) == canonical(frozen) == before
    detached = immutable.record()
    detached['training_manifest']['folds'].clear()
    assert immutable.record() == record['payload']
    with pytest.raises(TypeError, match='READ_ONLY_EVIDENCE'):
        frozen['payload']['training_manifest']['folds'][0]['fit_cutoff'] = '2027-01-01T00:00:00Z'


def test_r2_complete_engine_result_matches_frozen_and_mutable_context(fitted):
    from domain.les_v7.engine import evaluate
    from tests.strategies.test_low_entry_score_v7 import bars
    record = published_r2_record(fitted)
    context = dict(symbol='SYNTHETIC_ONLY', market='TWSE', as_of='2026-10-14T13:30:00+08:00',
        filings=[], receipts={}, external_evidence_ids=[], model_bundle=record,
        quality_policy=record['payload']['quality_policy'])
    frozen = ReadOnlyEvidence(context)
    before = canonical(context)
    native_result = evaluate(bars(540), context)
    assert evaluate(bars(540), frozen) == native_result
    assert native_result['envelope']['model']['bundle_sha256'] == record['sha256']
    assert native_result['envelope']['execution']['live_enabled'] is False
    # The container fix cannot turn missing financial/safety evidence into PASS.
    assert native_result['envelope']['quality']['status'] == 'DATA_INSUFFICIENT'
    assert native_result['envelope']['strategy']['strategy_decision'] not in {'BUY', 'BUY_SMALL'}
    assert canonical(context) == canonical(frozen) == before


def test_r2_daily_adapter_matches_native_kernel_and_keeps_missing_gates(fitted):
    from application.low_entry_v7_daily import daily_report, finalize_daily_report
    from domain.les_v7.engine import evaluate
    from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR, SIGNAL_DATE_ATTR
    from tests.strategies.test_low_entry_score_v7 import bars
    record = published_r2_record(fitted)
    raw = bars(540).reindex(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Trades', 'Amount', 'Bid', 'Ask', 'PE', 'TAIEX Close'])
    cutoff = '2026-10-14T13:30:00+08:00'
    history = raw.copy()
    for column in ('Open', 'High', 'Low', 'Close', 'Bid', 'Ask'):
        history[column] = history[column] * 1000.
    history.attrs.update(les_v6_daily_identity=dict(symbol='SYNTHETIC_ONLY', market='TWSE'),
        market_source='TWSE', les_v7_bundle_json=canonical(record),
        **{SIGNAL_DATE_ATTR: cutoff, PRICE_SCALE_ATTR: 1000.})
    # Match the actual adapter's legacy quote normalization, including its
    # floating-point operations, rather than silently tolerating score drift.
    normalized = history.copy()
    normalized.attrs = {}
    for column in ('Open', 'High', 'Low', 'Close', 'Bid', 'Ask'):
        normalized[column] = normalized[column] / 1000.
    context = dict(symbol='SYNTHETIC_ONLY', market='TWSE', as_of=cutoff, filings=[],
        receipts={}, external_evidence_ids=[], model_bundle=record,
        quality_policy=record['payload']['quality_policy'])
    expected = finalize_daily_report(evaluate(normalized, context), account={})
    before = canonical(record)
    actual = daily_report(history)
    assert actual == expected
    assert actual['envelope']['model']['bundle_sha256'] == record['sha256']
    assert actual['pae_authorization']['authorized'] is False
    assert canonical(record) == before


@pytest.mark.parametrize('frozen', [False, True], ids=['native-json', 'readonly-evidence'])
@pytest.mark.parametrize('damage', ['missing', 'empty', 'malformed', 'duplicate', 'out_of_order',
    'future', 'support', 'hash', 'calibration'])
def test_r2_fold_validation_stays_fail_closed_for_both_container_paths(fitted, frozen, damage):
    record = published_r2_record(fitted)
    payload = record['payload']
    manifest = payload['training_manifest']
    if damage == 'missing':
        manifest.pop('folds')
    elif damage == 'empty':
        manifest['folds'] = []
    elif damage == 'malformed':
        manifest['folds'] = [None]
    elif damage == 'duplicate':
        manifest['folds'] = [deepcopy(manifest['folds'][0])] * 2
    elif damage == 'out_of_order':
        manifest['folds'].reverse()
    elif damage == 'future':
        manifest['folds'][0]['last_training_label_available_at'] = '2027-01-01T00:00:00Z'
    elif damage == 'support':
        manifest['folds'][0]['training_row_ids']['count'] = 0
    elif damage == 'hash':
        record['sha256'] = '0' * 64
    else:
        payload['calibration']['fill'][0] += .001
    if damage != 'hash':
        record['sha256'] = identity(payload)
    supplied = ReadOnlyEvidence(record) if frozen else record
    before = canonical(supplied)
    with pytest.raises(ValueError, match='R2_|MODEL_BUNDLE_HASH_MISMATCH'):
        FrozenBundle.load(supplied)
    assert canonical(supplied) == before
