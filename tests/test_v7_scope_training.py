"""Explicit synthetic full-scope contracts, never a real trained-market result."""
from copy import deepcopy
from collections import Counter
import json
import os
from pathlib import Path
import weakref

import pytest

from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.training import _train_prepared, prepare_events, train_offline
from infrastructure import les_v7_scope_training as scope_training
from infrastructure.storage.les_v7_training import LesV7TrainingRepository
from infrastructure.storage.les_v7_prepared import read_prepared_packet
from tests.strategies.test_v7_offline_training import synthetic_config, synthetic_dataset
from tests.test_v7_dataset_scope import scope_inputs, freeze, NOW
from tests.test_v7_training_publication import Guard


@pytest.fixture
def inputs(tmp_path):
    source = scope_inputs(tmp_path, count=3)
    manifest = freeze(source, batch_size=2)
    config = synthetic_config()
    config.update(deepcopy(source['config']['seed']))
    config.update(evaluation_cutoff=source['config']['evaluation_cutoff'],
        fit_cutoff='2026-07-01T13:30:00+08:00', validation_start='2026-08-03T13:30:00+08:00',
        validation_end='2026-08-05T13:30:00+08:00', folds=[
            dict(fold_id='SYNTHETIC-march', fit_cutoff='2026-03-02T13:30:00+08:00',
                prediction_start='2026-03-03T13:30:00+08:00', prediction_end='2026-03-13T13:30:00+08:00'),
            dict(fold_id='SYNTHETIC-april', fit_cutoff='2026-04-20T13:30:00+08:00',
                prediction_start='2026-04-21T13:30:00+08:00', prediction_end='2026-04-30T13:30:00+08:00')])
    return source, manifest, config, tmp_path / 'SYNTHETIC-training-output'


def run(inputs, **kwargs):
    source, manifest, config, output = inputs
    return scope_training.train_elec_scope(manifest, source['config'], config, output,
        allow_synthetic=True, guard_factory=Guard, clock=lambda: NOW, **kwargs)


def prepare(inputs, **kwargs):
    source, manifest, config, output = inputs
    return scope_training.prepare_elec_scope_dataset(manifest, source['config'], config, output,
        allow_synthetic=True, guard_factory=Guard, clock=lambda: NOW, **kwargs)


def test_prepare_only_never_calls_model_training_or_holdout(inputs, monkeypatch):
    monkeypatch.setattr(scope_training, '_train_prepared', lambda *_: pytest.fail('model core called'))
    monkeypatch.setattr(scope_training.LesV7TrainingRepository, 'run',
        lambda *_args, **_kwargs: pytest.fail('model repository called'))
    result = prepare(inputs)
    root = inputs[3]
    assert result['status'] == 'DATASET_PREPARED'
    assert result['model_fitted'] is result['holdout_evaluated'] is result['model_published'] is False
    assert result['config_id'] == identity(inputs[2])
    assert Path(result['dataset_path']) == root / 'scope-training-dataset.json'
    assert not (root / 'models').exists()
    assert not (root / 'scope-training-outcome.json').exists()
    record = json.loads((root / 'scope-training-dataset.json').read_text())
    assert scope_training.verify_sharded_training_dataset(record, root, config=inputs[2]) == record['payload']


def test_prepare_only_recovery_reverifies_without_rebuilding_or_overwriting(inputs, monkeypatch):
    first = prepare(inputs)
    root = inputs[3]
    frozen = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*')
              if p.is_file() and p.suffix in {'.json', '.gz'}}
    monkeypatch.setattr(scope_training, 'prepare_events', lambda *_: pytest.fail('prepared rows rebuilt'))
    monkeypatch.setattr(scope_training.LesV7DatasetRepository, 'run',
        lambda *_args, **_kwargs: pytest.fail('dataset rebuilt'))
    assert prepare(inputs) == first
    assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*')
            if p.is_file() and p.suffix in {'.json', '.gz'}} == frozen
    record = json.loads((root / 'scope-training-dataset.json').read_text())
    (root / record['payload']['children'][0]['prepared_path']).write_bytes(b'truncated')
    with pytest.raises(ValueError, match='GZIP_INTEGRITY'):
        prepare(inputs)


def test_prepare_only_recovery_rejects_changed_config_and_source_pins(inputs):
    prepare(inputs)
    root, config = inputs[3], inputs[2]
    original = (root / 'scope-training-dataset.json').read_bytes()
    changed = deepcopy(config); changed['calibration_iterations'] += 1
    with pytest.raises(ValueError, match='RECOVERY_REQUEST_MISMATCH'):
        scope_training.prepare_elec_scope_dataset(inputs[1], inputs[0]['config'], changed, root,
            allow_synthetic=True, guard_factory=Guard, clock=lambda: NOW)
    class ChangedGuard(Guard):
        def retain(self, objects):
            pins = super().retain(objects)
            return dict(pins, source=objects.put(b'SYNTHETIC_TEST_ONLY changed source'))
    with pytest.raises(ValueError, match='RECOVERY_SOURCE_CHANGED'):
        scope_training.prepare_elec_scope_dataset(inputs[1], inputs[0]['config'], config, root,
            allow_synthetic=True, guard_factory=ChangedGuard, clock=lambda: NOW)
    assert (root / 'scope-training-dataset.json').read_bytes() == original
    assert not (root / 'models').exists()


def test_normal_training_continues_exact_prepared_dataset_without_rebuilding(inputs, monkeypatch):
    prepared = prepare(inputs)
    root = inputs[3]
    before = {str(p.relative_to(root)): p.read_bytes() for p in (root / 'prepared').iterdir()}
    monkeypatch.setattr(scope_training, 'prepare_events', lambda *_: pytest.fail('prepared rows rebuilt'))
    monkeypatch.setattr(scope_training.LesV7DatasetRepository, 'run',
        lambda *_args, **_kwargs: pytest.fail('dataset rebuilt'))
    outcome = run(inputs)
    assert outcome['dataset_id'] == prepared['dataset_id']
    assert outcome['status'] == 'NOT_READY'
    assert (root / 'scope-training-outcome.json').exists()
    assert len(list((root / 'models').glob('*/completion.json'))) == 1
    assert {str(p.relative_to(root)): p.read_bytes() for p in (root / 'prepared').iterdir()} == before


def test_light_inventory_is_manifest_verified_not_source_verified(inputs, monkeypatch):
    source, manifest, _, _ = inputs
    import infrastructure.les_v7_dataset_scope as module
    original = module.load_evidence
    monkeypatch.setattr(module, 'load_evidence', lambda *_args, **_kw: pytest.fail('light inventory read source bytes'))
    inventory = scope_training.inspect_elec_scope_freeze(manifest)
    assert inventory['symbols'] == source['symbols']
    assert inventory['batch_count'] == 2 and inventory['sources_verified'] is False
    monkeypatch.setattr(module, 'load_evidence', original)
    assert scope_training.inspect_elec_scope(manifest)['sources_verified'] is True


def test_two_shards_one_native_training_job_all_symbols_preserved(inputs):
    phases = []
    outcome = run(inputs, progress=phases.append)
    source, _, _, root = inputs
    assert outcome['symbols'] == source['symbols']
    assert outcome['symbol_count'] == 3 and outcome['batch_count'] == 2
    assert outcome['status'] == 'NOT_READY' and outcome['bundle_path'] is None
    assert len(list((root / 'models').glob('*/completion.json'))) == 1
    assert len(list((root / 'builds').glob('*/completion.json'))) == 2
    assert not list(root.rglob('frozen_bundle.json'))
    assert len(json.dumps(outcome).encode()) < 64 * 1024
    assert len(phases) == 3
    receipt = scope_training.verify_elec_scope_training(root, outcome)
    assert receipt['job_id'] == outcome['native_job_id'] and receipt['dataset_kind'] == 'SYNTHETIC_TEST_ONLY'
    record = json.loads((root / 'scope-training-dataset.json').read_text())
    body = record['payload']
    assert body['schema'] == scope_training.DATASET_SCHEMA
    assert [s for child in body['children'] for s in child['symbols']] == source['symbols']
    assert sorted(row['symbol'] for child in body['children'] for row in child['coverage']) == source['symbols']
    assert all('histories' not in child for child in body['children'])
    assert all('coverage' not in child for child in outcome['build_children'])
    assert all(child['prepared_path'].endswith('.json.gz') for child in body['children'])


def test_native_shared_core_matches_single_dataset_over_two_prepared_shards():
    config = synthetic_config()
    dataset = synthetic_dataset(config)
    expected = train_offline(dataset, config)
    histories = dataset['payload']['histories']
    rows, diagnostics = [], []
    for selected in (histories[:18], histories[18:]):
        payload = deepcopy(dataset['payload'])
        symbols = {item['symbol'] for item in selected}
        payload['histories'] = deepcopy(selected)
        payload['universe'] = sorted([[item['symbol'], item['market']] for item in selected])
        payload['events'] = [event for event in payload['events'] if event['symbol'] in symbols]
        native_rows, native_diagnostics = prepare_events(dict(payload=payload, sha256=identity(payload)), config)
        packet = scope_training._prepared_packet(dataset['sha256'], identity(config), native_rows, native_diagnostics)
        columns = scope_training._packet_feature_columns(packet)
        rows.extend(scope_training._unpack_row(row, columns) for row in packet['rows'])
        diagnostics.extend(native_diagnostics)
    rows.sort(key=lambda row: (timestamp(row['cutoff']), row['symbol'], row['market']))
    diagnostics.sort(key=lambda row: (timestamp(row['cutoff']), row['symbol'], row['market']))
    actual = _train_prepared(dataset, config, rows, diagnostics)
    assert actual == expected
    assert actual['status'] == 'READY_TO_FREEZE'


def test_lossless_prepared_columns_reduce_bytes_without_raising_native_budget():
    config = synthetic_config()
    dataset = synthetic_dataset(config)
    native_rows, diagnostics = prepare_events(dataset, config)
    # Full native rows remain exactly recoverable, including features not used
    # by head vectors and rows with missing/different feature fields.
    assert native_rows
    native_rows[0]['features']['unused_diagnostic_feature'] = None
    packet = scope_training._prepared_packet(dataset['sha256'], identity(config), native_rows, diagnostics)
    columns = scope_training._packet_feature_columns(packet)
    restored = [scope_training._unpack_row(row, columns) for row in packet['rows']]
    assert restored == native_rows
    assert isinstance(packet['rows'][0]['features'], list)
    assert isinstance(packet['rows'][1]['features'], dict)
    # Homogeneous production shards get their names stored once, while the
    # original 128 MiB and 100,000-row caps remain in force.
    homogeneous = [deepcopy(native_rows[1]) for _ in range(12)]
    compact = scope_training._prepared_packet('dataset', 'config', homogeneous, [])
    legacy = dict(schema=scope_training.LEGACY_PREPARED_SCHEMA, dataset_id='dataset', config_id='config',
                  rows=[scope_training._pack_row(row) for row in homogeneous], diagnostics=[])
    assert len(canonical(compact).encode()) < len(canonical(legacy).encode())
    assert scope_training.MAX_PREPARED_BYTES == 128 * 1024 * 1024
    assert scope_training.MAX_PREPARED_ROWS == 100000
    assert scope_training._packet_feature_columns(legacy) is None
    assert [scope_training._unpack_row(row) for row in legacy['rows']] == homogeneous


@pytest.mark.parametrize('change', ['duplicate', 'unordered', 'missing', 'length'])
def test_prepared_feature_columns_require_unambiguous_reconstruction(change):
    config = synthetic_config()
    dataset = synthetic_dataset(config)
    rows, diagnostics = prepare_events(dataset, config)
    packet = scope_training._prepared_packet(dataset['sha256'], identity(config), rows, diagnostics)
    if change == 'duplicate':
        packet['feature_columns'].append(packet['feature_columns'][0])
    elif change == 'unordered':
        packet['feature_columns'].reverse()
    elif change == 'missing':
        packet.pop('feature_columns')
    else:
        packet['rows'][0]['features'].pop()
    with pytest.raises(ValueError, match='PREPARED_FEATURE_BINDING_MISMATCH'):
        columns = scope_training._packet_feature_columns(packet)
        scope_training._unpack_row(packet['rows'][0], columns)


def test_explicit_synthetic_opt_in_and_config_alignment(inputs):
    source, manifest, config, root = inputs
    with pytest.raises(ValueError, match='SYNTHETIC_SCOPE_TRAINING_REQUIRES'):
        scope_training.train_elec_scope(manifest, source['config'], config, root)
    assert not root.exists()
    bad = deepcopy(config)
    bad['feature_bounds'][0][0] = -11
    with pytest.raises(ValueError, match='CONFIG_BINDING'):
        scope_training.train_elec_scope(manifest, source['config'], bad, root, allow_synthetic=True)
    assert not root.exists()


@pytest.mark.parametrize('target', ['source', 'child', 'prepared', 'outcome'])
def test_all_native_archives_and_full_coverage_rechecked(inputs, target):
    outcome = run(inputs)
    source, manifest, _, root = inputs
    if target == 'source':
        first = json.loads(Path(manifest).read_text())['body']['chunks'][0]
        path = source['root'] / 'evidence' / first['evidence_id'] / 'evidence.json'
        data = json.loads(path.read_text()); data['payload']['symbols'].append('9999')
        path.write_text(json.dumps(data))
    elif target == 'child':
        path = root / outcome['build_children'][0]['archive_path'] / 'dataset.json'
        path.write_text('{}')
    elif target == 'prepared':
        import gzip
        child = json.loads((root / 'scope-training-dataset.json').read_text())['payload']['children'][0]
        path = root / child['prepared_path']
        data = read_prepared_packet(path); data['rows'].append(dict(row_id='SYNTHETIC_FAKE_ROW'))
        path.write_bytes(gzip.compress(canonical(data).encode(), mtime=0))
    else:
        outcome = dict(outcome, symbols=outcome['symbols'][:-1])
    with pytest.raises((ValueError, KeyError)):
        scope_training.verify_elec_scope_training(root, outcome)
    if target != 'outcome':
        # Direct native model verification also checks sharded provenance.
        with pytest.raises((ValueError, KeyError)):
            LesV7TrainingRepository.verify(root / outcome['archive_path'])


def test_scope_retry_keeps_first_completion_and_never_refits(inputs, monkeypatch):
    first = run(inputs)
    root = inputs[3]
    frozen = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*.json')}
    monkeypatch.setattr(scope_training, '_train_prepared', lambda *_: pytest.fail('completed scope refitted'))
    assert run(inputs) == first
    assert {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*.json')} == frozen


def test_legacy_v1_prepared_archives_still_verify_without_migration(inputs, monkeypatch):
    current = scope_training._prepared_packet

    def legacy_packet(dataset_id, config_id, rows, diagnostics):
        packet = current(dataset_id, config_id, rows, diagnostics)
        packet.pop('feature_columns')
        packet.update(schema=scope_training.LEGACY_PREPARED_SCHEMA,
                      rows=[scope_training._pack_row(row) for row in rows])
        return packet

    monkeypatch.setattr(scope_training, '_prepared_packet', legacy_packet)
    outcome = run(inputs)
    root = inputs[3]
    archived = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*.json')}
    monkeypatch.setattr(scope_training, '_prepared_packet', current)
    assert scope_training.verify_elec_scope_training(root, outcome)['status'] == 'NOT_READY'
    assert {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*.json')} == archived


def test_prepared_budget_is_fail_closed_before_training_completion(inputs, monkeypatch):
    monkeypatch.setattr(scope_training, 'MAX_PREPARED_BYTES', 1)
    with pytest.raises(ValueError, match='PREPARED_BUDGET_EXCEEDED'):
        run(inputs)
    assert not list((inputs[3] / 'models').glob('*/completion.json'))
    assert not (inputs[3] / 'scope-training-outcome.json').exists()


def test_prepared_decoded_budget_is_independent_and_fail_closed(inputs, monkeypatch):
    assert scope_training.MAX_PREPARED_DECODED_BYTES == 256 * 1024 * 1024
    monkeypatch.setattr(scope_training, 'MAX_PREPARED_DECODED_BYTES', 1)
    with pytest.raises(ValueError, match='PREPARED_BUDGET_EXCEEDED'):
        run(inputs)
    assert not list((inputs[3] / 'models').glob('*/completion.json'))
    assert not (inputs[3] / 'scope-training-outcome.json').exists()


def test_reference_memo_does_not_add_build_budget_rejection(inputs, monkeypatch):
    """Simulate an owner-approved future budget; current native cap is 4096."""
    import application.les_v7_dataset as builder
    import infrastructure.storage.les_v7_dataset as storage
    from domain.les_v7.reference_memo import MAX_REFERENCE_ENTRIES, NativeReferenceMemo
    original_validate = builder.validate_build_config
    def future_owner_validate(config):
        # Only the native event-budget contract is extended in this fixture.
        # The memo must clamp its own cache, never become a new training gate.
        original_validate(dict(config, max_events=min(config['max_events'], 4096)))
        assert 1 <= config['max_events'] <= 8192
    for module in (builder, storage, scope_training):
        monkeypatch.setattr(module, 'validate_build_config', future_owner_validate)
    bounds = []
    def observed_memo(*, max_entries):
        bounds.append(max_entries)
        return NativeReferenceMemo(max_entries=max_entries)
    monkeypatch.setattr(scope_training, 'NativeReferenceMemo', observed_memo)
    inputs[0]['config']['max_events'] = 8192
    outcome = run(inputs)
    assert outcome['status'] == 'NOT_READY'
    assert bounds == [MAX_REFERENCE_ENTRIES, MAX_REFERENCE_ENTRIES]
    assert scope_training.verify_elec_scope_training(inputs[3], outcome)['status'] == 'NOT_READY'


def test_scope_archive_verifies_whole_sharded_source_once(inputs, monkeypatch):
    outcome = run(inputs)
    root = inputs[3]
    original = scope_training.verify_sharded_training_dataset
    calls = []

    def observed(record, directory, **kwargs):
        calls.append((record['sha256'], Path(directory), kwargs.get('config')))
        return original(record, directory, **kwargs)

    monkeypatch.setattr(scope_training, 'verify_sharded_training_dataset', observed)
    receipt = scope_training.verify_elec_scope_training(root, outcome)
    assert receipt['job_id'] == outcome['native_job_id']
    assert len(calls) == 1
    assert calls[0][0] == outcome['manifest_id'] and calls[0][1] == root
    assert calls[0][2] == inputs[2]


def test_fresh_scope_reuses_repository_verification_but_public_reads_repeat_it(inputs, monkeypatch):
    original = scope_training.verify_sharded_training_dataset
    calls = []

    def observed(record, directory, **kwargs):
        calls.append((record['sha256'], Path(directory)))
        return original(record, directory, **kwargs)

    monkeypatch.setattr(scope_training, 'verify_sharded_training_dataset', observed)
    outcome = run(inputs)
    root = inputs[3]
    # Native preflight, prepared-row loading, and repository completion each
    # verify the complete scope. The outer publication reuses that last result.
    assert len(calls) == 3
    assert all(item == (outcome['manifest_id'], root) for item in calls)
    saved = (root / 'scope-training-outcome.json').read_bytes()
    assert json.loads(saved) == outcome
    receipt = scope_training.verify_elec_scope_training(root, outcome)
    assert len(calls) == 4, 'an independent public read must scan every original again'
    assert receipt['job_id'] == outcome['native_job_id']
    assert (root / 'scope-training-outcome.json').read_bytes() == saved


def test_private_scope_binding_checks_root_and_fit_bytes_after_verified_receipt(inputs):
    outcome = run(inputs)
    root = inputs[3]
    receipt, body = LesV7TrainingRepository._verify_with_scope(root / outcome['archive_path'])
    bind = scope_training._bind_verified_scope_outcome
    assert bind(root, outcome, receipt, body) == receipt
    manifest = root / 'scope-training-dataset.json'
    fit_path = root / outcome['archive_path'] / 'fit_result.json'
    saved_manifest, saved_fit = manifest.read_bytes(), fit_path.read_bytes()
    original_outcome = (root / 'scope-training-outcome.json').read_bytes()

    # A corrupt root record and a differently valid root record must both be
    # rejected, even when the receipt and already-verified body remain intact.
    changed = json.loads(saved_manifest)
    changed['payload']['counts']['prepared_rows'] += 1
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='DATASET_BINDING_MISMATCH'):
        bind(root, outcome, receipt, body)
    changed['sha256'] = identity(changed['payload'])
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='DATASET_BINDING_MISMATCH'):
        bind(root, outcome, receipt, body)
    manifest.write_bytes(saved_manifest)

    # Hashing only the fit payload would be weaker than the receipt's exact
    # artifact identity; verify a rehashed changed record is also rejected.
    changed_fit = json.loads(saved_fit)
    changed_fit['payload']['counts']['prepared'] += 1
    changed_fit['sha256'] = identity(changed_fit['payload'])
    fit_path.write_text(json.dumps(changed_fit))
    with pytest.raises(ValueError, match='FIT_RESULT_HASH_MISMATCH'):
        bind(root, outcome, receipt, body)
    fit_path.write_bytes(saved_fit)

    with pytest.raises(ValueError, match='OUTCOME_BINDING_MISMATCH'):
        bind(root, dict(outcome, counts=dict(prepared=999999)), receipt, body)
    with pytest.raises(ValueError, match='OUTCOME_SCOPE_MISMATCH'):
        bind(root, dict(outcome, approved_for_live=True), receipt, body)
    assert bind(root, outcome, receipt, body) == receipt
    assert (root / 'scope-training-outcome.json').read_bytes() == original_outcome
    assert scope_training.verify_elec_scope_training(root, outcome) == receipt


@pytest.mark.parametrize('rehash', [False, True])
def test_root_scope_manifest_still_bound_to_fully_verified_model_dataset(inputs, rehash):
    outcome = run(inputs)
    root = inputs[3]
    manifest = root / 'scope-training-dataset.json'
    record = json.loads(manifest.read_text())
    record['payload']['counts']['prepared_rows'] += 1
    if rehash:
        record['sha256'] = identity(record['payload'])
    manifest.write_text(canonical(record))
    with pytest.raises(ValueError, match='ELEC_SCOPE_TRAINING_DATASET_BINDING_MISMATCH'):
        scope_training.verify_elec_scope_training(root, outcome)


def test_scope_consumes_exact_verified_child_artifacts_once_then_releases_them(inputs, monkeypatch):
    import infrastructure.storage.les_v7_dataset as storage

    outcome = run(inputs)
    root = inputs[3]
    folders = {root / child['archive_path'] for child in outcome['build_children']}
    native_read = storage.read_json_bounded
    native_scope_read = scope_training._read
    native_verify = storage.LesV7DatasetRepository._verify_with_artifacts
    reads, retained = Counter(), []

    def read_once(path, limit):
        path = Path(path)
        if path.parent in folders:
            reads[path] += 1
        return native_read(path, limit)

    def forbid_reread(path):
        assert Path(path).parent not in folders, 'Scope reread already verified child JSON'
        return native_scope_read(path)

    def verify_child(*args, **kwargs):
        assert not any(reference() is not None for reference in retained)
        receipt, artifacts = native_verify(*args, **kwargs)
        retained.append(weakref.ref(artifacts))
        return receipt, artifacts

    monkeypatch.setattr(storage, 'read_json_bounded', read_once)
    monkeypatch.setattr(scope_training, '_read', forbid_reread)
    monkeypatch.setattr(storage.LesV7DatasetRepository, '_verify_with_artifacts', verify_child)
    receipt = scope_training.verify_elec_scope_training(root, outcome)
    names = {'completion.json', 'request.json', 'source.json', 'config.json', 'dataset.json', 'build_result.json'}
    assert receipt['job_id'] == outcome['native_job_id']
    assert reads == {folder / name: 1 for folder in folders for name in names}
    assert len(retained) == len(folders) and not any(reference() is not None for reference in retained)


def test_scope_rechecks_originals_after_consuming_verified_child_payloads(inputs, monkeypatch):
    outcome = run(inputs)
    root = inputs[3]
    path = root / outcome['build_children'][0]['archive_path'] / 'source.json'
    native_coverage = scope_training._coverage
    replaced = []

    def replace_after_coverage(*args):
        value = native_coverage(*args)
        if not replaced:
            replacement = path.with_suffix('.replacement')
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
            replaced.append(True)
        return value

    monkeypatch.setattr(scope_training, '_coverage', replace_after_coverage)
    with pytest.raises(ValueError, match='DATASET_BUILD_ARCHIVE_CHANGED:source.json'):
        scope_training.verify_elec_scope_training(root, outcome)
