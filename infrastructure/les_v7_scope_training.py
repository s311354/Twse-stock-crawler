"""Verified frozen ELEC shards, one native global research fit, one bundle.

Source normalization and event construction remain the existing native ports.
Full bars are released after each shard. Only bounded prepared numeric rows are
assembled for the unchanged global OOF/calibration/weight/holdout core.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import date
import fcntl
from pathlib import Path

from application.les_v7_dataset import validate_build_config
from domain.les_v7.contracts import ReferencePlan, canonical, identity, timestamp
from domain.les_v7.reference_memo import MAX_REFERENCE_ENTRIES, NativeReferenceMemo
from domain.les_v7.training import _train_prepared, prepare_events, reference_config_id, validate_training_config
from infrastructure.les_v7_dataset_scope import _verify_freeze
from infrastructure.les_v7_training_source import FINAL_CLOSE, OBSERVED_ONLY, load_training_source
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.les_v7_dataset import LesV7DatasetRepository
from infrastructure.storage.les_v7_training import LesV7TrainingRepository, MAX_CONFIG_BYTES, MAX_DATASET_BYTES, read_json_bounded
from infrastructure.storage.les_v7_prepared import inspect_prepared_packet, write_prepared_packet

DATASET_SCHEMA = 'les-v7-sharded-training-dataset/v1'
LEGACY_OUTCOME_SCHEMA = 'les-v7-elec-scope-training-outcome/v1'
OUTCOME_SCHEMA = 'les-v7-elec-scope-training-outcome/v2'
MAX_LEGACY_OUTCOME_BYTES = 64 * 1024
MAX_OUTCOME_BYTES = 2 * 1024 * 1024
LEGACY_PREPARED_SCHEMA = 'les-v7-native-prepared-training-shard/v1'
PREPARED_SCHEMA = 'les-v7-native-prepared-training-shard/v2'
DIAGNOSTICS_POLICY = 'SLIM_NATIVE_EVENT_STATUS_FULL_CHILD_DIAGNOSTICS_RETAINED_R1'
MAX_SYMBOLS = 2048
MAX_PREPARED_ROWS = 100000
MAX_PREPARED_BYTES = 128 * 1024 * 1024
MAX_PREPARED_DECODED_BYTES = 256 * 1024 * 1024
MAX_L2_PREPARED_DECODED_BYTES = 384 * 1024 * 1024


def prepared_decoded_limit(config, *, legacy_limit=None):
    """Bound the new source-proof shape without expanding old archive budgets.

    L2 retains the original plan/source binding on every prepared row. The
    457-stock R1 preparation already occupies 240 MB decoded; those additional
    proofs require a distinct finite budget, not a larger row/evidence scope.
    Encoded bytes, per-shard bounds and sample minima are unchanged.
    """
    reference = config.get("reference_policy", {})
    if reference.get("version") == "FR3_SOURCE_BOUND_LIMIT_H20_L2":
        from domain.les_v7.fr3_plan import validate_policy
        validate_policy(reference)
        return MAX_L2_PREPARED_DECODED_BYTES
    return MAX_PREPARED_DECODED_BYTES if legacy_limit is None else legacy_limit


def _validate_outcome_metadata(outcome):
    """Versioned metadata-only bounds; never a larger evidence/sample budget.

    New v2 outcomes retain the exact complete child/reference census. Existing
    v1 files are read under their original bound and are never rewritten.
    """
    limits = {LEGACY_OUTCOME_SCHEMA: MAX_LEGACY_OUTCOME_BYTES, OUTCOME_SCHEMA: MAX_OUTCOME_BYTES}
    if (not isinstance(outcome, dict) or outcome.get('schema') not in limits
            or outcome.get('research_only') is not True
            or outcome.get('approved_for_live') is not False or outcome.get('execution_action') != 'NO_ACTION'):
        raise ValueError('ELEC_SCOPE_TRAINING_OUTCOME_SCOPE_MISMATCH')
    if len(canonical(outcome).encode()) > limits[outcome['schema']]:
        raise ValueError('ELEC_SCOPE_TRAINING_OUTCOME_SIZE_LIMIT')


def inspect_elec_scope_freeze(path, verify_sources=False):
    """Native manifest inventory; only True also validates every raw source.

    The default does not claim source integrity/readiness. It verifies canonical
    scope identity, original universe-row binding and exact ordered shard union.
    """
    path = Path(path).expanduser().resolve()
    root = path.parent.parent.parent
    record = _verify_freeze(path, root, verify_sources=verify_sources)
    body = record['body']
    return dict(freeze_id=record['freeze_id'], scope_id=body['scope_id'], freeze_path=str(path),
        source_root=str(root), history_start=body['history_start'], history_end=body['history_end'],
        universe_date=body['universe_record']['source_day'], symbols=list(body['universe']),
        universe=deepcopy(body['universe_record']['rows']), dataset_kind=body['dataset_kind'],
        reference_evidence_policy=body.get('reference_evidence_policy'), batch_count=len(body['chunks']),
        batch_size=body['batch_size'], sources_verified=bool(verify_sources),
        historical_information_set_certified=False, research_only=True, approved_for_live=False)


def inspect_elec_scope(path):
    return inspect_elec_scope_freeze(path, verify_sources=True)


def _read(path):
    return read_json_bounded(path, MAX_DATASET_BYTES)


def _write(path, value):
    if len(canonical(value).encode()) > MAX_DATASET_BYTES:
        raise ValueError('ELEC_SCOPE_TRAINING_ARTIFACT_SIZE_LIMIT:' + Path(path).name)
    write_once(Path(path), value)


@contextmanager
def _lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.scope-training.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('ELEC_SCOPE_TRAINING_BUSY') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class _SharedGuard:
    def __init__(self, guard):
        self.guard, self.pinned = guard, None

    def check(self):
        return self.guard.check()

    def retain(self, objects):
        self.check()
        pinned = self.guard.retain(objects)
        if self.pinned is not None and self.pinned != pinned:
            raise ValueError('ELEC_SCOPE_TRAINING_PINNED_OBJECTS_CHANGED')
        self.pinned = pinned
        return dict(pinned)


def _validate_configs(build, training):
    if any(len(canonical(value).encode()) > MAX_CONFIG_BYTES for value in (build, training)):
        raise ValueError('ELEC_SCOPE_TRAINING_CONFIG_SIZE_LIMIT')
    validate_build_config(build)
    validate_training_config(training)
    if (reference_config_id(build['seed']) != reference_config_id(training)
            or timestamp(build['evaluation_cutoff']) != timestamp(training['evaluation_cutoff'])
            or build.get('reference_evidence_policy') != training.get('reference_evidence_policy')):
        raise ValueError('ELEC_SCOPE_TRAINING_CONFIG_BINDING_MISMATCH')


def _synthetic_settings(value):
    if isinstance(value, dict):
        return any(_synthetic_settings(key) or _synthetic_settings(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_synthetic_settings(item) for item in value)
    return isinstance(value, str) and 'SYNTHETIC' in value.upper()


def _pack_row(row, feature_columns=None):
    packed = dict(row, plan=asdict(row['plan']))
    # Every event otherwise repeats over a hundred feature names. Store the
    # common names once per shard, without dropping or rounding any feature.
    # Unusual rows keep their own mapping, preserving absent vs. null fields.
    if feature_columns is not None and sorted(row['features']) == feature_columns:
        packed['features'] = [row['features'][name] for name in feature_columns]
    return packed


def _unpack_row(row, feature_columns=None):
    from domain.les_v7.fr3_plan import is_fr3_reference
    if is_fr3_reference({"version": row["plan"].get("policy_version")}):
        from domain.les_v7.fr3_plan import plan_from_record
        plan = plan_from_record(row["plan"])
    else:
        plan = ReferencePlan(**dict(row['plan'], band=tuple(row['plan']['band'])))
    unpacked = dict(row, plan=plan)
    if isinstance(row['features'], list):
        if feature_columns is None or len(row['features']) != len(feature_columns):
            raise ValueError('ELEC_SCOPE_TRAINING_PREPARED_FEATURE_BINDING_MISMATCH')
        unpacked['features'] = dict(zip(feature_columns, row['features']))
    elif not isinstance(row['features'], dict):
        raise ValueError('ELEC_SCOPE_TRAINING_PREPARED_FEATURE_BINDING_MISMATCH')
    return unpacked


def _prepared_packet(dataset_id, config_id, rows, diagnostics):
    columns = sorted(rows[0]['features']) if rows else []
    return dict(schema=PREPARED_SCHEMA, dataset_id=dataset_id, config_id=config_id,
                feature_columns=columns, rows=[_pack_row(row, columns) for row in rows],
                diagnostics=_slim_diagnostics(diagnostics))


def _packet_feature_columns(packet):
    if packet.get('schema') == LEGACY_PREPARED_SCHEMA:
        return None
    columns = packet.get('feature_columns')
    if (packet.get('schema') != PREPARED_SCHEMA or not isinstance(columns, list)
            or any(not isinstance(name, str) or not name for name in columns)
            or columns != sorted(set(columns))):
        raise ValueError('ELEC_SCOPE_TRAINING_PREPARED_FEATURE_BINDING_MISMATCH')
    return columns


def _slim_diagnostics(diagnostics):
    keys = ('row_id', 'symbol', 'market', 'cutoff', 'context', 'excluded_reason', 'label_status')
    return [{key: row[key] for key in keys if key in row} for row in diagnostics]


def _child_summary(child):
    keys = ('index', 'evidence_id', 'build_job_id', 'dataset_id', 'completion_sha256',
            'archive_path', 'prepared_sha256')
    return {key: child[key] for key in keys}


def _coverage(source, result, rows, diagnostics):
    by_symbol = Counter(row['symbol'] for row in rows)
    candidates = Counter(row['symbol'] for row in result['diagnostics'])
    output = []
    for history in source['payload']['histories']:
        symbol = history['symbol']
        build_reasons = Counter(reason for row in result['diagnostics'] if row['symbol'] == symbol for reason in row['reasons'])
        training_reasons = Counter(row['excluded_reason'] for row in diagnostics
            if row['symbol'] == symbol and row.get('excluded_reason'))
        output.append(dict(symbol=symbol, market=history['market'],
            security_type=history.get('security_type', 'COMMON'), normalized_bar_count=len(history['bars']),
            candidate_events=candidates[symbol], prepared_rows=by_symbol[symbol],
            build_reason_counts=dict(sorted(build_reasons.items())),
            training_reason_counts=dict(sorted(training_reasons.items()))))
    return output


def _relative(root, relative, prefix):
    part = Path(relative)
    path = (root / part).resolve()
    if part.is_absolute() or '..' in part.parts or not path.is_relative_to(root / prefix):
        raise ValueError('ELEC_SCOPE_TRAINING_REFERENCE_PATH_INVALID')
    return path


def verify_sharded_training_dataset(record, root, *, config=None):
    """Verify every child, prepared object and whole-universe binding, sequentially.

    This checks native immutable bytes and provenance; it does not refit or
    regenerate data. Public orchestration never accepts caller prepared rows.
    """
    root = Path(root).resolve()
    body = record.get('payload', {})
    if (record.get('sha256') != identity(body) or body.get('schema') != DATASET_SCHEMA
            or body.get('dataset_kind') not in {'RESEARCH', 'SYNTHETIC_TEST_ONLY'}
            or body.get('research_only') is not True or body.get('approved_for_live') is not False
            or body.get('execution_action') != 'NO_ACTION'
            or (body.get('price_unit'), body.get('volume_unit'), body.get('amount_unit')) != ('TWD', 'SHARES', 'TWD')
            or body.get('diagnostics_policy') != DIAGNOSTICS_POLICY):
        raise ValueError('ELEC_SCOPE_TRAINING_DATASET_BINDING_MISMATCH')
    scope = inspect_elec_scope_freeze(body['freeze_path'], verify_sources=True)
    if (scope['freeze_id'] != body.get('freeze_id') or scope['scope_id'] != body.get('scope_id')
            or scope['source_root'] != body.get('source_root') or scope['dataset_kind'] != body['dataset_kind']
            or body.get('universe') != [[symbol, 'TWSE'] for symbol in scope['symbols']]
            or body.get('reference_evidence_policy') != scope['reference_evidence_policy']
            or body.get('availability_policy') not in {OBSERVED_ONLY, FINAL_CLOSE}
            or not 1 <= len(body['universe']) <= MAX_SYMBOLS):
        raise ValueError('ELEC_SCOPE_TRAINING_SCOPE_BINDING_MISMATCH')
    build = _read(root / 'scope-build-config.json')
    training = _read(root / 'scope-training-config.json')
    _validate_configs(build, training)
    if (identity(build) != body.get('build_config_id') or identity(training) != body.get('training_config_id')
            or body.get('reference_config_id') != reference_config_id(training)
            or config is not None and config != training):
        raise ValueError('ELEC_SCOPE_TRAINING_CONFIG_BINDING_MISMATCH')
    freeze = _read(body['freeze_path'])['body']
    children = body.get('children')
    if not isinstance(children, list) or len(children) != len(freeze['chunks']):
        raise ValueError('ELEC_SCOPE_TRAINING_CHILD_COVERAGE_MISMATCH')
    seen, symbols, totals, coverage = set(), [], Counter(), []
    prepared_bytes = prepared_decoded_bytes = 0
    for child, chunk in zip(children, freeze['chunks']):
        if (child.get('index') != chunk['index'] or child.get('symbols') != chunk['symbols']
                or child.get('evidence_id') != chunk['evidence_id']):
            raise ValueError('ELEC_SCOPE_TRAINING_CHILD_COVERAGE_MISMATCH')
        folder = _relative(root, child['archive_path'], 'builds')
        receipt, artifacts = LesV7DatasetRepository._verify_with_artifacts(folder)
        if (receipt['job_id'] != child['build_job_id'] or receipt['dataset_id'] != child['dataset_id']
                or identity(artifacts['completion.json']) != child['completion_sha256']
                or receipt['source_packet_id'] != child['source_packet_id']
                or receipt['config_id'] != body['build_config_id']
                or receipt['pinned_objects'] != body['pinned_objects']
                or receipt['dataset_kind'] != body['dataset_kind'] or receipt['counts'] != child['build_counts']):
            raise ValueError('ELEC_SCOPE_TRAINING_CHILD_RECEIPT_MISMATCH')
        source = artifacts['source.json']
        dataset = artifacts['dataset.json']
        if (source['payload']['source_evidence_ids'] != [child['evidence_id']]
                or source['payload']['availability_policy'] != body['availability_policy']
                or source['payload']['universe'] != [[symbol, 'TWSE'] for symbol in child['symbols']]
                or timestamp(source['payload']['lineage']['evaluation_cutoff']) != timestamp(training['evaluation_cutoff'])
                or dataset['payload']['source_evidence_ids'] != [child['evidence_id']]
                or dataset['payload']['universe'] != source['payload']['universe']
                or source['payload'].get('reference_evidence_policy') != body.get('reference_evidence_policy')):
            raise ValueError('ELEC_SCOPE_TRAINING_CHILD_SOURCE_MISMATCH')
        prepared_path = _relative(root, child['prepared_path'], 'prepared')
        prepared = inspect_prepared_packet(prepared_path)
        prepared_bytes += prepared.encoded_bytes
        prepared_decoded_bytes += prepared.decoded_bytes
        packet = prepared.value
        if (identity(packet) != child['prepared_sha256']
                or packet.get('schema') not in {PREPARED_SCHEMA, LEGACY_PREPARED_SCHEMA}
                or packet.get('dataset_id') != child['dataset_id']
                or packet.get('config_id') != body['training_config_id']
                or len(packet['rows']) != child['prepared_count']
                or len(packet['diagnostics']) != child['diagnostic_count']
                or len(packet['rows']) != receipt['counts']['eligible_events']
                or len(packet['diagnostics']) != len(dataset['payload']['events'])):
            raise ValueError('ELEC_SCOPE_TRAINING_PREPARED_REFERENCE_MISMATCH')
        feature_columns = _packet_feature_columns(packet)
        source_events = {(event['symbol'], timestamp(event['cutoff']).isoformat()) for event in dataset['payload']['events']}
        for row in packet['rows']:
            expected_id = identity(dict(symbol=row['symbol'], market=row['market'], cutoff=timestamp(row['cutoff']).isoformat()))
            if (row['row_id'] != expected_id or expected_id in seen or row['symbol'] not in child['symbols']
                    or row['market'] != 'TWSE' or (row['symbol'], row['cutoff']) not in source_events
                    or timestamp(row['cutoff']) >= timestamp(row['label_available_at'])
                    or timestamp(row['label_available_at']) > timestamp(training['evaluation_cutoff'])):
                raise ValueError('ELEC_SCOPE_TRAINING_DUPLICATE_OR_UNBOUND_EVENT')
            _unpack_row(row, feature_columns)
            seen.add(expected_id)
        if child['coverage'] != _coverage(source, artifacts['build_result.json'], packet['rows'], packet['diagnostics']):
            raise ValueError('ELEC_SCOPE_TRAINING_COVERAGE_MISMATCH')
        coverage.extend(child['coverage'])
        symbols.extend(child['symbols'])
        totals.update(prepared_rows=len(packet['rows']), diagnostic_rows=len(packet['diagnostics']))
        for key in ('candidate_events', 'path_events', 'eligible_events'):
            totals[key] += child['build_counts'][key]
        artifacts.check()
        del source, dataset, packet, artifacts, prepared
        if (totals['prepared_rows'] > MAX_PREPARED_ROWS or prepared_bytes > MAX_PREPARED_BYTES
                or prepared_decoded_bytes > prepared_decoded_limit(training)):
            raise ValueError('ELEC_SCOPE_TRAINING_PREPARED_BUDGET_EXCEEDED')
    if (symbols != scope['symbols'] or len(symbols) != len(set(symbols))
            or sorted(row['symbol'] for row in coverage) != scope['symbols']
            or body.get('source_evidence_ids') != [chunk['evidence_id'] for chunk in freeze['chunks']]
            or body.get('counts') != dict(totals) or body.get('coverage_sha256') != identity(coverage)):
        raise ValueError('ELEC_SCOPE_TRAINING_FULL_COVERAGE_MISMATCH')
    return body


def _prepared_rows(record, config, root):
    body = verify_sharded_training_dataset(record, root, config=config)
    rows, diagnostics = [], []
    prepared_bytes = prepared_decoded_bytes = 0
    for child in body['children']:
        prepared = inspect_prepared_packet(_relative(root, child['prepared_path'], 'prepared'))
        prepared_bytes += prepared.encoded_bytes
        prepared_decoded_bytes += prepared.decoded_bytes
        packet = prepared.value
        if (identity(packet) != child['prepared_sha256'] or prepared_bytes > MAX_PREPARED_BYTES
                or prepared_decoded_bytes > prepared_decoded_limit(config)):
            raise ValueError('ELEC_SCOPE_TRAINING_PREPARED_REFERENCE_OR_BUDGET_CHANGED')
        feature_columns = _packet_feature_columns(packet)
        rows.extend(_unpack_row(row, feature_columns) for row in packet['rows'])
        diagnostics.extend(packet['diagnostics'])
        del packet, prepared
        if len(rows) > MAX_PREPARED_ROWS:
            raise ValueError('ELEC_SCOPE_TRAINING_PREPARED_BUDGET_EXCEEDED')
    rows.sort(key=lambda row: (timestamp(row['cutoff']), row['symbol'], row['market']))
    diagnostics.sort(key=lambda row: (timestamp(row['cutoff']), row['symbol'], row['market']))
    return rows, diagnostics


def _prepare_scope_shards(scope, build_config, training_config, root, *, availability_policy,
                          allow_synthetic, guard, pinned, clock, progress):
    """One guarded preparation implementation shared by prepare-only and fit."""
    freeze = _read(scope['freeze_path'])['body']
    children, totals, coverage = [], Counter(), []
    prepared_bytes = prepared_decoded_bytes = 0
    for chunk in freeze['chunks']:
        if progress:
            progress(f"全股票資料建置與驗證 {chunk['index'] + 1} / {len(freeze['chunks'])}")
        guard.check()
        source_path = Path(scope['source_root']) / 'evidence' / chunk['evidence_id'] / 'evidence.json'
        source = load_training_source(source_path, evaluation_cutoff=build_config['evaluation_cutoff'],
            availability_policy=availability_policy,
            **({'reference_evidence_policy': build_config['reference_evidence_policy']}
               if build_config.get('reference_evidence_policy') else {}))
        # The same native evidence/config may be prepared three times by
        # build, label verification and this shard. Retain detached numeric
        # results only, and release the bounded memo at every shard boundary.
        with NativeReferenceMemo(max_entries=min(build_config['max_events'], MAX_REFERENCE_ENTRIES)):
            receipt = LesV7DatasetRepository(root / 'builds', guard_factory=lambda: guard, clock=clock).run(
                source, build_config, allow_synthetic=allow_synthetic)
            dataset = _read(Path(receipt['output']) / 'dataset.json')
            rows, diagnostics = prepare_events(dataset, training_config)
        packet = _prepared_packet(dataset['sha256'], identity(training_config), rows, diagnostics)
        prepared_path = 'prepared/' + str(chunk['index']) + '.json.gz'
        prepared = write_prepared_packet(root / prepared_path, packet)
        child_coverage = _coverage(source, _read(Path(receipt['output']) / 'build_result.json'), rows, diagnostics)
        child = dict(index=chunk['index'], symbols=chunk['symbols'], evidence_id=chunk['evidence_id'],
            build_job_id=receipt['job_id'], dataset_id=receipt['dataset_id'], source_packet_id=receipt['source_packet_id'],
            completion_sha256=identity(_read(Path(receipt['output']) / 'completion.json')),
            archive_path=str(Path(receipt['output']).resolve().relative_to(root)), build_counts=receipt['counts'],
            prepared_path=prepared_path, prepared_sha256=identity(packet), prepared_count=len(rows),
            diagnostic_count=len(diagnostics), coverage=child_coverage)
        children.append(child)
        coverage.extend(child_coverage)
        totals.update(prepared_rows=len(rows), diagnostic_rows=len(diagnostics))
        for key in ('candidate_events', 'path_events', 'eligible_events'):
            totals[key] += receipt['counts'][key]
        prepared_bytes += prepared.encoded_bytes
        prepared_decoded_bytes += prepared.decoded_bytes
        del source, dataset, packet, rows, diagnostics, prepared
        if (totals['prepared_rows'] > MAX_PREPARED_ROWS or prepared_bytes > MAX_PREPARED_BYTES
                or prepared_decoded_bytes > prepared_decoded_limit(training_config)):
            raise ValueError('ELEC_SCOPE_TRAINING_PREPARED_BUDGET_EXCEEDED')
    payload = dict(schema=DATASET_SCHEMA, dataset_kind=scope['dataset_kind'], freeze_id=scope['freeze_id'],
        price_unit='TWD', volume_unit='SHARES', amount_unit='TWD',
        scope_id=scope['scope_id'], freeze_path=scope['freeze_path'], source_root=scope['source_root'],
        universe=[[symbol, 'TWSE'] for symbol in scope['symbols']],
        universe_policy='ALL_FROZEN_OFFICIAL_FINAL_ELEC_ROWS_NATIVE_EXCLUSIONS_NOT_CANDIDATE_PIT_MEMBERSHIP',
        source_evidence_ids=[child['evidence_id'] for child in children], children=children,
        build_config_id=identity(build_config), training_config_id=identity(training_config),
        reference_config_id=reference_config_id(training_config),
        reference_evidence_policy=build_config.get('reference_evidence_policy'), availability_policy=availability_policy,
        pinned_objects=pinned, counts=dict(totals), coverage_sha256=identity(coverage),
        diagnostics_policy=DIAGNOSTICS_POLICY, research_only=True, approved_for_live=False, execution_action='NO_ACTION')
    record = dict(payload=payload, sha256=identity(payload))
    _write(root / 'scope-training-dataset.json', record)
    return record


def _preparation_result(root, record):
    body = record['payload']
    return dict(schema='les-v7-elec-scope-preparation/v1', status='DATASET_PREPARED',
        dataset_id=record['sha256'], dataset_path=str(root / 'scope-training-dataset.json'),
        freeze_id=body['freeze_id'], scope_id=body['scope_id'],
        build_config_id=body['build_config_id'], config_id=body['training_config_id'],
        symbol_count=len(body['universe']), batch_count=len(body['children']), counts=deepcopy(body['counts']),
        pinned_objects=deepcopy(body['pinned_objects']), operation='PREPARE_ONLY',
        model_fitted=False, holdout_evaluated=False, model_published=False, model_activated=False,
        research_only=True, approved_for_live=False, execution_action='NO_ACTION')


def prepare_elec_scope_dataset(freeze_path, build_config, training_config, output_root, progress=None, *,
                              allow_synthetic=False, availability_policy=FINAL_CLOSE,
                              guard_factory=ResearchVersionGuard, clock=utc_now):
    """Build/verify native prepared evidence; never fit or evaluate any model.

    Recovery requires the exact frozen scope, complete config and pinned source
    identities. No model or training-completion outcome is written by this port.
    """
    return train_elec_scope(freeze_path, build_config, training_config, output_root, progress,
        allow_synthetic=allow_synthetic, availability_policy=availability_policy,
        guard_factory=guard_factory, clock=clock, prepare_only=True)


def train_elec_scope(freeze_path, build_config, training_config, output_root, progress=None, *,
                     allow_synthetic=False, availability_policy=FINAL_CLOSE,
                     guard_factory=ResearchVersionGuard, clock=utc_now, prepare_only=False):
    """Public all-frozen-symbols request; preparation and model fitting are distinct."""
    if type(prepare_only) is not bool:
        raise ValueError('ELEC_SCOPE_PREPARE_ONLY_BOOLEAN_REQUIRED')
    build_config, training_config = deepcopy(build_config), deepcopy(training_config)
    _validate_configs(build_config, training_config)
    if availability_policy not in {FINAL_CLOSE, OBSERVED_ONLY}:
        raise ValueError('ELEC_SCOPE_TRAINING_AVAILABILITY_POLICY_INVALID')
    scope = inspect_elec_scope_freeze(freeze_path, verify_sources=True)
    if scope['dataset_kind'] == 'SYNTHETIC_TEST_ONLY' and allow_synthetic is not True:
        raise ValueError('SYNTHETIC_SCOPE_TRAINING_REQUIRES_EXPLICIT_OPT_IN')
    if scope['dataset_kind'] == 'RESEARCH' and (_synthetic_settings(build_config) or _synthetic_settings(training_config)):
        raise ValueError('SYNTHETIC_SETTINGS_CANNOT_TRAIN_REAL_ELEC_SCOPE')
    if (build_config.get('reference_evidence_policy') != scope['reference_evidence_policy']
            or build_config.get('reference_evidence_policy') and availability_policy != FINAL_CLOSE):
        raise ValueError('ELEC_SCOPE_TRAINING_REFERENCE_POLICY_MISMATCH')
    if timestamp(clock()) < timestamp(training_config['evaluation_cutoff']):
        raise ValueError('ELEC_SCOPE_TRAINING_EVALUATION_AFTER_CLOCK')
    days = (date.fromisoformat(build_config['event_end']) - date.fromisoformat(build_config['event_start'])).days + 1
    if (not scope['history_start'] <= build_config['event_start'] <= build_config['event_end'] <= scope['history_end']
            or days * scope['batch_size'] > build_config['max_events']):
        raise ValueError('ELEC_SCOPE_TRAINING_WINDOW_OR_SHARD_BUDGET_MISMATCH')
    root = Path(output_root).resolve()
    with _lock(root):
        completion = root / 'scope-training-outcome.json'
        if completion.exists():
            old = read_json_bounded(completion, MAX_OUTCOME_BYTES)
            if (old['freeze_id'] != scope['freeze_id'] or old['build_config_id'] != identity(build_config)
                    or old['config_id'] != identity(training_config) or old['availability_policy'] != availability_policy):
                raise ValueError('ELEC_SCOPE_TRAINING_RECOVERY_REQUEST_MISMATCH')
            verify_elec_scope_training(root, old)
            return _preparation_result(root, _read(root / 'scope-training-dataset.json')) if prepare_only else old
        guard = _SharedGuard(guard_factory())
        pinned = guard.retain(ResearchObjects(root / 'builds' / 'objects'))
        dataset_path = root / 'scope-training-dataset.json'
        if dataset_path.exists():
            record = _read(dataset_path)
            body = record.get('payload', {})
            if (body.get('freeze_id') != scope['freeze_id'] or body.get('scope_id') != scope['scope_id']
                    or body.get('build_config_id') != identity(build_config)
                    or body.get('training_config_id') != identity(training_config)
                    or body.get('availability_policy') != availability_policy):
                raise ValueError('ELEC_SCOPE_TRAINING_RECOVERY_REQUEST_MISMATCH')
            if body.get('pinned_objects') != pinned:
                raise ValueError('ELEC_SCOPE_TRAINING_RECOVERY_SOURCE_CHANGED')
        else:
            _write(root / 'scope-build-config.json', build_config)
            _write(root / 'scope-training-config.json', training_config)
            record = _prepare_scope_shards(scope, build_config, training_config, root,
                availability_policy=availability_policy, allow_synthetic=allow_synthetic,
                guard=guard, pinned=pinned, clock=clock, progress=progress)
        children, totals = record['payload']['children'], record['payload']['counts']
        if prepare_only:
            verify_sharded_training_dataset(record, root, config=training_config)
            guard.check()
            return _preparation_result(root, record)
        guard.check()
        if progress:
            progress('全股票共用 OOF／校準／權重與獨立驗證')
        def trainer(dataset, config):
            rows, diagnostics = _prepared_rows(dataset, config, root)
            try:
                return _train_prepared(dataset, config, rows, diagnostics)
            finally:
                del rows, diagnostics
        receipt = LesV7TrainingRepository(root / 'models', guard_factory=lambda: guard, clock=clock).run(
            record, training_config, trainer=trainer)
        guard.check()
        folder = Path(receipt['output'])
        fit = _read(folder / 'fit_result.json')['payload']
        outcome = dict(schema=OUTCOME_SCHEMA, freeze_id=scope['freeze_id'], scope_id=scope['scope_id'],
            symbols=scope['symbols'], availability_policy=availability_policy,
            build_config_id=identity(build_config), config_id=identity(training_config),
            native_job_id=receipt['job_id'], dataset_id=receipt['dataset_id'], manifest_id=record['sha256'],
            status=receipt['status'], dataset_kind=scope['dataset_kind'], bundle_sha256=receipt['bundle_sha256'],
            first_published_at=receipt['first_published_at'], archive_path=str(folder.resolve().relative_to(root)),
            bundle_path=str((folder / 'frozen_bundle.json').resolve().relative_to(root)) if receipt['bundle_sha256'] else None,
            symbol_count=len(scope['symbols']), batch_count=len(children), counts=fit['counts'],
            reason_codes=fit['reason_codes'], report_references=fit['report_references'],
            build_children=[_child_summary(child) for child in children], coverage_counts=dict(totals),
            native_training={key: value for key, value in receipt.items() if key != 'output'},
            research_only=True, approved_for_live=False, execution_action='NO_ACTION')
        del fit
        _validate_outcome_metadata(outcome)
        # Repository.run just performed complete original/child/model verification.
        # Bind its exact verified dataset identity to the independently saved
        # scope outcome without immediately scanning the same files again.
        _bind_verified_scope_outcome(root, outcome, receipt, record['payload'])
        guard.check()
        _write(completion, outcome)
        return outcome


def verify_elec_scope_training(output_root, outcome):
    """Download/selection verification: all original sources, children and model."""
    root = Path(output_root).resolve()
    _validate_outcome_metadata(outcome)
    folder = _relative(root, outcome['archive_path'], 'models')
    receipt, body = LesV7TrainingRepository._verify_with_scope(folder)
    return _bind_verified_scope_outcome(root, outcome, receipt, body)


def _bind_verified_scope_outcome(output_root, outcome, receipt, body):
    """Private binding following a complete repository run/verify in this call.

    No persisted success cache or caller-provided verification flag exists.
    The public verifier always independently verifies every referenced source.
    """
    root = Path(output_root).resolve()
    _validate_outcome_metadata(outcome)
    record = _read(root / 'scope-training-dataset.json')
    if (not isinstance(record.get('payload'), dict)
            or record.get('sha256') != identity(record['payload'])):
        raise ValueError('ELEC_SCOPE_TRAINING_DATASET_BINDING_MISMATCH')
    folder = _relative(root, outcome['archive_path'], 'models')
    # The native model verifier already read every original/child/prepared
    # object. Bind that exact verified body to the independent root manifest;
    # a declared hash alone must never conceal a changed root payload.
    if body is None or record['payload'] != body:
        raise ValueError('ELEC_SCOPE_TRAINING_DATASET_BINDING_MISMATCH')
    fit_record = _read(folder / 'fit_result.json')
    if identity(fit_record) != receipt['artifacts'].get('fit_result.json'):
        raise ValueError('ELEC_SCOPE_TRAINING_FIT_RESULT_HASH_MISMATCH')
    fit = fit_record['payload']
    if (record['sha256'] != receipt['dataset_id'] or outcome.get('manifest_id') != record['sha256']
            or outcome.get('freeze_id') != body['freeze_id'] or outcome.get('scope_id') != body['scope_id']
            or outcome.get('symbols') != [row[0] for row in body['universe']]
            or outcome.get('build_config_id') != body['build_config_id'] or outcome.get('config_id') != body['training_config_id']
            or outcome.get('availability_policy') != body['availability_policy']
            or outcome.get('build_children') != [_child_summary(child) for child in body['children']]
            or outcome.get('coverage_counts') != body['counts']
            or outcome.get('symbol_count') != len(body['universe']) or outcome.get('batch_count') != len(body['children'])
            or outcome.get('counts') != fit['counts'] or outcome.get('reason_codes') != fit['reason_codes']
            or outcome.get('report_references') != fit['report_references']
            or outcome.get('native_training') != {key: value for key, value in receipt.items() if key != 'output'}
            or any(outcome.get(key) != receipt[target] for key, target in (
                ('native_job_id', 'job_id'), ('dataset_id', 'dataset_id'), ('status', 'status'),
                ('dataset_kind', 'dataset_kind'), ('bundle_sha256', 'bundle_sha256'), ('first_published_at', 'first_published_at')))
            or outcome.get('bundle_path') != (str((folder / 'frozen_bundle.json').relative_to(root))
                if receipt['bundle_sha256'] else None)):
        raise ValueError('ELEC_SCOPE_TRAINING_OUTCOME_BINDING_MISMATCH')
    return receipt
