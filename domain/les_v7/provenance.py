"""R2 runtime provenance: exact archived ID arrays, compact bound references.

No samples, time rules, acceptance criteria or numerical model fields change.
Raw reports/folds/OOF/weight provenance remain in the native immutable archive.
"""
from copy import deepcopy
import re

from domain.les_v7.contracts import CONTEXTS, PRIOR, identity, timestamp

RUNTIME_SCHEMA = 'LES_V7_RUNTIME_PROVENANCE_R2'
ARCHIVE_SCHEMA = 'les-v7-archived-training-provenance/v2'
REFERENCE_SCHEMA = 'les-v7-id-array-reference/v1'
REPORT_SCHEMA = 'les-v7-compact-validation-report/v2'
ID_FIELDS = {'training_row_ids', 'prediction_row_ids', 'row_ids', 'oof_prediction_ids',
             'weight_prediction_ids', 'child_dataset_ids'}
REPORT_NAMES = {'calibration', 'holdout', 'fractal_null', 'projection', 'ablation'}
SUPPORT_FIELDS = ('min_head_rows', 'min_calibration_rows', 'min_weight_rows',
                  'min_context_weight_rows', 'min_holdout_rows')


def id_reference(values):
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError('R2_ORDERED_ID_ARRAY_REQUIRED')
    return dict(schema=REFERENCE_SCHEMA, sha256=identity(values), count=len(values))


def _compact_arrays(value):
    if isinstance(value, dict):
        return {key: id_reference(item) if key in ID_FIELDS else _compact_arrays(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_arrays(item) for item in value]
    return deepcopy(value)


def compact_report(source_sha256, report):
    if identity(report) != source_sha256:
        raise ValueError('R2_RAW_VALIDATION_REPORT_HASH_MISMATCH')
    return dict(schema=REPORT_SCHEMA, source_report_sha256=source_sha256, report=_compact_arrays(report))


def archive_provenance(config, report_ids, weight_ids, weights):
    return dict(schema=ARCHIVE_SCHEMA, archive_validation_report_ids=deepcopy(report_ids),
        weight_prediction_ids=list(weight_ids), weight_fit=deepcopy(weights),
        support_requirements={key: config[key] for key in SUPPORT_FIELDS})


def compact_candidate(candidate, provenance):
    """Transform representation only; inputs stay detached and raw IDs retained."""
    final = deepcopy(candidate)
    final.pop('bundle_id', None)
    raw_ids = final['validation_report_ids']
    if raw_ids != provenance['archive_validation_report_ids']:
        raise ValueError('R2_ARCHIVE_REPORT_ID_BINDING_MISMATCH')
    reports, ids = {}, {}
    for name, raw_id in raw_ids.items():
        report = compact_report(raw_id, final['validation_reports'][raw_id])
        report_id = identity(report)
        reports[report_id], ids[name] = report, report_id
    final.update(runtime_provenance_schema=RUNTIME_SCHEMA,
        archive_validation_report_ids=deepcopy(raw_ids), validation_report_ids=ids, validation_reports=reports,
        fractal_null_validation_report_id=ids['fractal_null'])
    final['calibration'].update(validation_report_id=ids['calibration'], source_validation_report_sha256=raw_ids['calibration'])
    manifest = _compact_arrays(final['training_manifest'])
    manifest['support_requirements'] = deepcopy(provenance['support_requirements'])
    final['training_manifest'] = manifest
    final['bundle_id'] = identity(final)
    return final


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _ref(value, *, minimum=1, maximum=100000):
    if (not isinstance(value, dict) or set(value) != {'schema', 'sha256', 'count'}
            or value.get('schema') != REFERENCE_SCHEMA or not _hash(value.get('sha256'))
            or type(value.get('count')) is not int or not minimum <= value['count'] <= maximum):
        raise ValueError('R2_BOUNDED_ID_REFERENCE_REQUIRED')
    return value['count']


def validate_runtime_provenance(payload):
    """R2 standalone shape/time/support/gate checks, not raw-source certification.

    The native repository additionally reconstructs these exact digests and
    summaries from complete raw archive objects before publication or loading.
    """
    if (payload.get('runtime_provenance_schema') != RUNTIME_SCHEMA
            or payload.get('training_pipeline_schema') != 'LES_V7_OFFLINE_TRAINING_R1'
            or payload.get('research_only') is not True or payload.get('approved_for_live') is not False):
        raise ValueError('R2_RUNTIME_PROVENANCE_SCHEMA_REQUIRED')
    manifest = payload['training_manifest']
    if (manifest.get('upstream_predictions') != 'STRICTLY_PAST_OOF'
            or manifest.get('dataset_schema') != 'les-v7-sharded-training-dataset/v1'
            or not all(_hash(manifest.get(key)) for key in ('dataset_id', 'config_id', 'freeze_id', 'scope_id', 'full_universe_sha256'))):
        raise ValueError('R2_FULL_SCOPE_MANIFEST_REQUIRED')
    _ref(manifest.get('child_dataset_ids'), maximum=2048)
    oof_count = _ref(manifest.get('oof_prediction_ids'))
    weight_count = _ref(manifest.get('weight_prediction_ids'))
    counts = manifest['counts']
    count_keys = ('requested', 'prepared', 'final_training', 'holdout', 'oof', 'weight_rows')
    if (not isinstance(counts, dict)
            or any(type(counts.get(key)) is not int or not 0 <= counts[key] <=
                   (2048 * 4096 if key == 'requested' else 100000) for key in count_keys)
            or counts['final_training'] > counts['prepared'] or counts['holdout'] > counts['prepared']
            or counts['oof'] > counts['prepared'] or counts['prepared'] > counts['requested']
            or not isinstance(counts.get('weight_contexts'), dict)
            or any(type(value) is not int or not 0 <= value <= 100000 for value in counts['weight_contexts'].values())
            or set(counts['weight_contexts']) != set(CONTEXTS)
            or sum(counts['weight_contexts'].values()) != weight_count):
        raise ValueError('R2_BOUNDED_NATIVE_COUNTS_REQUIRED')
    requirements = manifest.get('support_requirements')
    if (not isinstance(requirements, dict) or set(requirements) != set(SUPPORT_FIELDS)
            or any(type(value) is not int or not 1 <= value <= 100000 for value in requirements.values())
            or counts.get('oof') != oof_count or counts.get('weight_rows') != weight_count
            or weight_count > oof_count or counts['final_training'] < requirements['min_head_rows']
            or counts['holdout'] < requirements['min_holdout_rows'] or weight_count < requirements['min_weight_rows']
            or any(counts.get('weight_contexts', {}).get(context, 0) < requirements['min_context_weight_rows'] for context in CONTEXTS)):
        raise ValueError('R2_SUPPORT_COUNT_BINDING_MISMATCH')
    folds = manifest.get('folds')
    # Native daily contexts freeze JSON arrays as tuples. Validate that same
    # ordered array in place; thawing the evidence graph is not required.
    if (not isinstance(folds, (list, tuple)) or not 1 <= len(folds) <= 64
            or any(not isinstance(fold, dict) for fold in folds)
            or len({fold.get('fold_id') for fold in folds}) != len(folds)):
        raise ValueError('R2_OOF_FOLDS_REQUIRED')
    previous = None
    for fold in folds:
        if (fold.get('status') != 'FITTED' or not _hash(fold.get('heads_sha256'))
                or not timestamp(fold['last_training_label_available_at']) <= timestamp(fold['fit_cutoff'])
                     < timestamp(fold['prediction_at']) <= timestamp(fold['prediction_end']) < timestamp(payload['fit_cutoff'])
                or _ref(fold.get('training_row_ids')) < requirements['min_head_rows']):
            raise ValueError('R2_OOF_TIME_OR_SUPPORT_MISMATCH')
        _ref(fold.get('prediction_row_ids'))
        if previous is not None and timestamp(fold['prediction_at']) <= previous:
            raise ValueError('R2_OOF_TIME_OR_SUPPORT_MISMATCH')
        previous = timestamp(fold['prediction_end'])
    reports, ids, raw_ids = payload.get('validation_reports', {}), payload.get('validation_report_ids', {}), payload.get('archive_validation_report_ids', {})
    if (set(ids) != REPORT_NAMES or set(raw_ids) != REPORT_NAMES
            or set(reports) != set(ids.values()) or any(not _hash(value) for value in raw_ids.values())):
        raise ValueError('R2_COMPLETE_BOUND_VALIDATION_REPORTS_REQUIRED')
    parsed = {}
    for name, ref in ids.items():
        wrapper = reports[ref]
        if (identity(wrapper) != ref or wrapper.get('schema') != REPORT_SCHEMA
                or wrapper.get('source_report_sha256') != raw_ids[name] or set(wrapper) != {'schema', 'source_report_sha256', 'report'}
                or not isinstance(wrapper.get('report'), dict)):
            raise ValueError('R2_VALIDATION_REPORT_HASH_BINDING_MISMATCH')
        parsed[name] = wrapper['report']
    calibration, holdout = parsed['calibration'], parsed['holdout']
    _ref(calibration.get('row_ids'))
    holdout_count = _ref(holdout.get('row_ids'))
    if (calibration['row_ids']['count'] < requirements['min_calibration_rows'] or holdout_count != counts['holdout']
            or timestamp(calibration['fit_cutoff']) != timestamp(payload['fit_cutoff'])
            or timestamp(calibration['last_label_available_at']) > timestamp(payload['fit_cutoff'])
            or timestamp(holdout['fit_cutoff']) != timestamp(payload['fit_cutoff'])
            or not timestamp(payload['fit_cutoff']) < timestamp(holdout['validation_start'])
                 <= timestamp(holdout['validation_end'])
            or not timestamp(payload['fit_cutoff']) < timestamp(holdout['last_label_available_at'])
            or timestamp(holdout['last_label_available_at']) != timestamp(payload['validation_available_at'])
            or holdout['heads_sha256'] != identity(payload['heads'])):
        raise ValueError('R2_VALIDATION_TIME_OR_COUNT_MISMATCH')
    expected_cal = dict(calibration['fitted_parameters'], projection_validated=True,
        validation_report_id=ids['calibration'], source_validation_report_sha256=raw_ids['calibration'])
    if payload['calibration'] != expected_cal:
        raise ValueError('R2_NUMERICAL_CALIBRATION_BINDING_MISMATCH')
    projection_checks = parsed['projection'].get('checks')
    if (not isinstance(projection_checks, dict)
            or set(projection_checks) != {'ordered_quantiles', 'known_low_upper_bound', 'joint_le_rebound', 'timing_sum_one'}
            or any(type(value) is not bool for value in projection_checks.values())
            or type(parsed['projection'].get('row_count')) is not int):
        raise ValueError('R2_VALIDATION_PROJECTION_CHECKS_REQUIRED')
    criteria, metrics = holdout['acceptance'], holdout['metrics']
    if (holdout.get('status') != 'COMPLETED' or holdout['pinball_loss'] > criteria['max_pinball_loss']
            or metrics['timing_nll'] > criteria['max_timing_nll']
            or any(value > criteria['max_brier'] for value in metrics['brier'].values())
            or parsed['fractal_null'].get('status') != 'PASS'
            or parsed['projection'].get('status') != 'PASS' or not all(parsed['projection']['checks'].values())
            or parsed['projection'].get('row_count') != counts['holdout']
            or parsed['ablation'].get('status') != 'COMPLETED' or parsed['ablation'].get('same_row_ids') is not True
            or parsed['ablation']['full']['row_ids'] != holdout['row_ids']
            or parsed['ablation']['baseline']['row_ids'] != holdout['row_ids']):
        raise ValueError('R2_VALIDATION_ACCEPTANCE_OR_GATE_FAILED')
    if (payload['calibration']['validation_report_id'] != ids['calibration']
            or payload['calibration'].get('source_validation_report_sha256') != raw_ids['calibration']
            or payload.get('fractal_null_validation_report_id') != ids['fractal_null']
            or manifest['weight_fit'].get('status') != 'CONVERGED' or manifest['weight_fit'].get('weights') != payload['weights']):
        raise ValueError('R2_NUMERICAL_ARTIFACT_BINDING_MISMATCH')


def verify_archived_compaction(candidate, result, config, dataset_record):
    """Exact reconstruction from retained native raw provenance and reports."""
    provenance = result.get('training_provenance')
    if not isinstance(provenance, dict) or provenance.get('schema') != ARCHIVE_SCHEMA:
        raise ValueError('R2_ARCHIVED_TRAINING_PROVENANCE_REQUIRED')
    raw_ids = provenance['archive_validation_report_ids']
    if (set(raw_ids) != REPORT_NAMES or any(value not in result['reports'] for value in raw_ids.values())
            or provenance['support_requirements'] != {key: config[key] for key in SUPPORT_FIELDS}):
        raise ValueError('R2_ARCHIVED_REPORT_OR_SUPPORT_BINDING_MISMATCH')
    _validate_raw_training(result, provenance, config)
    raw_manifest = dict(dataset_id=result['dataset_id'], config_id=result['config_id'],
        dataset_kind=dataset_record['payload']['dataset_kind'], source_authenticity='NOT_INDEPENDENTLY_CERTIFIED',
        upstream_predictions='STRICTLY_PAST_OOF', folds=[fold for fold in result['folds'] if fold['status'] == 'FITTED'],
        oof_prediction_ids=[prediction['prediction_id'] for prediction in result['oof_predictions']],
        weight_prediction_ids=provenance['weight_prediction_ids'], weight_fit=provenance['weight_fit'],
        prior=list(PRIOR), counts=result['counts'], transformations='NONE_RAW_FEATURE_COORDINATES',
        factor_weight_scope='FILLED_REALIZED_NET_PNL_CONDITIONAL_SAMPLES')
    body = dataset_record['payload']
    if body.get('schema') != 'les-v7-sharded-training-dataset/v1':
        raise ValueError('R2_SHARDED_DATASET_REQUIRED')
    raw_manifest.update(dataset_schema=body['schema'], freeze_id=body['freeze_id'], scope_id=body['scope_id'],
        full_universe_sha256=identity(body['universe']), child_dataset_ids=[child['dataset_id'] for child in body['children']],
        diagnostics_policy=body['diagnostics_policy'])
    expected_manifest = _compact_arrays(raw_manifest)
    expected_manifest['support_requirements'] = deepcopy(provenance['support_requirements'])
    wrappers, ids = {}, {}
    for name, raw_id in raw_ids.items():
        wrapper = compact_report(raw_id, result['reports'][raw_id])
        report_id = identity(wrapper)
        wrappers[report_id], ids[name] = wrapper, report_id
    if (candidate['training_manifest'] != expected_manifest or candidate['archive_validation_report_ids'] != raw_ids
            or candidate['validation_reports'] != wrappers or candidate['validation_report_ids'] != ids):
        raise ValueError('R2_EXACT_ARCHIVE_COMPACTION_MISMATCH')
    raw_cal = result['reports'][raw_ids['calibration']]
    expected_cal = dict(raw_cal['fitted_parameters'], projection_validated=True,
        validation_report_id=ids['calibration'], source_validation_report_sha256=raw_ids['calibration'])
    if candidate['calibration'] != expected_cal:
        raise ValueError('R2_ARCHIVED_CALIBRATION_BINDING_MISMATCH')
    validate_runtime_provenance(candidate)


def _validate_raw_training(result, provenance, config):
    """Prove past-only IDs against retained native objects, not digest claims."""
    fitted = {fold['fold_id']: fold for fold in result['folds'] if fold['status'] == 'FITTED'}
    declared = {fold['fold_id']: fold for fold in config['folds']}
    if len(fitted) > len(declared) or not fitted.keys() <= declared.keys():
        raise ValueError('R2_ARCHIVED_OOF_CONFIG_MISMATCH')
    for name, fold in fitted.items():
        expected = declared[name]
        if (any(timestamp(fold[key]) != timestamp(expected[source]) for key, source in (
                ('fit_cutoff', 'fit_cutoff'), ('prediction_at', 'prediction_start'), ('prediction_end', 'prediction_end')))
                or len(fold['training_row_ids']) != len(set(fold['training_row_ids']))
                or len(fold['prediction_row_ids']) != len(set(fold['prediction_row_ids']))):
            raise ValueError('R2_ARCHIVED_OOF_CONFIG_MISMATCH')
        ref = fold.get('calibration_report_id')
        if ref is not None and ref not in result['reports']:
            raise ValueError('R2_ARCHIVED_FOLD_CALIBRATION_REQUIRED')
    predictions = result['oof_predictions']
    if len(predictions) > 100000:
        raise ValueError('R2_ARCHIVED_OOF_BUDGET_EXCEEDED')
    prediction_rows = {name: set(fold['prediction_row_ids']) for name, fold in fitted.items()}
    seen, mature, valid = set(), [], {}
    fit = timestamp(config['fit_cutoff'])
    for prediction in predictions:
        pid, rid = prediction['prediction_id'], prediction['row_id']
        fold = fitted.get(prediction['fold_id'])
        if (pid in seen or fold is None or rid not in prediction_rows.get(prediction['fold_id'], set())
                or pid != identity({key: value for key, value in prediction.items()
                                     if key not in {'labels', 'label_available_at', 'label_id', 'prediction_id'}})
                or prediction['label_id'] != identity(dict(row_id=rid, labels=prediction['labels'],
                                                            available_at=prediction['label_available_at']))
                or prediction['heads_sha256'] != fold['heads_sha256']
                or timestamp(prediction['head_fit_cutoff']) != timestamp(fold['fit_cutoff'])
                or timestamp(prediction['head_last_label_available_at']) != timestamp(fold['last_training_label_available_at'])
                or not timestamp(prediction['head_last_label_available_at']) <= timestamp(prediction['head_fit_cutoff'])
                     < timestamp(prediction['cutoff']) < timestamp(prediction['label_available_at'])
                or not timestamp(fold['prediction_at']) <= timestamp(prediction['cutoff']) <= timestamp(fold['prediction_end'])):
            raise ValueError('R2_ARCHIVED_PAST_OOF_BINDING_MISMATCH')
        seen.add(pid)
        if timestamp(prediction['label_available_at']) <= fit:
            mature.append(prediction)
        if prediction['forecast']['status'] == 'VALID':
            report = result['reports'].get(prediction.get('calibration_report_id'))
            if (report is None or prediction.get('calibration_report_id') != fold.get('calibration_report_id')
                    or timestamp(prediction['calibration_fit_cutoff']) != timestamp(report['fit_cutoff'])
                    or timestamp(prediction['calibration_last_label_available_at']) != timestamp(report['last_label_available_at'])
                    or not timestamp(prediction['calibration_last_label_available_at'])
                         <= timestamp(prediction['calibration_fit_cutoff']) < timestamp(prediction['cutoff'])):
                raise ValueError('R2_ARCHIVED_PAST_CALIBRATION_BINDING_MISMATCH')
            if timestamp(prediction['label_available_at']) <= fit:
                valid[pid] = prediction
    weight_ids = provenance['weight_prediction_ids']
    if len(weight_ids) != len(set(weight_ids)) or any(pid not in valid for pid in weight_ids):
        raise ValueError('R2_ARCHIVED_WEIGHT_OOF_BINDING_MISMATCH')
    # Selection order is the native OOF order; no reordering or ID-set hashes.
    weight_set = set(weight_ids)
    if weight_ids != [prediction['prediction_id'] for prediction in mature if prediction['prediction_id'] in weight_set]:
        raise ValueError('R2_ARCHIVED_WEIGHT_OOF_ORDER_MISMATCH')
    for fold in fitted.values():
        report_id = fold.get('calibration_report_id')
        if report_id is None:
            continue
        past = [prediction for prediction in predictions
                if timestamp(prediction['cutoff']) < timestamp(fold['prediction_at'])
                and timestamp(prediction['label_available_at']) <= timestamp(fold['fit_cutoff'])]
        report = result['reports'][report_id]
        if (report['row_ids'] != [prediction['row_id'] for prediction in past]
                or report['data_sha256'] != identity(past)
                or timestamp(report['fit_cutoff']) != timestamp(fold['fit_cutoff'])):
            raise ValueError('R2_ARCHIVED_FOLD_CALIBRATION_BINDING_MISMATCH')
    raw_ids = provenance['archive_validation_report_ids']
    calibration = result['reports'][raw_ids['calibration']]
    holdout = result['reports'][raw_ids['holdout']]
    if (calibration['row_ids'] != [prediction['row_id'] for prediction in mature]
            or calibration['data_sha256'] != identity(mature)
            or holdout['acceptance'] != config['acceptance']
            or any(timestamp(holdout[key]) != timestamp(config[key]) for key in ('fit_cutoff', 'validation_start', 'validation_end'))):
        raise ValueError('R2_ARCHIVED_VALIDATION_CONFIG_MISMATCH')
