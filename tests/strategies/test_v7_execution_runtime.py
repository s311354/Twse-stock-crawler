"""Synthetic runtime/provenance checks; these do not assert market skill."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from domain.les_v7.contracts import GateReceipt, canonical, identity
from domain.les_v7.decision import make_plan, vetoes
from domain.les_v7.engine import evaluate
from domain.les_v7.execution_features import FEATURE_BOUNDS, FEATURE_NAMES, profile, compute_execution_features
from domain.les_v7.features import compute_features
from domain.les_v7.localization import FrozenBundle, model_input_profile
from domain.les_v7.native_evidence import REQUIRED_GATES, produce_native_receipts, validate_native_receipts
from domain.les_v7.report import validate_report
from tests.strategies.test_low_entry_score_v7 import bars, bundle, CUTOFF, filings, POLICY
from tests.test_v7_native_evidence import inputs


def extended_bundle():
    record = bundle(); model = record['payload']; selected = list(FEATURE_NAMES[:13])
    model.update(execution_feature_profile=profile(),
        fill_only_feature_extension=dict(base_feature_names=['close'], profile_id=identity(profile()), selected_features=selected),
        feature_names=['close'] + selected, feature_bounds=[[1., 200.]] + [list(x) for x in FEATURE_BOUNDS[:13]],
        imputation=[None] * 14, supported_masks=[[True] * 14])
    for name, value in model['heads'].items():
        array = np.asarray(value)
        model['heads'][name] = np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, 13)]).tolist()
    model['heads']['fill_only_base_count'] = 1
    model['heads']['fill'][-1] = 2.
    record['sha256'] = identity(model)
    return record


def evaluate_model(record, data=None):
    return evaluate(bars(600) if data is None else data,
        dict(symbol='SYNTHETIC_ONLY', market='TWSE', as_of=CUTOFF,
            filings=filings(), quality_policy=deepcopy(POLICY), model_bundle=record, receipts={}))


def test_runtime_keeps_original_features_receipts_and_nonfill_outputs_separate():
    base = evaluate_model(bundle()); extra = evaluate_model(extended_bundle())
    validate_report(extra)
    assert extra['diagnostics']['features'] == base['diagnostics']['features']
    assert extra['diagnostics']['feature_audit'] == base['diagnostics']['feature_audit']
    assert extra['diagnostics']['gate_bindings'] == base['diagnostics']['gate_bindings']
    assert extra['diagnostics']['reference_plan'] == base['diagnostics']['reference_plan']
    assert extra['diagnostics']['execution_feature_audit']['status'] == 'COMPUTED'
    first, second = base['diagnostics']['localization'], extra['diagnostics']['localization']
    for key in ('quantiles', 'timing', 'p_rebound', 'p_entry'):
        assert second[key] == first[key]
    assert second['p_fill'] != first['p_fill']
    assert extra['diagnostics']['model_input_profile']['schema'] == 'les-v7-model-input-profile/v2'
    future = bars(600).iloc[[-1]].copy(); future['Date'] = '2026-10-13'
    future[['Open', 'High', 'Low', 'Close']] = [10000., 20000., .01, 15000.]
    assert evaluate_model(extended_bundle(), pd.concat([bars(600), future], ignore_index=True)) == extra


@pytest.mark.parametrize('change', ['protected', 'count', 'profile', 'missing'])
def test_extended_loader_rejects_misdeclared_protected_coordinates(change):
    record = extended_bundle(); model = record['payload']
    if change == 'protected': model['heads']['price'][0][-1] = .1
    if change == 'count': model['heads']['fill_only_base_count'] = 2
    if change == 'profile': model['execution_feature_profile']['required_close_count'] = 120
    if change == 'missing': del model['fill_only_feature_extension']
    record['sha256'] = identity(model)
    with pytest.raises(ValueError): FrozenBundle.load(record)


def native_report():
    item = inputs(); data = bars(600); data['PE'] = 12.
    f, audit = compute_features(data)
    rows = data.to_dict('records'); packet = item['source_packet']
    packet['market_rows'] = rows; packet['market_evidence']['bars_sha256'] = identity(rows)
    base = dict(item['bindings']['DATA_INTEGRITY'], bars_hash=identity(rows))
    plan = asdict(make_plan('REVERSAL', f, item['reference_policy'], identity(base)))
    bindings = {gate: base for gate in REQUIRED_GATES}
    bindings['VALUATION'] = dict(base, price=f['close'], quality_result_id=identity(item['quality']))
    bindings['STRESS_RISK'] = dict(base, plan=plan)
    item.update(bars=data, features=f, plan=plan, risks=vetoes(f), bindings=bindings)
    produced = produce_native_receipts(**item)
    from domain.les_v7.contracts import ReferencePlan
    features, execution_audit = compute_execution_features(data, cutoff=CUTOFF,
        base_features=f, plan=ReferencePlan(**dict(plan, band=tuple(plan['band']))))
    return dict(symbol=item['symbol'], envelope=dict(event=dict(cutoff_at=item['cutoff'])),
        diagnostics=dict(features=f, feature_audit=audit, reference_plan=plan, quality=item['quality'],
            gate_bindings=bindings, gate_receipts=[asdict(GateReceipt.bind(produced[gate], gate_id=gate,
                inputs=bindings[gate], cutoff=item['cutoff'])) for gate in REQUIRED_GATES],
            model_input_profile=model_input_profile(extended_bundle()['payload'], extended_bundle()['sha256']),
            execution_features={name: features[name] for name in FEATURE_NAMES[:13]},
            execution_feature_audit=execution_audit,
            native_evidence=dict(schema='les-v7-native-evidence/v1', source_packet=packet,
                source_id=identity(packet), produced_receipts=produced, reference_policy=item['reference_policy'])))


def test_native_verification_recomputes_extension_and_rejects_rehashed_numeric_or_audit_tampering():
    original = native_report(); validate_native_receipts(original)
    for field in ('numeric', 'audit'):
        changed = deepcopy(original)
        if field == 'numeric': changed['diagnostics']['execution_features']['execution_gap_mean5'] += .001
        else: changed['diagnostics']['execution_feature_audit']['available_close_count'] += 1
        with pytest.raises(ValueError, match='EXECUTION_FEATURE_RECOMPUTATION'):
            validate_native_receipts(changed)
    assert canonical(original) == canonical(native_report())
