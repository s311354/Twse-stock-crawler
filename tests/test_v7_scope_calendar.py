"""Small synthetic archive/calendar contracts; no official certification or fit."""
from copy import deepcopy
import json

import pytest

from application.replay_contracts import canonical_json, digest
from infrastructure import les_v7_dataset_scope as scope_module
from infrastructure.les_v7_scope_calendar import inspect_elec_scope_calendar
from infrastructure.les_v7_training_source import FINAL_REFERENCE
from tests.test_v7_calendar_final import packet
from tests.test_v7_dataset_scope import scope_inputs, freeze


@pytest.fixture
def scoped(tmp_path):
    inputs = scope_inputs(tmp_path, count=3)
    parent = json.loads(inputs['source'].read_text())['payload']
    calendar, _ = packet(tmp_path)
    parent.update(calendar=calendar, reference_evidence_policy=FINAL_REFERENCE,
                  corporate_actions_final=None)
    inputs['source'].write_text(canonical_json(dict(evidence_id=digest(parent), payload=parent)))
    config = inputs['config']
    config['reference_evidence_policy'] = FINAL_REFERENCE
    for policy in (config['path_policy'], config['seed']['reference_policy']['cost_path_policy']):
        policy['coordinate_policy'] = 'OFFICIAL_FINAL_ACTIONS_RECONSTRUCTION_R1'
    manifest = freeze(inputs, batch_size=2)
    return inputs, manifest


def project(scoped, **changes):
    kwargs = dict(evaluation_cutoff='2026-08-06T13:30:00+08:00', start='2026-07-30',
                  end='2026-08-05', allow_synthetic=True)
    kwargs.update(changes)
    return inspect_elec_scope_calendar(scoped[1], **kwargs)


def first_packet(scoped):
    inputs, manifest = scoped
    body = json.loads(manifest.read_text())['body']
    chunk = body['chunks'][0]
    path = inputs['root'] / 'evidence' / chunk['evidence_id'] / 'evidence.json'
    return body, path


def test_projection_is_date_cutoff_scope_bound_and_reads_only_one_shard(scoped, monkeypatch):
    original = scope_module.load_evidence
    reads = []
    def observed(path, **kwargs):
        reads.append(path)
        return original(path, **kwargs)
    monkeypatch.setattr(scope_module, 'load_evidence', observed)
    result = project(scoped)
    record = json.loads(scoped[1].read_text())
    assert result['freeze_id'] == record['freeze_id']
    assert result['scope_id'] == record['body']['scope_id']
    assert result['start'] == '2026-07-30' and result['end'] == '2026-08-05'
    assert result['evaluation_cutoff'] == '2026-08-06T05:30:00+00:00'
    assert result['sessions'] == ['2026-07-30', '2026-07-31', '2026-08-03', '2026-08-04', '2026-08-05']
    assert result['days']['2026-08-01'] == result['days']['2026-08-02'] == 'CLOSED'
    assert result['unknown_dates'] == [] and len(set(reads)) == 1
    assert result['source_evidence_id'] in result['evidence_ids']
    assert result['historical_information_set_certified'] is False
    assert result['verification_scope'] == 'CALENDAR_AND_ONE_BOUND_SHARD_NOT_FULL_SCOPE_SOURCES'
    assert not any(key in result for key in ('bars', 'histories', 'symbols', 'calendar'))


def test_synthetic_opt_in_range_and_cutoff_are_explicit(scoped):
    with pytest.raises(ValueError, match='SYNTHETIC_SCOPE_CALENDAR_REQUIRES'):
        project(scoped, allow_synthetic=False)
    with pytest.raises(ValueError, match='OUTSIDE_FIXED_HISTORY'):
        project(scoped, start='2026-07-29')
    with pytest.raises(ValueError, match='CALENDAR_RANGE_INVALID'):
        project(scoped, start='20260730')
    with pytest.raises(ValueError, match='RANGE_AFTER_CUTOFF'):
        project(scoped, evaluation_cutoff='2026-08-04T13:30:00+08:00')
    with pytest.raises(ValueError):
        project(scoped, evaluation_cutoff='2026-08-06T13:30:00')


def test_changed_scope_calendar_packet_or_binding_is_rejected(scoped):
    _, path = first_packet(scoped)
    # A new correctly hashed packet is still not the source frozen in this scope.
    changed = json.loads(path.read_text())
    changed['payload']['calendar']['days'] = {'2026-08-03': 'CLOSED'}
    changed['evidence_id'] = digest(changed['payload'])
    path.write_text(canonical_json(changed))
    with pytest.raises(ValueError, match='SHARD_BINDING_MISMATCH'):
        project(scoped)


def test_calendar_raw_document_mutation_cannot_be_hidden_by_payload_hash(scoped):
    _, path = first_packet(scoped)
    packet_value = json.loads(path.read_text())['payload']
    ref = packet_value['calendar']['documents']['2026']['document']['body_ref']
    from infrastructure.research_documents import EvidenceDocuments
    docs = EvidenceDocuments.for_evidence(path, digest(packet_value))
    docs.objects.path(ref['sha256']).write_bytes(b'changed raw annual source')
    with pytest.raises(ValueError):
        project(scoped)


def test_unknown_native_calendar_days_are_preserved_not_inferred(scoped, monkeypatch):
    # The native selector deliberately reports UNKNOWN on missing annual/query
    # proof. Verify delegation without manufacturing a valid replacement scope.
    import infrastructure.les_v7_scope_calendar as module
    original = module.select_final_calendar
    def missing_query(calendar, *args, **kwargs):
        copy = deepcopy(calendar)
        copy['closure_queries'].clear()
        return original(copy, *args, **kwargs)
    monkeypatch.setattr(module, 'select_final_calendar', missing_query)
    result = project(scoped)
    assert result['sessions'] == []
    assert set(result['days'].values()) == {'UNKNOWN'}
    assert result['unknown_dates'] == sorted(result['days'])
