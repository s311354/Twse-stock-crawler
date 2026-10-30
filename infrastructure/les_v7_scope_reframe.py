"""Repartition retained official evidence for a longer research event window.

No HTTP, new prices, labels, fitting, or source-policy substitution. The frozen
universe, calendar, corporate actions and unadjusted bars remain exact. Only
shard boundaries change to obey the existing per-build event budget.
"""
from copy import deepcopy
from datetime import date
from pathlib import Path

from application.les_v7_dataset import validate_build_config
from domain.les_v7.contracts import identity
from infrastructure.les_v7_dataset_scope import SCOPE_FIELDS, SCOPE_SCHEMA, _verify_freeze
from infrastructure.les_v7_reference_repair import reference_document_reader
from infrastructure.les_v7_training_prepare import _repartition_shards
from infrastructure.research_archive import create_dataset, utc_now, write_once


def reframe_elec_scope(seed_path, build_config, output_root, *, allow_synthetic=False, progress=None):
    """Return a new immutable scope, verified against unchanged parent originals."""
    validate_build_config(build_config)
    seed_path, root = Path(seed_path).resolve(), Path(output_root).resolve()
    source_root = seed_path.parent.parent.parent
    original = _verify_freeze(seed_path, source_root, verify_sources=True)
    parent = original['body']
    if parent['dataset_kind'] == 'SYNTHETIC_TEST_ONLY' and not allow_synthetic:
        raise ValueError('SYNTHETIC_SCOPE_REFRAME_REQUIRES_EXPLICIT_OPT_IN')
    if not parent['history_start'] <= build_config['event_start'] <= build_config['event_end'] <= parent['history_end']:
        raise ValueError('ELEC_SCOPE_REFRAME_EVENT_WINDOW_OUTSIDE_HISTORY')
    if build_config.get('reference_evidence_policy') != parent.get('reference_evidence_policy'):
        raise ValueError('ELEC_SCOPE_REFRAME_REFERENCE_POLICY_MISMATCH')
    days = (date.fromisoformat(build_config['event_end']) - date.fromisoformat(build_config['event_start'])).days + 1
    batch_size = min(parent['batch_size'], build_config['max_events'] // days)
    if batch_size < 1:
        raise ValueError('ELEC_SCOPE_REFRAME_EVENT_BUDGET_INSUFFICIENT')
    scope = {key: deepcopy(parent[key]) for key in SCOPE_FIELDS}
    for key in ('market_supplement_sha256', 'reference_evidence_policy', 'corporate_actions_packet_sha256'):
        if key in parent:
            scope[key] = deepcopy(parent[key])
    scope['batch_size'] = batch_size
    scope_id, chunks = identity(scope), []
    for index, symbols, payload, parents, stores in _repartition_shards(parent, source_root, batch_size):
        if progress:
            progress(f'核對原始資料並重新分片 {index + 1} / {(len(parent["universe"]) + batch_size - 1) // batch_size}')
        payload['scope_binding'] = dict(scope_id=scope_id, chunk_index=index,
            complete_scope_size=len(parent['universe']),
            calendar_source_evidence_id=parent['calendar_evidence_id'],
            original_universe_day=parent['universe_record']['source_day'])
        evidence = create_dataset(payload, root / 'evidence', parent=Path(parents[0]['evidence_path']),
            purpose='ELEC_SAME_HISTORY_EXACT_UNION_REFRAME',
            document_reader=reference_document_reader(*stores))
        chunks.append(dict(index=index, symbols=symbols, evidence_id=evidence.parent.name))
        for store in stores:
            store.check()
        del payload
    body = dict(schema=SCOPE_SCHEMA, **scope, scope_id=scope_id, chunks=chunks,
        historical_information_set_certified=False, original_transport_bytes_retained=False,
        original_observation_receipts_available=False, source_authenticity='NOT_INDEPENDENTLY_CERTIFIED',
        reframe_parent=dict(freeze_id=original['freeze_id'], path=str(seed_path),
            build_config_id=identity(build_config), policy='SAME_HISTORY_EXACT_SYMBOL_UNION_ONLY'))
    identifier = identity(body)
    path = root / 'freezes' / identifier / 'manifest.json'
    candidate = path.parent / '.verification.json'
    if not candidate.exists():
        write_once(candidate, dict(freeze_id=identifier, body=body, created_at=utc_now()))
    verified = _verify_freeze(candidate, root, verify_sources=True)
    if verified['body'] != body:
        raise ValueError('ELEC_SCOPE_REFRAME_IMMUTABLE_CANDIDATE_CONFLICT')
    if _verify_freeze(seed_path, source_root, verify_sources=True) != original:
        raise ValueError('ELEC_SCOPE_REFRAME_PARENT_CHANGED')
    # Publication follows complete new-source verification and parent recheck.
    write_once(path, verified)
    return path
