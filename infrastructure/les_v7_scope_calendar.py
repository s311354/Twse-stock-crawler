"""Bounded calendar-only verification of a frozen ELEC training scope.

No market collection, bar normalization, fitting or full-scope verification.
One hash-bound source shard proves the common calendar's scope binding; the
existing final-calendar verifier remains the only calendar interpretation.
"""
from datetime import date
from pathlib import Path

from domain.les_v7.contracts import timestamp
from infrastructure.les_v7_dataset_scope import _load_bound_scope_shard, _verify_freeze
from infrastructure.les_v7_calendar_final import select_final_calendar
from infrastructure.les_v7_training_source import FINAL_REFERENCE, MAX_DAYS
from infrastructure.research_documents import EvidenceDocuments


def inspect_elec_scope_calendar(freeze_path, *, evaluation_cutoff, start, end,
                                allow_synthetic=False):
    """Return a small native-verified calendar projection, not source readiness.

    ``freeze_id`` binds the exact fixed snapshot. ``start``, ``end`` and the
    aware cutoff are retained in the result, so callers can compare all scoped
    day states (including CLOSED/UNKNOWN), rather than just matching quotes.
    Only final-reference reconstruction scopes are supported. UNKNOWN remains
    explicit; callers must refuse an incomplete planned session sequence.
    """
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    cutoff = timestamp(evaluation_cutoff)
    if (start != first.isoformat() or end != last.isoformat()
            or first > last or (last - first).days + 1 > MAX_DAYS):
        raise ValueError('ELEC_SCOPE_CALENDAR_RANGE_INVALID')
    path = Path(freeze_path).expanduser().resolve()
    root = path.parent.parent.parent
    record = _verify_freeze(path, root, verify_sources=False)
    body = record['body']
    if body['dataset_kind'] != 'RESEARCH' and allow_synthetic is not True:
        raise ValueError('SYNTHETIC_SCOPE_CALENDAR_REQUIRES_EXPLICIT_OPT_IN')
    if body.get('reference_evidence_policy') != FINAL_REFERENCE:
        raise ValueError('ELEC_SCOPE_CALENDAR_FINAL_REFERENCE_REQUIRED')
    if not body['history_start'] <= start <= end <= body['history_end']:
        raise ValueError('ELEC_SCOPE_CALENDAR_OUTSIDE_FIXED_HISTORY')
    source_path, evidence_id, source = _load_bound_scope_shard(body, root, body['chunks'][0])
    documents = EvidenceDocuments.for_evidence(source_path, evidence_id)
    documents.verify(source['calendar'])
    selected, diagnostics = select_final_calendar(source['calendar'], cutoff,
        documents=documents, start=start, end=end, evidence_id=evidence_id)
    result = dict(schema='les-v7-elec-scope-calendar-projection/v1',
        freeze_id=record['freeze_id'], scope_id=body['scope_id'],
        start=start, end=end, evaluation_cutoff=cutoff.isoformat(),
        calendar_sha256=body['calendar_sha256'], source_evidence_id=evidence_id,
        days=selected['days'], sessions=selected['sessions'],
        evidence_ids=sorted(set([record['freeze_id'], *selected['evidence_ids']])),
        unknown_dates=diagnostics['unknown_dates'],
        availability_policy=selected['availability_policy'],
        verification_scope='CALENDAR_AND_ONE_BOUND_SHARD_NOT_FULL_SCOPE_SOURCES',
        historical_information_set_certified=False, research_only=True, approved_for_live=False)
    documents.check()
    del source
    # Reject a concurrent replacement of the scope or its representative shard.
    current = _verify_freeze(path, root, verify_sources=False)
    if current != record:
        raise ValueError('ELEC_SCOPE_CALENDAR_INPUT_CHANGED')
    _, current_id, current_source = _load_bound_scope_shard(body, root, body['chunks'][0])
    if current_id != evidence_id:
        raise ValueError('ELEC_SCOPE_CALENDAR_INPUT_CHANGED')
    del current_source
    return result
