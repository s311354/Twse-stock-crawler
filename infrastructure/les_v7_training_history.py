"""Immutable historical extensions from already captured official FINAL data.

This port reuses the native calendar, market, action and ELEC scope verifiers.
It never fits, selects hyperparameters, promotes a model or rewrites the seed.
H20 maturity means 20 subsequent verified official OPEN sessions; retained final
bytes do not certify that the historical information set was available then.
"""
from copy import deepcopy
from datetime import date, time, timedelta
import fcntl
from pathlib import Path

from application.les_v7_dataset import validate_build_config
from application.replay_contracts import digest
from domain.les_v7.contracts import timestamp
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.les_v7_calendar_final import load_final_calendar, select_final_calendar
from infrastructure.les_v7_dataset_scope import _verify_freeze
from infrastructure.les_v7_market_final import load_final_market
from infrastructure.les_v7_training_prepare import _Budget, _extend_scope, _now, TAIPEI
from infrastructure.les_v7_training_source import FINAL_REFERENCE, MAX_DAYS
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments

SCHEMA = 'les-v7-elec-historical-extension/v1'


def _coverage(seed_scope, calendar_path, market_path, cutoff, *, verify_seed_sources=True):
    clock = timestamp(cutoff)
    if clock > _now():
        raise ValueError('HISTORICAL_EXTENSION_REAL_CUTOFF_REQUIRED')
    local = clock.astimezone(TAIPEI)
    through = local.date() if local.time().replace(tzinfo=None) >= time(13, 30) else local.date() - timedelta(days=1)
    seed_path = Path(seed_scope).expanduser().resolve()
    seed_root = seed_path.parent.parent.parent
    seed = _verify_freeze(seed_path, seed_root, verify_sources=verify_seed_sources)
    parent = seed['body']
    if parent['dataset_kind'] != 'RESEARCH' or parent.get('reference_evidence_policy') != FINAL_REFERENCE:
        raise ValueError('HISTORICAL_EXTENSION_REAL_FINAL_REFERENCE_SCOPE_REQUIRED')
    calendar_path = Path(calendar_path).expanduser().resolve()
    market_path = Path(market_path).expanduser().resolve()
    calendar, calendar_docs = load_final_calendar(calendar_path)
    market, market_docs = load_final_market(market_path)
    last = date.fromisoformat(market['end'])
    first_new = date.fromisoformat(parent['history_end']) + timedelta(days=1)
    if (market['start'] != first_new.isoformat() or first_new > last or last > through
            or (last - date.fromisoformat(parent['history_start'])).days + 1 > MAX_DAYS
            or market.get('calendar_sha256') != digest(calendar)):
        raise ValueError('HISTORICAL_EXTENSION_MARKET_RANGE_OR_CALENDAR_BINDING_INVALID')
    selected, diagnostics = select_final_calendar(calendar, clock, documents=calendar_docs,
        start=parent['history_start'], end=market['end'], evidence_id=digest(calendar))
    if diagnostics['unknown_dates'] or any(value == 'UNKNOWN' for value in selected['days'].values()):
        raise ValueError('HISTORICAL_EXTENSION_CALENDAR_UNKNOWN')
    required = {day for day in selected['sessions'] if first_new.isoformat() <= day <= market['end']}
    captured = [row['date'] for row in market['documents']]
    if (market['status'] != 'COMPLETE' or market.get('gaps') or len(captured) != len(set(captured))
            or set(captured) != required):
        raise ValueError('HISTORICAL_EXTENSION_COMPLETE_MARKET_REQUIRED')
    sessions = selected['sessions']
    if not sessions:
        raise ValueError('HISTORICAL_EXTENSION_VERIFIED_OPEN_SESSION_REQUIRED')
    latest = sessions[-1]
    evaluation = latest + 'T13:30:00+08:00'
    public = dict(schema=SCHEMA, seedScopeFile=str(seed_path), seedDatasetId=seed['freeze_id'],
        calendarFile=str(calendar_path), marketFile=str(market_path), verifiedThrough=latest,
        evaluationCutoff=evaluation, maturedEventThrough=sessions[-21] if len(sessions) > 20 else None,
        sessions=sessions, stockCount=len(parent['universe']), seedSourcesVerified=verify_seed_sources,
        historicalInformationSetCertified=False, researchOnly=True, approvedForLive=False,
        fittingPerformed=False, executionAction='NO_ACTION')
    calendar_docs.check()
    market_docs.check()
    return public, seed, market, market_docs


def inspect_historical_coverage(seed_scope, calendar_path, market_path, *, cutoff):
    """Verify retained originals and expose dates for a deterministic planner.

    The seed manifest is verified without loading every historical bar shard.
    New official calendar/market documents are fully byte-verified; immutable
    extension/publication repeats full native seed-source verification.
    No request or mutable-cache lookup occurs here. A caller must first finish
    the bounded native calendar/market preparation, including all UNKNOWN/gaps.
    The returned sessions are official calendar selections, not weekday guesses.
    """
    return _coverage(seed_scope, calendar_path, market_path, cutoff, verify_seed_sources=False)[0]


def _market_prefix(market, documents, latest, root):
    """Drop only a verified CLOSED tail, preserving every original byte."""
    if market['end'] == latest:
        return None
    body = deepcopy(market)
    body.update(end=latest, documents=[row for row in body['documents'] if row['date'] <= latest],
        parent_manifest_id=digest(market), parent_calendar_sha256=market['calendar_sha256'])
    store = EvidenceDocuments(root / 'objects')
    body = store.externalize(body, document_reader=documents.read)
    identifier = digest(body)
    path = root / 'manifests' / identifier / 'manifest.json'
    if not path.exists():
        write_once(path, dict(manifest_id=identifier, body=body, created_at=utc_now()))
    load_final_market(path)
    documents.check()
    return path


def extend_historical_scope(seed_scope, calendar_path, market_path, *, build, output_root,
        cutoff, max_requests=12, client=None, progress=None):
    """Append verified current coverage to a new immutable complete ELEC scope.

    ``build`` must use the inspected latest OPEN close as evaluation cutoff and
    mature events only. Native build/source checks remain mandatory. The only
    possible HTTP is native official action-coordinate collection, under one
    secure, rate-limited global request budget. This function never trains.
    """
    validate_build_config(build)
    if type(max_requests) is not int or not 1 <= max_requests <= 256:
        raise ValueError('HISTORICAL_EXTENSION_REQUEST_BUDGET_REQUIRED')
    coverage, seed, market, market_docs = _coverage(seed_scope, calendar_path, market_path, cutoff)
    parent = seed['body']
    if (timestamp(build['evaluation_cutoff']) != timestamp(coverage['evaluationCutoff'])
            or not parent['history_start'] <= build['event_start'] <= build['event_end']
            or coverage['maturedEventThrough'] is None or build['event_end'] > coverage['maturedEventThrough']):
        raise ValueError('HISTORICAL_EXTENSION_MATURE_H20_BUILD_RANGE_REQUIRED')
    request = dict(schema=SCHEMA, seedDatasetId=seed['freeze_id'], buildId=digest(build),
        calendarId=digest(load_final_calendar(coverage['calendarFile'])[0]), marketId=digest(market),
        verifiedThrough=coverage['verifiedThrough'])
    root = Path(output_root).expanduser().resolve() / digest(request)
    seed_path = Path(coverage['seedScopeFile'])
    if root.is_relative_to(seed_path.parent.parent.parent):
        raise ValueError('HISTORICAL_EXTENSION_OUTPUT_MUST_PRESERVE_SEED_ROOT')
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.history.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('HISTORICAL_EXTENSION_BUSY') from error
        write_once(root / 'request.json', request)
        budget = None
        if coverage['verifiedThrough'] <= parent['history_end']:
            path, identifier = seed_path, seed['freeze_id']
        else:
            transport = client or TwseHttpClient(TwseHttpConfig(timeout_seconds=20, max_attempts=1,
                min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
            if (transport.config.allow_insecure_ssl_fallback or transport.config.min_request_interval_seconds < 1
                    or transport.config.max_attempts != 1):
                raise ValueError('HISTORICAL_EXTENSION_SECURE_SINGLE_ATTEMPT_TRANSPORT_REQUIRED')
            budget = _Budget(transport, max_requests)
            prefix = _market_prefix(market, market_docs, coverage['verifiedThrough'], root / 'market')
            if progress:
                progress('核對官方行動座標並保留完整固定股票範圍')
            path, identifier = _extend_scope(seed_path, Path(coverage['calendarFile']),
                prefix or Path(coverage['marketFile']), build=build, end=coverage['verifiedThrough'],
                root=root / 'extension', budget=budget)
        verified = _verify_freeze(path, path.parent.parent.parent, verify_sources=True)
        if (verified['freeze_id'] != identifier or verified['body']['universe'] != parent['universe']
                or verified['body']['universe_record'] != parent['universe_record']
                or verified['body']['history_start'] != parent['history_start']
                or verified['body']['history_end'] < coverage['verifiedThrough']):
            raise ValueError('HISTORICAL_EXTENSION_EXACT_UNIVERSE_OR_COVERAGE_CHANGED')
        if _verify_freeze(seed_path, seed_path.parent.parent.parent, verify_sources=True) != seed:
            raise ValueError('HISTORICAL_EXTENSION_SEED_CHANGED')
        receipt = dict(coverage, scopeFile=str(path), datasetId=identifier,
            requestCount=budget.count if budget else 0, reusedSeed=path == seed_path,
            buildId=digest(build), requestId=digest(request))
        write_once(root / 'receipts' / digest(receipt) / 'receipt.json', receipt)
        return receipt
