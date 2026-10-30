"""Bounded, resumable official FINAL research preparation; never fits a model.

Only native calendar selection determines OPEN dates. An immutable seed fixes
the whole ELEC universe. Old scopes and originals are never rewritten. This is
retrospective reconstruction, not certified historical observation or FR2
preregistration. Full source expansion is deferred until all H20 labels mature.
"""
from copy import deepcopy
from datetime import date, datetime, time, timedelta
import fcntl
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from application.replay_contracts import digest
from domain.les_v7.contracts import timestamp
from domain.models import TwseColumns as C
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.les_v7_calendar_final import collect_final_calendars, load_final_calendar, select_final_calendar
from infrastructure.les_v7_dataset_scope import _verify_freeze, _load_bound_scope_shard, SCOPE_FIELDS, SCOPE_SCHEMA
from infrastructure.les_v7_market_final import collect_final_market, load_final_market, final_market_records
from infrastructure.les_v7_training_source import FINAL_REFERENCE, MAX_BYTES, MAX_DAYS
from infrastructure.research_archive import create_dataset, utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.storage.les_v7_training import read_json_bounded

SCHEMA = 'les-v7-elec-training-preparation/v1'
TAIPEI = ZoneInfo('Asia/Taipei')
MAX_STATE_BYTES = 2 * 1024 * 1024


def _now():
    return datetime.now(TAIPEI)


def _atomic(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(content) > MAX_STATE_BYTES:
        raise ValueError('TRAINING_PREPARATION_RECEIPT_SIZE_LIMIT')
    fd, name = tempfile.mkstemp(prefix='.prepare-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _checkpoint(path, root):
    if not path.exists():
        return {}
    state = read_json_bounded(path, MAX_STATE_BYTES)
    reference = Path(state.get('receiptFile', '')).resolve()
    if not reference.is_relative_to(root / 'receipts') or reference.name != 'receipt.json':
        raise ValueError('TRAINING_PREPARATION_CHECKPOINT_RECEIPT_PATH_INVALID')
    body = read_json_bounded(reference, MAX_STATE_BYTES)
    if reference.parent.name != digest(body) or state != dict(body, receiptFile=str(reference)):
        raise ValueError('TRAINING_PREPARATION_CHECKPOINT_RECEIPT_HASH_MISMATCH')
    return state


class _Budget:
    def __init__(self, client, limit):
        self.client, self.config, self.limit = client, client.config, limit
        self.count, self.blocked = 0, False

    def reset_session(self):
        reset = getattr(self.client, 'reset_session', None)
        if callable(reset):
            reset()

    def request_bytes(self, *args, before_attempt=None, **kwargs):
        def reserve():
            if self.blocked:
                raise ValueError('TRAINING_PREPARATION_STOPPED_ACCESS_BLOCKED')
            if self.count >= self.limit:
                raise ValueError('TRAINING_PREPARATION_REQUEST_BUDGET_EXHAUSTED')
            if before_attempt:
                before_attempt()
            self.count += 1
        try:
            return self.client.request_bytes(*args, before_attempt=reserve, **kwargs)
        except Exception as error:
            category = getattr(getattr(error, 'category', None), 'value', None)
            if category == 'access_blocked' or 'ACCESS_BLOCKED' in str(error) or 'AccessBlocked' in type(error).__name__:
                self.blocked = True
            raise


def _seed_calendar(record, source_root, target):
    """One representative bound shard; no full-scope loading while waiting."""
    _, evidence_id, source = _load_bound_scope_shard(record['body'], source_root, record['body']['chunks'][0])
    documents = EvidenceDocuments.for_evidence(source_root / 'evidence' / evidence_id / 'evidence.json', evidence_id)
    documents.verify(source['calendar'])
    store = EvidenceDocuments(target / 'objects')
    calendar = store.externalize(source['calendar'], document_reader=documents.read)
    identifier = digest(calendar)
    path = target / 'collections' / identifier / 'calendar.json'
    write_once(path, dict(calendar_id=identifier, payload=calendar))
    documents.check()
    return path


def _repartition_shards(body, source_root, batch_size):
    """Global exact-union re-shard, bounded by at most two original shards.

    Only native ELEC market-reference shards are merged. Common source fields
    must match exactly; conflicting evidence is rejected, never preferred.
    Raw bars stay unadjusted. Stock rows are selected by their native column;
    identical benchmark records are retained once by exact content identity.
    The complete original evidence IDs and paths remain in the lineage.
    """
    scoped = {'symbols', 'scope_binding', 'bars', 'filings', 'batches', 'security_classifications',
              'training_extension_parents'}
    universe = body['universe']
    for index, offset in enumerate(range(0, len(universe), batch_size)):
        symbols = universe[offset:offset + batch_size]
        wanted = set(symbols)
        merged, common, parents, stores, records = None, None, [], [], {}
        for chunk in body['chunks']:
            selected = wanted.intersection(chunk['symbols'])
            if not selected:
                continue
            path, eid, payload = _load_bound_scope_shard(body, source_root, chunk)
            documents = EvidenceDocuments.for_evidence(path, eid)
            documents.verify(payload)
            fields = {key: value for key, value in payload.items() if key not in scoped}
            if common is not None and fields != common:
                raise ValueError('TRAINING_PREPARATION_RESHARD_COMMON_EVIDENCE_CONFLICT')
            if merged is None:
                merged, common = deepcopy(fields), deepcopy(fields)
                merged.update(filings={}, batches={}, security_classifications={})
            for key in ('filings', 'batches', 'security_classifications'):
                values = payload.get(key, {})
                if not isinstance(values, dict) or any(symbol not in values for symbol in selected):
                    raise ValueError('TRAINING_PREPARATION_RESHARD_SYMBOL_EVIDENCE_MISSING')
                merged[key].update({symbol: deepcopy(values[symbol]) for symbol in selected})
            for original in payload['bars']:
                row = deepcopy(original)
                if row.get('dataset') == 'STOCK_TABLE':
                    row['data'] = [value for value in row['data'] if str(value[C.STOCK_NO]) in selected]
                    if not row['data']:
                        continue
                records[digest(row)] = row
            parents.append(dict(evidence_id=eid, evidence_path=str(path), symbols=sorted(selected)))
            stores.append(documents)
            del payload
        if merged is None or set(merged['security_classifications']) != wanted:
            raise ValueError('TRAINING_PREPARATION_RESHARD_EXACT_UNION_MISSING')
        merged.update(symbols=symbols, bars=[records[key] for key in sorted(records)])
        # Bind all parents, including sources crossing the new chunk boundary.
        merged['training_extension_parents'] = parents
        yield index, symbols, merged, parents, stores


def _extend_scope(seed_path, calendar_path, market_path, *, build, end, root, budget):
    """Append verified new bars to each original shard, retaining old evidence."""
    from application.les_v7_dataset import validate_build_config
    from infrastructure.les_v7_actions_final import collect_actions_final, load_actions_final
    from infrastructure.les_v7_reference_repair import reference_document_reader
    validate_build_config(build)
    seed_root = seed_path.parent.parent.parent
    original = _verify_freeze(seed_path, seed_root, verify_sources=True)
    parent = original['body']
    event_days = (date.fromisoformat(build['event_end']) - date.fromisoformat(build['event_start'])).days + 1
    effective_size = min(parent['batch_size'], build['max_events'] // event_days)
    if effective_size < 1:
        raise ValueError('TRAINING_PREPARATION_EVENT_BUDGET_INSUFFICIENT')
    calendar, calendar_docs = load_final_calendar(calendar_path)
    market, market_docs = load_final_market(market_path)
    if (market['status'] != 'COMPLETE' or market['end'] != end
            or market['start'] != (date.fromisoformat(parent['history_end']) + timedelta(days=1)).isoformat()):
        raise ValueError('TRAINING_PREPARATION_COMPLETE_MARKET_REQUIRED')
    selected, diagnostics = select_final_calendar(calendar, build['evaluation_cutoff'], documents=calendar_docs,
        start=parent['history_start'], end=end, evidence_id=digest(calendar))
    if diagnostics['unknown_dates']:
        raise ValueError('TRAINING_PREPARATION_FULL_HISTORY_CALENDAR_UNKNOWN')
    actions_path = collect_actions_final(parent['history_start'], end, root / 'actions',
        client=budget, max_requests=min(24, budget.limit), resume=True)
    actions = load_actions_final(actions_path)
    action_docs = EvidenceDocuments(actions_path.parent.parent / 'objects')
    first_path, _, _ = _load_bound_scope_shard(parent, seed_root, parent['chunks'][0])
    from infrastructure.les_v7_reference_repair import repair_reference_evidence
    repaired = repair_reference_evidence(first_path, calendar_path=calendar_path, actions_path=actions_path,
        store=root / 'reference')
    from infrastructure.replay_evidence import load_evidence
    calendar_source, _ = load_evidence(repaired, max_bytes=MAX_BYTES)
    calendar['days'] = selected['days']
    scope = {key: deepcopy(parent[key]) for key in SCOPE_FIELDS if key != 'schema'}
    scope.update(calendar_evidence_id=calendar_source, calendar_sha256=digest(calendar), history_end=end,
        batch_size=effective_size,
        market_supplement_sha256=digest(market), reference_evidence_policy=FINAL_REFERENCE,
        corporate_actions_packet_sha256=digest(actions))
    scope_id, chunks = digest(scope), []
    for index, symbols, payload, parents, stores in _repartition_shards(parent, seed_root, effective_size):
        payload['original_calendar_before_training_extension'] = payload['calendar']
        payload['original_actions_before_training_extension'] = payload.get('corporate_actions_final')
        if 'market_supplement_final' in payload:
            payload['original_market_supplement_before_training_extension'] = payload['market_supplement_final']
        payload.update(calendar=deepcopy(calendar), corporate_actions_final=deepcopy(actions),
            market_supplement_final=deepcopy(market), end=end)
        payload['scope_binding'] = dict(scope_id=scope_id, chunk_index=index,
            complete_scope_size=len(parent['universe']), calendar_source_evidence_id=calendar_source,
            original_universe_day=parent['universe_record']['source_day'])
        payload['bars'].extend(final_market_records(market, market_docs, symbols))
        evidence = create_dataset(payload, root / 'scope' / 'evidence', parent=Path(parents[0]['evidence_path']),
            purpose='ELEC_PROSPECTIVE_FINAL_REFERENCE_EXTENSION',
            document_reader=reference_document_reader(*stores, calendar_docs, action_docs, market_docs))
        chunks.append(dict(index=index, symbols=symbols, evidence_id=evidence.parent.name))
        for documents in stores:
            documents.check()
        del payload
    body = dict(schema=SCOPE_SCHEMA, **scope, scope_id=scope_id, chunks=chunks,
        historical_information_set_certified=False, original_transport_bytes_retained=False,
        original_observation_receipts_available=False, source_authenticity='NOT_INDEPENDENTLY_CERTIFIED')
    identifier = digest(body)
    path = root / 'scope' / 'freezes' / identifier / 'manifest.json'
    candidate = path.parent / '.verification.json'
    if not candidate.exists():
        write_once(candidate, dict(freeze_id=identifier, body=body, created_at=utc_now()))
    verified = _verify_freeze(candidate, root / 'scope', verify_sources=True)
    if verified['body'] != body:
        raise ValueError('TRAINING_PREPARATION_IMMUTABLE_SCOPE_CANDIDATE_CONFLICT')
    if _verify_freeze(seed_path, seed_root, verify_sources=False) != original:
        raise ValueError('TRAINING_PREPARATION_SEED_CHANGED')
    # Publish the discoverable scope only AFTER every child source and the
    # unchanged seed have passed. Failed attempts retain audit bytes only.
    write_once(path, verified)
    return path, identifier


def prepare_elec_training_data(seed_scope, build_template, *, plan_not_before, output_root, cutoff,
        validation_sessions=180, label_maturity_sessions=20, max_requests=12, market_cache=None,
        client=None, progress=None):
    """One bounded background pass; repeat later to resume immutable captures.

    ``cutoff`` must be aware and not after the real clock. The last candidate
    date is today only after 13:30 Taipei, otherwise yesterday. A still-future
    prospective anchor is valid and produces zero independent sessions.
    ``progress`` receives one human-readable phase string. ``market_cache`` is
    accepted for orchestration compatibility but never used:
    an extension appends to the pinned original shards instead of replacing
    historical rows from a mutable cache.
    """
    del market_cache
    clock = timestamp(cutoff)
    first = date.fromisoformat(plan_not_before)
    if clock > _now() or type(max_requests) is not int or not 1 <= max_requests <= 256:
        raise ValueError('TRAINING_PREPARATION_REAL_CUTOFF_AND_REQUEST_BUDGET_REQUIRED')
    if (type(validation_sessions) is not int or type(label_maturity_sessions) is not int
            or validation_sessions != 180 or label_maturity_sessions != 20):
        raise ValueError('TRAINING_PREPARATION_NATIVE_180_PLUS_20_REQUIRED')
    local = clock.astimezone(TAIPEI)
    through = local.date() if local.time().replace(tzinfo=None) >= time(13, 30) else local.date() - timedelta(days=1)
    seed_path = Path(seed_scope).expanduser().resolve()
    source_root = seed_path.parent.parent.parent
    seed = _verify_freeze(seed_path, source_root, verify_sources=False)
    body = seed['body']
    if body['dataset_kind'] != 'RESEARCH' or body.get('reference_evidence_policy') != FINAL_REFERENCE:
        raise ValueError('TRAINING_PREPARATION_REAL_FINAL_REFERENCE_SCOPE_REQUIRED')
    old_end = date.fromisoformat(body['history_end'])
    if first <= old_end or (through - date.fromisoformat(body['history_start'])).days + 1 > MAX_DAYS:
        raise ValueError('TRAINING_PREPARATION_FRESH_BOUNDED_PERIOD_REQUIRED')
    request = dict(schema=SCHEMA, seedFreezeId=seed['freeze_id'], buildTemplateId=digest(build_template),
        planNotBefore=first.isoformat(), validationSessions=validation_sessions,
        labelMaturitySessions=label_maturity_sessions)
    root = Path(output_root).expanduser().resolve() / digest(request)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.prepare.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('TRAINING_PREPARATION_BUSY') from error
        write_once(root / 'request.json', request)
        state_path = root / 'state.json'
        previous = _checkpoint(state_path, root)
        if previous.get('requestId') not in {None, digest(request)}:
            raise ValueError('TRAINING_PREPARATION_CHECKPOINT_BINDING_MISMATCH')
        if previous.get('calendarCheckedDate', through.isoformat()) > through.isoformat():
            raise ValueError('TRAINING_PREPARATION_CUTOFF_REGRESSION')
        if previous.get('status') == 'READY':
            # Source verification repeats at the native training boundary, not
            # on every status or data-preparation retry.
            return dict(previous, requestCount=0)
        transport = client or TwseHttpClient(TwseHttpConfig(timeout_seconds=20, max_attempts=1,
            min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
        if (transport.config.allow_insecure_ssl_fallback or transport.config.min_request_interval_seconds < 1
                or transport.config.max_attempts != 1):
            raise ValueError('TRAINING_PREPARATION_SECURE_SINGLE_ATTEMPT_TRANSPORT_REQUIRED')
        budget = _Budget(transport, max_requests)
        calendar_path = Path(previous['calendarFile']) if previous.get('calendarFile') else None
        receipt = dict(requestId=digest(request), schema=SCHEMA, status='WAITING', checkedAt=clock.isoformat(),
            seedScopeFile=str(seed_path), seedDatasetId=seed['freeze_id'], calendarFile=str(calendar_path) if calendar_path else None,
            marketFile=previous.get('marketFile'), scopeFile=None, datasetId=None,
            verifiedThrough=None, calendarThrough=None, independentSessions=0, maturedSessions=0,
            maturedEventSessions=0,
            validationSessions=validation_sessions, requiredSessions=validation_sessions + label_maturity_sessions,
            days={}, unknownDates=[], gaps=[], dependencies=[], requestCount=0,
            historicalInformationSetCertified=False, researchOnly=True, approvedForLive=False,
            fittingPerformed=False, executionAction='NO_ACTION')
        if through <= old_end:
            receipt.update(dependencies=['WAITING_FOR_NEW_COMPLETED_DATES_AFTER_FIXED_SEED_HISTORY'], accessBlocked=False)
            identifier = digest(receipt)
            write_once(root / 'receipts' / identifier / 'receipt.json', receipt)
            receipt['receiptFile'] = str(root / 'receipts' / identifier / 'receipt.json')
            _atomic(state_path, receipt)
            return receipt
        try:
            if calendar_path is None:
                calendar_path = _seed_calendar(seed, source_root, root / 'calendar')
            # Refresh present-year official news on each new local capture day.
            # Same-day resumptions reuse verified originals and repair gaps only.
            years = tuple(range(date.fromisoformat(body['history_start']).year, through.year + 1))
            previous_day = date.fromisoformat(previous.get('calendarCheckedDate', body['history_end']))
            refresh = (tuple(range(previous_day.year, through.year + 1))
                if previous.get('calendarCheckedDate') != through.isoformat() else ())
            calendar_path = collect_final_calendars(root / 'calendar', years=years,
                previous=calendar_path, refresh_years=refresh, client=budget,
                max_requests=min(max_requests, 20))
            receipt['calendarFile'] = str(calendar_path)
            calendar, documents = load_final_calendar(calendar_path)
            selected, diagnostic = select_final_calendar(calendar, clock, documents=documents,
                start=old_end + timedelta(days=1), end=through, evidence_id=digest(calendar))
            receipt['days'] = selected['days']
            receipt['unknownDates'] = diagnostic['unknown_dates']
            receipt['calendarCheckedDate'] = through.isoformat()
            errors = calendar.get('collection_errors', [])
            if any('ACCESS_BLOCKED' in str(row) or row.get('category') == 'access_blocked' for row in errors):
                budget.blocked = True
            sessions = [day for day in selected['sessions'] if day >= first.isoformat()]
            # Unknown dates break the prospective prefix; later OPEN dates do
            # not let the planner skip an unverified earlier session.
            for day in diagnostic['unknown_dates']:
                if day >= first.isoformat():
                    sessions = [item for item in sessions if item < day]
            receipt['independentSessions'] = min(validation_sessions, len(sessions))
            receipt['maturedSessions'] = min(label_maturity_sessions, max(0, len(sessions) - validation_sessions))
            receipt['maturedEventSessions'] = min(validation_sessions, max(0, len(sessions) - label_maturity_sessions))
            for day, value in sorted(selected['days'].items()):
                if value == 'UNKNOWN':
                    break
                receipt['calendarThrough'] = day
            if budget.blocked:
                raise ValueError('TRAINING_PREPARATION_STOPPED_ACCESS_BLOCKED')
            target_end = (date.fromisoformat(sessions[receipt['requiredSessions'] - 1])
                if len(sessions) >= receipt['requiredSessions'] else through)
            market_progress = (lambda day, calls, count:
                progress(f'補收官方行情 {day} · 已核對 {count} 日')) if progress else None
            market_path = collect_final_market(calendar_path, start=old_end + timedelta(days=1), end=target_end,
                root=root / 'market', previous=previous.get('marketFile'), client=budget,
                max_requests=min(max_requests, 256), progress=market_progress)
            receipt['marketFile'] = str(market_path)
            market, _ = load_final_market(market_path)
            receipt['gaps'] = market['gaps']
            captured = {entry['date'] for entry in market['documents']}
            for day, value in sorted(selected['days'].items()):
                if value == 'UNKNOWN' or (value == 'OPEN' and day not in captured):
                    break
                if value == 'OPEN':
                    receipt['verifiedThrough'] = day
            if (any(day <= target_end.isoformat() for day in diagnostic['unknown_dates'])
                    or market['status'] != 'COMPLETE'):
                receipt['status'] = 'PARTIAL'
            elif len(sessions) >= receipt['requiredSessions']:
                evaluation_day = sessions[receipt['requiredSessions'] - 1]
                # Pin the FIRST mature date. Later observed days must not alter
                # which events belong to this immutable prospective plan.
                build = deepcopy(build_template)
                build.update(event_end=sessions[validation_sessions - 1],
                    evaluation_cutoff=evaluation_day + 'T13:30:00+08:00')
                path, identifier = _extend_scope(seed_path, calendar_path, market_path, build=build,
                    end=evaluation_day, root=root, budget=budget)
                receipt.update(status='READY', scopeFile=str(path), datasetId=identifier,
                    evaluationCutoff=build['evaluation_cutoff'], validationStart=sessions[0],
                    validationEnd=sessions[validation_sessions - 1])
            else:
                receipt['dependencies'] = ['WAITING_FOR_180_UNSEEN_OFFICIAL_SESSIONS_PLUS_20_H20_MATURITY_SESSIONS']
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            receipt.update(status='PARTIAL', error=str(error))
        receipt['requestCount'] = budget.count
        receipt['accessBlocked'] = budget.blocked
        identifier = digest(receipt)
        write_once(root / 'receipts' / identifier / 'receipt.json', receipt)
        receipt['receiptFile'] = str(root / 'receipts' / identifier / 'receipt.json')
        _atomic(state_path, receipt)
        return receipt
