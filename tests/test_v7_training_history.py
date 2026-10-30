"""Historical planning-boundary regressions; synthetic inputs, no real HTTP."""
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from application.replay_contracts import digest
import infrastructure.les_v7_training_history as history


STAMP = '2026-10-18T18:00:00+08:00'


class Documents:
    def check(self):
        pass


class Client:
    config = SimpleNamespace(allow_insecure_ssl_fallback=False, min_request_interval_seconds=1, max_attempts=1)
    def request_bytes(self, *a, before_attempt, **k):
        before_attempt()
        return SimpleNamespace(content=b'SYNTHETIC')


@pytest.fixture
def ports(tmp_path, monkeypatch):
    seed = tmp_path / 'seed' / 'freezes' / ('a' * 64) / 'manifest.json'
    seed.parent.mkdir(parents=True)
    seed.write_bytes(b'SYNTHETIC_IMMUTABLE_SEED')
    record = dict(freeze_id='a' * 64, body=dict(history_start='2026-08-01', history_end='2026-10-12',
        universe=['3014', '3019', '5471'], universe_record={'SYNTHETIC': True},
        dataset_kind='RESEARCH', reference_evidence_policy=history.FINAL_REFERENCE))
    dates = [(date(2026, 8, 1) + timedelta(days=i)).isoformat() for i in range(79)]
    # Explicit source classifications, deliberately not a weekday heuristic.
    states = {day: 'OPEN' if day <= '2026-10-16' else 'CLOSED' for day in dates}
    calendar = dict(SYNTHETIC_TEST_ONLY=True)
    market = dict(start='2026-10-13', end='2026-10-18', status='COMPLETE', gaps=[],
        calendar_sha256=digest(calendar), documents=[{'date': day} for day in dates
            if '2026-10-13' <= day <= '2026-10-16'])
    calls = []
    monkeypatch.setattr(history, '_now', lambda: datetime.fromisoformat(STAMP))
    monkeypatch.setattr(history, 'validate_build_config', lambda config: None)
    def verify(path, root, **kwargs):
        calls.append(('verify', str(path), kwargs))
        return deepcopy(record)
    monkeypatch.setattr(history, '_verify_freeze', verify)
    monkeypatch.setattr(history, 'load_final_calendar', lambda path: (deepcopy(calendar), Documents()))
    monkeypatch.setattr(history, 'load_final_market', lambda path: (deepcopy(market), Documents()))
    monkeypatch.setattr(history, 'select_final_calendar', lambda *a, **k: (
        dict(days=deepcopy(states), sessions=sorted(day for day, value in states.items() if value == 'OPEN')),
        dict(unknown_dates=sorted(day for day, value in states.items() if value == 'UNKNOWN'))))
    return dict(seed=seed, record=record, calendar=calendar, market=market, states=states, calls=calls,
        calendar_path=tmp_path / 'calendar.json', market_path=tmp_path / 'market.json', root=tmp_path / 'output')


def inspect(ports, **kwargs):
    return history.inspect_historical_coverage(ports['seed'], ports['calendar_path'],
        ports['market_path'], cutoff=kwargs.get('cutoff', STAMP))


def build(ports):
    coverage = inspect(ports)
    return dict(event_start='2026-08-01', event_end=coverage['maturedEventThrough'],
        evaluation_cutoff=coverage['evaluationCutoff'], SYNTHETIC_TEST_ONLY=True)


def extend(ports, **kwargs):
    return history.extend_historical_scope(ports['seed'], ports['calendar_path'], ports['market_path'],
        build=kwargs.pop('build', build(ports)), output_root=ports['root'], cutoff=STAMP,
        client=kwargs.pop('client', Client()), **kwargs)


def test_full_official_session_list_and_h20_count_exclude_closed_tail(ports):
    out = inspect(ports)
    assert out['verifiedThrough'] == '2026-10-16'
    assert out['evaluationCutoff'] == '2026-10-16T13:30:00+08:00'
    assert out['maturedEventThrough'] == '2026-09-26'
    assert out['sessions'][0] == '2026-08-01'
    assert len(out['sessions'][out['sessions'].index(out['maturedEventThrough']) + 1:]) == 20
    assert out['stockCount'] == 3 and out['fittingPerformed'] is False
    assert out['seedSourcesVerified'] is False
    assert ports['calls'][0][2] == {'verify_sources': False}
    assert out['historicalInformationSetCertified'] is False and out['approvedForLive'] is False
    assert out['researchOnly'] is True


@pytest.mark.parametrize('failure', ['unknown', 'missing', 'closed', 'partial', 'calendar_binding', 'range'])
def test_unknown_missing_closed_or_mismatched_market_cannot_become_fixed_coverage(ports, failure):
    if failure == 'unknown':
        ports['states']['2026-10-14'] = 'UNKNOWN'
    elif failure == 'missing':
        ports['market']['documents'].pop(1)
    elif failure == 'closed':
        ports['market']['documents'].append({'date': '2026-10-18'})
    elif failure == 'partial':
        ports['market']['status'] = 'PARTIAL'
    elif failure == 'calendar_binding':
        ports['market']['calendar_sha256'] = 'x' * 64
    else:
        ports['market']['start'] = '2026-10-14'
    with pytest.raises(ValueError, match='HISTORICAL_EXTENSION_'):
        inspect(ports)
    assert not ports['root'].exists()


def test_corrupt_native_document_error_prevents_any_extension_or_publication(ports, monkeypatch):
    def corrupt(*a):
        raise ValueError('DOCUMENT_BODY_HASH_MISMATCH')
    monkeypatch.setattr(history, 'load_final_market', corrupt)
    monkeypatch.setattr(history, '_extend_scope', lambda *a, **k: pytest.fail('SYNTHETIC unsafe extension'))
    with pytest.raises(ValueError, match='HASH_MISMATCH'):
        inspect(ports)
    assert not ports['root'].exists()


def test_real_clock_and_nonmature_events_fail_before_action_collection(ports, monkeypatch):
    with pytest.raises(ValueError, match='REAL_CUTOFF'):
        inspect(ports, cutoff='2026-10-19T18:00:00+08:00')
    config = build(ports)
    config['event_end'] = '2026-10-16'
    monkeypatch.setattr(history, '_extend_scope', lambda *a, **k: pytest.fail('SYNTHETIC premature extension'))
    with pytest.raises(ValueError, match='MATURE_H20'):
        extend(ports, build=config)
    config['event_end'] = '2026-09-26'
    config['evaluation_cutoff'] = STAMP
    with pytest.raises(ValueError, match='MATURE_H20'):
        extend(ports, build=config)
    assert not ports['root'].exists()


def test_native_extension_keeps_exact_universe_immutable_seed_and_global_budget(ports, monkeypatch):
    seed_bytes = ports['seed'].read_bytes()
    original = deepcopy(ports['record'])
    target = ports['root'] / 'result' / 'freezes' / ('b' * 64) / 'manifest.json'
    new_record = deepcopy(original)
    new_record['freeze_id'] = 'b' * 64
    new_record['body']['history_end'] = '2026-10-16'
    prefix = ports['root'] / 'SYNTHETIC_PREFIX.json'
    events = []
    monkeypatch.setattr(history, '_market_prefix', lambda *a: prefix)
    def native(seed, calendar, market, **kwargs):
        events.append((seed, calendar, market, kwargs))
        kwargs['budget'].request_bytes('GET', 'SYNTHETIC')
        return target, 'b' * 64
    monkeypatch.setattr(history, '_extend_scope', native)
    monkeypatch.setattr(history, '_verify_freeze', lambda path, *a, **k:
        deepcopy(new_record if path == target else original))
    out = extend(ports, max_requests=1)
    assert out['datasetId'] == 'b' * 64 and out['requestCount'] == 1
    assert out['reusedSeed'] is False and out['scopeFile'] == str(target)
    assert events[0][2] == prefix and events[0][3]['end'] == '2026-10-16'
    assert ports['seed'].read_bytes() == seed_bytes
    assert list(ports['root'].glob('*/receipts/*/receipt.json'))


@pytest.mark.parametrize('failure', ['universe', 'seed_changed'])
def test_changed_universe_or_seed_is_rejected_before_completion_receipt(ports, monkeypatch, failure):
    original = deepcopy(ports['record'])
    target = ports['root'] / 'result' / 'freezes' / ('b' * 64) / 'manifest.json'
    new_record = deepcopy(original)
    new_record['freeze_id'] = 'b' * 64
    new_record['body']['history_end'] = '2026-10-16'
    if failure == 'universe':
        new_record['body']['universe'].pop()
    calls = []
    def verify(path, *a, **k):
        calls.append(path)
        if path == target:
            return deepcopy(new_record)
        changed = deepcopy(original)
        if failure == 'seed_changed' and target in calls:
            changed['body']['SYNTHETIC_MUTATION'] = True
        return changed
    monkeypatch.setattr(history, '_market_prefix', lambda *a: None)
    monkeypatch.setattr(history, '_verify_freeze', verify)
    monkeypatch.setattr(history, '_extend_scope', lambda *a, **k: (target, 'b' * 64))
    with pytest.raises(ValueError, match='CHANGED'):
        extend(ports)
    assert not list(ports['root'].glob('*/receipts/*/receipt.json'))


def test_no_new_open_day_reuses_seed_without_constructing_client_or_extension(ports, monkeypatch):
    for day in ports['states']:
        if day >= '2026-10-13':
            ports['states'][day] = 'CLOSED'
    ports['market']['documents'] = []
    monkeypatch.setattr(history, '_extend_scope', lambda *a, **k: pytest.fail('SYNTHETIC redundant extension'))
    monkeypatch.setattr(history, 'TwseHttpClient', lambda *a, **k: pytest.fail('SYNTHETIC redundant transport'))
    out = extend(ports)
    assert out['reusedSeed'] is True and out['requestCount'] == 0
    assert out['scopeFile'] == str(ports['seed']) and out['datasetId'] == 'a' * 64


def test_secure_transport_and_budget_are_mandatory_before_native_actions(ports, monkeypatch):
    bad = Client()
    bad.config = SimpleNamespace(allow_insecure_ssl_fallback=True, min_request_interval_seconds=1, max_attempts=1)
    monkeypatch.setattr(history, '_extend_scope', lambda *a, **k: pytest.fail('SYNTHETIC insecure extension'))
    with pytest.raises(ValueError, match='SECURE_SINGLE_ATTEMPT'):
        extend(ports, client=bad)
    with pytest.raises(ValueError, match='REQUEST_BUDGET'):
        extend(ports, max_requests=0)


def test_weekend_prefix_retains_native_raw_documents_and_resumes_without_rewrite(tmp_path, monkeypatch):
    import json
    import infrastructure.les_v7_calendar_final as calendar
    from infrastructure.crawler.twse_http_client import TwseBinaryResponse
    from infrastructure.les_v7_market_final import collect_final_market, load_final_market, STOCK_FIELDS
    states = {f'2026-10-{day:02d}': 'OPEN' if day <= 16 else 'CLOSED' for day in range(13, 19)}
    monkeypatch.setattr(calendar, 'load_final_calendar', lambda path: ({'SYNTHETIC_TEST_ONLY': True}, Documents()))
    monkeypatch.setattr(calendar, 'select_final_calendar', lambda *a, **k: (dict(days=states), {}))
    class Source(Client):
        def request_bytes(self, *a, params, before_attempt, **k):
            before_attempt()
            raw = json.dumps(dict(stat='OK', date=params['date'],
                params=dict(date=params['date'], type='ALLBUT0999'), tables=[
                    dict(fields=STOCK_FIELDS, data=[['3014', 'SYNTHETIC_TEST_ONLY', '1000', '1', '100000',
                        '100', '101', '99', '100', '+', '0', '99.9', '1', '100', '1', '']]),
                    dict(fields=['指數', '收盤指數'], data=[['發行量加權股價指數', '10000']])])).encode()
            return TwseBinaryResponse(raw, 200, 1)
    parent = collect_final_market('SYNTHETIC-calendar', start=date(2026, 10, 13), end=date(2026, 10, 18),
        root=tmp_path / 'parent', client=Source(), now=lambda: STAMP)
    original = parent.read_bytes()
    body, docs = load_final_market(parent)
    result = history._market_prefix(body, docs, '2026-10-16', tmp_path / 'prefix')
    trimmed, retained = load_final_market(result)
    assert trimmed['start'] == '2026-10-13' and trimmed['end'] == '2026-10-16'
    assert trimmed['status'] == 'COMPLETE' and not trimmed['gaps']
    assert trimmed['documents'] == body['documents']
    assert trimmed['parent_manifest_id'] == digest(body)
    assert parent.read_bytes() == original
    saved = result.read_bytes()
    assert history._market_prefix(body, docs, '2026-10-16', tmp_path / 'prefix') == result
    assert result.read_bytes() == saved
    raw = trimmed['documents'][0]['document']['sha256']
    retained.objects.path(raw).write_bytes(b'SYNTHETIC_CORRUPTION')
    with pytest.raises(ValueError, match='HASH_MISMATCH'):
        history._market_prefix(body, docs, '2026-10-16', tmp_path / 'prefix')
    assert parent.read_bytes() == original
