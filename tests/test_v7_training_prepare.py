"""Synthetic preparation orchestration; no real network, model or market data."""
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from domain.errors import AccessBlockedError
import infrastructure.les_v7_training_prepare as prepare


STAMP = '2026-10-18T18:00:00+08:00'


class Client:
    config = SimpleNamespace(allow_insecure_ssl_fallback=False, min_request_interval_seconds=1, max_attempts=1)
    def request_bytes(self, *args, before_attempt, **kwargs):
        before_attempt()
        return SimpleNamespace(content=b'SYNTHETIC')


@pytest.fixture
def ports(tmp_path, monkeypatch):
    seed = tmp_path / 'seed' / 'freezes' / ('a' * 64) / 'manifest.json'
    record = dict(freeze_id='a' * 64, body=dict(history_start='2023-07-03', history_end='2026-10-12',
        dataset_kind='RESEARCH', reference_evidence_policy=prepare.FINAL_REFERENCE))
    calendar = tmp_path / 'calendar.json'
    calendar.write_text('{}')
    market = tmp_path / 'market.json'
    market.write_text('{}')
    calls = []
    states = {f'2026-10-{d:02d}': 'OPEN' if d <= 16 else 'CLOSED' for d in range(13, 19)}
    monkeypatch.setattr(prepare, '_now', lambda: datetime.fromisoformat(STAMP))
    monkeypatch.setattr(prepare, '_verify_freeze', lambda *a, **k: deepcopy(record))
    monkeypatch.setattr(prepare, '_seed_calendar', lambda *a: calendar)
    def collect_calendar(*a, **kwargs):
        calls.append(('calendar', kwargs))
        return calendar
    monkeypatch.setattr(prepare, 'collect_final_calendars', collect_calendar)
    monkeypatch.setattr(prepare, 'load_final_calendar', lambda *a: ({'collection_errors': []}, None))
    monkeypatch.setattr(prepare, 'select_final_calendar', lambda *a, **k: (
        dict(days=deepcopy(states), sessions=[day for day, value in states.items() if value == 'OPEN']),
        dict(unknown_dates=[day for day, value in states.items() if value == 'UNKNOWN'])))
    def collect_market(*a, **kwargs):
        calls.append(('market', kwargs))
        return market
    monkeypatch.setattr(prepare, 'collect_final_market', collect_market)
    monkeypatch.setattr(prepare, 'load_final_market', lambda *a: (
        dict(status='COMPLETE', gaps=[], documents=[{'date': day} for day, value in states.items() if value == 'OPEN']), None))
    return dict(seed=seed, root=tmp_path / 'output', calls=calls, states=states, calendar=calendar, market=market)


def run(ports, **kwargs):
    return prepare.prepare_elec_training_data(ports['seed'], {'SYNTHETIC': True},
        plan_not_before='2026-10-19', output_root=ports['root'], cutoff=STAMP, client=Client(), **kwargs)


def test_future_anchor_collects_only_existing_closed_range_and_defers_full_scope(ports, monkeypatch):
    monkeypatch.setattr(prepare, '_extend_scope', lambda *a, **k: pytest.fail('SYNTHETIC premature expansion'))
    receipt = run(ports)
    assert receipt['status'] == 'WAITING'
    assert receipt['independentSessions'] == receipt['maturedSessions'] == 0
    assert receipt['requiredSessions'] == 200 and receipt['validationSessions'] == 180
    assert receipt['scopeFile'] is None and receipt['datasetId'] is None
    assert receipt['verifiedThrough'] == '2026-10-16'
    assert receipt['calendarThrough'] == '2026-10-18'
    market_call = ports['calls'][1][1]
    assert market_call['start'].isoformat() == '2026-10-13'
    assert market_call['end'].isoformat() == '2026-10-18'
    assert receipt['historicalInformationSetCertified'] is False
    assert receipt['fittingPerformed'] is False
    assert Path(receipt['receiptFile']).is_file()


def test_same_day_resume_reuses_calendar_and_market_checkpoint(ports):
    first = run(ports)
    second = run(ports)
    assert second['requestId'] == first['requestId']
    assert ports['calls'][0][1]['refresh_years'] == (2026,)
    assert ports['calls'][2][1]['refresh_years'] == ()
    assert ports['calls'][3][1]['previous'] == str(ports['market'])
    assert Path(first['receiptFile']).is_file()


def test_restart_across_year_refreshes_missing_elapsed_year_news_queries(ports, monkeypatch):
    run(ports)
    later = '2027-01-02T18:00:00+08:00'
    monkeypatch.setattr(prepare, '_now', lambda: datetime.fromisoformat(later))
    prepare.prepare_elec_training_data(ports['seed'], {'SYNTHETIC': True},
        plan_not_before='2026-10-19', output_root=ports['root'], cutoff=later, client=Client())
    assert ports['calls'][2][1]['refresh_years'] == (2026, 2027)


def test_corrupt_checkpoint_cannot_claim_ready_or_trigger_new_requests(ports):
    import json
    out = run(ports)
    state = Path(out['receiptFile']).parents[2] / 'state.json'
    payload = json.loads(state.read_text())
    payload['status'] = 'READY'
    state.write_text(json.dumps(payload))
    count = len(ports['calls'])
    with pytest.raises(ValueError, match='CHECKPOINT_RECEIPT_HASH_MISMATCH'):
        run(ports)
    assert len(ports['calls']) == count


def test_unknown_is_explicit_partial_and_breaks_verified_prefix(ports, monkeypatch):
    ports['states']['2026-10-14'] = 'UNKNOWN'
    out = run(ports)
    assert out['status'] == 'PARTIAL'
    assert out['unknownDates'] == ['2026-10-14']
    assert out['verifiedThrough'] == '2026-10-13'
    # The native market collector receives the calendar; its established
    # contract (tested in test_v7_market_final) excludes UNKNOWN requests.
    assert ports['calls'][1][0] == 'market'


def test_access_blocked_calendar_prevents_all_market_requests(ports, monkeypatch):
    monkeypatch.setattr(prepare, 'load_final_calendar', lambda *a: (
        {'collection_errors': [{'category': 'access_blocked', 'reason': 'SYNTHETIC ACCESS_BLOCKED'}]}, None))
    out = run(ports)
    assert out['status'] == 'PARTIAL' and out['accessBlocked'] is True
    assert [call[0] for call in ports['calls']] == ['calendar']


def test_global_budget_and_access_blocked_are_not_retried():
    budget = prepare._Budget(Client(), 1)
    budget.request_bytes('GET', 'SYNTHETIC')
    with pytest.raises(ValueError, match='REQUEST_BUDGET'):
        budget.request_bytes('GET', 'SYNTHETIC')
    assert budget.count == 1
    class Blocked(Client):
        def request_bytes(self, *a, before_attempt, **k):
            before_attempt()
            raise AccessBlockedError('SYNTHETIC denied')
    budget = prepare._Budget(Blocked(), 2)
    with pytest.raises(AccessBlockedError):
        budget.request_bytes('GET', 'SYNTHETIC')
    with pytest.raises(ValueError, match='STOPPED_ACCESS_BLOCKED'):
        budget.request_bytes('GET', 'SYNTHETIC')
    assert budget.count == 1


def test_future_clock_or_changed_native_session_counts_fail_before_collection(ports):
    with pytest.raises(ValueError, match='REAL_CUTOFF'):
        prepare.prepare_elec_training_data(ports['seed'], {}, plan_not_before='2026-10-19',
            output_root=ports['root'], cutoff='2026-10-19T18:00:00+08:00')
    with pytest.raises(ValueError, match='180_PLUS_20'):
        run(ports, validation_sessions=60)
    assert not ports['calls']


def test_no_new_completed_date_waits_without_source_loading_or_http(ports, monkeypatch):
    monkeypatch.setattr(prepare, '_seed_calendar', lambda *a: pytest.fail('SYNTHETIC unnecessary representative shard'))
    out = prepare.prepare_elec_training_data(ports['seed'], {'SYNTHETIC': True},
        plan_not_before='2026-10-19', output_root=ports['root'], cutoff='2026-10-12T18:00:00+08:00', client=Client())
    assert out['status'] == 'WAITING' and out['requestCount'] == 0
    assert out['calendarFile'] is None and out['verifiedThrough'] is None
    assert not ports['calls']


def test_only_first_200_native_sessions_pin_ready_dates(ports, monkeypatch):
    start = datetime.fromisoformat('2026-10-19T13:30:00+08:00')
    # Deliberately synthetic explicit OPEN dates; no weekday inference in code.
    dates = [(start + timedelta(days=i)).date().isoformat() for i in range(202)]
    ports['states'].clear()
    ports['states'].update({day: 'OPEN' for day in dates})
    ports['states'][dates[200]] = 'UNKNOWN'
    later = (start + timedelta(days=201, hours=1)).isoformat()
    monkeypatch.setattr(prepare, '_now', lambda: datetime.fromisoformat(later))
    expansions = []
    def extend(*a, **kwargs):
        expansions.append(kwargs)
        return Path('SYNTHETIC-new-fixed-scope'), 'b' * 64
    monkeypatch.setattr(prepare, '_extend_scope', extend)
    out = prepare.prepare_elec_training_data(ports['seed'], {'event_start': '2026-01-14'},
        plan_not_before='2026-10-19', output_root=ports['root'], cutoff=later, client=Client())
    assert out['status'] == 'READY'
    assert out['maturedSessions'] == 20 and out['maturedEventSessions'] == 180
    assert out['validationStart'] == dates[0] and out['validationEnd'] == dates[179]
    assert out['evaluationCutoff'] == dates[199] + 'T13:30:00+08:00'
    assert expansions[0]['end'] == dates[199]
    assert expansions[0]['build']['event_end'] == dates[179]
    assert ports['calls'][1][1]['end'].isoformat() == dates[199]
    calls = len(ports['calls'])
    again = prepare.prepare_elec_training_data(ports['seed'], {'event_start': '2026-01-14'},
        plan_not_before='2026-10-19', output_root=ports['root'], cutoff=later, client=Client())
    assert again['scopeFile'] == out['scopeFile'] and len(ports['calls']) == calls


def test_real_market_collection_adapts_three_argument_callback_to_one_phase(ports, monkeypatch):
    import infrastructure.les_v7_calendar_final as calendar
    from infrastructure.les_v7_market_final import collect_final_market, load_final_market
    from infrastructure.crawler.twse_http_client import TwseBinaryResponse
    from tests.test_v7_market_final import content
    monkeypatch.setattr(prepare, 'collect_final_market', collect_final_market)
    monkeypatch.setattr(prepare, 'load_final_market', load_final_market)
    monkeypatch.setattr(calendar, 'load_final_calendar', lambda *a: ({'SYNTHETIC_TEST_ONLY': True}, None))
    monkeypatch.setattr(calendar, 'select_final_calendar', lambda *a, **k: (
        {'days': {day: state for day, state in ports['states'].items()
            if k['start'].isoformat() <= day <= k['end'].isoformat()}}, {}))
    class MarketClient(Client):
        def request_bytes(self, *a, params, before_attempt, **kwargs):
            before_attempt()
            compact = params['date']
            return TwseBinaryResponse(content(compact[:4] + '-' + compact[4:6] + '-' + compact[6:]), 200, 1)
    phases = []
    out = prepare.prepare_elec_training_data(ports['seed'], {'SYNTHETIC': True},
        plan_not_before='2026-10-19', output_root=ports['root'], cutoff=STAMP,
        client=MarketClient(), progress=phases.append)
    assert out['status'] == 'WAITING' and out['requestCount'] == 4
    assert len(phases) == 4 and all(isinstance(phase, str) and '補收官方行情' in phase for phase in phases)
    assert len(list(Path(out['marketFile']).parents[2].glob('jobs/*/days/*.json'))) == 4


def test_longer_event_span_reshards_exact_global_union_preserving_parent_bytes(tmp_path, monkeypatch):
    symbols = [str(1000 + index) for index in range(7)]
    chunks = [dict(index=index, symbols=symbols[offset:offset+3], evidence_id=str(index)*64)
              for index, offset in enumerate(range(0, len(symbols), 3))]
    originals = {}
    for chunk in chunks:
        selected = chunk['symbols']
        originals[chunk['evidence_id']] = dict(schema='SYNTHETIC', symbols=selected,
            scope_binding={'SYNTHETIC': chunk['index']}, market='TWSE', sector='13',
            price_unit='TWD', volume_unit='SHARES', amount_unit='TWD',
            calendar={'SYNTHETIC': 'identical'}, corporate_actions_final={'SYNTHETIC': 'identical'},
            filings={s: [] for s in selected}, batches={s: [] for s in selected},
            security_classifications={s: {'symbol': s, 'security_type': 'COMMON'} for s in selected},
            bars=[dict(date='20261012', dataset='STOCK_TABLE', data=[[s, 'SYNTHETIC'] for s in selected]),
                  dict(date='20261012', dataset='TAIEX', data='10000')])
    before = deepcopy(originals)
    def load(body, root, chunk):
        return root / chunk['evidence_id'] / 'evidence.json', chunk['evidence_id'], deepcopy(originals[chunk['evidence_id']])
    class Documents:
        def verify(self, payload): pass
        def check(self): pass
    monkeypatch.setattr(prepare, '_load_bound_scope_shard', load)
    monkeypatch.setattr(prepare.EvidenceDocuments, 'for_evidence', lambda *a: Documents())
    body = dict(universe=symbols, chunks=chunks)
    # Native budget: floor(4096 / 1500 calendar event days) == two stocks.
    batch_size = min(3, 4096 // 1500)
    result = list(prepare._repartition_shards(body, tmp_path, batch_size))
    assert [row[1] for row in result] == [symbols[0:2], symbols[2:4], symbols[4:6], symbols[6:7]]
    assert [s for row in result for s in row[1]] == symbols
    assert len(result[1][3]) == 2  # Exact group crosses old chunk boundary.
    for index, selected, payload, parents, _ in result:
        assert payload['symbols'] == selected
        assert set(payload['security_classifications']) == set(selected)
        assert {row[0] for record in payload['bars'] if record['dataset'] == 'STOCK_TABLE' for row in record['data']} == set(selected)
        assert len([r for r in payload['bars'] if r['dataset'] == 'TAIEX']) == 1
        assert payload['price_unit'] == payload['amount_unit'] == 'TWD'
        assert payload['training_extension_parents'] == parents
    assert originals == before
    originals[chunks[1]['evidence_id']]['price_unit'] = 'SYNTHETIC_WRONG_UNIT'
    with pytest.raises(ValueError, match='COMMON_EVIDENCE_CONFLICT'):
        list(prepare._repartition_shards(body, tmp_path, batch_size))
