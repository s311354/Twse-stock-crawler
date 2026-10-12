"""Synthetic fixtures only; stable counts/equality, never wall-clock assertions."""
from copy import deepcopy
from dataclasses import replace
import datetime as dt
import json
from pathlib import Path
import sqlite3
import zipfile

import pytest

from application.historical_replay import replay_close_cutoff
from application.replay_cache import ReplayCache
from application.replay_contracts import ReplayAccount, digest
from application.replay_inputs import PinnedReplayEvidence
from application.research_audit import coverage_audit
from infrastructure.replay_runtime import current_versions, build_replay_engine
from infrastructure.research_archive import create_dataset
from infrastructure.research_job import ResearchJob
from infrastructure.research_objects import GRAPH_STORE_NAME, ResearchObjects
from tests.test_research_archive import raw_fixture, add_raw, annual_calendar
from tests.test_strict_replay import request


def setup(tmp_path):
    payload = raw_fixture(); add_raw(payload)
    evidence = create_dataset(payload, tmp_path/'datasets')
    req = replace(request(tmp_path, payload, day='2026-09-10'), versions=current_versions(),
                  evidence_id=evidence.parent.name)
    return evidence, req


def next_day(req):
    return replace(req, as_of=replay_close_cutoff('2026-09-11'), history_end=dt.date(2026,9,11))


def test_cache_on_off_legacy_full_result_and_coverage_equal(tmp_path):
    evidence, req = setup(tmp_path)
    legacy, _ = build_replay_engine(evidence)
    with ResearchJob(evidence, tmp_path/'on') as on, ResearchJob(evidence, tmp_path/'off', cache_enabled=False) as off:
        for day in (req, next_day(req)):
            expected = legacy.run(day)
            audit = coverage_audit(day, legacy.evidence.select(day), expected)
            a, b = on.run_day(day), off.run_day(day)
            assert a.result == b.result == expected
            assert a.coverage == b.coverage == audit
            assert a.state_id == b.state_id
            assert not day.state_path.parent.joinpath('on', 'source.zip').exists()
        assert on.export_snapshot() == legacy.state.export_snapshot(next_day(req))
        assert on.cache_stats['sqlite_ixbrl_miss'] == 1
        assert on.cache_stats['sqlite_index_hit'] == 1
        assert off.cache_stats.get('sqlite_index_hit', 0) == 0


def test_one_load_one_select_per_day_and_zero_on_retry(tmp_path, monkeypatch):
    evidence, req = setup(tmp_path)
    import infrastructure.research_job as module
    counts = dict(load=0, select=0)
    original_load, original_select = module.load_evidence, PinnedReplayEvidence.select
    def load(*args):
        counts['load'] += 1
        return original_load(*args)
    def select(self, *args):
        counts['select'] += 1
        return original_select(self, *args)
    monkeypatch.setattr(module, 'load_evidence', load)
    monkeypatch.setattr(PinnedReplayEvidence, 'select', select)
    with ResearchJob(evidence, tmp_path/'job') as job:
        first = job.run_day(req)
        assert job.run_day(req).result == first.result
        job.run_day(next_day(req))
        assert counts == dict(load=1, select=2)
    with ResearchJob(evidence, tmp_path/'job') as job:
        job.run_day(next_day(req))
        assert counts == dict(load=2, select=2)


def test_shared_inputs_cannot_be_mutated_and_future_cutoff_stays_separate(tmp_path):
    evidence, req = setup(tmp_path)
    with ResearchJob(evidence, tmp_path/'job') as job:
        first = job.evidence.select(req)
        baseline = deepcopy(first.evidence)
        first.evidence['filings']['2330'][0]['content_sha256'] = 'poison'
        first.diagnostics['financials']['2330']['missing_periods'] = ['poison']
        job.evidence.select(next_day(req))
        assert job.evidence.select(req).evidence == baseline
        assert job.cache.bytes <= job.cache.max_bytes
        job.cache.clear()
        assert job.evidence.select(req).evidence == baseline


def test_result_account_is_not_cached_by_symbol_date(tmp_path):
    evidence, req = setup(tmp_path)
    with ResearchJob(evidence, tmp_path/'a') as job:
        job.run_day(req)
        with pytest.raises(ValueError, match='INPUT_CHANGED'):
            job.run_day(replace(req, account=ReplayAccount(2e6, 2e6, ())))
        with pytest.raises(ValueError, match='IDENTITY'):
            job.run_day(replace(next_day(req), replay_id='different'))
    with ResearchJob(evidence, tmp_path/'b') as job:
        other = job.run_day(replace(req, account=ReplayAccount(2e6, 2e6, ())))
        assert other.result['account']['cash'] == 2e6


@pytest.mark.parametrize('stage', ['delta.json', 'complete.json', 'after_receipt'])
def test_interrupted_commit_recovers_exactly_once(tmp_path, monkeypatch, stage):
    evidence, req = setup(tmp_path)
    import infrastructure.research_job as module
    original = module.write_once
    def interrupt(path, value):
        if path.name == stage: raise RuntimeError('interruption')
        original(path, value)
        if stage == 'after_receipt' and path.name == 'complete.json': raise RuntimeError('interruption')
    with ResearchJob(evidence, tmp_path/'job') as job:
        monkeypatch.setattr(module, 'write_once', interrupt)
        with pytest.raises(RuntimeError): job.run_day(req)
        monkeypatch.setattr(module, 'write_once', original)
        if stage != 'after_receipt':
            with pytest.raises(ValueError, match='PENDING_DAY'):
                job.run_day(next_day(req))
    with ResearchJob(evidence, tmp_path/'job') as job:
        recovered = job.run_day(req)
        assert job.run_day(req).result == recovered.result
        with sqlite3.connect(tmp_path/'job'/'state.sqlite3') as conn:
            assert conn.execute('SELECT count(*) FROM replay_sessions').fetchone()[0] == 1
        second = job.run_day(next_day(req))
        assert second.result['pae_state']['previous_result_sha256'] == digest(recovered.result)
        assert job.reproduce()['equal']


def test_portable_bundle_rebuilds_index_and_reexecutes(tmp_path):
    evidence, req = setup(tmp_path)
    with ResearchJob(evidence, tmp_path/'job') as job:
        job.run_day(req); last = job.run_day(next_day(req))
        job.export_bundle(tmp_path/'portable.zip')
    with zipfile.ZipFile(tmp_path/'portable.zip') as archive:
        assert not any(name.endswith('.sqlite3') for name in archive.namelist())
        archive.extractall(tmp_path/'portable')
    with ResearchJob(tmp_path/'portable'/'evidence.json', tmp_path/'portable') as resumed:
        assert resumed.run_day(next_day(req)).result == last.result
        assert resumed.reproduce()['equal']


def test_object_corruption_and_old_date_refused(tmp_path):
    evidence, req = setup(tmp_path)
    with ResearchJob(evidence, tmp_path/'job') as job:
        first = job.run_day(req); job.run_day(next_day(req))
        with pytest.raises(ValueError, match='OUT_OF_ORDER'): job.run_day(req)
        object_path = job.objects.path(first.receipt['objects']['source'])
    object_path.write_bytes(b'corrupt synthetic test')
    with pytest.raises(ValueError, match='HASH_MISMATCH'):
        ResearchJob(evidence, tmp_path/'job')


def test_source_environment_objects_once_and_drift_rejected(tmp_path, monkeypatch):
    evidence, req = setup(tmp_path)
    import infrastructure.research_objects as objects
    calls = []
    original = objects.source_zip
    def compress():
        calls.append(True); return original()
    monkeypatch.setattr(objects, 'source_zip', compress)
    shared = tmp_path/'shared'/'objects'
    with ResearchJob(evidence, tmp_path/'a', object_root=shared) as job:
        a = job.run_day(req); b = job.run_day(next_day(req))
        assert a.receipt['objects']['source'] == b.receipt['objects']['source']
    with ResearchJob(evidence, tmp_path/'b', object_root=shared) as job:
        job.run_day(req)
        monkeypatch.setattr(objects.runtime, 'parameter_manifest', lambda: {'changed': True})
        with pytest.raises(ValueError, match='VERSION_CHANGED'): job.run_day(next_day(req))
    assert len(calls) == 1


def test_lru_budget_and_copy_on_hit():
    cache = ReplayCache(120)
    x = cache.memo('x', '1', lambda: {'a': [1]}); x['a'].append(2)
    assert cache.memo('x', '1', lambda: None) == {'a': [1]}
    for i in range(30): cache.memo('x', str(i), lambda: {'a': [1, 2, 3]})
    assert cache.bytes <= 120 and cache.stats['evictions'] > 0
    cache.clear(); assert cache.bytes == 0


def test_reference_state_does_not_copy_full_result_each_day(tmp_path):
    evidence, req = setup(tmp_path)
    with ResearchJob(evidence, tmp_path/'job') as job:
        a = job.run_day(req); b = job.run_day(next_day(req))
        delta = json.loads((b.folder/'delta.json').read_text())
        assert len(delta['session'][3]) < 200
        assert 'sessions' not in delta
        assert delta['previous_state_id'] == a.state_id
        with sqlite3.connect(tmp_path/'job'/'state.sqlite3') as conn:
            assert conn.execute('SELECT max(length(result_json)) FROM replay_sessions').fetchone()[0] < 200


def test_job_results_and_coverage_use_shared_nodes_without_changing_native_outputs(tmp_path):
    evidence, req = setup(tmp_path)
    legacy, _ = build_replay_engine(evidence)
    with ResearchJob(evidence, tmp_path/'job') as job:
        first = job.run_day(req)
        with sqlite3.connect(job.objects.root / GRAPH_STORE_NAME) as connection:
            before = dict(connection.execute('SELECT sha256, body FROM report_objects'))
        reuse = job.objects.stats['nodes_reused']
        second = job.run_day(next_day(req))
        assert job.objects.stats['nodes_reused'] > reuse
        with sqlite3.connect(job.objects.root / GRAPH_STORE_NAME) as connection:
            after = dict(connection.execute('SELECT sha256, body FROM report_objects'))
        assert all(after[sha] == body for sha, body in before.items())
        for day, stored in ((req, first), (next_day(req), second)):
            expected = legacy.run(day)
            assert stored.result == expected
            assert stored.coverage == coverage_audit(day, legacy.evidence.select(day), expected)
            for name in ('request', 'result', 'coverage', 'delta'):
                sha = stored.receipt['objects'][name]
                assert job.objects._index_path(sha).exists()
                assert not job.objects.path(sha).exists()
                assert digest(job.objects.get_json(sha)) == sha
            assert job.objects.get_json(stored.receipt['objects']['result']) == expected
        assert job.run_day(next_day(req)).result == second.result


def test_changed_evidence_and_state_chain_refused(tmp_path):
    evidence, req = setup(tmp_path)
    with ResearchJob(evidence, tmp_path/'job') as job:
        day = job.run_day(req)
        original = evidence.read_bytes()
        evidence.write_bytes(original + b' ')
        with pytest.raises(ValueError, match='DATASET_CHANGED'): job.run_day(next_day(req))
    evidence.write_bytes(original)
    delta = json.loads((day.folder/'delta.json').read_text())
    delta['previous_state_id'] = '0'*64
    (day.folder/'delta.json').write_text(json.dumps(delta))
    with ResearchJob(evidence, tmp_path/'job') as job:
        with pytest.raises(ValueError, match='CHAIN_INVALID'): job.run_day(req)


def test_v1_import_preserves_observations_and_full_export(tmp_path):
    from infrastructure.storage.replay_state import SQLiteReplayState
    from domain.strategies.portfolio_allocation_v2 import PersistenceObservationV2
    evidence, req = setup(tmp_path)
    store = SQLiteReplayState()
    with store.session(req, ('2026-09-10', '2026-09-11')) as session:
        session.record_session_v2(req.as_of.date(), (PersistenceObservationV2('2330', '3014', True),))
        session.finish(dict(synthetic=True))
    snapshot = store.export_snapshot(req)
    with ResearchJob(evidence, tmp_path/'job') as job:
        job.initialize_from_snapshot(next_day(req), snapshot)
        day = job.run_day(next_day(req))
        result = job.export_snapshot()
        assert result['observations'] == snapshot['observations']
        assert result['sessions'][0] == snapshot['sessions'][0]
        assert day.result['pae_state']['previous_result_sha256'] == snapshot['sessions'][0][2]
        assert job.reproduce()['equal']


def test_cli_job_and_shared_api_full_parity(tmp_path, capsys):
    from interface.research import main
    evidence, req = setup(tmp_path)
    requests = tmp_path/'requests.json'
    requests.write_text(json.dumps([req.inputs(), next_day(req).inputs()]))
    assert main(['job-capture', '--evidence', str(evidence), '--job', str(tmp_path/'cli'),
                 '--requests', str(requests)]) == 0
    with ResearchJob(evidence, tmp_path/'cli') as cli, ResearchJob(evidence, tmp_path/'api') as api:
        api.run_day(req)
        assert cli.run_day(next_day(req)).result == api.run_day(next_day(req)).result
    assert main(['job-reproduce', '--evidence', str(evidence), '--job', str(tmp_path/'cli'),
                 '--output', str(tmp_path/'check.json')]) == 0
    assert json.loads((tmp_path/'check.json').read_text())['equal']


def test_proof_and_parser_versions_are_cache_keys(tmp_path):
    from application.research_evidence import materialize_filings
    from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
    payload = raw_fixture(); row = add_raw(payload)
    cache, calls = ReplayCache(), []
    def parse(body):
        calls.append(1); return parse_ixbrl(body)
    first, _ = materialize_filings(payload, parse_filing=parse, cache=cache, version='one')
    row['publication_proof']['published_at'] = '2026-08-03'
    row['publication_proof']['quotes'] = ['SYNTHETIC TEST ONLY']
    changed, _ = materialize_filings(payload, parse_filing=parse, cache=cache, version='one')
    assert len(calls) == 1
    assert changed['2330'][0]['available_at'] != first['2330'][0]['available_at']
    materialize_filings(payload, parse_filing=parse, cache=cache, version='two')
    assert len(calls) == 2


def test_parallel_job_lock_refuses_second_writer(tmp_path):
    evidence, _ = setup(tmp_path)
    with ResearchJob(evidence, tmp_path/'job'):
        with pytest.raises(BlockingIOError): ResearchJob(evidence, tmp_path/'job')


def test_nonempty_observation_delta_matches_legacy_state(tmp_path):
    from infrastructure.storage.replay_state import SQLiteReplayState
    from infrastructure.storage.research_state import IncrementalResearchState
    from domain.strategies.portfolio_allocation_v2 import PersistenceObservationV2
    from application.replay_service import json_value
    evidence, req = setup(tmp_path)
    objects = ResearchObjects(tmp_path/'objects')
    incremental = IncrementalResearchState(result_store=objects)
    legacy = SQLiteReplayState()
    dates = ('2026-09-10', '2026-09-11')
    def commit(store, request):
        with store.session(request, dates) as session:
            if session.cached is not None: return session.cached
            session.load_recent_v2(('2330',), ('3014',), through_date=request.history_end, limit_per_pair=3)
            session.record_session_v2(request.history_end, (PersistenceObservationV2('2330', '3014', True),))
            # Match the native engine's JSON boundary (Persistence uses tuples
            # internally); this is not an alternate model result representation.
            result = json_value(dict(as_of=request.as_of.isoformat(), synthetic=True, pae_state=dict(
                previous_result_sha256=session.previous_result_sha256,
                observations=session.observations, persistence=session.reads)))
            session.finish(result)
            return result
    original = replace(req, state_path=tmp_path/'incremental.sqlite3')
    baseline = replace(req, state_path=tmp_path/'legacy.sqlite3')
    assert commit(incremental, original) == commit(legacy, baseline)
    base = incremental.base(original)
    delta = incremental.delta(original, base['state_id'])
    incremental.validate_delta(original, delta, base['state_id'])
    assert len(delta['observations']) == 1
    restored = replace(next_day(original), state_path=tmp_path/'restored.sqlite3')
    incremental.restore_rows(restored, [delta['session']], delta['observations'])
    expected = commit(legacy, next_day(baseline))
    assert commit(incremental, restored) == expected
    assert commit(incremental, restored) == expected  # Same-day retry: no second observation.
    assert incremental.export_snapshot(restored) == legacy.export_snapshot(next_day(baseline))
    assert len(incremental.export_snapshot(restored)['observations']) == 2
