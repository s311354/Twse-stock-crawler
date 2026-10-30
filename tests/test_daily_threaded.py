"""Synthetic bounds/ownership; native parity additionally requires real no-GIL.

The unit fixture patches only runtime admission and stock calculation. It does
not claim that a normal GIL interpreter is free-threaded or certify a model.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
import threading
import pickle

import pytest

from application.daily_performance import DailyPerformance, count_work
from application.daily_work import DailyWorkContext, current_work
from config.settings import AppSettings, SettingsError
from infrastructure import daily_threaded as threaded


OWNER = ContextVar('synthetic_thread_owner', default=None)


def identity(**updates):
    return dict(python_version='3.15.0', py_gil_disabled=1, gil_enabled=False, **updates)


@pytest.mark.parametrize('value,code', [
    (dict(python_version='3.14.0', py_gil_disabled=1, gil_enabled=False), 'PYTHON315_NOGIL_BUILD'),
    (dict(python_version='3.15.0', py_gil_disabled=0, gil_enabled=True), 'PYTHON315_NOGIL_BUILD'),
    (dict(python_version='3.15.0', py_gil_disabled=1, gil_enabled=True), 'EFFECTIVE_GIL_DISABLED'),
    (dict(python_version='3.15.0', py_gil_disabled=1, gil_enabled=None), 'EFFECTIVE_GIL_DISABLED'),
])
def test_thread_executor_rejects_unverified_or_effectively_enabled_gil(monkeypatch, value, code):
    monkeypatch.setattr(threaded, 'current_runtime_identity', lambda: value)
    with pytest.raises(ValueError, match=code):
        threaded.DailyThreadExecutor(2)


def test_dependency_import_must_not_reenable_the_gil(monkeypatch):
    replies = iter([identity(), dict(python_version='3.15.0', py_gil_disabled=1, gil_enabled=True)])
    monkeypatch.setattr(threaded, 'current_runtime_identity', lambda: next(replies))
    with pytest.raises(ValueError, match='EFFECTIVE_GIL_DISABLED'):
        threaded.DailyThreadExecutor(2)


@pytest.fixture
def synthetic_runtime(monkeypatch):
    @dataclass
    class Version:
        code: str = 'SYNTHETIC_FIXED_SOURCE'
    class Guard:
        versions = Version()
        def check(self):
            return self.versions
    monkeypatch.setattr(threaded, '_require_free_threading', lambda **kwargs: None)
    monkeypatch.setattr(threaded, 'ResearchVersionGuard', Guard)


def test_thread_tasks_own_work_data_context_and_cache_lifetime(synthetic_runtime, monkeypatch):
    from application.analysis_dataset import AnalysisDatasetBuilder
    from application import evidence_context
    class Evidence:
        def __init__(self, market):pass
        def __enter__(self):return self
        def __exit__(self, *args):pass
    monkeypatch.setattr(evidence_context, 'EvidenceContext', Evidence)
    barrier = threading.Barrier(2)
    observed, lock = [], threading.Lock()
    def kernel(self, market, item, watch, evidence):
        assert OWNER.get() is None
        OWNER.set('worker-change')
        work = current_work()
        assert work is not None and work.parse_memo is None
        assert not hasattr(work, 'snapshot_retention')
        with lock:
            observed.append((threading.get_ident(), work))
        if market.value < 2:
            barrier.wait(timeout=10)
        market.payload.append('private-worker-copy')
        count_work('SYNTHETIC_TASKS')
        return market.value, market.payload
    monkeypatch.setattr(AnalysisDatasetBuilder, 'stock_components', kernel)
    source = [SimpleNamespace(stock_count=1, value=n, payload=[]) for n in range(5)]
    outer, probe = DailyWorkContext('SYNTHETIC_PARENT'), DailyPerformance()
    token = OWNER.set('parent')
    try:
        with outer.activate(), probe.activate():
            result = threaded.DailyThreadExecutor(2).evaluate((n, value, {}) for n, value in enumerate(source))
            assert current_work() is outer
            assert OWNER.get() == 'parent'
        assert [result[n] for n in range(5)] == [(n, ['private-worker-copy']) for n in range(5)]
        assert all(value.payload == [] for value in source)
        assert len({tid for tid, work in observed}) == 2
        assert len({id(work) for tid, work in observed}) == 2
        assert all(work is not outer and work._closed and work.stats['validation_entries'] == 0
                   for tid, work in observed)
        assert probe.counters['SYNTHETIC_TASKS'] == 5
        assert probe.calls['worker_sum.stock_compute'] == 5
        assert probe.calls['stock_thread_pool'] == 1
        assert 'worker_max_process_peak_rss_bytes' not in probe.counters
    finally:
        OWNER.reset(token)
        outer.close()


def test_lazy_bounded_queue_and_out_of_order_completion(synthetic_runtime, monkeypatch):
    state = dict(produced=0, submitted=0, settled=0, peak=0, shutdown=0)
    class Pool:
        def __init__(self, **kwargs):
            assert kwargs['max_workers'] == 2
        def submit(self, function, calculate, encoded):
            assert OWNER.get() is None
            state['submitted'] += 1
            item, market, watch = pickle.loads(encoded)
            future = Future()
            future.item = item
            future.set_result((item, item, item, dict(stages={}, counters={}, process_peak_rss_bytes=None)))
            return future
        def shutdown(self, **kwargs):state['shutdown'] += 1
    def settled_last(pending, **kwargs):
        state['peak'] = max(state['peak'], len(pending))
        assert len(pending) <= 2
        state['settled'] += 1
        return {max(pending, key=lambda f: f.item)}, set()
    def jobs():
        for item in range(7):
            state['produced'] += 1
            assert state['produced'] - state['settled'] <= 2
            yield item, SimpleNamespace(stock_count=1), {}
    monkeypatch.setattr(threaded, 'ThreadPoolExecutor', Pool)
    monkeypatch.setattr(threaded, 'wait', settled_last)
    result = threaded.DailyThreadExecutor(2).evaluate(jobs())
    assert list(result) == list(range(7))
    assert [result[item] for item in range(7)] == [(item, item) for item in range(7)]
    assert state == dict(produced=7, submitted=7, settled=7, peak=2, shutdown=1)


@pytest.mark.parametrize('value', [1, 3, 4, True])
def test_thread_worker_bound_cannot_expand(value):
    with pytest.raises(ValueError, match='DAILY_THREAD_WORKERS_MUST_BE_2'):
        threaded.DailyThreadExecutor(value)


def test_capacity_duplicate_and_missing_index_fail_without_fallback(synthetic_runtime, monkeypatch):
    calls = []
    class Pool:
        def __init__(self, **kwargs):pass
        def submit(self, function, calculate, encoded):
            item, market, watch = pickle.loads(encoded)
            calls.append(item)
            future = Future()
            future.set_result((item, None, None, dict(stages={}, counters={}, process_peak_rss_bytes=None)))
            return future
        def shutdown(self, **kwargs):pass
    monkeypatch.setattr(threaded, 'ThreadPoolExecutor', Pool)
    with pytest.raises(ValueError, match='CAPACITY_EXCEEDED'):
        threaded.DailyThreadExecutor(2, max_packet_bytes=20, max_inflight_bytes=40).evaluate(
            [(0, SimpleNamespace(stock_count=1), {})])
    assert not calls
    with pytest.raises(ValueError, match='DUPLICATE_OR_INVALID'):
        threaded.DailyThreadExecutor(2).evaluate([(0, SimpleNamespace(stock_count=1), {})] * 2)
    with pytest.raises(ValueError, match='SEQUENCE_INCOMPLETE'):
        threaded.DailyThreadExecutor(2).evaluate([(1, SimpleNamespace(stock_count=1), {})])


def test_worker_error_closes_work_and_never_returns_partial_output(synthetic_runtime, monkeypatch):
    from application.analysis_dataset import AnalysisDatasetBuilder
    from application import evidence_context
    class Evidence:
        def __init__(self, market):pass
        def __enter__(self):return self
        def __exit__(self, *args):pass
    monkeypatch.setattr(evidence_context, 'EvidenceContext', Evidence)
    owners = []
    def fail(self, *args):
        owners.append(current_work())
        raise ValueError('SYNTHETIC_STOCK_FAILURE')
    monkeypatch.setattr(AnalysisDatasetBuilder, 'stock_components', fail)
    with pytest.raises(ValueError, match='SYNTHETIC_STOCK_FAILURE'):
        threaded.DailyThreadExecutor(2).evaluate([(0, SimpleNamespace(stock_count=1), {})])
    assert owners and all(work._closed for work in owners)


def test_executor_settings_default_serial_and_process_remain_compatible(synthetic_runtime):
    assert AppSettings.from_env({}, validate_paths=False).daily_executor == 'process'
    assert AppSettings.from_env({}, validate_paths=False).daily_strategy_workers == 1
    explicit = AppSettings.from_env({'TWSE_DAILY_EXECUTOR':'thread', 'TWSE_DAILY_STRATEGY_WORKERS':'2'}, validate_paths=False)
    assert explicit.daily_executor == 'thread' and explicit.daily_strategy_workers == 2
    from stockanalysis import TwseCrawker
    assert isinstance(TwseCrawker(0, app_settings=explicit)._analysis_dataset_builder.stock_executor,
                      threaded.DailyThreadExecutor)
    assert TwseCrawker(0, app_settings=AppSettings(daily_executor='serial'))._analysis_dataset_builder.stock_executor is None


@pytest.mark.parametrize('values', [
    {'TWSE_DAILY_EXECUTOR':'other'},
    {'TWSE_DAILY_EXECUTOR':'thread'},
    {'TWSE_DAILY_EXECUTOR':'thread', 'TWSE_DAILY_STRATEGY_WORKERS':'4'},
    {'TWSE_DAILY_EXECUTOR':'serial', 'TWSE_DAILY_STRATEGY_WORKERS':'2'},
])
def test_executor_setting_rejects_ambiguous_or_unsafe_combinations(values):
    with pytest.raises(SettingsError, match='TWSE_DAILY_EXECUTOR'):
        AppSettings.from_env(values, validate_paths=False)


def test_settings_preflight_rejects_a_normal_gil_build(monkeypatch):
    monkeypatch.setattr(threaded, 'current_runtime_identity', lambda: dict(
        python_version='3.15.0', py_gil_disabled=0, gil_enabled=True))
    with pytest.raises(SettingsError, match='PYTHON315_NOGIL_BUILD'):
        AppSettings.from_env({'TWSE_DAILY_EXECUTOR':'thread',
            'TWSE_DAILY_STRATEGY_WORKERS':'2'}, validate_paths=False)


def test_registry_lazy_import_and_registration_happen_once(monkeypatch):
    from domain import strategy
    class Synthetic(strategy.Strategy):
        name = 'SYNTHETIC_THREAD_REGISTRY'
        def run(self, data):return data
    calls = []
    def load(name):
        calls.append(name)
        return SimpleNamespace(Synthetic=Synthetic)
    monkeypatch.setattr(strategy, 'strategy_registry', strategy.StrategyRegistry())
    monkeypatch.setattr(strategy, '_BUILTIN_STRATEGIES', {Synthetic.name:'synthetic.Synthetic'})
    monkeypatch.setattr(strategy, '_BUILTINS_LOADED', set())
    monkeypatch.setattr(strategy, 'import_module', load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(strategy.get_strategy, [Synthetic.name] * 20))
    assert len(calls) == 1 and all(value is results[0] for value in results)


@pytest.mark.skipif(threaded.current_runtime_identity()['py_gil_disabled'] != 1
                   or threaded.current_runtime_identity()['gil_enabled'] is not False,
                   reason='requires a real free-threaded interpreter with an effective disabled GIL')
@pytest.mark.parametrize('cache', [False, True])
def test_real_nogil_thread_kernel_matches_complete_serial_native_results(cache):
    from application.analysis_dataset import AnalysisDatasetBuilder
    from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
    from application.portfolio_contracts import PortfolioRuntimeContext
    from domain.models import PortfolioPosition
    from tests.test_daily_computation_performance import two_stock_market
    market, profits = two_stock_market(), [[1,1,1,1,.01]] * 2
    work, probe = DailyWorkContext('SYNTHETIC_PARITY_PARENT', enabled=cache), DailyPerformance()
    try:
        with work.activate(), probe.activate():
            serial = AnalysisDatasetBuilder().build_decisions(market, profits)
            threaded_result = AnalysisDatasetBuilder(stock_executor=threaded.DailyThreadExecutor(2)).build_decisions(market, profits)
        assert threaded_result == serial
        context = PortfolioRuntimeContext(account_equity=1_000_000., current_cash=950_000.,
            positions=(PortfolioPosition('2330',100),), default_sector='24')
        assert PortfolioAllocationV2Pipeline().enrich(threaded_result, context) == PortfolioAllocationV2Pipeline().enrich(serial, context)
        assert probe.calls['worker_sum.stock_compute'] == 2
    finally:
        work.close()


@pytest.mark.skipif(threaded.current_runtime_identity()['py_gil_disabled'] != 1
                   or threaded.current_runtime_identity()['gil_enabled'] is not False,
                   reason='requires a real free-threaded interpreter with an effective disabled GIL')
@pytest.mark.parametrize('mode', ['complete_synthetic_model', 'retained_native_diagnostics'])
def test_real_nogil_v7_model_and_retained_native_evidence_keep_complete_serial_output(mode):
    from copy import deepcopy
    from dataclasses import replace
    import json
    from application.analysis_dataset import AnalysisDatasetBuilder
    from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
    from application.portfolio_contracts import PortfolioRuntimeContext
    from domain.les_v7.quality_policy import default_quality_policy
    from domain.les_v7.strategy_policy import make_strategy_policy
    from infrastructure.snapshot_retention import WorkSnapshotRetention
    from tests.strategies.test_daily_v7_pipeline import executable_synthetic_market, synthetic_market
    from tests.strategies.test_low_entry_score_v7 import bundle
    from tests.strategies.test_v7_reported_quality import source_records
    work = DailyWorkContext('SYNTHETIC_V7_THREAD_PARITY')
    owner = WorkSnapshotRetention(work.version, memory_bytes=0)
    work.snapshot_retention = owner
    try:
        with work.activate():
            if mode == 'complete_synthetic_model':
                market = executable_synthetic_market('BUY_SMALL')
            else:
                rules = bundle()['payload']
                policy = make_strategy_policy(policy_version='SYNTHETIC_NATIVE',
                    available_at='2026-01-03T00:00:00Z', quality_policy=default_quality_policy(),
                    **{k:rules[k] for k in ('context_rules','trigger_rules','reference_policy')})
                snapshot = dict(sector='24', les_v6_filings=source_records(), les_v6_mops_filings=source_records())
                observed, cutoff = '2026-10-12T13:35:00+08:00', '2026-10-12T15:00:00+08:00'
                retained = owner.snapshots()
                retained.retain('2330', snapshot, cutoff=observed, market='TWSE')
                market = replace(synthetic_market(), stock_numbers=['2330'], les_v7_bundle_json=None,
                    signal_dates={'2330':cutoff}, fundamental_snapshots=retained,
                    fundamental_capture_cutoffs={'2330':observed}, les_v7_native_evidence=True,
                    les_v7_diagnostics_only=True, les_v7_strategy_policy_json=json.dumps(policy),
                    market_observed_at=observed, market_breadth_snapshots={'2026-10-12':{'source_final':True}})
            profits = [[1,1,1,1,.01]]
            serial = AnalysisDatasetBuilder().build_decisions(market, profits)
            parallel = AnalysisDatasetBuilder(stock_executor=threaded.DailyThreadExecutor(2)).build_decisions(market, profits)
            assert parallel == serial
            context = PortfolioRuntimeContext(account_equity=1_000_000., current_cash=950_000., positions=(), default_sector='24')
            assert PortfolioAllocationV2Pipeline().enrich(parallel, context) == PortfolioAllocationV2Pipeline().enrich(serial, context)
            report = json.loads(serial[0].value('LES v7 Report'))
            assert report['model_version'] == '7.0.0'
            if mode == 'complete_synthetic_model':
                assert report['envelope']['localization']['status'] == 'VALID'
                assert report['envelope']['score']['final_score'] is not None
            else:
                assert report['diagnostics']['native_evidence']['source_packet']['financial_capture_cutoff'] == '2026-10-12T05:35:00+00:00'
                assert report['pae_authorization']['authorized'] is False
    finally:
        owner.close()
        work.close()
