"""Opt-in free-threaded stock kernel with isolated work and one result owner.

This backend requires a real Python 3.15+ free-threaded build and an effective
disabled GIL after the native numerical dependencies load. It never forces the
GIL off or falls back to another backend. Persistence and allocation remain in
the caller; only the existing stock-local serial kernel runs concurrently.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import Context
from dataclasses import asdict
import pickle
import threading

from application.daily_performance import DailyPerformance, count_work, measure_stage, merge_worker_performance
from application.daily_work import DailyWorkContext, current_work
from application.replay_contracts import digest
from infrastructure.research_objects import ResearchVersionGuard
from infrastructure.runtime_environment import current_runtime_identity


def _require_free_threading(*, import_dependencies=True):
    identity = current_runtime_identity()
    version = tuple(int(value) for value in identity['python_version'].split('.')[:2])
    if version < (3, 15) or identity['py_gil_disabled'] != 1:
        raise ValueError('FREE_THREAD_EXECUTOR_REQUIRES_PYTHON315_NOGIL_BUILD')
    if identity['gil_enabled'] is not False:
        raise ValueError('FREE_THREAD_EXECUTOR_REQUIRES_EFFECTIVE_GIL_DISABLED')
    if import_dependencies:
        # An unsupported C extension may re-enable the GIL at import. Load the
        # actual native kernel and numerical dependencies before accepting it.
        import numpy
        import pandas
        from application.analysis_dataset import AnalysisDatasetBuilder
        from application.evidence_context import EvidenceContext
        del numpy, pandas, AnalysisDatasetBuilder, EvidenceContext
        _require_free_threading(import_dependencies=False)


class DailyThreadExecutor:
    """Two bounded threads; every thread owns its guard, work and cache."""

    def __init__(self, workers=2, *, max_packet_bytes=64 * 1024 * 1024,
                 max_inflight_bytes=128 * 1024 * 1024, cache_bytes=16 * 1024 * 1024):
        if type(workers) is not int or workers != 2:
            raise ValueError('DAILY_THREAD_WORKERS_MUST_BE_2')
        if (min(max_packet_bytes, max_inflight_bytes) <= 0
                or max_packet_bytes > max_inflight_bytes or cache_bytes < 0):
            raise ValueError('INVALID_PARALLEL_CAPACITY')
        _require_free_threading()
        self.workers = workers
        self.max_packet_bytes = max_packet_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.cache_bytes = cache_bytes

    def evaluate(self, jobs):
        _require_free_threading()
        guard = ResearchVersionGuard()
        parent_work = current_work()
        expected_version = digest(asdict(guard.versions))
        cache_enabled = bool(parent_work and parent_work.enabled)
        owners, owners_lock, local = [], threading.Lock(), threading.local()

        def initialize():
            assert current_work() is None
            thread_guard = ResearchVersionGuard()
            if digest(asdict(thread_guard.versions)) != expected_version:
                raise ValueError('PARALLEL_SOURCE_VERSION_MISMATCH')
            work = DailyWorkContext(expected_version, enabled=cache_enabled,
                max_bytes=self.cache_bytes, check_version=thread_guard.check)
            local.work = work
            with owners_lock:
                owners.append(work)

        def calculate(encoded):
            from application.analysis_dataset import AnalysisDatasetBuilder
            from application.evidence_context import EvidenceContext
            _require_free_threading(import_dependencies=False)
            # Retain the existing internal trusted-byte boundary. Each task
            # decodes private DataFrames/proofs; parent graphs never escape.
            item, market, watch = pickle.loads(encoded)
            if market.stock_count != 1:
                raise ValueError('PARALLEL_REQUIRES_ONE_STOCK_PACKET')
            probe = DailyPerformance()
            with probe.activate(), local.work.activate(), measure_stage('stock_compute'):
                with EvidenceContext(market) as evidence:
                    entry, exit = AnalysisDatasetBuilder().stock_components(market, 0, watch, evidence)
                _require_free_threading(import_dependencies=False)
            performance = probe.snapshot()
            # Threads share the parent's RSS. The enclosing parent probe owns
            # process high-water memory; do not label it as per-worker RSS.
            performance['process_peak_rss_bytes'] = None
            performance['memory_scope'] = 'SHARED_PROCESS_NOT_PER_THREAD'
            return item, entry, exit, performance

        results, pending, seen = {}, {}, set()
        iterator, waiting, exhausted, inflight_bytes = iter(jobs), None, False, 0
        with measure_stage('stock_thread_pool'):
            pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='native-stock',
                initializer=lambda: Context().run(initialize))
            try:
                while not exhausted or pending or waiting is not None:
                    while len(pending) < self.workers and not exhausted:
                        if waiting is None:
                            if pending and inflight_bytes + self.max_packet_bytes > self.max_inflight_bytes:
                                break
                            try:
                                job = next(iterator)
                            except StopIteration:
                                exhausted = True
                                break
                            item = job[0]
                            if type(item) is not int or item < 0 or item in seen:
                                raise ValueError('DUPLICATE_OR_INVALID_STOCK_JOB')
                            seen.add(item)
                            with measure_stage('stock_packet_encode'):
                                encoded = pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL)
                            if len(encoded) > self.max_packet_bytes:
                                raise ValueError('STOCK_PACKET_CAPACITY_EXCEEDED')
                            waiting = item, encoded
                            del job, encoded
                        item, encoded = waiting
                        if inflight_bytes + len(encoded) > self.max_inflight_bytes:
                            break
                        # Empty submission context also prevents initial pool
                        # threads from retaining caller/request/work context.
                        # A separate fresh task Context can run before submit
                        # returns without re-entering the submission Context.
                        future = Context().run(pool.submit, Context().run, calculate, encoded)
                        pending[future] = item, len(encoded)
                        inflight_bytes += len(encoded)
                        count_work('stock_packets_submitted')
                        count_work('stock_packet_bytes', len(encoded))
                        waiting = None
                        del encoded
                    if pending:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            expected_item, size = pending.pop(future)
                            item, entry, exit, performance = future.result()
                            if item != expected_item or item in results:
                                raise ValueError('STOCK_WORKER_RESULT_BINDING_MISMATCH')
                            results[item] = entry, exit
                            inflight_bytes -= size
                            merge_worker_performance(performance)
                    elif waiting is not None:
                        raise ValueError('STOCK_PACKET_CAPACITY_EXCEEDED')
                if set(results) != set(range(len(results))):
                    raise ValueError('STOCK_JOB_SEQUENCE_INCOMPLETE')
                guard.check()
                _require_free_threading(import_dependencies=False)
            except BaseException:
                for future in pending:
                    future.cancel()
                # Python cannot kill a running thread safely. Wait for the
                # original stock kernel before cleanup; no partial output is
                # returned. Bank's outer process cancellation remains intact.
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
            finally:
                for work in owners:
                    work.close()
        return {item: results[item] for item in range(len(results))}
