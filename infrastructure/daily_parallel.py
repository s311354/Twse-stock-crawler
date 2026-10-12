"""Spawn-based, bounded native stock calculation; no worker I/O or PAE writes.

Only internally generated trusted packets use pickle (never an external archive).
Payload bytes and task count are bounded. Workers run the existing serial kernel
with isolated evidence and work caches. A failure aborts the entire calculation;
there is no partial-success or automatic serial/fallback-buy path.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
import multiprocessing
import pickle

from application.daily_performance import DailyPerformance, count_work, measure_stage, merge_worker_performance
from application.daily_work import DailyWorkContext, current_work
from application.replay_contracts import digest
from infrastructure.research_objects import ResearchVersionGuard

_GUARD = None
_WORK = None


def _initialize(expected_version, enabled, cache_bytes):
    global _GUARD, _WORK
    _GUARD = ResearchVersionGuard()
    actual = digest(asdict(_GUARD.versions))
    if actual != expected_version:
        raise ValueError("PARALLEL_SOURCE_VERSION_MISMATCH")
    _WORK = DailyWorkContext(actual, enabled=enabled, max_bytes=cache_bytes,
                             check_version=_GUARD.check)


def _calculate(encoded):
    from application.analysis_dataset import AnalysisDatasetBuilder
    from application.evidence_context import EvidenceContext
    item, market, watch = pickle.loads(encoded)
    if market.stock_count != 1:
        raise ValueError("PARALLEL_REQUIRES_ONE_STOCK_PACKET")
    probe = DailyPerformance()
    with probe.activate(), _WORK.activate(), measure_stage("stock_compute"):
        with EvidenceContext(market) as evidence:
            entry, exit = AnalysisDatasetBuilder().stock_components(market, 0, watch, evidence)
    return item, entry, exit, probe.snapshot()


class DailyProcessExecutor:
    """Per-analysis pool: bounded stock inputs, deterministic original order."""

    def __init__(self, workers=2, *, max_packet_bytes=64 * 1024 * 1024,
                 max_inflight_bytes=128 * 1024 * 1024, cache_bytes=16 * 1024 * 1024):
        if type(workers) is not int or not 2 <= workers <= 4:
            raise ValueError("DAILY_WORKERS_MUST_BE_2_TO_4")
        if min(max_packet_bytes, max_inflight_bytes) <= 0 or max_packet_bytes > max_inflight_bytes or cache_bytes < 0:
            raise ValueError("INVALID_PARALLEL_CAPACITY")
        self.workers = workers
        self.max_packet_bytes = max_packet_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.cache_bytes = cache_bytes

    def evaluate(self, jobs):
        guard = ResearchVersionGuard()
        work = current_work()
        version = digest(asdict(guard.versions))
        results, pending, seen = {}, {}, set()
        iterator = iter(jobs)
        waiting = None
        exhausted = False
        inflight_bytes = 0
        with measure_stage("stock_process_pool"):
            pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_initialize, initargs=(version, bool(work and work.enabled), self.cache_bytes))
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
                                raise ValueError("DUPLICATE_OR_INVALID_STOCK_JOB")
                            seen.add(item)
                            with measure_stage("stock_packet_encode"):
                                encoded = pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL)
                            if len(encoded) > self.max_packet_bytes:
                                raise ValueError("STOCK_PACKET_CAPACITY_EXCEEDED")
                            waiting = item, encoded
                            del job, encoded
                        item, encoded = waiting
                        if inflight_bytes + len(encoded) > self.max_inflight_bytes:
                            break
                        future = pool.submit(_calculate, encoded)
                        pending[future] = item, len(encoded)
                        inflight_bytes += len(encoded)
                        count_work("stock_packets_submitted")
                        count_work("stock_packet_bytes", len(encoded))
                        waiting = None
                        del encoded
                    if pending:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            expected_item, size = pending.pop(future)
                            item, entry, exit, performance = future.result()
                            if item != expected_item or item in results:
                                raise ValueError("STOCK_WORKER_RESULT_BINDING_MISMATCH")
                            results[item] = entry, exit
                            inflight_bytes -= size
                            merge_worker_performance(performance)
                    elif waiting is not None:
                        raise ValueError("STOCK_PACKET_CAPACITY_EXCEEDED")
                if set(results) != set(range(len(results))):
                    raise ValueError("STOCK_JOB_SEQUENCE_INCOMPLETE")
                guard.check()
            except BaseException:
                for future in pending:
                    future.cancel()
                # Do not publish partial results or leave workers running after
                # a source change/error/interruption. Python 3.14 supports an
                # immediate terminate; older runtimes use a waiting shutdown.
                terminate = getattr(pool, "terminate_workers", None)
                if callable(terminate):
                    terminate()
                else:
                    pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
        return results
