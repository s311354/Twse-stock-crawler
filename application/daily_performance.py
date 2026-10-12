"""Work-scoped diagnostics, never part of model inputs or result identities."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import sys
import time
try:
    import resource
except ImportError:  # Windows: do not make optional RSS telemetry a runtime gate.
    resource = None


_ACTIVE = ContextVar("daily_performance", default=None)
_PATH = ContextVar("daily_performance_path", default=())


class DailyPerformance:
    def __init__(self, *, clock=time.perf_counter, stage_callback=None):
        self.clock = clock
        self.seconds = Counter()
        self.calls = Counter()
        self.counters = Counter()
        self.stage_callback = stage_callback

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        path_token = _PATH.set(())
        try:
            yield self
        finally:
            _PATH.reset(path_token)
            _ACTIVE.reset(token)

    def notify(self, event):
        if self.stage_callback is not None:
            try:
                self.stage_callback(event)
            except Exception:
                # Observations cannot change research, gates or transactions.
                self.counters["performance_callback_errors"] += 1

    def snapshot(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
        return {"stages": {name: {"seconds": self.seconds[name], "calls": self.calls[name]}
                           for name in sorted(self.seconds)},
                "counters": dict(sorted(self.counters.items())),
                "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024) if rss is not None else None,
                "memory_scope": "PROCESS_HIGH_WATER_MARK_NOT_PER_RUN"}


def merge_worker_performance(snapshot):
    """Worker CPU-stage seconds are sums, explicitly NOT parent wall time."""
    probe = _ACTIVE.get()
    if probe is None:
        return
    for name, value in snapshot["stages"].items():
        key = "worker_sum." + name
        probe.seconds[key] += value["seconds"]
        probe.calls[key] += value["calls"]
    for name, value in snapshot["counters"].items():
        probe.counters[name] += value
    peak = snapshot.get("process_peak_rss_bytes")
    if peak is not None:
        probe.counters["worker_max_process_peak_rss_bytes"] = max(
            probe.counters["worker_max_process_peak_rss_bytes"], peak)


@contextmanager
def measure_stage(name):
    probe = _ACTIVE.get()
    if probe is None:
        yield
        return
    started = probe.clock()
    path = _PATH.get()
    active = path + ((name, started),)
    token = _PATH.set(active)
    probe.notify(dict(event="started", name=name, path=active, at=started))
    status = "complete"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        ended = probe.clock()
        probe.seconds[name] += max(0., ended - started)
        probe.calls[name] += 1
        _PATH.reset(token)
        probe.notify(dict(event="finished", name=name, path=path, at=ended,
                          status=status, seconds=probe.seconds[name], calls=probe.calls[name]))


def count_work(name, amount=1):
    probe = _ACTIVE.get()
    if probe is not None:
        probe.counters[name] += amount
