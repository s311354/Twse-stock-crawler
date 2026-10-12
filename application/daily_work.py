"""Bounded safe preprocessing shared within one native analysis job.

No account, EREM, PAE, availability selection or final decision is memoized.
JSON memo returns isolated copies; typed memo only holds immutable FeatureFrame
or frozen FiledFact tuples. Opening a new job gives a new version boundary.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

from application.daily_performance import count_work
from application.replay_cache import ReplayCache

_ACTIVE = ContextVar("native_daily_work", default=None)


def current_work():
    return _ACTIVE.get()


class DailyWorkContext:
    def __init__(self, version, *, enabled=True, max_bytes=64 * 1024 * 1024,
                 parse_memo=None, check_version=None):
        if max_bytes < 0:
            raise ValueError("INVALID_CACHE_CAPACITY")
        self.version = version
        self.enabled = enabled
        self.max_bytes = max_bytes
        # Native reports contain JSON strings too. Keep insertion order so a
        # memo hit is byte-equivalent, not just dictionary-equivalent.
        self.json_cache = ReplayCache(max_bytes // 2, enabled=enabled, preserve_order=True)
        self.parse_memo = parse_memo
        self.check_version = check_version or (lambda: None)
        self._values = OrderedDict()
        self._bytes = 0
        self._closed = False

    @contextmanager
    def activate(self):
        if self._closed:
            raise ValueError("DAILY_WORK_CONTEXT_CLOSED")
        self.check_version()
        token = _ACTIVE.set(self)
        try:
            yield self
            self.check_version()
        finally:
            _ACTIVE.reset(token)

    def memo_json(self, kind, key, compute):
        before = self.json_cache.counts[kind + "_hit"]
        value = self.json_cache.memo(kind, (self.version, key), compute)
        count_work(kind + ("_cache_hits" if self.json_cache.counts[kind + "_hit"] > before else "_cache_misses"))
        return value

    def memo_immutable(self, kind, key, compute, size):
        token = kind, self.version, key
        if self.enabled and token in self._values:
            self._values.move_to_end(token)
            count_work(kind + "_cache_hits")
            return self._values[token][0]
        count_work(kind + "_cache_misses")
        result = compute()
        weight = max(256 + len(str(token).encode()), size(result))
        capacity = self.max_bytes // 2
        if self.enabled and result is not None and weight <= capacity:
            while self._values and self._bytes + weight > capacity:
                _, (_, removed) = self._values.popitem(last=False)
                self._bytes -= removed
                count_work("daily_work_evictions")
            self._values[token] = result, weight
            self._bytes += weight
        return result

    def raw_parse(self, sha, compute, *, parser_version):
        if not self.enabled or self.parse_memo is None:
            return compute()
        return self.parse_memo.memo("ixbrl", ("daily:" + self.version + ":" + parser_version, sha), compute)

    def close(self):
        self._values.clear()
        self._bytes = 0
        self.json_cache.clear()
        self._closed = True

    @property
    def stats(self):
        return dict(immutable_bytes=self._bytes, immutable_entries=len(self._values),
                    json=self.json_cache.stats, max_bytes=self.max_bytes)
