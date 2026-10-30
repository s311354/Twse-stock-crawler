"""Bounded safe preprocessing shared within one native analysis job.

No account, EREM, PAE, availability selection or final decision is memoized.
JSON memo returns isolated copies; typed memo only holds immutable FeatureFrame
or frozen FiledFact tuples and compact exact-report binding operands. These
operands never replace checks against a caller's current decision or account.
Opening a new job gives a new version boundary.
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
                 parse_memo=None, check_version=None, validation_cache_bytes=1024 * 1024,
                 validation_cache_entries=4096):
        if (max_bytes < 0 or type(validation_cache_bytes) is not int or validation_cache_bytes < 0
                or type(validation_cache_entries) is not int or validation_cache_entries < 0):
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
        # Successful exact-JSON checks have an independent small budget. Heavy
        # feature/financial LRUs must not evict the whole universe before PAE.
        self.validation_cache_bytes = validation_cache_bytes
        self.validation_cache_entries = validation_cache_entries
        self._validation_tokens = OrderedDict()
        self._validation_bytes = 0

    @contextmanager
    def activate(self):
        if self._closed:
            raise ValueError("DAILY_WORK_CONTEXT_CLOSED")
        self.check_version()
        token = _ACTIVE.set(self)
        try:
            yield self
            self.check_version()
        except BaseException:
            self._validation_tokens.clear()
            self._validation_bytes = 0
            raise
        finally:
            _ACTIVE.reset(token)

    def validation_cached(self, kind, key):
        """Presence of an exact successful token, never an external-source verdict."""
        if self._closed:
            raise ValueError("DAILY_WORK_CONTEXT_CLOSED")
        return self.enabled and (kind, self.version, key) in self._validation_tokens

    def validate_once(self, kind, key, validate):
        """Memo successful pure validation only; no input/result graphs kept.

        Callers must bind actual immutable bytes and their validation policy.
        Account/source-file checks are not pure and cannot use this port.
        """
        if self._closed:
            raise ValueError("DAILY_WORK_CONTEXT_CLOSED")
        token = kind, self.version, key
        if self.enabled and token in self._validation_tokens:
            self._validation_tokens.move_to_end(token)
            count_work(kind + "_cache_hits")
            return
        count_work(kind + "_cache_misses")
        validate()  # Exceptions never enter the cache.
        weight = 256 + len(str(token).encode())
        if (not self.enabled or not self.validation_cache_entries
                or weight > self.validation_cache_bytes):
            return
        while self._validation_tokens and (len(self._validation_tokens) >= self.validation_cache_entries
                or self._validation_bytes + weight > self.validation_cache_bytes):
            _, removed = self._validation_tokens.popitem(last=False)
            self._validation_bytes -= removed
            count_work("daily_validation_evictions")
        self._validation_tokens[token] = weight
        self._validation_bytes += weight

    def memo_json(self, kind, key, compute):
        if self._closed:
            raise ValueError("DAILY_WORK_CONTEXT_CLOSED")
        before = self.json_cache.counts[kind + "_hit"]
        value = self.json_cache.memo(kind, (self.version, key), compute)
        count_work(kind + ("_cache_hits" if self.json_cache.counts[kind + "_hit"] > before else "_cache_misses"))
        return value

    def memo_immutable(self, kind, key, compute, size):
        if self._closed:
            raise ValueError("DAILY_WORK_CONTEXT_CLOSED")
        token = kind, self.version, key
        if self.enabled and token in self._values:
            self._values.move_to_end(token)
            count_work(kind + "_cache_hits")
            return self._values[token][0]
        count_work(kind + "_cache_misses")
        result = compute()
        capacity = self.max_bytes // 2
        overhead = 256 + len(str(token).encode())
        if not self.enabled or result is None or overhead > capacity:
            return result
        # Include entry/key overhead in addition to the value graph. The old
        # max() undercounted a large value paired with a substantial source key.
        weight = overhead + max(0, size(result))
        if weight <= capacity:
            while self._values and self._bytes + weight > capacity:
                _, (_, removed) = self._values.popitem(last=False)
                self._bytes -= removed
                count_work("daily_work_evictions")
            self._values[token] = result, weight
            self._bytes += weight
        return result

    def raw_parse(self, sha, compute, *, parser_version):
        if self._closed:
            raise ValueError("DAILY_WORK_CONTEXT_CLOSED")
        if not self.enabled or self.parse_memo is None:
            return compute()
        # The caller binds the actual parser/mapping/schema implementation.
        # Unrelated renderer, model or policy changes must not discard the raw
        # parse. The full job version guard remains active independently.
        return self.parse_memo.memo("ixbrl", ("daily-raw-parse/v2:" + parser_version, sha), compute)

    def raw_companion(self, sha, compute, *, derivation_version):
        """Cache pure source-bound financial companions, never cutoff selection.

        The caller binds all original metadata and derivation dependencies in
        its identity. Public results are detached JSON; availability, signed
        report selection, account inputs and final gates remain outside here.
        """
        if self._closed:
            raise ValueError("DAILY_WORK_CONTEXT_CLOSED")
        if not self.enabled or self.parse_memo is None:
            return compute()
        rows = self.parse_memo.memo("financial_companion",
            ("daily-financial-companion/v1:" + derivation_version, sha), lambda: [compute()])
        return rows[0]

    def close(self):
        self._values.clear()
        self._bytes = 0
        self.json_cache.clear()
        self._validation_tokens.clear()
        self._validation_bytes = 0
        self._closed = True

    @property
    def stats(self):
        return dict(immutable_bytes=self._bytes, immutable_entries=len(self._values),
                    json=self.json_cache.stats, max_bytes=self.max_bytes,
                    validation_bytes=self._validation_bytes,
                    validation_entries=len(self._validation_tokens),
                    validation_max_bytes=self.validation_cache_bytes,
                    validation_max_entries=self.validation_cache_entries)
