"""Bounded, context-local immutable JSON cache. Never caches account decisions."""
from collections import OrderedDict, Counter
import json

from application.replay_contracts import canonical_json


class ReplayCache:
    def __init__(self, max_bytes=32 * 1024 * 1024, *, enabled=True, preserve_order=False):
        if max_bytes < 0:
            raise ValueError('INVALID_CACHE_CAPACITY')
        self.max_bytes, self.enabled = max_bytes, enabled
        self.preserve_order = preserve_order
        self._values = OrderedDict()
        self.bytes = 0
        self.counts = Counter()

    def memo(self, kind, key, compute):
        token = (kind, key)
        if self.enabled and token in self._values:
            self.counts[kind + '_hit'] += 1
            self._values.move_to_end(token)
            return json.loads(self._values[token])
        self.counts[kind + '_miss'] += 1
        value = compute()
        if self.enabled:
            encoded = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                       if self.preserve_order else canonical_json(value)).encode()
            size = len(encoded) + len(str(token).encode())
            if size <= self.max_bytes:
                while self._values and self.bytes + size > self.max_bytes:
                    old, data = self._values.popitem(last=False)
                    self.bytes -= len(data) + len(str(old).encode())
                    self.counts['evictions'] += 1
                self._values[token] = encoded
                self.bytes += size
        return value

    def clear(self):
        self._values.clear()
        self.bytes = 0

    @property
    def stats(self):
        return dict(self.counts, bytes=self.bytes, entries=len(self._values), max_bytes=self.max_bytes)
