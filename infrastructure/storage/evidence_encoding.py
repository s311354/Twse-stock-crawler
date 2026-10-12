"""Bounded transaction-local structural interning, not a validation cache.

Tokens describe exact JSON structure; they are never evidence identities.
Children are visited every time, so mutable inputs cannot produce stale hits.
Evicted tokens are never recycled. Only canonical bytes/hashes are memoized,
not mutable nodes, restored reports, availability or model decisions.
"""
from collections import OrderedDict
import math
import sys

from application.daily_performance import count_work


class EncodingMemo:
    MAX_ENTRIES = 16384

    def __init__(self, capacity, prefix):
        self.capacity, self.prefix = capacity, prefix
        self.values = OrderedDict()
        self.bytes = self.sequence = 0

    def token(self):
        self.sequence += 1
        return self.sequence

    def get(self, key):
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
            count_work(self.prefix + "_encoding_hits")
            return value[0]
        count_work(self.prefix + "_encoding_misses")
        return None

    def put(self, key, value, weight):
        # Include key/tuple/int/OrderedDict overhead conservatively, plus the
        # encoded bytes and scalar strings owned by an entry.
        weight += 512 + (len(value[2]) if value[2] is not None else 0)
        if weight <= self.capacity:
            while self.values and (self.bytes + weight > self.capacity or len(self.values) >= self.MAX_ENTRIES):
                _, (_, removed) = self.values.popitem(last=False)
                self.bytes -= removed
                count_work(self.prefix + "_encoding_evictions")
            self.values[key] = value, weight
            self.bytes += weight

    def scalar(self, value):
        kind = type(value)
        if kind is float:
            if not math.isfinite(value):
                raise ValueError("Out of range float values are not JSON compliant")
            # Python equality conflates 0.0/-0.0 and int/float/bool; JSON does not.
            return ("value", kind, value.hex())
        elif kind in (str, int, bool, type(None)):
            return ("value", kind, value)
        else:
            # Do not intern arbitrary objects or change the native JSON encoder's
            # treatment of scalar subclasses / unsupported objects.
            return self.token()

    @staticmethod
    def scalar_weight(token):
        # Container tokens are integers; primitive tokens carry their exact
        # typed value. Account retained scalar keys without a second LRU lookup
        # or instrumentation event for each of millions of scalar leaves.
        return sys.getsizeof(token) + sys.getsizeof(token[2]) if isinstance(token, tuple) else 32

    def clear(self):
        self.values.clear()
        self.bytes = 0
