"""Bounded, context-local reuse of already keyed native reference preparation.

This cache is an execution optimization, not evidence or a model artifact. The
caller owns the exact source/cutoff/reference-policy identity. No bars, source
packets, report bodies, factories or process-global prepared values are kept.
"""
from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, fields
import json
from math import isfinite
from numbers import Real
from typing import Callable, Hashable, TypeVar

from domain.les_v7.contracts import ReferencePlan, canonical, identity, timestamp

MAX_REFERENCE_ENTRIES = 4096
MAX_REFERENCE_HISTORY_BYTES = 32 * 1024 * 1024
_CURRENT: ContextVar[NativeReferenceMemo | None] = ContextVar("native_v7_reference_memo", default=None)
_FIELDS = {"features", "feature_audit", "context", "context_matches", "plan", "vector", "feature_gap", "evidence_id"}
_AUDIT_FIELDS = {"profile", "status", "fractal", "fractal_history", "zero_range", "vwap_method", "cutoff_session", "execution_feature_audit", "fr3_reference"}
_MISSING = object()
T = TypeVar("T")


def _numeric(value):
    return value is None or isinstance(value, Real) and isfinite(value)


def _small_tree(value, depth=0):
    """Permit bounded native estimator diagnostics, never arbitrary objects."""
    if depth > 8:
        return False
    if value is None or isinstance(value, Real):
        return _numeric(value)
    if isinstance(value, str):
        return len(value) <= 1024
    if isinstance(value, dict):
        return (len(value) <= 128 and all(isinstance(key, str) and len(key) <= 128
                and _small_tree(item, depth + 1) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return len(value) <= 64 and all(_small_tree(item, depth + 1) for item in value)
    return False


def _cacheable(value):
    """Unsupported future prepared shapes simply continue without caching."""
    if not isinstance(value, dict) or set(value) != _FIELDS:
        return False
    features, audit, vector = value["features"], value["feature_audit"], value["vector"]
    if (not isinstance(features, dict) or len(features) > 256
            or not all(isinstance(key, str) and len(key) <= 128 and _numeric(item)
                       for key, item in features.items())):
        return False
    if not isinstance(audit, dict) or not set(audit) <= _AUDIT_FIELDS or not _small_tree(audit):
        return False
    if (not isinstance(value["context"], str) or len(value["context"]) > 128
            or not isinstance(value["context_matches"], (tuple, list))
            or len(value["context_matches"]) > 8
            or any(not isinstance(item, str) or len(item) > 128 for item in value["context_matches"])):
        return False
    plan = value["plan"]
    if plan is not None:
        plan_type = ReferencePlan
        if getattr(plan, "policy_version", None) == "FR3_SOURCE_BOUND_LIMIT_H20_R1":
            from domain.les_v7.fr3_plan import FR3ReferencePlan
            plan_type = FR3ReferencePlan
        elif getattr(plan, "policy_version", None) == "FR3_SOURCE_BOUND_LIMIT_H20_L2":
            from domain.les_v7.fr3_l2 import FR3L2ReferencePlan
            plan_type = FR3L2ReferencePlan
        if (type(plan) is not plan_type
                or set(vars(plan)) != {field.name for field in fields(plan_type)}
                or not _small_tree(vars(plan))):
            return False
    if vector is not None and (not isinstance(vector, (tuple, list)) or len(vector) > 64
                              or any(not _numeric(item) for item in vector)):
        return False
    return (isinstance(value["evidence_id"], str) and len(value["evidence_id"]) == 64
            and (value["feature_gap"] is None or isinstance(value["feature_gap"], str)
                 and len(value["feature_gap"]) <= 256))


class NativeReferenceMemo:
    """One job or shard's LRU; nested scopes restore the enclosing cache."""

    def __init__(self, max_entries=MAX_REFERENCE_ENTRIES):
        if type(max_entries) is not int or not 1 <= max_entries <= MAX_REFERENCE_ENTRIES:
            raise ValueError("REFERENCE_MEMO_ENTRY_BOUND_REQUIRED")
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._aliases = OrderedDict()
        self._history_sizes = {}
        self._history_bytes = 0
        self.fast_hits = 0
        self._token = None
        self._active = False
        self.hits = self.misses = self.stores = self.evictions = 0

    @property
    def stats(self):
        return dict(hits=self.hits, misses=self.misses, stores=self.stores,
                    evictions=self.evictions, current_entries=len(self._entries),
                    max_entries=self.max_entries, fast_hits=self.fast_hits,
                    history_bytes=self._history_bytes)

    def __enter__(self):
        if self._active:
            raise RuntimeError("REFERENCE_MEMO_CONTEXT_ALREADY_ENTERED")
        self._entries.clear()
        self._aliases.clear()
        self._history_sizes.clear()
        self._history_bytes = self.fast_hits = 0
        self.hits = self.misses = self.stores = self.evictions = 0
        self._token = _CURRENT.set(self)
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._entries.clear()
            self._aliases.clear()
            self._history_sizes.clear()
            self._history_bytes = 0
        finally:
            self._active = False
            token, self._token = self._token, None
            if token is not None:
                _CURRENT.reset(token)
        return False

    def _resolve(self, key: Hashable, factory: Callable[[], T]) -> T:
        stored = self._entries.get(key, _MISSING)
        if stored is not _MISSING:
            self.hits += 1
            self._entries.move_to_end(key)
            return deepcopy(stored)
        self.misses += 1
        # A failed/None computation is never inserted. The closure stays local
        # to this call, so its history/source input cannot outlive preparation.
        value = factory()
        if not _cacheable(value):
            return value
        self._entries[key] = deepcopy(value)
        self.stores += 1
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1
        return deepcopy(self._entries[key])

    def _reference_alias(self, alias, config_id, factory):
        key = self._aliases.get(alias, _MISSING)
        if key is not _MISSING:
            stored = self._entries.get(key, _MISSING)
            if stored is not _MISSING:
                self.hits += 1
                self.fast_hits += 1
                self._entries.move_to_end(key)
                self._aliases.move_to_end(alias)
                return deepcopy(stored)
            del self._aliases[alias]
        value = factory()
        # Only point to a value already admitted by the existing bounded
        # numeric cache. No frame, closure, source packet or second result is
        # retained here; unsupported future shapes simply miss as before.
        if isinstance(value, dict):
            key = value.get('evidence_id'), config_id
            if key in self._entries:
                self._aliases[alias] = key
                self._aliases.move_to_end(alias)
                if len(self._aliases) > self.max_entries:
                    self._aliases.popitem(last=False)
        return value


@dataclass(frozen=True, slots=True)
class _BoundReferenceHistory:
    """An owned JSON snapshot; its private frame never escapes its closure."""

    _source_json: str
    _prepare: Callable

    def __call__(self, symbol, market, cutoff, config, evidence_ids, *, session_evidence=None):
        return self._prepare(symbol, market, cutoff, config, evidence_ids,
            **(dict(session_evidence=session_evidence) if session_evidence is not None else {}))


def _bind_reference_history(bars, *, selection_cutoff=None):
    """Internal batch adapter, never a caller-supplied prepared-result port.

    Native builders supply the original JSON bar list once per history. Its
    detached snapshot has a content identity, unlike a mutable DataFrame id.
    First misses still call the complete public selector/preparer; subsequent
    identical references may reuse only its admitted detached numeric result.
    The source-byte budget is an optimization bound, never a training gate.
    """
    memo = _CURRENT.get()
    if memo is None or not memo._active:
        return None
    # Native source normalization emits JSON numbers/nulls. Arbitrary public
    # JSON may instead contain numeric strings: converting an earlier prefix
    # can infer integer dtype while converting the full evaluation history
    # first can infer float dtype. Those have distinct canonical evidence
    # bytes, so leave such histories on the unchanged public preparation path.
    numeric = {'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Trades', 'Bid', 'Ask', 'TAIEX Close'}
    if (not isinstance(bars, list) or any(not isinstance(row, dict) or any(
            key in numeric and value is not None and type(value) not in {int, float}
            for key, value in row.items()) for row in bars)):
        return None
    source_json = canonical(bars)
    size = len(source_json.encode('utf-8'))
    source_id = identity(bars)
    if source_id not in memo._history_sizes:
        if memo._history_bytes + size > MAX_REFERENCE_HISTORY_BYTES:
            return None
        memo._history_sizes[source_id] = size
        memo._history_bytes += size
    import pandas as pd
    from domain.les_v7.features import select_bars
    frame = pd.DataFrame(json.loads(source_json))
    if frame.empty:
        frame = frame.reindex(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Available At', 'Complete'])
    upper = timestamp(selection_cutoff) if selection_cutoff is not None else None
    if upper is not None:
        frame = select_bars(frame, selection_cutoff)
    def prepare(symbol, market, cutoff, config, evidence_ids, *, session_evidence=None):
        from domain.les_v7.training import prepare_reference, reference_config_id
        normalized = timestamp(cutoff)
        config_id = reference_config_id(config)
        # A post-evaluation event must retain the original selected-history
        # ceiling. Earlier events are identical to the builder's raw prefix.
        ceiling = upper.isoformat() if upper is not None and normalized > upper else None
        alias = (source_id, symbol, market, normalized.isoformat(), ceiling,
                 config_id, identity(evidence_ids))
        options = dict(session_evidence=session_evidence) if session_evidence is not None else {}
        if options:
            alias = (*alias, identity(session_evidence))
        factory = lambda: prepare_reference(symbol, market, frame, cutoff, config, evidence_ids, **options)
        if _CURRENT.get() is not memo or not memo._active:
            return factory()
        return memo._reference_alias(alias, config_id, factory)
    return _BoundReferenceHistory(source_json, prepare)


def memoize_reference(key: Hashable, factory: Callable[[], T]) -> T:
    memo = _CURRENT.get()
    if memo is None or not memo._active:
        return factory()
    return memo._resolve(key, factory)
