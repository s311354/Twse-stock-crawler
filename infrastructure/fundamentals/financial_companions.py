"""Bounded, source-bound preprocessing for immutable financial companions.

Only a repository which has just verified an original may call this boundary.
It does not select filings, infer historical publication, select signed PDFs,
evaluate Quality, or cache an account/model decision. Every collecting call
still performs cutoff/version selection and checks its original before reuse.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import sys

import pandas as pd

from application.daily_performance import count_work, measure_stage
from application.daily_work import current_work
from application.evidence_context import ReadOnlyEvidence
from application.replay_contracts import digest
from domain.fundamental_derivation import FiledFinancials
from domain.les_v7.financial_sources import (
    _financial_supplement_from_row, ixbrl_record_r2,
)
from infrastructure.fundamentals.parse_identity import (
    semantic_raw_parse_version, source_fingerprint,
)


COMPANION_SCHEMA = "verified-original-financial-companion/v1"
_SOURCE_METADATA = tuple(field.name for field in fields(FiledFinancials) if field.name != "facts")
# Actual implementations, not only hand-maintained labels. An unrelated HTML
# renderer change does not invalidate source mapping; mapping/derivation or
# source-verification changes do. The run's complete version guard is separate.
_SEMANTIC_SOURCES = (
    ("infrastructure/fundamentals/financial_companions.py", ()),
    ("domain/fundamental_derivation.py", ()),
    ("domain/financial_taxonomy.py", ()),
    ("domain/strategies/low_entry_v6_filings.py", ()),
    ("domain/strategies/low_entry_v6_ttm.py", ()),
    ("domain/strategies/low_entry_v6_quality_evidence.py", ()),
    ("domain/les_v7/financial_sources.py", ()),
    ("domain/les_v7/contracts.py", ()),
    ("domain/text_normalization.py", ()),
)


def companion_semantics():
    """Precise source/adapter/runtime identity; independent of a research run."""
    from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
    from infrastructure.fundamentals.ixbrl_parser import IXBRL_PARSER_VERSION
    work = current_work()
    def calculate():
        return digest(dict(schema=COMPANION_SCHEMA,
            source_sha256=source_fingerprint(_SEMANTIC_SOURCES),
            raw_parser=semantic_raw_parse_version(V6_FILING_SCHEMA_VERSION + ":" + IXBRL_PARSER_VERSION),
            python=list(sys.version_info), pandas=pd.__version__))
    # One semantic fingerprint per guarded work. Closing and reopening a work
    # rereads stat-bound implementation identities; the guard still detects
    # source drift during an active job rather than silently mixing versions.
    return (work.memo_immutable("financial_companion_semantics", COMPANION_SCHEMA,
                               calculate, sys.getsizeof) if work is not None else calculate())


def _retained_size(value):
    pending, seen, size = [value], set(), 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        size += sys.getsizeof(item)
        if isinstance(item, Mapping):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, tuple):
            pending.extend(item)
    return size


def _detach(value):
    if isinstance(value, Mapping):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_detach(item) for item in value]
    return value


def _tuple_paths(value, path=()):
    """Retain the public Python contract across a lossless JSON disk memo.

    FiledFact audit contexts contain tuples, whereas financial field contexts
    intentionally contain JSON lists. Remember container types rather than
    making warm cached calls silently differ from the uncached reference.
    """
    result = [path] if type(value) is tuple else []
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.extend(_tuple_paths(item, (*path, key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.extend(_tuple_paths(item, (*path, index)))
    return result


def _public_copy(packet):
    value = _detach(packet["value"])
    for path in sorted(packet["tuple_paths"], key=len, reverse=True):
        parent = value
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = tuple(parent[path[-1]])
    return value


def verified_source_companions(source, *, symbol, market="TWSE"):
    """Return detached legacy-compatible audit/supplement dictionaries.

    The cache holds one recursively immutable subtree, never a consumer's
    mutable projection. Exact source metadata (including availability, scope,
    revision, URL and taxonomy) participates in the key alongside the verified
    original hash and actual parser/mapping/derivation implementations. A
    different cutoff must still reselect sources, but cannot change this pure
    same-original mapping. Signed-report proofs remain outside this cache.
    """
    def build():
        with measure_stage("financial_companion_build"):
            count_work("financial_companion_builds")
            row = ixbrl_record_r2(symbol, source, market=market, original_verified=True)
            return dict(financial_audit=row["audit_evidence"],
                financial_supplement=_financial_supplement_from_row(row, original_verified=True))
    work = current_work()
    if work is None or not work.enabled:
        return build()
    version = companion_semantics()
    key = digest(dict(schema=COMPANION_SCHEMA, symbol=symbol, market=market,
                      original_verified=True,
                      metadata={name: getattr(source, name) for name in _SOURCE_METADATA}))
    def freeze():
        def prepare():
            value = build()
            return dict(value=value, tuple_paths=_tuple_paths(value))
        packet = work.raw_companion(key, prepare, derivation_version=version)
        return ReadOnlyEvidence(packet)
    frozen = work.memo_immutable("financial_companion", (version, key), freeze, _retained_size)
    with measure_stage("financial_companion_detach"):
        count_work("financial_companion_detaches")
        return _public_copy(frozen)
