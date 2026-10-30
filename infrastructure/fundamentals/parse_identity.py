"""Actual-source identities for rebuildable, raw-only iXBRL preprocessing.

This is deliberately narrower than the research run identity. The ordinary
ResearchVersionGuard still rejects any code/model/parameter drift during a job.
Original integrity and cutoff/revision selection are not cacheable outcomes.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys


RAW_PARSE_IDENTITY_SCHEMA = "ixbrl-raw-semantics/v2"
_ROOT = Path(__file__).resolve().parents[2]
# Raw parser dependencies, including financial mapping/schema metadata and the
# precise JSON-to-FiledFact adapter. Scope large derivation/query modules to
# their actual raw parsing definitions, not their unrelated financial rules.
_RAW_SOURCES = (
    ("infrastructure/fundamentals/parse_identity.py", ()),
    ("infrastructure/fundamentals/ixbrl_parser.py", ()),
    ("infrastructure/fundamentals/ixbrl_decoding.py", ()),
    ("infrastructure/fundamentals/ixbrl_units.py", ()),
    ("domain/financial_taxonomy.py", ()),
    ("domain/text_normalization.py", ()),
    ("domain/fundamental_models.py", ("NormalizedFact", "XbrlContext")),
    ("domain/fundamental_derivation.py", ("FiledFact",)),
    ("infrastructure/fundamentals/mops_queries.py", ("_read_raw_normalized_facts",)),
)


@lru_cache(maxsize=32)
def _fingerprint_verified_sources(entries):
    sources = []
    for name, path, signature, definitions in entries:
        content = Path(path).read_bytes()
        stat = Path(path).stat()
        if signature != (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size):
            raise ValueError("RAW_PARSE_IMPLEMENTATION_CHANGED_DURING_FINGERPRINT")
        if definitions:
            module = ast.parse(content, filename=name)
            selected = {node.name: node for node in module.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and node.name in definitions}
            if set(selected) != set(definitions):
                raise ValueError("RAW_PARSE_IMPLEMENTATION_DEFINITION_MISSING")
            # Ignore line offsets: an unrelated function added above a selected
            # adapter must not invalidate its identical implementation.
            content = json.dumps([ast.dump(selected[key], include_attributes=False)
                                  for key in definitions], separators=(",", ":")).encode()
        sources.append((name, sha256(content).hexdigest()))
    return sha256(json.dumps(sources, separators=(",", ":")).encode()).hexdigest()


def source_fingerprint(sources, *, root=None):
    """Fingerprint explicit source dependencies, with bounded stat-keyed reuse.

    Each entry is (repository-relative path, tuple of top-level definitions).
    An empty definition tuple hashes the complete module. Source paths are
    excluded from identity, so an identical isolated checkout shares semantics.
    """
    base = Path(root) if root is not None else _ROOT
    entries = []
    for name, definitions in sources:
        path = (base / name).resolve()
        stat = path.stat()
        entries.append((name, str(path), (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size), tuple(definitions)))
    return _fingerprint_verified_sources(tuple(entries))


def semantic_raw_parse_version(parser_version):
    """Bind explicit labels, parser code, units/schema and interpreter semantics."""
    payload = dict(schema=RAW_PARSE_IDENTITY_SCHEMA, caller_version=parser_version,
                   source_sha256=source_fingerprint(_RAW_SOURCES),
                   python=list(sys.version_info), implementation=sys.implementation.name)
    return RAW_PARSE_IDENTITY_SCHEMA + ":" + sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
