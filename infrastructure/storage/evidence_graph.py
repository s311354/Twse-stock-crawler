"""Lossless immutable JSON nodes shared by daily reports and research archives.

The existing report node encoding/table is retained for compatibility. Only
content storage is shared; availability, versions and model gates stay native.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from itertools import islice
import json
from pathlib import Path
import sqlite3
import sys
import zlib

from application.daily_performance import count_work, measure_stage
from application.replay_contracts import canonical_json, digest
from infrastructure.storage.evidence_encoding import EncodingMemo

SCHEMA = "shared-evidence-reference/v1"
MAX_NODE_BYTES = 8 * 1024 * 1024
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_CACHE_BYTES = 64 * 1024 * 1024
MAX_BATCH_NODES = 128
MAX_WRITE_BATCH_NODES = 1024
MAX_COMPRESSED_NODE_BYTES = MAX_NODE_BYTES + 8192


def object_id(value, prefix="EVIDENCE"):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"INVALID_{prefix}_ID")
    return value


def _sha(content):
    return hashlib.sha256(content).hexdigest()


class EvidenceGraphWriter:
    error_prefix = "EVIDENCE"
    counter_prefix = "evidence"

    def __init__(self, path, *, encoding_cache_bytes=8 * 1024 * 1024,
                 batch_nodes=MAX_WRITE_BATCH_NODES, ordered_lookup=True):
        if type(encoding_cache_bytes) is not int or not 0 <= encoding_cache_bytes <= MAX_CACHE_BYTES:
            raise ValueError(f"INVALID_{self.error_prefix}_ENCODING_CACHE_BYTES")
        if type(batch_nodes) is not int or not 1 <= batch_nodes <= MAX_WRITE_BATCH_NODES:
            raise ValueError(f"INVALID_{self.error_prefix}_WRITE_BATCH_NODES")
        if type(ordered_lookup) is not bool:
            raise ValueError(f"INVALID_{self.error_prefix}_ORDERED_LOOKUP")
        self.path = Path(path)
        self.connection = None
        self._seen = OrderedDict()
        self._pending = {}
        self._pending_bytes = 0
        self.stats = dict(nodes_written=0, nodes_reused=0, node_bytes_written=0)
        self._encoding = EncodingMemo(encoding_cache_bytes, self.counter_prefix)
        self.batch_nodes = batch_nodes
        self.ordered_lookup = ordered_lookup

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        try:
            self.connection.execute("PRAGMA cache_size=-32768")
            tables = {r[0] for r in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables - {"report_objects"}:
                raise ValueError(f"{self.error_prefix}_STORE_MUST_BE_SEPARATE")
            self.connection.execute("CREATE TABLE IF NOT EXISTS report_objects (sha256 TEXT PRIMARY KEY, body BLOB NOT NULL)")
            self.connection.execute("BEGIN IMMEDIATE")
            self._batch_nodes = min(self.batch_nodes,
                                   self.connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
        except BaseException:
            self.connection.close()
            raise
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if kind is None:
                self._flush()
                with measure_stage(self.counter_prefix + "_commit"):
                    self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()
            self._seen.clear()
            self._pending.clear()
            self._pending_bytes = 0
            self._encoding.clear()

    def _count(self, name, amount=1):
        self.stats[name] += amount
        count_work(f"{self.counter_prefix}_{name}", amount)

    def _flush(self):
        if not self._pending:
            return
        keys = sorted(self._pending)
        placeholders = ",".join("?" for _ in keys)
        with measure_stage(self.counter_prefix + "_lookup"):
            if self.ordered_lookup:
                # Read only the covering SHA/rowid index first, then visit body
                # rows in physical order. LIMIT prevents SQLite flattening the
                # ordered subquery; CROSS JOIN fixes the small index result as
                # the outer loop. No schema migration or evidence rewrite.
                query = f"""SELECT stored.sha256,
                    substr(stored.body, 1, {MAX_COMPRESSED_NODE_BYTES + 1})
                    FROM (SELECT rowid AS node_id FROM report_objects
                          WHERE sha256 IN ({placeholders}) ORDER BY rowid LIMIT -1) AS selected
                    CROSS JOIN report_objects AS stored ON stored.rowid = selected.node_id"""
            else:
                query = f"SELECT sha256, body FROM report_objects WHERE sha256 IN ({placeholders})"
            existing = {}
            for sha, body in self.connection.execute(query, keys):
                # A corrupt oversized BLOB cannot grow a batch without bound.
                # Valid zlib streams fit the raw length plus framing allowance.
                if len(body) > min(MAX_COMPRESSED_NODE_BYTES, len(self._pending[sha]) + 8192):
                    raise ValueError(f"{self.error_prefix}_HASH_MISMATCH")
                existing[sha] = body
            count_work(self.counter_prefix + "_lookup_keys", len(keys))
            count_work(self.counter_prefix + "_lookup_body_bytes", sum(map(len, existing.values())))
        new_nodes = []
        # One timing/progress event per bounded batch, not per node. Consumers
        # may persist progress callbacks; millions of notifications would turn
        # instrumentation itself into a new I/O bottleneck.
        with measure_stage(self.counter_prefix + "_stored_verify"):
            for sha in keys:
                encoded = self._pending[sha]
                if sha in existing:
                    stored = _decompress(existing[sha], sha, self.error_prefix)
                    if stored != encoded:
                        raise ValueError(f"{self.error_prefix}_HASH_MISMATCH")
                    self._count("nodes_reused")
                    continue
                portable = self.path.parent / (self.path.name + ".nodes") / sha
                if portable.exists():
                    # Extracted portable archives need no SQLite import/rewrite.
                    if not portable.resolve().is_relative_to(self.path.parent.resolve()):
                        raise ValueError(f"INVALID_{self.error_prefix}_STORE")
                    if portable.stat().st_size != len(encoded) or portable.read_bytes() != encoded:
                        raise ValueError(f"{self.error_prefix}_HASH_MISMATCH")
                    self._count("nodes_reused")
                else:
                    new_nodes.append((sha, encoded))
        inserts = []
        if new_nodes:
            with measure_stage(self.counter_prefix + "_compress"):
                inserts = [(sha, zlib.compress(encoded, 1)) for sha, encoded in new_nodes]
        if inserts:
            with measure_stage(self.counter_prefix + "_insert"):
                self.connection.executemany("INSERT INTO report_objects VALUES (?,?)", inserts)
            self._count("nodes_written", len(inserts))
            self._count("node_bytes_written", sum(len(self._pending[sha]) for sha, _ in inserts))
        count_work(f"{self.counter_prefix}_batches")
        self._pending.clear()
        self._pending_bytes = 0

    def _encode(self, value, depth=0):
        if self._encoding.capacity:
            return self._encode_cached(value, depth)[0]
        return self._encode_uncached(value, depth)

    def _encode_cached(self, value, depth):
        if depth > 128:
            raise ValueError(f"{self.error_prefix}_TOO_DEEP")
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError(f"INVALID_{self.error_prefix}_KEY")
            pairs, tokens = [], []
            for key, item in sorted(value.items()):
                child, token = self._encode_cached(item, depth + 1)
                pairs.append([key, child])
                tokens.append((key, token))
            node = ["dict", pairs]
            key = ("dict", tuple(tokens))
            weight = None
        elif isinstance(value, (list, tuple)):
            children = [self._encode_cached(item, depth + 1) for item in value]
            node = ["list", [child for child, _ in children]]
            key = ("list", tuple(token for _, token in children))
            weight = None
        else:
            return ["value", value], self._encoding.scalar(value)
        hit = self._encoding.get(key)
        if hit is None:
            count_work(self.counter_prefix + "_node_encodes")
            encoded = canonical_json(node).encode()
            if len(encoded) > MAX_NODE_BYTES:
                raise ValueError(f"{self.error_prefix}_NODE_TOO_LARGE")
            token = self._encoding.token()
            sha = None if len(encoded) <= 512 else _sha(encoded)
            hit = (token, sha, encoded if sha is not None else None)
            if key[0] == "dict":
                weight = sum(160 + sys.getsizeof(name) + self._encoding.scalar_weight(child)
                             for name, child in key[1])
            else:
                weight = sum(48 + self._encoding.scalar_weight(child) for child in key[1])
            self._encoding.put(key, hit, weight)
        token, sha, encoded = hit
        if sha is None:
            # Reconstruct fresh inline lists, never return memo-owned objects.
            return ["inline", node], token
        return self._retain_node(sha, encoded), token

    def _encode_uncached(self, value, depth=0):
        if depth > 128:
            raise ValueError(f"{self.error_prefix}_TOO_DEEP")
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError(f"INVALID_{self.error_prefix}_KEY")
            node = ["dict", [[key, self._encode(item, depth + 1)] for key, item in sorted(value.items())]]
        elif isinstance(value, (list, tuple)):
            # canonical_json treats tuples as arrays (e.g. dependency metadata).
            node = ["list", [self._encode(item, depth + 1) for item in value]]
        else:
            return ["value", value]
        count_work(self.counter_prefix + "_node_encodes")
        encoded = canonical_json(node).encode()
        if len(encoded) <= 512:
            return ["inline", node]
        if len(encoded) > MAX_NODE_BYTES:
            raise ValueError(f"{self.error_prefix}_NODE_TOO_LARGE")
        sha = _sha(encoded)
        return self._retain_node(sha, encoded)

    def _retain_node(self, sha, encoded):
        if sha not in self._seen:
            if self._pending and self._pending_bytes + len(encoded) > MAX_NODE_BYTES:
                self._flush()
            self._pending[sha] = encoded
            self._pending_bytes += len(encoded)
            self._seen[sha] = None
            if len(self._seen) > 65536:
                self._seen.popitem(last=False)
            if len(self._pending) >= self._batch_nodes or self._pending_bytes >= MAX_NODE_BYTES:
                self._flush()
        else:
            self._seen.move_to_end(sha)
            self._count("nodes_reused")
        return ["ref", sha]

    def retain(self, value):
        with measure_stage(self.counter_prefix + "_encode"):
            root = self._encode(value)
        with measure_stage(self.counter_prefix + "_value_hash"):
            sha = digest(value)
        return dict(schema=SCHEMA, store=self.path.name, root=root, value_sha256=sha)


def _decompress(body, sha, prefix):
    try:
        decoder = zlib.decompressobj()
        encoded = decoder.decompress(body, MAX_NODE_BYTES + 1)
        if len(encoded) > MAX_NODE_BYTES or not decoder.eof or _sha(encoded) != sha:
            raise ValueError(f"{prefix}_HASH_MISMATCH")
        return encoded
    except zlib.error as error:
        raise ValueError(f"{prefix}_HASH_MISMATCH") from error


class EvidenceGraphReader:
    """Bounded encoded-node cache; every resolve returns a detached JSON tree."""
    error_prefix = "EVIDENCE"
    counter_prefix = "evidence"

    def __init__(self, directory, *, cache_bytes=16 * 1024 * 1024,
                 max_value_bytes=MAX_REPORT_BYTES, batch_nodes=128, batch_bytes=4 * 1024 * 1024):
        for name, value, lower, upper in (
            ("cache_bytes", cache_bytes, 0, MAX_CACHE_BYTES),
            ("batch_nodes", batch_nodes, 1, MAX_BATCH_NODES),
            ("batch_bytes", batch_bytes, 1, MAX_NODE_BYTES),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"INVALID_{self.error_prefix}_{name.upper()}")
        self.directory = Path(directory)
        self.cache_bytes = cache_bytes
        self.max_value_bytes = max_value_bytes
        self.batch_nodes = batch_nodes
        self.batch_bytes = batch_bytes
        self._cache = OrderedDict()
        self._bytes = 0
        self._connections = {}
        self._data_versions = {}

    def _remember(self, key, encoded):
        if key in self._cache or len(encoded) > self.cache_bytes:
            return
        while self._cache and self._bytes + len(encoded) > self.cache_bytes:
            _, removed = self._cache.popitem(last=False)
            self._bytes -= len(removed)
        self._cache[key] = encoded
        self._bytes += len(encoded)

    def _references(self, descriptor, depth=0):
        if depth > 128 or not isinstance(descriptor, list) or len(descriptor) != 2:
            raise ValueError(f"INVALID_{self.error_prefix}_NODE")
        kind, value = descriptor
        if kind == "ref":
            yield object_id(value, self.error_prefix)
        elif kind == "inline" and isinstance(value, list) and len(value) == 2:
            if value[0] == "dict":
                for _, child in value[1]:
                    yield from self._references(child, depth + 1)
            elif value[0] == "list":
                for child in value[1]:
                    yield from self._references(child, depth + 1)

    def _prefetch(self, descriptors, store):
        """Read one bounded sibling batch, caching only hash-verified bytes."""
        connection = self._connections.get(store)
        limit = min(self.batch_bytes, self.cache_bytes)
        if connection is None or not limit:
            return
        ids = {}
        references = (sha for item in descriptors for sha in self._references(item))
        for sha in islice(references, self.batch_nodes):
            if (store, sha) not in self._cache:
                ids[sha] = None
        if not ids:
            return
        seen = set(ids)
        used = 0
        # Bounded breadth-first lookahead coalesces narrow nested proof branches,
        # not just wide sibling lists. Keep bytes, discovered IDs and work bounded.
        while ids and used < limit:
            placeholders = ",".join("?" for _ in ids)
            cursor = connection.execute(
                f"SELECT sha256, CASE WHEN length(body)<=? THEN body END "
                f"FROM report_objects WHERE sha256 IN ({placeholders})",
                (min(limit - used, MAX_COMPRESSED_NODE_BYTES), *ids),
            )
            count_work(f"{self.counter_prefix}_read_batches")
            following = {}
            try:
                for sha, body in cursor:
                    if body is None:
                        continue  # Demand reads still validate oversized/missing nodes.
                    encoded = _decompress(body, sha, self.error_prefix)
                    count_work(f"{self.counter_prefix}_nodes_read")
                    if used + len(encoded) > limit:
                        return
                    self._remember((store, sha), encoded)
                    used += len(encoded)
                    if len(seen) < self.batch_nodes:
                        for child in self._references(["inline", json.loads(encoded)]):
                            if child not in seen and (store, child) not in self._cache:
                                seen.add(child)
                                following[child] = None
                                if len(seen) >= self.batch_nodes:
                                    break
            finally:
                cursor.close()
            ids = following

    def _groups(self, values):
        iterator = iter(values)
        while group := list(islice(iterator, self.batch_nodes)):
            yield group

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()
        self._data_versions.clear()
        self._cache.clear()
        self._bytes = 0

    def validate_store(self, store):
        if not isinstance(store, str) or Path(store).name != store or store in {".", ".."}:
            raise ValueError(f"INVALID_{self.error_prefix}_STORE")
        path = self.directory / store
        if not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError(f"INVALID_{self.error_prefix}_STORE")
        return path

    def check_store(self, store):
        """Check external commits once per root, not once per proof node."""
        path = self.validate_store(store)
        if store in self._connections and not path.is_file():
            self._connections.pop(store).close()
            self._data_versions.pop(store, None)
            self._cache.clear()
            self._bytes = 0
        if store not in self._connections and path.is_file():
            if len(self._connections) >= 8:
                oldest = next(iter(self._connections))
                self._connections.pop(oldest).close()
                self._data_versions.pop(oldest, None)
                self._cache.clear()
                self._bytes = 0
            self._connections[store] = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        connection = self._connections.get(store)
        if connection is not None:
            # A committed external edit must not be hidden by a warm node cache.
            version = connection.execute("PRAGMA data_version").fetchone()[0]
            if self._data_versions.get(store, version) != version:
                self._cache.clear()
                self._bytes = 0
            self._data_versions[store] = version

    def _node_bytes(self, store, sha):
        path = self.validate_store(store)
        key = store, object_id(sha, self.error_prefix)
        if store not in self._connections and path.is_file():
            self.check_store(store)
        connection = self._connections.get(store)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        row = connection.execute(
            "SELECT CASE WHEN length(body)<=? THEN body END FROM report_objects WHERE sha256=?",
            (MAX_COMPRESSED_NODE_BYTES, sha),
        ).fetchone() if connection else None
        if row is not None:
            if row[0] is None:
                raise ValueError(f"{self.error_prefix}_NODE_TOO_LARGE")
            encoded = _decompress(row[0], sha, self.error_prefix)
        else:
            portable = self.directory / (store + ".nodes") / sha
            if not portable.resolve().is_relative_to(self.directory.resolve()):
                raise ValueError(f"INVALID_{self.error_prefix}_STORE")
            if not portable.is_file():
                suffix = "OBJECT_MISSING" if connection else "STORE_MISSING"
                raise ValueError(f"{self.error_prefix}_{suffix}")
            if portable.stat().st_size > MAX_NODE_BYTES:
                raise ValueError(f"{self.error_prefix}_NODE_TOO_LARGE")
            encoded = portable.read_bytes()
            if _sha(encoded) != sha:
                raise ValueError(f"{self.error_prefix}_HASH_MISMATCH")
            # Portable bytes are reverified, not memoized without a DB version.
            count_work(f"{self.counter_prefix}_nodes_read")
            return encoded
        count_work(f"{self.counter_prefix}_nodes_read")
        self._remember(key, encoded)
        return encoded

    def _node(self, store, sha):
        return json.loads(self._node_bytes(store, sha))

    def _decode(self, item, store, budget, depth=0):
        budget[0] += 64
        if self.max_value_bytes is not None and budget[0] > self.max_value_bytes:
            raise ValueError(f"{self.error_prefix}_EXPANSION_TOO_LARGE")
        if depth > 128 or not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"INVALID_{self.error_prefix}_NODE")
        kind, value = item
        if kind == "value":
            if isinstance(value, (dict, list)):
                raise ValueError(f"INVALID_{self.error_prefix}_SCALAR")
            budget[0] += len(canonical_json(value).encode("utf-8"))
            if self.max_value_bytes is not None and budget[0] > self.max_value_bytes:
                raise ValueError(f"{self.error_prefix}_EXPANSION_TOO_LARGE")
            return value
        node = self._node(store, value) if kind == "ref" else value if kind == "inline" else None
        if not isinstance(node, list) or len(node) != 2:
            raise ValueError(f"INVALID_{self.error_prefix}_NODE")
        if node[0] == "dict":
            result = {}
            for group in self._groups(node[1]):
                self._prefetch((descriptor for _, descriptor in group), store)
                for key, descriptor in group:
                    if not isinstance(key, str) or key in result:
                        raise ValueError(f"INVALID_{self.error_prefix}_KEY")
                    budget[0] += len(canonical_json(key).encode("utf-8"))
                    result[key] = self._decode(descriptor, store, budget, depth + 1)
            return result
        if node[0] == "list":
            result = []
            for group in self._groups(node[1]):
                self._prefetch(group, store)
                result.extend(self._decode(v, store, budget, depth + 1) for v in group)
            return result
        raise ValueError(f"INVALID_{self.error_prefix}_NODE")

    def resolve(self, reference):
        if not isinstance(reference, dict) or reference.get("schema") != SCHEMA:
            raise ValueError("INVALID_EVIDENCE_REFERENCE")
        self.check_store(reference.get("store"))
        value = self._decode(reference["root"], reference["store"], [0])
        if digest(value) != object_id(reference.get("value_sha256")):
            raise ValueError("EVIDENCE_VALUE_HASH_MISMATCH")
        return value

    def reachable_nodes(self, root, store):
        """Export each reachable node once, never unrelated database content."""
        seen = set()
        pending = [root]
        while pending:
            kind, value = pending.pop()
            if kind == "value":
                continue
            if kind == "ref":
                if value in seen:
                    continue
                seen.add(value)
                encoded = self._node_bytes(store, value)
                yield value, encoded
                node = json.loads(encoded)
            else:
                node = value
            if node[0] == "dict":
                pending.extend(item[1] for item in node[1])
            else:
                pending.extend(node[1])
