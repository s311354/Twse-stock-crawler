"""Private, bounded native-work spool; never reads an external pickle path.

Only already-collected internal snapshots are encoded. Descriptor hashes are
kept in memory and verified before decode. No archive, cache database, source
document, availability time or model result is written or reselected here.
"""
from collections.abc import MutableMapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import errno
import hashlib
from pathlib import Path
import pickle
from shutil import disk_usage
from tempfile import TemporaryDirectory, gettempdir

import pandas as pd

from application.daily_performance import count_work, measure_stage
from domain.fundamental_derivation import FundamentalBundle
from domain.numeric_history import metadata_free_frame


class SnapshotMetadata(dict):
    __slots__ = ("_nonempty",)

    def __init__(self, values, *, nonempty):
        super().__init__(values)
        self._nonempty = nonempty

    def __bool__(self):
        return self._nonempty


def _cutoff(value):
    if value is None:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="raise")
    if pd.isna(parsed):
        raise ValueError("INVALID_SNAPSHOT_CUTOFF")
    return parsed.isoformat()


def _metadata(snapshot):
    # Only real non-model consumers' identity/source fields belong here. A raw
    # document can itself be a scalar string; capturing all scalars would keep
    # large originals alive even though the financial graph is spooled.
    keys = ("sector", "available_at", "twse_financial_observed_at", "base_quality_feature_sources")
    values = {key: deepcopy(snapshot[key]) for key in keys if key in snapshot}
    return SnapshotMetadata(values, nonempty=bool(snapshot))


@dataclass(frozen=True, slots=True)
class _Descriptor:
    sha256: str
    size: int
    metadata: SnapshotMetadata
    cutoff: str | None
    market: str | None


class SnapshotRetentionStorageError(OSError):
    """A private work spool cannot preserve its free-space reserve."""

    def __init__(self, code, path, *, free_bytes, required_bytes):
        self.code = code
        self.path = str(path)
        self.free_bytes = free_bytes
        self.required_bytes = required_bytes
        super().__init__(errno.ENOSPC,
            f"{code}: free_bytes={free_bytes}; required_bytes={required_bytes}", self.path)


class WorkSnapshotRetention:
    """One work owns encoded bytes; only oversized universes spill to local tmp."""

    def __init__(self, version, *, memory_bytes=8 * 1024 * 1024,
                 max_spool_bytes=8 * 1024 * 1024 * 1024,
                 max_snapshot_bytes=128 * 1024 * 1024,
                 spool_parent=None, min_free_bytes=1024 * 1024 * 1024):
        if min(memory_bytes, max_spool_bytes, max_snapshot_bytes, min_free_bytes) < 0:
            raise ValueError("INVALID_SNAPSHOT_RETENTION_CAPACITY")
        self.version = version
        self.memory_bytes, self.max_spool_bytes = memory_bytes, max_spool_bytes
        self.max_snapshot_bytes = max_snapshot_bytes
        self.spool_parent = Path(spool_parent) if spool_parent is not None else None
        self.min_free_bytes = min_free_bytes
        self._memory = {}
        self._spilled = set()
        self._references = {}
        self._memory_size = self._spool_size = 0
        self._directory = None
        self._maps = []
        self._closed = False

    def _check(self):
        if self._closed:
            raise ValueError("SNAPSHOT_RETENTION_CLOSED")
        from application.daily_work import current_work
        work = current_work()
        if work is not None:
            if getattr(work, "_closed", False):
                raise ValueError("DAILY_WORK_CONTEXT_CLOSED")
            if work.version != self.version or getattr(work, "snapshot_retention", self) is not self:
                raise ValueError("SNAPSHOT_RETENTION_WORK_MISMATCH")

    def snapshots(self):
        self._check()
        result = SnapshotRetentionMap(self)
        self._maps.append(result)
        return result

    def bundles(self):
        self._check()
        result = BundleRetentionMap(self)
        self._maps.append(result)
        return result

    @contextmanager
    def scope(self):
        """Release one analysis' maps while keeping the outer work and memo alive."""
        self._check()
        parents = {id(mapping) for mapping in self._maps}
        try:
            yield self
        finally:
            if not self._closed:
                for mapping in tuple(self._maps):
                    if id(mapping) not in parents:
                        mapping.clear()
                        mapping._released = True
                        self._maps = [entry for entry in self._maps if entry is not mapping]

    def _storage_error(self, code, path, size):
        try:
            free = disk_usage(path).free
        except OSError:
            free = None
        return SnapshotRetentionStorageError(code, path,
            free_bytes=free, required_bytes=size + self.min_free_bytes)

    def _spill(self, sha, encoded):
        parent = self.spool_parent if self.spool_parent is not None else Path(gettempdir())
        location = Path(self._directory.name) if self._directory is not None else parent
        if disk_usage(location).free < len(encoded) + self.min_free_bytes:
            raise self._storage_error("SNAPSHOT_RETENTION_SPACE_REQUIRED", location, len(encoded))
        path, created = None, False
        try:
            if self._directory is None:
                # The optional parent chooses only the volume. Each work still
                # owns a fresh private directory and never imports other bytes.
                self._directory = TemporaryDirectory(prefix="twse-native-snapshots-", dir=self.spool_parent)
            path = Path(self._directory.name) / (sha + ".bin")
            output = path.open("xb")
            created = True
            with output:
                output.write(encoded)
        except OSError as error:
            if created:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            if error.errno in {errno.ENOSPC, errno.EDQUOT}:
                raise self._storage_error("SNAPSHOT_RETENTION_STORAGE_FULL", location, len(encoded)) from error
            raise

    def _retain(self, value):
        self._check()
        with measure_stage("snapshot_retain"):
            encoded = pickle.dumps(value, protocol=5)
            if len(encoded) > self.max_snapshot_bytes:
                raise ValueError("SNAPSHOT_RETENTION_SNAPSHOT_TOO_LARGE")
            sha = hashlib.sha256(encoded).hexdigest()
            if sha not in self._memory and sha not in self._spilled:
                if self._memory_size + len(encoded) <= self.memory_bytes:
                    self._memory[sha] = encoded
                    self._memory_size += len(encoded)
                else:
                    if self._spool_size + len(encoded) > self.max_spool_bytes:
                        raise ValueError("SNAPSHOT_RETENTION_CAPACITY_EXCEEDED")
                    self._spill(sha, encoded)
                    self._spilled.add(sha)
                    self._spool_size += len(encoded)
                    count_work("snapshot_spooled_bytes", len(encoded))
                count_work("snapshot_encoded_bytes", len(encoded))
            count_work("snapshots_retained")
            return sha, len(encoded)

    def _release(self, descriptor):
        sha = descriptor.sha256
        remaining = self._references[sha] - 1
        if remaining:
            self._references[sha] = remaining
            return
        del self._references[sha]
        if sha in self._memory:
            self._memory_size -= len(self._memory.pop(sha))
        else:
            self._spilled.remove(sha)
            self._spool_size -= descriptor.size
            path = Path(self._directory.name) / (sha + ".bin")
            path.unlink(missing_ok=True)

    def _materialize(self, descriptor):
        self._check()
        with measure_stage("snapshot_materialize"):
            encoded = self._memory.get(descriptor.sha256)
            if encoded is None:
                if descriptor.sha256 not in self._spilled or self._directory is None:
                    raise ValueError("SNAPSHOT_RETENTION_OBJECT_MISSING")
                path = Path(self._directory.name) / (descriptor.sha256 + ".bin")
                if path.is_symlink():
                    raise ValueError("SNAPSHOT_RETENTION_UNTRUSTED_PATH")
                try:
                    size = path.stat().st_size
                    if size != descriptor.size or size > self.max_snapshot_bytes:
                        raise ValueError("SNAPSHOT_RETENTION_SIZE_MISMATCH")
                    encoded = path.read_bytes()
                except OSError as error:
                    raise ValueError("SNAPSHOT_RETENTION_OBJECT_MISSING") from error
            if len(encoded) != descriptor.size or hashlib.sha256(encoded).hexdigest() != descriptor.sha256:
                raise ValueError("SNAPSHOT_RETENTION_HASH_MISMATCH")
            # Input can only be bytes this owner created. Hash verification
            # occurs before unpickle; there is no external file/import API.
            value = pickle.loads(encoded)
            if not isinstance(value, dict):
                raise ValueError("SNAPSHOT_RETENTION_TYPE_MISMATCH")
            count_work("snapshot_retention_decodes")
            return value

    def close(self):
        for mapping in self._maps:
            mapping._entries.clear()
            if isinstance(mapping, BundleRetentionMap):
                mapping._annual.clear()
        self._maps.clear()
        self._memory.clear()
        self._spilled.clear()
        self._references.clear()
        self._memory_size = self._spool_size = 0
        self._closed = True
        if self._directory is not None:
            self._directory.cleanup()

    @property
    def stats(self):
        return dict(memory_bytes=self._memory_size, spooled_bytes=self._spool_size,
                    objects=len(self._memory) + len(self._spilled), closed=self._closed)


class SnapshotRetentionMap(MutableMapping):
    def __init__(self, owner):
        self.owner = owner
        self._entries = {}
        self._released = False

    def _check(self):
        self.owner._check()
        if self._released:
            raise ValueError("SNAPSHOT_RETENTION_SCOPE_CLOSED")

    def retain(self, symbol, snapshot, *, cutoff=None, market=None):
        self._check()
        if not isinstance(snapshot, dict):
            snapshot = dict(snapshot)
        metadata, normalized_cutoff = _metadata(snapshot), _cutoff(cutoff)
        sha, size = self.owner._retain(snapshot)
        self.owner._references[sha] = self.owner._references.get(sha, 0) + 1
        previous = self._entries.get(symbol)
        self._entries[symbol] = _Descriptor(sha, size, metadata, normalized_cutoff, market)
        if previous is not None:
            self.owner._release(previous)

    def metadata(self, symbol):
        self._check()
        entry = self._entries.get(symbol)
        if entry is None:
            return {}
        values = entry.metadata
        return SnapshotMetadata(deepcopy(dict(values)), nonempty=bool(values))

    def materialize(self, symbol, *, cutoff=None, market=None):
        self._check()
        entry = self._entries.get(symbol)
        if entry is None:
            return {}
        if ((cutoff is not None and entry.cutoff is not None and _cutoff(cutoff) != entry.cutoff)
                or (market is not None and entry.market is not None and market != entry.market)):
            raise ValueError("SNAPSHOT_RETENTION_BINDING_MISMATCH")
        value = self.owner._materialize(entry)
        count_work("snapshot_materializations")
        return value

    def __getitem__(self, symbol):
        self._check()
        if symbol not in self._entries:
            raise KeyError(symbol)
        return self.materialize(symbol)

    def __setitem__(self, symbol, snapshot):
        self.retain(symbol, snapshot)

    def __delitem__(self, symbol):
        self._check()
        self.owner._release(self._entries.pop(symbol))

    def clear(self):
        self._check()
        for symbol in tuple(self._entries):
            del self[symbol]

    def __contains__(self, symbol):
        self._check()
        return symbol in self._entries

    def __iter__(self):
        self._check()
        return iter(self._entries)

    def __len__(self):
        self._check()
        return len(self._entries)


class BundleRetentionMap(SnapshotRetentionMap):
    """Small annual tables remain resident; their heavy snapshot is lazy."""
    def __init__(self, owner):
        super().__init__(owner)
        self._annual = {}

    def __setitem__(self, symbol, bundle):
        self.retain(symbol, bundle.snapshot, cutoff=bundle.as_of)
        self._annual[symbol] = (metadata_free_frame(bundle.annual_fundamentals), bundle.as_of)

    def __getitem__(self, symbol):
        self._check()
        if symbol not in self._entries:
            raise KeyError(symbol)
        snapshot = self.owner._materialize(self._entries[symbol])
        count_work("bundle_materializations")
        annual, as_of = self._annual[symbol]
        return FundamentalBundle(metadata_free_frame(annual), snapshot, as_of)

    def __delitem__(self, symbol):
        super().__delitem__(symbol)
        del self._annual[symbol]
