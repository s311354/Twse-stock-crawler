"""Closed SQLite staging on a configured local volume, then verified publication.

Only repairable work files live in the local directory. The published database
and its content-addressed references keep their existing location and format.
All crawler graph writers share a bounded-wait destination lock. Foreign
writers must obey that lock too; journal/WAL sources are deliberately refused.
"""
from __future__ import annotations

import fcntl
from contextlib import closing
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from application.daily_performance import count_work, measure_stage


def _stamp(path):
    if path.is_symlink():
        raise ValueError("LOCAL_WORKSPACE_SYMLINK_DESTINATION")
    if not path.exists():
        return None
    value = path.stat()
    if not path.is_file():
        raise ValueError("LOCAL_WORKSPACE_INVALID_DESTINATION")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _closed_source(path):
    # Replacing a database with outstanding WAL/journal files is not safe.
    for suffix in ("-wal", "-journal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.exists():
            raise ValueError("LOCAL_WORKSPACE_SOURCE_NOT_CLOSED:" + suffix)


def _hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _verify_database(path):
    _closed_source(path)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA cache_size=-32768")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("LOCAL_WORKSPACE_DATABASE_INTEGRITY_FAILED")


class LocalSQLiteWorkspace:
    """One writer's isolated local snapshot; never copies an open work DB.

    A failed seed/write/verification leaves the old destination intact. A
    failure after atomic replacement can leave a fully verified new DB without
    a report reference; retry is safe because graph objects are append-only.
    No recovery receipt is published until all destination durability checks
    succeed. Temporary files are bounded to this writer's private directory.
    """
    def __init__(self, destination, *, root=None, lock_timeout=30):
        self.destination = Path(destination)
        self.root = Path(root).expanduser() if root is not None else None
        self.path = self.destination
        self._lock = self._temporary = None
        self._baseline = None
        self.receipt = None
        if type(lock_timeout) not in (int, float) or not 0 <= lock_timeout <= 30:
            raise ValueError("INVALID_EVIDENCE_STORE_LOCK_TIMEOUT")
        self.lock_timeout = lock_timeout

    def __enter__(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.destination.with_name(self.destination.name + ".writer.lock")
        if lock_path.is_symlink():
            raise ValueError("LOCAL_WORKSPACE_SYMLINK_LOCK")
        self._lock = lock_path.open("a+b")  # never truncate existing metadata
        try:
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as error:
                    if time.monotonic() >= deadline:
                        raise ValueError("EVIDENCE_STORE_WRITER_BUSY") from error
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            if self.root is None:
                return self
            with measure_stage("evidence_workspace_seed"):
                self.root.mkdir(parents=True, exist_ok=True)
                self._baseline = _stamp(self.destination)
                _closed_source(self.destination)
                required = self._baseline[2] if self._baseline else 0
                if shutil.disk_usage(self.root).free < required + 16 * 1024 * 1024:
                    raise ValueError("LOCAL_WORKSPACE_INSUFFICIENT_SPACE")
                self._temporary = tempfile.TemporaryDirectory(prefix="twse-evidence-", dir=self.root)
                self.path = Path(self._temporary.name) / self.destination.name
                if self._baseline is not None:
                    with closing(sqlite3.connect(self.destination.resolve().as_uri() + "?mode=ro", uri=True)) as source:
                        if source.execute("PRAGMA journal_mode").fetchone()[0] == "wal":
                            raise ValueError("LOCAL_WORKSPACE_SOURCE_NOT_CLOSED:wal_mode")
                        with closing(sqlite3.connect(self.path)) as target:
                            source.backup(target, pages=1024)
                    if _stamp(self.destination) != self._baseline:
                        raise ValueError("LOCAL_WORKSPACE_SOURCE_CHANGED")
                    _verify_database(self.path)
                    count_work("evidence_workspace_seed_bytes", self.path.stat().st_size)
            return self
        except BaseException:
            self.close()
            raise

    def publish(self):
        """Call only after the writer has committed and closed its connection."""
        if self.root is None:
            return
        with measure_stage("evidence_workspace_verify"):
            _verify_database(self.path)
            sha = _hash(self.path)
            size = self.path.stat().st_size
        temporary = None
        try:
            with measure_stage("evidence_workspace_publish"):
                # The staging copy is on the destination volume so rename is
                # atomic even when workspace and archive are different devices.
                if shutil.disk_usage(self.destination.parent).free < size + 16 * 1024 * 1024:
                    raise ValueError("LOCAL_WORKSPACE_PUBLISH_INSUFFICIENT_SPACE")
                with tempfile.NamedTemporaryFile(prefix=".evidence-publish-", dir=self.destination.parent,
                                                  delete=False) as target:
                    temporary = Path(target.name)
                    if self._baseline is not None:
                        os.fchmod(target.fileno(), self.destination.stat().st_mode & 0o777)
                    with self.path.open("rb") as source:
                        shutil.copyfileobj(source, target, 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                if temporary.stat().st_size != size or _hash(temporary) != sha:
                    raise ValueError("LOCAL_WORKSPACE_PUBLISH_HASH_MISMATCH")
                _closed_source(self.destination)
                if _stamp(self.destination) != self._baseline:
                    raise ValueError("LOCAL_WORKSPACE_SOURCE_CHANGED")
                os.replace(temporary, self.destination)
                temporary = None
                descriptor = os.open(self.destination.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                self.receipt = dict(schema="local-sqlite-publication/v1", sha256=sha, bytes=size)
                count_work("evidence_workspace_published_bytes", size)
                count_work("evidence_workspace_publications")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def close(self):
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def __exit__(self, *exc):
        self.close()
