"""Single-process guard for the stateful daily analysis workflow."""

from __future__ import annotations

import datetime
import fcntl
import json
import os
import socket
from pathlib import Path
from types import TracebackType
from typing import TextIO


class RunAlreadyActiveError(RuntimeError):
    """Raised when another process owns the daily analysis lock."""


class ProcessRunLock:
    """Hold an advisory file lock for one complete analysis transaction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._handle: TextIO | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            metadata = self._read_metadata(handle)
            handle.close()
            raise RunAlreadyActiveError(
                "Another stock analysis run is active "
                "(pid={pid}, host={host}, started_at={started_at})".format(
                    pid=metadata.get("pid", "unknown"),
                    host=metadata.get("host", "unknown"),
                    started_at=metadata.get("started_at", "unknown"),
                )
            ) from error

        metadata = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(metadata, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "ProcessRunLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    @staticmethod
    def _read_metadata(handle: TextIO) -> dict[str, object]:
        try:
            handle.seek(0)
            payload = json.load(handle)
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}


__all__ = ["ProcessRunLock", "RunAlreadyActiveError"]
