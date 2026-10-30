"""Public native-job preprocessing lifetime and version guard."""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from application.daily_work import DailyWorkContext, current_work
from application.replay_contracts import digest
from infrastructure.research_objects import ResearchVersionGuard
from infrastructure.storage.research_cache import SQLiteRawParseCache
from infrastructure.snapshot_retention import WorkSnapshotRetention


DEFAULT_PARSE_CACHE_MAX_BYTES = 512 * 1024 * 1024


@contextmanager
def open_daily_work(*, parse_cache_path=None, enabled=True, max_bytes=64 * 1024 * 1024,
                    snapshot_spool_parent=None, parse_cache_max_bytes=DEFAULT_PARSE_CACHE_MAX_BYTES,
                    parse_cache_compression=True):
    """Open one guarded work with separate memory and rebuildable disk budgets.

    A larger on-disk parse working set does not enlarge the in-memory memo.
    Originals and semantic identities are still checked by the source reader.
    """
    if snapshot_spool_parent is not None:
        Path(snapshot_spool_parent).mkdir(mode=0o700, parents=True, exist_ok=True)
    guard = ResearchVersionGuard()
    cache = (SQLiteRawParseCache(Path(parse_cache_path), max_bytes=parse_cache_max_bytes,
                                compression=parse_cache_compression)
             if parse_cache_path and enabled else None)
    try:
        if cache is not None:
            cache.__enter__()
        work = DailyWorkContext(digest(asdict(guard.versions)), enabled=enabled, max_bytes=max_bytes,
                                parse_memo=cache, check_version=guard.check)
        retention = WorkSnapshotRetention(work.version, spool_parent=snapshot_spool_parent)
        work.snapshot_retention = retention
        try:
            with work.activate():
                yield work
        finally:
            work.close()
            retention.close()
    finally:
        if cache is not None:
            cache.__exit__()


@contextmanager
def native_work_scope(parse_cache_path=None, *, snapshot_spool_parent=None,
                      parse_cache_max_bytes=DEFAULT_PARSE_CACHE_MAX_BYTES):
    work = current_work()
    if work is not None:
        with work.activate():
            yield work
    else:
        with open_daily_work(parse_cache_path=parse_cache_path,
                             snapshot_spool_parent=snapshot_spool_parent,
                             parse_cache_max_bytes=parse_cache_max_bytes) as work:
            yield work
