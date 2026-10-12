"""Public native-job preprocessing lifetime and version guard."""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from application.daily_work import DailyWorkContext, current_work
from application.replay_contracts import digest
from infrastructure.research_objects import ResearchVersionGuard
from infrastructure.storage.research_cache import SQLiteRawParseCache


@contextmanager
def open_daily_work(*, parse_cache_path=None, enabled=True, max_bytes=64 * 1024 * 1024):
    guard = ResearchVersionGuard()
    cache = SQLiteRawParseCache(Path(parse_cache_path)) if parse_cache_path and enabled else None
    try:
        if cache is not None:
            cache.__enter__()
        work = DailyWorkContext(digest(asdict(guard.versions)), enabled=enabled, max_bytes=max_bytes,
                                parse_memo=cache, check_version=guard.check)
        try:
            with work.activate():
                yield work
        finally:
            work.close()
    finally:
        if cache is not None:
            cache.__exit__()


@contextmanager
def native_work_scope(parse_cache_path=None):
    work = current_work()
    if work is not None:
        with work.activate():
            yield work
    else:
        with open_daily_work(parse_cache_path=parse_cache_path) as work:
            yield work
