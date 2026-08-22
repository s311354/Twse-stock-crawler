"""Shared SQLite primitives for MOPS command and query repositories."""

from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Callable
from pathlib import Path

from domain.data_provenance import LEGACY_UNVERIFIED
from infrastructure.fundamentals.models import MopsFiling


class MopsStorageError(RuntimeError):
    """MOPS cache is unavailable or not writable."""


def to_epoch(value: str | dt.datetime) -> int:
    if isinstance(value, str):
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp())


def row_to_filing(row: tuple[object, ...] | None) -> MopsFiling | None:
    if row is None:
        return None
    return MopsFiling(
        filing_id=int(row[0]),
        symbol=str(row[1]),
        fiscal_year=int(row[2]),
        quarter=int(row[3]),
        report_id=str(row[4]),
        available_at=str(row[5]),
        fetched_at=str(row[6]),
        revision_number=int(row[7]),
        content_sha256=str(row[8]),
        source_url=str(row[9]),
        raw_path=str(row[10]),
        period_end=str(row[11]) if row[11] is not None else None,
        statement_scope=str(row[12]),
        audit_status=str(row[13]) if row[13] is not None else None,
        correction_description=(
            str(row[14]) if row[14] is not None else None
        ),
        availability_source=(
            str(row[15])
            if len(row) > 15 and row[15] is not None
            else LEGACY_UNVERIFIED
        ),
        correction_announced_on=(
            str(row[16])
            if len(row) > 16 and row[16] is not None
            else None
        ),
    )


def invalidate_active_bundle(
    connection: sqlite3.Connection,
    symbol: str,
) -> None:
    connection.execute(
        """
        UPDATE refresh_state
        SET active_filing_set_sha256 = NULL,
            active_snapshot_cache_key = NULL,
            active_derivation_version = NULL,
            active_snapshot_available_at_epoch = NULL,
            active_latest_fiscal_year = NULL,
            active_latest_quarter = NULL,
            active_annual_history_years = NULL
        WHERE symbol = ?
        """,
        (symbol,),
    )


def connect_sqlite(
    root: Path,
    database_path: Path,
    connect_attempts: int,
    sleeper: Callable[[float], None],
) -> sqlite3.Connection:
    last_error: OSError | sqlite3.OperationalError | None = None
    for attempt in range(1, connect_attempts + 1):
        connection: sqlite3.Connection | None = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise OSError("cache root is not a directory")
            connection = sqlite3.connect(database_path, timeout=30)
            connection.execute("PRAGMA busy_timeout = 30000")
            return connection
        except (OSError, sqlite3.OperationalError) as error:
            last_error = error
            if connection is not None:
                connection.close()
            if attempt < connect_attempts:
                sleeper(0.5 * attempt)
    raise MopsStorageError(
        "Unable to open MOPS SQLite cache at {} after {} attempt(s): {}. "
        "Check that Transcend-Dev is mounted and writable, or set "
        "MOPS_CACHE_DIR to a writable local directory.".format(
            database_path,
            connect_attempts,
            last_error,
        )
    ) from last_error


__all__ = [
    "MopsStorageError",
    "connect_sqlite",
    "invalidate_active_bundle",
    "row_to_filing",
    "to_epoch",
]
