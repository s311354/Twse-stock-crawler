"""Idempotent SQLite schema creation and migration for the MOPS cache."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Callable
from contextlib import closing

from infrastructure.fundamentals.mops_sqlite import to_epoch


EMPTY_FILING_SET_SHA256 = hashlib.sha256(b"").hexdigest()


class MopsSchemaMigrator:
    """Own all DDL and legacy cache repair operations."""

    def __init__(self, connect: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connect
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def ensure(self) -> None:
        if self._ready:
            return
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS filings (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    fiscal_year INTEGER NOT NULL,
                    quarter INTEGER NOT NULL,
                    report_id TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    available_at_epoch INTEGER NOT NULL,
                    fetched_at TEXT NOT NULL,
                    revision_number INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    raw_path TEXT NOT NULL,
                    period_end TEXT,
                    statement_scope TEXT NOT NULL DEFAULT 'consolidated',
                    audit_status TEXT,
                    correction_description TEXT,
                    availability_source TEXT NOT NULL DEFAULT 'LEGACY_UNVERIFIED',
                    correction_announced_on TEXT,
                    UNIQUE(symbol, fiscal_year, quarter, report_id, content_sha256)
                );
                CREATE INDEX IF NOT EXISTS filings_symbol_period
                    ON filings(symbol, fiscal_year, quarter, revision_number);
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY,
                    filing_id INTEGER NOT NULL REFERENCES filings(id),
                    concept TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    unit TEXT,
                    numeric_value REAL,
                    text_value TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    instant TEXT,
                    dimensions_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS facts_filing_concept
                    ON facts(filing_id, concept);
                CREATE TABLE IF NOT EXISTS refresh_state (
                    symbol TEXT PRIMARY KEY,
                    full_catalog_checked_at_epoch INTEGER NOT NULL,
                    recent_revision_checked_at_epoch INTEGER,
                    active_filing_set_sha256 TEXT,
                    active_snapshot_cache_key TEXT,
                    active_derivation_version TEXT,
                    active_snapshot_available_at_epoch INTEGER,
                    active_latest_fiscal_year INTEGER,
                    active_latest_quarter INTEGER,
                    active_annual_history_years INTEGER
                );
                CREATE TABLE IF NOT EXISTS remote_refresh_attempts (
                    refresh_key TEXT PRIMARY KEY,
                    attempted_at_epoch REAL NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS filing_catalog_cache (
                    symbol TEXT PRIMARY KEY,
                    checked_at_epoch INTEGER NOT NULL,
                    state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fundamental_snapshots (
                    symbol TEXT NOT NULL,
                    filing_set_sha256 TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    annual_json TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at_epoch INTEGER NOT NULL,
                    PRIMARY KEY(symbol, filing_set_sha256, sector)
                );
                CREATE TABLE IF NOT EXISTS financial_batch_state (
                    symbol TEXT PRIMARY KEY,
                    fiscal_year INTEGER NOT NULL,
                    quarter INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    observed_at_epoch INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS financial_batch_snapshots (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    fiscal_year INTEGER NOT NULL,
                    quarter INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    observed_at_epoch INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    UNIQUE(symbol, fiscal_year, quarter, content_sha256)
                );
                CREATE INDEX IF NOT EXISTS financial_batch_snapshots_point_in_time
                    ON financial_batch_snapshots(
                        symbol, observed_at_epoch, fiscal_year, quarter
                    );
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(filings)"
                ).fetchall()
            }
            migrations = {
                "period_end": "ALTER TABLE filings ADD COLUMN period_end TEXT",
                "statement_scope": (
                    "ALTER TABLE filings ADD COLUMN statement_scope TEXT "
                    "NOT NULL DEFAULT 'consolidated'"
                ),
                "audit_status": (
                    "ALTER TABLE filings ADD COLUMN audit_status TEXT"
                ),
                "correction_description": (
                    "ALTER TABLE filings ADD COLUMN correction_description TEXT"
                ),
                "availability_source": (
                    "ALTER TABLE filings ADD COLUMN availability_source TEXT "
                    "NOT NULL DEFAULT 'LEGACY_UNVERIFIED'"
                ),
                "correction_announced_on": (
                    "ALTER TABLE filings ADD COLUMN correction_announced_on TEXT"
                ),
                "available_at_epoch": (
                    "ALTER TABLE filings ADD COLUMN available_at_epoch INTEGER"
                ),
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)
            refresh_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(refresh_state)"
                ).fetchall()
            }
            if "recent_revision_checked_at_epoch" not in refresh_columns:
                connection.execute(
                    "ALTER TABLE refresh_state ADD COLUMN "
                    "recent_revision_checked_at_epoch INTEGER"
                )
            active_state_migrations = {
                "active_filing_set_sha256": (
                    "ALTER TABLE refresh_state ADD COLUMN "
                    "active_filing_set_sha256 TEXT"
                ),
                "active_snapshot_cache_key": (
                    "ALTER TABLE refresh_state ADD COLUMN "
                    "active_snapshot_cache_key TEXT"
                ),
                "active_derivation_version": (
                    "ALTER TABLE refresh_state ADD COLUMN "
                    "active_derivation_version TEXT"
                ),
                "active_snapshot_available_at_epoch": (
                    "ALTER TABLE refresh_state ADD COLUMN "
                    "active_snapshot_available_at_epoch INTEGER"
                ),
                "active_latest_fiscal_year": (
                    "ALTER TABLE refresh_state ADD COLUMN "
                    "active_latest_fiscal_year INTEGER"
                ),
                "active_latest_quarter": (
                    "ALTER TABLE refresh_state ADD COLUMN "
                    "active_latest_quarter INTEGER"
                ),
                "active_annual_history_years": (
                    "ALTER TABLE refresh_state ADD COLUMN "
                    "active_annual_history_years INTEGER"
                ),
            }
            for column, statement in active_state_migrations.items():
                if column not in refresh_columns:
                    connection.execute(statement)
            connection.execute(
                """
                UPDATE refresh_state
                SET recent_revision_checked_at_epoch = full_catalog_checked_at_epoch
                WHERE recent_revision_checked_at_epoch IS NULL
                """
            )
            missing_epochs = connection.execute(
                "SELECT id, available_at FROM filings "
                "WHERE available_at_epoch IS NULL"
            ).fetchall()
            connection.executemany(
                "UPDATE filings SET available_at_epoch = ? WHERE id = ?",
                [
                    (to_epoch(str(available_at)), int(filing_id))
                    for filing_id, available_at in missing_epochs
                ],
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS filings_symbol_available_epoch
                ON filings(
                    symbol, available_at_epoch, fiscal_year, quarter,
                    revision_number
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS refresh_state_active_snapshot
                ON refresh_state(
                    active_derivation_version,
                    active_snapshot_available_at_epoch,
                    symbol
                )
                """
            )
            self._remove_legacy_empty_state(connection)
        self._ready = True

    @staticmethod
    def _remove_legacy_empty_state(connection: sqlite3.Connection) -> None:
        removed_snapshots = connection.execute(
            """
            DELETE FROM fundamental_snapshots
            WHERE filing_set_sha256 = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM filings
                  WHERE filings.symbol = fundamental_snapshots.symbol
              )
            """,
            (EMPTY_FILING_SET_SHA256,),
        ).rowcount
        removed_refresh_states = connection.execute(
            """
            DELETE FROM refresh_state
            WHERE NOT EXISTS (
                      SELECT 1
                      FROM filings
                      WHERE filings.symbol = refresh_state.symbol
                  )
              AND NOT EXISTS (
                      SELECT 1
                      FROM fundamental_snapshots
                      WHERE fundamental_snapshots.symbol = refresh_state.symbol
                  )
            """
        ).rowcount
        if removed_snapshots or removed_refresh_states:
            logging.warning(
                "MOPS cache repair removed empty_snapshots=%s "
                "orphan_refresh_states=%s",
                removed_snapshots,
                removed_refresh_states,
            )


__all__ = ["EMPTY_FILING_SET_SHA256", "MopsSchemaMigrator"]
