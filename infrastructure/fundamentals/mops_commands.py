"""Write-side commands for versioned MOPS filing persistence."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from application.fundamental_ports import FundamentalUnitOfWork
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FundamentalSnapshot
from domain.fundamental_models import NormalizedFact
from domain.fundamental_models import FilingCatalogState
from infrastructure.fundamentals.models import MopsFiling
from infrastructure.fundamentals.models import MopsFilingCandidate
from infrastructure.fundamentals.mops_queries import MopsQueryRepository
from infrastructure.fundamentals.mops_sqlite import MopsStorageError
from infrastructure.fundamentals.mops_sqlite import invalidate_active_bundle
from infrastructure.fundamentals.mops_sqlite import to_epoch
from infrastructure.fundamentals.mops_unit_of_work import borrow_connection
from infrastructure.fundamentals.mops_unit_of_work import write_transaction


class MopsRevisionWriter:
    """Persist filing revisions and derived snapshots atomically."""

    def __init__(
        self,
        root: Path,
        connect: Callable[[], sqlite3.Connection],
        ensure_schema: Callable[[], None],
        queries: MopsQueryRepository,
    ) -> None:
        self.root = root
        self._connect = connect
        self._ensure_schema = ensure_schema
        self._queries = queries

    def save_filing_catalog(
        self, state: FilingCatalogState, *, uow: FundamentalUnitOfWork | None = None,
    ) -> None:
        self._ensure_schema()
        if any(item.symbol != state.symbol for item in (*state.candidates, *state.corrections, *state.verified_corrections)):
            raise ValueError("Filing catalog contains another symbol")
        payload = json.dumps(asdict(state), ensure_ascii=False, sort_keys=True)
        with borrow_connection(self._connect, None, uow) as connection:
            with write_transaction(connection, externally_managed=uow is not None):
                connection.execute(
                    "INSERT INTO filing_catalog_cache (symbol, checked_at_epoch, state_json) VALUES (?, ?, ?) "
                    "ON CONFLICT(symbol) DO UPDATE SET "
                    "checked_at_epoch=excluded.checked_at_epoch, state_json=excluded.state_json",
                    (state.symbol, to_epoch(state.checked_at), payload),
                )

    def save_refresh_attempt(
        self, key: str, attempted_at: float, status: str, *,
        uow: FundamentalUnitOfWork | None = None,
    ) -> None:
        self._ensure_schema()
        with borrow_connection(self._connect, None, uow) as connection:
            with write_transaction(connection, externally_managed=uow is not None):
                connection.execute(
                    "INSERT INTO remote_refresh_attempts VALUES (?, ?, ?) "
                    "ON CONFLICT(refresh_key) DO UPDATE SET "
                    "attempted_at_epoch=excluded.attempted_at_epoch, status=excluded.status",
                    (key, attempted_at, status),
                )

    def save_revision(
        self,
        candidate: MopsFilingCandidate,
        content: bytes,
        facts: list[NormalizedFact],
        *,
        available_at: str,
        fetched_at: str,
        availability_source: str | None = None,
        minimum_revision_number: int = 1,
        correction_description: str | None = None,
        correction_announced_on: str | None = None,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> MopsFiling:
        content_sha256 = hashlib.sha256(content).hexdigest()
        self._ensure_schema()

        relative_path = Path("raw") / candidate.symbol / (
            f"{candidate.fiscal_year}Q{candidate.quarter}"
        ) / f"{content_sha256}.html"
        raw_path = self.root / relative_path
        try:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = raw_path.with_suffix(".tmp")
            temporary_path.write_bytes(content)
            temporary_path.replace(raw_path)
        except OSError as error:
            raise MopsStorageError(
                f"Unable to write MOPS raw cache at {raw_path}: {error}"
            ) from error

        externally_managed = uow is not None or connection is not None
        with borrow_connection(self._connect, connection, uow) as active:
            existing = self._queries.find_by_hash_on_connection(
                active,
                candidate,
                content_sha256,
            )
            if existing is not None:
                return existing

            with write_transaction(
                active,
                externally_managed=externally_managed,
            ):
                row = active.execute(
                    """
                    SELECT COALESCE(MAX(revision_number), 0)
                    FROM filings
                    WHERE symbol = ? AND fiscal_year = ? AND quarter = ?
                      AND report_id = ?
                    """,
                    (
                        candidate.symbol,
                        candidate.fiscal_year,
                        candidate.quarter,
                        candidate.report_id,
                    ),
                ).fetchone()
                revision_number = max(
                    int(row[0]) + 1,
                    minimum_revision_number,
                )
                cursor = active.execute(
                    """
                    INSERT INTO filings (
                        symbol, fiscal_year, quarter, report_id, available_at,
                        available_at_epoch, fetched_at, revision_number,
                        content_sha256, source_url, raw_path, period_end,
                        statement_scope, audit_status, correction_description,
                        availability_source, correction_announced_on
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.symbol,
                        candidate.fiscal_year,
                        candidate.quarter,
                        candidate.report_id,
                        available_at,
                        to_epoch(available_at),
                        fetched_at,
                        revision_number,
                        content_sha256,
                        candidate.source_url,
                        str(relative_path),
                        candidate.period_end,
                        candidate.statement_scope,
                        candidate.audit_status,
                        correction_description,
                        availability_source or candidate.availability_source,
                        correction_announced_on,
                    ),
                )
                filing_id = int(cursor.lastrowid)
                active.executemany(
                    """
                    INSERT INTO facts (
                        filing_id, concept, context_id, unit, numeric_value,
                        text_value, start_date, end_date, instant,
                        dimensions_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            filing_id,
                            fact.concept,
                            fact.context_id,
                            fact.unit,
                            fact.numeric_value,
                            fact.text_value,
                            fact.start_date,
                            fact.end_date,
                            fact.instant,
                            json.dumps(fact.dimensions, ensure_ascii=True),
                        )
                        for fact in facts
                    ],
                )
                invalidate_active_bundle(active, candidate.symbol)

        return MopsFiling(
            filing_id=filing_id,
            symbol=candidate.symbol,
            fiscal_year=candidate.fiscal_year,
            quarter=candidate.quarter,
            report_id=candidate.report_id,
            available_at=available_at,
            fetched_at=fetched_at,
            revision_number=revision_number,
            content_sha256=content_sha256,
            source_url=candidate.source_url,
            raw_path=str(relative_path),
            period_end=candidate.period_end,
            statement_scope=candidate.statement_scope,
            audit_status=candidate.audit_status,
            correction_description=correction_description,
            availability_source=(
                availability_source or candidate.availability_source
            ),
            correction_announced_on=correction_announced_on,
        )

    def update_availability_provenance(
        self,
        filing: MopsFiling,
        *,
        available_at: str,
        availability_source: str,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> MopsFiling:
        self._ensure_schema()
        externally_managed = uow is not None or connection is not None
        with borrow_connection(self._connect, connection, uow) as active:
            with write_transaction(
                active,
                externally_managed=externally_managed,
            ):
                active.execute(
                    """
                    UPDATE filings
                    SET available_at = ?, available_at_epoch = ?,
                        availability_source = ?
                    WHERE id = ?
                    """,
                    (
                        available_at,
                        to_epoch(available_at),
                        availability_source,
                        filing.filing_id,
                    ),
                )
                invalidate_active_bundle(active, filing.symbol)
        return MopsFiling(
            filing_id=filing.filing_id,
            symbol=filing.symbol,
            fiscal_year=filing.fiscal_year,
            quarter=filing.quarter,
            report_id=filing.report_id,
            available_at=available_at,
            fetched_at=filing.fetched_at,
            revision_number=filing.revision_number,
            content_sha256=filing.content_sha256,
            source_url=filing.source_url,
            raw_path=filing.raw_path,
            period_end=filing.period_end,
            statement_scope=filing.statement_scope,
            audit_status=filing.audit_status,
            correction_description=filing.correction_description,
            availability_source=availability_source,
            correction_announced_on=filing.correction_announced_on,
        )

    def mark_full_refresh(
        self,
        symbol: str,
        refreshed_at: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._ensure_schema()
        externally_managed = uow is not None or connection is not None
        with borrow_connection(self._connect, connection, uow) as active:
            with write_transaction(active, externally_managed=externally_managed):
                active.execute(
                    """
                    INSERT INTO refresh_state (
                        symbol, full_catalog_checked_at_epoch
                    ) VALUES (?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        full_catalog_checked_at_epoch =
                            excluded.full_catalog_checked_at_epoch
                    """,
                    (symbol, to_epoch(refreshed_at)),
                )

    def mark_recent_revision_check(
        self,
        symbol: str,
        checked_at: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._ensure_schema()
        externally_managed = uow is not None or connection is not None
        checked_at_epoch = to_epoch(checked_at)
        with borrow_connection(self._connect, connection, uow) as active:
            with write_transaction(active, externally_managed=externally_managed):
                active.execute(
                    """
                    INSERT INTO refresh_state (
                        symbol, full_catalog_checked_at_epoch,
                        recent_revision_checked_at_epoch
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        recent_revision_checked_at_epoch =
                            excluded.recent_revision_checked_at_epoch
                    """,
                    (symbol, 0, checked_at_epoch),
                )

    def save_financial_batch_state(
        self,
        symbol: str,
        fiscal_year: int,
        quarter: int,
        content_sha256: str,
        observed_at: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._ensure_schema()
        externally_managed = uow is not None or connection is not None
        with borrow_connection(self._connect, connection, uow) as active:
            with write_transaction(active, externally_managed=externally_managed):
                active.execute(
                    """
                    INSERT INTO financial_batch_state (
                        symbol, fiscal_year, quarter, content_sha256,
                        observed_at_epoch
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        fiscal_year = excluded.fiscal_year,
                        quarter = excluded.quarter,
                        content_sha256 = excluded.content_sha256,
                        observed_at_epoch = excluded.observed_at_epoch
                    """,
                    (
                        symbol,
                        fiscal_year,
                        quarter,
                        content_sha256,
                        to_epoch(observed_at),
                    ),
                )

    def save_financial_batch_snapshot(
        self,
        snapshot: FundamentalSnapshot,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._ensure_schema()
        externally_managed = uow is not None or connection is not None
        payload = json.dumps(
            asdict(snapshot),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        with borrow_connection(self._connect, connection, uow) as active:
            with write_transaction(active, externally_managed=externally_managed):
                active.execute(
                    """
                    INSERT OR IGNORE INTO financial_batch_snapshots (
                        symbol, fiscal_year, quarter, content_sha256,
                        observed_at, observed_at_epoch, snapshot_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.symbol,
                        snapshot.fiscal_year,
                        snapshot.quarter,
                        snapshot.content_sha256,
                        snapshot.observed_at,
                        to_epoch(snapshot.observed_at),
                        payload,
                    ),
                )

    def mark_active_fundamental_bundle(
        self,
        symbol: str,
        *,
        filing_set_sha256: str,
        snapshot_cache_key: str,
        derivation_version: str,
        available_at: str,
        latest_period: tuple[int, int],
        annual_history_years: int,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._ensure_schema()
        externally_managed = uow is not None or connection is not None
        with borrow_connection(self._connect, connection, uow) as active:
            with write_transaction(active, externally_managed=externally_managed):
                active.execute(
                    "INSERT OR IGNORE INTO refresh_state (symbol, full_catalog_checked_at_epoch) VALUES (?, 0)",
                    (symbol,),
                )
                active.execute(
                    """
                    UPDATE refresh_state
                    SET active_filing_set_sha256 = ?,
                        active_snapshot_cache_key = ?,
                        active_derivation_version = ?,
                        active_snapshot_available_at_epoch = ?,
                        active_latest_fiscal_year = ?,
                        active_latest_quarter = ?,
                        active_annual_history_years = ?
                    WHERE symbol = ?
                    """,
                    (
                        filing_set_sha256,
                        snapshot_cache_key,
                        derivation_version,
                        to_epoch(available_at),
                        latest_period[0],
                        latest_period[1],
                        annual_history_years,
                        symbol,
                    ),
                )

    def save_fundamental_bundle(
        self,
        symbol: str,
        filing_set_sha256: str,
        sector: str,
        bundle: FundamentalBundle,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._ensure_schema()
        created_at_epoch = int(dt.datetime.now(dt.timezone.utc).timestamp())
        externally_managed = uow is not None or connection is not None
        with borrow_connection(self._connect, connection, uow) as active:
            with write_transaction(active, externally_managed=externally_managed):
                active.execute(
                    """
                    INSERT INTO fundamental_snapshots (
                        symbol, filing_set_sha256, sector, annual_json,
                        snapshot_json, created_at_epoch
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, filing_set_sha256, sector) DO UPDATE SET
                        annual_json = excluded.annual_json,
                        snapshot_json = excluded.snapshot_json,
                        created_at_epoch = excluded.created_at_epoch
                    """,
                    (
                        symbol,
                        filing_set_sha256,
                        sector,
                        bundle.annual_fundamentals.to_json(
                            orient="records",
                            date_format="iso",
                        ),
                        json.dumps(bundle.snapshot, ensure_ascii=True),
                        created_at_epoch,
                    ),
                )


__all__ = ["MopsRevisionWriter"]
