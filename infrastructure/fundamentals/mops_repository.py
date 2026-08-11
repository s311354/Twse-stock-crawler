"""SQLite metadata/fact store with content-addressed raw iXBRL cache."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import logging
import sqlite3
import time
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import closing
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import TracebackType

import pandas as pd

from application.fundamental_ports import FundamentalUnitOfWork
from config.settings import DATA_DIR
from config.settings import AppSettings
from domain.data_provenance import LEGACY_UNVERIFIED
from domain.fundamental_derivation import CachedFundamentalBundle
from domain.fundamental_derivation import FiledFact
from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FundamentalSnapshot
from domain.fundamental_models import NormalizedFact
from infrastructure.fundamentals.models import MopsFiling
from infrastructure.fundamentals.models import MopsFilingCandidate


EMPTY_FILING_SET_SHA256 = hashlib.sha256(b"").hexdigest()


class MopsStorageError(RuntimeError):
    """MOPS cache is unavailable or not writable."""


class SqliteFundamentalUnitOfWork:
    """Own one explicit SQLite transaction and connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._finished = False

    def __enter__(self) -> "SqliteFundamentalUnitOfWork":
        self.connection.execute("BEGIN")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_value, traceback
        try:
            if not self._finished:
                if exc_type is None:
                    self.commit()
                else:
                    self.rollback()
        finally:
            self.connection.close()
        return False

    def commit(self) -> None:
        if not self._finished:
            self.connection.commit()
            self._finished = True

    def rollback(self) -> None:
        if not self._finished:
            self.connection.rollback()
            self._finished = True


class MopsRepository:
    """Preserve every observed filing revision and its normalized facts."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        connect_attempts: int | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        settings = (
            AppSettings.from_env(
                mops_enabled=False,
                validate_paths=False,
            )
            if root is None or connect_attempts is None
            else None
        )
        self.root = Path(
            root
            or (settings.mops_cache_dir if settings is not None else None)
            or DATA_DIR / "mops"
        ).expanduser()
        self.database_path = self.root / "mops.sqlite3"
        self.raw_root = self.root / "raw"
        self.connect_attempts = (
            (
                settings.mops_sqlite_connect_attempts
                if settings is not None
                else 3
            )
            if connect_attempts is None
            else connect_attempts
        )
        self._sleep = sleeper
        self._schema_ready = False
        if self.connect_attempts < 1:
            raise ValueError("connect_attempts must be at least one")

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
            "{}Q{}".format(candidate.fiscal_year, candidate.quarter)
        ) / "{}.html".format(content_sha256)
        raw_path = self.root / relative_path
        try:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = raw_path.with_suffix(".tmp")
            temporary_path.write_bytes(content)
            temporary_path.replace(raw_path)
        except OSError as error:
            raise MopsStorageError(
                "Unable to write MOPS raw cache at {}: {}".format(raw_path, error)
            ) from error

        externally_managed = uow is not None or connection is not None
        with self._borrow_connection(connection, uow) as active_connection:
            existing = self._find_by_hash(
                active_connection,
                candidate,
                content_sha256,
            )
            if existing is not None:
                return existing

            # The filing row and all normalized facts commit or roll back together.
            with self._write_transaction(
                active_connection,
                externally_managed=externally_managed,
            ):
                row = active_connection.execute(
                    """
                    SELECT COALESCE(MAX(revision_number), 0)
                    FROM filings
                    WHERE symbol = ? AND fiscal_year = ? AND quarter = ? AND report_id = ?
                    """,
                    (
                        candidate.symbol,
                        candidate.fiscal_year,
                        candidate.quarter,
                        candidate.report_id,
                    ),
                ).fetchone()
                revision_number = max(int(row[0]) + 1, minimum_revision_number)
                cursor = active_connection.execute(
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
                        self._to_epoch(available_at),
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
                active_connection.executemany(
                    """
                    INSERT INTO facts (
                        filing_id, concept, context_id, unit, numeric_value,
                        text_value, start_date, end_date, instant, dimensions_json
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
                self._invalidate_active_bundle(
                    active_connection,
                    candidate.symbol,
                )

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
            availability_source=availability_source or candidate.availability_source,
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
        """Replace legacy timing metadata with a conservative observed timestamp."""

        self._ensure_schema()
        externally_managed = uow is not None or connection is not None
        with self._borrow_connection(connection, uow) as active_connection:
            with self._write_transaction(
                active_connection,
                externally_managed=externally_managed,
            ):
                active_connection.execute(
                    """
                    UPDATE filings
                    SET available_at = ?, available_at_epoch = ?, availability_source = ?
                    WHERE id = ?
                    """,
                    (
                        available_at,
                        self._to_epoch(available_at),
                        availability_source,
                        filing.filing_id,
                    ),
                )
                self._invalidate_active_bundle(
                    active_connection,
                    filing.symbol,
                )
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

    def latest_filing(
        self,
        symbol: str,
        fiscal_year: int,
        quarter: int,
        report_id: str = "C",
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> MopsFiling | None:
        self._ensure_schema()
        with self._borrow_connection(connection, uow) as active_connection:
            row = active_connection.execute(
                """
                SELECT id, symbol, fiscal_year, quarter, report_id, available_at,
                       fetched_at, revision_number, content_sha256, source_url, raw_path,
                       period_end, statement_scope, audit_status, correction_description,
                       availability_source, correction_announced_on
                FROM filings
                WHERE symbol = ? AND fiscal_year = ? AND quarter = ? AND report_id = ?
                ORDER BY revision_number DESC
                LIMIT 1
                """,
                (symbol, fiscal_year, quarter, report_id),
            ).fetchone()
        return self._row_to_filing(row)

    def find_by_hash(
        self,
        candidate: MopsFilingCandidate,
        content_sha256: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> MopsFiling | None:
        self._ensure_schema()
        with self._borrow_connection(connection, uow) as active_connection:
            return self._find_by_hash(active_connection, candidate, content_sha256)

    def load_filings(
        self,
        symbol: str,
        *,
        as_of: str | None = None,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[MopsFiling]:
        self._ensure_schema()
        query = """
            SELECT id, symbol, fiscal_year, quarter, report_id, available_at,
                   fetched_at, revision_number, content_sha256, source_url, raw_path,
                   period_end, statement_scope, audit_status, correction_description,
                   availability_source, correction_announced_on
            FROM filings WHERE symbol = ?
        """
        params: list[object] = [symbol]
        if as_of is not None:
            query += " AND available_at_epoch <= ?"
            params.append(self._to_epoch(as_of))
        query += " ORDER BY fiscal_year, quarter, available_at_epoch, revision_number"
        with self._borrow_connection(connection, uow) as active_connection:
            rows = active_connection.execute(query, params).fetchall()
        return [filing for row in rows if (filing := self._row_to_filing(row)) is not None]

    def load_facts(
        self,
        filing_id: int,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[NormalizedFact]:
        self._ensure_schema()
        with self._borrow_connection(connection, uow) as active_connection:
            rows = active_connection.execute(
                """
                SELECT concept, context_id, unit, numeric_value, text_value,
                       start_date, end_date, instant, dimensions_json
                FROM facts WHERE filing_id = ? ORDER BY id
                """,
                (filing_id,),
            ).fetchall()
        return [
            NormalizedFact(
                concept=row[0],
                context_id=row[1],
                unit=row[2],
                numeric_value=row[3],
                text_value=row[4],
                start_date=row[5],
                end_date=row[6],
                instant=row[7],
                dimensions=tuple(tuple(item) for item in json.loads(row[8])),
            )
            for row in rows
        ]

    def load_filed_financials(
        self,
        symbol: str,
        as_of: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[FiledFinancials]:
        """Load point-in-time filings and facts with one indexed JOIN query."""

        self._ensure_schema()
        with self._borrow_connection(connection, uow) as active_connection:
            rows = active_connection.execute(
                """
                SELECT f.id, f.symbol, f.fiscal_year, f.quarter, f.report_id,
                       f.available_at, f.fetched_at, f.revision_number,
                       f.content_sha256, f.source_url, f.raw_path, f.period_end,
                       f.statement_scope, f.audit_status, f.correction_description,
                       f.availability_source, f.correction_announced_on,
                       x.id, x.concept, x.context_id, x.unit, x.numeric_value,
                       x.text_value, x.start_date, x.end_date, x.instant,
                       x.dimensions_json
                FROM filings AS f
                LEFT JOIN facts AS x ON x.filing_id = f.id
                WHERE f.symbol = ?
                  AND f.available_at_epoch <= ?
                ORDER BY f.fiscal_year, f.quarter, f.revision_number, x.id
                """,
                (symbol, self._to_epoch(as_of)),
            ).fetchall()

        grouped: dict[int, tuple[MopsFiling, list[FiledFact]]] = {}
        for row in rows:
            filing_id = int(row[0])
            if filing_id not in grouped:
                filing = self._row_to_filing(row[:17])
                if filing is None:
                    continue
                grouped[filing_id] = (filing, [])
            if row[17] is not None:
                grouped[filing_id][1].append(
                    FiledFact(
                        concept=str(row[18]),
                        numeric_value=(
                            float(row[21]) if row[21] is not None else None
                        ),
                        text_value=str(row[22]) if row[22] is not None else None,
                        start_date=str(row[23]) if row[23] is not None else None,
                        end_date=str(row[24]) if row[24] is not None else None,
                        instant=str(row[25]) if row[25] is not None else None,
                        dimensions=tuple(
                            tuple(str(value) for value in item)
                            for item in json.loads(str(row[26]))
                        ),
                        unit=str(row[20]) if row[20] is not None else None,
                        context_id=str(row[19]) if row[19] is not None else None,
                    )
                )

        return [
            FiledFinancials(
                fiscal_year=filing.fiscal_year,
                quarter=filing.quarter,
                available_at=filing.available_at,
                revision_number=filing.revision_number,
                facts=tuple(facts),
                correction_description=filing.correction_description,
                availability_source=filing.availability_source,
                filing_id=filing.filing_id,
                content_sha256=filing.content_sha256,
                source_url=filing.source_url,
                correction_announced_on=filing.correction_announced_on,
                period_end=filing.period_end,
                statement_scope=filing.statement_scope,
                audit_status=filing.audit_status,
            )
            for filing, facts in grouped.values()
        ]

    def last_full_refresh_epoch(
        self,
        symbol: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> int | None:
        self._ensure_schema()
        with self._borrow_connection(connection, uow) as active_connection:
            row = active_connection.execute(
                "SELECT full_catalog_checked_at_epoch FROM refresh_state WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return int(row[0]) if row is not None else None

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
        with self._borrow_connection(connection, uow) as active_connection:
            with self._write_transaction(
                active_connection,
                externally_managed=externally_managed,
            ):
                active_connection.execute(
                    """
                    INSERT INTO refresh_state (symbol, full_catalog_checked_at_epoch)
                    VALUES (?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        full_catalog_checked_at_epoch = excluded.full_catalog_checked_at_epoch
                    """,
                    (symbol, self._to_epoch(refreshed_at)),
                )

    def last_recent_revision_check_epoch(
        self,
        symbol: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> int | None:
        self._ensure_schema()
        with self._borrow_connection(connection, uow) as active_connection:
            row = active_connection.execute(
                "SELECT recent_revision_checked_at_epoch "
                "FROM refresh_state WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return int(row[0]) if row is not None and row[0] is not None else None

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
        checked_at_epoch = self._to_epoch(checked_at)
        with self._borrow_connection(connection, uow) as active_connection:
            with self._write_transaction(
                active_connection,
                externally_managed=externally_managed,
            ):
                active_connection.execute(
                    """
                    INSERT INTO refresh_state (
                        symbol,
                        full_catalog_checked_at_epoch,
                        recent_revision_checked_at_epoch
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        recent_revision_checked_at_epoch = excluded.recent_revision_checked_at_epoch
                    """,
                    (symbol, checked_at_epoch, checked_at_epoch),
                )

    def load_financial_batch_state(
        self,
        symbol: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[tuple[int, int], str] | None:
        self._ensure_schema()
        with self._borrow_connection(connection, uow) as active_connection:
            row = active_connection.execute(
                """
                SELECT fiscal_year, quarter, content_sha256
                FROM financial_batch_state
                WHERE symbol = ?
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return (int(row[0]), int(row[1])), str(row[2])

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
        with self._borrow_connection(connection, uow) as active_connection:
            with self._write_transaction(
                active_connection,
                externally_managed=externally_managed,
            ):
                active_connection.execute(
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
                        self._to_epoch(observed_at),
                    ),
                )

    def save_financial_batch_snapshot(
        self,
        snapshot: FundamentalSnapshot,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        """Store the first observed copy of each immutable TWSE revision."""

        self._ensure_schema()
        externally_managed = uow is not None or connection is not None
        payload = json.dumps(
            asdict(snapshot),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._borrow_connection(connection, uow) as active_connection:
            with self._write_transaction(
                active_connection,
                externally_managed=externally_managed,
            ):
                active_connection.execute(
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
                        self._to_epoch(snapshot.observed_at),
                        payload,
                    ),
                )

    def load_financial_batch_snapshots(
        self,
        symbol: str,
        *,
        as_of: str,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[FundamentalSnapshot]:
        """Return one latest eligible revision per period without look-ahead."""

        self._ensure_schema()
        with self._borrow_connection(connection, uow) as active_connection:
            rows = active_connection.execute(
                """
                SELECT fiscal_year, quarter, observed_at_epoch, id, snapshot_json
                FROM financial_batch_snapshots
                WHERE symbol = ? AND observed_at_epoch <= ?
                ORDER BY fiscal_year, quarter, observed_at_epoch, id
                """,
                (symbol, self._to_epoch(as_of)),
            ).fetchall()
        latest_by_period: dict[tuple[int, int], FundamentalSnapshot] = {}
        for row in rows:
            payload = json.loads(str(row[4]))
            latest_by_period[(int(row[0]), int(row[1]))] = FundamentalSnapshot(
                **payload
            )
        return [latest_by_period[period] for period in sorted(latest_by_period)]

    @staticmethod
    def filing_set_hash(filings: list[FiledFinancials]) -> str:
        """Hash all inputs that can change point-in-time fundamental output."""

        components = [
            "|".join(
                (
                    str(filing.fiscal_year),
                    str(filing.quarter),
                    str(filing.revision_number),
                    filing.available_at,
                    filing.availability_source,
                    filing.content_sha256 or "",
                    filing.correction_description or "",
                )
            )
            for filing in filings
        ]
        return hashlib.sha256("\n".join(components).encode("utf-8")).hexdigest()

    def load_cached_bundles(
        self,
        symbol_sectors: Mapping[str, str],
        *,
        as_of: str,
        derivation_version: str,
    ) -> dict[str, CachedFundamentalBundle]:
        """Load active snapshots in one query without touching normalized facts."""

        self._ensure_schema()
        requested = {
            str(symbol).strip(): str(sector).strip()
            for symbol, sector in symbol_sectors.items()
            if str(symbol).strip()
        }
        if not requested:
            return {}
        placeholders = ",".join("?" for _ in requested)
        query = f"""
            SELECT rs.symbol, rs.active_filing_set_sha256,
                   rs.active_derivation_version,
                   rs.active_snapshot_available_at_epoch,
                   rs.active_latest_fiscal_year, rs.active_latest_quarter,
                   rs.active_annual_history_years,
                   rs.full_catalog_checked_at_epoch,
                   fs.sector, fs.annual_json, fs.snapshot_json
            FROM refresh_state AS rs
            JOIN fundamental_snapshots AS fs
              ON fs.symbol = rs.symbol
             AND fs.filing_set_sha256 = rs.active_snapshot_cache_key
            WHERE rs.symbol IN ({placeholders})
              AND rs.active_derivation_version = ?
              AND rs.active_snapshot_available_at_epoch <= ?
        """
        params = [
            *requested,
            derivation_version,
            self._to_epoch(as_of),
        ]
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()

        bundles: dict[str, CachedFundamentalBundle] = {}
        for row in rows:
            symbol = str(row[0])
            stored_sector = str(row[8])
            if requested.get(symbol) != stored_sector:
                continue
            annual = pd.read_json(io.StringIO(str(row[9])), orient="records")
            snapshot = json.loads(str(row[10]))
            bundles[symbol] = CachedFundamentalBundle(
                symbol=symbol,
                sector=stored_sector,
                filing_set_sha256=str(row[1]),
                derivation_version=str(row[2]),
                available_at_epoch=int(row[3]),
                latest_period=(int(row[4]), int(row[5])),
                annual_history_years=int(row[6]),
                full_catalog_checked_at_epoch=int(row[7]),
                bundle=FundamentalBundle(
                    annual_fundamentals=annual,
                    snapshot=snapshot,
                ),
            )
        return bundles

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
        """Activate a snapshot only when catalog refresh state already exists."""

        self._ensure_schema()
        externally_managed = uow is not None or connection is not None
        with self._borrow_connection(connection, uow) as active_connection:
            with self._write_transaction(
                active_connection,
                externally_managed=externally_managed,
            ):
                active_connection.execute(
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
                        self._to_epoch(available_at),
                        latest_period[0],
                        latest_period[1],
                        annual_history_years,
                        symbol,
                    ),
                )

    def load_fundamental_bundle(
        self,
        symbol: str,
        filing_set_sha256: str,
        sector: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> FundamentalBundle | None:
        self._ensure_schema()
        with self._borrow_connection(connection, uow) as active_connection:
            row = active_connection.execute(
                """
                SELECT annual_json, snapshot_json
                FROM fundamental_snapshots
                WHERE symbol = ? AND filing_set_sha256 = ? AND sector = ?
                """,
                (symbol, filing_set_sha256, sector),
            ).fetchone()
        if row is None:
            return None
        annual = pd.read_json(io.StringIO(str(row[0])), orient="records")
        snapshot = json.loads(str(row[1]))
        return FundamentalBundle(annual_fundamentals=annual, snapshot=snapshot)

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
        with self._borrow_connection(connection, uow) as active_connection:
            with self._write_transaction(
                active_connection,
                externally_managed=externally_managed,
            ):
                active_connection.execute(
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

    def unit_of_work(self) -> SqliteFundamentalUnitOfWork:
        """Create one explicit transaction for all work on a company."""

        self._ensure_schema()
        return SqliteFundamentalUnitOfWork(self._connect())

    @contextmanager
    def company_connection(self) -> Iterator[sqlite3.Connection]:
        """Backward-compatible access to the unit-of-work connection."""

        with self.unit_of_work() as uow:
            yield uow.connection

    def raw_path(self, filing: MopsFiling) -> Path:
        return self.root / filing.raw_path

    def _ensure_schema(self) -> None:
        if self._schema_ready:
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
                for row in connection.execute("PRAGMA table_info(filings)").fetchall()
            }
            migrations = {
                "period_end": "ALTER TABLE filings ADD COLUMN period_end TEXT",
                "statement_scope": (
                    "ALTER TABLE filings ADD COLUMN statement_scope TEXT "
                    "NOT NULL DEFAULT 'consolidated'"
                ),
                "audit_status": "ALTER TABLE filings ADD COLUMN audit_status TEXT",
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
                "SELECT id, available_at FROM filings WHERE available_at_epoch IS NULL"
            ).fetchall()
            connection.executemany(
                "UPDATE filings SET available_at_epoch = ? WHERE id = ?",
                [
                    (self._to_epoch(str(available_at)), int(filing_id))
                    for filing_id, available_at in missing_epochs
                ],
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS filings_symbol_available_epoch
                ON filings(symbol, available_at_epoch, fiscal_year, quarter, revision_number)
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
        self._schema_ready = True

    @staticmethod
    def _remove_legacy_empty_state(connection: sqlite3.Connection) -> None:
        """Remove only artifacts created without an underlying filing."""

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
                "MOPS cache repair removed empty_snapshots=%s orphan_refresh_states=%s",
                removed_snapshots,
                removed_refresh_states,
            )

    @staticmethod
    def _invalidate_active_bundle(
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

    @contextmanager
    def _borrow_connection(
        self,
        connection: sqlite3.Connection | None,
        uow: FundamentalUnitOfWork | None,
    ) -> Iterator[sqlite3.Connection]:
        if connection is not None and uow is not None:
            raise ValueError("Pass either uow or connection, not both")
        if uow is not None:
            if not isinstance(uow, SqliteFundamentalUnitOfWork):
                raise TypeError(
                    "MopsRepository requires SqliteFundamentalUnitOfWork"
                )
            yield uow.connection
            return
        if connection is not None:
            yield connection
            return
        with closing(self._connect()) as owned_connection:
            yield owned_connection

    @staticmethod
    @contextmanager
    def _write_transaction(
        connection: sqlite3.Connection,
        *,
        externally_managed: bool,
    ) -> Iterator[None]:
        if externally_managed:
            yield
            return
        with connection:
            yield

    def _find_by_hash(
        self,
        connection: sqlite3.Connection,
        candidate: MopsFilingCandidate,
        content_sha256: str,
    ) -> MopsFiling | None:
        row = connection.execute(
            """
            SELECT id, symbol, fiscal_year, quarter, report_id, available_at,
                   fetched_at, revision_number, content_sha256, source_url, raw_path,
                   period_end, statement_scope, audit_status, correction_description,
                   availability_source, correction_announced_on
            FROM filings
            WHERE symbol = ? AND fiscal_year = ? AND quarter = ?
              AND report_id = ? AND content_sha256 = ?
            LIMIT 1
            """,
            (
                candidate.symbol,
                candidate.fiscal_year,
                candidate.quarter,
                candidate.report_id,
                content_sha256,
            ),
        ).fetchone()
        return self._row_to_filing(row)

    @staticmethod
    def _to_epoch(value: str | dt.datetime) -> int:
        if isinstance(value, str):
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            parsed = value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return int(parsed.timestamp())

    def _connect(self) -> sqlite3.Connection:
        last_error: OSError | sqlite3.OperationalError | None = None
        for attempt in range(1, self.connect_attempts + 1):
            connection: sqlite3.Connection | None = None
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                if not self.root.is_dir():
                    raise OSError("cache root is not a directory")
                connection = sqlite3.connect(self.database_path, timeout=30)
                connection.execute("PRAGMA busy_timeout = 30000")
                return connection
            except (OSError, sqlite3.OperationalError) as error:
                last_error = error
                if connection is not None:
                    connection.close()
                if attempt < self.connect_attempts:
                    self._sleep(0.5 * attempt)
        raise MopsStorageError(
            "Unable to open MOPS SQLite cache at {} after {} attempt(s): {}. "
            "Check that Transcend-Dev is mounted and writable, or set MOPS_CACHE_DIR "
            "to a writable local directory.".format(
                self.database_path,
                self.connect_attempts,
                last_error,
            )
        ) from last_error

    @staticmethod
    def _row_to_filing(row: tuple[object, ...] | None) -> MopsFiling | None:
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
            correction_description=str(row[14]) if row[14] is not None else None,
            availability_source=(
                str(row[15]) if len(row) > 15 and row[15] is not None else LEGACY_UNVERIFIED
            ),
            correction_announced_on=(
                str(row[16]) if len(row) > 16 and row[16] is not None else None
            ),
        )


__all__ = [
    "MopsRepository",
    "MopsStorageError",
    "SqliteFundamentalUnitOfWork",
]
