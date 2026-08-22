"""Compatibility facade for command/query-separated MOPS persistence."""

from __future__ import annotations

import datetime as dt
import sqlite3
import time
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

from application.fundamental_ports import FundamentalUnitOfWork
from config.settings import DATA_DIR
from config.settings import AppSettings
from domain.fundamental_derivation import CachedFundamentalBundle
from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FundamentalSnapshot
from domain.fundamental_models import NormalizedFact
from infrastructure.fundamentals.models import MopsFiling
from infrastructure.fundamentals.models import MopsFilingCandidate
from infrastructure.fundamentals.mops_commands import MopsRevisionWriter
from infrastructure.fundamentals.mops_queries import MopsQueryRepository
from infrastructure.fundamentals.mops_schema import EMPTY_FILING_SET_SHA256
from infrastructure.fundamentals.mops_schema import MopsSchemaMigrator
from infrastructure.fundamentals.mops_sqlite import MopsStorageError
from infrastructure.fundamentals.mops_sqlite import connect_sqlite
from infrastructure.fundamentals.mops_sqlite import invalidate_active_bundle
from infrastructure.fundamentals.mops_sqlite import row_to_filing
from infrastructure.fundamentals.mops_sqlite import to_epoch
from infrastructure.fundamentals.mops_unit_of_work import (
    SqliteFundamentalUnitOfWork,
)
from infrastructure.fundamentals.mops_unit_of_work import borrow_connection
from infrastructure.fundamentals.mops_unit_of_work import write_transaction


class MopsRepository:
    """Keep the legacy repository API while delegating reads and writes."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        connect_attempts: int | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        settings = (
            AppSettings.from_env(mops_enabled=False, validate_paths=False)
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
        if self.connect_attempts < 1:
            raise ValueError("connect_attempts must be at least one")

        # Resolve the facade method at call time so existing diagnostics can
        # monkeypatch _connect without bypassing either persistence side.
        self.schema_migrator = MopsSchemaMigrator(lambda: self._connect())
        self.queries = MopsQueryRepository(
            lambda: self._connect(),
            lambda: self._ensure_schema(),
        )
        self.revision_writer = MopsRevisionWriter(
            self.root,
            lambda: self._connect(),
            lambda: self._ensure_schema(),
            self.queries,
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
        return self.revision_writer.save_revision(
            candidate,
            content,
            facts,
            available_at=available_at,
            fetched_at=fetched_at,
            availability_source=availability_source,
            minimum_revision_number=minimum_revision_number,
            correction_description=correction_description,
            correction_announced_on=correction_announced_on,
            uow=uow,
            connection=connection,
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
        return self.revision_writer.update_availability_provenance(
            filing,
            available_at=available_at,
            availability_source=availability_source,
            uow=uow,
            connection=connection,
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
        return self.queries.latest_filing(
            symbol,
            fiscal_year,
            quarter,
            report_id,
            uow=uow,
            connection=connection,
        )

    def find_by_hash(
        self,
        candidate: MopsFilingCandidate,
        content_sha256: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> MopsFiling | None:
        return self.queries.find_by_hash(
            candidate,
            content_sha256,
            uow=uow,
            connection=connection,
        )

    def load_filings(
        self,
        symbol: str,
        *,
        as_of: str | None = None,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[MopsFiling]:
        return self.queries.load_filings(
            symbol,
            as_of=as_of,
            uow=uow,
            connection=connection,
        )

    def load_facts(
        self,
        filing_id: int,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[NormalizedFact]:
        return self.queries.load_facts(
            filing_id,
            uow=uow,
            connection=connection,
        )

    def load_filed_financials(
        self,
        symbol: str,
        as_of: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[FiledFinancials]:
        return self.queries.load_filed_financials(
            symbol,
            as_of,
            uow=uow,
            connection=connection,
        )

    def last_full_refresh_epoch(
        self,
        symbol: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> int | None:
        return self.queries.last_full_refresh_epoch(
            symbol,
            uow=uow,
            connection=connection,
        )

    def mark_full_refresh(
        self,
        symbol: str,
        refreshed_at: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self.revision_writer.mark_full_refresh(
            symbol,
            refreshed_at,
            uow=uow,
            connection=connection,
        )

    def last_recent_revision_check_epoch(
        self,
        symbol: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> int | None:
        return self.queries.last_recent_revision_check_epoch(
            symbol,
            uow=uow,
            connection=connection,
        )

    def mark_recent_revision_check(
        self,
        symbol: str,
        checked_at: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self.revision_writer.mark_recent_revision_check(
            symbol,
            checked_at,
            uow=uow,
            connection=connection,
        )

    def load_financial_batch_state(
        self,
        symbol: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[tuple[int, int], str] | None:
        return self.queries.load_financial_batch_state(
            symbol,
            uow=uow,
            connection=connection,
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
        self.revision_writer.save_financial_batch_state(
            symbol,
            fiscal_year,
            quarter,
            content_sha256,
            observed_at,
            uow=uow,
            connection=connection,
        )

    def save_financial_batch_snapshot(
        self,
        snapshot: FundamentalSnapshot,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self.revision_writer.save_financial_batch_snapshot(
            snapshot,
            uow=uow,
            connection=connection,
        )

    def load_financial_batch_snapshots(
        self,
        symbol: str,
        *,
        as_of: str,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[FundamentalSnapshot]:
        return self.queries.load_financial_batch_snapshots(
            symbol,
            as_of=as_of,
            uow=uow,
            connection=connection,
        )

    @staticmethod
    def filing_set_hash(filings: list[FiledFinancials]) -> str:
        return MopsQueryRepository.filing_set_hash(filings)

    def load_cached_bundles(
        self,
        symbol_sectors: Mapping[str, str],
        *,
        as_of: str,
        derivation_version: str,
    ) -> dict[str, CachedFundamentalBundle]:
        return self.queries.load_cached_bundles(
            symbol_sectors,
            as_of=as_of,
            derivation_version=derivation_version,
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
        self.revision_writer.mark_active_fundamental_bundle(
            symbol,
            filing_set_sha256=filing_set_sha256,
            snapshot_cache_key=snapshot_cache_key,
            derivation_version=derivation_version,
            available_at=available_at,
            latest_period=latest_period,
            annual_history_years=annual_history_years,
            uow=uow,
            connection=connection,
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
        return self.queries.load_fundamental_bundle(
            symbol,
            filing_set_sha256,
            sector,
            uow=uow,
            connection=connection,
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
        self.revision_writer.save_fundamental_bundle(
            symbol,
            filing_set_sha256,
            sector,
            bundle,
            uow=uow,
            connection=connection,
        )

    def unit_of_work(self) -> SqliteFundamentalUnitOfWork:
        self._ensure_schema()
        return SqliteFundamentalUnitOfWork(self._connect())

    @contextmanager
    def company_connection(self) -> Iterator[sqlite3.Connection]:
        with self.unit_of_work() as uow:
            yield uow.connection

    def raw_path(self, filing: MopsFiling) -> Path:
        return self.root / filing.raw_path

    def _ensure_schema(self) -> None:
        self.schema_migrator.ensure()

    @property
    def _schema_ready(self) -> bool:
        return self.schema_migrator.ready

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(
            self.root,
            self.database_path,
            self.connect_attempts,
            self._sleep,
        )

    def _find_by_hash(
        self,
        connection: sqlite3.Connection,
        candidate: MopsFilingCandidate,
        content_sha256: str,
    ) -> MopsFiling | None:
        return self.queries.find_by_hash_on_connection(
            connection,
            candidate,
            content_sha256,
        )

    @staticmethod
    def _to_epoch(value: str | dt.datetime) -> int:
        return to_epoch(value)

    @staticmethod
    def _row_to_filing(row: tuple[object, ...] | None) -> MopsFiling | None:
        return row_to_filing(row)

    @staticmethod
    def _invalidate_active_bundle(
        connection: sqlite3.Connection,
        symbol: str,
    ) -> None:
        invalidate_active_bundle(connection, symbol)

    @contextmanager
    def _borrow_connection(
        self,
        connection: sqlite3.Connection | None,
        uow: FundamentalUnitOfWork | None,
    ) -> Iterator[sqlite3.Connection]:
        with borrow_connection(self._connect, connection, uow) as active:
            yield active

    @staticmethod
    @contextmanager
    def _write_transaction(
        connection: sqlite3.Connection,
        *,
        externally_managed: bool,
    ) -> Iterator[None]:
        with write_transaction(
            connection,
            externally_managed=externally_managed,
        ):
            yield


__all__ = [
    "EMPTY_FILING_SET_SHA256",
    "MopsRepository",
    "MopsStorageError",
    "SqliteFundamentalUnitOfWork",
]
