"""Read-only SQLite query side of MOPS persistence."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from collections.abc import Callable
from collections.abc import Mapping
from contextlib import closing

import pandas as pd

from application.fundamental_ports import FundamentalUnitOfWork
from domain.fundamental_derivation import CachedFundamentalBundle
from domain.fundamental_derivation import FiledFact
from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FundamentalSnapshot
from domain.fundamental_models import NormalizedFact
from infrastructure.fundamentals.models import MopsFiling
from infrastructure.fundamentals.models import MopsFilingCandidate
from infrastructure.fundamentals.mops_sqlite import row_to_filing
from infrastructure.fundamentals.mops_sqlite import to_epoch
from infrastructure.fundamentals.mops_unit_of_work import borrow_connection


class MopsQueryRepository:
    """Execute point-in-time reads without owning any persistence command."""

    _BULK_SYMBOL_CHUNK_SIZE = 400

    def __init__(
        self,
        connect: Callable[[], sqlite3.Connection],
        ensure_schema: Callable[[], None],
    ) -> None:
        self._connect = connect
        self._ensure_schema = ensure_schema

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
        with borrow_connection(self._connect, connection, uow) as active:
            row = active.execute(
                """
                SELECT id, symbol, fiscal_year, quarter, report_id, available_at,
                       fetched_at, revision_number, content_sha256, source_url,
                       raw_path, period_end, statement_scope, audit_status,
                       correction_description, availability_source,
                       correction_announced_on
                FROM filings
                WHERE symbol = ? AND fiscal_year = ? AND quarter = ?
                  AND report_id = ?
                ORDER BY revision_number DESC
                LIMIT 1
                """,
                (symbol, fiscal_year, quarter, report_id),
            ).fetchone()
        return row_to_filing(row)

    def find_by_hash(
        self,
        candidate: MopsFilingCandidate,
        content_sha256: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> MopsFiling | None:
        self._ensure_schema()
        with borrow_connection(self._connect, connection, uow) as active:
            return self.find_by_hash_on_connection(
                active,
                candidate,
                content_sha256,
            )

    def find_by_hash_on_connection(
        self,
        connection: sqlite3.Connection,
        candidate: MopsFilingCandidate,
        content_sha256: str,
    ) -> MopsFiling | None:
        row = connection.execute(
            """
            SELECT id, symbol, fiscal_year, quarter, report_id, available_at,
                   fetched_at, revision_number, content_sha256, source_url,
                   raw_path, period_end, statement_scope, audit_status,
                   correction_description, availability_source,
                   correction_announced_on
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
        return row_to_filing(row)

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
                   fetched_at, revision_number, content_sha256, source_url,
                   raw_path, period_end, statement_scope, audit_status,
                   correction_description, availability_source,
                   correction_announced_on
            FROM filings WHERE symbol = ?
        """
        params: list[object] = [symbol]
        if as_of is not None:
            query += " AND available_at_epoch <= ?"
            params.append(to_epoch(as_of))
        query += (
            " ORDER BY fiscal_year, quarter, available_at_epoch, "
            "revision_number"
        )
        with borrow_connection(self._connect, connection, uow) as active:
            rows = active.execute(query, params).fetchall()
        return [
            filing
            for row in rows
            if (filing := row_to_filing(row)) is not None
        ]

    def load_facts(
        self,
        filing_id: int,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[NormalizedFact]:
        self._ensure_schema()
        with borrow_connection(self._connect, connection, uow) as active:
            rows = active.execute(
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
        self._ensure_schema()
        with borrow_connection(self._connect, connection, uow) as active:
            rows = active.execute(
                """
                SELECT f.id, f.symbol, f.fiscal_year, f.quarter, f.report_id,
                       f.available_at, f.fetched_at, f.revision_number,
                       f.content_sha256, f.source_url, f.raw_path, f.period_end,
                       f.statement_scope, f.audit_status,
                       f.correction_description, f.availability_source,
                       f.correction_announced_on, x.id, x.concept, x.context_id,
                       x.unit, x.numeric_value, x.text_value, x.start_date,
                       x.end_date, x.instant, x.dimensions_json
                FROM filings AS f
                LEFT JOIN facts AS x ON x.filing_id = f.id
                WHERE f.symbol = ? AND f.available_at_epoch <= ?
                ORDER BY f.fiscal_year, f.quarter, f.revision_number, x.id
                """,
                (symbol, to_epoch(as_of)),
            ).fetchall()

        grouped: dict[int, tuple[MopsFiling, list[FiledFact]]] = {}
        for row in rows:
            filing_id = int(row[0])
            if filing_id not in grouped:
                filing = row_to_filing(row[:17])
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
                        text_value=(
                            str(row[22]) if row[22] is not None else None
                        ),
                        start_date=(
                            str(row[23]) if row[23] is not None else None
                        ),
                        end_date=(
                            str(row[24]) if row[24] is not None else None
                        ),
                        instant=str(row[25]) if row[25] is not None else None,
                        dimensions=tuple(
                            tuple(str(value) for value in item)
                            for item in json.loads(str(row[26]))
                        ),
                        unit=str(row[20]) if row[20] is not None else None,
                        context_id=(
                            str(row[19]) if row[19] is not None else None
                        ),
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
        with borrow_connection(self._connect, connection, uow) as active:
            row = active.execute(
                "SELECT full_catalog_checked_at_epoch "
                "FROM refresh_state WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return int(row[0]) if row is not None else None

    def last_recent_revision_check_epoch(
        self,
        symbol: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> int | None:
        self._ensure_schema()
        with borrow_connection(self._connect, connection, uow) as active:
            row = active.execute(
                "SELECT recent_revision_checked_at_epoch "
                "FROM refresh_state WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return int(row[0]) if row is not None and row[0] is not None else None

    def load_financial_batch_state(
        self,
        symbol: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[tuple[int, int], str] | None:
        self._ensure_schema()
        with borrow_connection(self._connect, connection, uow) as active:
            row = active.execute(
                """
                SELECT fiscal_year, quarter, content_sha256
                FROM financial_batch_state WHERE symbol = ?
                """,
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return (int(row[0]), int(row[1])), str(row[2])

    def load_financial_batch_states(
        self,
        symbols: list[str],
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, tuple[tuple[int, int], str]]:
        normalized = list(dict.fromkeys(symbol for symbol in symbols if symbol))
        if not normalized:
            return {}
        self._ensure_schema()
        loaded: dict[str, tuple[tuple[int, int], str]] = {}
        with borrow_connection(self._connect, connection, uow) as active:
            for offset in range(0, len(normalized), self._BULK_SYMBOL_CHUNK_SIZE):
                chunk = normalized[offset : offset + self._BULK_SYMBOL_CHUNK_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                rows = active.execute(
                    """
                    SELECT symbol, fiscal_year, quarter, content_sha256
                    FROM financial_batch_state
                    WHERE symbol IN ({})
                    """.format(placeholders),
                    chunk,
                ).fetchall()
                for row in rows:
                    loaded[str(row[0])] = (
                        (int(row[1]), int(row[2])),
                        str(row[3]),
                    )
        return loaded

    def load_financial_batch_snapshots(
        self,
        symbol: str,
        *,
        as_of: str,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[FundamentalSnapshot]:
        self._ensure_schema()
        with borrow_connection(self._connect, connection, uow) as active:
            rows = active.execute(
                """
                SELECT fiscal_year, quarter, observed_at_epoch, id,
                       snapshot_json
                FROM financial_batch_snapshots
                WHERE symbol = ? AND observed_at_epoch <= ?
                ORDER BY fiscal_year, quarter, observed_at_epoch, id
                """,
                (symbol, to_epoch(as_of)),
            ).fetchall()
        latest_by_period: dict[tuple[int, int], FundamentalSnapshot] = {}
        for row in rows:
            payload = json.loads(str(row[4]))
            latest_by_period[(int(row[0]), int(row[1]))] = (
                FundamentalSnapshot(**payload)
            )
        return [latest_by_period[period] for period in sorted(latest_by_period)]

    def load_financial_batch_snapshots_many(
        self,
        symbols: list[str],
        *,
        as_of: str,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, list[FundamentalSnapshot]]:
        normalized = list(dict.fromkeys(symbol for symbol in symbols if symbol))
        if not normalized:
            return {}
        self._ensure_schema()
        grouped: dict[
            str,
            dict[tuple[int, int], FundamentalSnapshot],
        ] = {symbol: {} for symbol in normalized}
        with borrow_connection(self._connect, connection, uow) as active:
            for offset in range(0, len(normalized), self._BULK_SYMBOL_CHUNK_SIZE):
                chunk = normalized[offset : offset + self._BULK_SYMBOL_CHUNK_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                rows = active.execute(
                    """
                    SELECT symbol, fiscal_year, quarter, observed_at_epoch, id,
                           snapshot_json
                    FROM financial_batch_snapshots
                    WHERE symbol IN ({}) AND observed_at_epoch <= ?
                    ORDER BY symbol, fiscal_year, quarter, observed_at_epoch, id
                    """.format(placeholders),
                    [*chunk, to_epoch(as_of)],
                ).fetchall()
                for row in rows:
                    symbol = str(row[0])
                    payload = json.loads(str(row[5]))
                    grouped[symbol][(int(row[1]), int(row[2]))] = (
                        FundamentalSnapshot(**payload)
                    )
        return {
            symbol: [periods[period] for period in sorted(periods)]
            for symbol, periods in grouped.items()
        }

    @staticmethod
    def filing_set_hash(filings: list[FiledFinancials]) -> str:
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
        return hashlib.sha256(
            "\n".join(components).encode("utf-8")
        ).hexdigest()

    def load_cached_bundles(
        self,
        symbol_sectors: Mapping[str, str],
        *,
        as_of: str,
        derivation_version: str,
    ) -> dict[str, CachedFundamentalBundle]:
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
        params = [*requested, derivation_version, to_epoch(as_of)]
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
        with borrow_connection(self._connect, connection, uow) as active:
            row = active.execute(
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
        return FundamentalBundle(
            annual_fundamentals=annual,
            snapshot=snapshot,
        )


__all__ = ["MopsQueryRepository"]
