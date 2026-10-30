"""SQLite persistence for PAE v1's cross-session switch evidence."""

from __future__ import annotations

import datetime
import sqlite3
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

from application.portfolio_contracts import PersistenceObservation
from application.portfolio_contracts import PortfolioPersistenceError
from domain.strategies.portfolio_allocation_v1 import PAE_V1_MODEL_VERSION
from domain.strategies.portfolio_allocation_v1 import PersistenceEvidence
from domain.strategies.portfolio_allocation_v2 import PAE_V2_MODEL_VERSION
from domain.strategies.portfolio_allocation_v2 import PersistenceEvidenceV2
from domain.strategies.portfolio_allocation_v2 import PersistenceObservationV2


class SQLitePortfolioPersistence:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def load_recent(
        self,
        incumbent_symbols: Sequence[str],
        challenger_symbols: Sequence[str],
        *,
        through_date: datetime.date,
        limit_per_pair: int,
    ) -> tuple[PersistenceEvidence, ...]:
        incumbents = tuple(dict.fromkeys(str(item) for item in incumbent_symbols))
        challengers = tuple(dict.fromkeys(str(item) for item in challenger_symbols))
        if not incumbents or not challengers or limit_per_pair < 1:
            return ()
        incumbent_marks = ",".join("?" for _ in incumbents)
        challenger_marks = ",".join("?" for _ in challengers)
        query = """
            SELECT trade_date, incumbent_symbol, challenger_symbol,
                   excess_opportunity
            FROM pae_opportunity_evidence
            WHERE model_version = ?
              AND trade_date <= ?
              AND incumbent_symbol IN ({})
              AND challenger_symbol IN ({})
            ORDER BY trade_date, incumbent_symbol, challenger_symbol
        """.format(incumbent_marks, challenger_marks)
        try:
            with sqlite3.connect(self.database_path) as connection:
                self._assert_live_database(connection)
                rows = connection.execute(
                    query,
                    (
                        PAE_V1_MODEL_VERSION,
                        through_date.isoformat(),
                        *incumbents,
                        *challengers,
                    ),
                ).fetchall()
        except sqlite3.Error as error:
            raise PortfolioPersistenceError(
                "PAE_PERSISTENCE_READ_FAILED: {}".format(error)
            ) from error

        grouped: dict[tuple[str, str], list[float]] = {}
        for _, incumbent, challenger, excess in rows:
            grouped.setdefault((incumbent, challenger), []).append(float(excess))
        return tuple(
            PersistenceEvidence(
                incumbent_symbol=incumbent,
                challenger_symbol=challenger,
                recent_excess_opportunities=tuple(values[-limit_per_pair:]),
            )
            for (incumbent, challenger), values in sorted(grouped.items())
        )

    def record_session(
        self,
        trade_date: datetime.date,
        observations: Sequence[PersistenceObservation],
    ) -> None:
        if not observations:
            return
        rows = [
            (
                trade_date.isoformat(),
                observation.incumbent_symbol,
                observation.challenger_symbol,
                float(observation.excess_opportunity),
                PAE_V1_MODEL_VERSION,
            )
            for observation in observations
        ]
        try:
            with sqlite3.connect(self.database_path) as connection:
                self._assert_live_database(connection)
                connection.executemany(
                    """
                    INSERT INTO pae_opportunity_evidence (
                        trade_date,
                        incumbent_symbol,
                        challenger_symbol,
                        excess_opportunity,
                        model_version
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(
                        trade_date,
                        incumbent_symbol,
                        challenger_symbol,
                        model_version
                    ) DO UPDATE SET
                        excess_opportunity = excluded.excess_opportunity
                    """,
                    rows,
                )
        except sqlite3.Error as error:
            raise PortfolioPersistenceError(
                "PAE_PERSISTENCE_WRITE_FAILED: {}".format(error)
            ) from error

    def load_recent_v2(
        self,
        incumbent_symbols: Sequence[str],
        challenger_symbols: Sequence[str],
        *,
        through_date: datetime.date,
        limit_per_pair: int,
    ) -> tuple[PersistenceEvidenceV2, ...]:
        rows = self._load_rows(
            incumbent_symbols,
            challenger_symbols,
            through_date=through_date,
            limit_per_pair=limit_per_pair,
            model_version=PAE_V2_MODEL_VERSION,
        )
        return tuple(
            PersistenceEvidenceV2(
                incumbent_stock_id=incumbent,
                challenger_stock_id=challenger,
                recent_passes=tuple(bool(value) for value in values),
            )
            for (incumbent, challenger), values in sorted(rows.items())
        )

    def record_session_v2(
        self,
        trade_date: datetime.date,
        observations: Sequence[PersistenceObservationV2],
    ) -> None:
        self._record_rows(
            trade_date,
            (
                (
                    item.incumbent_stock_id,
                    item.challenger_stock_id,
                    1.0 if item.passed else 0.0,
                )
                for item in observations
            ),
            model_version=PAE_V2_MODEL_VERSION,
        )

    def _load_rows(
        self,
        incumbent_symbols: Sequence[str],
        challenger_symbols: Sequence[str],
        *,
        through_date: datetime.date,
        limit_per_pair: int,
        model_version: str,
    ) -> dict[tuple[str, str], list[float]]:
        incumbents = tuple(dict.fromkeys(str(item) for item in incumbent_symbols))
        challengers = tuple(dict.fromkeys(str(item) for item in challenger_symbols))
        if not incumbents or not challengers or limit_per_pair < 1:
            return {}
        incumbent_marks = ",".join("?" for _ in incumbents)
        challenger_marks = ",".join("?" for _ in challengers)
        query = """
            SELECT trade_date, incumbent_symbol, challenger_symbol,
                   excess_opportunity
            FROM pae_opportunity_evidence
            WHERE model_version = ?
              AND trade_date <= ?
              AND incumbent_symbol IN ({})
              AND challenger_symbol IN ({})
            ORDER BY trade_date, incumbent_symbol, challenger_symbol
        """.format(incumbent_marks, challenger_marks)
        try:
            with sqlite3.connect(self.database_path) as connection:
                self._assert_live_database(connection)
                records = connection.execute(
                    query,
                    (
                        model_version,
                        through_date.isoformat(),
                        *incumbents,
                        *challengers,
                    ),
                ).fetchall()
        except sqlite3.Error as error:
            raise PortfolioPersistenceError(
                "PAE_PERSISTENCE_READ_FAILED: {}".format(error)
            ) from error
        grouped: dict[tuple[str, str], list[float]] = {}
        for _, incumbent, challenger, value in records:
            grouped.setdefault((incumbent, challenger), []).append(float(value))
        return {
            pair: values[-limit_per_pair:] for pair, values in grouped.items()
        }

    def _record_rows(
        self,
        trade_date: datetime.date,
        observations: Iterable[tuple[str, str, float]],
        *,
        model_version: str,
    ) -> None:
        rows = [
            (
                trade_date.isoformat(),
                incumbent,
                challenger,
                float(value),
                model_version,
            )
            for incumbent, challenger, value in observations
        ]
        if not rows:
            return
        try:
            with sqlite3.connect(self.database_path) as connection:
                self._assert_live_database(connection)
                connection.executemany(
                    """
                    INSERT INTO pae_opportunity_evidence (
                        trade_date,
                        incumbent_symbol,
                        challenger_symbol,
                        excess_opportunity,
                        model_version
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(
                        trade_date,
                        incumbent_symbol,
                        challenger_symbol,
                        model_version
                    ) DO UPDATE SET
                        excess_opportunity = excluded.excess_opportunity
                    """,
                    rows,
                )
        except sqlite3.Error as error:
            raise PortfolioPersistenceError(
                "PAE_PERSISTENCE_WRITE_FAILED: {}".format(error)
            ) from error

    def _initialize(self) -> None:
        try:
            with sqlite3.connect(self.database_path) as connection:
                self._assert_live_database(connection)
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pae_opportunity_evidence (
                        trade_date TEXT NOT NULL,
                        incumbent_symbol TEXT NOT NULL,
                        challenger_symbol TEXT NOT NULL,
                        excess_opportunity REAL NOT NULL,
                        model_version TEXT NOT NULL,
                        PRIMARY KEY (
                            trade_date,
                            incumbent_symbol,
                            challenger_symbol,
                            model_version
                        )
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_pae_pair_date
                    ON pae_opportunity_evidence (
                        model_version,
                        incumbent_symbol,
                        challenger_symbol,
                        trade_date
                    )
                    """
                )
        except sqlite3.Error as error:
            raise PortfolioPersistenceError(
                "PAE_PERSISTENCE_INIT_FAILED: {}".format(error)
            ) from error

    @staticmethod
    def _assert_live_database(connection: sqlite3.Connection) -> None:
        # Research and foreign stores must never be reused as ordinary state,
        # including after an existing repository object has been redirected.
        if connection.execute("PRAGMA application_id").fetchone()[0] != 0:
            raise PortfolioPersistenceError(
                "PAE_PERSISTENCE_RESEARCH_OR_FOREIGN_DATABASE_FORBIDDEN"
            )


__all__ = ["SQLitePortfolioPersistence"]
