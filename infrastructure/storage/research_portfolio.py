"""Identity-bound, transactional PAE persistence for current-data research.

This store cannot open the ordinary portfolio database. It preserves native
pair-history semantics while sealing one complete observation set per data day.
Account-sensitive allocation is recomputed on every invocation, never cached.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
import datetime as dt
import json
from pathlib import Path
import sqlite3

from application.replay_contracts import canonical_json, digest
from domain.strategies.portfolio_allocation_v2 import (
    PAE_V2_MODEL_VERSION, PersistenceEvidenceV2, PersistenceObservationV2,
)


APPLICATION_ID = 0x52504145
SCHEMA_VERSION = 1
TABLES = frozenset({"research_identity", "research_sessions", "pae_opportunity_evidence"})


class ResearchStateError(RuntimeError):
    """Research identity/order/integrity failure; never downgrade to PAE UNKNOWN."""


def _day(value):
    if type(value) is not dt.date:
        raise ResearchStateError("RESEARCH_PAE_INVALID_SESSION_DATE")
    return value.isoformat()


def _observation_rows(observations):
    rows, pairs = [], set()
    for item in observations:
        if (not isinstance(item, PersistenceObservationV2)
                or not isinstance(item.incumbent_stock_id, str) or not item.incumbent_stock_id
                or not isinstance(item.challenger_stock_id, str) or not item.challenger_stock_id
                or type(item.passed) is not bool):
            raise ResearchStateError("RESEARCH_PAE_INVALID_OBSERVATION")
        pair = item.incumbent_stock_id, item.challenger_stock_id
        if pair in pairs:
            raise ResearchStateError("RESEARCH_PAE_DUPLICATE_OBSERVATION_PAIR")
        pairs.add(pair)
        rows.append(dict(incumbent=pair[0], challenger=pair[1], passed=item.passed))
    return sorted(rows, key=lambda row: (row["incumbent"], row["challenger"]))


class _ResearchSession:
    def __init__(self, connection, trade_date, symbols=None):
        self.connection, self.trade_date = connection, trade_date
        self.symbols = symbols
        self.recorded = False
        self._verified_days = set()

    def _verify_day(self, day):
        if day in self._verified_days:
            return
        header = self.connection.execute(
            "SELECT observations_sha256,observation_count FROM research_sessions WHERE trade_date=?",
            (day,)).fetchone()
        actual = []
        for incumbent, challenger, value, version in self.connection.execute(
                "SELECT incumbent_symbol,challenger_symbol,excess_opportunity,model_version "
                "FROM pae_opportunity_evidence WHERE trade_date=? "
                "ORDER BY incumbent_symbol,challenger_symbol,model_version", (day,)):
            if version != PAE_V2_MODEL_VERSION or value not in (0., 1.):
                raise ResearchStateError("RESEARCH_PAE_OBSERVATION_CORRUPT")
            actual.append(dict(incumbent=incumbent, challenger=challenger, passed=bool(value)))
        if header is None or header != (digest(actual), len(actual)):
            raise ResearchStateError("RESEARCH_PAE_SESSION_HASH_MISMATCH")
        self._verified_days.add(day)

    def record_session_v2(self, trade_date, observations):
        day = _day(trade_date)
        if trade_date != self.trade_date:
            raise ResearchStateError("RESEARCH_PAE_SESSION_DATE_MISMATCH")
        rows = _observation_rows(observations)
        if self.symbols is not None and any(
                row["incumbent"] not in self.symbols or row["challenger"] not in self.symbols for row in rows):
            raise ResearchStateError("RESEARCH_PAE_OBSERVATION_OUTSIDE_FROZEN_UNIVERSE")
        expected = digest(rows), len(rows)
        previous = self.connection.execute(
            "SELECT observations_sha256,observation_count FROM research_sessions WHERE trade_date=?",
            (day,)).fetchone()
        if previous is not None:
            self._verify_day(day)
            if previous != expected:
                raise ResearchStateError("RESEARCH_PAE_SAME_DAY_OBSERVATIONS_CHANGED")
        else:
            if self.connection.execute(
                    "SELECT 1 FROM pae_opportunity_evidence WHERE trade_date=? LIMIT 1", (day,)
                    ).fetchone():
                raise ResearchStateError("RESEARCH_PAE_ORPHAN_OBSERVATION")
            self.connection.execute("INSERT INTO research_sessions VALUES (?,?,?)", (day, *expected))
            self.connection.executemany(
                "INSERT INTO pae_opportunity_evidence VALUES (?,?,?,?,?)",
                [(day, row["incumbent"], row["challenger"], float(row["passed"]), PAE_V2_MODEL_VERSION)
                 for row in rows])
            self._verified_days.add(day)
        self.recorded = True

    def load_recent_v2(self, incumbent_symbols: Sequence[str], challenger_symbols: Sequence[str],
                       *, through_date: dt.date, limit_per_pair: int):
        day = _day(through_date)
        if through_date > self.trade_date:
            raise ResearchStateError("RESEARCH_PAE_FUTURE_HISTORY_REQUESTED")
        incumbents = tuple(dict.fromkeys(str(item) for item in incumbent_symbols))
        challengers = tuple(dict.fromkeys(str(item) for item in challenger_symbols))
        if not incumbents or not challengers or limit_per_pair < 1:
            return ()
        query = """SELECT trade_date,incumbent_symbol,challenger_symbol,excess_opportunity
            FROM pae_opportunity_evidence WHERE model_version=? AND trade_date<=?
              AND incumbent_symbol IN ({}) AND challenger_symbol IN ({})
            ORDER BY trade_date,incumbent_symbol,challenger_symbol""".format(
                ",".join("?" for _ in incumbents), ",".join("?" for _ in challengers))
        grouped = {}
        for observed, incumbent, challenger, value in self.connection.execute(
                query, (PAE_V2_MODEL_VERSION, day, *incumbents, *challengers)):
            grouped.setdefault((incumbent, challenger), []).append((observed, value))
        recent = {pair: values[-limit_per_pair:] for pair, values in grouped.items()}
        for observed in sorted({item[0] for values in recent.values() for item in values}):
            self._verify_day(observed)
        # Match the ordinary native port: latest N recorded observations per
        # pair, not a newly introduced calendar-filled N-session policy.
        return tuple(PersistenceEvidenceV2(
            incumbent_stock_id=incumbent, challenger_stock_id=challenger,
            recent_passes=tuple(bool(value) for _, value in values))
            for (incumbent, challenger), values in sorted(recent.items()))


class SQLiteResearchPortfolioPersistence:
    def __init__(self, database_path: str | Path, identity: dict, *, check_version=None):
        if type(identity) is not dict or not identity:
            raise ResearchStateError("RESEARCH_PAE_IDENTITY_REQUIRED")
        self.database_path = Path(database_path).absolute()
        self.identity_json = canonical_json(identity)
        detached = json.loads(self.identity_json)
        self.identity_sha256 = digest(detached)
        self.symbols = None
        if "symbols" in detached:
            symbols = detached["symbols"]
            if (not isinstance(symbols, list) or not symbols
                    or any(not isinstance(item, str) or not item for item in symbols)
                    or len(set(symbols)) != len(symbols)):
                raise ResearchStateError("RESEARCH_PAE_IDENTITY_SYMBOLS_INVALID")
            self.symbols = frozenset(symbols)
        self.check_version = check_version or (lambda: None)
        self._check_existing()

    @staticmethod
    def _status(connection):
        tables = frozenset(row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
        app = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if not tables and app == 0 and version == 0:
            return False
        if tables != TABLES or app != APPLICATION_ID or version != SCHEMA_VERSION:
            raise ResearchStateError("RESEARCH_PAE_MUST_NOT_USE_LIVE_OR_FOREIGN_DATABASE")
        return True

    def _assert_identity(self, connection):
        rows = connection.execute(
            "SELECT singleton,identity_json,identity_sha256 FROM research_identity").fetchall()
        if rows != [(1, self.identity_json, self.identity_sha256)]:
            raise ResearchStateError("RESEARCH_PAE_VERSION_MODEL_OR_UNIVERSE_CHANGED")

    def _check_existing(self):
        if self.database_path.is_symlink():
            raise ResearchStateError("RESEARCH_PAE_SYMLINK_DATABASE_FORBIDDEN")
        if not self.database_path.exists():
            return
        try:
            with closing(sqlite3.connect(self.database_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
                if self._status(connection):
                    self._assert_identity(connection)
        except sqlite3.Error as error:
            raise ResearchStateError("RESEARCH_PAE_STATE_READ_FAILED") from error

    def _initialize_or_verify(self, connection):
        if self._status(connection):
            self._assert_identity(connection)
            return
        # Individual statements retain the caller's BEGIN IMMEDIATE; unlike
        # executescript(), none implicitly commits a partially created store.
        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        connection.execute("CREATE TABLE research_identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                           "identity_json TEXT NOT NULL,identity_sha256 TEXT NOT NULL)")
        connection.execute("CREATE TABLE research_sessions (trade_date TEXT PRIMARY KEY, "
                           "observations_sha256 TEXT NOT NULL,observation_count INTEGER NOT NULL)")
        connection.execute("""CREATE TABLE pae_opportunity_evidence (
            trade_date TEXT NOT NULL,incumbent_symbol TEXT NOT NULL,challenger_symbol TEXT NOT NULL,
            excess_opportunity REAL NOT NULL CHECK(excess_opportunity IN (0,1)),model_version TEXT NOT NULL,
            PRIMARY KEY(trade_date,incumbent_symbol,challenger_symbol,model_version))""")
        connection.execute("CREATE INDEX idx_pae_pair_date ON pae_opportunity_evidence "
                           "(model_version,incumbent_symbol,challenger_symbol,trade_date)")
        connection.execute("INSERT INTO research_identity VALUES (1,?,?)", (self.identity_json, self.identity_sha256))

    def _check_version(self):
        try:
            self.check_version()
        except ResearchStateError:
            raise
        except Exception as error:
            raise ResearchStateError("RESEARCH_PAE_VERSION_CHANGED_DURING_SESSION") from error

    def _check_request_scope(self, request):
        if self.symbols is None or request is None:
            return
        try:
            securities = (*request.holdings, *request.candidates)
            if any(item.stock_id not in self.symbols for item in securities):
                raise ResearchStateError("RESEARCH_PAE_REQUEST_OUTSIDE_FROZEN_UNIVERSE")
        except (AttributeError, TypeError) as error:
            raise ResearchStateError("RESEARCH_PAE_REQUEST_SCOPE_UNVERIFIED") from error

    def evaluate_session_v2(self, request, trade_date: dt.date, *, evaluate):
        day = _day(trade_date)
        self._check_existing()  # Reject foreign DBs before any write/DDL.
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path, timeout=30)) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._initialize_or_verify(connection)
                self._check_request_scope(request)
                latest = connection.execute("SELECT MAX(trade_date) FROM research_sessions").fetchone()[0]
                if latest is not None and day < latest:
                    raise ResearchStateError("RESEARCH_PAE_OUT_OF_ORDER")
                self._check_version()
                session = _ResearchSession(connection, trade_date, self.symbols)
                result = evaluate(session)
                if not session.recorded:
                    session.record_session_v2(trade_date, ())
                self._check_version()
                connection.commit()
                return result
            except BaseException as error:
                connection.rollback()
                if isinstance(error, sqlite3.Error):
                    raise ResearchStateError("RESEARCH_PAE_STATE_TRANSACTION_FAILED") from error
                raise


__all__ = ["SQLiteResearchPortfolioPersistence", "ResearchStateError"]
