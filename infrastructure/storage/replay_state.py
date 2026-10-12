"""Transactional, job-isolated PAE research state; never a live portfolio DB."""
from __future__ import annotations

import json
import sqlite3
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from contextlib import closing

from application.replay_contracts import ReplayRequest, aware_timestamp, canonical_json, digest
from domain.strategies.portfolio_allocation_v2 import PersistenceEvidenceV2

APPLICATION_ID = 0x52504C59


class ReplaySession:
    def __init__(self, connection, request, sessions, previous, cached, result_store=None):
        self.connection, self.request, self.sessions = connection, request, sessions
        self.previous_result_sha256, self.cached = previous, cached
        self.observations = []
        self.reads = []
        self.finished = False
        self.result_store = result_store

    def record_session_v2(self, trade_date, observations):
        if trade_date.isoformat() not in self.sessions:
            raise ValueError("PAE_OBSERVATION_NOT_A_TRADING_SESSION")
        # A weekend review must not count again or rewrite Friday's observation.
        if trade_date != self.request.as_of.date():
            return
        self.observations = [asdict(item) for item in observations]
        self.connection.executemany("""INSERT INTO replay_observations
            (replay_id, trade_date, incumbent, challenger, passed, as_of)
            VALUES (?,?,?,?,?,?)""", [
                (self.request.replay_id, trade_date.isoformat(), item.incumbent_stock_id,
                 item.challenger_stock_id, int(item.passed), self.request.as_of.isoformat()) for item in observations])

    def load_recent_v2(self, incumbent_symbols, challenger_symbols, *, through_date, limit_per_pair):
        dates = [day for day in self.sessions if day <= through_date.isoformat()][-limit_per_pair:]
        rows = self.connection.execute("""SELECT trade_date,incumbent,challenger,passed
            FROM replay_observations WHERE replay_id=? AND trade_date>=? AND trade_date<=? AND as_of<=?
            ORDER BY trade_date,incumbent,challenger""",
            (self.request.replay_id, dates[0] if dates else '9999', through_date.isoformat(), self.request.as_of.isoformat())).fetchall()
        by_pair = {}
        for day, incumbent, challenger, passed in rows:
            if day in dates:
                by_pair.setdefault((incumbent, challenger), {})[day] = bool(passed)
        # Missing evaluated sessions are not compressed out of the native N-of-M
        # window, and are never represented as successful persistence.
        evidence = tuple(PersistenceEvidenceV2(i, c, tuple(by_pair.get((i,c), {}).get(day, False) for day in dates))
                         for i in sorted(incumbent_symbols) for c in sorted(challenger_symbols))
        self.reads = [dict(asdict(item), trading_dates=dates) for item in evidence]
        return evidence

    def finish(self, result):
        serialized = canonical_json(result)
        if self.result_store is not None:
            serialized = canonical_json(dict(schema='replay-result-ref/v1', sha256=self.result_store.put_json(result)))
        self.connection.execute("INSERT INTO replay_sessions VALUES (?,?,?,?,?)",
            (self.request.replay_id, self.request.as_of.isoformat(), digest(self.request.inputs()), digest(result), serialized))
        self.finished = True


class SQLiteReplayState:
    def __init__(self, *, result_store=None):
        self.result_store = result_store

    def _result(self, serialized):
        value = json.loads(serialized)
        if value.get('schema') == 'replay-result-ref/v1':
            if self.result_store is None:
                raise ValueError('REPLAY_RESULT_OBJECT_STORE_REQUIRED')
            return self.result_store.get_json(value['sha256'])
        return value

    @staticmethod
    def empty_snapshot(request: ReplayRequest) -> dict:
        body = dict(schema="replay-state/v1", replay_id=request.replay_id,
                    identity=request.identity(), sessions=[], observations=[])
        return dict(body, state_id=digest(body))

    def export_snapshot(self, request: ReplayRequest, *, before: str | None = None) -> dict:
        """Consistent logical read transaction. Never copy an open SQLite file."""
        if not request.state_path.exists():
            return self.empty_snapshot(request)
        with closing(sqlite3.connect(request.state_path.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
                raise ValueError("REPLAY_STATE_MUST_NOT_USE_LIVE_OR_FOREIGN_DATABASE")
            row = connection.execute("SELECT identity_json FROM replay_runs WHERE replay_id=?", (request.replay_id,)).fetchone()
            if row is None:
                return self.empty_snapshot(request)
            if json.loads(row[0]) != request.identity():
                raise ValueError("REPLAY_VERSIONS_DATA_OR_UNIVERSE_CHANGED")
            bound = before or "9999"
            sessions = [list(row) for row in connection.execute(
                "SELECT as_of,input_hash,result_hash,result_json FROM replay_sessions WHERE replay_id=? AND as_of<? ORDER BY as_of",
                (request.replay_id, bound))]
            observations = [list(row) for row in connection.execute(
                "SELECT trade_date,incumbent,challenger,passed,as_of FROM replay_observations WHERE replay_id=? AND as_of<? ORDER BY trade_date,incumbent,challenger",
                (request.replay_id, bound))]
        body = dict(schema="replay-state/v1", replay_id=request.replay_id,
                    identity=request.identity(), sessions=sessions, observations=observations)
        for row in sessions:
            result = self._result(row[3])
            if digest(result) != row[2]:
                raise ValueError("REPLAY_STATE_RESULT_HASH_MISMATCH")
            row[3] = canonical_json(result)
        return dict(body, state_id=digest(body))

    def restore_snapshot(self, request: ReplayRequest, snapshot: dict) -> None:
        """Explicit fresh research fork, retaining exact version/evidence identity.

        Restoration never rebases old persistence onto a different dataset or
        model. Recompute predecessors for those changes, just like strict replay.
        """
        body = {key: value for key, value in snapshot.items() if key != "state_id"}
        if snapshot.get("state_id") != digest(body) or body.get("schema") != "replay-state/v1":
            raise ValueError("REPLAY_STATE_SNAPSHOT_HASH_MISMATCH")
        if body["identity"] != request.identity() or body["replay_id"] != request.replay_id:
            raise ValueError("REPLAY_PREDECESSOR_IDENTITY_MISMATCH")
        for row in body["sessions"]:
            if aware_timestamp(row[0]) >= request.as_of or digest(json.loads(row[3])) != row[2]:
                raise ValueError("INVALID_PREDECESSOR_SESSION")
        times = {r[0] for r in body["sessions"]}
        if any(r[4] not in times for r in body["observations"]):
            raise ValueError("ORPHAN_PREDECESSOR_OBSERVATION")
        if any(r[0] != aware_timestamp(r[4]).date().isoformat() or r[3] not in (0, 1)
               for r in body["observations"]):
            raise ValueError("INVALID_PREDECESSOR_OBSERVATION")
        if request.state_path.exists():
            if self.export_snapshot(request) == snapshot:
                return
            raise ValueError("RESTORE_REQUIRES_EMPTY_RESEARCH_DESTINATION")
        # Publish only a fully committed database. Interrupted restoration cannot
        # leave an empty destination that prevents safe retry.
        request.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".restore-", dir=request.state_path.parent)
        os.close(fd)
        try:
            with closing(sqlite3.connect(temporary)) as connection:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    self._initialize(connection)
                    connection.execute("INSERT INTO replay_runs VALUES (?,?)", (request.replay_id, canonical_json(request.identity())))
                    connection.executemany("INSERT INTO replay_sessions VALUES (?,?,?,?,?)",
                        [(request.replay_id, *row) for row in body["sessions"]])
                    connection.executemany("INSERT INTO replay_observations VALUES (?,?,?,?,?,?)",
                        [(request.replay_id, *row) for row in body["observations"]])
            os.link(temporary, request.state_path)
        finally:
            os.unlink(temporary)

    @staticmethod
    def _initialize(connection):
        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        connection.execute("CREATE TABLE IF NOT EXISTS replay_runs (replay_id TEXT PRIMARY KEY, identity_json TEXT NOT NULL)")
        connection.execute("""CREATE TABLE IF NOT EXISTS replay_sessions (
            replay_id TEXT NOT NULL, as_of TEXT NOT NULL, input_hash TEXT NOT NULL,
            result_hash TEXT NOT NULL, result_json TEXT NOT NULL, PRIMARY KEY(replay_id,as_of))""")
        connection.execute("""CREATE TABLE IF NOT EXISTS replay_observations (
            replay_id TEXT NOT NULL, trade_date TEXT NOT NULL, incumbent TEXT NOT NULL,
            challenger TEXT NOT NULL, passed INTEGER NOT NULL, as_of TEXT NOT NULL,
            PRIMARY KEY(replay_id,trade_date,incumbent,challenger))""")

    @contextmanager
    def session(self, request: ReplayRequest, trading_sessions: tuple[str, ...]):
        request.state_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(request.state_path, timeout=30)
        try:
            connection.execute("BEGIN IMMEDIATE")
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            app_id = connection.execute("PRAGMA application_id").fetchone()[0]
            if tables and (app_id != APPLICATION_ID or not tables <= {"replay_runs", "replay_sessions", "replay_observations"}):
                raise ValueError("REPLAY_STATE_MUST_NOT_USE_LIVE_OR_FOREIGN_DATABASE")
            self._initialize(connection)
            identity = canonical_json(request.identity())
            row = connection.execute("SELECT identity_json FROM replay_runs WHERE replay_id=?", (request.replay_id,)).fetchone()
            if row and row[0] != identity:
                raise ValueError("REPLAY_VERSIONS_DATA_OR_UNIVERSE_CHANGED: start a new replay_id")
            connection.execute("INSERT OR IGNORE INTO replay_runs VALUES (?,?)", (request.replay_id, identity))
            latest = connection.execute("SELECT as_of,input_hash,result_hash,result_json FROM replay_sessions WHERE replay_id=? ORDER BY as_of DESC LIMIT 1",
                                        (request.replay_id,)).fetchone()
            cached = None
            if latest:
                last_result = self._result(latest[3])
                if digest(last_result) != latest[2]:
                    raise ValueError("REPLAY_STATE_RESULT_HASH_MISMATCH")
                if request.as_of.isoformat() < latest[0]:
                    raise ValueError("REPLAY_OUT_OF_ORDER: start a new replay_id to recompute predecessors")
                if request.as_of.isoformat() == latest[0]:
                    if digest(request.inputs()) != latest[1]:
                        raise ValueError("REPLAY_PREDECESSOR_INPUT_CHANGED: start a new replay_id")
                    cached = last_result
                    if digest(cached) != latest[2]:
                        raise ValueError("REPLAY_STATE_RESULT_HASH_MISMATCH")
            session = ReplaySession(connection, request, trading_sessions, latest[2] if latest else None, cached,
                                    result_store=self.result_store)
            yield session
            if session.cached is None and not session.finished:
                raise ValueError("REPLAY_SESSION_NOT_FINISHED")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
