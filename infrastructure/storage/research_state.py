"""Incremental archive state, with the existing transactional PAE state engine.

The SQLite file is a rebuildable index. Immutable daily deltas are the public
state contract; full result bodies live once in the content-addressed store.
"""
from contextlib import closing
import json
import sqlite3

from application.replay_contracts import canonical_json, digest
from infrastructure.storage.replay_state import SQLiteReplayState, APPLICATION_ID


class IncrementalResearchState(SQLiteReplayState):
    @staticmethod
    def base(request):
        body = dict(schema='research-state-base/v2', identity=request.identity(), replay_id=request.replay_id)
        return dict(body, state_id=digest(body))

    def read_rows(self, request, *, cutoff=None):
        if not request.state_path.exists():
            return [], []
        with closing(sqlite3.connect(request.state_path.as_uri() + '?mode=ro', uri=True)) as conn:
            conn.execute('BEGIN')
            if conn.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID:
                raise ValueError('FOREIGN_RESEARCH_STATE')
            row = conn.execute('SELECT identity_json FROM replay_runs WHERE replay_id=?', (request.replay_id,)).fetchone()
            if row is None or json.loads(row[0]) != request.identity():
                raise ValueError('REPLAY_PREDECESSOR_IDENTITY_MISMATCH')
            predicate = ' AND as_of=?' if cutoff else ''
            args = (request.replay_id, cutoff) if cutoff else (request.replay_id,)
            sessions = [list(r) for r in conn.execute('SELECT as_of,input_hash,result_hash,result_json FROM replay_sessions '
                'WHERE replay_id=?' + predicate + ' ORDER BY as_of', args)]
            observations = [list(r) for r in conn.execute('SELECT trade_date,incumbent,challenger,passed,as_of FROM replay_observations '
                'WHERE replay_id=?' + predicate + ' ORDER BY trade_date,incumbent,challenger', args)]
        return sessions, observations

    def delta(self, request, previous_state_id):
        sessions, observations = self.read_rows(request, cutoff=request.as_of.isoformat())
        if len(sessions) != 1 or sessions[0][1] != digest(request.inputs()):
            raise ValueError('RESEARCH_COMMIT_MISSING_OR_CHANGED')
        body = dict(schema='research-state-delta/v2', identity=request.identity(), replay_id=request.replay_id,
                    previous_state_id=previous_state_id, session=sessions[0], observations=observations)
        return dict(body, state_id=digest(body))

    def validate_delta(self, request, delta, previous_state_id, previous_date=None):
        body = {k:v for k,v in delta.items() if k != 'state_id'}
        if (body.get('schema') != 'research-state-delta/v2' or digest(body) != delta.get('state_id') or
                body.get('previous_state_id') != previous_state_id or body.get('identity') != request.identity() or
                body.get('replay_id') != request.replay_id):
            raise ValueError('RESEARCH_STATE_CHAIN_INVALID')
        row = body['session']
        if previous_date is not None and row[0] <= previous_date:
            raise ValueError('RESEARCH_STATE_CHAIN_OUT_OF_ORDER')
        result = self._result(row[3])
        if digest(result) != row[2] or result['as_of'] != row[0]:
            raise ValueError('RESEARCH_STATE_RESULT_HASH_MISMATCH')
        if any(r[4] != row[0] or r[0] != row[0][:10] or r[3] not in (0, 1) for r in body['observations']):
            raise ValueError('INVALID_PREDECESSOR_OBSERVATION')
        expected = [[row[0][:10], o['incumbent_stock_id'], o['challenger_stock_id'], int(o['passed']), row[0]]
                    for o in result['pae_state']['observations']]
        if sorted(expected) != sorted(body['observations']):
            raise ValueError('RESEARCH_STATE_OBSERVATION_MISMATCH')
        return result

    def restore_rows(self, request, sessions, observations):
        """Rebuild index only from a caller-validated public chain, never copy SQLite."""
        if request.state_path.exists():
            raise ValueError('RESTORE_REQUIRES_EMPTY_RESEARCH_DESTINATION')
        request.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Reuse the already-tested atomic snapshot restore. Reference JSON is
        # expanded only for validation, then inserted as compact references.
        import os
        import tempfile
        fd, name = tempfile.mkstemp(prefix='.research-restore-', dir=request.state_path.parent)
        os.close(fd)
        try:
            with closing(sqlite3.connect(name)) as conn:
                with conn:
                    conn.execute('BEGIN IMMEDIATE')
                    self._initialize(conn)
                    conn.execute('INSERT INTO replay_runs VALUES (?,?)', (request.replay_id, canonical_json(request.identity())))
                    conn.executemany('INSERT INTO replay_sessions VALUES (?,?,?,?,?)', [(request.replay_id,*r) for r in sessions])
                    conn.executemany('INSERT INTO replay_observations VALUES (?,?,?,?,?,?)', [(request.replay_id,*r) for r in observations])
            os.link(name, request.state_path)
        finally:
            os.unlink(name)
