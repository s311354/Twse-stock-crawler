"""Append-only official restriction observations; replay never creates evidence."""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from pathlib import Path
import sqlite3

import pandas as pd

from domain.trading_restrictions import MAX_RESTRICTION_AGE, restriction_check


class TradingRestrictionRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, evidence: dict, *, market: str, market_data_date: str,
               decision_time: object) -> int:
        """Persist only complete, validated observations, without redating them.

        Identical observations are idempotent. Conflicting observations at the
        same timestamp are both retained, so replay cannot cherry-pick CLEAR.
        This is called by the live collector, not by the replay file importer.
        """
        rows = []
        for symbol, item in evidence.items():
            if restriction_check(item, market=market, market_data_date=market_data_date,
                                 decision_time=decision_time)["status"] == "UNKNOWN":
                continue
            payload = json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False)
            rows.append((market, str(symbol), pd.Timestamp(item["available_at"]).value,
                         hashlib.sha256(payload.encode()).hexdigest(), payload))
        if not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS restriction_observations (
                market TEXT NOT NULL, symbol TEXT NOT NULL,
                available_at_ns INTEGER NOT NULL, digest TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                PRIMARY KEY (market, symbol, available_at_ns, digest))""")
            connection.execute("""CREATE INDEX IF NOT EXISTS restriction_cutoff
                ON restriction_observations(market, available_at_ns, symbol)""")
            before = connection.total_changes
            connection.executemany("INSERT OR IGNORE INTO restriction_observations VALUES (?,?,?,?,?)", rows)
            return connection.total_changes - before

    def load_as_of(self, *, market: str, market_data_date: str, as_of: object) -> dict:
        """Read latest eligible evidence per symbol using one indexed JOIN query.

        Revalidate the original timestamps/source contract at the caller's
        cutoff. No older CLEAR fallback after a newer restriction or conflict.
        The read-only connection never initializes or migrates a replay cache.
        """
        cutoff = pd.Timestamp(as_of)
        if pd.isna(cutoff) or cutoff.tzinfo is None:
            raise ValueError("RESTRICTION_ARCHIVE_REQUIRES_AWARE_CUTOFF")
        if not self.path.exists():
            return {}
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            rows = connection.execute("""SELECT r.symbol, r.digest, r.evidence_json
                FROM restriction_observations AS r
                JOIN (
                    SELECT symbol, MAX(available_at_ns) AS latest
                    FROM restriction_observations
                    WHERE market = ? AND available_at_ns BETWEEN ? AND ?
                    GROUP BY symbol
                ) AS selected ON r.symbol = selected.symbol AND r.available_at_ns = selected.latest
                WHERE r.market = ? ORDER BY r.symbol, r.digest""",
                (market, (cutoff - MAX_RESTRICTION_AGE).value, cutoff.value, market)).fetchall()
        result, seen = {}, set()
        for symbol, digest, payload in rows:
            if symbol in seen:
                result.pop(symbol, None)
                continue
            seen.add(symbol)
            if hashlib.sha256(payload.encode()).hexdigest() != digest:
                continue
            try:
                item = json.loads(payload)
            except (ValueError, TypeError):
                continue
            checked = restriction_check(item, market=market, market_data_date=market_data_date,
                                        decision_time=cutoff)
            if checked["status"] != "UNKNOWN":
                result[symbol] = item
        return result
