"""Bounded dated P/E capture with the existing exchange transport and SQLite."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from domain.les_v7.contracts import canonical, timestamp
from domain.les_v7.official_pe import ENDPOINTS, build_snapshot, request_parameters, session_date, validate_snapshot
from infrastructure.crawler.twse_http_client import TwseHttpClient


class OfficialPeClient:
    """One full-market dated request; caller controls budget, no history loop."""
    def __init__(self, http_client: TwseHttpClient | None = None, *, clock=None):
        self.http_client = http_client or TwseHttpClient()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def fetch(self, *, market: str, trade_date: str) -> dict:
        day = session_date(trade_date)
        params = request_parameters(market, day)
        response = self.http_client.request_bytes("GET", ENDPOINTS[market], params=params,
            context=f"{market} dated official P/E at {day}")
        # Capture after the response completes. Do not date availability at
        # request start or the table's historic session date.
        return build_snapshot(response.content, market=market, trade_date=day,
                              fetched_at=self.clock().isoformat())


class OfficialPeRepository:
    """Append-only, content-identified observations; read does not create DB."""
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, snapshot: dict) -> str:
        checked = validate_snapshot(snapshot)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS official_pe_observations (
                evidence_id TEXT PRIMARY KEY, market TEXT NOT NULL,
                session_date TEXT NOT NULL, available_at_us INTEGER NOT NULL,
                observation_json TEXT NOT NULL)""")
            connection.execute("""CREATE INDEX IF NOT EXISTS official_pe_cutoff
                ON official_pe_observations(market, session_date, available_at_us)""")
            encoded = canonical(checked)
            existing = connection.execute("SELECT observation_json FROM official_pe_observations WHERE evidence_id=?",
                                          (checked["evidence_id"],)).fetchone()
            if existing and existing[0] != encoded:
                raise ValueError("IMMUTABLE_OFFICIAL_PE_CONFLICT")
            connection.execute("INSERT OR IGNORE INTO official_pe_observations VALUES (?,?,?,?,?)",
                (checked["evidence_id"], checked["market"], checked["session_date"],
                 int(timestamp(checked["available_at"]).timestamp() * 1_000_000), encoded))
        return checked["evidence_id"]

    def load_as_of(self, *, market: str, trade_dates: list[str], cutoff: str) -> list[dict]:
        """One cutoff-indexed read for exact dates, retaining same-time conflicts."""
        import json
        dates = sorted({session_date(day) for day in trade_dates})
        if not dates:
            return []
        request_parameters(market, dates[0])
        cutoff_us = int(timestamp(cutoff).timestamp() * 1_000_000)
        if not self.path.exists():
            return []
        placeholders = ",".join("?" for _ in dates)
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='official_pe_observations'").fetchone():
                return []
            rows = connection.execute(f"""SELECT o.evidence_id, o.market, o.session_date,
                       o.available_at_us, o.observation_json
                FROM official_pe_observations AS o
                JOIN (
                    SELECT market, session_date, MAX(available_at_us) AS latest
                    FROM official_pe_observations
                    WHERE market=? AND session_date IN ({placeholders}) AND available_at_us<=?
                    GROUP BY market, session_date
                ) AS selected ON o.market=selected.market AND o.session_date=selected.session_date
                              AND o.available_at_us=selected.latest
                ORDER BY o.session_date, o.evidence_id""", [market, *dates, cutoff_us]).fetchall()
        result = []
        for evidence_id, stored_market, day, available_us, encoded in rows:
            checked = validate_snapshot(json.loads(encoded), cutoff=cutoff)
            if (checked["evidence_id"] != evidence_id or checked["market"] != stored_market
                    or checked["session_date"] != day
                    or int(timestamp(checked["available_at"]).timestamp() * 1_000_000) != available_us):
                raise ValueError("OFFICIAL_PE_DATABASE_BINDING_INVALID")
            result.append(checked)
        return result
