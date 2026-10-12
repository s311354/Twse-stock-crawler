"""Rebuildable research-only SQLite parse cache and point-in-time filing index.

Original documents remain authoritative. No account, decisions or PAE state is
stored here; the ordinary materializer remains the uncached reference path.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3

from application.replay_contracts import aware_timestamp, canonical_json, digest
from application.research_evidence import materialize_filings

APPLICATION_ID = 0x52534348
SCHEMA_VERSION = 1
TABLES = {"research_parses", "research_filing_versions", "research_catalogs"}


def epoch_microseconds(value: str | dt.datetime) -> int:
    at = aware_timestamp(value.isoformat() if isinstance(value, dt.datetime) else value)
    delta = at.astimezone(dt.timezone.utc) - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _json_hash(encoded: str) -> str:
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class _InvalidCache(ValueError):
    pass


class _ParseMemo:
    def __init__(self, owner, connection):
        self.owner, self.connection = owner, connection

    def memo(self, kind, key, compute):
        if kind != "ixbrl":
            raise ValueError("ONLY_RAW_PARSE_RESULTS_MAY_BE_PERSISTED")
        version, sha = key
        row = self.connection.execute(
            "SELECT facts_json, facts_sha256 FROM research_parses WHERE version=? AND content_sha256=?",
            (version, sha)).fetchone()
        if row is not None and _json_hash(row[0]) == row[1]:
            try:
                value = json.loads(row[0])
                if not isinstance(value, list):
                    raise ValueError("INVALID_FACTS")
            except (ValueError, TypeError):
                pass
            else:
                self.owner.counts["sqlite_ixbrl_hit"] += 1
                return value
        self.owner.counts["sqlite_ixbrl_miss"] += 1
        value = compute()
        encoded = canonical_json(value)
        self.connection.execute("INSERT OR REPLACE INTO research_parses VALUES (?,?,?,?)",
                                (version, sha, encoded, _json_hash(encoded)))
        return json.loads(encoded)


class SQLiteRawParseCache:
    """Public raw-only memo using the existing research cache schema.

    One connection per work. Its repairable rows contain no availability proof,
    account, gate or decision. Callers must verify the original hash first.
    """
    def __init__(self, path, *, max_bytes=512 * 1024 * 1024):
        if max_bytes < 0:
            raise ValueError("INVALID_CACHE_CAPACITY")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.bytes = 0
        self.counts = Counter()
        self.connection = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=5)
        SQLiteFilingHistory._ensure_schema(self.connection)
        if self.connection.execute("SELECT COUNT(*) FROM research_filing_versions").fetchone()[0]:
            self.connection.close()
            raise ValueError("RAW_PARSE_CACHE_MUST_NOT_USE_INDEXED_RESEARCH_DATABASE")
        self.bytes = self.connection.execute("SELECT COALESCE(SUM(length(CAST(facts_json AS BLOB))),0) FROM research_parses").fetchone()[0]
        self.connection.commit()
        return self

    def memo(self, kind, key, compute):
        with self.connection:
            before = self.counts["sqlite_ixbrl_hit"]
            old = self.connection.execute("SELECT length(CAST(facts_json AS BLOB)) FROM research_parses WHERE version=? AND content_sha256=?", key).fetchone()
            value = _ParseMemo(self, self.connection).memo(kind, key, compute)
            from application.daily_performance import count_work
            hit = self.counts["sqlite_ixbrl_hit"] > before
            count_work("raw_parse_disk_hits" if hit else "raw_parse_disk_misses")
            if not hit:
                new = self.connection.execute("SELECT length(CAST(facts_json AS BLOB)) FROM research_parses WHERE version=? AND content_sha256=?", key).fetchone()[0]
                self.bytes += new - (old[0] if old else 0)
            # FIFO eviction of repairable raw parses only. No committed research
            # state, results, originals or report evidence is ever deleted.
            while self.bytes > self.max_bytes:
                oldest = self.connection.execute("SELECT rowid,length(CAST(facts_json AS BLOB)) FROM research_parses ORDER BY rowid LIMIT 1").fetchone()
                if oldest is None:
                    break
                self.connection.execute("DELETE FROM research_parses WHERE rowid=?", (oldest[0],))
                self.bytes -= oldest[1]
                count_work("raw_parse_disk_evictions")
            return value

    def __exit__(self, *args):
        if self.connection is not None:
            self.connection.close()


class SQLiteFilingHistory:
    """Versioned derived data, scoped to an immutable dataset, not today's cache."""

    def __init__(self, path: Path, *, evidence_id: str, version: str, parse_filing, document_reader=None):
        self.path = Path(path)
        self.evidence_id = evidence_id
        # Conservative invalidation includes model, parameters, code and format.
        self.version = digest(dict(schema=SCHEMA_VERSION, execution_version=version))
        self.parse_filing, self.document_reader = parse_filing, document_reader
        self.counts = Counter()

    @property
    def stats(self) -> dict:
        return dict(self.counts)

    @staticmethod
    def _ensure_schema(connection) -> None:
        tables = {r[0] for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        app = connection.execute("PRAGMA application_id").fetchone()[0]
        if (tables or app) and (app != APPLICATION_ID or not tables <= TABLES):
            raise ValueError("RESEARCH_CACHE_MUST_NOT_USE_LIVE_OR_FOREIGN_DATABASE")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise ValueError("UNSUPPORTED_RESEARCH_CACHE_SCHEMA")
        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        connection.execute("""CREATE TABLE IF NOT EXISTS research_parses (
            version TEXT NOT NULL, content_sha256 TEXT NOT NULL,
            facts_json TEXT NOT NULL, facts_sha256 TEXT NOT NULL,
            PRIMARY KEY(version,content_sha256))""")
        connection.execute("""CREATE TABLE IF NOT EXISTS research_filing_versions (
            version TEXT NOT NULL, evidence_id TEXT NOT NULL, symbol TEXT NOT NULL,
            ordinal INTEGER NOT NULL, content_sha256 TEXT, available_at_epoch_us INTEGER,
            metadata_json TEXT NOT NULL, error_json TEXT NOT NULL, facts_sha256 TEXT,
            PRIMARY KEY(version,evidence_id,symbol,ordinal))""")
        connection.execute("""CREATE INDEX IF NOT EXISTS research_filings_cutoff
            ON research_filing_versions(version,evidence_id,symbol,available_at_epoch_us)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS research_catalogs (
            version TEXT NOT NULL, evidence_id TEXT NOT NULL, symbol TEXT NOT NULL,
            catalog_sha256 TEXT NOT NULL, rows_sha256 TEXT NOT NULL,
            PRIMARY KEY(version,evidence_id,symbol))""")
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _key(self, symbol):
        return self.version, self.evidence_id, symbol

    def _index_valid(self, connection, symbol, catalog_sha) -> bool:
        key = self._key(symbol)
        receipt = connection.execute("""SELECT catalog_sha256,rows_sha256 FROM research_catalogs
            WHERE version=? AND evidence_id=? AND symbol=?""", key).fetchone()
        if receipt is None or receipt[0] != catalog_sha:
            return False
        rows = connection.execute("""SELECT ordinal,content_sha256,available_at_epoch_us,
            metadata_json,error_json,facts_sha256 FROM research_filing_versions
            WHERE version=? AND evidence_id=? AND symbol=? ORDER BY ordinal""", key).fetchall()
        return digest(rows) == receipt[1]

    def _rebuild(self, connection, payload, symbol, catalog_sha):
        key = self._key(symbol)
        memo = _ParseMemo(self, connection)
        rows = []
        for ordinal, item in enumerate(payload.get("filings", {}).get(symbol, [])):
            records, errors = materialize_filings(
                dict(raw_documents=payload.get("raw_documents", {}), filings={symbol: [item]}),
                parse_filing=self.parse_filing, document_reader=self.document_reader,
                cache=memo, version=self.version)
            if records[symbol]:
                metadata = records[symbol][0]
                facts = metadata.pop("facts")
                row = (ordinal, item["content_sha256"], epoch_microseconds(metadata["available_at"]),
                       canonical_json(metadata), "null", digest(facts))
            else:
                if errors[symbol][0]["reason"].startswith("SOURCE_DOCUMENT_UNAVAILABLE"):
                    # A removable disk or temporary read failure is not an
                    # immutable parser outcome; roll back instead of pinning it.
                    raise ValueError(errors[symbol][0]["reason"])
                row = (ordinal, item.get("content_sha256"), None, "null", canonical_json(errors[symbol][0]), None)
            rows.append(row)
        connection.execute("DELETE FROM research_filing_versions WHERE version=? AND evidence_id=? AND symbol=?", key)
        connection.executemany("INSERT INTO research_filing_versions VALUES (?,?,?,?,?,?,?,?,?)",
                               [(*key, *row) for row in rows])
        connection.execute("INSERT OR REPLACE INTO research_catalogs VALUES (?,?,?,?,?)",
                           (*key, catalog_sha, digest(rows)))

    def _select(self, connection, symbol, cutoff):
        key = self._key(symbol)
        # Fetch facts ONLY for evidence available at this cutoff. Keep future
        # metadata/errors for the exact same exclusion diagnostics as the oracle.
        eligible = connection.execute("""SELECT f.ordinal,f.metadata_json,f.error_json,
            f.content_sha256,f.facts_sha256,p.facts_json,p.facts_sha256
            FROM research_filing_versions f LEFT JOIN research_parses p
              ON p.version=f.version AND p.content_sha256=f.content_sha256
            WHERE f.version=? AND f.evidence_id=? AND f.symbol=? AND f.available_at_epoch_us<=?
            ORDER BY f.ordinal""", (*key, cutoff)).fetchall()
        excluded = connection.execute("""SELECT ordinal,metadata_json,error_json,content_sha256,
            facts_sha256,NULL,NULL FROM research_filing_versions
            WHERE version=? AND evidence_id=? AND symbol=?
              AND (available_at_epoch_us>? OR available_at_epoch_us IS NULL) ORDER BY ordinal""",
            (*key, cutoff)).fetchall()
        records, errors = [], []
        for _, metadata, error, sha, expected_hash, facts, stored_hash in sorted(eligible + excluded):
            if error != "null":
                errors.append(json.loads(error))
                continue
            record = json.loads(metadata)
            # Independent of the SQL predicate: never trust the derived index
            # to move an original available_at timestamp across the cutoff.
            if epoch_microseconds(record["available_at"]) <= cutoff:
                if facts is None or stored_hash != expected_hash or _json_hash(facts) != expected_hash:
                    connection.execute("DELETE FROM research_parses WHERE version=? AND content_sha256=?",
                                       (self.version, sha))
                    raise _InvalidCache("RESEARCH_PARSE_CACHE_CORRUPT")
                record["facts"] = json.loads(facts)
            else:
                record["facts"] = []
            records.append(record)
        return records, errors

    def load(self, payload: dict, symbols: tuple[str, ...], as_of: dt.datetime) -> tuple[dict, dict]:
        cutoff = epoch_microseconds(as_of)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                self._ensure_schema(connection)
                records, errors = {}, {}
                for symbol in symbols:
                    catalog_sha = digest(payload.get("filings", {}).get(symbol, []))
                    valid = self._index_valid(connection, symbol, catalog_sha)
                    self.counts["sqlite_index_hit" if valid else "sqlite_index_miss"] += 1
                    if not valid:
                        self._rebuild(connection, payload, symbol, catalog_sha)
                    try:
                        records[symbol], errors[symbol] = self._select(connection, symbol, cutoff)
                    except _InvalidCache:
                        self.counts["sqlite_rebuild"] += 1
                        self._rebuild(connection, payload, symbol, catalog_sha)
                        records[symbol], errors[symbol] = self._select(connection, symbol, cutoff)
                return records, errors
        except (sqlite3.DatabaseError, OSError):
            # A locked/unavailable optimization must not change model semantics.
            self.counts["sqlite_fallback"] += 1
            records, errors = materialize_filings(dict(raw_documents=payload.get("raw_documents", {}),
                filings={s: payload.get("filings", {}).get(s, []) for s in symbols}),
                parse_filing=self.parse_filing, document_reader=self.document_reader)
            for rows in records.values():
                for row in rows:
                    if epoch_microseconds(row["available_at"]) > cutoff:
                        row["facts"] = []
            return records, errors
