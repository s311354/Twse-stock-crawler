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
import zlib

from application.replay_contracts import aware_timestamp, canonical_json, digest
from application.research_evidence import materialize_filings

APPLICATION_ID = 0x52534348
SCHEMA_VERSION = 1
TABLES = {"research_parses", "research_filing_versions", "research_catalogs"}
_RAW_ZLIB_MAGIC = b"\x00TWSE_RAW_JSON_ZLIB_1\x00"
_RAW_LENGTH_BYTES = 8
DEFAULT_MAX_DECODED_PARSE_BYTES = 32 * 1024 * 1024
_MIN_COMPRESS_BYTES = 1024


def epoch_microseconds(value: str | dt.datetime) -> int:
    at = aware_timestamp(value.isoformat() if isinstance(value, dt.datetime) else value)
    delta = at.astimezone(dt.timezone.utc) - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _json_hash(encoded: str) -> str:
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class _InvalidCache(ValueError):
    pass


class _ParseMemo:
    def __init__(self, owner, connection, *, admit=None):
        self.owner, self.connection = owner, connection
        self.admit = admit
        self.old_bytes = self.new_bytes = 0
        self.admitted = False
        self.miss_reason = None

    def memo(self, kind, key, compute):
        if kind != "ixbrl":
            raise ValueError("ONLY_RAW_PARSE_RESULTS_MAY_BE_PERSISTED")
        version, sha = key
        row = self.connection.execute(
            "SELECT facts_json, facts_sha256 FROM research_parses WHERE version=? AND content_sha256=?",
            (version, sha)).fetchone()
        self.old_bytes = len(row[0].encode("utf-8")) if row is not None else 0
        if row is not None and _json_hash(row[0]) == row[1]:
            try:
                value = json.loads(row[0])
                if not isinstance(value, list):
                    raise ValueError("INVALID_FACTS")
            except (ValueError, TypeError):
                pass
            else:
                self.owner.counts["sqlite_ixbrl_hit"] += 1
                self.new_bytes = self.old_bytes
                return value
        self.owner.counts["sqlite_ixbrl_miss"] += 1
        self.miss_reason = "absent" if row is None else "corrupt"
        self.owner.counts["sqlite_ixbrl_miss_" + self.miss_reason] += 1
        value = compute()
        encoded = canonical_json(value)
        self.new_bytes = len(encoded.encode("utf-8"))
        if self.admit is not None and not self.admit(self.new_bytes):
            # A single oversized result must not flush the useful working set.
            # Invalid preexisting rows are repairable, never authoritative.
            if row is not None:
                self.connection.execute("DELETE FROM research_parses WHERE version=? AND content_sha256=?",
                                        (version, sha))
            self.new_bytes = 0
            return json.loads(encoded)
        self.connection.execute("INSERT OR REPLACE INTO research_parses VALUES (?,?,?,?)",
                                (version, sha, encoded, _json_hash(encoded)))
        self.admitted = True
        return json.loads(encoded)


class _RawParseMemo(_ParseMemo):
    """Disposable source cache codec; indexed research facts keep their format.

    The stored SHA is always that of the original canonical JSON, not its
    compression. Legacy TEXT rows remain readable. BLOBs are bounded before
    retrieval and inflation; a bad/unknown codec is a miss, never evidence.
    """

    def memo(self, kind, key, compute):
        if kind != "ixbrl":
            raise ValueError("ONLY_RAW_PARSE_RESULTS_MAY_BE_PERSISTED")
        from application.daily_performance import measure_stage
        version, sha = key
        limit = self.owner.max_decoded_bytes
        stored_limit = limit + len(_RAW_ZLIB_MAGIC) + _RAW_LENGTH_BYTES
        with measure_stage("source_cache_lookup"):
            row = self.connection.execute("""SELECT
                CASE WHEN length(CAST(facts_json AS BLOB))<=? THEN facts_json END,
                facts_sha256,length(CAST(facts_json AS BLOB))
                FROM research_parses WHERE version=? AND content_sha256=?""",
                (stored_limit, version, sha)).fetchone()
        self.old_bytes = row[2] if row is not None else 0
        if row is not None:
            try:
                with measure_stage("source_cache_decode"):
                    encoded = self._decode(row[0], limit)
                    if hashlib.sha256(encoded).hexdigest() != row[1]:
                        raise ValueError("RAW_PARSE_CACHE_HASH_MISMATCH")
                    value = json.loads(encoded)
                    if not isinstance(value, list):
                        raise ValueError("INVALID_FACTS")
            except (ValueError, TypeError, UnicodeError, zlib.error, OverflowError):
                pass
            else:
                self.owner.counts["sqlite_ixbrl_hit"] += 1
                self._count("source_cache_stored_bytes_read", self.old_bytes)
                self._count("source_cache_decoded_bytes_read", len(encoded))
                self.new_bytes = self.old_bytes
                return value
        self.owner.counts["sqlite_ixbrl_miss"] += 1
        self.miss_reason = "absent" if row is None else "corrupt"
        self.owner.counts["sqlite_ixbrl_miss_" + self.miss_reason] += 1
        if row is not None:
            # Repairable rows may have been damaged outside this connection.
            # Reconcile only on corruption, not with a full SUM on every hit.
            self.owner.bytes = self.connection.execute(
                "SELECT COALESCE(SUM(length(CAST(facts_json AS BLOB))),0) FROM research_parses"
            ).fetchone()[0]
        value = compute()
        with measure_stage("source_cache_encode"):
            encoded = canonical_json(value).encode("utf-8")
            payload = self._encode(encoded)
        self.new_bytes = len(payload.encode("utf-8")) if isinstance(payload, str) else len(payload)
        if len(encoded) > limit or self.admit is not None and not self.admit(self.new_bytes):
            if len(encoded) > limit:
                self._count("source_cache_decoded_oversized_bypass")
            if row is not None:
                self.connection.execute("DELETE FROM research_parses WHERE version=? AND content_sha256=?",
                                        (version, sha))
            self.new_bytes = 0
            return json.loads(encoded)
        self.connection.execute("INSERT OR REPLACE INTO research_parses VALUES (?,?,?,?)",
                                (version, sha, payload, hashlib.sha256(encoded).hexdigest()))
        self._count("source_cache_json_bytes_admitted", len(encoded))
        if isinstance(payload, bytes):
            self._count("source_cache_compressed_admissions")
        self.admitted = True
        return json.loads(encoded)

    def _count(self, name, value=1):
        from application.daily_performance import count_work
        self.owner.counts[name] += value
        count_work(name, value)

    def _encode(self, encoded):
        if (not self.owner.compression or len(encoded) < _MIN_COMPRESS_BYTES
                or len(encoded) > self.owner.max_decoded_bytes):
            return encoded.decode("utf-8")
        compressed = (_RAW_ZLIB_MAGIC + len(encoded).to_bytes(_RAW_LENGTH_BYTES, "big")
                      + zlib.compress(encoded, level=1))
        return compressed if len(compressed) < len(encoded) else encoded.decode("utf-8")

    @staticmethod
    def _decode(payload, limit):
        if isinstance(payload, str):
            encoded = payload.encode("utf-8")
            if len(encoded) > limit:
                raise ValueError("RAW_PARSE_CACHE_DECODE_LIMIT")
            return encoded
        if not isinstance(payload, bytes) or not payload.startswith(_RAW_ZLIB_MAGIC):
            raise ValueError("RAW_PARSE_CACHE_CODEC_UNKNOWN")
        end = len(_RAW_ZLIB_MAGIC) + _RAW_LENGTH_BYTES
        if len(payload) <= end:
            raise ValueError("RAW_PARSE_CACHE_HEADER_INVALID")
        size = int.from_bytes(payload[len(_RAW_ZLIB_MAGIC):end], "big")
        if size > limit:
            raise ValueError("RAW_PARSE_CACHE_DECODE_LIMIT")
        decoder = zlib.decompressobj()
        encoded = decoder.decompress(payload[end:], size + 1)
        if (len(encoded) != size or not decoder.eof or decoder.unused_data
                or decoder.unconsumed_tail):
            raise ValueError("RAW_PARSE_CACHE_COMPRESSED_BODY_INVALID")
        return encoded


class SQLiteRawParseCache:
    """Public source-preprocessing memo using the existing research cache schema.

    One connection per work. Repairable rows contain raw parses or pure
    source-bound companions, not cutoff selections, account, gates or decisions.
    Callers must verify the original hash first. Capacity bounds stored payload
    bytes, with a separate per-entry decoded bound; SQLite free pages may be
    larger. Compression changes neither canonical JSON nor its source key/SHA.
    """
    def __init__(self, path, *, max_bytes=512 * 1024 * 1024, compression=True,
                 max_decoded_bytes=DEFAULT_MAX_DECODED_PARSE_BYTES):
        if max_bytes < 0:
            raise ValueError("INVALID_CACHE_CAPACITY")
        self.path = Path(path)
        self.max_bytes = max_bytes
        if type(compression) is not bool or type(max_decoded_bytes) is not int or max_decoded_bytes <= 0:
            raise ValueError("INVALID_RAW_PARSE_CODEC_CONFIGURATION")
        self.compression, self.max_decoded_bytes = compression, max_decoded_bytes
        self.bytes = 0
        self.counts = Counter()
        self.connection = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=5)
        try:
            # Reject an indexed research database before schema PRAGMAs/DDL:
            # even a rejected cache open must not touch committed research.
            indexed = self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_filing_versions'"
            ).fetchone()
            if indexed and self.connection.execute("SELECT 1 FROM research_filing_versions LIMIT 1").fetchone():
                raise ValueError("RAW_PARSE_CACHE_MUST_NOT_USE_INDEXED_RESEARCH_DATABASE")
            SQLiteFilingHistory._ensure_schema(self.connection)
            self.bytes = self.connection.execute("SELECT COALESCE(SUM(length(CAST(facts_json AS BLOB))),0) FROM research_parses").fetchone()[0]
            self.connection.commit()
        except BaseException:
            self.connection.close()
            self.connection = None
            raise
        return self

    def memo(self, kind, key, compute):
        if kind not in {"ixbrl", "financial_companion"}:
            raise ValueError("ONLY_SOURCE_PREPROCESSING_MAY_BE_PERSISTED")
        if self.connection is None:
            raise ValueError("RAW_PARSE_CACHE_NOT_OPEN")
        # Companion rows cannot collide with legacy/raw parser versions even
        # when a caller accidentally supplies the same version/hash pair.
        stored_key = key if kind == "ixbrl" else ("financial-companion:" + key[0], key[1])
        prefix = "raw_parse_disk" if kind == "ixbrl" else "financial_companion_disk"
        with self.connection:
            before = self.counts["sqlite_ixbrl_hit"]
            memo = _RawParseMemo(self, self.connection, admit=lambda size: size <= self.max_bytes)
            value = memo.memo("ixbrl", stored_key, compute)
            from application.daily_performance import count_work
            hit = self.counts["sqlite_ixbrl_hit"] > before
            count_work(prefix + ("_hits" if hit else "_misses"))
            self.counts[prefix + ("_hits" if hit else "_misses")] += 1
            if not hit:
                self.bytes += memo.new_bytes - memo.old_bytes
                count_work(prefix + "_miss_" + memo.miss_reason)
                self.counts[prefix + "_miss_" + memo.miss_reason] += 1
                if memo.admitted:
                    count_work(prefix + "_admissions")
                    count_work(prefix + "_bytes_admitted", memo.new_bytes)
                    self.counts[prefix + "_admissions"] += 1
                    self.counts[prefix + "_bytes_admitted"] += memo.new_bytes
                else:
                    count_work(prefix + "_oversized_bypass")
                    self.counts[prefix + "_oversized_bypass"] += 1
            # FIFO eviction of repairable raw parses only. No committed research
            # state, results, originals or report evidence is ever deleted.
            while self.bytes > self.max_bytes:
                oldest = self.connection.execute("SELECT rowid,length(CAST(facts_json AS BLOB)) FROM research_parses ORDER BY rowid LIMIT 1").fetchone()
                if oldest is None:
                    break
                self.connection.execute("DELETE FROM research_parses WHERE rowid=?", (oldest[0],))
                self.bytes -= oldest[1]
                count_work("raw_parse_disk_evictions")
                count_work("raw_parse_disk_bytes_evicted", oldest[1])
                self.counts["raw_parse_disk_evictions"] += 1
                self.counts["raw_parse_disk_bytes_evicted"] += oldest[1]
            return value

    @property
    def stats(self):
        return dict(self.counts, encoded_bytes=self.bytes, max_bytes=self.max_bytes,
                    stored_payload_bytes=self.bytes, capacity_unit="STORED_PAYLOAD_BYTES",
                    compression_enabled=self.compression, max_decoded_entry_bytes=self.max_decoded_bytes,
                    disk_bytes=self.path.stat().st_size if self.path.exists() else 0)

    def __exit__(self, *args):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


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
                parses = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_parses'"
                ).fetchone()
                if parses and connection.execute(
                        "SELECT 1 FROM research_parses WHERE typeof(facts_json)!='text' LIMIT 1"
                        ).fetchone() is not None:
                    raise ValueError("INDEXED_RESEARCH_MUST_NOT_USE_RAW_SOURCE_CACHE_DATABASE")
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
