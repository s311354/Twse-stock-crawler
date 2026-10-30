"""Read-only SQLite query side of MOPS persistence."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import sys
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import asdict, fields, replace
from functools import lru_cache
from pathlib import Path

import pandas as pd

from application.fundamental_ports import FundamentalUnitOfWork
from application.daily_performance import count_work, measure_stage
from application.daily_work import current_work
from domain.fundamental_derivation import CachedFundamentalBundle
from domain.fundamental_derivation import CachedFundamentalHeader
from domain.fundamental_derivation import FiledFact
from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FilingCatalogState
from domain.fundamental_models import FilingCorrection
from domain.fundamental_models import FundamentalSnapshot
from domain.fundamental_models import NormalizedFact
from infrastructure.fundamentals.models import MopsFiling
from infrastructure.fundamentals.models import MopsFilingCandidate
from infrastructure.fundamentals.ixbrl_parser import (
    IXBRL_PARSER_VERSION, ixbrl_schema_refs, parse_ixbrl_evidence,
)
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
from infrastructure.fundamentals.mops_sqlite import row_to_filing
from infrastructure.fundamentals.mops_sqlite import to_epoch
from infrastructure.fundamentals.mops_unit_of_work import borrow_connection
from infrastructure.fundamentals.parse_identity import semantic_raw_parse_version


_FILED_FACT_FIELD_NAMES = tuple(field.name for field in fields(FiledFact))


def _raw_normalized_facts(
    path: Path, expected_hash: str, signature: tuple[int, int, int], parser_version: str,
    *, include_decoding: bool = False,
) -> tuple | None:
    parser_version = semantic_raw_parse_version(parser_version + ":" + IXBRL_PARSER_VERSION)
    work = current_work()
    if work is None:
        result = _raw_normalized_facts_cached(path, expected_hash, signature, parser_version,
                                             "legacy-single-call")
        return result if result is None or include_decoding else result[:2]
    # A native job has exactly one owner for parsed facts. Do not populate or
    # consult the process-global legacy LRU, even when its work cache is disabled.
    # Check existence/stat on every access: a cached parse cannot hide a removed
    # or replaced original, or share facts between distinct source paths.
    path = path.resolve()
    stat = path.stat()
    actual_signature = stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
    key = str(path), expected_hash, actual_signature, parser_version
    result = work.memo_immutable("raw_parse", key,
        lambda: _read_raw_normalized_facts(path, expected_hash, parser_version, work),
        _raw_facts_size)
    return result if result is None or include_decoding else result[:2]


@lru_cache(maxsize=128)
def _raw_normalized_facts_cached(path, expected_hash, signature, parser_version, work_version):
    """Legacy single-call LRU only; native jobs never retain values here."""
    return _read_raw_normalized_facts(path, expected_hash, parser_version, None)


def _raw_facts_size(value):
    """Reachable immutable parse bytes, including strings and dimension tuples."""
    pending, seen, size = [value], set(), 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        size += sys.getsizeof(item)
        if isinstance(item, tuple):
            pending.extend(item)
        elif isinstance(item, FiledFact):
            pending.extend(getattr(item, name) for name in _FILED_FACT_FIELD_NAMES)
    return size


def _read_raw_normalized_facts(path, expected_hash, parser_version, work):
    """Verify this source first, then optionally reuse raw-only disk parsing."""
    with measure_stage("filing_original_read_verify"):
        content = path.read_bytes()
        count_work("raw_original_reads")
    if hashlib.sha256(content).hexdigest() != expected_hash:
        count_work("raw_unit_hash_rejections")
        return None
    def parse():
        with measure_stage("filing_unit_parse"):
            count_work("raw_unit_parse_calls")
            normalized, decoding = parse_ixbrl_evidence(content)
        return [dict(facts=[asdict(f) for f in normalized], schema_refs=list(ixbrl_schema_refs(content)),
                     decoding=decoding)]
    rows = work.raw_parse(expected_hash, parse, parser_version=parser_version) if work is not None else parse()
    facts = tuple(FiledFact(concept=f["concept"], context_id=f.get("context_id"), unit=f.get("unit"),
        numeric_value=f["numeric_value"], text_value=f["text_value"], start_date=f["start_date"],
        end_date=f["end_date"], instant=f["instant"],
        dimensions=tuple(tuple(d) for d in f.get("dimensions", ()))) for f in rows[0]["facts"])
    # JSON string keeps the work-owned result immutable and bounded. Consumers
    # detach only the small receipt; neither facts nor original graphs copy.
    return facts, tuple(rows[0]["schema_refs"]), json.dumps(rows[0]["decoding"], ensure_ascii=False, sort_keys=True)


# Preserve diagnostic/test hooks for the legacy cache. Work statistics and
# lifetime are owned by DailyWorkContext instead of a second process-global LRU.
_raw_normalized_facts.cache_clear = _raw_normalized_facts_cached.cache_clear
_raw_normalized_facts.cache_info = _raw_normalized_facts_cached.cache_info


class MopsQueryRepository:
    """Execute point-in-time reads without owning any persistence command."""

    _BULK_SYMBOL_CHUNK_SIZE = 400

    def __init__(
        self,
        connect: Callable[[], sqlite3.Connection],
        ensure_schema: Callable[[], None],
        *,
        raw_root: Path | None = None,
    ) -> None:
        self._connect = connect
        self._ensure_schema = ensure_schema
        self._raw_root = raw_root

    def _resolve_cached_document(self, filing: MopsFiling, facts: list[FiledFact]) -> tuple[tuple[FiledFact, ...], tuple[str, ...]]:
        """Reparse verified local originals without rewriting facts/provenance.

        Cached facts may predate context/number fixes, even with canonical units.
        Originals are verified by hash, read locally with a bounded versioned
        cache, and never trigger a network request or change availability.
        Missing/corrupt originals cannot turn an opaque EPS unit into evidence.
        """
        if self._raw_root is None:
            return tuple(facts), ()
        original_verified = False
        schema_refs = ()
        try:
            path = (self._raw_root.parent / filing.raw_path).resolve()
            if path.is_relative_to(self._raw_root.resolve()):
                stat = path.stat()
                reparsed = _raw_normalized_facts(
                    path, filing.content_sha256, (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size),
                    V6_FILING_SCHEMA_VERSION,
                )
                original_verified = reparsed is not None
                if reparsed is not None:
                    normalized, schema_refs = reparsed[:2]
                    if normalized:
                        return normalized, schema_refs
        except OSError:
            pass
        result = []
        for fact in facts:
            if not original_verified and fact.numeric_value is not None:
                # Persisted amounts cannot hide a missing/replaced original.
                # Keep SQLite/filing identity intact; this read model is UNKNOWN.
                fact = replace(fact, numeric_value=None, unit=None)
            elif original_verified:
                # Legacy storage callers may retain their persisted amounts
                # when a verified original has no parseable iXBRL facts. Such
                # amounts do NOT have re-established unit evidence for v6.
                fact = replace(fact, unit=None)
            if fact.unit and not fact.unit.startswith(("iso4217:", "xbrli:")):
                # Even a familiar unitRef label is not a unit declaration.
                fact = replace(fact, unit=None)
            result.append(fact)
        return tuple(result), schema_refs

    def load_filing_catalog(
        self, symbol: str, *, as_of: str, uow: FundamentalUnitOfWork | None = None,
    ) -> FilingCatalogState | None:
        self._ensure_schema()
        with borrow_connection(self._connect, None, uow) as connection:
            row = connection.execute(
                "SELECT state_json FROM filing_catalog_cache "
                "WHERE symbol = ? AND checked_at_epoch <= ?", (symbol, to_epoch(as_of)),
            ).fetchone()
        if row is None:
            return None
        return self._decode_catalog(row[0])

    def load_filing_catalogs_many(self, symbols: list[str], *, as_of: str,
                                 uow: FundamentalUnitOfWork | None = None) -> dict[str, FilingCatalogState]:
        """Small metadata only, same cutoff and decoding as the single query."""
        normalized = list(dict.fromkeys(str(item).strip() for item in symbols if str(item).strip()))
        if not normalized:
            return {}
        self._ensure_schema()
        result = {}
        with measure_stage("fundamental_catalog_preflight"), borrow_connection(self._connect, None, uow) as active:
            for offset in range(0, len(normalized), self._BULK_SYMBOL_CHUNK_SIZE):
                chunk = normalized[offset:offset + self._BULK_SYMBOL_CHUNK_SIZE]
                rows = active.execute("SELECT symbol,state_json FROM filing_catalog_cache "
                    "WHERE symbol IN ({}) AND checked_at_epoch <= ?".format(",".join("?" for _ in chunk)),
                    [*chunk, to_epoch(as_of)])
                count_work("fundamental_catalog_queries")
                for symbol, raw in rows:
                    result[str(symbol)] = self._decode_catalog(raw)
        return result

    @staticmethod
    def _decode_catalog(raw):
        values = json.loads(raw)
        return FilingCatalogState(
            symbol=values["symbol"], checked_at=values["checked_at"],
            candidates=tuple(FilingCandidate(**item) for item in values["candidates"]),
            scanned_periods=tuple(tuple(item) for item in values["scanned_periods"]),
            corrections=tuple(FilingCorrection(**item) for item in values["corrections"]),
            verified_corrections=tuple(FilingCorrection(**item) for item in values["verified_corrections"]),
            announcements_pending=values.get("announcements_pending", False),
            observed_batch_period=(tuple(values["observed_batch_period"])
                                   if values.get("observed_batch_period") else None),
        )

    def load_refresh_attempt(
        self, key: str, *, uow: FundamentalUnitOfWork | None = None,
    ) -> tuple[float, str] | None:
        self._ensure_schema()
        with borrow_connection(self._connect, None, uow) as connection:
            row = connection.execute(
                "SELECT attempted_at_epoch, status FROM remote_refresh_attempts "
                "WHERE refresh_key = ?", (key,),
            ).fetchone()
        return (float(row[0]), str(row[1])) if row else None

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

    def load_filings_many(
        self, symbols: list[str], *, as_of: str,
        uow: FundamentalUnitOfWork | None = None,
    ) -> dict[str, list[MopsFiling]]:
        """One borrowed connection, bounded queries, no facts or raw documents.

        The identity hash has canonical ordering, independent of facts JOIN
        order (including filings with no facts). Model input order is unchanged.
        """
        normalized = list(dict.fromkeys(str(item).strip() for item in symbols if str(item).strip()))
        if not normalized:
            return {}
        self._ensure_schema()
        result: dict[str, list[MopsFiling]] = {symbol: [] for symbol in normalized}
        with borrow_connection(self._connect, None, uow) as active:
            for offset in range(0, len(normalized), self._BULK_SYMBOL_CHUNK_SIZE):
                chunk = normalized[offset:offset + self._BULK_SYMBOL_CHUNK_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                rows = active.execute(
                    f"""
                    SELECT id, symbol, fiscal_year, quarter, report_id, available_at,
                           fetched_at, revision_number, content_sha256, source_url,
                           raw_path, period_end, statement_scope, audit_status,
                           correction_description, availability_source,
                           correction_announced_on
                    FROM filings
                    WHERE symbol IN ({placeholders}) AND available_at_epoch <= ?
                    ORDER BY symbol, fiscal_year, quarter, revision_number, id
                    """,
                    [*chunk, to_epoch(as_of)],
                ).fetchall()
                for row in rows:
                    filing = row_to_filing(row)
                    if filing is not None:
                        result[filing.symbol].append(filing)
        return result

    def load_financial_audit_evidence(self, symbol, as_of, *, periods, uow=None):
        """Read verified originals only; never certifies persisted audit flags.

        This small native R2 companion does not activate or write a snapshot.
        The work-local raw parse memo owns any reused facts. Callers match each
        proof to the exact selected filing hash/revision, not merely its period.
        """
        return self.load_v7_financial_sources(symbol, as_of, periods=periods, uow=uow)["financial_audits"]

    def load_v7_financial_sources(self, symbol, as_of, *, periods, uow=None, source_repair=False, market="TWSE", audit_repair_r7=False):
        """Bounded R2 source proofs plus annual EPS; no network or snapshot writes."""
        from infrastructure.fundamentals.financial_companions import verified_source_companions
        from domain.les_v7.valuation_earnings import valuation_earnings
        from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
        selected = set(periods)
        rows = self.load_filings_many([symbol], as_of=as_of, uow=uow).get(symbol, [])
        proofs, supplements, earnings, repaired, signed, r7_audits = {}, {}, {}, {}, {}, {}
        if self._raw_root is None:
            return dict(financial_audits=proofs, financial_supplements=supplements,
                        **({"financial_audits_r7": r7_audits} if audit_repair_r7 else {}))
        for filing in rows:
            if (filing.fiscal_year, filing.quarter) not in selected:
                continue
            if (source_repair or audit_repair_r7) and pd.Timestamp(filing.available_at) > pd.Timestamp(as_of):
                # The header index is second-granularity; the new source port
                # must not expose a later subsecond observation at this cutoff.
                continue
            path = (self._raw_root.parent / filing.raw_path).resolve()
            if not path.is_relative_to(self._raw_root.resolve()):
                continue
            decode_options = ({"include_decoding": True}
                              if source_repair and (filing.fiscal_year, filing.quarter) == max(selected) else {})
            try:
                stat = path.stat()
                parsed = _raw_normalized_facts(path, filing.content_sha256,
                    (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size), V6_FILING_SCHEMA_VERSION,
                    **decode_options)
            except OSError:
                continue
            if parsed is None:
                continue
            facts, refs = parsed[:2]
            source = FiledFinancials(fiscal_year=filing.fiscal_year, quarter=filing.quarter,
                available_at=filing.available_at, revision_number=filing.revision_number,
                facts=facts, correction_description=filing.correction_description,
                availability_source=filing.availability_source, filing_id=filing.filing_id,
                content_sha256=filing.content_sha256, source_url=filing.source_url,
                period_end=filing.period_end, statement_scope=filing.statement_scope,
                audit_status=filing.audit_status, taxonomy_schema_refs=refs)
            companion = verified_source_companions(source, symbol=symbol)
            proofs[filing.content_sha256] = companion["financial_audit"]
            supplements[filing.content_sha256] = companion["financial_supplement"]
            if audit_repair_r7 and (filing.fiscal_year, filing.quarter) == max(selected):
                # An additive same-original interpretation, never a substituted
                # report or a rewritten availability timestamp. Historical
                # companions and their R5/R6 semantics remain unchanged.
                from domain.les_v7.financial_audit_r7 import audit_source_r7
                r7_audits[filing.content_sha256] = audit_source_r7(
                    source, symbol=symbol, market=market, original_verified=True)
            # Only current financial safety consumes the new audit/cash proof.
            # Annual EPS companions remain independent; do not rebuild every
            # historical financial graph just to repair the current report.
            if source_repair and (filing.fiscal_year, filing.quarter) == max(selected):
                from domain.les_v7.financial_sources import ixbrl_record_r2
                from domain.les_v7.financial_sources_r5 import repair_source_record_r5, _CASH_FIELDS
                canonical_row = ixbrl_record_r2(symbol, source, market=market, original_verified=True)
                fixed = repair_source_record_r5(canonical_row, source, symbol=symbol, market=market, original_verified=True)
                decoding = json.loads(parsed[2])
                if decoding["mode"] == "STRICT_FACT_RANGES":
                    fixed["source_repair_r5"]["source_decoding"] = decoding
                keys = ("symbol", "market", "source", "source_url", "period_end", "available_at",
                        "content_sha256", "revision_id", "scope", "currency", "unit")
                repaired[filing.content_sha256] = {
                    **{key: fixed[key] for key in keys},
                    "fields": {key: fixed.get(key) for key in _CASH_FIELDS},
                    "field_evidence": {key: fixed["field_evidence"].get(key, {}) for key in _CASH_FIELDS},
                    "audit_evidence_r5": fixed["audit_evidence_r5"],
                    "source_repair_r5": fixed["source_repair_r5"],
                }
                from infrastructure.fundamentals.signed_audit import SignedAuditStore
                from domain.les_v7.signed_audit import audit_proof_r6, AVAILABLE_AT
                if pd.Timestamp(as_of) >= pd.Timestamp(AVAILABLE_AT):
                    packet = SignedAuditStore(self._raw_root.parent / "signed-audit").select(fixed, cutoff=as_of)
                    if packet is not None:
                        signed[filing.content_sha256] = audit_proof_r6(source, fixed, packet,
                            symbol=symbol, market=market, cutoff=as_of)
            if filing.quarter == 4:
                earnings[filing.content_sha256] = valuation_earnings(
                    source, symbol=symbol, original_verified=True, as_of=as_of)
        result = dict(financial_audits=proofs, financial_supplements=supplements)
        if earnings:
            result["valuation_earnings"] = earnings
        if source_repair:
            result["financial_repairs_r5"] = repaired
            if signed:
                result["signed_financial_audits"] = signed
        if audit_repair_r7:
            result["financial_audits_r7"] = r7_audits
        return result

    def load_filed_financials(
        self,
        symbol: str,
        as_of: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
        connection: sqlite3.Connection | None = None,
        periods: tuple[tuple[int, int], ...] | None = None,
    ) -> list[FiledFinancials]:
        if periods is not None and not periods:
            return []
        period_predicate, params = "", [symbol, to_epoch(as_of)]
        if periods is not None:
            if any(type(y) is not int or type(q) is not int or not 1 <= q <= 4 for y, q in periods):
                raise ValueError("INVALID_FILING_PERIOD_FILTER")
            period_predicate = " AND (" + " OR ".join("(f.fiscal_year=? AND f.quarter=?)" for _ in periods) + ")"
            params.extend(v for item in periods for v in item)
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
                """ + period_predicate + """
                ORDER BY f.fiscal_year, f.quarter, f.revision_number, x.id
                """,
                params,
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

        result = []
        for filing, facts in grouped.values():
            normalized, schema_refs = self._resolve_cached_document(filing, facts)
            result.append(FiledFinancials(
                fiscal_year=filing.fiscal_year,
                quarter=filing.quarter,
                available_at=filing.available_at,
                revision_number=filing.revision_number,
                facts=normalized,
                correction_description=filing.correction_description,
                availability_source=filing.availability_source,
                filing_id=filing.filing_id,
                content_sha256=filing.content_sha256,
                source_url=filing.source_url,
                correction_announced_on=filing.correction_announced_on,
                period_end=filing.period_end,
                statement_scope=filing.statement_scope,
                audit_status=filing.audit_status,
                taxonomy_schema_refs=schema_refs,
            ))
        return result

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
    def filing_set_hash(filings: list[FiledFinancials] | list[MopsFiling]) -> str:
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
            for filing in sorted(filings, key=lambda item: (
                item.fiscal_year, item.quarter, item.revision_number, item.filing_id or 0,
            ))
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
        uow: FundamentalUnitOfWork | None = None,
    ) -> dict[str, CachedFundamentalBundle]:
        self._ensure_schema()
        requested = {
            str(symbol).strip(): str(sector).strip()
            for symbol, sector in symbol_sectors.items()
            if str(symbol).strip()
        }
        if not requested:
            return {}
        query = """
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
        symbols = list(requested)
        bundles: dict[str, CachedFundamentalBundle] = {}
        with borrow_connection(self._connect, None, uow) as connection:
            for offset in range(0, len(symbols), self._BULK_SYMBOL_CHUNK_SIZE):
                chunk = symbols[offset:offset + self._BULK_SYMBOL_CHUNK_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    query.format(placeholders=placeholders),
                    [*chunk, derivation_version, to_epoch(as_of)],
                )
                for row in rows:
                    symbol = str(row[0])
                    stored_sector = str(row[8])
                    if requested.get(symbol) != stored_sector:
                        continue
                    bundles[symbol] = CachedFundamentalBundle(
                        symbol=symbol, sector=stored_sector,
                        filing_set_sha256=str(row[1]), derivation_version=str(row[2]),
                        available_at_epoch=int(row[3]), latest_period=(int(row[4]), int(row[5])),
                        annual_history_years=int(row[6]), full_catalog_checked_at_epoch=int(row[7]),
                        bundle=self._decode_bundle(str(row[9]), str(row[10])),
                    )
        return bundles

    def load_cached_bundle_headers(
        self, symbol_sectors: Mapping[str, str], *, as_of: str,
        derivation_version: str, uow: FundamentalUnitOfWork | None = None,
    ) -> dict[str, CachedFundamentalHeader]:
        """Bounded metadata JOIN; deliberately excludes both large JSON columns."""
        requested = {str(s).strip(): str(sector).strip() for s, sector in symbol_sectors.items()
                     if str(s).strip()}
        if not requested:
            return {}
        self._ensure_schema()
        result = {}
        symbols = list(requested)
        with measure_stage("fundamental_snapshot_headers"), borrow_connection(self._connect, None, uow) as active:
            for offset in range(0, len(symbols), self._BULK_SYMBOL_CHUNK_SIZE):
                chunk = symbols[offset:offset + self._BULK_SYMBOL_CHUNK_SIZE]
                rows = active.execute(
                    """SELECT rs.symbol, rs.active_filing_set_sha256,
                              rs.active_derivation_version, rs.active_snapshot_available_at_epoch,
                              rs.active_latest_fiscal_year, rs.active_latest_quarter,
                              rs.active_annual_history_years, rs.full_catalog_checked_at_epoch,
                              fs.sector, rs.active_snapshot_cache_key,
                              fs.filing_schema_version, fs.has_mops_filings
                       FROM refresh_state AS rs JOIN fundamental_snapshots AS fs
                         INDEXED BY fundamental_snapshots_header
                         ON fs.symbol = rs.symbol AND fs.filing_set_sha256 = rs.active_snapshot_cache_key
                       WHERE rs.symbol IN ({}) AND rs.active_derivation_version = ?
                         AND rs.active_snapshot_available_at_epoch <= ?""".format(",".join("?" for _ in chunk)),
                    [*chunk, derivation_version, to_epoch(as_of)],
                )
                count_work("fundamental_header_queries")
                for row in rows:
                    if requested.get(str(row[0])) != str(row[8]):
                        continue
                    result[str(row[0])] = CachedFundamentalHeader(
                        symbol=str(row[0]), sector=str(row[8]), filing_set_sha256=str(row[1]),
                        derivation_version=str(row[2]), available_at_epoch=int(row[3]),
                        latest_period=(int(row[4]), int(row[5])), annual_history_years=int(row[6]),
                        full_catalog_checked_at_epoch=int(row[7]), snapshot_cache_key=str(row[9]),
                        filing_schema_version=str(row[10]) if row[10] is not None else None,
                        has_mops_filings=bool(row[11]) if row[11] is not None else None,
                    )
                    count_work("fundamental_headers_loaded")
        return result

    @staticmethod
    def _decode_bundle(annual_json: str, snapshot_json: str) -> FundamentalBundle:
        with measure_stage("fundamental_snapshot_decode"):
            count_work("fundamental_snapshot_decodes")
            count_work("fundamental_snapshot_json_characters", len(annual_json) + len(snapshot_json))
            return FundamentalBundle(
                annual_fundamentals=pd.read_json(io.StringIO(annual_json), orient="records"),
                snapshot=json.loads(snapshot_json),
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
        self._ensure_schema()
        with measure_stage("fundamental_snapshot_read"), borrow_connection(self._connect, connection, uow) as active:
            row = active.execute(
                """
                SELECT annual_json, snapshot_json
                FROM fundamental_snapshots
                WHERE symbol = ? AND filing_set_sha256 = ? AND sector = ?
                """,
                (symbol, filing_set_sha256, sector),
            ).fetchone()
            count_work("fundamental_snapshot_reads")
        if row is None:
            return None
        return self._decode_bundle(str(row[0]), str(row[1]))


__all__ = ["MopsQueryRepository"]
