"""Prepare portable evidence from read-only archives. No research-time HTTP."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import base64
import hashlib
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from application.replay_contracts import canonical_json, digest
from application.research_evidence import RAW_EVIDENCE_SCHEMA
from domain.market_breadth import derive_twse_market_breadth
from domain.models import Stocktype
from infrastructure.fundamentals.mops_queries import MopsQueryRepository

EVIDENCE_SCHEMA = "replay-evidence/v1"


def read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def save_evidence(payload: dict, destination: Path, *, document_reader=None) -> str:
    """Never overwrite a previously pinned evidence artifact."""
    from infrastructure.research_documents import EvidenceDocuments
    payload = EvidenceDocuments(destination.parent / "objects").externalize(payload, document_reader=document_reader)
    evidence_id = digest(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json({"evidence_id": evidence_id, "payload": payload}))
    return evidence_id


def load_evidence(path: str | Path) -> tuple[str, dict]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = document["payload"]
    if payload.get("schema") not in {EVIDENCE_SCHEMA, RAW_EVIDENCE_SCHEMA} or digest(payload) != document["evidence_id"]:
        raise ValueError("REPLAY_EVIDENCE_SIGNATURE_MISMATCH")
    return document["evidence_id"], payload


def collect_evidence(*, market_cache: Path, mops_cache: Path | None,
                     restriction_cache: Path | None, restriction_file: Path | None,
                     calendar: dict, symbols: tuple[str, ...], market: str, sector: str,
                     start: dt.date, end: dt.date, calibration_file: Path | None = None,
                     raw: bool = False, publication_file: Path | None = None, document_store=None) -> dict:
    """Freeze raw cached bars, filing versions/facts and restriction observations.

    Calendar input must be an explicitly prepared official-calendar artifact.
    Missing databases are documented gaps, not empty successful observations.
    Normalized facts retain original provenance; export time is never available_at.
    """
    if start > end or not symbols or market not in {"TWSE", "TPEX"}:
        raise ValueError("INVALID_EVIDENCE_RANGE_OR_UNIVERSE")
    if calendar.get("source") not in {"TWSE_OFFICIAL", "TPEX_OFFICIAL"}:
        raise ValueError("OFFICIAL_CALENDAR_SOURCE_REQUIRED")
    days = calendar.get("days", {})
    for offset in range((end - start).days + 1):
        day = (start + dt.timedelta(days=offset)).isoformat()
        if days.get(day) not in {"OPEN", "CLOSED"} and not raw:
            raise ValueError(f"CALENDAR_COVERAGE_UNKNOWN:{day}")
    payload = dict(schema=RAW_EVIDENCE_SCHEMA if raw else EVIDENCE_SCHEMA, market=market, sector=sector, symbols=sorted(set(symbols)),
                   start=start.isoformat(), end=end.isoformat(), calendar=calendar,
                   bars=[], breadth={}, filings={}, batches={}, restrictions=[], restriction_ledger=None,
                   preparation_gaps=[], market_policy="PINNED_FINAL_DAILY_ARCHIVE_NOT_ORIGINAL_VINTAGE")
    if raw:
        payload.update(raw_documents={}, breadth_sources={}, universe_policy="EXPLICIT_FIXED_UNIVERSE_NO_WINNER_FILTER",
                       price_unit="TWD", volume_unit="SHARES", amount_unit="TWD")
    # Preserve complete source rows. The original schema decoder remains the
    # authority, including TPEx's canonical prefix and audit suffix.
    market_type = "TPEX:24:v1" if market == "TPEX" else sector
    if market_cache.is_file():
        with closing(read_only(market_cache)) as connection:
            connection.execute("BEGIN")
            records = connection.execute("""SELECT trade_date, dataset, market_type, payload_json
                FROM market_history_cache WHERE trade_date BETWEEN ? AND ? AND schema_version=1
                AND ((dataset='STOCK_TABLE' AND market_type IN (?, ?))
                  OR (dataset='TAIEX' AND market_type='IND'))
                ORDER BY trade_date, dataset, market_type""",
                (start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), market_type,
                 "ALLBUT0999" if market == "TWSE" else market_type)).fetchall()
            for date, dataset, source_type, serialized in records:
                data = json.loads(serialized)
                if dataset == "STOCK_TABLE":
                    # Match the daily model's industry breadth, before selecting
                    # tracked symbols. A full-market fallback is not an industry.
                    if source_type == market_type:
                        universe = "TPEX_SEMI" if market == "TPEX" else next(
                            (s.name for s in Stocktype if str(s.value[0]) == sector), sector)
                        day = dt.datetime.strptime(date, "%Y%m%d").date().isoformat()
                        if raw:
                            payload["breadth_sources"][day] = dict(rows=data, universe=universe, market_type=source_type)
                        else:
                            payload["breadth"][day] = dict(
                                snapshot=derive_twse_market_breadth(data, as_of=day, universe=universe, source_final=True).to_mapping(),
                                source_rows_sha256=digest(data), market_type=source_type)
                    data = [row for row in data if str(row[0]) in symbols]
                payload["bars"].append(dict(date=date, dataset=dataset, market_type=source_type,
                                            data=data, source=market if dataset == "STOCK_TABLE" else "TWSE_TAIEX"))
    else:
        payload["preparation_gaps"].append("MARKET_CACHE_MISSING")
    if mops_cache is not None and mops_cache.is_file():
        with closing(read_only(mops_cache)) as connection:
            connection.execute("BEGIN")
            query = MopsQueryRepository(lambda: read_only(mops_cache), lambda: None,
                                        raw_root=mops_cache.parent / "raw")
            for symbol in symbols:
                if raw:
                    # Query metadata only: raw files are reparsed by the chosen
                    # model version, not old normalized facts or snapshots.
                    connection.row_factory = sqlite3.Row
                    rows = connection.execute("SELECT * FROM filings WHERE symbol=? ORDER BY fiscal_year,quarter,report_id,revision_number,content_sha256", (symbol,)).fetchall()
                    payload["filings"][symbol] = []
                    for row in rows:
                        values = {key: row[key] for key in (
                            "fiscal_year", "quarter", "available_at", "revision_number", "correction_description",
                            "availability_source", "content_sha256", "source_url", "correction_announced_on",
                            "period_end", "statement_scope", "audit_status")}
                        values.update(receipt=dict(fetched_at=row["fetched_at"], report_id=row["report_id"]),
                                      source_version=f"observed-content:{row['content_sha256']}")
                        path = (mops_cache.parent / row["raw_path"]).resolve()
                        try:
                            if not path.is_relative_to((mops_cache.parent / "raw").resolve()):
                                raise ValueError("RAW_PATH_OUTSIDE_CACHE")
                            content = path.read_bytes()
                            if hashlib.sha256(content).hexdigest() != row["content_sha256"]:
                                raise ValueError("RAW_FILING_HASH_MISMATCH")
                            original = dict(url=row["source_url"], sha256=row["content_sha256"],
                                            retrieved_at=row["fetched_at"])
                            if document_store is not None:
                                original["body_ref"] = document_store(content)
                            else:
                                # Compatibility for in-memory callers; persisted archives externalize this.
                                original["body_base64"] = base64.b64encode(content).decode("ascii")
                            payload["raw_documents"][row["content_sha256"]] = original
                        except (OSError, ValueError) as error:
                            payload["preparation_gaps"].append(dict(symbol=symbol, period=f"{row['fiscal_year']}Q{row['quarter']}", reason=str(error)))
                        payload["filings"][symbol].append(values)
                else:
                    records = query.load_filed_financials(symbol, "2100-01-01T00:00:00Z", connection=connection)
                    receipts = {row[0]: dict(fetched_at=row[1], report_id=row[2]) for row in connection.execute(
                        "SELECT id, fetched_at, report_id FROM filings WHERE symbol=?", (symbol,))}
                    payload["filings"][symbol] = [dict(asdict(record), receipt=receipts.get(record.filing_id, {})) for record in records]
                payload["batches"][symbol] = [json.loads(row[0]) for row in connection.execute(
                    "SELECT snapshot_json FROM financial_batch_snapshots WHERE symbol=? ORDER BY observed_at_epoch,id", (symbol,))]
    else:
        payload["preparation_gaps"].append("MOPS_CACHE_MISSING")
    if restriction_cache is not None and restriction_cache.is_file():
        with closing(read_only(restriction_cache)) as connection:
            for symbol, sha, serialized in connection.execute(
                "SELECT symbol, digest, evidence_json FROM restriction_observations WHERE market=? ORDER BY available_at_ns,digest", (market,)):
                if symbol in symbols:
                    if hashlib.sha256(serialized.encode()).hexdigest() != sha:
                        raise ValueError("RESTRICTION_ARCHIVE_HASH_MISMATCH")
                    payload["restrictions"].append(dict(symbol=symbol, evidence=json.loads(serialized)))
    if restriction_file is not None:
        from application.restriction_history import HistoricalRestrictionLedger
        ledger = json.loads(restriction_file.read_text(encoding="utf-8"))
        HistoricalRestrictionLedger(ledger)  # Source documents, quotes, coverage and hashes.
        payload["restriction_ledger"] = ledger
    if not payload["restrictions"] and payload["restriction_ledger"] is None:
        payload["preparation_gaps"].append("HISTORICAL_RESTRICTIONS_MISSING")
    payload["kelly_calibrations"] = []
    if calibration_file is not None:
        records = json.loads(calibration_file.read_text(encoding="utf-8"))
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise ValueError("CALIBRATION_FILE_REQUIRES_JSON_RECORDS")
        payload["kelly_calibrations"] = records
    if publication_file is not None:
        from application.research_evidence import filing_binding, verify_publication
        proofs = json.loads(publication_file.read_text(encoding="utf-8"))
        if not raw or not isinstance(proofs, dict):
            raise ValueError("PUBLICATION_IMPORT_REQUIRES_V2_HASH_MAPPING")
        matched = set()
        for symbol, records in payload["filings"].items():
            for record in records:
                sha = record["content_sha256"]
                if sha in proofs:
                    proof = proofs[sha]
                    record["source_version"] = proof["binding"]["source_version"]
                    verify_publication(proof, binding=filing_binding(symbol, record))
                    record["publication_proof"] = proof
                    matched.add(sha)
        if matched != set(proofs):
            raise ValueError("PUBLICATION_PROOF_WITHOUT_ORIGINAL_DOCUMENT")
    return payload


def prepare_evidence(*, destination: Path, **kwargs) -> str:
    if kwargs.get("raw") and "document_store" not in kwargs:
        from infrastructure.research_documents import EvidenceDocuments
        kwargs["document_store"] = EvidenceDocuments(destination.parent / "objects").store
    return save_evidence(collect_evidence(**kwargs), destination)
