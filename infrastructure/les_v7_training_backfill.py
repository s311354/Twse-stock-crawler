"""Bounded cache-only market repair; preserve original evidence and unresolved gaps.

No HTTP, inferred sessions, adjusted prices, financial replacements or account
state. A conflicting cache row is retained beside the old row for the normal
source adapter to reject, never silently preferred as a correction.
"""
from copy import deepcopy
from contextlib import closing
from datetime import date, timedelta
import json
from pathlib import Path
import sqlite3

from application.replay_contracts import canonical_json, digest
from application.research_evidence import RAW_EVIDENCE_SCHEMA
from domain.models import TwseColumns as C
from infrastructure.les_v7_training_source import MAX_BYTES, MAX_DAYS, MAX_SYMBOLS, FINAL_VINTAGE
from infrastructure.replay_evidence import read_only, load_evidence
from infrastructure.research_archive import create_dataset, utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments

VERSION = "LES_V7_LOCAL_MARKET_BACKFILL_R1"
MAX_CACHE_DAY_BYTES = 16 * 1024 * 1024
MAX_CACHE_ROW_BYTES = 64 * 1024


def _cached_market_records(cache, parent, history_start, *, connection=None):
    """Filter selected rows in SQLite, stream days, never expand sector breadth."""
    if connection is None:
        with closing(read_only(cache)) as snapshot:
            snapshot.execute("BEGIN")
            return _cached_market_records(cache, parent, history_start, connection=snapshot)
    market_type = "TPEX:24:v1" if parent["market"] == "TPEX" else parent["sector"]
    if parent["market"] not in {"TWSE", "TPEX"}:
        raise ValueError("BACKFILL_SOURCE_IDENTITY_OR_UNITS_MISMATCH")
    placeholders = ",".join("?" for _ in parent["symbols"])
    records, size = [], 0
    try:
        if connection is not None:
            headers = connection.execute("""SELECT trade_date,dataset,market_type,length(CAST(payload_json AS BLOB))
                FROM market_history_cache WHERE trade_date BETWEEN ? AND ? AND schema_version=1
                AND ((dataset='STOCK_TABLE' AND market_type IN (?,?)) OR (dataset='TAIEX' AND market_type='IND'))
                ORDER BY trade_date,dataset,market_type""",
                (history_start.strftime("%Y%m%d"), parent["end"].replace("-", ""), market_type,
                 "ALLBUT0999" if parent["market"] == "TWSE" else market_type))
            for count, (day, dataset, source_type, byte_count) in enumerate(headers, 1):
                if count > MAX_DAYS * 3 or byte_count > MAX_CACHE_DAY_BYTES:
                    raise ValueError("BACKFILL_CACHE_DAY_OR_COUNT_LIMIT")
                if dataset == "STOCK_TABLE":
                    query = """SELECT stock_row.value FROM market_history_cache AS cache
                        JOIN json_each(cache.payload_json) AS stock_row
                        WHERE cache.trade_date=? AND cache.dataset=? AND cache.market_type=? AND cache.schema_version=1
                        AND CAST(json_extract(stock_row.value,'$[0]') AS TEXT) IN ({})""".format(placeholders)
                    data = []
                    for row_count, (serialized,) in enumerate(connection.execute(query,
                            (day, dataset, source_type, *parent["symbols"])), 1):
                        if row_count > MAX_SYMBOLS * 4 or len(serialized.encode("utf-8")) > MAX_CACHE_ROW_BYTES:
                            raise ValueError("BACKFILL_SELECTED_ROW_SIZE_OR_COUNT_LIMIT")
                        data.append(json.loads(serialized))
                else:
                    serialized, = connection.execute("""SELECT payload_json FROM market_history_cache
                        WHERE trade_date=? AND dataset=? AND market_type=? AND schema_version=1""",
                        (day, dataset, source_type)).fetchone()
                    data = json.loads(serialized)
                item = dict(date=day, dataset=dataset, market_type=source_type, data=data,
                            source=parent["market"] if dataset == "STOCK_TABLE" else "TWSE_TAIEX")
                size += len(canonical_json(item).encode("utf-8"))
                if size > MAX_BYTES:
                    raise ValueError("BACKFILL_SELECTED_MARKET_SIZE_LIMIT")
                records.append(item)
    except sqlite3.Error as error:
        raise ValueError("BACKFILL_MARKET_CACHE_QUERY_FAILED") from error
    return records


def merge_market_backfill(parent_id, parent, supplement, *, history_start):
    """Merge a same-universe final-daily snapshot without replacing old records."""
    if digest(parent) != parent_id:
        raise ValueError("BACKFILL_PARENT_HASH_MISMATCH")
    fields = ("schema", "market", "sector", "symbols", "price_unit", "volume_unit", "amount_unit", "market_policy")
    if (parent.get("schema") != RAW_EVIDENCE_SCHEMA or parent.get("market_policy") != FINAL_VINTAGE
            or any(parent.get(k) != supplement.get(k) for k in fields)
            or (parent.get("price_unit"), parent.get("volume_unit"), parent.get("amount_unit")) != ("TWD", "SHARES", "TWD")):
        raise ValueError("BACKFILL_SOURCE_IDENTITY_OR_UNITS_MISMATCH")
    first, end = date.fromisoformat(parent["start"]), date.fromisoformat(parent["end"])
    if (type(history_start) is not date or history_start > first or history_start > end
            or (end - history_start).days + 1 > MAX_DAYS
            or supplement.get("start") != history_start.isoformat() or supplement.get("end") != end.isoformat()
            or not 1 <= len(parent["symbols"]) <= MAX_SYMBOLS):
        raise ValueError("BACKFILL_BOUNDED_PAST_RANGE_REQUIRED")
    payload = deepcopy(parent)
    original_ids = {digest(row) for row in parent["bars"]}
    additions = {}
    for item in supplement["bars"]:
        day = date.fromisoformat(item["date"][:4] + "-" + item["date"][4:6] + "-" + item["date"][6:])
        if not history_start <= day <= end:
            raise ValueError("BACKFILL_BAR_OUTSIDE_PINNED_RANGE")
        if item.get("dataset") not in {"STOCK_TABLE", "TAIEX"}:
            raise ValueError("BACKFILL_MARKET_DATASET_UNSUPPORTED")
        if item["dataset"] == "STOCK_TABLE":
            if item.get("source") != parent["market"] or any(
                    not isinstance(r, list) or len(r) < C.REQUIRED_WIDTH or str(r[C.STOCK_NO]) not in parent["symbols"]
                    for r in item["data"]):
                raise ValueError("BACKFILL_STOCK_SOURCE_OR_UNIVERSE_MISMATCH")
        elif item.get("source") != "TWSE_TAIEX":
            raise ValueError("BACKFILL_BENCHMARK_SOURCE_MISMATCH")
        sha = digest(item)
        if sha not in original_ids:
            additions[sha] = deepcopy(item)
    payload["bars"].extend(additions[k] for k in sorted(additions))
    payload["start"] = history_start.isoformat()
    # The raw tri-state list is not a substitute for verified annual documents.
    # calendar_at() still proves every date from the original bound sources.
    days = payload["calendar"].setdefault("days", {})
    new_calendar_dates = []
    for i in range((end - history_start).days + 1):
        day = (history_start + timedelta(days=i)).isoformat()
        if day not in days:
            days[day] = "UNKNOWN"
            new_calendar_dates.append(day)
    markers = {}
    for item in payload["bars"]:
        if item.get("dataset") == "STOCK_TABLE":
            for row in item["data"]:
                sign = str(row[C.CHANGE_SIGN]).replace("<p>", "").replace("</p>", "").strip()
                if sign == "X":
                    detail = dict(symbol=str(row[C.STOCK_NO]), date=item["date"], source=item.get("source"),
                        source_row_sha256=digest(row), source_record_sha256=digest(item),
                        reason="SOURCE_ACTION_MARKER_PENDING_VERIFICATION")
                    markers[digest(detail)] = detail
    repair = dict(schema="les-v7-market-backfill/v1", version=VERSION, parent_evidence_id=parent_id,
        requested_start=history_start.isoformat(), fixed_end=end.isoformat(), fixed_universe=parent["symbols"],
        original_record_count=len(parent["bars"]), added_record_count=len(additions),
        added_records_sha256=digest([additions[k] for k in sorted(additions)]),
        unknown_calendar_dates_added=len(new_calendar_dates), pending_action_markers=[markers[k] for k in sorted(markers)],
        policy="LOCAL_CACHE_ONLY_APPEND_NO_OVERRIDE_NO_AVAILABILITY_BACKDATE",
        original_observation_receipts_added=False, historical_information_set_certified=False)
    payload["market_backfill"] = repair
    return payload, repair


def backfill_training_evidence(evidence_path, *, market_cache, history_start, store, allow_synthetic=False):
    """Read the cache in one snapshot; publish a NEW hash-bound evidence dataset."""
    path, cache, store = Path(evidence_path), Path(market_cache), Path(store)
    if not cache.is_file():
        raise ValueError("BACKFILL_MARKET_CACHE_MISSING")
    parent_id, parent = load_evidence(path, max_bytes=MAX_BYTES)
    if parent.get("purpose") == "SYNTHETIC_TEST_ONLY" and not allow_synthetic:
        raise ValueError("SYNTHETIC_BACKFILL_REQUIRES_EXPLICIT_OPT_IN")
    if parent.get("schema") != RAW_EVIDENCE_SCHEMA or parent.get("market_policy") != FINAL_VINTAGE:
        raise ValueError("BACKFILL_SOURCE_IDENTITY_OR_UNITS_MISMATCH")
    if type(history_start) is not date or history_start > date.fromisoformat(parent["start"]):
        raise ValueError("BACKFILL_BOUNDED_PAST_RANGE_REQUIRED")
    if ((date.fromisoformat(parent["end"]) - history_start).days + 1 > MAX_DAYS
            or not 1 <= len(parent["symbols"]) <= MAX_SYMBOLS):
        raise ValueError("BACKFILL_BOUNDED_PAST_RANGE_REQUIRED")
    documents = EvidenceDocuments.for_evidence(path, parent_id)
    documents.verify(parent)
    collected_at = utc_now()
    fields = ("schema", "market", "sector", "symbols", "price_unit", "volume_unit", "amount_unit", "market_policy", "end")
    supplement = {key: parent[key] for key in fields}
    supplement.update(start=history_start.isoformat(), bars=_cached_market_records(cache, parent, history_start))
    payload, report = merge_market_backfill(parent_id, parent, supplement, history_start=history_start)
    if len(canonical_json(dict(evidence_id=digest(payload), payload=payload)).encode("utf-8")) > MAX_BYTES:
        raise ValueError("BACKFILL_EVIDENCE_OUTPUT_SIZE_LIMIT")
    if load_evidence(path, max_bytes=MAX_BYTES)[0] != parent_id:
        raise ValueError("BACKFILL_PARENT_CHANGED_DURING_JOB")
    documents.check()
    # Original reports/revisions/proofs are copied verbatim, not refreshed from
    # today's financial cache. Collection wall time is outside content identity.
    destination = create_dataset(payload, store, parent=path, collected_at=collected_at,
        purpose="V7_TRAINING_MARKET_BACKFILL_RESEARCH_ONLY", document_reader=documents.read)
    write_once(destination.parent / "market_backfill.json", report)
    return dict(evidence=str(destination), evidence_id=destination.parent.name,
                parent_evidence_id=parent_id, repair_report=str(destination.parent / "market_backfill.json"))
