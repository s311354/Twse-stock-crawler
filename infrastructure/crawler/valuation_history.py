"""Cache-only monthly pricing evidence for native R12 valuation research.

This does not expand technical histories or certify historical observation.
Calendar observations and cache payloads are read locally, without DDL, HTTP,
TAIEX requirements or writes. Missing exact month-end quotes stay missing.
"""
from __future__ import annotations

import calendar as calendar_module
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import sqlite3

from domain.les_v7.contracts import identity, timestamp
from domain.models import TwseColumns
from twse.parser import parse_price


SCHEMA = "les-v7-monthly-valuation-history/v1"
SELECTION_SCHEMA = "les-v7-monthly-valuation-selection/v1"
MANIFEST_SCHEMA = "les-v7-monthly-valuation-history-manifest/v1"
CALENDAR_SCHEMA = "les-v7-monthly-valuation-calendar/v1"
AVAILABILITY = "CURRENT_OBSERVATION_OF_OFFICIAL_FINAL_DAILY_CACHE_NOT_HISTORICAL_OBSERVATION"
MARKET_TYPE = "13"
MAX_MONTHS = 36
MAX_SYMBOLS = 2000
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


def _window(reference_session, months):
    if type(months) is not int or not 1 <= months <= MAX_MONTHS:
        raise ValueError("VALUATION_HISTORY_MONTH_LIMIT_INVALID")
    reference = date.fromisoformat(str(reference_session)[:10])
    end = reference.replace(day=1) - timedelta(days=1)
    absolute = reference.year * 12 + reference.month - 1 - months
    year, month = divmod(absolute, 12)
    return date(year, month + 1, 1), end


def offline_calendar_projection(*, reference_session, observed_at, months=MAX_MONTHS):
    """Use verified observed annual documents and reviewed exceptions only.

    A missing official year is UNKNOWN. The existing calendar interprets its
    verified annual schedule; this helper never guesses open weekdays and
    never instantiates the calendar's default network transport.
    """
    from infrastructure.calendar.annual_observations import ObservedAnnualSchedules
    from infrastructure.calendar.emergency_announcements import ReviewedEmergencyAnnouncements
    from infrastructure.calendar.twse_calendar import TwseCalendar
    observed = timestamp(observed_at)
    start, end = _window(reference_session, months)
    archive = ObservedAnnualSchedules(cutoff=observed)
    emergency = ReviewedEmergencyAnnouncements(cutoff=observed)
    emergency_evidence = emergency.evidence()
    calendars, documents, unavailable = {}, {}, []
    for year in range(start.year, end.year + 1):
        archived = archive.get(year)
        if archived is None:
            unavailable.append(year); continue
        payload, header = archived
        calendars[year] = TwseCalendar(fetcher=lambda payload=payload: payload,
            emergency_announcements=emergency, annual_archive=False)
        documents[str(year)] = {key: value for key, value in header.items() if key != "verified_at"}
    days = {}
    for offset in range((end - start).days + 1):
        current = start + timedelta(days=offset)
        known = calendars.get(current.year)
        days[current.isoformat()] = known.trading_day_status(current).value.upper() if known else "UNKNOWN"
    source_ids = sorted({header["sha256"] for header in documents.values()} | {emergency_evidence["manifestSha256"]})
    result = dict(schema=CALENDAR_SCHEMA, source="TWSE_OFFICIAL", reference_session=str(reference_session)[:10],
        window=dict(start=start.isoformat(), end=end.isoformat(), months=months),
        days=days, days_sha256=identity(days), source_ids=source_ids,
        available_at=observed.isoformat(), annual_documents=documents,
        emergency_evidence=emergency_evidence, missing_annual_years=unavailable,
        availability_basis="VERIFIED_OBSERVED_CALENDAR_NOT_HISTORICAL_PUBLICATION")
    result["calendar_id"] = identity(result)
    return result


def _month_ends(projection, start, end, observed):
    if (projection.get("schema") != CALENDAR_SCHEMA or projection.get("source") != "TWSE_OFFICIAL"
            or projection.get("days_sha256") != identity(projection.get("days"))
            or projection.get("calendar_id") != identity({key: value for key, value in projection.items() if key != "calendar_id"})
            or timestamp(projection["available_at"]) > observed
            or projection.get("window", {}).get("start") != start.isoformat()
            or projection.get("window", {}).get("end") != end.isoformat()
            or not projection.get("source_ids")
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in projection["source_ids"])):
        raise ValueError("VALUATION_HISTORY_CALENDAR_BINDING_INVALID")
    result, unknown = {}, []
    cursor = start
    while cursor <= end:
        key = cursor.strftime("%Y-%m")
        last = calendar_module.monthrange(cursor.year, cursor.month)[1]
        chosen = None
        for day in range(last, 0, -1):
            current = cursor.replace(day=day)
            state = projection["days"].get(current.isoformat(), "UNKNOWN")
            if state == "UNKNOWN":
                unknown.append(key); break
            if state == "OPEN":
                chosen = current.isoformat(); break
            if state != "CLOSED":
                raise ValueError("VALUATION_HISTORY_CALENDAR_STATE_INVALID")
        if chosen is not None:
            result[key] = chosen
        elif key not in unknown:
            unknown.append(key)  # No verified open session, not a guessed one.
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
    return result, unknown


def _read_days(database_path, dates, symbols):
    """One batch query, filtered to requested equities; no schema bootstrap."""
    path = Path(database_path)
    if not path.exists() or not dates or not symbols:
        return {}, {}
    date_slots, symbol_slots = ",".join("?" for _ in dates), ",".join("?" for _ in symbols)
    # The correlated aggregation returns requested raw rows only. The full
    # payload is retained once per month for source/membership verification,
    # not repeated once per symbol or copied into technical DataFrame attrs.
    query = f"""SELECT c.trade_date,c.cached_at_epoch,c.payload_json,
        (SELECT json_group_array(json_object('index',r.key,'row',json(r.value)))
           FROM json_each(c.payload_json) r
          WHERE CAST(json_extract(r.value,'$[0]') AS TEXT) IN ({symbol_slots}))
        FROM market_history_cache c
        WHERE c.dataset='STOCK_TABLE' AND c.market_type=? AND c.schema_version=1
          AND c.trade_date IN ({date_slots}) ORDER BY c.trade_date"""
    sources, selected = {}, {}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        for day, cached_at, payload, matches in connection.execute(query, [*symbols, MARKET_TYPE, *dates]):
            encoded = payload.encode("utf-8")
            if len(encoded) > MAX_PAYLOAD_BYTES:
                raise ValueError("VALUATION_HISTORY_CACHE_PAYLOAD_TOO_LARGE")
            payload_hash = hashlib.sha256(encoded).hexdigest()
            # Validate the complete envelope before accepting any projected row.
            original = json.loads(payload)
            if not isinstance(original, list) or not original or any(not isinstance(row, list) for row in original):
                raise ValueError("VALUATION_HISTORY_CACHE_PAYLOAD_SCHEMA_INVALID")
            source = dict(dataset="STOCK_TABLE", market_type=MARKET_TYPE, schema_version=1,
                trade_date=day, cached_at_epoch=cached_at, payload_sha256=payload_hash,
                payload_json=payload, cached_at_semantics="LAST_CACHE_UPSERT_NOT_ORIGINAL_PUBLICATION")
            source_id = identity(source)
            sources[source_id] = source
            selected[day] = dict(source_id=source_id, cached_at_epoch=cached_at, rows=json.loads(matches))
    return selected, sources


def collect_valuation_history(database_path, *, symbols, reference_session, observed_at,
                             calendar_projection, months=MAX_MONTHS, capture_clock=None):
    """Read existing monthly quotes, preserving source/gap diagnostics.

    The caller supplies the actual completed read observation clock. This is
    not an annual EPS availability assertion or a substitute for PIT checks.
    """
    requested = sorted(set(str(value).strip() for value in symbols))
    if not requested or len(requested) > MAX_SYMBOLS or any(not re.fullmatch(r"[0-9A-Z]{4,12}", value) for value in requested):
        raise ValueError("VALUATION_HISTORY_SYMBOL_SCOPE_INVALID")
    observed = timestamp(observed_at)
    start, end = _window(reference_session, months)
    ends, unknown = _month_ends(calendar_projection, start, end, observed)
    dates = sorted(day.replace("-", "") for day in ends.values())
    read_error = None
    try:
        loaded, documents = _read_days(database_path, dates, requested)
    except (sqlite3.Error, UnicodeError, ValueError, TypeError) as error:
        loaded, documents = {}, {}
        read_error = "VALUATION_HISTORY_CACHE_UNVERIFIED:" + type(error).__name__
    if capture_clock is not None:
        completed = timestamp(capture_clock())
        if completed < observed:
            raise ValueError("VALUATION_HISTORY_CAPTURE_CLOCK_REVERSED")
        observed = completed
    by_symbol = {}
    for symbol in requested:
        rows, records, gaps = [], [], []
        for month, day in ends.items():
            cached = loaded.get(day.replace("-", ""))
            if cached is None:
                gaps.append(dict(month=month, expected_session=day,
                    reason=read_error or "OFFICIAL_MONTH_END_CACHE_DAY_MISSING")); continue
            epoch = cached["cached_at_epoch"]
            if type(epoch) is not int or epoch < 0 or epoch > observed.timestamp():
                gaps.append(dict(month=month, expected_session=day, reason="CACHE_LAST_WRITE_TIME_UNVERIFIED_OR_FUTURE")); continue
            matching = [item for item in cached["rows"] if str(item["row"][0]) == symbol]
            if len(matching) != 1:
                gaps.append(dict(month=month, expected_session=day,
                    reason="SYMBOL_NOT_IN_MONTH_END_TABLE" if not matching else "CONFLICTING_SYMBOL_ROWS")); continue
            entry = matching[0]; raw = entry["row"]
            close = parse_price(raw[TwseColumns.CLOSE]) if len(raw) > TwseColumns.CLOSE else None
            if close is None or not isfinite(close) or close <= 0:
                gaps.append(dict(month=month, expected_session=day, reason="MONTH_END_CLOSE_UNVERIFIED")); continue
            rows.append(dict(Date=day, Close=close))  # Official raw table is TWD, never legacy ×1000.
            records.append(dict(symbol=symbol, session_date=day, price_unit="TWD", close=close,
                cache_source_id=cached["source_id"], raw_row_index=entry["index"], raw_row_sha256=identity(raw),
                raw_row=raw, cached_at_epoch=epoch))
        status = "UNKNOWN" if read_error or any(gap["reason"] in {"CONFLICTING_SYMBOL_ROWS", "CACHE_LAST_WRITE_TIME_UNVERIFIED_OR_FUTURE"} for gap in gaps) else "PARTIAL" if gaps or unknown else "COMPLETE"
        item = dict(symbol=symbol, market="TWSE", status=status, rows=rows, records=records,
            coverage=dict(months_requested=months, months_valid=len(rows), gaps=gaps,
                          unknown_calendar_months=list(unknown)),
            price_unit="TWD", availability_basis=AVAILABILITY)
        item["source_id"] = identity(item)
        by_symbol[symbol] = item
    result = dict(schema=SCHEMA, market="TWSE", market_type=MARKET_TYPE, symbols=requested,
        reference_session=str(reference_session)[:10], window=dict(start=start.isoformat(), end=end.isoformat(), months=months),
        observed_at=observed.isoformat(), availability_basis=AVAILABILITY,
        source_authenticity="CACHE_CONTENT_INTEGRITY_NOT_OFFICIAL_TRANSPORT_AUTHENTICITY_CERTIFICATION",
        historical_information_set_certified=False,
        calendar_evidence=calendar_projection, by_symbol=by_symbol, documents=documents,
        summary=dict(symbols=len(requested), calendar_months_known=len(ends), calendar_months_unknown=len(unknown),
                     cached_month_end_days=len(loaded), cache_error=read_error))
    result["evidence_id"] = identity(result)
    return result


def capture_valuation_history(database_path, *, symbols, reference_session, clock=None):
    """Native capture before the final research cutoff; no transport calls."""
    clock = clock or (lambda: datetime.now(timezone.utc))
    started = timestamp(clock())
    projection = offline_calendar_projection(reference_session=reference_session, observed_at=started)
    result = collect_valuation_history(database_path, symbols=symbols, reference_session=reference_session,
        observed_at=started, calendar_projection=projection, capture_clock=clock)
    result.pop("evidence_id")
    result["capture_started_at"] = started.isoformat()
    result["evidence_id"] = identity(result)
    return result


def verify_valuation_history(result):
    """Verify retained payload membership/units and outer immutable identity."""
    if result.get("schema") != SCHEMA or result.get("evidence_id") != identity({key: value for key, value in result.items() if key != "evidence_id"}):
        raise ValueError("VALUATION_HISTORY_CONTENT_ID_MISMATCH")
    observed = timestamp(result["observed_at"])
    start, end = _window(result["reference_session"], result["window"]["months"])
    ends, unknown = _month_ends(result["calendar_evidence"], start, end, observed)
    requested = result.get("symbols")
    if (not isinstance(requested, (list, tuple)) or not requested or len(requested) > MAX_SYMBOLS
            or list(requested) != sorted(set(requested)) or set(requested) != set(result["by_symbol"])
            or result.get("market") != "TWSE" or result.get("market_type") != MARKET_TYPE
            or result.get("window") != dict(start=start.isoformat(), end=end.isoformat(), months=result["window"]["months"])
            or len(result["documents"]) > MAX_MONTHS
            or result.get("historical_information_set_certified") is not False
            or result.get("source_authenticity") != "CACHE_CONTENT_INTEGRITY_NOT_OFFICIAL_TRANSPORT_AUTHENTICITY_CERTIFICATION"):
        raise ValueError("VALUATION_HISTORY_SCOPE_OR_WINDOW_INVALID")
    decoded = {}
    for source_id, source in result["documents"].items():
        if (identity(source) != source_id or source.get("market_type") != MARKET_TYPE
                or source.get("dataset") != "STOCK_TABLE" or source.get("schema_version") != 1
                or hashlib.sha256(source["payload_json"].encode("utf-8")).hexdigest() != source["payload_sha256"]):
            raise ValueError("VALUATION_HISTORY_CACHE_SOURCE_MISMATCH")
        rows = json.loads(source["payload_json"])
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise ValueError("VALUATION_HISTORY_CACHE_SOURCE_MISMATCH")
        decoded[source_id] = rows  # Each complete batch is parsed only once.
    for symbol, item in result["by_symbol"].items():
        if (item["source_id"] != identity({key: value for key, value in item.items() if key != "source_id"})
                or item["symbol"] != symbol or item.get("market") != "TWSE" or item.get("price_unit") != "TWD"
                or item["coverage"]["unknown_calendar_months"] != unknown or len(item["rows"]) != len(item["records"])):
            raise ValueError("VALUATION_HISTORY_SYMBOL_BINDING_INVALID")
        dates = [row["Date"] for row in item["rows"]]
        gap_months = [gap["month"] for gap in item["coverage"]["gaps"]]
        covered_months = [value[:7] for value in dates]
        if (dates != sorted(set(dates)) or len(dates) != item["coverage"]["months_valid"]
                or len(gap_months + covered_months) != len(set(gap_months + covered_months))
                or set(gap_months + covered_months) != set(ends)
                or any(gap.get("expected_session") != ends.get(gap["month"]) for gap in item["coverage"]["gaps"])
                or any(not isinstance(row.get("Close"), (int, float)) or isinstance(row["Close"], bool)
                       or not isfinite(row["Close"]) or row["Close"] <= 0 for row in item["rows"])):
            raise ValueError("VALUATION_HISTORY_QUOTE_ORDER_OR_VALUE_INVALID")
        for quote, record in zip(item["rows"], item["records"]):
            source = result["documents"][record["cache_source_id"]]
            if type(record.get("raw_row_index")) is not int or not 0 <= record["raw_row_index"] < len(decoded[record["cache_source_id"]]):
                raise ValueError("VALUATION_HISTORY_RAW_PRICE_BINDING_INVALID")
            raw = decoded[record["cache_source_id"]][record["raw_row_index"]]
            if (raw != record["raw_row"] or identity(raw) != record["raw_row_sha256"]
                    or raw[TwseColumns.STOCK_NO] != symbol or record.get("price_unit") != "TWD"
                    or quote != dict(Date=record["session_date"], Close=record["close"])
                    or ends.get(quote["Date"][:7]) != quote["Date"]
                    or source["trade_date"] != quote["Date"].replace("-", "")
                    or parse_price(raw[TwseColumns.CLOSE]) != quote["Close"]
                    or record["cached_at_epoch"] != source["cached_at_epoch"]
                    or source["cached_at_epoch"] > observed.timestamp()):
                raise ValueError("VALUATION_HISTORY_RAW_PRICE_BINDING_INVALID")
    return result


def retain_valuation_history(result, root):
    """Store each raw monthly cache payload once, then an immutable manifest."""
    from infrastructure.research_archive import write_once, write_bytes_once
    verify_valuation_history(result)
    root = Path(root)
    payload = dict(result)
    documents = {}
    for source_id, source in result["documents"].items():
        body = source["payload_json"].encode("utf-8")
        sha = source["payload_sha256"]
        relative = f"objects/{sha}.json"
        target = root / relative
        if target.exists():
            retained = target.read_bytes()
            if len(retained) != len(body) or hashlib.sha256(retained).hexdigest() != sha:
                raise ValueError("VALUATION_HISTORY_RETAINED_OBJECT_CORRUPT")
        else:
            write_bytes_once(target, body)
        documents[source_id] = dict({key: value for key, value in source.items() if key != "payload_json"},
            body_ref=dict(path=relative, sha256=sha, bytes=len(body)))
    payload["documents"] = documents
    manifest = dict(schema=MANIFEST_SCHEMA, evidence_id=result["evidence_id"], payload=payload)
    manifest["manifest_id"] = identity(manifest)
    path = root / "manifests" / (result["evidence_id"] + ".json")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("VALUATION_HISTORY_IMMUTABLE_MANIFEST_CONFLICT")
    else:
        write_once(path, manifest)
    return path


def load_valuation_history(manifest_path, root, *, _verified_files=None):
    """Restore and verify shared source bodies; no runtime-cache substitution."""
    root, path = Path(root).resolve(), Path(manifest_path).resolve()
    if not path.is_relative_to(root / "manifests") or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("VALUATION_HISTORY_MANIFEST_LOCATION_OR_SIZE_INVALID")
    signatures = {path: _file_signature(path)}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != MANIFEST_SCHEMA or path.stem != manifest.get("evidence_id")
            or manifest.get("manifest_id") != identity({key: value for key, value in manifest.items() if key != "manifest_id"})):
        raise ValueError("VALUATION_HISTORY_MANIFEST_BINDING_INVALID")
    result = dict(manifest["payload"])
    documents = {}
    for source_id, source in result["documents"].items():
        reference = source["body_ref"]; sha = reference["sha256"]
        if (not re.fullmatch(r"[0-9a-f]{64}", str(sha)) or reference.get("path") != f"objects/{sha}.json"
                or type(reference.get("bytes")) is not int or not 0 <= reference["bytes"] <= MAX_PAYLOAD_BYTES):
            raise ValueError("VALUATION_HISTORY_OBJECT_REFERENCE_INVALID")
        target = root / reference["path"]
        if not target.resolve().is_relative_to(root / "objects"):
            raise ValueError("VALUATION_HISTORY_OBJECT_REFERENCE_INVALID")
        signatures[target] = _file_signature(target)
        body = target.read_bytes()
        if len(body) != reference["bytes"] or hashlib.sha256(body).hexdigest() != sha:
            raise ValueError("VALUATION_HISTORY_RETAINED_OBJECT_CORRUPT")
        documents[source_id] = dict({key: value for key, value in source.items() if key != "body_ref"}, payload_json=body.decode("utf-8"))
    result["documents"] = documents
    if result.get("evidence_id") != manifest["evidence_id"]:
        raise ValueError("VALUATION_HISTORY_MANIFEST_BINDING_INVALID")
    verified = verify_valuation_history(result)
    if any(_file_signature(value) != signature for value, signature in signatures.items()):
        raise ValueError("VALUATION_SELECTION_RETAINED_REFERENCE_MISSING_OR_CHANGED")
    if _verified_files is not None:
        _verified_files.update(signatures)
    return verified


def _compact_selection(capture, symbol, manifest):
    item = capture["by_symbol"][symbol]
    result = dict(schema=SELECTION_SCHEMA, symbol=symbol, market="TWSE",
        observed_at=capture["observed_at"], capture_evidence_id=capture["evidence_id"],
        source_id=item["source_id"], rows=item["rows"], status=item["status"],
        coverage=item["coverage"], price_unit="TWD", availability_basis=capture["availability_basis"],
        source_authenticity=capture["source_authenticity"], historical_information_set_certified=False,
        source_record_ids=sorted({record["cache_source_id"] for record in item["records"]}),
        calendar_id=capture["calendar_evidence"]["calendar_id"], manifest=str(manifest))
    result["selection_id"] = identity(result)
    # Do not expose the retained capture's mutable rows/coverage to consumers.
    return json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))


def project_valuation_selections(capture, *, manifest):
    """Validate once and return detached compact projections for all symbols."""
    verify_valuation_history(capture)
    return {symbol: _compact_selection(capture, symbol, manifest) for symbol in capture["symbols"]}


def _selection_header(selection, *, symbol=None, cutoff=None):
    try:
        if (selection.get("schema") != SELECTION_SCHEMA or selection.get("market") != "TWSE"
                or not re.fullmatch(r"[0-9A-Z]{4,12}", str(selection.get("symbol", "")))
                or symbol is not None and selection["symbol"] != symbol
                or selection.get("selection_id") != identity({key: value for key, value in selection.items() if key != "selection_id"})):
            raise ValueError("VALUATION_SELECTION_BINDING_INVALID")
        observed = timestamp(selection["observed_at"])
        if cutoff is not None and observed > timestamp(cutoff):
            raise ValueError("VALUATION_SELECTION_AFTER_CUTOFF")
        if not selection.get("capture_evidence_id"):
            diagnostic = selection.get("diagnostics")
            if (selection.get("status") != "UNKNOWN" or selection.get("rows") not in ([], ())
                    or selection.get("source_id") or selection.get("source_record_ids") or selection.get("manifest")
                    or selection.get("historical_information_set_certified") is not False
                    or not isinstance(diagnostic, (list, tuple)) or not diagnostic
                    or any(not isinstance(reason, str) or not reason.strip() for reason in diagnostic)
                    or selection.get("coverage", {}).get("months_valid", 0) != 0):
                raise ValueError("VALUATION_SELECTION_UNBOUND_FAILURE_INVALID")
            return False
        if (not re.fullmatch(r"[0-9a-f]{64}", str(selection["capture_evidence_id"]))
                or not isinstance(selection.get("manifest"), str) or not selection["manifest"]):
            raise ValueError("VALUATION_SELECTION_RETAINED_REFERENCE_INVALID")
        return True
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("VALUATION_SELECTION_BINDING_INVALID") from error


def _file_signature(path):
    try:
        value = path.stat()
    except OSError as error:
        raise ValueError("VALUATION_SELECTION_RETAINED_REFERENCE_MISSING_OR_CHANGED") from error
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class ValuationSelectionVerifier:
    """Bounded job-local retained-source verifier; no domain filesystem calls.

    Full immutable captures are verified once, then discarded after generating
    canonical compact projections. Cache hits still stat the manifest and all
    referenced bodies, so later deletion/modification is not silently ignored.
    Instantiate a fresh verifier for another job/process; close drops all data.
    """
    def __init__(self, *, max_captures=2, max_bytes=16 * 1024 * 1024, manifest_loader=None,
                 manifest_path_resolver=None):
        if type(max_captures) is not int or max_captures < 0 or type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("VALUATION_SELECTION_CACHE_LIMIT_INVALID")
        self.max_captures, self.max_bytes = max_captures, max_bytes
        self.manifest_loader = manifest_loader
        # A portable archive may retain the same hash-bound logical manifest
        # name at a different physical location. Only I/O is resolved here;
        # the original text remains part of every selection identity.
        self.manifest_path_resolver = manifest_path_resolver
        self._entries = OrderedDict()
        self._bytes = 0
        self._closed = False
        self._counts = dict(loads=0, hits=0, evictions=0)

    def _load(self, path, capture_id, manifest_text):
        # Existing load_valuation_history() includes complete verification.
        verified_files = {}
        try:
            capture = (load_valuation_history(path, path.parent.parent, _verified_files=verified_files) if self.manifest_loader is None
                       else verify_valuation_history(self.manifest_loader(path)))
        except (OSError, KeyError, IndexError, TypeError) as error:
            raise ValueError("VALUATION_SELECTION_RETAINED_REFERENCE_INVALID") from error
        self._counts["loads"] += 1
        if capture["evidence_id"] != capture_id:
            raise ValueError("VALUATION_SELECTION_CAPTURE_ID_MISMATCH")
        projections = {}
        for symbol in capture["symbols"]:
            value = _compact_selection(capture, symbol, manifest_text)
            projections[symbol] = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        paths = {path, *(path.parent.parent / "objects" / (source["payload_sha256"] + ".json")
                         for source in capture["documents"].values())}
        signatures = verified_files or {value: _file_signature(value) for value in paths}
        weight = sum(len(value.encode("utf-8")) + len(key) + 256 for key, value in projections.items())
        weight += sum(len(str(value).encode("utf-8")) + 128 for value in signatures)
        return projections, signatures, weight

    def verify(self, selection, *, symbol=None, cutoff=None):
        if self._closed:
            raise ValueError("VALUATION_SELECTION_VERIFIER_CLOSED")
        if not _selection_header(selection, symbol=symbol, cutoff=cutoff):
            return json.loads(json.dumps(selection, ensure_ascii=False, allow_nan=False))
        path = Path(selection["manifest"]).resolve()
        if self.manifest_path_resolver is not None:
            path = Path(self.manifest_path_resolver(selection["manifest"])).resolve()
        token = selection["capture_evidence_id"], str(path), selection["manifest"]
        if token in self._entries:
            projections, signatures, weight = self._entries[token]
            if any(_file_signature(value) != signature for value, signature in signatures.items()):
                raise ValueError("VALUATION_SELECTION_RETAINED_REFERENCE_MISSING_OR_CHANGED")
            self._entries.move_to_end(token)
            self._counts["hits"] += 1
        else:
            projections, signatures, weight = self._load(path, token[0], selection["manifest"])
            if self.max_captures and weight <= self.max_bytes:
                while self._entries and (len(self._entries) >= self.max_captures or self._bytes + weight > self.max_bytes):
                    _, (_, _, removed) = self._entries.popitem(last=False)
                    self._bytes -= removed
                    self._counts["evictions"] += 1
                self._entries[token] = (projections, signatures, weight)
                self._bytes += weight
        expected = projections.get(selection["symbol"])
        actual = json.dumps(selection, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if expected is None or actual != expected:
            raise ValueError("VALUATION_SELECTION_RETAINED_PROJECTION_MISMATCH")
        return json.loads(expected)

    def clear(self):
        self._entries.clear()
        self._bytes = 0

    def close(self):
        self.clear()
        self._closed = True

    @property
    def stats(self):
        return dict(self._counts, entries=len(self._entries), bytes=self._bytes,
                    max_captures=self.max_captures, max_bytes=self.max_bytes)

    def __enter__(self):
        if self._closed:
            raise ValueError("VALUATION_SELECTION_VERIFIER_CLOSED")
        return self

    def __exit__(self, *exc):
        self.close()


def verify_valuation_selection(selection, *, manifest_loader=None, symbol=None, cutoff=None):
    """Single-shot public port; batch/work callers should reuse the verifier."""
    with ValuationSelectionVerifier(manifest_loader=manifest_loader) as verifier:
        return verifier.verify(selection, symbol=symbol, cutoff=cutoff)
