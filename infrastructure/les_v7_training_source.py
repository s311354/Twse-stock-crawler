"""Bounded, offline normalization of immutable replay archives for v7 training.

The existing final-day cache has no historical observation receipts. The default
therefore excludes those bars. The optional close reconstruction is a declared
retrospective calculation convention, never a publication/observation proof.
No live data, financial parsing, model scoring or Bank account mutation occurs.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

from application.replay_contracts import aware_timestamp, digest
from application.research_evidence import RAW_EVIDENCE_SCHEMA, calendar_at, verify_publication
from domain.models import TwseColumns as C
from infrastructure.calendar.twse_calendar import TwseCalendar
from infrastructure.replay_evidence import load_evidence
from infrastructure.research_documents import EvidenceDocuments
from twse.parser import clean_cell

SCHEMA = "les-v7-training-source/v1"
VERSION = "LES_V7_REPLAY_TRAINING_SOURCE_R2"
OBSERVED_ONLY = "VERIFIED_OBSERVATION_ONLY"
FINAL_CLOSE = "FINAL_DAILY_CLOSE_RECONSTRUCTION_R1"
FINAL_VINTAGE = "PINNED_FINAL_DAILY_ARCHIVE_NOT_ORIGINAL_VINTAGE"
FINAL_REFERENCE = "OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_R1"
TAIPEI = ZoneInfo("Asia/Taipei")
MAX_BYTES = 128 * 1024 * 1024
MAX_SYMBOLS = 32
MAX_DAYS = 3660
TECHNICAL_COLUMNS = (C.OPEN, C.HIGH, C.LOW, C.CLOSE, C.VOLUME,
                     C.TRADE_VALUE, C.TRADE_COUNT, C.BID_PRICE, C.ASK_PRICE)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(str(value).replace(",", ""))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _stamp(path):
    value = path.stat()
    return value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _technical_values(row):
    # P/E, company name and bid/ask sizes do not feed this training adapter.
    # Preserve their conflicts, but do not erase equal verified OHLCV because
    # an unrelated column is blank in one official table and zero in another.
    return tuple(_number(row[i]) for i in TECHNICAL_COLUMNS)


def _calendar(payload, cutoff, documents, start, end, evidence_id):
    calendar = payload["calendar"]
    market = payload["market"]
    if calendar.get("source") != market + "_OFFICIAL":
        raise ValueError("TRAINING_CALENDAR_MARKET_SOURCE_MISMATCH")
    # Verify only documents used by this adapter, not unrelated financial bytes.
    integrity = documents.verify(calendar)
    verified = []

    def verifier(proof, *, binding):
        at = verify_publication(proof, binding=binding, document_reader=documents.read)
        verified.append(dict(binding=dict(binding), available_at=at, proof_sha256=digest(proof),
                             document_sha256=proof["document"]["sha256"]))
        return at

    days, rejected = calendar_at(calendar, cutoff, strict=True,
        annual_calendar=lambda data: TwseCalendar(fetcher=lambda: data), verifier=verifier)
    wanted = [(start + dt.timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
    scoped = {day: days.get(day, "UNKNOWN") for day in wanted}
    if any(value not in {"OPEN", "CLOSED", "UNKNOWN"} for value in scoped.values()):
        raise ValueError("INVALID_TRAINING_CALENDAR_STATUS")
    eligible = [row for row in verified if aware_timestamp(row["available_at"]) <= cutoff]
    available = max((aware_timestamp(row["available_at"]) for row in eligible), default=None)
    per_date = {}
    for day, status in scoped.items():
        if status == "UNKNOWN":
            per_date[day] = None
            continue
        exceptions = [row for row in eligible if row["binding"].get("kind") == "CALENDAR_EXCEPTION"
                      and row["binding"].get("date") == day]
        annual = [row for row in eligible if row["binding"].get("kind") == "ANNUAL_CALENDAR"
                  and row["binding"].get("year") == int(day[:4])]
        needed = exceptions or annual
        known_at = max((aware_timestamp(row["available_at"]) for row in needed), default=None)
        per_date[day] = known_at.isoformat() if known_at else None
    result = dict(source=market + "_OFFICIAL", sessions=sorted(day for day, status in scoped.items() if status == "OPEN"),
                  days=scoped, available_at=available.isoformat() if available else None,
                  evidence_ids=[evidence_id, digest(calendar)], publication_events=sorted(eligible, key=digest),
                  availability_by_date=per_date, scope_start=start.isoformat(), scope_end=end.isoformat())
    return result, dict(unknown_dates=[day for day, status in scoped.items() if status == "UNKNOWN"],
                       excluded=rejected, document_integrity=integrity)


def load_training_source(evidence_path, *, evaluation_cutoff, symbols=None, start=None, end=None,
                         availability_policy=OBSERVED_ONLY, max_bytes=MAX_BYTES,
                         reference_evidence_policy=None):
    """Return a deterministic hash-bound source packet, with unresolved gaps.

    Existing replay-evidence/v2 archived rows are TWD, shares and TWD amount,
    unlike the legacy strategy input's TWD*1000. No scaling or quote imputation
    is performed. A subset is explicit and is recorded alongside archive scope.

    FINAL_CLOSE is only for the documented final archive vintage. Its Available
    At is a reconstruction convention (close), not a changed retrieved_at. The
    output explicitly cannot certify historical observed information sets.
    Missing calendar proof, benchmark, quotes and source conflicts remain gaps.
    """
    if (type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES
            or availability_policy not in {OBSERVED_ONLY, FINAL_CLOSE}):
        raise ValueError("INVALID_TRAINING_SOURCE_LIMIT_OR_POLICY")
    if reference_evidence_policy not in {None, FINAL_REFERENCE}:
        raise ValueError("INVALID_REFERENCE_EVIDENCE_POLICY")
    if reference_evidence_policy and availability_policy != FINAL_CLOSE:
        raise ValueError("FINAL_REFERENCE_REQUIRES_EXPLICIT_FINAL_DAILY_RECONSTRUCTION")
    cutoff = aware_timestamp(evaluation_cutoff)
    path = Path(evidence_path)
    before = _stamp(path)
    if before[1] > max_bytes:
        raise ValueError("TRAINING_SOURCE_ARCHIVE_TOO_LARGE")
    evidence_id, source = load_evidence(path, max_bytes=max_bytes)
    if _stamp(path) != before:
        raise ValueError("TRAINING_SOURCE_ARCHIVE_CHANGED")
    if source.get("schema") != RAW_EVIDENCE_SCHEMA:
        raise ValueError("RAW_REPLAY_EVIDENCE_V2_REQUIRED")
    market = source.get("market")
    if market not in {"TWSE", "TPEX"}:
        raise ValueError("INVALID_TRAINING_SOURCE_MARKET")
    if (source.get("price_unit"), source.get("volume_unit"), source.get("amount_unit")) != ("TWD", "SHARES", "TWD"):
        raise ValueError("TRAINING_SOURCE_UNITS_UNVERIFIED")
    if availability_policy == FINAL_CLOSE and source.get("market_policy") != FINAL_VINTAGE:
        raise ValueError("FINAL_CLOSE_REQUIRES_DECLARED_FINAL_ARCHIVE_VINTAGE")
    original_symbols = source["symbols"]
    if (not isinstance(original_symbols, list) or not original_symbols
            or any(not isinstance(s, str) or not s or s != s.strip() for s in original_symbols)
            or len(set(original_symbols)) != len(original_symbols)):
        raise ValueError("INVALID_TRAINING_ARCHIVE_UNIVERSE")
    supplied = list(original_symbols if symbols is None else symbols)
    if any(not isinstance(s, str) or not s or s != s.strip() for s in supplied):
        raise ValueError("TRAINING_SOURCE_EXPLICIT_BOUNDED_UNIVERSE_REQUIRED")
    selected = sorted(set(supplied))
    if (not selected or len(selected) > MAX_SYMBOLS or not set(selected).issubset(original_symbols)
            or any(not isinstance(s, str) or not s or s != s.strip() for s in selected)):
        raise ValueError("TRAINING_SOURCE_EXPLICIT_BOUNDED_UNIVERSE_REQUIRED")
    first = dt.date.fromisoformat(start or source["start"])
    last = dt.date.fromisoformat(end or source["end"])
    if (first < dt.date.fromisoformat(source["start"]) or last > dt.date.fromisoformat(source["end"])
            or first > last or (last - first).days + 1 > MAX_DAYS):
        raise ValueError("TRAINING_SOURCE_RANGE_OUTSIDE_ARCHIVE_OR_LIMIT")
    documents = EvidenceDocuments.for_evidence(path, evidence_id)
    if reference_evidence_policy == FINAL_REFERENCE:
        from infrastructure.les_v7_calendar_final import select_final_calendar
        documents.verify(dict(calendar=source["calendar"], actions=source.get("corporate_actions_final"),
            market=source.get("market_supplement_final")))
        calendar, calendar_audit = select_final_calendar(source["calendar"], cutoff,
            documents=documents, start=first, end=last, evidence_id=evidence_id)
    else:
        calendar, calendar_audit = _calendar(source, cutoff, documents, first, last, evidence_id)
    invalid, duplicates, raw_rows, benchmark_rows = [], 0, {}, {}
    source_variants, ancillary_conflicts = {}, []
    conflicts, benchmark_conflicts = set(), set()
    source_count = 0
    for item in source["bars"]:
        try:
            day = dt.datetime.strptime(item["date"], "%Y%m%d").date()
            if day.strftime("%Y%m%d") != item["date"]:
                raise ValueError("NON_CANONICAL_SOURCE_DATE")
        except (KeyError, TypeError, ValueError):
            invalid.append(dict(reason="INVALID_SOURCE_SESSION_DATE"))
            continue
        if not first <= day <= last:
            continue
        date = day.isoformat()
        if item.get("dataset") == "TAIEX":
            if item.get("source") != "TWSE_TAIEX":
                invalid.append(dict(date=date, reason="BENCHMARK_SOURCE_MARKET_MISMATCH"))
                continue
            if date in benchmark_rows and benchmark_rows[date][0] != item["data"]:
                benchmark_conflicts.add(date)
            benchmark_rows[date] = (item["data"], digest(item))
            continue
        if item.get("dataset") != "STOCK_TABLE":
            continue
        if item.get("source") != market:
            invalid.append(dict(date=date, reason="BAR_SOURCE_MARKET_MISMATCH"))
            continue
        if not isinstance(item.get("data"), list):
            invalid.append(dict(date=date, reason="SOURCE_ROWS_SCHEMA_CHANGED"))
            continue
        header = {key: item.get(key) for key in ("date", "dataset", "market_type", "source")}
        for row in item["data"]:
            if not isinstance(row, list) or len(row) < C.REQUIRED_WIDTH:
                invalid.append(dict(date=date, reason="SOURCE_ROW_SCHEMA_CHANGED"))
                continue
            symbol = str(row[C.STOCK_NO])
            if symbol not in selected:
                continue
            source_count += 1
            key = symbol, date
            row_id = digest(dict(header=header, row=row))
            source_variants.setdefault(key, {})[row_id] = dict(
                source_row_sha256=digest(row), source_header=header)
            if key in raw_rows:
                previous_row = raw_rows[key][0]
                if _technical_values(previous_row) != _technical_values(row):
                    conflicts.add(key)
                else:
                    duplicates += 1
                    raw_rows[key][1].add(row_id)
                    if previous_row != row:
                        ancillary_conflicts.append(dict(symbol=symbol, date=date,
                            reason="NON_TECHNICAL_SOURCE_FIELDS_CONFLICT",
                            differing_columns=[i for i, (left, right) in enumerate(zip(previous_row, row)) if left != right],
                            source_row_sha256s=sorted({digest(previous_row), digest(row)}),
                            resolution="NO_FINANCIAL_VALUE_SELECTED_TECHNICAL_INPUTS_EQUAL"))
                    continue
            raw_rows[key] = (row, {row_id}, header)
    for symbol, date in sorted(conflicts):
        raw_rows.pop((symbol, date), None)
        invalid.append(dict(symbol=symbol, date=date, reason="CONFLICTING_MARKET_ROWS"))
    for date in sorted(benchmark_conflicts):
        benchmark_rows.pop(date, None)
        invalid.append(dict(date=date, reason="CONFLICTING_BENCHMARK_ROWS"))
    histories = {symbol: dict(symbol=symbol, market=market, bars=[], bar_lineage=[]) for symbol in selected}
    for (symbol, date), (row, row_ids, header) in sorted(raw_rows.items()):
        o, h, l, c, v = (_number(row[i]) for i in (C.OPEN, C.HIGH, C.LOW, C.CLOSE, C.VOLUME))
        if any(value is None for value in (o, h, l, c, v)) or not (0 < l <= min(o, c) <= max(o, c) <= h) or v < 0:
            invalid.append(dict(symbol=symbol, date=date, reason="OHLCV_UNAVAILABLE"))
            continue
        close_at = dt.datetime.combine(dt.date.fromisoformat(date), dt.time(13, 30), TAIPEI)
        if close_at > cutoff:
            invalid.append(dict(symbol=symbol, date=date, reason="BAR_AFTER_EVALUATION_CUTOFF"))
            continue
        # No availability field/receipt exists in collect_evidence()'s bars.
        # Never infer observation availability from an unrelated filing receipt.
        if availability_policy == OBSERVED_ONLY:
            invalid.append(dict(symbol=symbol, date=date, reason="BAR_AVAILABILITY_UNVERIFIED"))
            continue
        if calendar["days"].get(date) == "CLOSED":
            invalid.append(dict(symbol=symbol, date=date, reason="BAR_ON_VERIFIED_CLOSED_SESSION"))
            continue
        benchmark = benchmark_rows.get(date)
        taiex = _number(benchmark[0]) if benchmark else None
        if taiex is None or taiex <= 0:
            taiex = None
            invalid.append(dict(symbol=symbol, date=date, reason="BENCHMARK_UNAVAILABLE"))
        values = dict(Date=date, Open=o, High=h, Low=l, Close=c, Volume=v,
                      Amount=_number(row[C.TRADE_VALUE]), Trades=_number(row[C.TRADE_COUNT]),
                      Bid=_number(row[C.BID_PRICE]), Ask=_number(row[C.ASK_PRICE]), **{"TAIEX Close": taiex})
        if (values["Bid"] is not None and values["Ask"] is not None
                and values["Bid"] > values["Ask"]):
            invalid.append(dict(symbol=symbol, date=date, reason="CROSSED_SOURCE_QUOTES"))
        for field in ("Amount", "Trades"):
            if values[field] is None or values[field] < 0:
                values[field] = None
                invalid.append(dict(symbol=symbol, date=date, reason=field.upper() + "_UNAVAILABLE"))
        for field in ("Bid", "Ask"):
            if values[field] is None or values[field] <= 0:
                values[field] = None
                invalid.append(dict(symbol=symbol, date=date, reason=field.upper() + "_UNAVAILABLE"))
        values.update({"Available At": close_at.isoformat(), "Complete": True,
                       "Availability Policy": FINAL_CLOSE, "Vintage": FINAL_VINTAGE})
        histories[symbol]["bars"].append(values)
        histories[symbol]["bar_lineage"].append(dict(date=date, source_row_ids=sorted(row_ids),
            source_row_sha256=digest(row), source_header=header,
            source_variants=[source_variants[(symbol, date)][i] for i in sorted(source_variants[(symbol, date)])],
            benchmark_sha256=benchmark[1] if benchmark else None,
            original_observed_at=None, original_available_at=None,
            availability_basis="RETROSPECTIVE_FINAL_DAILY_CLOSE_CONVENTION_NOT_OBSERVATION_PROOF",
            calendar_status=calendar["days"].get(date, "UNKNOWN")))
    output_histories = []
    for symbol, history in histories.items():
        classification = source.get("security_classifications", {}).get(symbol)
        if classification is not None:
            history["security_type"] = classification["security_type"]
        history["coordinate_id"] = digest(dict(symbol=symbol, market=market, source_evidence_id=evidence_id,
            price_unit="TWD", volume_unit="SHARES", amount_unit="TWD", vintage=source.get("market_policy"),
            availability_policy=availability_policy, rows=history["bar_lineage"]))
        output_histories.append(history)
    documents.check()
    if _stamp(path) != before:
        raise ValueError("TRAINING_SOURCE_ARCHIVE_CHANGED")
    missing = {symbol: [day for day in calendar["sessions"] if (symbol, day) not in raw_rows]
               for symbol in selected}
    normalized_missing = {symbol: sorted(set(calendar["sessions"]) - {row["Date"] for row in history["bars"]})
                          for symbol, history in histories.items()}
    payload = dict(schema=SCHEMA, adapter_version=VERSION,
        dataset_kind="SYNTHETIC_TEST_ONLY" if source.get("purpose") == "SYNTHETIC_TEST_ONLY" else "RESEARCH",
        price_unit="TWD", volume_unit="SHARES", amount_unit="TWD",
        universe=[[s, market] for s in selected], universe_policy=source.get("universe_policy", "EXPLICIT_FIXED_ARCHIVE_UNIVERSE"),
        source_evidence_ids=[evidence_id], histories=output_histories, calendars={market: calendar},
        availability_policy=availability_policy, historical_information_set_certified=False,
        lineage=dict(archive_schema=source["schema"], archive_market_policy=source.get("market_policy"),
            archive_universe=list(original_symbols), archive_universe_sha256=digest(original_symbols),
            explicit_subset=selected != sorted(original_symbols), range=[first.isoformat(), last.isoformat()],
            evaluation_cutoff=cutoff.isoformat(), source_archive_observation="BAR_RECEIPTS_NOT_RETAINED",
            calendar_document_receipts={year: {key: doc.get(key) for key in ("retrieved_at", "source_url", "url")}
                for year, doc in source["calendar"].get("documents", {}).items()},
            official_transport_bytes_retained=False, source_authenticity="NOT_INDEPENDENTLY_CERTIFIED"),
        diagnostics=dict(source_row_count=source_count, duplicate_rows=duplicates,
            non_technical_conflicts=ancillary_conflicts,
            normalized_bar_count=sum(len(h["bars"]) for h in output_histories),
            calendar=calendar_audit, missing_source_rows=missing, excluded=invalid,
            missing_normalized_bars=normalized_missing,
            reason_counts=dict(sorted(Counter(row["reason"] for row in invalid).items())),
            preparation_gaps=source.get("preparation_gaps", [])))
    if "scope_binding" in source:
        payload["lineage"]["scope_binding"] = source["scope_binding"]
    if reference_evidence_policy == FINAL_REFERENCE:
        from infrastructure.les_v7_actions_final import build_final_coordinate_coverages
        packet = source.get("corporate_actions_final")
        payload["reference_evidence_policy"] = FINAL_REFERENCE
        payload["adapter_version"] = "LES_V7_REPLAY_TRAINING_SOURCE_R3_FINAL_REFERENCE"
        payload["lineage"]["original_historical_reference_availability"] = None
        payload["lineage"]["final_reference_evidence_policy"] = FINAL_REFERENCE
        if packet is not None:
            markers = [dict(symbol=str(row[C.STOCK_NO]), date=item["date"])
                for item in source["bars"] if item.get("dataset") == "STOCK_TABLE"
                for row in item["data"] if re.sub(r"<[^>]*>", "", clean_cell(row[C.CHANGE_SIGN])) == "X"]
            payload["coordinate_coverages"] = build_final_coordinate_coverages(
                output_histories, packet, pending_markers=markers, document_reader=documents.read)
            payload["lineage"]["corporate_actions_packet_sha256"] = digest(packet)
        else:
            payload["coordinate_coverages"] = []
            payload["diagnostics"]["preparation_gaps"].append("FINAL_ACTIONS_SOURCE_NOT_COLLECTED")
        documents.check()
        if _stamp(path) != before:
            raise ValueError("TRAINING_SOURCE_ARCHIVE_CHANGED")
    return dict(payload=payload, sha256=digest(payload))
