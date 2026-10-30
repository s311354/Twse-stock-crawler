"""R17 current official observations; never historical-publication backfill.

Strict publication packets retain the R16 verifier. Current packets instead
bind original annual JSON, the complete official year/tag closure query, and
each required reviewed closure document to their actual collection clocks.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
import json
import re
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from domain.evidence_publication import verified_document, verify_publication
from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.session_coverage_r16 import verify_window as verify_strict_window
from domain.trading_calendar import OfficialAnnualCalendar

SCHEMA = "les-v7-current-official-calendar-observation/v1"
POLICY = "OFFICIAL_ANNUAL_AND_COMPLETE_CLOSURE_QUERY_AT_ACTUAL_OBSERVATION_R17"
TAIPEI = ZoneInfo("Asia/Taipei")
TITLE = re.compile(r"^臺灣證券交易所集中交易市場(\d{3})年(\d{1,2})月(\d{1,2})日休市一天$")
MAX_BYTES = 16 * 1024 * 1024


def _json(row, *, path, params, observed, cutoff):
    document = row["document"]
    parsed = urlparse(document["url"])
    if (parsed.scheme != "https" or parsed.netloc != "www.twse.com.tw" or parsed.path != path
            or parse_qs(parsed.query) != {key: [str(value)] for key, value in params.items()}):
        raise ValueError("R17_OFFICIAL_QUERY_SCOPE_MISMATCH")
    body = verified_document(document)
    if len(body) > 2 * 1024 * 1024:
        raise ValueError("R17_CALENDAR_DOCUMENT_TOO_LARGE")
    payload = json.loads(body)
    retrieved = timestamp(document["retrieved_at"])
    if retrieved > observed or retrieved > cutoff:
        raise ValueError("R17_CALENDAR_DOCUMENT_AFTER_CUTOFF")
    if payload != row["payload"] or identity(payload) != row["payload_sha256"]:
        raise ValueError("R17_CALENDAR_ORIGINAL_BINDING_MISMATCH")
    return payload, document


def _annual(row, year, observed, cutoff):
    payload, document = _json(row, path="/holidaySchedule/holidaySchedule",
        params=dict(response="json", date=f"{year}0101"), observed=observed, cutoff=cutoff)
    if (payload.get("stat") != "ok" or payload.get("queryYear") != year
            or payload.get("date") != f"{year}0101" or payload.get("title") != f"{year-1911} 年市場開休市日期"
            or payload.get("fields") != ["日期", "名稱", "說明"]
            or payload.get("total") != len(payload.get("data", [])) or not payload.get("data")):
        raise ValueError("R17_OFFICIAL_ANNUAL_SCOPE_INCOMPLETE")
    if any(not isinstance(row, list) or len(row) != 3 or date.fromisoformat(row[0]).year != year
           for row in payload["data"]):
        raise ValueError("R17_OFFICIAL_ANNUAL_YEAR_MISMATCH")
    adapter = OfficialAnnualCalendar(payload)
    if adapter.trading_day_status(date(year, 1, 1)).value == "unknown":
        raise ValueError("R17_OFFICIAL_ANNUAL_SCHEMA_UNVERIFIED")
    return adapter, document["sha256"]


def _query(row, year, observed, cutoff, last):
    payload, document = _json(row, path="/rwd/zh/news/newsList",
        params=dict(startDate=year, endDate=year, tag="休市", response="json"), observed=observed, cutoff=cutoff)
    if (payload.get("stat") != "ok" or payload.get("fields") != ["項次", "標題", "日期", "zhId", "enId"]
            or payload.get("totalCount") != len(payload.get("data", []))):
        raise ValueError("R17_OFFICIAL_CLOSURE_QUERY_INCOMPLETE")
    # An earlier empty list cannot establish completeness for a later session.
    if timestamp(document["retrieved_at"]) < datetime.combine(last, time(13, 30), TAIPEI):
        raise ValueError("R17_CLOSURE_QUERY_PRECEDES_WINDOW_CLOSE")
    found = {}
    for item in payload["data"]:
        if not isinstance(item, list) or len(item) != 5:
            raise ValueError("R17_OFFICIAL_CLOSURE_QUERY_ROW_INVALID")
        match = TITLE.fullmatch(item[1])
        if match is None:
            if "集中交易市場" in item[1] and any(word in item[1] for word in ("休市一天", "全日休市", "停止交易")):
                raise ValueError("R17_UNRECOGNIZED_MARKET_CLOSURE_NEEDS_REVIEW")
            continue  # Other news mentioning closure is not a market closure.
        day = date(int(match[1]) + 1911, int(match[2]), int(match[3])).isoformat()
        if int(day[:4]) != year or not re.fullmatch(r"[0-9a-f]{32}", item[3]) or item[3] in found:
            raise ValueError("R17_OFFICIAL_CLOSURE_QUERY_CONFLICT")
        found[item[3]] = day
    return found, document["sha256"]


def _closure(event, observed, cutoff):
    day = date.fromisoformat(event["date"])
    if event.get("market") != "TWSE" or event.get("status") != "CLOSED":
        raise ValueError("R17_CLOSURE_MARKET_OR_STATUS_MISMATCH")
    detail, document = _json(event["detail"], path="/rwd/zh/news/newsDetail",
        params=dict(id=event["news_id"], response="json"), observed=observed, cutoff=cutoff)
    if detail.get("stat") != "ok" or not re.fullmatch(r"\d{12}", detail.get("date", "")):
        raise ValueError("R17_CLOSURE_DETAIL_INVALID")
    released = datetime.strptime(detail["date"], "%Y%m%d%H%M").replace(tzinfo=TAIPEI)
    title = f"臺灣證券交易所集中交易市場{day.year-1911}年{day.month}月{day.day}日休市一天"
    table = detail["tables"][0]
    pdf_path = table["data"][2]
    if (table.get("title") != title or table.get("fields") != ["text", "html", "pdf", "lang"]
            or not re.fullmatch(r"/news/news/tsecnews/[0-9a-f]{32}\.pdf", pdf_path)):
        raise ValueError("R17_CLOSURE_DETAIL_SOURCE_MISMATCH")
    proof = event["proof"]
    if (proof["document"]["url"] != "https://www.twse.com.tw/staticFiles" + pdf_path
            or timestamp(proof["published_at"]) != released or released.date() > day
            or timestamp(document["retrieved_at"]) < released
            or timestamp(proof["verified_at"]) > observed
            or timestamp(proof["document"]["retrieved_at"]) > observed):
        raise ValueError("R17_CLOSURE_ORIGINAL_TIME_BINDING_MISMATCH")
    available = timestamp(verify_publication(proof,
        binding=dict(kind="CALENDAR_EXCEPTION", market="TWSE", date=day.isoformat(), status="CLOSED")))
    if available > cutoff:
        raise ValueError("R17_CLOSURE_PUBLICATION_AFTER_CUTOFF")
    body = verified_document(proof["document"])
    compact = re.sub(r"\s+", "", proof["document"]["extracted_text"])
    if (not body.startswith(b"%PDF-") or title not in compact
            or f"中華民國{released.year-1911}年{released.month}月{released.day}日" not in compact
            or f"{day.year-1911}年{day.month}月{day.day}日集中交易市場全日休市" not in compact):
        raise ValueError("R17_CLOSURE_TEXT_EFFECTIVE_DATE_MISMATCH")
    return [document["sha256"], proof["document"]["sha256"], identity(proof)]


def verify_window(calendar_evidence, *, market, cutoff, observed_sessions, prior_sessions=120):
    if not isinstance(calendar_evidence, Mapping) or calendar_evidence.get("schema") != SCHEMA:
        return verify_strict_window(calendar_evidence, market=market, cutoff=cutoff,
            observed_sessions=observed_sessions, prior_sessions=prior_sessions)
    audit = dict(schema="les-v7-r17-official-session-window/v1", policy=POLICY, market=market,
        cutoff=None, state="UNKNOWN", prior_sessions=prior_sessions, expected_sessions=[], observed_sessions=[],
        missing=[], extra=[], unknown_dates=[], calendar_days={}, source_id=None, source_proof_ids=[],
        excluded_evidence=[], reasons=[], original_historical_available_at=None,
        historical_information_set_certified=False, availability_basis="ACTUAL_CURRENT_OBSERVATION_NOT_BACKFILL")

    def finish(state, *reasons):
        audit.update(state=state, reasons=list(reasons))
        return audit

    try:
        at = timestamp(cutoff)
        audit["cutoff"] = at.isoformat()
        if market != "TWSE":
            return finish("UNKNOWN", "R17_MARKET_SPECIFIC_CALENDAR_CAPTURE_UNAVAILABLE")
        if (type(prior_sessions) is not int or not 1 <= prior_sessions <= 1024
                or not isinstance(observed_sessions, (list, tuple)) or len(observed_sessions) > 10000):
            return finish("UNKNOWN", "R17_SESSION_WINDOW_INPUT_INVALID")
        dates = [date.fromisoformat(day) for day in observed_sessions]
        if dates != sorted(set(dates)) or [d.isoformat() for d in dates] != list(observed_sessions):
            return finish("UNKNOWN", "R17_SESSION_DATES_NOT_SORTED_UNIQUE_ISO_DATES")
        if len(dates) < prior_sessions + 1:
            return finish("UNKNOWN", "R17_OBSERVED_PRIOR_SESSION_HISTORY_MISSING")
        dates = dates[-prior_sessions-1:]
        first, last = dates[0], dates[-1]
        audit.update(start=first.isoformat(), end=last.isoformat(), observed_sessions=[d.isoformat() for d in dates])
        if (last-first).days > 3660 or datetime.combine(last, time(13, 30), TAIPEI) > at:
            return finish("UNKNOWN", "R17_SESSION_WINDOW_FUTURE_OR_EXCESSIVE_SPAN")
        encoded = canonical(calendar_evidence)
        if len(encoded.encode()) > MAX_BYTES:
            return finish("UNKNOWN", "R17_CALENDAR_EVIDENCE_TOO_LARGE")
        calendar = json.loads(encoded)
        claimed_id = calendar.pop("evidence_id", None)
        if claimed_id != identity(calendar):
            return finish("UNKNOWN", "R17_CALENDAR_EVIDENCE_ID_MISMATCH")
        audit["source_id"] = claimed_id
        if (calendar.get("policy") != POLICY or calendar.get("market") != market
                or calendar.get("source") != "TWSE_OFFICIAL" or calendar.get("historical_information_set_certified") is not False
                or calendar.get("original_historical_available_at") is not None):
            return finish("UNKNOWN", "R17_CURRENT_OBSERVATION_POLICY_REQUIRED")
        observed = timestamp(calendar["observed_at"])
        audit["observed_at"] = observed.isoformat()
        if observed > at:
            return finish("UNKNOWN", "R17_CALENDAR_OBSERVATION_AFTER_CUTOFF")
        audit["selected_publications"] = [dict(binding=dict(kind="CURRENT_CALENDAR_OBSERVATION", market=market),
            proof_id=claimed_id, available_at=observed.isoformat(),
            availability_basis="ACTUAL_OBSERVATION_NOT_ORIGINAL_PUBLICATION")]
        adapters, closures, proofs, excluded = {}, {}, [], []
        for year in range(first.year, last.year+1):
            try:
                adapter, annual_id = _annual(calendar["documents"][str(year)], year, observed, at)
                query, query_id = _query(calendar["closure_queries"][str(year)], year, observed, at,
                    min(last, date(year, 12, 31)))
                adapters[year], closures[year] = adapter, query
                proofs.extend([annual_id, query_id])
            except (KeyError, ValueError, TypeError, IndexError) as error:
                excluded.append(dict(year=year, reason="R17_ANNUAL_OR_CLOSURE_COVERAGE_UNVERIFIED", detail=str(error)))
        days = {(first+timedelta(days=i)).isoformat(): (adapters[(first+timedelta(days=i)).year]
                .trading_day_status(first+timedelta(days=i)).value.upper()
                if (first+timedelta(days=i)).year in adapters else "UNKNOWN") for i in range((last-first).days+1)}
        events = calendar.get("announcements", [])
        for year, identified in closures.items():
            for news_id, day in identified.items():
                if day not in days:
                    continue
                try:
                    matches = [event for event in events if event.get("news_id") == news_id]
                    if len(matches) != 1 or matches[0].get("date") != day:
                        raise ValueError("R17_REQUIRED_CLOSURE_ORIGINAL_MISSING_OR_CONFLICTING")
                    proofs.extend(_closure(matches[0], observed, at))
                    days[day] = "CLOSED"
                except (KeyError, ValueError, TypeError, IndexError) as error:
                    days[day] = "UNKNOWN"
                    excluded.append(dict(date=day, reason="R17_CLOSURE_ORIGINAL_UNVERIFIED", detail=str(error)))
        for event in events:
            day = event.get("date")
            if day in days and closures.get(int(day[:4]), {}).get(event.get("news_id")) != day:
                days[day] = "UNKNOWN"
                excluded.append(dict(date=day, reason="R17_CLOSURE_NOT_BOUND_TO_COMPLETE_QUERY"))
        observations = audit["observed_sessions"]
        expected = [day for day, state in days.items() if state == "OPEN"]
        missing = [day for day in expected if day not in observations]
        extra = [day for day in observations if days[day] == "CLOSED"]
        unknown = [day for day, state in days.items() if state == "UNKNOWN"]
        audit.update(calendar_days=days, expected_sessions=expected, missing=missing, extra=extra,
            unknown_dates=unknown, source_proof_ids=sorted(set(proofs)), excluded_evidence=excluded,
            expected_open_count=len(expected), observed_count=len(observations),
            closure_query_scope="COMPLETE_OFFICIAL_YEAR_TAG_CLOSURE_QUERY_NOT_ALL_EXCHANGE_NEWS")
        if missing or extra:
            return finish("FAIL", *(["R17_OFFICIAL_OPEN_SESSION_BAR_MISSING"] if missing else []),
                          *(["R17_BAR_ON_OFFICIAL_CLOSED_DAY"] if extra else []))
        if unknown or not proofs:
            return finish("UNKNOWN", "R17_CALENDAR_DATES_UNKNOWN")
        if expected != observations or len(expected) != prior_sessions + 1:
            return finish("FAIL", "R17_OFFICIAL_SESSION_SEQUENCE_MISMATCH")
        return finish("PASS")
    except (KeyError, ValueError, TypeError, OverflowError, AttributeError, IndexError, RecursionError) as error:
        audit["validation_error"] = str(error)
        return finish("UNKNOWN", "R17_CURRENT_CALENDAR_EVIDENCE_INVALID")
