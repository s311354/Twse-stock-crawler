"""R16 source-bound official session coverage, without fetching or guessing.

A bar count is not a complete trading-session window. Reuse the strict replay
annual-calendar/publication contract and its raw annual adapter, then compare
every calendar day and the exact OPEN sequence. Content hashes identify the
retained source; they do not themselves authenticate its official publication.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
import json
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from domain.les_v7.contracts import canonical, identity, timestamp

SCHEMA = "les-v7-r16-official-session-window/v1"
POLICY = "STRICT_REPLAY_VERIFIED_ANNUAL_AND_EXCEPTIONS_EXACT_OPEN_SEQUENCE_R1"
MAX_PRIOR_SESSIONS = 1024
MAX_OBSERVED_SESSIONS = 10000
MAX_CALENDAR_SPAN_DAYS = 3660
MARKET_HOSTS = {
    "TWSE": frozenset({"www.twse.com.tw", "twse.com.tw", "wwwc.twse.com.tw", "doc.twse.com.tw",
                       "openapi.twse.com.tw"}),
    "TPEX": frozenset({"www.tpex.org.tw", "tpex.org.tw"}),
}


def verify_window(calendar_evidence, *, market, cutoff, observed_sessions, prior_sessions=120):
    """Verify a signal session plus its fixed prior official OPEN sessions.

    ``calendar_evidence`` is the existing replay v2 calendar object: ``source``,
    raw ``documents[year].payload`` plus exact ``publication_proof`` and reviewed
    ``announcements``. A provided ``days`` mapping is not an authority. Every
    date between the first and last selected observation is rederived.

    Later collection/review of a verifiable historical original retains the
    existing strict-replay policy. Publication bounds after ``cutoff`` are
    excluded; fetched_at/verified_at are neither rewritten nor misrepresented.
    No document reader or transport is supplied: unresolved external document
    references fail closed instead of causing I/O or using an unchecked flag.
    """
    audit = dict(schema=SCHEMA, policy=POLICY, market=market, cutoff=None,
        state="UNKNOWN", prior_sessions=prior_sessions, expected_sessions=[], observed_sessions=[],
        missing=[], extra=[], unknown_dates=[], calendar_days={}, source_id=None,
        source_proof_ids=[], excluded_evidence=[], reasons=[],
        source_authenticity="EXISTING_REVIEWED_PUBLICATION_CONTRACT_NOT_HASH_ALONE")

    def finish(state, *reasons):
        audit.update(state=state, reasons=list(dict.fromkeys(reasons)))
        return audit

    try:
        at = timestamp(cutoff)
        audit["cutoff"] = at.isoformat()
        if market not in MARKET_HOSTS:
            return finish("UNKNOWN", "R16_SESSION_MARKET_NOT_SUPPORTED")
        if (type(prior_sessions) is not int or not 1 <= prior_sessions <= MAX_PRIOR_SESSIONS
                or not isinstance(observed_sessions, (list, tuple))
                or len(observed_sessions) > MAX_OBSERVED_SESSIONS):
            return finish("UNKNOWN", "R16_SESSION_WINDOW_INPUT_INVALID")
        dates = [date.fromisoformat(value) if isinstance(value, str) else None for value in observed_sessions]
        if (any(value is None or value.isoformat() != original for value, original in zip(dates, observed_sessions))
                or dates != sorted(set(dates))):
            return finish("UNKNOWN", "R16_SESSION_DATES_NOT_SORTED_UNIQUE_ISO_DATES")
        if len(dates) < prior_sessions + 1:
            return finish("UNKNOWN", "R16_OBSERVED_PRIOR_SESSION_HISTORY_MISSING")
        dates = dates[-(prior_sessions + 1):]
        observations = [value.isoformat() for value in dates]
        audit["observed_sessions"] = observations
        first, last = dates[0], dates[-1]
        audit.update(start=first.isoformat(), end=last.isoformat())
        if (datetime.combine(last, time(13, 30), tzinfo=ZoneInfo("Asia/Taipei")) > at
                or (last - first).days > MAX_CALENDAR_SPAN_DAYS):
            return finish("UNKNOWN", "R16_SESSION_WINDOW_FUTURE_OR_EXCESSIVE_SPAN")
        if not isinstance(calendar_evidence, Mapping):
            return finish("UNKNOWN", "R16_OFFICIAL_SESSION_EVIDENCE_MISSING")
        # Frozen JSON mappings use tuples. Normalize the bounded calendar only
        # so the existing raw adapter sees its documented list/dict payload.
        calendar = json.loads(canonical(calendar_evidence))
        audit["source_id"] = identity(calendar)
        if (not isinstance(calendar.get("source"), str)
                or calendar["source"].split("_")[0] != market):
            return finish("UNKNOWN", "R16_CALENDAR_MARKET_SCOPE_UNVERIFIED")
        raw_days = calendar.get("days", {})
        if not isinstance(raw_days, dict):
            return finish("UNKNOWN", "R16_CALENDAR_DAY_MAPPING_INVALID")
        window_days = [(first + timedelta(days=offset)).isoformat()
                       for offset in range((last - first).days + 1)]
        calendar["days"] = {day: raw_days.get(day, "UNKNOWN") for day in window_days}
        # Validate publications with their existing exact binding, quotes,
        # document hash, official URL and conservative date-only policy.
        from domain.evidence_publication import calendar_at, verify_publication
        from domain.trading_calendar import OfficialAnnualCalendar
        accepted, future = [], []

        def verifier(proof, *, binding):
            document = proof.get("document", {})
            if urlparse(document.get("url", "")).hostname not in MARKET_HOSTS[market]:
                raise ValueError("R16_CALENDAR_DOCUMENT_MARKET_SCOPE_UNVERIFIED")
            available = verify_publication(proof, binding=binding)
            detail = dict(binding=binding, proof_id=identity(proof), available_at=available)
            if timestamp(available) <= at:
                accepted.append(detail)
            else:
                future.append(dict(detail, reason="R16_CALENDAR_PUBLICATION_AFTER_CUTOFF"))
            return available

        days, excluded = calendar_at(calendar, at, strict=True,
            annual_calendar=OfficialAnnualCalendar,
            verifier=verifier)
        days = {day: days.get(day, "UNKNOWN") for day in window_days}
        # A supplied but unverifiable exception is not evidence of closure,
        # nor permission to certify the old annual OPEN state as complete.
        # A *verified* later publication is different: it was not part of the
        # cutoff information set and must not rewrite that earlier calendar.
        for item in excluded:
            day = item.get("date")
            if day in days and item.get("reason") != "CALENDAR_ANNOUNCEMENT_AFTER_CUTOFF":
                days[day] = "UNKNOWN"
        if any(status not in {"OPEN", "CLOSED", "UNKNOWN"} for status in days.values()):
            return finish("UNKNOWN", "R16_OFFICIAL_CALENDAR_STATUS_INVALID")
        expected = [day for day in window_days if days[day] == "OPEN"]
        unknown = [day for day in window_days if days[day] == "UNKNOWN"]
        missing = [day for day in expected if day not in observations]
        extra = [day for day in observations if days[day] == "CLOSED"]
        audit.update(calendar_days=days, expected_sessions=expected, missing=missing, extra=extra,
            unknown_dates=unknown, source_proof_ids=sorted({entry["proof_id"] for entry in accepted}),
            selected_publications=accepted, excluded_evidence=[*excluded, *future],
            expected_open_count=len(expected), observed_count=len(observations))
        if missing or extra:
            return finish("FAIL", *(["R16_OFFICIAL_OPEN_SESSION_BAR_MISSING"] if missing else []),
                          *(["R16_BAR_ON_OFFICIAL_CLOSED_DAY"] if extra else []),
                          *(["R16_CALENDAR_DATES_UNKNOWN"] if unknown else []))
        if unknown or not accepted:
            return finish("UNKNOWN", "R16_CALENDAR_DATES_UNKNOWN")
        if expected != observations or len(expected) != prior_sessions + 1:
            return finish("FAIL", "R16_OFFICIAL_SESSION_SEQUENCE_MISMATCH")
        return finish("PASS")
    except (KeyError, ValueError, TypeError, OverflowError, AttributeError, RecursionError) as error:
        audit["validation_error"] = str(error)
        return finish("UNKNOWN", "R16_OFFICIAL_SESSION_EVIDENCE_INVALID")
