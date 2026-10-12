"""Bounded, fail-closed LES v6 execution bridge. Never submits orders."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from domain.strategies.low_entry_score_v6 import assess_execution
from domain.strategies.low_entry_v6_report import report_failure
from infrastructure.storage.report_evidence import load_report_bundle

MAX_BYTES = 8 * 1024 * 1024


def is_supported_execution_symbol(value: object) -> bool:
    return isinstance(value, str) and value.isdigit() and 4 <= len(value) <= 6


def execute_request(payload: dict) -> dict:
    try:
        failure = report_failure(payload.get("report"))
        if failure:
            return {"action": "SKIP", "shares": 0, "reason": failure}
        if not isinstance(payload.get("hard_risk"), bool):
            return {"action": "SKIP", "shares": 0, "reason": "HARD_RISK_UNKNOWN"}
        sessions = payload["trading_sessions"]
        if not isinstance(sessions, list) or not 2 <= len(sessions) <= 32:
            raise ValueError("INVALID_SESSION_CALENDAR")
        for day in sessions:
            if not isinstance(day, str) or dt.date.fromisoformat(day).isoformat() != day:
                raise ValueError("INVALID_SESSION_CALENDAR")
        return assess_execution(payload["report"], payload["quote"], now=payload["now"],
                                trading_sessions=sessions, portfolio=payload["portfolio"],
                                hard_risk=payload["hard_risk"])
    except (KeyError, TypeError, ValueError, AttributeError, ArithmeticError):
        return {"action": "SKIP", "shares": 0, "reason": "INVALID_EXECUTION_INPUT"}


def load_reports(path: str, data_date: str) -> dict[str, dict]:
    source = Path(path)
    if source.stat().st_size > MAX_BYTES:
        raise ValueError("V6_REPORTS_TOO_LARGE")
    payload = load_report_bundle(source)
    reports = payload.get("reports")
    if payload.get("model") != "low_entry_score_v6" or not isinstance(reports, list) or not 1 <= len(reports) <= 2000:
        raise ValueError("INVALID_V6_REPORTS")
    policy = payload.get("score_policy")
    if policy is not None and policy not in {"legacy", "unified", "hybrid", "hybrid-v2"}:
        raise ValueError("INVALID_V6_SCORE_POLICY")
    output = {}
    for report in reports:
        if policy is not None and report.get("parameters", {}).get("decision_policy_version", 2) != {
                "legacy": 1, "unified": 2, "hybrid": 3, "hybrid-v2": 4}[policy]:
            raise ValueError("V6_REPORT_SCORE_POLICY_MISMATCH")
        if report.get("evaluation_mode") == "HISTORICAL_REPLAY":
            raise ValueError("HISTORICAL_REPLAY_NOT_EXECUTABLE")
        code = report.get("symbol")
        if not is_supported_execution_symbol(code):
            if (report.get("decision", {}).get("signal") in {"BUY", "BUY_SMALL"}
                    or report.get("sizing", {}).get("indicative_shares") != 0):
                raise ValueError("V6_REPORT_IDENTITY_OR_DATE_MISMATCH")
            continue
        if (code in output or report.get("signal_date") != data_date
                or report.get("model_version") != "6.0.0"):
            raise ValueError("V6_REPORT_IDENTITY_OR_DATE_MISMATCH")
        output[code] = report
    return output


def next_sessions(signal_date: str, trade_date: str, calendar=None) -> list[str]:
    # The production schedule comes from the existing official TWSE calendar;
    # UNKNOWN never becomes a weekday-based assumed trading day.
    from infrastructure.calendar import TwseCalendar
    calendar = calendar or TwseCalendar()
    start = dt.date.fromisoformat(signal_date)
    if not calendar.is_trading_day(start):
        raise ValueError("INVALID_SIGNAL_SESSION")
    for days in range(1, 32):
        candidate = start + dt.timedelta(days=days)
        if calendar.is_trading_day(candidate):
            if candidate.isoformat() != trade_date:
                raise ValueError("V6_REPORT_EXPIRED_OR_WRONG_SESSION")
            return [signal_date, trade_date]
    raise ValueError("NO_VERIFIED_NEXT_SESSION")


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("INPUT_TOO_LARGE")
        result = execute_request(json.loads(raw))
    except (ValueError, TypeError, AttributeError):
        result = {"action": "SKIP", "shares": 0, "reason": "INVALID_EXECUTION_INPUT"}
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
