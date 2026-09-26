"""Strict, timezone-aware historical decision boundaries (not a fill simulator)."""

from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo


TAIPEI = ZoneInfo("Asia/Taipei")
SESSION_CLOSE = dt.time(13, 30)


def replay_close_cutoff(value: str) -> dt.datetime:
    """Convert a replay end date to the internal Taipei session-close cutoff.

    This is not an evidence timestamp parser: original publication and
    observation times retain their full precision in the point-in-time gates.
    """
    try:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError("date only")
        return dt.datetime.combine(dt.date.fromisoformat(value), SESSION_CLOSE, TAIPEI)
    except ValueError as error:
        raise ValueError("Replay end date requires YYYY-MM-DD (Taipei 13:30 close); timestamps are not supported") from error


def completed_session(cutoff: dt.datetime, calendar: object) -> dt.date:
    """Find the latest exchange session whose closing bar is visible."""
    local = cutoff.astimezone(TAIPEI)
    day = local.date()
    if local.time().replace(tzinfo=None) < SESSION_CLOSE:
        day -= dt.timedelta(days=1)
    for _ in range(370):
        if calendar.is_trading_day(day):
            return day
        day -= dt.timedelta(days=1)
    raise ValueError("REPLAY_COMPLETED_SESSION_UNAVAILABLE")
