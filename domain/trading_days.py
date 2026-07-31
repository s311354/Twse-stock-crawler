"""Deterministic calendar-day ranges for TWSE backtracking."""

from __future__ import annotations

import datetime


def build_backtrack_dates(
    reference_date: datetime.date,
    begin_backtrack: int,
    end_backtrack: int,
    *,
    include_reference: bool = False,
) -> list[datetime.date]:
    """Return oldest-to-newest dates inside the backtrack boundaries.

    `end_backtrack` is the older inclusive boundary. `begin_backtrack` is the
    newer exclusive boundary. The historical default excludes the reference
    date; callers may opt in so a newly published completed daily session can
    be discovered without changing existing consumers.
    """

    if isinstance(reference_date, datetime.datetime):
        reference_date = reference_date.date()
    if not isinstance(reference_date, datetime.date):
        raise TypeError("reference_date must be a datetime.date")
    if begin_backtrack < 0 or end_backtrack < 0:
        raise ValueError("backtrack boundaries cannot be negative")
    if end_backtrack <= begin_backtrack:
        raise ValueError("end_backtrack must be greater than begin_backtrack")

    dates = [
        reference_date - datetime.timedelta(days=offset)
        for offset in range(end_backtrack, begin_backtrack, -1)
    ]
    if include_reference and begin_backtrack == 0:
        dates.append(reference_date)
    return dates
