"""Deterministic calendar-day ranges for TWSE backtracking."""

from __future__ import annotations

import datetime


def build_backtrack_dates(
    reference_date: datetime.date,
    begin_backtrack: int,
    end_backtrack: int,
    *,
    include_reference: bool = False,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> list[datetime.date]:
    """Return oldest-to-newest dates inside the backtrack boundaries.

    `end_backtrack` is the older inclusive boundary. `begin_backtrack` is the
    newer exclusive boundary. The historical default excludes the reference
    date; callers may opt in so a newly published completed daily session can
    be discovered without changing existing consumers.

    An explicit start_date/end_date pair instead selects an inclusive date
    window. It is intersected with completed dates relative to reference_date;
    the legacy integer offset semantics are unchanged when the pair is absent.
    """

    if isinstance(reference_date, datetime.datetime):
        reference_date = reference_date.date()
    if not isinstance(reference_date, datetime.date):
        raise TypeError("reference_date must be a datetime.date")
    if start_date is not None or end_date is not None:
        if (type(start_date) is not datetime.date or type(end_date) is not datetime.date
                or start_date > end_date or end_date > reference_date):
            raise ValueError("Explicit date range requires start <= end <= reference_date")
        # Date bounds are inclusive, but an unfinished reference-day bar is
        # never selected. The caller still filters exchange holidays/weekends.
        complete_through = reference_date if include_reference else reference_date - datetime.timedelta(days=1)
        last = min(end_date, complete_through)
        return [start_date + datetime.timedelta(days=offset)
                for offset in range((last - start_date).days + 1)]
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
