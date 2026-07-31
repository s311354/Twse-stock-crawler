"""Point-in-time market-breadth facts derived from completed TWSE daily rows."""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

from domain.models import TwseColumns


MARKET_BREADTH_ATTR = "les_market_breadth"
MARKET_DATA_THROUGH_ATTR = "les_market_data_through"
MARKET_SESSION_FINAL_ATTR = "les_market_session_final"

_COMMON_STOCK_PATTERN = re.compile(r"^[1-9]\d{3}$")


@dataclass(frozen=True)
class MarketBreadthSnapshot:
    """Auditable cross-sectional facts for one completed TWSE session."""

    as_of: str
    universe: str
    eligible_count: int
    valid_return_count: int
    advancers: int
    decliners: int
    unchanged: int
    advancer_ratio: float | None
    median_return: float | None
    coverage: float
    source_final: bool
    taiex_return: float | None = None

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def derive_twse_market_breadth(
    rows: Iterable[Sequence[object]],
    *,
    as_of: str,
    universe: str,
    source_final: bool,
    taiex_return: float | None = None,
) -> MarketBreadthSnapshot:
    """Derive breadth from the full official industry table.

    Only four-digit common-stock identifiers are included. Duplicate symbols
    are counted once. Rows whose prior close cannot be reconstructed from the
    official close/change fields remain in the eligible universe but reduce
    coverage rather than being treated as unchanged.
    """

    seen: set[str] = set()
    eligible_count = 0
    returns: list[float] = []

    for row in rows:
        if not row:
            continue
        stock_no = str(row[TwseColumns.STOCK_NO]).replace(",", "").strip()
        if not _COMMON_STOCK_PATTERN.fullmatch(stock_no) or stock_no in seen:
            continue
        seen.add(stock_no)
        eligible_count += 1

        daily_return = _daily_return_from_twse_row(row)
        if daily_return is not None:
            returns.append(daily_return)

    advancers = sum(value > 0 for value in returns)
    decliners = sum(value < 0 for value in returns)
    unchanged = len(returns) - advancers - decliners
    valid_return_count = len(returns)
    coverage = (
        valid_return_count / eligible_count
        if eligible_count > 0
        else 0.0
    )
    advancer_ratio = (
        advancers / valid_return_count
        if valid_return_count > 0
        else None
    )
    median_return = statistics.median(returns) if returns else None

    return MarketBreadthSnapshot(
        as_of=str(as_of),
        universe=str(universe).strip().upper() or "UNKNOWN",
        eligible_count=eligible_count,
        valid_return_count=valid_return_count,
        advancers=advancers,
        decliners=decliners,
        unchanged=unchanged,
        advancer_ratio=advancer_ratio,
        median_return=median_return,
        coverage=coverage,
        source_final=bool(source_final),
        taiex_return=taiex_return,
    )


def normalize_market_breadth_snapshot(
    value: Mapping[str, Any] | MarketBreadthSnapshot | None,
) -> dict[str, object]:
    if isinstance(value, MarketBreadthSnapshot):
        return value.to_mapping()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _daily_return_from_twse_row(row: Sequence[object]) -> float | None:
    if len(row) < TwseColumns.REQUIRED_WIDTH:
        return None

    close = _parse_number(row[TwseColumns.CLOSE])
    change = _parse_number(row[TwseColumns.PRICE_CHANGE])
    if close is None or close <= 0 or change is None or change < 0:
        return None

    direction = _parse_change_direction(
        row[TwseColumns.CHANGE_SIGN],
        change=change,
    )
    if direction is None:
        return None

    signed_change = direction * change
    previous_close = close - signed_change
    if previous_close <= 0:
        return None
    return signed_change / previous_close


def _parse_change_direction(value: object, *, change: float) -> int | None:
    text = str(value).strip().lower()
    if "+" in text or "red" in text:
        return 1
    if "-" in text or "green" in text:
        return -1
    if change == 0:
        return 0
    return None


def _parse_number(value: object) -> float | None:
    text = str(value).replace(",", "").strip()
    if text in {"", "--"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


__all__ = [
    "MARKET_BREADTH_ATTR",
    "MARKET_DATA_THROUGH_ATTR",
    "MARKET_SESSION_FINAL_ATTR",
    "MarketBreadthSnapshot",
    "derive_twse_market_breadth",
    "normalize_market_breadth_snapshot",
]
