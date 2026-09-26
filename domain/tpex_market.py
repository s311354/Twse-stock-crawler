"""TPEx daily quotes in original units, independent of legacy row layout."""

from __future__ import annotations

from dataclasses import dataclass

TPEX_FIELDS = (
    "代號", "名稱", "收盤", "漲跌", "開盤", "最高", "最低", "成交股數",
    "成交金額(元)", "成交筆數", "最後買價", "最後買量(張數)", "最後賣價",
    "最後賣量(張數)", "發行股數", "次日漲停價", "次日跌停價",
)
TPEX_RAW_OFFSET = 16


@dataclass(frozen=True)
class TpexDailyQuote:
    trade_date: str
    stock_no: str
    stock_name: str
    close: float | None
    change: str
    open: float | None
    high: float | None
    low: float | None
    volume: int
    amount: float
    trades: int
    bid: float | None
    bid_lots: float | None
    ask: float | None
    ask_lots: float | None
    issued_shares: int | None
    next_limit_up: float | None
    next_limit_down: float | None
    # Preserve all 17 official cells for audit/export, including change markers.
    raw_cells: tuple[str, ...]
