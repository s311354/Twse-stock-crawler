"""Low-coupling TWSE client boundary."""

from __future__ import annotations

from domain.models import StockRows
from twstockcrawler import TwStockCrawler


class TwseClient:
    """Fetch TWSE rows without exposing the legacy crawler implementation."""

    def __init__(self, crawler: TwStockCrawler | None = None) -> None:
        self._crawler = crawler or TwStockCrawler()

    def get_daily_stock_rows(self, date_time: str, stocktype: int) -> StockRows:
        return self._crawler.get_stocktype_data(date_time, stocktype)

