"""Low-coupling TWSE client boundary."""

from __future__ import annotations

from domain.models import StockRows
from infrastructure.crawler.twse_http_client import TwseHttpClient
from twstockcrawler import TwStockCrawler


class TwseClient:
    """Fetch TWSE rows without exposing the legacy crawler implementation."""

    def __init__(
        self,
        crawler: TwStockCrawler | None = None,
        http_client: TwseHttpClient | None = None,
    ) -> None:
        if crawler is not None and http_client is not None:
            raise ValueError("Pass either crawler or http_client, not both")
        self._crawler = crawler or TwStockCrawler(http_client=http_client)

    def get_daily_stock_rows(
        self,
        date_time: str,
        stocktype: int | str,
    ) -> StockRows:
        return self._crawler.get_stocktype_data(date_time, stocktype)

    def get_daily_taiex_close(self, date_time: str) -> float | None:
        return self._crawler.get_taiex_close(date_time)
