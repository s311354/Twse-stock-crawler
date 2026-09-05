"""Backward-compatible imports for stock-analysis runtime adapters.

Runtime implementations live in ``infrastructure.runtime``. This module stays
as a stable facade for integrations that still import the historical path.
"""

from infrastructure.runtime.fundamental_adapter import CrawlerFundamentalAdapter
from infrastructure.runtime.market_data_adapter import CrawlerMarketDataAdapter
from infrastructure.runtime.output_adapters import CrawlerLineChartAdapter
from infrastructure.runtime.output_adapters import CrawlerNotificationAdapter
from infrastructure.runtime.output_adapters import CrawlerReportAdapter
from infrastructure.runtime.strategy_adapter import CrawlerStrategyAdapter

__all__ = [
    "CrawlerFundamentalAdapter",
    "CrawlerLineChartAdapter",
    "CrawlerMarketDataAdapter",
    "CrawlerNotificationAdapter",
    "CrawlerReportAdapter",
    "CrawlerStrategyAdapter",
]
