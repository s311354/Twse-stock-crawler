"""Domain-visible data-source failures independent of HTTP implementations."""

from __future__ import annotations

import enum


class DataErrorCategory(str, enum.Enum):
    MARKET_CLOSED = "market_closed"
    NO_DATA = "no_data"
    SCHEMA_CHANGED = "schema_changed"
    NETWORK_ERROR = "network_error"


class DataSourceError(RuntimeError):
    category: DataErrorCategory

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class MarketClosedError(DataSourceError):
    category = DataErrorCategory.MARKET_CLOSED


class NoDataError(DataSourceError):
    category = DataErrorCategory.NO_DATA


class SchemaChangedError(DataSourceError):
    category = DataErrorCategory.SCHEMA_CHANGED


class NetworkError(DataSourceError):
    category = DataErrorCategory.NETWORK_ERROR
