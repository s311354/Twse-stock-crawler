"""SQLite-backed incremental cache for TWSE market history."""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from pathlib import Path

from application.market_history_cache import CachedMarketDay
from application.market_history_cache import MARKET_HISTORY_SCHEMA_VERSION
from application.market_history_cache import MarketHistoryCacheError
from domain.models import StockRows


STOCK_TABLE_DATASET = "STOCK_TABLE"
TAIEX_DATASET = "TAIEX"
TAIEX_MARKET_TYPE = "IND"


class SQLiteMarketHistoryCache:
    """Store complete market days using one row per dataset and cache key."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._schema_ready = False

    def load_many(
        self,
        trade_dates: list[str],
        *,
        market_type: str,
        schema_version: int = MARKET_HISTORY_SCHEMA_VERSION,
    ) -> dict[str, CachedMarketDay]:
        normalized_dates = sorted(set(trade_dates))
        if not normalized_dates or not self.database_path.exists():
            return {}
        self._validate_market_type(market_type)
        self._validate_schema_version(schema_version)
        for trade_date in normalized_dates:
            self._validate_trade_date(trade_date)

        placeholders = ",".join("?" for _ in normalized_dates)
        query = """
            SELECT trade_date, dataset, payload_json
            FROM market_history_cache
            WHERE trade_date IN ({})
              AND schema_version = ?
              AND (
                    (dataset = ? AND market_type = ?)
                 OR (dataset = ? AND market_type = ?)
              )
            ORDER BY trade_date, dataset
        """.format(placeholders)
        parameters: list[object] = list(normalized_dates)
        parameters.extend(
            [
                schema_version,
                STOCK_TABLE_DATASET,
                market_type,
                TAIEX_DATASET,
                TAIEX_MARKET_TYPE,
            ]
        )
        try:
            self._ensure_schema()
            with sqlite3.connect(self.database_path) as connection:
                records = connection.execute(query, parameters).fetchall()
        except sqlite3.Error as error:
            raise MarketHistoryCacheError(
                "Unable to load TWSE market-history cache at {}: {}".format(
                    self.database_path,
                    error,
                )
            ) from error

        payloads: dict[str, dict[str, str]] = {}
        for trade_date, dataset, payload_json in records:
            payloads.setdefault(str(trade_date), {})[str(dataset)] = str(
                payload_json
            )

        loaded: dict[str, CachedMarketDay] = {}
        for trade_date, datasets in payloads.items():
            day = self._decode_day(trade_date, datasets)
            if day is not None:
                loaded[trade_date] = day
        return loaded

    def upsert_day(
        self,
        day: CachedMarketDay,
        *,
        market_type: str,
        schema_version: int = MARKET_HISTORY_SCHEMA_VERSION,
    ) -> None:
        self._validate_trade_date(day.trade_date)
        self._validate_market_type(market_type)
        self._validate_schema_version(schema_version)
        stock_rows = self._normalize_stock_rows(day.stock_rows)
        taiex_close = self._normalize_taiex(day.taiex_close)
        cached_at_epoch = int(dt.datetime.now(dt.timezone.utc).timestamp())
        rows = [
            (
                day.trade_date,
                STOCK_TABLE_DATASET,
                market_type,
                schema_version,
                json.dumps(stock_rows, ensure_ascii=False, separators=(",", ":")),
                cached_at_epoch,
            ),
            (
                day.trade_date,
                TAIEX_DATASET,
                TAIEX_MARKET_TYPE,
                schema_version,
                json.dumps(taiex_close),
                cached_at_epoch,
            ),
        ]
        try:
            self._ensure_schema()
            with sqlite3.connect(self.database_path) as connection:
                connection.executemany(
                    """
                    INSERT INTO market_history_cache (
                        trade_date,
                        dataset,
                        market_type,
                        schema_version,
                        payload_json,
                        cached_at_epoch
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(
                        trade_date,
                        dataset,
                        market_type,
                        schema_version
                    ) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        cached_at_epoch = excluded.cached_at_epoch
                    """,
                    rows,
                )
        except sqlite3.Error as error:
            raise MarketHistoryCacheError(
                "Unable to save TWSE market-history cache at {}: {}".format(
                    self.database_path,
                    error,
                )
            ) from error

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.database_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS market_history_cache (
                        trade_date TEXT NOT NULL,
                        dataset TEXT NOT NULL,
                        market_type TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        cached_at_epoch INTEGER NOT NULL,
                        PRIMARY KEY (
                            trade_date,
                            dataset,
                            market_type,
                            schema_version
                        )
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS market_history_cache_lookup
                    ON market_history_cache (
                        schema_version,
                        trade_date,
                        dataset,
                        market_type
                    )
                    """
                )
        except sqlite3.Error as error:
            raise MarketHistoryCacheError(
                "Unable to initialize TWSE market-history cache at {}: {}".format(
                    self.database_path,
                    error,
                )
            ) from error
        self._schema_ready = True

    @classmethod
    def _decode_day(
        cls,
        trade_date: str,
        datasets: dict[str, str],
    ) -> CachedMarketDay | None:
        if STOCK_TABLE_DATASET not in datasets or TAIEX_DATASET not in datasets:
            return None
        try:
            stock_rows = cls._normalize_stock_rows(
                json.loads(datasets[STOCK_TABLE_DATASET])
            )
            taiex_close = cls._normalize_taiex(
                json.loads(datasets[TAIEX_DATASET])
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return CachedMarketDay(
            trade_date=trade_date,
            stock_rows=stock_rows,
            taiex_close=taiex_close,
        )

    @staticmethod
    def _normalize_stock_rows(value: object) -> StockRows:
        if not isinstance(value, list) or not value:
            raise ValueError("stock_rows must be a non-empty list")
        normalized: StockRows = []
        for row in value:
            if not isinstance(row, list) or not row:
                raise ValueError("each stock row must be a non-empty list")
            normalized.append([str(cell) for cell in row])
        return normalized

    @staticmethod
    def _normalize_taiex(value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("taiex_close must be numeric")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError("taiex_close must be positive and finite")
        return parsed

    @staticmethod
    def _validate_trade_date(value: str) -> None:
        try:
            parsed = dt.datetime.strptime(value, "%Y%m%d")
        except (TypeError, ValueError) as error:
            raise ValueError("trade_date must use YYYYMMDD format") from error
        if parsed.strftime("%Y%m%d") != value:
            raise ValueError("trade_date must use YYYYMMDD format")

    @staticmethod
    def _validate_market_type(value: str) -> None:
        if not value or not value.strip():
            raise ValueError("market_type cannot be empty")

    @staticmethod
    def _validate_schema_version(value: int) -> None:
        if value < 1:
            raise ValueError("schema_version must be positive")


__all__ = [
    "SQLiteMarketHistoryCache",
    "STOCK_TABLE_DATASET",
    "TAIEX_DATASET",
    "TAIEX_MARKET_TYPE",
]
