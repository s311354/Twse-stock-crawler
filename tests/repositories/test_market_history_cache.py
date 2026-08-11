from __future__ import annotations

import sqlite3

import pytest

from application.market_history_cache import CachedMarketDay
from infrastructure.storage.market_history_cache import SQLiteMarketHistoryCache


def _day(trade_date: str = "20260807") -> CachedMarketDay:
    return CachedMarketDay(
        trade_date=trade_date,
        stock_rows=[
            [
                "2330",
                "台積電",
                "1,000",
                "10",
                "100,000",
                "100",
                "105",
                "99",
                "104",
            ]
        ],
        taiex_close=24_000.5,
    )


def test_sqlite_market_history_cache_round_trip_and_schema_isolation(
    tmp_path,
) -> None:
    cache = SQLiteMarketHistoryCache(tmp_path / "market_history.sqlite3")
    day = _day()
    cache.upsert_day(day, market_type="ALLBUT0999", schema_version=1)

    loaded = cache.load_many(
        ["20260806", "20260807", "20260807"],
        market_type="ALLBUT0999",
        schema_version=1,
    )

    assert loaded == {"20260807": day}
    assert cache.load_many(
        ["20260807"],
        market_type="ALLBUT0999",
        schema_version=2,
    ) == {}
    assert cache.load_many(
        ["20260807"],
        market_type="24",
        schema_version=1,
    ) == {}


def test_sqlite_market_history_cache_upserts_both_datasets_atomically(
    tmp_path,
) -> None:
    database = tmp_path / "market_history.sqlite3"
    cache = SQLiteMarketHistoryCache(database)
    original = _day()
    replacement = CachedMarketDay(
        trade_date=original.trade_date,
        stock_rows=[["2330", "台積電", "2,000"]],
        taiex_close=24_100.0,
    )
    cache.upsert_day(original, market_type="ALLBUT0999")
    cache.upsert_day(replacement, market_type="ALLBUT0999")

    with sqlite3.connect(database) as connection:
        datasets = connection.execute(
            """
            SELECT dataset
            FROM market_history_cache
            WHERE trade_date = ?
            ORDER BY dataset
            """,
            (original.trade_date,),
        ).fetchall()

    assert datasets == [("STOCK_TABLE",), ("TAIEX",)]
    assert cache.load_many(
        [original.trade_date],
        market_type="ALLBUT0999",
    ) == {original.trade_date: replacement}


def test_sqlite_market_history_cache_ignores_partial_or_corrupt_days(
    tmp_path,
) -> None:
    database = tmp_path / "market_history.sqlite3"
    cache = SQLiteMarketHistoryCache(database)
    cache.upsert_day(_day(), market_type="ALLBUT0999")
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            DELETE FROM market_history_cache
            WHERE trade_date = ? AND dataset = 'TAIEX'
            """,
            ("20260807",),
        )

    assert cache.load_many(
        ["20260807"],
        market_type="ALLBUT0999",
    ) == {}


def test_sqlite_market_history_cache_rejects_invalid_complete_day(tmp_path) -> None:
    cache = SQLiteMarketHistoryCache(tmp_path / "market_history.sqlite3")

    with pytest.raises(ValueError, match="taiex_close"):
        cache.upsert_day(
            CachedMarketDay(
                trade_date="20260807",
                stock_rows=[["2330"]],
                taiex_close=0,
            ),
            market_type="ALLBUT0999",
        )

    assert not (tmp_path / "market_history.sqlite3").exists()
