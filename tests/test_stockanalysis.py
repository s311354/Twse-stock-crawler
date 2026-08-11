import datetime
import logging

import pandas as pd
import pytest

import stockanalysis
from application.analysis_dataset import AnalysisDatasetBuilder
from application.market_history_cache import CachedMarketDay
from application.market_history_cache import MarketHistoryCachePort
from application.strategy_execution import StrategyExecutionService
from domain.features import SharedFeaturePipeline
from domain.market_breadth import MARKET_BREADTH_ATTR
from domain.market_breadth import MARKET_BREADTH_HISTORY_ATTR
from domain.market_breadth import MARKET_DATA_THROUGH_ATTR
from domain.market_breadth import MARKET_SESSION_FINAL_ATTR
from infrastructure.calendar.twse_calendar import TradingDayStatus
from infrastructure.calendar.twse_calendar import TwseCalendar
from infrastructure.crawler.twse_http_client import TwseHttpResponse
from infrastructure.crawler.twse_http_client import TwseMarketClosedError
from infrastructure.storage.market_history_cache import SQLiteMarketHistoryCache


class DummyStockType:
    value = (13,)


class FakeTwseClient:
    def __init__(
        self,
        rows_by_date: dict[str, stockanalysis.StockRows],
        taiex_by_date: dict[str, float | None] | None = None,
    ) -> None:
        self.rows_by_date = rows_by_date
        self.taiex_by_date = taiex_by_date or {}
        self.stock_calls: list[str] = []
        self.taiex_calls: list[str] = []

    def get_daily_stock_rows(self, date_time: str, stocktype: int) -> stockanalysis.StockRows:
        self.stock_calls.append(date_time)
        return self.rows_by_date[date_time]

    def get_daily_taiex_close(self, date_time: str) -> float | None:
        self.taiex_calls.append(date_time)
        return self.taiex_by_date.get(date_time, 20000.0)


class CapturingTwseCrawler(stockanalysis.TwseCrawker):
    def __init__(
        self,
        stocklistsize: int,
        rows_by_date: dict[str, stockanalysis.StockRows] | None = None,
        taiex_by_date: dict[str, float | None] | None = None,
        analysis_dataset_builder: AnalysisDatasetBuilder | None = None,
        market_history_cache: MarketHistoryCachePort | None = None,
    ) -> None:
        super().__init__(
            stocklistsize,
            twse_client=FakeTwseClient(rows_by_date or {}, taiex_by_date),
            analysis_dataset_builder=analysis_dataset_builder,
            market_history_cache=market_history_cache,
        )
        self.records: list[tuple[str, stockanalysis.StockRows]] = []

    def record(self, file_name: str, scheduled_time: str, row_data: stockanalysis.StockRows) -> None:
        self.records.append((scheduled_time, row_data))


class FakeBacktrackCalendar:
    def __init__(self, statuses: dict[datetime.date, TradingDayStatus]) -> None:
        self.statuses = statuses

    def trading_day_status(self, target_date: datetime.date) -> TradingDayStatus:
        return self.statuses.get(target_date, TradingDayStatus.UNKNOWN)


def twse_row(
    open_price: str,
    high_price: str,
    low_price: str,
    close_price: str,
    volume: str = "1,000",
    pe_ratio: str = "12.3",
    stock_no: str = "3701",
    stock_name: str = "大眾控",
) -> stockanalysis.StockRow:
    return [
        stock_no,
        stock_name,
        volume,
        "10",
        "58,000",
        open_price,
        high_price,
        low_price,
        close_price,
        "+",
        "0.5",
        "58.5",
        "1",
        "57.2",
        "1",
        pe_ratio,
    ]


def test_market_history_cache_fetches_only_missing_dates(tmp_path) -> None:
    cache = SQLiteMarketHistoryCache(tmp_path / "market_history.sqlite3")
    cached_row = twse_row("58.60", "58.70", "56.30", "57.10")
    cache.upsert_day(
        CachedMarketDay(
            trade_date="20260525",
            stock_rows=[cached_row],
            taiex_close=19_900.0,
        ),
        market_type="13",
    )
    rows_by_date = {
        "20260526": [twse_row("59.00", "60.00", "58.00", "59.50")],
    }
    crawler = CapturingTwseCrawler(
        1,
        rows_by_date,
        taiex_by_date={"20260526": 20_000.0},
        market_history_cache=cache,
    )
    crawler.iso_scheduled_times = ["2026-05-25", "2026-05-26"]

    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["3701"])

    assert crawler.twse_client.stock_calls == ["20260526"]
    assert crawler.twse_client.taiex_calls == ["20260526"]
    assert crawler.daily_dates == [["2026-05-25", "2026-05-26"]]
    assert crawler.daily_taiex_closes == [[19_900.0, 20_000.0]]
    assert set(
        cache.load_many(
            ["20260525", "20260526"],
            market_type="13",
        )
    ) == {"20260525", "20260526"}


def test_market_history_cache_refreshes_current_reference_date(tmp_path) -> None:
    cache = SQLiteMarketHistoryCache(tmp_path / "market_history.sqlite3")
    cache.upsert_day(
        CachedMarketDay(
            trade_date="20260526",
            stock_rows=[twse_row("58", "59", "57", "58.5")],
            taiex_close=19_900.0,
        ),
        market_type="13",
    )
    refreshed_row = twse_row("60", "61", "59", "60.5")
    crawler = CapturingTwseCrawler(
        1,
        {"20260526": [refreshed_row]},
        taiex_by_date={"20260526": 20_100.0},
        market_history_cache=cache,
    )
    crawler.iso_scheduled_times = ["2026-05-26"]
    crawler._market_history_refresh_dates = {"20260526"}

    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["3701"])

    assert crawler.twse_client.stock_calls == ["20260526"]
    assert crawler.daily_closes == [[60_500.0]]
    refreshed = cache.load_many(["20260526"], market_type="13")["20260526"]
    assert refreshed.taiex_close == 20_100.0
    assert refreshed.stock_rows == [refreshed_row]


def test_missing_ohlc_does_not_reuse_previous_daily_row() -> None:
    rows_by_date = {
        "20260525": [twse_row("58.60", "58.70", "56.30", "57.10")],
        "20260526": [twse_row("--", "--", "--", "--")],
    }

    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.iso_scheduled_times = ["2026-05-25", "2026-05-26"]
    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["0"])

    assert crawler.records[0][0] == "20260525"
    assert crawler.records[0][1][0][:9] == ["3701", "大眾控", "1000", "10", "58000", "58.60", "58.70", "56.30", "57.10"]
    assert len(crawler.records) == 1
    assert crawler.iso_scheduled_times == ["2026-05-25"]
    assert crawler.daily_stocks == [[58600.0]]
    assert crawler.daily_volumes == [[1000.0]]
    assert crawler.daily_trade_counts == [[10]]
    assert crawler.daily_trade_values == [[58000.0]]
    assert crawler.daily_bid_prices == [[58500.0]]
    assert crawler.daily_ask_prices == [[57200.0]]
    assert crawler.daily_pe_ratios == [[12.3]]
    assert crawler.daily_taiex_closes == [[20000.0]]
    assert crawler.daily_dates == [["2026-05-25"]]


def test_missing_ohlc_logs_debug_details_and_one_release_summary(
    caplog,
) -> None:
    rows_by_date = {
        "20260525": [
            twse_row("--", "--", "--", "--", stock_no="2321"),
            twse_row(
                "106",
                "110",
                "105",
                "109",
                stock_no="2382",
                stock_name="廣達",
            ),
        ],
        "20260526": [
            twse_row("--", "--", "--", "--", stock_no="2321"),
            twse_row(
                "107",
                "111",
                "106",
                "110",
                stock_no="2382",
                stock_name="廣達",
            ),
        ],
    }
    crawler = CapturingTwseCrawler(2, rows_by_date)
    crawler.iso_scheduled_times = ["2026-05-25", "2026-05-26"]

    with caplog.at_level(logging.DEBUG):
        crawler.get_twse_daily_stocks(
            "shirong",
            DummyStockType,
            ["2321", "2382"],
        )

    details = [
        record
        for record in caplog.records
        if getattr(record, "status", None) == "OHLC_MISSING"
    ]
    summaries = [
        record
        for record in caplog.records
        if getattr(record, "status", None) == "OHLC_MISSING_SUMMARY"
    ]
    assert len(details) == 2
    assert all(record.levelno == logging.DEBUG for record in details)
    assert len(summaries) == 1
    assert summaries[0].levelno == logging.WARNING
    assert summaries[0].source == "TWSE"
    assert (
        "records=2 stocks=1 first_date=20260525 last_date=20260526"
        in summaries[0].getMessage()
    )


def test_no_data_date_logs_one_final_warning_and_is_collected(caplog) -> None:
    rows_by_date = {
        "20260525": [],
        "20260526": [twse_row("58.60", "58.70", "56.30", "57.10")],
    }
    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.iso_scheduled_times = ["2026-05-25", "2026-05-26"]

    with caplog.at_level(logging.INFO):
        crawler.get_twse_daily_stocks(
            "shirong",
            DummyStockType,
            ["0"],
        )

    no_data_warnings = [
        record
        for record in caplog.records
        if getattr(record, "status", None) == "NO_DATA_SUMMARY"
    ]
    assert len(no_data_warnings) == 1
    assert no_data_warnings[0].levelno == logging.WARNING
    assert "dates=1 first_date=20260525 last_date=20260525" in (
        no_data_warnings[0].getMessage()
    )
    assert crawler.twse_no_data_dates == {"2026-05-25"}


def test_unscheduled_market_closure_is_not_reported_as_no_data(caplog) -> None:
    class MarketClosedClient(FakeTwseClient):
        def get_daily_stock_rows(self, date_time, stocktype):
            if date_time == "20260710":
                raise TwseMarketClosedError("full market contains no rows")
            return super().get_daily_stock_rows(date_time, stocktype)

    rows_by_date = {
        "20260713": [twse_row("58.60", "58.70", "56.30", "57.10")],
    }
    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.twse_client = MarketClosedClient(rows_by_date)
    crawler.iso_scheduled_times = ["2026-07-10", "2026-07-13"]

    with caplog.at_level(logging.INFO):
        crawler.get_twse_daily_stocks(
            "shirong",
            DummyStockType,
            ["0"],
        )

    summaries = [
        record
        for record in caplog.records
        if getattr(record, "status", None)
        == "CLOSED_UNSCHEDULED_SUMMARY"
    ]
    assert len(summaries) == 1
    assert summaries[0].levelno == logging.INFO
    assert crawler.twse_closed_unscheduled_dates == {"2026-07-10"}
    assert crawler.twse_no_data_dates == set()


def test_invalid_twse_row_shape_is_skipped_gracefully() -> None:
    crawler = stockanalysis.TwseCrawker(1)

    assert crawler.parse_ohlc([["3701", "大眾控"]], 0, "20260526") is None
    assert crawler.parse_ohlc([], 0, "20260526") is None


def test_empty_twse_date_is_skipped_without_recording() -> None:
    rows_by_date = {
        "20260515": [],
        "20260518": [twse_row("58.60", "58.70", "56.30", "57.10")],
    }

    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.iso_scheduled_times = ["2026-05-15", "2026-05-18"]
    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["0"])

    assert crawler.records[0][0] == "20260518"
    assert crawler.records[0][1][0][:9] == ["3701", "大眾控", "1000", "10", "58000", "58.60", "58.70", "56.30", "57.10"]
    assert crawler.iso_scheduled_times == ["2026-05-18"]
    assert crawler.daily_stocks == [[58600.0]]


def test_legacy_index_selector_locks_stock_number_when_twse_order_changes() -> None:
    rows_by_date = {
        "20260616": [
            twse_row("100", "105", "99", "104", stock_no="2382", stock_name="廣達"),
            twse_row("200", "205", "198", "204", stock_no="2383", stock_name="台光電"),
        ],
        "20260617": [
            twse_row("210", "215", "208", "214", stock_no="2383", stock_name="台光電"),
            twse_row("106", "110", "105", "109", stock_no="2382", stock_name="廣達"),
        ],
    }

    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.iso_scheduled_times = ["2026-06-16", "2026-06-17"]
    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["0"])

    assert [record[1][0][0] for record in crawler.records] == ["2382", "2382"]
    assert crawler.tracked_stock_numbers == ["2382"]
    assert crawler.stocknames == ["廣達"]
    assert crawler.daily_closes == [[104000.0, 109000.0]]
    assert crawler.daily_taiex_closes == [[20000.0, 20000.0]]


def test_stock_number_selector_does_not_depend_on_twse_row_index() -> None:
    rows_by_date = {
        "20260617": [
            twse_row("210", "215", "208", "214", stock_no="2383", stock_name="台光電"),
            twse_row("106", "110", "105", "109", stock_no="2382", stock_name="廣達"),
        ],
    }

    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.iso_scheduled_times = ["2026-06-17"]
    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["2382"])

    assert crawler.records[0][1][0][0] == "2382"
    assert crawler.daily_closes == [[109000.0]]


def test_tracked_stock_missing_does_not_fallback_to_same_index() -> None:
    rows_by_date = {
        "20260616": [
            twse_row("100", "105", "99", "104", stock_no="2382", stock_name="廣達"),
        ],
        "20260617": [
            twse_row("210", "215", "208", "214", stock_no="2383", stock_name="台光電"),
        ],
    }

    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.iso_scheduled_times = ["2026-06-16", "2026-06-17"]
    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["0"])

    assert [record[0] for record in crawler.records] == ["20260616"]
    assert crawler.daily_closes == [[104000.0]]
    assert crawler.iso_scheduled_times == ["2026-06-16"]


def test_consecutive_missing_stock_logs_one_warning_and_recovery(caplog) -> None:
    crawler = stockanalysis.TwseCrawker(1)
    crawler.tracked_stock_numbers[0] = "2382"
    row = twse_row("106", "110", "105", "109", stock_no="2382", stock_name="廣達")

    with caplog.at_level(logging.DEBUG):
        assert crawler.resolve_stock_row([], {}, "0", 0, "20260713") is None
        assert crawler.resolve_stock_row([], {}, "0", 0, "20260714") is None
        assert crawler.resolve_stock_row([row], {"2382": row}, "0", 0, "20260715") == row

    missing_details = [
        record
        for record in caplog.records
        if getattr(record, "status", None) == "TRACKED_STOCK_MISSING"
    ]
    assert len(missing_details) == 1
    assert missing_details[0].levelno == logging.DEBUG
    assert missing_details[0].symbol == "2382"
    assert missing_details[0].trade_date == "20260713"
    assert "Tracked stock 2382 remains missing" in caplog.text
    assert "Tracked stock 2382 returned" in caplog.text
    assert crawler.missing_tracked_stock_first_dates == {
        "2382": "20260713",
    }


def test_missing_tracked_stock_details_are_suppressed_in_release(caplog) -> None:
    rows_by_date = {
        "20260616": [
            twse_row(
                "100", "105", "99", "104",
                stock_no="2382", stock_name="廣達",
            ),
        ],
        "20260617": [
            twse_row(
                "210", "215", "208", "214",
                stock_no="2383", stock_name="台光電",
            ),
        ],
    }
    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.iso_scheduled_times = ["2026-06-16", "2026-06-17"]

    with caplog.at_level(logging.INFO):
        crawler.get_twse_daily_stocks("shirong", DummyStockType, ["0"])

    details = [
        record
        for record in caplog.records
        if getattr(record, "status", None) == "TRACKED_STOCK_MISSING"
    ]
    assert details == []
    assert not any(
        "Skipping tracked stock" in record.getMessage()
        for record in caplog.records
    )
    assert crawler.missing_tracked_stock_first_dates == {
        "2382": "20260617",
    }


def test_invalid_twse_schema_rows_are_not_written() -> None:
    rows_by_date = {
        "20260616": [["2382", "廣達", "1000", "10", "58000", "100", "105", "99", "104"]],
        "20260617": [
            twse_row("106", "110", "105", "109", stock_no="2382", stock_name="廣達"),
        ],
    }

    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.iso_scheduled_times = ["2026-06-16", "2026-06-17"]
    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["2382"])

    assert [record[0] for record in crawler.records] == ["20260617"]
    assert crawler.daily_closes == [[109000.0]]


def test_taiex_history_aligns_with_each_stocks_valid_rows() -> None:
    rows_by_date = {
        "20260616": [
            twse_row("100", "105", "99", "104", stock_no="2382", stock_name="廣達"),
            twse_row("200", "205", "198", "204", stock_no="2383", stock_name="台光電"),
        ],
        "20260617": [
            twse_row("106", "110", "105", "109", stock_no="2382", stock_name="廣達"),
        ],
    }
    crawler = CapturingTwseCrawler(
        2,
        rows_by_date,
        {"20260616": 20000.0, "20260617": 20100.0},
    )
    crawler.iso_scheduled_times = ["2026-06-16", "2026-06-17"]

    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["2382", "2383"])

    assert crawler.daily_closes == [[104000.0, 109000.0], [204000.0]]
    assert crawler.daily_taiex_closes == [[20000.0, 20100.0], [20000.0]]
    assert crawler.build_strategy_history(0)["Date"].tolist() == ["2026-06-16", "2026-06-17"]
    assert crawler.build_strategy_history(1)["Date"].tolist() == ["2026-06-16"]
    assert len(crawler.build_strategy_history(0)) == 2
    assert len(crawler.build_strategy_history(1)) == 1


def test_completed_daily_rows_capture_full_universe_market_breadth() -> None:
    rows_by_date = {
        "20260729": [
            twse_row(
                "99",
                "101",
                "98",
                "100",
                stock_no="2382",
                stock_name="廣達",
            ),
        ],
        "20260730": [
            twse_row(
                "100",
                "105",
                "99",
                "104",
                stock_no="2382",
                stock_name="廣達",
            ),
            twse_row(
                "200",
                "205",
                "198",
                "204",
                stock_no="2383",
                stock_name="台光電",
            ),
            twse_row(
                "50",
                "52",
                "49",
                "51",
                stock_no="2303",
                stock_name="聯電",
            ),
        ],
    }
    crawler = CapturingTwseCrawler(
        1,
        rows_by_date,
        {"20260729": 40_000.0, "20260730": 42_000.0},
    )
    crawler.iso_scheduled_times = ["2026-07-29", "2026-07-30"]

    crawler.get_twse_daily_stocks(
        "shirong",
        stockanalysis.Stocktype.ELEC,
        ["2382"],
    )

    snapshot = crawler.market_breadth_snapshots_by_date["2026-07-30"]
    assert snapshot["universe"] == "ELEC"
    assert snapshot["eligible_count"] == 3
    assert snapshot["valid_return_count"] == 3
    assert snapshot["source_final"] is True
    assert snapshot["taiex_return"] == pytest.approx(0.05)

    history = crawler.build_strategy_history(0)
    assert history.attrs[MARKET_DATA_THROUGH_ATTR] == "2026-07-30"
    assert history.attrs[MARKET_SESSION_FINAL_ATTR] is True
    assert history.attrs[MARKET_BREADTH_ATTR]["valid_return_count"] == 3
    assert set(history.attrs[MARKET_BREADTH_HISTORY_ATTR]) == {
        "2026-07-29",
        "2026-07-30",
    }


def test_missing_taiex_skips_date_without_misaligned_stock_data() -> None:
    rows_by_date = {
        "20260616": [twse_row("100", "105", "99", "104", stock_no="2382", stock_name="廣達")],
        "20260617": [twse_row("106", "110", "105", "109", stock_no="2382", stock_name="廣達")],
    }
    crawler = CapturingTwseCrawler(
        1,
        rows_by_date,
        {"20260616": None, "20260617": 20100.0},
    )
    crawler.iso_scheduled_times = ["2026-06-16", "2026-06-17"]

    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["2382"])

    assert [record[0] for record in crawler.records] == ["20260617"]
    assert crawler.daily_closes == [[109000.0]]
    assert crawler.daily_taiex_closes == [[20100.0]]
    assert crawler.iso_scheduled_times == ["2026-06-17"]


def test_daily_analysis_dataset_uses_low_entry_score_v5() -> None:
    crawler = CapturingTwseCrawler(1)
    crawler.iso_scheduled_times = ["2026-06-17"]
    crawler.stocknumbers = ["2382"]
    crawler.stocknames = ["廣達"]
    closes = [100000.0 + index * 200 for index in range(220)]
    crawler.daily_stocks = [[close - 100 for close in closes]]
    crawler.daily_highs = [[close + 200 for close in closes]]
    crawler.daily_lows = [[close - 200 for close in closes]]
    crawler.daily_closes = [closes]
    crawler.daily_volumes = [[1000.0] * 219 + [3000.0]]
    crawler.daily_pe_ratios = [[None] * 220]
    taiex_closes = [20000.0 + index * 10 for index in range(220)]
    crawler.daily_taiex_closes = [taiex_closes]

    decisions = crawler.build_strategy_decisions(
        [[1, 1, 1, 1, 0.01]]
    )
    dataset = crawler.project_strategy_decisions(decisions)

    assert "LES v5 Final Quality Status" in dataset.columns
    assert "Base Quality Status" in dataset.columns
    assert "Base Data Confidence" in dataset.columns
    assert "Extended Quality Status" in dataset.columns
    assert "Extended Data Confidence" in dataset.columns
    assert "Falling Knife State" in dataset.columns
    assert "LES v5 Trigger" in dataset.columns
    assert "動能分數" in dataset.columns
    assert "Risk Gate" in dataset.columns
    assert "RS20" in dataset.columns
    assert dataset.loc[0, "LES v5 Final Quality Status"] == "DATA_INSUFFICIENT"
    assert dataset.loc[0, "低點決策"] == "DATA_INSUFFICIENT"
    assert dataset.loc[0, "Base Data Confidence"] == 0
    assert dataset.loc[0, "Extended Data Confidence"] == 0
    rendered = crawler.build_entry_signal_analysis(decisions)
    assert "Low-Entry Model v5" in rendered
    assert "Quality-First Panic Reversal" in rendered
    assert "DATA_INSUFFICIENT" in rendered
    assert "EREM 分數" in dataset.columns
    assert "EREM 決策訊號" in dataset.columns
    assert "EREM RewardRisk" in dataset.columns
    assert "EREM Highest120" in dataset.columns
    assert "EREM MACD Histogram Rising" in dataset.columns
    assert "EREM Holding Confidence" in dataset.columns
    assert "EREM Relative Strength60" in dataset.columns
    assert "EREM VWAP60" in dataset.columns
    expected_rs60 = round(
        (closes[-1] / closes[-61] - 1) - (taiex_closes[-1] / taiex_closes[-61] - 1),
        4,
    )
    assert dataset.loc[0, "EREM Relative Strength60"] == expected_rs60
    assert dataset.loc[0, "EREM Relative Strength Score"] == 10
    assert dataset.loc[0, "EREM 決策訊號"] in {
        "STRONG HOLD",
        "HOLD",
        "REDUCE 25%",
        "EXIT",
    }


def test_analysis_builds_shared_features_once_per_stock() -> None:
    builds = 0

    class CountingPipeline:
        def __init__(self, *args, **kwargs) -> None:
            self.delegate = SharedFeaturePipeline(*args, **kwargs)

        def build(self, data):
            nonlocal builds
            builds += 1
            return self.delegate.build(data)

    analysis_dataset_builder = AnalysisDatasetBuilder(
        strategy_execution=StrategyExecutionService(
            feature_pipeline_factory=CountingPipeline,
        ),
    )
    crawler = CapturingTwseCrawler(
        1,
        analysis_dataset_builder=analysis_dataset_builder,
    )
    crawler.iso_scheduled_times = ["2026-06-17"]
    crawler.stocknumbers = ["2382"]
    crawler.stocknames = ["廣達"]
    closes = [100_000.0 + index * 100 for index in range(220)]
    crawler.daily_stocks = [[close - 100 for close in closes]]
    crawler.daily_highs = [[close + 200 for close in closes]]
    crawler.daily_lows = [[close - 200 for close in closes]]
    crawler.daily_closes = [closes]
    crawler.daily_volumes = [[1_000.0] * 220]
    crawler.daily_pe_ratios = [[15.0] * 220]
    crawler.daily_taiex_closes = [
        [20_000.0 + index * 5 for index in range(220)]
    ]
    crawler.daily_dates = [
        [
            date.strftime("%Y-%m-%d")
            for date in pd.date_range(
                "2025-08-15",
                periods=220,
                freq="B",
            )
        ]
    ]

    crawler.build_analysis_dataset([[1, 1, 1, 1, 0.01]])

    assert builds == 1


def test_backtrack_dates_log_debug_status_and_skip_closed_days(
    caplog,
    capsys,
) -> None:
    caplog.set_level("DEBUG")
    reference_date = datetime.date(2026, 7, 20)
    calendar = FakeBacktrackCalendar(
        {
            datetime.date(2026, 7, 15): TradingDayStatus.OPEN,
            datetime.date(2026, 7, 16): TradingDayStatus.CLOSED,
            datetime.date(2026, 7, 17): TradingDayStatus.OPEN,
            datetime.date(2026, 7, 18): TradingDayStatus.UNKNOWN,
            datetime.date(2026, 7, 19): TradingDayStatus.CLOSED,
        }
    )
    crawler = CapturingTwseCrawler(1)

    crawler.get_date_times(
        begin_backtrack=0,
        end_backtrack=5,
        calendar=calendar,
        reference_date=reference_date,
    )

    assert crawler.iso_scheduled_times == ["2026-07-15", "2026-07-17"]
    assert capsys.readouterr().out == ""
    output = caplog.text
    assert "TWSE backtrack date 2026-07-15: OPEN" in output
    assert "TWSE backtrack date 2026-07-16: CLOSED" in output
    assert "TWSE backtrack date 2026-07-18: UNKNOWN" in output
    assert "candidates=5 open=2 closed=2 unknown=1 selected=2" in output


def test_backtrack_dates_can_probe_completed_reference_session(
    caplog,
    capsys,
) -> None:
    caplog.set_level("DEBUG")
    reference_date = datetime.date(2026, 7, 20)
    calendar = FakeBacktrackCalendar(
        {
            datetime.date(2026, 7, 19): TradingDayStatus.CLOSED,
            reference_date: TradingDayStatus.OPEN,
        }
    )
    crawler = CapturingTwseCrawler(1)

    crawler.get_date_times(
        begin_backtrack=0,
        end_backtrack=1,
        calendar=calendar,
        reference_date=reference_date,
        include_reference=True,
    )

    assert crawler.iso_scheduled_times == ["2026-07-20"]
    assert crawler._market_history_refresh_dates == {"20260720"}
    assert capsys.readouterr().out == ""
    assert "TWSE backtrack date 2026-07-20: OPEN" in caplog.text


def test_historical_holiday_is_removed_before_market_data_request() -> None:
    class HistoricalCalendarHttpClient:
        def get_json(self, url: str, **kwargs: object) -> TwseHttpResponse:
            return TwseHttpResponse(
                {
                    "stat": "ok",
                    "fields": ["日期", "名稱", "說明"],
                    "data": [
                        ["2025-05-01", "勞動節", "依規定放假1日。"],
                    ],
                },
                200,
                1,
            )

    rows_by_date = {
        "20250430": [twse_row("57.80", "58.20", "57.10", "58.00")],
        "20250502": [twse_row("58.60", "58.70", "56.30", "57.10")],
    }
    market_client = FakeTwseClient(rows_by_date)
    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.twse_client = market_client
    calendar = TwseCalendar(http_client=HistoricalCalendarHttpClient())

    crawler.get_date_times(
        begin_backtrack=0,
        end_backtrack=2,
        calendar=calendar,
        reference_date=datetime.date(2025, 5, 2),
        include_reference=True,
    )
    crawler.get_twse_daily_stocks("shirong", DummyStockType, ["0"])

    assert crawler.iso_scheduled_times == ["2025-04-30", "2025-05-02"]
    assert [date for date, _ in crawler.records] == ["20250430", "20250502"]
    assert crawler.twse_no_data_dates == set()
