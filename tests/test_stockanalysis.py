import stockanalysis


class DummyStockType:
    value = (13,)


class FakeTwseClient:
    def __init__(self, rows_by_date: dict[str, stockanalysis.StockRows]) -> None:
        self.rows_by_date = rows_by_date

    def get_daily_stock_rows(self, date_time: str, stocktype: int) -> stockanalysis.StockRows:
        return self.rows_by_date[date_time]


class CapturingTwseCrawler(stockanalysis.TwseCrawker):
    def __init__(self, stocklistsize: int, rows_by_date: dict[str, stockanalysis.StockRows] | None = None) -> None:
        super().__init__(stocklistsize, twse_client=FakeTwseClient(rows_by_date or {}))
        self.records: list[tuple[str, stockanalysis.StockRows]] = []

    def record(self, file_name: str, scheduled_time: str, row_data: stockanalysis.StockRows) -> None:
        self.records.append((scheduled_time, row_data))


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
    assert crawler.daily_pe_ratios == [[12.3]]


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


def test_analysis_dataset_includes_low_entry_score_v3_columns() -> None:
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

    dataset = crawler.build_analysis_dataset([[1, 1, 1, 1, 0.01]])

    assert "EMA Alignment" in dataset.columns
    assert "Volume Above MA20" in dataset.columns
    assert "OBV Rising" in dataset.columns
    assert dataset.loc[0, "EMA Alignment"] == 1
