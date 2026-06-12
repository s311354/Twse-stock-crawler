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


def test_missing_ohlc_does_not_reuse_previous_daily_row() -> None:
    rows_by_date = {
        "20260525": [["3701", "大眾控", "", "", "", "58.60", "58.70", "56.30", "57.10"]],
        "20260526": [["3701", "大眾控", "", "", "", "--", "--", "--", "--"]],
    }

    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.iso_scheduled_times = ["2026-05-25", "2026-05-26"]
    crawler.get_twse_daily_stocks("shirong", DummyStockType, [0])

    assert crawler.records[0] == ("20260525", [["3701", "大眾控", "58.60", "58.70", "56.30", "57.10"]])
    assert crawler.records[1] == ("20260526", [["3701", "大眾控", "--", "--", "--", "--"]])
    assert crawler.daily_stocks == [[58600.0]]


def test_invalid_twse_row_shape_is_skipped_gracefully() -> None:
    crawler = stockanalysis.TwseCrawker(1)

    assert crawler.parse_ohlc([["3701", "大眾控"]], 0, "20260526") is None
    assert crawler.parse_ohlc([], 0, "20260526") is None


def test_empty_twse_date_is_skipped_without_recording() -> None:
    rows_by_date = {
        "20260515": [],
        "20260518": [["3701", "大眾控", "", "", "", "58.60", "58.70", "56.30", "57.10"]],
    }

    crawler = CapturingTwseCrawler(1, rows_by_date)
    crawler.iso_scheduled_times = ["2026-05-15", "2026-05-18"]
    crawler.get_twse_daily_stocks("shirong", DummyStockType, [0])

    assert crawler.records == [("20260518", [["3701", "大眾控", "58.60", "58.70", "56.30", "57.10"]])]
    assert crawler.iso_scheduled_times == ["2026-05-18"]
    assert crawler.daily_stocks == [[58600.0]]
