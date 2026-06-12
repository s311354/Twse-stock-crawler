from application.stock_service import StockAnalysisService
from interface.cli import build_request


class DummyStockType:
    value = (13,)


class FakeCrawler:
    def __init__(self, stocklistsize: int) -> None:
        self.stocklistsize = stocklistsize
        self.called: list[str] = []

    def get_date_times(self, start_date: int, backtrack_days: int, holidays: list[str]) -> None:
        self.called.append("get_date_times")

    def get_twse_daily_stocks(self, file_name: str, stocktype: DummyStockType, stocks: list[int]) -> None:
        self.called.append("get_twse_daily_stocks")

    def cal_max_profit(self) -> list[list[int]]:
        self.called.append("cal_max_profit")
        return [[1, 1, 1, 1, 1]]

    def record_analysis_dataset(self, file_name: str, maxprofits: list[list[int]]) -> None:
        self.called.append("record_analysis_dataset")


def test_interface_builds_stock_analysis_request(tmp_path) -> None:
    stocklist = tmp_path / "stocklist"
    holidays = tmp_path / "holidays"
    stocklist.write_text("0;1;2")
    holidays.write_text("20260101,20260216")

    request = build_request([
        "-o",
        "shirong",
        "-e",
        "3",
        "-b",
        "0",
        "-t",
        "ELEC",
        str(stocklist),
        str(holidays),
    ])

    assert request.stocklist == [0, 1, 2]
    assert request.holidays == ["20260101", "20260216"]
    assert request.output_file_names == "shirong"
    assert request.endbacktrack == 3


def test_application_service_orchestrates_injected_crawler() -> None:
    crawlers: list[FakeCrawler] = []

    def crawler_factory(stocklistsize: int) -> FakeCrawler:
        crawler = FakeCrawler(stocklistsize)
        crawlers.append(crawler)
        return crawler

    request = build_request(["-o", "shirong", "-e", "3", "-b", "0", "-t", "ELEC", "0", "20260101"])
    StockAnalysisService(crawler_factory=crawler_factory).run(request)

    assert crawlers[0].stocklistsize == 1
    assert crawlers[0].called == [
        "get_date_times",
        "get_twse_daily_stocks",
        "cal_max_profit",
        "record_analysis_dataset",
    ]
