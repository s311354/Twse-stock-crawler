import datetime

import pytest
import pandas as pd

import interface.cli as cli
from domain.fundamental_derivation import FundamentalBundle
from infrastructure.calendar.twse_calendar import TradingDayStatus
from infrastructure.calendar.twse_calendar import TwseCalendarUnavailableError
from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.crawler.twse_http_client import TwseHttpConfig
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.fundamentals.mops_repository import MopsStorageError
from infrastructure.bootstrap import build_stock_analysis_runtime
from infrastructure.bootstrap import build_stock_analysis_service
from interface.cli import build_request


class DummyStockType:
    value = (13,)


class FakeCrawler:
    def __init__(self, stocklistsize: int) -> None:
        self.stocklistsize = stocklistsize
        self.called: list[str] = []
        self.reference_date: datetime.date | None = None
        self.include_reference = False

    def get_date_times(
        self,
        begin_backtrack: int,
        end_backtrack: int,
        calendar: object,
        reference_date: datetime.date | None = None,
        include_reference: bool = False,
    ) -> None:
        self.reference_date = reference_date
        self.include_reference = include_reference
        self.called.append("get_date_times")

    def get_twse_daily_stocks(self, file_name: str, stocktype: DummyStockType, stocks: list[str]) -> None:
        self.called.append("get_twse_daily_stocks")

    def cal_max_profit(self) -> list[list[int]]:
        self.called.append("cal_max_profit")
        return [[1, 1, 1, 1, 1]]

    def record_analysis_dataset(self, file_name: str, maxprofits: list[list[int]]) -> None:
        self.called.append("record_analysis_dataset")


class FakeCalendar:
    def __init__(self, is_open: bool = True) -> None:
        self.is_open = is_open
        self.requested_dates: list[datetime.date] = []

    def is_trading_day(self, target_date: datetime.date) -> bool:
        self.requested_dates.append(target_date)
        return self.is_open

    def trading_day_status(self, target_date: datetime.date) -> TradingDayStatus:
        return TradingDayStatus.OPEN if self.is_open else TradingDayStatus.CLOSED


def test_interface_builds_stock_analysis_request(tmp_path) -> None:
    stocklist = tmp_path / "stocklist"
    stocklist.write_text("0;1;2")

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
    ])

    assert request.stocklist == ["0", "1", "2"]
    assert request.output_file_names == "shirong"
    assert request.endbacktrack == 3


def test_application_service_orchestrates_injected_crawler() -> None:
    crawlers: list[FakeCrawler] = []

    def crawler_factory(stocklistsize: int) -> FakeCrawler:
        crawler = FakeCrawler(stocklistsize)
        crawlers.append(crawler)
        return crawler

    calendar = FakeCalendar()
    today = datetime.date(2026, 7, 20)
    request = build_request(["-o", "shirong", "-e", "3", "-b", "0", "-t", "ELEC", "0"])
    build_stock_analysis_service(
        crawler_factory=crawler_factory,
        market_calendar=calendar,
        today_provider=lambda: today,
    ).run(request)

    assert crawlers[0].stocklistsize == 1
    assert crawlers[0].reference_date == today
    assert crawlers[0].include_reference is True
    assert calendar.requested_dates == [today]
    assert crawlers[0].called == [
        "get_date_times",
        "get_twse_daily_stocks",
        "cal_max_profit",
        "record_analysis_dataset",
    ]


def test_application_service_runs_backtrack_analysis_when_market_is_closed(
    capsys,
) -> None:
    crawler = FakeCrawler(1)
    request = build_request(["-o", "shirong", "-e", "3", "-t", "ELEC", "2330"])
    result = build_stock_analysis_service(
        crawler_factory=lambda size: crawler,
        market_calendar=FakeCalendar(is_open=False),
        today_provider=lambda: datetime.date(2026, 2, 16),
    ).run(request)

    assert result is True
    assert crawler.called == [
        "get_date_times",
        "get_twse_daily_stocks",
        "cal_max_profit",
        "record_analysis_dataset",
    ]
    assert "TWSE market status for 2026-02-16: CLOSED" in capsys.readouterr().out


def test_application_service_does_not_treat_calendar_failure_as_closed() -> None:
    class FailingCalendar:
        def is_trading_day(self, target_date: datetime.date) -> bool:
            raise TwseCalendarUnavailableError("network unavailable")

    request = build_request(["-o", "shirong", "-e", "3", "-t", "ELEC", "2330"])

    with pytest.raises(TwseCalendarUnavailableError, match="network unavailable"):
        build_stock_analysis_service(
            crawler_factory=lambda size: FakeCrawler(size),
            market_calendar=FailingCalendar(),
            today_provider=lambda: datetime.date(2026, 7, 20),
        ).run(request)


def test_default_calendar_and_crawler_share_one_http_client() -> None:
    http_client = TwseHttpClient(
        config=TwseHttpConfig(min_request_interval_seconds=0),
    )
    runtime = build_stock_analysis_runtime(http_client=http_client)

    crawler = runtime.crawler_factory(1)

    assert runtime.market_calendar._http_client is http_client
    assert crawler.twse_client._crawler.http_client is http_client
    assert (
        runtime.fundamental_service.filing_catalog.http_client
        is http_client
    )


def test_application_loads_mops_context_before_analysis() -> None:
    class MopsAwareCrawler(FakeCrawler):
        def __init__(self, stocklistsize: int) -> None:
            super().__init__(stocklistsize)
            self.stocknumbers = ["2330"]
            self.daily_dates = [["2026-07-20"]]
            self.fundamentals = None

        def set_low_entry_v4_fundamentals(
            self,
            stock_no,
            annual_fundamentals,
            fundamental_snapshot,
            account_equity,
            signal_date=None,
        ):
            self.called.append("set_low_entry_v4_fundamentals")
            self.fundamentals = (
                stock_no,
                annual_fundamentals,
                fundamental_snapshot,
                account_equity,
                signal_date,
            )

    class FakeFundamentalService:
        def __init__(self) -> None:
            self.calls = []
            self.batch_times = []

        def prepare_daily_batch(self, observed_at):
            self.batch_times.append(observed_at)
            return 1

        def load(self, symbol, *, signal_date, sector, history_years):
            self.calls.append((symbol, signal_date, sector, history_years))
            return FundamentalBundle(
                annual_fundamentals=pd.DataFrame({"fiscal_year": [2025]}),
                snapshot={"available_at": "2026-02-10T18:16:19+08:00"},
            )

    crawler = MopsAwareCrawler(1)
    fundamentals = FakeFundamentalService()
    request = build_request([
        "-o", "shirong", "-e", "3", "-t", "ELEC",
        "--mops-history-years", "4", "--account-equity", "2000000", "2330",
    ])

    build_stock_analysis_service(
        crawler_factory=lambda size: crawler,
        market_calendar=FakeCalendar(),
        today_provider=lambda: datetime.date(2026, 7, 20),
        fundamental_service=fundamentals,
    ).run(request)

    assert fundamentals.calls == [("2330", "2026-07-20T13:30:00+08:00", "ELEC", 4)]
    assert fundamentals.batch_times == [
        datetime.datetime(
            2026,
            7,
            20,
            13,
            30,
            tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
        )
    ]
    assert crawler.fundamentals[0] == "2330"
    assert crawler.fundamentals[3] == 2_000_000
    assert crawler.fundamentals[4] == "2026-07-20T13:30:00+08:00"
    assert crawler.called.index("set_low_entry_v4_fundamentals") < crawler.called.index("cal_max_profit")


def test_application_continues_when_mops_cache_is_unavailable(caplog) -> None:
    class StorageAwareCrawler(FakeCrawler):
        def __init__(self) -> None:
            super().__init__(1)
            self.stocknumbers = ["2330"]
            self.daily_dates = [["2026-07-20"]]

        def set_low_entry_v4_fundamentals(self, *args, **kwargs) -> None:
            raise AssertionError("storage failure must not inject incomplete fundamentals")

    class FailingFundamentalService:
        def load(self, *args, **kwargs):
            raise MopsStorageError("unable to open cache")

    crawler = StorageAwareCrawler()
    request = build_request(["-o", "shirong", "-e", "3", "-t", "ELEC", "2330"])

    result = build_stock_analysis_service(
        crawler_factory=lambda size: crawler,
        market_calendar=FakeCalendar(),
        today_provider=lambda: datetime.date(2026, 7, 20),
        fundamental_service=FailingFundamentalService(),
    ).run(request)

    assert result is True
    assert "LES v4 remains WATCH_ONLY" in caplog.text
    assert "record_analysis_dataset" in crawler.called


def test_application_opens_mops_circuit_after_consecutive_network_failures(caplog) -> None:
    class NetworkAwareCrawler(FakeCrawler):
        def __init__(self) -> None:
            super().__init__(3)
            self.stocknumbers = ["2330", "2317", "2308"]
            self.daily_dates = [["2026-07-20"] for _ in self.stocknumbers]

        def set_low_entry_v4_fundamentals(self, *args, **kwargs) -> None:
            raise AssertionError("network failure must not inject fundamentals")

    class OfflineFundamentalService:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def load(self, symbol, **kwargs):
            self.calls.append(symbol)
            raise TwseNetworkError("DNS unavailable")

    crawler = NetworkAwareCrawler()
    fundamentals = OfflineFundamentalService()
    request = build_request([
        "-o", "shirong", "-e", "3", "-t", "ELEC", "2330", "2317", "2308",
    ])

    build_stock_analysis_service(
        crawler_factory=lambda size: crawler,
        market_calendar=FakeCalendar(),
        today_provider=lambda: datetime.date(2026, 7, 20),
        fundamental_service=fundamentals,
    ).run(request)

    assert fundamentals.calls == ["2330", "2317"]
    assert "MOPS circuit open after 2 consecutive network failures" in caplog.text
    assert "record_analysis_dataset" in crawler.called


def test_application_marks_remaining_symbols_when_mops_schema_circuit_opens(caplog) -> None:
    class SchemaAwareCrawler(FakeCrawler):
        def __init__(self) -> None:
            super().__init__(3)
            self.stocknumbers = ["2330", "2317", "2308"]
            self.daily_dates = [["2026-07-20"] for _ in self.stocknumbers]
            self.unavailable: dict[str, str] = {}

        def set_low_entry_v4_fundamentals(self, *args, **kwargs) -> None:
            raise AssertionError("schema failure must not inject fundamentals")

        def set_low_entry_v4_fundamentals_unavailable(
            self,
            stock_no,
            *,
            status,
            **kwargs,
        ) -> None:
            self.unavailable[stock_no] = status

    class BrokenSchemaFundamentalService:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def load(self, symbol, **kwargs):
            self.calls.append(symbol)
            raise TwseSchemaChangedError("MOPS guard page")

    crawler = SchemaAwareCrawler()
    fundamentals = BrokenSchemaFundamentalService()
    request = build_request([
        "-o", "shirong", "-e", "3", "-t", "ELEC", "2330", "2317", "2308",
    ])

    build_stock_analysis_service(
        crawler_factory=lambda size: crawler,
        market_calendar=FakeCalendar(),
        today_provider=lambda: datetime.date(2026, 7, 20),
        fundamental_service=fundamentals,
    ).run(request)

    assert fundamentals.calls == ["2330", "2317"]
    assert crawler.unavailable == {
        "2330": "SCHEMA_CHANGED",
        "2317": "SCHEMA_CHANGED",
        "2308": "CIRCUIT_OPEN",
    }
    assert "MOPS schema circuit open after 2 consecutive failures" in caplog.text


def test_cli_renders_typed_http_failure(monkeypatch, capsys) -> None:
    class FailingService:
        def run(self, request) -> None:
            raise TwseNetworkError("offline")

    monkeypatch.setattr(
        cli,
        "build_stock_analysis_service",
        lambda **kwargs: FailingService(),
    )

    exit_code = cli.main(["-e", "3", "-t", "ELEC", "2330"])

    assert exit_code == 3
    assert "TWSE request status: NETWORK_ERROR (offline)" in capsys.readouterr().err
