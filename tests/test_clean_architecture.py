import datetime
import logging

import pytest
import pandas as pd

import interface.cli as cli
from application.contracts import AnalysisContext
from application.contracts import MarketStatus
from application.contracts import StockAnalysisRequest
from domain.fundamental_derivation import FundamentalBundle
from infrastructure.calendar.twse_calendar import TradingDayStatus
from infrastructure.calendar.twse_calendar import TwseCalendarUnavailableError
from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.crawler.twse_http_client import TwseHttpConfig
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.fundamentals.mops_repository import MopsStorageError
from infrastructure.bootstrap import build_stock_analysis_runtime
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from infrastructure.bootstrap import build_stock_analysis_service
from infrastructure.stock_analysis_adapters import CrawlerFundamentalAdapter
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
        now_provider=lambda: datetime.datetime(
            2026,
            7,
            20,
            14,
            0,
            tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
        ),
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


def test_market_data_no_data_dates_are_added_to_run_summary() -> None:
    class NoDataCrawler(FakeCrawler):
        def __init__(self, stocklistsize: int) -> None:
            super().__init__(stocklistsize)
            self.twse_no_data_dates = set()
            self.twse_closed_unscheduled_dates = set()
            self.missing_tracked_stock_first_dates = {}

        def get_twse_daily_stocks(self, file_name, stocktype, stocks) -> None:
            super().get_twse_daily_stocks(file_name, stocktype, stocks)
            self.twse_no_data_dates = {"2025-10-06", "2025-10-07"}
            self.twse_closed_unscheduled_dates = {"2026-07-10"}
            self.missing_tracked_stock_first_dates = {
                "2327": "2025-08-14",
                "3321": "2025-09-08",
            }

    runtime = build_stock_analysis_runtime(
        crawler_factory=NoDataCrawler,
        market_calendar=FakeCalendar(),
        today_provider=lambda: datetime.date(2026, 7, 20),
    )
    runtime.service.run(
        build_request(["-o", "shirong", "-e", "3", "-t", "ELEC", "0"])
    )

    summary = runtime.observer.snapshot()
    assert summary["no_data_date_count"] == 2
    assert summary["no_data_dates"] == ["2025-10-06", "2025-10-07"]
    assert summary["closed_unscheduled_date_count"] == 1
    assert summary["closed_unscheduled_dates"] == ["2026-07-10"]
    assert summary["missing_tracked_stock_count"] == 2
    assert summary["missing_tracked_stocks"] == {
        "2327": "2025-08-14",
        "3321": "2025-09-08",
    }


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


def test_production_composition_root_uses_pae_v2_pipeline() -> None:
    runtime = build_stock_analysis_runtime()

    strategy_adapter = runtime.service._evaluate_strategies._strategies

    assert isinstance(
        strategy_adapter._portfolio_allocator,
        PortfolioAllocationV2Pipeline,
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
            self.revision_times = []

        def prepare_daily_batch(self, observed_at):
            self.batch_times.append(observed_at)
            return 1

        def prepare_daily_revisions(self, observed_at):
            self.revision_times.append(observed_at)
            return 0

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
    assert fundamentals.revision_times == fundamentals.batch_times
    assert crawler.fundamentals[0] == "2330"
    assert crawler.fundamentals[3] == 2_000_000
    assert crawler.fundamentals[4] == "2026-07-20T13:30:00+08:00"
    assert crawler.called.index("set_low_entry_v4_fundamentals") < crawler.called.index("cal_max_profit")


def test_fundamental_adapter_uses_bulk_snapshot_and_falls_back_per_symbol() -> None:
    class BulkCrawler:
        stocknumbers = ["2330", "2317"]
        stocknames = ["台積電", "鴻海"]
        daily_dates = [["2026-08-03"], ["2026-08-03"]]

        def __init__(self) -> None:
            self.attached: list[str] = []

        def set_low_entry_v4_fundamentals(self, stock_no, **kwargs):
            self.attached.append(stock_no)

    class BulkService:
        def __init__(self) -> None:
            self.preload_calls = []
            self.load_calls = []

        def load_cached_bundles(self, symbols, **kwargs):
            self.preload_calls.append((symbols, kwargs))
            return {
                "2330": FundamentalBundle(
                    annual_fundamentals=pd.DataFrame({"fiscal_year": [2025]}),
                    snapshot={"remote_refresh_skipped": True},
                )
            }

        def load(self, symbol, **kwargs):
            self.load_calls.append((symbol, kwargs))
            return FundamentalBundle(
                annual_fundamentals=pd.DataFrame({"fiscal_year": [2025]}),
                snapshot={},
            )

    crawler = BulkCrawler()
    service = BulkService()
    adapter = CrawlerFundamentalAdapter(service)
    context = AnalysisContext(
        crawler,
        datetime.date(2026, 8, 3),
        MarketStatus.OPEN,
    )
    request = StockAnalysisRequest(
        stocklist=["2330", "2317"],
        stocktype="ELEC",
        output_file_names="shirong",
        endbacktrack=3,
    )

    adapter.refresh(
        context,
        request,
        analysis_time=datetime.datetime(
            2026,
            8,
            3,
            13,
            45,
            tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
        ),
    )

    assert service.preload_calls[0][0] == ["2330", "2317"]
    assert [call[0] for call in service.load_calls] == ["2317"]
    assert crawler.attached == ["2330", "2317"]


def test_fundamental_refresh_logs_one_individual_scope_fallback_summary(
    caplog,
) -> None:
    class ScopeCrawler:
        stocknumbers = ["7631", "6281"]
        stocknames = ["測試一", "測試二"]
        daily_dates = [["2026-08-03"], ["2026-08-03"]]

        def set_low_entry_v4_fundamentals(self, *args, **kwargs):
            pass

    class ScopeFallbackService:
        def __init__(self) -> None:
            self.symbols: set[str] = set()

        def reset_scope_fallback_diagnostics(self):
            self.symbols.clear()

        def consume_scope_fallback_symbols(self):
            result = tuple(sorted(self.symbols))
            self.symbols.clear()
            return result

        def load(self, symbol, **kwargs):
            self.symbols.add(symbol)
            return FundamentalBundle(
                annual_fundamentals=pd.DataFrame({"fiscal_year": [2025]}),
                snapshot={"available_at": "2026-08-03T13:45:00+08:00"},
            )

    class Observer:
        def __init__(self) -> None:
            self.records = []

        def record(self, event, **kwargs):
            self.records.append((event, kwargs))

    service = ScopeFallbackService()
    observer = Observer()
    adapter = CrawlerFundamentalAdapter(service, observer=observer)
    context = AnalysisContext(
        ScopeCrawler(),
        datetime.date(2026, 8, 3),
        MarketStatus.OPEN,
    )
    analysis_request = StockAnalysisRequest(
        stocklist=["7631", "6281"],
        stocktype="ELEC",
        output_file_names="shirong",
        endbacktrack=3,
    )

    with caplog.at_level("INFO"):
        adapter.refresh(
            context,
            analysis_request,
            analysis_time=datetime.datetime(
                2026,
                8,
                3,
                13,
                45,
                tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
            ),
        )

    summaries = [
        record
        for record in caplog.records
        if getattr(record, "status", None)
        == "INDIVIDUAL_SCOPE_FALLBACK_SUMMARY"
    ]
    assert len(summaries) == 1
    assert "stocks=2 symbols=6281,7631" in summaries[0].getMessage()
    assert service.symbols == set()
    assert any(
        event == "fundamental_individual_scope_fallback_summary"
        and values["fallback_symbols"] == ["6281", "7631"]
        for event, values in observer.records
    )


def test_fundamental_refresh_logs_one_access_blocked_cache_summary(
    caplog,
) -> None:
    class CachedCrawler:
        stocknumbers = ["6924", "6962", "7631"]
        stocknames = ["測試一", "測試二", "測試三"]
        daily_dates = [["2026-08-31"] for _ in stocknumbers]

        def set_low_entry_v4_fundamentals(self, *args, **kwargs):
            pass

    class AccessBlockedCacheService:
        def __init__(self) -> None:
            self.symbols: set[str] = set()

        def reset_access_blocked_diagnostics(self):
            self.symbols.clear()

        def consume_access_blocked_cache_fallback_symbols(self):
            result = tuple(sorted(self.symbols))
            self.symbols.clear()
            return result

        def load(self, symbol, **kwargs):
            self.symbols.add(symbol)
            return FundamentalBundle(
                annual_fundamentals=pd.DataFrame({"fiscal_year": [2025]}),
                snapshot={"available_at": "2026-08-31T13:45:00+08:00"},
            )

    service = AccessBlockedCacheService()
    adapter = CrawlerFundamentalAdapter(service)
    context = AnalysisContext(
        CachedCrawler(),
        datetime.date(2026, 8, 31),
        MarketStatus.OPEN,
    )
    request = StockAnalysisRequest(
        stocklist=["6924", "6962", "7631"],
        stocktype="ELEC",
        output_file_names="shirong",
        endbacktrack=3,
    )

    with caplog.at_level("INFO"):
        adapter.refresh(
            context,
            request,
            analysis_time=datetime.datetime(
                2026,
                8,
                31,
                13,
                45,
                tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
            ),
        )

    summaries = [
        record
        for record in caplog.records
        if getattr(record, "status", None)
        == "ACCESS_BLOCKED_CACHE_FALLBACK_SUMMARY"
    ]
    assert len(summaries) == 1
    assert summaries[0].levelno == logging.WARNING
    assert summaries[0].source == "MOPS_CACHE"
    assert summaries[0].cache_hit is True
    assert "stocks=3 symbols=6924,6962,7631" in summaries[0].getMessage()
    assert service.symbols == set()


def test_application_separates_preferred_security_from_mops_issuer() -> None:
    class PreferredCrawler(FakeCrawler):
        def __init__(self) -> None:
            super().__init__(1)
            self.stocknumbers = ["3036A"]
            self.stocknames = ["文曄甲特"]
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
            self.fundamentals = (
                stock_no,
                annual_fundamentals,
                fundamental_snapshot,
                account_equity,
                signal_date,
            )

    class RecordingFundamentalService:
        def __init__(self) -> None:
            self.symbols = []

        def load(self, symbol, *, signal_date, sector, history_years):
            self.symbols.append(symbol)
            return FundamentalBundle(
                annual_fundamentals=pd.DataFrame({"fiscal_year": [2025]}),
                snapshot={"available_at": signal_date},
                as_of=signal_date,
            )

    crawler = PreferredCrawler()
    fundamentals = RecordingFundamentalService()
    request = build_request([
        "-o", "shirong", "-e", "3", "-t", "ELEC", "3036A",
    ])

    build_stock_analysis_service(
        crawler_factory=lambda size: crawler,
        market_calendar=FakeCalendar(),
        today_provider=lambda: datetime.date(2026, 7, 20),
        fundamental_service=fundamentals,
    ).run(request)

    assert fundamentals.symbols == ["3036"]
    assert crawler.fundamentals[0] == "3036A"
    assert crawler.fundamentals[2]["market_symbol"] == "3036A"
    assert crawler.fundamentals[2]["issuer_symbol"] == "3036"
    assert crawler.fundamentals[2]["security_type"] == "PREFERRED"


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
    assert "LES v5 formal BUY disabled" in caplog.text
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


def test_cli_renders_typed_http_failure(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("TWSE_RUN_LOCK_PATH", str(tmp_path / "analysis.lock"))

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
