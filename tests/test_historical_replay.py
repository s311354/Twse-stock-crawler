"""Strict cutoff replay: no future evidence, live persistence, email or orders."""

import datetime as dt
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from application.contracts import AnalysisContext, MarketStatus, StockAnalysisRequest
from application.historical_replay import replay_close_cutoff, completed_session
from application.use_cases import FetchMarketData, RefreshFundamentals
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from domain.models import Stocktype
from domain.strategies.low_entry_score_v6 import assess_execution
from domain.strategies.low_entry_v6_report import report_failure
from infrastructure.runtime.market_data_adapter import CrawlerMarketDataAdapter
from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator
from infrastructure.runtime.strategy_adapter import CrawlerStrategyAdapter
from infrastructure.runtime.output_adapters import CrawlerReportAdapter
from infrastructure.storage.csv_repository import CsvRepository
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.report.html_renderer import RendererFactory
from aistock_twse_integration.les_v6_execution import load_reports
from interface.cli import build_run_configuration, build_preflight_summary
from tests.fundamentals.test_filing_refresh_planner import _candidate, _fact, _FakeMopsClient, _fundamental_service
from tests.fundamentals.test_fundamental_pipeline_components import _snapshot
from tests.strategies.test_daily_v6_pipeline import daily_market, build


def request(**kwargs):
    return StockAnalysisRequest(stocklist=["2330"], stocktype=Stocktype.ELEC,
                                output_file_names="shirong", endbacktrack=792,
                                as_of=replay_close_cutoff("2026-09-16"), **kwargs)


def close_replay_market():
    """Synthetic close-time evidence; never retimestamp real archived evidence."""
    market = daily_market()
    cutoff = replay_close_cutoff("2026-09-11").isoformat()
    evidence = dict(market.trading_restrictions["2330"], available_at=cutoff)
    return replace(market, replay_as_of=cutoff, signal_dates={"2330": cutoff},
                   trading_restrictions={"2330": evidence})


class WeekdayCalendar:
    def is_trading_day(self, day):
        return day.weekday() < 5


@pytest.mark.parametrize("value,expected", [
    ("2026-09-16", "2026-09-16"),
    ("2026-09-16T09:00:00+08:00", "2026-09-15"),
    ("2026-09-16T05:29:59Z", "2026-09-15"),
    ("2026-09-16T05:30:00Z", "2026-09-16"),
    ("2026-09-20", "2026-09-18"),
])
def test_completed_daily_session_boundary(value, expected):
    # The internal visibility helper retains timestamp precision, independently
    # of the date-only CLI contract.
    cutoff = pd.Timestamp(value).to_pydatetime() if "T" in value else replay_close_cutoff(value)
    assert completed_session(cutoff, WeekdayCalendar()).isoformat() == expected


@pytest.mark.parametrize("value", ["2026-09-16T09:00:00", "2026-09-16T09:00:00+08:00",
                                 "2026-09-16T13:30:00+08:00", "2026-09-16T05:30:00Z",
                                 "yesterday", "2026-02-30", "20260916", "2026-W38-3",
                                 "2026-9-16", " 2026-09-16", "2026-09-16 "])
def test_cutoff_rejects_ambiguous_or_invalid_timestamp(value):
    with pytest.raises(ValueError, match="Replay end date"):
        replay_close_cutoff(value)


def replay_args(tmp_path, content=""):
    holdings = tmp_path / "historical.txt"
    holdings.write_text(content)
    return ["2330", "-b", "2024-10-25", "-e", "2026-09-16",
            "--pae-holdings-file", str(holdings), "--pae-current-cash", "900000",
            "--account-equity", "1000000"]


def test_cli_preserves_day_offsets_and_requires_historical_portfolio(tmp_path, monkeypatch):
    monkeypatch.setenv("TWSE_PAE_HOLDINGS_JSON", '[{"symbol":"3008","shares":10,"entry_date":"2026-09-25"}]')
    monkeypatch.setenv("TWSE_PAE_CURRENT_CASH", "9999999")
    req, settings = build_run_configuration(replay_args(tmp_path))
    assert (req.beginbacktrack, req.endbacktrack) == (0, 691)
    assert settings.pae_positions == ()
    assert req.as_of.isoformat() == "2026-09-16T13:30:00+08:00"
    assert req.report_file_prefix == "SHIRONG_replay_20260916T133000+0800"
    assert build_preflight_summary(req, settings)["pae_persistence"] == "ISOLATED_NO_HISTORY"
    with pytest.raises(SystemExit):
        build_run_configuration(["2330", "-b", "2024-10-25", "-e", "2026-09-16"])
    for content in ("2330:100", "2330:100:2026-09-23"):
        with pytest.raises(SystemExit):
            build_run_configuration(replay_args(tmp_path, content))
    with pytest.raises(SystemExit):
        build_run_configuration(replay_args(tmp_path) + ["--mail"])
    monkeypatch.delenv("TWSE_PAE_CURRENT_CASH")
    with pytest.raises(SystemExit):
        build_run_configuration(["2330", "-b", "9", "-e", "792"])
    live, _ = build_run_configuration(["2330", "-b", "0", "-e", "792"])
    assert live.as_of is None and live.beginbacktrack == 0
    assert live.report_file_prefix == live.output_file_names


@pytest.mark.parametrize("cutoff,include", [("2026-09-16", True), ("2026-09-20", False)])
def test_use_cases_do_not_read_wall_clock_for_replay(cutoff, include):
    req = replace(request(), as_of=replay_close_cutoff(cutoff))
    clock = Mock(side_effect=AssertionError("live clock used"))
    port = Mock()
    context = FetchMarketData(port, WeekdayCalendar(), clock, clock).execute(req)
    assert port.fetch.call_args.kwargs["market_date"] == dt.date.fromisoformat(cutoff)
    assert port.fetch.call_args.kwargs["include_reference"] is include
    fundamental = Mock()
    RefreshFundamentals(fundamental, clock).execute(context, req)
    assert fundamental.refresh.call_args.kwargs["analysis_time"] == req.as_of
    clock.assert_not_called()


@pytest.mark.parametrize("cutoff", ["2026-09-16T09:00:00+08:00", "2026-09-16T13:29:59+08:00",
                                  "2026-09-16T13:30:00.000001+08:00", "2026-09-16T15:00:00+08:00"])
def test_programmatic_replay_rejects_non_close_before_calendar_or_fetch(cutoff):
    req = replace(request(), as_of=dt.datetime.fromisoformat(cutoff))
    calendar, port, clock = Mock(), Mock(), Mock()
    with pytest.raises(ValueError, match="REPLAY_CLOSE_ONLY"):
        FetchMarketData(port, calendar, clock, clock).execute(req)
    calendar.is_trading_day.assert_not_called()
    port.fetch.assert_not_called()
    clock.assert_not_called()


def test_programmatic_replay_accepts_equivalent_utc_close():
    req = replace(request(), as_of=dt.datetime.fromisoformat("2026-09-16T05:30:00+00:00"))
    port, clock = Mock(), Mock()
    FetchMarketData(port, WeekdayCalendar(), clock, clock).execute(req)
    assert port.fetch.call_args.kwargs["include_reference"] is True
    assert port.fetch.call_args.kwargs["market_date"] == dt.date(2026, 9, 16)
    clock.assert_not_called()


def test_market_adapter_freezes_clock_even_without_mops_and_checks_range(tmp_path):
    crawler = SimpleNamespace(iso_scheduled_times=["2026-09-16"], stocknumbers=["2330"],
                              get_date_times=Mock(), get_twse_daily_stocks=Mock())
    adapter = CrawlerMarketDataAdapter(lambda count: crawler)
    req = request(mops_enabled=False)
    context = adapter.fetch(req, calendar=WeekdayCalendar(), market_date=req.as_of.date(),
                            is_trading_day=True, include_reference=True)
    assert context.as_of == req.as_of
    assert crawler.low_entry_v4_signal_dates_by_stock == {"2330": req.as_of.isoformat()}
    assert crawler.trading_restrictions == {}
    crawler.iso_scheduled_times = ["2026-09-15"]
    with pytest.raises(ValueError, match="REPLAY_RANGE_MISSES_CUTOFF_SESSION"):
        adapter.fetch(req, calendar=WeekdayCalendar(), market_date=req.as_of.date(),
                      is_trading_day=True, include_reference=True)


def test_historical_mops_excludes_sept23_revision_and_never_activates_snapshot(tmp_path, monkeypatch):
    repo = MopsRepository(tmp_path / "mops")
    for observed, value in (("2026-09-13T01:00:00Z", 100.), ("2026-09-23T01:00:00Z", 999.)):
        candidate = replace(_candidate(2025, 4), available_at=observed, period_end="2025-12-31")
        repo.save_revision(candidate, str(value).encode(), [_fact(value)],
                           available_at=observed, fetched_at=observed)
    client = _FakeMopsClient(repo, [], [])
    service = _fundamental_service(client, repo, include_current_observations=True,
                                  now_provider=lambda: dt.datetime(2026, 9, 26, tzinfo=dt.timezone.utc))
    for method in ("mark_active_fundamental_bundle", "save_fundamental_bundle", "save_financial_batch_snapshot"):
        monkeypatch.setattr(repo, method, Mock(side_effect=AssertionError("live cache pointer mutation")))
    service._resolve_sector = Mock(side_effect=AssertionError("current company master used"))
    bundle = service.load_historical("2330", as_of=replay_close_cutoff("2026-09-16"), sector="ELEC")
    records = bundle.snapshot["les_v6_filings"]
    assert len(records) == 1
    assert pd.Timestamp(records[0]["available_at"]) == pd.Timestamp("2026-09-13T01:00:00Z")
    assert bundle.annual_fundamentals.iloc[0]["revenue"] == 100.
    assert pd.Timestamp(bundle.as_of) == pd.Timestamp("2026-09-16T13:30:00+08:00")
    assert client.discovery_calls == 0 and client.downloaded_periods == []
    late = service.load_historical("2330", as_of=replay_close_cutoff("2026-09-24"), sector="ELEC")
    assert late.annual_fundamentals.iloc[0]["revenue"] == 999.
    empty = service.load_historical("6834", as_of=replay_close_cutoff("2026-09-16"), sector="ELEC")
    assert empty.snapshot["les_v6_filings"] == []


def test_date_only_replay_keeps_same_day_filing_timestamp_precision(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    for observed, value in (("2026-09-16T05:30:00Z", 100.), ("2026-09-16T05:30:01Z", 999.)):
        candidate = replace(_candidate(2025, 4), available_at=observed, period_end="2025-12-31")
        repo.save_revision(candidate, str(value).encode(), [_fact(value)],
                           available_at=observed, fetched_at=observed)
    service = _fundamental_service(_FakeMopsClient(repo, [], []), repo)
    bundle = service.load_historical("2330", as_of=replay_close_cutoff("2026-09-16"), sector="ELEC")
    assert bundle.annual_fundamentals.iloc[0]["revenue"] == 100.
    assert len(bundle.snapshot["les_v6_filings"]) == 1
    assert pd.Timestamp(bundle.snapshot["les_v6_filings"][0]["available_at"]) == pd.Timestamp("2026-09-16T05:30:00Z")


def test_replay_refresh_does_not_call_live_batch_catalog_or_restrictions(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    service = _fundamental_service(_FakeMopsClient(repo, [], []), repo)
    service.financial_batch_client = Mock()
    service.begin_run = Mock(side_effect=AssertionError("live refresh used"))
    crawler = SimpleNamespace(stocknumbers=["6834"], stocknames=["test"],
                              set_low_entry_v4_fundamentals=Mock(), trading_restrictions={})
    req = request()
    context = AnalysisContext(crawler, req.as_of.date(), MarketStatus.OPEN, req.as_of)
    FundamentalRefreshCoordinator(service).refresh(context, req, analysis_time=req.as_of)
    service.financial_batch_client.load.assert_not_called()
    assert crawler.trading_restrictions == {}
    assert crawler.set_low_entry_v4_fundamentals.call_args.kwargs["signal_date"] == req.as_of.isoformat()


def test_historical_twse_batch_filters_future_observations_and_tpex_excludes_twse(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    older = _snapshot(fiscal_year=2026, quarter=2, observed_at="2026-09-13T01:00:00Z")
    future = replace(older, observed_at="2026-09-23T01:00:00Z", revenue_ytd=999999.)
    repo.save_financial_batch_snapshot(older)
    repo.save_financial_batch_snapshot(future)
    service = _fundamental_service(_FakeMopsClient(repo, [], []), repo)
    # A live in-memory batch must be ignored as well as future SQLite versions.
    service._daily_financials = {older.symbol: future}
    bundle = service.load_historical(older.symbol, as_of=replay_close_cutoff("2026-09-16"), sector="ELEC")
    assert bundle.snapshot["twse_revenue_ytd"] == older.revenue_ytd
    assert all(pd.Timestamp(row["available_at"]) <= pd.Timestamp("2026-09-16T13:30:00+08:00")
               for row in bundle.snapshot["les_v6_filings"])
    tpex = service.load_historical(older.symbol, as_of=replay_close_cutoff("2026-09-16"), sector="24", market="TPEX")
    assert tpex.snapshot["les_v6_filings"] == []


@pytest.mark.usefixtures("active_industry_profiles")
def test_replay_les_erem_pae_report_isolated_and_not_live_executable(tmp_path):
    market = close_replay_market()
    cutoff = market.signal_dates["2330"]
    market = replace(market, replay_as_of=cutoff)
    builder, decisions = build(market)
    assert decisions[0].value("低點決策") == "BUY_SMALL"
    assert decisions[0].value("EREM 分數") is not None
    assert decisions[0].evaluated_at == pd.Timestamp(cutoff).to_pydatetime()
    persistence = Mock()
    allocator = PortfolioAllocationV2Pipeline(persistence=persistence)
    crawler = SimpleNamespace(cal_max_profit=Mock(return_value=[]), build_strategy_decisions=Mock(return_value=decisions))
    context = AnalysisContext(crawler, dt.date(2026, 9, 11), MarketStatus.OPEN, dt.datetime.fromisoformat(cutoff))
    evaluation = CrawlerStrategyAdapter(portfolio_allocator=allocator, portfolio_current_cash=1_000_000).evaluate(context)
    persistence.load_recent_v2.assert_not_called()
    persistence.record_session_v2.assert_not_called()
    assert evaluation.decisions[0].value("PAE Status") is not None
    report = json.loads(evaluation.decisions[0].value("LES v6 Report"))
    assert report_failure(report, allow_replay=True) is None
    assert report_failure(report) == "HISTORICAL_REPLAY_NOT_EXECUTABLE"
    assert assess_execution(report, {}, now=cutoff, trading_sessions=[], portfolio={})["reason"] == "HISTORICAL_REPLAY_NOT_EXECUTABLE"
    prefix = replace(request(), as_of=dt.datetime.fromisoformat(cutoff)).report_file_prefix
    CsvRepository(tmp_path).write_analysis_dataset(prefix, builder.project_decisions(evaluation.decisions))
    assert not (tmp_path / "les_v6_leadership_history.sqlite3").exists()
    assert not (tmp_path / "shirong_les_v6_reports.json").exists()
    with pytest.raises(ValueError, match="HISTORICAL_REPLAY_NOT_EXECUTABLE"):
        load_reports(str(tmp_path / f"{prefix}_les_v6_reports.json"), "2026-09-11")
    rendered = RendererFactory.get_renderer("daily_strategy_email").render(evaluation.decisions)
    assert "HISTORICAL_REPLAY" in rendered and "歷史研究，不可下單" in rendered


def test_report_adapter_uses_separate_prefix():
    crawler = SimpleNamespace(record_strategy_decisions=Mock())
    req = request()
    CrawlerReportAdapter().generate(AnalysisContext(crawler, req.as_of.date(), MarketStatus.OPEN, req.as_of),
                                    req, SimpleNamespace(decisions=("test",)))
    assert crawler.record_strategy_decisions.call_args.kwargs["file_name"] == req.report_file_prefix


@pytest.mark.parametrize("source", ["missing", "future"])
def test_historical_restrictions_never_borrow_current_clear_list(source):
    market = daily_market()
    evidence = json.loads(json.dumps(market.trading_restrictions))
    if source == "missing":
        evidence = {}
    else:
        evidence["2330"]["available_at"] = "2026-09-26T10:00:00+08:00"
        evidence["2330"]["session_date"] = "2026-09-26"
    _, decisions = build(replace(market, replay_as_of=market.signal_dates["2330"], trading_restrictions=evidence))
    assert decisions[0].value("低點決策") not in {"BUY", "BUY_SMALL"}
    assert decisions[0].value("建議股數") == 0


def test_same_cutoff_ignores_process_clock_and_future_portfolio_entries(tmp_path):
    args = replay_args(tmp_path, "2330:100:2026-09-15:80:BUY")
    req, settings = build_run_configuration(args)
    assert settings.pae_positions[0].entry_date == dt.date(2026, 9, 15)
    args[args.index("-e") + 1] = "2026-09-14"
    with pytest.raises(SystemExit):
        build_run_configuration(args)
    args[args.index("-e") + 1] = "2026-09-15"
    _, settings = build_run_configuration(args)
    assert settings.pae_positions[0].entry_date == dt.date(2026, 9, 15)
