"""TPEx -> isolated SQLite history -> existing strategy history -> CSV."""
import csv
import datetime as dt
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from application.contracts import AnalysisContext, MarketStatus, StockAnalysisRequest
from application.market_history_cache import CachedMarketDay
from domain.fundamental_derivation import FundamentalBundle
from domain.models import Stocktype
from domain.model_contracts import FieldSource
from domain.tpex_market import TPEX_FIELDS
from infrastructure.crawler.tpex_client import TpexClient, parse_tpex_quotes, to_legacy_row
from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator
from infrastructure.runtime.market_data_adapter import CrawlerMarketDataAdapter
from infrastructure.storage.csv_repository import TpexCsvRepository
from infrastructure.storage.market_history_cache import SQLiteMarketHistoryCache
from interface.cli import build_run_configuration, build_preflight_summary
from stockanalysis import TwseCrawker
from tests.test_tpex_client import payload
from tests.test_twse_http_client import FakeResponse, FakeSession, config


def crawler(tmp_path, session):
    benchmark = Mock()
    benchmark.get_daily_taiex_close.return_value = 20_000.
    client = TpexClient(TwseHttpClient(config=config(), session=session), benchmark=benchmark)
    result = TwseCrawker(1, twse_client=client, market_source="TPEX",
                         market_history_cache=SQLiteMarketHistoryCache(tmp_path / "market.sqlite3"),
                         csv_repository=TpexCsvRepository(tmp_path))
    result.analysis_universe = "SEMI"
    return result, benchmark


def test_symbol_identity_cache_namespace_new_listings_and_raw_export(tmp_path):
    first = payload("20260923", ("3105", "5347"))
    second = payload("20260924", ("5347", "3105", "6423"))
    second["tables"][0]["data"][1][2] = "503.00"
    session = FakeSession([FakeResponse(first), FakeResponse(second)])
    source, benchmark = crawler(tmp_path, session)
    # Old TWSE sector 24 must not satisfy a TPEx cache lookup.
    source.market_history_cache.upsert_day(CachedMarketDay("20260923", [["2330"]], 19000.), market_type="24")
    source.iso_scheduled_times = ["2026-09-23", "2026-09-24"]
    source.explicit_stock_numbers = {"2330"}
    source.get_twse_daily_stocks("test_tpex_semi", Stocktype.SEMI, ["ALL", "2330"])
    assert source.stocknumbers == ["3105", "5347", "6423"]
    assert source.daily_closes == [[502_000., 503_000.], [502_000., 502_000.], [502_000.]]
    assert source.daily_dates[2] == ["2026-09-24"]
    assert len(session.calls) == benchmark.get_daily_taiex_close.call_count == 2
    assert all(call["params"]["type"] == "24" for call in session.calls)
    history = source.build_strategy_history(0)
    assert history.attrs["market_source"] == "TPEX"
    assert history.attrs["les_v6_daily_identity"]["market"] == "TPEX"
    assert history["Volume"].iloc[-1] == 21_240_000
    assert history["PE"].isna().all()
    assert history["TAIEX Close"].tolist() == [20_000., 20_000.]
    assert source.market_breadth_snapshots_by_date["2026-09-24"]["universe"] == "TPEX_SEMI"
    with (tmp_path / "test_tpex_semi_20260924.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["日期", *TPEX_FIELDS]
    assert rows[1][1] == "3105" and rows[1][8] == "21,240,000"
    assert rows[1][12] == "33" and rows[1][15] == "423,940,384"

    cached, no_benchmark = crawler(tmp_path, FakeSession([]))
    cached.iso_scheduled_times = ["2026-09-23", "2026-09-24"]
    cached.get_twse_daily_stocks("test_tpex_semi", Stocktype.SEMI, ["ALL"])
    assert cached.stocknumbers == source.stocknumbers
    assert cached.daily_closes == source.daily_closes
    no_benchmark.get_daily_taiex_close.assert_not_called()
    # Existing downstream analysis runs on the normalized data without a new strategy.
    decisions = cached.build_strategy_decisions(cached.cal_max_profit())
    assert len(decisions) == 3
    assert all(decision.value("低點決策") not in {"BUY", "BUY_SMALL"} for decision in decisions)
    assert all(decision.field("收盤").source == FieldSource.TPEX_MARKET for decision in decisions)
    assert all(decision.field("收盤").model_version == "tpex-daily-v1" for decision in decisions)


def test_bad_ohlc_empty_dates_and_explicit_symbols_do_not_write_empty_records(tmp_path):
    data = payload("20260924", ("3105", "5347"))
    data["tables"][0]["data"][0][2] = "---"
    source, benchmark = crawler(tmp_path, FakeSession([FakeResponse(payload("20260923", ())), FakeResponse(data)]))
    source.iso_scheduled_times = ["2026-09-23", "2026-09-24"]
    source.get_twse_daily_stocks("selected", Stocktype.SEMI, ["3105", "5347", "2330"])
    assert source.daily_dates == [[], ["2026-09-24"], []]
    assert source.twse_no_data_dates == {"2026-09-23"}
    assert source.twse_closed_unscheduled_dates == set()
    assert not (tmp_path / "selected_20260923.csv").exists()
    with (tmp_path / "selected_20260924.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    assert len(rows) == 2 and rows[1][1] == "5347"


def test_latest_session_refresh_and_legacy_index_rejection(tmp_path):
    source, benchmark = crawler(tmp_path, FakeSession([FakeResponse(payload())]))
    source.iso_scheduled_times = ["2026-09-24"]
    source.get_twse_daily_stocks("one", Stocktype.SEMI, ["3105"])
    data = payload()
    data["tables"][0]["data"][0][2] = "504.00"
    refreshed, _ = crawler(tmp_path, FakeSession([FakeResponse(data)]))
    refreshed.iso_scheduled_times = ["2026-09-24"]
    refreshed._market_history_refresh_dates = {"20260924"}
    refreshed.get_twse_daily_stocks("one", Stocktype.SEMI, ["3105"])
    assert refreshed.daily_closes == [[504_000.]]
    with pytest.raises(ValueError, match="not TWSE row indexes"):
        refreshed.get_twse_daily_stocks("one", Stocktype.SEMI, ["0"])


def test_semi_cli_routes_to_tpex_without_touching_other_profiles():
    request, settings = build_run_configuration(["-t", "SEMI", "--skip-mops"])
    summary = build_preflight_summary(request, settings)
    assert request.stocklist == ["ALL"]
    assert summary["market_source"] == "TPEX"
    assert summary["output"] == "SHIRONG_tpex_semi"
    assert summary["universe"] == "HISTORICAL_SECTOR_UNION"
    with pytest.raises(SystemExit):
        build_run_configuration(["-t", "SEMI", "0", "1"])
    with pytest.raises(SystemExit):
        build_run_configuration(["-t", "SEMI", "-l", "ALL"])
    other, _ = build_run_configuration(["-t", "ELEC", "2330"])
    assert other.market_source == "TWSE" and other.output_file_names == "SHIRONG"


def test_runtime_adapter_selects_tpex_factory(tmp_path):
    source, _ = crawler(tmp_path, FakeSession([FakeResponse(payload())]))
    source.get_date_times = Mock(side_effect=lambda **_: setattr(source, "iso_scheduled_times", ["2026-09-24"]))
    twse_factory, tpex_factory = Mock(), Mock(return_value=source)
    adapter = CrawlerMarketDataAdapter(twse_factory, tpex_crawler_factory=tpex_factory)
    request = StockAnalysisRequest(["ALL"], Stocktype.SEMI, "semi", 3)
    context = adapter.fetch(request, calendar=Mock(), market_date=dt.date(2026, 9, 24),
                            is_trading_day=True, include_reference=True)
    assert context.resource is source
    twse_factory.assert_not_called()
    tpex_factory.assert_called_once_with(1)


def test_tpex_fundamentals_do_not_certify_using_twse_batch_or_restrictions(tmp_path):
    source, _ = crawler(tmp_path, FakeSession([FakeResponse(payload(symbols=("3105",)))]))
    source.iso_scheduled_times = ["2026-09-24"]
    source.get_twse_daily_stocks("semi", Stocktype.SEMI, ["ALL"])
    mops_row = {"symbol": "3105", "source": "MOPS_IXBRL", "market": "TWSE", "period_end": "2026-06-30"}
    snapshot = {"les_v6_mops_filings": [mops_row], "les_v6_filings": [{"source": "TWSE_OPENAPI"}]}
    def restriction_payload(url, **kwargs):
        rows = ([{"Date": "20260924", "SecuritiesCompanyCode": "", "CompanyName": "",
                  "暫停交易": "", "恢復交易": ""}] if url.endswith("tpex_spendi_today") else [])
        return SimpleNamespace(payload=rows)
    batch = SimpleNamespace(http_client=SimpleNamespace(get_json=Mock(side_effect=restriction_payload)))
    service = SimpleNamespace(
        financial_batch_client=batch, prepare_daily_batch=Mock(),
        filing_catalog=SimpleNamespace(select_market=Mock()),
        load=Mock(return_value=FundamentalBundle(pd.DataFrame(), snapshot, "2026-09-24T15:00:00+08:00")),
    )
    request = StockAnalysisRequest(["ALL"], Stocktype.SEMI, "semi", 3)
    FundamentalRefreshCoordinator(service).refresh(
        AnalysisContext(source, dt.date(2026, 9, 24), MarketStatus.OPEN), request,
        analysis_time=dt.datetime.fromisoformat("2026-09-24T15:00:00+08:00"),
    )
    service.prepare_daily_batch.assert_not_called()
    assert batch.http_client.get_json.call_count == 4
    assert all(call.args[0].startswith("https://www.tpex.org.tw/openapi/v1/tpex_")
               for call in batch.http_client.get_json.call_args_list)
    service.filing_catalog.select_market.assert_called_once_with("TPEX")
    saved = source.fundamental_snapshots_by_stock["3105"]
    assert saved["sector"] == "24"
    assert saved["les_v6_filings"] == [dict(mops_row, market="TPEX")]
    assert mops_row["market"] == "TWSE"  # Cached evidence was not mutated.
    assert source.trading_restrictions["3105"]["source"] == "TPEX_OPENAPI"
    assert source.trading_restrictions["3105"]["market"] == "TPEX"


@pytest.mark.usefixtures("active_industry_profiles")
@pytest.mark.parametrize("restriction_kind", ["twse", "tpex_clear", "tpex_restricted", "tpex_partial"])
def test_tpex_daily_report_requires_complete_market_bound_restrictions(restriction_kind):
    from tests.strategies.test_daily_v6_pipeline import daily_market, build
    from tests.strategies.test_restriction_asof import tpex_evidence

    base = daily_market()
    snapshot = dict(base.fundamental_snapshots["2330"])
    snapshot["les_v6_filings"] = [dict(row, symbol="3105", market="TPEX", source="MOPS_IXBRL")
                                  for row in snapshot["les_v6_filings"]]
    restriction = base.trading_restrictions["2330"] if restriction_kind == "twse" else tpex_evidence(
        base.signal_dates["2330"], restricted=restriction_kind == "tpex_restricted")
    if restriction_kind == "tpex_partial":
        restriction["endpoints"] = restriction["endpoints"][:-1]
    market = replace(base, stock_numbers=["3105"], market_source="TPEX", default_sector="SEMI",
                     fundamental_snapshots={"3105": snapshot}, signal_dates={"3105": base.signal_dates["2330"]},
                     trading_restrictions={"3105": restriction})
    _, decisions = build(market)
    report = json.loads(decisions[0].value("LES v6 Report"))
    assert report["market"]["exchange"] == "TPEX"
    assert report["quality"]["status"] == "PASS"
    if restriction_kind == "tpex_clear":
        assert report["data"]["restriction_check"]["status"] == "CLEAR"
        assert report["decision"]["signal"] in {"BUY", "BUY_SMALL"}
        assert report["sizing"]["indicative_shares"] > 0
    else:
        expected = "RESTRICTED" if restriction_kind == "tpex_restricted" else "UNKNOWN"
        assert report["data"]["restriction_check"]["status"] == expected
        reason = "LIQUIDITY_OR_RESTRICTION_FAIL" if expected == "RESTRICTED" else "TRADING_RESTRICTION_UNKNOWN"
        assert reason in report["decision"]["reasons"]
        assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}
        assert report["sizing"]["indicative_shares"] == 0


def test_default_composition_runs_tpex_to_report_without_twse_stock_download(tmp_path, monkeypatch):
    from infrastructure import bootstrap
    from infrastructure.crawler.twse_client import TwseClient
    from infrastructure.calendar.twse_calendar import TradingDayStatus
    from config.settings import AppSettings

    monkeypatch.setattr(bootstrap, "TpexCsvRepository", lambda: TpexCsvRepository(tmp_path))
    monkeypatch.setattr(TwseClient, "get_daily_taiex_close", lambda *args: 20_000.)
    monkeypatch.setattr(TwseClient, "get_daily_stock_rows", Mock(side_effect=AssertionError("TWSE stocks must not be requested")))
    session = FakeSession([FakeResponse(payload("20260923")), FakeResponse(payload())])
    calendar = Mock()
    calendar.is_trading_day.return_value = True
    calendar.trading_day_status.return_value = TradingDayStatus.OPEN
    settings = replace(AppSettings.from_env(mail_enabled=False, mops_enabled=False, validate_paths=False),
                       market_history_cache_path=tmp_path / "market.sqlite3",
                       pae_persistence_path=tmp_path / "pae.sqlite3", pae_positions=())
    runtime = bootstrap.build_stock_analysis_runtime(
        settings=settings, http_client=TwseHttpClient(config=config(), session=session),
        market_calendar=calendar,
        now_provider=lambda: dt.datetime.fromisoformat("2026-09-24T15:00:00+08:00"),
    )
    request = StockAnalysisRequest(["ALL"], Stocktype.SEMI, "integration_tpex_semi", 1, mops_enabled=False)
    assert runtime.service.run(request)
    assert len(session.calls) == 2 and all("tpex.org.tw" in item["url"] for item in session.calls)
    dataset = pd.read_csv(tmp_path / "integration_tpex_semi_analysis_dataset.csv", dtype={"證券代號": str})
    assert set(dataset["證券代號"]) == {"3105", "5347"}
    artifact = json.loads((tmp_path / "integration_tpex_semi_les_v6_reports.json").read_text())
    assert all(report["market"]["exchange"] == "TPEX" for report in artifact["reports"])
