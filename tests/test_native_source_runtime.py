"""SYNTHETIC ONLY: bounded current source capture; no HTTP, mail or accounts."""
from copy import deepcopy
from dataclasses import replace
import datetime as dt
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from application.contracts import StockAnalysisRequest
from application.daily_stock import isolate_stock_market
from application.les_v7_preflight import daily_v7_preflight
from application.market_history import MarketHistoryAssembler
from config.settings import AppSettings, SettingsError
from domain.les_v7.contracts import identity
from domain.les_v7.rule_baseline import BASELINE_POLICY, SOURCE_BASELINE_POLICY
from domain.models import Stocktype
from infrastructure.runtime.market_data_adapter import CrawlerMarketDataAdapter
from infrastructure.runtime.native_source_capture import NativeSourceCapture
from tests.test_daily_computation_performance import two_stock_market
from tests.test_official_pe import snapshot
from tests.test_security_master import packet
from tests.test_v7_daily_preflight import isolated_env


CUTOFF = "2026-10-14T15:30:00+08:00"
BEFORE_OBSERVATION = "2026-10-13T13:30:00+08:00"
SESSION = "2026-10-13"


def crawler_stub(*, dates=None, stocknumbers=None, daily_dates=None):
    dates = dates if dates is not None else [SESSION]
    symbols = stocknumbers if stocknumbers is not None else ["2330"]
    return SimpleNamespace(iso_scheduled_times=dates, stocknumbers=symbols,
        daily_dates=daily_dates if daily_dates is not None else [list(dates) for _ in symbols],
        get_date_times=Mock(), get_twse_daily_stocks=Mock())


@pytest.mark.parametrize("market", ["TWSE", "TPEX"])
def test_adapter_current_opt_in_collects_after_market_rows_only_once(market):
    crawler = crawler_stub()
    capture = Mock(); capture.collect.return_value = capture
    calls = []
    crawler.get_twse_daily_stocks.side_effect = lambda **kwargs: calls.append("market")
    capture.collect.side_effect = lambda *args, **kwargs: calls.append("native") or capture
    adapter = CrawlerMarketDataAdapter(lambda count: crawler,
        tpex_crawler_factory=lambda count: crawler, native_source_capture=capture)
    request = StockAnalysisRequest(stocklist=["2330"], stocktype=Stocktype.SEMI if market == "TPEX" else "ELEC",
                                  output_file_names="SYNTHETIC_ONLY", endbacktrack=700)
    context = adapter.fetch(request, calendar=Mock(), market_date=dt.date(2026, 10, 14),
                            is_trading_day=True, include_reference=True)
    assert calls == ["market", "native"]
    capture.collect.assert_called_once_with(crawler, market=market)
    assert context.resource.les_v7_source_capture is capture
    assert dt.datetime.fromisoformat(crawler.market_observed_at).tzinfo is not None
    assert not request.mail and request.as_of is None


def test_adapter_opt_out_does_not_create_capture_or_mutate_old_source_contract():
    crawler = crawler_stub()
    request = StockAnalysisRequest(stocklist=["2330"], stocktype="ELEC",
                                  output_file_names="SYNTHETIC_ONLY", endbacktrack=700)
    CrawlerMarketDataAdapter(lambda count: crawler).fetch(request, calendar=Mock(),
        market_date=dt.date(2026, 10, 14), is_trading_day=True, include_reference=True)
    assert not hasattr(crawler, "les_v7_source_capture")


@pytest.mark.parametrize("market", ["TWSE", "TPEX"])
def test_adapter_historical_never_downloads_current_master_or_pe(market):
    crawler = crawler_stub(); capture = Mock()
    cutoff = dt.datetime.fromisoformat(BEFORE_OBSERVATION)
    request = StockAnalysisRequest(stocklist=["2330"], stocktype=Stocktype.SEMI if market == "TPEX" else "ELEC",
        output_file_names="SYNTHETIC_ONLY", endbacktrack=700, as_of=cutoff)
    calendar = Mock(); calendar.is_trading_day.return_value = True
    context = CrawlerMarketDataAdapter(lambda count: crawler,
        tpex_crawler_factory=lambda count: crawler, native_source_capture=capture).fetch(request,
            calendar=calendar, market_date=cutoff.date(), is_trading_day=True, include_reference=True)
    capture.collect.assert_not_called()
    assert not hasattr(crawler, "les_v7_source_capture")
    assert not hasattr(crawler, "market_observed_at")
    assert context.as_of == cutoff
    assert crawler.low_entry_v4_signal_dates_by_stock == {"2330": cutoff.isoformat()}


def capture_fixture(tmp_path, *, cached=None):
    collector = Mock(return_value=tmp_path / "SYNTHETIC_MASTER_MANIFEST.json")
    documents = SimpleNamespace(read=Mock(), check=Mock())
    loader = Mock(return_value=(packet(), documents))
    pe_client = Mock(); pe_client.fetch.return_value = snapshot(symbol="2330")
    repository = Mock()
    repository.load_as_of.return_value = cached if cached is not None else [snapshot(symbol="2330")]
    capture = NativeSourceCapture(tmp_path, http_client=Mock(), collector=collector,
        master_loader=loader, pe_client=pe_client, pe_repository=repository,
        clock=lambda: dt.datetime.fromisoformat(CUTOFF))
    return capture, collector, loader, pe_client, repository, documents


def test_capture_latest_only_loads_last_sixty_dates_in_one_query_and_reselects_each_cutoff(tmp_path):
    final = dt.date.fromisoformat(SESSION)
    dates = [(final - dt.timedelta(days=offset)).isoformat() for offset in reversed(range(120))]
    crawler = crawler_stub(dates=dates)
    capture, collector, loader, pe_client, repo, documents = capture_fixture(tmp_path)
    assert capture.collect(crawler, market="TWSE") is capture
    collector.assert_called_once_with(tmp_path / "security-master", "TWSE", client=capture.http_client,
                                      now=collector.call_args.kwargs["now"])
    loader.assert_called_once_with(tmp_path / "SYNTHETIC_MASTER_MANIFEST.json")
    pe_client.fetch.assert_called_once_with(market="TWSE", trade_date=SESSION)
    repo.record.assert_called_once_with(pe_client.fetch.return_value)
    repo.load_as_of.assert_called_once_with(market="TWSE", trade_dates=dates[-60:], cutoff=CUTOFF)
    old = capture.selections(crawler, cutoff=BEFORE_OBSERVATION)["2330"]
    assert old["security"]["status"] == "UNKNOWN"
    assert all(item["status"] == "MISSING" for item in old["official_pe"])
    current = capture.selections(crawler, cutoff=CUTOFF)["2330"]
    assert current["security"]["status"] == "VERIFIED"
    assert len(current["official_pe"]) == 60
    assert current["official_pe"][-1]["status"] == "REPORTED"
    assert sum(item["status"] == "REPORTED" for item in current["official_pe"]) == 1
    assert "raw_response_utf8" not in str(current)
    assert current["availability_policy"] == "CURRENT_OBSERVATIONS_NOT_HISTORICAL_BACKFILL"
    assert capture.master_manifest == str(collector.return_value)
    assert pe_client.fetch.call_count == repo.load_as_of.call_count == collector.call_count == 1
    assert documents.check.call_count == 2  # One cheap integrity guard per stock selection.


def test_capture_selections_are_detached_from_reused_work_index(tmp_path):
    capture, *_ = capture_fixture(tmp_path)
    crawler = crawler_stub()
    capture.collect(crawler, market="TWSE")
    first = capture.selections(crawler, cutoff=CUTOFF)
    baseline = deepcopy(first)
    first["2330"]["official_pe"][0]["pe"] = 999.
    first["2330"]["security"]["instrument"]["isin"] = "OTHER"
    first["2330"]["capture_diagnostics"].append("UNTRUSTED")
    assert capture.selections(crawler, cutoff=CUTOFF) == baseline


def test_capture_failed_current_source_does_not_erase_valid_prior_observations_or_impute(tmp_path):
    capture, collector, _, pe_client, repo, _ = capture_fixture(tmp_path)
    collector.side_effect = ValueError("SYNTHETIC_MISSING_MASTER")
    pe_client.fetch.side_effect = ValueError("SYNTHETIC_MISSING_RESPONSE")
    capture.collect(crawler_stub(), market="TWSE")
    assert capture.master is None
    repo.record.assert_not_called()
    result = capture.selections(crawler_stub(), cutoff=CUTOFF)["2330"]
    assert result["security"] is None
    assert len(result["capture_diagnostics"]) == 2
    assert result["official_pe"][0]["status"] == "REPORTED"  # Previously observed exact date, not fallback price.
    assert capture.selections(crawler_stub(), cutoff=BEFORE_OBSERVATION)["2330"]["official_pe"][0]["status"] == "MISSING"


def test_corrupt_retained_pe_cache_is_not_replaced_with_live_data(tmp_path):
    capture, _, _, pe_client, repository, _ = capture_fixture(tmp_path)
    repository.load_as_of.side_effect = ValueError("SYNTHETIC_CACHE_HASH_MISMATCH")
    with pytest.raises(ValueError, match="CACHE_HASH_MISMATCH"):
        capture.collect(crawler_stub(), market="TWSE")
    pe_client.fetch.assert_called_once()
    assert repository.load_as_of.call_count == 1


def test_reusing_capture_without_completed_sessions_clears_prior_source_index(tmp_path):
    capture, collector, _, pe_client, repo, _ = capture_fixture(tmp_path)
    capture.collect(crawler_stub(), market="TWSE")
    assert capture.selections(crawler_stub(), cutoff=CUTOFF)["2330"]["official_pe"][0]["status"] == "REPORTED"
    # A malformed/stale collector has daily_dates but no verified completed
    # batch schedule. No previous job's source index may survive the reset.
    empty = crawler_stub(dates=[], daily_dates=[[SESSION]])
    capture.collect(empty, market="TWSE")
    result = capture.selections(empty, cutoff=CUTOFF)["2330"]
    assert result["security"] is None
    assert result["official_pe"][0]["status"] == "MISSING"
    assert result["capture_diagnostics"] == ["NO_COMPLETED_SESSION_FOR_SOURCE_CAPTURE"]
    assert collector.call_count == pe_client.fetch.call_count == repo.load_as_of.call_count == 1


def test_history_and_worker_transport_retain_only_current_symbol_source_proofs():
    source = {"2330": dict(security={"market_symbol": "2330", "proof": [1]}, official_pe=[{"symbol": "2330"}]),
              "2317": dict(security={"market_symbol": "2317", "proof": [2]}, official_pe=[{"symbol": "2317"}])}
    market = replace(two_stock_market(), entry_model="v7", les_v7_native_evidence=True,
        les_v7_rule_baseline=True, les_v7_source_repair=True, native_source_selections=source)
    before = deepcopy(source)
    local = isolate_stock_market(market, 0)
    assert local.les_v7_source_repair and set(local.native_source_selections) == {"2330"}
    history = MarketHistoryAssembler().build(local, 0)
    assert history.attrs["les_v7_native_sources"] == source["2330"]
    restored = pickle.loads(pickle.dumps(local))
    restored.native_source_selections["2330"]["security"]["proof"].append(99)
    assert source == before
    legacy = replace(market, les_v7_source_repair=False)
    assert "les_v7_native_sources" not in MarketHistoryAssembler().build(legacy, 0).attrs


@pytest.mark.parametrize("repair,version,profile", [("true", SOURCE_BASELINE_POLICY, "QG_V7_REPORTED_PERIOD_R5"),
                                                   ("false", BASELINE_POLICY, "QG_V7_REPORTED_PERIOD_R4")])
def test_settings_source_repair_preflight_is_explicit_and_old_r5_is_still_available(tmp_path, repair, version, profile):
    settings = AppSettings.from_env({**isolated_env(tmp_path), "TWSE_LES_V7_SOURCE_REPAIR": repair}, validate_paths=False)
    result = daily_v7_preflight(settings, checked_at=CUTOFF)
    assert result["state"] == "READY" and result["mode"] == "RULE_BASED_BASELINE"
    assert settings.les_v7_source_repair is (repair == "true")
    assert result["rule_baseline_policy"]["policy_version"] == version
    assert result["rule_baseline_policy"]["quality_policy"]["profile_id"] == profile
    assert result["source_repair_enabled"] is (repair == "true")
    assert result["order_authority"] is False
    if repair == "false":
        assert identity(result["rule_baseline_policy"]) == "2401d1f1185716e7f7b5b456b310c73d8136ac73b8860963f3dd017755c4fd2e"


def test_source_repair_flag_does_not_relax_publication_time_or_legacy_modes(tmp_path):
    settings = AppSettings.from_env(isolated_env(tmp_path), validate_paths=False)
    early = daily_v7_preflight(settings, checked_at="2026-10-14T14:00:00+08:00")
    assert early["state"] == "BLOCKED" and not early["formal_entry_enabled"]
    assert "NOT_AVAILABLE" in early["reason_codes"][0]
    legacy = replace(settings, daily_entry_model="v6")
    assert daily_v7_preflight(legacy, checked_at=CUTOFF)["state"] == "NOT_APPLICABLE"
    with pytest.raises(SettingsError, match="SOURCE_REPAIR"):
        AppSettings.from_env({**isolated_env(tmp_path), "TWSE_LES_V7_SOURCE_REPAIR": "maybe"}, validate_paths=False)
