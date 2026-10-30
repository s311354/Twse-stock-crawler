"""Synthetic clock/source wiring; no network, SMTP or real research run."""
from dataclasses import replace
import datetime as dt
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from application.contracts import AnalysisContext, MarketStatus, StockAnalysisRequest
from application.market_history import MarketHistoryAssembler
from config.settings import AppSettings
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from infrastructure.legacy_crawler_facade import LegacyCrawlerFacade
from infrastructure.runtime.market_data_adapter import CrawlerMarketDataAdapter
from infrastructure.runtime.strategy_adapter import CrawlerStrategyAdapter
from tests.strategies.test_daily_v6_pipeline import daily_market


COLLECTED = "2026-10-13T08:00:00+00:00"
DECIDED = "2026-10-13T08:05:00+00:00"


def facade(settings, *, replay=None):
    source = daily_market()
    result = SimpleNamespace(stocklistsize=source.stock_count, stocknumbers=source.stock_numbers,
        stocknames=source.stock_names, daily_dates=source.dates, daily_stocks=source.opens,
        daily_highs=source.highs, daily_lows=source.lows, daily_closes=source.closes,
        daily_volumes=source.volumes, daily_trade_counts=source.trade_counts,
        daily_trade_values=source.trade_values, daily_bid_prices=source.bids,
        daily_ask_prices=source.asks, daily_pe_ratios=source.pe_ratios,
        daily_taiex_closes=source.taiex_closes, annual_fundamentals_by_stock={},
        fundamental_snapshots_by_stock={}, fundamental_status_by_stock={},
        fundamental_status_detail_by_stock={}, low_entry_v4_signal_dates_by_stock={"2330": "2026-09-11T13:30:00+08:00"},
        low_entry_v4_account_equity=1_000_000., app_settings=settings,
        iso_scheduled_times=["2026-09-11"], replay_as_of=replay,
        market_observed_at=COLLECTED, les_v7_decision_time=DECIDED)
    return result


def test_native_daily_uses_one_actual_clock_not_latest_bar_close():
    settings = AppSettings(daily_entry_model="v7", les_v7_native_evidence=True,
        les_v7_strategy_policy_json="SYNTHETIC_POLICY_FORWARDING_ONLY", les_v7_diagnostics_only=True)
    crawler = facade(settings)
    market = LegacyCrawlerFacade._market_history_input(crawler)
    assert market.signal_dates == {"2330": DECIDED}
    assert crawler.low_entry_v4_signal_dates_by_stock["2330"] == "2026-09-11T13:30:00+08:00"
    assert market.market_observed_at == COLLECTED
    assert market.les_v7_strategy_policy_json == settings.les_v7_strategy_policy_json
    assert market.les_v7_diagnostics_only is True
    frame = MarketHistoryAssembler().build(market, 0)
    assert frame.attrs[SIGNAL_DATE_ATTR] == DECIDED
    proof = frame.attrs["les_v7_market_evidence"]
    assert proof["source_available_at"] == COLLECTED
    assert proof["session_date"] == "2026-09-11"
    assert proof["historical_original_observation_verified"] is False
    assert proof["source_final"] is False  # No batch finality proof, not a default PASS.
    assert proof["price_unit"] == "TWD" and proof["volume_unit"] == "shares"
    assert frame.Close.iloc[-1] == market.closes[0][-1]  # Legacy scale is not altered.


@pytest.mark.parametrize("entry_model,replay", [("v6", None), ("v7", "2026-09-11T13:30:00+08:00")])
def test_legacy_and_historical_clocks_are_not_replaced(entry_model, replay):
    crawler = facade(AppSettings(daily_entry_model=entry_model, les_v7_native_evidence=True), replay=replay)
    market = LegacyCrawlerFacade._market_history_input(crawler)
    assert market.signal_dates["2330"] == "2026-09-11T13:30:00+08:00"
    assert market.les_v7_native_evidence is False and market.market_observed_at is None
    assert "les_v7_market_evidence" not in MarketHistoryAssembler().build(market, 0).attrs


def test_market_observation_is_real_collection_clock_not_requested_bar_date():
    crawler = SimpleNamespace(iso_scheduled_times=["2026-09-11"], stocknumbers=["2330"],
        get_date_times=Mock(), get_twse_daily_stocks=Mock())
    request = StockAnalysisRequest(stocklist=["2330"], stocktype="ELEC", output_file_names="SYNTHETIC", endbacktrack=700)
    before = dt.datetime.now(dt.timezone.utc)
    CrawlerMarketDataAdapter(lambda count: crawler).fetch(request, calendar=Mock(),
        market_date=dt.date(2026, 9, 11), is_trading_day=True, include_reference=True)
    observed = dt.datetime.fromisoformat(crawler.market_observed_at)
    assert before <= observed <= dt.datetime.now(dt.timezone.utc)
    assert observed.date() != dt.date(2026, 9, 11)


@pytest.mark.parametrize("replay", [None, "2026-09-11T13:30:00+08:00"])
def test_strategy_fixes_clock_after_collection_before_decision_assembly(replay):
    settings = AppSettings(daily_entry_model="v7", les_v7_native_evidence=True)
    crawler = SimpleNamespace(app_settings=settings, market_observed_at=COLLECTED)
    crawler.cal_max_profit = Mock(return_value=[])
    captured = []
    crawler.build_strategy_decisions = lambda _: captured.append(getattr(crawler, "les_v7_decision_time", None)) or ()
    context = AnalysisContext(crawler, dt.date(2026, 9, 11), MarketStatus.OPEN,
        dt.datetime.fromisoformat(replay) if replay else None)
    before = dt.datetime.now(dt.timezone.utc)
    CrawlerStrategyAdapter(portfolio_current_cash=1_000_000.).evaluate(context)
    if replay:
        assert captured == [None]
    else:
        assert before <= dt.datetime.fromisoformat(captured[0]) <= dt.datetime.now(dt.timezone.utc)
        assert dt.datetime.fromisoformat(captured[0]) > dt.datetime.fromisoformat(COLLECTED)
    crawler.cal_max_profit.assert_called_once()


def test_final_batch_marker_and_native_global_fields_survive_stock_isolation():
    from application.daily_stock import isolate_stock_market
    market = LegacyCrawlerFacade._market_history_input(facade(AppSettings(
        daily_entry_model="v7", les_v7_native_evidence=True)))
    market = replace(market, market_breadth_snapshots={"2026-09-11": {"source_final": True}})
    frame = MarketHistoryAssembler().build(isolate_stock_market(market, 0), 0)
    assert frame.attrs["les_v7_market_evidence"]["source_final"] is True
    assert frame.attrs["les_v7_market_evidence"]["source_available_at"] == COLLECTED


@pytest.mark.parametrize("memory_bytes", [0, 100_000])
def test_native_later_cutoff_uses_exact_retained_capture_not_a_relaxed_spool(memory_bytes):
    from application.evidence_context import EvidenceContext
    from application.daily_stock import isolate_stock_market
    from infrastructure.snapshot_retention import WorkSnapshotRetention
    owner = WorkSnapshotRetention("SYNTHETIC_CAPTURE", memory_bytes=memory_bytes)
    source = {"sector": "24", "les_v6_filings": [], "proof": [1, 2]}
    crawler = facade(AppSettings(daily_entry_model="v7", les_v7_native_evidence=True))
    crawler.low_entry_v4_signal_dates_by_stock = {"2330": COLLECTED}
    retained = owner.snapshots()
    retained.retain("2330", source, cutoff=COLLECTED, market="TWSE")
    crawler.fundamental_snapshots_by_stock = retained
    try:
        market = LegacyCrawlerFacade._market_history_input(crawler)
        assert market.fundamental_capture_cutoffs == {"2330": COLLECTED}
        # The underlying exact binding still refuses the evaluation clock.
        with pytest.raises(ValueError, match="SNAPSHOT_RETENTION_BINDING_MISMATCH"):
            retained.materialize("2330", cutoff=DECIDED, market="TWSE")
        history = MarketHistoryAssembler().build(market, 0, include_evidence=False)
        with EvidenceContext(market) as context, context.stock(0, history) as packet:
            assert list(packet.snapshot["proof"]) == source["proof"]
            assert packet.cutoff == DECIDED
        local = isolate_stock_market(market, 0)
        assert local.fundamental_snapshots == {"2330": source}
        assert local.fundamental_capture_cutoffs == {"2330": COLLECTED}
        assert MarketHistoryAssembler().build(market, 0).attrs["les_v7_fundamental_capture_cutoff"] == COLLECTED
        for invalid in (replace(market, fundamental_capture_cutoffs={"2330": "2027-01-01T00:00:00Z"}),
                        replace(market, replay_as_of=DECIDED), replace(market, entry_model="v6")):
            with pytest.raises(ValueError, match="FUNDAMENTAL_CAPTURE_CUTOFF_BINDING_MISMATCH"):
                isolate_stock_market(invalid, 0)
    finally:
        owner.close()


def test_native_retained_daily_complete_serial_spawn_results_are_identical():
    import json
    from application.analysis_dataset import AnalysisDatasetBuilder
    from infrastructure.daily_parallel import DailyProcessExecutor
    from infrastructure.snapshot_retention import WorkSnapshotRetention
    from domain.les_v7.quality_policy import default_quality_policy
    from domain.les_v7.strategy_policy import make_strategy_policy
    from tests.strategies.test_daily_v7_pipeline import synthetic_market
    from tests.strategies.test_low_entry_score_v7 import bundle
    from tests.strategies.test_v7_reported_quality import source_records
    rules = bundle()["payload"]
    policy = make_strategy_policy(policy_version="SYNTHETIC_NATIVE", available_at="2026-01-03T00:00:00Z",
        quality_policy=default_quality_policy(), **{k: rules[k] for k in ("context_rules", "trigger_rules", "reference_policy")})
    original = {"sector": "24", "les_v6_filings": source_records(), "les_v6_mops_filings": source_records()}
    observed, cutoff = "2026-10-12T13:35:00+08:00", "2026-10-12T15:00:00+08:00"
    market = replace(synthetic_market(), stock_numbers=["2330"], les_v7_bundle_json=None,
        signal_dates={"2330": cutoff}, fundamental_snapshots={"2330": original},
        fundamental_capture_cutoffs={"2330": observed}, les_v7_native_evidence=True,
        les_v7_diagnostics_only=True, les_v7_strategy_policy_json=json.dumps(policy),
        market_observed_at=observed, market_breadth_snapshots={"2026-10-12": {"source_final": True}})
    profits = [[1, 1, 1, 1, .01]]
    expected = AnalysisDatasetBuilder().build_decisions(market, profits)
    owner = WorkSnapshotRetention("SYNTHETIC_SERIAL_SPAWN", memory_bytes=0)
    retained = owner.snapshots()
    retained.retain("2330", original, cutoff=observed, market="TWSE")
    try:
        retained_market = replace(market, fundamental_snapshots=retained)
        serial = AnalysisDatasetBuilder().build_decisions(retained_market, profits)
        parallel = AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)).build_decisions(retained_market, profits)
        assert serial == parallel == expected  # Includes all scores, gates, provenance/account fields.
        report = json.loads(serial[0].value("LES v7 Report"))
        assert report["diagnostics"]["native_evidence"]["source_packet"]["financial_capture_cutoff"] == "2026-10-12T05:35:00+00:00"
        assert report["pae_authorization"]["authorized"] is False
    finally:
        owner.close()
