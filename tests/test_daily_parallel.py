"""Spawn equivalence/isolation tests; synthetic inputs, not investment results."""
from copy import deepcopy
from dataclasses import fields, replace
import pickle

import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_entry import DailyEntryService
from application.daily_performance import DailyPerformance
from application.daily_stock import PER_STOCK_MAPPINGS, PER_STOCK_SEQUENCES, isolate_stock_market
from application.daily_work import DailyWorkContext
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from application.market_history import MarketHistoryInput
from config.settings import AppSettings, SettingsError
from domain.models import PortfolioPosition
from infrastructure.daily_parallel import DailyProcessExecutor
from tests.test_daily_computation_performance import two_stock_market


def test_stock_packet_preserves_global_context_and_excludes_other_financials():
    market = two_stock_market()
    market = replace(market, fallback_date="", industry_benchmarks={"24": {"benchmark": 2}, "99": {"unrelated": 9}},
        fundamental_snapshots={"2330": {"sector": "24", "proof": [1]}, "2317": {"secret_other_issuer": [2]}},
        entry_les_by_stock={"2330": 78, "2317": 95})
    isolated = isolate_stock_market(market, 0)
    for name in PER_STOCK_SEQUENCES:
        assert len(getattr(isolated, name)) == 1
        assert getattr(isolated, name)[0] == getattr(market, name)[0]
    for name in PER_STOCK_MAPPINGS:
        assert set(getattr(isolated, name)) <= {"2330"}
    assert isolated.fallback_date == max(row[-1] for row in market.dates)
    assert isolated.market_breadth_snapshots == market.market_breadth_snapshots
    assert isolated.industry_benchmarks == {"24": {"benchmark": 2}}
    # Transport is detached; editing worker input cannot mutate the parent.
    restored = pickle.loads(pickle.dumps(isolated))
    restored.fundamental_snapshots["2330"]["proof"].append(3)
    assert market.fundamental_snapshots["2330"]["proof"] == [1]
    # Fail visibly in review when the public input gains a new field.
    assert {f.name for f in fields(MarketHistoryInput)} == set(PER_STOCK_SEQUENCES + PER_STOCK_MAPPINGS) | {
        "stock_count", "account_equity", "default_sector", "les_v6_portfolio", "fallback_date",
        "market_breadth_snapshots", "market_source", "replay_as_of", "industry_benchmarks"}


@pytest.mark.parametrize("cache", [False, True])
def test_spawn_matches_complete_native_results_and_pae_with_account(cache):
    market = two_stock_market()
    # Explicit account-sensitive Entry LES/age/decision, restriction and cutoff
    # differences are part of the full equality comparison, not dropped fields.
    snapshots = deepcopy(market.fundamental_snapshots)
    snapshots["2317"] = deepcopy(snapshots["2330"])
    snapshots["2317"]["going_concern"] = True
    market = replace(market, fundamental_snapshots=snapshots,
        entry_les_by_stock={"2330": 76, "2317": 59},
        entry_dates_by_stock={"2330": "2026-09-09", "2317": "2026-09-10"},
        entry_decisions_by_stock={"2330": "BUY", "2317": "BUY_SMALL"},
        signal_dates={"2330": "2026-09-11T15:00:00+08:00", "2317": "2026-09-11T15:30:00+08:00"})
    profits = [[1, 1, 1, 1, .01]] * 2
    work = DailyWorkContext("test-parent", enabled=cache)
    probe = DailyPerformance()
    try:
        with work.activate(), probe.activate():
            serial = AnalysisDatasetBuilder().build_decisions(market, profits)
            parallel = AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)).build_decisions(market, profits)
    finally:
        work.close()
    assert parallel == serial  # all native fields, including embedded report
    allocator = PortfolioAllocationV2Pipeline()
    context = PortfolioRuntimeContext(account_equity=1_000_000., current_cash=950_000.,
        positions=(PortfolioPosition("2330", 100),), default_sector="24")
    assert allocator.enrich(parallel, context) == allocator.enrich(serial, context)
    assert probe.counters["stock_packets_submitted"] == 2
    assert probe.calls["worker_sum.stock_compute"] == 2
    assert probe.counters["worker_max_process_peak_rss_bytes"] > 0


def test_stale_symbol_empty_history_and_replay_cutoff_keep_serial_semantics():
    market = two_stock_market()
    updates = {name: (getattr(market, name)[0], ()) for name in PER_STOCK_SEQUENCES
               if name not in {"stock_numbers", "stock_names"}}
    market = replace(market, **updates, fallback_date="2026-09-14", replay_as_of="2026-09-14T13:30:00+08:00",
        signal_dates={"2330": "2026-09-14T13:30:00+08:00", "2317": "2026-09-14T13:30:00+08:00"})
    profits = [[1, 1, 1, 1, .01]] * 2
    assert AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)).build_decisions(market, profits) == (
        AnalysisDatasetBuilder().build_decisions(market, profits))


def test_invalid_history_fails_whole_pool_without_partial_publication():
    market = two_stock_market()
    market = replace(market, opens=((), market.opens[1]))
    executor = DailyProcessExecutor(2)
    with pytest.raises(ValueError, match="Open history length"):
        # Bypass the earlier screen validation to exercise an actual worker.
        executor.evaluate([(0, isolate_stock_market(market, 0), {}), (1, isolate_stock_market(market, 1), {})])


def test_capacity_exceeded_does_not_execute_or_fallback():
    executor = DailyProcessExecutor(2, max_packet_bytes=20, max_inflight_bytes=40)
    with pytest.raises(ValueError, match="CAPACITY_EXCEEDED"):
        executor.evaluate([(0, isolate_stock_market(two_stock_market(), 0), {})])


def test_injected_components_are_not_silently_replaced_by_worker_defaults():
    builder = AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2),
        daily_entry=DailyEntryService(entry_evaluator=lambda histories: pd.DataFrame()))
    with pytest.raises(ValueError, match="REQUIRES_NATIVE"):
        builder.build_decisions(two_stock_market(), [[1, 1, 1, 1, .01]] * 2)


@pytest.mark.parametrize("value", ["0", "5", "bad"])
def test_worker_setting_rejects_invalid_values(value):
    with pytest.raises(SettingsError, match="TWSE_DAILY_STRATEGY_WORKERS"):
        AppSettings.from_env({"TWSE_DAILY_STRATEGY_WORKERS": value}, validate_paths=False)


def test_worker_configuration_is_opt_in_and_existing_default_is_serial():
    assert AppSettings.from_env({}, validate_paths=False).daily_strategy_workers == 1
    assert AppSettings.from_env({"TWSE_DAILY_STRATEGY_WORKERS": "2"}, validate_paths=False).daily_strategy_workers == 2
    from stockanalysis import TwseCrawker
    crawler = TwseCrawker(0, app_settings=AppSettings(daily_strategy_workers=2))
    assert crawler._analysis_dataset_builder.stock_executor.workers == 2


def test_worker_startup_rejects_different_content_version(monkeypatch):
    import infrastructure.daily_parallel as parallel
    from dataclasses import dataclass
    @dataclass
    class Version:
        code: str = "different-source"
    monkeypatch.setattr(parallel, "ResearchVersionGuard", lambda: type("Guard", (), {"versions": Version()})())
    with pytest.raises(ValueError, match="SOURCE_VERSION_MISMATCH"):
        parallel._initialize("not-the-version-hash", False, 0)


def test_queue_is_lazy_bounded_and_worker_completion_order_does_not_reorder(monkeypatch):
    import infrastructure.daily_parallel as parallel
    from concurrent.futures import Future
    from types import SimpleNamespace
    state = dict(produced=0, submitted=0, settled=0, peak=0, shutdown=0)
    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"
        def submit(self, calculate, encoded):
            state["submitted"] += 1
            item, market, watch = pickle.loads(encoded)
            future = Future()
            future.item = item
            future.set_result((item, pd.DataFrame({"item": [item]}), pd.DataFrame({"item": [item]}),
                               dict(stages={}, counters={}, process_peak_rss_bytes=None)))
            return future
        def shutdown(self, **kwargs):
            state["shutdown"] += 1
    def settled_last(pending, **kwargs):
        state["peak"] = max(state["peak"], len(pending))
        assert len(pending) <= 2
        state["settled"] += 1
        # Deliberately finish later input first.
        return {max(pending, key=lambda f: f.item)}, set()
    def jobs():
        for item in range(7):
            state["produced"] += 1
            assert state["produced"] - state["settled"] <= 2
            yield item, SimpleNamespace(stock_count=1), {}
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(parallel, "wait", settled_last)
    results = DailyProcessExecutor(2).evaluate(jobs())
    assert [results[item][0].iloc[0]["item"] for item in range(7)] == list(range(7))
    assert state == dict(produced=7, submitted=7, settled=7, peak=2, shutdown=1)


def test_same_job_cannot_omit_or_duplicate_index(monkeypatch):
    from types import SimpleNamespace
    import infrastructure.daily_parallel as parallel
    from concurrent.futures import Future
    class Pool:
        def __init__(self, **kwargs):
            pass
        def submit(self, calculate, encoded):
            item, _, _ = pickle.loads(encoded)
            future = Future()
            future.set_result((item, None, None, dict(stages={}, counters={}, process_peak_rss_bytes=None)))
            return future
        def shutdown(self, **kwargs):
            pass
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Pool)
    executor = DailyProcessExecutor(2)
    with pytest.raises(ValueError, match="DUPLICATE_OR_INVALID"):
        executor.evaluate([(0, SimpleNamespace(stock_count=1), {}), (0, SimpleNamespace(stock_count=1), {})])
    with pytest.raises(ValueError, match="SEQUENCE_INCOMPLETE"):
        executor.evaluate([(1, SimpleNamespace(stock_count=1), {})])
