"""Synthetic equivalence/call-count checks, not investment performance tests."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance, count_work, measure_stage
from application.leadership_screen import screen_once, watch_result_key
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from domain.features import SharedFeatureConfig, SharedFeaturePipeline
from domain.models import PortfolioPosition
from domain.strategies.expected_return_exit_v3 import ExpectedReturnExitV3Strategy
from domain.strategies.low_entry_v6_watch import screen_leadership
from tests.indicators.test_shared_feature_pipeline import _history
from tests.strategies.test_daily_v6_pipeline import daily_market
from tests.strategies.test_expected_return_exit_v3 import make_history


@pytest.mark.parametrize("size", [0, 1, 19, 61, 240])
@pytest.mark.parametrize("missing", [False, True])
def test_batched_units_are_exactly_equal_to_independent_features(size, missing):
    history = _history(size)
    if missing and size:
        history.loc[history.index[0], ["Close", "High", "Volume", "TAIEX Close"]] = np.nan
        history.loc[history.index[-1], ["Volume", "Bid", "Ask"]] = 0
    normalized = history.copy()
    for field in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        normalized[field] = pd.to_numeric(normalized[field], errors="coerce") / 1000.
    pipeline = SharedFeaturePipeline(SharedFeatureConfig(ema_long_window=120))
    native, raw = pipeline.build_price_pair(history)
    for actual, source in ((native, history), (raw, normalized)):
        pd.testing.assert_frame_equal(actual.to_pandas(), pipeline.build(source).to_pandas(), check_exact=True)
        actual.assert_compatible(source)
        changed = actual.to_pandas()
        changed.loc[:, "close"] = -1
        pd.testing.assert_frame_equal(actual.to_pandas(), pipeline.build(source).to_pandas(), check_exact=True)
    if size:
        with pytest.raises(ValueError, match="source data"):
            raw.assert_compatible(history)


@pytest.mark.parametrize("seed", [0, 19, 103])
def test_price_pair_matches_independent_arithmetic_with_random_missing_and_tiny_prices(seed):
    history = _history(240)
    rng = np.random.default_rng(seed)
    close = rng.uniform(.00000001, 100000., len(history))
    for name, offset in (("Open", -.0001), ("High", .01), ("Low", -.01), ("Close", 0), ("Bid", -.01), ("Ask", .01)):
        history[name] = close + offset
    history.loc[[18, 85, 213], ["Close", "Volume", "TAIEX Close"]] = np.nan
    pipeline = SharedFeaturePipeline()
    pair = pipeline.build_price_pair(history)
    normalized = history.copy()
    for name in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        normalized[name] /= 1000.
    for actual, source in zip(pair, (history, normalized), strict=True):
        pd.testing.assert_frame_equal(actual.to_pandas(), pipeline.build(source).to_pandas(), check_exact=True)


@pytest.mark.parametrize("decision", ["BUY", "BUY_SMALL", "EARLY_BUY", None])
@pytest.mark.parametrize("optional", [False, True])
def test_erem_latest_keeps_every_field_and_exit_confirmation(decision, optional, monkeypatch):
    history = make_history(optional)
    history.index = history.index * 2 + 17
    if optional:
        history["Entry Decision"] = decision
    features = SharedFeaturePipeline().build(history)
    strategy = ExpectedReturnExitV3Strategy()
    expected = strategy.run_with_features(history, features).tail(1)
    lengths = []
    apply = pd.DataFrame.apply
    def tracked(frame, *args, **kwargs):
        if "exit_confirmed" in frame:
            lengths.append(len(frame))
        return apply(frame, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "apply", tracked)
    actual = strategy.run_latest_with_features(history, features)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert lengths == [1]


def two_stock_market():
    market = daily_market()
    updates = {name: [list(getattr(market, name)[0]), list(getattr(market, name)[0])]
               for name in ("dates", "opens", "highs", "lows", "closes", "volumes", "trade_counts",
                            "trade_values", "bids", "asks", "pe_ratios", "taiex_closes")}
    return replace(market, stock_count=2, stock_numbers=["2330", "2317"], stock_names=["Synthetic A", "Synthetic B"],
                   **updates)


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("change", ["none", "missing", "reject", "restriction", "stale"])
def test_all_daily_and_pae_fields_match_nonoptimized_path(held, change):
    market = two_stock_market()
    snapshots = deepcopy(market.fundamental_snapshots)
    if change == "missing":
        snapshots = {}
    elif change == "reject":
        snapshots["2330"]["going_concern"] = True
    elif change == "restriction":
        restrictions = deepcopy(market.trading_restrictions)
        restrictions["2330"]["restricted"] = True
        market = replace(market, trading_restrictions=restrictions)
    elif change == "stale":
        market = replace(market, fallback_date="2026-09-14")
    market = replace(market, fundamental_snapshots=snapshots)
    results = [AnalysisDatasetBuilder(optimized_daily=mode).build_decisions(market, [[1, 1, 1, 1, .01]] * 2)
               for mode in (False, True)]
    assert results[0] == results[1]
    context = PortfolioRuntimeContext(account_equity=1_000_000., current_cash=990_330. if held else 1_000_000.,
        positions=(PortfolioPosition("2330", 100),) if held else (), default_sector="24")
    assert PortfolioAllocationV2Pipeline().enrich(results[0], context) == PortfolioAllocationV2Pipeline().enrich(results[1], context)


def test_daily_uses_one_feature_pair_and_one_erem_row_per_stock():
    probe = DailyPerformance()
    with probe.activate():
        AnalysisDatasetBuilder().build_decisions(two_stock_market(), [[1, 1, 1, 1, .01]] * 2)
    assert probe.counters["feature_pair_builds"] == 2
    assert probe.counters["v6_fallback_feature_builds"] == 0
    assert probe.counters["strategy_output_rows"] == 2


def watched_histories():
    return AnalysisDatasetBuilder().history_assembler.build_all(two_stock_market())


@pytest.mark.parametrize("change", ["none", "financial", "price", "peer", "restriction", "cutoff", "complete", "scale", "tampered"])
def test_watch_reuses_only_identical_nonfinancial_universe(change):
    histories = watched_histories()
    session, cutoff = "2026-09-11", "2026-09-11T14:00:00+08:00"
    original, key = screen_once(histories, session_date=session, as_of=cutoff)
    for history, watch in zip(histories, original, strict=True):
        history.attrs["les_v6_research_context"] = {"watch": deepcopy(watch), "watch_input_sha256": key,
                                                    "watch_result_sha256": watch_result_key(watch)}
    if change == "financial":
        histories[0].attrs["fundamental_snapshot"] = {"quality": 100}
    elif change == "price":
        histories[0].loc[histories[0].index[-1], "Close"] += 1
    elif change == "peer":
        histories = histories[:1]
    elif change == "restriction":
        histories[0].attrs["trading_restriction"] = {}
    elif change == "cutoff":
        cutoff = "2026-09-11T14:01:00+08:00"
    elif change == "complete":
        histories[0]["Complete"] = False
    elif change == "scale":
        from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
        histories[0].attrs[PRICE_SCALE_ATTR] = 1.
    elif change == "tampered":
        histories[0].attrs["les_v6_research_context"]["watch"]["status"] = "tampered"
    probe = DailyPerformance()
    with probe.activate():
        actual, _ = screen_once(histories, session_date=session, as_of=cutoff)
    assert actual == screen_leadership(histories, session_date=session, as_of=cutoff)
    assert probe.counters["leadership_cache_hits"] == (1 if change in {"none", "financial"} else 0)
    actual[0]["status"] = "mutated"
    assert histories[0].attrs["les_v6_research_context"]["watch"]["status"] != "mutated"


def test_probe_is_scoped_and_records_failures_without_affecting_results():
    probe = DailyPerformance(clock=iter([1., 3.]).__next__)
    with probe.activate():
        with pytest.raises(RuntimeError):
            with measure_stage("synthetic_failure"):
                count_work("attempts")
                raise RuntimeError("synthetic")
    count_work("attempts")
    assert probe.counters["attempts"] == 1
    assert probe.snapshot()["stages"]["synthetic_failure"] == {"seconds": 2., "calls": 1}


@pytest.mark.parametrize("failure", [False, True])
def test_native_service_publishes_stage_metrics_on_success_and_failure_without_notifications(failure):
    from types import SimpleNamespace
    from application.contracts import StockAnalysisRequest
    from application.stock_service import StockAnalysisService
    class Observer:
        def __init__(self):
            self.events, self.statuses = [], []
        def start(self, request):
            pass
        def record(self, name, **metrics):
            self.events.append((name, metrics))
        def finish(self, status):
            self.statuses.append(status)
    observer = Observer()
    def compute(context):
        if failure:
            raise RuntimeError("synthetic_failure")
        return object()
    receipts = []
    service = StockAnalysisService(
        fetch_market_data=SimpleNamespace(execute=lambda request: object()),
        refresh_fundamentals=SimpleNamespace(execute=lambda context, request: None),
        evaluate_strategies=SimpleNamespace(execute=compute),
        generate_report=SimpleNamespace(execute=lambda *args: None),
        send_notification=SimpleNamespace(execute=lambda *args: None),
        line_chart=SimpleNamespace(generate=lambda request: None), observer=observer,
        performance_sink=lambda request, context, metrics: receipts.append(metrics))
    request = StockAnalysisRequest(stocklist=["2330"], stocktype="ELEC", output_file_names="synthetic", endbacktrack=700)
    if failure:
        with pytest.raises(RuntimeError, match="synthetic_failure"):
            service.run(request)
    else:
        assert service.run(request)
    metrics = next(value for name, value in observer.events if name == "daily_performance")
    assert metrics["stages"]["total"]["calls"] == 1
    assert metrics["stages"]["strategies"]["calls"] == 1
    assert "notification" not in metrics["stages"]
    assert metrics["status"] == ("FAILED" if failure else "SUCCESS")
    assert receipts == [service.last_performance] and receipts[0] == {k: v for k, v in metrics.items() if k != "source"}
    assert observer.statuses == ["FAILED" if failure else "SUCCESS"]


def test_raw_original_parse_counter_counts_only_real_parses_not_lru_hits(tmp_path):
    import hashlib
    from infrastructure.fundamentals.mops_queries import _raw_normalized_facts
    from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
    from tests.fundamentals.test_ixbrl_units import document
    content = document()
    path = tmp_path / "synthetic.xml"
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    stat = path.stat()
    signature = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    probe = DailyPerformance()
    with probe.activate():
        first = _raw_normalized_facts(path, sha, signature, V6_FILING_SCHEMA_VERSION)
        assert _raw_normalized_facts(path, sha, signature, V6_FILING_SCHEMA_VERSION) == first
        assert _raw_normalized_facts(path, "0" * 64, signature, V6_FILING_SCHEMA_VERSION) is None
    facts, schema_refs = first
    assert facts[0].unit == "iso4217:TWD" and facts[0].numeric_value == 5_000_000
    assert schema_refs == ()  # This synthetic parser-counter fixture has no schemaRef.
    assert probe.counters["raw_unit_parse_calls"] == 1
    assert probe.counters["raw_unit_hash_rejections"] == 1
