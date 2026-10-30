"""SYNTHETIC: daily preload spool ownership, not financial certification."""

import copy
import datetime as dt
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

from application.contracts import AnalysisContext, StockAnalysisRequest
from application.daily_performance import DailyPerformance
from domain.fundamental_derivation import FundamentalBundle
from infrastructure.fundamentals.mops_repository import MopsStorageError
from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator
from tests.test_snapshot_retention import retained_work


CUTOFF = "2026-09-11T15:00:00+08:00"


def bundle(symbol, origin):
    return FundamentalBundle(pd.DataFrame({"fiscal_year": [2025]}),
        {"sector": "24", "available_at": CUTOFF,
         "origin": origin, "values": [symbol], "remote_refresh_skipped": True}, CUTOFF)


def exercise(*, streaming, partial_failure=False, refresh_failure=False, retained=True,
             mutating_consumer=True):
    outputs, unavailable, events, loads, schedules = {}, {}, [], [], []

    def attach(symbol, **kwargs):
        outputs[symbol] = copy.deepcopy(kwargs)
        # An instrument consumer cannot poison reuse for a second instrument
        # belonging to the same issuer.
        if mutating_consumer:
            kwargs["fundamental_snapshot"]["values"].append("CONSUMER_MUTATION")

    crawler = SimpleNamespace(stocknumbers=["2330", "2317", "2330A"],
        stocknames=["Common A", "Common B", "Preferred 特"],
        daily_dates=[["2026-09-11"]] * 3,
        set_low_entry_v4_fundamentals=attach,
        set_low_entry_v4_fundamentals_unavailable=lambda symbol, **kw: unavailable.update({symbol: kw}))

    class Control:
        scheduled_order = []

        def schedule(self, symbols, **kwargs):
            schedules.append((list(symbols), kwargs))

        def summary(self):
            return {}

    class Service:
        refresh_control = Control()

        def load_cached_bundles(self, symbols, **kwargs):
            assert symbols == ["2330", "2317"]
            assert kwargs["signal_date"] == CUTOFF
            return {"2330": bundle("2330", "PRELOAD")}

        def load_cached_bundles_into(self, symbols, *, sink, **kwargs):
            assert symbols == ["2330", "2317"]
            assert kwargs["signal_date"] == CUTOFF
            sink("2330", bundle("2330", "PRELOAD"))
            if partial_failure:
                raise MopsStorageError("SYNTHETIC_PARTIAL_PRELOAD")
            return ("2330",)

        def load(self, symbol, **kwargs):
            assert kwargs["signal_date"] == CUTOFF
            loads.append(symbol)
            if refresh_failure and symbol == "2317":
                raise MopsStorageError("SYNTHETIC_REFRESH_FAILURE")
            return bundle(symbol, "REFRESH")

    service = Service()
    if not streaming:
        service.load_cached_bundles_into = None
    observer = SimpleNamespace(record=lambda event, **kw: events.append((event, kw)))
    probe = DailyPerformance()
    scope = retained_work(memory_bytes=0) if retained else nullcontext((None, None))
    with probe.activate(), scope as (_, owner):
        FundamentalRefreshCoordinator(service, observer=observer).refresh(
            AnalysisContext(crawler, dt.date(2026, 9, 11), "CLOSED"),
            StockAnalysisRequest([], "ELEC", "SYNTHETIC", 700),
            analysis_time=dt.datetime.fromisoformat(CUTOFF))
        # Both the failed-preload cleanup and normal completion release all
        # issuer encodings; downstream snapshots are owned by the consumer.
        if owner is not None:
            assert owner.stats["objects"] == owner.stats["spooled_bytes"] == 0
    return outputs, unavailable, events, loads, schedules, probe.snapshot()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("refresh_failure", [False, True])
def test_preloaded_issuer_retained_once_preserves_outputs_counts_and_later_failure(streaming, refresh_failure):
    outputs, unavailable, events, loads, schedules, metrics = exercise(
        streaming=streaming, refresh_failure=refresh_failure)
    assert loads == ["2317"]
    assert schedules == [(["2317"], {"priority_symbols": []})]
    assert outputs["2330"]["fundamental_snapshot"]["origin"] == "PRELOAD"
    assert outputs["2330A"]["fundamental_snapshot"]["origin"] == "PRELOAD"
    assert outputs["2330A"]["fundamental_snapshot"]["values"] == ["2330"]
    assert outputs["2330A"]["fundamental_snapshot"]["market_symbol"] == "2330A"
    assert outputs["2330A"]["fundamental_snapshot"]["security_type"] == "PREFERRED"
    assert all(row["signal_date"] == CUTOFF for row in outputs.values())
    assert set(unavailable) == ({"2317"} if refresh_failure else set())
    if refresh_failure:
        assert unavailable["2317"]["status"] == "STORAGE_ERROR"
    else:
        assert outputs["2317"]["fundamental_snapshot"]["origin"] == "REFRESH"
    preload = next(value for event, value in events if event == "fundamental_cache_preload_completed")
    complete = next(value for event, value in events if event == "fundamental_runtime_completed")
    assert preload["reused_symbol_count"] == complete["preloaded_symbol_count"] == 1
    assert complete["issuer_symbol_count"] == 2
    # Stable CI criterion: one pickle retention per unique issuer, not a timing
    # assertion; the old two-map path retained the preload a second time.
    assert metrics["counters"]["snapshots_retained"] == (1 if refresh_failure else 2)
    assert metrics["counters"]["bundle_materializations"] == 2


def test_partial_streaming_preload_is_discarded_before_same_fallback():
    outputs, unavailable, events, loads, schedules, metrics = exercise(
        streaming=True, partial_failure=True)
    assert loads == ["2330", "2317"]
    assert unavailable == {}
    assert schedules == [(["2330", "2317", "2330"], {"priority_symbols": []})]
    assert all(row["fundamental_snapshot"]["origin"] == "REFRESH" for row in outputs.values())
    complete = next(value for event, value in events if event == "fundamental_runtime_completed")
    assert complete["preloaded_symbol_count"] == 0
    assert metrics["counters"]["snapshots_retained"] == 3
    assert any(event == "fundamental_cache_preload_failed" for event, _ in events)


@pytest.mark.parametrize("refresh_failure", [False, True])
def test_shared_native_map_matches_all_plain_dictionary_business_fields(refresh_failure):
    native = exercise(streaming=True, refresh_failure=refresh_failure, mutating_consumer=False)
    legacy = exercise(streaming=False, refresh_failure=refresh_failure, retained=False,
                      mutating_consumer=False)
    assert native[1] == legacy[1]  # Same unavailable reasons and causal times.
    assert native[3:5] == legacy[3:5]  # Same requests and refresh queue.
    assert set(native[0]) == set(legacy[0])
    for symbol, value in native[0].items():
        expected = dict(legacy[0][symbol])
        actual = dict(value)
        pd.testing.assert_frame_equal(actual.pop("annual_fundamentals"),
                                      expected.pop("annual_fundamentals"), check_exact=True)
        assert actual == expected


def test_component_benchmark_has_equal_complete_bundles_and_half_retain_calls():
    from examples.benchmark_preload_retention import compare
    rows = compare({symbol: bundle(symbol, "SYNTHETIC") for symbol in ("2330", "2317")}, rounds=1)
    assert all(row["complete_bundle_equality"] for row in rows)
    assert [row["metrics"]["counters"]["snapshots_retained"] for row in rows] == [4, 2]
    assert [row["metrics"]["counters"]["bundle_materializations"] for row in rows] == [2, 2]
    assert rows[0]["retained_storage"] == rows[1]["retained_storage"]
    with pytest.raises(ValueError, match="POSITIVE_ROUNDS"):
        compare({"2330": bundle("2330", "SYNTHETIC")}, rounds=0)
