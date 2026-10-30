"""Synthetic bounded restriction refresh; no live HTTP or persistent accounts."""
import datetime as dt
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest

from application.contracts import AnalysisContext, MarketStatus, StockAnalysisRequest
from domain.trading_restrictions import RESTRICTION_CONTRACTS, restriction_check
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator
from infrastructure.runtime.strategy_adapter import CrawlerStrategyAdapter


NOW = dt.datetime.fromisoformat("2026-10-16T14:00:00+08:00")


class Clock:
    def __init__(self, value=NOW):
        self.value = value

    def __call__(self):
        return self.value


def proof(at=NOW, *, market="TWSE", restricted=False):
    source, endpoints = RESTRICTION_CONTRACTS[market]
    day = at.astimezone(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    return dict(market=market, source=source, endpoints=list(endpoints), restricted=restricted,
        available_at=at.isoformat(), session_date=day, observation_date=day, scope="CURRENT_AS_OF")


def context(*, market="TWSE", observed=NOW, restricted=False, armed=True, native=True, model="v7", replay=False):
    crawler = SimpleNamespace(stocknumbers=["2330", "2303"], iso_scheduled_times=["2026-10-15"],
        market_source=market, app_settings=SimpleNamespace(daily_entry_model=model, les_v7_native_evidence=native),
        trading_restrictions={s: proof(observed, market=market, restricted=restricted) for s in ("2330", "2303")},
        cal_max_profit=Mock(return_value=[]), build_strategy_decisions=Mock(return_value=[]))
    if armed:
        crawler._restriction_refresh_market = market
        crawler._restriction_recapture_attempted = False
    result = AnalysisContext(crawler, NOW.date(), MarketStatus.OPEN, NOW if replay else None)
    if armed:
        crawler._restriction_refresh_context = weakref.ref(result)
    return result


def coordinator(clock, *, client=None, repository=None):
    http = client or Mock(get_json=Mock(return_value=SimpleNamespace(payload=[])))
    service = SimpleNamespace(financial_batch_client=SimpleNamespace(http_client=http))
    return FundamentalRefreshCoordinator(service, now_provider=clock, restriction_repository=repository), http


@pytest.mark.parametrize("restricted", [False, True])
def test_fresh_complete_clear_or_restricted_needs_no_extra_http(restricted):
    job, http = coordinator(Clock())
    ctx = context(restricted=restricted)
    original = ctx.resource.trading_restrictions
    job.ensure_current_restrictions(ctx)
    assert ctx.resource.trading_restrictions is original
    assert ctx.resource._restriction_recapture_attempted is False
    http.get_json.assert_not_called()


@pytest.mark.parametrize("market,count", [("TWSE", 3), ("TPEX", 4)])
def test_stale_complete_capture_uses_final_response_time_and_one_market_batch(market, count):
    clock, repo = Clock(), Mock()
    def response(url, **_):
        clock.value += dt.timedelta(seconds=1)
        rows = ([dict(Date="20261016", SecuritiesCompanyCode="", CompanyName="", 暫停交易="", 恢復交易="")]
                if url.endswith("tpex_spendi_today") else [])
        return SimpleNamespace(payload=rows)
    job, http = coordinator(clock, client=Mock(get_json=Mock(side_effect=response)), repository=repo)
    ctx = context(market=market, observed=NOW - dt.timedelta(hours=5))
    job.ensure_current_restrictions(ctx)
    assert http.get_json.call_count == count
    expected = NOW + dt.timedelta(seconds=count)
    for item in ctx.resource.trading_restrictions.values():
        assert item["available_at"] == expected.isoformat()
        assert restriction_check(item, market=market, market_data_date="2026-10-15",
            decision_time=expected)["status"] == "CLEAR"
    assert repo.record.call_args.kwargs["decision_time"] == expected
    clock.value += dt.timedelta(hours=5)
    job.ensure_current_restrictions(ctx)  # No second recapture in the same job.
    assert http.get_json.call_count == count


def test_new_complete_restricted_observation_is_retained_not_replaced_with_clear():
    http = Mock(get_json=Mock(side_effect=lambda url, **_: SimpleNamespace(
        payload=[{"Code": "2330"}] if url.endswith("TWT85U") else [])))
    job, _ = coordinator(Clock(), client=http)
    ctx = context(observed=NOW - dt.timedelta(hours=5))
    job.ensure_current_restrictions(ctx)
    assert ctx.resource.trading_restrictions["2330"]["restricted"] is True
    assert ctx.resource.trading_restrictions["2303"]["restricted"] is False
    assert http.get_json.call_count == 3


@pytest.mark.parametrize("missing", [{}, None, "invalid"])
def test_missing_or_invalid_container_gets_one_complete_recapture(missing):
    job, http = coordinator(Clock())
    ctx = context()
    ctx.resource.trading_restrictions = missing
    job.ensure_current_restrictions(ctx)
    assert http.get_json.call_count == 3
    assert set(ctx.resource.trading_restrictions) == {"2330", "2303"}


def test_live_refresh_arms_the_late_hook_without_resetting_source_timestamps():
    clock = Clock(NOW - dt.timedelta(hours=5))
    job, http = coordinator(clock)
    ctx = context(armed=False)
    ctx.resource.stocknames = ["synthetic", "synthetic"]
    ctx.resource.daily_dates = [["2026-10-15"], ["2026-10-15"]]
    ctx.resource.set_low_entry_v4_fundamentals = Mock()
    class StopAfterCapture(Exception):
        pass
    job.fundamental_service.begin_run = Mock(side_effect=StopAfterCapture)
    with pytest.raises(StopAfterCapture):
        job.refresh(ctx, StockAnalysisRequest([], "ELEC", "SYNTHETIC_ONLY", 700), analysis_time=clock())
    original = ctx.resource.trading_restrictions
    assert http.get_json.call_count == 3
    assert ctx.resource._restriction_recapture_attempted is False
    clock.value = NOW
    job.ensure_current_restrictions(ctx)
    assert http.get_json.call_count == 6
    assert original["2330"]["available_at"] == (NOW - dt.timedelta(hours=5)).isoformat()
    assert ctx.resource.trading_restrictions["2330"]["available_at"] == NOW.isoformat()


def test_public_fallback_clock_is_read_after_final_http_not_request_start(monkeypatch):
    import infrastructure.runtime.fundamental_refresh as module
    readings = iter([NOW, NOW + dt.timedelta(seconds=3)])
    class ActualClock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(next(readings).timestamp(), tz)
    monkeypatch.setattr(module, "datetime", SimpleNamespace(datetime=ActualClock))
    job, http = coordinator(None)
    ctx = context(observed=NOW - dt.timedelta(hours=5))
    job.ensure_current_restrictions(ctx)
    assert http.get_json.call_count == 3
    assert ctx.resource.trading_restrictions["2330"]["available_at"] == (NOW + dt.timedelta(seconds=3)).isoformat()


@pytest.mark.parametrize("empty", ["universe", "bars"])
def test_no_universe_or_completed_bar_cannot_trigger_pointless_http(empty):
    job, http = coordinator(Clock())
    ctx = context(observed=NOW - dt.timedelta(hours=5))
    if empty == "universe":
        ctx.resource.stocknumbers = []
    else:
        ctx.resource.iso_scheduled_times = []
    job.ensure_current_restrictions(ctx)
    assert ctx.resource.trading_restrictions == {}
    http.get_json.assert_not_called()


@pytest.mark.parametrize("failure", ["network", "partial", "bad_source"])
def test_failed_or_partial_recapture_has_no_old_clear_fallback_or_retry(monkeypatch, failure):
    repo = Mock()
    job, http = coordinator(Clock(), repository=repo)
    if failure == "network":
        http.get_json.side_effect = TwseSchemaChangedError("synthetic unavailable", retryable=False)
    elif failure == "partial":
        http.get_json.side_effect = [SimpleNamespace(payload=[]), SimpleNamespace(payload={})]
    else:
        monkeypatch.setattr("infrastructure.crawler.trading_restrictions.TradingRestrictionClient.load",
            lambda *args, **kwargs: {"2330": dict(proof(), source="UNVERIFIED")})
    ctx = context(observed=NOW - dt.timedelta(hours=5))
    job.ensure_current_restrictions(ctx)
    assert ctx.resource.trading_restrictions == {}
    requests = http.get_json.call_count
    job.ensure_current_restrictions(ctx)
    assert http.get_json.call_count == requests
    repo.record.assert_not_called()


@pytest.mark.parametrize("completion", [NOW - dt.timedelta(seconds=1), NOW.replace(tzinfo=None),
    dt.datetime.fromisoformat("2026-10-17T00:00:01+08:00")])
def test_backwards_naive_or_cross_midnight_completion_cannot_make_fresh_evidence(completion):
    clock = Mock(side_effect=[NOW, completion])
    job, http = coordinator(clock)
    ctx = context(observed=NOW - dt.timedelta(hours=5))
    job.ensure_current_restrictions(ctx)
    assert http.get_json.call_count == 3
    assert ctx.resource.trading_restrictions == {}


def test_future_observation_is_replaced_only_by_a_real_current_capture():
    job, http = coordinator(Clock())
    ctx = context(observed=NOW + dt.timedelta(seconds=1))
    job.ensure_current_restrictions(ctx)
    assert http.get_json.call_count == 3
    assert all(row["available_at"] == NOW.isoformat() for row in ctx.resource.trading_restrictions.values())


@pytest.mark.parametrize("invalid", [None, NOW.replace(tzinfo=None)])
def test_invalid_precheck_clock_clears_evidence_without_http(invalid):
    job, http = coordinator(lambda: invalid)
    ctx = context()
    job.ensure_current_restrictions(ctx)
    assert ctx.resource.trading_restrictions == {}
    http.get_json.assert_not_called()


@pytest.mark.parametrize("options", [dict(replay=True), dict(armed=False), dict(native=False), dict(model="v6")])
def test_replay_skip_mops_and_legacy_paths_do_not_collect(options):
    job, http = coordinator(Clock())
    ctx = context(observed=NOW - dt.timedelta(hours=5), **options)
    original = ctx.resource.trading_restrictions
    job.ensure_current_restrictions(ctx)
    assert ctx.resource.trading_restrictions is original
    http.get_json.assert_not_called()


def test_reused_crawler_cannot_arm_a_new_skip_mops_context_from_an_old_job():
    job, http = coordinator(Clock())
    previous = context(observed=NOW - dt.timedelta(hours=5))
    current = AnalysisContext(previous.resource, NOW.date(), MarketStatus.OPEN)
    original = previous.resource.trading_restrictions
    job.ensure_current_restrictions(current)
    assert previous.resource.trading_restrictions is original
    http.get_json.assert_not_called()
    job.ensure_current_restrictions(previous)
    assert http.get_json.call_count == 3
    job.ensure_current_restrictions(previous)
    assert http.get_json.call_count == 3


def test_adapter_refreshes_before_one_cutoff_source_selection_and_decisions():
    clock = Clock()
    job, http = coordinator(clock)
    ctx = context(observed=NOW - dt.timedelta(hours=5))
    order = []
    def refresh(value):
        order.append("refresh")
        job.ensure_current_restrictions(value)
        clock.value += dt.timedelta(seconds=1)
    def select(crawler, *, cutoff):
        order.append("freeze_sources")
        assert dt.datetime.fromisoformat(cutoff) == clock.value
        assert all(dt.datetime.fromisoformat(x["available_at"]) < clock.value
                   for x in crawler.trading_restrictions.values())
        return {"verified": True}
    ctx.resource.les_v7_source_capture = SimpleNamespace(selections=select)
    ctx.resource.build_strategy_decisions = lambda _: order.append("decisions") or []
    adapter = CrawlerStrategyAdapter(portfolio_current_cash=1_000_000., restriction_refresh=refresh, now_provider=clock)
    adapter.evaluate(ctx)
    assert order == ["refresh", "freeze_sources", "decisions"]
    assert http.get_json.call_count == 3
    assert ctx.resource.les_v7_source_selections == {"verified": True}


@pytest.mark.parametrize("options", [dict(replay=True), dict(native=False), dict(model="v6")])
def test_adapter_never_invokes_live_hook_for_non_native_or_historical_work(options):
    hook = Mock()
    ctx = context(**options)
    CrawlerStrategyAdapter(portfolio_current_cash=1_000_000., restriction_refresh=hook).evaluate(ctx)
    hook.assert_not_called()


def test_future_final_observation_still_fails_existing_cutoff_gate():
    job, _ = coordinator(Mock(side_effect=[NOW, NOW + dt.timedelta(seconds=3)]))
    ctx = context(observed=NOW - dt.timedelta(hours=5))
    CrawlerStrategyAdapter(portfolio_current_cash=1_000_000., restriction_refresh=job.ensure_current_restrictions,
        now_provider=lambda: NOW).evaluate(ctx)
    assert restriction_check(ctx.resource.trading_restrictions["2330"], market_data_date="2026-10-15",
        decision_time=ctx.resource.les_v7_decision_time)["reason"] == "FUTURE_RESTRICTION_EVIDENCE"


@pytest.mark.parametrize("invalid", [None, NOW.replace(tzinfo=None)])
def test_adapter_rejects_invalid_final_clock_before_source_freeze(invalid):
    ctx = context()
    ctx.resource.les_v7_source_capture = Mock()
    with pytest.raises(ValueError, match="timezone-aware clock"):
        CrawlerStrategyAdapter(portfolio_current_cash=1_000_000., now_provider=lambda: invalid).evaluate(ctx)
    ctx.resource.les_v7_source_capture.selections.assert_not_called()
    ctx.resource.build_strategy_decisions.assert_not_called()


def test_production_bootstrap_shares_coordinator_and_actual_clock(monkeypatch, tmp_path):
    import infrastructure.bootstrap as bootstrap
    from config.settings import AppSettings
    captured = {}
    original = bootstrap.CrawlerStrategyAdapter
    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(bootstrap, "CrawlerStrategyAdapter", spy)
    clock = Clock()
    runtime = bootstrap.build_stock_analysis_runtime(crawler_factory=lambda _: SimpleNamespace(),
        fundamental_service=SimpleNamespace(), market_calendar=Mock(), http_client=Mock(),
        now_provider=clock, settings=AppSettings(mops_cache_dir=tmp_path / "mops",
            pae_persistence_path=tmp_path / "pae.sqlite3"))
    assert captured["restriction_refresh"].__self__.fundamental_service is runtime.fundamental_service
    assert captured["restriction_refresh"].__self__._now_provider is clock
    assert captured["now_provider"] is clock
