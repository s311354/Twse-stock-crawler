"""No future CLEAR, replay writes, cross-market borrowing or stale risk ratios."""

import datetime as dt
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from application.contracts import AnalysisContext, MarketStatus
from domain.trading_restrictions import RESTRICTION_CONTRACTS
from domain.strategies import low_entry_v6_signals as signals
from infrastructure.runtime.market_data_adapter import CrawlerMarketDataAdapter
from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator
from infrastructure.storage.trading_restriction_repository import TradingRestrictionRepository
from tests.test_historical_replay import request, WeekdayCalendar
from tests.strategies.test_low_entry_score_v6 import CONFIG, setup_features


def evidence(at="2026-09-16T13:29:00+08:00", *, market="TWSE", restricted=False):
    source, endpoints = RESTRICTION_CONTRACTS[market]
    day = dt.datetime.fromisoformat(at).astimezone(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    return dict(market=market, source=source, endpoints=list(endpoints), restricted=restricted,
                session_date=day, observation_date=day, scope="CURRENT_AS_OF", available_at=at)


def record(repo, item):
    return repo.record({"3016": item}, market=item["market"], market_data_date="2026-09-15",
                       decision_time=item["available_at"])


def load(repo, cutoff="2026-09-16T13:30:00+08:00", market="TWSE"):
    return repo.load_as_of(market=market, market_data_date="2026-09-15", as_of=cutoff)


def test_replay_missing_archive_does_not_create_database(tmp_path):
    repo = TradingRestrictionRepository(tmp_path / "missing" / "restrictions.sqlite3")
    assert load(repo) == {}
    assert not repo.path.parent.exists()
    with pytest.raises(ValueError, match="AWARE"):
        load(repo, "2026-09-16")


def test_latest_asof_market_isolation_and_no_replay_mutation(tmp_path):
    repo = TradingRestrictionRepository(tmp_path / "restrictions.sqlite3")
    old = evidence("2026-09-16T12:00:00+08:00")
    current = evidence(restricted=True)
    future = evidence("2026-09-16T13:30:00.000001+08:00")
    for item in (old, current, future, evidence(market="TPEX"), evidence("2026-09-26T13:29:00+08:00")):
        assert record(repo, item) == 1
    assert record(repo, current) == 0
    before = repo.path.read_bytes()
    assert load(repo)["3016"] == current
    assert load(repo, market="TPEX")["3016"]["restricted"] is False
    assert load(repo, "2026-09-16T13:30:01+08:00")["3016"] == future
    assert load(repo, "2026-09-16T11:59:00+08:00") == {}
    assert load(repo, "2026-09-17T13:30:00+08:00") == {}
    assert repo.path.read_bytes() == before


def test_conflicting_or_corrupt_latest_record_cannot_fallback_to_clear(tmp_path):
    repo = TradingRestrictionRepository(tmp_path / "restrictions.sqlite3")
    record(repo, evidence("2026-09-16T12:00:00+08:00"))
    record(repo, evidence())
    record(repo, evidence(restricted=True))
    assert load(repo) == {}
    record(repo, evidence("2026-09-16T13:29:59+08:00"))
    with sqlite3.connect(repo.path) as connection:
        connection.execute("UPDATE restriction_observations SET evidence_json='{}' WHERE available_at_ns=(SELECT MAX(available_at_ns) FROM restriction_observations)")
    assert load(repo) == {}


@pytest.mark.parametrize("patch", [dict(endpoints=[]), dict(source="UNVERIFIED"),
                                    dict(restricted=None), dict(available_at="2026-09-16T13:29:00")])
def test_invalid_evidence_is_not_archived(tmp_path, patch):
    repo = TradingRestrictionRepository(tmp_path / "restrictions.sqlite3")
    assert repo.record({"3016": dict(evidence(), **patch)}, market="TWSE", market_data_date="2026-09-15",
                       decision_time="2026-09-16T13:30:00+08:00") == 0
    assert not repo.path.exists()


def test_freshness_boundary_and_same_day_rule(tmp_path):
    repo = TradingRestrictionRepository(tmp_path / "restrictions.sqlite3")
    record(repo, evidence("2026-09-16T09:30:00+08:00"))
    assert "3016" in load(repo)
    assert load(repo, "2026-09-16T13:30:00.000001+08:00") == {}
    record(repo, evidence("2026-09-16T23:59:00+08:00"))
    assert load(repo, "2026-09-17T00:01:00+08:00") == {}


def test_replay_adapter_automatically_reads_archive_and_explicit_file_overrides(tmp_path):
    repo = TradingRestrictionRepository(tmp_path / "restrictions.sqlite3")
    record(repo, evidence())
    crawler = SimpleNamespace(iso_scheduled_times=["2026-09-16"], stocknumbers=["3016"],
                              get_date_times=Mock(), get_twse_daily_stocks=Mock())
    adapter = CrawlerMarketDataAdapter(lambda count: crawler, restriction_repository=repo)
    def fetch(req):
        adapter.fetch(req, calendar=WeekdayCalendar(), market_date=req.as_of.date(),
                      is_trading_day=True, include_reference=True)
    fetch(request(mops_enabled=False))
    assert crawler.trading_restrictions == {"3016": evidence()}
    override = tmp_path / "explicit.json"
    override.write_text("{}")
    fetch(request(replay_restrictions_file=str(override)))
    assert crawler.trading_restrictions == {}  # no silent merge with a CLEAR cache
    with sqlite3.connect(repo.path) as connection:
        connection.execute("DROP TABLE restriction_observations")
    fetch(request())
    assert crawler.trading_restrictions == {}


def test_live_collection_archives_at_response_completion_not_request_start(tmp_path):
    repo = TradingRestrictionRepository(tmp_path / "restrictions.sqlite3")
    service = SimpleNamespace(financial_batch_client=SimpleNamespace(
        http_client=SimpleNamespace(get_json=lambda *a, **k: SimpleNamespace(payload=[]))))
    crawler = SimpleNamespace(stocknumbers=["3016"], stocknames=["test"], daily_dates=[[]],
                              iso_scheduled_times=["2026-09-16"], set_low_entry_v4_fundamentals=Mock())
    class Collected(Exception):
        pass
    service.begin_run = Mock(side_effect=Collected)
    finished = dt.datetime.fromisoformat("2026-09-16T13:30:01+08:00")
    coordinator = FundamentalRefreshCoordinator(service, restriction_repository=repo,
                                                now_provider=lambda: finished)
    from dataclasses import replace
    req = replace(request(), as_of=None)
    with pytest.raises(Collected):
        coordinator.refresh(AnalysisContext(crawler, finished.date(), MarketStatus.OPEN), req,
                            analysis_time=finished - dt.timedelta(seconds=1))
    assert load(repo) == {}  # not available at the 13:30 strict replay boundary
    assert load(repo, finished)["3016"]["available_at"] == finished.isoformat()


@pytest.mark.parametrize("setup", signals.SETUPS)
@pytest.mark.parametrize("patch", [dict(stop_distance_atr=.1), dict(stop_distance_fraction=.001),
                                    dict(entry_price=115.), dict(stop_price=50.)])
def test_stop_gate_recomputes_prices_instead_of_trusting_cached_ratios(setup, patch):
    f = setup_features(setup, swing_high=140., hh20=145., hh60=150.)
    plan = signals.trade_plan(f, config=CONFIG)
    assert signals.risk_pass(setup, "BUY_SMALL", f, plan, "NEUTRAL", CONFIG)[0]
    assert not signals.risk_pass(setup, "BUY_SMALL", f, dict(plan, **patch), "NEUTRAL", CONFIG)[0]


@pytest.mark.parametrize("atr", [None, 0., -1., float("nan"), float("inf")])
def test_missing_or_invalid_atr_cannot_reuse_old_stop_ratios(atr):
    f = setup_features("LEADERSHIP_CONTINUATION")
    plan = signals.trade_plan(f, config=CONFIG)
    assert not signals.risk_pass("LEADERSHIP_CONTINUATION", "BUY_SMALL", dict(f, atr14=atr), plan,
                                 "NEUTRAL", CONFIG)[0]
