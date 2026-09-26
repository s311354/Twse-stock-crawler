"""Offline old-cache migrations, bounded backfill, and restart-safe cooldown."""

import datetime as dt
from types import SimpleNamespace

import pytest
import requests

from application.fundamental_service import MopsFundamentalService
from application.mops_refresh_control import MopsRefreshControl, MopsRefreshPolicy, RefreshDeferred
from config.settings import AppSettings, SettingsError
from domain.errors import AccessBlockedError, NoDataError
from domain.fundamental_models import FilingAnnouncement
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from domain.strategies.low_entry_v6_quality import evaluate_quality
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.fundamentals.mops_client import MopsClient
from infrastructure.fundamentals.mops_repository import MopsRepository
from tests.fundamentals.test_filing_refresh_planner import (
    _FakeMopsClient, _FakeFilingParser, _candidate, _fact,
)


SIGNAL = "2026-09-13T06:00:00+00:00"


class Clock:
    value = dt.datetime.fromisoformat(SIGNAL)

    def __call__(self):
        return self.value


def service(repo, client, clock, policy=None):
    result = MopsFundamentalService(
        client, repository=repo, filing_documents=client, filing_parser=_FakeFilingParser(),
        now_provider=clock, refresh_policy=policy,
    )
    result._company_sectors = {}
    return result


def seed(repo, symbol, year=2025, quarter=4):
    candidate = _candidate(year, quarter, symbol=symbol)
    repo.save_revision(
        candidate, f"{symbol}-{year}Q{quarter}".encode(), [_fact(1000.)],
        available_at="2026-02-15T05:00:00+00:00", fetched_at="2026-02-15T05:00:00+00:00",
        availability_source=candidate.availability_source,
    )


def load(svc, symbol):
    return svc.load(symbol, signal_date=SIGNAL, sector="24")


def test_existing_v6_cache_with_old_schema_is_migrated_locally_once(tmp_path, monkeypatch):
    repo = MopsRepository(tmp_path / "mops")
    seed(repo, "2330")
    client = _FakeMopsClient(repo, [], [])
    svc = service(repo, client, Clock())
    assert svc.migrate_local_cache(["2330"], signal_date=SIGNAL, sector="24") == 1
    real_load = repo.load_cached_bundles
    from dataclasses import replace
    def old_schema(*args, **kwargs):
        entries = real_load(*args, **kwargs)
        entry = entries["2330"]
        snapshot = dict(entry.bundle.snapshot, les_v6_filing_schema_version="old")
        entries["2330"] = replace(entry, bundle=replace(entry.bundle, snapshot=snapshot))
        return entries
    migrated = []
    real_migrate = svc._load_cached_bundle
    def record_migration(*args, **kwargs):
        migrated.append(args[0])
        return real_migrate(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(repo, "load_cached_bundles", old_schema)
        patch.setattr(svc, "_load_cached_bundle", record_migration)
        assert svc.migrate_local_cache(["2330"], signal_date=SIGNAL, sector="24") == 1
    assert migrated == ["2330"]
    assert svc.migrate_local_cache(["2330"], signal_date=SIGNAL, sector="24") == 0
    entries = real_load({"2330": "24"}, as_of=SIGNAL,
                        derivation_version=FUNDAMENTAL_DERIVATION_VERSION)
    assert entries["2330"].bundle.snapshot["les_v6_filing_schema_version"] == V6_FILING_SCHEMA_VERSION
    assert client.discovery_calls == 0


def test_233_legacy_raw_caches_migrate_before_remote_and_breaker_survives_restart(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    symbols = [str(4000 + i) for i in range(233)]
    for symbol in symbols:
        seed(repo, symbol)

    class Blocked(_FakeMopsClient):
        def discover(self, symbol):
            self.discovery_calls += 1
            raise AccessBlockedError("MOPS security guard")

    clock = Clock()
    client = Blocked(repo, [], [])
    svc = service(repo, client, clock)
    assert svc.migrate_local_cache(symbols, signal_date=SIGNAL, sector="24") == 233
    assert client.discovery_calls == 0
    for symbol in symbols:
        bundle = load(svc, symbol)
        assert len(bundle.snapshot["les_v6_mops_filings"]) == 1
        quality = evaluate_quality(
            bundle.snapshot["les_v6_mops_filings"], symbol=symbol, market="TWSE", as_of=SIGNAL,
        )
        assert quality.status != "PASS"
    assert client.discovery_calls == 2
    assert svc.refresh_control.circuit_open
    assert svc.refresh_control.deferred["ACCESS_BLOCKED_CIRCUIT_OPEN"] == 231
    assert svc.consume_access_blocked_cache_fallback_symbols() == tuple(symbols[:2])

    # A fresh process/service must not forget host cooldown or re-request each issuer.
    restarted_client = Blocked(repo, [], [])
    restarted = service(MopsRepository(repo.root), restarted_client, clock)
    blocked_at = repo.load_refresh_attempt(MopsRefreshControl.HOST_KEY)[0]
    clock.value += dt.timedelta(hours=12)
    with pytest.raises(RefreshDeferred):
        restarted.prepare_daily_revisions(clock())
    assert repo.load_refresh_attempt(MopsRefreshControl.HOST_KEY)[0] == blocked_at
    assert load(restarted, symbols[-1]).snapshot["remote_refresh_status"] == "ACCESS_BLOCKED_CIRCUIT_OPEN"
    assert restarted_client.discovery_calls == 0
    clock.value += dt.timedelta(days=1, seconds=1)
    resumed = service(repo, restarted_client, clock)
    load(resumed, symbols[-1])
    assert restarted_client.discovery_calls == 1


def test_missing_quarters_are_downloaded_in_batches_and_cooldown_persists(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    for year in range(2021, 2026):
        seed(repo, "2330", year)
    candidates = [_candidate(year, 4) for year in range(2021, 2026)] + [
        _candidate(year, quarter) for year in (2024, 2025) for quarter in (1, 2, 3)
    ]
    clock = Clock()
    policy = MopsRefreshPolicy(max_filings_per_symbol=2, cooldown_seconds=60)
    announcements = [FilingAnnouncement(
        symbol="2330", fiscal_year=item.fiscal_year, quarter=item.quarter,
        available_at=item.available_at, title="財務報告",
    ) for item in candidates]
    client = _FakeMopsClient(repo, candidates, announcements)
    load(service(repo, client, clock, policy), "2330")
    assert client.downloaded_periods == [(2025, 3), (2025, 2)]
    assert repo.last_full_refresh_epoch("2330") in (None, 0)
    assert repo.load_refresh_attempt("2330")[1] == "PARTIAL"
    load(service(MopsRepository(repo.root), client, clock, policy), "2330")
    assert client.discovery_calls == 1
    clock.value += dt.timedelta(seconds=61)
    load(service(repo, client, clock, policy), "2330")
    assert client.discovery_calls == 1  # Resume the persisted catalog across restarts.
    assert client.downloaded_periods == [(2025, 3), (2025, 2), (2025, 1), (2024, 3)]


def test_symbol_budget_advances_to_next_batch_and_negative_cache_is_persistent(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    class Empty(_FakeMopsClient):
        def discover(self, symbol):
            self.discovery_calls += 1
            raise NoDataError("No eligible remote filings")
    clock = Clock()
    client = Empty(repo, [], [])
    policy = MopsRefreshPolicy(max_symbols=2)
    first = service(repo, client, clock, policy)
    for symbol in ("4001", "4002"):
        with pytest.raises(NoDataError, match="No eligible"):
            load(first, symbol)
        assert repo.load_refresh_attempt(symbol) is not None
    with pytest.raises(RefreshDeferred, match="SYMBOL_BUDGET"):
        load(first, "4003")
    assert repo.load_refresh_attempt("4003") is None
    assert client.discovery_calls == 2
    second = service(MopsRepository(repo.root), client, clock, policy)
    with pytest.raises(RefreshDeferred, match="REFRESH_COOLDOWN"):
        load(second, "4001")
    with pytest.raises(NoDataError, match="No eligible"):
        load(second, "4003")
    assert client.discovery_calls == 3


def test_daily_schedule_prefers_unattempted_symbols_even_after_cooldown(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    clock = Clock()
    repo.save_refresh_attempt("4001", clock().timestamp() - 100000, "PARTIAL")
    control = MopsRefreshControl(repo, MopsRefreshPolicy(max_symbols=1), clock)
    control.schedule(["4001", "4002"])
    assert control.scheduled_symbols == {"4002"}
    with repo.unit_of_work() as uow, pytest.raises(RefreshDeferred, match="SYMBOL_BATCH_DEFERRED"):
        control.start_symbol("4001", uow=uow)


def test_empty_catalog_cools_down_without_claiming_complete_refresh(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    client = _FakeMopsClient(repo, [], [])
    with pytest.raises(NoDataError, match="no point-in-time eligible"):
        load(service(repo, client, Clock()), "2330")
    assert repo.last_full_refresh_epoch("2330") is None
    assert repo.load_refresh_attempt("2330")[1] == "NO_ELIGIBLE_FILINGS"


def test_priority_backfill_reserves_half_the_batch_for_general_rotation(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    control = MopsRefreshControl(repo, MopsRefreshPolicy(max_symbols=4), Clock())
    control.schedule(["4001", "4002", "4003", "4004", "4005", "4006"],
                     priority_symbols=["4004", "4005", "4006"])
    assert control.scheduled_order == ["4004", "4001", "4005", "4002"]


def test_one_symbol_priority_batch_alternates_across_restarts_and_keeps_cooldown(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    clock = Clock()
    policy = MopsRefreshPolicy(max_symbols=1)
    for expected in ("4003", "4001", "4004", "4002"):
        control = MopsRefreshControl(MopsRepository(repo.root), policy, clock)
        control.schedule(["4001", "4002", "4003", "4004"], priority_symbols=["4003", "4004"])
        assert control.scheduled_order == [expected]
        with control.repository.unit_of_work() as uow:
            control.start_symbol(expected, uow=uow)
            control.finish(expected, "PARTIAL", uow=uow)
        # Fresh service, same wall clock: attempted names remain cooling down.
    control = MopsRefreshControl(repo, policy, clock)
    control.schedule(["4001", "4002", "4003", "4004"], priority_symbols=["4003", "4004"])
    assert control.scheduled_order == []


def test_priority_never_bypasses_request_budget_or_host_breaker(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    clock = Clock()
    control = MopsRefreshControl(repo, MopsRefreshPolicy(max_requests=0), clock)
    control.schedule(["2330"], priority_symbols=["2330"])
    with repo.unit_of_work() as uow, pytest.raises(RefreshDeferred, match="REQUEST_BUDGET_EXHAUSTED"):
        control.start_symbol("2330", uow=uow)
    repo.save_refresh_attempt(control.HOST_KEY, clock().timestamp(), "ACCESS_BLOCKED")
    control = MopsRefreshControl(repo, MopsRefreshPolicy(), clock)
    control.schedule(["2330"], priority_symbols=["2330"])
    with repo.unit_of_work() as uow, pytest.raises(RefreshDeferred, match="CIRCUIT_OPEN"):
        control.start_symbol("2330", uow=uow)


def test_runtime_screens_then_inspects_local_evidence_before_priority_backfill(tmp_path):
    from application.contracts import AnalysisContext, MarketStatus, StockAnalysisRequest
    from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator
    from tests.strategies.test_leadership_verification import watch_histories

    histories = watch_histories(True)
    # Keep the strong candidate at the back of input to prove real request order.
    histories = histories[1:] + histories[:1]
    symbols = [h.attrs["les_v6_daily_identity"]["symbol"] for h in histories]
    repo = MopsRepository(tmp_path / "mops")
    calls = []

    class Empty(_FakeMopsClient):
        def discover(self, symbol):
            calls.append(symbol)
            raise NoDataError("No eligible filings")

    svc = service(repo, Empty(repo, [], []), Clock(), MopsRefreshPolicy(max_symbols=2))
    inspect = svc.inspect_local_evidence
    def checked_inspection(symbols, **kwargs):
        assert calls == []
        result = inspect(symbols, **kwargs)
        assert result["2330"]["backfill_needed"]
        return result
    svc.inspect_local_evidence = checked_inspection
    crawler = SimpleNamespace(
        stocknumbers=symbols, stocknames=["測試"] * 4,
        daily_dates=[h["Date"].dt.strftime("%Y-%m-%d").tolist() for h in histories],
        build_strategy_history=lambda item: histories[item],
        set_low_entry_v4_fundamentals=lambda *args, **kwargs: None,
    )
    context = AnalysisContext(crawler, dt.date(2026, 9, 11), MarketStatus.CLOSED)
    FundamentalRefreshCoordinator(svc).refresh(context, StockAnalysisRequest([], "ELEC", "test", 500),
                                              analysis_time=dt.datetime.fromisoformat("2026-09-11T15:00:00+08:00"))
    assert calls == ["2330", "2454"]
    assert crawler.les_v6_research_context["2330"]["before_refresh"]["quality_status"] == "DATA_INSUFFICIENT"


@pytest.mark.parametrize("market", ["TWSE", "TPEX"])
def test_runtime_migrates_all_raw_cache_before_revision_or_catalog_requests(tmp_path, market):
    from application.contracts import AnalysisContext, MarketStatus, StockAnalysisRequest
    from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator

    repo = MopsRepository(tmp_path / "mops")
    symbols = ["4001", "4002", "4003"]
    for symbol in symbols:
        seed(repo, symbol)
    # Expired attempts at the front must not consume the next batch first.
    for symbol in symbols[:2]:
        repo.save_refresh_attempt(symbol, Clock()().timestamp() - 100000, "PARTIAL")
    events = []
    class Checked(_FakeMopsClient):
        def discover(self, symbol):
            events.append(symbol)
            return super().discover(symbol)

        def discover_corrections(self, periods):
            bundles = repo.load_cached_bundles(
                {symbol: "24" for symbol in symbols}, as_of=SIGNAL,
                derivation_version=FUNDAMENTAL_DERIVATION_VERSION,
            )
            assert set(bundles) == set(symbols)
            assert all("les_v6_mops_filings" in entry.bundle.snapshot for entry in bundles.values())
            events.append("remote")
            return []
    snapshots = {}
    crawler = SimpleNamespace(
        stocknumbers=symbols, stocknames=["" for _ in symbols], daily_dates=[["2026-09-11"] for _ in symbols],
        set_low_entry_v4_fundamentals=lambda symbol, **kwargs: snapshots.update({symbol: kwargs["fundamental_snapshot"]}),
    )
    client = Checked(repo, [], [])
    svc = service(repo, client, Clock(), MopsRefreshPolicy(max_symbols=1))
    context = AnalysisContext(crawler, dt.date(2026, 9, 11), MarketStatus.CLOSED)
    from domain.models import Stocktype
    request = StockAnalysisRequest([], Stocktype.SEMI if market == "TPEX" else "24", "test", 500)
    FundamentalRefreshCoordinator(svc).refresh(context, request, analysis_time=Clock()())
    assert events == ["remote", "4003"]
    assert len(snapshots) == 3 and client.discovery_calls == 1
    assert snapshots["4002"]["remote_refresh_status"] == "SYMBOL_BUDGET_EXHAUSTED"
    if market == "TPEX":
        assert all(snapshot["les_v6_filings"][0]["market"] == "TPEX" for snapshot in snapshots.values())


def test_otc_local_evidence_uses_mops_without_twse_batch_contamination(monkeypatch):
    from contextlib import nullcontext
    from unittest.mock import Mock
    from tests.strategies.test_low_entry_score_v6 import financials

    records = [dict(row, symbol="3105", source="MOPS_IXBRL") for row in financials()]
    repo = Mock()
    repo.unit_of_work.side_effect = lambda: nullcontext(None)
    repo.load_filed_financials.return_value = records
    # Must not even read the TWSE-only batch table for an OTC issuer.
    repo.load_financial_batch_snapshots.side_effect = AssertionError("TWSE batch read for TPEX")
    monkeypatch.setattr("application.fundamental_service.ixbrl_record", lambda symbol, row: dict(row))
    svc = service(repo, _FakeMopsClient(repo, [], []), Clock())

    result = svc.inspect_local_evidence(["3105"], signal_date=SIGNAL, market="TPEX")["3105"]

    assert result["quality_status"] == "PASS"
    assert not result["local_review_required"]
    assert not result["backfill_needed"]
    assert all(row["market"] == "TWSE" for row in records)  # Caller-owned cache unchanged.
    repo.load_financial_batch_snapshots.assert_not_called()


def test_otc_missing_filings_remain_insufficient_despite_twse_batch():
    from contextlib import nullcontext
    from unittest.mock import Mock

    repo = Mock()
    repo.unit_of_work.side_effect = lambda: nullcontext(None)
    repo.load_filed_financials.return_value = []
    svc = service(repo, _FakeMopsClient(repo, [], []), Clock())
    result = svc.inspect_local_evidence(["3105"], signal_date=SIGNAL, market="TPEX")["3105"]

    assert result["quality_status"] == "DATA_INSUFFICIENT"
    assert result["backfill_needed"]
    assert result["quality_reasons"] == ["NO_POINT_IN_TIME_FINANCIALS"]
    repo.load_financial_batch_snapshots.assert_not_called()


def test_zero_request_budget_converts_cache_without_remote_or_fabricating_facts(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    seed(repo, "2330")
    client = _FakeMopsClient(repo, [], [])
    svc = service(repo, client, Clock(), MopsRefreshPolicy(max_requests=0))
    bundle = load(svc, "2330")
    assert client.discovery_calls == 0
    assert bundle.snapshot["remote_refresh_status"] == "REQUEST_BUDGET_EXHAUSTED"
    record = bundle.snapshot["les_v6_mops_filings"][0]
    assert record["available_at"] == "2026-02-15T05:00:00+00:00"
    assert record.get("cfo") is None
    assert repo.last_full_refresh_epoch("2330") == 0


def test_partial_download_survives_later_budget_exhaustion_without_full_refresh(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    candidates = [_candidate(2025, quarter) for quarter in (1, 2, 3, 4)]
    announcements = [FilingAnnouncement(
        symbol="2330", fiscal_year=item.fiscal_year, quarter=item.quarter,
        available_at=item.available_at, title="財務報告",
    ) for item in candidates]
    class Interrupted(_FakeMopsClient):
        def download(self, candidate):
            if self.downloaded_periods:
                raise RefreshDeferred("REQUEST_BUDGET_EXHAUSTED")
            return super().download(candidate)
    client = Interrupted(repo, candidates, announcements)
    clock = Clock()
    result = load(service(repo, client, clock), "2330")
    assert result.snapshot["remote_refresh_status"] == "REQUEST_BUDGET_EXHAUSTED"
    assert len(repo.load_filings("2330")) == 1
    assert repo.last_full_refresh_epoch("2330") == 0
    load(service(MopsRepository(repo.root), client, clock), "2330")
    assert client.discovery_calls == 1


class FakeSession:
    def __init__(self, outcomes):
        self.headers = {}
        self.outcomes = iter(outcomes)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        value = next(self.outcomes)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, SimpleNamespace) else SimpleNamespace(status_code=200, content=value)

    post = get

    def close(self):
        pass


def controlled_client(repo, session, *, max_requests=3):
    control = MopsRefreshControl(repo, MopsRefreshPolicy(max_requests=max_requests), Clock())
    http = TwseHttpClient(
        config=TwseHttpConfig(max_attempts=4, min_request_interval_seconds=0, backoff_seconds=0),
        session=session, sleeper=lambda seconds: None,
    )
    client = MopsClient(
        http_client=http, refresh_control=control, network_rounds=2, network_backoff_seconds=0,
        catalog_empty_attempts=2, catalog_empty_backoff_seconds=0, min_request_interval_seconds=0,
        sleeper=lambda seconds: None,
    )
    return client, control


def test_request_budget_counts_transport_retries_and_host_failover(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    session = FakeSession([requests.exceptions.ConnectionError("offline")] * 20)
    client, control = controlled_client(repo, session, max_requests=5)
    with pytest.raises(RefreshDeferred, match="REQUEST_BUDGET"):
        client.discover("2330")
    assert session.calls == control.requests == 5


def test_ssl_fallback_counts_against_budget(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    session = FakeSession([requests.exceptions.SSLError("Missing Subject Key Identifier"), b"<table/>"])
    client, control = controlled_client(repo, session, max_requests=1)
    with pytest.raises(RefreshDeferred, match="REQUEST_BUDGET"):
        client.discover("2330")
    assert session.calls == control.requests == 1


def test_redirects_cannot_bypass_request_budget(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    redirect = SimpleNamespace(status_code=302, headers={"Location": "/next"}, content=b"")
    session = FakeSession([redirect] * 10)
    client, control = controlled_client(repo, session, max_requests=2)
    with pytest.raises(RefreshDeferred, match="REQUEST_BUDGET"):
        client.discover("2330")
    assert session.calls == control.requests == 2


@pytest.mark.parametrize("endpoint", ["catalog", "corrections", "announcements", "download"])
def test_guard_pages_at_every_mops_endpoint_trip_without_retry_storm(tmp_path, endpoint):
    repo = MopsRepository(tmp_path / "mops")
    guard = "<html>因為安全性考量，所執行的頁面無法呈現</html>".encode()
    session = FakeSession([guard] * 10)
    client, control = controlled_client(repo, session)
    def invoke():
        if endpoint == "catalog":
            return client.discover("2330")
        if endpoint == "corrections":
            return client.discover_corrections([(2025, 4)])
        if endpoint == "announcements":
            return client.discover_announcements("2330", [2026])
        return client.download(_candidate(2025, 4))
    for _ in range(2):
        with pytest.raises(AccessBlockedError):
            invoke()
    with pytest.raises(RefreshDeferred, match="CIRCUIT_OPEN"):
        invoke()
    assert session.calls == control.requests == 2


def test_success_resets_consecutive_guard_counter(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    guard = "因為安全性考量".encode()
    session = FakeSession([guard, b"<table/>", guard])
    client, control = controlled_client(repo, session, max_requests=10)
    with pytest.raises(AccessBlockedError):
        client.discover_corrections([(2025, 1)])
    client.discover_corrections([(2025, 2)])
    with pytest.raises(AccessBlockedError):
        client.discover_corrections([(2025, 3)])
    assert control.blocked_streak == 1 and not control.circuit_open


def test_real_ixbrl_security_discussion_is_not_a_guard_page(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    filing = '<html><ix:header/><ix:nonfraction>1</ix:nonfraction>因為安全性考量</html>'.encode()
    client, control = controlled_client(repo, FakeSession([filing]))
    assert client.download(_candidate(2025, 4)).content == filing
    assert control.blocked_streak == 0


def test_settings_expose_bounded_defaults_and_offline_mode():
    defaults = AppSettings.from_env({}, validate_paths=False)
    assert defaults.mops_refresh_max_symbols == 20
    assert defaults.mops_refresh_max_requests == 120
    assert defaults.mops_refresh_max_filings_per_symbol == 4
    assert defaults.mops_refresh_cooldown_seconds == 86400
    assert defaults.mops_access_blocked_threshold == 2
    offline = AppSettings.from_env({"MOPS_REFRESH_MAX_REQUESTS": "0"}, validate_paths=False)
    assert offline.mops_refresh_max_requests == 0
    with pytest.raises(SettingsError):
        AppSettings.from_env({"MOPS_REFRESH_MAX_SYMBOLS": "-1"}, validate_paths=False)
