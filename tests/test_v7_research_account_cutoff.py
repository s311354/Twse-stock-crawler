"""SYNTHETIC ONLY: native account/cutoff wiring; no state, mail or orders."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from application.contracts import AnalysisContext, MarketStatus
from application.low_entry_v7_daily import daily_report, finalize_daily_report
from application.market_history import MarketHistoryAssembler
from domain.les_v7.contracts import canonical
from domain.les_v7.fr2_policy import restore_original
from domain.les_v7.report import validate_report
from domain.models import PortfolioPosition
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from infrastructure.legacy_crawler_facade import LegacyCrawlerFacade
from infrastructure.runtime.strategy_adapter import CrawlerStrategyAdapter
from tests.test_v7_fr2_eps_daily import research_history, research_market


def _crawler(source):
    """Adapt synthetic collected inputs through the real legacy facade port."""
    fields = {
        "stocklistsize": "stock_count", "stocknumbers": "stock_numbers",
        "stocknames": "stock_names", "daily_dates": "dates", "daily_stocks": "opens",
        "daily_highs": "highs", "daily_lows": "lows", "daily_closes": "closes",
        "daily_volumes": "volumes", "daily_trade_counts": "trade_counts",
        "daily_trade_values": "trade_values", "daily_bid_prices": "bids",
        "daily_ask_prices": "asks", "daily_pe_ratios": "pe_ratios",
        "daily_taiex_closes": "taiex_closes", "annual_fundamentals_by_stock": "annual_fundamentals",
        "fundamental_snapshots_by_stock": "fundamental_snapshots",
        "fundamental_status_by_stock": "fundamental_statuses",
        "fundamental_status_detail_by_stock": "fundamental_status_details",
        "low_entry_v4_signal_dates_by_stock": "signal_dates",
        "low_entry_v4_account_equity": "account_equity",
        "trading_restrictions": "trading_restrictions",
        "market_breadth_snapshots_by_date": "market_breadth_snapshots",
        "market_observed_at": "market_observed_at", "market_source": "market_source",
    }
    crawler = SimpleNamespace(**{target: getattr(source, original) for target, original in fields.items()})
    crawler.app_settings = SimpleNamespace(daily_entry_model=source.entry_model,
        **{key: getattr(source, key) for key in (
            "les_v7_bundle_json", "les_v7_research_metadata_json", "les_v7_strategy_policy_json",
            "les_v7_diagnostics_only", "les_v7_native_evidence", "les_v7_rule_baseline")})
    crawler.iso_scheduled_times = [source.fallback_date]
    crawler.replay_as_of = source.replay_as_of
    crawler.analysis_universe = source.default_sector
    crawler.cal_max_profit = Mock(return_value=[])
    return crawler


def _account(cutoff):
    return dict(value=1_000_000., available_cash=1_000_000., intended_weight=.25,
                capacity_weight=.25, positions=[], as_of=cutoff)


def test_native_adapter_facade_research_uses_the_one_frozen_account_cutoff():
    source = research_market()
    cutoff = datetime.fromisoformat(source.signal_dates["2330"])
    canonical_cutoff = cutoff.astimezone(timezone.utc).isoformat()
    clock = Mock(return_value=cutoff)
    crawler = _crawler(source)
    captured = {}

    def build(_):
        market = LegacyCrawlerFacade._market_history_input(crawler)
        history = MarketHistoryAssembler().build(market, 0)
        captured.update(history=history, report=daily_report(history))
        return ()

    crawler.build_strategy_decisions = build
    CrawlerStrategyAdapter(portfolio_current_cash=1_000_000., now_provider=clock).evaluate(
        AnalysisContext(crawler, cutoff.date(), MarketStatus.OPEN))
    clock.assert_called_once_with()
    assert crawler.les_v6_portfolio == _account(canonical_cutoff)
    assert crawler.les_v7_decision_time == captured["history"].attrs[SIGNAL_DATE_ATTR] == canonical_cutoff
    report = captured["report"]
    validate_report(report)
    execution = report["fr2_execution"]
    assert execution["supplied_account"] == crawler.les_v6_portfolio
    assert "ACCOUNT_INPUTS_UNKNOWN" not in execution["reason_codes"]
    assert not any(reason.startswith("ACCOUNT_INVALID:") for reason in execution["reason_codes"])
    # A timestamp is not a safety/Quality/plan override or an authorization.
    assert not execution["allocation_authorized"]
    assert not execution["live_order_authorized"] and execution["filled_shares"] == 0
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}

    # Supplying the identical explicit account through the public daily entry
    # must reproduce the entire result, not only a selected score subset.
    history = captured["history"]
    history.attrs["les_v6_portfolio"] = _account(canonical_cutoff)
    assert canonical(daily_report(history)) == canonical(report)

    # Reconstruct the pre-fix failure without rerunning or changing any model:
    # only the execution receipt differs; native result/admission/score do not.
    missing_time = dict(crawler.les_v6_portfolio)
    missing_time.pop("as_of")
    old = finalize_daily_report(deepcopy(report), account=missing_time)
    assert "ACCOUNT_INVALID:'as_of'" in old["fr2_execution"]["reason_codes"]
    assert restore_original(old) == restore_original(report)
    assert old["diagnostics"]["fr2"]["admission"] == report["diagnostics"]["fr2"]["admission"]
    assert {k: v for k, v in old.items() if k != "fr2_execution"} == {
        k: v for k, v in report.items() if k != "fr2_execution"}


@pytest.mark.parametrize("missing", ["as_of", "available_cash"])
def test_external_daily_accounts_are_not_timestamped_or_completed(missing):
    history = research_history()
    account = _account(history.attrs[SIGNAL_DATE_ATTR])
    account.pop(missing)
    history.attrs["les_v6_portfolio"] = account
    before = deepcopy(account)
    report = daily_report(history)
    validate_report(report)
    assert report["fr2_execution"]["supplied_account"] == before
    assert "ACCOUNT_INPUTS_UNKNOWN" in report["fr2_execution"]["reason_codes"]
    assert not report["fr2_execution"]["allocation_authorized"]
    assert history.attrs["les_v6_portfolio"] == before


def test_external_daily_account_cutoff_mismatch_remains_rejected():
    history = research_history(account=_account("2026-10-13T07:00:00+00:00"))
    report = daily_report(history)
    validate_report(report)
    assert "ACCOUNT_INVALID:FR2_INVALID_ACCOUNT_OR_CUTOFF" in report["fr2_execution"]["reason_codes"]
    assert not report["fr2_execution"]["allocation_authorized"]


def test_native_account_with_holdings_does_not_invent_existing_position_risk_proof():
    source = research_market()
    cutoff = datetime.fromisoformat(source.signal_dates["2330"])
    crawler = _crawler(source)
    reports = []

    def build(_):
        market = LegacyCrawlerFacade._market_history_input(crawler)
        reports.append(daily_report(MarketHistoryAssembler().build(market, 0)))
        return ()

    crawler.build_strategy_decisions = build
    CrawlerStrategyAdapter(portfolio_positions=(PortfolioPosition("2330", shares=10),),
        portfolio_current_cash=1_000_000. - 10 * source.closes[0][-1] / 1000.,
        now_provider=lambda: cutoff).evaluate(
            AnalysisContext(crawler, cutoff.date(), MarketStatus.OPEN))
    report = reports[0]
    validate_report(report)
    execution = report["fr2_execution"]
    assert execution["supplied_account"]["positions"] == [{"symbol": "2330", "shares": 10}]
    assert "ACCOUNT_INPUTS_UNKNOWN" not in execution["reason_codes"]
    assert "EXISTING_POSITION_RISK_UNKNOWN" in execution["reason_codes"]
    assert execution["state"] == "UNKNOWN" and not execution["allocation_authorized"]
    assert execution["account"]["positions"][0]["risk_evidence"] is None


@pytest.mark.parametrize("model,native,replay", [
    ("v6", True, None), ("v7", False, None),
    ("v7", True, "2026-10-14T13:30:00+08:00"),
])
def test_legacy_non_native_and_historical_accounts_are_not_backfilled(model, native, replay):
    settings = SimpleNamespace(daily_entry_model=model, les_v7_native_evidence=native)
    crawler = SimpleNamespace(app_settings=settings, cal_max_profit=Mock(return_value=[]),
                              build_strategy_decisions=Mock(return_value=()))
    clock = Mock(side_effect=AssertionError("This path must not read the current clock"))
    at = datetime.fromisoformat(replay) if replay else None
    CrawlerStrategyAdapter(portfolio_current_cash=1_000_000., now_provider=clock).evaluate(
        AnalysisContext(crawler, datetime(2026, 10, 14).date(), MarketStatus.OPEN, at))
    assert "as_of" not in crawler.les_v6_portfolio
    clock.assert_not_called()


def test_public_research_daily_entry_does_not_accept_historical_account_backfill():
    history = research_history()
    history.attrs["replay_as_of"] = history.attrs[SIGNAL_DATE_ATTR]
    history.attrs["les_v6_portfolio"] = _account(history.attrs[SIGNAL_DATE_ATTR])
    with pytest.raises(ValueError, match="V7_RESEARCH_BUNDLE_DAILY_BINDING_MISMATCH"):
        daily_report(history)
