"""Strong research candidates must never bypass evidence or execution gates."""
from copy import deepcopy
from dataclasses import replace
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from application.low_entry_v6_daily import daily_report, overlay_daily_v6, project_report
from application.market_history import MarketHistoryAssembler
from application.mops_refresh_control import MopsRefreshControl, MopsRefreshPolicy
from domain.strategies.low_entry_score_v4 import ANNUAL_FUNDAMENTALS_ATTR, FUNDAMENTAL_SNAPSHOT_ATTR, PRICE_SCALE_ATTR, SIGNAL_DATE_ATTR
from domain.strategies.low_entry_v6_quality import verification_diagnostics
from domain.strategies.low_entry_v6_watch import screen_leadership
from domain.trading_restrictions import RESTRICTION_ENDPOINTS
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.storage.leadership_watch_repository import LeadershipWatchRepository
from aistock_twse_integration.les_v6_execution import execute_request
from tests.strategies.test_low_entry_score_v6 import ACCOUNT, financials
from tests.strategies.test_daily_v6_pipeline import daily_market, build


SESSION = "2026-09-11"
AS_OF = "2026-09-11T15:00:00+08:00"


def watch_histories(breakout=False):
    histories = []
    for i, symbol in enumerate(("2330", "2454", "3008", "2404")):
        c = np.r_[np.linspace(90., 100., 80), 100 + .3 * np.sin(np.arange(20))]
        if i:
            c[-20:] = np.linspace(100., 94., 20)
        if breakout and i == 0:
            c[-1] = 101.
        v = np.full(100, 1_000_000.)
        if breakout and i == 0:
            v[-1] = 1_300_000.
        history = pd.DataFrame({
            "Date": pd.bdate_range(end=SESSION, periods=100),
            "Open": c - .3, "High": c + .1, "Low": c - .6, "Close": c,
            "Volume": v, "Amount": c * v, "Trades": 500,
            "Bid": c - .05, "Ask": c + .05, "PE": 20.,
            "TAIEX Close": np.r_[np.linspace(20000., 21000., 80), np.linspace(21000., 18800., 20)],
        })
        history.attrs.update({
            "les_v6_daily_identity": {"symbol": symbol, "name": "測試", "sector": "ELEC"},
            "les_v6_analysis_session": SESSION, SIGNAL_DATE_ATTR: AS_OF, PRICE_SCALE_ATTR: 1,
            "trading_restriction": {"restricted": False, "source": "TWSE_OPENAPI", "session_date": SESSION,
                                    "available_at": AS_OF, "endpoints": list(RESTRICTION_ENDPOINTS)},
        })
        histories.append(history)
    return histories


def screen(histories):
    return screen_leadership(histories, session_date=SESSION, as_of=AS_OF)


@pytest.mark.parametrize("breakout,pattern", [(False, "RESILIENT_BASE"), (True, "LEADING_BREAKOUT_60")])
def test_relative_leadership_requires_market_and_same_universe_peers(breakout, pattern):
    result = screen(watch_histories(breakout))[0]
    assert result["status"] == "LEADERSHIP_WATCH"
    assert pattern in result["patterns"]
    assert result["priority_eligible"] is True
    assert result["is_entry_signal"] is False
    assert result["peer_count"] == 3 and result["peer_symbols"] == ["2404", "2454", "3008"]
    assert result["metrics"]["peer_excess_return20"] > .05


@pytest.mark.parametrize("breakout", [False, True])
def test_ohlc_check_ignores_dataframe_metadata_and_preserves_input(breakout):
    histories = watch_histories(breakout)
    expected = screen(histories)
    annual = pd.DataFrame({"year": [2024, 2025], "eps": [2., 3.]})
    for history in histories:
        history.index = pd.RangeIndex(100, 200)
        history.attrs[ANNUAL_FUNDAMENTALS_ATTR] = annual
    before = [history.copy(deep=True) for history in histories]

    assert screen(histories) == expected
    for history, original in zip(histories, before):
        pd.testing.assert_frame_equal(history, original)
        assert history.attrs[ANNUAL_FUNDAMENTALS_ATTR] is annual
    pd.testing.assert_frame_equal(annual, pd.DataFrame({"year": [2024, 2025], "eps": [2., 3.]}))


@pytest.mark.parametrize("open_price,high,low,close", [
    (102., 101., 99., 100.),  # High below Open only.
    (100., 101., 99., 102.),  # High below Close only.
    (100., 103., 101., 102.),  # Low above Open only.
    (102., 103., 101., 100.),  # Low above Close only.
])
def test_dataframe_metadata_does_not_disable_ohlc_rejection(open_price, high, low, close):
    histories = watch_histories(True)
    histories[0].attrs[ANNUAL_FUNDAMENTALS_ATTR] = pd.DataFrame({"eps": [3.]})
    histories[0].loc[99, ["Open", "High", "Low", "Close"]] = [open_price, high, low, close]
    result = screen(histories)[0]
    assert result["status"] == "UNAVAILABLE"
    assert result["reasons"] == ["INVALID_OHLC"]
    assert not result["priority_eligible"] and not result["is_entry_signal"]


@pytest.mark.parametrize("has_filings", [False, True])
@pytest.mark.usefixtures("active_industry_profiles")
def test_daily_overlay_preserves_v6_decision_with_annual_dataframe_metadata(has_filings):
    market = daily_market()
    if not has_filings:
        market = replace(market, fundamental_snapshots={})
    history = MarketHistoryAssembler().build(market, 0)
    expected = overlay_daily_v6([history], pd.DataFrame(index=[0]))
    annual = pd.DataFrame({"year": [2024, 2025], "eps": [2., 3.]})
    history.attrs[ANNUAL_FUNDAMENTALS_ATTR] = annual
    original = history.copy(deep=True)

    actual = overlay_daily_v6([history], pd.DataFrame(index=[0]))

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(history, original)
    assert history.attrs[ANNUAL_FUNDAMENTALS_ATTR] is annual
    report = json.loads(actual.iloc[0]["LES v6 Report"])
    assert report["decision"]["signal"] == ("BUY_SMALL" if has_filings else "WATCH")
    if not has_filings:
        assert report["sizing"]["indicative_shares"] == 0
        assert report["decision"]["safety_cap"] == 61


@pytest.mark.parametrize("change,reason", [
    ("stale", "STALE_MARKET_DATA"), ("volume", "MISSING_OR_INACTIVE_MARKET_DATA"),
    ("amount", "ILLIQUID_WATCH_HISTORY"), ("halt", "TRADING_RESTRICTION"),
    ("restriction_missing", "TRADING_RESTRICTION_UNKNOWN"),
    ("future_restriction", "TRADING_RESTRICTION_UNKNOWN"),
    ("future_bar", "INCOMPLETE_DAILY_BAR"), ("incomplete", "INCOMPLETE_DAILY_BAR"),
    ("taiex", "MISSING_OR_INACTIVE_MARKET_DATA"), ("ohlc", "INVALID_OHLC"),
])
def test_no_false_resilience_from_bad_or_stale_evidence(change, reason):
    histories = watch_histories()
    h = histories[0]
    if change == "stale":
        histories[0] = h.iloc[:-1].copy()
    elif change == "volume": h.loc[99, "Volume"] = 0
    elif change == "amount": h["Amount"] = 10_000.
    elif change == "halt": h.attrs["trading_restriction"]["restricted"] = True
    elif change == "restriction_missing": h.attrs["trading_restriction"] = {}
    elif change == "future_restriction": h.attrs["trading_restriction"]["available_at"] = "2026-09-12T15:00:00+08:00"
    elif change == "incomplete": h["Complete"] = False
    elif change == "taiex": h.loc[85, "TAIEX Close"] = np.nan
    elif change == "ohlc": h.loc[99, "High"] = 1
    else:
        result = screen_leadership(histories, session_date=SESSION, as_of="2026-09-11T13:00:00+08:00")[0]
        assert result["priority_eligible"] is False and reason in result["reasons"]
        return
    result = screen(histories)[0]
    assert result["status"] == "UNAVAILABLE" and not result["priority_eligible"]
    assert reason in result["reasons"]


def test_peer_unknown_or_outperformance_failure_does_not_claim_priority():
    histories = watch_histories()
    assert screen(histories[:1])[0]["reasons"] == ["PEER_COMPARISON_INSUFFICIENT"]
    histories[-1].attrs["les_v6_daily_identity"]["sector"] = "AIR"
    assert screen(histories)[0]["priority_eligible"] is False
    histories = watch_histories()
    for h in histories[1:]:
        for key in ("Open", "High", "Low", "Close"):
            h.loc[99, key] += 20
    result = screen(histories)[0]
    assert result["status"] == "LEADERSHIP_WATCH" and "BELOW_PEER_MEDIAN" in result["reasons"]
    assert not result["priority_eligible"]


def test_diagnostics_separate_missing_quarters_fields_and_optional_evidence():
    records = financials()
    base = verification_diagnostics(records, symbol="2330", as_of=AS_OF)
    assert base["next_action"] == "NORMAL_MODEL_EVALUATION"
    assert not base["backfill_needed"] and not base["local_review_required"]
    missing = verification_diagnostics(records[:-2] + [records[-1]], symbol="2330", as_of=AS_OF)
    assert missing["missing_periods"] == ["2026Q1"] and missing["backfill_needed"]
    records[-1]["current_assets"] = None
    fields = verification_diagnostics(records, symbol="2330", as_of=AS_OF)
    assert fields["next_action"] == "REVIEW_LOCAL_SOURCE_FIELDS" and not fields["backfill_needed"]
    assert fields["unresolved_fields"] == ["2026Q2:current_assets"]
    mixed = verification_diagnostics(records[:-2] + [records[-1]], symbol="2330", as_of=AS_OF)
    assert mixed["backfill_needed"] and mixed["local_review_required"]
    assert mixed["next_action"] == "REVIEW_LOCAL_SOURCE_FIELDS"
    # A future filing cannot fill an old gap, even if already present on disk.
    records[-1]["available_at"] = "2026-09-15T00:00:00+08:00"
    assert verification_diagnostics(records, symbol="2330", as_of=AS_OF)["latest_period"] == "2026Q1"
    malformed = verification_diagnostics([dict(financials()[-1], available_at=None)], symbol="2330", as_of=AS_OF)
    assert malformed["local_review_required"] and not malformed["backfill_needed"]


def test_ttm_ytd_predecessor_is_diagnostic_but_missing_reported_comparisons_need_review():
    records = financials()[-8:]
    for record in records:
        record["flow_basis"] = "YTD"
        record["basic_eps_quarter"] = record["basic_eps"]
    diagnostic = verification_diagnostics(records, symbol="2330", as_of=AS_OF)
    assert diagnostic["missing_periods"] == []
    assert diagnostic["diagnostic_missing_periods"] == ["2024Q2"]
    # Source/revision labels alone cannot attest that two YTD reports have the
    # same accounting revision basis. Presence of the predecessor is not PASS.
    assert diagnostic["local_review_required"]
    assert any(c["reason"] == "REVISION_BASIS_UNVERIFIED" and c["status"] == "UNKNOWN" and not c["required_by_quality"]
               for c in diagnostic["field_checks"])
    assert "2026Q2:comparative:revenue" in diagnostic["unresolved_fields"]


def test_watch_is_additive_never_turns_missing_data_into_a_buy():
    histories = watch_histories(True)
    # The production daily boundary expects scaled legacy prices.
    for history in histories:
        for field in ("Open", "High", "Low", "Close", "Bid", "Ask"):
            history[field] *= 1000
        history.attrs[PRICE_SCALE_ATTR] = 1000
    output = overlay_daily_v6(histories, pd.DataFrame(index=range(4)))
    report = json.loads(output.iloc[0]["LES v6 Report"])
    assert output.iloc[0]["LES v6 Watch Status"] == "LEADERSHIP_WATCH"
    assert report["research"]["verification"]["backfill_needed"]
    assert report["decision"]["signal"] == "WATCH"
    assert report["decision"]["safety_cap"] == 61
    assert output.iloc[0]["Position Fraction"] == output.iloc[0]["建議股數"] == 0
    assert execute_request(dict(report=report, quote={}, now=AS_OF,
                                trading_sessions=[SESSION], portfolio=ACCOUNT, hard_risk=False))["action"] == "SKIP"


@pytest.mark.usefixtures("active_industry_profiles")
def test_reassessment_preserves_old_block_and_execution_rechecks_new_risk(tmp_path):
    history = MarketHistoryAssembler().build(daily_market(), 0)
    complete = deepcopy(history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR])
    history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR] = {}
    before = daily_report(history)
    history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR] = complete
    history.attrs[SIGNAL_DATE_ATTR] = "2026-09-11T16:00:00+08:00"
    after = daily_report(history)
    assert before["decision"]["signal"] == "WATCH"
    assert before["quality"]["status"] == "DATA_INSUFFICIENT"
    assert after["decision"]["signal"] == "BUY_SMALL" and after["entry"]["is_fill"] is False
    repository = LeadershipWatchRepository(tmp_path / "research.sqlite3")
    repository.record("ELEC", [before, after, after])
    events = repository.timeline("ELEC", "2330")
    assert len(events) == 2
    assert [event["decision"]["signal"] for event in events] == ["WATCH", "BUY_SMALL"]
    assert events[0]["research"]["verification"]["backfill_needed"]
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10:00+08:00", price=140, bid=139.9, ask=140, restricted=False)
    result = execute_request(dict(report=after, quote=quote, now=quote["timestamp"],
                                  trading_sessions=[SESSION, "2026-09-14"], portfolio=ACCOUNT, hard_risk=False))
    assert result["action"] == "SKIP" and result["reason"] == "REPRICED_RISK_OR_LIQUIDITY_FAIL"
    # Cash flow now participates in native Quality; missing optional audit
    # evidence does not fabricate it or waive any execution risk check.
    assert after["quality"]["metrics"]["cfo_ttm"] > 0
    assert after["quality"]["coverage"]["cash_flow"] == 100
    assert after["quality"]["optional"]["auditor_opinion"] == "NOT_AVAILABLE"


def research_market():
    histories = watch_histories(True)
    market = daily_market()
    from_data = {name: [h[column].tolist() for h in histories] for name, column in (
        ("dates", "Date"), ("opens", "Open"), ("highs", "High"), ("lows", "Low"),
        ("closes", "Close"), ("volumes", "Volume"), ("trade_counts", "Trades"),
        ("trade_values", "Amount"), ("bids", "Bid"), ("asks", "Ask"), ("pe_ratios", "PE"),
        ("taiex_closes", "TAIEX Close"),
    )}
    for key in ("opens", "highs", "lows", "closes", "bids", "asks"):
        from_data[key] = [[value * 1000 for value in values] for values in from_data[key]]
    from_data["dates"] = [[pd.Timestamp(value).date().isoformat() for value in values] for values in from_data["dates"]]
    symbols = [h.attrs["les_v6_daily_identity"]["symbol"] for h in histories]
    return replace(market, stock_count=4, stock_numbers=symbols, stock_names=["測試<script>" for _ in symbols],
                   fundamental_snapshots={}, signal_dates={symbol: AS_OF for symbol in symbols},
                   trading_restrictions={h.attrs["les_v6_daily_identity"]["symbol"]: h.attrs["trading_restriction"] for h in histories},
                   **from_data)


def test_full_daily_report_omits_research_but_preserves_internal_audit(tmp_path):
    from application.analysis_dataset import AnalysisDatasetBuilder
    from infrastructure.report.html_renderer import RendererFactory
    from infrastructure.storage.csv_repository import CsvRepository
    from aistock_twse_integration.les_v6_execution import load_reports

    market = research_market()
    builder = AnalysisDatasetBuilder()
    decisions = builder.build_decisions(market, [[1, 1, 1, 1, .01]] * 4)
    html = RendererFactory.get_renderer("daily_strategy_email").render(decisions)
    assert 'data-section="leadership-watch"' not in html
    assert 'data-section="technical-observations"' not in html
    assert "LEADING_BREAKOUT_60" not in html and "BACKFILL_MISSING_QUARTERS" not in html
    assert "強勢候選優先驗證" not in html and "本次沒有 BUY / BUY_SMALL" in html
    assert "測試<script>" not in html and "&lt;script&gt;" in html
    assert 'data-label="待驗證資料"' not in html
    CsvRepository(tmp_path).write_analysis_dataset("elec", builder.project_decisions(decisions))
    reports = load_reports(str(tmp_path / "elec_les_v6_reports.json"), SESSION)
    assert all(report["decision"]["signal"] not in {"BUY", "BUY_SMALL"} for report in reports.values())
    assert all(report["sizing"]["indicative_shares"] == 0 for report in reports.values())
    audit = LeadershipWatchRepository(tmp_path / "les_v6_leadership_history.sqlite3")
    assert audit.timeline("elec", "2330")[0]["research"]["watch"]["status"] == "LEADERSHIP_WATCH"
    # Non-leaders are retained as controls, avoiding a winners-only record.
    assert audit.timeline("elec", "2454")[0]["research"]["watch"]["status"] == "NONE"
