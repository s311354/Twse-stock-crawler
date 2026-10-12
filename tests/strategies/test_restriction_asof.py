"""Restriction as-of dates must not be confused with the last completed bar."""
from dataclasses import replace
import datetime as dt
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from application.contracts import AnalysisContext, MarketStatus, StockAnalysisRequest
from application.low_entry_v6_daily import daily_report
from application.market_history import MarketHistoryAssembler
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from domain.strategies.low_entry_score_v6 import LowEntryScoreV6Strategy, V6_CONTEXT_ATTR, attach_low_entry_v6_context, assess_execution
from domain.strategies.low_entry_v6_report import report_failure
from domain.strategies.low_entry_v6_watch import screen_leadership
from domain.trading_restrictions import CURRENT_RESTRICTION_SCOPE, RESTRICTION_ENDPOINTS, TPEX_RESTRICTION_ENDPOINTS, restriction_check
from infrastructure.crawler.trading_restrictions import TradingRestrictionClient, TpexTradingRestrictionClient
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator
from aistock_twse_integration.les_v6_execution import execute_request
from tests.strategies.test_daily_v6_pipeline import build, daily_market
from tests.strategies.test_leadership_verification import AS_OF, SESSION, watch_histories
from tests.strategies.test_low_entry_score_v6 import ACCOUNT, financials


WEEKEND = "2026-09-12T15:00:00+08:00"


def evidence(at=WEEKEND, restricted=False):
    date = pd.Timestamp(at).tz_convert("Asia/Taipei").date().isoformat()
    return dict(restricted=restricted, source="TWSE_OPENAPI", available_at=at,
                session_date=date, observation_date=date, scope=CURRENT_RESTRICTION_SCOPE,
                endpoints=list(RESTRICTION_ENDPOINTS))


def checked(record, *, cutoff=WEEKEND, bar=SESSION):
    return restriction_check(record, market_data_date=bar, decision_time=cutoff)


@pytest.mark.parametrize("cutoff", [
    AS_OF, WEEKEND, "2026-09-13T22:30:00+08:00", "2026-09-14T08:30:00+08:00",
    "2026-09-14T00:30:00+08:00", "2026-09-13T16:30:00+00:00",
])
def test_fresh_current_evidence_uses_decision_date_not_bar_date(cutoff):
    record = evidence(cutoff)
    result = checked(record, cutoff=cutoff)
    assert result["status"] == "CLEAR" and result["restricted"] is False
    assert result["market_data_date"] == SESSION
    assert result["observation_date"] == record["observation_date"]
    assert result["decision_date"] == record["observation_date"]


@pytest.mark.parametrize("patch,reason", [
    ({"available_at": "2026-09-12T15:00:01+08:00"}, "FUTURE_RESTRICTION_EVIDENCE"),
    ({"available_at": "2026-09-12T10:59:59+08:00"}, "STALE_RESTRICTION_EVIDENCE"),
    ({"available_at": AS_OF}, "STALE_RESTRICTION_EVIDENCE"),
    ({"available_at": "2026-09-12T15:00:00"}, "INVALID_RESTRICTION_TIMESTAMP"),
    ({"available_at": "invalid"}, "INVALID_RESTRICTION_TIMESTAMP"),
    ({"available_at": None}, "INVALID_RESTRICTION_TIMESTAMP"),
    ({"session_date": SESSION}, "RESTRICTION_DATE_MISMATCH"),
    ({"observation_date": SESSION}, "RESTRICTION_DATE_MISMATCH"),
    ({"scope": None}, "RESTRICTION_SCOPE_REQUIRED"),
    ({"scope": "HISTORICAL"}, "UNSUPPORTED_RESTRICTION_SCOPE"),
    ({"source": "unverified"}, "INCOMPLETE_RESTRICTION_EVIDENCE"),
    ({"restricted": None}, "INCOMPLETE_RESTRICTION_EVIDENCE"),
    ({"restricted": 0}, "INCOMPLETE_RESTRICTION_EVIDENCE"),
    ({"endpoints": []}, "INCOMPLETE_RESTRICTION_EVIDENCE"),
    ({"endpoints": list(RESTRICTION_ENDPOINTS[:2])}, "INCOMPLETE_RESTRICTION_EVIDENCE"),
    ({"endpoints": "ALL"}, "INCOMPLETE_RESTRICTION_EVIDENCE"),
])
def test_invalid_or_stale_evidence_remains_unknown(patch, reason):
    result = checked(dict(evidence(), **patch))
    assert result["status"] == "UNKNOWN" and result["restricted"] is None
    assert result["reason"] == reason


def test_cutoff_and_bar_validation_and_legacy_same_session_compatibility():
    assert checked(evidence("2026-09-12T11:00:00+08:00"))["status"] == "CLEAR"
    for cutoff in (None, "invalid", "2026-09-12T15:00:00"):
        assert checked(evidence(), cutoff=cutoff)["status"] == "UNKNOWN"
    for bar in (None, "invalid", "2026-09-12T15:00:00+08:00"):
        assert checked(evidence(), bar=bar)["status"] == "UNKNOWN"
    assert checked(evidence(), bar="2026-09-14")["reason"] == "INCOMPLETE_OR_FUTURE_MARKET_DATA"
    for missing in ({}, None, []):
        assert checked(missing)["restricted"] is None
    old = daily_market().trading_restrictions["2330"]
    assert checked(old, cutoff=AS_OF)["status"] == "CLEAR"
    assert checked(old)["status"] == "UNKNOWN"  # Friday cannot authorize Saturday.
    assert checked(evidence(restricted=True))["status"] == "RESTRICTED"


@pytest.mark.parametrize("cutoff", [WEEKEND, "2026-09-13T22:30:00+08:00", "2026-09-14T08:30:00+08:00"])
@pytest.mark.usefixtures("active_industry_profiles")
def test_daily_pipeline_rechecks_current_status_without_changing_ohlc_date(cutoff):
    market = replace(daily_market(), signal_dates={"2330": cutoff},
                     trading_restrictions={"2330": evidence(cutoff)})
    _, decisions = build(market)
    report = json.loads(decisions[0].value("LES v6 Report"))
    assert report["decision"]["signal"] == "BUY_SMALL"
    assert report["signal_date"] == report["data"]["market_data_date"] == SESSION
    assert pd.Timestamp(report["decision_time"]) == pd.Timestamp(cutoff)
    assert report["data"]["restriction_check"]["status"] == "CLEAR"
    assert report["data"]["restriction_evidence"] == evidence(cutoff)
    assert report_failure(report) is None
    assert "TRADING_RESTRICTION_UNKNOWN" not in report["decision"]["reasons"]
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10:00+08:00", price=96.9, bid=96.8, ask=96.9, restricted=False)
    request = dict(report=report, quote=quote, now=quote["timestamp"],
                   trading_sessions=[SESSION, "2026-09-14"], portfolio=ACCOUNT, hard_risk=False)
    assert execute_request(request)["action"] == "ELIGIBLE"
    for restricted in (None, True):
        assert execute_request(dict(request, quote=dict(quote, restricted=restricted)))["action"] == "SKIP"
    assert execute_request(dict(request, now="2026-09-14T09:12:00+08:00"))["action"] == "SKIP"


@pytest.mark.parametrize("record", [evidence(restricted=True), {}, evidence(AS_OF), dict(evidence(), endpoints=[])])
def test_daily_pipeline_blocks_restrictions_and_invalid_evidence(record):
    _, decisions = build(replace(daily_market(), signal_dates={"2330": WEEKEND}, trading_restrictions={"2330": record}))
    report = json.loads(decisions[0].value("LES v6 Report"))
    assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}
    assert report["sizing"]["indicative_shares"] == 0
    assert report["data"]["restriction_check"]["status"] != "CLEAR"


def test_current_evidence_cannot_authorize_a_stale_symbol_or_an_incomplete_bar():
    market = replace(daily_market(), signal_dates={"2330": WEEKEND}, trading_restrictions={"2330": evidence()})
    history = MarketHistoryAssembler().build(market, 0)
    stale = history.iloc[:-1].copy()
    report = daily_report(stale)
    assert report["decision"]["reasons"] == ["STALE_MARKET_DATA"]
    incomplete_at = "2026-09-11T12:00:00+08:00"
    history.attrs.update({SIGNAL_DATE_ATTR: incomplete_at, "trading_restriction": evidence(incomplete_at)})
    incomplete = daily_report(history)
    assert incomplete["decision"]["signal"] == "WATCH"
    assert incomplete["decision"]["final_score"] is None


def test_weekend_restriction_context_never_leaks_into_historical_close_decisions():
    history = MarketHistoryAssembler().build(daily_market(), 0)
    for key in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        history[key] /= 1000
    history["Restricted"] = False  # Even a conflicting row value cannot override the context gate.
    data = attach_low_entry_v6_context(history.tail(65), symbol="2330", sector="ELEC", as_of=WEEKEND,
                                       filings=financials(), portfolio=ACCOUNT)
    data.attrs[V6_CONTEXT_ATTR]["restriction_evidence"] = evidence()
    reports = LowEntryScoreV6Strategy().run(data)["LES v6 Report"]
    assert all(r["data"]["restriction_check"]["status"] == "UNKNOWN" for r in reports)
    assert all(r["technical"]["restricted"] is None for r in reports)
    assert all(r["decision"]["signal"] not in {"BUY", "BUY_SMALL"} for r in reports)


@pytest.mark.usefixtures("active_industry_profiles")
def test_report_and_execution_revalidate_evidence_not_a_forged_clear_label():
    _, decisions = build(replace(daily_market(), signal_dates={"2330": WEEKEND}, trading_restrictions={"2330": evidence()}))
    report = json.loads(decisions[0].value("LES v6 Report"))
    report["data"]["restriction_evidence"]["session_date"] = SESSION
    assert report["data"]["restriction_check"]["status"] == "CLEAR"  # Intentionally stale label.
    assert report_failure(report) == "INVALID_PUBLISHED_RESTRICTION_EVIDENCE"
    assert execute_request(dict(report=report))["action"] == "SKIP"
    assert assess_execution(report, {}, now=WEEKEND, trading_sessions=[SESSION], portfolio=ACCOUNT,
                            hard_risk=False)["reason"] == "INVALID_PUBLISHED_RESTRICTION_EVIDENCE"
    for malformed_market in (None, [], {"exchange": []}, {"exchange": "UNKNOWN"}):
        report["market"] = malformed_market
        assert report_failure(report) == "INVALID_PUBLISHED_RESTRICTION_EVIDENCE"
        assert assess_execution(report, {}, now=WEEKEND, trading_sessions=[SESSION], portfolio=ACCOUNT,
                                hard_risk=False)["reason"] == "INVALID_PUBLISHED_RESTRICTION_EVIDENCE"


@pytest.mark.usefixtures("active_industry_profiles")
def test_execution_rejects_a_report_from_after_the_quote_time():
    _, decisions = build(replace(daily_market(), signal_dates={"2330": "2026-09-14T09:11:00+08:00"},
                                 trading_restrictions={"2330": evidence("2026-09-14T09:11:00+08:00")}))
    report = json.loads(decisions[0].value("LES v6 Report"))
    quote = dict(symbol="2330", timestamp="2026-09-14T09:10:00+08:00", price=96.9, bid=96.8, ask=96.9, restricted=False)
    result = execute_request(dict(report=report, quote=quote, now=quote["timestamp"],
                                  trading_sessions=[SESSION, "2026-09-14"], portfolio=ACCOUNT, hard_risk=False))
    assert result["action"] == "SKIP" and result["reason"] == "FUTURE_OR_MISSING_PUBLISHED_DECISION"


@pytest.mark.parametrize("breakout", [False, True])
def test_weekend_watch_uses_same_current_evidence_policy(breakout):
    histories = watch_histories(breakout)
    for h in histories:
        h.attrs["trading_restriction"] = evidence()
    result = screen_leadership(histories, session_date=SESSION, as_of=WEEKEND)
    assert result[0]["priority_eligible"] is True
    assert result[0]["analysis_session_date"] == SESSION
    assert result[0]["restriction_check"]["observation_date"] == "2026-09-12"
    histories[0].attrs["trading_restriction"] = evidence(restricted=True)
    assert screen_leadership(histories, session_date=SESSION, as_of=WEEKEND)[0]["reasons"] == ["TRADING_RESTRICTION"]


def test_runtime_uses_batch_bar_date_for_watch_and_current_date_for_official_lists():
    histories = watch_histories(True)
    calls = []
    def get_json(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(payload=[])
    service = SimpleNamespace(financial_batch_client=SimpleNamespace(http_client=SimpleNamespace(get_json=get_json)))
    crawler = SimpleNamespace(stocknumbers=[h.attrs["les_v6_daily_identity"]["symbol"] for h in histories],
                              stocknames=["測試"] * 4, daily_dates=[[]] * 4,
                              set_low_entry_v4_fundamentals=lambda *a, **kw: None)
    def history_at(index):
        h = histories[index].copy()
        h.attrs["trading_restriction"] = crawler.trading_restrictions[crawler.stocknumbers[index]]
        return h
    crawler.build_strategy_history = history_at
    # Stop immediately after screening: no MOPS requests are needed to test ordering.
    class Screened(Exception):
        pass
    def begin_run():
        raise Screened
    service.begin_run = begin_run
    crawler.daily_dates = [[SESSION]] * 4
    context = AnalysisContext(crawler, dt.date(2026, 9, 12), MarketStatus.CLOSED)
    with pytest.raises(Screened):
        FundamentalRefreshCoordinator(service).refresh(context, StockAnalysisRequest([], "ELEC", "test", 500),
                                                      analysis_time=dt.datetime.fromisoformat(WEEKEND))
    assert len(calls) == 3
    watch = crawler.les_v6_research_context["2330"]["watch"]
    assert watch["priority_eligible"] is True and watch["analysis_session_date"] == SESSION
    assert crawler.trading_restrictions["2330"]["session_date"] == "2026-09-12"


def test_client_preserves_observation_scope_and_requires_all_lists():
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(payload=[])
    client = TradingRestrictionClient(SimpleNamespace(get_json=get))
    observed = dt.datetime.fromisoformat(WEEKEND)
    records = client.load(["2330"], session_date="2026-09-12", observed_at=observed)
    assert records["2330"] == evidence()
    assert client.load(["2330"], session_date=SESSION, observed_at=observed) == {}
    assert len(calls) == 3
    def partial(url, **kwargs):
        return SimpleNamespace(payload={} if url.endswith("punish") else [])
    with pytest.raises(TwseSchemaChangedError):
        TradingRestrictionClient(SimpleNamespace(get_json=partial)).load(["2330"], session_date="2026-09-12", observed_at=observed)


def test_current_halt_and_disposition_intervals_use_observation_date():
    rows = {
        "TWTAWU": [{"Code": "2330", "TradingHaltDate": "1150912", "TradingResumptionDate": "1150914"}],
        "TWT85U": [{"Code": "3008"}],
        "punish": [{"Code": "2454", "DispositionPeriod": "115/09/12~115/09/14"}],
    }
    client = TradingRestrictionClient(SimpleNamespace(get_json=lambda url, **kw: SimpleNamespace(payload=rows[url.rsplit("/", 1)[-1]])))
    result = client.load(["2330", "2454", "3008", "2404"], session_date="2026-09-12", observed_at=dt.datetime.fromisoformat(WEEKEND))
    assert all(checked(result[s])["restricted"] is True for s in ("2330", "2454", "3008"))
    assert checked(result["2404"])["restricted"] is False


def tpex_evidence(at=WEEKEND, restricted=False):
    return dict(evidence(at, restricted), market="TPEX", source="TPEX_OPENAPI",
                endpoints=list(TPEX_RESTRICTION_ENDPOINTS))


def tpex_feeds():
    return {
        "tpex_spendi_today": [{"Date": "20260912", "SecuritiesCompanyCode": "", "CompanyName": "",
                               "暫停交易": "", "恢復交易": ""}],
        "tpex_spendi_history": [], "tpex_cmode": [], "tpex_disposal_information": [],
    }


def load_tpex(feeds, symbols=("3105",)):
    return TpexTradingRestrictionClient(SimpleNamespace(
        get_json=lambda url, **kw: SimpleNamespace(payload=feeds[url.rsplit("/", 1)[-1]]),
    )).load(symbols, session_date="2026-09-12", observed_at=dt.datetime.fromisoformat(WEEKEND))


def halt_event(symbol, date, *, resume=False):
    return {"Date": "115", "SecuritiesCompanyCode": symbol,
            "DateOfSuspendedTrading": "" if resume else date,
            "TimeOfSuspendedTrading": "" if resume else "080000",
            "DateOfResumedTrading": date if resume else "",
            "TimeOfResumedTrading": "080000" if resume else ""}


def test_tpex_requires_all_official_feeds_and_preserves_current_scope():
    feeds, calls = tpex_feeds(), []
    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(payload=feeds[url.rsplit("/", 1)[-1]])
    client = TpexTradingRestrictionClient(SimpleNamespace(get_json=get))
    observed = dt.datetime.fromisoformat(WEEKEND)
    record = client.load(["3105"], session_date="2026-09-12", observed_at=observed)["3105"]
    assert record == tpex_evidence()
    assert len(calls) == 4 and all(url.startswith("https://www.tpex.org.tw/openapi/v1/") for url in calls)
    assert client.load(["3105"], session_date=SESSION, observed_at=observed) == {}
    assert len(calls) == 4
    assert restriction_check(record, market="TPEX", market_data_date=SESSION, decision_time=WEEKEND)["status"] == "CLEAR"
    for market, record in (("TWSE", record), ("TPEX", evidence())):
        result = restriction_check(record, market=market, market_data_date=SESSION, decision_time=WEEKEND)
        assert result["status"] == "UNKNOWN" and result["reason"] == "RESTRICTION_MARKET_MISMATCH"
    for patch in ({"source": "TWSE_OPENAPI"}, {"endpoints": list(RESTRICTION_ENDPOINTS)},
                  {"endpoints": list(TPEX_RESTRICTION_ENDPOINTS[:-1])}, {"available_at": AS_OF}):
        assert restriction_check(dict(tpex_evidence(), **patch), market="TPEX", market_data_date=SESSION,
                                 decision_time=WEEKEND)["status"] == "UNKNOWN"


def test_tpex_halt_event_pairs_full_width_modes_and_compact_disposition_periods():
    feeds = tpex_feeds()
    feeds["tpex_spendi_today"] = [{"Date": "20260912", "SecuritiesCompanyCode": "today",
                                   "暫停交易": "", "恢復交易": "08:00:00"}]
    feeds["tpex_spendi_history"] = [
        halt_event("active", "1150911"), halt_event("resumed", "1150909"),
        halt_event("resumed", "1150910", resume=True), halt_event("resuming", "1150909"),
        halt_event("resuming", "1150912", resume=True), halt_event("future", "1150914"),
    ]
    feeds["tpex_cmode"] = [{"Date": "1150911", "SecuritiesCompanyCode": "mode",
                            "AlteredTrading": "Ｙ", "PeriodicTrading": "", "ManagedStock": "",
                            "SuspensionOfTrading": ""}]
    feeds["tpex_disposal_information"] = [
        {"Date": "1150911", "SecuritiesCompanyCode": "disposed", "DispositionPeriod": "1150912~1150914"},
        {"Date": "1150909", "SecuritiesCompanyCode": "expired", "DispositionPeriod": "1150909~1150911"},
    ]
    symbols = ("today", "active", "resuming", "mode", "disposed", "resumed", "future", "expired", "3105")
    result = load_tpex(feeds, symbols)
    assert {symbol for symbol, record in result.items() if record["restricted"]} == {
        "today", "active", "resuming", "mode", "disposed",
    }


@pytest.mark.parametrize("endpoint,rows", [
    ("tpex_spendi_today", []),
    ("tpex_spendi_today", [{"Date": "20260911", "SecuritiesCompanyCode": "", "暫停交易": "", "恢復交易": ""}]),
    ("tpex_spendi_today", [{"Date": "20260912", "SecuritiesCompanyCode": "", "暫停交易": "Y", "恢復交易": ""}]),
    ("tpex_spendi_today", [{"Date": "20260912", "SecuritiesCompanyCode": "3105", "暫停交易": "", "恢復交易": ""}]),
    ("tpex_spendi_history", {}),
    ("tpex_spendi_history", [dict(halt_event("3105", "1150911"), Date="114")]),
    ("tpex_spendi_history", [dict(halt_event("3105", "1150911"), TimeOfSuspendedTrading="bad")]),
    ("tpex_spendi_history", [halt_event("3105", "1150911"), halt_event("3105", "1150911")]),
    ("tpex_spendi_history", [dict(halt_event("3105", ""), TimeOfSuspendedTrading="")]),
    ("tpex_cmode", [{"Date": "1150911", "SecuritiesCompanyCode": "3105", "AlteredTrading": "?",
                     "PeriodicTrading": "", "ManagedStock": "", "SuspensionOfTrading": ""}]),
    ("tpex_cmode", [{"Date": "1150911", "SecuritiesCompanyCode": "3105"}]),
    ("tpex_cmode", [{"Date": "1150913", "SecuritiesCompanyCode": "3105", "AlteredTrading": "Ｙ",
                     "PeriodicTrading": "", "ManagedStock": "", "SuspensionOfTrading": ""}]),
    ("tpex_disposal_information", [{"Date": "1150911", "SecuritiesCompanyCode": "3105",
                                    "DispositionPeriod": "1150914~1150912"}]),
    ("tpex_disposal_information", ["malformed"]),
    ("tpex_disposal_information", [{"Date": "1150913", "SecuritiesCompanyCode": "3105",
                                    "DispositionPeriod": "1150913~1150914"}]),
])
def test_tpex_partial_or_malformed_feed_cannot_clear_any_symbol(endpoint, rows):
    feeds = dict(tpex_feeds(), **{endpoint: rows})
    with pytest.raises(TwseSchemaChangedError):
        load_tpex(feeds)


@pytest.mark.usefixtures("active_industry_profiles")
def test_strategy_report_execution_and_watch_bind_restrictions_to_expected_market():
    # An otherwise executable TWSE report cannot be relabelled as TPEx while
    # retaining TWSE eligibility evidence, including direct execution callers.
    _, decisions = build(replace(daily_market(), signal_dates={"2330": WEEKEND},
                                 trading_restrictions={"2330": evidence()}))
    report = json.loads(decisions[0].value("LES v6 Report"))
    assert report["market"]["exchange"] == "TWSE"
    report["market"]["exchange"] = "TPEX"
    assert report_failure(report) == "INVALID_PUBLISHED_RESTRICTION_EVIDENCE"
    assert execute_request(dict(report=report))["action"] == "SKIP"
    assert assess_execution(report, {}, now=WEEKEND, trading_sessions=[SESSION], portfolio=ACCOUNT,
                            hard_risk=False)["reason"] == "INVALID_PUBLISHED_RESTRICTION_EVIDENCE"
    histories = watch_histories(True)
    for history in histories:
        history.attrs["les_v6_daily_identity"] = dict(history.attrs["les_v6_daily_identity"], market="TPEX")
        history.attrs["trading_restriction"] = evidence()
    watches = screen_leadership(histories, session_date=SESSION, as_of=WEEKEND)
    assert all(w["reasons"] == ["TRADING_RESTRICTION_UNKNOWN"] for w in watches)
    for history in histories:
        history.attrs["trading_restriction"] = tpex_evidence()
    watches = screen_leadership(histories, session_date=SESSION, as_of=WEEKEND)
    assert watches[0]["priority_eligible"] is True


def test_direct_strategy_publishes_market_and_rejects_other_market_evidence():
    history = MarketHistoryAssembler().build(daily_market(), 0)
    for key in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        history[key] /= 1000
    data = attach_low_entry_v6_context(
        history, symbol="2330", sector="ELEC", as_of=WEEKEND, market="TPEX",
        filings=[dict(row, market="TPEX") for row in financials()], portfolio=ACCOUNT,
    )
    for record, status in ((evidence(), "UNKNOWN"), (tpex_evidence(), "CLEAR")):
        data.attrs[V6_CONTEXT_ATTR]["restriction_evidence"] = record
        report = LowEntryScoreV6Strategy().run_latest(data).iloc[-1]["LES v6 Report"]
        assert report["market"]["exchange"] == "TPEX"
        assert report["data"]["restriction_check"]["status"] == status
        if status == "UNKNOWN":
            assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}
