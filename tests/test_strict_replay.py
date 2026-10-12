"""Strict research contracts, native model parity and transactional isolation."""
from copy import deepcopy
from dataclasses import asdict, replace
import datetime as dt
import json
from pathlib import Path
import sqlite3
from unittest.mock import Mock

import pandas as pd
import pytest

from application.historical_replay import replay_close_cutoff
from application.replay_contracts import ReplayAccount, ReplayHolding, ReplayRequest, ReplayVersions, calendar_window, digest
from application.replay_inputs import PinnedReplayEvidence
from application.replay_service import StrictReplayEngine
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.strategies.portfolio_allocation_v2 import PersistenceObservationV2
from infrastructure.replay_evidence import EVIDENCE_SCHEMA, save_evidence, load_evidence, prepare_evidence
from infrastructure.replay_runtime import build_replay_engine, current_versions
from infrastructure.storage.replay_state import SQLiteReplayState
from interface.cli import main, create_parser
from interface.replay import request_from_args
from tests.strategies.test_daily_v6_pipeline import daily_market
from tests.strategies.test_low_entry_score_v6 import financials


VERSIONS = ReplayVersions("test-native-model", "test-parameters", "test-code")


def payload_fixture():
    market = daily_market()
    bars = []
    for i, day in enumerate(market.dates[0]):
        row = ["2330", "test", str(market.volumes[0][i]), "500", str(market.trade_values[0][i]),
               *(str(getattr(market, name)[0][i]/1000) for name in ("opens", "highs", "lows", "closes")),
               "+", "0", str(market.bids[0][i]/1000), "1", str(market.asks[0][i]/1000), "1", "20"]
        bars += [dict(date=day.replace("-", ""), dataset="STOCK_TABLE", market_type="24", data=[row]),
                 dict(date=day.replace("-", ""), dataset="TAIEX", market_type="IND", data=market.taiex_closes[0][i])]
    records, flows = [], {}
    for i, source in enumerate(financials()):
        period = pd.Period(source["period_end"], freq="Q")
        if period.quarter == 1:
            flows = {}
        facts = []
        for key, concept in {"revenue":"Revenue", "operating_income":"ProfitLossFromOperatingActivities", "net_income":"ProfitLoss", "basic_eps":"BasicEarningsLossPerShare"}.items():
            flows[key] = flows.get(key, 0) + source[key]
            unit = "iso4217:TWD/xbrli:shares" if key == "basic_eps" else "iso4217:TWD"
            value = flows[key] * (1 if key == "basic_eps" else 1000)
            facts.append(FiledFact("ifrs-full:"+concept, value, None, f"{period.year}-01-01", source["period_end"], None, unit=unit))
        flows["gross"] = flows.get("gross", 0) + source["revenue"]-source["cost_of_revenue"]
        facts.append(FiledFact("ifrs-full:GrossProfit", flows["gross"]*1000, None, f"{period.year}-01-01", source["period_end"], None, unit="iso4217:TWD"))
        for key, concept in {"current_assets":"CurrentAssets", "total_assets":"Assets", "current_liabilities":"CurrentLiabilities", "total_liabilities":"Liabilities", "total_equity":"Equity"}.items():
            facts.append(FiledFact("ifrs-full:"+concept, source[key]*1000, None, None, None, source["period_end"], unit="iso4217:TWD"))
        record = FiledFinancials(period.year, period.quarter, source["available_at"], 1, tuple(facts),
                                 availability_source="OFFICIAL_FILING_TIMESTAMP", filing_id=i+1, content_sha256=f"test-{i}",
                                 source_url="https://mops.twse.com.tw/test-fixture", period_end=source["period_end"])
        records.append(dict(asdict(record), receipt=dict(fetched_at=source["available_at"], report_id="C")))
    start, end = market.dates[0][0], "2026-09-14"
    days = {day.date().isoformat(): "OPEN" if day.weekday()<5 else "CLOSED" for day in pd.date_range(start, end)}
    evidence = dict(market.trading_restrictions["2330"], available_at="2026-09-11T13:30:00+08:00")
    return dict(schema=EVIDENCE_SCHEMA, start=start, end=end, market="TWSE", sector="24", symbols=["2330"],
                calendar=dict(source="TWSE_OFFICIAL", days=days), bars=bars, filings={"2330":records}, batches={},
                restrictions=[dict(symbol="2330", evidence=evidence)], restriction_ledger=None, preparation_gaps=[])


def request(tmp_path, payload, *, day="2026-09-11", replay_id="test", account=None):
    return ReplayRequest(replay_id, replay_close_cutoff(day), dt.date.fromisoformat(payload["start"]), dt.date.fromisoformat(day),
        ("2330",), "TWSE", "24", account or ReplayAccount(1_000_000., 1_000_000., ()),
        VERSIONS, digest(payload), tmp_path / "replay.sqlite3")


def engine(payload, **kwargs):
    return StrictReplayEngine(PinnedReplayEvidence(digest(payload), payload), SQLiteReplayState(), lambda: VERSIONS, **kwargs)


def test_cli_and_shared_engine_equal_and_no_live_ports(tmp_path, monkeypatch, capsys):
    payload = payload_fixture()
    evidence = tmp_path / "evidence.json"
    save_evidence(payload, evidence)
    holdings = tmp_path / "empty.txt"
    holdings.write_text("")
    args = ["2330", "-b", payload["start"], "-e", "2026-09-11", "--replay-id", "parity",
            "--replay-evidence-file", str(evidence), "--replay-state-path", str(tmp_path/"cli.sqlite3"),
            "--replay-output-dir", str(tmp_path/"out"), "--pae-holdings-file", str(holdings),
            "--pae-current-cash", "1000000", "--account-equity", "1000000"]
    monkeypatch.setattr("requests.sessions.Session.request", Mock(side_effect=AssertionError("network")))
    monkeypatch.setattr("interface.cli.build_stock_analysis_service", Mock(side_effect=AssertionError("live runtime")))
    monkeypatch.setenv("TWSE_SMTP_EMAIL", "not-an-address")
    assert main(args) == 0
    shared, _ = build_replay_engine(evidence)
    req = request_from_args(create_parser().parse_args(args))
    expected = shared.run(replace(req, state_path=tmp_path/"shared.sqlite3"))
    actual = json.loads(next((tmp_path/"out").glob("*.json")).read_text())
    assert actual == expected
    assert actual["live_order_authorized"] is False
    assert actual["decisions"][0]["raw"]["erem_score"] is not None
    assert actual["decisions"][0]["hard_gates"]["kelly"] == "UNKNOWN"
    assert next((tmp_path/"out").glob("*.csv")).exists()


@pytest.mark.usefixtures("active_industry_profiles")
def test_future_data_invariant_and_missing_restrictions_remain_unknown(tmp_path):
    payload = payload_fixture()
    req = request(tmp_path, payload)
    before = engine(payload).run(req)
    after_payload = deepcopy(payload)
    revision = deepcopy(payload["filings"]["2330"][-1])
    revision.update(available_at="2026-09-12T13:30:00+08:00", content_sha256="future", revision_number=2)
    after_payload["filings"]["2330"].append(revision)
    after_payload["restrictions"].append(dict(symbol="2330", evidence=dict(payload["restrictions"][0]["evidence"], available_at="2026-09-12T13:30:00+08:00")))
    after_payload["bars"].append(dict(date="20260914", dataset="TAIEX", data=999999))
    after = engine(after_payload).run(request(tmp_path, after_payload, replay_id="future-check"))
    assert before["rows"] == after["rows"]
    assert before["selected_evidence_sha256"] == after["selected_evidence_sha256"]
    assert any(e["reason"]=="AFTER_CUTOFF" for e in after["diagnostics"]["excluded_evidence"]["2330"])
    payload["restrictions"] = []
    blocked = engine(payload).run(request(tmp_path, payload, replay_id="missing"))
    result = blocked["decisions"][0]
    # This fixture has unbound YTD quarters and no CFO. The previous normalized
    # QS looked high despite missing sources; fixed-budget QS must not do that.
    q = result["les_report"]["quality"]
    assert q["status"] == "DATA_INSUFFICIENT"
    assert q["metrics"]["quality_score_basis"] == "FIXED_100_BUDGET_MINUS_EVIDENCE_PENALTY"
    assert q["score"] == max(0., q["metrics"]["quality_raw_score"] - q["metrics"]["quality_penalty"])
    assert "cfo_reported" in q["metrics"]["quality_required_missing"]
    assert result["raw"]["les"] not in {"BUY", "BUY_SMALL"}
    assert not result["research_entry_eligible"]
    assert result["hard_gates"]["restrictions"] == "UNKNOWN"


def test_filing_revision_scope_and_cutoff_are_applied_before_native_derivation(tmp_path):
    payload = payload_fixture()
    old = deepcopy(payload["filings"]["2330"][-1])
    individual = deepcopy(old)
    individual.update(statement_scope="individual", revision_number=99, content_sha256="individual")
    changed = deepcopy(old)
    changed.update(available_at="2026-09-12T10:00:00+08:00", content_sha256="revised", revision_number=2)
    changed["facts"][0]["numeric_value"] *= 2
    payload["filings"]["2330"] += [individual, changed]
    evidence = PinnedReplayEvidence(digest(payload), payload)
    early = evidence.select(request(tmp_path,payload))
    late = evidence.select(request(tmp_path,payload,day="2026-09-14"))
    selected = lambda result: [r for r in result.evidence["filings"]["2330"] if r["selected"]]
    assert selected(early)[-1]["content_sha256"] == old["content_sha256"]
    assert selected(late)[-1]["content_sha256"] == "revised"
    assert early.market.fundamental_snapshots["2330"]["les_v6_filings"][-1]["revenue"] != late.market.fundamental_snapshots["2330"]["les_v6_filings"][-1]["revenue"]


def test_conflicting_restriction_evidence_cannot_be_clear(tmp_path):
    payload = payload_fixture()
    payload["restrictions"].append(dict(symbol="2330", evidence=dict(payload["restrictions"][0]["evidence"], restricted=True)))
    selected = PinnedReplayEvidence(digest(payload),payload).select(request(tmp_path,payload))
    assert selected.market.trading_restrictions == {}
    assert selected.diagnostics["restrictions"]["2330"]["reason"] == "CONFLICTING_RESTRICTION_OBSERVATIONS"


@pytest.mark.parametrize("holding", [
    ReplayHolding("2330", 1, "2026-09-12"), ReplayHolding("2330", 1, "2026-09-11T13:30:01+08:00"),
    ReplayHolding("2330", 1, "2026-09-11T13:00:00"), ReplayHolding("2330", -1, "2026-09-10"),
    ReplayHolding("2330", float("nan"), "2026-09-10"), ReplayHolding("2330", 1.5, "2026-09-10"),
    ReplayHolding("2330", True, "2026-09-10"),
])
def test_invalid_account_entries_rejected(tmp_path, holding):
    with pytest.raises(ValueError):
        request(tmp_path,payload_fixture(),account=ReplayAccount(0,100,(holding,)))


def test_accounts_are_explicit_and_reconciled_using_dated_prices(tmp_path):
    payload=payload_fixture()
    position=ReplayHolding("2330",100,"2026-08-05",82,"BUY")
    account=ReplayAccount(990330,1_000_000,(position,))
    req=request(tmp_path,payload,account=account)
    inputs= PinnedReplayEvidence(digest(payload),payload).select(req)
    assert inputs.account_validation["marks"]["2330"] == dict(date="2026-09-11",price=96.7,source="TWSE")
    assert inputs.market.entry_les_by_stock == {"2330":82}
    assert inputs.market.les_v6_portfolio["available_cash"] == 990330
    result=engine(payload).run(req)
    assert result["rows"][0]["PAE Role"] == "HOLDING"
    assert result["rows"][0]["EREM Holding Confidence"] is not None
    with pytest.raises(ValueError,match="RECONCILIATION"):
        engine(payload).run(replace(req,account=replace(account,cash=900000),replay_id="bad"))
    for bad in (ReplayAccount(float("inf"),1_000_000,()), ReplayAccount(-1,1_000_000,()),
                ReplayAccount(900000,1_000_000,()), ReplayAccount(0,1_000_000,(position,position))):
        with pytest.raises(ValueError):
            request(tmp_path,payload,account=bad)
    with pytest.raises(ValueError,match="REQUIRES_EXECUTION_TIME"):
        ReplayHolding("2330",1,"2026-09-11").validate(dt.datetime.fromisoformat("2026-09-11T08:30:00+08:00"))


def test_dates_no_hidden_warmup_no_silent_market_date_substitution(tmp_path):
    payload=payload_fixture()
    req=request(tmp_path,payload)
    start,end=calendar_window(req.history_end)
    assert (end-start).days==699
    evidence=PinnedReplayEvidence(digest(payload),payload)
    same=evidence.select(replace(req,history_start=req.history_end))
    assert same.market.dates == (("2026-09-11",),)
    weekend=evidence.select(replace(req,as_of=replay_close_cutoff("2026-09-13"),history_end=dt.date(2026,9,13)))
    assert weekend.signal_date=="2026-09-11" and weekend.diagnostics["calendar_status"]=="CLOSED"
    missing=evidence.select(replace(req,as_of=replay_close_cutoff("2026-09-14"),history_end=dt.date(2026,9,14)))
    assert missing.signal_date=="2026-09-14"
    assert "2026-09-14" in missing.diagnostics["missing_bars"]["2330"]
    closed=engine(payload).run(replace(req,replay_id="closed",as_of=replay_close_cutoff("2026-09-13"),history_start=dt.date(2026,9,13),history_end=dt.date(2026,9,13)))
    assert closed["status"]=="NO_TRADING_SESSION"
    assert closed["decisions"][0]["raw"]["les"] is None


def test_persistence_native_window_idempotence_resume_and_isolation(tmp_path):
    payload=payload_fixture()
    dates=("2026-09-09","2026-09-10","2026-09-11")
    observation=PersistenceObservationV2("2330","3014",True)
    def step(req,store):
        with store.session(req,dates) as session:
            if session.cached:
                return session.cached
            session.record_session_v2(req.as_of.date(),(observation,))
            recent=session.load_recent_v2(("2330",),("3014",),through_date=req.as_of.date(),limit_per_pair=3)
            report=dict(previous=session.previous_result_sha256, passes=list(recent[0].recent_passes))
            session.finish(report)
            return report
    store=SQLiteReplayState()
    continuous=[]
    for day in dates:
        continuous.append(step(request(tmp_path,payload,day=day,replay_id="continuous"),store))
    resumed=[]
    for day in dates:
        req=request(tmp_path,payload,day=day,replay_id="resumed")
        resumed.append(step(req,SQLiteReplayState()))
        assert step(req,SQLiteReplayState())==resumed[-1]
    assert continuous==resumed
    assert resumed[-1]["passes"]==[True,True,True]
    cold=step(request(tmp_path,payload,replay_id="new"),store)
    assert cold["passes"]==[False,False,True]
    with sqlite3.connect(tmp_path/"replay.sqlite3") as c:
        assert c.execute("SELECT count(*) FROM replay_observations WHERE replay_id='resumed'").fetchone()[0]==3


def test_state_rollback_versions_reverse_and_foreign_database(tmp_path):
    payload=payload_fixture(); req=request(tmp_path,payload); store=SQLiteReplayState()
    with pytest.raises(RuntimeError):
        with store.session(req,("2026-09-11",)) as session:
            session.record_session_v2(req.as_of.date(),(PersistenceObservationV2("2330","3014",True),))
            raise RuntimeError("interrupted")
    with store.session(req,("2026-09-11",)) as session:
        assert session.cached is None
        session.finish({"ok":True})
    for changed,reason in ((replace(req,versions=replace(VERSIONS,code="changed")),"VERSIONS"),
                           (replace(req,evidence_id="different"),"VERSIONS"),
                           (replace(req,history_start=req.history_start+dt.timedelta(days=1)),"INPUT_CHANGED"),
                           (replace(req,as_of=replay_close_cutoff("2026-09-10"),history_end=dt.date(2026,9,10)),"OUT_OF_ORDER")):
        with pytest.raises(ValueError,match=reason):
            with store.session(changed,("2026-09-11",)):
                pass
    live=tmp_path/"live.sqlite3"
    with sqlite3.connect(live) as c:
        c.execute("CREATE TABLE pae_opportunity_evidence (value TEXT)")
        c.execute("INSERT INTO pae_opportunity_evidence VALUES ('live')")
    before=live.read_bytes()
    with pytest.raises(ValueError,match="LIVE_OR_FOREIGN"):
        with store.session(replace(req,state_path=live),("2026-09-11",)):
            pass
    assert before==live.read_bytes()


def test_tampered_evidence_and_model_versions_rejected(tmp_path):
    payload=payload_fixture(); path=tmp_path/"evidence.json"
    save_evidence(payload,path)
    doc=json.loads(path.read_text());doc["payload"]["sector"]="13";path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="SIGNATURE"):
        load_evidence(path)
    with pytest.raises(ValueError,match="MODEL_OR_CODE"):
        engine(payload).run(replace(request(tmp_path,payload),versions=replace(VERSIONS,parameters="changed")))


def test_native_pae_persistence_remains_two_of_three_not_a_fast_path(tmp_path):
    from tests.strategies.test_portfolio_allocation_v2 import security, request as pae_request
    from domain.strategies.portfolio_allocation_v2 import PortfolioAllocationEngineV2
    from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
    holdings=tuple(security(str(1000+i),current_weight=.2,quality_score=80,bear_cagr=.05,base_cagr=.08,bull_cagr=.11) for i in range(4))
    candidate=security("2330",quality_score=84,data_confidence=80,bear_cagr=.18,base_cagr=.25,bull_cagr=.32)
    native=pae_request(holdings=holdings,candidates=(candidate,),cash_weight=.2)
    payload=payload_fixture(); dates=("2026-09-09","2026-09-10","2026-09-11")
    results=[]
    for day in dates[:2]:
        req=request(tmp_path,payload,day=day)
        with SQLiteReplayState().session(req,dates) as session:
            pipeline=PortfolioAllocationV2Pipeline(engine=PortfolioAllocationEngineV2(),persistence=session,portfolio_id=req.replay_id)
            result=pipeline._evaluate_with_persistence(replace(native,decision_time=req.as_of),req.as_of.date())
            line=next(line for line in result.lines if line.stock_id=="2330")
            results.append(line)
            session.finish({"decision":line.action.value})
    assert results[0].action.value=="NO_ACTION"
    assert results[1].action.value=="ROTATE_75"
    assert results[1].persistence_pass is True
    assert results[1].atomic_group_id


def test_skipped_sessions_and_weekend_cannot_inflate_persistence(tmp_path):
    payload=payload_fixture(); store=SQLiteReplayState()
    dates=("2026-09-07","2026-09-08","2026-09-09","2026-09-10","2026-09-11")
    observation=PersistenceObservationV2("2330","3014",True)
    for day in ("2026-09-07","2026-09-11","2026-09-13"):
        req=request(tmp_path,payload,day=day)
        with store.session(req,dates) as session:
            date=min(req.as_of.date(),dt.date(2026,9,11))
            session.record_session_v2(date,(observation,))
            evidence=session.load_recent_v2(("2330",),("3014",),through_date=date,limit_per_pair=3)
            session.finish({"passes":evidence[0].recent_passes})
    assert evidence[0].recent_passes==(False,False,True)
    with sqlite3.connect(req.state_path) as c:
        assert c.execute("SELECT count(*) FROM replay_observations").fetchone()[0]==2


def test_preparation_reuses_sqlite_queries_without_writes_or_http(tmp_path,monkeypatch):
    from infrastructure.storage.market_history_cache import SQLiteMarketHistoryCache
    from application.market_history_cache import CachedMarketDay
    from infrastructure.fundamentals.mops_repository import MopsRepository
    from tests.fundamentals.test_filing_refresh_planner import _candidate,_fact
    payload=payload_fixture()
    cache=tmp_path/"market.sqlite3"; repo=SQLiteMarketHistoryCache(cache)
    row=next(b["data"][0] for b in reversed(payload["bars"]) if b["dataset"]=="STOCK_TABLE")
    repo.upsert_day(CachedMarketDay("20260911",[row],20200.),market_type="24")
    mops=MopsRepository(tmp_path/"mops",connect_attempts=1)
    candidate=replace(_candidate(2025,4),symbol="2330",available_at="2026-08-01T00:00:00Z")
    mops.save_revision(candidate,b"fixture",[_fact(100.)],available_at=candidate.available_at,
                       fetched_at=candidate.available_at,availability_source="OFFICIAL_FILING_TIMESTAMP")
    before=(cache.read_bytes(),mops.database_path.read_bytes())
    monkeypatch.setattr("requests.sessions.Session.request",Mock(side_effect=AssertionError("HTTP")))
    path=tmp_path/"archive.json"
    eid=prepare_evidence(destination=path,market_cache=cache,mops_cache=mops.database_path,
        restriction_cache=None,restriction_file=None,calendar=dict(source="TWSE_OFFICIAL",days={"2026-09-11":"OPEN"}),
        symbols=("2330",),market="TWSE",sector="24",start=dt.date(2026,9,11),end=dt.date(2026,9,11))
    assert (cache.read_bytes(),mops.database_path.read_bytes())==before
    restored_id,restored=load_evidence(path)
    assert restored_id==eid
    selected=PinnedReplayEvidence(eid,restored).select(replace(request(tmp_path,restored),history_start=dt.date(2026,9,11)))
    assert selected.market.dates==(("2026-09-11",),)
    assert selected.evidence["filings"]["2330"][0]["fetched_at"]==candidate.available_at
    assert "HISTORICAL_RESTRICTIONS_MISSING" in restored["preparation_gaps"]
    assert selected.market.market_breadth_snapshots["2026-09-11"]["eligible_count"] == 1


def test_native_engine_continuous_and_process_resume_are_identical(tmp_path):
    payload = payload_fixture()
    continuous = engine(payload)
    for day in ("2026-09-10", "2026-09-11"):
        req = request(tmp_path, payload, day=day)
        expected = continuous.run(replace(req, state_path=tmp_path/"continuous.sqlite3"))
        resumed_request = replace(req, state_path=tmp_path/"resumed.sqlite3")
        assert engine(payload).run(resumed_request) == expected
        assert engine(payload).run(resumed_request) == expected


def test_empirical_kelly_is_pinned_cutoff_checked_and_not_inferred(tmp_path):
    payload = payload_fixture()
    calibration = dict(symbol="2330", market="TWSE", available_at="2026-08-31T13:30:00+08:00",
        training_end="2026-08-28T13:30:00+08:00", half_kelly_weight=.1, sample_count=500,
        method="empirical-test-fixture", source_id="test-only-calibration", model_version=VERSIONS.model,
        parameter_sha256=VERSIONS.parameters)
    payload["kelly_calibrations"] = [calibration]
    result = engine(payload).run(request(tmp_path, payload))
    assert result["rows"][0]["Half Kelly Weight"] == .1
    assert result["decisions"][0]["hard_gates"]["kelly"] == "PASS"
    for change in ({"available_at":"2026-09-12T13:30:00+08:00"}, {"half_kelly_weight":None},
                   {"parameter_sha256":"wrong"}, {"training_end":"2027-01-01T13:30:00+08:00"}):
        changed = dict(payload, kelly_calibrations=[dict(calibration, **change)])
        selected = PinnedReplayEvidence(digest(changed),changed).select(request(tmp_path,changed))
        assert selected.calibrations == {}
        assert selected.diagnostics["kelly"]["2330"]["excluded"]


def test_date_cutoff_is_close_only_and_never_naive(tmp_path):
    req = request(tmp_path, payload_fixture())
    for value in ("2026-09-11T08:30:00+08:00", "2026-09-11T12:00:00+08:00", "2026-09-11T13:30:00"):
        with pytest.raises(ValueError):
            replace(req, as_of=dt.datetime.fromisoformat(value))
    assert replace(req, as_of=dt.datetime.fromisoformat("2026-09-11T05:30:00+00:00")).as_of == req.as_of


def test_unfinished_research_transaction_is_rolled_back(tmp_path):
    req=request(tmp_path,payload_fixture())
    with pytest.raises(ValueError,match="NOT_FINISHED"):
        with SQLiteReplayState().session(req,("2026-09-11",)) as session:
            session.record_session_v2(req.as_of.date(),(PersistenceObservationV2("2330","3014",True),))
    with SQLiteReplayState().session(req,("2026-09-11",)) as session:
        assert session.cached is None
        assert session.load_recent_v2(("2330",),("3014",),through_date=req.as_of.date(),limit_per_pair=3)[0].recent_passes == (False,)
        session.finish({"verified":True})
