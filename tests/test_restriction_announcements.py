"""Offline evidence contracts, temporal reconstruction and replay-only safety."""
from dataclasses import replace
import datetime as dt
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from application.restriction_history import HistoricalRestrictionLedger
from domain.restriction_announcements import BUNDLE_SCHEMA, CATEGORIES
from domain.trading_restrictions import restriction_check
from domain.strategies.low_entry_score_v6 import assess_execution
from domain.strategies.low_entry_v6_report import report_failure
from infrastructure.crawler.restriction_history import HistoricalRestrictionCollector
from infrastructure.crawler.twse_http_client import TwseAccessBlockedError, TwseSchemaChangedError
from infrastructure.runtime.market_data_adapter import CrawlerMarketDataAdapter
from infrastructure.runtime.strategy_adapter import CrawlerStrategyAdapter
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.contracts import AnalysisContext, MarketStatus
from interface.restriction_history import main
from tests.test_historical_replay import close_replay_market, request, WeekdayCalendar, replay_args
from tests.strategies.test_daily_v6_pipeline import daily_market, build
from interface.cli import build_run_configuration


CUTOFF = "2026-09-16T13:30:00+08:00"
RETRIEVED = "2026-09-26T12:00:00+08:00"


def bundle(*events, market="TWSE"):
    # Synthetic official-shaped records, not actual exchange evidence.
    body = "SYNTHETIC TEST ONLY: complete catalog; carry-in and all original revisions verified. Notice text."
    return {
        "schema_version": BUNDLE_SCHEMA, "market": market,
        "documents": {"source": {"url": "https://www." + ("twse.com.tw" if market == "TWSE" else "tpex.org.tw") + "/test",
                                  "body_utf8": body, "sha256": hashlib.sha256(body.encode()).hexdigest(),
                                  "retrieved_at": RETRIEVED}},
        "coverage": [{"category": category, "start_at": "2026-09-01T00:00:00+08:00",
                      "end_at": "2026-09-30T23:59:59+08:00", "status": "VERIFIED",
                      "complete": True, "includes_carry_in": True, "includes_all_revisions": True,
                      "universe_scope": "MARKET_WIDE",
                      "verified_by": "SYNTHETIC_TEST", "verified_at": RETRIEVED,
                      "proofs": [{"document_id": "source", "quote": "complete catalog"}]}
                     for category in CATEGORIES],
        "announcements": list(events),
    }


def event(**patch):
    return dict({"id": "NOTICE-1", "version": 1, "category": "DISPOSITION", "symbol": "2330",
                 "published_at": "2026-09-10T16:00:00+08:00", "publication_precision": "TIMESTAMP",
                 "effective_from": "2026-09-11T00:00:00+08:00", "effective_until": None, "withdrawn": False,
                 "proofs": [{"document_id": "source", "quote": "Notice text"}]}, **patch)


def check(raw=None, *, as_of=CUTOFF, market="TWSE", symbol="2330", bar="2026-09-16"):
    evidence = HistoricalRestrictionLedger(bundle() if raw is None else raw).reconstruct(
        [symbol], market=market, as_of=as_of, market_data_date=bar)[symbol]
    return restriction_check(evidence, market=market, market_data_date=bar, decision_time=as_of,
                             allow_historical=True, symbol=symbol)


def test_later_retrieval_does_not_become_publication_or_current_observation():
    ledger = HistoricalRestrictionLedger(bundle())
    evidence = ledger.reconstruct(["2330"], market="TWSE", as_of=CUTOFF, market_data_date="2026-09-16")["2330"]
    assert evidence["documents"]["source"]["retrieved_at"] == RETRIEVED
    assert "available_at" not in evidence
    assert check()["status"] == "CLEAR"
    assert restriction_check(evidence, decision_time=CUTOFF, market_data_date="2026-09-16")["reason"] == "HISTORICAL_RESTRICTIONS_REPLAY_ONLY"
    assert restriction_check(evidence, decision_time=CUTOFF, market_data_date="2026-09-16",
                             allow_historical=True, symbol="3016")["reason"] == "HISTORICAL_RESTRICTION_SYMBOL_MISMATCH"
    assert restriction_check(evidence, decision_time=CUTOFF, market_data_date="2026-09-16", market="TPEX",
                             allow_historical=True, symbol="2330")["status"] == "UNKNOWN"
    assert restriction_check(evidence, decision_time="2026-09-17T13:30:00+08:00", market_data_date="2026-09-16",
                             allow_historical=True, symbol="2330")["reason"] == "HISTORICAL_RESTRICTION_CUTOFF_MISMATCH"


@pytest.mark.parametrize("category", CATEGORIES)
def test_each_category_and_carry_in_restriction_blocks(category):
    result = check(bundle(event(category=category, published_at="2026-08-01T08:00:00+08:00",
                                effective_from="2026-08-02T00:00:00+08:00")))
    assert result["status"] == "RESTRICTED"
    assert result["categories"][category]["active_ids"] == ["NOTICE-1"]


@pytest.mark.parametrize("change", ["missing", "unverified", "range", "all_missing"])
def test_absence_in_incomplete_catalog_never_proves_clear(change):
    raw = bundle()
    if change == "missing":
        raw["coverage"].pop()
    elif change == "all_missing":
        raw["coverage"] = []
    elif change == "range":
        raw["coverage"][0]["end_at"] = "2026-09-15T23:59:59+08:00"
    else:
        raw["coverage"][0]["status"] = "UNVERIFIED"
    assert check(raw)["status"] == "UNKNOWN"


def test_known_active_restriction_not_hidden_by_other_missing_category():
    raw = bundle(event())
    raw["coverage"] = []
    assert check(raw)["restricted"] is True


@pytest.mark.parametrize("field", ["complete", "includes_carry_in", "includes_all_revisions"])
def test_verified_requires_all_completeness_attestations(field):
    raw = bundle()
    raw["coverage"][0][field] = False
    with pytest.raises(ValueError, match="COMPLETE_COVERAGE_ATTESTATION_REQUIRED"):
        check(raw)


@pytest.mark.parametrize("end,status", [(None, "RESTRICTED"), (CUTOFF, "CLEAR"),
                                       ("2026-09-16T13:30:01+08:00", "RESTRICTED")])
def test_end_is_exclusive_and_must_belong_to_original_published_version(end, status):
    assert check(bundle(event(effective_until=end)))["status"] == status


def test_later_announced_resumption_cannot_rewrite_earlier_history():
    original = event(category="TRADING_HALT")
    revised = event(category="TRADING_HALT", version=2, published_at="2026-09-17T08:00:00+08:00",
                    effective_until="2026-09-16T09:00:00+08:00")
    raw = bundle(original, revised)
    assert check(raw)["status"] == "RESTRICTED"
    assert check(raw, as_of="2026-09-17T13:30:00+08:00")["status"] == "CLEAR"


def test_future_announcements_and_future_effective_dates_do_not_rewrite_cutoff():
    assert check(bundle(event(published_at="2026-09-17T08:00:00+08:00")))["status"] == "CLEAR"
    assert check(bundle(event(effective_from="2026-09-17T08:00:00+08:00")))["status"] == "CLEAR"


@pytest.mark.parametrize("published,status", [(CUTOFF, "RESTRICTED"),
                                             ("2026-09-16T13:30:01+08:00", "CLEAR")])
def test_same_day_publication_keeps_second_precision_at_close(published, status):
    assert check(bundle(event(published_at=published)))["status"] == status


def test_withdrawals_overlap_and_per_symbol_independence():
    withdrawn = event(version=2, published_at="2026-09-15T08:00:00+08:00", withdrawn=True)
    assert check(bundle(event(), withdrawn))["status"] == "CLEAR"
    assert check(bundle(event(), withdrawn, event(id="OTHER")))["status"] == "RESTRICTED"
    assert check(bundle(event(symbol="3016")))["status"] == "CLEAR"


@pytest.mark.parametrize("published,status", [("2026-09-15", "RESTRICTED"), ("2026-09-16", "UNKNOWN"),
                                             ("2026-09-17", "CLEAR")])
def test_date_only_publication_never_assumes_midnight_release(published, status):
    assert check(bundle(event(published_at=published, publication_precision="DATE")))["status"] == status


def test_missing_prior_version_never_falls_back_to_clear():
    assert check(bundle(event(version=2, withdrawn=True)))["status"] == "UNKNOWN"
    assert check(bundle(event(version=2, published_at="2026-09-17T08:00:00+08:00")))["status"] == "UNKNOWN"


def test_coverage_at_later_cutoff_cannot_certify_uncovered_signal_day():
    raw = bundle()
    for row in raw['coverage']:
        row['start_at'] = '2026-09-12T00:00:00+08:00'
    result = check(raw, as_of='2026-09-13T13:30:00+08:00', bar='2026-09-11')
    assert result['status'] == 'UNKNOWN'
    assert result['market_effective_at'] == '2026-09-11T13:30:00+08:00'


@pytest.mark.parametrize('start,end,market_ids,decision_ids', [
    ('2026-09-11T00:00:00+08:00', '2026-09-12T00:00:00+08:00', ['NOTICE-1'], []),
    ('2026-09-12T00:00:00+08:00', None, [], ['NOTICE-1']),
])
def test_weekend_cutoff_checks_both_signal_day_and_current_restrictions(start, end, market_ids, decision_ids):
    result = check(bundle(event(effective_from=start, effective_until=end)),
                   as_of='2026-09-13T13:30:00+08:00', bar='2026-09-11')
    assert result['status'] == 'RESTRICTED'
    state = result['categories']['DISPOSITION']
    assert state['market_active_ids'] == market_ids
    assert state['decision_active_ids'] == decision_ids


def test_complete_empty_historical_coverage_allows_clear_on_weekend():
    assert check(bundle(), as_of='2026-09-13T13:30:00+08:00', bar='2026-09-11')['status'] == 'CLEAR'


@pytest.mark.parametrize("kind", ["hash", "quote", "url", "naive", "precision", "duplicate", "identity", "future_download"])
def test_import_rejects_unverifiable_or_conflicting_records(kind):
    raw = bundle(event())
    if kind == "hash":
        raw["documents"]["source"]["body_utf8"] += "tampered"
    elif kind == "quote":
        raw["announcements"][0]["proofs"][0]["quote"] = "invented text"
    elif kind == "url":
        raw["documents"]["source"]["url"] = "https://twse.com.tw.evil.example/source"
    elif kind == "naive":
        raw["announcements"][0]["effective_from"] = "2026-09-11"
    elif kind == "precision":
        raw["announcements"][0].pop("publication_precision")
    elif kind == "duplicate":
        raw["announcements"].append(event())
    elif kind == "identity":
        raw["announcements"].append(event(version=2, symbol="3016"))
    else:
        raw["documents"]["source"]["retrieved_at"] = "2026-01-01T00:00:00Z"
    with pytest.raises(ValueError):
        HistoricalRestrictionLedger(raw)


def test_derived_status_is_never_trusted_and_market_data_cannot_be_future():
    evidence = HistoricalRestrictionLedger(bundle(event())).reconstruct(
        ["2330"], market="TWSE", as_of=CUTOFF, market_data_date="2026-09-16")["2330"]
    evidence["restricted"] = False
    evidence["reconstruction_result"] = {"status": "CLEAR"}
    assert restriction_check(evidence, decision_time=CUTOFF, market_data_date="2026-09-16",
                             allow_historical=True, symbol="2330")["restricted"] is True
    assert check(bar="2026-09-17")["status"] == "UNKNOWN"
    assert check(as_of="2026-09-16T09:00:00+08:00")["status"] == "UNKNOWN"


def test_tpex_verified_import_is_independent_of_twse_collector():
    assert check(bundle(market="TPEX"), market="TPEX")["status"] == "CLEAR"
    with pytest.raises(ValueError, match="MARKET_MISMATCH"):
        HistoricalRestrictionLedger(bundle()).reconstruct(["2330"], market="TPEX", as_of=CUTOFF,
                                                         market_data_date="2026-09-16")


def test_collector_is_bounded_and_does_not_auto_certify_empty_queries():
    http = Mock()
    http.request_bytes.return_value = SimpleNamespace(content=b'{"stat":"OK","data":[],"fields":[]}')
    raw = HistoricalRestrictionCollector(http, clock=lambda: dt.datetime.fromisoformat(RETRIEVED)).collect(
        start=dt.date(2026, 9, 1), end=dt.date(2026, 9, 16))
    assert http.request_bytes.call_count == 3
    assert all(row["status"] == "UNVERIFIED" for row in raw["coverage"])
    assert check(raw)["status"] == "UNKNOWN"


@pytest.mark.parametrize("error", [TwseAccessBlockedError("blocked", retryable=False),
                                  TwseSchemaChangedError("changed")])
def test_collector_stops_at_first_error(error):
    http = Mock()
    http.request_bytes.side_effect = error
    with pytest.raises(type(error)):
        HistoricalRestrictionCollector(http).collect(start=dt.date(2026, 9, 1), end=dt.date(2026, 9, 16))
    assert http.request_bytes.call_count == 1


def test_cli_reconstruction_deterministic_immutable_and_accepted_by_replay(tmp_path):
    source, output = tmp_path / "sources.json", tmp_path / "evidence.json"
    source.write_text(json.dumps(bundle()))
    args = ["reconstruct", "--bundle", str(source), "--output", str(output), "--market", "TWSE",
            "-e", "2026-09-16", "--market-data-date", "2026-09-16", "--symbols", "2330"]
    assert main(args) == 0
    first = output.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == first
    assert HistoricalRestrictionLedger.from_file(source).reconstruct(
        ["2330"], market="TWSE", as_of=CUTOFF, market_data_date="2026-09-16") == json.loads(first)
    req, _ = build_run_configuration(replay_args(tmp_path) + ["--replay-restrictions-file", str(source)])
    assert req.replay_restrictions_file == str(source)


@pytest.mark.parametrize("cutoff", ["2026-09-16T09:00:00+08:00", "2026-09-16T13:30:00+08:00",
                                   "2026-09-16T05:30:00Z"])
def test_reconstruction_cli_rejects_timestamps_before_reading_or_writing(tmp_path, cutoff, monkeypatch):
    load = Mock(side_effect=AssertionError("invalid cutoff must be rejected before loading evidence"))
    monkeypatch.setattr(HistoricalRestrictionLedger, "from_file", load)
    output = tmp_path / "evidence.json"
    with pytest.raises(SystemExit) as error:
        main(["reconstruct", "--bundle", str(tmp_path / "missing.json"), "--output", str(output),
              "--market", "TWSE", "-e", cutoff, "--market-data-date", "2026-09-16", "--symbols", "2330"])
    assert error.value.code == 2
    load.assert_not_called()
    assert not output.exists()


def test_reconstruction_cli_rejects_removed_as_of_even_with_valid_end(tmp_path, monkeypatch, capsys):
    load = Mock(side_effect=AssertionError("removed option must be rejected before loading evidence"))
    monkeypatch.setattr(HistoricalRestrictionLedger, "from_file", load)
    output = tmp_path / "evidence.json"
    with pytest.raises(SystemExit) as error:
        main(["reconstruct", "--bundle", str(tmp_path / "missing.json"), "--output", str(output),
              "--market", "TWSE", "-e", "2026-09-16", "--as-of", "2026-09-16",
              "--market-data-date", "2026-09-16", "--symbols", "2330"])
    assert error.value.code == 2
    assert "unrecognized arguments: --as-of" in capsys.readouterr().err
    load.assert_not_called()
    assert not output.exists()


def test_runtime_reconstructs_once_after_universe_resolution_and_never_fetches_live(tmp_path):
    source = tmp_path / "sources.json"
    source.write_text(json.dumps(bundle()))
    crawler = SimpleNamespace(iso_scheduled_times=["2026-09-16"], stocknumbers=[], get_date_times=Mock())
    crawler.get_twse_daily_stocks = Mock(side_effect=lambda **kwargs: setattr(crawler, "stocknumbers", ["2330", "3016"]))
    repository = Mock()
    req = request(replay_restrictions_file=str(source))
    adapter = CrawlerMarketDataAdapter(lambda count: crawler, restriction_repository=repository)
    adapter.fetch(req, calendar=WeekdayCalendar(), market_date=req.as_of.date(), is_trading_day=True, include_reference=True)
    assert set(crawler.trading_restrictions) == {"2330", "3016"}
    assert all(row["reconstruction_result"]["status"] == "CLEAR" for row in crawler.trading_restrictions.values())
    repository.load_as_of.assert_not_called()


@pytest.mark.usefixtures("active_industry_profiles")
@pytest.mark.parametrize("complete", [True, False])
def test_les_erem_pae_share_reconstructed_cutoff_and_remain_research_only(complete):
    market = close_replay_market()
    cutoff = market.signal_dates["2330"]
    raw = bundle()
    if not complete:
        raw["coverage"][0]["status"] = "UNVERIFIED"
    restrictions = HistoricalRestrictionLedger(raw).reconstruct(
        ["2330"], market="TWSE", as_of=cutoff, market_data_date="2026-09-11")
    market = replace(market, trading_restrictions=restrictions, replay_as_of=cutoff)
    builder, decisions = build(market)
    persistence = Mock()
    allocator = PortfolioAllocationV2Pipeline(persistence=persistence)
    crawler = SimpleNamespace(cal_max_profit=Mock(return_value=[]), build_strategy_decisions=Mock(return_value=decisions))
    context = AnalysisContext(crawler, dt.date(2026, 9, 11), MarketStatus.OPEN, dt.datetime.fromisoformat(cutoff))
    evaluation = CrawlerStrategyAdapter(portfolio_allocator=allocator, portfolio_current_cash=1_000_000).evaluate(context)
    decision = evaluation.decisions[0]
    report = json.loads(decision.value("LES v6 Report"))
    assert decision.value("EREM 分數") is not None
    assert decision.value("PAE Status") is not None
    assert report["data"]["restriction_check"]["status"] == ("CLEAR" if complete else "UNKNOWN")
    if complete:
        assert report["decision"]["signal"] == "BUY_SMALL"
        assert report_failure(report, allow_replay=True) is None
    else:
        assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}
        assert report["sizing"]["indicative_shares"] == 0
        assert decision.value("PAE Executable Candidate") is False
        assert decision.value("PAE Target Weight") == 0
    assert report_failure(report) == "HISTORICAL_REPLAY_NOT_EXECUTABLE"
    assert assess_execution(report, {}, now=cutoff, trading_sessions=[], portfolio={})["action"] == "SKIP"
    persistence.load_recent_v2.assert_not_called()
    persistence.record_session_v2.assert_not_called()


def test_removing_replay_label_cannot_make_historical_evidence_live_executable():
    market = daily_market()
    cutoff = market.signal_dates["2330"]
    historical = HistoricalRestrictionLedger(bundle()).reconstruct(
        ["2330"], market="TWSE", as_of=cutoff, market_data_date="2026-09-11")
    _, decisions = build(replace(market, trading_restrictions=historical))
    report = json.loads(decisions[0].value("LES v6 Report"))
    assert report["data"]["restriction_check"]["reason"] == "HISTORICAL_RESTRICTIONS_REPLAY_ONLY"
    assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}


def test_partial_universe_cannot_certify_whole_market_absence():
    raw = bundle()
    raw["coverage"][0]["universe_scope"] = "SELECTED_SYMBOLS"
    with pytest.raises(ValueError, match="COMPLETE_COVERAGE_ATTESTATION_REQUIRED"):
        HistoricalRestrictionLedger(raw)


def test_version_publication_time_cannot_run_backwards():
    raw = bundle(event(), event(version=2, published_at="2026-09-09T08:00:00+08:00", withdrawn=True))
    assert check(raw)["reason"] == "NON_MONOTONIC_ANNOUNCEMENT_VERSIONS"


@pytest.mark.parametrize("body", [b'<html>Access Denied</html>', b'{"stat":"OK"}',
                                  b'{"stat":"not OK","data":[]}', b'[]'])
def test_collector_rejects_blocked_html_or_incomplete_response(body):
    http = Mock()
    http.request_bytes.return_value = SimpleNamespace(content=body)
    with pytest.raises(TwseSchemaChangedError):
        HistoricalRestrictionCollector(http).collect(start=dt.date(2026, 9, 1), end=dt.date(2026, 9, 16))
    assert http.request_bytes.call_count == 1
