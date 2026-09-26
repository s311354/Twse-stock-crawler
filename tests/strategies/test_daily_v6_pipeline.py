"""Offline daily raw data -> v6 -> typed decisions -> files -> HTML/MIME."""
from copy import deepcopy
from dataclasses import replace
import datetime as dt
from email import message_from_string
from email.header import decode_header, make_header
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.market_history import MarketHistoryInput
from application.strategy_contracts import StrategyContractError, StrategyDecisionAssembler
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from domain.model_contracts import FieldSource, FieldUnit
from domain.trading_restrictions import RESTRICTION_ENDPOINTS
from domain.strategies.low_entry_score_v6 import LowEntryScoreV6Strategy, attach_low_entry_v6_context
from domain.strategies.low_entry_v6_report import report_failure
from infrastructure.crawler.trading_restrictions import TradingRestrictionClient
from infrastructure.report.html_renderer import RendererFactory
from infrastructure.storage.csv_repository import CsvRepository
from aistock_twse_integration.les_v6_execution import load_reports, execute_request
from tests.strategies.test_low_entry_score_v6 import financials, ACCOUNT


def daily_market():
    count = 500
    dates = pd.bdate_range(end="2026-09-11", periods=count).strftime("%Y-%m-%d").tolist()
    close = np.r_[np.linspace(70., 115., 400), np.linspace(115., 120., 60), np.linspace(120., 95., 30),
                  [95., 94.8, 95.1, 94.9, 95., 95.3, 95.6, 95.9, 96.2, 96.7]]
    low = close - 3.
    low[-5:] = [94.5, 94.8, 94.4, 95.7, 96.1]
    high = close + .1
    high[-20] += 3.  # Frozen prior wick; sufficient room after configured costs.
    scaled = lambda values: [(values * 1000).tolist()]
    observed = "2026-09-11T15:00:00+08:00"
    return MarketHistoryInput(
        stock_count=1, stock_numbers=["2330"], stock_names=["測試公司"], dates=[dates],
        opens=scaled(close-.4), highs=scaled(high), lows=scaled(low), closes=scaled(close),
        volumes=[[1_000_000.]*499+[1_200_000.]], trade_counts=[[500]*count],
        trade_values=[(close * 1_000_000.).tolist()], bids=scaled(close-.05), asks=scaled(close+.05),
        pe_ratios=[[20.]*count], taiex_closes=[np.linspace(19000., 20200., count).tolist()],
        annual_fundamentals={}, fundamental_snapshots={"2330": {"sector": "24", "les_v6_filings": financials()}},
        fundamental_statuses={}, fundamental_status_details={}, signal_dates={"2330": observed},
        account_equity=1_000_000., default_sector="ELEC", les_v6_portfolio=ACCOUNT,
        trading_restrictions={"2330": {"restricted": False, "session_date": dates[-1], "available_at": observed,
                                       "source": "TWSE_OPENAPI", "endpoints": list(RESTRICTION_ENDPOINTS)}},
    )


def build(market=None):
    builder = AnalysisDatasetBuilder()
    decisions = builder.build_decisions(market or daily_market(), [[1, 1, 1, 1, .01]])
    return builder, decisions


@pytest.mark.usefixtures("active_industry_profiles")
def test_raw_daily_generation_is_real_v6_not_relabelled_v5(tmp_path):
    builder, decisions = build()
    decision = decisions[0]
    report = json.loads(decision.value("LES v6 Report"))
    assert report["decision"]["signal"] == decision.value("低點決策") == "BUY"
    assert report["decision"]["score"] >= 80
    assert report["quality"]["status"] == decision.value("Quality Status") == "PASS"
    # Intentionally absent legacy annual snapshot: v5 would NOT issue this BUY.
    assert decision.value("LES v5 Final Quality Status") == "DATA_INSUFFICIENT"
    assert decision.field("LES v5 Final Quality Status").source == FieldSource.LES_V5
    assert decision.field("低點決策").source == FieldSource.LES_V6
    assert decision.field("低點決策").model_version == "6.0.0"
    assert decision.field("Valuation Model Version").source == FieldSource.LES_V5_VALUATION
    assert decision.field("EREM 分數").source == FieldSource.EREM_V3
    assert decision.field("策略停損").unit == FieldUnit.TWD_SCALED_1000
    assert decision.value("策略停損") == pytest.approx(report["entry"]["stop_price"] * 1000)
    assert report["technical"]["close"] == pytest.approx(96.7)
    assert report["technical"]["volume"] == 1_200_000.
    assert report_failure(report) is None
    assert report["sizing"]["indicative_shares"] > 0 and report["entry"]["is_fill"] is False

    dataset = builder.project_decisions(decisions)
    CsvRepository(tmp_path).write_analysis_dataset("elec", dataset)
    restored = pd.read_csv(tmp_path / "elec_analysis_dataset.csv", dtype={"證券代號": str})
    artifact = load_reports(str(tmp_path / "elec_les_v6_reports.json"), "2026-09-11")
    assert artifact["2330"] == report == json.loads(restored.iloc[0]["LES v6 Report"])
    rendered = RendererFactory.get_renderer("daily_strategy_email").render(decisions)
    assert "Low-Entry Model v6" in rendered and "Low-Entry Model v5" not in rendered
    assert "REVERSAL" in rendered and "EREM v3" in rendered and "下一交易日" in rendered
    assert "responsive-decision-table" in rendered and 'data-les-version="6.0.0"' in rendered
    assert "96.80" in rendered and "96,800" not in rendered


@pytest.mark.parametrize("change,reason", [
    ("filings", "QUALITY_DATA_INSUFFICIENT"),
    ("restriction", "TRADING_RESTRICTION_UNKNOWN"),
    ("future_restriction", "TRADING_RESTRICTION_UNKNOWN"),
    ("prices", "V6_INPUT_INVALID"),
])
def test_missing_or_future_data_never_falls_back_to_v5_buy(change, reason):
    market = daily_market()
    if change == "filings":
        market = replace(market, fundamental_snapshots={})
    elif change == "restriction":
        market = replace(market, trading_restrictions={})
    elif change == "future_restriction":
        evidence = deepcopy(market.trading_restrictions)
        evidence["2330"]["available_at"] = "2026-09-12T09:00:00+08:00"
        market = replace(market, trading_restrictions=evidence)
    else:
        market = replace(market, signal_dates={"2330": "2026-09-11T12:00:00+08:00"})
        # Typed daily contracts prohibit market data from after evaluation time.
        with pytest.raises(StrategyContractError, match="future point-in-time"):
            build(market)
        return
    _, decisions = build(market)
    assert decisions[0].value("低點決策") not in {"BUY", "BUY_SMALL"}
    assert reason in decisions[0].value("低點理由")
    assert decisions[0].value("建議股數") == 0


def test_daily_veto_and_known_fundamental_hard_risk_are_not_score_overrides():
    market = daily_market()
    snapshot = deepcopy(market.fundamental_snapshots)
    snapshot["2330"].update(going_concern=True, available_at="2026-09-11T14:00:00+08:00")
    _, decisions = build(replace(market, fundamental_snapshots=snapshot))
    assert decisions[0].value("Quality Status") == "REJECT"
    assert decisions[0].value("低點決策") == "NO_BUY"
    market = replace(market, opens=deepcopy(market.opens), highs=deepcopy(market.highs), lows=deepcopy(market.lows), closes=deepcopy(market.closes))
    for field, value in ((market.opens, 95_000.), (market.highs, 96_000.), (market.lows, 69_000.), (market.closes, 70_000.)):
        field[0][-1] = value
    _, decisions = build(market)
    assert decisions[0].value("Falling Knife Veto") is True
    assert decisions[0].value("低點決策") == "NO_BUY"


@pytest.mark.parametrize("evidence", [
    {"audit_review_required": True},
    {"current_audit_opinion": "UNKNOWN", "audit_evidence_available": True,
     "material_accounting_issue": None},
    {"audit_opinion": "QUALIFIED", "audit_evidence_available": True},
])
def test_unresolved_known_audit_evidence_cannot_be_optional_missing_data(evidence):
    market = daily_market()
    snapshot = deepcopy(market.fundamental_snapshots)
    snapshot["2330"].update(evidence, available_at="2026-09-11T14:00:00+08:00")
    _, decisions = build(replace(market, fundamental_snapshots=snapshot))
    report = json.loads(decisions[0].value("LES v6 Report"))
    assert report["quality"]["score"] >= 75
    assert report["quality"]["status"] == "DATA_INSUFFICIENT"
    assert report["quality"]["reasons"] == ["AUDIT_REVIEW_REQUIRED"]
    assert report["decision"]["signal"] == "DATA_INSUFFICIENT"
    assert report["sizing"]["indicative_shares"] == 0


@pytest.mark.parametrize("evidence", [{"audit_review_required": True}, {"material_accounting_issue": True}])
def test_undated_audit_evidence_blocks_entry_instead_of_being_ignored(evidence):
    market = daily_market()
    snapshot = deepcopy(market.fundamental_snapshots)
    snapshot["2330"].update(evidence)
    _, decisions = build(replace(market, fundamental_snapshots=snapshot))
    report = json.loads(decisions[0].value("LES v6 Report"))
    assert report["decision"]["signal"] == "DATA_INSUFFICIENT"
    assert report["quality"]["reasons"] == ["AUDIT_EVIDENCE_DATE_UNKNOWN"]


def test_future_snapshot_is_rejected_and_known_hard_risk_has_priority():
    market = daily_market()
    snapshot = deepcopy(market.fundamental_snapshots)
    snapshot["2330"].update(audit_review_required=True, available_at="2026-09-12T09:00:00+08:00")
    with pytest.raises(StrategyContractError, match="future point-in-time evidence"):
        build(replace(market, fundamental_snapshots=snapshot))
    snapshot["2330"].update(material_accounting_issue=True, available_at="2026-09-11T14:00:00+08:00")
    _, decisions = build(replace(market, fundamental_snapshots=snapshot))
    assert decisions[0].value("Quality Status") == "REJECT"
    assert decisions[0].value("低點決策") == "NO_BUY"


@pytest.mark.usefixtures("active_industry_profiles")
def test_pae_consumes_v6_quality_but_keeps_independent_valuation_gate():
    _, decisions = build()
    pipeline = PortfolioAllocationV2Pipeline()
    context = PortfolioRuntimeContext(account_equity=1_000_000., current_cash=500_000.)
    security = pipeline._security(decisions[0], context, position=None)
    assert security.quality_status == "PASS" and security.quality_score == decisions[0].value("Quality Score")
    assert security.les_decision == "BUY"
    enriched = pipeline.enrich(decisions, context)
    assert enriched[0].value("低點決策") == "BUY"
    assert enriched[0].value("PAE Executable Candidate") is not True  # No manufactured DCF/CER.


@pytest.mark.usefixtures("active_industry_profiles")
def test_no_cash_means_no_executable_shares_not_no_signal():
    _, decisions = build(replace(daily_market(), les_v6_portfolio={}))
    assert decisions[0].value("低點決策") == "BUY"
    assert decisions[0].value("建議股數") == 0


@pytest.mark.usefixtures("active_industry_profiles")
def test_daily_report_execution_checks_stale_gap_and_cash():
    _, decisions = build()
    report = json.loads(decisions[0].value("LES v6 Report"))
    quote = {"symbol": "2330", "timestamp": "2026-09-14T09:10:00+08:00", "price": 96.9, "bid": 96.8, "ask": 96.9, "restricted": False}
    request = dict(report=report, quote=quote, now=quote["timestamp"], trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT, hard_risk=False)
    assert execute_request(request)["action"] == "ELIGIBLE"
    assert execute_request(dict(request, now="2026-09-14T09:12:00+08:00"))["reason"] == "STALE_OR_MISSING_QUOTE"
    assert execute_request(dict(request, quote=dict(quote, price=140, bid=139.9, ask=140)))["action"] == "SKIP"
    assert execute_request(dict(request, portfolio=dict(ACCOUNT, available_cash=0)))["action"] == "SKIP"


def test_subject_and_html_come_from_same_v6_decisions_without_sending(monkeypatch):
    import twstockcrawler
    from textmewhenitsdone import TextMeWhenItsDone
    from tests.test_stockanalysis import CapturingTwseCrawler
    _, decisions = build()
    transport = object.__new__(TextMeWhenItsDone)
    transport.server = Mock()
    class Mail:
        def __init__(self, *args, **kwargs): pass
        def smtpauthentication(self): pass
        def textstockprofittable(self, **kwargs):
            transport.textstockprofittableme(subject="test", email="test@example.invalid", password="unused", receiver="test@example.invalid", **kwargs)
    monkeypatch.setattr("infrastructure.legacy_crawler_facade.SMTPEmail", Mail)
    crawler = CapturingTwseCrawler(1)
    crawler.iso_scheduled_times = ["2026-09-11"]
    crawler.smtp_strategy_email("test", None, "ELEC", decisions)
    message = message_from_string(transport.server.sendmail.call_args.kwargs["msg"])
    assert "LES v6 / EREM v3" in str(make_header(decode_header(message["Subject"])))
    assert "Low-Entry Model v6" in message.get_payload()[0].get_payload(decode=True).decode("utf-8")


def test_restriction_lists_must_all_succeed_and_never_backfill():
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        rows = [{"Code": "2330"}] if url.endswith("TWT85U") else []
        return SimpleNamespace(payload=rows)
    client = TradingRestrictionClient(SimpleNamespace(get_json=get))
    observed = dt.datetime.fromisoformat("2026-09-11T15:00:00+08:00")
    result = client.load(["2330", "2454"], session_date="2026-09-11", observed_at=observed)
    assert len(calls) == 3 and result["2330"]["restricted"] is True and result["2454"]["restricted"] is False
    assert client.load(["2330"], session_date="2026-09-10", observed_at=observed) == {}
    assert len(calls) == 3


def test_ixbrl_adapter_preserves_period_scope_units_and_availability():
    from domain.fundamental_derivation import FiledFact, FiledFinancials
    from domain.strategies.low_entry_v6_filings import ixbrl_record
    facts = (
        FiledFact("ifrs-full:Revenue", 5_000_000., None, "2026-01-01", "2026-06-30", None, unit="iso4217:TWD"),
        FiledFact("ifrs-full:GrossProfit", 2_000_000., None, "2026-01-01", "2026-06-30", None, unit="iso4217:TWD"),
        FiledFact("ifrs-full:BasicEarningsLossPerShare", 2.5, None, "2026-01-01", "2026-06-30", None, unit="iso4217:TWD/xbrli:shares"),
        FiledFact("ifrs-full:Assets", 10_000_000., None, None, None, "2026-06-30", unit="iso4217:TWD"),
        # A non-current comparative context must never override this period.
        FiledFact("ifrs-full:Assets", 999_000_000., None, None, None, "2025-06-30", unit="iso4217:TWD"),
    )
    filing = FiledFinancials(2026, 2, "2026-08-10T18:00:00+08:00", 2, facts, content_sha256="r2")
    row = ixbrl_record("2330", filing)
    assert row["revenue"] == 5000. and row["cost_of_revenue"] == 3000.
    assert row["basic_eps"] == 2.5 and row["total_assets"] == 10_000.
    assert row["net_income"] is None
    assert row["available_at"] == filing.available_at and row["revision_id"] == "r2"
    assert row["period_end"] == "2026-06-30" and row["flow_basis"] == "YTD"
    assert row["unit"] == "thousand" and row["scope"] == "consolidated"


def test_v6_projection_tampering_and_mixed_versions_are_rejected():
    builder, decisions = build()
    frame = builder.project_decisions(decisions)
    frame.at[0, "低點分數"] = 100
    with pytest.raises(StrategyContractError, match="projection mismatch"):
        StrategyDecisionAssembler().assemble(frame, daily_market())
    legacy = replace(decisions[0], fields=tuple(replace(field, value="5.0.0") if field.name == "LES Model Version" else field for field in decisions[0].fields))
    # Give the legacy record a different symbol so latest-by-symbol cannot hide it.
    legacy = replace(legacy, fields=tuple(replace(field, value="2454") if field.name == "證券代號" else field for field in legacy.fields))
    with pytest.raises(ValueError, match="Mixed LES versions"):
        RendererFactory.get_renderer("daily_strategy_email").render((*decisions, legacy))


def test_empty_daily_report_still_reports_v6_not_legacy_v5():
    rendered = RendererFactory.get_renderer("daily_strategy_email").render(())
    assert "Low-Entry Model v6" in rendered and "Low-Entry Model v5" not in rendered
    assert "本次沒有 BUY / BUY_SMALL" in rendered


def test_published_daily_csv_reaches_owner_pae_and_rejects_conflicting_artifact(tmp_path, monkeypatch, capsys):
    from aistock_twse_integration import owner_research
    builder, decisions = build()
    enriched = PortfolioAllocationV2Pipeline().enrich(decisions, PortfolioRuntimeContext(account_equity=1_000_000., current_cash=500_000.))
    CsvRepository(tmp_path).write_analysis_dataset("elec", builder.project_decisions(enriched))
    request = {"ownerId": "test:daily-v6", "tradeDate": "2026-09-14", "account": {"cash": 500_000., "positions": []}}
    monkeypatch.setattr(owner_research, "parse_args", lambda: SimpleNamespace(crawler_root=Path.cwd(), dataset=tmp_path / "elec_analysis_dataset.csv", history_db=tmp_path / "unused.db", market_type="13", industry="ELEC"))
    monkeypatch.setattr(owner_research, "load_request", lambda: request)
    monkeypatch.setattr(owner_research, "load_history_sessions", lambda *args: 500)
    monkeypatch.setattr("aistock_twse_integration.les_v6_execution.next_sessions", lambda *args: ["2026-09-11", "2026-09-14"])
    assert owner_research.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["models"]["les"] == "6.0.0"
    assert output["lines"][0]["lowEntry"]["v6Report"] == json.loads(decisions[0].value("LES v6 Report"))
    request["lesV6ReportsFile"] = str(tmp_path / "elec_les_v6_reports.json")
    assert owner_research.main() == 0
    capsys.readouterr()
    conflicting = load_reports(request["lesV6ReportsFile"], "2026-09-11")
    conflicting["2330"]["decision"]["score"] = 99
    monkeypatch.setattr("aistock_twse_integration.les_v6_execution.load_reports", lambda *args: conflicting)
    with pytest.raises(ValueError, match="V6_CSV_ARTIFACT_MISMATCH"):
        owner_research.main()
