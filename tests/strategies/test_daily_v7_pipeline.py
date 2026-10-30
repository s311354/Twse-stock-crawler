"""SYNTHETIC ONLY: daily v7 wiring, not deployment/calibration approval."""
from copy import deepcopy
from dataclasses import replace
from email import message_from_string
from email.header import decode_header, make_header
import json
from unittest.mock import Mock
from types import SimpleNamespace

import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.contracts import AnalysisContext, MarketStatus
from application.daily_stock import isolate_stock_market
from application.les_v7 import analyze_request
from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from application.strategy_contracts import StrategyContractError, StrategyDecisionAssembler
from config.settings import AppSettings, SettingsError
from domain.model_contracts import FieldSource, FieldUnit
from domain.models import PortfolioPosition
from infrastructure.daily_parallel import DailyProcessExecutor
from infrastructure.report.html_renderer import RendererFactory
from infrastructure.storage.csv_repository import CsvRepository
from infrastructure.storage.report_evidence import iter_report_bundle, resolve_csv_report
from tests.strategies.test_daily_v6_pipeline import daily_market
from tests.strategies.test_low_entry_score_v7 import bars, bundle, filings, receipt, seal, CUTOFF


def synthetic_market():
    raw = bars(90)
    scaled = lambda key: [(raw[key] * 1000).tolist()]
    symbol = "SYNTHETIC_ONLY"
    return replace(daily_market(), stock_numbers=[symbol], stock_names=["合成測試"],
        dates=[raw.Date.tolist()], opens=scaled("Open"), highs=scaled("High"),
        lows=scaled("Low"), closes=scaled("Close"), bids=scaled("Bid"), asks=scaled("Ask"),
        volumes=[raw.Volume.tolist()], trade_counts=[[100] * len(raw)],
        trade_values=[raw.Amount.tolist()], pe_ratios=[[20.] * len(raw)],
        taiex_closes=[raw["TAIEX Close"].tolist()], entry_model="v7",
        les_v7_bundle_json=json.dumps(bundle()), fallback_date="2026-10-12",
        signal_dates={symbol: CUTOFF}, trading_restrictions={},
        fundamental_snapshots={symbol: {"sector": "24", "les_v6_filings": filings()}})


def prepared_synthetic_market(market=None):
    market = market or synthetic_market()
    snapshot = deepcopy(market.fundamental_snapshots)
    gates = {}
    for name in ("SAFETY", "DATA_INTEGRITY", "TRADING_STATUS", "VALUATION", "STRESS_RISK"):
        snapshot["SYNTHETIC_ONLY"]["les_v7_context"] = {"receipts": gates}
        candidate = replace(market, fundamental_snapshots=snapshot)
        report = daily_report(MarketHistoryAssembler().build(candidate, 0))
        gates[name] = receipt(name, report["diagnostics"]["gate_bindings"][name])
    return replace(market, fundamental_snapshots=snapshot)


def executable_synthetic_market(signal="BUY"):
    # Explicit test-only artifact, not a production calibration/approval claim.
    market = daily_market()
    model = bundle()
    for context in model["payload"]["weights"]:
        model["payload"]["weights"][context] = [0., 1., 0., 0., 0., 0.]
    if signal == "BUY_SMALL":
        model["payload"]["heads"]["joint"] = [.2006706954621514, 0.]
    symbol = "SYNTHETIC_ONLY"
    dates = pd.bdate_range(end="2026-10-12", periods=len(market.dates[0])).strftime("%Y-%m-%d").tolist()
    lows = deepcopy(market.lows)
    # Confirm the support before cutoff (not the pivot first confirmed today).
    lows[0][-6:] = [price * 1000 for price in (94.5, 94.7, 94.4, 95.5, 95.7, 96.1)]
    market = replace(market, stock_numbers=[symbol], stock_names=["合成候選"], dates=[dates],
        lows=lows,
        entry_model="v7", les_v7_bundle_json=json.dumps(seal(model)),
        signal_dates={symbol: CUTOFF}, fallback_date="2026-10-12", trading_restrictions={},
        les_v6_portfolio=dict(value=1_000_000., available_cash=1_000_000., intended_weight=.25, capacity_weight=.25, positions=[]),
        fundamental_snapshots={symbol: {"sector": "24", "les_v6_filings": filings()}})
    return prepared_synthetic_market(market)


def build(market):
    builder = AnalysisDatasetBuilder()
    return builder, builder.build_decisions(market, [[1, 1, 1, 1, .01]])


def test_missing_bundle_uses_native_v7_null_not_v6_score(monkeypatch):
    from domain.strategies.low_entry_score_v6 import LowEntryScoreV6Strategy
    def forbidden(*args, **kwargs):
        raise AssertionError("Daily v7 must not calculate v6 scores")
    monkeypatch.setattr(LowEntryScoreV6Strategy, "run_latest", forbidden)
    _, decisions = build(replace(daily_market(), entry_model="v7"))
    d = decisions[0]
    report = json.loads(d.value("LES v7 Report"))
    assert d.value("LES Model Version") == "7.0.0"
    assert d.value("低點決策") == "DATA_INSUFFICIENT"
    assert d.value("低點分數") is None and d.value("LES v6 Report") is None
    assert "MODEL_ARTIFACTS_NOT_PROVIDED" in report["envelope"]["strategy"]["reason_codes"]
    assert d.field("低點決策").source == FieldSource.LES_V7
    assert d.field("EREM 分數").source == FieldSource.EREM_V3
    assert d.field("LES v7 Execution Eligible").unit == FieldUnit.BOOLEAN
    assert d.value("Position Fraction") == 0 and d.value("建議股數") == 0


def test_daily_adapter_matches_public_v7_kernel_and_native_price_units():
    market = prepared_synthetic_market()
    history = MarketHistoryAssembler().build(market, 0)
    raw = history.iloc[:, :12].copy()
    raw.attrs = {}
    for key in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        raw[key] /= 1000
    snapshot = market.fundamental_snapshots["SYNTHETIC_ONLY"]
    context = dict(symbol="SYNTHETIC_ONLY", market="TWSE", as_of=CUTOFF,
        filings=filings(), model_bundle=bundle(), quality_policy=bundle()["payload"]["quality_policy"],
        receipts=snapshot["les_v7_context"]["receipts"], external_evidence_ids=[])
    expected = analyze_request(dict(bars=raw.to_dict("records"), context=context))["reports"][0]
    _, decisions = build(market)
    report = json.loads(decisions[0].value("LES v7 Report"))
    assert report["envelope"] == expected["envelope"]
    assert report["diagnostics"]["features"] == expected["diagnostics"]["features"]
    assert report["envelope"]["quality"]["status"] == "PASS"
    assert report["envelope"]["localization"]["status"] == "VALID"
    assert report["envelope"]["score"]["final_score"] is not None
    assert decisions[0].value("策略停損") == pytest.approx(report["diagnostics"]["reference_plan"]["stop"] * 1000)
    assert decisions[0].value("收盤") == pytest.approx(raw.Close.iloc[-1] * 1000)
    assert decisions[0].value("LES v7 Execution Eligible") is False


def test_v7_serial_and_spawn_workers_preserve_model_and_evidence():
    market = prepared_synthetic_market()
    local = isolate_stock_market(market, 0)
    assert local.entry_model == "v7" and local.les_v7_bundle_json == market.les_v7_bundle_json
    profits = [[1, 1, 1, 1, .01]]
    serial = AnalysisDatasetBuilder().build_decisions(market, profits)
    parallel = AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)).build_decisions(market, profits)
    assert parallel == serial


def test_csv_json_graph_and_email_use_v7_not_relabelled_v6(tmp_path):
    builder, decisions = build(prepared_synthetic_market())
    report = json.loads(decisions[0].value("LES v7 Report"))
    CsvRepository(tmp_path).write_analysis_dataset("synthetic", builder.project_decisions(decisions))
    path = tmp_path / "synthetic_les_v7_reports.json"
    payload = json.loads(path.read_text())
    assert payload["model"] == "low_entry_score_v7" and payload["execution_enabled"] is False
    assert "diagnostics" not in payload["reports"][0]
    assert list(iter_report_bundle(path, require_reference=True)) == [report]
    data = pd.read_csv(tmp_path / "synthetic_analysis_dataset.csv", dtype={"證券代號": str})
    assert resolve_csv_report(data.iloc[0]["LES v7 Report"], tmp_path) == report
    assert not (tmp_path / "synthetic_les_v6_reports.json").exists()
    html = RendererFactory.get_renderer("daily_strategy_email").render(decisions)
    assert "Low-Entry Model v7" in html and "Low-Entry Model v6" not in html
    assert 'data-les-version="7.0.0"' in html and "EREM v3" in html and "候選授權" in html
    assert f'現價 NTD {decisions[0].value("收盤"):,.2f}' in html
    assert "股價 ×1000（台幣尺度）" not in html

    # The CSV is a convenient latest view; the immutable candidate snapshot
    # retains every per-stock outcome and joins EREM to the full native report.
    from infrastructure.storage.report_evidence import ReportEvidenceReader
    snapshots = list((tmp_path / "synthetic_les_v7_candidates").glob("*.json"))
    assert len(snapshots) == 1
    snapshot = json.loads(snapshots[0].read_text())
    candidate = snapshot["candidates"][0]
    assert snapshot["schema"] == "les-v7-daily-candidates/v1"
    assert snapshot["snapshot_id"] == snapshots[0].stem
    assert snapshot["candidate_count"] == 1
    assert candidate["decision"] == report["envelope"]["strategy"]["strategy_decision"]
    assert candidate["quality"] == report["diagnostics"]["quality"]
    assert candidate["gate_receipts"] == report["diagnostics"]["gate_receipts"]
    assert candidate["six_factor_ledger"] == report["diagnostics"]["factor_ledger"]
    assert candidate["erem"]["EREM 分數"] == decisions[0].value("EREM 分數")
    assert candidate["erem"]["EREM 理由"] == decisions[0].value("EREM 理由")
    assert candidate["candidate_state"] == (
        "BLOCKED_DATA_INSUFFICIENT" if candidate["decision"] == "DATA_INSUFFICIENT" else "NOT_ADMITTED")
    assert candidate["fill_observation"]["status"] == "NOT_CAPTURED_BY_CRAWLER"
    with ReportEvidenceReader(tmp_path) as reader:
        assert reader.resolve(candidate["report_reference"]) == report
    immutable_bytes = snapshots[0].read_bytes()
    CsvRepository(tmp_path).write_analysis_dataset("synthetic", builder.project_decisions(decisions))
    assert snapshots[0].read_bytes() == immutable_bytes
    assert len(list((tmp_path / "synthetic_les_v7_candidates").glob("*.json"))) == 1


@pytest.mark.parametrize("scaled_close, displayed", [(9300., "9,300.00"), (1215000., "1,215,000.00"), (None, "N/A")])
def test_v7_current_price_display_keeps_thousand_scale_without_mutating_native_plan(scaled_close, displayed):
    # Synthetic renderer-only projection cases; not market prices or fills.
    _, decisions = build(executable_synthetic_market())
    original = decisions[0]
    item = replace(original, fields=tuple(replace(f, value=scaled_close) if f.name == "收盤" else f
        for f in original.fields))
    report_before = item.value("LES v7 Report")
    report = json.loads(report_before)
    html = RendererFactory.get_renderer("daily_strategy_email").render([item])
    assert f'現價 NTD {displayed}' in html
    assert "股價 ×1000（台幣尺度）" not in html
    plan = report["diagnostics"]["reference_plan"]
    assert f'進場帶 NT${plan["band"][0]:,.2f}' in html
    assert item.value("LES v7 Report") == report_before
    assert original in decisions


def test_v7_email_subject_and_body_agree_without_sending(monkeypatch):
    from textmewhenitsdone import TextMeWhenItsDone
    from tests.test_stockanalysis import CapturingTwseCrawler
    _, decisions = build(replace(daily_market(), entry_model="v7"))
    transport = object.__new__(TextMeWhenItsDone)
    transport.server = Mock()
    class Mail:
        def __init__(self, *args, **kwargs): pass
        def smtpauthentication(self): pass
        def textstockprofittable(self, **kwargs):
            transport.textstockprofittableme(subject="test", email="test@example.invalid", password="unused",
                receiver="test@example.invalid", **kwargs)
    monkeypatch.setattr("infrastructure.legacy_crawler_facade.SMTPEmail", Mail)
    crawler = CapturingTwseCrawler(1)
    crawler.iso_scheduled_times = ["2026-09-11"]
    crawler.smtp_strategy_email("test", None, "ELEC", decisions)
    message = message_from_string(transport.server.sendmail.call_args.kwargs["msg"])
    assert "LES v7 / EREM v3" in str(make_header(decode_header(message["Subject"])))
    assert "Low-Entry Model v7" in message.get_payload()[0].get_payload(decode=True).decode("utf-8")


def test_missing_receipts_never_borrow_old_v6_clear_flags():
    market = synthetic_market()
    _, decisions = build(market)
    report = json.loads(decisions[0].value("LES v7 Report"))
    assert {g["state"] for g in report["diagnostics"]["gate_receipts"]} == {"UNKNOWN"}
    assert report["envelope"]["score"]["final_score"] is None
    assert decisions[0].value("Position Fraction") == 0


def test_pae_consumes_native_v7_quality_without_authorizing_allocation():
    _, decisions = build(prepared_synthetic_market())
    context = PortfolioRuntimeContext(account_equity=1_000_000., current_cash=1_000_000., default_sector="24")
    pipeline = PortfolioAllocationV2Pipeline()
    request, _, _ = pipeline._build_request(decisions, context, decisions[0].evaluated_at)
    candidate = request.candidates[0]
    assert candidate.quality_score == decisions[0].value("Quality Score")
    assert candidate.data_confidence == decisions[0].value("Data Confidence")
    assert candidate.executable_shares == 0 and candidate.les_position_cap is None
    assert pipeline.enrich(decisions, context)[0].value("PAE Model Version") == "2.0.0"


@pytest.mark.parametrize("signal", ["BUY", "BUY_SMALL"])
def test_qualified_v7_signal_authorizes_pae_candidate_not_order(signal, tmp_path):
    market = executable_synthetic_market(signal)
    builder, decisions = build(market)
    decision = decisions[0]
    report = json.loads(decision.value("LES v7 Report"))
    authorization = report["pae_authorization"]
    assert report["envelope"]["strategy"]["strategy_decision"] == signal
    assert authorization["authorized"] is True and decision.value("LES v7 Execution Eligible") is True
    assert decision.value("建議股數") == authorization["indicative_shares"] > 0
    assert decision.value("Position Fraction") == authorization["position_fraction"] > 0
    assert authorization["indicative_shares"] == (authorization["full_shares"] // 4 if signal == "BUY_SMALL" else authorization["full_shares"])
    assert authorization["live_order_authorized"] is False
    assert report["envelope"]["execution"]["allocated_shares"] == 0
    assert report["envelope"]["execution"]["live_enabled"] is False
    pipeline = PortfolioAllocationV2Pipeline()
    context = PortfolioRuntimeContext(account_equity=1_000_000., current_cash=1_000_000., default_sector="24")
    request, _, _ = pipeline._build_request(decisions, context, decision.evaluated_at)
    candidate = request.candidates[0]
    assert candidate.executable_shares == authorization["indicative_shares"]
    assert candidate.les_position_cap == authorization["position_fraction"]
    assert candidate.risk_gate == candidate.liquidity_status == "PASS"
    # Missing independent valuation/CER is not forged by candidate authorization.
    enriched = pipeline.enrich(decisions, context)[0]
    assert enriched.value("PAE Decision") not in {"ADD", "STRONG_ADD", "ROTATE_25", "ROTATE_50", "ROTATE_75"}
    # A separately supplied, complete synthetic valuation unlocks the unchanged
    # native PAE allocation rules; no production valuation is fabricated.
    complete = replace(candidate, valuation_status="COMPLETE", valuation_gate="PASS",
        bear_cagr=.08, base_cagr=.22, bull_cagr=.35, valuation_confidence=.95)
    allocation = pipeline.engine.allocate(replace(request, candidates=(complete,)))
    line = allocation.lines[0]
    assert line.action.value == "STARTER" and line.target_weight > 0
    assert line.execution_status == "READY"
    assert line.target_weight <= authorization["position_fraction"]
    html = RendererFactory.get_renderer("daily_strategy_email").render(decisions)
    assert "PAE 候選已授權" in html and "尚未成交" in html
    CsvRepository(tmp_path).write_analysis_dataset("synthetic", builder.project_decisions(decisions))
    assert list(iter_report_bundle(tmp_path / "synthetic_les_v7_reports.json")) == [report]
    assert json.loads((tmp_path / "synthetic_les_v7_reports.json").read_text())["pae_candidate_authorization_enabled"] is True


def test_candidate_quota_respects_cash_risk_and_adv_without_changing_signal():
    from domain.les_v7.allocation import authorize_candidate
    _, decisions = build(executable_synthetic_market())
    report = json.loads(decisions[0].value("LES v7 Report"))
    original = deepcopy(report["envelope"])
    account = report["pae_authorization"]["account"]
    full = authorize_candidate(report, account)
    assert full["indicative_shares"] * full["bound_plan_loss_per_share"] <= account["value"] * .005
    assert full["indicative_shares"] * full["sizing_entry_limit"] <= report["diagnostics"]["features"]["adv"] * .01
    limited = authorize_candidate(report, dict(account, available_cash=1000.))
    assert 0 < limited["indicative_shares"] < full["indicative_shares"]
    assert limited["indicative_shares"] * (limited["sizing_entry_limit"] + report["diagnostics"]["reference_plan"]["cost_up"]) <= 1000.
    assert limited["authorization_id"] != full["authorization_id"]
    assert report["envelope"] == original


@pytest.mark.parametrize("values", [dict(available_cash=0.), dict(capacity_weight=0.), dict(available_cash=None),
    dict(positions=None), dict(positions=[dict(symbol="2330", shares=None)])])
def test_buy_without_account_capacity_remains_raw_buy_with_zero_quota(values):
    from domain.les_v7.allocation import authorize_candidate
    _, decisions = build(executable_synthetic_market())
    report = json.loads(decisions[0].value("LES v7 Report"))
    result = authorize_candidate(report, dict(report["pae_authorization"]["account"], **values))
    assert result["authorized"] is False and result["indicative_shares"] == 0
    assert report["envelope"]["strategy"]["strategy_decision"] == "BUY"


@pytest.mark.parametrize("values", [dict(value=float("nan")), dict(available_cash=-1.), dict(value=True),
    dict(available_cash=2_000_000.), dict(capacity_weight=2.), dict(as_of="2026-10-13T13:30:00+08:00"),
    dict(positions=[dict(symbol="2330", shares=1.5)]), dict(positions=[dict(symbol="2330", shares=-1)]),
    dict(positions=[dict(symbol="2330", shares=1)] * 2)])
def test_invalid_account_cannot_authorize_v7(values):
    from domain.les_v7.allocation import authorize_candidate
    _, decisions = build(executable_synthetic_market())
    report = json.loads(decisions[0].value("LES v7 Report"))
    with pytest.raises(ValueError):
        authorize_candidate(report, dict(report["pae_authorization"]["account"], **values))


@pytest.mark.parametrize("field", ["indicative_shares", "position_fraction", "account_sha256", "signal_sha256"])
def test_pae_authorization_tampering_is_rejected(field):
    from domain.les_v7.report import validate_report
    _, decisions = build(executable_synthetic_market())
    report = json.loads(decisions[0].value("LES v7 Report"))
    report["pae_authorization"][field] = "forged"
    with pytest.raises(ValueError, match="AUTHORIZATION_BINDING"):
        validate_report(report)


def test_pae_rejects_authorized_quota_when_account_has_changed():
    _, decisions = build(executable_synthetic_market())
    context = PortfolioRuntimeContext(account_equity=1_000_000., current_cash=990_000., default_sector="24")
    result = PortfolioAllocationV2Pipeline().enrich(decisions, context)[0]
    assert result.value("PAE Status") == "DATA_INSUFFICIENT"
    assert "V7_PAE_ACCOUNT_BINDING_MISMATCH" in result.value("PAE Blocking Reasons")


def test_same_cash_and_equity_with_changed_holdings_reject_old_quota():
    _, decisions = build(executable_synthetic_market())
    context = PortfolioRuntimeContext(account_equity=1_000_000., current_cash=1_000_000., default_sector="24",
        positions=(PortfolioPosition("SYNTHETIC_ONLY", 10),))
    result = PortfolioAllocationV2Pipeline().enrich(decisions, context)[0]
    assert "V7_PAE_ACCOUNT_BINDING_MISMATCH" in result.value("PAE Blocking Reasons")


def test_high_score_cannot_authorize_native_unknown_or_failed_gates():
    from domain.les_v7.allocation import authorize_candidate
    _, decisions = build(executable_synthetic_market())
    report = json.loads(decisions[0].value("LES v7 Report"))
    report["diagnostics"]["gate_receipts"][0]["state"] = "UNKNOWN"
    with pytest.raises(ValueError, match="FAILED_OR_UNKNOWN"):
        authorize_candidate(report, report["pae_authorization"]["account"])


def test_price_and_cutoff_cannot_be_changed_after_v7_authorization():
    market = executable_synthetic_market()
    builder, decisions = build(market)
    data = builder.project_decisions(decisions)
    data.loc[0, "收盤"] += 1000
    with pytest.raises(StrategyContractError, match="price mismatch"):
        StrategyDecisionAssembler().assemble(data, market)


def test_missing_historical_account_binding_cannot_inherit_daily_authorization():
    market = replace(executable_synthetic_market(), replay_as_of=CUTOFF)
    _, decisions = build(market)
    report = json.loads(decisions[0].value("LES v7 Report"))
    assert report["envelope"]["strategy"]["strategy_decision"] == "BUY"
    assert report["pae_authorization"]["authorized"] is False
    assert "ACCOUNT_INPUTS_MISSING" in report["pae_authorization"]["reason_codes"]


@pytest.mark.parametrize("version", ["v6", "v7"])
def test_runtime_captures_explicit_holdings_only_for_v7(version):
    from infrastructure.runtime.strategy_adapter import CrawlerStrategyAdapter
    crawler = SimpleNamespace(app_settings=SimpleNamespace(daily_entry_model=version),
        stocknumbers=["2330"], daily_closes=[[100_000.]], cal_max_profit=lambda: [],
        build_strategy_decisions=lambda profits: ())
    context = AnalysisContext(resource=crawler, market_date=pd.Timestamp("2026-10-12").date(),
                              market_status=MarketStatus.OPEN)
    CrawlerStrategyAdapter(portfolio_positions=(PortfolioPosition("2330", 1000),),
        portfolio_current_cash=900_000., account_equity=1_000_000.).evaluate(context)
    assert crawler.les_v6_portfolio["available_cash"] == 900_000.
    if version == "v7":
        assert crawler.les_v6_portfolio["positions"] == [dict(symbol="2330", shares=1000)]
    else:
        assert "positions" not in crawler.les_v6_portfolio


def test_tpex_daily_history_binds_the_actual_market():
    market = replace(daily_market(), entry_model="v7", market_source="TPEX", fundamental_snapshots={})
    report = daily_report(MarketHistoryAssembler().build(market, 0))
    assert report["diagnostics"]["gate_bindings"]["TRADING_STATUS"]["market"] == "TPEX"
    assert report["envelope"]["score"]["final_score"] is None


@pytest.mark.parametrize("change,error", [("stale", "STALE_MARKET_DATA"), ("time", "INVALID_FINANCIAL_AVAILABILITY")])
def test_daily_invalid_or_stale_inputs_cannot_use_valid_receipts(change, error):
    market = prepared_synthetic_market()
    if change == "stale":
        market = replace(market, fallback_date="2026-10-13")
    else:
        snapshots = deepcopy(market.fundamental_snapshots)
        snapshots["SYNTHETIC_ONLY"]["les_v6_filings"][-1]["available_at"] = "2026-01-01"
        market = replace(market, fundamental_snapshots=snapshots)
    report = daily_report(MarketHistoryAssembler().build(market, 0))
    assert error in report["diagnostics"]["daily_input_errors"]
    assert report["envelope"]["score"]["final_score"] is None


def test_daily_projection_tampering_is_rejected():
    builder, decisions = build(prepared_synthetic_market())
    data = builder.project_decisions(decisions)
    data.loc[0, "LES v7 Execution Eligible"] = True
    with pytest.raises(StrategyContractError, match="v7"):
        StrategyDecisionAssembler().assemble(data, prepared_synthetic_market())


def test_v7_bundle_captured_once_and_legacy_mode_stays_explicit(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle()))
    env = {"TWSE_DAILY_ENTRY_MODEL": "v7", "TWSE_LES_V7_BUNDLE": str(path)}
    settings = AppSettings.from_env(env, validate_paths=False)
    path.write_text("not json")
    assert json.loads(settings.les_v7_bundle_json)["sha256"] == bundle()["sha256"]
    with pytest.raises(SettingsError, match="TWSE_LES_V7_BUNDLE"):
        AppSettings.from_env(env, validate_paths=False)
    assert AppSettings.from_env({}, validate_paths=False).daily_entry_model == "v6"
    assert AppSettings.from_env({"TWSE_DAILY_ENTRY_MODEL": "v7"}, validate_paths=False).les_v7_bundle_json is None
    with pytest.raises(SettingsError, match="v6 or v7"):
        AppSettings.from_env({"TWSE_DAILY_ENTRY_MODEL": "v5"}, validate_paths=False)
