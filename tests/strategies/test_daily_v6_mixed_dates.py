"""A missing latest quote must block that symbol, not the whole daily report."""
from copy import deepcopy
from dataclasses import replace
import json

import pandas as pd
import pytest

from aistock_twse_integration.les_v6_execution import execute_request, load_reports
from application.analysis_dataset import AnalysisDatasetBuilder
from application.low_entry_v6_daily import project_report
from infrastructure.report.html_renderer import RendererFactory
from infrastructure.storage.csv_repository import CsvRepository
from tests.strategies.test_daily_v6_pipeline import daily_market


def build(market):
    builder = AnalysisDatasetBuilder()
    return builder, builder.build_decisions(market, [[1, 1, 1, 1, .01] for _ in range(market.stock_count)])


def mixed_market():
    market = daily_market()
    fields = ("dates", "opens", "highs", "lows", "closes", "volumes", "trade_counts",
              "trade_values", "bids", "asks", "pe_ratios", "taiex_closes")
    histories = {name: [deepcopy(getattr(market, name)[0]), deepcopy(getattr(market, name)[0][:-2])]
                 for name in fields}
    snapshots = deepcopy(market.fundamental_snapshots)
    snapshots["2454"] = deepcopy(snapshots["2330"])
    for filing in snapshots["2454"]["les_v6_filings"]:
        filing["symbol"] = "2454"
    return replace(market, stock_count=2, stock_numbers=["2330", "2454"],
                   stock_names=["當日股票", "缺少最新報價"], fallback_date="2026-09-11",
                   fundamental_snapshots=snapshots,
                   signal_dates=dict(market.signal_dates, **{"2454": market.signal_dates["2330"]}),
                   **histories)


@pytest.mark.usefixtures("active_industry_profiles")
def test_mixed_bar_dates_publish_current_reports_without_relabelling_or_dropping_diagnostics(tmp_path, caplog):
    builder, decisions = build(mixed_market())
    current, stale = decisions
    assert current.value("低點決策") == "BUY_SMALL"
    assert stale.value("低點決策") == "WATCH"
    assert stale.value("LES v6 Final Score") is None
    assert stale.value("低點理由") == "STALE_MARKET_DATA"
    assert stale.value("建議股數") == 0 and stale.value("Position Fraction") == 0
    assert stale.value("Risk Gate") != "PASS"
    frame = builder.project_decisions(decisions)
    CsvRepository(tmp_path).write_analysis_dataset("elec", frame)
    csv = pd.read_csv(tmp_path / "elec_analysis_dataset.csv", dtype={"證券代號": str})
    assert csv["日期"].tolist() == ["2026-09-11", "2026-09-09"]
    payload = json.loads((tmp_path / "elec_les_v6_reports.json").read_text())
    assert payload["signal_date"] == "2026-09-11"
    assert [r["symbol"] for r in payload["reports"]] == ["2330"]
    assert [r["symbol"] for r in payload["stale_reports"]] == ["2454"]
    stale_report = payload["stale_reports"][0]
    assert stale_report["signal_date"] == stale_report["data"]["market_data_date"] == "2026-09-09"
    assert stale_report["data"]["analysis_session_date"] == "2026-09-11"
    assert execute_request({"report": stale_report})["action"] == "SKIP"
    assert set(load_reports(str(tmp_path / "elec_les_v6_reports.json"), "2026-09-11")) == {"2330"}
    assert "count=1 symbols=2454" in caplog.text
    rendered = RendererFactory.get_renderer("daily_strategy_email").render(decisions)
    assert "2454 缺少最新報價" in rendered and "資料日 2026-09-09" in rendered
    assert "STALE_MARKET_DATA" in rendered and "共 1 檔" in rendered


def test_all_stale_preserves_scheduled_session_and_publishes_no_executable_report(tmp_path):
    market = mixed_market()
    fields = ("dates", "opens", "highs", "lows", "closes", "volumes", "trade_counts",
              "trade_values", "bids", "asks", "pe_ratios", "taiex_closes")
    market = replace(market, **{name: [getattr(market, name)[0][:-2], getattr(market, name)[1]] for name in fields})
    builder, decisions = build(market)
    assert all(d.value("低點決策") == "WATCH" and d.value("LES v6 Final Score") is None for d in decisions)
    CsvRepository(tmp_path).write_analysis_dataset("elec", builder.project_decisions(decisions))
    payload = json.loads((tmp_path / "elec_les_v6_reports.json").read_text())
    assert payload["signal_date"] == "2026-09-11" and payload["reports"] == []
    assert len(payload["stale_reports"]) == 2
    with pytest.raises(ValueError, match="INVALID_V6_REPORTS"):
        load_reports(str(tmp_path / "elec_les_v6_reports.json"), "2026-09-11")


def test_unsupported_current_security_is_kept_for_diagnostics_but_excluded_from_execution(tmp_path):
    builder, decisions = build(mixed_market())
    frame = builder.project_decisions(decisions[:1])
    report = json.loads(frame.at[0, "LES v6 Report"])
    report["symbol"] = "2330A"
    report["decision"]["signal"] = "DATA_INSUFFICIENT"
    report["sizing"]["indicative_shares"] = 0
    frame.at[0, "證券代號"] = "2330A"
    for name, value in project_report(report).items():
        frame.at[0, name] = value

    CsvRepository(tmp_path).write_analysis_dataset("elec", frame)
    payload = json.loads((tmp_path / "elec_les_v6_reports.json").read_text())
    assert payload["reports"] == []
    assert [item["symbol"] for item in payload["stale_reports"]] == ["2330A"]


def test_loader_skips_an_unsupported_non_executable_report(tmp_path):
    builder, decisions = build(mixed_market())
    frame = builder.project_decisions(decisions[:1])
    report = json.loads(frame.at[0, "LES v6 Report"])
    report["symbol"] = "2330A"
    report["decision"]["signal"] = "DATA_INSUFFICIENT"
    report["sizing"]["indicative_shares"] = 0
    artifact = {
        "model": "low_entry_score_v6",
        "signal_date": report["signal_date"],
        "reports": [report],
        "stale_reports": [],
    }
    path = tmp_path / "reports.json"
    path.write_text(json.dumps(artifact))
    assert load_reports(str(path), report["signal_date"]) == {}

    report["decision"]["signal"] = "BUY_SMALL"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="V6_REPORT_IDENTITY_OR_DATE_MISMATCH"):
        load_reports(str(path), report["signal_date"])


@pytest.mark.parametrize("mutation,reason", [
    ("stale_buy", "STALE_EXECUTABLE_V6_REPORT"),
    ("stale_shares", "STALE_EXECUTABLE_V6_REPORT"),
    ("different_sessions", "MIXED_DAILY_ANALYSIS_SESSIONS"),
    ("different_policy", "MIXED_DAILY_V6_SCORE_POLICIES"),
    ("identity", "V6_REPORT_IDENTITY_OR_DATE_MISMATCH"),
    ("future_date", "FUTURE_DAILY_SIGNAL_DATE"),
])
def test_writer_still_rejects_unsafe_reports_before_overwriting_files(tmp_path, mutation, reason):
    builder, decisions = build(mixed_market())
    original = builder.project_decisions(decisions)
    repository = CsvRepository(tmp_path)
    repository.write_analysis_dataset("elec", original)
    paths = [tmp_path / "elec_analysis_dataset.csv", tmp_path / "elec_les_v6_reports.json"]
    before = [path.read_bytes() for path in paths]
    frame = original.copy(deep=True)
    report = json.loads(frame.at[1, "LES v6 Report"])
    if mutation == "stale_buy":
        report["decision"]["signal"] = "BUY"
    elif mutation == "stale_shares":
        report["sizing"]["indicative_shares"] = 10
    elif mutation == "different_sessions":
        report["data"]["analysis_session_date"] = "2026-09-10"
    elif mutation == "different_policy":
        report["parameters"]["decision_policy_version"] = 2
    elif mutation == "identity":
        report["symbol"] = "9999"
    else:
        report["signal_date"] = "2026-09-14"
        frame.at[1, "日期"] = "2026-09-14"
    for name, value in project_report(report).items():
        frame.at[1, name] = value
    with pytest.raises(ValueError, match=reason):
        repository.write_analysis_dataset("elec", frame)
    assert [path.read_bytes() for path in paths] == before
