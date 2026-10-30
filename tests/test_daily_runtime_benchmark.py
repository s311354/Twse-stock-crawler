"""Synthetic storage fixtures; test counters/parity, not financial readiness."""
import json

import pytest

from application.replay_contracts import digest
from examples.benchmark_daily_runtime import benchmark_reports
from infrastructure.storage.report_evidence import ReportEvidenceWriter


def source(tmp_path):
    reports = [dict(symbol=symbol, signal_date="2026-10-19", model_version="7.0.0",
        parameter_version="SYNTHETIC_TEST_ONLY", diagnostics=dict(native_evidence=dict(
            source_packet=dict(scope="SYNTHETIC_TEST_ONLY", financials=[], restrictions="UNKNOWN"))),
        envelope=dict(strategy=dict(strategy_decision="DATA_INSUFFICIENT")),
        pae_authorization=dict(authorized=False, account_sha256=digest(dict(synthetic_only=True)),
                               account=dict(value=1000000, available_cash=1000000, positions=[])))
        for symbol in ("TEST_A", "TEST_B")]
    for report in reports:
        report["diagnostics"].update(gate_bindings=dict(DATA_INTEGRITY=dict(market="TWSE")),
            quality=dict(status="UNKNOWN"), gate_receipts=[], factor_ledger={}, reference_plan=None)
        report["envelope"].update(event=dict(cutoff_at="2026-10-19T13:30:00+08:00"), score={}, quality={})
    with ReportEvidenceWriter(tmp_path / "report_evidence.sqlite3") as writer:
        refs = [writer.retain(report) for report in reports]
    path = tmp_path / "reports.json"
    path.write_text(json.dumps(dict(storage_schema="daily-reports/v2", reports=refs)))
    return path, reports


def test_small_benchmark_compares_entire_reports_and_rechecks_in_new_work(tmp_path, monkeypatch):
    path, reports = source(tmp_path)
    validated = []
    def validate(value):
        assert value["parameter_version"] == "SYNTHETIC_TEST_ONLY"
        validated.append(digest(value))
    monkeypatch.setattr("domain.les_v7.report.validate_report", validate)
    result = benchmark_reports(path)
    assert result["full_native_results_identical"] and result["source_bundle_and_evidence_store_unchanged"]
    expected = [digest(report) for report in reports]
    for mode in ("reference", "optimized", "restarted"):
        assert all(row["full_report_sha256"] == expected for row in result[mode]["passes"])
    counters = lambda mode: result[mode]["metrics"]["counters"]
    assert counters("reference")["v7_report_full_validations"] == 6
    assert counters("optimized")["v7_report_full_validations"] == 2
    assert counters("optimized")["v7_report_validation_cache_hits"] == 4
    assert counters("restarted")["v7_report_full_validations"] == 2
    assert len(validated) == 10
    assert [row["source_packet_sha256"] for row in result["identities"]] == [
        digest(report["diagnostics"]["native_evidence"]["source_packet"]) for report in reports]
    assert [row["account_authorization_sha256"] for row in result["identities"]] == [
        digest(report["pae_authorization"]) for report in reports]


@pytest.mark.parametrize("stocks,passes", [(0, 3), (4, 3), (True, 3), (2, 1), (2, 6)])
def test_scope_is_bounded_before_reading_any_source(tmp_path, stocks, passes):
    with pytest.raises(ValueError, match="SMALL_BENCHMARK_ONLY"):
        benchmark_reports(tmp_path / "not-read.json", stocks=stocks, passes=passes)


def test_source_change_is_not_published_as_complete(tmp_path, monkeypatch):
    from examples import benchmark_daily_runtime as benchmark
    path, _ = source(tmp_path)
    monkeypatch.setattr("domain.les_v7.report.validate_report", lambda value: None)
    original = benchmark._measure_validation
    def change_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_text(path.read_text() + " ")
        return result
    monkeypatch.setattr(benchmark, "_measure_validation", change_after_read)
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        benchmark_reports(path)


def test_missing_references_never_become_empty_success(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(dict(storage_schema="daily-reports/v2", reports=[])))
    with pytest.raises(ValueError, match="REFERENCED_REPORTS_REQUIRED"):
        benchmark_reports(path)


def test_csv_reuses_native_token_but_rechecks_external_source_and_projection(tmp_path, monkeypatch):
    import pandas as pd
    from application.daily_work import DailyWorkContext
    from application.v7_report_input import read_v7_report_json
    from infrastructure.storage.csv_repository import CsvRepository
    from infrastructure.storage import valuation_source_verification as external
    _, reports = source(tmp_path)
    report = reports[0]
    encoded = json.dumps(report)
    validated, checked = [], []
    monkeypatch.setattr("domain.les_v7.report.validate_report", lambda value: validated.append(digest(value)))
    monkeypatch.setattr(external, "verify_report_valuation_sources", lambda value, **kw: checked.append(digest(value)))
    dataset = pd.DataFrame([dict({"LES Model Version": "7.0.0", "LES v7 Report": encoded,
        "證券代號": report["symbol"], "日期": report["signal_date"], "低點決策": "DATA_INSUFFICIENT",
        "EREM 分數": None, "EREM 決策訊號": "EXIT", "EREM 理由": "SYNTHETIC_ONLY"})])
    with DailyWorkContext("SYNTHETIC_ONLY").activate():
        read_v7_report_json(encoded)
        repository = CsvRepository(tmp_path)
        repository.write_analysis_dataset("first", dataset)
        repository.write_analysis_dataset("second", dataset)
        damaged = dataset.copy()
        damaged["證券代號"] = "OTHER"
        with pytest.raises(ValueError, match="V7_REPORT_IDENTITY_DATE_OR_DECISION_MISMATCH"):
            repository.write_analysis_dataset("damaged", damaged)
    assert validated == [digest(report)]
    assert checked == [digest(report)] * 3
    assert not (tmp_path / "damaged_les_v7_reports.json").exists()
