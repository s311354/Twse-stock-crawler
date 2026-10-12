"""Synthetic retained reports only; no live account, network or research state."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from application.daily_report_audit import audit_daily_report
from infrastructure.daily_report_audit import retain_daily_report_audit
from interface.research import main


def fixture():
    reports = []
    for symbol, signal in (("3305", "BUY_SMALL"), ("6196", "WATCH"), ("3036A", "DATA_INSUFFICIENT")):
        reports.append(dict(symbol=symbol, name="SYNTHETIC TEST", signal_date="2026-09-18",
            model_version="test-model", parameter_version="test-parameters", parameters={"threshold": 65},
            decision_time="2026-09-29T01:00:00+00:00",
            decision=dict(signal=signal, score=65, risk_gate="PASS"),
            quality=dict(status="PASS", score=80, coverage={"profitability": 75}),
            sizing=dict(indicative_shares=371, portfolio_data_pass=True),
            data=dict(restriction_check=dict(status="CLEAR"), restriction_evidence=dict(
                scope="CURRENT_AS_OF", available_at="2026-09-29T08:00:00+08:00")),
            technical=dict(rsi=45.0),
            research=dict(verification=dict(missing_periods=[],
                unresolved_fields=["2025Q4:basic_eps:PERIOD_BASIS_REVIEW"],
                latest_available_at="2026-09-20T12:00:00+08:00"))))
    reports[-1]["signal_date"] = "2026-02-11"
    rows = []
    for report in reports:
        rows.append({"證券代號": report["symbol"], "證券名稱": report["name"],
            "日期": report["signal_date"], "低點決策": report["decision"]["signal"],
            "低點分數": "65.0", "LES v6 Score": "65", "LES v6 Decision": report["decision"]["signal"],
            "LES v6 Report": json.dumps(report), "Quality Status": "PASS", "Quality Score": "80.0",
            "LES Model Version": report["model_version"], "LES Parameter Version": report["parameter_version"],
            "Quality Gate": "PASS", "EREM 決策訊號": "HOLD", "PAE Decision": "NO_ACTION",
            "PAE Portfolio Input": "COLD_START_ALL_CASH", "PAE Binding": "NON_BINDING",
            "PAE Blocking Reasons": "VALUATION_FAIL;CER_BELOW_REQUIRED_RETURN"})
    return rows, dict(model="test-model", signal_date="2026-09-18", reports=reports[:2], stale_reports=reports[2:])


def audit(rows=None, document=None):
    original_rows, original_document = fixture()
    return audit_daily_report(rows if rows is not None else original_rows,
        document if document is not None else original_document, "0;1;2", stocklist_format="legacy-indices")


def test_full_universe_includes_stale_and_preserves_raw_vs_allocation():
    result = audit()
    assert result["summary"]["audited_symbols"] == 3
    assert result["summary"]["export_parity"] == {"MATCH": 3}
    assert result["entry_anchor_symbols"] == ["3305"]
    assert next(r for r in result["rows"] if r["symbol"] == "3036A")["report_bucket"] == "stale_reports"
    assert result["summary"]["original_pae_decisions"] == {"NO_ACTION": 3}
    assert result["stocklist"]["identity_status"] == "UNVERIFIED_LEGACY_INDEX_MAPPING"
    assert not {"0", "1", "2"} & {r["symbol"] for r in result["rows"]}


def test_summary_never_proves_replay_account_or_financial_completeness():
    rows, document = fixture()
    document["reports"][0]["research"]["verification"]["unresolved_fields"] = []
    result = audit(rows, document)
    assert result["dataset_id"] is None
    assert result["stages"]["same_input_replay"] == "BLOCKED_MISSING_ORIGINAL_INPUTS"
    assert "ACCOUNT_SNAPSHOT_NOT_RETAINED" in result["missing_original_inputs"]
    assert "PAE_PREDECESSOR_NOT_RETAINED" in result["missing_original_inputs"]
    assert all(r["research_entry_eligibility"] == "NOT_EVALUATED" and not r["is_fill"] for r in result["rows"])
    row = next(r for r in result["rows"] if r["symbol"] == "3305")
    assert row["financial_comparability_audit"] == "NOT_INDEPENDENTLY_VERIFIED"
    assert row["historical_restrictions_audit"] != "CLEAR"


def test_full_nested_parity_not_only_scores_or_signals():
    rows, document = fixture()
    document["reports"][0]["technical"]["rsi"] = 46.0
    rows[0]["Quality Score"] = "79"
    result = audit(rows, document)
    row = next(r for r in result["rows"] if r["symbol"] == "3305")
    assert row["export_parity"]["differences"] == ["LES v6 Report/technical/rsi", "Quality Score"]
    assert result["summary"]["export_parity"] == {"INCOMPLETE_OR_MISMATCH": 1, "MATCH": 2}


def test_missing_and_extra_symbols_remain_visible():
    rows, document = fixture()
    del rows[0]
    del document["reports"][1]
    result = audit_daily_report(rows, document, "3305;6196;3036A;2330", stocklist_format="symbols")
    assert result["summary"]["audited_symbols"] == 4
    assert result["stocklist"]["identity_status"] == "MISMATCH"
    assert result["summary"]["csv_only_symbols"] == ["6196"]
    assert result["summary"]["json_only_symbols"] == ["3305"]
    missing = next(r for r in result["rows"] if r["symbol"] == "2330")
    assert missing["issues"] == ["CSV_ROW_MISSING", "JSON_REPORT_MISSING"]


@pytest.mark.parametrize("duplicate", ["csv", "json", "cross_bucket"])
def test_duplicate_symbols_rejected(duplicate):
    rows, document = fixture()
    if duplicate == "csv": rows.append(deepcopy(rows[0]))
    elif duplicate == "json": document["reports"].append(deepcopy(document["reports"][0]))
    else: document["stale_reports"].append(deepcopy(document["reports"][0]))
    with pytest.raises(ValueError, match="DUPLICATE_SYMBOL"):
        audit(rows, document)


@pytest.mark.parametrize("stamp,after", [("2026-09-18T05:30:00+00:00", False),
    ("2026-09-18T05:30:00.000001+00:00", True), (None, False)])
def test_availability_cutoff_instant_and_timezone_are_explicit(stamp, after):
    rows, document = fixture()
    document["reports"][0]["research"]["verification"]["latest_available_at"] = stamp
    result = audit(rows, document)
    row = next(r for r in result["rows"] if r["symbol"] == "3305")
    assert ("REPORTED_FINANCIAL_AVAILABILITY_AFTER_SIGNAL_CLOSE" in row["issues"]) is after
    assert row["signal_to_decision_calendar_days"] == 11
    assert row["hypothetical_signal_close_cutoff"] == "2026-09-18T13:30:00+08:00"
    assert row["financial_availability_audit"] == "UNVERIFIABLE_WITHOUT_SELECTED_VERSIONS_AND_PROOFS"


def test_naive_timestamp_and_nonfinite_score_rejected():
    rows, document = fixture()
    document["reports"][0]["decision_time"] = "2026-09-29T13:30:00"
    with pytest.raises(ValueError, match="TIMEZONE"):
        audit(rows, document)
    rows[0]["Quality Score"] = "nan"
    with pytest.raises(ValueError, match="NON_FINITE"):
        audit(rows)


@pytest.mark.parametrize("stocklist,form", [("", "symbols"), ("0;00", "legacy-indices"),
    ("-1;0", "legacy-indices"), ("0;1;1", "legacy-indices"), ("0;1", "auto")])
def test_stocklist_format_and_duplicates_are_explicit(stocklist, form):
    rows, document = fixture()
    with pytest.raises(ValueError):
        audit_daily_report(rows, document, stocklist, stocklist_format=form)


def intake(tmp_path):
    rows, document = fixture()
    flat = io.StringIO(newline="")
    writer = csv.DictWriter(flat, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    paths = {"analysis_csv": tmp_path/"daily.csv", "reports": tmp_path/"reports.json", "stocklist": tmp_path/"stocklist"}
    paths["analysis_csv"].write_text(flat.getvalue())
    paths["reports"].write_text(json.dumps(document))
    paths["stocklist"].write_text("0;1;2")
    return paths


def test_publication_idempotent_sources_unchanged_and_cli_equal(tmp_path, capsys):
    paths = intake(tmp_path)
    before = {key: path.read_bytes() for key, path in paths.items()}
    output = tmp_path/"audit"
    folder = retain_daily_report_audit(**paths, output=output, stocklist_format="legacy-indices")
    artifacts = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert retain_daily_report_audit(**paths, output=output, stocklist_format="legacy-indices") == folder
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == artifacts
    assert {key: path.read_bytes() for key, path in paths.items()} == before
    result = json.loads((folder/"audit.json").read_text())
    for key, ref in result["source_objects"].items():
        body = (folder/ref["path"]).read_bytes()
        assert body == before[key]
        assert hashlib.sha256(body).hexdigest() == ref["sha256"]
    assert main(["daily-audit", "--analysis-csv", str(paths["analysis_csv"]), "--reports", str(paths["reports"]),
        "--stocklist", str(paths["stocklist"]), "--stocklist-format", "legacy-indices", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["baseline_id"] == folder.name
    assert (folder/"audit.json").read_bytes() == artifacts["audit.json"]


def test_resume_after_partial_publication_and_reject_tampered_object(tmp_path):
    paths = intake(tmp_path)
    output = tmp_path/"audit"
    folder = retain_daily_report_audit(**paths, output=output, stocklist_format="legacy-indices")
    receipt = (folder/"receipt.json").read_bytes()
    (folder/"receipt.json").unlink()
    retain_daily_report_audit(**paths, output=output, stocklist_format="legacy-indices")
    assert (folder/"receipt.json").read_bytes() == receipt
    ref = json.loads(receipt)["source_objects"]["reports"]["path"]
    (folder/ref).write_bytes(b"TAMPERED SYNTHETIC TEST")
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        retain_daily_report_audit(**paths, output=output, stocklist_format="legacy-indices")


def test_changed_source_uses_new_baseline_and_preserves_old(tmp_path):
    paths = intake(tmp_path)
    output = tmp_path/"audit"
    first = retain_daily_report_audit(**paths, output=output, stocklist_format="legacy-indices")
    old = (first/"audit.json").read_bytes()
    paths["stocklist"].write_text("0;1")
    second = retain_daily_report_audit(**paths, output=output, stocklist_format="legacy-indices")
    assert second != first
    assert not json.loads((second/"audit.json").read_text())["stocklist"]["report_count_matches"]
    assert (first/"audit.json").read_bytes() == old


def test_live_file_change_during_intake_rejects_before_publication(tmp_path, monkeypatch):
    paths = intake(tmp_path)
    original_read = Path.read_bytes
    reads = 0
    def changing_read(path):
        nonlocal reads
        content = original_read(path)
        if path == paths["reports"]:
            reads += 1
            if reads > 1:
                return content + b" "
        return content
    monkeypatch.setattr(Path, "read_bytes", changing_read)
    output = tmp_path/"audit"
    with pytest.raises(ValueError, match="SOURCE_CHANGED_DURING_AUDIT"):
        retain_daily_report_audit(**paths, output=output, stocklist_format="legacy-indices")
    assert not output.exists()
