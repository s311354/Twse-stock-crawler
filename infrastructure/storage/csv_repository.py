"""CSV storage repository for TWSE daily and analysis outputs."""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR
from domain.models import StockRows
from domain.tpex_market import TPEX_FIELDS, TPEX_RAW_OFFSET
from infrastructure.storage.stock_repository import CsvStockRepository
from infrastructure.storage.leadership_watch_repository import LeadershipWatchRepository
from infrastructure.storage.report_evidence import ReportEvidenceWriter, STORE_NAME, compact_report
from application.daily_performance import count_work, measure_stage
from application.replay_contracts import digest


def _is_supported_execution_symbol(value: object) -> bool:
    return isinstance(value, str) and value.isdigit() and 4 <= len(value) <= 6


class CsvRepository(CsvStockRepository):
    """Owns CSV filesystem writes so application code stays storage-agnostic."""

    def __init__(self, output_dir: str | Path = DATA_DIR) -> None:
        super().__init__(output_dir)

    def write_daily_rows(self, file_name: str, scheduled_time: str, rows: StockRows) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{file_name}_{scheduled_time}.csv"

        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows([row for row in rows if row])

        return output_path

    def write_analysis_dataset(self, file_name: str, dataset: pd.DataFrame) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{file_name}_analysis_dataset.csv"
        payload = None
        exported = dataset
        if "LES Model Version" in dataset and "6.0.0" in set(dataset["LES Model Version"]):
            if set(dataset["LES Model Version"]) != {"6.0.0"}:
                raise ValueError("MIXED_DAILY_LES_VERSIONS")
            reports = []
            seen = {}
            references, summaries = {}, {}
            # Validate and retain each decoded report once. A later batch error
            # rolls back all nodes, including already flushed SQLite batches.
            with measure_stage("report_native_publish"), ReportEvidenceWriter(self.output_dir / STORE_NAME) as store:
                for _, row in dataset.iterrows():
                    with measure_stage("report_native_parse"):
                        report = json.loads(row["LES v6 Report"])
                        count_work("report_native_parses")
                    code = str(row["證券代號"])
                    if report.get("model_version") != "6.0.0" or report.get("symbol") != code or report.get("signal_date") != str(row["日期"])[:10]:
                        raise ValueError("V6_REPORT_IDENTITY_OR_DATE_MISMATCH")
                    if code in seen:
                        if digest(report) != seen[code]:
                            raise ValueError("CONFLICTING_V6_REPORTS")
                        del report
                        continue
                    references[code] = store.retain(report)
                    seen[code] = references[code]["report_sha256"]
                    summaries[code] = compact_report(report, references[code])
                    # Keep only publication metadata in the validation pass. The
                    # original JSON already lives in dataset; do not expand all
                    # financial proof trees simultaneously (several GB in ELEC).
                    reports.append({"symbol": code, "signal_date": report["signal_date"],
                        "parameter_version": report["parameter_version"],
                        "parameters": {"decision_policy_version": report.get("parameters", {}).get("decision_policy_version", 2)},
                        "data": {"analysis_session_date": report.get("data", {}).get("analysis_session_date")},
                        "decision": {"signal": report["decision"]["signal"]},
                        "sizing": {"indicative_shares": report["sizing"]["indicative_shares"]},
                        "evaluation_mode": report.get("evaluation_mode")})
                    del report
                dates = {dt.date.fromisoformat(r["signal_date"]) for r in reports}
                sessions = {
                    dt.date.fromisoformat(r["data"]["analysis_session_date"])
                    for r in reports if r.get("data", {}).get("analysis_session_date")
                }
                if len(sessions) > 1:
                    raise ValueError("MIXED_DAILY_ANALYSIS_SESSIONS")
                # Older per-stock histories remain in CSV/diagnostics. Publish only
                # the batch's completed session to the executable report collection.
                session_date = (next(iter(sessions)) if sessions else max(dates)).isoformat()
                policies = {r.get("parameters", {}).get("decision_policy_version", 2) for r in reports}
                if len(policies) != 1 or not policies <= {1, 2, 3, 4}:
                    raise ValueError("MIXED_DAILY_V6_SCORE_POLICIES")
                policy = {1: "legacy", 2: "unified", 3: "hybrid", 4: "hybrid-v2"}[next(iter(policies))]
                current_reports, stale_reports = [], []
                for report in reports:
                    date = report["signal_date"]
                    if date > session_date:
                        raise ValueError("FUTURE_DAILY_SIGNAL_DATE")
                    if date < session_date or not _is_supported_execution_symbol(report.get("symbol")):
                        if report["decision"]["signal"] in {"BUY", "BUY_SMALL"} or report["sizing"]["indicative_shares"] != 0:
                            raise ValueError("STALE_EXECUTABLE_V6_REPORT")
                        stale_reports.append(report)
                    else:
                        current_reports.append(report)
            # Do not mutate the in-memory native results used by the renderer,
            # PAE or notification gates. Only the persisted CSV uses references.
            exported = dataset.copy()
            exported["LES v6 Report"] = [json.dumps(references[str(code)], ensure_ascii=False, separators=(",", ":"))
                                         for code in exported["證券代號"]]
            payload = json.dumps({"model": "low_entry_score_v6", "storage_schema": "daily-reports/v2",
                                  "score_policy": policy,
                                  "parameter_versions": sorted({r["parameter_version"] for r in reports}),
                                  "signal_date": session_date,
                                  "reports": [summaries[r["symbol"]] for r in current_reports],
                                  "stale_reports": [summaries[r["symbol"]] for r in stale_reports]},
                                 ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            # Preserve all observations, not only eventual winners. Research
            # history is not read by load_reports or the order bridge.
            if not any(r.get("evaluation_mode") == "HISTORICAL_REPLAY" for r in reports):
                LeadershipWatchRepository(self.output_dir / "les_v6_leadership_history.sqlite3").record(file_name, list(summaries.values()))
            if stale_reports:
                logging.warning("LES v6 stale histories excluded from entry artifact: session=%s count=%s symbols=%s",
                                session_date, len(stale_reports), ",".join(r["symbol"] for r in stale_reports))
        with measure_stage("report_csv_write"):
            exported.to_csv(output_path, index=False)
            count_work("report_csv_bytes", output_path.stat().st_size)
        if payload is not None:
            target = self.output_dir / f"{file_name}_les_v6_reports.json"
            # Publish a complete snapshot atomically; readers cannot see half JSON.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile("w", dir=self.output_dir, encoding="utf-8", delete=False) as file:
                    temporary = file.name
                    with measure_stage("report_json_write"):
                        file.write(payload)
                os.replace(temporary, target)
                count_work("report_json_bytes", target.stat().st_size)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        return output_path


class TpexCsvRepository(CsvRepository):
    """Export original TPEx fields; normal analysis/report APIs stay unchanged."""

    def write_daily_rows(self, file_name: str, scheduled_time: str, rows: StockRows) -> Path:
        records = [row[TPEX_RAW_OFFSET:] for row in rows if row]
        if not records or any(len(row) != len(TPEX_FIELDS) for row in records):
            raise ValueError("TPEx export requires nonempty validated 17-field records")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / f"{file_name}_{scheduled_time}.csv"
        with target.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["日期", *TPEX_FIELDS])
            writer.writerows([scheduled_time, *row] for row in records)
        return target
