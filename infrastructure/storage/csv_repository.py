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

    def __init__(self, output_dir: str | Path = DATA_DIR, *, workspace_root: str | Path | None = None,
                 valuation_history_verifier_scope=None, report_json_block_bytes=0) -> None:
        super().__init__(output_dir)
        self.workspace_root = workspace_root
        self.valuation_history_verifier_scope = valuation_history_verifier_scope
        self.report_json_block_bytes = report_json_block_bytes

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
        artifact_version = "v6"
        exported = dataset
        v7_candidate_records = None
        if "LES Model Version" in dataset and "7.0.0" in set(dataset["LES Model Version"]):
            if set(dataset["LES Model Version"]) != {"7.0.0"}: raise ValueError("MIXED_DAILY_LES_VERSIONS")
            required_erem = {"EREM 分數", "EREM 決策訊號", "EREM 理由"}
            if not required_erem <= set(dataset.columns):
                raise ValueError("V7_CANDIDATE_SNAPSHOT_EREM_FIELDS_REQUIRED")
            from application.v7_report_input import read_v7_report_json
            from infrastructure.storage.valuation_source_verification import (
                valuation_history_verifier_scope, verify_report_valuation_sources,
            )
            references, versions, seen, candidate_seen = {}, set(), {}, {}
            v7_candidate_records = []
            from infrastructure.storage.daily_candidate_archive import build_candidate
            scope = self.valuation_history_verifier_scope or valuation_history_verifier_scope
            with ReportEvidenceWriter(self.output_dir / STORE_NAME, workspace_root=self.workspace_root,
                    json_block_bytes=self.report_json_block_bytes) as store, scope() as verifier:
                for _, row in dataset.iterrows():
                    report = read_v7_report_json(row["LES v7 Report"])
                    verify_report_valuation_sources(report, verifier=verifier)
                    code = str(row["證券代號"])
                    if (report["symbol"] != code or report["signal_date"] != str(row["日期"])[:10]
                            or report["envelope"]["strategy"]["strategy_decision"] != row["低點決策"]):
                        raise ValueError("V7_REPORT_IDENTITY_DATE_OR_DECISION_MISMATCH")
                    if code in seen and digest(report) != seen[code]: raise ValueError("CONFLICTING_V7_REPORTS")
                    if code not in seen:
                        references[code] = store.retain(report)
                        seen[code] = references[code]["report_sha256"]
                    erem = {column: row[column] for column in dataset.columns if column.startswith("EREM ")}
                    candidate = build_candidate(report, references[code], erem)
                    candidate_identity = (candidate["signal_date"], code)
                    previous_candidate = candidate_seen.get(candidate_identity)
                    if previous_candidate is not None:
                        if digest(previous_candidate) != digest(candidate):
                            raise ValueError("CONFLICTING_V7_CANDIDATE_ROWS")
                    else:
                        candidate_seen[candidate_identity] = candidate
                        v7_candidate_records.append(candidate)
                    versions.add(report["parameter_version"])
                    del report
            exported = dataset.copy()
            exported["LES v7 Report"] = [json.dumps(references[str(code)], ensure_ascii=False) for code in exported["證券代號"]]
            artifact_version = "v7"
            payload = json.dumps(dict(model="low_entry_score_v7", storage_schema="daily-reports/v2",
                parameter_versions=sorted(versions), research_only=True, execution_enabled=False,
                pae_candidate_authorization_enabled=True,
                reports=list(references.values())), ensure_ascii=False, allow_nan=False)
        elif "LES Model Version" in dataset and "6.0.0" in set(dataset["LES Model Version"]):
            if set(dataset["LES Model Version"]) != {"6.0.0"}:
                raise ValueError("MIXED_DAILY_LES_VERSIONS")
            reports = []
            seen = {}
            references, summaries = {}, {}
            # Validate and retain each decoded report once. A later batch error
            # rolls back all nodes, including already flushed SQLite batches.
            with measure_stage("report_native_publish"), ReportEvidenceWriter(self.output_dir / STORE_NAME,
                    workspace_root=self.workspace_root, json_block_bytes=self.report_json_block_bytes) as store:
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
            target = self.output_dir / f"{file_name}_les_{artifact_version}_reports.json"
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
        if v7_candidate_records is not None:
            from infrastructure.storage.daily_candidate_archive import publish_snapshot
            snapshot_path = publish_snapshot(self.output_dir, file_name, v7_candidate_records)
            count_work("v7_daily_candidate_snapshots")
            count_work("v7_daily_candidate_rows", len(v7_candidate_records))
            logging.info("LES v7 immutable candidate snapshot published path=%s candidates=%s",
                         snapshot_path, len(v7_candidate_records))
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
