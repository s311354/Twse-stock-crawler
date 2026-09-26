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
        if "LES Model Version" in dataset and "6.0.0" in set(dataset["LES Model Version"]):
            if set(dataset["LES Model Version"]) != {"6.0.0"}:
                raise ValueError("MIXED_DAILY_LES_VERSIONS")
            reports = []
            seen = {}
            for _, row in dataset.iterrows():
                report = json.loads(row["LES v6 Report"])
                code = str(row["證券代號"])
                if report.get("model_version") != "6.0.0" or report.get("symbol") != code or report.get("signal_date") != str(row["日期"])[:10]:
                    raise ValueError("V6_REPORT_IDENTITY_OR_DATE_MISMATCH")
                if code in seen:
                    if report != seen[code]:
                        raise ValueError("CONFLICTING_V6_REPORTS")
                    continue
                seen[code] = report
                reports.append(report)
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
            payload = json.dumps({"model": "low_entry_score_v6", "signal_date": session_date,
                                  "reports": current_reports, "stale_reports": stale_reports},
                                 ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            # Preserve all observations, not only eventual winners. Research
            # history is not read by load_reports or the order bridge.
            if not any(r.get("evaluation_mode") == "HISTORICAL_REPLAY" for r in reports):
                LeadershipWatchRepository(self.output_dir / "les_v6_leadership_history.sqlite3").record(file_name, reports)
            if stale_reports:
                logging.warning("LES v6 stale histories excluded from entry artifact: session=%s count=%s symbols=%s",
                                session_date, len(stale_reports), ",".join(r["symbol"] for r in stale_reports))
        dataset.to_csv(output_path, index=False)
        if payload is not None:
            target = self.output_dir / f"{file_name}_les_v6_reports.json"
            # Publish a complete snapshot atomically; readers cannot see half JSON.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile("w", dir=self.output_dir, encoding="utf-8", delete=False) as file:
                    temporary = file.name
                    file.write(payload)
                os.replace(temporary, target)
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
