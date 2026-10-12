"""Append-only research observations, separate from order/execution artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from infrastructure.storage.report_evidence import ReportEvidenceReader


class LeadershipWatchRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, profile: str, reports: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS observations (
                profile TEXT NOT NULL, symbol TEXT NOT NULL, signal_date TEXT NOT NULL,
                decision_time TEXT NOT NULL, digest TEXT NOT NULL, watch_status TEXT NOT NULL,
                quality_status TEXT NOT NULL, entry_signal TEXT NOT NULL, close REAL,
                evidence_json TEXT NOT NULL,
                PRIMARY KEY(profile, symbol, signal_date, decision_time, digest))""")
            for report in reports:
                if "research" not in report:
                    continue
                evidence = {key: report[key] for key in ("research", "quality", "decision", "entry")}
                evidence["market_data_date"] = report["signal_date"]
                evidence["model_version"] = report["model_version"]
                evidence["parameter_version"] = report["parameter_version"]
                evidence["close"] = report.get("technical", {}).get("close")
                if "_report_reference" in report:
                    evidence = {"_report_reference": report["_report_reference"]}
                payload = json.dumps(evidence, ensure_ascii=False, sort_keys=True, allow_nan=False)
                digest = hashlib.sha256(payload.encode()).hexdigest()
                connection.execute("INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    str(profile), report["symbol"], report["signal_date"], report["decision_time"], digest,
                    report["research"]["watch"]["status"], report["quality"]["status"],
                    report["decision"]["signal"], report.get("technical", {}).get("close"), payload,
                ))

    def timeline(self, profile: str, symbol: str) -> list[dict]:
        """Retain failed/disappearing setups too; never backfill a historical BUY."""
        if not self.path.exists():
            return []
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute("""SELECT signal_date, decision_time, evidence_json
                FROM observations WHERE profile=? AND symbol=?
                ORDER BY signal_date, decision_time, digest""", (str(profile), str(symbol))).fetchall()
        result = []
        with ReportEvidenceReader(self.path.parent) as reader:
            for date, at, payload in rows:
                evidence = json.loads(payload)
                if "_report_reference" in evidence:
                    report = reader.resolve(evidence["_report_reference"])
                    evidence = {key: report[key] for key in ("research", "quality", "decision", "entry")}
                    evidence.update(market_data_date=report["signal_date"], model_version=report["model_version"],
                                    parameter_version=report["parameter_version"], close=report.get("technical", {}).get("close"))
                result.append(dict(signal_date=date, decision_time=at, **evidence))
        return result
