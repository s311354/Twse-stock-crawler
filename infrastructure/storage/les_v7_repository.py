"""Optional research-only immutable signal index over the existing evidence store."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from domain.les_v7.contracts import identity, quality_extension_schema
from domain.les_v7.report import validate_report
from infrastructure.storage.report_evidence import ReportEvidenceReader, ReportEvidenceWriter, STORE_NAME
from infrastructure.storage.valuation_source_verification import verify_report_valuation_sources


def validate_retained_report(report):
    validate_report(report)
    verify_report_valuation_sources(report)


class LesV7SignalRepository:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    _validate = staticmethod(validate_retained_report)

    def publish(self, report: dict) -> dict:
        self._validate(report)
        envelope = report["envelope"]
        event = envelope["event"]
        bundle = envelope["model"]["bundle_sha256"] or "NO_MODEL"
        signal_binding = dict(evidence_id=event["evidence_id"], bundle_hash=envelope["model"]["bundle_sha256"],
                              decision_policy_version=envelope["strategy"]["decision_policy_version"])
        if report["diagnostics"].get("fr2") is not None:
            signal_binding["components_id"] = identity(report["diagnostics"]["research_components"])
            bundle += ":FR2:" + signal_binding["components_id"]
            if report.get("fr2_execution") is not None:
                # Signal IDs are account-independent. Retain account-bound
                # variants in separate immutable research result archives.
                raise ValueError("FR2_ACCOUNT_RESULT_USE_RESEARCH_ARCHIVE")
        expected = identity(signal_binding)
        if event["signal_id"] != expected: raise ValueError("V7_SIGNAL_ID_BINDING_MISMATCH")
        if quality_extension_schema(report["diagnostics"]["quality"].get("profile_id")):
            # Keep the v1 index/table and all committed R1 rows untouched.
            # Each independently registered policy has its own namespace,
            # including R2 and R3 without a trained model.
            bundle += ":" + report["diagnostics"]["quality"]["quality_policy_id"]
        key = (event["security_id"], event["cutoff_at"], bundle)
        digest = identity(report)
        self.directory.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.directory / "les_v7_signal_index.sqlite3", timeout=30) as connection:
            tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables - {"signals"}: raise ValueError("V7_INDEX_MUST_BE_SEPARATE")
            connection.execute("CREATE TABLE IF NOT EXISTS signals (security TEXT, cutoff TEXT, bundle TEXT, "
                "signal_id TEXT UNIQUE NOT NULL, report_hash TEXT NOT NULL, reference TEXT NOT NULL, "
                "PRIMARY KEY(security,cutoff,bundle))")
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT signal_id,report_hash,reference FROM signals WHERE security=? AND cutoff=? AND bundle=?", key).fetchone()
            if existing:
                if existing[0] != expected or existing[1] != digest: raise ValueError("NONDETERMINISTIC_REPLAY")
                ref = json.loads(existing[2])
                with ReportEvidenceReader(self.directory) as reader:
                    if identity(reader.resolve(ref)) != digest: raise ValueError("V7_STORED_SIGNAL_HASH_MISMATCH")
            else:
                # Nodes commit before the atomic index. An interrupted index may
                # leave unreferenced immutable nodes, never a partially published signal.
                with ReportEvidenceWriter(self.directory / STORE_NAME) as writer:
                    ref = writer.retain(report)
                connection.execute("INSERT INTO signals VALUES (?,?,?,?,?,?)", (*key, expected, digest, json.dumps(ref)))
            return dict(signal_id=expected, report_sha256=digest, reference=ref)

    def load(self, signal_id: str) -> dict:
        path = self.directory / "les_v7_signal_index.sqlite3"
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            record = connection.execute("SELECT report_hash,reference FROM signals WHERE signal_id=?", (signal_id,)).fetchone()
        if record is None: raise KeyError("V7_SIGNAL_NOT_FOUND")
        with ReportEvidenceReader(self.directory) as reader:
            report = reader.resolve(json.loads(record[1]))
        if identity(report) != record[0]: raise ValueError("V7_STORED_SIGNAL_HASH_MISMATCH")
        self._validate(report)
        return report
