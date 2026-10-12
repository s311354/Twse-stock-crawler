"""Audit retained outputs, never reconstruct missing historical model inputs."""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time
import json
import math
import re
from zoneinfo import ZoneInfo

from application.replay_contracts import digest
from application.research_audit import compare_results

SCHEMA = "daily-report-audit/v1"
TAIPEI = ZoneInfo("Asia/Taipei")
ENTRY_SIGNALS = {"BUY", "BUY_SMALL"}
INPUT_GAPS = {
    "ACCOUNT_SNAPSHOT_NOT_RETAINED": "Retain explicit cash, equity, holdings and their valuation evidence; do not infer amounts from sizing.",
    "PAE_PREDECESSOR_NOT_RETAINED": "Retain the original predecessor observations, including an explicit empty state if applicable.",
    "FILING_SELECTION_NOT_RETAINED": "Retain every selected filing version, scope, original document and publication/availability evidence.",
    "MARKET_INPUTS_NOT_RETAINED": "Retain the exact OHLCV window, calendar, benchmark, breadth and peer-universe inputs.",
    "RESTRICTION_SOURCE_DOCUMENTS_NOT_RETAINED": "Retain dated official source documents and coverage, not only the reported CLEAR/RESTRICTED result.",
    "CALIBRATION_INPUTS_NOT_RETAINED": "Retain cutoff-valid empirical calibration and its model/parameter binding, if used.",
    "FULL_MODEL_CODE_IDENTITY_NOT_RETAINED": "Retain source/dependency hashes and LES/EREM/PAE parameters; report version labels alone are insufficient.",
}


def _index(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label}_ROW_MUST_BE_OBJECT")
        symbol = row.get(key)
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"{label}_SYMBOL_REQUIRED")
        symbol = symbol.strip()
        if symbol in result:
            raise ValueError(f"{label}_DUPLICATE_SYMBOL:{symbol}")
        result[symbol] = row
    return result


def _instant(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("AUDIT_TIMESTAMP_REQUIRES_TIMEZONE")
    return parsed


def _number(value: object) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("AUDIT_NON_FINITE_SCORE")
    return number


def audit_daily_report(csv_rows: list[dict], document: dict, stocklist: str,
                       *, stocklist_format: str) -> dict:
    """Field parity covers exports only; input and daily/replay parity stay UNKNOWN.

    Legacy stocklist numbers are row selectors, NOT security identifiers. Without
    the original lookup receipt only their count can be reconciled to the report.
    Missing/extra report symbols remain in the audit instead of being dropped.
    """
    if not csv_rows or not isinstance(document, dict) or not isinstance(document.get("reports"), list):
        raise ValueError("DAILY_REPORT_ROWS_REQUIRED")
    tokens = re.split(r"[;,\s]+", stocklist.strip()) if stocklist.strip() else []
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("STOCKLIST_EMPTY_OR_DUPLICATE")
    if stocklist_format not in {"symbols", "legacy-indices"}:
        raise ValueError("EXPLICIT_STOCKLIST_FORMAT_REQUIRED")
    if stocklist_format == "legacy-indices" and any(not t.isascii() or not t.isdigit() for t in tokens):
        raise ValueError("STOCKLIST_INVALID_LEGACY_INDEX")
    if stocklist_format == "legacy-indices" and len({int(t) for t in tokens}) != len(tokens):
        raise ValueError("STOCKLIST_DUPLICATE_LEGACY_INDEX")
    csv_index = _index(csv_rows, "證券代號", "CSV")
    reports, locations = [], {}
    for bucket in ("reports", "stale_reports"):
        values = document.get(bucket, [])
        if not isinstance(values, list):
            raise ValueError("REPORT_BUCKET_MUST_BE_ARRAY")
        for i, report in enumerate(values):
            if not isinstance(report, dict):
                raise ValueError("JSON_ROW_MUST_BE_OBJECT")
            reports.append(report)
            locations[report.get("symbol")] = f"/{bucket}/{i}"
    report_index = _index(reports, "symbol", "JSON")
    expected = set(tokens) if stocklist_format == "symbols" else set()
    symbols = sorted(csv_index.keys() | report_index.keys() | expected)
    rows = []
    for symbol in symbols:
        csv_row, report = csv_index.get(symbol, {}), report_index.get(symbol, {})
        decision, data = report.get("decision", {}), report.get("data", {})
        quality = report.get("quality", {})
        verification = report.get("research", {}).get("verification", {})
        signal_date = report.get("signal_date")
        cutoff = datetime.combine(date.fromisoformat(signal_date), time(13, 30), TAIPEI) if signal_date else None
        decision_at = _instant(report.get("decision_time"))
        latest_at = _instant(verification.get("latest_available_at"))
        evidence = data.get("restriction_evidence", {})
        restriction_at = _instant(evidence.get("available_at"))
        differences, issues = [], []
        if not csv_row:
            issues.append("CSV_ROW_MISSING")
        if not report:
            issues.append("JSON_REPORT_MISSING")
        if csv_row and report:
            embedded = csv_row.get("LES v6 Report")
            if embedded:
                differences.extend("LES v6 Report/" + p for p in
                                   compare_results(json.loads(embedded), report)["differences"])
            else:
                issues.append("CSV_EMBEDDED_REPORT_MISSING")
            fields = {
                "日期": signal_date, "證券名稱": report.get("name"),
                "低點決策": decision.get("signal"), "LES v6 Decision": decision.get("signal"),
                "LES Model Version": report.get("model_version"),
                "LES Parameter Version": report.get("parameter_version"),
                "Quality Status": quality.get("status"),
            }
            for key, value in fields.items():
                if csv_row.get(key) != value:
                    differences.append(key)
            for key, value in {"低點分數": decision.get("score"), "LES v6 Score": decision.get("score"),
                               "Quality Score": quality.get("score")}.items():
                if key not in csv_row or _number(csv_row[key]) != _number(value):
                    differences.append(key)
        if differences:
            issues.append("EXPORT_CONTENT_MISMATCH")
        if cutoff and decision_at and decision_at.astimezone(TAIPEI).date() != cutoff.date():
            issues.append("MARKET_AND_DECISION_DATE_DIFFER")
        if signal_date and signal_date != document.get("signal_date"):
            issues.append("MARKET_DATE_BEFORE_OR_DIFFERENT_FROM_REPORT_SESSION")
        if cutoff and latest_at and latest_at > cutoff:
            issues.append("REPORTED_FINANCIAL_AVAILABILITY_AFTER_SIGNAL_CLOSE")
        if cutoff and restriction_at and restriction_at > cutoff:
            issues.append("REPORTED_RESTRICTIONS_AFTER_SIGNAL_CLOSE")
        if verification.get("missing_periods"):
            issues.append("REPORTED_MISSING_FINANCIAL_PERIODS")
        if verification.get("unresolved_fields"):
            issues.append("REPORTED_FINANCIAL_FIELDS_UNRESOLVED")
        rows.append(dict(
            symbol=symbol, name=report.get("name", csv_row.get("證券名稱")),
            report_pointer=locations.get(symbol),
            report_bucket=locations[symbol].split("/")[1] if symbol in locations else None,
            signal_date=signal_date, csv_signal_date=csv_row.get("日期"),
            hypothetical_signal_close_cutoff=cutoff.isoformat() if cutoff else None,
            decision_time=report.get("decision_time"),
            signal_to_decision_calendar_days=(decision_at.astimezone(TAIPEI).date() - cutoff.date()).days
                if decision_at and cutoff else None,
            model_version=report.get("model_version"), parameter_version=report.get("parameter_version"),
            reported_parameters_sha256=digest(report["parameters"]) if "parameters" in report else None,
            original_les=decision, original_quality=quality,
            original_erem={k: v for k, v in csv_row.items() if k.startswith("EREM ")},
            original_pae={k: v for k, v in csv_row.items() if k.startswith("PAE ")},
            reported_gates={k: v for k, v in csv_row.items() if k.endswith(" Gate")},
            reported_entry=report.get("entry", {}), reported_sizing=report.get("sizing", {}),
            reported_data=data, financial_verification=verification,
            financial_availability_audit="UNVERIFIABLE_WITHOUT_SELECTED_VERSIONS_AND_PROOFS",
            financial_comparability_audit="REPORTED_GAPS" if verification.get("missing_periods") or
                verification.get("unresolved_fields") else "NOT_INDEPENDENTLY_VERIFIED",
            historical_restrictions_audit="NOT_VERIFIED_FROM_REPORT_SUMMARY",
            export_parity=dict(scope="LES_EMBEDDED_REPORT_AND_MAPPED_CSV_FIELDS",
                status="MATCH" if csv_row and report and not differences and
                "CSV_EMBEDDED_REPORT_MISSING" not in issues else "INCOMPLETE_OR_MISMATCH", differences=differences),
            issues=issues, research_entry_eligibility="NOT_EVALUATED", is_fill=False,
        ))

    def counts(values):
        return dict(sorted(Counter(str(v) if v is not None else "MISSING" for v in values).items()))

    summary = dict(
        audited_symbols=len(rows), csv_rows=len(csv_rows), json_reports=len(reports),
        report_buckets={key: len(document.get(key, [])) for key in ("reports", "stale_reports")},
        csv_only_symbols=sorted(csv_index.keys() - report_index.keys()),
        json_only_symbols=sorted(report_index.keys() - csv_index.keys()),
        original_les_decisions=counts(r["original_les"].get("signal") for r in rows),
        original_quality_statuses=counts(r["original_quality"].get("status") for r in rows),
        original_erem_decisions=counts(r["original_erem"].get("EREM 決策訊號") for r in rows),
        original_pae_decisions=counts(r["original_pae"].get("PAE Decision") for r in rows),
        signal_dates=counts(r["signal_date"] for r in rows),
        decision_dates_taipei=counts(_instant(r["decision_time"]).astimezone(TAIPEI).date().isoformat()
            if r["decision_time"] else None for r in rows),
        model_versions=counts(r["model_version"] for r in rows),
        parameter_versions=counts(r["parameter_version"] for r in rows),
        restriction_scopes=counts(r["reported_data"].get("restriction_evidence", {}).get("scope") for r in rows),
        reported_restriction_statuses=counts(r["reported_data"].get("restriction_check", {}).get("status") for r in rows),
        export_parity=counts(r["export_parity"]["status"] for r in rows),
        issue_counts=counts(issue for r in rows for issue in r["issues"]),
        missing_period_counts=counts(v for r in rows for v in r["financial_verification"].get("missing_periods", [])),
        unresolved_field_counts=counts(v for r in rows for v in r["financial_verification"].get("unresolved_fields", [])),
    )
    return dict(schema=SCHEMA, status="REPORT_ONLY_INPUT_PARITY_UNVERIFIABLE",
        declared_signal_date=document.get("signal_date"), dataset_id=None,
        stocklist=dict(format=stocklist_format, token_count=len(tokens),
            report_count_matches=len(tokens) == len(csv_index) == len(report_index),
            identity_status="UNVERIFIED_LEGACY_INDEX_MAPPING" if stocklist_format == "legacy-indices" else
                "MATCH" if expected == csv_index.keys() == report_index.keys() else "MISMATCH",
            missing_requested_symbols=sorted(expected - csv_index.keys()) if expected else [],
            unexpected_report_symbols=sorted(set(symbols) - expected) if expected else []),
        summary=summary, rows=rows,
        entry_anchor_symbols=[r["symbol"] for r in rows if r["original_les"].get("signal") in ENTRY_SIGNALS],
        missing_original_inputs=INPUT_GAPS,
        stages=dict(original_report="AUDITED", same_input_replay="BLOCKED_MISSING_ORIGINAL_INPUTS",
            historical_reconstruction="NOT_RUN", repaired_dataset="NOT_PUBLISHED", simulation="NOT_RUN"),
        limitations=["Export parity is not daily-versus-replay parity.",
            "EREM/PAE outputs have only the retained CSV; there is no independent output or input parity evidence.",
            "Signal-date close is a diagnostic comparison boundary, not the original decision cutoff.",
            "Reported latest_available_at does not identify every selected filing or prove its publication time.",
            "Missing originals are not replaced with current cache, account, restriction lists or formal PAE state.",
            "A native BUY/BUY_SMALL and indicative sizing are not a fill, historical eligibility or live authorization."])
