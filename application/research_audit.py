"""Coverage is not profitability. Keep conditions, evidence gaps and decisions apart."""
from __future__ import annotations

from collections import Counter
from application.replay_contracts import digest
from application.research_acceptance import acceptance_checks, decision_stages

POLICY = {
    "id": "research-coverage/v3", "default_calendar_days": 700,
    "range": "Explicit inclusive range; 700 calendar days is a default, not a completeness gate",
    "calendar": "All dates known from verified official schedule and exceptions",
    "bars": "All expected eligible sessions valid; proven prelisting/halt separately disclosed",
    "financials": "No unresolved reported-period scoring inputs; TTM/quarter gaps retained as diagnostics only",
    "restrictions": "CLEAR or RESTRICTED from complete scoped evidence, never missing=clear",
    "kelly": "Dated, sourced, version-matched empirical calibration ending before cutoff",
    "acceptance": "No unresolved evidence blockers. BUY count and profit are NOT criteria",
}


def coverage_audit(request, selected, result: dict | None = None) -> dict:
    # Native research annotates financial coverage. A standalone caller passing
    # result + a fresh selection must see the same annotations as the one-select
    # API, not silently omit the native component coverage.
    diag = (result or {}).get('diagnostics', selected.diagnostics)
    dates = dict(zip(selected.market.stock_numbers, selected.market.dates))
    start, end = request.history_start, request.history_end
    calendar_days = (end - start).days + 1
    unknown = diag.get("calendar_unknown_dates", [])
    sessions = [d for d in selected.sessions if start.isoformat() <= d <= end.isoformat()]
    models = {d["symbol"]: d for d in (result or {}).get("decisions", [])}
    rows = []
    for symbol in request.symbols:
        financial = diag["financials"].get(symbol, {})
        model = models.get(symbol, {})
        quality = model.get("les_report", {}).get("quality", {})
        gates = model.get("hard_gates", {})
        failures, insufficient = [], []
        for key, status in sorted(gates.items()):
            if status == "PASS":
                continue
            if (status in {"UNKNOWN", "NOT_EVALUATED", "ASSUMED_NOT_VERIFIED"} or key == "published_report" or
                    key == "quality" and quality.get("status") in {"PROVISIONAL", "UNKNOWN", "DATA_INSUFFICIENT"}):
                insufficient.append(key)
            else:
                failures.append(key)
        missing = diag["missing_bars"].get(symbol, [])
        restrictions, kelly = diag["restrictions"][symbol], diag["kelly"][symbol]
        problems = []
        if result is None: problems.append("NATIVE_GATES_NOT_EVALUATED")
        if unknown: problems.append("CALENDAR_UNKNOWN")
        if missing: problems.append("MARKET_GAPS")
        if any(item.get("symbol") in (None, symbol) for item in diag["invalid_bars"]): problems.append("INVALID_OR_CONFLICTING_BARS")
        if diag.get("missing_taiex_dates"): problems.append("MISSING_BENCHMARK")
        if diag.get("missing_breadth_dates"): problems.append("MISSING_BREADTH")
        if financial.get("missing_periods"): problems.append("MISSING_FINANCIAL_PERIODS")
        if financial.get("unresolved_fields"): problems.append("INCOMPARABLE_OR_UNRESOLVED_FIELDS")
        if restrictions["status"] == "UNKNOWN": problems.append("RESTRICTION_EVIDENCE_UNKNOWN")
        if kelly["status"] != "PASS": problems.append("KELLY_UNKNOWN")
        if diag.get("raw_filing_errors", {}).get(symbol): problems.append("RAW_FILING_UNVERIFIABLE")
        if diag.get("publication_time_assumptions", {}).get(symbol): problems.append("ASSUMED_PUBLICATION_TIME")
        if insufficient: problems.append("UNRESOLVED_NATIVE_GATES")
        rows.append(dict(symbol=symbol, cutoff=request.as_of.isoformat(),
            window_start=start.isoformat(), window_end=end.isoformat(), inclusive_calendar_days=calendar_days,
            actual_history_start=request.history_start.isoformat(),
            expected_market_sessions=None if unknown else len(sessions), known_open_sessions=len(sessions),
            valid_bar_days=len(dates.get(symbol, ())), missing_bars=missing,
            bar_exemptions=diag.get("bar_exemptions", {}).get(symbol, {}),
            financial=financial, quality_status=quality.get("status"), quality_coverage=quality.get("coverage", {}),
            data_confidence=model.get("les_report", {}).get("data", {}).get("confidence"),
            liquidity=model.get("les_report", {}).get("liquidity", {}),
            restrictions=restrictions, kelly=kelly, final_les=model.get("raw", {}).get("les"),
            condition_failures=failures, insufficient_gates=insufficient,
            excluded_evidence=diag["excluded_evidence"].get(symbol, []),
            source_errors=diag.get("raw_filing_errors", {}).get(symbol, []),
            coverage_problems=problems, coverage_status="PARTIAL" if problems else "COMPLETE"))
        # Evidence completeness is separate from research eligibility. Accepted
        # assumptions remain visible/PARTIAL without becoming execution gates.
        rows[-1]["evidence_warnings"] = model.get("evidence_warnings", [])
        rows[-1]["research_entry_eligible"] = model.get("research_entry_eligible", False)
        rows[-1]["acceptance_checks"] = acceptance_checks(request, selected, diag, symbol)
        rows[-1]["decision_stages"] = decision_stages(model)
        availability = rows[-1]["acceptance_checks"]["financial_availability"]
        if availability["pending_proof_count"]:
            problems.append("PUBLICATION_PROOF_PENDING")
        if not availability["selected_versions"]:
            problems.append("NO_SELECTED_FINANCIAL_VERSIONS")
        rows[-1]["coverage_status"] = "PARTIAL" if problems else "COMPLETE"
    return dict(schema="research-coverage/v1", policy=POLICY, dataset_id=request.evidence_id,
        as_of=request.as_of.isoformat(), calendar_unknown_dates=unknown,
        calendar_excluded=diag.get("calendar_excluded", []), invalid_bars=diag["invalid_bars"],
        lifecycle_excluded=diag.get("lifecycle_excluded", []),
        missing_benchmark_dates=diag.get("missing_taiex_dates", []), missing_breadth_dates=diag.get("missing_breadth_dates", []),
        rows=rows, status="PARTIAL" if any(r["coverage_problems"] for r in rows) else "COMPLETE",
        condition_failure_counts=dict(Counter(k for r in rows for k in r["condition_failures"])),
        data_gap_counts=dict(Counter(k for r in rows for k in r["coverage_problems"])),
        unsupported_conclusions=["No unbiased net-performance or model-ranking conclusion from incomplete coverage",
                                 "Saved output is not proof of historical deployment or executable fills"])


def summarize(audit: dict) -> str:
    lines = [f"# Evidence coverage: {audit['status']}", "", f"Cutoff: {audit['as_of']}",
             f"Dataset: {audit['dataset_id']}", "", "| Symbol | Valid bars | Expected | Missing quarters | Coverage |", "|---|---:|---:|---|---|"]
    for row in audit["rows"]:
        lines.append(f"| {row['symbol']} | {row['valid_bar_days']} | {row['expected_market_sessions']} | "
                     f"{', '.join(row['financial'].get('missing_periods', []))} | {', '.join(row['coverage_problems']) or 'COMPLETE'} |")
    lines += ["", "## Four Evidence Checks", "",
              "| Stock | Availability | Comparability | Restrictions | Market / Kelly | Raw LES / EREM / PAE | Eligible |",
              "|---|---|---|---|---|---|---|"]
    for row in audit["rows"]:
        checks = row.get("acceptance_checks", {})
        stages = row.get("decision_stages", {})
        statuses = [checks.get(key, {}).get("status", "NOT_EVALUATED") for key in
                    ("financial_availability", "financial_comparability", "historical_restrictions", "market_and_calibration")]
        raw = stages.get("raw", {})
        signals = " / ".join(str(raw.get(key) or "N/A") for key in ("les", "erem", "pae"))
        lines.append(f"| {row['symbol']} | {' | '.join(statuses)} | {signals} | {row.get('research_entry_eligible', False)} |")
    for row in audit["rows"]:
        checks = row.get("acceptance_checks", {})
        stages = row.get("decision_stages", {})
        lines += ["", f"### {row['symbol']} Next Steps"]
        for key, check in checks.items():
            if isinstance(check, dict) and check.get("next_action") not in (None, "NONE"):
                lines.append(f"- {key}: {check['next_action']}")
        lines.append("- Missing quarters: " + ", ".join(row['financial'].get('missing_periods', [])))
        lines.append("- Unresolved fields: " + ", ".join(row['financial'].get('unresolved_fields', [])))
        for group, blockers in stages.get("blockers", {}).items():
            lines.append(f"- {group}: " + (", ".join(f"{key}={value}" for key, value in blockers.items()) or "NONE"))
    lines += ["", "PDF assumptions remain disclosed, not VERIFIED. Research eligibility is not a fill or live order authorization."]
    lines += ["", "Condition failures: " + str(audit["condition_failure_counts"]),
              "Data gaps: " + str(audit["data_gap_counts"])]
    return "\n".join(lines + ["", "UNKNOWN is not PASS. A higher BUY count is not evidence repair.", ""])


def compare_results(left: dict, right: dict, *, different_execution: bool = False) -> dict:
    """Compare ALL result fields; no selective signal/score-only parity shortcut."""
    # Archive IDs and timestamps live outside engine results. Changing portfolio
    # IDs can affect atomic plans; those nested differences remain visible.
    allowed = {"replay_id": "Pure caller execution label; nested portfolio/plan IDs remain compared"} if different_execution else {}
    differences = []
    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                if not path and key in allowed: continue
                if key not in a or key not in b:
                    differences.append("/".join((*path, key)))
                else: walk(a[key], b[key], (*path, key))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)): walk(x, y, (*path, str(i)))
        elif a != b:
            differences.append("/".join(path))
    walk(left, right, ())
    return dict(equal=not differences, differences=differences, allowed_metadata_differences=allowed,
                left_sha256=digest(left), right_sha256=digest(right))
