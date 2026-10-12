"""Project validated replay diagnostics into four evidence acceptance checks.

This is a report adapter, not a second evaluator. It never grants eligibility,
changes publication times, derives financial ratios, or authorizes fills.
"""
from __future__ import annotations

from copy import deepcopy


def _excluded_filing(row: dict) -> dict:
    result = deepcopy(row)
    if row.get("reason") == "AFTER_CUTOFF":
        if row.get("availability_source") == "FIRST_OBSERVED_DOWNLOAD":
            result.update(gap_status="DOCUMENT_PENDING_PUBLICATION_PROOF",
                          next_action="COLLECT_EXACT_VERSION_PUBLICATION_PROOF")
        elif row.get("publication_policy"):
            result.update(gap_status="POLICY_AFTER_CUTOFF",
                          next_action="COLLECT_EXACT_VERSION_PUBLICATION_PROOF")
        elif row.get("availability_source") == "OFFICIAL_FILING_TIMESTAMP":
            result.update(gap_status="PUBLISHED_AFTER_CUTOFF",
                          next_action="KEEP_EXCLUDED_USE_VERIFIED_PREDECESSOR")
        else:
            result.update(gap_status="UNVERIFIED_AVAILABILITY",
                          next_action="REVIEW_VERSION_AND_SOURCE_EVIDENCE")
    else:
        result.update(gap_status="UNRESOLVED_VERSION_OR_AVAILABILITY",
                      next_action="REVIEW_VERSION_AND_SOURCE_EVIDENCE")
    return result


def acceptance_checks(request, selected, diagnostics: dict, symbol: str) -> dict:
    financial = deepcopy(diagnostics.get("financials", {}).get(symbol, {}))
    versions = [deepcopy(row) for row in selected.evidence.get("filings", {}).get(symbol, [])
                if row.get("selected")]
    excluded = [_excluded_filing(row) for row in diagnostics.get("excluded_evidence", {}).get(symbol, [])
                if row.get("fiscal_year") is not None]
    errors = deepcopy(diagnostics.get("raw_filing_errors", {}).get(symbol, []))
    assumptions = deepcopy(diagnostics.get("publication_time_assumptions", {}).get(symbol, []))
    pending = [row for row in excluded if row["gap_status"] not in {"PUBLISHED_AFTER_CUTOFF", "POLICY_AFTER_CUTOFF"}]
    availability = dict(status="UNKNOWN" if not versions else "PARTIAL" if pending or errors or assumptions else "PASS",
                        selected_versions=versions, excluded_versions=excluded, source_errors=errors,
                        assumptions=assumptions, pending_proof_count=len(pending),
                        next_action="COLLECT_EXACT_VERSION_PUBLICATION_PROOF" if pending or assumptions else
                                    "REVIEW_SOURCE_ERRORS" if errors else
                                    "ACQUIRE_ORIGINAL_AND_PUBLICATION_EVIDENCE" if not versions else "NONE")
    comparable = bool(versions and financial.get("latest_period") and not financial.get("missing_periods")
                      and not financial.get("unresolved_fields"))
    comparability = dict(financial, status="PASS" if comparable else "UNKNOWN")
    comparability["next_action"] = "NONE" if comparable else "REVIEW_NATIVE_REPORTED_PERIOD_SCOPE_UNIT_AND_SOURCE_CONTEXTS"

    restriction = deepcopy(diagnostics.get("restrictions", {}).get(symbol, {}))
    restrictions = dict(status={"CLEAR": "PASS", "RESTRICTED": "FAIL"}.get(restriction.get("status"), "UNKNOWN"),
                        check=restriction, source_evidence=deepcopy(selected.evidence.get("restrictions", {}).get(symbol)),
                        next_action="NONE" if restriction.get("status") in {"CLEAR", "RESTRICTED"} else
                                    "IMPORT_OFFICIAL_COVERAGE_CARRY_IN_AND_REVISION_EVIDENCE")
    histories = dict(zip(selected.market.stock_numbers, selected.market.dates))
    dates = histories.get(symbol, ())
    unknown_calendar = diagnostics.get("calendar_unknown_dates", [])
    sessions = [day for day in selected.sessions if str(request.history_start) <= day <= str(request.history_end)]
    expected_latest = sessions[-1] if sessions and not unknown_calendar else None
    latest_bar = max(dates) if dates else None
    invalid = [row for row in diagnostics.get("invalid_bars", []) if row.get("symbol") in (None, symbol)]
    missing = diagnostics.get("missing_bars", {}).get(symbol, [])
    market = dict(
        calendar=dict(status="UNKNOWN" if unknown_calendar else "PASS", unknown_dates=unknown_calendar),
        prices=dict(status="PASS" if expected_latest is not None and latest_bar == expected_latest
                    and not missing and not invalid else "UNKNOWN",
                    expected_latest_session=expected_latest, latest_bar=latest_bar,
                    missing_dates=missing, invalid_rows=invalid),
        benchmark=dict(status="UNKNOWN" if unknown_calendar or diagnostics.get("missing_taiex_dates") else "PASS",
                       missing_dates=diagnostics.get("missing_taiex_dates", [])),
        breadth=dict(status="UNKNOWN" if unknown_calendar or diagnostics.get("missing_breadth_dates") else "PASS",
                     missing_dates=diagnostics.get("missing_breadth_dates", [])),
        kelly=deepcopy(diagnostics.get("kelly", {}).get(symbol, {"status": "UNKNOWN"})))
    market["status"] = "PASS" if all(check["status"] == "PASS" for check in market.values()) else "UNKNOWN"
    market["next_action"] = "NONE" if market["status"] == "PASS" else "IMPORT_MISSING_OFFICIAL_MARKET_CALENDAR_OR_DATED_CALIBRATION"
    return dict(schema="research-acceptance/v1", financial_availability=availability,
                financial_comparability=comparability, historical_restrictions=restrictions,
                market_and_calibration=market)


def decision_stages(model: dict) -> dict:
    """Keep native rules, evidence, account allocation and fills distinguishable."""
    groups = {"native_model": {}, "historical_evidence": {}, "account_allocation": {}}
    evidence = {"restrictions", "market_history", "published_report", "kelly", "calendar", "market_context"}
    for gate, status in model.get("hard_gates", {}).items():
        group = "account_allocation" if gate in {"account", "allocation"} else (
            "historical_evidence" if gate in evidence else "native_model")
        if status != "PASS":
            groups[group][gate] = status
    return dict(raw=deepcopy(model.get("raw", {})), blockers=groups,
                blocking_reasons=list(model.get("blocking_reasons", [])),
                research_entry_eligible=model.get("research_entry_eligible", False),
                execution_status="NOT_SIMULATED", live_order_authorized=False, is_fill=False)
