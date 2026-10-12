"""Source-bound reported-period inputs for the shared Quality evaluator.

No TTM reconstruction, annualized amounts, missing-value substitution or I/O.
Canonical comparisons/opening balances come from the selected original itself.
The explicitly legacy QUARTER contract remains readable, not a proof that two
original versions have an unchanged accounting basis.
"""
import pandas as pd

from domain.strategies.low_entry_v6_quality_evidence import validate_field


REPORTED_FLOW_FIELDS = ("revenue", "cost_of_revenue", "operating_income", "net_income", "cfo", "capex")
OPENING_FIELDS = ("total_assets", "total_liabilities", "total_equity")


def reported_period_inputs(periods, *, cutoff, ledger):
    latest_period = max(periods)
    row = periods[latest_period]
    basis = row["flow_basis"]
    current = {field: ledger[f"{latest_period}:{field}"]["value"] for field in REPORTED_FLOW_FIELDS}
    comparisons, opening, comparison_keys, opening_keys = {}, {}, {}, {}
    canonical = bool(row.get("filing_schema") or row.get("flow_derivation_policy")
                     or "comparative_ytd_field_evidence" in row)
    origin = row.get("quarter_flow_source", row)

    def bound_value(field, group, period, *, instant=False):
        proof = row.get(group + "_field_evidence", {}).get(field, {})
        value = row.get(group + "_fields", {}).get(field)
        # validate_field rechecks the value, hash, version, time, unit and exact
        # original context. The view's period is the context's reported period;
        # its receipt separately retains the original filing's period.
        view = {**origin, "period_end": period.end_time.date().isoformat(),
                "flow_basis": basis, "filing_schema": row.get("filing_schema") or "BOUND_ORIGINAL",
                field: value, "field_evidence": {field: proof}}
        if proof.get("basis") != ("INSTANT" if instant else basis):
            view[field] = None
        check = validate_field(view, field, period=period, cutoff=cutoff, instant=instant)
        check.update(filing_period=str(latest_period), comparison_policy="SELECTED_ORIGINAL_EXACT_CONTEXT")
        if view[field] is None and value is not None:
            check.update(status="PENDING_VERIFICATION", category="INCOMPARABLE",
                         reason="REPORTED_COMPARISON_BASIS_MISMATCH")
        return check

    for field in REPORTED_FLOW_FIELDS:
        key = f"{latest_period}:comparative:{field}"
        if basis == "YTD" or canonical:
            # QUARTER canonical sources would require an explicit quarter
            # comparative, never a YTD comparative disguised as a quarter.
            group = "comparative_ytd" if basis == "YTD" else "comparative_quarter"
            check = bound_value(field, group, latest_period - 4)
        else:
            prior = periods.get(latest_period - 4, {})
            check = dict(ledger[f"{latest_period - 4}:{field}"]) if prior else validate_field(
                {}, field, period=latest_period - 4, cutoff=cutoff)
            if prior and prior.get("flow_basis") != basis:
                check.update(value=None, status="INCOMPARABLE", category="INCOMPARABLE",
                             reason="REPORTED_COMPARISON_BASIS_MISMATCH")
            check.update(comparison_policy="LEGACY_SAME_PERIOD_NORMALIZED_CONTRACT")
        ledger[key] = check
        comparisons[field], comparison_keys[field] = check.get("value"), key

    opening_period = (pd.Period(year=latest_period.year - 1, quarter=4, freq="Q")
                      if basis == "YTD" else latest_period - 1)
    for field in OPENING_FIELDS:
        key = f"{latest_period}:opening:{field}"
        if basis == "YTD" or canonical:
            group = "comparative_year_end" if basis == "YTD" else "prior_quarter_end"
            check = bound_value(field, group, opening_period, instant=True)
        else:
            # A directly reported legacy quarter uses its preceding instant,
            # not a full-year average with a quarter's net income.
            opening_period = latest_period - 1
            check = dict(ledger[f"{opening_period}:{field}"]) if opening_period in periods else validate_field(
                {}, field, period=opening_period, cutoff=cutoff, instant=True)
            check.update(comparison_policy="LEGACY_PRECEDING_INSTANT_NORMALIZED_CONTRACT")
        ledger[key] = check
        opening[field], opening_keys[field] = check.get("value"), key
    return dict(period=str(latest_period), basis="ANNUAL" if basis == "YTD" and latest_period.quarter == 4 else basis,
                duration_fraction=latest_period.quarter / 4 if basis == "YTD" else .25,
                available_at=row["_available"].isoformat(), current=current, comparative=comparisons,
                opening=opening, opening_period=str(opening_period),
                current_keys={f: f"{latest_period}:{f}" for f in REPORTED_FLOW_FIELDS},
                comparison_keys=comparison_keys, opening_keys=opening_keys,
                comparison_policy="SELECTED_ORIGINAL_EXACT_CONTEXT" if basis == "YTD" or canonical
                else "LEGACY_SAME_PERIOD_NORMALIZED_CONTRACT")
