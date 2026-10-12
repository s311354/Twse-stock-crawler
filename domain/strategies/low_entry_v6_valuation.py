"""PIT fundamental valuation references, separate from technical P/E timing.

Eight quarters are NOT proof of a full economic cycle. These references never
supply a target multiple, expected return, fair value or allocation permission.
"""
from statistics import median

import pandas as pd

from domain.strategies.low_entry_v6_quality import (
    _eligible, _quality_preprocess, select_quality_inputs,
    normalize_quarters_with_diagnostics, number, ratio,
)


def valuation_context(records, *, symbol, market, as_of, quality, preprocessing=None) -> dict:
    result = {"policy_version": "pit-valuation-reference/v1", "purpose": "PORTFOLIO_DILIGENCE",
              "allocation_authority": "PAE", "status": "UNKNOWN", "affects_les_score": False,
              "fair_value": None, "expected_cagr": None, "reference_metrics": {},
              "sources": [], "missing_evidence": ["FULL_CYCLE_NORMALIZATION", "VALUATION_ASSUMPTIONS",
                                                  "DOWNSIDE_AND_RETURN_CALIBRATION"]}
    if quality.integrity != "PASS" or quality.status != "PASS":
        result["reason"] = "QUALITY_NOT_VERIFIED"
        return result
    try:
        cutoff = pd.to_datetime(as_of, utc=True, errors="raise")
        if preprocessing is not None:
            periods, _, excluded = select_quality_inputs(
                records, symbol=symbol, market=market, cutoff=cutoff, preprocessing=preprocessing)
            if excluded:
                # Valuation's historical contract is stricter than Quality's
                # scoped exclusion. Do not make formerly incomparable reference
                # data usable by sharing a shortened Quality inventory.
                periods = _eligible(records, symbol, market, cutoff)
                preprocessing = None
        else:
            periods = _eligible(records, symbol, market, cutoff)
        latest = max(periods)
        inputs = [latest - i for i in range(8)]
        quarters, _ = _quality_preprocess(preprocessing, "quarters",
            lambda: normalize_quarters_with_diagnostics(periods),
            records=records, symbol=symbol, market=market, cutoff=cutoff)
        margins = [ratio(quarters.get(p, {}).get("operating_income"), quarters.get(p, {}).get("revenue"))
                   for p in inputs]
        # Only directly reported quarter EPS, never differenced cumulative EPS.
        eps = [number(quarters.get(p, {}).get("basic_eps")) for p in inputs]
        row = periods[latest]
        unit_scale = {"dollar": 1., "thousand": 1000.}.get(row["unit"])
        # Consolidated total equity can include non-controlling interests;
        # those do not belong to the listed parent's shareholders.
        equity, shares = number(row.get("parent_equity")), number(row.get("shares_outstanding"))
        book = (ratio(equity * unit_scale, shares)
                if equity is not None and unit_scale is not None and row.get("shares_unit") == "shares" else None)
        result["reference_metrics"] = {
            "ttm_eps": quality.metrics.get("basic_eps_ttm"),
            "median_annualized_quarter_eps_8q": 4 * median(eps) if all(v is not None for v in eps) else None,
            "median_operating_margin_8q": median(margins) if all(v is not None for v in margins) else None,
            "book_value_per_share": book,
            "roe_ttm": quality.metrics.get("roe_ttm"),
        }
        result["sources"] = [{"period": str(p), "revision_id": r["revision_id"], "source": r["source"],
                              "available_at": r["_available"].isoformat(), "scope": r["scope"], "unit": r["unit"],
                              "quarter_eps_provenance": r.get("quarter_eps_provenance")}
                             for p, r in sorted(periods.items())]
        availability = [r["_available"] for r in periods.values()]
        availability.extend(pd.to_datetime(r["quarter_eps_provenance"]["available_at"], utc=True)
                            for r in periods.values() if r.get("quarter_eps_provenance"))
        result["references_available_at"] = max(availability).isoformat()
        result["reason"] = "REFERENCES_ONLY_NOT_FULL_CYCLE_OR_FAIR_VALUE"
        if book is None:
            result["missing_evidence"].append("VERIFIED_PARENT_EQUITY_AND_SHARE_UNITS")
    except (ValueError, TypeError, KeyError):
        result["reason"] = "INCOMPARABLE_OR_MISSING_VALUATION_SOURCES"
    return result
