"""Point-in-time, OpenAPI-only core quality for LES v6 (no I/O)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite

import pandas as pd

from domain.fundamental_models import FundamentalSnapshot


FLOW_FIELDS = ("revenue", "cost_of_revenue", "operating_income", "net_income", "basic_eps")
BALANCE_FIELDS = (
    "current_assets", "non_current_assets", "total_assets",
    "current_liabilities", "total_liabilities", "total_equity",
)
OPTIONAL_FIELDS = (
    "cfo", "capex", "fcf", "cash", "interest_income", "interest_expense",
    "accounts_receivable", "inventory", "net_ppe", "construction_in_progress",
    "shares_outstanding", "dividends", "share_repurchases", "auditor_opinion",
    "going_concern", "material_accounting_issue", "audit_review_required",
)
OFFICIAL_SOURCES = {"TWSE_OPENAPI", "TPEX_OPENAPI", "MOPS_IXBRL"}
MINIMUM_ENTRY_QUALITY = 70.


def number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if isfinite(result) else None


def ratio(numerator: object, denominator: object) -> float | None:
    n, d = number(numerator), number(denominator)
    return number(n / d) if n is not None and d is not None and d > 0 else None


def margin_drawdown(values: Sequence[float | None]) -> float | None:
    """Largest chronological margin deterioration, in percentage points, not %.

    Improving margins are not downside instability. Missing quarters remain
    unknown, and a peak followed by a decline still counts even in a net uptrend.
    """
    if not values or any(number(value) is None for value in values):
        return None
    peak, drawdown = values[0], 0.
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, peak - value)
    return drawdown


def snapshot_record(snapshot: FundamentalSnapshot, *, market: str = "TWSE") -> dict:
    """Adapt existing official snapshots; first observation is conservative availability.

    Amounts remain in the source's TWD-thousands unit; EPS remains TWD/share.
    Never synthesize CFO, audit evidence, or historical availability.
    """
    period = pd.Period(year=snapshot.fiscal_year, quarter=snapshot.quarter, freq="Q")
    revenue, gross = snapshot.revenue_ytd, snapshot.gross_profit_ytd
    assets, current = snapshot.total_assets, snapshot.current_assets
    return {
        "symbol": snapshot.symbol, "market": market, "currency": "TWD",
        "unit": "thousand", "scope": "consolidated", "source": "TWSE_OPENAPI",
        "period_end": period.end_time.date().isoformat(), "flow_basis": "YTD",
        "available_at": snapshot.observed_at, "revision_id": snapshot.content_sha256,
        "revenue": revenue,
        "cost_of_revenue": revenue - gross if revenue is not None and gross is not None else None,
        "operating_income": snapshot.operating_income_ytd,
        "net_income": snapshot.net_income_ytd, "basic_eps": snapshot.basic_eps_ytd,
        "current_assets": current, "total_assets": assets,
        "non_current_assets": assets - current if assets is not None and current is not None else None,
        "current_liabilities": snapshot.current_liabilities,
        "total_liabilities": snapshot.total_liabilities, "total_equity": snapshot.total_equity,
    }


def combine_filing_sources(mops: Sequence[Mapping], batch: Sequence[Mapping], *, as_of: object) -> list[dict]:
    """Keep batch core precedence, retaining separately dated quarter-EPS evidence.

    Never splice cash flows/balances or subtract YTD EPS. The latest eligible
    MOPS revision must tie to the selected batch before its quarter EPS is used.
    The consumer rechecks the supplement's timestamp at its own decision cutoff.
    """
    cutoff = pd.to_datetime(as_of, utc=True, errors="coerce")
    eligible = []
    for record in mops:
        available = pd.to_datetime(record.get("available_at"), utc=True, errors="coerce")
        if pd.notna(cutoff) and pd.notna(available) and available <= cutoff:
            eligible.append((available, record))
    result = []
    for record in batch:
        row = dict(record)
        candidates = [(date, item) for date, item in eligible
                      if all(item.get(key) == row.get(key) for key in ("symbol", "market", "period_end"))]
        if candidates:
            latest = max(date for date, _ in candidates)
            versions = [item for date, item in candidates if date == latest]
            # Ambiguous same-time revisions cannot supply an enrichment.
            if all(item == versions[0] for item in versions):
                row["quarter_eps_evidence"] = dict(versions[0])
        result.append(row)
    periods = {row["period_end"] for row in batch}
    return [dict(row) for row in mops if row["period_end"] not in periods] + result


def _quarter_eps_supplement(row: Mapping, as_of: pd.Timestamp) -> Mapping | None:
    evidence = row.get("quarter_eps_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("source") != "MOPS_IXBRL":
        return None
    available = pd.to_datetime(evidence.get("available_at"), utc=True, errors="coerce")
    period_end = pd.to_datetime(evidence.get("period_end"), utc=True, errors="coerce")
    if (pd.isna(available) or pd.isna(period_end) or available < period_end
            or available > as_of or not evidence.get("revision_id")
            or row.get("source") not in {"TWSE_OPENAPI", "TPEX_OPENAPI"}
            or row.get("unit") != "thousand" or row.get("flow_basis") != "YTD"):
        return None
    if any(not row.get(key) or evidence.get(key) != row.get(key) for key in
           ("symbol", "market", "currency", "unit", "scope", "period_end", "flow_basis")):
        return None
    # Amounts are TWD thousands; tolerate only half a displayed unit of rounding.
    for field in (*BALANCE_FIELDS, *FLOW_FIELDS):
        left, right = number(row.get(field)), number(evidence.get(field))
        tolerance = 1e-9 if field == "basic_eps" else .5
        if left is None or right is None or not isclose(left, right, rel_tol=0., abs_tol=tolerance):
            return None
    return evidence if number(evidence.get("basic_eps_quarter")) is not None else None


@dataclass(frozen=True)
class QualityResult:
    status: str
    score: float | None
    confidence: float
    integrity: str
    components: dict
    coverage: dict
    metrics: dict
    warnings: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    optional: dict | None = None


def growth(current: object, previous: object) -> tuple[float | None, str]:
    c, p = number(current), number(previous)
    if c is None or p is None:
        return None, "NOT_AVAILABLE"
    if p > 0:
        return c / p - 1, "COMPARABLE"
    if c > 0:
        return None, "LOSS_TO_PROFIT"
    return None, "IMPROVING_LOSS" if c > p else "DETERIORATING_LOSS"


def _eligible(records: Sequence[Mapping], symbol: str, market: str, as_of: pd.Timestamp) -> dict:
    periods: dict[pd.Period, dict] = {}
    for raw in records:
        available = pd.to_datetime(raw.get("available_at"), utc=True, errors="coerce")
        if pd.isna(available):
            raise ValueError("MISSING_AVAILABLE_AT")
        if available > as_of:
            continue
        required = ("symbol", "market", "currency", "unit", "scope", "revision_id", "source")
        if any(not raw.get(key) for key in required):
            raise ValueError("MISSING_FILING_METADATA")
        if str(raw["symbol"]) != symbol or raw["market"] != market:
            raise ValueError("SECURITY_IDENTITY_AMBIGUOUS")
        end = pd.to_datetime(raw.get("period_end"), errors="coerce")
        if pd.isna(end) or end.date() > available.date() or end.date() > as_of.date():
            raise ValueError("INVALID_REPORTING_PERIOD")
        period = pd.Period(end, freq="Q")
        if end.date() != period.end_time.date():
            raise ValueError("NON_QUARTER_END")
        row = dict(raw, _available=available)
        old = periods.get(period)
        if old and available == old["_available"] and row != old:
            raise ValueError("AMBIGUOUS_FILING_VERSION")
        if old is None or available > old["_available"]:
            periods[period] = row
    comparable = {(r["currency"], r["unit"], r["scope"]) for r in periods.values()}
    if len(comparable) > 1:
        raise ValueError("INCOMPARABLE_FINANCIAL_STATEMENTS")
    for row in periods.values():
        if number(row.get("basic_eps_quarter")) is None:
            evidence = _quarter_eps_supplement(row, as_of)
            if evidence is not None:
                row["basic_eps_quarter"] = number(evidence["basic_eps_quarter"])
                row["quarter_eps_provenance"] = {
                    key: evidence[key] for key in ("source", "revision_id", "available_at", "period_end")
                }
    return periods


def normalize_quarters(periods: Mapping) -> dict:
    """Subtract additive flows only; EPS has a period-specific share denominator."""
    result = {}
    for period, row in sorted(periods.items()):
        basis = row.get("flow_basis")
        if basis not in {"YTD", "QUARTER"}:
            raise ValueError("UNKNOWN_FLOW_BASIS")
        previous = periods.get(period - 1, {}) if period.quarter != 1 else {}
        flows = {}
        for key in FLOW_FIELDS:
            value = number(row.get(key))
            if basis == "YTD" and period.quarter != 1:
                if key == "basic_eps":
                    # A share issue can make H1 EPS < Q1 EPS even when Q2 earns
                    # a profit. Use the reported single-quarter context or None.
                    value = number(row.get("basic_eps_quarter"))
                else:
                    prior = number(previous.get(key)) if previous.get("flow_basis") == "YTD" else None
                    value = value - prior if value is not None and prior is not None else None
            flows[key] = value
        result[period] = flows
    return result


def verification_diagnostics(records: Sequence[Mapping], *, symbol: str, as_of: object, market: str = "TWSE") -> dict:
    """Explain evidence gaps without changing evaluate_quality or granting PASS.

    Missing raw quarters need backfill; present quarters with unresolved fields
    need source/parser/period review. Optional fields never create a backfill need.
    """
    result = {"missing_periods": [], "unresolved_fields": [], "sources": [],
              "latest_available_at": None, "latest_period": None,
              "backfill_needed": False, "local_review_required": False,
              "next_action": "NORMAL_MODEL_EVALUATION"}
    try:
        timestamp = pd.to_datetime(as_of, utc=True, errors="raise")
        if pd.isna(timestamp):
            raise ValueError("INVALID_DECISION_TIME")
        periods = _eligible(records, symbol, market, timestamp)
        if not periods:
            result.update(backfill_needed=True, next_action="BACKFILL_MISSING_QUARTERS")
            result["unresolved_fields"] = ["NO_POINT_IN_TIME_FINANCIALS"]
            return result
        latest = max(periods)
        result.update(latest_period=str(latest),
                      latest_available_at=max(r["_available"] for r in periods.values()).isoformat(),
                      sources=sorted({r["source"] for r in periods.values()}))
        needed = {latest - i for i in range(8)}
        for period in tuple(needed):
            row = periods.get(period, {})
            if row.get("flow_basis") == "YTD" and period.quarter != 1:
                needed.add(period - 1)
        result["missing_periods"] = [str(p) for p in sorted(needed - periods.keys())]
        for key in BALANCE_FIELDS:
            if number(periods[latest].get(key)) is None:
                result["unresolved_fields"].append(f"{latest}:{key}")
        quarters = normalize_quarters(periods)
        for period in sorted(needed & periods.keys()):
            row = periods[period]
            for key in FLOW_FIELDS:
                if number(row.get(key)) is None:
                    result["unresolved_fields"].append(f"{period}:{key}")
                elif period in {latest - i for i in range(8)} and quarters[period][key] is None and period - 1 in periods:
                    result["unresolved_fields"].append(f"{period}:{key}:PERIOD_BASIS_REVIEW")
        result["backfill_needed"] = bool(result["missing_periods"])
        result["local_review_required"] = bool(result["unresolved_fields"])
        if result["local_review_required"]:
            result["next_action"] = "REVIEW_LOCAL_SOURCE_FIELDS"
        elif result["backfill_needed"]:
            result["next_action"] = "BACKFILL_MISSING_QUARTERS"
        return result
    except (ValueError, TypeError, AttributeError) as error:
        result.update(local_review_required=True, next_action="REVIEW_LOCAL_SOURCE_FIELDS")
        result["unresolved_fields"] = [str(error)]
        return result


def evaluate_quality(records: Sequence[Mapping], *, symbol: str, market: str, as_of: object) -> QualityResult:
    """Evaluate only versions published by decision time, never today's latest retroactively."""
    def failed(reason: str, status: str = "REJECT", integrity: str = "FAIL") -> QualityResult:
        return QualityResult(status, None, 0, integrity, {}, {}, {}, reasons=(reason,))

    timestamp = pd.to_datetime(as_of, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return failed("INVALID_DECISION_TIME")
    try:
        periods = _eligible(records, symbol, market, timestamp)
        if not periods:
            return failed("NO_POINT_IN_TIME_FINANCIALS", "DATA_INSUFFICIENT", "UNKNOWN")
        latest_period = max(periods)
        latest = periods[latest_period]
        for period, row in periods.items():
            if period == latest_period:
                continue
            assets, liabilities, equity = (number(row.get(k)) for k in ("total_assets", "total_liabilities", "total_equity"))
            if all(v is not None for v in (assets, liabilities, equity)):
                if assets <= 0 or abs(assets - liabilities - equity) / assets >= .01:
                    return failed("HISTORICAL_BALANCE_SHEET_IDENTITY_FAIL")
        balances = {key: number(latest.get(key)) for key in BALANCE_FIELDS}
        if any(value is None for value in balances.values()):
            return failed("REQUIRED_BALANCE_SHEET_FIELDS_UNRESOLVABLE")
        a, l, e = (balances[k] for k in ("total_assets", "total_liabilities", "total_equity"))
        if a <= 0 or e <= 0 or any(value < 0 for value in balances.values()):
            return failed("INVALID_BALANCE_OR_NONPOSITIVE_EQUITY")
        if abs(a - l - e) / a >= .01 or abs(a - balances["current_assets"] - balances["non_current_assets"]) / a >= .01:
            return failed("BALANCE_SHEET_IDENTITY_FAIL")
        if balances["current_liabilities"] > l:
            return failed("CURRENT_LIABILITIES_EXCEED_TOTAL")
        quarters = normalize_quarters(periods)
    except (ValueError, TypeError) as error:
        return failed(str(error))

    # Optional evidence is never required, but a confirmed current hard risk cannot be ignored.
    optional = {key: latest.get(key) if latest.get(key) is not None else "NOT_AVAILABLE" for key in OPTIONAL_FIELDS}
    if (latest.get("auditor_opinion") in {"ADVERSE", "DISCLAIMER", "QUALIFIED_MATERIAL_MISSTATEMENT"}
            or latest.get("going_concern") is True or latest.get("material_accounting_issue") is True):
        return failed("CONFIRMED_OPTIONAL_HARD_RISK")
    review_required = latest.get("audit_review_required") is True or latest.get("auditor_opinion") in {
        "UNKNOWN", "QUALIFIED", "QUALIFIED_UNKNOWN",
    }

    def ttm(field: str, offset: int = 0) -> float | None:
        values = [quarters.get(latest_period - offset - i, {}).get(field) for i in range(4)]
        return sum(values) if all(v is not None for v in values) else None

    components = {"balance_sheet": [0., 0., 30.], "profitability": [0., 0., 25.],
                  "growth": [0., 0., 25.], "return_on_capital": [0., 0., 20.]}
    metrics: dict = {f"{key}_ttm": ttm(key) for key in FLOW_FIELDS}
    # A sum of reported quarterly EPS is a screening proxy, not an IAS 33
    # annual EPS with one weighted denominator. Never sum differenced YTD EPS.
    metrics["basic_eps_ttm_basis"] = "SUM_REPORTED_QUARTER_EPS" if metrics["basic_eps_ttm"] is not None else "UNAVAILABLE"
    metrics["quarter_eps_supplements"] = {
        str(period): row["quarter_eps_provenance"] for period, row in periods.items()
        if "quarter_eps_provenance" in row
    }
    if latest_period.quarter == 4 and latest.get("flow_basis") == "YTD":
        metrics["basic_eps_ttm"] = number(latest.get("basic_eps"))
        metrics["basic_eps_ttm_basis"] = "REPORTED_FULL_YEAR" if metrics["basic_eps_ttm"] is not None else "UNAVAILABLE"

    def award(group: str, weight: float, fraction: float | None) -> None:
        if fraction is not None:
            components[group][0] += weight * fraction
            components[group][1] += weight

    def bands(value: float | None, limits: tuple, fractions: tuple, *, ascending: bool = False):
        if value is None:
            return None
        for limit, fraction in zip(limits, fractions):
            if (value <= limit) if ascending else (value >= limit):
                return fraction
        return 0.

    prior = periods.get(latest_period - 4, {})
    debt = ratio(l, a)
    current_ratio = ratio(balances["current_assets"], balances["current_liabilities"])
    rev_growth, _ = growth(ttm("revenue"), ttm("revenue", 4))
    liability_growth, _ = growth(l, prior.get("total_liabilities"))
    divergence = liability_growth - rev_growth if liability_growth is not None and rev_growth is not None else None
    metrics.update(debt_ratio=debt, current_ratio=current_ratio, equity_ratio=e / a,
                   revenue_growth=rev_growth, liability_growth=liability_growth, liability_divergence=divergence)
    award("balance_sheet", 8, 1.)
    award("balance_sheet", 8, bands(debt, (.4, .5, .6, .7), (1, .75, .5, .25), ascending=True))
    award("balance_sheet", 7, bands(current_ratio, (1.5, 1.2, 1), (1, 5/7, 2/7)))
    award("balance_sheet", 7, bands(divergence, (0, .1, .2), (1, 5/7, 2/7), ascending=True))

    margin_fields = (("gross_margin", 8), ("operating_margin", 9), ("net_margin", 8))
    for name, weight in margin_fields:
        values = []
        for i in reversed(range(8)):
            q = quarters.get(latest_period - i, {})
            numerator = q.get("operating_income" if name == "operating_margin" else "net_income")
            if name == "gross_margin":
                r, c = q.get("revenue"), q.get("cost_of_revenue")
                numerator = r - c if r is not None and c is not None else None
            values.append(ratio(numerator, q.get("revenue")))
        now = values[-1]
        metrics[name] = now
        # Versioned provisional mapping: positive level, history, stability, peer rank (25% each).
        award("profitability", weight / 4, float(now > 0) if now is not None else None)
        history = values[:-1]
        history_rank = sum((x < now) + .5 * (x == now) for x in history) / 7 if now is not None and all(x is not None for x in history) else None
        award("profitability", weight / 4, history_rank if now is not None and now > 0 else (0. if history_rank is not None else None))
        recent = values[-4:]
        drawdown = margin_drawdown(recent)
        metrics[f"{name}_drawdown_4q"] = drawdown
        stable = float(now > 0 and drawdown <= .05 + 1e-12 and now >= recent[0] - .01 - 1e-12) if drawdown is not None else None
        award("profitability", weight / 4, stable)
        peer = number(latest.get(f"{name}_subsector_percentile"))
        award("profitability", weight / 4, peer if peer is not None and 0 <= peer <= 1 and now is not None else None)

    for field, weight in (("revenue", 8), ("operating_income", 7), ("net_income", 5)):
        value, state = growth(ttm(field), ttm(field, 4))
        metrics[f"{field}_growth"] = value
        metrics[f"{field}_growth_state"] = state
        fraction = bands(value, (.2, .1, 0, -.1), (1, .8, .6, .2))
        if state != "COMPARABLE":
            fraction = {"LOSS_TO_PROFIT": .6, "IMPROVING_LOSS": .2, "DETERIORATING_LOSS": 0}.get(state)
        award("growth", weight, fraction)
    eps = [quarters.get(latest_period - i, {}).get("basic_eps") for i in range(8)]
    positive_eps = sum(v > 0 for v in eps) if all(v is not None for v in eps) else None
    metrics["positive_eps_quarters"] = positive_eps
    award("growth", 5, bands(positive_eps, (8, 6, 4), (1, .8, .4)))
    for metric, field, cutoffs in (("roe", "total_equity", (.2, .15, .1, .05)),
                                  ("roa", "total_assets", (.1, .07, .05, .02))):
        previous = number(prior.get(field))
        average = (balances[field] + previous) / 2 if previous is not None and previous > 0 else None
        metrics[metric] = ratio(ttm("net_income"), average)
        award("return_on_capital", 10, bands(metrics[metric], cutoffs, (1, .8, .5, .2)))

    available = sum(c[1] for c in components.values())
    score = 100 * sum(c[0] for c in components.values()) / available if available else None
    coverage = {name: 100 * c[1] / c[2] for name, c in components.items()}
    age = (timestamp.date() - latest_period.end_time.date()).days
    freshness = 100 if age <= 180 else 70 if age <= 275 else 0
    reliability = 100 if all(r["source"] == "MOPS_IXBRL" for r in periods.values()) else 95
    official = all(r["source"] in OFFICIAL_SOURCES for r in periods.values())
    if not official:
        reliability = 0
    confidence = .4 * available + .25 * freshness + 25 + .1 * reliability
    enough = all(coverage[k] >= limit for k, limit in (("balance_sheet", 70), ("profitability", 70), ("growth", 60), ("return_on_capital", 60)))
    if confidence < 50 or score is None:
        status = "DATA_INSUFFICIENT"
    elif score < 60:
        status = "REJECT"
    elif score < MINIMUM_ENTRY_QUALITY:
        status = "WATCH_QUALITY"
    elif confidence < 70 or not enough or not official or age > 275:
        status = "PROVISIONAL"
    else:
        status = "PASS"
    warnings = []
    if positive_eps is None:
        warnings.append("QUARTER_EPS_EVIDENCE_INCOMPLETE")
    if divergence is not None and divergence > .1:
        warnings.append("LIABILITY_GROWTH_DIVERGENCE")
    if not enough:
        warnings.append("INSUFFICIENT_COMPONENT_COVERAGE")
    if age > 180:
        warnings.append("STALE_FINANCIALS")
    if not official:
        warnings.append("UNTRUSTED_FINANCIAL_SOURCE")
    reasons = ()
    if review_required:
        reasons = ("AUDIT_REVIEW_REQUIRED",)
        if status != "REJECT":
            status = "DATA_INSUFFICIENT"
    return QualityResult(status, score, confidence, "PASS",
                         {k: c[0] for k, c in components.items()}, coverage, metrics,
                         tuple(warnings), reasons, optional)
