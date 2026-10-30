"""Point-in-time official-source core quality for LES v6 (no I/O)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite

import pandas as pd

from domain.fundamental_models import FundamentalSnapshot
from domain.strategies.low_entry_v6_ttm import comparative_ttm
from domain.strategies.low_entry_v6_quality_evidence import (
    validate_fields, validated_proof_value, verification_summary,
)
from domain.strategies.low_entry_v6_reported_period import reported_period_inputs


CORE_FLOW_FIELDS = ("revenue", "cost_of_revenue", "operating_income", "net_income", "basic_eps")
CAPITAL_FLOW_FIELDS = ("cfo", "capex", "income_tax", "pretax_income")
FLOW_FIELDS = CORE_FLOW_FIELDS + CAPITAL_FLOW_FIELDS
CAPITAL_BALANCE_FIELDS = ("cash", "interest_bearing_debt")
QUALITY_POLICY_VERSION = "2026-10-04-v6-reported-period-quality-v7"
QUALITY_PROFITABILITY_BASIS = "REPORTED_SAME_PERIOD_MARGIN"
QUALITY_PROFITABILITY_WINDOW = 2  # prior-year same duration and current, not independent samples
QUALITY_DC_WEIGHTS = {"coverage": .40, "freshness": .25, "verification": .25, "reliability": .10}
QUALITY_BUDGETS = {"balance_sheet": 25., "profitability": 20., "growth": 20.,
                   "return_on_capital": 20., "cash_flow": 15.}
QUALITY_COVERAGE_MINIMA = {"balance_sheet": 70., "profitability": 60., "growth": 50.,
                          "return_on_capital": 50., "cash_flow": 60.}
QUALITY_MINIMUM_AVAILABLE_WEIGHT = 75.
QUALITY_MINIMUM_CONFIDENCE = 65.
QUALITY_MISSING_PENALTY_RATE = .25
QUALITY_REQUIRED_REPORTED_FIELDS = ("revenue", "cost_of_revenue", "operating_income", "net_income", "cfo")
DIAGNOSTIC_TTM_FIELDS = QUALITY_REQUIRED_REPORTED_FIELDS
QUALITY_REQUIRED_TTM_FIELDS = DIAGNOSTIC_TTM_FIELDS  # compatibility for diagnostic inventory only
# Fixed policy for every stock, not a per-stock available-data denominator.
# Removed scores remain source-validated diagnostics, never assumed PASS/zero.
QUALITY_PROFITABILITY_PARTS = {"positive": 1 / 3, "history": 1 / 3, "stability": 1 / 3}
QUALITY_GROWTH_WEIGHTS = {"revenue": 7.5, "operating_income": 7.5, "net_income": 5.}
QUALITY_RETURN_WEIGHTS = {"roe": 10., "roa": 10.}
QUALITY_DIAGNOSTIC_ONLY = (
    "positive_eps_quarters", "roic", "gross_margin_peer", "operating_margin_peer", "net_margin_peer",
    "ttm", "liability_divergence",
)
QUALITY_SCORING_POLICY = {
    "flow_basis": "UNIFORM_REPORTED_YTD_ANNUAL_OR_EXPLICIT_QUARTER",
    "comparison_basis": "SELECTED_ORIGINAL_SAME_DURATION",
    "return_basis": "REPORTED_NET_INCOME_OVER_OPENING_CLOSING_AVERAGE",
    "return_threshold_policy": "ANNUAL_THRESHOLDS_TIMES_REPORTED_DURATION_FRACTION_NOT_ANNUALIZED_VALUES",
    "balance_weights": {"identity": 6., "debt_ratio": 10., "current_ratio": 9.},
    "profitability_parts": QUALITY_PROFITABILITY_PARTS,
    "growth_weights": QUALITY_GROWTH_WEIGHTS,
    "return_weights": QUALITY_RETURN_WEIGHTS,
    "diagnostic_only": list(QUALITY_DIAGNOSTIC_ONLY),
}
# Explicit opt-in for v7 R4. The v6/R3 constants and default calculation remain
# frozen: old results must not inherit a new coverage or ratio interpretation.
QUALITY_R4_POLICY_VERSION = "2026-10-13-v7-reported-limited-gaps-r4"
QUALITY_R4_COVERAGE_MINIMA = {name: 0. for name in QUALITY_BUDGETS}
QUALITY_R4_SCORING_POLICY = dict(QUALITY_SCORING_POLICY,
    optional_coverage_policy="TOTAL_75_NO_OPTIONAL_DOMAIN_FLOORS",
    minimum_available_weight=75., minimum_confidence=65., minimum_score=65.,
    missing_penalty_rate=.25, observed_reject_floor=60.,
    known_nonpositive_denominator_policy="NOT_MEANINGFUL_ZERO_POINTS_COVERED_WITH_VERIFIED_DEPENDENCIES")


def quality_scoring_rules(policy_version=None):
    """Resolve only registered native rules; never silently fall back."""
    if policy_version in (None, QUALITY_POLICY_VERSION):
        return QUALITY_POLICY_VERSION, QUALITY_SCORING_POLICY, QUALITY_COVERAGE_MINIMA
    if policy_version == QUALITY_R4_POLICY_VERSION:
        return QUALITY_R4_POLICY_VERSION, QUALITY_R4_SCORING_POLICY, QUALITY_R4_COVERAGE_MINIMA
    raise ValueError("UNREGISTERED_NATIVE_QUALITY_POLICY")


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
MINIMUM_ENTRY_QUALITY = 65.


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
        "scope_basis": "LEGACY_DEFAULT_NOT_SOURCE_REPORTED",
        "period_end": period.end_time.date().isoformat(), "flow_basis": "YTD",
        "flow_derivation_policy": "SAME_DOCUMENT_ONLY",
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


QUALITY_SOURCE_SELECTION_POLICY = "CANONICAL_ORIGINAL_OVER_UNVERIFIED_BATCH/v2"


def combine_filing_sources(mops: Sequence[Mapping], batch: Sequence[Mapping], *, as_of: object) -> list[dict]:
    """Prefer a complete canonical original to an unverified batch scope.

    Never splice cash flows/balances or subtract YTD EPS. The latest eligible
    MOPS revision must tie to the selected batch before its quarter EPS is used.
    The consumer rechecks the supplement's timestamp at its own decision cutoff.
    Selection never depends on the sign of CFO, quality score or completeness.
    Preserve batch differences for diagnosis, not as substitute source values.
    Explicitly scoped batches retain the existing full-core tie-out contract.
    """
    cutoff = pd.to_datetime(as_of, utc=True, errors="coerce")
    eligible = []
    for record in mops:
        available = pd.to_datetime(record.get("available_at"), utc=True, errors="coerce")
        if pd.notna(cutoff) and pd.notna(available) and available <= cutoff:
            eligible.append((available, record))
    result = []
    for record in batch:
        available = pd.to_datetime(record.get("available_at"), utc=True, errors="coerce")
        if pd.notna(cutoff) and pd.notna(available) and available > cutoff:
            # A later observed batch must not erase an eligible original MOPS
            # version before the consumer applies its PIT selection.
            continue
        row = dict(record)
        candidates = [(date, item) for date, item in eligible
                      if all(item.get(key) == row.get(key) for key in ("symbol", "market", "period_end"))]
        if candidates:
            latest = max(date for date, _ in candidates)
            versions = [item for date, item in candidates if date == latest]
            # Ambiguous same-time revisions cannot supply an enrichment.
            if (row.get("scope_basis") == "LEGACY_DEFAULT_NOT_SOURCE_REPORTED"
                    and any(item.get("filing_schema") and item.get("field_evidence") for item in versions)
                    and not all(item == versions[0] for item in versions)):
                # Do not hide an original-version conflict behind an incomplete
                # summary. _eligible must reject the ambiguous originals.
                result.extend(dict(item) for item in versions)
                continue
            if all(item == versions[0] for item in versions):
                original = versions[0]
                canonical = (original.get("filing_schema")
                             and original.get("flow_derivation_policy") == "SAME_DOCUMENT_ONLY"
                             and original.get("content_sha256")
                             and original.get("field_evidence"))
                if (row.get("scope_basis") == "LEGACY_DEFAULT_NOT_SOURCE_REPORTED"
                        and original.get("source") == "MOPS_IXBRL"
                        and original.get("scope") in {"individual", "consolidated"}
                        and (canonical or original["scope"] != row.get("scope"))):
                    # Use the COMPLETE original MOPS record, never relabel
                    # batch values or mix individual EPS/consolidated flows.
                    row = dict(original, source_selection={
                        "policy": QUALITY_SOURCE_SELECTION_POLICY,
                        "excluded_batch_revision_id": row.get("revision_id"),
                        "excluded_batch_available_at": row.get("available_at"),
                        "batch_differences": {key: {"batch": row.get(key), "original": original.get(key)}
                                              for key in (*BALANCE_FIELDS, *CORE_FLOW_FIELDS)
                                              if row.get(key) != original.get(key)},
                    })
                else:
                    row["quarter_eps_evidence"] = dict(original)
        result.append(row)
    identities = {(row.get("symbol"), row.get("market"), row["period_end"]) for row in result}
    return [dict(row) for row in mops
            if (row.get("symbol"), row.get("market"), row["period_end"]) not in identities] + result


def _quarter_eps_supplement(row: Mapping, as_of: pd.Timestamp, *, require_eps: bool = True) -> Mapping | None:
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
    # Cash-flow enrichment ties the scored financial fields, not diagnostic EPS.
    # EPS enrichment still needs its own exact EPS tie; neither path may relax
    # the balance-sheet, income-statement, scope or version checks.
    fields = (*BALANCE_FIELDS, *(field for field in CORE_FLOW_FIELDS
                                if require_eps or field != "basic_eps"))
    # Amounts are TWD thousands; tolerate only half a displayed unit of rounding.
    for field in fields:
        left, right = number(row.get(field)), number(evidence.get(field))
        tolerance = 1e-9 if field == "basic_eps" else .5
        if left is None or right is None or not isclose(left, right, rel_tol=0., abs_tol=tolerance):
            return None
    return evidence if not require_eps or number(evidence.get("basic_eps_quarter")) is not None else None


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


def quality_evidence_complete(quality: QualityResult, *, policy_version=None) -> bool:
    """Shared evidence floor for scoring, rendering and execution validation.

    Missing soft metrics can be penalized without becoming observed values.
    The new ledger must agree with coverage and cannot conceal required gaps.
    Legacy manually supplied quality contracts retain the basic coverage check.
    """
    version, scoring_policy, coverage_minima = quality_scoring_rules(policy_version)
    confidence = number(quality.confidence)
    if quality.integrity != "PASS" or confidence is None or not QUALITY_MINIMUM_CONFIDENCE <= confidence <= 100:
        return False
    if not isinstance(quality.coverage, Mapping) or not isinstance(quality.metrics, Mapping):
        return False
    covered = 0.
    coverage_values = {}
    for group, minimum in coverage_minima.items():
        value = number(quality.coverage.get(group))
        if value is None or not minimum <= value <= 100:
            return False
        coverage_values[group] = value
        covered += QUALITY_BUDGETS[group] * value / 100
    if covered + 1e-9 < QUALITY_MINIMUM_AVAILABLE_WEIGHT:
        return False
    metrics = quality.metrics
    if metrics.get("quality_policy_version") not in ({None, version} if version == QUALITY_POLICY_VERSION else {version}):
        # Retain the metadata-free legacy contract, not a previous version's
        # penalty ledger masquerading as this version's execution evidence.
        return False
    if metrics.get("quality_policy_version") == version:
        if metrics.get("quality_scoring_policy") != scoring_policy:
            return False
        reported = metrics.get("quality_reported_inputs")
        ledger = metrics.get("quality_field_checks")
        if (not isinstance(reported, Mapping) or not isinstance(ledger, Mapping)
                or reported.get("basis") not in {"YTD", "ANNUAL", "QUARTER"}
                or metrics.get("quality_reported_basis") != reported["basis"]):
            return False
        for field in QUALITY_REQUIRED_REPORTED_FIELDS:
            key = reported.get("current_keys", {}).get(field)
            check = ledger.get(key, {})
            value = number(metrics.get(field + "_reported"))
            if (value is None or number(reported.get("current", {}).get(field)) != value
                    or number(check.get("value")) != value
                    or check.get("status") not in {"VERIFIED_REPORTED", "VERIFIED_DERIVED"}):
                return False
        raw = number(metrics.get("quality_raw_score"))
        penalty = number(metrics.get("quality_penalty"))
        available = number(metrics.get("quality_available_weight"))
        score = number(quality.score)
        components = quality.components
        if not isinstance(components, Mapping) or set(components) != set(QUALITY_BUDGETS):
            return False
        earned = [number(components[group]) for group in QUALITY_BUDGETS]
        if any(value is None or not 0 <= value <= QUALITY_BUDGETS[group]
               for group, value in zip(QUALITY_BUDGETS, earned)):
            return False
        expected_penalties = [dict(group=group, reason="MISSING_OR_INCOMPARABLE_METRICS",
            missing_weight=budget * (1. - coverage_values[group] / 100),
            points=budget * (1. - coverage_values[group] / 100) * QUALITY_MISSING_PENALTY_RATE)
            for group, budget in QUALITY_BUDGETS.items() if coverage_values[group] < 100]
        actual_penalties = metrics.get("quality_penalties")
        if not isinstance(actual_penalties, list) or len(actual_penalties) != len(expected_penalties):
            return False
        for actual, expected in zip(actual_penalties, expected_penalties):
            if (not isinstance(actual, Mapping) or actual.get("group") != expected["group"]
                    or actual.get("reason") != expected["reason"]
                    or any(number(actual.get(key)) is None or not isclose(number(actual[key]), expected[key], abs_tol=1e-8)
                           for key in ("missing_weight", "points"))):
                return False
        if (raw is None or penalty is None or available is None or score is None
                or metrics.get("quality_required_missing") != []
                or any(number(metrics.get(key + "_reported")) is None for key in QUALITY_REQUIRED_REPORTED_FIELDS)
                or not 0 <= raw <= available + 1e-8 or not 0 <= available <= 100 + 1e-8
                or not isclose(raw, sum(earned), abs_tol=1e-8)
                or not isclose(available, covered, abs_tol=1e-8)
                or not isclose(penalty, (100. - available) * QUALITY_MISSING_PENALTY_RATE, abs_tol=1e-8)
                or not isclose(score, max(0., raw - penalty), abs_tol=1e-8)
                or metrics.get("quality_final_score") != quality.score):
            return False
        dc = metrics.get("quality_dc_inputs")
        verification = metrics.get("quality_verification")
        ledger = metrics.get("quality_field_checks")
        if (not isinstance(dc, Mapping) or set(dc) != set(QUALITY_DC_WEIGHTS)
                or not isinstance(verification, Mapping) or not isinstance(ledger, Mapping)
                or metrics.get("quality_dc_weights") != QUALITY_DC_WEIGHTS
                or any(number(v) is None or not 0 <= number(v) <= 100 for v in dc.values())
                or not isinstance(verification.get("dependencies"), list)):
            return False
        expected_verification = verification_summary(ledger, verification["dependencies"])
        if (verification != expected_verification
                or not isclose(number(dc["coverage"]), covered, abs_tol=1e-8)
                or not isclose(number(dc["verification"]), verification["score"], abs_tol=1e-8)
                or not isclose(confidence, sum(QUALITY_DC_WEIGHTS[k] * number(dc[k]) for k in dc), abs_tol=1e-8)):
            return False
    return True


def growth(current: object, previous: object) -> tuple[float | None, str]:
    c, p = number(current), number(previous)
    if c is None or p is None:
        return None, "NOT_AVAILABLE"
    if p > 0:
        return c / p - 1, "COMPARABLE"
    if c > 0:
        return None, "LOSS_TO_PROFIT"
    return None, "IMPROVING_LOSS" if c > p else "DETERIORATING_LOSS"


def required_quality_periods(periods: Mapping) -> set[pd.Period]:
    """Legacy-named diagnostic inventory, not reported scoring requirements.

    Eight quarters plus TTM/YTD dependencies remain source-validated for
    diagnostics. The scorer separately uses current-original reported inputs.
    """
    if not periods:
        return set()
    latest = max(periods)
    needed = {latest - i for i in range(8)}
    # Prior-year TTM growth has an additional comparative interim dependency.
    # This is an input to the verified bridge, not permission to mix five years.
    if latest.quarter != 4 and periods[latest].get("comparative_ytd_field_evidence"):
        needed.add(latest - 8)
    for period in tuple(needed):
        if (periods.get(period, {}).get("flow_basis") == "YTD" and period.quarter != 1
                and not periods[period].get("flow_derivation_policy")):
            needed.add(period - 1)
    return needed


def _eligible(records: Sequence[Mapping], symbol: str, market: str, as_of: pd.Timestamp,
              *, excluded_inputs: list | None = None, field_ledger: dict | None = None) -> dict:
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
    # Retain version selection first, then bound the diagnostic inventory.
    # Excluded incompatible history remains a gap, never a substitute source
    # for the current original's reported-period scoring dependencies.
    needed = required_quality_periods(periods)
    periods = {period: row for period, row in periods.items() if period in needed}
    if excluded_inputs is not None and periods:
        latest_period = max(periods)
        latest = periods[latest_period]
        # Do not mix statements or fall back to an older revision. Incompatible
        # historical inputs become explicit gaps, not company financial risks.
        for period, row in tuple(periods.items()):
            if period == latest_period:
                continue
            reason = None
            if any(row[key] != latest[key] for key in ("currency", "unit", "scope")):
                reason = "INCOMPARABLE_FINANCIAL_STATEMENTS"
            elif row.get("flow_basis") not in {"QUARTER", "YTD"}:
                reason = "UNKNOWN_FLOW_BASIS"
            if reason:
                excluded_inputs.append(dict(period=str(period), reason=reason,
                    **{key: row.get(key) for key in ("revision_id", "source", "source_url",
                        "content_sha256", "available_at", "currency", "unit", "scope")}))
                del periods[period]
    comparable = {(r["currency"], r["unit"], r["scope"]) for r in periods.values()}
    if len(comparable) > 1:
        raise ValueError("INCOMPARABLE_FINANCIAL_STATEMENTS")
    # Validate only versions actually needed by the calculation. An unrelated
    # old currency/unit/scope must not poison today's eight-quarter window.
    for row in periods.values():
        if row["scope"] not in {"consolidated", "individual"}:
            raise ValueError("UNVERIFIED_FILING_SCOPE")
        if row["currency"] != "TWD" or row["unit"] != "thousand":
            raise ValueError("UNSUPPORTED_CANONICAL_CURRENCY_OR_UNIT")
    for row in periods.values():
        supplement = _quarter_eps_supplement(row, as_of, require_eps=False)
        if supplement is not None and supplement.get("flow_derivation_policy") == "SAME_DOCUMENT_ONLY":
            # The complete source must tie to the selected batch, not just EPS.
            # Quarter amounts remain bound to that original document/version.
            for key in ("quarter_fields", "quarter_field_evidence", "prior_ytd_fields", "prior_ytd_field_evidence",
                        "comparative_ytd_fields", "comparative_ytd_field_evidence",
                        "comparative_year_end_fields", "comparative_year_end_field_evidence",
                        "statement_ledger", "balance_ledger", "comparative_statement_ledger",
                        "comparative_year_end_ledger", "accounting_basis_evidence", "correction_description",
                        "field_impact_evidence"):
                row[key] = supplement.get(key, {})
            row["quarter_flow_source"] = {key: supplement.get(key) for key in
                ("symbol", "market", "period_end", "content_sha256", "revision_id", "source", "source_url",
                 "available_at", "published_at", "fetched_at", "availability_source", "scope", "currency", "unit")}
            # Batch statements do not supply cash flows/debt. Only enrich from
            # the complete, core-tied original selected at this same cutoff.
            row["field_evidence"] = dict(row.get("field_evidence", {}))
            for key in (*CAPITAL_FLOW_FIELDS, *CAPITAL_BALANCE_FIELDS):
                row[key] = supplement.get(key)
                row["field_evidence"][key] = supplement.get("field_evidence", {}).get(key, {})
            row["_available"] = max(row["_available"], pd.to_datetime(supplement["available_at"], utc=True))
        if number(row.get("basic_eps_quarter")) is None:
            evidence = _quarter_eps_supplement(row, as_of)
            if evidence is not None:
                row["basic_eps_quarter"] = number(evidence["basic_eps_quarter"])
                row["quarter_eps_provenance"] = {
                    key: evidence[key] for key in ("source", "revision_id", "available_at", "period_end")
                }
    selected, ledger = validate_fields(periods, fields=(*FLOW_FIELDS, *BALANCE_FIELDS, *CAPITAL_BALANCE_FIELDS),
        instant_fields=(*BALANCE_FIELDS, *CAPITAL_BALANCE_FIELDS), cutoff=as_of)
    if field_ledger is not None:
        field_ledger.update(ledger)
    return selected


def select_quality_inputs(records, *, symbol, market, cutoff, preprocessing=None,
                          field_ledger=None, excluded_inputs=None):
    """Select and validate once for scoring and audit at the exact cutoff.

    The optional work-local hook only reuses immutable preprocessing, never a
    QualityResult. Standalone callers retain the original selection algorithm.
    Return fresh outer containers: reported comparisons extend the local ledger.
    """
    def compute():
        excluded, ledger = [], {}
        try:
            periods = _eligible(records, symbol, market, cutoff,
                                excluded_inputs=excluded, field_ledger=ledger)
        except (ValueError, TypeError, AttributeError):
            # Preserve partial failure attribution from the original validator.
            # Failed computations are not inserted into the preprocessing cache.
            if field_ledger is not None:
                field_ledger.update(ledger)
            if excluded_inputs is not None:
                excluded_inputs.extend(excluded)
            raise
        return periods, ledger, excluded
    value = _quality_preprocess(preprocessing, "selection", compute, records=records,
                                symbol=symbol, market=market, cutoff=cutoff)
    return dict(value[0]), dict(value[1]), list(value[2])


def _quality_preprocess(preprocessing, kind, compute, *, records, symbol, market, cutoff):
    return (preprocessing(kind, compute, records=records, symbol=symbol, market=market,
                          cutoff=cutoff, policy=QUALITY_POLICY_VERSION)
            if preprocessing is not None else compute())


def _reported_inputs(periods, ledger, *, prepare, cutoff):
    def compute():
        local = dict(ledger)
        reported = reported_period_inputs(periods, cutoff=cutoff, ledger=local)
        return reported, local
    reported, checks = prepare("reported", compute)
    ledger.update(checks)
    return reported


def normalize_quarters_with_diagnostics(periods: Mapping) -> tuple[dict, dict]:
    """One normalization core shared by Quality and gap audit.

    Canonical sources never subtract unverified revisions from other documents.
    Legacy metadata-free arithmetic inputs remain readable, but cannot establish
    publication/Quality eligibility. Balances are deliberately absent here.
    """
    result, diagnostics = {}, {}
    for period, row in sorted(periods.items()):
        basis = row.get("flow_basis")
        if basis not in {"YTD", "QUARTER"}:
            raise ValueError("UNKNOWN_FLOW_BASIS")
        previous = periods.get(period - 1, {}) if period.quarter != 1 else {}
        flows, trace = {}, {}
        origin = row.get("quarter_flow_source", row)

        def same_document(field, group):
            evidence = row.get(group + "_field_evidence", {}).get(field, {})
            value = number(row.get(group + "_fields", {}).get(field))
            expected_start = period.start_time.date().isoformat() if group == "quarter" else f"{period.year}-01-01"
            expected_end = (period if group == "quarter" else period - 1).end_time.date().isoformat()
            if (value is None or evidence.get("status") != "RESOLVED"
                    or not origin.get("content_sha256")
                    or evidence.get("content_sha256") != origin["content_sha256"]
                    or evidence.get("revision_id") != origin.get("revision_id")
                    or any(evidence.get(k) != origin.get(k) for k in ("scope", "currency", "available_at"))
                    or evidence.get("unit") != ("TWD/share" if field == "basic_eps" else "thousand")
                    or evidence.get("start_date") != expected_start or evidence.get("end_date") != expected_end):
                return None, evidence
            parsed = validated_proof_value(evidence, origin, field, start=expected_start, end=expected_end)
            if parsed is None or not isclose(parsed, value, rel_tol=0., abs_tol=1e-9 if field == "basic_eps" else .5):
                return None, dict(evidence, status="UNKNOWN", category="PENDING_VERIFICATION",
                                  reason="FIELD_PROOF_BINDING_OR_VALUE_MISMATCH")
            return value, evidence

        for key in FLOW_FIELDS:
            value = number(row.get(key))
            detail = dict(status="RESOLVED" if value is not None else "UNKNOWN",
                          category=None if value is not None else row.get("field_evidence", {}).get(key, {}).get("category", "MISSING_DATA"),
                          reason="REPORTED_QUARTER" if basis == "QUARTER" or period.quarter == 1 else "YTD_REQUIRES_NORMALIZATION",
                          inputs=[row.get("field_evidence", {}).get(key, {})])
            if basis == "YTD" and period.quarter != 1:
                reported, evidence = same_document(key, "quarter")
                if reported is not None:
                    value = reported
                    detail.update(status="RESOLVED", category=None, reason="REPORTED_SINGLE_QUARTER", inputs=[evidence])
                elif key == "basic_eps":
                    # A share issue can make H1 EPS < Q1 EPS even when Q2 earns
                    # a profit. Use the reported single-quarter context or None.
                    value = number(row.get("basic_eps_quarter"))
                    # Canonical failed/conflicting quarter contexts cannot fall
                    # back to an unbound EPS number; old EPS contract is readable.
                    if "quarter_field_evidence" in row:
                        value = None
                    detail.update(status="RESOLVED" if value is not None else "UNKNOWN",
                        category=None if value is not None else "PENDING_VERIFICATION",
                        reason="REPORTED_SINGLE_QUARTER_EPS" if value is not None else "EPS_NOT_ADDITIVE", inputs=[evidence])
                else:
                    prior, prior_evidence = same_document(key, "prior_ytd")
                    own_evidence = row.get("field_evidence", {}).get(key, {})
                    if row.get("quarter_flow_source"):
                        own_evidence = row["quarter_eps_evidence"].get("field_evidence", {}).get(key, {})
                    own_bound = (own_evidence.get("status") == "RESOLVED" and
                                 own_evidence.get("content_sha256") == origin.get("content_sha256") and
                                 own_evidence.get("revision_id") == origin.get("revision_id") and
                                 own_evidence.get("scope") == origin.get("scope") and
                                 own_evidence.get("currency") == origin.get("currency") and
                                 own_evidence.get("available_at") == origin.get("available_at") and
                                 own_evidence.get("unit") == "thousand" and
                                 own_evidence.get("start_date") == f"{period.year}-01-01" and
                                 own_evidence.get("end_date") == period.end_time.date().isoformat())
                    reported_invalid = evidence.get("status") == "UNKNOWN" and evidence.get("reason") not in {
                        "NO_EXACT_AGGREGATE_DURATION", "CONCEPT_NOT_REPORTED"}
                    if reported_invalid:
                        value = None
                        detail.update(status="UNKNOWN", category=evidence.get("category") or "PENDING_VERIFICATION",
                                      reason=evidence["reason"], inputs=[evidence])
                    elif value is None and own_evidence.get("category"):
                        detail.update(status="UNKNOWN", category=own_evidence["category"], reason=own_evidence["reason"], inputs=[own_evidence])
                    elif prior is not None and value is not None and own_bound:
                        value -= prior
                        detail.update(status="RESOLVED", category=None, reason="SAME_DOCUMENT_YTD_DIFFERENCE",
                                      inputs=[own_evidence, prior_evidence])
                    elif row.get("flow_derivation_policy") or row.get("source"):
                        value = None
                        detail.update(status="UNKNOWN", category="PENDING_VERIFICATION", reason="REVISION_BASIS_UNVERIFIED",
                                      inputs=[own_evidence, prior_evidence])
                    else:
                        compatible = all(row.get(k) == previous.get(k) for k in ("currency", "unit", "scope"))
                        prior = number(previous.get(key)) if previous.get("flow_basis") == "YTD" and compatible else None
                        value = value - prior if value is not None and prior is not None else None
                        detail.update(status="RESOLVED" if value is not None else "UNKNOWN",
                            category=None if value is not None else "INCOMPARABLE", reason="LEGACY_UNVERIFIED_YTD_DIFFERENCE" if value is not None else "MISSING_OR_INCOMPARABLE_YTD_PREDECESSOR")
            flows[key] = value
            trace[key] = detail
        result[period] = flows
        diagnostics[period] = trace
    return result, diagnostics


def normalize_quarters(periods: Mapping) -> dict:
    return normalize_quarters_with_diagnostics(periods)[0]


def reported_annual_flow(row: Mapping, period: pd.Period, field: str) -> float | None:
    """A verified 12-month original is already TTM at year-end.

    This does NOT reconstruct a Q4 amount, bridge two documents, or supply
    missing quarterly margins/EPS. Batch values need the existing full-core
    tie to an original before using its field provenance.
    """
    if period.quarter != 4 or row.get("flow_basis") != "YTD" or field == "basic_eps":
        return None
    origin = row.get("quarter_flow_source", row)
    original = row.get("quarter_eps_evidence", row) if row.get("quarter_flow_source") else row
    evidence = original.get("field_evidence", {}).get(field, {})
    value = number(row.get(field))
    if (value is None or origin.get("source") not in OFFICIAL_SOURCES
            or not origin.get("content_sha256") or not origin.get("revision_id")
            or evidence.get("status") != "RESOLVED"
            or any(evidence.get(key) != origin.get(key) for key in
                   ("content_sha256", "revision_id", "scope", "currency", "available_at"))
            or evidence.get("unit") != "thousand" or evidence.get("basis") != "YTD"
            or evidence.get("start_date") != f"{period.year}-01-01"
            or evidence.get("end_date") != period.end_time.date().isoformat()):
        return None
    parsed = validated_proof_value(evidence, origin, field, start=f"{period.year}-01-01",
                                   end=period.end_time.date().isoformat())
    return value if parsed is not None and isclose(parsed, value, rel_tol=0., abs_tol=.5) else None


def financial_field_checks(periods: Mapping, needed: set, *, symbol: str,
                           selection_error: str | None = None, normalization=None) -> list[dict]:
    """Diagnostic inventory using the shared safe quarter-normalization core."""
    try:
        quarters, traces = (normalization() if normalization is not None
                            else normalize_quarters_with_diagnostics(periods))
    except ValueError:
        quarters, traces = {}, {}
    latest = max(needed) if needed else None
    checks = []
    for period in sorted(needed):
        row = periods.get(period, {})
        for field in (*FLOW_FIELDS, *BALANCE_FIELDS, *CAPITAL_BALANCE_FIELDS):
            evidence = row.get("field_evidence", {}).get(field, {})
            trace = traces.get(period, {}).get(field, {}) if field in FLOW_FIELDS else evidence
            value = number(quarters.get(period, {}).get(field)) if field in FLOW_FIELDS else number(row.get(field))
            if not row:
                category, reason = "MISSING_DATA", "NO_ELIGIBLE_FILING_FOR_PERIOD"
            elif selection_error:
                category, reason = "INCOMPARABLE" if selection_error in {"INCOMPARABLE_FINANCIAL_STATEMENTS", "UNSUPPORTED_CANONICAL_CURRENCY_OR_UNIT"} else "PENDING_VERIFICATION", selection_error
            elif value is not None:
                category, reason = None, trace.get("reason", "REPORTED_VALUE")
            else:
                category = trace.get("category") or evidence.get("category") or "MISSING_DATA"
                reason = trace.get("reason") or evidence.get("reason") or "FIELD_VALUE_MISSING"
            checks.append(dict(symbol=symbol, period=str(period), field=field,
                status="RESOLVED" if category is None else "UNKNOWN", category=category, reason=reason,
                value=value, required_by_quality=False, purpose="TTM_QUARTER_DIAGNOSTIC_ONLY",
                basis="QUARTER" if field in FLOW_FIELDS else "INSTANT", source_basis=row.get("flow_basis"),
                source=row.get("source"), source_url=row.get("source_url"), currency=row.get("currency"), unit="TWD/share" if field == "basic_eps" else row.get("unit"),
                scope=row.get("scope"), revision_id=row.get("revision_id"), content_sha256=row.get("content_sha256"),
                available_at=row.get("available_at"), evidence=trace or evidence))
    return checks


def verification_diagnostics(records: Sequence[Mapping], *, symbol: str, as_of: object,
                             market: str = "TWSE", preprocessing=None) -> dict:
    """Explain evidence gaps without changing evaluate_quality or granting PASS.

    Required reported inputs schedule remediation. Historical quarter/TTM gaps
    remain a separate inventory, not an additional eligibility gate.
    """
    result = {"missing_periods": [], "unusable_periods": [], "unresolved_fields": [], "sources": [],
              "latest_available_at": None, "latest_period": None,
              "backfill_needed": False, "local_review_required": False,
              "next_action": "NORMAL_MODEL_EVALUATION", "field_checks": [], "field_gaps": [],
              "period_policy": "SELECTED_REPORTED_PERIOD_AND_BOUND_COMPARISONS",
              "diagnostic_missing_periods": [], "diagnostic_unresolved_fields": []}
    excluded_inputs: list[dict] = []
    result["excluded_quality_inputs"] = excluded_inputs
    periods = {}
    needed = set()
    timestamp = pd.NaT
    try:
        timestamp = pd.to_datetime(as_of, utc=True, errors="raise")
        if pd.isna(timestamp):
            raise ValueError("INVALID_DECISION_TIME")
        periods, ledger, excluded = select_quality_inputs(
            records, symbol=symbol, market=market, cutoff=timestamp, preprocessing=preprocessing,
            excluded_inputs=excluded_inputs)
        excluded_inputs.extend(excluded)
        def prepare(kind, compute):
            return _quality_preprocess(preprocessing, kind, compute, records=records,
                                       symbol=symbol, market=market, cutoff=timestamp)
        if not periods:
            result.update(backfill_needed=True, next_action="BACKFILL_MISSING_QUARTERS")
            result["unresolved_fields"] = ["NO_POINT_IN_TIME_FINANCIALS"]
            latest = pd.Period(timestamp.tz_convert("Asia/Taipei").date(), freq="Q") - 1
            needed = {latest - i for i in range(8)}
            result["diagnostic_missing_periods"] = [str(p) for p in sorted(needed)]
            result["comparison_window_basis"] = "LAST_COMPLETED_QUARTER_DIAGNOSTIC_ONLY_NOT_PUBLICATION_DEADLINE"
            result["field_checks"] = financial_field_checks({}, needed, symbol=symbol)
            result["field_gaps"] = result["field_checks"]
            return result
        latest = max(periods)
        result.update(latest_period=str(latest),
                      latest_available_at=max(r["_available"] for r in periods.values()).isoformat(),
                      sources=sorted({r["source"] for r in periods.values()}))
        needed = required_quality_periods(periods)
        result["comparison_periods"] = [str(p) for p in sorted(needed)]
        excluded_periods = {pd.Period(row["period"], freq="Q") for row in excluded_inputs}
        result["diagnostic_missing_periods"] = [str(p) for p in sorted(needed - periods.keys() - excluded_periods)]
        result["unusable_periods"] = [str(p) for p in sorted(excluded_periods & needed)]
        result["field_checks"] = financial_field_checks(periods, needed, symbol=symbol,
            normalization=lambda: prepare("quarters", lambda: normalize_quarters_with_diagnostics(periods)))
        excluded = {row["period"]: row for row in excluded_inputs}
        # Preserve the public aggregate reason while adding period/field detail.
        result["diagnostic_unresolved_fields"].extend(dict.fromkeys(row["reason"] for row in excluded_inputs))
        for check in result["field_checks"]:
            if check["period"] in excluded:
                evidence = excluded[check["period"]]
                check.update(status="UNKNOWN", category="INCOMPARABLE", reason=evidence["reason"],
                             evidence=evidence,
                             **{key: evidence.get(key) for key in ("source", "source_url", "currency",
                                 "unit", "scope", "revision_id", "content_sha256", "available_at")})
                result["diagnostic_unresolved_fields"].append(f'{check["period"]}:{check["field"]}:INCOMPARABLE')
        reported = _reported_inputs(periods, ledger, prepare=prepare, cutoff=timestamp)
        result["reported_basis"] = reported["basis"]
        result["reported_period"] = reported["period"]
        core_keys = {f"{latest}:{field}" for field in BALANCE_FIELDS}
        core_keys.update(reported["current_keys"].values())
        core_keys.update(reported["comparison_keys"][field] for field in
                         ("revenue", "cost_of_revenue", "operating_income", "net_income"))
        core_keys.update(reported["opening_keys"][field] for field in ("total_assets", "total_equity"))
        if reported["comparison_policy"] == "LEGACY_SAME_PERIOD_NORMALIZED_CONTRACT":
            result["missing_periods"] = [str(p) for p in (latest - 4, latest - 1) if p not in periods]
        for key in sorted(core_keys):
            check = dict(ledger[key])
            check.update(symbol=symbol, field=check.get("field", key.split(":")[-1]),
                         period=check.get("period", str(latest)), dependency_id=key,
                         required_by_quality=True, purpose="REPORTED_PERIOD_SCORING")
            if check.get("value") is None:
                check.update(status="UNKNOWN", category=check.get("category") or "MISSING_DATA",
                             reason=check.get("reason") or "REPORTED_SCORING_INPUT_MISSING")
                result["unresolved_fields"].append(key)
            else:
                check["status"] = "RESOLVED"
            result["field_checks"].append(check)
        result["field_gaps"] = [r for r in result["field_checks"] if r["status"] == "UNKNOWN"]
        result["diagnostic_unresolved_fields"].extend(
            f'{check["period"]}:{check["field"]}:{check["reason"]}' for check in result["field_gaps"]
            if not check["required_by_quality"])
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
        # Inventory for attribution only. Do not use this best-effort view to
        # bypass the native selection failure or supply Quality calculation.
        for raw in records:
            available = pd.to_datetime(raw.get("available_at"), utc=True, errors="coerce")
            end = pd.to_datetime(raw.get("period_end"), errors="coerce")
            if pd.notna(timestamp) and pd.notna(available) and pd.notna(end) and available <= timestamp:
                period = pd.Period(end, freq="Q")
                if period not in periods or available > pd.to_datetime(periods[period]["available_at"], utc=True):
                    periods[period] = raw
        needed = required_quality_periods(periods)
        result["field_checks"] = financial_field_checks(periods, needed, symbol=symbol, selection_error=str(error))
        result["field_gaps"] = [r for r in result["field_checks"] if r["status"] == "UNKNOWN"]
        return result


def evaluate_quality(records: Sequence[Mapping], *, symbol: str, market: str,
                     as_of: object, preprocessing=None, policy_version=None) -> QualityResult:
    """Evaluate only versions published by decision time, never today's latest retroactively."""
    version, scoring_policy, coverage_minima = quality_scoring_rules(policy_version)
    limited_gaps = version == QUALITY_R4_POLICY_VERSION
    field_ledger, metric_checks = {}, {}
    def failed(reason: str, status: str = "REJECT", integrity: str = "FAIL") -> QualityResult:
        return QualityResult(status, None, 0, integrity, {}, {},
            {"quality_field_checks": field_ledger, "quality_metric_checks": metric_checks}, reasons=(reason,))

    timestamp = pd.to_datetime(as_of, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return failed("INVALID_DECISION_TIME", "DATA_INSUFFICIENT", "UNKNOWN")
    excluded_inputs: list[dict] = []
    def prepare(kind, compute):
        return _quality_preprocess(preprocessing, kind, compute, records=records,
                                   symbol=symbol, market=market, cutoff=timestamp)
    try:
        periods, checks, excluded = select_quality_inputs(
            records, symbol=symbol, market=market, cutoff=timestamp, preprocessing=preprocessing,
            field_ledger=field_ledger, excluded_inputs=excluded_inputs)
        field_ledger.update(checks)
        excluded_inputs.extend(excluded)
        if not periods:
            return failed("NO_POINT_IN_TIME_FINANCIALS", "DATA_INSUFFICIENT", "UNKNOWN")
        latest_period = max(periods)
        latest = periods[latest_period]
        balances = {key: number(latest.get(key)) for key in BALANCE_FIELDS}
        if any(value is None for value in balances.values()):
            return failed("REQUIRED_BALANCE_SHEET_FIELDS_UNRESOLVABLE", "DATA_INSUFFICIENT", "UNKNOWN")
        a, l, e = (balances[k] for k in ("total_assets", "total_liabilities", "total_equity"))
        if a <= 0 or e <= 0 or any(value < 0 for value in balances.values()):
            return failed("INVALID_BALANCE_OR_NONPOSITIVE_EQUITY")
        if abs(a - l - e) / a >= .01 or abs(a - balances["current_assets"] - balances["non_current_assets"]) / a >= .01:
            return failed("BALANCE_SHEET_IDENTITY_FAIL")
        if balances["current_liabilities"] > l:
            return failed("CURRENT_LIABILITIES_EXCEED_TOTAL")
        quarters, flow_traces = prepare("quarters", lambda: normalize_quarters_with_diagnostics(periods))
    except (ValueError, TypeError) as error:
        return failed(str(error), "DATA_INSUFFICIENT", "UNKNOWN")

    # Optional evidence is never required, but a confirmed current hard risk cannot be ignored.
    optional = {key: latest.get(key) if latest.get(key) is not None else "NOT_AVAILABLE" for key in OPTIONAL_FIELDS}
    if (latest.get("auditor_opinion") in {"ADVERSE", "DISCLAIMER", "QUALIFIED_MATERIAL_MISSTATEMENT"}
            or latest.get("going_concern") is True or latest.get("material_accounting_issue") is True):
        return failed("CONFIRMED_OPTIONAL_HARD_RISK")
    review_required = latest.get("audit_review_required") is True or latest.get("auditor_opinion") in {
        "UNKNOWN", "QUALIFIED", "QUALIFIED_UNKNOWN",
    }

    ttm_inputs, ttm_values, dependencies = {}, {}, set()
    def ttm(field: str, offset: int = 0) -> float | None:
        end_period = latest_period - offset
        key = (f"{field}_ttm" if offset == 0 else f"{field}_previous_ttm" if offset == 4
               else f"{field}_ttm_at_{end_period}")
        if key in ttm_values:
            return ttm_values[key]
        annual_row = periods.get(end_period, {})
        annual = reported_annual_flow(annual_row, end_period, field)
        if annual is not None:
            origin = annual_row.get("quarter_flow_source", annual_row)
            ttm_inputs[key] = dict(method="REPORTED_FULL_YEAR", period=str(end_period),
                source=origin.get("source"), content_sha256=origin.get("content_sha256"),
                revision_id=origin.get("revision_id"), available_at=annual_row["_available"].isoformat(),
                status="VERIFIED_REPORTED", value=annual, formula="reported_full_year", gaps=[])
            dependencies.add(f"{end_period}:{field}")
            ttm_values[key] = annual
            return annual
        needed = [end_period - i for i in range(4)]
        values = [quarters.get(period, {}).get(field) for period in needed]
        gaps = [dict(period=str(period), category=flow_traces.get(period, {}).get(field, {}).get("category") or "MISSING_DATA",
                     reason=flow_traces.get(period, {}).get(field, {}).get("reason") or "NO_ELIGIBLE_FILING_FOR_PERIOD")
                for period, value in zip(needed, values) if value is None]
        contradictions = {"CONFLICTING_AGGREGATE_VALUES", "UNIT_MISSING_OR_INCOMPATIBLE",
                          "NUMERIC_TEXT_UNRESOLVED", "CASH_FLOW_RECONCILIATION_FAILED", "FILING_PERIOD_END_MISMATCH",
                          "FIELD_PROOF_BINDING_OR_VALUE_MISMATCH"}
        if (gaps and periods.get(end_period, {}).get("comparative_ytd_field_evidence")
                and not any(gap["reason"] in contradictions for gap in gaps)
                and field != "basic_eps" and end_period.quarter != 4):
            bridged, trace = comparative_ttm(periods, end_period, field, as_of)
            if bridged is not None:
                for source in trace["sources"]:
                    dependencies.add(f"{pd.Period(source['period_end'], freq='Q')}:{field}")
                trace.update(value=bridged)
                ttm_inputs[key], ttm_values[key] = trace, bridged
                return bridged
            bridge_gaps = trace["gaps"]
        else:
            bridge_gaps = []
        ttm_inputs[key] = dict(method="SUM_VERIFIED_QUARTERS" if not gaps else "UNRESOLVED_QUARTERS",
            periods=[str(period) for period in needed], gaps=gaps, comparative_bridge_gaps=bridge_gaps,
            formula="sum_four_comparable_quarters", operands=values,
            status="VERIFIED_DERIVED" if not gaps else "PENDING_VERIFICATION",
            available_at=max((periods[p]["_available"] for p in needed if p in periods), default=timestamp).isoformat())
        dependencies.update(f"{p}:{field}" for p in needed)
        ttm_values[key] = sum(values) if not gaps else None
        ttm_inputs[key]["value"] = ttm_values[key]
        return ttm_values[key]

    components = {name: [0., 0., budget] for name, budget in QUALITY_BUDGETS.items()}
    # Only scored dependencies feed measured DC. EPS/tax/debt diagnostics are
    # evaluated later without diluting (or inflating) the core evidence floor.
    metrics: dict = {f"{key}_ttm": ttm(key) for key in (*DIAGNOSTIC_TTM_FIELDS, "capex")}
    metrics.update({key + "_previous_ttm": ttm(key, 4) for key in DIAGNOSTIC_TTM_FIELDS[:4]})
    metrics["quality_policy_version"] = version
    metrics["quality_scoring_policy"] = {key: dict(value) if isinstance(value, dict) else
        list(value) if isinstance(value, list) else value for key, value in scoring_policy.items()}
    metrics["quality_profitability_basis"] = QUALITY_PROFITABILITY_BASIS
    metrics["quality_profitability_window"] = QUALITY_PROFITABILITY_WINDOW
    metrics["quality_field_checks"] = field_ledger
    metrics["quality_metric_checks"] = metric_checks
    metrics["quality_budgets"] = dict(QUALITY_BUDGETS)
    metrics["quality_ttm_inputs"] = ttm_inputs
    metrics["comparison_periods"] = [str(p) for p in sorted(periods)]
    metrics["excluded_quality_inputs"] = excluded_inputs
    metrics["quarter_flow_methods"] = {str(p): {key: value["reason"] for key, value in fields.items()}
        for p, fields in flow_traces.items()}
    metrics["quarter_eps_supplements"] = {
        str(period): row["quarter_eps_provenance"] for period, row in periods.items()
        if "quarter_eps_provenance" in row
    }
    reported = _reported_inputs(periods, field_ledger, prepare=prepare, cutoff=timestamp)
    metrics["quality_reported_inputs"] = reported
    metrics["quality_reported_basis"] = reported["basis"]
    metrics.update({key + "_reported": value for key, value in reported["current"].items()})
    # TTM still validates its own multi-document bridge. Its dependencies do
    # not determine measured confidence or missing weight in reported scoring.
    dependencies.clear()
    opening = reported["opening"]
    opening_values = [number(opening.get(key)) for key in ("total_assets", "total_liabilities", "total_equity")]
    if all(value is not None for value in opening_values):
        oa, ol, oe = opening_values
        if oa <= 0 or oe <= 0 or ol < 0 or abs(oa - ol - oe) / oa >= .01:
            return failed("HISTORICAL_BALANCE_SHEET_IDENTITY_FAIL")

    def award(group: str, weight: float, fraction: float | None, *, name: str = "", state: str | None = None) -> None:
        metric_checks[name or f"{group}_{len(metric_checks) + 1}"] = dict(
            group=group, weight=weight, fraction=fraction,
            status=state or ("EVALUATED" if fraction is not None else "MISSING_OR_INCOMPARABLE"),
            available=fraction is not None, points=weight * fraction if fraction is not None else 0.)
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
    dependencies.update(f"{latest_period}:{field}" for field in BALANCE_FIELDS)
    debt = ratio(l, a)
    current_ratio = ratio(balances["current_assets"], balances["current_liabilities"])
    rev_growth, _ = growth(reported["current"]["revenue"], reported["comparative"]["revenue"])
    liability_growth, _ = growth(l, prior.get("total_liabilities"))
    divergence = liability_growth - rev_growth if liability_growth is not None and rev_growth is not None else None
    metrics.update(debt_ratio=debt, current_ratio=current_ratio, equity_ratio=e / a,
                   revenue_growth=rev_growth, liability_growth=liability_growth, liability_divergence=divergence)
    award("balance_sheet", 6, 1., name="balance_sheet_identity")
    award("balance_sheet", 10, bands(debt, (.4, .5, .6, .7), (1, .75, .5, .25), ascending=True), name="debt_ratio")
    current_not_meaningful = limited_gaps and balances["current_liabilities"] == 0
    award("balance_sheet", 9, 0. if current_not_meaningful else bands(current_ratio, (1.5, 1.2, 1), (1, 5/7, 2/7)),
          name="current_ratio", state="NOT_MEANINGFUL" if current_not_meaningful else None)
    metrics["quality_diagnostic_checks"] = {"liability_divergence": dict(
        value=divergence, status="EVALUATED" if divergence is not None else "PENDING_VERIFICATION",
        scored=False, affects_gate=False)}

    margin_fields = (("gross_margin", 6), ("operating_margin", 8), ("net_margin", 6))
    for name, weight in margin_fields:
        values = []
        known_nonpositive = []
        numerator_field = "operating_income" if name == "operating_margin" else "net_income"
        needed_fields = ("revenue", "cost_of_revenue" if name == "gross_margin" else numerator_field)
        for group, keys in (("comparative", "comparison_keys"), ("current", "current_keys")):
            dependencies.update(reported[keys][field] for field in needed_fields)
            revenue = reported[group]["revenue"]
            numerator = reported[group][numerator_field]
            if name == "gross_margin":
                cost = reported[group]["cost_of_revenue"]
                numerator = revenue - cost if revenue is not None and cost is not None else None
            values.append(ratio(numerator, revenue))
            known_nonpositive.append(limited_gaps and numerator is not None
                                     and revenue is not None and revenue <= 0)
        now = values[-1]
        metrics[name] = now
        # Uniform three-part policy; peer rank is diagnostic only. The budget
        # is fixed before inspecting this stock's data, never renormalized.
        metrics[f"{name}_reported_history"] = values
        # Preserve genuine rolling TTM diagnostics, never relabel YTD as TTM.
        ttm_history = []
        for i in reversed(range(4)):
            revenue = ttm("revenue", i)
            numerator = ttm(numerator_field, i)
            if name == "gross_margin":
                cost = ttm("cost_of_revenue", i)
                numerator = revenue - cost if revenue is not None and cost is not None else None
            ttm_history.append(ratio(numerator, revenue))
        metrics[f"{name}_ttm_history"] = ttm_history
        metrics[f"{name}_drawdown_4q"] = margin_drawdown(ttm_history)  # diagnostic only
        award("profitability", weight * QUALITY_PROFITABILITY_PARTS["positive"],
              0. if known_nonpositive[-1] else float(now > 0) if now is not None else None,
              name=name + "_positive", state="NOT_MEANINGFUL" if known_nonpositive[-1] else None)
        history = values[:-1]
        history_rank = sum(.5 if isclose(x, now, abs_tol=1e-12) else float(x < now)
                           for x in history) / len(history) if now is not None and all(x is not None for x in history) else None
        history_not_meaningful = (any(known_nonpositive)
            and all(value is not None or known for value, known in zip(values, known_nonpositive)))
        award("profitability", weight * QUALITY_PROFITABILITY_PARTS["history"],
              0. if history_not_meaningful else history_rank if now is not None and now > 0 else (0. if history_rank is not None else None),
              name=name + "_history", state="NOT_MEANINGFUL" if history_not_meaningful else None)
        recent = values
        drawdown = margin_drawdown(recent)
        metrics[f"{name}_reported_drawdown"] = drawdown
        stable = float(now > 0 and drawdown <= .05 + 1e-12 and now >= recent[0] - .01 - 1e-12) if drawdown is not None else None
        award("profitability", weight * QUALITY_PROFITABILITY_PARTS["stability"],
              0. if history_not_meaningful else stable, name=name + "_stability",
              state="NOT_MEANINGFUL" if history_not_meaningful else None)
        # Peer diagnostics retain their explicit TTM receipt; do not mix
        # reported-period scoring margins with a TTM comparison universe.
        peer_evidence = latest.get("quality_peer_evidence", {}).get(name, {})
        peer = number(peer_evidence.get("percentile"))
        peer_at = pd.to_datetime(peer_evidence.get("available_at"), utc=True, errors="coerce")
        peer_valid = (peer_evidence.get("basis") == "TTM" and peer_evidence.get("status") == "VERIFIED"
            and peer_evidence.get("symbol") == symbol and peer_evidence.get("market") == market
            and peer_evidence.get("scope") == latest["scope"] and peer_evidence.get("period") == str(latest_period)
            and peer_evidence.get("universe_id") and peer_evidence.get("source_hashes")
            and number(peer_evidence.get("metric_value")) is not None and ttm_history[-1] is not None
            and isclose(number(peer_evidence["metric_value"]), ttm_history[-1], abs_tol=1e-12)
            and pd.notna(peer_at) and peer_at <= timestamp and peer_at >= latest["_available"]
            and peer is not None and 0 <= peer <= 1)
        metrics.setdefault("quality_diagnostic_checks", {})[name + "_peer"] = dict(
            value=peer if peer_valid else None,
            status="VERIFIED_DERIVED" if peer_valid else "PENDING_VERIFICATION",
            scored=False, affects_gate=False,
        )

    for field, weight in QUALITY_GROWTH_WEIGHTS.items():
        dependencies.update((reported["current_keys"][field], reported["comparison_keys"][field]))
        value, state = growth(reported["current"][field], reported["comparative"][field])
        metrics[f"{field}_growth"] = value
        metrics[f"{field}_growth_state"] = state
        fraction = bands(value, (.2, .1, 0, -.1), (1, .8, .6, .2))
        if state != "COMPARABLE":
            fraction = {"LOSS_TO_PROFIT": .6, "IMPROVING_LOSS": .2, "DETERIORATING_LOSS": 0}.get(state)
        award("growth", weight, fraction, name=field + "_growth", state=state)
    eps = [quarters.get(latest_period - i, {}).get("basic_eps") for i in range(8)]
    positive_eps = sum(v > 0 for v in eps) if all(v is not None for v in eps) else None
    metrics["positive_eps_quarters"] = positive_eps
    metrics["quality_diagnostic_checks"]["positive_eps_quarters"] = dict(
        value=positive_eps, status="EVALUATED" if positive_eps is not None else "PENDING_VERIFICATION",
        scored=False, affects_gate=False,
    )
    for metric, field, cutoffs in (("roe", "total_equity", (.2, .15, .1, .05)),
                                  ("roa", "total_assets", (.1, .07, .05, .02))):
        dependencies.add(reported["current_keys"]["net_income"])
        dependencies.add(reported["opening_keys"][field])
        previous = number(opening.get(field))
        average = (balances[field] + previous) / 2 if previous is not None and previous > 0 else None
        metrics[metric] = ratio(reported["current"]["net_income"], average)
        metrics[metric + "_reported"] = metrics[metric]
        scaled = tuple(value * reported["duration_fraction"] for value in cutoffs)
        award("return_on_capital", QUALITY_RETURN_WEIGHTS[metric], bands(metrics[metric], scaled, (1, .8, .5, .2)), name=metric)

    # PP&E acquisition payments only, not investing cash flow or intangible
    # additions. Normalize each reported quarter's outflow sign AFTER safe
    # quarter derivation; missing quarters must never become zero CAPEX.
    capex_quarters = [number(quarters.get(latest_period - i, {}).get("capex")) for i in range(4)]
    annual_capex = reported_annual_flow(latest, latest_period, "capex")
    capex_ttm = (abs(annual_capex) if annual_capex is not None else
                 sum(abs(value) for value in capex_quarters) if all(v is not None for v in capex_quarters) else None)
    if capex_ttm is None and ttm_inputs.get("capex_ttm", {}).get("method") == "COMPARATIVE_ORIGINAL_TTM":
        # A sign-consistent acquisition-payment bridge, not total investing CF.
        capex_ttm = abs(metrics["capex_ttm"])
    cfo_ttm, net_ttm = metrics["cfo_ttm"], metrics["net_income_ttm"]
    fcf_ttm = cfo_ttm - capex_ttm if cfo_ttm is not None and capex_ttm is not None else None
    cfo_reported, net_reported = reported["current"]["cfo"], reported["current"]["net_income"]
    capex_reported = reported["current"]["capex"]
    capex_reported = abs(capex_reported) if capex_reported is not None else None
    fcf_reported = cfo_reported - capex_reported if cfo_reported is not None and capex_reported is not None else None
    cash_conversion = ratio(cfo_reported, net_reported)
    conversion_not_meaningful = cfo_reported is not None and net_reported is not None and net_reported <= 0
    dependencies.update(reported["current_keys"][field] for field in ("cfo", "capex", "net_income"))
    award("cash_flow", 5, float(cfo_reported > 0) if cfo_reported is not None else None, name="cfo_positive")
    award("cash_flow", 5, 0. if conversion_not_meaningful else bands(cash_conversion, (1., .8, .5), (1., .8, .4)),
          name="cash_conversion", state="NOT_MEANINGFUL" if conversion_not_meaningful else None)
    metrics["cash_conversion_state"] = "NOT_MEANINGFUL" if conversion_not_meaningful else "EVALUATED" if cash_conversion is not None else "MISSING_OR_INCOMPARABLE"
    award("cash_flow", 5, float(fcf_reported > 0) if fcf_reported is not None else None, name="fcf_positive")
    # Diagnostic TTM calls above add dependencies; do not allow those to leak
    # into the actual measured DC ledger. Rebuild it from scored inputs only.
    dependencies = {f"{latest_period}:{field}" for field in BALANCE_FIELDS}
    dependencies.update(reported["current_keys"].values())
    dependencies.update(reported["comparison_keys"][field] for field in
                        ("revenue", "cost_of_revenue", "operating_income", "net_income"))
    dependencies.update(reported["opening_keys"][field] for field in ("total_assets", "total_equity"))

    # Freeze the actual scored dependencies before any optional diagnostic
    # calculation. These calculations remain available for audit/valuation.
    scored_dependencies = set(dependencies)
    for metric, field in (("roe", "total_equity"), ("roa", "total_assets")):
        old = number(prior.get(field))
        average = (balances[field] + old) / 2 if old is not None and old > 0 else None
        pa, pl, pe = (number(prior.get(k)) for k in ("total_assets", "total_liabilities", "total_equity"))
        if any(v is None for v in (pa, pl, pe)) or pa <= 0 or pe <= 0 or pl < 0 or abs(pa - pl - pe) / pa >= .01:
            average = None
        metrics[metric + "_ttm"] = ratio(net_ttm, average)
    metrics["basic_eps_ttm"] = ttm("basic_eps")
    metrics["basic_eps_ttm_basis"] = "SUM_REPORTED_QUARTER_EPS" if metrics["basic_eps_ttm"] is not None else "UNAVAILABLE"
    if latest_period.quarter == 4 and latest.get("flow_basis") == "YTD":
        metrics["basic_eps_ttm"] = number(latest.get("basic_eps"))
        metrics["basic_eps_ttm_basis"] = "REPORTED_FULL_YEAR" if metrics["basic_eps_ttm"] is not None else "UNAVAILABLE"
    metrics["income_tax_ttm"], metrics["pretax_income_ttm"] = ttm("income_tax"), ttm("pretax_income")

    def invested_capital(row):
        equity, borrowings, cash = (number(row.get(k)) for k in ("total_equity", "interest_bearing_debt", "cash"))
        if any(v is None for v in (equity, borrowings, cash)) or equity <= 0 or borrowings < 0 or cash < 0:
            return None
        capital = equity + borrowings - cash
        return capital if capital > 0 else None

    current_capital, previous_capital = invested_capital(latest), invested_capital(prior)
    average_capital = ((current_capital + previous_capital) / 2
                       if current_capital is not None and previous_capital is not None else None)
    effective_tax = ratio(metrics["income_tax_ttm"], metrics["pretax_income_ttm"])
    tax_rate = min(.35, max(0., effective_tax)) if effective_tax is not None else None
    nopat = (metrics["operating_income_ttm"] * (1 - tax_rate)
             if metrics["operating_income_ttm"] is not None and tax_rate is not None else None)
    roic = ratio(nopat, average_capital)
    metrics["quality_diagnostic_checks"]["roic"] = dict(
        value=roic, status="EVALUATED" if roic is not None else "PENDING_VERIFICATION",
        scored=False, affects_gate=False,
    )
    metrics.update(capex_ttm=capex_ttm, fcf_ttm=fcf_ttm, cash_conversion=cash_conversion,
                   cash_conversion_ttm=ratio(cfo_ttm, net_ttm), capex_reported=capex_reported, fcf_reported=fcf_reported,
                   effective_tax_rate=effective_tax, normalized_tax_rate=tax_rate, nopat_ttm=nopat,
                   invested_capital=current_capital, prior_invested_capital=previous_capital,
                   average_invested_capital=average_capital, roic=roic,
                   capex_basis="REPORTED_PPE_ACQUISITION_PAYMENTS",
                   invested_capital_basis="TOTAL_EQUITY_PLUS_BORROWINGS_AND_LEASES_MINUS_TOTAL_CASH")

    available = sum(c[1] for c in components.values())
    raw_score = sum(c[0] for c in components.values())
    # Fixed budget: missing data earns no points and does not shrink the
    # denominator. The additional uncertainty deduction is explicit and bounded.
    penalties = [dict(group=name, reason="MISSING_OR_INCOMPARABLE_METRICS",
                      missing_weight=c[2] - c[1], points=(c[2] - c[1]) * QUALITY_MISSING_PENALTY_RATE)
                 for name, c in components.items() if c[1] < c[2]]
    penalty = sum(item["points"] for item in penalties)
    score = max(0., raw_score - penalty) if available else None
    coverage = {name: 100 * c[1] / c[2] for name, c in components.items()}
    required_missing = [key + "_reported" for key in QUALITY_REQUIRED_REPORTED_FIELDS
                        if metrics[key + "_reported"] is None]
    metrics.update(quality_raw_score=raw_score, quality_penalty=penalty, quality_penalties=penalties,
                   quality_available_weight=available, quality_missing_weight=100. - available,
                   quality_final_score=score, quality_required_missing=required_missing,
                   quality_missing_penalty_rate=QUALITY_MISSING_PENALTY_RATE,
                   quality_minimum_available_weight=QUALITY_MINIMUM_AVAILABLE_WEIGHT,
                   quality_minimum_confidence=QUALITY_MINIMUM_CONFIDENCE,
                   quality_minimum_score=MINIMUM_ENTRY_QUALITY,
                   quality_coverage_minima=dict(coverage_minima),
                   quality_required_gaps={key: [field_ledger[f"{latest_period}:{key.removesuffix('_reported')}"]]
                                          for key in required_missing},
                   quality_score_basis="FIXED_100_BUDGET_MINUS_EVIDENCE_PENALTY")
    age = (timestamp.date() - latest_period.end_time.date()).days
    freshness = 100 if age <= 180 else 70 if age <= 275 else 0
    scored_sources = {latest["source"], *(field_ledger[key].get("source") for key in scored_dependencies
                       if field_ledger.get(key, {}).get("source"))}
    reliability = 100 if scored_sources == {"MOPS_IXBRL"} else 95
    official = scored_sources <= OFFICIAL_SOURCES
    if not official:
        reliability = 0
    verification = verification_summary(field_ledger, scored_dependencies)
    dc_inputs = dict(coverage=available, freshness=freshness, verification=verification["score"], reliability=reliability)
    confidence = sum(QUALITY_DC_WEIGHTS[key] * value for key, value in dc_inputs.items())
    metrics.update(quality_verification=verification, quality_dc_inputs=dc_inputs,
                   quality_dc_weights=dict(QUALITY_DC_WEIGHTS))
    enough = (available >= QUALITY_MINIMUM_AVAILABLE_WEIGHT
              and all(coverage[k] >= limit for k, limit in coverage_minima.items()))
    if confidence < 50 or score is None:
        status = "DATA_INSUFFICIENT"
    elif required_missing:
        status = "DATA_INSUFFICIENT"
    elif confidence < QUALITY_MINIMUM_CONFIDENCE or not enough or not official or age > 275:
        status = "PROVISIONAL"
    elif score < 60 and raw_score + (100. - available) < 60:
        # Even awarding all unknown items could not clear the quality floor:
        # poor observed financial conditions, not merely a missing-data penalty.
        status = "REJECT"
    elif score < MINIMUM_ENTRY_QUALITY:
        status = "WATCH_QUALITY"
    else:
        status = "PASS"
    warnings = []
    if divergence is not None and divergence > .1:
        warnings.append("LIABILITY_GROWTH_DIVERGENCE")
    if not enough:
        warnings.append("INSUFFICIENT_COMPONENT_COVERAGE")
    if penalties:
        warnings.append("QUALITY_EVIDENCE_PENALTY_APPLIED")
    if excluded_inputs:
        warnings.append("INCOMPARABLE_HISTORICAL_INPUTS_EXCLUDED")
    if required_missing:
        warnings.append("REQUIRED_QUALITY_SOURCES_UNKNOWN")
    if age > 180:
        warnings.append("STALE_FINANCIALS")
    if not official:
        warnings.append("UNTRUSTED_FINANCIAL_SOURCE")
    reasons = ("REQUIRED_QUALITY_SOURCES_UNKNOWN",) if required_missing else ()
    if review_required:
        reasons += ("AUDIT_REVIEW_REQUIRED",)
        if status != "REJECT":
            status = "DATA_INSUFFICIENT"
    return QualityResult(status, score, confidence, "PASS",
                         {k: c[0] for k, c in components.items()}, coverage, metrics,
                         tuple(warnings), reasons, optional)
