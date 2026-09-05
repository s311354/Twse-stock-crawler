"""Point-in-time Base and Extended quality model for Low Entry v5."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable

import pandas as pd

from domain.data_provenance import TRUSTED_AVAILABILITY_SOURCES
from domain.security_identity import is_airline_issuer


class BaseQualityStatus(str, Enum):
    PASS = "BASE_PASS"
    WATCH = "BASE_WATCH"
    PROVISIONAL = "BASE_PROVISIONAL"
    UNKNOWN = "BASE_UNKNOWN"
    REJECT = "BASE_REJECT"


class ExtendedQualityStatus(str, Enum):
    PASS = "EXTENDED_PASS"
    PROVISIONAL = "EXTENDED_PROVISIONAL"
    LIMITED = "EXTENDED_LIMITED"
    UNKNOWN = "EXTENDED_UNKNOWN"
    REJECT = "EXTENDED_REJECT"


class FinalQualityStatus(str, Enum):
    CONFIRMED = "QUALITY_CONFIRMED"
    PROVISIONAL = "QUALITY_PROVISIONAL"
    BASE_ONLY = "BASE_ONLY"
    WATCH = "QUALITY_WATCH"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    HARD_NO_BUY = "HARD_NO_BUY"
    SECTOR_UNSUPPORTED = "SECTOR_UNSUPPORTED"


@dataclass(frozen=True)
class QualityMetric:
    name: str
    maximum: float
    value: object
    passed: bool | None

    @property
    def available(self) -> bool:
        return self.passed is not None

    @property
    def earned(self) -> float:
        return self.maximum if self.passed else 0.0


@dataclass(frozen=True)
class LowEntryV5QualityResult:
    base_score: float | None
    base_data_confidence: float
    base_available_weight: float
    base_status: BaseQualityStatus
    extended_score: float | None
    extended_data_confidence: float
    extended_available_weight: float
    critical_field_coverage: float
    extended_status: ExtendedQualityStatus
    effective_extended_weight: float
    final_score: float | None
    final_status: FinalQualityStatus
    base_metrics: tuple[QualityMetric, ...]
    extended_metrics: tuple[QualityMetric, ...]
    hard_rejects: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class LowEntryV5QualityConfig:
    minimum_base_weight: float = 50.0
    minimum_base_pass_weight: float = 70.0
    minimum_extended_weight: float = 40.0
    maximum_extended_weight: float = 0.40
    unsupported_sector_codes: frozenset[str] = frozenset({"17"})
    balance_error_limit: float = 0.01
    air_max_debt_ratio: float = 0.80
    air_min_current_ratio: float = 0.60
    air_min_roe_ttm: float = 0.08
    air_min_roa_ttm: float = 0.025
    air_max_net_debt_to_equity: float = 0.80


@dataclass(frozen=True)
class _Rule:
    name: str
    maximum: float
    key: str
    predicate: Callable[[Any], bool]


@dataclass(frozen=True)
class _BaseEvidence:
    snapshot: dict[str, Any]
    point_in_time: bool
    provisional_only: bool
    warnings: tuple[str, ...]


_BASE_DERIVED_FIELDS = frozenset(
    {
        "debt_to_equity_below_industry_median",
        "equity_growth_yoy",
        "share_dilution_yoy",
        "revenue_growth_ttm_yoy",
        "positive_operating_quarters_8",
        "gross_margin_change",
        "operating_margin_change",
        "net_margin_above_industry_median",
        "roe_change_4q",
        "asset_growth_supported",
        "revenue_growth_above_industry_median",
        "eps_growth_ttm_yoy",
        "asset_growth_yoy",
    }
)

_MOPS_BASE_DIRECT_FIELDS = frozenset(
    {
        "equity",
        "total_equity",
        "total_assets",
        "total_liabilities",
        "current_assets",
        "current_liabilities",
        "debt_ratio",
        "current_ratio",
        "roe_ttm",
        "roa_ttm",
    }
)


def evaluate_v5_quality(
    annual_df: pd.DataFrame,
    snapshot: dict[str, Any],
    signal_date: str | pd.Timestamp,
    config: LowEntryV5QualityConfig | None = None,
) -> LowEntryV5QualityResult:
    """Evaluate v5 quality without converting missing evidence into failure."""

    cfg = config or LowEntryV5QualityConfig()
    signal = pd.to_datetime(signal_date, utc=True, errors="coerce")
    if pd.isna(signal):
        return _insufficient_result("INVALID_SIGNAL_DATE")

    sector = _text(snapshot.get("sector"))
    if not sector:
        return _insufficient_result("OFFICIAL_SECTOR_UNKNOWN")
    if sector in cfg.unsupported_sector_codes:
        return _unsupported_result()
    air_profile = is_airline_issuer(snapshot.get("issuer_symbol"), sector)

    annual = _point_in_time_annual(annual_df, signal)
    base_evidence = _resolve_base_evidence(snapshot, signal)
    extended_available_at = pd.to_datetime(
        snapshot.get("available_at"),
        utc=True,
        errors="coerce",
    )
    base_point_in_time = base_evidence.point_in_time
    extended_point_in_time = (
        not pd.isna(extended_available_at) and extended_available_at <= signal
    )
    warnings = list(base_evidence.warnings)
    if air_profile:
        warnings.append("AIR_QUALITY_PROFILE_ACTIVE")
    if not extended_point_in_time:
        warnings.append("MOPS_EXTENDED_NOT_POINT_IN_TIME")

    base_snapshot = base_evidence.snapshot
    extended_snapshot = snapshot if extended_point_in_time else {}
    balance_valid = _balance_sheet_valid(base_snapshot, cfg.balance_error_limit)
    if balance_valid is False:
        warnings.append("BALANCE_SHEET_NOT_BALANCED")

    base_values = _base_values(
        base_snapshot,
        prefer_ttm_returns=air_profile,
    )
    base_metrics = _evaluate_rules(
        base_values,
        _base_rules(
            max_debt_ratio=(
                cfg.air_max_debt_ratio if air_profile else 0.60
            ),
            min_current_ratio=(
                cfg.air_min_current_ratio if air_profile else 1.20
            ),
            min_roe=(cfg.air_min_roe_ttm if air_profile else 0.10),
            min_roa=(cfg.air_min_roa_ttm if air_profile else 0.05),
        ),
    )
    base_available = sum(item.maximum for item in base_metrics if item.available)
    if cfg.minimum_base_weight <= base_available < cfg.minimum_base_pass_weight:
        warnings.append(
            "BASE_PARTIAL_COVERAGE={:.0f}/{:.0f}".format(
                base_available,
                cfg.minimum_base_pass_weight,
            )
        )
    base_score = _normalized_score(base_metrics, minimum_weight=0)
    base_confidence = _base_confidence(
        base_snapshot,
        signal,
        coverage=base_available,
        balance_valid=balance_valid,
        point_in_time=base_point_in_time,
    )
    base_hard_rejects = _base_hard_rejects(base_snapshot)
    base_status = _classify_base(
        score=base_score,
        confidence=base_confidence,
        available_weight=base_available,
        hard_rejects=base_hard_rejects,
        integrity_valid=base_point_in_time and balance_valid is not False,
        provisional_only=base_evidence.provisional_only,
        config=cfg,
    )

    extended_values = _extended_values(annual, extended_snapshot)
    extended_not_applicable = _extended_not_applicable(
        extended_snapshot,
        extended_values,
    )
    extended_metrics = _evaluate_rules(
        extended_values,
        _extended_rules(
            max_net_debt_to_equity=(
                cfg.air_max_net_debt_to_equity if air_profile else 0.50
            )
        ),
    )
    extended_available = sum(
        item.maximum for item in extended_metrics if item.available
    )
    extended_score = _normalized_score(
        extended_metrics,
        minimum_weight=cfg.minimum_extended_weight,
    )
    critical_coverage = _critical_field_coverage(extended_snapshot)
    extended_confidence = _extended_confidence(
        extended_snapshot,
        signal,
        coverage=extended_available,
        critical_coverage=critical_coverage,
        balance_valid=balance_valid,
        point_in_time=extended_point_in_time,
    )
    extended_hard_rejects = _extended_hard_rejects(annual, extended_snapshot)
    extended_status = _classify_extended(
        score=extended_score,
        confidence=extended_confidence,
        available_weight=extended_available,
        critical_coverage=critical_coverage,
        hard_rejects=extended_hard_rejects,
        config=cfg,
    )
    warnings.extend(_metric_diagnostics("BASE", base_metrics))
    warnings.extend(
        _metric_diagnostics(
            "EXTENDED",
            extended_metrics,
            unavailable_reasons=extended_not_applicable,
        )
    )

    effective_extended_weight = (
        cfg.maximum_extended_weight * extended_confidence / 100
        if extended_score is not None
        else 0.0
    )
    final_score = _combine_scores(
        base_score,
        extended_score,
        effective_extended_weight,
    )
    final_status = _classify_final(
        base_status,
        extended_status,
        final_score,
    )

    return LowEntryV5QualityResult(
        base_score=base_score,
        base_data_confidence=round(base_confidence, 2),
        base_available_weight=round(base_available, 2),
        base_status=base_status,
        extended_score=extended_score,
        extended_data_confidence=round(extended_confidence, 2),
        extended_available_weight=round(extended_available, 2),
        critical_field_coverage=round(critical_coverage, 2),
        extended_status=extended_status,
        effective_extended_weight=round(effective_extended_weight, 4),
        final_score=final_score,
        final_status=final_status,
        base_metrics=base_metrics,
        extended_metrics=extended_metrics,
        hard_rejects=tuple(base_hard_rejects + extended_hard_rejects),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _resolve_base_evidence(
    snapshot: dict[str, Any],
    signal: pd.Timestamp,
) -> _BaseEvidence:
    """Keep only Base fields whose source was available at the signal time."""

    resolved = dict(snapshot)
    warnings: list[str] = []
    twse_source = _text(snapshot.get("twse_financial_source"))
    twse_observed_at = pd.to_datetime(
        snapshot.get("twse_financial_observed_at"),
        utc=True,
        errors="coerce",
    )
    twse_present = bool(twse_source) or "twse_financial_observed_at" in snapshot
    twse_valid = (
        twse_present
        and not pd.isna(twse_observed_at)
        and twse_observed_at <= signal
    )
    twse_stale = False
    if not twse_present or pd.isna(twse_observed_at):
        warnings.append("TWSE_BASE_SNAPSHOT_MISSING")
    elif twse_observed_at > signal:
        warnings.append("TWSE_BASE_OBSERVED_AFTER_SIGNAL")
    else:
        age_days = (signal - twse_observed_at).days
        if age_days > 450:
            twse_stale = True
            warnings.append("TWSE_BASE_STALE={}D".format(age_days))

    mops_available_at = pd.to_datetime(
        snapshot.get("available_at"),
        utc=True,
        errors="coerce",
    )
    mops_valid = (
        not pd.isna(mops_available_at)
        and mops_available_at <= signal
        and _trusted_source(snapshot)
    )

    if not twse_valid:
        for key in tuple(resolved):
            if key.startswith("twse_"):
                resolved.pop(key, None)
    if not mops_valid:
        for key in _MOPS_BASE_DIRECT_FIELDS:
            resolved.pop(key, None)

    feature_sources = _parse_evidence_pairs(
        snapshot.get("base_quality_feature_sources")
    )
    feature_available_at = _parse_evidence_pairs(
        snapshot.get("base_quality_feature_available_at")
    )
    rejected_fields: list[str] = []
    for key in _BASE_DERIVED_FIELDS:
        if key not in resolved:
            continue
        source = feature_sources.get(key, "")
        available_at = pd.to_datetime(
            feature_available_at.get(key),
            utc=True,
            errors="coerce",
        )
        source_eligible = (
            twse_valid
            if source.startswith("TWSE_OPENAPI")
            else mops_valid
            if source.startswith("MOPS_POINT_IN_TIME")
            else twse_valid or mops_valid
        )
        time_eligible = pd.isna(available_at) or available_at <= signal
        if not source_eligible or not time_eligible:
            resolved.pop(key, None)
            rejected_fields.append(key)
    if rejected_fields:
        warnings.append(
            "BASE_FIELDS_NOT_POINT_IN_TIME={}".format(
                ",".join(sorted(rejected_fields))
            )
        )

    if not twse_valid and mops_valid:
        warnings.append("MOPS_POINT_IN_TIME_BASE_FALLBACK")

    return _BaseEvidence(
        snapshot=resolved,
        point_in_time=twse_valid or mops_valid,
        provisional_only=(not twse_valid and mops_valid) or twse_stale,
        warnings=tuple(warnings),
    )


def _parse_evidence_pairs(value: object) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in _text(value).split(";"):
        key, separator, item_value = item.partition("=")
        if separator and key.strip() and item_value.strip():
            pairs[key.strip()] = item_value.strip()
    return pairs


def _base_rules(
    *,
    max_debt_ratio: float = 0.60,
    min_current_ratio: float = 1.20,
    min_roe: float = 0.10,
    min_roa: float = 0.05,
) -> tuple[_Rule, ...]:
    positive = lambda value: float(value) > 0
    non_negative = lambda value: float(value) >= 0
    true = lambda value: bool(value)
    return (
        _Rule("positive_equity", 7, "equity", positive),
        _Rule("debt_ratio", 7, "debt_ratio", lambda value: float(value) <= max_debt_ratio),
        _Rule("current_ratio", 6, "current_ratio", lambda value: float(value) >= min_current_ratio),
        _Rule("debt_to_equity_industry", 5, "debt_to_equity_below_industry_median", true),
        _Rule("equity_growth", 5, "equity_growth_yoy", positive),
        _Rule("capital_not_inflated", 5, "share_dilution_yoy", lambda value: float(value) <= 0.05),
        _Rule("revenue_growth", 6, "revenue_growth_ttm_yoy", positive),
        _Rule("positive_operating_quarters", 6, "positive_operating_quarters_8", lambda value: float(value) >= 6),
        _Rule("gross_margin_stable", 6, "gross_margin_change", lambda value: float(value) >= -0.01),
        _Rule("operating_margin_stable", 6, "operating_margin_stable", true),
        _Rule("net_margin_industry", 6, "net_margin_above_industry_median", true),
        _Rule("roe", 8, "roe", lambda value: float(value) >= min_roe),
        _Rule("roa", 6, "roa", lambda value: float(value) >= min_roa),
        _Rule("roe_not_deteriorating", 3, "roe_change_4q", lambda value: float(value) >= -0.02),
        _Rule("asset_productivity", 3, "asset_growth_supported", true),
        _Rule("revenue_growth_industry", 4, "revenue_growth_above_industry_median", true),
        _Rule("eps_growth", 4, "eps_growth_ttm_yoy", positive),
        _Rule("equity_growth_structure", 3, "equity_growth_yoy", positive),
        _Rule("asset_growth_rational", 2, "asset_growth_rational", true),
        _Rule("limited_dilution", 2, "share_dilution_yoy", lambda value: float(value) <= 0.02),
    )


def _extended_rules(
    *,
    max_net_debt_to_equity: float = 0.50,
) -> tuple[_Rule, ...]:
    positive = lambda value: float(value) > 0
    true = lambda value: bool(value)
    return (
        _Rule("positive_cfo", 8, "cfo_ttm", positive),
        _Rule("cash_conversion", 8, "cash_conversion", lambda value: float(value) >= 0.80),
        _Rule("positive_fcf", 7, "fcf_ttm", positive),
        _Rule("fcf_to_cfo", 5, "fcf_to_cfo", lambda value: float(value) >= 0.50),
        _Rule("accrual_ratio", 4, "accrual_ratio", lambda value: float(value) <= 0.10),
        _Rule("positive_cfo_history", 3, "positive_cfo_year_ratio", lambda value: float(value) >= 2 / 3),
        _Rule("net_debt_to_equity", 7, "net_debt_to_equity", lambda value: float(value) <= max_net_debt_to_equity),
        _Rule("interest_coverage", 7, "interest_coverage", lambda value: float(value) >= 5),
        _Rule("short_debt_coverage", 4, "cash_to_short_term_debt", lambda value: float(value) >= 1),
        _Rule("debt_growth", 4, "debt_growth_supported", true),
        _Rule("interest_expense_valid", 3, "interest_expense_valid", true),
        _Rule("receivable_growth", 5, "receivable_growth_supported", true),
        _Rule("inventory_growth", 5, "inventory_growth_supported", true),
        _Rule("receivable_days", 3, "receivable_days_change", lambda value: float(value) <= 0),
        _Rule("inventory_days", 2, "inventory_days_change", lambda value: float(value) <= 0),
        _Rule("capex_to_cfo", 4, "capex_to_cfo", lambda value: 0 <= float(value) <= 0.80),
        _Rule("fixed_asset_turnover", 3, "fixed_asset_turnover_change", lambda value: float(value) >= 0),
        _Rule("construction_supported", 3, "construction_growth_supported", true),
        _Rule("distribution_coverage", 4, "distribution_coverage", lambda value: 0 <= float(value) <= 1),
        _Rule("stable_shares", 3, "share_dilution_yoy", lambda value: float(value) <= 0),
        _Rule("retained_returns", 3, "retained_returns_improving", true),
        _Rule("audit_opinion", 2, "audit_clean_enough", true),
        _Rule("no_material_issue", 1, "no_material_issue", true),
        _Rule("no_going_concern", 1, "no_going_concern", true),
        _Rule("filing_traceable", 1, "filing_traceable", true),
    )


def _evaluate_rules(
    values: dict[str, object],
    rules: tuple[_Rule, ...],
) -> tuple[QualityMetric, ...]:
    metrics: list[QualityMetric] = []
    for rule in rules:
        value = values.get(rule.key)
        passed = None if _missing(value) else bool(rule.predicate(value))
        metrics.append(QualityMetric(rule.name, rule.maximum, value, passed))
    return tuple(metrics)


def _metric_diagnostics(
    prefix: str,
    metrics: tuple[QualityMetric, ...],
    *,
    unavailable_reasons: dict[str, str] | None = None,
) -> list[str]:
    reasons = unavailable_reasons or {}
    missing = [
        metric.name
        for metric in metrics
        if not metric.available and metric.name not in reasons
    ]
    failed = [metric.name for metric in metrics if metric.passed is False]
    diagnostics: list[str] = []
    if missing:
        diagnostics.append("{}_MISSING={}".format(prefix, ",".join(missing)))
    if failed:
        diagnostics.append("{}_FAILED={}".format(prefix, ",".join(failed)))
    not_applicable = [
        "{}({})".format(metric.name, reasons[metric.name])
        for metric in metrics
        if not metric.available and metric.name in reasons
    ]
    if not_applicable:
        diagnostics.append(
            "{}_NOT_APPLICABLE={}".format(prefix, ",".join(not_applicable))
        )
    return diagnostics


def _base_values(
    snapshot: dict[str, Any],
    *,
    prefer_ttm_returns: bool = False,
) -> dict[str, object]:
    equity = _first_number(snapshot, "twse_parent_equity", "equity", "total_equity")
    assets = _first_number(snapshot, "twse_total_assets", "total_assets")
    liabilities = _first_number(snapshot, "twse_total_liabilities", "total_liabilities")
    debt_ratio = _first_number(snapshot, "twse_debt_ratio", "debt_ratio")
    if debt_ratio is None:
        debt_ratio = _safe_div(liabilities, assets)
    current_ratio = _first_number(snapshot, "twse_current_ratio", "current_ratio")
    operating_margin = _first_number(
        snapshot,
        "twse_operating_margin_ytd",
        "operating_margin_ttm",
    )
    operating_margin_change = _first_number(snapshot, "operating_margin_change")
    operating_margin_stable = (
        None
        if operating_margin is None or operating_margin_change is None
        else operating_margin > 0
        and operating_margin_change >= -0.01
    )
    asset_growth = _first_number(snapshot, "asset_growth_yoy")
    revenue_growth = _first_number(snapshot, "revenue_growth_ttm_yoy")
    asset_growth_rational = (
        None
        if asset_growth is None or revenue_growth is None
        else asset_growth <= revenue_growth + 0.05
    )
    roe_keys = (
        ("roe_ttm", "twse_roe_ytd_end_balance")
        if prefer_ttm_returns
        else ("twse_roe_ytd_end_balance", "roe_ttm")
    )
    roa_keys = (
        ("roa_ttm", "twse_roa_ytd_end_balance")
        if prefer_ttm_returns
        else ("twse_roa_ytd_end_balance", "roa_ttm")
    )
    return {
        "equity": equity,
        "debt_ratio": debt_ratio,
        "current_ratio": current_ratio,
        "debt_to_equity_below_industry_median": snapshot.get(
            "debt_to_equity_below_industry_median"
        ),
        "equity_growth_yoy": _first_number(snapshot, "equity_growth_yoy"),
        "share_dilution_yoy": _first_number(snapshot, "share_dilution_yoy"),
        "revenue_growth_ttm_yoy": revenue_growth,
        "positive_operating_quarters_8": _first_number(
            snapshot, "positive_operating_quarters_8"
        ),
        "gross_margin_change": _first_number(snapshot, "gross_margin_change"),
        "operating_margin_stable": operating_margin_stable,
        "net_margin_above_industry_median": snapshot.get(
            "net_margin_above_industry_median"
        ),
        "roe": _first_number(snapshot, *roe_keys),
        "roa": _first_number(snapshot, *roa_keys),
        "roe_change_4q": _first_number(snapshot, "roe_change_4q"),
        "asset_growth_supported": snapshot.get("asset_growth_supported"),
        "revenue_growth_above_industry_median": snapshot.get(
            "revenue_growth_above_industry_median"
        ),
        "eps_growth_ttm_yoy": _first_number(snapshot, "eps_growth_ttm_yoy"),
        "asset_growth_rational": asset_growth_rational,
    }


def _extended_values(
    annual: pd.DataFrame,
    snapshot: dict[str, Any],
) -> dict[str, object]:
    cfo = _first_number(snapshot, "operating_cash_flow_ttm", "cfo_ttm")
    capex = _first_number(snapshot, "capex_ttm")
    fcf = _first_number(snapshot, "free_cash_flow_ttm", "fcf_ttm")
    net_income = _first_number(snapshot, "net_income_ttm")
    cash_conversion = _first_number(snapshot, "cash_conversion")
    if net_income is not None and net_income <= 0:
        cash_conversion = None
    audit_opinion = _text(snapshot.get("audit_opinion")).upper()
    material_issue = _flag(snapshot.get("material_accounting_issue"))
    going_concern = _flag(snapshot.get("going_concern"))
    positive_cfo_ratio = _positive_year_ratio(annual, "operating_cash_flow")
    debt_growth = _first_number(snapshot, "interest_bearing_debt_growth_yoy")
    revenue_growth = _first_number(snapshot, "revenue_growth_ttm_yoy")
    return {
        "cfo_ttm": cfo,
        "cash_conversion": cash_conversion,
        "fcf_ttm": fcf,
        "fcf_to_cfo": _safe_div(fcf, cfo) if cfo is not None and cfo > 0 else None,
        "accrual_ratio": _first_number(snapshot, "accrual_ratio"),
        "positive_cfo_year_ratio": positive_cfo_ratio,
        "net_debt_to_equity": _first_number(snapshot, "net_debt_to_equity"),
        "interest_coverage": _first_number(snapshot, "interest_coverage"),
        "cash_to_short_term_debt": _first_number(snapshot, "cash_to_short_term_debt"),
        "debt_growth_supported": (
            None
            if debt_growth is None or revenue_growth is None
            else debt_growth <= revenue_growth
        ),
        "interest_expense_valid": _interest_expense_valid(snapshot),
        "receivable_growth_supported": snapshot.get("receivable_growth_supported"),
        "inventory_growth_supported": snapshot.get("inventory_growth_supported"),
        "receivable_days_change": _first_number(snapshot, "receivable_days_change"),
        "inventory_days_change": _first_number(snapshot, "inventory_days_change"),
        "capex_to_cfo": _safe_div(capex, cfo) if cfo is not None and cfo > 0 else None,
        "fixed_asset_turnover_change": _first_number(
            snapshot, "fixed_asset_turnover_change"
        ),
        "construction_growth_supported": snapshot.get(
            "construction_growth_supported"
        ),
        "distribution_coverage": _first_number(snapshot, "distribution_coverage"),
        "share_dilution_yoy": _first_number(snapshot, "share_dilution_yoy"),
        "retained_returns_improving": _retained_returns_improving(snapshot),
        "audit_clean_enough": (
            None
            if not audit_opinion or audit_opinion in {"UNKNOWN", "N/A"}
            else audit_opinion not in {"ADVERSE", "DISCLAIMER"}
        ),
        "no_material_issue": None if material_issue is None else not material_issue,
        "no_going_concern": None if going_concern is None else not going_concern,
        "filing_traceable": _filing_traceable(snapshot),
    }


def _extended_not_applicable(
    snapshot: dict[str, Any],
    values: dict[str, object],
) -> dict[str, str]:
    reasons: dict[str, str] = {}
    cfo = _number(values.get("cfo_ttm"))
    if cfo is not None and cfo <= 0:
        reasons["fcf_to_cfo"] = "BLOCKED_BY_NON_POSITIVE_CFO"
        reasons["capex_to_cfo"] = "BLOCKED_BY_NON_POSITIVE_CFO"
    net_income = _first_number(snapshot, "net_income_ttm")
    if net_income is not None and net_income <= 0:
        reasons["cash_conversion"] = "BLOCKED_BY_NON_POSITIVE_TTM_NET_INCOME"
    return reasons


def _base_confidence(
    snapshot: dict[str, Any],
    signal: pd.Timestamp,
    *,
    coverage: float,
    balance_valid: bool | None,
    point_in_time: bool,
) -> float:
    freshness = _freshness(snapshot, signal, "twse_financial_observed_at")
    validation = 100.0 if balance_valid is True and point_in_time else 50.0 if balance_valid is None and point_in_time else 0.0
    source = _text(snapshot.get("twse_financial_source"))
    derived_reliability = _number(
        snapshot.get("base_quality_feature_source_reliability")
    )
    source_reliability = (
        derived_reliability
        if derived_reliability is not None
        else 95.0
        if source.startswith("TWSE_OPENAPI")
        else 90.0
        if _trusted_source(snapshot)
        else 0.0
    )
    return _clamp(0.40 * coverage + 0.25 * freshness + 0.20 * validation + 0.15 * source_reliability)


def _extended_confidence(
    snapshot: dict[str, Any],
    signal: pd.Timestamp,
    *,
    coverage: float,
    critical_coverage: float,
    balance_valid: bool | None,
    point_in_time: bool,
) -> float:
    freshness = _freshness(snapshot, signal, "available_at")
    checks = (
        point_in_time,
        balance_valid is not False,
        _number(snapshot.get("capex_ttm")) is None
        or _number(snapshot.get("capex_ttm")) >= 0,
        _text(snapshot.get("statement_scope")) in {"consolidated", "individual"},
        _number(snapshot.get("revision_number")) is not None,
        bool(_text(snapshot.get("derivation_version"))),
    )
    validation = 100.0 * sum(checks) / len(checks)
    source_reliability = 100.0 if _trusted_source(snapshot) else 0.0
    return _clamp(
        0.45 * coverage
        + 0.20 * critical_coverage
        + 0.15 * freshness
        + 0.10 * validation
        + 0.10 * source_reliability
    )


def _critical_field_coverage(snapshot: dict[str, Any]) -> float:
    opinion = _text(snapshot.get("audit_opinion")).upper()
    fields = (
        (20, _first_number(snapshot, "operating_cash_flow_ttm", "cfo_ttm") is not None),
        (15, _first_number(snapshot, "capex_ttm") is not None),
        (10, _first_number(snapshot, "cash") is not None),
        (15, _first_number(snapshot, "total_debt") is not None),
        (15, _first_number(snapshot, "interest_expense_ttm") is not None),
        (15, bool(opinion and opinion not in {"UNKNOWN", "N/A"})),
        (10, _filing_traceable(snapshot) is True),
    )
    return float(sum(weight for weight, available in fields if available))


def _classify_base(
    *,
    score: float | None,
    confidence: float,
    available_weight: float,
    hard_rejects: list[str],
    integrity_valid: bool,
    provisional_only: bool,
    config: LowEntryV5QualityConfig,
) -> BaseQualityStatus:
    if hard_rejects:
        return BaseQualityStatus.REJECT
    if not integrity_valid or available_weight < config.minimum_base_weight or confidence < 50 or score is None:
        return BaseQualityStatus.UNKNOWN
    if score < 55 and confidence >= 70:
        return BaseQualityStatus.REJECT
    if (
        score >= 70
        and confidence >= 70
        and available_weight >= config.minimum_base_pass_weight
    ):
        return (
            BaseQualityStatus.PROVISIONAL
            if provisional_only
            else BaseQualityStatus.PASS
        )
    if score >= 70 and confidence >= 50:
        return BaseQualityStatus.PROVISIONAL
    if score >= 55 and confidence >= 65:
        return BaseQualityStatus.WATCH
    return BaseQualityStatus.UNKNOWN


def _classify_extended(
    *,
    score: float | None,
    confidence: float,
    available_weight: float,
    critical_coverage: float,
    hard_rejects: list[str],
    config: LowEntryV5QualityConfig,
) -> ExtendedQualityStatus:
    if hard_rejects:
        return ExtendedQualityStatus.REJECT
    if available_weight < config.minimum_extended_weight or score is None or confidence < 30:
        return ExtendedQualityStatus.UNKNOWN
    if score >= 70 and confidence >= 70 and critical_coverage >= 70:
        return ExtendedQualityStatus.PASS
    if score >= 70 and confidence >= 50 and critical_coverage >= 50:
        return ExtendedQualityStatus.PROVISIONAL
    if confidence >= 30 and critical_coverage >= 30:
        return ExtendedQualityStatus.LIMITED
    return ExtendedQualityStatus.UNKNOWN


def _classify_final(
    base: BaseQualityStatus,
    extended: ExtendedQualityStatus,
    final_score: float | None,
) -> FinalQualityStatus:
    if base is BaseQualityStatus.REJECT or extended is ExtendedQualityStatus.REJECT:
        return FinalQualityStatus.HARD_NO_BUY
    if base is BaseQualityStatus.UNKNOWN:
        return FinalQualityStatus.DATA_INSUFFICIENT
    if base is BaseQualityStatus.WATCH:
        return FinalQualityStatus.WATCH
    if base is BaseQualityStatus.PROVISIONAL:
        return FinalQualityStatus.WATCH
    if base is BaseQualityStatus.PASS and extended is ExtendedQualityStatus.PASS:
        return FinalQualityStatus.CONFIRMED if final_score is not None and final_score >= 70 else FinalQualityStatus.WATCH
    if base is BaseQualityStatus.PASS and extended is ExtendedQualityStatus.PROVISIONAL:
        return FinalQualityStatus.PROVISIONAL if final_score is not None and final_score >= 70 else FinalQualityStatus.WATCH
    return FinalQualityStatus.BASE_ONLY


def _base_hard_rejects(snapshot: dict[str, Any]) -> list[str]:
    equity = _first_number(snapshot, "twse_parent_equity", "equity", "total_equity")
    rejects: list[str] = []
    if equity is not None and equity <= 0:
        rejects.append("NON_POSITIVE_PARENT_EQUITY")
    if _flag(snapshot.get("permanent_business_impairment")) is True:
        rejects.append("CONFIRMED_PERMANENT_BUSINESS_IMPAIRMENT")
    return rejects


def _extended_hard_rejects(
    annual: pd.DataFrame,
    snapshot: dict[str, Any],
) -> list[str]:
    rejects: list[str] = []
    if _latest_years_all_negative(annual, "operating_cash_flow", 3):
        rejects.append("LONG_TERM_NEGATIVE_CFO")
    if _latest_years_all_negative(annual, "free_cash_flow", 3):
        rejects.append("LONG_TERM_NEGATIVE_FCF")
    interest_coverage = _first_number(snapshot, "interest_coverage")
    if interest_coverage is not None and interest_coverage < 1:
        rejects.append("INTEREST_COVERAGE_BELOW_1")
    debt_growth = _first_number(snapshot, "interest_bearing_debt_growth_yoy")
    revenue_growth = _first_number(snapshot, "revenue_growth_ttm_yoy")
    if debt_growth is not None and revenue_growth is not None and debt_growth > revenue_growth + 0.20:
        rejects.append("INTEREST_BEARING_DEBT_RAPID_INCREASE")
    opinion = _text(snapshot.get("audit_opinion")).upper()
    if opinion == "ADVERSE":
        rejects.append("ADVERSE_AUDIT_OPINION")
    if opinion == "DISCLAIMER":
        rejects.append("DISCLAIMER_OF_OPINION")
    if _flag(snapshot.get("material_accounting_issue")) is True:
        rejects.append("UNRESOLVED_MATERIAL_RESTATEMENT")
    if _flag(snapshot.get("going_concern")) is True:
        rejects.append("GOING_CONCERN_MATERIAL_UNCERTAINTY")
    return rejects


def _balance_sheet_valid(
    snapshot: dict[str, Any],
    limit: float,
) -> bool | None:
    assets = _first_number(snapshot, "twse_total_assets", "total_assets")
    liabilities = _first_number(snapshot, "twse_total_liabilities", "total_liabilities")
    equity = _first_number(snapshot, "twse_total_equity", "total_equity", "equity")
    if assets is None or liabilities is None or equity is None or assets <= 0:
        return None
    return abs(assets - liabilities - equity) / assets < limit


def _normalized_score(
    metrics: tuple[QualityMetric, ...],
    *,
    minimum_weight: float,
) -> float | None:
    maximum = sum(metric.maximum for metric in metrics if metric.available)
    if maximum < minimum_weight or maximum <= 0:
        return None
    earned = sum(metric.earned for metric in metrics if metric.available)
    return round(100 * earned / maximum, 2)


def _combine_scores(
    base: float | None,
    extended: float | None,
    extended_weight: float,
) -> float | None:
    if base is None:
        return None
    if extended is None:
        return base
    return round(base * (1 - extended_weight) + extended * extended_weight, 2)


def _point_in_time_annual(
    annual_df: pd.DataFrame,
    signal: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(annual_df, pd.DataFrame) or annual_df.empty:
        return pd.DataFrame()
    annual = annual_df.copy()
    if "available_at" not in annual:
        return pd.DataFrame()
    available = pd.to_datetime(annual["available_at"], utc=True, errors="coerce")
    annual = annual.loc[available.notna() & (available <= signal)].copy()
    if "fiscal_year" in annual:
        annual = annual.sort_values("fiscal_year", kind="stable")
    else:
        annual = annual.assign(_available_at=available.loc[annual.index]).sort_values(
            "_available_at",
            kind="stable",
        )
        annual = annual.drop(columns="_available_at")
    return annual


def _freshness(
    snapshot: dict[str, Any],
    signal: pd.Timestamp,
    preferred_key: str,
) -> float:
    observed = pd.to_datetime(snapshot.get(preferred_key), utc=True, errors="coerce")
    if pd.isna(observed):
        observed = pd.to_datetime(snapshot.get("available_at"), utc=True, errors="coerce")
    if pd.isna(observed) or observed > signal:
        return 0.0
    age = (signal - observed).days
    if age <= 180:
        return 100.0
    if age <= 300:
        return 70.0
    if age <= 450:
        return 40.0
    return 0.0


def _trusted_source(snapshot: dict[str, Any]) -> bool:
    return _text(snapshot.get("availability_source")) in TRUSTED_AVAILABILITY_SOURCES


def _filing_traceable(snapshot: dict[str, Any]) -> bool | None:
    available = pd.to_datetime(snapshot.get("available_at"), utc=True, errors="coerce")
    revision = _number(snapshot.get("revision_number"))
    if pd.isna(available) and revision is None:
        return None
    return not pd.isna(available) and revision is not None


def _interest_expense_valid(snapshot: dict[str, Any]) -> bool | None:
    value = _first_number(snapshot, "interest_expense_ttm")
    if value is None:
        return None
    if value > 0:
        return True
    reported_zero = _flag(snapshot.get("interest_expense_reported_zero"))
    return False if reported_zero is True else None


def _retained_returns_improving(snapshot: dict[str, Any]) -> bool | None:
    roic_change = _first_number(snapshot, "roic_change_yoy")
    eps_growth = _first_number(snapshot, "eps_growth_ttm_yoy")
    if roic_change is None and eps_growth is None:
        return None
    return bool((roic_change is not None and roic_change > 0) or (eps_growth is not None and eps_growth > 0))


def _positive_year_ratio(annual: pd.DataFrame, column: str) -> float | None:
    if annual.empty or column not in annual:
        return None
    values = pd.to_numeric(annual[column], errors="coerce").dropna().tail(3)
    if values.empty:
        return None
    return float((values > 0).mean())


def _latest_years_all_negative(
    annual: pd.DataFrame,
    column: str,
    years: int,
) -> bool:
    if annual.empty or column not in annual:
        return False
    values = pd.to_numeric(annual[column], errors="coerce").dropna().tail(years)
    return len(values) >= years and bool((values < 0).all())


def _first_number(snapshot: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(snapshot.get(key))
        if value is not None:
            return value
    return None


def _number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) and parsed not in {float("inf"), float("-inf")} else None


def _flag(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = _text(value).lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _text(value: object) -> str:
    return "" if value is None or _missing(value) else str(value).strip()


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def _clamp(value: float) -> float:
    return min(100.0, max(0.0, value))


def _insufficient_result(reason: str) -> LowEntryV5QualityResult:
    return LowEntryV5QualityResult(
        base_score=None,
        base_data_confidence=0,
        base_available_weight=0,
        base_status=BaseQualityStatus.UNKNOWN,
        extended_score=None,
        extended_data_confidence=0,
        extended_available_weight=0,
        critical_field_coverage=0,
        extended_status=ExtendedQualityStatus.UNKNOWN,
        effective_extended_weight=0,
        final_score=None,
        final_status=FinalQualityStatus.DATA_INSUFFICIENT,
        base_metrics=(),
        extended_metrics=(),
        hard_rejects=(),
        warnings=(reason,),
    )


def _unsupported_result() -> LowEntryV5QualityResult:
    result = _insufficient_result("SECTOR_UNSUPPORTED")
    return replace(result, final_status=FinalQualityStatus.SECTOR_UNSUPPORTED)


__all__ = [
    "BaseQualityStatus",
    "ExtendedQualityStatus",
    "FinalQualityStatus",
    "LowEntryV5QualityConfig",
    "LowEntryV5QualityResult",
    "QualityMetric",
    "evaluate_v5_quality",
]
