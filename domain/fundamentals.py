"""Point-in-time Quality Score and Data Confidence for Low Entry Model v4."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import pandas as pd

from domain.data_provenance import TRUSTED_AVAILABILITY_SOURCES
from domain.industry_profiles import ELECTRONICS_PROFILE
from domain.industry_profiles import IndustryProfileMode
from domain.industry_profiles import IndustryProfileRegistry
from domain.industry_profiles import IndustryQualityProfile


ELECTRONIC_INDUSTRY_CODES = ELECTRONICS_PROFILE.sector_codes
FILING_CALENDAR_RULE_VERSION = "twse-general-2026-v1"
_EXPECTED_PERIOD_CUTOFFS = (
    (4, 1, -1, 4),
    (5, 16, 0, 1),
    (8, 15, 0, 2),
    (11, 15, 0, 3),
)
_MODIFIED_AUDIT_OPINIONS = {
    "ADVERSE",
    "DISCLAIMER",
    "QUALIFIED",
    "QUALIFIED_MATERIAL_MISSTATEMENT",
    "QUALIFIED_SCOPE_LIMITATION",
    "QUALIFIED_UNKNOWN",
}


class GateStatus(str, Enum):
    PASS = "PASS"
    PROVISIONAL = "PROVISIONAL"
    WATCH_QUALITY = "WATCH_QUALITY"
    REJECT = "REJECT"
    FAIL = "REJECT"  # Backward-compatible alias.
    UNKNOWN = "UNKNOWN"
    SECTOR_UNSUPPORTED = "SECTOR_UNSUPPORTED"


@dataclass(frozen=True)
class QualityGateConfig:
    min_history_years: int = 3
    max_net_debt_to_ebitda: float = 3.0
    min_interest_coverage: float = 3.0
    pass_score: float = 70.0
    fail_score: float = 55.0
    minimum_data_confidence: float = 50.0
    pass_data_confidence: float = 70.0
    minimum_available_metric_weight: float = 25.0
    qualified_scope_limitation_dc_penalty: float = 10.0
    audit_opinion_unknown_dc_penalty: float = 8.0
    unknown_audit_flag_dc_penalty: float = 4.0


@dataclass(frozen=True)
class MetricScore:
    name: str
    category: str
    earned: float
    maximum: float
    available: bool
    reason: str


@dataclass
class QualityGateResult:
    status: GateStatus
    score: float | None
    data_confidence: float = 0.0
    available_metric_weight: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)
    metric_scores: list[MetricScore] = field(default_factory=list)
    confidence_components: dict[str, float] = field(default_factory=dict)
    hard_red_flags: list[str] = field(default_factory=list)
    qa_flags: list[str] = field(default_factory=list)
    industry_profile: str = "UNSUPPORTED"
    industry_profile_version: str = "N/A"
    industry_profile_mode: str = "UNSUPPORTED"


def safe_div(numerator: float, denominator: float) -> float | None:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return None
    value = float(numerator / denominator)
    return value if math.isfinite(value) else None


def cagr(first: float, last: float, years: int) -> float | None:
    if years <= 0 or pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0:
        return None
    return float((last / first) ** (1 / years) - 1)


def flag_is_true(value: Any) -> bool:
    return parse_flag(value) is True


def parse_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def scalar_is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def evaluate_quality_gate(
    annual_df: pd.DataFrame,
    snapshot: dict[str, Any],
    signal_date: str | pd.Timestamp,
    config: QualityGateConfig | None = None,
) -> QualityGateResult:
    """Compute company quality independently from data confidence.

    Missing values remove only their metric weight. They lower Data Confidence
    and never become zero, false, a hard red flag, or an implicit PASS.
    """

    cfg = config or QualityGateConfig()
    parsed_signal = pd.to_datetime(signal_date, utc=True, errors="coerce")
    if pd.isna(parsed_signal):
        return QualityGateResult(
            status=GateStatus.UNKNOWN,
            score=None,
            reasons=["訊號日期無效"],
            qa_flags=["INVALID_SIGNAL_DATE"],
        )

    sector_value = snapshot.get("sector")
    sector = "" if sector_value is None else str(sector_value).strip()
    sector_known = bool(sector)
    industry_profile = IndustryProfileRegistry.resolve(sector)
    sector_supported = industry_profile is not None
    scoring_profile = industry_profile or ELECTRONICS_PROFILE
    annual, annual_qa = _prepare_annual(annual_df, parsed_signal)
    snapshot_available_at = pd.to_datetime(
        snapshot.get("available_at"),
        utc=True,
        errors="coerce",
    )
    snapshot_source = str(snapshot.get("availability_source", "")).strip()
    snapshot_point_in_time = (
        not pd.isna(snapshot_available_at) and snapshot_available_at <= parsed_signal
    )
    snapshot_source_trusted = snapshot_source in TRUSTED_AVAILABILITY_SOURCES
    annual_sources_trusted = (
        not annual.empty
        and annual.get("availability_source", pd.Series(dtype="object"))
        .isin(TRUSTED_AVAILABILITY_SOURCES)
        .all()
    )

    qa_flags = list(annual_qa)
    if not snapshot_point_in_time:
        qa_flags.append("SNAPSHOT_NOT_POINT_IN_TIME")
    if not snapshot_source_trusted:
        qa_flags.append("UNTRUSTED_SNAPSHOT_SOURCE")
    if not annual_sources_trusted:
        qa_flags.append("UNTRUSTED_ANNUAL_SOURCE")
    if len(annual) < cfg.min_history_years:
        qa_flags.append("INSUFFICIENT_ANNUAL_HISTORY")
    if not sector_known:
        qa_flags.append("OFFICIAL_SECTOR_UNKNOWN")
    elif not sector_supported:
        qa_flags.append("SECTOR_UNSUPPORTED")
    elif industry_profile.mode is IndustryProfileMode.SHADOW:
        qa_flags.append("INDUSTRY_PROFILE_SHADOW_MODE")

    values = _quality_values(annual, snapshot)
    metric_scores = _score_quality_metrics(values, scoring_profile)
    available_metric_weight = sum(
        metric.maximum for metric in metric_scores if metric.available
    )
    earned_points = sum(metric.earned for metric in metric_scores if metric.available)
    quality_score = (
        round(100.0 * earned_points / available_metric_weight, 2)
        if available_metric_weight > 0
        else None
    )

    hard_red_flags = _hard_red_flags(
        annual,
        snapshot,
        values,
        cfg,
        decision_grade=snapshot_point_in_time and snapshot_source_trusted,
        annual_sources_trusted=annual_sources_trusted,
    )
    quality_risk_flags = _quality_risk_flags(
        annual,
        values,
        cfg,
        scoring_profile,
        annual_sources_trusted=annual_sources_trusted,
    )
    qa_flags.extend(quality_risk_flags)
    confidence_components, confidence_qa = _data_confidence_components(
        annual,
        snapshot,
        parsed_signal,
        snapshot_point_in_time=snapshot_point_in_time,
        snapshot_source_trusted=snapshot_source_trusted,
        annual_sources_trusted=annual_sources_trusted,
    )
    qa_flags.extend(confidence_qa)
    data_confidence = _compute_data_confidence(confidence_components)

    opinion = str(snapshot.get("audit_opinion", "")).strip().upper()
    if opinion == "QUALIFIED_SCOPE_LIMITATION":
        qa_flags.append("QUALIFIED_SCOPE_LIMITATION")
        data_confidence -= cfg.qualified_scope_limitation_dc_penalty
    elif opinion in {"QUALIFIED", "QUALIFIED_UNKNOWN"}:
        qa_flags.append("QUALIFIED_OPINION_REASON_UNKNOWN")
        data_confidence = min(data_confidence, cfg.pass_data_confidence - 1)
    elif opinion == "UNKNOWN":
        qa_flags.append("AUDIT_OPINION_UNKNOWN")
        if (
            str(snapshot.get("audit_opinion_source", "")).strip().upper()
            == "ABSTRACT_CONFLICT"
        ):
            qa_flags.append("AUDIT_ABSTRACT_CONFLICT")
            data_confidence = min(
                data_confidence,
                cfg.pass_data_confidence - 1,
            )
        else:
            # Missing or unclassifiable audit text is an evidence gap, not
            # affirmative proof of a modified opinion.
            data_confidence -= cfg.audit_opinion_unknown_dc_penalty

    comparative_raw = snapshot.get("comparative_audit_opinion")
    comparative_opinion = (
        ""
        if scalar_is_missing(comparative_raw)
        else str(comparative_raw).strip().upper()
    )
    if comparative_opinion in _MODIFIED_AUDIT_OPINIONS:
        # Comparative opinions remain visible for follow-up, but current-period
        # Quality Gate and hard-red-flag decisions use only the current opinion.
        qa_flags.append(
            "PRIOR_PERIOD_AUDIT_OPINION={}".format(comparative_opinion)
        )

    unknown_audit_flags = [
        name
        for name in ("going_concern", "material_accounting_issue")
        if parse_flag(snapshot.get(name)) is None
    ]
    if unknown_audit_flags:
        qa_flags.append("AUDIT_FLAGS_UNKNOWN={}".format(sorted(unknown_audit_flags)))
        data_confidence -= (
            cfg.unknown_audit_flag_dc_penalty * len(unknown_audit_flags)
        )
    if (
        not snapshot_point_in_time
        or not snapshot_source_trusted
        or not annual_sources_trusted
        or len(annual) < cfg.min_history_years
        or available_metric_weight < cfg.minimum_available_metric_weight
        or not sector_known
    ):
        data_confidence = min(data_confidence, cfg.minimum_data_confidence - 1)
    data_confidence = float(
        round(max(0.0, min(data_confidence, 100.0)), 2)
    )

    status = _determine_status(
        sector_supported=sector_supported,
        sector_known=sector_known,
        quality_score=quality_score,
        data_confidence=data_confidence,
        available_metric_weight=available_metric_weight,
        hard_red_flags=hard_red_flags,
        config=cfg,
        profile=scoring_profile,
    )
    if (
        status in {GateStatus.PASS, GateStatus.PROVISIONAL}
        and {
            "NON_POSITIVE_TTM_EPS",
            "NON_POSITIVE_THREE_YEAR_CFO",
        }.intersection(quality_risk_flags)
    ):
        status = GateStatus.WATCH_QUALITY
    reasons = _status_reasons(status, quality_score, data_confidence, hard_red_flags)
    warnings = list(dict.fromkeys(qa_flags))
    metrics = {
        **values,
        "coverage": confidence_components["coverage"],
        "freshness": confidence_components["freshness"],
        "validation": confidence_components["validation"],
        "source_reliability": confidence_components["source_reliability"],
    }
    return QualityGateResult(
        status=status,
        score=quality_score,
        data_confidence=data_confidence,
        available_metric_weight=round(available_metric_weight, 2),
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
        metric_scores=metric_scores,
        confidence_components=confidence_components,
        hard_red_flags=hard_red_flags,
        qa_flags=warnings,
        industry_profile=scoring_profile.name if sector_supported else "UNSUPPORTED",
        industry_profile_version=(
            scoring_profile.version if sector_supported else "N/A"
        ),
        industry_profile_mode=(
            scoring_profile.mode.value if sector_supported else "UNSUPPORTED"
        ),
    )


def _prepare_annual(
    annual_df: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    required_identity = {"fiscal_year", "available_at", "availability_source"}
    if annual_df.empty or not required_identity.issubset(annual_df.columns):
        return pd.DataFrame(), [
            "MISSING_ANNUAL_IDENTITY_FIELDS={}".format(
                sorted(required_identity - set(annual_df.columns))
            )
        ]
    annual = annual_df.copy()
    annual["available_at"] = pd.to_datetime(
        annual["available_at"],
        utc=True,
        errors="coerce",
    )
    annual["fiscal_year"] = pd.to_numeric(annual["fiscal_year"], errors="coerce")
    qa_flags: list[str] = []
    if annual["available_at"].isna().any():
        qa_flags.append("INVALID_ANNUAL_AVAILABLE_AT")
    if (annual["available_at"] > signal_date).any():
        qa_flags.append("FUTURE_ANNUAL_DATA_EXCLUDED")
    annual = annual.dropna(subset=["fiscal_year", "available_at"])
    annual = (
        annual[annual["available_at"] <= signal_date]
        .sort_values(["fiscal_year", "available_at"])
        .drop_duplicates("fiscal_year", keep="last")
    )
    for column in (
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "roic",
        "gross_margin",
        "operating_margin",
        "diluted_shares",
        "gross_profit",
        "ebit",
        "cash",
        "total_debt",
        "total_assets",
        "total_liabilities",
        "equity",
        "total_equity",
        "current_assets",
        "current_liabilities",
        "dividends_paid",
    ):
        if column not in annual:
            annual[column] = float("nan")
        annual[column] = pd.to_numeric(annual[column], errors="coerce")
    revenue = annual["revenue"].where(annual["revenue"].abs() > 0)
    annual["gross_margin"] = annual["gross_margin"].fillna(
        annual["gross_profit"] / revenue
    )
    return annual, qa_flags


def _quality_values(
    annual: pd.DataFrame,
    snapshot: dict[str, Any],
) -> dict[str, float | None]:
    recent = annual.tail(5)
    ttm_eps = _number(snapshot.get("ttm_eps"))
    ebit_ttm = _number(snapshot.get("ebit_ttm"))
    cfo_ttm = _number(snapshot.get("operating_cash_flow_ttm"))
    net_income_ttm = _number(snapshot.get("net_income_ttm"))
    cash = _number(snapshot.get("cash"))
    total_debt = _number(snapshot.get("total_debt"))
    ebitda_ttm = _number(snapshot.get("ebitda_ttm"))
    interest_expense = _number(snapshot.get("interest_expense_ttm"))
    cumulative_cfo_3y = _sum_complete(annual.tail(3).get("operating_cash_flow"), 3)
    cumulative_net_income_3y = _sum_complete(annual.tail(3).get("net_income"), 3)
    net_debt = (
        total_debt - cash if total_debt is not None and cash is not None else None
    )
    net_debt_to_ebitda = (
        safe_div(net_debt, ebitda_ttm)
        if net_debt is not None and ebitda_ttm is not None and ebitda_ttm > 0
        else 0.0
        if net_debt is not None and net_debt <= 0
        else None
    )
    interest_coverage = _number(snapshot.get("interest_coverage"), allow_infinite=True)
    if interest_coverage is None and ebit_ttm is not None and interest_expense is not None:
        if (
            interest_expense == 0
            and ebit_ttm > 0
            and parse_flag(snapshot.get("interest_expense_reported_zero")) is True
        ):
            interest_coverage = math.inf
        elif interest_expense > 0:
            interest_coverage = ebit_ttm / interest_expense
    latest_roic = _latest_number(recent, "roic")
    latest_margin = _latest_number(recent, "operating_margin")
    current_gross_margin = _coalesce(
        _number(snapshot.get("gross_margin_ttm")),
        safe_div(
            _number(snapshot.get("gross_profit_ttm")),
            _number(snapshot.get("revenue_ttm")),
        ),
        _latest_number(recent, "gross_margin"),
    )
    positive_cfo_ratio = _positive_ratio(recent.get("operating_cash_flow"))
    positive_income_ratio = _positive_ratio(recent.get("net_income"))
    return {
        "ttm_eps": ttm_eps,
        "operating_income_ttm": ebit_ttm,
        "positive_operating_quarters_8": _number(snapshot.get("positive_operating_quarters_8")),
        "roe_ttm": _number(snapshot.get("roe_ttm")),
        "roa_ttm": _number(snapshot.get("roa_ttm")),
        "roic_ttm": _coalesce(_number(snapshot.get("roic_ttm")), latest_roic),
        "roic_5y_median": _median(recent.get("roic")),
        "cfo_ttm": cfo_ttm,
        "cash_conversion": _coalesce(
            _number(snapshot.get("cash_conversion")),
            safe_div(cfo_ttm, net_income_ttm)
            if cfo_ttm is not None and net_income_ttm is not None and net_income_ttm > 0
            else None,
        ),
        "fcf_ttm": _number(snapshot.get("free_cash_flow_ttm")),
        "accrual_ratio": _number(snapshot.get("accrual_ratio")),
        "debt_ratio": _number(snapshot.get("debt_ratio")),
        "net_debt_to_equity": _number(snapshot.get("net_debt_to_equity")),
        "interest_coverage": interest_coverage,
        "current_ratio": _number(snapshot.get("current_ratio")),
        "cash_to_current_liabilities": _number(snapshot.get("cash_to_current_liabilities")),
        "revenue_growth_ttm_yoy": _coalesce(
            _number(snapshot.get("revenue_growth_ttm_yoy")),
            _annual_growth(recent, "revenue"),
        ),
        "operating_income_growth_ttm_yoy": _coalesce(
            _number(snapshot.get("operating_income_growth_ttm_yoy")),
            _annual_growth(recent, "ebit"),
        ),
        "positive_eps_quarters_8": _number(snapshot.get("positive_eps_quarters_8")),
        "gross_margin_change": _number(snapshot.get("gross_margin_change")),
        "share_dilution_yoy": _coalesce(
            _number(snapshot.get("share_dilution_yoy")),
            _annual_growth(recent, "diluted_shares"),
        ),
        "distribution_coverage": _number(snapshot.get("distribution_coverage")),
        "bvps_trend": _number(snapshot.get("bvps_trend")),
        "cumulative_cfo_3y": cumulative_cfo_3y,
        "cumulative_net_income_3y": cumulative_net_income_3y,
        "net_debt": net_debt,
        "net_debt_to_ebitda": net_debt_to_ebitda,
        "positive_cfo_ratio": positive_cfo_ratio,
        "positive_cfo_year_ratio": positive_cfo_ratio,
        "positive_income_ratio": positive_income_ratio,
        "latest_operating_margin": latest_margin,
        "gross_margin_5y_percentile": _percentile_rank(
            recent.get("gross_margin"),
            current_gross_margin,
        ),
        "operating_margin_trend": _recent_trend(
            recent.get("operating_margin"),
        ),
        "ebitda_ttm": ebitda_ttm,
    }


def _score_quality_metrics(
    values: dict[str, float | None],
    profile: IndustryQualityProfile,
) -> list[MetricScore]:
    return [
        _tier(
            rule.name,
            rule.category,
            values.get(rule.name),
            rule.maximum,
            rule.scorer,
            allow_infinite=rule.name == "interest_coverage",
        )
        for rule in profile.metric_rules
    ]


def _tier(
    name: str,
    category: str,
    value: float | None,
    maximum: float,
    rule: Callable[[float], float],
    *,
    allow_infinite: bool = False,
) -> MetricScore:
    if value is None or (not allow_infinite and not math.isfinite(value)):
        return MetricScore(name, category, 0.0, maximum, False, "missing_or_inapplicable")
    earned = min(max(float(rule(value)), 0.0), maximum)
    return MetricScore(name, category, earned, maximum, True, "value={}".format(value))


def _hard_red_flags(
    annual: pd.DataFrame,
    snapshot: dict[str, Any],
    values: dict[str, float | None],
    config: QualityGateConfig,
    *,
    decision_grade: bool,
    annual_sources_trusted: bool,
) -> list[str]:
    flags: list[str] = []
    if not decision_grade:
        return flags
    equity = _number(snapshot.get("equity"))
    if equity is not None and equity <= 0:
        flags.append("NON_POSITIVE_PARENT_EQUITY")
    opinion = str(snapshot.get("audit_opinion", "")).upper()
    if opinion == "ADVERSE":
        flags.append("ADVERSE_AUDIT_OPINION")
    if opinion == "DISCLAIMER":
        flags.append("DISCLAIMER_OF_OPINION")
    if opinion == "QUALIFIED_MATERIAL_MISSTATEMENT":
        flags.append("QUALIFIED_MATERIAL_MISSTATEMENT")
    if parse_flag(snapshot.get("going_concern")) is True:
        flags.append("GOING_CONCERN_MATERIAL_UNCERTAINTY")
    if parse_flag(snapshot.get("material_accounting_issue")) is True:
        flags.append("MATERIAL_ACCOUNTING_OR_FILING_CONCERN")
    return flags


def _quality_risk_flags(
    annual: pd.DataFrame,
    values: dict[str, float | None],
    config: QualityGateConfig,
    profile: IndustryQualityProfile,
    *,
    annual_sources_trusted: bool,
) -> list[str]:
    del annual, config
    flags: list[str] = []
    if values["ttm_eps"] is not None and values["ttm_eps"] <= 0:
        flags.append("NON_POSITIVE_TTM_EPS")
    if (
        annual_sources_trusted
        and values["cumulative_cfo_3y"] is not None
        and values["cumulative_cfo_3y"] <= 0
    ):
        flags.append("NON_POSITIVE_THREE_YEAR_CFO")
    if (
        values["net_debt_to_ebitda"] is not None
        and values["net_debt_to_ebitda"] > profile.max_net_debt_to_ebitda
    ):
        flags.append(
            "NET_DEBT_TO_EBITDA_ABOVE_{}".format(
                _threshold_label(profile.max_net_debt_to_ebitda)
            )
        )
    if (
        values["net_debt"] is not None
        and values["net_debt"] > 0
        and values["ebitda_ttm"] is not None
        and values["ebitda_ttm"] <= 0
    ):
        flags.append("NET_DEBT_TO_EBITDA_NOT_MEANINGFUL")
    if (
        values["interest_coverage"] is not None
        and values["interest_coverage"] < profile.min_interest_coverage
    ):
        flags.append(
            "INTEREST_COVERAGE_BELOW_{}".format(
                _threshold_label(profile.min_interest_coverage)
            )
        )
    return flags


def _data_confidence_components(
    annual: pd.DataFrame,
    snapshot: dict[str, Any],
    signal_date: pd.Timestamp,
    *,
    snapshot_point_in_time: bool,
    snapshot_source_trusted: bool,
    annual_sources_trusted: bool,
) -> tuple[dict[str, float], list[str]]:
    metadata = (
        "available_at",
        "availability_source",
        "filing_id",
        "revision_number",
        "content_sha256",
        "source_url",
        "derivation_version",
        "period_end",
        "statement_scope",
        "audit_status",
    )
    income = (
        "ttm_eps",
        "revenue_ttm",
        "net_income_ttm",
        "ebit_ttm",
        "ebitda_ttm",
        "gross_profit_ttm",
    )
    balance = (
        "cash",
        "total_debt",
        "total_assets",
        "total_liabilities",
        "equity",
        "total_equity",
        "current_assets",
        "current_liabilities",
    )
    cash_flow = ("operating_cash_flow_ttm", "capex_ttm", "free_cash_flow_ttm")
    audit = (
        "going_concern",
        "material_accounting_issue",
        "audit_opinion",
        "audit_evidence_available",
    )
    coverage = (
        .20 * _field_coverage(snapshot, metadata)
        + .25 * _field_coverage(snapshot, income)
        + .25 * _field_coverage(snapshot, balance)
        + .20 * _field_coverage(snapshot, cash_flow)
        + .10 * _field_coverage(snapshot, audit)
    )
    freshness = _freshness_score(snapshot, annual, signal_date)
    checks: list[tuple[str, bool]] = [
        ("snapshot_point_in_time", snapshot_point_in_time),
        ("snapshot_source_trusted", snapshot_source_trusted),
        ("annual_sources_trusted", annual_sources_trusted),
        ("annual_history", len(annual) >= 3),
        ("consolidated_scope", str(snapshot.get("statement_scope", "consolidated")) == "consolidated"),
    ]
    qa_flags: list[str] = []
    assets = _number(snapshot.get("total_assets"))
    liabilities = _number(snapshot.get("total_liabilities"))
    total_equity = _number(snapshot.get("total_equity"))
    major_validation_failure = False
    if assets is not None and liabilities is not None and total_equity is not None:
        balance_ok = abs(assets - liabilities - total_equity) / max(abs(assets), 1) <= .01
        checks.append(("balance_sheet_identity", balance_ok))
        if not balance_ok:
            qa_flags.append("BALANCE_SHEET_NOT_BALANCED")
            major_validation_failure = True
    validation = 100.0 * sum(passed for _, passed in checks) / len(checks)
    if major_validation_failure:
        validation = min(validation, 40.0)
    source_reliability = 100.0 if snapshot_source_trusted else 40.0 if snapshot.get("source_url") else 0.0
    return (
        {
            "coverage": float(round(coverage, 2)),
            "freshness": float(round(freshness, 2)),
            "validation": float(round(validation, 2)),
            "source_reliability": float(round(source_reliability, 2)),
        },
        qa_flags,
    )


def _compute_data_confidence(components: dict[str, float]) -> float:
    return (
        .40 * components["coverage"]
        + .25 * components["freshness"]
        + .25 * components["validation"]
        + .10 * components["source_reliability"]
    )


def _determine_status(
    *,
    sector_supported: bool,
    sector_known: bool,
    quality_score: float | None,
    data_confidence: float,
    available_metric_weight: float,
    hard_red_flags: list[str],
    config: QualityGateConfig,
    profile: IndustryQualityProfile,
) -> GateStatus:
    if sector_known and not sector_supported:
        return GateStatus.SECTOR_UNSUPPORTED
    if hard_red_flags:
        return GateStatus.REJECT
    if (
        not sector_known
        or quality_score is None
        or available_metric_weight < config.minimum_available_metric_weight
        or data_confidence < config.minimum_data_confidence
    ):
        return GateStatus.UNKNOWN
    if quality_score < profile.fail_score:
        return GateStatus.REJECT
    if quality_score < profile.pass_score:
        return GateStatus.WATCH_QUALITY
    if data_confidence < config.pass_data_confidence:
        return GateStatus.PROVISIONAL
    return GateStatus.PASS


def _status_reasons(
    status: GateStatus,
    score: float | None,
    data_confidence: float,
    hard_red_flags: list[str],
) -> list[str]:
    if status == GateStatus.SECTOR_UNSUPPORTED:
        return ["產業模型不支援"]
    if hard_red_flags:
        return list(hard_red_flags)
    score_text = "N/A" if score is None else "{:.1f}".format(score)
    labels = {
        GateStatus.PASS: "品質與資料可信度通過",
        GateStatus.PROVISIONAL: "品質通過但資料可信度僅供觀察",
        GateStatus.WATCH_QUALITY: "品質分數需觀察",
        GateStatus.REJECT: "品質分數未達最低門檻",
        GateStatus.UNKNOWN: "資料可信度或可用指標權重不足",
    }
    return ["{}；QS={}；DC={:.1f}".format(labels[status], score_text, data_confidence)]


def _field_coverage(snapshot: dict[str, Any], names: tuple[str, ...]) -> float:
    usable = sum(
        bool(snapshot.get(name))
        if name == "audit_evidence_available"
        else not _value_missing(snapshot.get(name))
        for name in names
    )
    return 100.0 * usable / len(names)


def _value_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().upper() in {"UNKNOWN", "N/A"}
    return scalar_is_missing(value)


def _freshness_score(
    snapshot: dict[str, Any],
    annual: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> float:
    period = pd.to_datetime(snapshot.get("period_end"), errors="coerce")
    if pd.isna(period) and not annual.empty:
        period = pd.Timestamp(year=int(annual.iloc[-1]["fiscal_year"]), month=12, day=31)
    if pd.isna(period):
        return 0.0
    expected_year, expected_quarter = _expected_period(signal_date.tz_convert("Asia/Taipei"))
    actual_quarter = (period.month - 1) // 3 + 1
    lag = (expected_year * 4 + expected_quarter) - (period.year * 4 + actual_quarter)
    if lag <= 0:
        return 100.0
    return {1: 60.0, 2: 30.0}.get(lag, 0.0)


def _expected_period(signal_date: pd.Timestamp) -> tuple[int, int]:
    current = signal_date.date()
    year = current.year
    selected = (year - 1, 3)
    for month, day, year_offset, quarter in _EXPECTED_PERIOD_CUTOFFS:
        if current >= dt.date(year, month, day):
            selected = year + year_offset, quarter
    return selected


def _number(value: Any, *, allow_infinite: bool = False) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    if scalar_is_missing(parsed):
        return None
    result = float(parsed)
    if not allow_infinite and not math.isfinite(result):
        return None
    return result


def _coalesce(*values: float | None) -> float | None:
    return next((value for value in values if value is not None), None)


def _latest_number(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else None


def _sum_complete(series: pd.Series | None, required: int) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce")
    if len(values) < required or values.tail(required).isna().any():
        return None
    return float(values.tail(required).sum())


def _positive_ratio(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((values > 0).mean()) if not values.empty else None


def _median(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def _percentile_rank(
    series: pd.Series | None,
    current: float | None,
) -> float | None:
    if series is None or current is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float((values <= current).mean())


def _recent_trend(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna().tail(3)
    if len(values) < 2:
        return None
    return float((values.iloc[-1] - values.iloc[0]) / (len(values) - 1))


def _threshold_label(value: float) -> str:
    return "{:g}".format(value).replace(".", "_")


def _annual_growth(frame: pd.DataFrame, column: str) -> float | None:
    if len(frame) < 2 or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if len(values) < 2 or values.iloc[-2] <= 0:
        return None
    return float(values.iloc[-1] / values.iloc[-2] - 1)


__all__ = [
    "FILING_CALENDAR_RULE_VERSION",
    "GateStatus",
    "MetricScore",
    "QualityGateConfig",
    "QualityGateResult",
    "cagr",
    "evaluate_quality_gate",
    "flag_is_true",
    "parse_flag",
    "safe_div",
    "scalar_is_missing",
]
