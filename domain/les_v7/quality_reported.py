"""Versioned R3/R4 projections of one source-bound reported Quality engine.

The native evaluator owns the fixed 100-point budget, missing-data deduction
and measured DC ledger. TTM/debt/ROIC diagnostics are retained, not relabelled
as reported values or promoted into new core requirements. This adapter does
not authorize orders, manufacture audit opinions or alter R1/R2 policies.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields
from math import isclose, isfinite

import pandas as pd

from domain.les_v7.contracts import finite, identity, timestamp
from domain.strategies.low_entry_v6_quality import (
    BALANCE_FIELDS, QUALITY_BUDGETS, QUALITY_DC_WEIGHTS,
    QUALITY_MISSING_PENALTY_RATE, QUALITY_POLICY_VERSION,
    QUALITY_PROFITABILITY_PARTS, QUALITY_RETURN_WEIGHTS, QUALITY_GROWTH_WEIGHTS,
    QUALITY_REQUIRED_REPORTED_FIELDS, QUALITY_SCORING_POLICY, OFFICIAL_SOURCES, QualityResult,
    evaluate_quality, quality_evidence_complete, select_quality_inputs, quality_scoring_rules,
    QUALITY_R4_POLICY_VERSION,
)
from domain.strategies.low_entry_v6_quality_evidence import verification_summary, validate_field


PROFILE = "QG_V7_REPORTED_PERIOD_R3"
MAPPING_VERSION = "QG_MAP_R3"
R4_PROFILE = "QG_V7_REPORTED_PERIOD_R4"
R5_PROFILE = "QG_V7_REPORTED_PERIOD_R5"
R6_PROFILE = "QG_V7_REPORTED_PERIOD_R6"
R7_PROFILE = "QG_V7_REPORTED_PERIOD_R7"
_LIMITED_GAPS_PROFILES = frozenset((R4_PROFILE, R5_PROFILE, R6_PROFILE, R7_PROFILE))
SCORING_BASIS = "REPORTED_SAME_PERIOD"
_AUDIT_FIELDS = ("auditor_opinion", "audit_review_required", "going_concern", "material_accounting_issue")


def _profile_rules(policy):
    profile = policy.get("profile_id") if isinstance(policy, Mapping) else None
    if profile == R4_PROFILE:
        return R4_PROFILE, "QG_MAP_R4", QUALITY_R4_POLICY_VERSION
    if profile == R5_PROFILE:
        return R5_PROFILE, "QG_MAP_R5", QUALITY_R4_POLICY_VERSION
    if profile == R6_PROFILE:
        return R6_PROFILE, "QG_MAP_R6", QUALITY_R4_POLICY_VERSION
    if profile == R7_PROFILE:
        return R7_PROFILE, "QG_MAP_R7", QUALITY_R4_POLICY_VERSION
    return PROFILE, MAPPING_VERSION, QUALITY_POLICY_VERSION


def _financial_source(latest, profile):
    """R4 adds bounded original operands, never an issuer's historical graph."""
    from domain.les_v7.quality_policy import financial_safety_source
    source = financial_safety_source(latest)
    if profile in _LIMITED_GAPS_PROFILES:
        fields = ("revenue", "cost_of_revenue", "operating_income", "net_income")
        for field in fields:
            source[field] = latest.get(field)
            source["field_evidence"][field] = deepcopy(latest.get("field_evidence", {}).get(field, {}))
        for group in ("comparative_ytd", "comparative_quarter"):
            source[group + "_fields"] = {field: latest.get(group + "_fields", {}).get(field) for field in fields}
            source[group + "_field_evidence"] = {
                field: deepcopy(latest.get(group + "_field_evidence", {}).get(field, {})) for field in fields}
    return source


def _native_metric_budgets():
    metrics = dict(balance_sheet_identity=("balance_sheet", 6.), debt_ratio=("balance_sheet", 10.),
                   current_ratio=("balance_sheet", 9.))
    for name, weight in (("gross_margin", 6.), ("operating_margin", 8.), ("net_margin", 6.)):
        metrics.update((name + "_" + part, ("profitability", weight * fraction))
                       for part, fraction in QUALITY_PROFITABILITY_PARTS.items())
    metrics.update((field + "_growth", ("growth", weight)) for field, weight in QUALITY_GROWTH_WEIGHTS.items())
    metrics.update((name, ("return_on_capital", weight)) for name, weight in QUALITY_RETURN_WEIGHTS.items())
    metrics.update((name, ("cash_flow", 5.)) for name in ("cfo_positive", "cash_conversion", "fcf_positive"))
    return metrics


def _empty(reason, policy, *, status="DATA_INSUFFICIENT", cutoff=None):
    try:
        cutoff_at = timestamp(cutoff).isoformat()
    except (ValueError, TypeError, AttributeError):
        cutoff_at = None
    profile, mapping, _ = _profile_rules(policy)
    return dict(profile_id=profile, mapping_version=mapping,
        quality_policy_id=identity(policy), resolved_policy=policy,
        cutoff_at=cutoff_at, financial_safety_source=None,
        status=status, quality_score=None, earned_points=None,
        missing_penalty_points=None, data_confidence=None, coverage_total=None,
        coverage_by_domain={}, metric_ledger=[], gap_groups=[],
        gap_qualifier="NOT_EVALUATED", reason_codes=[reason], warnings=[],
        core_ttm_pass=False, core_reported_pass=False, core_inputs_pass=False,
        native_quality_complete=False, mandatory_safety_pass=False,
        core_source_basis=SCORING_BASIS, scoring_basis=SCORING_BASIS,
        quality_les_cap=None, core_ttm={}, core_reported={}, field_checks={},
        financial_safety=dict(state="UNKNOWN", reasons=[reason]),
        ttm_traces={}, reported_result=None,
        readiness=dict(financial="NOT_READY"))


def _source_complete(result):
    """Only verified native core receipts qualify; a finite number is not proof."""
    reported = result.metrics.get("quality_reported_inputs", {})
    ledger = result.metrics.get("quality_field_checks", {})
    if not isinstance(reported, Mapping) or not isinstance(ledger, Mapping):
        return False
    for field in QUALITY_REQUIRED_REPORTED_FIELDS:
        key = reported.get("current_keys", {}).get(field)
        check = ledger.get(key, {})
        if (reported.get("current", {}).get(field) is None
                or check.get("value") != reported.get("current", {}).get(field)
                or check.get("status") not in {"VERIFIED_REPORTED", "VERIFIED_DERIVED"}
                or check.get("evidence_level") != "CANONICAL_ORIGINAL"
                or not check.get("checks") or not all(check["checks"].values())):
            return False
    return True


def _metric_ledger(result):
    """Expose native metric budgets without masquerading as R1/R2's 19 curves."""
    return [dict(metric_id=name, domain=node["group"], budget=node["weight"],
        mapped_score=node["fraction"], earned_points=node["points"],
        value_status=node["status"], available=node["available"])
        for name, node in sorted(result.metrics.get("quality_metric_checks", {}).items())]


def _audited_view(records, latest, safety):
    """Shallow local scalar overlay; shared raw proofs are never rewritten.

    The old generic mapper's UNKNOWN must not overwrite a newer source-bound
    classification. Conversely missing proof cannot be turned into UNMODIFIED
    or false risk flags. The native safety receipt supplies the classification
    only after rebuilding its original contexts and cutoff binding.
    """
    classification = safety.get("audit_classification")
    view = []
    for record in records:
        local = dict(record)
        if all(local.get(key) == latest.get(key) for key in
               ("symbol", "market", "period_end", "available_at", "content_sha256", "revision_id")):
            for field in _AUDIT_FIELDS:
                local.pop(field, None)
            if isinstance(classification, Mapping):
                local["auditor_opinion"] = classification.get("audit_opinion")
                for field in _AUDIT_FIELDS[1:]:
                    local[field] = classification.get(field)
        view.append(local)
    return view


def _status(native, safety, core_reported_pass, complete):
    status = native.status
    if safety.get("state") == "FAIL":
        return "REJECT"
    if status != "REJECT" and (safety.get("state") != "PASS" or not core_reported_pass):
        return "DATA_INSUFFICIENT"
    if status == "PROVISIONAL" or status == "PASS" and not complete:
        return "DATA_INSUFFICIENT"
    return status


def validate_reported_result(result):
    """Validate R3's business arithmetic, not merely an envelope content hash.

    Insufficient results may have partial/no native output, but cannot invent a
    score, covered budget, verification credit or a PASS source-readiness flag.
    R1/R2 reports never enter this validator.
    """
    try:
        profile, mapping, native_version = _profile_rules(result)
        _, native_scoring, coverage_minima = quality_scoring_rules(native_version)
        if (result.get("profile_id") != profile or result.get("mapping_version") != mapping
                or result.get("scoring_basis") != SCORING_BASIS or result.get("core_source_basis") != SCORING_BASIS):
            raise ValueError("PROFILE_OR_BASIS")
        stored = result.get("reported_result")
        if stored is None:
            if (result.get("status") not in {"DATA_INSUFFICIENT", "CONFIG_INCOMPLETE"}
                    or any(result.get(key) is not None for key in
                           ("quality_score", "earned_points", "missing_penalty_points", "data_confidence", "coverage_total"))
                    or result.get("coverage_by_domain") != {} or result.get("metric_ledger") != []
                    or any(result.get(key) is not False for key in
                           ("core_reported_pass", "core_inputs_pass", "core_ttm_pass", "native_quality_complete", "mandatory_safety_pass"))):
                raise ValueError("EMPTY_RESULT")
            return
        if not isinstance(stored, Mapping):
            raise ValueError("NATIVE_RESULT")
        from domain.les_v7.quality_policy import validate_quality_policy, native_financial_safety
        source = result["financial_safety_source"]
        if not isinstance(source, Mapping):
            raise ValueError("FINANCIAL_PROOF_SOURCE_MISSING")
        resolved = validate_quality_policy(result["resolved_policy"], result["cutoff_at"],
            market=source["market"], records=[source], symbol=source["symbol"])
        if identity(resolved) != identity(result["resolved_policy"]) or resolved["profile_id"] != profile:
            raise ValueError("FINANCIAL_PROOF_POLICY_BINDING")
        rebuilt_safety = native_financial_safety(source, symbol=source["symbol"], market=source["market"],
            cutoff=result["cutoff_at"], policy=resolved)
        if identity(rebuilt_safety) != identity(result["financial_safety"]):
            raise ValueError("FINANCIAL_PROOF_RECEIPT_MISMATCH")
        native = QualityResult(**stored)
        metrics = native.metrics
        safety = result["financial_safety"]
        core_reported = {field: metrics.get(field + "_reported") for field in QUALITY_REQUIRED_REPORTED_FIELDS}
        core_ttm = {field: metrics.get(field + "_ttm") for field in QUALITY_REQUIRED_REPORTED_FIELDS}
        complete, core_pass = quality_evidence_complete(native, policy_version=native_version), _source_complete(native)
        expected = dict(quality_score=native.score, data_confidence=native.confidence,
            earned_points=metrics.get("quality_raw_score"), missing_penalty_points=metrics.get("quality_penalty"),
            coverage_total=(metrics["quality_available_weight"] / 100. if "quality_available_weight" in metrics else None),
            coverage_by_domain={name: value / 100. for name, value in native.coverage.items()},
            metric_ledger=_metric_ledger(native), core_reported=core_reported, core_ttm=core_ttm,
            core_reported_pass=core_pass, core_inputs_pass=core_pass and safety.get("state") == "PASS",
            core_ttm_pass=all(value is not None for value in core_ttm.values()),
            native_quality_complete=complete, mandatory_safety_pass=safety.get("state") == "PASS",
            status=_status(native, safety, core_pass, complete))
        for key, value in expected.items():
            if identity(result.get(key)) != identity(value):
                raise ValueError("PROJECTION:" + key)
        if identity(result.get("financial_inputs")) != identity(metrics):
            raise ValueError("NATIVE_METRICS_PROJECTION")
        if result["status"] == "PASS" and (not complete or not core_pass or safety.get("state") != "PASS"):
            raise ValueError("PASS_WITH_INCOMPLETE_SOURCE")
        if native.score is None:
            if (native.status not in {"DATA_INSUFFICIENT", "REJECT"} or native.confidence != 0
                    or native.components != {} or native.coverage != {}
                    or metrics.get("quality_metric_checks", {}) != {}):
                raise ValueError("UNSCORED_NATIVE_RESULT")
            return
        if (metrics.get("quality_policy_version") != native_version
                or metrics.get("quality_scoring_policy") != native_scoring
                or metrics.get("quality_dc_weights") != QUALITY_DC_WEIGHTS
                or metrics.get("quality_budgets") != QUALITY_BUDGETS):
            raise ValueError("NATIVE_POLICY")
        budgets, checks = _native_metric_budgets(), metrics["quality_metric_checks"]
        if set(checks) != set(budgets) or set(native.components) != set(QUALITY_BUDGETS) or set(native.coverage) != set(QUALITY_BUDGETS):
            raise ValueError("FIXED_METRIC_BUDGET")
        earned = {name: 0. for name in QUALITY_BUDGETS}
        covered = {name: 0. for name in QUALITY_BUDGETS}
        for key, (group, weight) in budgets.items():
            check = checks[key]
            if check["group"] != group or check["weight"] != weight:
                raise ValueError("METRIC_WEIGHT:" + key)
            fraction = check["fraction"]
            if check["available"] is not (fraction is not None):
                raise ValueError("METRIC_AVAILABILITY:" + key)
            points = weight * finite(fraction, key, 0, 1) if fraction is not None else 0.
            if not isclose(check["points"], points, abs_tol=1e-8):
                raise ValueError("METRIC_POINTS:" + key)
            earned[group] += points
            if fraction is not None:
                covered[group] += weight
        if profile in _LIMITED_GAPS_PROFILES:
            if (metrics.get("quality_coverage_minima") != coverage_minima
                    or metrics.get("quality_minimum_available_weight") != 75.
                    or metrics.get("quality_minimum_confidence") != 65.
                    or metrics.get("quality_minimum_score") != 65.):
                raise ValueError("R4_COVERAGE_POLICY")
            _validate_not_meaningful(checks, metrics, source, result["cutoff_at"])
        for group, budget in QUALITY_BUDGETS.items():
            if (not isclose(native.components[group], earned[group], abs_tol=1e-8)
                    or not isclose(native.coverage[group], 100. * covered[group] / budget, abs_tol=1e-8)):
                raise ValueError("DOMAIN_LEDGER:" + group)
        available, raw = sum(covered.values()), sum(earned.values())
        penalty = (100. - available) * QUALITY_MISSING_PENALTY_RATE
        if (not isclose(metrics["quality_raw_score"], raw, abs_tol=1e-8)
                or not isclose(metrics["quality_available_weight"], available, abs_tol=1e-8)
                or not isclose(metrics["quality_missing_weight"], 100. - available, abs_tol=1e-8)
                or not isclose(metrics["quality_penalty"], penalty, abs_tol=1e-8)
                or not isclose(native.score, max(0., raw - penalty), abs_tol=1e-8)
                or metrics["quality_final_score"] != native.score):
            raise ValueError("SCORE_ARITHMETIC")
        expected_penalties = [dict(group=group, reason="MISSING_OR_INCOMPARABLE_METRICS",
            missing_weight=QUALITY_BUDGETS[group] - covered[group],
            points=(QUALITY_BUDGETS[group] - covered[group]) * QUALITY_MISSING_PENALTY_RATE)
            for group in QUALITY_BUDGETS if covered[group] < QUALITY_BUDGETS[group]]
        if identity(metrics.get("quality_penalties")) != identity(expected_penalties):
            raise ValueError("MISSING_PENALTY_LEDGER")
        reported, ledger = metrics["quality_reported_inputs"], metrics["quality_field_checks"]
        period = reported["period"]
        dependencies = {f"{period}:{field}" for field in BALANCE_FIELDS}
        dependencies.update(reported["current_keys"].values())
        dependencies.update(reported["comparison_keys"][field] for field in
                            ("revenue", "cost_of_revenue", "operating_income", "net_income"))
        dependencies.update(reported["opening_keys"][field] for field in ("total_assets", "total_equity"))
        verification = verification_summary(ledger, dependencies)
        if identity(metrics["quality_verification"]) != identity(verification):
            raise ValueError("SOURCE_VERIFICATION_LEDGER")
        dc = metrics["quality_dc_inputs"]
        age = (pd.Timestamp(timestamp(safety["checked_at"])).date() - pd.Period(period, freq="Q").end_time.date()).days
        freshness = 100. if age <= 180 else 70. if age <= 275 else 0.
        sources = {ledger[key].get("source") for key in dependencies if ledger.get(key, {}).get("source")}
        reliability = 100. if sources == {"MOPS_IXBRL"} else 95. if sources <= OFFICIAL_SOURCES else 0.
        if (set(dc) != set(QUALITY_DC_WEIGHTS) or dc["coverage"] != available
                or dc["verification"] != verification["score"]
                or dc["freshness"] != freshness or dc["reliability"] != reliability
                or any(isinstance(value, bool) or not isfinite(value) or not 0 <= value <= 100 for value in dc.values())
                or not isclose(native.confidence, sum(QUALITY_DC_WEIGHTS[key] * dc[key] for key in dc), abs_tol=1e-8)):
            raise ValueError("DC_ARITHMETIC")
        if profile in _LIMITED_GAPS_PROFILES:
            if native.confidence < 50:
                expected_status = "DATA_INSUFFICIENT"
            elif metrics["quality_required_missing"]:
                expected_status = "DATA_INSUFFICIENT"
            elif native.confidence < 65. or available < 75. or not sources <= OFFICIAL_SOURCES or age > 275:
                expected_status = "PROVISIONAL"
            elif native.score < 60. and raw + (100. - available) < 60.:
                expected_status = "REJECT"
            elif native.score < 65.:
                expected_status = "WATCH_QUALITY"
            else:
                expected_status = "PASS"
            optional = native.optional or {}
            if expected_status != "REJECT" and (optional.get("audit_review_required") is True
                    or optional.get("auditor_opinion") in {"UNKNOWN", "QUALIFIED", "QUALIFIED_UNKNOWN"}):
                expected_status = "DATA_INSUFFICIENT"
            if native.status != expected_status:
                raise ValueError("R4_NATIVE_STATUS_ARITHMETIC")
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValueError("V7_REPORTED_QUALITY_INVALID:" + str(error)) from error


def _validate_not_meaningful(checks, metrics, source, cutoff):
    """Known zero denominators are not missing or a licence to award points."""
    reported = metrics["quality_reported_inputs"]
    ledger = metrics["quality_field_checks"]
    period = reported["period"]
    at = pd.Timestamp(timestamp(cutoff))
    def original_value(group, field):
        p = pd.Period(period, freq="Q")
        if group == "current":
            row = source
        else:
            p -= 4
            prefix = "comparative_ytd" if source["flow_basis"] == "YTD" else "comparative_quarter"
            row = dict(source, period_end=p.end_time.date().isoformat(),
                       field_evidence={field: source[prefix + "_field_evidence"][field]},
                       **{field: source[prefix + "_fields"][field]})
        receipt = validate_field(row, field, period=p, cutoff=at)
        if (receipt["value"] is None or receipt["evidence_level"] != "CANONICAL_ORIGINAL"
                or not all(receipt["checks"].values())):
            return None
        return receipt["value"]
    for name, check in checks.items():
        if check["status"] != "NOT_MEANINGFUL":
            continue
        if not check["available"] or check["fraction"] != 0. or check["points"] != 0.:
            raise ValueError("NOT_MEANINGFUL_POINTS:" + name)
        if name == "current_ratio":
            proof = ledger[f"{period}:current_liabilities"]
            original = validate_field(source, "current_liabilities", period=pd.Period(period, freq="Q"),
                                      cutoff=pd.Timestamp(timestamp(cutoff)), instant=True)
            assets = validate_field(source, "current_assets", period=pd.Period(period, freq="Q"),
                                    cutoff=pd.Timestamp(timestamp(cutoff)), instant=True)
            if (proof["value"] != 0. or original["value"] != 0. or assets["value"] is None
                    or original["status"] not in {"VERIFIED_REPORTED", "VERIFIED_DERIVED"}
                    or proof["status"] not in {"VERIFIED_REPORTED", "VERIFIED_DERIVED"}):
                raise ValueError("NOT_MEANINGFUL_SOURCE:" + name)
        elif name == "cash_conversion":
            cfo, profit = original_value("current", "cfo"), original_value("current", "net_income")
            if cfo is None or profit is None or profit > 0 or cfo != reported["current"]["cfo"] or profit != reported["current"]["net_income"]:
                raise ValueError("NOT_MEANINGFUL_SOURCE:" + name)
        elif name.startswith(("gross_margin_", "operating_margin_", "net_margin_")):
            margin, part = name.rsplit("_", 1)
            numerator = "cost_of_revenue" if margin == "gross_margin" else "operating_income" if margin == "operating_margin" else "net_income"
            groups = ("current",) if part == "positive" else ("comparative", "current")
            operands = {(group, field): original_value(group, field) for group in groups for field in ("revenue", numerator)}
            if (not any(operands[group, "revenue"] is not None and operands[group, "revenue"] <= 0 for group in groups)
                    or any(value is None or value != reported[group][field] for (group, field), value in operands.items())):
                raise ValueError("NOT_MEANINGFUL_SOURCE:" + name)
        else:
            raise ValueError("NOT_MEANINGFUL_METRIC:" + name)


def evaluate_financials_reported(records, *, symbol, market, cutoff,
                                 safety_receipt=None, policy=None):
    """Reuse one native QualityResult at the fixed historical cutoff.

    ``safety_receipt`` remains API-compatible but is never accepted as financial
    proof. In particular an unrelated market SAFETY PASS cannot replace source
    validation, audit evidence or the six required balance-sheet receipts.
    """
    from domain.les_v7.quality_policy import validate_quality_policy, native_financial_safety
    try:
        resolved = validate_quality_policy(policy, cutoff, market=market, records=records, symbol=symbol)
        if resolved.get("profile_id") not in {PROFILE, *_LIMITED_GAPS_PROFILES}:
            raise ValueError("REPORTED_QUALITY_R3_R4_OR_R5_POLICY_REQUIRED")
    except (ValueError, TypeError, AttributeError) as error:
        return _empty(str(error), policy, status="CONFIG_INCOMPLETE", cutoff=cutoff)
    profile, mapping, native_version = _profile_rules(resolved)
    at = pd.Timestamp(timestamp(cutoff))
    try:
        periods, ledger, excluded = select_quality_inputs(records, symbol=symbol, market=market, cutoff=at)
        if not periods:
            return _empty("NO_POINT_IN_TIME_FINANCIALS", resolved, cutoff=cutoff)
        latest_period = max(periods)
        latest = periods[latest_period]
        if str(latest_period) != resolved["latest_required_period"]:
            return _empty("LATEST_REQUIRED_REPORTING_PERIOD_UNAVAILABLE", resolved, cutoff=cutoff)
        source = _financial_source(latest, profile)
        safety = native_financial_safety(source, symbol=symbol, market=market,
                                        cutoff=cutoff, policy=resolved)
        native = evaluate_quality(_audited_view(records, latest, safety), symbol=symbol, market=market,
                                  as_of=cutoff, policy_version=native_version)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return _empty(str(error), resolved, cutoff=cutoff)

    metrics = native.metrics
    reported = metrics.get("quality_reported_inputs", {})
    core_reported = {field: metrics.get(field + "_reported") for field in QUALITY_REQUIRED_REPORTED_FIELDS}
    core_ttm = {field: metrics.get(field + "_ttm") for field in QUALITY_REQUIRED_REPORTED_FIELDS}
    core_reported_pass = _source_complete(native)
    safety_pass = safety.get("state") == "PASS"
    native_complete = quality_evidence_complete(native, policy_version=native_version)
    status = _status(native, safety, core_reported_pass, native_complete)
    reasons = list(native.reasons)
    if not core_reported_pass:
        reasons.append("REQUIRED_REPORTED_SOURCES_UNVERIFIED")
    if not safety_pass:
        reasons.extend(safety.get("reasons", ()))
    if native.status == "PROVISIONAL" or native.status == "PASS" and not native_complete:
        reasons.append("NATIVE_REPORTED_QUALITY_COVERAGE_OR_LEDGER_INCOMPLETE")
    if status == "DATA_INSUFFICIENT" and not reasons:
        reasons.append("REPORTED_QUALITY_DATA_INSUFFICIENT")

    available = metrics.get("quality_available_weight")
    total = available / 100. if available is not None else None
    age = (at.date() - latest_period.end_time.date()).days
    return dict(profile_id=profile, mapping_version=mapping,
        quality_policy_id=identity(resolved), resolved_policy=resolved,
        cutoff_at=timestamp(cutoff).isoformat(), financial_safety_source=source,
        status=status, quality_score=native.score,
        earned_points=metrics.get("quality_raw_score"),
        missing_penalty_points=metrics.get("quality_penalty"),
        data_confidence=native.confidence, coverage_total=total,
        coverage_by_domain={name: value / 100. for name, value in native.coverage.items()},
        metric_ledger=_metric_ledger(native), gap_groups=[],
        gap_qualifier=("COMPLETE" if available == 100 else "REPORTED_CORE_WITH_BOUNDED_MISSING_WEIGHT"
                       if available is not None else "NOT_EVALUATED"),
        reason_codes=sorted(set(reasons)), warnings=list(native.warnings),
        core_reported_pass=core_reported_pass, core_inputs_pass=core_reported_pass and safety_pass,
        native_quality_complete=native_complete, mandatory_safety_pass=safety_pass,
        core_ttm_pass=all(value is not None for value in core_ttm.values()),
        core_source_basis=SCORING_BASIS, scoring_basis=SCORING_BASIS,
        quality_les_cap=100 if status == "PASS" else None,
        core_ttm=core_ttm, core_reported=core_reported, reported_inputs=reported,
        latest_period=str(latest_period), core_ttm_quarters=[str(latest_period - i) for i in range(4)],
        missing_quarters=[str(latest_period - i) for i in range(4) if latest_period - i not in periods],
        field_checks=metrics.get("quality_field_checks", ledger),
        excluded_history=metrics.get("excluded_quality_inputs", excluded),
        ttm_traces=metrics.get("quality_ttm_inputs", {}),
        # Native owns this newly derived graph. Keep one graph in this work;
        # public JSON serialization later detaches it from external callers.
        financial_inputs=metrics,
        reported_result={field.name: getattr(native, field.name) for field in fields(native)},
        capex_ttm=metrics.get("capex_ttm"), fcf_ttm=metrics.get("fcf_ttm"),
        financial_safety=safety, verification=dict(
            freshness=metrics.get("quality_dc_inputs", {}).get("freshness"),
            validation=metrics.get("quality_dc_inputs", {}).get("verification"),
            source_reliability=metrics.get("quality_dc_inputs", {}).get("reliability"),
            source_age_days=age, native=metrics.get("quality_verification", {})),
        readiness=dict(financial="READY" if status == "PASS" else "NOT_READY"))
