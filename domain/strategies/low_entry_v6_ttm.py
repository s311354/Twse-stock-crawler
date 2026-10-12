"""Verified comparative TTM, without inventing missing single quarters.

The bridge validates the actual field dependencies, not unrelated statement
lines. Unrelated comparative differences remain diagnostics. A positive,
source-bound accounting-policy attestation is still required; local revision
counters are NOT evidence of a common accounting basis.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite
import re

import pandas as pd

from domain.text_normalization import normalize_cjk_text
from domain.data_provenance import OFFICIAL_FILING_TIMESTAMP, MOPS_PUBLIC_ANNOUNCEMENT, MOPS_CORRECTION_ANNOUNCEMENT


TTM_BRIDGE_POLICY_VERSION = "comparative-original-ttm/scoped-v2"
_POLICY_CONCEPT = "tifrs-notes:ApplicationOfNewlyIssuedOrAmendedStandardsAndInterpretations"
_ANCHORS = ("current_assets", "total_assets", "current_liabilities", "total_liabilities", "total_equity", "cash")


def accounting_basis_evidence(filing, start: str, end: str, origin: dict) -> dict:
    """Read only the current-adoption subsection, not future IFRS discussion.

    Exceptions, reclassification or restatement require review. A missing note
    is not equivalent to 'no change'. Evidence stays bound to the original.
    """
    candidates = [f for f in filing.facts if f.concept == _POLICY_CONCEPT and not f.dimensions
                  and f.start_date == start and f.end_date == end and f.text_value]
    sections = []
    for fact in candidates:
        text = normalize_cjk_text(fact.text_value)
        headings = list(re.finditer(r"\(([一二三123])\)|(?<!\d)([123])\.(?=\D)", text))
        label = lambda heading: heading.group(1) or heading.group(2)
        first = next((h for h in headings if label(h) in {"一", "1"}), None)
        following = next((h for h in headings if first and h.start() > first.start()
                          and label(h) in {"二", "2"}), None)
        section = text[first.end():following.start()] if first and following else ""
        sections.append((fact.context_id, section))
    texts = {text for _, text in sections if text}
    status, reason = "UNKNOWN", "CURRENT_ACCOUNTING_POLICY_ATTESTATION_MISSING"
    if len(texts) > 1:
        reason = "CONFLICTING_CURRENT_ACCOUNTING_POLICY_ATTESTATIONS"
    elif len(texts) == 1:
        text = next(iter(texts))
        if any(word in text for word in ("除下列", "除以下", "除上述", "不重編", "重分類", "追溯", "重編", "更正",
                                         "尚未", "未採用", "未適用", "預計")):
            reason = "ACCOUNTING_POLICY_EXCEPTION_REQUIRES_REVIEW"
        elif re.search(r"(?:不致|不會|未|無)[^。;；]{0,120}(?:重大變動|重大影響|重大變更)", text):
            status, reason = "RESOLVED", "CURRENT_ADOPTION_NO_MATERIAL_CHANGE_REPORTED"
    if filing.correction_description:
        status, reason = "UNKNOWN", "FILING_CORRECTION_REQUIRES_REVIEW"
    return dict(origin, status=status, reason=reason, policy=TTM_BRIDGE_POLICY_VERSION,
                concept=_POLICY_CONCEPT, start_date=start, end_date=end,
                contexts=[dict(context_id=context, current_adoption_text=text) for context, text in sections])


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _original(row):
    return row.get("quarter_eps_evidence", row) if row.get("quarter_flow_source") else row


def _bound(proof, row, start, end, *, instant=False):
    return (isinstance(proof, Mapping) and proof.get("status") == "RESOLVED" and row.get("source") == "MOPS_IXBRL"
            and row.get("content_sha256") and row.get("revision_id")
            and all(proof.get(k) == row.get(k) for k in
                    ("content_sha256", "revision_id", "scope", "currency", "available_at"))
            and proof.get("unit") == "thousand" and proof.get("end_date") == end
            and (proof.get("basis") == "INSTANT" if instant else
                 proof.get("start_date") == start and proof.get("basis") in {"YTD", "QUARTER"}))


def _ledger_match(left, right):
    # Each ledger includes aggregate statement facts only, not note dimensions
    # or EPS with its non-additive weighted denominator.
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return "COMPARATIVE_STATEMENT_LEDGER_MISSING", []
    common = sorted(set(left) & set(right))
    mismatches = [key for key in common if _number(left[key]) is None or _number(right[key]) is None
                  or not isclose(_number(left[key]), _number(right[key]), rel_tol=0., abs_tol=.5)]
    if mismatches:
        return "COMPARATIVE_STATEMENT_VALUES_CHANGED", mismatches
    return (None if common else "COMPARATIVE_STATEMENT_LEDGER_MISSING"), common


def _field_dependencies(proof, row, start, end):
    """Return the source amount and exact concepts used by a verified field.

    Context lists contain other periods too: only the exact aggregate duration
    is relevant here. Derived cost has two reported operands; their concepts
    must reconcile as well. No arbitrary expression evaluator is introduced.
    """
    if not _bound(proof, row, start, end):
        return None, {}
    if proof.get("reason") == "REVENUE_MINUS_GROSS_PROFIT_SAME_DOCUMENT":
        inputs = proof.get("inputs", [])
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            return None, {}
        left, left_dependencies = _field_dependencies(inputs[0], row, start, end)
        right, right_dependencies = _field_dependencies(inputs[1], row, start, end)
        if left is None or right is None or any(
                key in right_dependencies and not isclose(value, right_dependencies[key], rel_tol=0., abs_tol=.5)
                for key, value in left_dependencies.items()):
            return None, {}
        return left - right, dict(left_dependencies, **right_dependencies)
    contexts = proof.get("contexts", [])
    if not isinstance(contexts, (list, tuple)):
        return None, {}
    matching = [context for context in contexts if isinstance(context, Mapping)
                and not context.get("dimensions") and context.get("start_date") == start
                and context.get("end_date") == end and context.get("instant") is None]
    if not matching or any(context.get("unit") != "iso4217:TWD"
                           or not context.get("concept") or _number(context.get("numeric_value")) is None
                           for context in matching):
        return None, {}
    values = {_number(context["numeric_value"]) / 1000 for context in matching}
    if len(values) != 1:
        return None, {}
    return next(iter(values)), {context["concept"]: _number(context["numeric_value"]) / 1000
                               for context in matching}


def _impact_attestation(row, field, issues, dependencies, cutoff):
    """Validate an optional explicit review; never infer safety from a keyword.

    Normalized callers may supply ``field_impact_evidence[field]`` after a
    verified, source-bound field impact review. It must quote original policy
    contexts, explicitly cover the issues and concepts, and not be available
    before the source or after cutoff. The original is never rewritten.
    """
    evidence = row.get("field_impact_evidence", {})
    proof = evidence.get(field, {}) if isinstance(evidence, Mapping) else {}
    if not isinstance(proof, Mapping):
        return None
    issue_list, concepts = proof.get("issues"), proof.get("dependency_concepts")
    if (not isinstance(issue_list, (list, tuple)) or not all(isinstance(issue, str) for issue in issue_list)
            or not isinstance(concepts, (list, tuple)) or not all(isinstance(concept, str) for concept in concepts)):
        return None
    if (proof.get("status") != "RESOLVED"
            or proof.get("impact") != "UNAFFECTED" or proof.get("field") != field
            or proof.get("policy") != TTM_BRIDGE_POLICY_VERSION
            or proof.get("verification_method") != "SOURCE_BOUND_EXPLICIT_FIELD_IMPACT"
            or any(proof.get(key) != row.get(key) for key in
                   ("symbol", "market", "source", "source_url", "content_sha256", "revision_id",
                    "scope", "currency", "unit", "period_end", "available_at"))
            or not row.get("content_sha256") or not row.get("revision_id") or not row.get("source_url")
            or set(issue_list) != set(issues) or set(concepts) != set(dependencies)):
        return None
    available = pd.to_datetime(proof.get("evidence_available_at"), utc=True, errors="coerce")
    verified = pd.to_datetime(proof.get("verified_at"), utc=True, errors="coerce")
    source_available = pd.to_datetime(row.get("available_at"), utc=True, errors="coerce")
    # A new review is not silently made historical. A historical review proof
    # requires its own verifiable timestamp, no earlier than its bound source.
    if (pd.isna(available) or pd.isna(verified) or pd.isna(source_available)
            or available < source_available or verified < source_available or available < verified
            or available > cutoff or verified > cutoff):
        return None
    policy = row.get("accounting_basis_evidence", {})
    if "ACCOUNTING_POLICY" in issues:
        source_contexts = {context.get("context_id"): context.get("current_adoption_text", "")
                           for context in policy.get("contexts", []) if isinstance(context, Mapping)}
        contexts = proof.get("contexts", [])
        if not isinstance(contexts, (list, tuple)) or not contexts or any(not isinstance(context, Mapping)
                               or not isinstance(context.get("statement"), str) or not context["statement"]
                               or context["statement"] not in source_contexts.get(context.get("context_id"), "")
                               for context in contexts):
            return None
    if "FILING_CORRECTION" in issues and (
            not row.get("correction_description")
            or proof.get("correction_description") != row["correction_description"]):
        return None
    return dict(proof)


def comparative_ttm(periods: Mapping, period: pd.Period, field: str, as_of) -> tuple[float | None, dict]:
    """FY + current YTD - current-original prior-year YTD, when verified.

    No EPS subtraction, no quarter filling, no latest-data fallback. Both
    overlapping prior interim field dependencies must reconcile. Unrelated
    flows and annual ending balances are diagnostic, not additional operands.
    All sources must be the cutoff-selected originals with compatible scope.
    """
    trace = dict(method="COMPARATIVE_ORIGINAL_TTM", policy=TTM_BRIDGE_POLICY_VERSION,
                 period=str(period), field=field, status="UNKNOWN", gaps=[], diagnostics=[])

    def fail(reason, **details):
        category = "INCOMPARABLE" if reason in {
            "TTM_BRIDGE_STATEMENTS_INCOMPARABLE", "COMPARATIVE_FIELD_DEPENDENCY_VALUES_CHANGED",
            "PRIOR_YEAR_YTD_COMPARATIVE_CHANGED", "CAPEX_SIGN_BASIS_INCONSISTENT",
        } else "PENDING_VERIFICATION"
        trace["gaps"].append(dict(category=category, reason=reason, **details))
        return None, trace

    if field == "basic_eps" or period.quarter == 4:
        return fail("NON_ADDITIVE_EPS_OR_YEAR_END_USE_NATIVE_PATH")
    prior_period = period - 4
    annual_period = pd.Period(year=period.year - 1, quarter=4, freq="Q")
    selected = [periods.get(p) for p in (period, annual_period, prior_period)]
    if any(row is None for row in selected):
        return fail("TTM_BRIDGE_SOURCE_PERIOD_MISSING", periods=[str(p) for p, row in
                    zip((period, annual_period, prior_period), selected) if row is None])
    current, annual, prior = [_original(row) for row in selected]
    cutoff = pd.to_datetime(as_of, utc=True)
    for row in (current, annual, prior):
        available = pd.to_datetime(row.get("available_at"), utc=True, errors="coerce")
        if pd.isna(available) or available > cutoff:
            return fail("TTM_BRIDGE_SOURCE_NOT_AVAILABLE_AT_CUTOFF")
    # Publication/revision time is not acquisition time. A later local download
    # of an old original is allowed AFTER that observation and the full checks
    # below. It cannot be backdated, nor mistaken for a later published revision.
    publication_sources = {OFFICIAL_FILING_TIMESTAMP, MOPS_PUBLIC_ANNOUNCEMENT, MOPS_CORRECTION_ANNOUNCEMENT}
    if current.get("availability_source") in publication_sources and any(
            row.get("availability_source") in publication_sources
            and pd.to_datetime(row["available_at"], utc=True) > pd.to_datetime(current["available_at"], utc=True)
            for row in (annual, prior)):
        return fail("TTM_ANCHOR_VERSION_LATER_THAN_CURRENT_ORIGINAL")
    if any(row.get(key) != current.get(key) for row in (annual, prior) for key in
           ("symbol", "market", "scope", "currency", "unit")):
        return fail("TTM_BRIDGE_STATEMENTS_INCOMPARABLE")
    policy = current.get("accounting_basis_evidence", {})
    if (not isinstance(policy, Mapping) or policy.get("policy") != TTM_BRIDGE_POLICY_VERSION
            or policy.get("start_date") != f"{period.year}-01-01"
            or policy.get("end_date") != period.end_time.date().isoformat()
            or any(policy.get(k) != current.get(k) for k in
                   ("content_sha256", "revision_id", "available_at", "scope", "currency"))):
        return fail("CURRENT_ACCOUNTING_POLICY_PROVENANCE_UNVERIFIED")
    start, end = f"{period.year}-01-01", period.end_time.date().isoformat()
    old_start, old_end = f"{period.year-1}-01-01", prior_period.end_time.date().isoformat()
    year_end = annual_period.end_time.date().isoformat()
    groups = [(current, "field_evidence", current.get(field), start, end),
              (annual, "field_evidence", annual.get(field), old_start, year_end),
              (current, "comparative_ytd_field_evidence", current.get("comparative_ytd_fields", {}).get(field), old_start, old_end),
              (prior, "field_evidence", prior.get(field), old_start, old_end)]
    values, source_dependencies = [], {}
    ledger_groups = {"field_evidence": "statement_ledger",
                     "comparative_ytd_field_evidence": "comparative_statement_ledger"}
    for row, group, value, first, last in groups:
        proof = row.get(group, {}).get(field, {})
        if _number(value) is None or not _bound(proof, row, first, last):
            return fail("TTM_BRIDGE_FIELD_PROVENANCE_UNVERIFIED", source_period=row.get("period_end"), group=group)
        parsed_value, dependencies = _field_dependencies(proof, row, first, last)
        if parsed_value is None or not isclose(parsed_value, _number(value), rel_tol=0., abs_tol=.5):
            return fail("TTM_BRIDGE_FIELD_SOURCE_VALUE_UNVERIFIED", source_period=row.get("period_end"), group=group)
        ledger = row.get(ledger_groups[group], {})
        if not isinstance(ledger, Mapping) or any(_number(ledger.get(key)) is None for key in dependencies):
            return fail("TTM_BRIDGE_DEPENDENCY_LEDGER_UNVERIFIED", source_period=row.get("period_end"), group=group,
                        fields=sorted(dependencies))
        changed = [key for key, expected in dependencies.items()
                   if not isclose(_number(ledger[key]), expected, rel_tol=0., abs_tol=.5)]
        if changed:
            return fail("COMPARATIVE_FIELD_DEPENDENCY_VALUES_CHANGED", source_period=row.get("period_end"),
                        group=group, fields=sorted(changed))
        source_dependencies.setdefault(id(row), set()).update(dependencies)
        values.append(_number(value))
    if not isclose(values[2], values[3], rel_tol=0., abs_tol=.5):
        return fail("PRIOR_YEAR_YTD_COMPARATIVE_CHANGED")
    attestations = []
    for row in (current, annual, prior):
        issues = []
        if row is current and policy.get("status") != "RESOLVED":
            issues.append("ACCOUNTING_POLICY")
        if row.get("correction_description"):
            issues.append("FILING_CORRECTION")
        if issues:
            attestation = _impact_attestation(row, field, issues, source_dependencies[id(row)], cutoff)
            if attestation is None:
                return fail("FILING_CORRECTION_REQUIRES_REVIEW" if row.get("correction_description")
                            else policy.get("reason") or "CURRENT_ACCOUNTING_POLICY_ATTESTATION_MISSING",
                            source_period=row.get("period_end"), field=field)
            attestations.append(attestation)
    # Balance lines do not enter an additive flow bridge. Their discrepancies
    # are retained for audit and any separate capital/BS metric, not promoted
    # to an unsupported whole-statement restatement conclusion.
    for key in _ANCHORS:
        proof = current.get("comparative_year_end_field_evidence", {}).get(key, {})
        original_proof = annual.get("field_evidence", {}).get(key, {})
        left = _number(current.get("comparative_year_end_fields", {}).get(key))
        right = _number(annual.get(key))
        if (left is None or right is None or not _bound(proof, current, None, year_end, instant=True)
                or not _bound(original_proof, annual, None, year_end, instant=True)):
            trace["diagnostics"].append(dict(reason="ANNUAL_COMPARATIVE_ANCHOR_UNVERIFIED", field=key,
                                             affects_ttm_field=False))
        elif not isclose(left, right, rel_tol=0., abs_tol=.5):
            trace["diagnostics"].append(dict(reason="ANNUAL_COMPARATIVE_ANCHOR_CHANGED", field=key,
                                             affects_ttm_field=False))
    for label, left, right in (
        ("prior_ytd", current.get("comparative_statement_ledger"), prior.get("statement_ledger")),
        ("annual_end", current.get("comparative_year_end_ledger"), annual.get("balance_ledger")),
    ):
        reason, keys = _ledger_match(left, right)
        if reason:
            relevant = sorted(set(keys) & source_dependencies[id(current)]) if label == "prior_ytd" else []
            if relevant and reason == "COMPARATIVE_STATEMENT_VALUES_CHANGED":
                return fail("COMPARATIVE_FIELD_DEPENDENCY_VALUES_CHANGED", comparison=label, fields=relevant)
            trace["diagnostics"].append(dict(reason=reason, comparison=label, fields=keys,
                                             affects_ttm_field=False))
    value = values[1] + values[0] - values[2]
    if not isfinite(value):
        return fail("NONFINITE_TTM_DERIVATION")
    if field == "capex" and ((any(v < 0 for v in values) and any(v > 0 for v in values))
                             or (value < 0 and any(v > 0 for v in values))
                             or (value > 0 and any(v < 0 for v in values))):
        return fail("CAPEX_SIGN_BASIS_INCONSISTENT")
    sources = [dict(period_end=row["period_end"], **{k: row.get(k) for k in
               ("source", "source_url", "content_sha256", "revision_id", "available_at", "scope", "currency", "unit")})
               for row in (current, annual, prior)]
    trace.update(status="RESOLVED", formula="prior_full_year + current_ytd - comparative_prior_year_ytd",
                 operands=dict(prior_full_year=values[1], current_ytd=values[0], comparative_prior_year_ytd=values[2]),
                 sources=sources, available_at=max(
                     [pd.to_datetime(r["available_at"], utc=True) for r in (current, annual, prior)]
                     + [r.get("_available", pd.to_datetime(r["available_at"], utc=True)) for r in selected]
                     + [pd.to_datetime(proof[key], utc=True) for proof in attestations
                        for key in ("evidence_available_at", "verified_at")]).isoformat(),
                 accounting_basis="SOURCE_BOUND_FIELD_IMPACT_REVIEW" if attestations else policy["reason"],
                 field_impact_attestations=attestations,
                 dependency_concepts=sorted(set().union(*source_dependencies.values())), quarter_values_filled=False)
    return value, trace
