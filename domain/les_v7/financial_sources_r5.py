"""R5 source repair: current report sections, bilingual binding and cash proofs.

Only a verified same-original filing may rebuild a cached field. Publication
times, financial amounts, taxonomy mappings and the legacy audit engine stay
unchanged. Interpretation labels below are analysis-only; original contexts and
their hashes are retained verbatim in the new proof.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import re

import pandas as pd

from domain.financial_taxonomy import CASH_AUDIT_FLOWS, CASH_RECONCILING_COMPONENTS
from domain.fundamental_derivation import _AUDIT_CONCEPTS, _derive_period_audit_evidence
from domain.les_v7.contracts import identity
from domain.les_v7.financial_sources import (
    _period_quotes, _scope_for_period, ixbrl_record_r2,
)
from domain.text_normalization import normalize_cjk_text

AUDIT_POLICY_R5 = "native-audit-evidence/v4"
AUDIT_CLASSIFICATION_R5 = "SOURCE_BOUND_CURRENT_REPORT_SECTIONS_R5"
SOURCE_REPAIR_POLICY_R5 = "SAME_ORIGINAL_AUDIT_CASH_REBUILD_R5"
R5_AVAILABLE_AT = "2026-10-14T06:33:18Z"
_BODY = "tifrs-ar:AccountantsReportBody"
_CASH_FIELDS = tuple(dict.fromkeys(("cfo", *CASH_AUDIT_FLOWS, "cash_beginning",
                                  "cash_ending", "cash", *CASH_RECONCILING_COMPONENTS)))


def _scope_for_period_r5(text, quote):
    value = _scope_for_period(text, quote)
    if value is not None or "年" in quote:
        return value
    # The verified English original says "as at June 30, 2026". Do not accept
    # arbitrary nearby "consolidated" text, or a report for a different date.
    normalized = normalize_cjk_text(text)
    scopes = set()
    description = re.compile(
        r"\b(consolidated|separate|individual)\s+(?:(?:interim|condensed)\s+)*"
        r"(?:financial statements?|balance sheets?|statements of "
        r"(?:comprehensive income|cash flows|changes in equity))\b", re.I)
    for match in re.finditer(re.escape(quote), normalized, re.I):
        prefix = normalized[max(0, match.start() - 500):match.start()]
        prefix = re.sub(r"\b(Inc|Ltd|Co|Corp)\.", r"\1", prefix, flags=re.I)
        prefix = re.split(r"[.;。；!?]", prefix)[-1]
        for bound in description.finditer(prefix):
            if re.search(r"\b(?:as at|as of|ended)\b", prefix[bound.end():], re.I):
                scopes.add("consolidated" if bound[1].lower() == "consolidated" else "individual")
    return next(iter(scopes)) if len(scopes) == 1 else None


def _body_bindings_r5(facts, period, scope):
    bindings = []
    for fact in facts:
        if fact.concept != _BODY or not fact.text_value:
            continue
        text = normalize_cjk_text(fact.text_value)[:2000]
        quotes = _period_quotes(text, period)
        if not quotes or any(_scope_for_period_r5(text, quote) != scope for quote in quotes):
            return None  # A mismatched translation is not silently discarded.
        bindings.append(dict(context_id=fact.context_id, financial_period_quote=quotes[0],
            financial_period_quotes=list(quotes), body_sha256=identity(fact.text_value), scope=scope))
    return bindings


def select_audit_contexts_r5(filing):
    """Bound current fiscal endpoint or one source-visible signed-report date."""
    period = pd.Period(year=filing.fiscal_year, quarter=filing.quarter, freq="Q")
    end = period.end_time.date().isoformat()
    facts = tuple(f for f in filing.facts if f.concept in _AUDIT_CONCEPTS and not f.dimensions)
    binding = dict(method="FINANCIAL_PERIOD_CONTEXT", financial_period_end=end,
                   auditor_report_date=None, body_bindings=[])
    available = pd.to_datetime(filing.available_at, utc=True, errors="coerce")
    if pd.isna(available):
        return (), binding
    last = available.tz_convert("Asia/Taipei").date().isoformat()
    dates = set()
    for fact in facts:
        endpoint = fact.end_date or fact.instant
        if endpoint is None:
            continue
        try:
            checked = pd.Timestamp(endpoint).date().isoformat()
        except (TypeError, ValueError):
            return (), binding
        if checked != endpoint:
            return (), binding
        if end < checked <= last:
            dates.add(checked)
    current = tuple(f for f in facts if (f.end_date or f.instant) == end)
    if current:
        bodies = _body_bindings_r5(current, period, filing.statement_scope)
        return ((), binding) if dates or bodies is None else (current, dict(binding, body_bindings=bodies))
    if len(dates) != 1:
        return (), binding
    signed = next(iter(dates))
    selected = tuple(f for f in facts if (f.end_date or f.instant) == signed)
    bodies = _body_bindings_r5(selected, period, filing.statement_scope)
    if not bodies:
        return (), binding
    return selected, dict(binding, method="SOURCE_BOUND_ACCOUNTANT_REPORT_DATE",
                          auditor_report_date=signed, body_bindings=bodies)


_CURRENT_HEADINGS = frozenset((
    "保留結論之基礎", "保留意見之基礎", "保留結論", "保留意見", "結論", "查核意見",
    "否定意見之基礎", "無法表示意見之基礎", "無法表示結論之基礎",
    "繼續經營重大不確定性", "持續經營重大不確定性",
    "basisforqualifiedconclusion", "basisforqualifiedopinion", "qualifiedconclusion",
    "qualifiedopinion", "conclusion", "opinion", "basisforadverseopinion",
    "basisfordisclaimerofopinion", "materialuncertaintyrelatedtogoingconcern",
))


def _section_view(text):
    """Current report headings reset context, not every incidental year value.

    Only standalone headings qualify. Explicit comparative/prior headings are
    never relabelled, and original report text is never modified or persisted as
    an original. This fixes a current 115/114 dual-year introduction making its
    actual current basis look like a separate prior auditor's opinion.
    """
    lines, prior_section = [], False
    for line in text.splitlines():
        normalized = normalize_cjk_text(line)
        key = re.sub(r"\s+", "", normalized).lower().rstrip(":：")
        if re.match(r"^(?:前期|上期|前一年度|上年度|priorperiod|prioryear|previousperiod|previousyear|comparativeperiod)", key):
            prior_section = True
        elif re.match(r"^(?:本期|本年度|當期|currentperiod|currentyear)", key):
            prior_section = False
        if key in _CURRENT_HEADINGS and not prior_section:
            line = "本期:" + line
        lines.append(line)
    value = "\n".join(lines)
    # Exact observed English synonym; only interpretation text is normalized.
    return re.sub(r"\binsignificant\s+consolidated\s+subsidiar", "nonsignificant consolidated subsidiar",
                  value, flags=re.I)


def classify_audit_contexts_r5(filing, current, binding):
    bodies = tuple(f for f in current if f.concept == _BODY and f.text_value)
    base = tuple(f for f in current if f.concept != _BODY)
    classifications = []
    for body in bodies:
        interpreted = replace(body, text_value=_section_view(body.text_value))
        audit = _derive_period_audit_evidence(replace(filing, facts=(*base, interpreted)),
                                              allow_unreviewed_components=True)
        classifications.append(dict(body_sha256=identity(body.text_value),
                                    interpretation_sha256=identity(interpreted.text_value), audit=audit))
    audit = (_derive_period_audit_evidence(replace(filing, facts=current), allow_unreviewed_components=True)
             if not classifications else dict(classifications[0]["audit"]))
    # Do not choose a favorable translation. A demonstrated hard risk always
    # survives; otherwise inconsistent source interpretations remain unknown.
    safety = ("audit_opinion", "going_concern", "material_accounting_issue", "audit_review_required")
    hard = [node["audit"] for node in classifications if node["audit"]["audit_opinion"] in
            {"ADVERSE", "DISCLAIMER", "QUALIFIED_MATERIAL_MISSTATEMENT"}
            or node["audit"]["going_concern"] is True or node["audit"]["material_accounting_issue"] is True]
    if hard:
        audit = dict(hard[0])
    elif any(tuple(node["audit"][key] for key in safety) != tuple(audit[key] for key in safety)
             for node in classifications):
        audit.update(audit_opinion="UNKNOWN", current_audit_opinion="UNKNOWN",
                     material_accounting_issue=None, audit_review_required=True,
                     accounting_concern_classification="SOURCE_LANGUAGE_CLASSIFICATION_CONFLICT")
    raw = " ".join(f.text_value for f in bodies)
    return dict(audit, audit_report_text_raw=raw, audit_report_text_normalized=normalize_cjk_text(raw),
                classification_policy=AUDIT_CLASSIFICATION_R5, body_classifications=classifications)


def audit_proof_r5(filing, *, symbol, market="TWSE", original_verified=False):
    period = pd.Period(year=filing.fiscal_year, quarter=filing.quarter, freq="Q")
    end = period.end_time.date().isoformat()
    current, binding = select_audit_contexts_r5(filing)
    audit = classify_audit_contexts_r5(filing, current, binding)
    verified = bool(original_verified and filing.content_sha256 and filing.source_url
                    and (filing.period_end is None or filing.period_end == end)
                    and current and audit["audit_evidence_available"])
    return dict(schema="financial-audit-proof/v2", policy_version=AUDIT_POLICY_R5,
        source="MOPS_IXBRL", source_url=filing.source_url, symbol=str(symbol), market=market,
        period_end=end, available_at=filing.available_at, scope=filing.statement_scope,
        audit_status=filing.audit_status, currency="TWD", unit="thousand",
        content_sha256=filing.content_sha256,
        revision_id=filing.content_sha256 or f"filing-{filing.filing_id}-r{filing.revision_number}",
        status="VERIFIED_REPORTED" if verified else "PENDING_VERIFICATION",
        reason="CURRENT_PERIOD_AUDIT_SOURCE_VERIFIED" if verified else
            "RAW_AUDIT_SOURCE_NOT_VERIFIED" if not original_verified else "CURRENT_PERIOD_AUDIT_CONTEXT_UNAVAILABLE",
        audit=audit, period_binding=binding, contexts=[asdict(f) for f in current], result_id=identity(audit))


def repair_source_record_r5(row, filing, *, symbol, market="TWSE", original_verified=False):
    """Public source overlay; collector must verify the raw bytes.

    Field selection reuses the existing IAS 7/disposal mapper. No arithmetic
    difference is inserted as an adjustment and no absent component is zero.
    An independently source-bound new audit proof is additive to the old one.
    Changed/current audit packets are detached; unrelated original evidence
    graphs remain shared read-only, as in the native reported-source adapter.
    """
    origin = row.get("quarter_flow_source", row)
    expected = dict(symbol=str(symbol), market=market, source="MOPS_IXBRL",
        content_sha256=filing.content_sha256, revision_id=filing.content_sha256,
        source_url=filing.source_url, available_at=filing.available_at,
        period_end=pd.Period(year=filing.fiscal_year, quarter=filing.quarter, freq="Q").end_time.date().isoformat(),
        scope=filing.statement_scope, currency="TWD", unit="thousand")
    if not original_verified or any(origin.get(key) != value for key, value in expected.items()):
        raise ValueError("R5_SAME_ORIGINAL_REPAIR_BINDING_REQUIRED")
    if row.get("quarter_flow_source"):
        raise ValueError("R5_REPORTED_ORIGINAL_REQUIRED_NOT_DERIVED_QUARTER")
    rebuilt = ixbrl_record_r2(symbol, filing, market=market, original_verified=True)
    result = dict(row)
    if "audit_evidence" in row:
        result["audit_evidence"] = deepcopy(row["audit_evidence"])
    result["field_evidence"] = dict(row.get("field_evidence", {}))
    for field in _CASH_FIELDS:
        result[field] = rebuilt.get(field)
        result["field_evidence"][field] = deepcopy(rebuilt.get("field_evidence", {}).get(field, {}))
    result["audit_evidence_r5"] = audit_proof_r5(filing, symbol=symbol, market=market, original_verified=True)
    result["source_repair_r5"] = dict(schema="financial-source-repair/v1", policy=SOURCE_REPAIR_POLICY_R5,
        content_sha256=filing.content_sha256, available_at=filing.available_at,
        old_audit_proof_id=identity(row.get("audit_evidence")),
        new_audit_proof_id=identity(result["audit_evidence_r5"]),
        repaired_fields=list(_CASH_FIELDS),
        old_cash_proof_id=identity({f: row.get("field_evidence", {}).get(f) for f in _CASH_FIELDS}),
        new_cash_proof_id=identity({f: result["field_evidence"].get(f) for f in _CASH_FIELDS}))
    return result
