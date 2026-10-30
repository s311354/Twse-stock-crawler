"""Opt-in audit parser candidate; not a replacement Quality/entry engine.

Preserve R5/R6 proofs and their replay arithmetic. Bind actual statement
descriptions, not every incidental date in an inventory/investment note. The
unchanged native audit classifier still owns opinion and hard-risk semantics.
This component is deliberately not selected by the daily R23 policy.
"""
from dataclasses import asdict, replace
from datetime import date
import re
from zoneinfo import ZoneInfo

import pandas as pd

from domain.fundamental_derivation import (
    _AUDIT_CONCEPTS, _derive_period_audit_evidence, _qualified_basis_text,
)
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.financial_sources_r5 import _BODY, _section_view, audit_proof_r5
from domain.text_normalization import normalize_cjk_text

VERSION = "SOURCE_BOUND_STATEMENT_DESCRIPTIONS_AUDIT_R7_CANDIDATE"
SOURCE_VERSION = "SOURCE_BOUND_STATEMENT_DESCRIPTIONS_AUDIT_R7"
AVAILABLE_AT = "2026-10-28T12:41:27Z"
PLACEHOLDERS = frozenset({"", "na", "n/a"})
_CN_DIGITS = "零一二三四五六七八九"
_CN_YEAR = r"[0-9零〇一二三四五六七八九]{3,4}"
_CN_SCOPE = re.compile(
    r"[民國0-9零〇一二三四五六七八九十百千年月日及、與/\-]{0,180}"
    r"(?:之|的)?(?:期中|簡明)?(合併|個體|個別)(?:期中|簡明)?"
    r"(?:資產負債表|財務報表|綜合損益表|權益變動表|現金流量表|財務狀況|財務績效)")
_EN_SCOPE = re.compile(
    r"\b(consolidated|separate|individual)\s+(?:(?:interim|condensed)\s+)*"
    r"(?:financial statements?|balance sheets?|financial position|statements of "
    r"(?:comprehensive income|cash flows|changes in equity))\b", re.I)
_ACTIVE_UNREVIEWED = re.compile(
    r"\bwe\s+did\s+not\s+review\s+the\s+financial\s+statements\s+of\s+certain\s+"
    r"(?=non-?significant\s+consolidated\s+subsidiar)", re.I)
_ACCESS_FAILURE = re.compile(
    r"未能接觸|無法接觸|無法取得|未能取得|無法判斷|"
    r"\b(?:unable|not able)\s+to\s+(?:access|obtain|contact|determine)\b|"
    r"\b(?:could not|cannot|can not|can't)\s+(?:access|obtain|contact|determine)\b|"
    r"\b(?:denied|no)\s+access\b|\bdid not obtain\s+(?:sufficient|audit|review|financial)\b", re.I)


def policy():
    return dict(version=VERSION, period="EXPLICIT_CURRENT_ENDPOINT_AND_REPORT_SCOPE",
        descriptions="ALL_BOUND_REPORT_DESCRIPTIONS_MUST_AGREE_NOT_INCIDENTAL_NOTE_DATES",
        placeholders=sorted(PLACEHOLDERS), placeholder_effect="MISSING_TRANSLATION_NOT_OPINION_EVIDENCE",
        classification="NATIVE_CLASSIFIER_BOUND_BASIS_LEXICAL_EQUIVALENCE_ACCESS_FAILURE_STAYS_UNKNOWN",
        missing_policy="UNKNOWN_NO_SCOPE_OR_OPINION_IMPUTATION", daily_enabled=False)


def _cn_number(number):
    return (_CN_DIGITS[number] if number < 10 else
        ("" if number < 20 else _CN_DIGITS[number // 10]) + "十" +
        (_CN_DIGITS[number % 10] if number % 10 else ""))


def statement_bindings(text, period):
    """Return explicit current-period scope assertions plus ignored date count.

    A second explicit assertion with a different scope is a conflict. An
    incidental date without a statement description supplies no scope at all.
    """
    normalized = normalize_cjk_text(text)
    compact = re.sub(r"\s+", "", normalized)
    roc = period.year - 1911
    year = rf"(?:{period.year}|{roc}|{''.join(_CN_DIGITS[int(c)] for c in str(roc))})"
    month, day = period.end_time.month, period.end_time.day
    cn = re.compile(r"(?<![0-9零〇一二三四五六七八九十百千])" + year +
        rf"(?:年(?:(?:及|、|與)(?:民國)?{_CN_YEAR}年)?|(?:及|、|與)(?:民國)?{_CN_YEAR}年)"
        rf"(?:{month}|{_cn_number(month)})月(?:{day}|{_cn_number(day)})日")
    bound, ignored = [], 0
    for match in cn.finditer(compact):
        description = _CN_SCOPE.match(compact, match.end())
        if description is None:
            ignored += 1
            continue
        bound.append(dict(period_quote=match.group(), description_quote=description.group(),
            scope="consolidated" if description[1] == "合併" else "individual"))
    en = re.compile(rf"\b{period.end_time.strftime('%B')}\s+{day},?\s+{period.year}\b", re.I)
    for match in en.finditer(normalized):
        prefix = normalized[max(0, match.start() - 500):match.start()]
        prefix = re.sub(r"\b(Inc|Ltd|Co|Corp)\.", r"\1", prefix, flags=re.I)
        prefix = re.split(r"[.;。；!?]", prefix)[-1]
        descriptions = [m for m in _EN_SCOPE.finditer(prefix)
            if re.search(r"\b(?:as at|as of|ended)\s*$", prefix[m.end():], re.I)]
        if not descriptions:
            ignored += 1
        for description in descriptions:
            bound.append(dict(period_quote=match.group(), description_quote=prefix[description.start():],
                scope="consolidated" if description[1].lower() == "consolidated" else "individual"))
    return bound, ignored


def _body_bindings(facts, period, scope):
    bindings, excluded, errors = [], [], []
    for fact in facts:
        if fact.concept != _BODY:
            continue
        text = fact.text_value or ""
        if normalize_cjk_text(text).strip().lower() in PLACEHOLDERS:
            excluded.append(dict(context_id=fact.context_id, body_sha256=identity(text),
                reason="MISSING_TRANSLATION_PLACEHOLDER_RETAINED_NOT_OPINION"))
            continue
        assertions, ignored = statement_bindings(text, period)
        if not assertions:
            errors.append("CURRENT_PERIOD_EXPLICIT_STATEMENT_SCOPE_NOT_BOUND")
        elif {a["scope"] for a in assertions} != {scope}:
            errors.append("CURRENT_PERIOD_STATEMENT_SCOPE_CONFLICT")
        bindings.append(dict(context_id=fact.context_id, body_sha256=identity(text),
            assertions=assertions, incidental_date_mentions=ignored, scope=scope))
    if not bindings:
        errors.append("CURRENT_PERIOD_AUDIT_BODY_UNAVAILABLE")
    return bindings, excluded, sorted(set(errors))


def select_audit_contexts_r7(filing, *, cutoff):
    period = pd.Period(year=filing.fiscal_year, quarter=filing.quarter, freq="Q")
    end = period.end_time.date().isoformat()
    binding = dict(method="FINANCIAL_PERIOD_CONTEXT", financial_period_end=end,
        auditor_report_date=None, body_bindings=[], excluded_bodies=[], reasons=[])
    available, at = timestamp(filing.available_at), timestamp(cutoff)
    if available > at:
        return (), dict(binding, reasons=["ORIGINAL_NOT_AVAILABLE_AT_CUTOFF"])
    facts = tuple(f for f in filing.facts if f.concept in _AUDIT_CONCEPTS and not f.dimensions)
    last = available.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
    dates = set()
    for fact in facts:
        endpoint = fact.end_date or fact.instant
        if endpoint is None:
            continue
        try:
            if date.fromisoformat(endpoint).isoformat() != endpoint:
                raise ValueError("NON_CANONICAL_DATE")
        except (TypeError, ValueError):
            return (), dict(binding, reasons=["AUDIT_CONTEXT_DATE_INVALID"])
        if end < endpoint <= last:
            dates.add(endpoint)
    current = tuple(f for f in facts if (f.end_date or f.instant) == end)
    if current and dates or not current and len(dates) != 1:
        return (), dict(binding, reasons=["CURRENT_AUDIT_CONTEXT_AMBIGUOUS_OR_MISSING"])
    if not current:
        signed = next(iter(dates))
        current = tuple(f for f in facts if (f.end_date or f.instant) == signed)
        binding.update(method="SOURCE_BOUND_ACCOUNTANT_REPORT_DATE", auditor_report_date=signed)
    bodies, excluded, errors = _body_bindings(current, period, filing.statement_scope)
    binding.update(body_bindings=bodies, excluded_bodies=excluded, reasons=errors)
    if errors:
        return (), binding
    excluded_ids = {(b["context_id"], b["body_sha256"]) for b in excluded}
    selected = tuple(f for f in current if (f.context_id, identity(f.text_value or "")) not in excluded_ids)
    return selected, binding


def classify_audit_contexts_r7(filing, current, binding):
    bodies = tuple(f for f in current if f.concept == _BODY and f.text_value)
    base = tuple(f for f in current if f.concept != _BODY)
    nodes = []
    for body in bodies:
        view = _section_view(body.text_value)
        basis = _qualified_basis_text(view, current_fiscal_year=filing.fiscal_year)
        lexical = bool(_ACTIVE_UNREVIEWED.search(basis))
        # Only enable this equivalent wording when the existing classifier
        # itself found it in the current qualified basis. Original bytes and
        # every inability-to-access/material-misstatement phrase stay intact.
        interpreted = (_ACTIVE_UNREVIEWED.sub("unreviewed financial statements of certain ", view)
            if lexical else view)
        audit = _derive_period_audit_evidence(replace(filing,
            facts=(*base, replace(body, text_value=interpreted))), allow_unreviewed_components=True)
        # An inability appearing after the conclusion still belongs to this
        # selected current report. Do not discard it because a section lexer
        # stopped at the qualified-conclusion heading.
        access_failure = bool(_ACCESS_FAILURE.search(view))
        if access_failure and audit["audit_opinion"] not in {"ADVERSE", "DISCLAIMER", "QUALIFIED_MATERIAL_MISSTATEMENT"}:
            audit.update(audit_opinion="UNKNOWN", current_audit_opinion="UNKNOWN", material_accounting_issue=None,
                audit_review_required=True, accounting_concern_classification="EVIDENCE_ACCESS_FAILURE_REQUIRES_REVIEW")
        nodes.append(dict(body_sha256=identity(body.text_value),
            interpretation_sha256=identity(interpreted), active_voice_basis_equivalence=lexical,
            evidence_access_failure=access_failure, audit=audit))
    audit = (dict(nodes[0]["audit"]) if nodes else
        _derive_period_audit_evidence(replace(filing, facts=current), allow_unreviewed_components=True))
    safety = ("audit_opinion", "going_concern", "material_accounting_issue", "audit_review_required")
    hard = [n["audit"] for n in nodes if n["audit"]["audit_opinion"] in
        {"ADVERSE", "DISCLAIMER", "QUALIFIED_MATERIAL_MISSTATEMENT"} or
        n["audit"]["going_concern"] is True or n["audit"]["material_accounting_issue"] is True]
    if hard:
        audit = dict(hard[0])
    elif any(tuple(n["audit"][k] for k in safety) != tuple(audit[k] for k in safety) for n in nodes):
        audit.update(audit_opinion="UNKNOWN", current_audit_opinion="UNKNOWN", material_accounting_issue=None,
            audit_review_required=True, accounting_concern_classification="SOURCE_LANGUAGE_CLASSIFICATION_CONFLICT")
    raw = " ".join(f.text_value for f in bodies)
    return dict(audit, audit_report_text_raw=raw, audit_report_text_normalized=normalize_cjk_text(raw),
        classification_policy=VERSION, body_classifications=nodes)


def audit_proof_r7(filing, *, symbol, cutoff, market="TWSE", original_verified=False):
    """Additive proof candidate; never mutate the original or its R5/R6 proof."""
    current, binding = select_audit_contexts_r7(filing, cutoff=cutoff)
    audit = classify_audit_contexts_r7(filing, current, binding)
    period_end = pd.Period(year=filing.fiscal_year, quarter=filing.quarter, freq="Q").end_time.date().isoformat()
    verified = bool(original_verified and filing.content_sha256 and filing.source_url and current
        and filing.period_end in {None, period_end} and audit["audit_evidence_available"])
    old = audit_proof_r5(filing, symbol=symbol, market=market, original_verified=original_verified)
    return dict(schema="financial-audit-proof/r7-candidate-v1", policy=policy(), policy_id=identity(policy()),
        source="MOPS_IXBRL", symbol=str(symbol), market=market, scope=filing.statement_scope,
        period_end=period_end, source_url=filing.source_url, content_sha256=filing.content_sha256,
        revision_number=filing.revision_number, original_available_at=filing.available_at,
        cutoff_at=timestamp(cutoff).isoformat(), original_r5_proof_id=identity(old),
        status="VERIFIED_REPORTED" if verified else "PENDING_VERIFICATION",
        reason="BOUND_CURRENT_STATEMENT_SOURCE" if verified else "ORIGINAL_OR_PERIOD_BINDING_UNVERIFIED",
        period_binding=binding, contexts=[asdict(f) for f in current], audit=audit,
        result_id=identity(audit), quality_gate_recomputed=False, entry_authority=False)


def source_policy():
    """Frozen production source interpretation; R5/R6 identities stay intact."""
    return dict(policy(), version=SOURCE_VERSION, daily_enabled="REGISTERED_R7_PROFILE_ONLY",
        parser_algorithm_id=identity(policy()), availability="ORIGINAL_OBSERVATION_UNCHANGED",
        access_failure_scope="ALL_SELECTED_CURRENT_BODY_REQUIRES_REVIEW",
        existing_verified_flags="UNCHANGED_R5_CURRENT_PERIOD_FLAGS_ONLY_WHEN_NO_REPORT_BODY_EXISTS",
        signed_report_precedence="UNCHANGED_VERIFIED_R6_SIGNED_SOURCE")


def audit_source_r7(filing, *, symbol, market="TWSE", original_verified=False):
    """Stable same-original companion, independent of each evaluation cutoff.

    Source binding is assessed as of this original's actual availability. The
    Quality policy and consumer independently enforce their evaluation cutoff;
    a parser revision never backdates a new financial observation.
    """
    candidate = audit_proof_r7(filing, symbol=symbol, market=market,
        cutoff=filing.available_at, original_verified=original_verified)
    if not any(f.concept == _BODY for f in filing.facts):
        # A previously verified, exact current-period typed opinion does not
        # need a new textual body merely because R7 repairs body parsing. No
        # fallback is permitted when an actual body is present but unbound,
        # conflicting, inaccessible or otherwise unverified.
        old = audit_proof_r5(filing, symbol=symbol, market=market, original_verified=original_verified)
        if old["status"] == "VERIFIED_REPORTED":
            candidate = dict(candidate, audit=old["audit"], contexts=old["contexts"],
                period_binding=old["period_binding"], status=old["status"],
                reason="PRESERVED_VERIFIED_R5_CURRENT_PERIOD_TYPED_FLAGS_NO_BODY")
    audit = dict(candidate["audit"], classification_policy=SOURCE_VERSION)
    return dict(schema="financial-audit-proof/v4", policy_version=SOURCE_VERSION,
        source_policy_id=identity(source_policy()), source="MOPS_IXBRL", symbol=str(symbol), market=market,
        scope=filing.statement_scope, currency="TWD", unit="thousand", source_url=filing.source_url,
        content_sha256=filing.content_sha256, revision_id=filing.content_sha256,
        revision_number=filing.revision_number, period_end=candidate["period_end"],
        available_at=filing.available_at, audit_status=filing.audit_status,
        status=candidate["status"], reason=candidate["reason"],
        period_binding=candidate["period_binding"], contexts=candidate["contexts"],
        original_contexts=[asdict(f) for f in filing.facts if f.concept in _AUDIT_CONCEPTS],
        original_r5_proof_id=candidate["original_r5_proof_id"], audit=audit, result_id=identity(audit))
