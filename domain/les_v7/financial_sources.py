"""Source-bound audit evidence for the R2 financial port, not order authority.

The existing audit classifier remains the only interpretation engine. Raw
document verification belongs to the collecting repository. Undated/prior
audit facts cannot establish the current reporting period's safety here.
"""
from dataclasses import asdict, replace
from datetime import date
from collections.abc import Mapping
from math import isclose, isfinite
from types import MappingProxyType
import re
import pandas as pd

from domain.fundamental_derivation import _AUDIT_CONCEPTS, _derive_audit_evidence, _derive_period_audit_evidence
from domain.text_normalization import normalize_cjk_text
from domain.les_v7.contracts import identity
from domain.strategies.low_entry_v6_filings import FIELD_CONCEPTS, ixbrl_record
from domain.strategies.low_entry_v6_quality_evidence import validated_proof_value


AUDIT_POLICY = "native-audit-evidence/v3"
FINANCIAL_MAPPING = "v7-financial-exact-context/r2-v1"
PAYMENT_POLICY = "VERIFIED_REPORTED_ACQUISITION_PAYMENT_MAGNITUDE_R2"
# FinanceCosts and CF AdjustmentsForInterestExpense are intentionally NOT
# aliases for reported interest expense. Treasury-share BS balances likewise
# cannot stand in for cash payments. No missing child is inferred to be zero.
EXTRA_FIELDS = MappingProxyType({
    "parent_equity": ("ifrs-full:EquityAttributableToOwnersOfParent",),
    "parent_net_income": ("ifrs-full:ProfitLossAttributableToOwnersOfParent",),
    "interest_expense": ("tifrs-notes:InterestExpense_n",),
    "intangible_capex": ("ifrs-full:PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities",),
    "accounts_receivable": ("tifrs-bsci-ci:AccountsReceivableNet",),
    "inventory": ("ifrs-full:Inventories",),
    "dividends": ("ifrs-full:DividendsPaidClassifiedAsFinancingActivities",),
    "share_repurchases": ("ifrs-full:PurchaseOfTreasuryShares", "tifrs-SCF:PaymentsToAcquireTreasuryShares"),
})
EXTRA_INSTANT = ("parent_equity", "accounts_receivable", "inventory")
PAYMENT_CONCEPTS = MappingProxyType({"capex": FIELD_CONCEPTS["capex"], **{
    field: EXTRA_FIELDS[field] for field in ("intangible_capex", "dividends", "share_repurchases")}})


def ixbrl_record_r2(symbol, filing, *, market="TWSE", original_verified=False):
    """R2-only mapping; old ixbrl_record callers keep their exact field set.

    DSO consistently uses reported ordinary net trade receivables, excluding
    related-party receivables. It is NOT labelled a total receivables measure.
    Unreported acquisition/payment fields and units remain explicit gaps.
    """
    row = ixbrl_record(symbol, filing, extra_fields=EXTRA_FIELDS, extra_instant_fields=EXTRA_INSTANT)
    row.update(market=market, financial_mapping_version=FINANCIAL_MAPPING,
               payment_policy=PAYMENT_POLICY,
               receivable_basis="REPORTED_ORDINARY_NET_TRADE_EXCLUDING_RELATED_PARTIES",
               audit_evidence=audit_proof(filing, symbol=symbol, market=market,
                                          original_verified=original_verified))
    return row


def financial_supplement(filing, *, symbol, market="TWSE", original_verified=False):
    """Thin source-bound additions to cached v6 rows; never duplicate core graphs."""
    row = ixbrl_record_r2(symbol, filing, market=market, original_verified=original_verified)
    return _financial_supplement_from_row(row, original_verified=original_verified)


def _financial_supplement_from_row(row, *, original_verified=False):
    """Project an already mapped original; no additional source or gate claim.

    The repository's verified-original companion path can share one canonical
    mapping with the audit projection. Public filing callers still enter via
    ``financial_supplement`` and retain exactly the same field/proof contract.
    """
    fields = set(EXTRA_FIELDS)
    result = {key: row.get(key) for key in ("symbol", "market", "source", "source_url", "period_end",
        "available_at", "content_sha256", "revision_id", "scope", "currency", "unit",
        "financial_mapping_version", "payment_policy", "receivable_basis")}
    result.update(schema="financial-supplement/v1", status="VERIFIED_ORIGINAL" if original_verified else "PENDING_VERIFICATION",
                  fields={field: row.get(field) for field in fields})
    for group in ("field_evidence", "quarter_fields", "quarter_field_evidence", "prior_ytd_fields",
                  "prior_ytd_field_evidence", "comparative_ytd_fields", "comparative_ytd_field_evidence",
                  "comparative_year_end_fields", "comparative_year_end_field_evidence"):
        result[group] = {field: value for field, value in row.get(group, {}).items() if field in fields}
    # Exact additive note/payment QNames participate in the same field-scoped
    # TTM ledger; no unrelated note or synthetic amount is added.
    concepts = {concept for field, names in EXTRA_FIELDS.items() if field not in EXTRA_INSTANT for concept in names}
    for group in ("statement_ledger", "comparative_statement_ledger"):
        result[group] = {key: value for key, value in row[group].items() if key in concepts}
    return result


def payment_outflow(value, *, field, periods, trace, cutoff):
    """Normalize only verified acquisition/distribution payment contexts.

    The raw additive TTM remains intact. Every successful path and source is
    checked; mixed signed conventions or a BS/proxy QName cannot earn a value.
    This is not a generic abs() repair for negative financial inputs.
    """
    receipt = dict(policy=PAYMENT_POLICY, field=field, raw_value=value, value=None,
                   status="UNKNOWN", reason="SOURCE_PAYMENT_MAPPING_UNVERIFIED")
    if field not in PAYMENT_CONCEPTS or isinstance(value, bool):
        return None, receipt
    try:
        amount, stop = float(value), pd.Timestamp(cutoff)
        if stop.tzinfo is None or not isfinite(amount) or trace.get("status") not in {"VERIFIED_REPORTED", "VERIFIED_DERIVED"}:
            return None, receipt
        if (not isclose(float(trace["value"]), amount, rel_tol=0., abs_tol=.5)
                or not trace.get("paths") or any(not isclose(float(path["value"]), amount,
                    rel_tol=0., abs_tol=.5) for path in trace["paths"])):
            return None, receipt
    except (ValueError, TypeError, KeyError):
        return None, receipt
    selected = {}
    for row in periods.values():
        origin = row.get("quarter_eps_evidence", row) if row.get("quarter_flow_source") else row
        selected[origin.get("content_sha256")] = origin
    ids = trace.get("source_ids", ())
    if not ids or any(source not in selected for source in ids):
        return None, receipt
    signs, checked = set(), []
    for source in ids:
        row = selected[source]
        if (row.get("financial_mapping_version") != FINANCIAL_MAPPING or row.get("payment_policy") != PAYMENT_POLICY
                or row.get("currency") != "TWD" or row.get("unit") != "thousand"):
            return None, receipt
        available = pd.to_datetime(row.get("available_at"), utc=True, errors="coerce")
        if pd.isna(available) or available > stop:
            return None, receipt
        # Inspect exact contexts from the bound source, including comparative
        # operands. Non-successful/missing contexts do not establish a value.
        for group in ("field_evidence", "quarter_field_evidence", "prior_ytd_field_evidence", "comparative_ytd_field_evidence"):
            proof = row.get(group, {}).get(field, {})
            if proof.get("status") != "RESOLVED":
                continue
            start, end = proof.get("start_date"), proof.get("end_date")
            parsed = validated_proof_value(proof, row, field, start=start, end=end)
            matching = [c for c in proof.get("contexts", ()) if not c.get("dimensions")
                        and c.get("start_date") == start and c.get("end_date") == end]
            if parsed is None or not matching or any(c.get("concept") not in PAYMENT_CONCEPTS[field] for c in matching):
                return None, receipt
            if parsed: signs.add(1 if parsed > 0 else -1)
            checked.append(dict(content_sha256=source, group=group, start_date=start, end_date=end,
                                concepts=sorted({c["concept"] for c in matching})))
    if not checked or len(signs) > 1 or (amount and signs and (1 if amount > 0 else -1) not in signs):
        receipt["reason"] = "PAYMENT_SIGN_OR_SOURCE_BASIS_CONFLICT"
        return None, receipt
    normalized = -amount if signs == {-1} else amount
    if normalized < 0:
        return None, receipt
    receipt.update(status="VERIFIED_DERIVED", reason="SOURCE_BOUND_PAYMENT_MAGNITUDE",
                   value=normalized, source_ids=list(ids), available_at=trace.get("available_at"), contexts=checked)
    return normalized, receipt


def _period_quotes(text, period):
    compact = re.sub(r"\s+", "", normalize_cjk_text(text))
    digits = "零一二三四五六七八九"
    small = lambda n: digits[n] if n < 10 else (("" if n < 20 else digits[n // 10]) + "十" + (digits[n % 10] if n % 10 else ""))
    roc = period.year - 1911
    years = rf"(?:{period.year}|{roc}|{''.join(digits[int(n)] for n in str(roc))})"
    month, day = period.end_time.month, period.end_time.day
    pattern = r"(?<![0-9零〇一二三四五六七八九十百千])" + years + r"年(?:及(?:\d{3,4}|[零〇一二三四五六七八九]{3,4})年)?" + rf"(?:{month}|{small(month)})月(?:{day}|{small(day)})日"
    quotes = [match.group() for match in re.finditer(pattern, compact)]
    month_name = period.end_time.strftime("%B")
    quotes.extend(match.group() for match in re.finditer(
        rf"{month_name}\s+{day},?\s+{period.year}\b", normalize_cjk_text(text), re.I))
    return tuple(dict.fromkeys(quotes))


def _period_quote(text, period):
    return next(iter(_period_quotes(text, period)), None)


def _scope_for_period(text, quote):
    """Bind scope to a financial description, never a whole-body keyword.

    A separate report that merely mentions a consolidated report cannot attest
    consolidated safety. Conflicting descriptions of the same endpoint are
    likewise not resolved by picking the more favorable sentence.
    """
    normalized = normalize_cjk_text(text)
    scopes = set()
    if "年" in quote:
        compact = re.sub(r"\s+", "", normalized)
        # Only comparative dates/connectors may separate endpoint and scope.
        # This supports both numeric and spelled-out ROC report periods.
        description = re.compile(
            r"[民國0-9零〇一二三四五六七八九十百千年月日及、與/\-]{0,180}"
            r"(?:之|的)?(?:期中|簡明)?(合併|個體|個別)(?:期中|簡明)?"
            r"(?:資產負債表|財務報表|綜合損益表|權益變動表|現金流量表)")
        for match in re.finditer(re.escape(quote), compact):
            bound = description.match(compact, match.end())
            if bound:
                scopes.add("consolidated" if bound[1] == "合併" else "individual")
    else:
        description = re.compile(
            r"\b(consolidated|separate|individual)\s+(?:(?:interim|condensed)\s+)*"
            r"(?:financial statements?|balance sheets?|statements of (?:comprehensive income|cash flows|changes in equity))\b",
            re.I)
        for match in re.finditer(re.escape(quote), normalized, re.I):
            prefix = normalized[max(0, match.start() - 500):match.start()]
            # A verified sentence can contain a company's legal abbreviation;
            # its full stop is not a report-description boundary.
            prefix = re.sub(r"\b(Inc|Ltd|Co|Corp)\.", r"\1", prefix, flags=re.I)
            prefix = re.split(r"[.;。；!?]", prefix)[-1]
            for bound in description.finditer(prefix):
                if re.search(r"\b(?:as of|ended)\b", prefix[bound.end():], re.I):
                    scopes.add("consolidated" if bound[1].lower() == "consolidated" else "individual")
    return next(iter(scopes)) if len(scopes) == 1 else None


def _audit_body_bindings(facts, period, scope):
    bodies = [f for f in facts if f.concept == "tifrs-ar:AccountantsReportBody" and f.text_value]
    bindings = []
    for fact in bodies:
        text = normalize_cjk_text(fact.text_value)[:2000]
        quotes = _period_quotes(text, period)
        if not quotes or any(_scope_for_period(text, quote) != scope for quote in quotes):
            return None  # Do not silently choose one of conflicting bodies.
        bindings.append(dict(context_id=fact.context_id, financial_period_quote=quotes[0],
            financial_period_quotes=list(quotes),
            body_sha256=identity(fact.text_value), scope=scope))
    return bindings


def select_audit_contexts(filing):
    """Fiscal context and signed-report date are distinct source semantics.

    A post-period auditor context is usable only when the same original's
    report body explicitly identifies the financial end and statement scope.
    The signing date is never used as publication/availability time.
    """
    period = pd.Period(year=filing.fiscal_year, quarter=filing.quarter, freq="Q")
    end = period.end_time.date().isoformat()
    facts = tuple(f for f in filing.facts if f.concept in _AUDIT_CONCEPTS and not f.dimensions)
    binding = dict(method="FINANCIAL_PERIOD_CONTEXT", financial_period_end=end, auditor_report_date=None, body_bindings=[])
    current = tuple(f for f in facts if (f.end_date or f.instant) == end)
    available = pd.to_datetime(filing.available_at, utc=True, errors="coerce")
    if pd.isna(available): return (), binding
    last = available.tz_convert("Asia/Taipei").date().isoformat()
    dates = set()
    for fact in facts:
        endpoint = fact.end_date or fact.instant
        if endpoint is None: continue
        try: checked = date.fromisoformat(endpoint).isoformat()
        except (TypeError, ValueError): return (), binding
        if end < checked <= last: dates.add(checked)
    if current:
        # Two different present-available opinion context interpretations are
        # not silently prioritized. Future contexts remain cutoff-ineligible.
        bodies = _audit_body_bindings(current, period, filing.statement_scope)
        return ((), binding) if dates or bodies is None else (current, dict(binding, body_bindings=bodies))
    if len(dates) != 1: return (), binding
    signed = next(iter(dates))
    selected = tuple(f for f in facts if (f.end_date or f.instant) == signed)
    bindings = _audit_body_bindings(selected, period, filing.statement_scope)
    if not bindings: return (), binding
    return selected, dict(binding, method="SOURCE_BOUND_ACCOUNTANT_REPORT_DATE", auditor_report_date=signed, body_bindings=bindings)


def classify_audit_contexts(filing, current, binding, *, allow_unreviewed_components=False):
    if binding["method"] == "FINANCIAL_PERIOD_CONTEXT":
        return _derive_audit_evidence(replace(filing, facts=current),
                                     allow_unreviewed_components=allow_unreviewed_components)
    audit = _derive_period_audit_evidence(replace(filing, facts=current),
                                         allow_unreviewed_components=allow_unreviewed_components)
    raw = " ".join(f.text_value for f in current if f.concept == "tifrs-ar:AccountantsReportBody" and f.text_value)
    return dict(audit, audit_report_text_raw=raw, audit_report_text_normalized=normalize_cjk_text(raw))


def audit_proof(filing, *, symbol, market="TWSE", original_verified=False):
    period = pd.Period(year=filing.fiscal_year, quarter=filing.quarter, freq="Q")
    end = period.end_time.date().isoformat()
    current, binding = select_audit_contexts(filing)
    audit = classify_audit_contexts(filing, current, binding)
    verified = bool(original_verified and filing.content_sha256 and filing.source_url
                    and (filing.period_end is None or filing.period_end == end)
                    and current and audit["audit_evidence_available"])
    return dict(schema="financial-audit-proof/v1", policy_version=AUDIT_POLICY,
        source="MOPS_IXBRL", source_url=filing.source_url, symbol=str(symbol), market=market,
        period_end=end, available_at=filing.available_at, scope=filing.statement_scope,
        audit_status=filing.audit_status,
        currency="TWD", unit="thousand", content_sha256=filing.content_sha256,
        revision_id=filing.content_sha256 or f"filing-{filing.filing_id}-r{filing.revision_number}",
        status="VERIFIED_REPORTED" if verified else "PENDING_VERIFICATION",
        reason="CURRENT_PERIOD_AUDIT_SOURCE_VERIFIED" if verified else
            "RAW_AUDIT_SOURCE_NOT_VERIFIED" if not original_verified else "CURRENT_PERIOD_AUDIT_CONTEXT_UNAVAILABLE",
        audit=audit, period_binding=binding, contexts=[asdict(fact) for fact in current], result_id=identity(audit))
