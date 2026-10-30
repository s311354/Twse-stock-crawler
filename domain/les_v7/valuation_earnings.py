"""Thin, source-bound total annual EPS; never infer a share-count denominator."""
from __future__ import annotations

from math import isfinite
import re
from types import MappingProxyType

from domain.les_v7.contracts import timestamp


SCHEMA = "valuation-earnings-source/v1"
MAPPING_VERSION = "V7_TOTAL_ANNUAL_EPS_SOURCE_R1"
EPS_CONCEPTS = MappingProxyType({
    "basic_eps": "ifrs-full:BasicEarningsLossPerShare",
    "diluted_eps": "ifrs-full:DilutedEarningsLossPerShare",
})
EPS_UNIT = "iso4217:TWD/xbrli:shares"


def _number(value):
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) else None)


def valuation_earnings(filing, *, symbol, market="TWSE", original_verified=False, as_of=None):
    """Map Q4 current/prior total EPS from one independently verified original.

    Continuing/discontinued operations are not aggregate aliases. Prior-year
    comparisons retain this document's hash and availability, never the earlier
    period's publication time. EPS is already per-share; no thousand conversion
    or net-income/EPS denominator inference is performed. A collecting caller
    supplies ``as_of``; consumers must also validate the retained availability.
    """
    if filing.quarter != 4:
        raise ValueError("V7_VALUATION_ANNUAL_FILING_REQUIRED")
    if not isinstance(symbol, str) or not symbol or market not in {"TWSE", "TPEX"}:
        raise ValueError("V7_VALUATION_EARNINGS_IDENTITY_REQUIRED")
    stop = timestamp(as_of) if as_of is not None else None
    end = f"{filing.fiscal_year}-12-31"
    origin = dict(symbol=symbol, source="MOPS_IXBRL", source_url=filing.source_url,
        period_end=end, available_at=filing.available_at, content_sha256=filing.content_sha256,
        revision_id=filing.content_sha256, revision_number=filing.revision_number,
        scope=filing.statement_scope, currency="TWD", unit="thousand")
    metadata_reason = None
    if original_verified is not True:
        metadata_reason = "ORIGINAL_NOT_VERIFIED"
    elif (not isinstance(filing.content_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", filing.content_sha256) is None
            or not isinstance(filing.source_url, str) or not filing.source_url
            or filing.statement_scope not in {"consolidated", "individual"}):
        metadata_reason = "ORIGINAL_IDENTITY_UNVERIFIED"
    elif filing.period_end not in (None, end):
        metadata_reason = "FILING_PERIOD_END_MISMATCH"
    else:
        try:
            available = timestamp(filing.available_at)
            if stop is not None and available > stop:
                metadata_reason = "FINANCIAL_AVAILABILITY_AFTER_CUTOFF"
        except (TypeError, ValueError, AttributeError):
            metadata_reason = "FINANCIAL_AVAILABILITY_UNVERIFIED"

    def resolve(field, year):
        start, period_end = f"{year}-01-01", f"{year}-12-31"
        candidates = [fact for fact in filing.facts if fact.concept == EPS_CONCEPTS[field]]
        matching = [fact for fact in candidates if not fact.dimensions
            and fact.start_date == start and fact.end_date == period_end and fact.instant is None]
        valid = [fact for fact in matching if fact.unit == EPS_UNIT and _number(fact.numeric_value) is not None]
        amounts = {_number(fact.numeric_value) for fact in valid}
        category, reason, value = None, "REPORTED_EXACT_CONTEXT", None
        if metadata_reason:
            category, reason = "PENDING_VERIFICATION", metadata_reason
        elif not candidates:
            category, reason = "MISSING_DATA", "TOTAL_EPS_CONCEPT_NOT_REPORTED"
        elif not matching:
            category, reason = "INCOMPARABLE", "NO_EXACT_AGGREGATE_ANNUAL_DURATION"
        elif any(fact.unit != EPS_UNIT for fact in matching):
            category, reason = "INCOMPARABLE", "EPS_UNIT_MISSING_OR_INCOMPATIBLE"
        elif len(amounts) > 1:
            category, reason = "CONFLICT", "CONFLICTING_AGGREGATE_EPS_VALUES"
        elif len(valid) != len(matching) or not amounts:
            category, reason = "PENDING_VERIFICATION", "NIL_OR_NONFINITE_EPS_VALUE"
        else:
            value = next(iter(amounts))
        # Relevant exact-duration facts only; no raw document or financial graph.
        contexts = [dict(concept=fact.concept, context_id=fact.context_id,
            start_date=fact.start_date, end_date=fact.end_date, instant=fact.instant,
            dimensions=list(fact.dimensions), unit=fact.unit,
            numeric_value=_number(fact.numeric_value),
            unresolved_text=fact.text_value if fact.numeric_value is None else None)
            for fact in candidates if fact.start_date == start and fact.end_date == period_end]
        proof = dict(origin, unit="TWD/share", basis="ANNUAL", start_date=start, end_date=period_end,
            status="RESOLVED" if value is not None else "UNKNOWN", category=category,
            reason=reason, mapping_version=MAPPING_VERSION, contexts=contexts)
        return value, proof

    result = dict(origin, market=market, schema=SCHEMA, mapping_version=MAPPING_VERSION,
                  status="VERIFIED_ORIGINAL" if metadata_reason is None else "PENDING_VERIFICATION")
    for year, values_key, proof_key in (
        (filing.fiscal_year, "fields", "field_evidence"),
        (filing.fiscal_year - 1, "comparative_ytd_fields", "comparative_ytd_field_evidence"),
    ):
        values, proofs = {}, {}
        for field in EPS_CONCEPTS:
            values[field], proofs[field] = resolve(field, year)
        result[values_key], result[proof_key] = values, proofs
    return result
