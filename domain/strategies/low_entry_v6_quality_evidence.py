"""Pure field validation for the existing Quality evaluator; never repairs facts.

Legacy normalized contracts remain readable, but their metadata is not a raw
document/context proof. Explicit canonical evidence cannot fall back to that
legacy path when a proof fails. Shared input mappings are never modified.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite

import pandas as pd

QUALITY_EVIDENCE_SCHEMA = "quality-field-evidence/v1"


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _context_value(proof, *, eps=False):
    values = []
    for context in proof.get("contexts", ()):
        if not isinstance(context, Mapping) or context.get("dimensions"):
            continue
        expected_unit = "iso4217:TWD/xbrli:shares" if eps else "iso4217:TWD"
        end = proof.get("end_date") or proof.get("instant") or proof.get("period_end")
        if proof.get("basis") == "INSTANT":
            matches = context.get("instant") == end
        else:
            matches = (context.get("start_date") == proof.get("start_date")
                       and context.get("end_date") == end)
        value = _number(context.get("numeric_value"))
        if matches and (context.get("unit") != expected_unit or value is None):
            return None
        if matches:
            values.append(value / (1. if eps else 1000.))
    if not values or any(not isclose(v, values[0], rel_tol=0., abs_tol=1e-9) for v in values):
        return None
    return values[0]


def _proof_value(proof, row, field):
    """Recheck reported values and the three existing, explicit derivations."""
    reason = proof.get("reason")
    inputs = proof.get("inputs", ())
    if reason in {"REVENUE_MINUS_GROSS_PROFIT_SAME_DOCUMENT", "ASSETS_MINUS_CURRENT_ASSETS_SAME_INSTANT"} and inputs and any(p.get("status") != "RESOLVED" or any(p.get(k) != proof.get(k) for k in
            ("content_sha256", "revision_id", "scope", "currency", "available_at", "start_date"))
            or (p.get("end_date") or p.get("instant") or p.get("period_end")) !=
               (proof.get("end_date") or proof.get("instant") or proof.get("period_end"))
            or p.get("unit") != "thousand" for p in inputs):
        return None
    if reason == "REVENUE_MINUS_GROSS_PROFIT_SAME_DOCUMENT":
        inputs = proof.get("inputs", ())
        values = [_context_value(p) for p in inputs]
        return values[0] - values[1] if len(values) == 2 and all(v is not None for v in values) else None
    if reason == "ASSETS_MINUS_CURRENT_ASSETS_SAME_INSTANT":
        inputs = proof.get("inputs", ())
        values = [_context_value(p) for p in inputs]
        return values[0] - values[1] if len(values) == 2 and all(v is not None for v in values) else None
    if reason == "VERIFIED_NONOVERLAPPING_DEBT_COMPONENTS":
        proofs = row.get("field_evidence", {})
        used = [proofs.get(key, {}) for key in proof.get("used_fields", ())]
        if any(p.get("status") != "RESOLVED" or any(p.get(k) != proof.get(k) for k in
                ("content_sha256", "revision_id", "scope", "currency", "available_at"))
                or p.get("end_date") != proof.get("instant") or p.get("unit") != "thousand" for p in used):
            return None
        values = [_context_value(p) for p in used]
        return sum(values) if values and all(v is not None for v in values) else None
    return _context_value(proof, eps=field == "basic_eps")


def validated_proof_value(proof, origin, field, *, start, end, instant=False, field_evidence=None):
    """Resolve only source-bound exact contexts, including derivation operands."""
    if (not isinstance(proof, Mapping) or proof.get("status") != "RESOLVED"
            or not origin.get("content_sha256")
            or any(proof.get(k) != origin.get(k) for k in
                   ("content_sha256", "revision_id", "scope", "currency", "available_at"))
            or proof.get("unit") != ("TWD/share" if field == "basic_eps" else "thousand")):
        return None
    if any(origin.get(k) and proof.get(k) != origin[k] for k in ("symbol", "source")):
        return None
    if instant:
        if proof.get("basis") != "INSTANT" or (proof.get("end_date") or proof.get("instant") or proof.get("period_end")) != end:
            return None
    elif proof.get("start_date") != start or proof.get("end_date") != end:
        return None
    return _proof_value(proof, {"field_evidence": field_evidence or {}}, field)


def _receipt(proof, fields=None):
    """Keep exact relevant context refs, not every comparative fact in a report."""
    if not proof:
        return {}
    end = proof.get("end_date") or proof.get("instant") or proof.get("period_end")
    contexts = [dict(context) for context in proof.get("contexts", ()) if isinstance(context, Mapping)
        and not context.get("dimensions") and
        (context.get("instant") == end if proof.get("basis") == "INSTANT" else
         context.get("start_date") == proof.get("start_date") and context.get("end_date") == end)]
    inputs = ([fields.get(k, {}) for k in proof.get("used_fields", ())]
              if proof.get("reason") == "VERIFIED_NONOVERLAPPING_DEBT_COMPONENTS" and fields is not None
              else proof.get("inputs", ()))
    return {**{key: proof.get(key) for key in ("status", "category", "reason", "basis", "start_date", "end_date",
        "instant", "period_end", "content_sha256", "revision_id", "available_at", "source", "source_url",
        "scope", "currency", "unit")}, "contexts": contexts,
        "inputs": [_receipt(p) for p in inputs], "used_fields": list(proof.get("used_fields", ())) }


def validate_field(row: Mapping, field: str, *, period, cutoff, instant=False) -> dict:
    value = _number(row.get(field))
    origin = row.get("quarter_flow_source", row) if field in {
        "cfo", "capex", "income_tax", "pretax_income", "cash", "interest_bearing_debt"} else row
    proof = row.get("field_evidence", {}).get(field, {})
    available = pd.to_datetime(origin.get("available_at"), utc=True, errors="coerce")
    end = period.end_time.date().isoformat()
    checks = dict(
        identity=bool(row.get("symbol") and row.get("market") and row.get("source")),
        period=row.get("period_end") == end,
        currency_unit=row.get("currency") == "TWD" and row.get("unit") == "thousand",
        version=bool(origin.get("revision_id")),
        availability=pd.notna(available) and available <= cutoff,
        original_context=False,
    )
    category = proof.get("category") or "MISSING_DATA"
    reason = proof.get("reason") or "FIELD_VALUE_MISSING"
    canonical = bool(row.get("filing_schema") or proof)
    if value is not None and canonical:
        start = (period.start_time.date().isoformat() if row.get("flow_basis") == "QUARTER"
                 else f"{period.year}-01-01")
        parsed = validated_proof_value(proof, origin, field, start=start, end=end,
                                      instant=instant, field_evidence=row.get("field_evidence"))
        checks["original_context"] = (parsed is not None and isclose(
            parsed, value, rel_tol=1e-12, abs_tol=1e-9 if field == "basic_eps" else .5))
        if not checks["original_context"]:
            value = None
            category = proof.get("category") or "PENDING_VERIFICATION"
            reason = proof.get("reason") if proof.get("status") != "RESOLVED" else "FIELD_PROOF_BINDING_OR_VALUE_MISMATCH"
    status = ("VERIFIED_DERIVED" if proof.get("inputs") else "VERIFIED_REPORTED") if value is not None else {
        "MISSING_DATA": "MISSING", "PARSE_FAILED": "PARSE_FAILED",
        "INCOMPARABLE": "CONFLICT" if "CONFLICT" in str(reason) else "INCOMPARABLE",
    }.get(category, "PENDING_VERIFICATION")
    if value is not None and not canonical:
        reason = "LEGACY_NORMALIZED_CONTRACT_NO_RAW_CONTEXT_PROOF"
    return dict(schema=QUALITY_EVIDENCE_SCHEMA, field=field, period=str(period), value=value,
        status=status, category=None if value is not None else category, reason=reason,
        basis="INSTANT" if instant else row.get("flow_basis"),
        evidence_level="CANONICAL_ORIGINAL" if canonical else "LEGACY_NORMALIZED_CONTRACT",
        **{k: origin.get(k) for k in ("symbol", "market", "source", "source_url", "scope", "currency",
            "unit", "revision_id", "content_sha256", "available_at", "published_at", "fetched_at", "availability_source")},
        checks=checks, evidence=_receipt(proof, row.get("field_evidence")),
        value_unit="TWD/share" if field == "basic_eps" else "TWD thousand",
        duration="INSTANT" if instant else "ANNUAL" if row.get("flow_basis") == "YTD" and period.quarter == 4 else row.get("flow_basis"))


def validate_fields(periods, *, fields, instant_fields, cutoff):
    selected, ledger = {}, {}
    for period, raw in periods.items():
        row = dict(raw)
        for field in fields:
            check = validate_field(raw, field, period=period, cutoff=cutoff, instant=field in instant_fields)
            ledger[f"{period}:{field}"] = check
            if _number(raw.get(field)) is not None and check["value"] is None:
                row[field] = None
                evidence = dict(row.get("field_evidence", {}))
                evidence[field] = dict(evidence.get(field, {}), status="UNKNOWN", category=check["category"],
                                       reason=check["reason"])
                row["field_evidence"] = evidence
        selected[period] = row
    return selected, ledger


def verification_summary(ledger: Mapping, dependencies) -> dict:
    """Count actual checks once per dependency, not once per score award."""
    keys = sorted(set(dependencies))
    passed, total = 0, 6 * len(keys)
    for key in keys:
        node = ledger.get(key, {})
        # Missing amounts do not erase identity/version checks that really ran.
        # Coverage and required-source gates still independently reject gaps.
        passed += sum(value is True for value in node.get("checks", {}).values())
    return dict(schema=QUALITY_EVIDENCE_SCHEMA, passed_checks=passed, total_checks=total,
                score=100. * passed / total if total else 0., dependencies=keys)
