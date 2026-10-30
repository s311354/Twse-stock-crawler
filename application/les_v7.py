"""Shared offline v7 request entry; no infrastructure or broker dependency."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import pandas as pd

from domain.strategy import get_strategy
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.quality_policy import default_quality_policy, is_source_evidence_policy, validate_quality_policy
from domain.les_v7.financial_sources import EXTRA_FIELDS, FINANCIAL_MAPPING, PAYMENT_POLICY


def _source_bound(proof, origin):
    keys = ("symbol", "market", "scope", "currency", "unit", "source", "source_url",
            "period_end", "content_sha256", "revision_id")
    return (all(proof.get(key) == origin.get(key) for key in keys)
            and timestamp(proof["available_at"]) == timestamp(origin["available_at"]))


def _merge_financial_supplement(row, supplement):
    origin = row.get("quarter_flow_source", row)
    if (not isinstance(supplement, Mapping) or supplement.get("schema") != "financial-supplement/v1"
            or not _source_bound(supplement, origin)
            or supplement.get("financial_mapping_version") != FINANCIAL_MAPPING
            or supplement.get("payment_policy") != PAYMENT_POLICY
            or supplement.get("receivable_basis") != "REPORTED_ORDINARY_NET_TRADE_EXCLUDING_RELATED_PARTIES"):
        raise ValueError("V7_FINANCIAL_SUPPLEMENT_SOURCE_BINDING_MISMATCH")
    if supplement.get("status") != "VERIFIED_ORIGINAL":
        # Do not consume computed fields or payment authority from an original
        # that the collector could not verify. Preserve the gap and diagnostic.
        return dict(row, financial_supplement_status=dict(schema=supplement["schema"],
            status=supplement.get("status"), content_sha256=supplement.get("content_sha256"),
            available_at=supplement.get("available_at")))
    fields = supplement.get("fields", {})
    if not isinstance(fields, Mapping) or set(fields) - set(EXTRA_FIELDS):
        raise ValueError("V7_FINANCIAL_SUPPLEMENT_FIELD_SCOPE_INVALID")
    merged = dict(row)
    for field, value in fields.items():
        if row.get(field) is not None and value != row[field]:
            raise ValueError("V7_FINANCIAL_SUPPLEMENT_VALUE_CONFLICT:" + field)
        merged[field] = value
    for group in ("field_evidence", "quarter_fields", "quarter_field_evidence", "prior_ytd_fields",
                  "prior_ytd_field_evidence", "comparative_ytd_fields", "comparative_ytd_field_evidence",
                  "comparative_year_end_fields", "comparative_year_end_field_evidence"):
        additions = supplement.get(group, {})
        if not isinstance(additions, Mapping) or set(additions) - set(EXTRA_FIELDS):
            raise ValueError("V7_FINANCIAL_SUPPLEMENT_FIELD_SCOPE_INVALID:" + group)
        current = row.get(group, {})
        for field, value in additions.items():
            if current.get(field) is not None and identity(current[field]) != identity(value):
                raise ValueError("V7_FINANCIAL_SUPPLEMENT_PROOF_CONFLICT:" + group + ":" + field)
        if additions:
            merged[group] = dict(current, **additions)
    flow_concepts = {name for field, names in EXTRA_FIELDS.items() if field not in {"parent_equity", "accounts_receivable", "inventory"} for name in names}
    for group in ("statement_ledger", "comparative_statement_ledger"):
        additions = supplement.get(group, {})
        if not isinstance(additions, Mapping) or set(additions) - flow_concepts:
            raise ValueError("V7_FINANCIAL_SUPPLEMENT_LEDGER_SCOPE_INVALID")
        current = row.get(group, {})
        if any(key in current and current[key] != value for key, value in additions.items()):
            raise ValueError("V7_FINANCIAL_SUPPLEMENT_LEDGER_CONFLICT")
        if additions: merged[group] = dict(current, **additions)
    for key in ("financial_mapping_version", "payment_policy", "receivable_basis"):
        if row.get(key) is not None and row[key] != supplement[key]:
            raise ValueError("V7_FINANCIAL_SUPPLEMENT_POLICY_CONFLICT:" + key)
        merged[key] = supplement[key]
    return merged


def prepare_quality_context(context, *, financial_audits=None, financial_supplements=None, financial_repairs_r5=None, signed_financial_audits=None, financial_audits_r7=None):
    """Inject the registered independent policy only at native/public entries.

    Explicit R1/R2 policies and bundle equality retain their original semantics.
    Optional audit proofs are attached by exact original identity, never by a
    snapshot PASS flag; they are financial evidence, not market GateReceipts.
    The caller retains ownership of its original filing rows.
    """
    if context.get("strategy_policy") is not None:
        from domain.les_v7.strategy_policy import StrategyPolicy
        strategy = StrategyPolicy.load(context["strategy_policy"], cutoff=context["as_of"])
        declared = strategy.record()["quality_policy"]
        if context.get("quality_policy") is not None and identity(context["quality_policy"]) != identity(declared):
            raise ValueError("STRATEGY_QUALITY_POLICY_MISMATCH")
        context["quality_policy"] = declared
    if context.get("model_bundle") is None and context.get("quality_policy") is None:
        from domain.les_v7.rule_baseline import SOURCE_POLICIES, baseline_policy
        version = context.get("rule_baseline_policy_version")
        context["quality_policy"] = (baseline_policy(context["market"], version=version)["quality_policy"]
            if version in SOURCE_POLICIES else default_quality_policy())
    policy = context.get("quality_policy")
    if not is_source_evidence_policy(policy):
        return context
    validate_quality_policy(policy, context["as_of"], market=context["market"])
    audits = financial_audits if financial_audits is not None else context.get("financial_audits", {})
    if not isinstance(audits, Mapping):
        raise ValueError("V7_FINANCIAL_AUDIT_SOURCE_MAP_REQUIRED")
    supplements = financial_supplements if financial_supplements is not None else context.get("financial_supplements", {})
    if not isinstance(supplements, Mapping):
        raise ValueError("V7_FINANCIAL_SUPPLEMENT_SOURCE_MAP_REQUIRED")
    repairs = financial_repairs_r5 if financial_repairs_r5 is not None else context.get("financial_repairs_r5", {})
    if not isinstance(repairs, Mapping):
        raise ValueError("V7_FINANCIAL_REPAIR_SOURCE_MAP_REQUIRED")
    signed = signed_financial_audits if signed_financial_audits is not None else context.get("signed_financial_audits", {})
    if not isinstance(signed, Mapping):
        raise ValueError("V7_SIGNED_FINANCIAL_AUDIT_MAP_REQUIRED")
    r7_audits = financial_audits_r7 if financial_audits_r7 is not None else context.get("financial_audits_r7", {})
    if not isinstance(r7_audits, Mapping):
        raise ValueError("V7_R7_FINANCIAL_AUDIT_MAP_REQUIRED")
    rows, changed = [], False
    for row in context.get("filings", ()):
        origin = row.get("quarter_flow_source", row)
        supplement = supplements.get(origin.get("content_sha256"))
        if supplement is not None:
            row = _merge_financial_supplement(row, supplement)
            changed = True
        from domain.les_v7.quality_policy import is_r5_policy, is_r6_policy, is_r7_policy, _audit_proof
        repair = repairs.get(origin.get("content_sha256")) if is_r5_policy(policy) or is_r6_policy(policy) or is_r7_policy(policy) else None
        if repair is not None:
            from domain.les_v7.financial_sources_r5 import _CASH_FIELDS, SOURCE_REPAIR_POLICY_R5
            if (not isinstance(repair, Mapping) or not _source_bound(repair, origin)
                    or repair.get("source_repair_r5", {}).get("policy") != SOURCE_REPAIR_POLICY_R5
                    or set(repair.get("fields", {})) != set(_CASH_FIELDS)
                    or set(repair.get("field_evidence", {})) != set(_CASH_FIELDS)
                    or not _source_bound(repair.get("audit_evidence_r5", {}), origin)):
                raise ValueError("V7_R5_FINANCIAL_REPAIR_BINDING_MISMATCH")
            provenance = repair["source_repair_r5"]
            old_audit = audits.get(origin.get("content_sha256"), row.get("audit_evidence"))
            if (provenance.get("content_sha256") != origin.get("content_sha256")
                    or timestamp(provenance["available_at"]) != timestamp(origin["available_at"])
                    or provenance.get("old_audit_proof_id") != identity(old_audit)
                    or provenance.get("new_audit_proof_id") != identity(repair["audit_evidence_r5"])
                    or provenance.get("new_cash_proof_id") != identity(dict(repair["field_evidence"]))):
                raise ValueError("V7_R5_FINANCIAL_REPAIR_PROVENANCE_MISMATCH")
            row = dict(row, **repair["fields"],
                field_evidence=dict(row.get("field_evidence", {}), **repair["field_evidence"]),
                audit_evidence_r5=repair["audit_evidence_r5"], source_repair_r5=repair["source_repair_r5"])
            changed = True
        signed_proof = signed.get(origin.get("content_sha256")) if is_r6_policy(policy) or is_r7_policy(policy) else None
        if signed_proof is not None:
            candidate = dict(row, audit_evidence_r6=signed_proof)
            _, error = _audit_proof(candidate, period=pd.Period(origin["period_end"], freq="Q"),
                cutoff=timestamp(context["as_of"]), r5=True, r6=True)
            if error:
                raise ValueError("V7_SIGNED_AUDIT_BINDING_INVALID:" + error)
            row, changed = candidate, True
        r7_proof = r7_audits.get(origin.get("content_sha256")) if is_r7_policy(policy) else None
        if r7_proof is not None:
            if not isinstance(r7_proof, Mapping) or not _source_bound(r7_proof, origin):
                raise ValueError("V7_R7_FINANCIAL_AUDIT_BINDING_MISMATCH")
            existing = row.get("audit_evidence_r7")
            if existing is not None and identity(existing) != identity(r7_proof):
                raise ValueError("V7_R7_FINANCIAL_AUDIT_SOURCE_CONFLICT")
            candidate = dict(row, audit_evidence_r7=r7_proof)
            # Validate even if a valid R6 signed report will take precedence.
            # A genuine pending interpretation remains UNKNOWN in native
            # Quality; it is not a malformed source packet or a whole-run error.
            _, error = _audit_proof(candidate, period=pd.Period(origin["period_end"], freq="Q"),
                cutoff=timestamp(context["as_of"]), r5=True, r6=False, r7=True)
            if error and not error.startswith("AUDIT_SOURCE_PENDING:"):
                raise ValueError("V7_R7_FINANCIAL_AUDIT_BINDING_INVALID:" + error)
            row, changed = candidate, True
        proof = audits.get(origin.get("content_sha256"))
        if proof is None:
            rows.append(row)
            continue
        if not isinstance(proof, Mapping):
            raise ValueError("V7_FINANCIAL_AUDIT_SOURCE_OBJECT_REQUIRED")
        if not _source_bound(proof, origin):
            raise ValueError("V7_FINANCIAL_AUDIT_SOURCE_BINDING_MISMATCH")
        existing = row.get("audit_evidence")
        if existing is not None:
            if identity(existing) != identity(proof):
                raise ValueError("V7_FINANCIAL_AUDIT_SOURCE_CONFLICT")
            rows.append(row)
        else:
            rows.append(dict(row, audit_evidence=proof))
            changed = True
    if changed:
        context["filings"] = tuple(rows)
    return context


def analyze_request(payload: dict) -> dict:
    if not isinstance(payload, dict): raise ValueError("OBJECT_REQUEST_REQUIRED")
    plugin = get_strategy("low_entry_score_v7")
    requests = payload.get("securities", [payload])
    if not isinstance(requests, list) or not 1 <= len(requests) <= 2000:
        raise ValueError("BOUNDED_NONEMPTY_SECURITY_REQUESTS_REQUIRED")
    reports, seen = [], set()
    for raw in requests:
        request = deepcopy(raw)
        context = request["context"]
        key = (context["symbol"], context["market"], timestamp(context["as_of"]))
        if key in seen: raise ValueError("DUPLICATE_SECURITY_CUTOFF")
        seen.add(key)
        bars = pd.DataFrame(request["bars"])
        if "taiex_bars" in request:
            benchmark = pd.DataFrame(request["taiex_bars"])
            dates = pd.to_datetime(benchmark["Date"])
            if dates.duplicated().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
                raise ValueError("UNIQUE_LOCAL_BENCHMARK_DATES_REQUIRED")
            mapped = pd.Series(pd.to_numeric(benchmark["Close"], errors="raise").to_numpy(), index=dates)
            if not bars.empty:
                bars["TAIEX Close"] = pd.to_datetime(bars["Date"]).map(mapped)
        if "model_bundle" in payload:
            if "model_bundle" in context and context["model_bundle"] != payload["model_bundle"]:
                raise ValueError("CONFLICTING_MODEL_BUNDLES")
            context["model_bundle"] = deepcopy(payload["model_bundle"])
        prepare_quality_context(context)
        reports.append(plugin.evaluate(bars, context=context))
    return dict(model="low_entry_score_v7", reports=reports, research_only=True,
                execution_enabled=False, evaluation_scope="FROZEN_CUTOFF_SNAPSHOT_NOT_BROKER_EXECUTION")
