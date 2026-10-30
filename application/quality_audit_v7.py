"""Quality-only audit over a fixed universe and one cutoff-selected stock at a time.

Source ports own local reads/verification; this application never refreshes a
cache, calculates LES/EREM, resolves an account or writes portfolio state.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from application.replay_contracts import aware_timestamp, canonical_json, digest

SCHEMA = "local-v7-quality-audit/v1"


@dataclass(frozen=True)
class QualityAuditStock:
    records: tuple[Mapping, ...]
    provenance: tuple[dict, ...] = ()
    diagnostics: tuple[dict, ...] = ()
    preflight: Mapping | None = None
    baseline_records: tuple[Mapping, ...] | None = None
    source_baseline_records: tuple[Mapping, ...] | None = None
    industry_evidence: Mapping | None = None


class QualityAuditSource(Protocol):
    def load_stock(self, symbol: str, cutoff: str, *, period_count: int) -> QualityAuditStock: ...


def select_audit_symbols(universe: Sequence[str], *, symbols: Sequence[str] | None = None,
                         limit: int | None = None) -> tuple[str, ...]:
    """Explicit subsets preserve fixed-universe order, never select by score."""
    fixed = tuple(universe)
    if not fixed or any(not isinstance(s, str) or not s.strip() or s != s.strip() for s in fixed) or len(set(fixed)) != len(fixed):
        raise ValueError("QUALITY_AUDIT_EXPLICIT_UNIQUE_UNIVERSE_REQUIRED")
    requested = tuple(symbols) if symbols is not None else fixed
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(fixed):
        raise ValueError("QUALITY_AUDIT_SUBSET_MUST_BELONG_TO_FIXED_UNIVERSE")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("QUALITY_AUDIT_LIMIT_MUST_BE_POSITIVE")
    selected = tuple(symbol for symbol in fixed if symbol in set(requested))
    return selected[:limit] if limit is not None else selected


def _unknown(reason: str) -> dict:
    return dict(status="DATA_INSUFFICIENT", quality_score=None, reason_codes=[reason],
                metric_ledger=[], coverage_total=None, coverage_by_domain={}, data_confidence=None)


def _field_gaps(quality):
    """Expose native field diagnostics without retaining original context graphs."""
    checks = quality.get("field_checks", ())
    from domain.les_v7.contracts import QUALITY_REPORTED_PROFILES
    reported = quality.get("profile_id") in QUALITY_REPORTED_PROFILES
    dependencies = set((quality.get("reported_result") or {}).get("metrics", {}).get(
        "quality_verification", {}).get("dependencies", ()))
    checks = checks.items() if isinstance(checks, Mapping) else ((None, check) for check in checks)
    gaps = []
    for key, check in checks:
        if not isinstance(check, Mapping): continue
        if check.get("status") in {"PASS", "VERIFIED", "RESOLVED"} or check.get("category") is None:
            continue
        gap = {name: check.get(name) for name in
            ("field", "period", "status", "category", "reason", "value_status", "evidence_level",
             "symbol", "market", "source", "source_url", "scope", "currency", "unit", "value_unit",
             "basis", "duration", "revision_id", "content_sha256", "available_at", "published_at",
             "fetched_at", "availability_source")}
        if isinstance(check.get("checks"), Mapping):
            gap["verification_checks"] = dict(check["checks"])
        if reported:
            gap.update(check_id=key, affects_gate=key in dependencies,
                       purpose="REPORTED_CORE_DEPENDENCY" if key in dependencies else "DIAGNOSTIC_ONLY")
        gaps.append(gap)
    return gaps


def _cfo_comparability(quality):
    """Diagnose the current and year-ago CFO evidence pair; never changes Quality."""
    checks = quality.get("field_checks", {})
    reported = quality.get("reported_inputs")
    if not isinstance(reported, Mapping):
        reported = (quality.get("reported_result") or {}).get("metrics", {}).get("quality_reported_inputs", {})
    if not isinstance(checks, Mapping) or not isinstance(reported, Mapping):
        return dict(status="UNKNOWN", reason="CFO_COMPARISON_LEDGER_UNAVAILABLE")
    period = reported.get("period")
    current_key = (reported.get("current_keys") or {}).get("cfo")
    comparison_key = (reported.get("comparison_keys") or {}).get("cfo")
    current, comparison = checks.get(current_key), checks.get(comparison_key)
    if not isinstance(current, Mapping) or not isinstance(comparison, Mapping):
        return dict(status="UNKNOWN", period=period, current_check_id=current_key,
                    comparison_check_id=comparison_key, reason="CFO_CURRENT_OR_COMPARATIVE_PROOF_MISSING")

    def selected(check):
        return {name: check.get(name) for name in
            ("field", "period", "status", "category", "reason", "symbol", "market", "source", "source_url", "scope",
             "currency", "unit", "basis", "revision_id", "content_sha256", "available_at",
             "published_at", "fetched_at", "availability_source")}

    current_view, comparison_view = selected(current), selected(comparison)
    expected = None
    if isinstance(period, str) and len(period) == 6 and period[4] == "Q" and period[:4].isdigit() and period[5] in "1234":
        expected = f"{int(period[:4]) - 1}Q{period[5]}"
    comparison_fields = ("symbol", "market", "source", "basis", "scope", "currency", "unit")
    same_basis = all(current.get(key) is not None and current.get(key) == comparison.get(key)
                     for key in comparison_fields)
    verified = all(check.get("status") in {"VERIFIED_REPORTED", "VERIFIED_DERIVED"}
                   and isinstance(check.get("checks"), Mapping)
                   and bool(check["checks"]) and all(check["checks"].values())
                   for check in (current, comparison))
    same_period = current.get("period") == period and comparison.get("period") == expected
    return dict(status="VERIFIED_SAME_PERIOD_BASIS" if verified and same_basis and same_period else "UNKNOWN",
        period=period, expected_comparative_period=expected, current_check_id=current_key,
        comparison_check_id=comparison_key, same_accounting_basis=same_basis,
        same_period=same_period, current=current_view, comparative=comparison_view,
        reason=None if verified and same_basis and same_period else
            "CFO_PROOF_OR_PERIOD_BASIS_COMPARABILITY_UNVERIFIED")


def _industry_classification(evidence, *, symbol, market, cutoff):
    """Verify a cutoff-eligible official sector label; never infer a numeric code.

    A current TWSE/TPEx security-master observation can establish the official
    label only from its observed availability onward. It cannot establish past
    membership or a numeric industry code without a separately verified codebook.
    """
    if not isinstance(evidence, Mapping):
        return dict(status="UNKNOWN", reason="OFFICIAL_INDUSTRY_MASTER_NOT_SUPPLIED",
                    industry_code=None, industry_code_status="UNKNOWN")
    try:
        from domain.security_master import verify_compact_security_selection
        selection = verify_compact_security_selection(dict(evidence), cutoff=cutoff)
    except (TypeError, ValueError) as error:
        return dict(status="UNKNOWN", reason="INDUSTRY_SELECTION_INVALID:" + str(error),
                    industry_code=None, industry_code_status="UNKNOWN")
    master = selection.get("master", {})
    instrument = selection.get("instrument")
    if selection.get("market") != market or selection.get("market_symbol") != symbol:
        return dict(status="CONFLICT", reason="INDUSTRY_SELECTION_MARKET_OR_SYMBOL_MISMATCH",
                    master_evidence_id=selection.get("master_evidence_id"),
                    industry_code=None, industry_code_status="UNKNOWN")
    sector = instrument.get("sector") if isinstance(instrument, Mapping) else None
    sector = sector.strip() if isinstance(sector, str) else ""
    if selection.get("status") != "VERIFIED" or not sector:
        return dict(status="UNKNOWN", reason=selection.get("reason", "OFFICIAL_SECTOR_LABEL_MISSING"),
            security_status=selection.get("status"), security_type=selection.get("security_type"),
            industry_name=sector or None, industry_code=None, industry_code_status="UNKNOWN",
            master_evidence_id=selection.get("master_evidence_id"), selection_id=selection.get("selection_id"),
            available_at=master.get("available_at"), fetched_at=master.get("fetched_at"),
            original_publication_at=master.get("original_publication_at"),
            historical_information_set_certified=master.get("historical_information_set_certified"))
    document = master.get("instruments_document", {})
    return dict(status="VERIFIED_SOURCE_LABEL", reason="OFFICIAL_ISIN_INDUSTRY_LABEL_BOUND_TO_SECURITY_ROW",
        industry_name=sector, industry_code=None, industry_code_status="UNKNOWN_NO_VERIFIED_CODEBOOK_MAPPING",
        security_status=selection["status"], security_type=selection.get("security_type"),
        source="TWSE_TPEX_OFFICIAL_ISIN_MASTER", source_url=document.get("url"),
        source_document_sha256=document.get("sha256"), source_row_sha256=instrument.get("source_row_sha256"),
        master_evidence_id=selection.get("master_evidence_id"), selection_id=selection.get("selection_id"),
        market=market, symbol=symbol, cutoff=cutoff, available_at=master.get("available_at"),
        fetched_at=master.get("fetched_at"), original_publication_at=master.get("original_publication_at"),
        availability_basis=master.get("availability_basis"), time_policy=master.get("time_policy"),
        historical_information_set_certified=master.get("historical_information_set_certified"),
        inclusion_proof=selection.get("instrument_inclusion"))


def _financial_source_diagnostics(records):
    """Compact, source-bound audit-period/opinion and cash-mapping receipts."""
    from domain.financial_taxonomy import CASH_AUDIT_FLOWS, CASH_RECONCILING_COMPONENTS, TAXONOMY_PROFILES
    from domain.strategies.low_entry_v6_filings import FIELD_CONCEPTS

    cash_fields = tuple(dict.fromkeys(("cfo", *CASH_AUDIT_FLOWS, "cash_beginning", "cash_ending",
        "cash", *CASH_RECONCILING_COMPONENTS)))
    result = []
    for record in records:
        taxonomy = record.get("taxonomy") or {}
        schema = taxonomy.get("schema_name")
        profile = TAXONOMY_PROFILES.get(schema, {})
        mappings = []
        evidence_groups = (("CURRENT", "field_evidence", ""),
                           ("COMPARATIVE_YTD", "comparative_ytd_field_evidence", "comparative_ytd_fields"))
        for group_name, evidence_name, values_name in evidence_groups:
            evidence_group = record.get(evidence_name) or {}
            values_group = record.get(values_name) or {}
            for field in cash_fields:
                proof = evidence_group.get(field) or {}
                allowed = tuple(dict.fromkeys((*FIELD_CONCEPTS.get(field, ()), *profile.get(field, ()))))
                contexts = [context for context in proof.get("contexts", ()) if isinstance(context, Mapping)]
                observed = sorted({context["concept"] for context in contexts if context.get("concept")})
                mappings.append(dict(group=group_name, field=field,
                    value=values_group.get(field) if values_name else record.get(field), expected_concepts=list(allowed),
                    observed_concepts=observed, mapping_status="MAPPING_CONFIGURED" if schema in TAXONOMY_PROFILES else
                    "BASE_MAPPING_ONLY" if FIELD_CONCEPTS.get(field) else "UNKNOWN",
                    taxonomy_schema=schema, taxonomy_mapping_version=taxonomy.get("mapping_version"),
                    taxonomy_status=taxonomy.get("status"), taxonomy_reason=taxonomy.get("reason"),
                    evidence_status=proof.get("status"), evidence_category=proof.get("category"),
                    evidence_reason=proof.get("reason"), basis=proof.get("basis"),
                    start_date=proof.get("start_date"), end_date=proof.get("end_date"),
                    reported_instant=proof.get("reported_instant"), cash_flow_period_start=proof.get("cash_flow_period_start"),
                    source=proof.get("source"), source_url=proof.get("source_url"), symbol=proof.get("symbol"),
                    scope=proof.get("scope"), currency=proof.get("currency"), unit=proof.get("unit"),
                    content_sha256=proof.get("content_sha256"), revision_id=proof.get("revision_id"),
                    available_at=proof.get("available_at"), fetched_at=record.get("fetched_at"),
                    contexts=[dict(concept=context.get("concept"), context_id=context.get("context_id"),
                        start_date=context.get("start_date"), end_date=context.get("end_date"),
                        instant=context.get("instant"), unit=context.get("unit"),
                        dimensions=context.get("dimensions"), numeric_value=context.get("numeric_value"))
                        for context in contexts]))
        checks = record.get("validation_checks") or {}
        audit_proof = (record.get("audit_evidence_r6") or record.get("audit_evidence_r5")
                       or record.get("audit_evidence") or {})
        audit = audit_proof.get("audit") or {}
        binding = audit_proof.get("period_binding") or {}
        opinion_keys = ("audit_opinion", "current_audit_opinion", "comparative_audit_opinion",
            "audit_opinion_source", "comparative_audit_opinion_source", "audit_evidence_available",
            "audit_review_required", "material_accounting_issue", "going_concern",
            "accounting_concern_classification", "classification_policy")
        body_bindings = binding.get("body_bindings", ())
        body_classifications = audit.get("body_classifications", ())
        source_binding_keys = ("symbol", "market", "scope", "currency", "unit", "period_end", "source",
                               "content_sha256", "available_at", "source_url")
        source_binding_checks = {key: audit_proof.get(key) == record.get(key)
                                 for key in source_binding_keys if key in audit_proof}
        source_binding_complete = set(source_binding_checks) == set(source_binding_keys)
        period_binding_ok = binding.get("financial_period_end") == record.get("period_end")
        binding_method = binding.get("method")
        if binding_method == "SOURCE_BOUND_ACCOUNTANT_REPORT_DATE":
            report_context_ok = bool(binding.get("auditor_report_date") and body_bindings and all(
                item.get("scope") == record.get("scope") and item.get("financial_period_quotes")
                for item in body_bindings))
        elif binding_method == "SOURCE_BOUND_SIGNED_REPORT_PRECEDENCE_R1":
            report_context_ok = bool(binding.get("auditor_report_date") and binding.get("signed_source_id")
                and body_bindings and all(item.get("scope") == record.get("scope")
                                          and item.get("financial_period_quotes") for item in body_bindings))
        elif binding_method == "FINANCIAL_PERIOD_CONTEXT":
            report_context_ok = binding.get("auditor_report_date") is None and period_binding_ok
        else:
            report_context_ok = False
        bound_opinion = audit.get("audit_opinion")
        opinion_parsed = bound_opinion not in (None, "UNKNOWN", "QUALIFIED_UNKNOWN")
        audit_binding_status = ("VERIFIED_BOUND_AND_PARSED" if audit_proof.get("status") == "VERIFIED_REPORTED"
            and source_binding_complete and all(source_binding_checks.values()) and period_binding_ok
            and report_context_ok and opinion_parsed else
            "CONFLICT" if source_binding_checks and not all(source_binding_checks.values())
            or binding.get("financial_period_end") and not period_binding_ok else "UNKNOWN")
        result.append(dict(symbol=record.get("symbol"), market=record.get("market"),
            period_end=record.get("period_end"), available_at=record.get("available_at"),
            fetched_at=record.get("fetched_at"), availability_source=record.get("availability_source"),
            source=record.get("source"), source_url=record.get("source_url"), scope=record.get("scope"),
            currency=record.get("currency"), unit=record.get("unit"), content_sha256=record.get("content_sha256"),
            revision_id=record.get("revision_id"), revision_number=record.get("revision_number"),
            taxonomy=dict(status=taxonomy.get("status"), reason=taxonomy.get("reason"), schema_name=schema,
                schema_refs=taxonomy.get("schema_refs"), mapping_version=taxonomy.get("mapping_version")),
            audit_report=dict(status=audit_proof.get("status"), reason=audit_proof.get("reason"),
                policy_version=audit_proof.get("policy_version"), source=audit_proof.get("source"),
                source_url=audit_proof.get("source_url"), content_sha256=audit_proof.get("content_sha256"),
                available_at=audit_proof.get("available_at"), scope=audit_proof.get("scope"),
                period_end=audit_proof.get("period_end"), period_binding=dict(binding),
                binding_verification=dict(status=audit_binding_status,
                    source_fields=source_binding_checks, financial_period_end_matches=period_binding_ok,
                    report_body_matches_period_and_scope=report_context_ok,
                    opinion_parsed=opinion_parsed),
                opinions={key: audit.get(key) for key in opinion_keys if key in audit},
                body_sha256=sorted({item.get("body_sha256") for item in body_bindings if item.get("body_sha256")} | {
                    item.get("body_sha256") for item in body_classifications if item.get("body_sha256")} ),
                body_classifications=[dict(body_sha256=item.get("body_sha256"),
                    interpretation_sha256=item.get("interpretation_sha256"),
                    opinions={key: item.get("audit", {}).get(key) for key in opinion_keys
                              if key in item.get("audit", {})}) for item in body_classifications]),
            cash_flow_mapping=dict(schema_name=schema, mapping_version=taxonomy.get("mapping_version"),
                fields=mappings, current_reconciliation=checks.get("cash_flow", []),
                comparative_reconciliation=checks.get("comparative_cash_flow", []))))
    return result


def audit_quality_v7(source: QualityAuditSource, universe: Sequence[str], cutoff: str, *,
                     market: str = "TWSE", symbols: Sequence[str] | None = None,
                     limit: int | None = None, period_count: int = 9,
                     policy: Mapping | None = None, data_kind: str = "LOCAL_CACHE",
                     source_metadata: Mapping | None = None, progress: Callable | None = None,
                     evaluator: Callable | None = None, policy_resolver: Callable | None = None) -> dict:
    """Recompute native financial Quality only; R1 is a same-original diagnostic.

    R1 receives no invented market SAFETY receipt. Its result is not a restored
    historical deployed run or evidence that a market gate passed. R2 financial
    safety is established by its native source-bound adapter, not this audit.
    """
    from domain.les_v7.quality import evaluate_financials
    from domain.les_v7.quality_policy import (default_quality_policy, r2_quality_policy, r3_quality_policy, r5_quality_policy, r6_quality_policy,
        is_r3_policy, is_r4_policy, is_r5_policy, is_r6_policy, is_reported_policy, validate_quality_policy)
    from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION
    cutoff = aware_timestamp(cutoff).isoformat()
    if market not in {"TWSE", "TPEX"}: raise ValueError("QUALITY_AUDIT_MARKET_REQUIRED")
    if data_kind not in {"LOCAL_CACHE", "SYNTHETIC_ONLY"}: raise ValueError("QUALITY_AUDIT_DATA_KIND_REQUIRED")
    if type(period_count) is not int or not 8 <= period_count <= 20:
        raise ValueError("QUALITY_AUDIT_PERIOD_COUNT_MUST_BE_8_TO_20")
    fixed = tuple(universe)
    selected = select_audit_symbols(fixed, symbols=symbols, limit=limit)
    requested = tuple(symbol for symbol in fixed if symbols is None or symbol in set(symbols))
    evaluator = evaluator or evaluate_financials
    resolver = policy_resolver or default_quality_policy
    rows = []
    for index, symbol in enumerate(selected):
        try:
            stock = source.load_stock(symbol, cutoff, period_count=period_count)
        except (OSError, ValueError) as error:
            reason = "LOCAL_EVIDENCE_READ_FAILED:" + str(error)
            rows.append(dict(symbol=symbol, market=market, before=_unknown(reason), after=_unknown(reason),
                policy=None, originals=[], diagnostics=[dict(category="LOCAL_READ_FAILURE", reason=str(error))],
                field_gaps=[], cfo_comparability=dict(status="UNKNOWN", reason="LOCAL_EVIDENCE_READ_FAILED"),
                industry_classification=dict(status="NOT_EVALUATED",
                    reason="NO_SOURCE_BOUND_INDUSTRY_CLASSIFICATION_IN_FINANCIAL_AUDIT"),
                financial_source_evidence=[], preflight={}))
        else:
            resolved = (validate_quality_policy(policy, cutoff, market=market, records=stock.records, symbol=symbol)
                if policy is not None else resolver(stock.records, symbol=symbol, market=market, cutoff=cutoff))
            # Both profiles consume the same selected original versions; the
            # R2-only field mapping is not retroactively applied to R1. Neither
            # side creates a bars-based Safety receipt for the R1 engine.
            # An R2 payment policy must not be passed off as an R1 policy: the
            # frozen legacy evaluator intentionally rejects that rule name.
            # This is an explicit compatibility diagnostic, not a recreation
            # of an old deployed report whose policy/model may be absent.
            reported = is_reported_policy(resolved)
            repaired = is_r5_policy(resolved) or is_r6_policy(resolved)
            baseline_resolver = r5_quality_policy if is_r6_policy(resolved) else default_quality_policy if repaired else r3_quality_policy if is_r4_policy(resolved) else r2_quality_policy
            baseline_policy = (baseline_resolver(stock.records, symbol=symbol, market=market, cutoff=cutoff)
                if reported else dict(profile_id="QG_V7_BOUNDED_GAPS_R1",
                source_mapping_id="LEGACY_IXBRL_RECORD:" + V6_FILING_SCHEMA_VERSION,
                filing_schedule_version="LATEST_VISIBLE_PERIOD_DIAGNOSTIC_NOT_OFFICIAL_FILING_SCHEDULE",
                latest_required_period=resolved["latest_required_period"],
                tax_rule="VERIFIED_TTM_EXPENSE_OVER_PRETAX", capex_rule="PPE_PLUS_INTANGIBLES_POSITIVE_OUTFLOW"))
            before = evaluate_financials(stock.source_baseline_records if repaired and stock.source_baseline_records is not None else stock.records if reported else
                                        stock.baseline_records if stock.baseline_records is not None else stock.records,
                                        symbol=symbol, market=market,
                                        cutoff=cutoff, policy=baseline_policy)
            after = evaluator(stock.records, symbol=symbol, market=market, cutoff=cutoff, policy=resolved)
            canonical_json(before)
            canonical_json(after)
            row = dict(symbol=symbol, market=market, before=before, after=after, policy=resolved,
                baseline_policy=baseline_policy,
                originals=list(stock.provenance), diagnostics=list(stock.diagnostics),
                field_gaps=_field_gaps(after), cfo_comparability=_cfo_comparability(after),
                industry_classification=_industry_classification(stock.industry_evidence,
                    symbol=symbol, market=market, cutoff=cutoff),
                financial_source_evidence=_financial_source_diagnostics(stock.records),
                preflight=dict(stock.preflight or {}))
            if repaired:
                row.update(before_field_gaps=_field_gaps(before),
                    source_policy_ids=dict(before_policy_sha256=baseline_policy["policy_sha256"],
                        after_policy_sha256=resolved["policy_sha256"],
                        source_mapping_sha256=resolved["source_mapping_sha256"]),
                    source_repairs=[dict(r["source_repair_r5"]) for r in stock.records if r.get("source_repair_r5")],
                    before_result_id=digest(before), after_result_id=digest(after))
                if is_r6_policy(resolved):
                    row["signed_reports"] = [dict(r["audit_evidence_r6"]) for r in stock.records if r.get("audit_evidence_r6")]
            rows.append(row)
            del stock
        if progress is not None: progress(index + 1, len(selected), symbol)
    summary = dict(fixed_universe_count=len(fixed), requested_count=len(requested), processed_count=len(rows),
        unprocessed_count=len(fixed) - len(rows), full_universe_processed=len(rows) == len(fixed),
        before_status_counts=dict(Counter(row["before"].get("status", "UNKNOWN") for row in rows)),
        after_status_counts=dict(Counter(row["after"].get("status", "UNKNOWN") for row in rows)),
        after_reason_counts=dict(Counter(reason for row in rows for reason in row["after"].get("reason_codes", ()))),
        source_diagnostic_counts=dict(Counter(item.get("category", "UNKNOWN") for row in rows for item in row["diagnostics"])),
        field_gap_category_counts=dict(Counter(item.get("category", "UNKNOWN") for row in rows for item in row["field_gaps"])),
        industry_classification_status_counts=dict(Counter(row.get("industry_classification", {}).get("status", "UNKNOWN")
            for row in rows)),
        audit_opinion_status_counts=dict(Counter(item.get("audit_report", {}).get("opinions", {}).get("audit_opinion", "UNKNOWN")
            for row in rows for item in row.get("financial_source_evidence", ()))),
        cash_flow_mapping_status_counts=dict(Counter(field.get("mapping_status", "UNKNOWN")
            for row in rows for item in row.get("financial_source_evidence", ())
            for field in item.get("cash_flow_mapping", {}).get("fields", ()))))
    r5 = (any(is_r5_policy(row.get("policy")) for row in rows)
          or is_r5_policy(policy) or policy is None and resolver is r5_quality_policy)
    r6 = (any(is_r6_policy(row.get("policy")) for row in rows)
          or is_r6_policy(policy) or policy is None and resolver is r6_quality_policy)
    r4 = (any(is_r4_policy(row.get("policy")) for row in rows)
          or is_r4_policy(policy) or policy is None and resolver is default_quality_policy)
    reported = (r6 or r5 or r4 or any(is_r3_policy(row.get("policy")) for row in rows)
                or is_r3_policy(policy) or policy is None and resolver is r3_quality_policy)
    body = dict(schema="local-v7-quality-audit/v5" if r6 else "local-v7-quality-audit/v4" if r5 else "local-v7-quality-audit/v3" if r4 else "local-v7-quality-audit/v2" if reported else SCHEMA,
        field_evidence_schema="quality-field-audit/v1",
        source_evidence_schema="quality-source-evidence/v1",
        data_kind=data_kind, cutoff=cutoff, market=market, period_count=period_count,
        fixed_universe=list(fixed), requested_symbols=list(requested), processed_symbols=list(selected),
        unprocessed_symbols=[symbol for symbol in fixed if symbol not in selected],
        selection_policy="EXPLICIT_FIXED_UNIVERSE_ORDER_OPTIONAL_EXPLICIT_SUBSET_AND_PREFIX_LIMIT_NEVER_QUALITY_FILTER",
        source_metadata=dict(source_metadata or {}), rows=rows, summary=summary,
        availability_policy="STORED_OBSERVATION_CUTOFF_NOT_A_HISTORICAL_PUBLICATION_PROOF",
        comparison="SAME_LOCAL_ORIGINALS_R5_VS_R6_SIGNED_REPORT_SOURCE_PRECEDENCE_FINANCIAL_QUALITY_ONLY" if r6 else
            "SAME_LOCAL_ORIGINALS_R4_VS_R5_SOURCE_BOUND_AUDIT_CASH_REPAIR_FINANCIAL_QUALITY_ONLY" if r5 else
            "SAME_LOCAL_ORIGINALS_R3_REPORTED_PERIOD_VS_R4_LIMITED_MISSING_FINANCIAL_QUALITY_ONLY" if r4 else
            "SAME_LOCAL_ORIGINALS_R2_TTM_REQUIRED_VS_R3_REPORTED_PERIOD_FINANCIAL_QUALITY_ONLY" if reported else
            "SAME_LOCAL_ORIGINALS_LEGACY_R1_MAPPING_WITHOUT_MARKET_SAFETY_RECEIPT_VS_R2_MAPPING_FINANCIAL_QUALITY_ONLY",
        no_network=True, no_source_cache_writes=True, no_account_or_pae_state=True,
        not_a_full_les_erem_pae_run=True, no_performance_or_survivorship_claim=True)
    if reported:
        body["quality_profile"] = "QG_V7_REPORTED_PERIOD_R6" if r6 else "QG_V7_REPORTED_PERIOD_R5" if r5 else "QG_V7_REPORTED_PERIOD_R4" if r4 else "QG_V7_REPORTED_PERIOD_R3"
        body["baseline_profile"] = "QG_V7_REPORTED_PERIOD_R5" if r6 else "QG_V7_REPORTED_PERIOD_R4" if r5 else "QG_V7_REPORTED_PERIOD_R3" if r4 else "QG_V7_EVIDENCE_R2"
        summary["core_field_gap_category_counts"] = dict(Counter(item["category"] for row in rows
            for item in row["field_gaps"] if item.get("affects_gate") is True))
        summary["diagnostic_field_gap_category_counts"] = dict(Counter(item["category"] for row in rows
            for item in row["field_gaps"] if item.get("affects_gate") is False))
    if r5 or r6:
        summary["before_financial_safety_counts"] = dict(Counter(row["before"].get("financial_safety", {}).get("state", "UNKNOWN") for row in rows))
        summary["after_financial_safety_counts"] = dict(Counter(row["after"].get("financial_safety", {}).get("state", "UNKNOWN") for row in rows))
        summary["before_reason_counts"] = dict(Counter(reason for row in rows for reason in row["before"].get("reason_codes", ())))
        body.update(account=None, source_repair_policy="SAME_ORIGINAL_AUDIT_CASH_REBUILD_R5",
                    source_repair_scope="CURRENT_AND_COMPARATIVE_ORIGINAL_CONTEXTS_NO_HISTORY_BACKDATE")
        if r6:
            body["source_precedence_policy"] = "SOURCE_BOUND_SIGNED_REPORT_PRECEDENCE_R1"
    return dict(body, audit_id=digest(body))


def quality_audit_summary(report: Mapping) -> str:
    summary = report["summary"]
    statuses = ", ".join(f"{key}={value}" for key, value in sorted(summary["after_status_counts"].items()))
    return (f"Quality v7 local audit {report['cutoff']} [{report['data_kind']}]\n"
            f"Fixed universe {summary['fixed_universe_count']}; requested {summary['requested_count']}; "
            f"processed {summary['processed_count']}; unprocessed {summary['unprocessed_count']}\n"
            f"{'R6' if report.get('quality_profile') == 'QG_V7_REPORTED_PERIOD_R6' else 'R5' if report.get('quality_profile') == 'QG_V7_REPORTED_PERIOD_R5' else 'R4' if report.get('quality_profile') == 'QG_V7_REPORTED_PERIOD_R4' else 'R3' if report.get('quality_profile') == 'QG_V7_REPORTED_PERIOD_R3' else 'R2'}: {statuses or 'no processed stocks'}\n"
            "Financial Quality only; no LES/EREM/PAE, network, mail, accounts or cache updates.\n"
            "Stored availability does not establish historical publication or unbiased universe membership.")
