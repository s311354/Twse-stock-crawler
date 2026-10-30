"""Canonical iXBRL fields with exact-context provenance; no guessed values."""
from __future__ import annotations

from math import isfinite

import pandas as pd

from domain.fundamental_derivation import FiledFinancials
from domain.financial_taxonomy import (
    CASH_AUDIT_FLOWS, EXTENDED_FIELDS, EXTENDED_INSTANT_FIELDS,
    OPERATING_COST_CONCEPTS, OPERATING_COST_RECONCILIATION_POLICY,
    cash_flow_checks, interest_bearing_debt, operating_cost_checks, taxonomy_mapping,
)
from domain.strategies.low_entry_v6_ttm import accounting_basis_evidence


V6_FILING_SCHEMA_VERSION = "2026-10-05-source-bound-cost-hierarchy-field-evidence-v10"
FIELD_CONCEPTS = {
    "revenue": ("ifrs-full:Revenue", "ifrs-full:RevenueFromContractsWithCustomers"),
    "cost_of_revenue": ("ifrs-full:CostOfRevenue", "tifrs-bsci-ci:OperatingCosts"),
    # CostOfSales is a direct child when OperatingCosts is reported. Legacy
    # IFRS-only statements without other cost categories remain compatible.
    **OPERATING_COST_CONCEPTS,
    # These are NOT aliases: Taiwanese statements can report both, separated
    # by unrealized/realized sales-profit adjustments (e.g. rows 5900/5950).
    "gross_profit": ("tifrs-bsci-ci:GrossProfitLossFromOperations",),
    "gross_profit_net": ("ifrs-full:GrossProfit",),
    "unrealized_sales_profit": ("tifrs-bsci-ci:UnrealizedProfitLossFromSales",),
    "realized_sales_profit": ("tifrs-bsci-ci:RealizedProfitLossFromSales",),
    "operating_income": ("ifrs-full:ProfitLossFromOperatingActivities",),
    "net_income": ("ifrs-full:ProfitLoss",),
    "basic_eps": ("ifrs-full:BasicEarningsLossPerShare",),
    "current_assets": ("ifrs-full:CurrentAssets",),
    "total_assets": ("ifrs-full:Assets",),
    "current_liabilities": ("ifrs-full:CurrentLiabilities",),
    "total_liabilities": ("ifrs-full:Liabilities",),
    "total_equity": ("ifrs-full:Equity",),
    "cfo": ("ifrs-full:CashFlowsFromUsedInOperatingActivities",),
    "capex": ("ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
              "ifrs-full:PurchaseOfPropertyPlantAndEquipment"),
    "income_tax": ("ifrs-full:IncomeTaxExpenseContinuingOperations", "ifrs-full:IncomeTaxExpense"),
    "pretax_income": ("ifrs-full:ProfitLossBeforeTax",),
    "cash": ("ifrs-full:CashAndCashEquivalents",),
    "borrowings": ("ifrs-full:Borrowings",),
    "current_borrowings": ("ifrs-full:CurrentBorrowings",),
    "noncurrent_borrowings": ("ifrs-full:NoncurrentBorrowings",),
    "current_lease_liabilities": ("ifrs-full:CurrentLeaseLiabilities",),
    "noncurrent_lease_liabilities": ("ifrs-full:NoncurrentLeaseLiabilities",),
}
INSTANT_FIELDS = frozenset({"current_assets", "total_assets", "current_liabilities", "total_liabilities", "total_equity",
    "cash", "borrowings", "current_borrowings", "noncurrent_borrowings", "current_lease_liabilities", "noncurrent_lease_liabilities", *EXTENDED_INSTANT_FIELDS})
FLOW_FIELDS = ("revenue", "cost_of_revenue", "operating_income", "net_income", "basic_eps",
               "cfo", "capex", "income_tax", "pretax_income", *CASH_AUDIT_FLOWS)


def ixbrl_record(symbol: str, filing: FiledFinancials, *, extra_fields=None, extra_instant_fields=()) -> dict:
    period = pd.Period(year=filing.fiscal_year, quarter=filing.quarter, freq="Q")
    end = period.end_time.date().isoformat()
    start = f"{filing.fiscal_year}-01-01"
    extra_mapping, taxonomy = taxonomy_mapping(filing.taxonomy_schema_refs)
    concepts = dict(FIELD_CONCEPTS, **{field: tuple(dict.fromkeys(
        (*FIELD_CONCEPTS.get(field, ()), *extra_mapping.get(field, ())))) for field in EXTENDED_FIELDS})
    # Versioned adapters may add exact-context fields, never change the legacy
    # aliases. The default remains byte-for-byte compatible with v6/R1.
    extra_fields = extra_fields or {}
    if set(extra_fields) & set(concepts):
        raise ValueError("ADDITIONAL_FINANCIAL_MAPPING_CANNOT_OVERRIDE_NATIVE_FIELDS")
    concepts.update(extra_fields)
    instant_fields = INSTANT_FIELDS | frozenset(extra_instant_fields)
    flow_fields = (*FLOW_FIELDS, *(field for field in extra_fields if field not in instant_fields))
    period_valid = filing.period_end is None or filing.period_end == end

    quarter_start = period.start_time.date().isoformat()
    revision = filing.content_sha256 or f"filing-{filing.filing_id}-r{filing.revision_number}"
    origin = dict(source="MOPS_IXBRL", source_url=filing.source_url, symbol=symbol,
                  period_end=end, available_at=filing.available_at, content_sha256=filing.content_sha256,
                  revision_id=revision, revision_number=filing.revision_number,
                  scope=filing.statement_scope, currency="TWD", unit="thousand",
                  taxonomy_schema=taxonomy["schema_name"], taxonomy_mapping_version=taxonomy["mapping_version"])

    def resolve(field, *, flow_start=start, flow_end=end):
        instant = field in instant_fields
        unit = "iso4217:TWD/xbrli:shares" if field == "basic_eps" else "iso4217:TWD"
        candidates = [f for f in filing.facts if f.concept in concepts[field]]
        matching = [f for f in candidates if not f.dimensions and
                    ((f.instant == flow_end) if instant else
                     (f.start_date == flow_start and f.end_date == flow_end))]
        valid = [f for f in matching if f.unit == unit and f.numeric_value is not None
                 and isfinite(f.numeric_value)]
        values = {float(f.numeric_value) for f in valid}
        # Never resolve contradictory aggregate aliases by silently picking one.
        if len(values) == 1 and len(valid) == len(matching) and period_valid:
            status, category, reason = "RESOLVED", None, "REPORTED_EXACT_CONTEXT"
            value = values.pop() / (1 if field == "basic_eps" else 1000)
        else:
            value, status = None, "UNKNOWN"
            if not period_valid:
                category, reason = "INCOMPARABLE", "FILING_PERIOD_END_MISMATCH"
            elif field in {*EXTENDED_FIELDS, *extra_fields} and taxonomy["status"] != "RESOLVED":
                category, reason = "PENDING_VERIFICATION", taxonomy["reason"]
            elif not candidates:
                category, reason = "MISSING_DATA", "CONCEPT_NOT_REPORTED"
            elif not matching:
                category, reason = "INCOMPARABLE", "NO_AGGREGATE_INSTANT_CONTEXT" if instant else "NO_EXACT_AGGREGATE_DURATION"
            elif any(f.unit != unit for f in matching):
                category, reason = "INCOMPARABLE", "UNIT_MISSING_OR_INCOMPATIBLE"
            elif len(values) > 1:
                category, reason = "INCOMPARABLE", "CONFLICTING_AGGREGATE_VALUES"
            elif any(f.numeric_value is None and f.text_value for f in matching):
                category, reason = "PARSE_FAILED", "NUMERIC_TEXT_UNRESOLVED"
            else:
                category, reason = "PENDING_VERIFICATION", "NIL_OR_NONFINITE_VALUE"
        contexts = [dict(concept=f.concept, context_id=f.context_id, start_date=f.start_date,
                         end_date=f.end_date, instant=f.instant, unit=f.unit, dimensions=list(f.dimensions),
                         numeric_value=f.numeric_value if f.numeric_value is not None and isfinite(f.numeric_value) else None,
                         unresolved_text=f.text_value if f.numeric_value is None else None)
                    for f in candidates]
        anchor = field in {"cash_beginning", "cash_ending"}
        return value, dict(origin, status=status, category=category, reason=reason,
                           basis="CASH_FLOW_ANCHOR" if anchor else "INSTANT" if instant else "QUARTER" if flow_start == quarter_start else "YTD",
                           start_date=None if instant else flow_start, end_date=flow_end,
                           cash_flow_period_start=flow_start if anchor else None,
                           reported_instant=flow_end if instant else None,
                           unit="TWD/share" if field == "basic_eps" else "thousand", contexts=contexts)

    profitability_checks = {}

    def amounts(flow_start, flow_end):
        values, evidence = {}, {}
        for field in concepts:
            if field in instant_fields:
                continue
            values[field], evidence[field] = resolve(field, flow_start=flow_start, flow_end=flow_end)
        def reported(field):
            return any(not c.get("dimensions") and c.get("start_date") == flow_start
                       and c.get("end_date") == flow_end for c in evidence[field].get("contexts", ()))
        if evidence["cost_of_revenue"]["reason"] in {"CONCEPT_NOT_REPORTED", "NO_EXACT_AGGREGATE_DURATION"}:
            other_categories = any(reported(k) for k in OPERATING_COST_CONCEPTS if k != "cost_of_sales")
            if reported("cost_of_sales") and not other_categories:
                values["cost_of_revenue"] = values["cost_of_sales"]
                evidence["cost_of_revenue"] = dict(evidence["cost_of_sales"],
                    selection_policy="IFRS_COST_OF_SALES_WITHOUT_OTHER_COST_CATEGORIES",
                    selection_field="cost_of_sales")
        if values["cost_of_revenue"] is None and evidence["cost_of_revenue"]["reason"] in {
                "CONCEPT_NOT_REPORTED", "NO_EXACT_AGGREGATE_DURATION"}:
            revenue, gross = values["revenue"], values["gross_profit"]
            gross_evidence = evidence["gross_profit"]
            # Ordinary IFRS statements may have only GrossProfit. It is safe
            # as a fallback only when no pre-adjustment/adjustment concepts
            # exist in this exact duration. Never hide a conflicting value.
            adjustments_present = any(
                fact.concept in (*concepts["unrealized_sales_profit"], *concepts["realized_sales_profit"])
                and fact.start_date == flow_start and fact.end_date == flow_end
                for fact in filing.facts)
            if (gross is None and gross_evidence["reason"] in {
                    "CONCEPT_NOT_REPORTED", "NO_EXACT_AGGREGATE_DURATION"}
                    and not adjustments_present):
                gross, gross_evidence = values["gross_profit_net"], evidence["gross_profit_net"]
            if revenue is not None and gross is not None:
                values["cost_of_revenue"] = revenue - gross
                evidence["cost_of_revenue"] = dict(origin, status="RESOLVED", category=None,
                    reason="REVENUE_MINUS_GROSS_PROFIT_SAME_DOCUMENT", basis="QUARTER" if flow_start == quarter_start else "YTD",
                    start_date=flow_start, end_date=flow_end, inputs=[evidence["revenue"], gross_evidence])
        checks = operating_cost_checks(values, evidence, start_date=flow_start, end_date=flow_end, taxonomy=taxonomy)
        profitability_checks[f"{flow_start}/{flow_end}"] = {
            "policy": OPERATING_COST_RECONCILIATION_POLICY,
            "margin_basis": "REVENUE_MINUS_COST_OF_REVENUE",
            "checks": checks,
            "fields": {field: {"value": values[field], "evidence": evidence[field]}
                       for field in ("cost_of_revenue", "gross_profit", "gross_profit_net",
                                     "unrealized_sales_profit", "realized_sales_profit", *OPERATING_COST_CONCEPTS)},
        }
        if any(check["blocks_cost_inputs"] for check in checks):
            failed = any(check["status"] == "FAIL" for check in checks)
            # Retain extracted operands above for audit. Invalid total/detail
            # evidence cannot be repaired by silently using revenue minus gross.
            values["cost_of_revenue"] = None
            evidence["cost_of_revenue"] = dict(evidence["cost_of_revenue"], status="UNKNOWN",
                category="INCOMPARABLE" if failed else "PENDING_VERIFICATION",
                reason="OPERATING_COST_RECONCILIATION_FAILED" if failed else "OPERATING_COST_RECONCILIATION_UNVERIFIED")
        return {k: values[k] for k in flow_fields}, {k: evidence[k] for k in flow_fields}

    values, evidence = amounts(start, end)
    quarters, quarter_evidence = amounts(quarter_start, end)
    prior, prior_evidence = ({}, {})
    if period.quarter != 1:
        prior, prior_evidence = amounts(start, (period - 1).end_time.date().isoformat())
    comparative_period = period - 4
    comparative_start = f"{filing.fiscal_year - 1}-01-01"
    comparative_end = comparative_period.end_time.date().isoformat()
    comparative, comparative_evidence = amounts(comparative_start, comparative_end)
    # CF/BS reconciliation uses the actual comparative instant in this same
    # original, not another year's independently revised balance sheet.
    for field in ("cash", "cash_beginning", "cash_ending", "cash_ias7_adjustment", "cash_disposal_group"):
        comparative[field], comparative_evidence[field] = resolve(
            field, flow_start=comparative_start, flow_end=comparative_end)
    year_end = f"{filing.fiscal_year - 1}-12-31"
    year_end_fields, year_end_evidence = {}, {}
    for field in instant_fields:
        year_end_fields[field], year_end_evidence[field] = resolve(field, flow_end=year_end)
    for field in instant_fields:
        values[field], evidence[field] = resolve(field)
    values["interest_bearing_debt"], debt_check = interest_bearing_debt(values, evidence)
    evidence["interest_bearing_debt"] = dict(origin, basis="INSTANT", instant=end, **debt_check)
    cash_checks = cash_flow_checks(values, start_date=start, end_date=end, evidence=evidence)
    if any(check["blocks_cash_inputs"] for check in cash_checks):
        # A confirmed contradiction is different from a missing audit-only
        # field. Preserve extracted amounts in diagnostics, not model inputs.
        failed = any(check["status"] == "FAIL" for check in cash_checks)
        for amounts_group, evidence_group in ((values, evidence), (quarters, quarter_evidence), (prior, prior_evidence)):
            for field in ("cfo", "capex"):
                if field in evidence_group:
                    evidence_group[field] = dict(evidence_group[field], status="UNKNOWN",
                        category="INCOMPARABLE" if failed else "PENDING_VERIFICATION",
                        reason="CASH_FLOW_RECONCILIATION_FAILED" if failed else "CASH_FLOW_RECONCILIATION_UNVERIFIED",
                        parsed_value=amounts_group.get(field))
                    amounts_group[field] = None
    comparative_checks = cash_flow_checks(comparative, start_date=comparative_start,
        end_date=comparative_end, evidence=comparative_evidence)
    if any(check["blocks_cash_inputs"] for check in comparative_checks):
        failed = any(check["status"] == "FAIL" for check in comparative_checks)
        for field in ("cfo", "capex"):
            comparative_evidence[field] = dict(comparative_evidence[field], status="UNKNOWN",
                category="INCOMPARABLE" if failed else "PENDING_VERIFICATION",
                reason="CASH_FLOW_RECONCILIATION_FAILED" if failed else "CASH_FLOW_RECONCILIATION_UNVERIFIED",
                parsed_value=comparative.get(field))
            comparative[field] = None
    assets, current = values["total_assets"], values["current_assets"]
    values["non_current_assets"] = assets - current if assets is not None and current is not None else None
    input_categories = {evidence[k]["category"] for k in ("total_assets", "current_assets")}
    missing_category = next((c for c in ("INCOMPARABLE", "PARSE_FAILED", "PENDING_VERIFICATION", "MISSING_DATA") if c in input_categories), "PENDING_VERIFICATION")
    evidence["non_current_assets"] = dict(origin, status="RESOLVED" if values["non_current_assets"] is not None else "UNKNOWN",
        category=None if values["non_current_assets"] is not None else missing_category,
        reason="ASSETS_MINUS_CURRENT_ASSETS_SAME_INSTANT" if values["non_current_assets"] is not None else "UNRESOLVED_BALANCE_INPUTS",
        basis="INSTANT", inputs=[evidence["total_assets"], evidence["current_assets"]])
    def ledger(*, first=None, last, instant=False):
        grouped = {}
        additive_extras = {name for field, names in extra_fields.items() if field not in instant_fields for name in names}
        for fact in filing.facts:
            if (fact.dimensions or not (fact.concept.startswith(("ifrs-full:", "tifrs-bsci-")) or fact.concept in additive_extras)
                    or "EarningsLossPerShare" in fact.concept or "EarningsPerShare" in fact.concept
                    or fact.unit != "iso4217:TWD"
                    or (fact.instant != last if instant else fact.start_date != first or fact.end_date != last)):
                continue
            if fact.unit != "iso4217:TWD" or fact.numeric_value is None or not isfinite(fact.numeric_value):
                grouped.setdefault(fact.concept, set()).add(None)
            else:
                grouped.setdefault(fact.concept, set()).add(float(fact.numeric_value) / 1000)
        return {key: next(iter(amounts)) if len(amounts) == 1 else None for key, amounts in grouped.items()}

    return dict(symbol=symbol, market="TWSE", currency="TWD", unit="thousand", scope=filing.statement_scope,
        source="MOPS_IXBRL", source_url=filing.source_url, flow_basis="YTD", period_end=end,
        available_at=filing.available_at, revision_id=revision, content_sha256=filing.content_sha256,
        availability_source=filing.availability_source,
        revision_number=filing.revision_number, filing_schema=V6_FILING_SCHEMA_VERSION,
        taxonomy=taxonomy, validation_checks=dict(
            period=dict(status="PASS" if period_valid else "FAIL", expected_end=end, reported_end=filing.period_end,
                        policy="EXACT_DURATION_AND_INSTANT_NO_FISCAL_DATE_SUBSTITUTION"),
            cash_flow=cash_checks, comparative_cash_flow=comparative_checks, debt=debt_check,
            profitability=profitability_checks),
        flow_derivation_policy="SAME_DOCUMENT_ONLY", field_evidence=evidence,
        quarter_fields=quarters, quarter_field_evidence=quarter_evidence,
        prior_ytd_fields=prior, prior_ytd_field_evidence=prior_evidence,
        comparative_ytd_fields=comparative, comparative_ytd_field_evidence=comparative_evidence,
        comparative_year_end_fields=year_end_fields, comparative_year_end_field_evidence=year_end_evidence,
        statement_ledger=ledger(first=start, last=end), balance_ledger=ledger(last=end, instant=True),
        comparative_statement_ledger=ledger(first=comparative_start, last=comparative_end),
        comparative_year_end_ledger=ledger(last=year_end, instant=True),
        accounting_basis_evidence=accounting_basis_evidence(filing, start, end, origin),
        correction_description=filing.correction_description,
        basic_eps_quarter=quarters["basic_eps"], **values)
