"""Version-bound Taiwan iXBRL mappings and deterministic financial controls.

Only the reviewed subset used by v6 is mapped, not the entire taxonomy. A new
schema date is not silently treated as the latest supported schema. These
controls do not infer publication times, omitted amounts, or investment PASS.
"""
from __future__ import annotations

from html.parser import HTMLParser
from datetime import datetime
from math import isclose, isfinite
from types import MappingProxyType
from urllib.parse import urlsplit

TAXONOMY_MAPPING_VERSION = "2026-10-05-tifrs-cost-hierarchy-v4"
CASH_RECONCILIATION_POLICY = "source-bound-cash-reconciliation/v3"
OPERATING_COST_RECONCILIATION_POLICY = "source-bound-operating-cost-hierarchy/v1"
IAS7_OTHER_CASH_CONCEPT = "tifrs-SCF:OtherItemsQualifyingForCashAndCashEquivalentsUnderTheDefinitionOfIas7"
DISPOSAL_GROUP_CASH_CONCEPT = "ifrs-full:CashAndCashEquivalentsClassifiedAsPartOfDisposalGroupHeldForSale"
CASH_RECONCILING_COMPONENTS = ("cash_ias7_adjustment", "cash_disposal_group")

# Direct children of OperatingCosts in the reviewed 2020/2025/2026 CI
# calculation linkbases. CostOfSales is a CHILD here, not an alias for its
# parent. Preserve signed StandardCostVariances; missing children are not zero.
OPERATING_COST_CONCEPTS = {
    "cost_of_sales": ("ifrs-full:CostOfSales",),
    "cost_of_tenant_counter_sales": ("tifrs-bsci-ci:CostOfTenantCounterSalesForDepartmentStores",),
    "investment_costs": ("tifrs-bsci-ci:InvestmentCostsForInvestmentBusiness",),
    "cost_of_rental_sales": ("tifrs-bsci-ci:CostOfRentalSales",),
    "cost_of_travel_services": ("tifrs-bsci-ci:CostOfTravelServicesForHotelBusiness",),
    "cost_of_construction_and_engineering": ("tifrs-bsci-ci:CostOfConstructionAndEngineeringServiceSales",),
    "cost_of_services": ("tifrs-bsci-ci:CostOfServices",),
    "standard_cost_variances": ("tifrs-bsci-ci:StandardCostVariances",),
    "other_operating_costs": ("tifrs-bsci-ci:OtherOperatingCosts",),
    "cost_of_construction": ("tifrs-bsci-ci:CostOfConstructionSales",),
    "cost_of_engineering": ("tifrs-bsci-ci:CostOfEngineeringSales",),
}
OPERATING_COST_CHILDREN = tuple(OPERATING_COST_CONCEPTS)[:9]
CONSTRUCTION_COST_CHILDREN = ("cost_of_construction", "cost_of_engineering")

# Declared in the three reviewed official taxonomy archives; the schemaRefs
# and reported subset were checked in local 2330/2357/3014/3016/6834 originals.
# Registry entries are explicit even when this reviewed subset is unchanged.
# In particular LongtermBorrowings and NoncurrentPortionOfNoncurrentLoansReceived
# are alternatives for the SAME noncurrent loan balance, not additive items.
_TIFRS_FIELDS = {
    **OPERATING_COST_CONCEPTS,
    "current_borrowings": ("ifrs-full:CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings",),
    "investing_cash_flow": ("tifrs-SCF:NetCashFlowsFromUsedInInvestingActivities",),
    "financing_cash_flow": ("tifrs-SCF:CashFlowsFromUsedInFinancingActivities",),
    "cash_fx_effect": ("ifrs-full:EffectOfExchangeRateChangesOnCashAndCashEquivalents",),
    "cash_change": ("ifrs-full:IncreaseDecreaseInCashAndCashEquivalents",),
    "cash_beginning": ("tifrs-SCF:CashAndCashEquivalentsAtBeginningOfPeriod",),
    "cash_ending": ("tifrs-SCF:CashAndCashEquivalentsAtEndOfPeriod",),
    "cash_ias7_adjustment": (IAS7_OTHER_CASH_CONCEPT,),
    # A separate reported cash asset, NOT an alias for the IAS 7 other item or
    # the total assets of a disposal group. Count each bound component once.
    "cash_disposal_group": (DISPOSAL_GROUP_CASH_CONCEPT,),
    "short_term_borrowings": ("ifrs-full:ShorttermBorrowings",),
    "short_term_notes": ("tifrs-bsci-ci:ShorttermNotesAndBillsPayable",),
    "current_portion_borrowings": ("tifrs-bsci-ci:LongtermBorrowingsCurrentPortion",),
    "current_bonds": ("tifrs-bsci-ci:BondsPayableCurrentPortion",),
    "long_term_borrowings": ("ifrs-full:LongtermBorrowings", "ifrs-full:NoncurrentPortionOfNoncurrentLoansReceived"),
    "noncurrent_bonds": ("ifrs-full:NoncurrentPortionOfNoncurrentBondsIssued",),
    # This is NOT an interest-bearing-debt alias: it may contain non-loan items.
    "current_portion_longterm_liabilities": ("tifrs-bsci-ci:LongtermLiabilitiesCurrentPortion",),
}
TAXONOMY_PROFILES = MappingProxyType({
    "tifrs-ci-cr-2020-06-30.xsd": MappingProxyType(dict(_TIFRS_FIELDS)),
    "tifrs-ci-cr-2025-06-30.xsd": MappingProxyType(dict(_TIFRS_FIELDS)),
    "tifrs-ci-cr-2026-03-31.xsd": MappingProxyType(dict(_TIFRS_FIELDS)),
})
EXTENDED_FIELDS = tuple(_TIFRS_FIELDS)
CASH_AUDIT_FLOWS = ("investing_cash_flow", "financing_cash_flow", "cash_fx_effect", "cash_change")
DEBT_COMPONENTS = ("short_term_borrowings", "short_term_notes", "current_portion_borrowings",
                   "current_bonds", "long_term_borrowings", "noncurrent_bonds")
EXTENDED_INSTANT_FIELDS = frozenset((*DEBT_COMPONENTS, "current_portion_longterm_liabilities",
                                   "cash_beginning", "cash_ending", *CASH_RECONCILING_COMPONENTS))


class _SchemaReferences(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.refs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "link:schemaref":
            href = (dict(attrs).get("xlink:href") or "").strip()
            if href and href not in self.refs:
                self.refs.append(href)


def ixbrl_schema_refs(content: bytes | str) -> tuple[str, ...]:
    """Read raw document metadata; ignore comments/escaped text, never fetch XSDs."""
    if isinstance(content, bytes):
        for encoding in ("utf-8", "big5", "cp950", "latin-1"):
            try:
                content = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
    parser = _SchemaReferences()
    parser.feed(content)
    parser.close()
    return tuple(parser.refs)


def taxonomy_mapping(schema_refs) -> tuple[dict, dict]:
    refs = tuple(schema_refs or ())
    names = {urlsplit(ref).path.rsplit("/", 1)[-1] for ref in refs}
    supported = names & TAXONOMY_PROFILES.keys()
    trusted_hosts = {"www.xbrl.org", "xbrl.org", "mops.twse.com.tw", "mopsov.twse.com.tw", "mopsfin.twse.com.tw"}
    trusted = all(not urlsplit(ref).netloc or urlsplit(ref).hostname in trusted_hosts for ref in refs)
    valid = len(names) == len(supported) == 1 and trusted
    name = next(iter(supported)) if valid else None
    reason = ("REVIEWED_TIFRS_PROFILE" if valid else "MISSING_SCHEMA_REF" if not names
              else "UNTRUSTED_SCHEMA_REF" if not trusted else "CONFLICTING_SCHEMA_REFS" if len(names) > 1
              else "UNSUPPORTED_TAXONOMY_VERSION")
    return (dict(TAXONOMY_PROFILES[name]) if valid else {}), dict(
        status="RESOLVED" if valid else "UNKNOWN", reason=reason, schema_refs=list(refs),
        schema_name=name, mapping_version=TAXONOMY_MAPPING_VERSION,
        legacy_common_concepts_only=not refs)


def operating_cost_checks(values, evidence, *, start_date, end_date, taxonomy):
    """Source-bound total/detail/gross checks, never an absent-zero sum.

    A partial detail tie proves only the reported subset's arithmetic, not
    exhaustive scope. A partial mismatch is UNKNOWN (unreported signed costs
    may exist); a COMPLETE hierarchy mismatch or gross mismatch is FAIL.
    """
    from domain.strategies.low_entry_v6_quality_evidence import validated_proof_value

    origin = evidence["cost_of_revenue"]
    binding_keys = ("source", "source_url", "symbol", "content_sha256", "revision_id",
                    "revision_number", "scope", "currency", "unit", "available_at",
                    "period_end", "taxonomy_schema", "taxonomy_mapping_version")
    sha = origin.get("content_sha256") or ""
    verified_origin = (len(sha) == 64 and all(c in "0123456789abcdef" for c in sha.lower())
        and origin.get("source") == "MOPS_IXBRL" and bool(origin.get("source_url"))
        and bool(origin.get("symbol")) and bool(origin.get("revision_id"))
        and origin.get("scope") in {"consolidated", "individual"}
        and origin.get("currency") == "TWD" and origin.get("unit") == "thousand")
    try:
        at = datetime.fromisoformat(origin.get("available_at", "").replace("Z", "+00:00"))
        verified_origin = verified_origin and at.utcoffset() is not None
    except (ValueError, TypeError, AttributeError):
        verified_origin = False

    def reported(field):
        return any(not c.get("dimensions") and c.get("start_date") == start_date
                   and c.get("end_date") == end_date for c in evidence[field].get("contexts", ()))

    def check(name, left, right, *, missing=(), hierarchy=False):
        fields = (*left, *right)
        errors, receipts = [], []
        if not verified_origin:
            errors.append(dict(field="cost_of_revenue", reason="UNVERIFIED_RECONCILIATION_ORIGIN"))
        if hierarchy and taxonomy.get("status") != "RESOLVED":
            errors.append(dict(field="cost_of_revenue", reason="UNVERIFIED_OPERATING_COST_HIERARCHY"))
        for field in fields:
            proof = evidence[field]
            parsed = validated_proof_value(proof, origin, field, start=start_date, end=end_date)
            valid = (parsed is not None and values[field] is not None
                and isclose(parsed, values[field], rel_tol=0., abs_tol=1e-9)
                and all(proof.get(k) == origin.get(k) for k in binding_keys))
            if not valid:
                errors.append(dict(field=field, reason=proof.get("reason") if proof.get("status") != "RESOLVED"
                                   else "RECONCILIATION_SOURCE_BINDING_OR_VALUE_MISMATCH"))
            receipts.append(dict(field=field, value=values[field], **{k: proof.get(k) for k in
                (*binding_keys, "status", "reason", "basis", "start_date", "end_date")},
                contexts=[dict(c) for c in proof.get("contexts", ()) if not c.get("dimensions")
                          and c.get("start_date") == start_date and c.get("end_date") == end_date]))
        available = all(values[k] is not None for k in fields)
        expected = sum(values[k] for k in left) if available else None
        observed = sum(values[k] for k in right) if available else None
        variance = observed - expected if available else None
        tolerance = .5 * len(fields)
        complete = not missing
        status = ("UNKNOWN" if errors or not available else "PASS" if abs(variance) <= tolerance
                  else "FAIL" if complete else "UNKNOWN")
        # Explicit invalid reported operands are not mere absent categories.
        invalid_reported = any(reported(k) and values[k] is None for k in fields)
        blocks = status == "FAIL" or invalid_reported or (
            hierarchy and taxonomy.get("status") != "RESOLVED") or any(
            e["reason"] == "RECONCILIATION_SOURCE_BINDING_OR_VALUE_MISMATCH" for e in errors)
        return dict(check_id=name, policy=OPERATING_COST_RECONCILIATION_POLICY,
            start_date=start_date, end_date=end_date, inputs=list(fields),
            expected_value=expected, observed_value=observed, variance=variance, tolerance=tolerance,
            status=status, complete=complete, missing_fields=list(missing),
            scope="COMPLETE_REVIEWED_HIERARCHY" if hierarchy and complete else
                  "REPORTED_COMPONENTS_ONLY_NOT_PROOF_OF_EXHAUSTIVE_SCOPE" if hierarchy else "EXACT_REPORTED_GROSS_BASIS",
            source_binding_errors=errors, source_inputs=receipts, blocks_cost_inputs=blocks)

    checks = []
    for parent, children, name in (
        ("cost_of_construction_and_engineering", CONSTRUCTION_COST_CHILDREN, "CONSTRUCTION_COST_DETAIL_TIE"),
        ("cost_of_revenue", OPERATING_COST_CHILDREN, "OPERATING_COST_DETAIL_TIE"),
    ):
        present = tuple(k for k in children if reported(k))
        if (reported(parent) or values[parent] is not None) and present and not (
                parent == "cost_of_revenue" and origin.get("selection_field") == "cost_of_sales"):
            checks.append(check(name, present, (parent,), missing=tuple(k for k in children if not reported(k)), hierarchy=True))
    checked = {field for item in checks for field in item["inputs"]}
    # A reported leaf with an absent subtotal cannot be silently discarded if
    # its own exact-context value/unit is contradictory or unresolved.
    for field in OPERATING_COST_CONCEPTS:
        if reported(field) and field not in checked and origin.get("selection_field") != field:
            checks.append(dict(check("COST_COMPONENT_SOURCE_CHECK", (field,), (field,), hierarchy=True),
                               scope="EXACT_REPORTED_COMPONENT_SOURCE"))
    gross = "gross_profit"
    if not reported(gross) and not any(reported(k) for k in ("unrealized_sales_profit", "realized_sales_profit")):
        gross = "gross_profit_net"
    if reported(gross) and reported("revenue") and values["cost_of_revenue"] is not None:
        checks.append(check("REVENUE_COST_GROSS_TIE", ("cost_of_revenue", gross), ("revenue",)))
    # Neither adjustment is supplied as zero. Only reconcile the net bridge
    # when BOTH adjustment amounts are explicitly reported in this duration.
    if all(reported(k) for k in ("gross_profit", "gross_profit_net", "unrealized_sales_profit", "realized_sales_profit")):
        checks.append(check("GROSS_SALES_ADJUSTMENT_TIE", ("gross_profit", "realized_sales_profit"),
                            ("gross_profit_net", "unrealized_sales_profit")))
    return checks


def _cash_source_binding(values, evidence, *, components, start_date, end_date):
    """Verify every reported operand, never a plug computed from a gap.

    All amounts are from one original/revision and one closing instant. The CF
    closing anchor also binds the flow window without rewriting its context.
    Publication eligibility remains the caller's cutoff-selection responsibility.
    """
    fields = ("cash", *components, "cash_ending")
    origin = (evidence or {}).get("cash_ending", {})
    binding_keys = ("source", "source_url", "symbol", "content_sha256", "revision_id",
                    "revision_number", "scope", "currency", "unit", "available_at",
                    "period_end", "taxonomy_schema", "taxonomy_mapping_version")
    errors, receipts = [], []
    sha = origin.get("content_sha256") or ""
    if (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower())
            or not origin.get("revision_id") or not origin.get("source_url") or not origin.get("symbol")
            or origin.get("source") != "MOPS_IXBRL"
            or origin.get("scope") not in {"consolidated", "individual"}
            or origin.get("currency") != "TWD" or origin.get("unit") != "thousand"
            or origin.get("taxonomy_schema") not in TAXONOMY_PROFILES
            or origin.get("taxonomy_mapping_version") != TAXONOMY_MAPPING_VERSION):
        errors.append(dict(field="cash_ending", reason="UNVERIFIED_RECONCILIATION_ORIGIN"))
    try:
        at = datetime.fromisoformat(origin.get("available_at", "").replace("Z", "+00:00"))
        if at.utcoffset() is None:
            raise ValueError("NAIVE_AVAILABILITY")
    except (ValueError, TypeError, AttributeError):
        errors.append(dict(field="cash_ending", reason="INVALID_RECONCILIATION_AVAILABILITY"))
    for field in fields:
        proof = (evidence or {}).get(field, {})
        allowed = ("ifrs-full:CashAndCashEquivalents",) if field == "cash" else _TIFRS_FIELDS[field]
        contexts = [dict(c) for c in proof.get("contexts", ()) if not c.get("dimensions")
                    and c.get("instant") == end_date and c.get("concept") in allowed]
        valid_contexts = bool(contexts) and all(
            c.get("unit") == "iso4217:TWD" and isinstance(c.get("numeric_value"), (int, float))
            and not isinstance(c["numeric_value"], bool) and isfinite(c["numeric_value"])
            and values.get(field) is not None
            and isclose(c["numeric_value"] / 1000., values[field], rel_tol=0., abs_tol=1e-9)
            for c in contexts)
        valid = (proof.get("status") == "RESOLVED" and proof.get("reason") == "REPORTED_EXACT_CONTEXT"
                 and all(proof.get(k) == origin.get(k) for k in binding_keys)
                 and proof.get("end_date") == end_date and proof.get("reported_instant") == end_date
                 and proof.get("basis") == ("CASH_FLOW_ANCHOR" if field == "cash_ending" else "INSTANT")
                 and (field != "cash_ending" or proof.get("cash_flow_period_start") == start_date)
                 and valid_contexts)
        if not valid:
            errors.append(dict(field=field, reason=proof.get("reason") if proof.get("status") != "RESOLVED"
                               else "RECONCILIATION_SOURCE_BINDING_OR_VALUE_MISMATCH"))
        receipts.append(dict(field=field, value=values.get(field), **{k: proof.get(k) for k in
            (*binding_keys, "status", "reason", "basis", "end_date", "reported_instant", "cash_flow_period_start")},
            contexts=contexts))
    return errors, receipts


def cash_flow_checks(values, *, start_date, end_date, evidence=None) -> list[dict]:
    """Cash anchors retain their reported context; checks bind the flow window.

    Allow at most half a thousand TWD per rounded input. UNKNOWN is distinct
    from a numerically confirmed FAIL; absent FX/cash components are not zero.
    Reported IAS 7/disposal cash may bridge CF/BS only with bound raw proofs.
    """
    checks = []
    for name, left, right in (
        ("CASH_ACTIVITY_BRIDGE", ("cfo", *CASH_AUDIT_FLOWS[:3]), ("cash_change",)),
        ("CASH_ROLL_FORWARD", ("cash_beginning", "cash_change"), ("cash_ending",)),
        ("CASH_BALANCE_TIE", ("cash_ending",), ("cash",)),
    ):
        method, binding_errors, receipts = "DIRECT", [], []
        components = tuple(field for field in CASH_RECONCILING_COMPONENTS
            if values.get(field) is not None or bool((evidence or {}).get(field, {}).get("contexts")))
        if name == "CASH_BALANCE_TIE" and components:
            left, right = ("cash", *components), ("cash_ending",)
            method = ("IAS7_SOURCE_BOUND_ADJUSTMENT" if components == ("cash_ias7_adjustment",)
                      else "DISPOSAL_GROUP_SOURCE_BOUND_ADJUSTMENT" if components == ("cash_disposal_group",)
                      else "SOURCE_BOUND_CASH_COMPONENTS")
            binding_errors, receipts = _cash_source_binding(values, evidence,
                components=components, start_date=start_date, end_date=end_date)
        inputs = (*left, *right)
        available = all(values.get(k) is not None for k in inputs)
        expected = sum(values[k] for k in left) if available else None
        observed = sum(values[k] for k in right) if available else None
        variance = observed - expected if available else None
        tolerance = len(inputs) * .5
        status = "UNKNOWN" if not available or binding_errors else "PASS" if abs(variance) <= tolerance else "FAIL"
        checks.append(dict(check_id=name, start_date=start_date, end_date=end_date,
            policy=CASH_RECONCILIATION_POLICY, method=method,
            inputs=list(inputs), expected_value=expected, observed_value=observed,
            variance=variance, tolerance=tolerance,
            status=status, source_binding_errors=binding_errors, source_inputs=receipts,
            blocks_cash_inputs=bool(binding_errors) or status == "FAIL",
            missing_fields=[k for k in inputs if values.get(k) is None]))
    return checks


def interest_bearing_debt(values, evidence) -> tuple[float | None, dict]:
    """Aggregate-first; missing leaves are not zero and aliases are never summed."""
    leases = ("current_lease_liabilities", "noncurrent_lease_liabilities")
    used, conflicts, missing = [], [], []
    monitored = ("borrowings", "current_borrowings", "noncurrent_borrowings", *DEBT_COMPONENTS,
                 "current_portion_longterm_liabilities", *leases)
    for key in monitored:
        if values.get(key) is not None and values[key] < 0:
            conflicts.append(dict(field=key, reason="NEGATIVE_DEBT_BALANCE"))
        elif evidence.get(key, {}).get("reason") in {
                "CONFLICTING_AGGREGATE_VALUES", "UNIT_MISSING_OR_INCOMPATIBLE", "NUMERIC_TEXT_UNRESOLVED"}:
            conflicts.append(dict(field=key, reason=evidence[key]["reason"]))

    def complete_sum(keys):
        return sum(values[k] for k in keys) if all(values.get(k) is not None for k in keys) else None

    current_keys = DEBT_COMPONENTS[:4]
    noncurrent_keys = DEBT_COMPONENTS[4:]
    for key, children in (("current_borrowings", current_keys), ("noncurrent_borrowings", noncurrent_keys)):
        parent = values.get(key)
        known_minimum = sum(values[k] for k in children if values.get(k) is not None)
        tolerance = (len(children) + 1) * .5
        if parent is not None and known_minimum > parent + tolerance:
            conflicts.append(dict(field=key, reason="DEBT_PARENT_COMPONENT_CONFLICT"))

    # The generic current portion is disclosed for audit, not added to debt.
    # Where no complete debt aggregate is reported, unidentified maturities
    # remain unverified instead of being counted as classified loan/bond debt.
    maturity = values.get("current_portion_longterm_liabilities")
    classified = complete_sum(("current_portion_borrowings", "current_bonds"))
    if maturity is not None and (classified is None or abs(maturity - classified) > 1.5):
        if values.get("borrowings") is None and values.get("current_borrowings") is None:
            conflicts.append(dict(field="current_portion_longterm_liabilities", reason="UNCLASSIFIED_CURRENT_PORTION"))

    components = complete_sum(("current_borrowings", "noncurrent_borrowings"))
    classified_total = complete_sum(DEBT_COMPONENTS)
    total = values.get("borrowings")
    if total is not None:
        known_minimum = sum(values[parent] if values.get(parent) is not None else
            sum(values[k] for k in children if values.get(k) is not None)
            for parent, children in (("current_borrowings", current_keys), ("noncurrent_borrowings", noncurrent_keys)))
        if known_minimum > total + 3.5:
            conflicts.append(dict(field="borrowings", reason="DEBT_PARENT_COMPONENT_CONFLICT"))
        if components is not None and abs(total - components) > 1.5:
            conflicts.append(dict(field="borrowings", reason="CONFLICTING_DEBT_TOTAL"))
        used.append("borrowings")
    else:
        # The reviewed leaf subset is not proof of all financing liabilities.
        # Even six explicit amounts cannot exclude another debt category. Only
        # the complete non-lease total or BOTH complete classification totals
        # establish scope; leaf sums remain an audit-only reconciliation aid.
        total = components
        for parent, children in (("current_borrowings", current_keys), ("noncurrent_borrowings", noncurrent_keys)):
            if values.get(parent) is not None:
                used.append(parent)
            else:
                missing.append(parent)
                missing.extend(k for k in children if values.get(k) is None)
    missing.extend(k for k in leases if values.get(k) is None)
    used.extend(leases)
    result = total + sum(values[k] for k in leases) if total is not None and not missing and not conflicts else None
    reason = ("VERIFIED_NONOVERLAPPING_DEBT_COMPONENTS" if result is not None else
              conflicts[0]["reason"] if conflicts else "INCOMPLETE_DEBT_COMPONENTS")
    return result, dict(status="RESOLVED" if result is not None else "UNKNOWN",
        category=None if result is not None else "INCOMPARABLE" if conflicts else "PENDING_VERIFICATION",
        reason=reason, used_fields=list(dict.fromkeys(used)), missing_fields=list(dict.fromkeys(missing)),
        conflicts=conflicts, aggregation_policy="AGGREGATE_FIRST_NO_ABSENT_ZERO_NO_ALIAS_SUM",
        reported_component_sum=classified_total,
        component_sum_policy="AUDIT_ONLY_NOT_PROOF_OF_EXHAUSTIVE_FINANCING_SCOPE",
        inputs=[evidence[k] for k in monitored if k in evidence])
