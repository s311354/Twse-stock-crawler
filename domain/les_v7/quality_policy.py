"""Frozen, independently registered Quality R2 policy and financial proof.

No trained LES artifact, wall clock, network, legal filing deadline or old PASS
flag participates. The period policy deliberately does not claim to verify
late filing: it evaluates the newest source-visible period at the cutoff, and
refuses to substitute an older original when that newest period is unverified.
Financial safety is not the bars-bound market SAFETY gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from types import MappingProxyType
from zoneinfo import ZoneInfo

import pandas as pd

from domain.fundamental_derivation import FiledFact, FiledFinancials, _AUDIT_CONCEPTS, _derive_audit_evidence
from domain.financial_taxonomy import (CASH_AUDIT_FLOWS, CASH_RECONCILING_COMPONENTS,
    TAXONOMY_PROFILES, TAXONOMY_MAPPING_VERSION, cash_flow_checks, _cash_source_binding)
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.financial_sources import (FINANCIAL_MAPPING, EXTRA_FIELDS, EXTRA_INSTANT,
                                           PAYMENT_CONCEPTS, PAYMENT_POLICY, AUDIT_POLICY, audit_proof)
from domain.strategies.low_entry_v6_quality_evidence import validate_field, validated_proof_value


R1_PROFILE = "QG_V7_BOUNDED_GAPS_R1"
R2_PROFILE = "QG_V7_EVIDENCE_R2"
R3_PROFILE = "QG_V7_REPORTED_PERIOD_R3"
R4_PROFILE = "QG_V7_REPORTED_PERIOD_R4"
R5_PROFILE = "QG_V7_REPORTED_PERIOD_R5"
R6_PROFILE = "QG_V7_REPORTED_PERIOD_R6"
R7_PROFILE = "QG_V7_REPORTED_PERIOD_R7"
SOURCE_MAPPING_ID = "MOPS_TYPED_FIELD_EVIDENCE_R2"
SOURCE_MAPPING = MappingProxyType(dict(
    mapping_id=SOURCE_MAPPING_ID, version="QG_SOURCE_MAP_R2",
    amount_unit="TWD_THOUSAND", eps_unit="TWD_PER_SHARE",
    scopes=("consolidated", "individual"),
    sources=("MOPS_IXBRL", "TWSE_OPENAPI", "TPEX_OPENAPI"),
    original_proof="quality-field-evidence/v1",
    audit_proof="financial-audit-proof/v1",
    audit_selection_policy=AUDIT_POLICY,
    additive_ledger_policy="EXACT_MAPPED_FLOW_QNAMES_NO_UNRELATED_NOTE_VALUES",
    debt_policy="AGGREGATE_FIRST_NO_ABSENT_ZERO_NO_ALIAS_SUM",
    financial_mapping_version=FINANCIAL_MAPPING,
    extra_fields=tuple((field, tuple(concepts)) for field, concepts in sorted(EXTRA_FIELDS.items())),
    extra_instant_fields=tuple(EXTRA_INSTANT),
    payment_concepts=tuple((field, tuple(concepts)) for field, concepts in sorted(PAYMENT_CONCEPTS.items())),
    capex_policy=PAYMENT_POLICY,
    receivable_basis="REPORTED_ORDINARY_NET_TRADE_EXCLUDING_RELATED_PARTIES",
    cash_policy="SOURCE_BOUND_IAS7_AND_DISPOSAL_RECONCILIATION",
))

# The explicit R2 manifest stays byte-identical. R3 adopts the existing v6
# reported-period kernel uniformly, not a fallback for individual stocks.
from domain.strategies.low_entry_v6_quality import (
    QUALITY_POLICY_VERSION, QUALITY_SCORING_POLICY, QUALITY_BUDGETS,
    QUALITY_COVERAGE_MINIMA, QUALITY_DC_WEIGHTS, QUALITY_REQUIRED_REPORTED_FIELDS,
    BALANCE_FIELDS,
    QUALITY_R4_POLICY_VERSION, QUALITY_R4_SCORING_POLICY, QUALITY_R4_COVERAGE_MINIMA,
)
R3_AUDIT_CLASSIFICATION = "SOURCE_BOUND_UNREVIEWED_COMPONENT_EN_R3"


def _freeze_manifest(value):
    if isinstance(value, Mapping):
        return tuple((key, _freeze_manifest(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_manifest(item) for item in value)
    return value


REPORTED_SOURCE_MAPPING = MappingProxyType(dict(SOURCE_MAPPING,
    mapping_id="MOPS_REPORTED_PERIOD_EVIDENCE_R3", version="QG_SOURCE_MAP_R3",
    native_quality_policy=QUALITY_POLICY_VERSION,
    native_scoring_manifest=_freeze_manifest(QUALITY_SCORING_POLICY),
    budgets=_freeze_manifest(QUALITY_BUDGETS), coverage_minima=_freeze_manifest(QUALITY_COVERAGE_MINIMA),
    dc_weights=_freeze_manifest(QUALITY_DC_WEIGHTS),
    required_reported_fields=QUALITY_REQUIRED_REPORTED_FIELDS,
    required_balance_fields=BALANCE_FIELDS,
    scoring_basis="REPORTED_SAME_PERIOD",
    ttm_policy="DIAGNOSTIC_ONLY_NOT_CORE_SCORE_COVERAGE_OR_DC",
    debt_policy="TOTAL_LIABILITIES_CORE_COMPLETE_INTEREST_BEARING_DEBT_DIAGNOSTIC_ONLY",
    audit_classification_policy=R3_AUDIT_CLASSIFICATION,
))

# R3's manifest above remains immutable. R4 uses the same original-source and
# financial-safety contracts, but a separately frozen missing-data policy.
R4_REPORTED_SOURCE_MAPPING = MappingProxyType(dict(REPORTED_SOURCE_MAPPING,
    mapping_id="MOPS_REPORTED_PERIOD_EVIDENCE_R4", version="QG_SOURCE_MAP_R4",
    native_quality_policy=QUALITY_R4_POLICY_VERSION,
    native_scoring_manifest=_freeze_manifest(QUALITY_R4_SCORING_POLICY),
    coverage_minima=_freeze_manifest(QUALITY_R4_COVERAGE_MINIMA),
))

from domain.les_v7.financial_sources_r5 import (
    AUDIT_POLICY_R5, AUDIT_CLASSIFICATION_R5, SOURCE_REPAIR_POLICY_R5, R5_AVAILABLE_AT,
    audit_proof_r5,
)
R5_REPORTED_SOURCE_MAPPING = MappingProxyType(dict(R4_REPORTED_SOURCE_MAPPING,
    mapping_id="MOPS_REPORTED_PERIOD_EVIDENCE_R5", version="QG_SOURCE_MAP_R5",
    audit_proof="financial-audit-proof/v2", audit_selection_policy=AUDIT_POLICY_R5,
    audit_classification_policy=AUDIT_CLASSIFICATION_R5,
    source_repair_policy=SOURCE_REPAIR_POLICY_R5,
    cash_policy="SAME_ORIGINAL_REPARSE_SOURCE_BOUND_IAS7_AND_DISPOSAL_NO_PLUG",
))
from domain.les_v7.signed_audit import POLICY as SIGNED_AUDIT_POLICY, AVAILABLE_AT as R6_AVAILABLE_AT
R6_REPORTED_SOURCE_MAPPING = MappingProxyType(dict(R5_REPORTED_SOURCE_MAPPING,
    mapping_id="MOPS_REPORTED_PERIOD_EVIDENCE_R6", version="QG_SOURCE_MAP_R6",
    signed_report_policy=SIGNED_AUDIT_POLICY,
    source_precedence="VERIFIED_SIGNED_PDF_FULL_BODY_AND_FINANCIAL_ANCHORS_ELSE_UNCHANGED_R5",
    signed_report_availability="MAX_ACTUAL_FETCH_AND_TIEOUT_NO_HISTORICAL_BACKDATE",
))
from domain.les_v7.financial_audit_r7 import (
    SOURCE_VERSION as R7_AUDIT_POLICY, AVAILABLE_AT as R7_AVAILABLE_AT,
    source_policy as audit_source_policy_r7,
)
R7_REPORTED_SOURCE_MAPPING = MappingProxyType(dict(R6_REPORTED_SOURCE_MAPPING,
    mapping_id="MOPS_REPORTED_PERIOD_EVIDENCE_R7", version="QG_SOURCE_MAP_R7",
    audit_proof="financial-audit-proof/v4", audit_selection_policy=R7_AUDIT_POLICY,
    audit_classification_policy=R7_AUDIT_POLICY,
    audit_source_policy=_freeze_manifest(audit_source_policy_r7()),
    source_precedence="VERIFIED_SIGNED_R6_ELSE_SOURCE_BOUND_STATEMENT_DESCRIPTIONS_R7",
))


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    profile_id: str = R2_PROFILE
    policy_id: str = "QG_V7_EVIDENCE_R2_20261012"
    policy_version: str = "QG_EVIDENCE_POLICY_R2"
    mapping_version: str = "QG_MAP_R2"
    source_mapping_id: str = SOURCE_MAPPING_ID
    filing_schedule_version: str = "LATEST_VERIFIED_PUBLISHED_PERIOD_R1"
    latest_required_period_policy: str = "LATEST_VERIFIED_PUBLISHED_PERIOD_NO_OLDER_FALLBACK"
    tax_rule: str = "VERIFIED_TTM_EXPENSE_OVER_PRETAX"
    capex_rule: str = "VERIFIED_REPORTED_ACQUISITION_PAYMENT_MAGNITUDE_R2"
    freshness_full_age_days: int = 180
    freshness_max_age_days: int = 275
    freshness_policy: str = "PERIOD_END_180_FULL_THEN_LINEAR_TO_275_ZERO"
    overdue_filing_verification: str = "NOT_VERIFIED_NO_OFFICIAL_SCHEDULE_EVIDENCE"

    def record(self):
        value = asdict(self)
        value["latest_required_period"] = "LATEST_VERIFIED_PUBLISHED_PERIOD"
        mapping = (R7_REPORTED_SOURCE_MAPPING if self.profile_id == R7_PROFILE else
                   R6_REPORTED_SOURCE_MAPPING if self.profile_id == R6_PROFILE else
                   R5_REPORTED_SOURCE_MAPPING if self.profile_id == R5_PROFILE else
                   R4_REPORTED_SOURCE_MAPPING if self.profile_id == R4_PROFILE else
                   REPORTED_SOURCE_MAPPING if self.profile_id == R3_PROFILE else SOURCE_MAPPING)
        value["source_mapping_sha256"] = identity(dict(mapping))
        value["policy_sha256"] = identity(value)
        return value


BUILTIN_QUALITY_POLICY = QualityPolicy()


@dataclass(frozen=True, slots=True)
class ReportedQualityPolicy(QualityPolicy):
    profile_id: str = R3_PROFILE
    policy_id: str = "QG_V7_REPORTED_PERIOD_R3_20261012"
    policy_version: str = "QG_REPORTED_POLICY_R3"
    mapping_version: str = "QG_MAP_R3"
    source_mapping_id: str = "MOPS_REPORTED_PERIOD_EVIDENCE_R3"
    tax_rule: str = "VERIFIED_TTM_TAX_DIAGNOSTIC_ONLY"
    capex_rule: str = "VERIFIED_REPORTED_PPE_ACQUISITION_PAYMENT"
    freshness_policy: str = "NATIVE_REPORTED_PERIOD_180_FULL_275_REDUCED_THEN_BLOCKED"


BUILTIN_REPORTED_QUALITY_POLICY = ReportedQualityPolicy()


@dataclass(frozen=True, slots=True)
class LimitedGapsQualityPolicy(ReportedQualityPolicy):
    profile_id: str = R4_PROFILE
    policy_id: str = "QG_V7_REPORTED_PERIOD_R4_20261013"
    policy_version: str = "QG_REPORTED_POLICY_R4"
    mapping_version: str = "QG_MAP_R4"
    source_mapping_id: str = "MOPS_REPORTED_PERIOD_EVIDENCE_R4"


BUILTIN_LIMITED_GAPS_QUALITY_POLICY = LimitedGapsQualityPolicy()


@dataclass(frozen=True, slots=True)
class SourceRepairQualityPolicy(LimitedGapsQualityPolicy):
    profile_id: str = R5_PROFILE
    policy_id: str = "QG_V7_REPORTED_PERIOD_R5_20261014"
    policy_version: str = "QG_REPORTED_POLICY_R5"
    mapping_version: str = "QG_MAP_R5"
    source_mapping_id: str = "MOPS_REPORTED_PERIOD_EVIDENCE_R5"
    available_at: str = R5_AVAILABLE_AT


BUILTIN_SOURCE_REPAIR_QUALITY_POLICY = SourceRepairQualityPolicy()


@dataclass(frozen=True, slots=True)
class SignedReportQualityPolicy(SourceRepairQualityPolicy):
    profile_id: str = R6_PROFILE
    policy_id: str = "QG_V7_REPORTED_PERIOD_R6_20261014"
    policy_version: str = "QG_REPORTED_POLICY_R6"
    mapping_version: str = "QG_MAP_R6"
    source_mapping_id: str = "MOPS_REPORTED_PERIOD_EVIDENCE_R6"
    available_at: str = R6_AVAILABLE_AT


BUILTIN_SIGNED_REPORT_QUALITY_POLICY = SignedReportQualityPolicy()


@dataclass(frozen=True, slots=True)
class StatementAuditQualityPolicy(SignedReportQualityPolicy):
    profile_id: str = R7_PROFILE
    policy_id: str = "QG_V7_REPORTED_PERIOD_R7_20261028"
    policy_version: str = "QG_REPORTED_POLICY_R7"
    mapping_version: str = "QG_MAP_R7"
    source_mapping_id: str = "MOPS_REPORTED_PERIOD_EVIDENCE_R7"
    available_at: str = R7_AVAILABLE_AT


BUILTIN_STATEMENT_AUDIT_QUALITY_POLICY = StatementAuditQualityPolicy()
_REGISTERED = MappingProxyType({p.policy_id: p for p in
    (BUILTIN_QUALITY_POLICY, BUILTIN_REPORTED_QUALITY_POLICY, BUILTIN_LIMITED_GAPS_QUALITY_POLICY,
     BUILTIN_SOURCE_REPAIR_QUALITY_POLICY, BUILTIN_SIGNED_REPORT_QUALITY_POLICY,
     BUILTIN_STATEMENT_AUDIT_QUALITY_POLICY)})


def is_r2_policy(policy):
    return isinstance(policy, Mapping) and policy.get("profile_id") == R2_PROFILE


def is_r3_policy(policy):
    return isinstance(policy, Mapping) and policy.get("profile_id") == R3_PROFILE


def is_r4_policy(policy):
    return isinstance(policy, Mapping) and policy.get("profile_id") == R4_PROFILE


def is_r5_policy(policy):
    return isinstance(policy, Mapping) and policy.get("profile_id") == R5_PROFILE


def is_r6_policy(policy):
    return isinstance(policy, Mapping) and policy.get("profile_id") == R6_PROFILE


def is_r7_policy(policy):
    return isinstance(policy, Mapping) and policy.get("profile_id") == R7_PROFILE


def is_reported_policy(policy):
    return is_r3_policy(policy) or is_r4_policy(policy) or is_r5_policy(policy) or is_r6_policy(policy) or is_r7_policy(policy)


def is_source_evidence_policy(policy):
    return is_r2_policy(policy) or is_reported_policy(policy)


def _published_period(records, *, symbol, market, cutoff):
    visible, errors = {}, []
    for row in records:
        try:
            at = timestamp(row["available_at"])
            if at > cutoff:
                continue
            if row.get("symbol") != symbol or row.get("market") != market:
                raise ValueError("SECURITY_IDENTITY_AMBIGUOUS")
            end = pd.Timestamp(row["period_end"])
            period = pd.Period(end, freq="Q")
            if (end.date() != period.end_time.date()
                    or end.date() > at.astimezone(ZoneInfo("Asia/Taipei")).date()
                    or end.date() > cutoff.astimezone(ZoneInfo("Asia/Taipei")).date()):
                raise ValueError("INVALID_REPORTING_PERIOD")
            visible.setdefault(period, []).append(row)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            errors.append(str(error))
    if errors:
        return None, "VISIBLE_SOURCE_METADATA_UNVERIFIED", sorted(set(errors))
    if not visible:
        return None, "NO_VERIFIED_PUBLISHED_PERIOD", ["NO_POINT_IN_TIME_FINANCIALS"]
    newest = max(visible)
    rows = visible[newest]
    verified = [row for row in rows if (
        row.get("source") in SOURCE_MAPPING["sources"] and row.get("scope") in SOURCE_MAPPING["scopes"]
        and row.get("scope_basis") != "LEGACY_DEFAULT_NOT_SOURCE_REPORTED"
        and row.get("currency") == "TWD" and row.get("unit") == "thousand"
        and row.get("revision_id") and isinstance(row.get("content_sha256"), str)
        and len(row["content_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in row["content_sha256"].lower())
        and bool(row.get("field_evidence")))]
    if not verified:
        return None, "LATEST_PERIOD_ORIGINAL_UNVERIFIED", ["LATEST_PERIOD_ORIGINAL_UNVERIFIED:" + str(newest)]
    return str(newest), "VERIFIED_PUBLISHED_ORIGINAL_PERIOD", []


def validate_quality_policy(policy, cutoff, *, market=None, records=None, symbol=None):
    """One registered config preflight, independent of model training readiness.

    R1 dictionaries are returned unchanged for exact legacy bundle equality.
    R2 dynamic period annotations are recomputed from fixed cutoff inputs; they
    are not part of the frozen config hash and never use today's quarter.
    """
    at = timestamp(cutoff)
    if market is not None and market not in {"TWSE", "TPEX"}:
        raise ValueError("QUALITY_POLICY_MARKET_UNSUPPORTED")
    if isinstance(policy, QualityPolicy):
        policy = policy.record()
    elif policy is None:
        policy = BUILTIN_LIMITED_GAPS_QUALITY_POLICY.record()
    elif not isinstance(policy, Mapping):
        raise ValueError("QUALITY_POLICY_OBJECT_REQUIRED")
    if not is_source_evidence_policy(policy):
        return deepcopy(dict(policy))  # Explicit legacy policy, never upgraded.
    known = _REGISTERED.get(policy.get("policy_id"))
    if known is None:
        raise ValueError("UNREGISTERED_QUALITY_POLICY")
    expected = known.record()
    if (is_r5_policy(expected) or is_r6_policy(expected) or is_r7_policy(expected)) and timestamp(expected["available_at"]) > at:
        raise ValueError("QUALITY_POLICY_NOT_AVAILABLE_AT_CUTOFF")
    for key, value in expected.items():
        if key == "latest_required_period":
            continue  # Runtime source selection, not a configurable deadline.
        if policy.get(key) != value:
            raise ValueError("QUALITY_POLICY_HASH_OR_MAPPING_MISMATCH:" + key)
    extras = set(policy) - set(expected) - {"latest_required_period", "period_selection_status", "period_selection_reasons"}
    if extras:
        raise ValueError("UNKNOWN_QUALITY_POLICY_FIELDS:" + ",".join(sorted(extras)))
    result = dict(expected)
    if records is not None:
        if not symbol or market is None:
            raise ValueError("QUALITY_POLICY_SECURITY_BINDING_REQUIRED")
        required, status, reasons = _published_period(records, symbol=symbol, market=market, cutoff=at)
        declared = policy.get("latest_required_period")
        if declared not in (None, "LATEST_VERIFIED_PUBLISHED_PERIOD") and declared != required:
            raise ValueError("QUALITY_POLICY_REQUIRED_PERIOD_MISMATCH")
        result.update(latest_required_period=required, period_selection_status=status, period_selection_reasons=reasons)
    else:
        required = policy.get("latest_required_period")
        if (required not in (None, "LATEST_VERIFIED_PUBLISHED_PERIOD")
                and pd.Period(required, freq="Q").end_time.date() > at.astimezone(ZoneInfo("Asia/Taipei")).date()):
            raise ValueError("QUALITY_POLICY_FUTURE_REQUIRED_PERIOD")
        result.update(latest_required_period=required,
            period_selection_status=policy.get("period_selection_status", "REQUIRES_SOURCE_SELECTION"),
            period_selection_reasons=list(policy.get("period_selection_reasons", ())))
    return result


def default_quality_policy(records=None, *, symbol=None, market=None, cutoff=None):
    """Stable registered template for preflight; optional cutoff-bound resolve."""
    if cutoff is None:
        if records is not None or symbol is not None or market is not None:
            raise ValueError("QUALITY_POLICY_CUTOFF_REQUIRED_FOR_SOURCE_SELECTION")
        return BUILTIN_LIMITED_GAPS_QUALITY_POLICY.record()
    return validate_quality_policy(BUILTIN_LIMITED_GAPS_QUALITY_POLICY.record(), cutoff,
                                   market=market, records=records, symbol=symbol)


def r3_quality_policy(records=None, *, symbol=None, market=None, cutoff=None):
    """Explicit frozen R3; old bundles and reports are never upgraded in place."""
    if cutoff is None:
        if records is not None or symbol is not None or market is not None:
            raise ValueError("QUALITY_POLICY_CUTOFF_REQUIRED_FOR_SOURCE_SELECTION")
        return BUILTIN_REPORTED_QUALITY_POLICY.record()
    return validate_quality_policy(BUILTIN_REPORTED_QUALITY_POLICY.record(), cutoff,
                                   market=market, records=records, symbol=symbol)


def r5_quality_policy(records=None, *, symbol=None, market=None, cutoff=None):
    """Explicit source-repair R5; default R4 and frozen earlier hashes stay intact."""
    if cutoff is None:
        if records is not None or symbol is not None or market is not None:
            raise ValueError("QUALITY_POLICY_CUTOFF_REQUIRED_FOR_SOURCE_SELECTION")
        return BUILTIN_SOURCE_REPAIR_QUALITY_POLICY.record()
    return validate_quality_policy(BUILTIN_SOURCE_REPAIR_QUALITY_POLICY.record(), cutoff,
                                   market=market, records=records, symbol=symbol)


def r6_quality_policy(records=None, *, symbol=None, market=None, cutoff=None):
    if cutoff is None:
        if records is not None or symbol is not None or market is not None:
            raise ValueError("QUALITY_POLICY_CUTOFF_REQUIRED_FOR_SOURCE_SELECTION")
        return BUILTIN_SIGNED_REPORT_QUALITY_POLICY.record()
    return validate_quality_policy(BUILTIN_SIGNED_REPORT_QUALITY_POLICY.record(), cutoff,
                                   market=market, records=records, symbol=symbol)


def r7_quality_policy(records=None, *, symbol=None, market=None, cutoff=None):
    """Explicit new source parser; no upgrade of old bundle/Quality proofs."""
    if cutoff is None:
        if records is not None or symbol is not None or market is not None:
            raise ValueError("QUALITY_POLICY_CUTOFF_REQUIRED_FOR_SOURCE_SELECTION")
        return BUILTIN_STATEMENT_AUDIT_QUALITY_POLICY.record()
    return validate_quality_policy(BUILTIN_STATEMENT_AUDIT_QUALITY_POLICY.record(), cutoff,
                                   market=market, records=records, symbol=symbol)


def r2_quality_policy(records=None, *, symbol=None, market=None, cutoff=None):
    """Explicit legacy R2 template; never silently migrated to R3."""
    if cutoff is None:
        if records is not None or symbol is not None or market is not None:
            raise ValueError("QUALITY_POLICY_CUTOFF_REQUIRED_FOR_SOURCE_SELECTION")
        return BUILTIN_QUALITY_POLICY.record()
    return validate_quality_policy(BUILTIN_QUALITY_POLICY.record(), cutoff,
                                   market=market, records=records, symbol=symbol)


def quality_freshness(period, cutoff, policy):
    age = (timestamp(cutoff).astimezone(ZoneInfo("Asia/Taipei")).date() - pd.Period(period, freq="Q").end_time.date()).days
    full, maximum = policy["freshness_full_age_days"], policy["freshness_max_age_days"]
    value = 100. if age <= full else max(0., 100. * (maximum - age) / (maximum - full))
    return age, value


def _audit_proof(row, *, period, cutoff, r5=False, r6=False, r7=False):
    if r6 and row.get("audit_evidence_r6") is not None:
        proof = row["audit_evidence_r6"]
        contexts = proof.get("contexts", ())
        facts = tuple(FiledFact(**dict(node, dimensions=tuple(tuple(d) for d in node.get("dimensions", ())))) for node in contexts)
        origin = row.get("quarter_flow_source", row)
        if any(proof.get(k) != origin.get(k) for k in ("symbol", "market", "scope", "currency", "unit", "source", "source_url", "period_end", "available_at", "content_sha256", "revision_id")):
            return None, "SIGNED_REPORT_AUDIT_ORIGIN_BINDING_INVALID"
        filing = FiledFinancials(period.year, period.quarter, proof["available_at"], 1, facts,
            content_sha256=proof["content_sha256"], statement_scope=proof["scope"],
            source_url=proof["source_url"], audit_status=proof.get("audit_status"), period_end=proof["period_end"])
        from domain.les_v7.signed_audit import audit_proof_r6
        rebuilt = audit_proof_r6(filing, row, proof["signed_source"], symbol=proof["symbol"], market=proof["market"], cutoff=cutoff)
        if identity(rebuilt) != identity(proof):
            return None, "SIGNED_REPORT_AUDIT_REBUILD_MISMATCH"
        return rebuilt["audit"], None
    if r7:
        proof = row.get("audit_evidence_r7")
        if not isinstance(proof, Mapping):
            return None, "R7_AUDIT_ORIGINAL_EVIDENCE_MISSING"
        origin = row.get("quarter_flow_source", row)
        fields = ("symbol", "market", "scope", "currency", "unit", "source", "source_url",
                  "period_end", "available_at", "content_sha256", "revision_id")
        if (proof.get("schema") != "financial-audit-proof/v4" or proof.get("policy_version") != R7_AUDIT_POLICY
                or any(proof.get(k) != origin.get(k) for k in fields)
                or proof.get("period_end") != period.end_time.date().isoformat()
                or timestamp(proof["available_at"]) > cutoff):
            return None, "R7_AUDIT_ORIGINAL_BINDING_INVALID"
        contexts = proof.get("original_contexts")
        if not isinstance(contexts, (list, tuple)) or any(
                not isinstance(node, Mapping) or node.get("concept") not in _AUDIT_CONCEPTS for node in contexts):
            return None, "R7_AUDIT_ORIGINAL_CONTEXTS_INVALID"
        facts = tuple(FiledFact(**dict(node, dimensions=tuple(tuple(d) for d in node.get("dimensions", ()))))
                      for node in contexts)
        source = FiledFinancials(period.year, period.quarter, proof["available_at"], proof["revision_number"], facts,
            content_sha256=proof["content_sha256"], statement_scope=proof["scope"], source_url=proof["source_url"],
            audit_status=proof.get("audit_status"), period_end=proof["period_end"])
        from domain.les_v7.financial_audit_r7 import audit_source_r7
        rebuilt = audit_source_r7(source, symbol=proof["symbol"], market=proof["market"], original_verified=True)
        if identity(rebuilt) != identity(proof):
            return None, "R7_AUDIT_ORIGINAL_REBUILD_MISMATCH"
        if rebuilt["status"] != "VERIFIED_REPORTED":
            return None, "AUDIT_SOURCE_PENDING:" + rebuilt["reason"]
        return rebuilt["audit"], None
    proof = row.get("audit_evidence_r5") if r5 else row.get("audit_evidence")
    if not isinstance(proof, Mapping):
        return None, "AUDIT_ORIGINAL_EVIDENCE_MISSING"
    origin = row.get("quarter_flow_source", row)
    fields = ("symbol", "market", "scope", "currency", "unit", "source", "source_url", "period_end", "available_at", "content_sha256", "revision_id")
    schema, version = ("financial-audit-proof/v2", AUDIT_POLICY_R5) if r5 else ("financial-audit-proof/v1", AUDIT_POLICY)
    if (proof.get("schema") != schema or proof.get("policy_version") != version
            or any(proof.get(k) != origin.get(k) for k in fields)
            or proof.get("period_end") != period.end_time.date().isoformat()
            or timestamp(proof["available_at"]) > cutoff or proof.get("result_id") != identity(proof.get("audit"))):
        return None, "AUDIT_ORIGINAL_EVIDENCE_BINDING_INVALID"
    if proof.get("status") != "VERIFIED_REPORTED":
        return None, "AUDIT_SOURCE_PENDING:" + str(proof.get("reason", "UNVERIFIED_CURRENT_AUDIT"))
    contexts = proof.get("contexts", ())
    if not contexts:
        return None, "AUDIT_ORIGINAL_CONTEXTS_MISSING"
    if any(node.get("concept") not in _AUDIT_CONCEPTS or node.get("dimensions") for node in contexts):
        return None, "AUDIT_ORIGINAL_CONTEXT_PERIOD_UNVERIFIED"
    facts = tuple(FiledFact(concept=node["concept"], numeric_value=node.get("numeric_value"),
        text_value=node.get("text_value"), start_date=node.get("start_date"), end_date=node.get("end_date"),
        instant=node.get("instant"), unit=node.get("unit"), context_id=node.get("context_id"),
        dimensions=tuple(tuple(item) for item in node.get("dimensions", ()))) for node in contexts)
    source = FiledFinancials(period.year, period.quarter, proof["available_at"], 1, facts,
        content_sha256=proof["content_sha256"], statement_scope=proof["scope"],
        source_url=proof["source_url"], audit_status=proof.get("audit_status"), period_end=proof["period_end"])
    build = audit_proof_r5 if r5 else audit_proof
    rebuilt = build(source, symbol=proof["symbol"], market=proof["market"], original_verified=True)
    derived = rebuilt["audit"]
    if (rebuilt["status"] != "VERIFIED_REPORTED"
            or identity(rebuilt["period_binding"]) != identity(proof.get("period_binding"))
            or identity(rebuilt["contexts"]) != identity(contexts) or identity(derived) != identity(proof["audit"])):
        return None, "AUDIT_ORIGINAL_CLASSIFICATION_MISMATCH"
    return derived, None


def _cash_anchor_value(proof, origin, field, *, start, end):
    """Validate a native CF anchor without relabelling its instant as a flow.

    The SCF beginning balance is reported at the CF closing instant with a
    distinct concept, not a previous balance-sheet row. Its flow-window binding
    is explicit in the mapper's CASH_FLOW_ANCHOR proof.
    """
    if (not isinstance(proof, Mapping) or proof.get("status") != "RESOLVED"
            or proof.get("reason") != "REPORTED_EXACT_CONTEXT"
            or proof.get("basis") != "CASH_FLOW_ANCHOR"
            or proof.get("cash_flow_period_start") != start
            or proof.get("end_date") != end or proof.get("reported_instant") != end
            or proof.get("taxonomy_schema") not in TAXONOMY_PROFILES
            or proof.get("taxonomy_mapping_version") != TAXONOMY_MAPPING_VERSION
            or proof.get("unit") != "thousand"
            or any(proof.get(k) != origin.get(k) for k in
                   ("content_sha256", "revision_id", "scope", "currency", "available_at", "symbol", "source", "source_url"))):
        return None
    allowed = TAXONOMY_PROFILES[proof["taxonomy_schema"]][field]
    contexts = [node for node in proof.get("contexts", ()) if not node.get("dimensions")
                and node.get("instant") == end and node.get("concept") in allowed]
    if not contexts or any(node.get("unit") != "iso4217:TWD"
            or isinstance(node.get("numeric_value"), bool)
            or not isinstance(node.get("numeric_value"), (int, float))
            or not isfinite(node["numeric_value"]) for node in contexts):
        return None
    values = {float(node["numeric_value"]) / 1000. for node in contexts}
    return next(iter(values)) if len(values) == 1 else None


def financial_safety_source(row):
    """Thin current-original proof inputs for independently checking a receipt.

    Original bytes stay in the existing evidence store. This view carries only
    header identity, necessary CF/BS values/contexts and the bound audit proof;
    it never retains the issuer's whole historical financial graph.
    """
    headers = ("symbol", "market", "source", "source_url", "period_end", "scope", "currency",
               "unit", "available_at", "content_sha256", "revision_id", "flow_basis", "scope_basis")
    fields = tuple(dict.fromkeys((*BALANCE_FIELDS, "cash", "cfo", *CASH_AUDIT_FLOWS,
                                 "cash_beginning", "cash_ending", *CASH_RECONCILING_COMPONENTS)))
    view = {key: row.get(key) for key in (*headers, *fields)}
    view["field_evidence"] = {field: row.get("field_evidence", {}).get(field, {}) for field in fields}
    view["audit_evidence"] = row.get("audit_evidence")
    for key in ("audit_evidence_r5", "source_repair_r5", "audit_evidence_r6", "audit_evidence_r7"):
        if key in row:
            view[key] = row[key]
    origin = row.get("quarter_flow_source")
    if origin is not None:
        view["quarter_flow_source"] = {key: origin.get(key) for key in headers}
    # Only this bounded current-period packet is detached for mutable domain
    # callers. Immutable work-owned proof nodes may share safely; no issuer
    # history, DataFrame metadata or full native-result graph is copied here.
    return deepcopy(view)


def native_financial_safety(latest, checks=None, *, symbol, market, cutoff, policy=None):
    """Derive financial-only proof from original contexts, never old gate flags.

    A missing upstream audit/context remains UNKNOWN. This result cannot be
    supplied as the separately bars-bound GateReceipt named ``SAFETY``.
    """
    at = timestamp(cutoff)
    reported = is_reported_policy(policy)
    r7 = is_r7_policy(policy)
    r6 = is_r6_policy(policy) or r7
    r5 = is_r5_policy(policy) or r6
    classification_policy = R7_AUDIT_POLICY if r7 else AUDIT_CLASSIFICATION_R5 if r5 else R3_AUDIT_CLASSIFICATION
    result = dict(gate_id="FINANCIAL_SAFETY", gate_version="NATIVE_FINANCIAL_SAFETY_R7" if r7 else "NATIVE_FINANCIAL_SAFETY_R6" if r6 else "NATIVE_FINANCIAL_SAFETY_R5" if r5 else "NATIVE_FINANCIAL_SAFETY_R3" if reported else "NATIVE_FINANCIAL_SAFETY_R2",
        state="UNKNOWN", reasons=[], proof_record_ids=[], input_ids=[], source_available_at=None,
        checked_at=at.isoformat(), input_hash=None)
    ledger, reasons, failures = {}, [], []
    try:
        if r5:
            # The public financial-safety port must not interpret a newly
            # published source policy at a pre-publication cutoff on its own.
            validate_quality_policy(policy, cutoff, market=market)
        if (latest.get("symbol") != symbol or latest.get("market") != market
                or latest.get("scope") not in SOURCE_MAPPING["scopes"]
                or latest.get("source") not in SOURCE_MAPPING["sources"]):
            raise ValueError("FINANCIAL_SAFETY_IDENTITY_UNVERIFIED")
        period = pd.Period(latest["period_end"], freq="Q")
        if latest["period_end"] != period.end_time.date().isoformat() or timestamp(latest["available_at"]) > at:
            raise ValueError("FINANCIAL_SAFETY_PERIOD_OR_TIME_UNVERIFIED")
        balance_fields = (tuple(BALANCE_FIELDS) if reported else
            ("total_assets", "current_assets", "non_current_assets", "total_liabilities", "current_liabilities", "total_equity", "cash", "interest_bearing_debt"))
        values = {}
        for field in balance_fields:
            check = validate_field(latest, field, period=period, cutoff=pd.Timestamp(at), instant=True)
            if (check["evidence_level"] != "CANONICAL_ORIGINAL" or check["value"] is None
                    or not all(check["checks"].values())):
                reasons.append("FINANCIAL_SAFETY_FIELD_UNVERIFIED:" + field)
            else:
                prior = (checks or {}).get(str(period) + ":" + field)
                if prior is not None and prior != check:
                    reasons.append("FINANCIAL_SAFETY_LEDGER_CONFLICT:" + field)
                values[field] = check["value"]
            ledger[field] = check
        if len(values) == len(balance_fields):
            a, l, e = (values[field] for field in ("total_assets", "total_liabilities", "total_equity"))
            if a <= 0 or e <= 0 or any(value < 0 for value in values.values()):
                failures.append("INVALID_BALANCE_OR_NONPOSITIVE_EQUITY")
            elif (abs(a - l - e) / a >= .01 or abs(a - values["current_assets"] - values["non_current_assets"]) / a >= .01
                    or values["current_liabilities"] > l
                    or not reported and (values["interest_bearing_debt"] > l or values["cash"] > a)):
                failures.append("FINANCIAL_BALANCE_IDENTITY_FAIL")
        debt = latest.get("field_evidence", {}).get("interest_bearing_debt", {})
        if not reported and (debt.get("reason") != "VERIFIED_NONOVERLAPPING_DEBT_COMPONENTS" or not debt.get("used_fields")):
            reasons.append("FINANCIAL_DEBT_COMPLETENESS_UNVERIFIED")
        start = (period.start_time.date().isoformat() if latest.get("flow_basis") == "QUARTER" else f"{period.year}-01-01")
        cf_values = {}
        origin = latest.get("quarter_flow_source", latest)
        for field in ("cfo", *CASH_AUDIT_FLOWS, "cash_beginning", "cash_ending", "cash", *CASH_RECONCILING_COMPONENTS):
            proof = latest.get("field_evidence", {}).get(field, {})
            end = period.end_time.date().isoformat()  # Expected selected endpoint, not the proof's own claim.
            value = (_cash_anchor_value(proof, origin, field, start=start, end=end)
                     if field in {"cash_beginning", "cash_ending"} else
                     validated_proof_value(proof, origin, field, start=start, end=end,
                         instant=proof.get("basis") == "INSTANT", field_evidence=latest.get("field_evidence")))
            claimed = latest.get(field)
            if value is not None and claimed is not None and isclose(value, float(claimed), rel_tol=1e-12, abs_tol=.5):
                cf_values[field] = value
        # Native arithmetic alone is not source validation for the direct CF/BS
        # tie. Validate its operands even when no IAS 7/disposal bridge exists.
        components = tuple(field for field in CASH_RECONCILING_COMPONENTS
            if latest.get(field) is not None or latest.get("field_evidence", {}).get(field, {}).get("contexts"))
        cash_binding_errors, cash_source_receipts = _cash_source_binding(cf_values,
            latest.get("field_evidence"), components=components,
            start_date=start, end_date=period.end_time.date().isoformat())
        if cash_binding_errors:
            reasons.append("FINANCIAL_CASH_SOURCE_BINDING_UNVERIFIED")
        cash_checks = cash_flow_checks(cf_values, start_date=start, end_date=period.end_time.date().isoformat(),
                                      evidence=latest.get("field_evidence"))
        if any(check["status"] == "FAIL" for check in cash_checks):
            failures.append("FINANCIAL_CASH_RECONCILIATION_FAIL")
        if any(check["status"] != "PASS" for check in cash_checks):
            reasons.append("FINANCIAL_CASH_RECONCILIATION_UNVERIFIED")
        audit, audit_reason = _audit_proof(latest, period=period, cutoff=at, r5=r5, r6=r6, r7=r7)
        signed_used = r6 and latest.get("audit_evidence_r6") is not None
        if r5 and audit_reason is None:
            proof = latest["audit_evidence_r6" if signed_used else "audit_evidence_r7" if r7 else "audit_evidence_r5"]
            result.update(audit_classification=audit, audit_classification_policy=classification_policy,
                          original_audit_result_id=proof["result_id"], source_repair=latest.get("source_repair_r5"))
            if signed_used:
                result.update(audit_classification_policy=SIGNED_AUDIT_POLICY,
                    signed_report_source_id=proof["signed_source"]["source_id"],
                    retained_translation_conflicts=proof["retained_translation_conflicts"])
        elif reported and audit_reason is None:
            # Use the same classifier with one versioned, explicit language
            # extension; the stored v3 original classification remains intact.
            from domain.les_v7.financial_sources import classify_audit_contexts
            proof = latest["audit_evidence"]
            facts = tuple(FiledFact(**dict(node, dimensions=tuple(tuple(d) for d in node.get("dimensions", ()))))
                          for node in proof["contexts"])
            filing = FiledFinancials(period.year, period.quarter, proof["available_at"], 1, facts,
                statement_scope=proof["scope"], audit_status=proof.get("audit_status"), period_end=proof["period_end"])
            audit = classify_audit_contexts(filing, facts, proof["period_binding"],
                                            allow_unreviewed_components=True)
            result.update(audit_classification=audit, audit_classification_policy=R3_AUDIT_CLASSIFICATION,
                          original_audit_result_id=proof["result_id"])
        if audit_reason:
            reasons.append(audit_reason)
        elif (audit["audit_opinion"] in {"ADVERSE", "DISCLAIMER", "QUALIFIED_MATERIAL_MISSTATEMENT"}
              or audit["going_concern"] is True or audit["material_accounting_issue"] is True):
            failures.append("CONFIRMED_FINANCIAL_AUDIT_HARD_RISK")
        elif (audit["audit_opinion"] not in ({"UNMODIFIED", "REVIEWED_UNMODIFIED", "QUALIFIED_SCOPE_LIMITATION"}
                                           if reported else {"UNMODIFIED", "REVIEWED_UNMODIFIED"})
              or audit["audit_review_required"] is True or audit["going_concern"] is not False
              or audit["material_accounting_issue"] is not False):
            reasons.append("FINANCIAL_AUDIT_REVIEW_REQUIRED")
        dates = [timestamp(node["available_at"]) for node in ledger.values() if node.get("available_at")]
        if dates:
            result["source_available_at"] = max(dates).isoformat()
        audit_key = "audit_evidence_r6" if signed_used else "audit_evidence_r7" if r7 else "audit_evidence_r5" if r5 else "audit_evidence"
        if signed_used:
            dates.append(timestamp(latest[audit_key]["evidence_available_at"]))
            result["source_available_at"] = max(dates).isoformat()
            result["input_ids"] = [latest[audit_key]["signed_source"]["pdf_sha256"]]
        result["proof_record_ids"] = sorted({identity(node) for node in ledger.values()} | {identity(cash_checks), identity(cash_source_receipts)}
                                            | ({identity(latest[audit_key])} if latest.get(audit_key) else set()))
        result["input_ids"] = sorted(set(result["input_ids"]) | {node["content_sha256"] for node in ledger.values() if node.get("content_sha256")})
        result["input_hash"] = identity(dict(symbol=symbol, market=market, cutoff=at.isoformat(),
                                             period=str(period), field_checks=ledger, cash_checks=cash_checks,
                                             cash_source_receipts=cash_source_receipts,
                                             audit_result_id=latest.get(audit_key, {}).get("result_id")))
        if reported:
            result["input_hash"] = identity(dict(base_input_hash=result["input_hash"],
                audit_classification_policy=classification_policy,
                audit_classification=result.get("audit_classification")))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        reasons.append(str(error))
    result["reasons"] = sorted(set(failures + reasons))
    result["state"] = "FAIL" if failures else "UNKNOWN" if reasons else "PASS"
    return result
