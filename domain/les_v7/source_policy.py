"""Frozen R6 source ports; no relaxed financial or valuation assumptions."""
from collections.abc import Mapping

from domain.les_v7.contracts import identity, timestamp

VERSION = "V7_NATIVE_SOURCE_PORTS_R1"
AVAILABLE_AT = "2026-10-14T06:38:00Z"
SIGNED_VERSION = "V7_NATIVE_SOURCE_PORTS_R2"
SCOPED_VERSION = "V7_NATIVE_SOURCE_PORTS_R3_SCOPED_PE"
AUDIT_SOURCE_VERSION = "V7_NATIVE_SOURCE_PORTS_R4_STATEMENT_AUDIT"


def source_policy(*, version=VERSION):
    if version not in {VERSION, SIGNED_VERSION, SCOPED_VERSION, AUDIT_SOURCE_VERSION}:
        raise ValueError("V7_SOURCE_PORT_POLICY_VERSION_UNKNOWN")
    value = dict(schema="les-v7-source-ports/v1", policy_version=VERSION,
        available_at=AVAILABLE_AT,
        identity="OFFICIAL_INSTRUMENT_AND_ISSUER_MASTER_EXACT_MARKET_CUTOFF",
        supported_security_types=["COMMON"],
        unsupported_policy="DATA_INTEGRITY_FAIL_OUT_OF_SCOPE_NOT_FINANCIAL_REJECT",
        pe="DATED_OFFICIAL_OBSERVATION_EXACT_SYMBOL_MARKET_SESSION",
        quote_conflict="UNKNOWN_NO_FAVORABLE_SELECTION",
        pe_window="UNCHANGED_LAST_60_MINIMUM_20_POSITIVE_CURRENT_REQUIRED",
        annual_scope="SAME_ORIGINAL_CURRENT_COMPARATIVE_NO_SHARE_CLASS_SUBSTITUTION",
        availability="ACTUAL_OBSERVATION_NO_HISTORICAL_BACKDATING",
        financial_safety="SOURCE_BOUND_QUALITY_R5_AUDIT_AND_CASH",
        missing="UNKNOWN_NO_PRICE_PE_CASH_EPS_IMPUTATION")
    if version in {SIGNED_VERSION, SCOPED_VERSION, AUDIT_SOURCE_VERSION}:
        from domain.les_v7.signed_audit import AVAILABLE_AT as signed_at, POLICY
        value.update(policy_version=SIGNED_VERSION, available_at=signed_at,
            financial_safety="SOURCE_BOUND_QUALITY_R6_SIGNED_REPORT_ELSE_UNCHANGED_R5_AUDIT_CASH",
            signed_report_precedence=POLICY)
    if version in {SCOPED_VERSION, AUDIT_SOURCE_VERSION}:
        from domain.les_v7.policy_r23 import AVAILABLE_AT as scoped_at
        value.update(policy_version=SCOPED_VERSION, available_at=scoped_at,
            quote_conflict="DEPENDENCY_SCOPED_QUARANTINE_NO_FAVORABLE_SELECTION",
            pe_window="PE_DEPENDENT_COMPONENT_ONLY_LAST60_MIN20_CURRENT_REQUIRED_NOT_INDEPENDENT_EPS_GATE",
            pe_conflict_scope=dict(version="EXACT_SESSION_PE_QUARANTINE_R1",
                blocking_data_integrity_fields=["Close"],
                ratio_only_state="UNKNOWN_COMPONENT_NOT_OHLC_IDENTITY",
                conflicting_value="PE_NONE_NO_SOURCE_PRIORITY_OR_TOLERANCE_RELAXATION",
                source_values="PRESERVE_BOTH_AND_ALL_SELECTION_PROOFS",
                required_component_gate="PE_DEPENDENT_VALUATION_UNKNOWN_IF_CURRENT_OR_HISTORY_UNAVAILABLE",
                independent_eps_method="OWN_VERIFIED_INPUTS_NO_P_E_IMPUTATION_OR_ADDITIONAL_P_E_GATE",
                current_pe_and_history_requirements="UNCHANGED_NO_POSITIVE_OR_COUNT_IMPUTATION"))
    if version == AUDIT_SOURCE_VERSION:
        from domain.les_v7.financial_audit_r7 import AVAILABLE_AT as audit_at, SOURCE_VERSION as audit_policy
        value.update(policy_version=AUDIT_SOURCE_VERSION, available_at=audit_at,
            financial_safety="SOURCE_BOUND_QUALITY_R7_STATEMENT_AUDIT_WITH_VERIFIED_SIGNED_R6_PRECEDENCE_UNCHANGED_CASH",
            statement_audit_policy=audit_policy)
    return value


def require_source_policy(record, *, cutoff):
    expected = source_policy(version=record.get("policy_version") if isinstance(record, Mapping) else None)
    if not isinstance(record, Mapping) or identity(record) != identity(expected):
        raise ValueError("V7_SOURCE_PORT_POLICY_MISMATCH")
    if timestamp(expected["available_at"]) > timestamp(cutoff):
        raise ValueError("V7_SOURCE_PORT_POLICY_AFTER_CUTOFF")
    return expected
