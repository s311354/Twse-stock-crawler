"""Frozen valuation-source dependencies, separate from entry Quality approval.

This changes no valuation economics and supplies no missing financial values.
Quality, financial safety and every other entry gate remain independently required.
"""
from collections.abc import Mapping

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.policy_r13 import INPUT_VERSION as RELAXED_VERSION
from domain.les_v7.policy_r14 import INPUT_VERSION as COMPONENT_VERSION
from domain.les_v7.policy_r15 import INPUT_VERSION as TACTICAL_VERSION
from domain.les_v7.valuation_price import policy as price_policy


VERSION = "V7_PRICE_INPUT_READINESS_R1"
AVAILABLE_AT = "2026-10-14T04:05:10Z"
RECOVERY_VERSION = "V7_PRICE_INPUT_READINESS_R10"


def input_policy(*, version=VERSION):
    from domain.les_v7.fr3_policy import INPUT_VERSION as fr3_version
    if version == fr3_version:
        from domain.les_v7.fr3_policy import eps_policy, AVAILABLE_AT as fr3_release
        return dict(input_policy(), policy_version=fr3_version, available_at=fr3_release,
            valuation_method_policy_id=identity(eps_policy()),
            annual_only=True, no_recovery_or_peer_override=True)
    if version == TACTICAL_VERSION:
        from domain.les_v7.valuation_recovery_r15 import policy, AVAILABLE_AT as release
        return dict(input_policy(), policy_version=TACTICAL_VERSION, available_at=release,
            valuation_method_policy_id=identity(policy()),
            optional_recovery="INDEPENDENT_SOURCE_BOUND_SAME_ORIGINAL_YTD_NO_ANNUALIZATION",
            optional_comparables="FIXED_UNIVERSE_ROLE_BOUND_PEER_QUARANTINE_WITH_REASONS",
            missing_comparables="VERIFIED_ANNUAL_REFERENCE_FALLBACK_NOT_PASS_SUBSTITUTION",
            issuer_conflicts="UNKNOWN_NOT_QUARANTINED",
            peer_only="VERIFIED_TEN_PEERS_SMALL_ONLY_ABSOLUTE_CAP_0_0125")
    if version == COMPONENT_VERSION:
        from domain.les_v7.valuation_recovery_r14 import policy, AVAILABLE_AT as release
        return dict(input_policy(), policy_version=COMPONENT_VERSION, available_at=release,
            valuation_method_policy_id=identity(policy()),
            optional_recovery="INDEPENDENT_SOURCE_BOUND_SAME_ORIGINAL_YTD_NO_ANNUALIZATION",
            optional_comparables="INDEPENDENT_FIXED_UNIVERSE_ISSUER_AND_PEER_COMPONENTS",
            missing_comparables="EXPLICIT_ANNUAL_MULTIPLE_REFERENCE_WITH_COMMON_FIVE_YEAR_GROWTH",
            peer_only="VERIFIED_TEN_PEERS_SMALL_ONLY_METHOD_WEIGHT_MULTIPLIER_0_5")
    if version == RELAXED_VERSION:
        from domain.les_v7.valuation_relaxed_r13 import policy, AVAILABLE_AT as release
        return dict(input_policy(), policy_version=RELAXED_VERSION, available_at=release,
            valuation_method_policy_id=identity(policy()),
            optional_recovery="SOURCE_BOUND_SAME_ORIGINAL_YTD_NO_ANNUALIZATION",
            optional_comparables="FIXED_UNIVERSE_SOURCE_BOUND_ANNUAL_EPS_PE",
            missing_comparables="EXPLICIT_R13_ANNUAL_FALLBACK_NOT_PASS_SUBSTITUTION")
    if version == RECOVERY_VERSION:
        from domain.les_v7.valuation_recovery_r10 import policy, AVAILABLE_AT as release
        return dict(input_policy(), policy_version=RECOVERY_VERSION, available_at=release,
            valuation_method_policy_id=identity(policy()),
            optional_recovery="SOURCE_BOUND_SAME_ORIGINAL_YTD_NO_ANNUALIZATION",
            optional_comparables="FIXED_UNIVERSE_SOURCE_BOUND_ANNUAL_EPS_PE",
            missing_comparables="EXPLICIT_ORIGINAL_R1_FALLBACK_NOT_PASS_SUBSTITUTION")
    if version != VERSION:
        raise ValueError("V7_PRICE_INPUT_POLICY_VERSION_UNKNOWN")
    return dict(schema="les-v7-price-input-readiness/v1", policy_version=VERSION,
        available_at=AVAILABLE_AT, valuation_method_policy_id=identity(price_policy()),
        required_annual_operands=["current_eps", "prior_eps", "current_revenue", "prior_revenue",
                                 "current_operating_income", "prior_operating_income"],
        annual_basis="VERIFIED_LATEST_SAME_ORIGINAL_CURRENT_AND_COMPARATIVE",
        required_market_operands=["current_price", "official_current_pe", "official_recent_pe_median"],
        quality_dependency="RESULT_ID_BINDING_NOT_ANNUAL_INPUT_READINESS",
        entry_quality_required=True, missing_policy="UNKNOWN_NO_SUBSTITUTION",
        revision_policy="CUTOFF_VISIBLE_LATEST_NO_OLDER_FALLBACK")


def require_input_policy(record, *, cutoff):
    fixed = input_policy(version=record.get("policy_version") if isinstance(record, Mapping) else VERSION)
    if not isinstance(record, Mapping) or identity(record) != identity(fixed):
        raise ValueError("V7_PRICE_INPUT_POLICY_MISMATCH")
    if timestamp(fixed["available_at"]) > timestamp(cutoff):
        raise ValueError("V7_PRICE_INPUT_POLICY_NOT_AVAILABLE_AT_CUTOFF")
    return fixed
