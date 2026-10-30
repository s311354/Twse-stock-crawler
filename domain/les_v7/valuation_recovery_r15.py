"""R15 role-bound peers with the unchanged R14 economic scenario arithmetic.

Only the retained, independently validated peer set changes. There is no new
multiple, growth assumption, annualization or probability in this policy.
"""
from collections.abc import Mapping

from domain.les_v7.contracts import identity
from domain.les_v7.policy_r15 import AVAILABLE_AT, VALUATION_VERSION
from domain.les_v7 import valuation_recovery_r14 as previous


POLICY_VERSION = VALUATION_VERSION
COMPONENT_SCHEMA = "les-v7-comparable-components/v2"
PEER_VALIDATION_POLICY = "R15_FIXED_UNIVERSE_ROLE_BOUND_PEER_QUARANTINE"
FULL_METHOD = "RECOVERY_FULL_COMPARABLES_R15"
PEER_METHOD = "RECOVERY_PEER_ONLY_R15"
ANNUAL_METHOD = "RECOVERY_ANNUAL_MULTIPLE_FALLBACK_R15"


def policy():
    fixed = previous.policy()
    fixed.update(schema="les-v7-price-return-valuation/v5", policy_version=POLICY_VERSION,
        available_at=AVAILABLE_AT,
        unknown_component="OWN_ISSUER_HISTORY_CONFLICT_BLOCKS_THIRD_PARTY_PEERS_QUARANTINED",
        economic_scenario_policy="R14_UNCHANGED_ARITHMETIC_NOT_A_PRICE_FORECAST")
    fixed["comparables"].update(component_schema=COMPONENT_SCHEMA,
        peer_validation_policy=PEER_VALIDATION_POLICY,
        peer_quarantine=dict(role="THIRD_PARTY_ONLY", minimum_verified_peers=10,
            selection="FIXED_UNIVERSE_ALL_MEMBERS_SAME_VALIDATION_NO_OUTCOME_SELECTION",
            exclusions="RETAIN_SYMBOL_REASON_SOURCE_IDS_AND_CONTENT_HASHES",
            own_or_issuer_history_conflict="UNKNOWN_NO_QUARANTINE",
            insufficient="EXPLICIT_VERIFIED_ANNUAL_FALLBACK_ONLY"))
    return fixed


def _components_for_math(comparables):
    if not isinstance(comparables, Mapping) or comparables.get("schema") != COMPONENT_SCHEMA:
        return dict(schema=previous.COMPONENT_SCHEMA, status="UNKNOWN",
                    reasons=["R15_COMPARABLE_COMPONENT_SCHEMA_UNVERIFIED"])
    # No source certification occurs here. The receipt producer and independent
    # validator must rerun the public role-aware selector from retained proofs.
    value = previous._plain(comparables)
    value["schema"] = previous.COMPONENT_SCHEMA
    if value.get("peer_validation_policy") != PEER_VALIDATION_POLICY:
        value.update(status="UNKNOWN", reasons=["R15_PEER_VALIDATION_POLICY_UNVERIFIED"])
    return value


def evaluate_recovery_price_valuation(*, legacy_result, comparables, recovery=None):
    """Recompute all scenarios, then bind the unchanged arithmetic to R15."""
    result = previous.evaluate_recovery_price_valuation(legacy_result=legacy_result,
        comparables=_components_for_math(comparables), recovery=recovery)
    fixed = policy()
    result.update(schema=fixed["schema"], policy_version=POLICY_VERSION,
                  policy=fixed, policy_id=identity(fixed), comparables=previous._plain(comparables))
    result["selected_method"] = {previous.FULL_METHOD: FULL_METHOD,
        previous.PEER_METHOD: PEER_METHOD, previous.ANNUAL_METHOD: ANNUAL_METHOD}[result["selected_method"]]
    result["reasons"] = [reason.replace("R14_", "R15_") for reason in result["reasons"]]
    if result.get("multiple_selection"):
        result["multiple_selection"]["component_result_id"] = identity(comparables)
    return result


def evaluate_annual_price_valuation(**inputs):
    missing = dict(schema=COMPONENT_SCHEMA, peer_validation_policy=PEER_VALIDATION_POLICY,
        status="INSUFFICIENT", reasons=["R15_COMPARABLE_CONTEXT_MISSING"],
        issuer_history=dict(status="INSUFFICIENT", count=0, minimum=24, median=None, sources=[]),
        peers=dict(status="INSUFFICIENT", count=0, minimum=10, median=None, sources=[]),
        peer_quarantine=[])
    return evaluate_recovery_price_valuation(legacy_result=dict(inputs=inputs), comparables=missing)


evaluate_relaxed_price_valuation = evaluate_recovery_price_valuation
