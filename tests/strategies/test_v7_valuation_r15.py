"""SYNTHETIC ONLY: R15 metadata cannot alter R14 scenario economics."""
from copy import deepcopy

import pytest

from application.evidence_context import ReadOnlyEvidence
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.valuation_recovery_r14 import evaluate_recovery_price_valuation as old_evaluate
from domain.les_v7.valuation_recovery_r14 import policy as old_policy
from domain.les_v7.valuation_recovery_r15 import (ANNUAL_METHOD, COMPONENT_SCHEMA, PEER_METHOD,
    PEER_VALIDATION_POLICY, evaluate_annual_price_valuation, evaluate_recovery_price_valuation, policy)
from tests.strategies.test_v7_valuation_r13 import inputs
from tests.strategies.test_v7_valuation_r14 import components as old_components, recent


def components(**options):
    return dict(old_components(**options), schema=COMPONENT_SCHEMA,
                peer_validation_policy=PEER_VALIDATION_POLICY, peer_quarantine=[])


def evaluate(*, c=None, r=None, args=None):
    return evaluate_recovery_price_valuation(legacy_result=dict(inputs=args or inputs()),
        comparables=components() if c is None else c, recovery=r)


@pytest.mark.parametrize("issuer_count,peer_count", [(24, 10), (3, 10), (3, 9)])
def test_r15_reuses_identical_r14_numerics_not_new_growth_or_multiple(issuer_count, peer_count):
    c = components(issuer_count=issuer_count, peer_count=peer_count)
    new = evaluate(c=c, r=recent())
    old = old_evaluate(legacy_result=dict(inputs=inputs()),
        comparables=old_components(issuer_count=issuer_count, peer_count=peer_count), recovery=recent())
    for key in ("anchor", "observed_growth", "scenarios", "status", "gate_checks", "bear_permanent_loss"):
        assert new[key] == old[key]
    assert new["multiple_selection"]["component_result_id"] == identity(c)
    assert new["policy_version"].endswith("R15")


def test_immutable_policies_readonly_inputs_and_quarantine_audit_are_preserved():
    prior = identity(old_policy()); c = components(issuer_count=3)
    c["peer_quarantine"] = [dict(symbol="SYNTHETIC_BAD_PEER", block_id="a" * 64, reason="CONFLICT")]
    before = deepcopy(c)
    result = evaluate(c=ReadOnlyEvidence(c), r=ReadOnlyEvidence(recent()))
    assert result["selected_method"] == PEER_METHOD and result["comparables"] == before
    canonical(result)
    assert c == before and identity(old_policy()) == prior
    fixed = policy(); fixed["comparables"]["peer_quarantine"]["minimum_verified_peers"] = 0
    assert policy()["comparables"]["peer_quarantine"]["minimum_verified_peers"] == 10


@pytest.mark.parametrize("damage", ["schema", "peer_policy", "own_unknown", "recovery_unknown", "annual"])
def test_sources_and_own_conflicts_never_become_permissive_fallback(damage):
    c, r, args = components(issuer_count=3), recent(), inputs()
    if damage == "schema": c["schema"] = "les-v7-comparable-components/v1"
    elif damage == "peer_policy": c["peer_validation_policy"] = "UNVERIFIED"
    elif damage == "own_unknown": c.update(status="UNKNOWN", reasons=["OWN_SOURCE_CONFLICT"])
    elif damage == "recovery_unknown": r.update(status="UNKNOWN", reasons=["OWN_YTD_CONFLICT"])
    else: args["current_eps"] = None
    result = evaluate(c=c, r=r, args=args)
    assert result["status"] == "UNKNOWN" and result["scenarios"] is None


def test_under_ten_verified_peers_only_uses_existing_explicit_annual_fallback():
    result = evaluate(c=components(issuer_count=3, peer_count=9), r=recent())
    assert result["selected_method"] == ANNUAL_METHOD
    assert result["comparables"]["peers"]["count"] == 9
    assert result["observed_growth"]["recent_recovery"] == .25
    assert evaluate_annual_price_valuation(**inputs())["comparables"]["peers"]["count"] == 0
