"""SYNTHETIC arithmetic only, not historical-source or return validation."""
from copy import deepcopy
from math import prod

import pytest

from domain.les_v7.contracts import canonical, identity
from domain.les_v7.valuation_price import evaluate_price_valuation
from domain.les_v7.valuation_recovery_r14 import (
    ANNUAL_METHOD, COMPONENT_SCHEMA, FULL_METHOD, PEER_METHOD,
    evaluate_annual_price_valuation, evaluate_recovery_price_valuation, policy,
)
from domain.les_v7.valuation_relaxed_r13 import evaluate_relaxed_price_valuation, policy as prior_policy
from tests.strategies.test_v7_valuation_r13 import inputs


def components(*, issuer_count=24, peer_count=10, issuer_pe=25., peer_pe=35., industry="半導體業"):
    issuer, peers = issuer_count >= 24, peer_count >= 10
    return dict(schema=COMPONENT_SCHEMA,
        status="VERIFIED" if issuer and peers else "PARTIAL" if peers else "INSUFFICIENT",
        reasons=[], industry=industry, issuer_month_count=issuer_count, peer_count=peer_count,
        issuer_history=dict(status="VERIFIED" if issuer else "INSUFFICIENT", count=issuer_count,
            minimum=24, median=issuer_pe if issuer_count else None, sources=[]),
        peers=dict(status="VERIFIED" if peers else "INSUFFICIENT", count=peer_count,
            minimum=10, median=peer_pe if peer_count else None, sources=[]))


def recent(status="VERIFIED"):
    return dict(status=status, reasons=[], values=dict(current_revenue=130., prior_revenue=100.,
        current_operating_income=25., prior_operating_income=20., current_cfo=5.))


def evaluate(*, args=None, peers=None, recovery=None):
    return evaluate_recovery_price_valuation(legacy_result=evaluate_price_valuation(**(args or inputs())),
        comparables=components() if peers is None else peers, recovery=recovery)


def test_r13_identity_and_economics_are_not_modified():
    fixed = prior_policy(); before = identity(fixed)
    old = evaluate_relaxed_price_valuation(legacy_result=evaluate_price_valuation(**inputs()),
        comparables=dict(status="INSUFFICIENT", reasons=[]), recovery=recent())
    evaluate(peers=components(issuer_count=3), recovery=recent())
    assert identity(prior_policy()) == before
    assert old == evaluate_relaxed_price_valuation(legacy_result=evaluate_price_valuation(**inputs()),
        comparables=dict(status="INSUFFICIENT", reasons=[]), recovery=recent())
    assert old["scenarios"]["BEAR"]["years"] == 3


def test_policy_is_detached_and_all_routes_have_explicit_five_year_assumptions():
    fixed = policy(); before = identity(fixed)
    fixed["comparables"]["peer_only"]["method_position_multiplier"] = 999
    assert identity(policy()) == before
    assert policy()["performance_validated"] is False
    for c in (components(), components(issuer_count=3), components(issuer_count=3, peer_count=9)):
        result = evaluate(peers=c)
        assert {s["years"] for s in result["scenarios"].values()} == {5}
        for scenario in result["scenarios"].values():
            assert scenario["fair_price"] == pytest.approx(scenario["future_price"] / 1.08 ** 5)


def test_full_components_preserve_r13_recovery_growth_and_multiples():
    result = evaluate(recovery=recent())
    old = evaluate_relaxed_price_valuation(legacy_result=evaluate_price_valuation(**inputs()),
        comparables=dict(status="VERIFIED", industry="半導體業", issuer_pe_median=25.,
            peer_pe_median=35., issuer_month_count=24, peer_count=10, reasons=[]), recovery=recent())
    assert result["selected_method"] == FULL_METHOD
    for key in ("anchor", "observed_growth", "scenarios", "status", "gate_checks", "bear_permanent_loss"):
        assert result[key] == old[key]
    assert result["method_small_only"] is False and result["method_position_multiplier"] == 1.


def test_three_issuer_months_do_not_discard_verified_peers_or_recovery():
    args = inputs(current_eps=10.5, current_revenue=1050., recent_pe_median=8.)
    result = evaluate(args=args, peers=components(issuer_count=3, peer_pe=18.), recovery=recent())
    assert result["selected_method"] == PEER_METHOD
    assert result["scenarios"]["BASE"]["exit_pe"] == 18.
    assert result["observed_growth"]["recent_recovery"] == pytest.approx(.25)
    assert result["observed_growth"]["blended"] == pytest.approx(.75 * .05 + .25 * .25)
    assert result["method_small_only"] is True and result["method_position_multiplier"] == .5
    assert result["multiple_selection"]["issuer_history_required"] is False
    assert "OFFICIAL" not in result["multiple_selection"]["basis"]


def test_insufficient_peers_use_official_reference_but_still_receive_verified_recovery():
    args = inputs(current_eps=10.5, current_revenue=1050., recent_pe_median=12.)
    c = components(issuer_count=3, peer_count=9)
    result = evaluate(args=args, peers=c, recovery=recent())
    without = evaluate(args=args, peers=c)
    assert result["selected_method"] == ANNUAL_METHOD
    assert result["scenarios"]["BASE"]["exit_pe"] == 12.
    assert result["observed_growth"]["recent_recovery"] == pytest.approx(.25)
    assert result["scenarios"]["BASE"]["future_price"] > without["scenarios"]["BASE"]["future_price"]
    assert result["method_small_only"] is False and result["method_position_multiplier"] == 1.
    assert "NOT_ANNUAL_COMPARABLE" in result["multiple_selection"]["basis"]


@pytest.mark.parametrize("issuer_count,peer_count", [(36, 10), (3, 10), (3, 9)])
def test_verified_official_industry_without_policy_profile_uses_existing_fallback_not_peer_privilege(issuer_count, peer_count):
    c = components(issuer_count=issuer_count, peer_count=peer_count, industry="資訊服務業", peer_pe=60.)
    c.update(industry_source_status="VERIFIED_OFFICIAL_CLASSIFICATION", industry_profile_status="NOT_DEFINED")
    args = inputs(current_eps=10.5, current_revenue=1050., recent_pe_median=50.)
    result = evaluate(args=args, peers=c, recovery=recent())
    assert result["status"] != "UNKNOWN"
    assert result["selected_method"] == ANNUAL_METHOD
    assert result["comparables"] == c
    assert result["scenarios"]["BASE"]["exit_pe"] == 20.
    assert result["observed_growth"]["blended"] == pytest.approx(.75 * .05 + .25 * .25)
    assert result["observed_growth"]["bounded_first_year"] == .10
    assert result["method_small_only"] is False and result["method_position_multiplier"] == 1.
    assert result["multiple_selection"]["industry"] == "資訊服務業"
    assert result["multiple_selection"]["fallback_reason"] == "INDUSTRY_PROFILE_NOT_DEFINED"


@pytest.mark.parametrize("changes", [dict(industry_source_status="UNKNOWN"),
    dict(industry_profile_status="DEFINED"), dict(industry="")])
def test_unverified_or_contradictory_official_industry_does_not_borrow_default_profile(changes):
    c = components(industry="資訊服務業")
    c.update(industry_source_status="VERIFIED_OFFICIAL_CLASSIFICATION", industry_profile_status="NOT_DEFINED")
    c.update(changes)
    assert evaluate(peers=c)["status"] == "UNKNOWN"


@pytest.mark.parametrize("status", ["NO_RECOVERY", "NOT_AVAILABLE"])
def test_missing_or_proven_unfavorable_recent_conditions_get_no_recovery_credit(status):
    a, b = evaluate(recovery=recent(status)), evaluate()
    assert a["scenarios"] == b["scenarios"]
    assert a["observed_growth"]["recent_recovery"] is None


@pytest.mark.parametrize("where", ["comparables", "recovery"])
@pytest.mark.parametrize("status", ["UNKNOWN", "CONFLICT", "UNRECOGNIZED"])
def test_conflict_or_unverified_present_component_never_picks_favorable_fallback(where, status):
    args = {"peers" if where == "comparables" else "recovery": dict(status=status, reasons=["SYNTHETIC_CONFLICT"])}
    result = evaluate(**args)
    assert result["status"] == "UNKNOWN" and result["scenarios"] is None
    canonical(result)


@pytest.mark.parametrize("field", ["current_eps", "prior_eps", "current_revenue", "prior_revenue", "current_margin", "prior_margin", "current_price", "current_pe", "recent_pe_median"])
@pytest.mark.parametrize("value,status", [(None, "UNKNOWN"), (True, "UNKNOWN"), (float("nan"), "UNKNOWN"), (0., "FAIL"), (-1., "FAIL")])
def test_required_positive_annual_and_market_sources_cannot_be_replaced(field, value, status):
    result = evaluate(args=inputs(**{field: value}), peers=components(issuer_count=3), recovery=recent())
    assert result["status"] == status and result["scenarios"] is None
    canonical(result)


@pytest.mark.parametrize("damage", ["issuer_status", "peer_count", "top_count", "nan_pe", "unknown_industry"])
def test_inconsistent_component_summary_is_unknown(damage):
    c = components(issuer_count=3)
    if damage == "issuer_status": c["issuer_history"]["status"] = "VERIFIED"
    elif damage == "peer_count": c["peers"]["count"] = 9
    elif damage == "top_count": c["peer_count"] = 100
    elif damage == "nan_pe": c["peers"]["median"] = float("nan")
    else: c["industry"] = "UNVERIFIED_SECTOR"
    assert evaluate(peers=c)["status"] == "UNKNOWN"


def test_all_routes_rebuild_from_annual_inputs_and_do_not_mutate_callers():
    old, c, r = evaluate_price_valuation(**inputs(eps_basis="BASIC")), components(issuer_count=3), recent()
    before = deepcopy((old, c, r)); old["anchor"]["scenario_anchor"] = 99999
    result = evaluate_recovery_price_valuation(legacy_result=old, comparables=c, recovery=r)
    assert result["anchor"]["scenario_anchor"] == 10.8
    assert c == before[1] and r == before[2] and old["anchor"]["scenario_anchor"] == 99999
    for scenario in result["scenarios"].values():
        assert scenario["terminal_eps"] == pytest.approx(10.8 * scenario["eps_factor"] * prod(1 + rate for rate in scenario["annual_growth"]))
    assert evaluate_annual_price_valuation(**inputs())["selected_method"] == ANNUAL_METHOD


def test_6277_saved_operands_are_not_forced_to_pass_by_recovery_decoupling():
    # Readable retained-operand counterfactual only; no authenticated stock fixture.
    args = inputs(current_eps=3.04, prior_eps=4.08, current_revenue=4918944., prior_revenue=4993174.,
        current_margin=.11703182634321513, prior_margin=.13388457922756147,
        current_price=83., current_pe=20.05, recent_pe_median=19.505)
    r = dict(status="VERIFIED", reasons=[], values=dict(current_revenue=2635317., prior_revenue=2336384.,
        current_operating_income=408478., prior_operating_income=240131., current_cfo=219723.))
    result = evaluate(args=args, peers=components(issuer_count=3, peer_count=9, industry="電腦及週邊設備業"), recovery=r)
    assert result["observed_growth"]["blended"] == pytest.approx(-.15918975521952878)
    assert result["scenarios"]["BASE"]["fair_price"] == pytest.approx(36.216971347669965)
    assert result["discounted_bear_value_shortfall"] == pytest.approx(.7441574419160081)
    assert result["status"] == "FAIL" and result["gate_checks"]["BEAR_LOSS_LTE_HARD_MAXIMUM"] is False
