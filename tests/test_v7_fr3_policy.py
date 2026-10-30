"""SYNTHETIC ONLY: approved FR3 policy arithmetic, not predictive efficacy."""
from copy import deepcopy

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.fr3_policy import (AVAILABLE_AT, INPUT_VERSION, components,
    eps_policy, evaluate_eps_r2, entry_policy, execution_policy, valuation_admission)
from domain.les_v7.valuation_eps_receipt import produce_price_valuation_receipt
from domain.les_v7.valuation_input_policy import input_policy, require_input_policy
from domain.les_v7.valuation_price import policy as old_policy
from domain.les_v7.valuation_relaxed_r13 import evaluate_annual_price_valuation
from tests.test_v7_eps_price_receipts import item, source


def operands(**changes):
    return dict(dict(current_eps=12., prior_eps=10., current_revenue=120., prior_revenue=100.,
        current_margin=.3, prior_margin=.3, current_price=30., current_pe=6.,
        recent_pe_median=15., eps_basis="DILUTED"), **changes)


def test_r2_reuses_exact_r13_annual_math_not_another_growth_forecast():
    data = operands()
    old = evaluate_annual_price_valuation(**data)
    new = evaluate_eps_r2(**data)
    for key in ("inputs", "scenarios", "anchor", "gate_checks", "bear_permanent_loss", "status"):
        assert new[key] == old[key]
    assert eps_policy()["required_return"] == .08
    assert eps_policy()["maximum_bear_discounted_shortfall"] == .45
    assert old_policy()["required_return"] == .10  # Original policy unchanged.
    assert eps_policy()["basic_eps_haircut"] == .9
    assert new["policy_id"] == identity(eps_policy())


def test_input_policy_has_own_binding_and_real_publication_no_backdating():
    record = input_policy(version=INPUT_VERSION)
    assert record["valuation_method_policy_id"] == identity(eps_policy())
    assert require_input_policy(record, cutoff="2026-10-24T00:00:00Z") == record
    with pytest.raises(ValueError, match="NOT_AVAILABLE"):
        require_input_policy(record, cutoff="2026-09-21T05:30:00Z")


def run(row=None):
    args = item(row)
    args.update(cutoff="2026-10-24T00:00:00Z", policy=eps_policy(),
                input_policy=input_policy(version=INPUT_VERSION))
    return produce_price_valuation_receipt(**args)


def test_native_verified_annual_source_is_reused_without_share_or_dividend_imputation():
    annual = source()
    before = deepcopy(annual)
    receipt = run(annual)
    assert receipt["state"] == "PASS"
    assert receipt["valuation_diagnostics"]["policy_id"] == identity(eps_policy())
    assert receipt["valuation_diagnostics"]["anchor"]["scenario_anchor"] == 12.
    assert annual == before
    assert annual.get("dividends") is None and annual.get("diluted_shares") is None


@pytest.mark.parametrize("damage", ["missing", "unit", "revision", "url", "future"])
def test_unknown_evidence_remains_unknown_not_a_relaxed_warning(damage):
    row = source()
    if damage == "missing": row.pop("earnings_source")
    else:
        proof = row["earnings_source"]["field_evidence"]["diluted_eps"]
        if damage == "unit": proof["contexts"][0]["unit"] = "iso4217:TWD"
        elif damage == "revision": proof["revision_number"] += 1
        elif damage == "url": proof["source_url"] += "OTHER"
        else: proof["available_at"] = "2099-01-01T00:00:00Z"
    receipt = run(row)
    assert receipt["state"] == "UNKNOWN"
    assert valuation_admission(receipt)["warning"] is False


@pytest.mark.parametrize("eps", [0., -1.])
def test_known_nonpositive_eps_is_fail_not_missing_or_warning(eps):
    receipt = run(source(current_eps=eps))
    assert receipt["state"] == "FAIL" and not valuation_admission(receipt)["warning"]


@pytest.mark.parametrize("loss,warning,state", [(.35, False, "PASS"), (.4, True, "ALLOWED_WITH_WARNING"),
                                              (.45, True, "ALLOWED_WITH_WARNING"), (.450001, False, "FAIL")])
def test_approved_bear_bands_are_explicit_and_do_not_rewrite_native_fail(loss, warning, state):
    checks = dict(BASE_PRICE_CAGR_GTE_REQUIRED_RETURN=True, BASE_FAIR_PRICE_GT_CURRENT_PRICE=True,
                  BEAR_LOSS_LTE_MAXIMUM=loss <= .35, BEAR_LOSS_LTE_HARD_MAXIMUM=loss <= .45)
    receipt = dict(state="PASS" if all(checks.values()) else "FAIL", valuation_diagnostics=dict(
        policy_id=identity(eps_policy()), scenarios={"SYNTHETIC": {}}, gate_checks=checks))
    before = deepcopy(receipt)
    result = valuation_admission(receipt)
    assert result["warning"] is warning and result["state"] == state
    assert result["penalty"] == (5. if warning else 0.)
    assert receipt == before


def test_hash_only_or_empty_checks_cannot_certify_a_valuation_pass():
    forged = dict(state="PASS", valuation_diagnostics=dict(policy_id=identity(eps_policy()),
                                                           scenarios={"forged": {}}, gate_checks={}))
    assert valuation_admission(forged)["state"] == "UNKNOWN"
    assert not valuation_admission(forged)["warning"]
    assert valuation_admission(None)["state"] == "UNKNOWN"


def test_component_records_are_detached_and_limit_has_no_score_operand():
    first = components(); first["entry"]["policy"]["small_minimum"] = 0.
    assert components()["entry"]["policy"]["small_minimum"] == 65.
    assert entry_policy()["position_limits_affect_score"] is False
    assert execution_policy()["total_symbol_cap"] == .01
    assert entry_policy()["maximum_decision"] == "BUY_SMALL"
