"""SYNTHETIC ONLY: R13 source/cutoff bindings, no real source certification."""
from copy import deepcopy

import pandas as pd
import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.policy_r13 import AVAILABLE_AT, INPUT_VERSION
from domain.les_v7.valuation_eps_receipt import RELAXED_PRODUCER, produce_price_valuation_receipt
from domain.les_v7.valuation_input_policy import input_policy, require_input_policy
from domain.les_v7.valuation_relaxed_r13 import ANNUAL_METHOD, RECOVERY_METHOD, policy
from tests.test_v7_eps_price_receipts import item, source
from tests.test_v7_valuation_r10_receipts import context, SYMBOLS, CUTOFF, recent


def data(row=None):
    result = item(row)
    result.update(policy=policy(), cutoff=CUTOFF, input_policy=input_policy(version=INPUT_VERSION))
    return result


def evaluate(value):
    return produce_price_valuation_receipt(**value)


def test_new_producer_uses_latest_proven_annual_eps_without_changing_native_quality():
    value = data(); value["quality"] = dict(core_inputs_pass=False, status="UNKNOWN")
    original = deepcopy(value["packet"])
    result = evaluate(value)
    assert result["gate_version"] == RELAXED_PRODUCER
    diagnostic = result["valuation_diagnostics"]
    assert diagnostic["selected_method"] == ANNUAL_METHOD
    assert diagnostic["anchor"]["scenario_anchor"] == 12.
    assert diagnostic["policy_id"] == identity(policy())
    assert diagnostic["input_policy_id"] == identity(input_policy(version=INPUT_VERSION))
    assert diagnostic["entry_quality_required"] is True
    assert value["quality"]["status"] == "UNKNOWN" and value["packet"] == original


def test_complete_source_bound_comparables_use_r13_recovery_and_actual_strategy_quote():
    ctx = context(); row = ctx["issuer"]["annual_filings"][-1]
    value = data(row); value.update(symbol=SYMBOLS[0], features=dict(close=120.))
    value["bars"]["Date"] = "2026-10-15"
    value["packet"]["r10_valuation_context"] = ctx
    result = evaluate(value)
    assert result["valuation_diagnostics"]["selected_method"] == RECOVERY_METHOD
    assert result["valuation_diagnostics"]["comparables"]["status"] == "VERIFIED"
    assert pd.Timestamp(result["source_available_at"]) >= pd.Timestamp(ctx["issuer"]["market_evidence"]["source_available_at"])
    value["features"]["close"] = 121.
    assert evaluate(value)["state"] == "UNKNOWN"


@pytest.mark.parametrize("damage", ["missing_original", "future_original", "companion_hash", "eps_unit", "money_hash", "annual_scope", "missing_current_pe"])
def test_r13_never_substitutes_missing_or_conflicting_annual_or_quote_evidence(damage):
    value = data(); row = value["packet"]["valuation_filings"][0]
    if damage == "missing_original": value["packet"]["valuation_filings"] = []
    elif damage == "future_original": row["available_at"] = "2027-01-01T00:00:00Z"
    elif damage == "companion_hash": row["earnings_source"]["content_sha256"] = "0" * 64
    elif damage == "eps_unit": row["earnings_source"]["field_evidence"]["diluted_eps"]["contexts"][0]["unit"] = "iso4217:TWD"
    elif damage == "money_hash": row["comparative_ytd_field_evidence"]["revenue"]["content_sha256"] = "0" * 64
    elif damage == "annual_scope": row["scope"] = "individual"
    else: value["bars"]["PE"] = None
    result = evaluate(value)
    assert result["state"] == "UNKNOWN" and result["reason_codes"]


@pytest.mark.parametrize("row", [source(current_eps=-1.), source(prior_eps=0.),
    source(money_overrides={("operating_income", 2025): -1.})])
def test_known_bad_annual_premise_stays_fail_even_without_quality(row):
    value = data(row); value["quality"] = dict(status="UNKNOWN", core_inputs_pass=False)
    assert evaluate(value)["state"] == "FAIL"


def test_future_annual_revision_is_excluded_not_backdated():
    value = data(); before = evaluate(value)
    future = deepcopy(value["packet"]["valuation_filings"][0])
    future.update(available_at="2027-01-01T00:00:00Z", revision_id="SYNTHETIC_FUTURE")
    value["packet"]["valuation_filings"].append(future)
    assert evaluate(value) == before


def test_conflicting_comparable_proof_does_not_fall_back_to_good_annual_result():
    ctx = context(); row = ctx["issuer"]["annual_filings"][-1]
    value = data(row); value.update(symbol=SYMBOLS[0], features=dict(close=120.))
    value["bars"]["Date"] = "2026-10-15"
    ctx["peers"][0]["annual_filings"][-1]["earnings_source"]["content_sha256"] = "0" * 64
    value["packet"]["r10_valuation_context"] = ctx
    result = evaluate(value)
    assert result["state"] == "UNKNOWN" and result["valuation_diagnostics"]["scenarios"] is None


def test_verified_no_recovery_source_time_is_retained_without_recovery_credit():
    value = data(); value["symbol"] = SYMBOLS[0]
    annual = context()["issuer"]["annual_filings"][-1]
    value["packet"]["valuation_filings"] = [annual]
    row = recent(cfo=-1.)
    value["packet"]["filings"] = [row]
    result = evaluate(value)
    recovery = result["valuation_diagnostics"]["recovery"]
    assert recovery["status"] == "NO_RECOVERY"
    assert pd.Timestamp(result["source_available_at"]) >= pd.Timestamp(recovery["available_at"])


def test_backdated_or_altered_r13_policy_and_wrong_method_input_policy_are_rejected():
    value = data()
    value["cutoff"] = (pd.Timestamp(AVAILABLE_AT) - pd.Timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"): evaluate(value)
    value = data(); value["policy"]["required_return"] = 0.
    with pytest.raises(ValueError, match="POLICY_MISMATCH"): evaluate(value)
    value = data(); value["input_policy"] = input_policy()
    with pytest.raises(ValueError, match="METHOD_MISMATCH"): evaluate(value)
    fixed = input_policy(version=INPUT_VERSION)
    assert require_input_policy(fixed, cutoff=CUTOFF) == fixed
    fixed["annual_basis"] = "SYNTHETIC_UNVERIFIED"
    with pytest.raises(ValueError, match="POLICY_MISMATCH"): require_input_policy(fixed, cutoff=CUTOFF)
