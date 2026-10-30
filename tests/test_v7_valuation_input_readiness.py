"""SYNTHETIC ONLY: separate annual-price evidence from quarterly Quality.

These checks do not establish a real financial source, entry eligibility,
historical deployment, or investment performance. Quality and all other gates
remain independently evaluated by the engine.
"""
from copy import deepcopy

import pandas as pd
import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.valuation_eps_receipt import (
    INDEPENDENT_PRODUCER,
    PRODUCER,
    produce_price_valuation_receipt,
)
from domain.les_v7.valuation_input_policy import input_policy, require_input_policy
from domain.les_v7.valuation_price import policy as price_policy
from tests.test_v7_eps_price_receipts import item, source


CUTOFF = "2026-10-14T23:30:00+08:00"


def synthetic_item(row=None, *, quality=None):
    data = item(row)
    data["cutoff"] = CUTOFF
    data["market_source"]["source_available_at"] = CUTOFF
    if quality is not None:
        data["quality"] = quality
    return data


def evaluate(data, *, independent=True):
    return produce_price_valuation_receipt(
        **data, **({"input_policy": input_policy()} if independent else {}))


def test_legacy_quality_dependency_remains_unknown_and_requires_explicit_opt_in():
    data = synthetic_item(quality=dict(core_inputs_pass=False,
        status="DATA_INSUFFICIENT", reason="SYNTHETIC_CURRENT_QUARTER_MISSING"))
    legacy = evaluate(data, independent=False)
    current = evaluate(data)
    assert legacy["gate_version"] == PRODUCER
    assert legacy["state"] == "UNKNOWN"
    assert legacy["reason_codes"] == ["PRICE_VALUATION_QUALITY_INPUTS_UNVERIFIED"]
    assert current["gate_version"] == INDEPENDENT_PRODUCER
    assert current["state"] == "PASS"
    assert current["valuation_diagnostics"]["policy"] == price_policy()
    assert current["valuation_diagnostics"]["policy_id"] == identity(price_policy())


@pytest.mark.parametrize("quality", [
    {},
    dict(core_inputs_pass=False),
    dict(core_inputs_pass=None, status="UNKNOWN"),
    dict(core_inputs_pass=False, status="DATA_INSUFFICIENT"),
    dict(core_inputs_pass=False, status="REJECT", score=0.),
])
def test_complete_annual_operands_are_evaluated_without_changing_quality_or_inputs(quality):
    data = synthetic_item(quality=quality)
    before = deepcopy({key: value for key, value in data.items() if key != "bars"})
    bars_before = data["bars"].copy(deep=True)
    accepted_quality = synthetic_item(quality=dict(core_inputs_pass=True, status="PASS"))
    expected = evaluate(accepted_quality)["valuation_diagnostics"]
    result = evaluate(data)
    assert result["state"] == expected["status"] == "PASS"
    assert result["valuation_diagnostics"] == expected
    assert {key: value for key, value in data.items() if key != "bars"} == before
    pd.testing.assert_frame_equal(data["bars"], bars_before)
    assert data["quality"] == quality


@pytest.mark.parametrize("damage", [
    "missing_original", "future_original", "missing_companion", "conflicting_original",
    "bad_eps_unit", "future_eps_proof", "missing_current_revenue",
    "missing_comparative_operating_income", "wrong_money_original", "wrong_scope",
])
def test_independent_readiness_never_replaces_missing_or_unverified_annual_evidence(damage):
    data = synthetic_item(quality=dict(core_inputs_pass=False, status="UNKNOWN"))
    row = data["packet"]["valuation_filings"][0]
    if damage == "missing_original":
        data["packet"]["valuation_filings"] = []
    elif damage == "future_original":
        row["available_at"] = "2026-10-15T00:00:00+08:00"
    elif damage == "missing_companion":
        row.pop("earnings_source")
    elif damage == "conflicting_original":
        other = deepcopy(row)
        other["revision_id"] = "SYNTHETIC_CONFLICTING_ORIGINAL"
        data["packet"]["valuation_filings"].append(other)
    elif damage == "bad_eps_unit":
        row["earnings_source"]["field_evidence"]["diluted_eps"]["contexts"][0]["unit"] = "iso4217:TWD"
    elif damage == "future_eps_proof":
        row["earnings_source"]["comparative_ytd_field_evidence"]["diluted_eps"]["available_at"] = "2026-10-15T00:00:00+08:00"
    elif damage == "missing_current_revenue":
        row.pop("revenue")
    elif damage == "missing_comparative_operating_income":
        row["comparative_ytd_fields"].pop("operating_income")
    elif damage == "wrong_money_original":
        row["comparative_ytd_field_evidence"]["revenue"]["content_sha256"] = "0" * 64
    else:
        row["scope"] = "individual"
    result = evaluate(data)
    assert result["state"] == "UNKNOWN"
    assert result["reason_codes"]
    assert "PRICE_VALUATION_QUALITY_INPUTS_UNVERIFIED" not in result["reason_codes"]


@pytest.mark.parametrize("value", [None, 0., float("nan"), True])
def test_independent_current_pe_missing_does_not_fall_back_to_old_positive_quote(value):
    data = synthetic_item(quality=dict(core_inputs_pass=False))
    data["bars"] = pd.DataFrame(dict(PE=[15.] * 59 + [value]))
    result = evaluate(data)
    assert result["state"] == "UNKNOWN"
    assert result["reason_codes"] == ["OFFICIAL_CURRENT_OR_HISTORY_PE_UNVERIFIED"]


def test_independent_history_pe_still_requires_the_original_minimum_observations():
    data = synthetic_item(quality=dict(core_inputs_pass=False))
    data["bars"] = pd.DataFrame(dict(PE=[None] * 59 + [6.]))
    result = evaluate(data)
    assert result["state"] == "UNKNOWN"
    assert result["reason_codes"] == ["OFFICIAL_CURRENT_OR_HISTORY_PE_UNVERIFIED"]


@pytest.mark.parametrize("row", [
    source(current_eps=-1.),
    source(prior_eps=0.),
    source(money_overrides={("operating_income", 2025): 0.}),
    source(money_overrides={("operating_income", 2024): -10.}),
])
def test_verified_nonpositive_earnings_remain_financial_fail_in_both_producers(row):
    data = synthetic_item(row, quality=dict(core_inputs_pass=False, status="DATA_INSUFFICIENT"))
    legacy = evaluate(data, independent=False)
    independent = evaluate(data)
    assert independent["state"] == legacy["state"] == "FAIL"
    assert independent["reason_codes"] == legacy["reason_codes"]
    assert any(reason.startswith("NON_POSITIVE_") for reason in independent["reason_codes"])


def test_complete_but_expensive_price_scenarios_remain_fail_not_automatically_pass():
    data = synthetic_item(quality=dict(core_inputs_pass=False, status="UNKNOWN"))
    data["features"]["close"] = 3000.
    data["bars"] = pd.DataFrame(dict(PE=[15.] * 59 + [600.]))
    independent = evaluate(data)
    true_quality = deepcopy(data)
    true_quality["quality"] = dict(core_inputs_pass=True)
    assert independent["state"] == "FAIL"
    assert independent["valuation_diagnostics"] == evaluate(true_quality)["valuation_diagnostics"]
    assert independent["valuation_diagnostics"]["policy"] == price_policy()


def test_input_policy_is_a_detached_content_identified_contract():
    first, second = input_policy(), input_policy()
    assert first == second and first is not second
    assert identity(first) == identity(second)
    first["schema"] = "SYNTHETIC_CHANGED_SCHEMA"
    assert input_policy() == second
    require_input_policy(second, cutoff=CUTOFF)


@pytest.mark.parametrize("damage", ["schema", "version", "availability", "additional_field", "missing_field"])
def test_modified_input_policy_is_rejected_even_if_caller_recomputes_its_identity(damage):
    frozen = input_policy()
    if damage == "schema":
        frozen["schema"] = "SYNTHETIC_UNKNOWN_SCHEMA"
    elif damage == "version":
        key = "policy_version" if "policy_version" in frozen else "version"
        frozen[key] = "SYNTHETIC_CHANGED_VERSION"
    elif damage == "availability":
        frozen["available_at"] = "2026-01-01T00:00:00Z"
    elif damage == "additional_field":
        frozen["pretend_verified"] = True
    else:
        frozen.pop("schema")
    assert identity(frozen) != identity(input_policy())
    with pytest.raises(ValueError):
        require_input_policy(frozen, cutoff=CUTOFF)
    with pytest.raises(ValueError):
        produce_price_valuation_receipt(**synthetic_item(), input_policy=frozen)


def test_new_input_policy_cannot_be_backdated_into_an_earlier_research_cutoff():
    frozen = input_policy()
    before = (pd.Timestamp(frozen["available_at"]) - pd.Timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError):
        require_input_policy(frozen, cutoff=before)
    data = synthetic_item()
    data["cutoff"] = before
    data["market_source"]["source_available_at"] = before
    with pytest.raises(ValueError):
        produce_price_valuation_receipt(**data, input_policy=frozen)


def test_independent_source_policy_does_not_relax_the_frozen_economic_price_policy():
    data = synthetic_item()
    data["policy"]["required_return"] = 0.
    with pytest.raises(ValueError, match="VALUATION_POLICY_MISMATCH"):
        evaluate(data)
