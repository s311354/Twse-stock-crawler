"""SYNTHETIC ONLY: rejected EPS premises are not missing valuation scenarios.

No live originals, database or network are used. Receipt content hashes are
bindings, not source signatures; native validation must still rederive sources.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest
import domain.strategies  # noqa: F401 -- preserve the existing plugin import order

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.entry_policy_r15 import (EARLY_EPS_FAILURE_AVAILABLE_AT,
    EARLY_EPS_FAILURE_SCHEMA, assess,
    tactical_valuation_grade)
from domain.les_v7.policy_r15 import INPUT_VERSION
from domain.les_v7.valuation_eps_receipt import produce_price_valuation_receipt
from domain.les_v7.valuation_input_policy import input_policy
from domain.les_v7.valuation_recovery_r15 import policy as valuation_policy
from tests.strategies.test_v7_entry_policy_r13 import gates, quality
from tests.test_v7_eps_price_receipts import item, source


CUTOFF = "2026-10-20T05:30:00Z"


def arguments(*, cutoff=CUTOFF, **changes):
    value = item(source(**changes))
    value.update(cutoff=cutoff, policy=valuation_policy(),
                 input_policy=input_policy(version=INPUT_VERSION))
    value["market_source"] = dict(source_available_at=cutoff, source="TWSE")
    value["binding"] = dict(symbol="SYNTHETIC", market="TWSE", cutoff=cutoff,
                            provenance="SYNTHETIC_ONLY")
    return value


def grade(receipt, *, cutoff=CUTOFF):
    return tactical_valuation_grade(receipt, current_price=30., cutoff=cutoff)


def rehash_receipt(receipt):
    receipt["proof_record_ids"] = [identity(dict(producer=receipt["gate_version"],
        gate_id="VALUATION", source_id=receipt["input_ids"][0],
        input_hash=receipt["input_hash"], result=receipt["valuation_diagnostics"]))]


@pytest.mark.parametrize("current,prior", [(-1., 10.), (0., 10.), (12., 0.), (-6.11, -10.8)])
@pytest.mark.parametrize("basic_only", [False, True])
def test_real_native_producer_early_eps_rejection_is_blocked_not_missing(current, prior, basic_only):
    args = arguments(current_eps=current, prior_eps=prior, basic_only=basic_only)
    receipt = produce_price_valuation_receipt(**args)
    before = deepcopy(receipt)
    result = grade(receipt)
    assert receipt["state"] == "FAIL" and result["state"] == "BLOCKED"
    assert result["reason"] == "VALUATION_VERIFIED_EPS_PREREQUISITE_FAIL"
    assert result["prerequisite_failure_schema"] == EARLY_EPS_FAILURE_SCHEMA
    assert result["points"] == result["multiplier"] == result["total_cap"] == 0.
    assert len(result["prerequisite_failure_id"]) == 64
    assert "inputs" not in receipt["valuation_diagnostics"]
    assert "scenarios" not in receipt["valuation_diagnostics"]
    assert receipt == before  # No mutation or new diagnostic in producer bytes.
    admitted = assess(quality(), gates(receipt), receipt,
        symbol="SYNTHETIC", cutoff=CUTOFF, current_price=30.)
    assert admitted["state"] == "BLOCKED"
    assert not admitted["buy_allowed"] and not admitted["small_allowed"]


@pytest.mark.parametrize("damage", ["value", "reason", "proof", "source", "input_hash",
    "producer", "state", "policy", "policy_id", "input_policy", "input_policy_id",
    "basis", "nan", "bool", "missing_operand", "future_available", "checked_cutoff",
    "missing_checked", "scenario_added"])
def test_changed_or_unresolved_early_receipts_cannot_claim_verified_rejection(damage):
    receipt = produce_price_valuation_receipt(**arguments(current_eps=-1.))
    d = receipt["valuation_diagnostics"]
    if damage == "value": d["observed_inputs"]["current_eps"] = -2.
    elif damage == "reason": d["reasons"] = ["NON_POSITIVE_PRIOR_EPS"]
    elif damage == "proof": receipt["proof_record_ids"] = ["0" * 64]
    elif damage == "source": receipt["input_ids"] = ["0" * 64]
    elif damage == "input_hash": receipt["input_hash"] = "0" * 64
    elif damage == "producer": receipt["gate_version"] = "UNVERIFIED_PRODUCER"
    elif damage == "state": receipt["state"] = "UNKNOWN"
    elif damage == "policy": d["policy"]["required_return"] = 0.
    elif damage == "policy_id": d["policy_id"] = "0" * 64
    elif damage == "input_policy": d["input_policy"]["revision_policy"] = "LATEST_NOW"
    elif damage == "input_policy_id": d["input_policy_id"] = "0" * 64
    elif damage == "basis": d["observed_inputs"]["eps_basis"] = "YTD_ANNUALIZED"
    elif damage == "nan": d["observed_inputs"]["current_eps"] = float("nan")
    elif damage == "bool": d["observed_inputs"]["current_eps"] = False
    elif damage == "missing_operand": d["observed_inputs"].pop("prior_eps")
    elif damage == "future_available": receipt["source_available_at"] = "2027-01-01T00:00:00Z"
    elif damage == "checked_cutoff": receipt["checked_at"] = "2027-01-01T00:00:00Z"
    elif damage == "missing_checked": receipt.pop("checked_at")
    else: d["scenarios"] = {}
    assert grade(receipt)["state"] == "UNKNOWN"


@pytest.mark.parametrize("damage", ["policy", "reason", "positive", "input_policy", "basis"])
def test_rehashed_non_native_fail_predicates_do_not_pass_exact_policy_and_operand_checks(damage):
    receipt = produce_price_valuation_receipt(**arguments(current_eps=-1.))
    d = receipt["valuation_diagnostics"]
    if damage == "policy":
        d["policy"]["required_return"] = 0.
        d["policy_id"] = identity(d["policy"])
    elif damage == "reason":
        d["reasons"] = receipt["reason_codes"] = ["NON_POSITIVE_PRIOR_EPS"]
    elif damage == "positive": d["observed_inputs"]["current_eps"] = 1.
    elif damage == "input_policy":
        d["input_policy"]["revision_policy"] = "LATEST_NOW"
        d["input_policy_id"] = identity(d["input_policy"])
    else: d["observed_inputs"]["eps_basis"] = "YTD"
    rehash_receipt(receipt)
    assert grade(receipt)["state"] == "UNKNOWN"


@pytest.mark.parametrize("damage", ["unit", "proof_hash", "companion", "future", "scope"])
def test_upstream_original_proof_errors_remain_unknown_not_verified_eps_failure(damage):
    args = arguments(current_eps=-1.)
    row = args["packet"]["valuation_filings"][0]
    earnings = row["earnings_source"]
    proof = earnings["field_evidence"]["diluted_eps"]
    if damage == "unit": proof["contexts"][0]["unit"] = "iso4217:TWD"
    elif damage == "proof_hash": proof["content_sha256"] = "0" * 64
    elif damage == "companion": earnings["content_sha256"] = "0" * 64
    elif damage == "future": proof["available_at"] = "2027-01-01T00:00:00Z"
    else: proof["scope"] = "individual"
    receipt = produce_price_valuation_receipt(**args)
    assert receipt["state"] == "UNKNOWN"
    assert grade(receipt)["state"] == "UNKNOWN"


@pytest.mark.parametrize("field", ["revenue", "operating_income"])
def test_legacy_nonpositive_money_receipt_without_observed_operands_stays_unknown(field):
    receipt = produce_price_valuation_receipt(**arguments(money_overrides={(field, 2025): 0.}))
    assert receipt["state"] == "FAIL"
    assert "observed_inputs" not in receipt["valuation_diagnostics"]
    assert grade(receipt)["state"] == "UNKNOWN"


def test_explicit_requested_cutoff_required_and_full_scenario_path_unchanged():
    receipt = produce_price_valuation_receipt(**arguments(current_eps=-1.))
    assert grade(receipt, cutoff=None)["state"] == "UNKNOWN"
    assert grade(receipt, cutoff="2026-10-20T05:30:01Z")["state"] == "UNKNOWN"
    from tests.strategies.test_v7_entry_policy_r15 import valuation
    complete = valuation(ratio=2.)
    price = complete["valuation_diagnostics"]["inputs"]["current_price"]
    assert tactical_valuation_grade(complete, current_price=price) == tactical_valuation_grade(
        complete, current_price=price, cutoff=CUTOFF)


@pytest.mark.parametrize("cutoff,state", [
    ((timestamp(EARLY_EPS_FAILURE_AVAILABLE_AT) - timedelta(microseconds=1)).isoformat(), "UNKNOWN"),
    (EARLY_EPS_FAILURE_AVAILABLE_AT, "BLOCKED"),
    (timestamp(EARLY_EPS_FAILURE_AVAILABLE_AT).astimezone(
        ZoneInfo("Asia/Taipei")).isoformat(), "BLOCKED"),
])
def test_diagnostic_code_activation_cutoff_is_explicit_and_timezone_equivalent(cutoff, state):
    receipt = produce_price_valuation_receipt(**arguments(cutoff=cutoff, current_eps=-1.))
    assert grade(receipt, cutoff=cutoff)["state"] == state


def test_old_r15_assessment_is_exactly_unchanged_before_diagnostic_activation(monkeypatch):
    cutoff = "2026-10-16T09:23:53Z"  # Retained full-universe comparison cutoff.
    receipt = produce_price_valuation_receipt(**arguments(cutoff=cutoff, current_eps=-1.))
    args = dict(symbol="SYNTHETIC", cutoff=cutoff, current_price=30.)
    current = assess(quality(), gates(receipt), receipt, **args)
    assert current["tactical_grade"]["state"] == "UNKNOWN"
    assert "prerequisite_failure_id" not in current["tactical_grade"]
    with monkeypatch.context() as old:
        old.setattr("domain.les_v7.entry_policy_r15._verified_early_eps_failure", lambda *args, **kwargs: None)
        previous = assess(quality(), gates(receipt), receipt, **args)
    assert current == previous  # Includes assessment_id and every cap/grade field.


def test_new_diagnostic_does_not_require_old_original_information_to_be_newly_available():
    args = arguments(cutoff=EARLY_EPS_FAILURE_AVAILABLE_AT, current_eps=-1.)
    args["market_source"]["source_available_at"] = "2026-10-16T09:23:53Z"
    receipt = produce_price_valuation_receipt(**args)
    assert timestamp(receipt["source_available_at"]) < timestamp(EARLY_EPS_FAILURE_AVAILABLE_AT)
    assert grade(receipt, cutoff=EARLY_EPS_FAILURE_AVAILABLE_AT)["state"] == "BLOCKED"


def test_native_validator_rejects_rehashed_early_fail_claim_not_derived_from_original():
    from domain.les_v7.native_evidence import validate_native_receipts
    from tests.test_v7_r15_pipeline import fresh_market, native
    from infrastructure.crawler.security_master import select_security_master
    from tests.test_security_master import packet
    value = fresh_market()
    at = (timestamp(EARLY_EPS_FAILURE_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    selections = deepcopy(value.native_source_selections)
    selections["2330"]["security"] = select_security_master(packet(), "2330", at)
    restrictions = deepcopy(value.trading_restrictions)
    restrictions["2330"]["available_at"] = at
    value = replace(value, signal_dates={"2330": at}, native_source_selections=selections,
        trading_restrictions=restrictions, les_v6_portfolio=dict(value.les_v6_portfolio, as_of=at))
    report = native(value)
    validate_native_receipts(report)
    receipt = report["diagnostics"]["native_evidence"]["produced_receipts"]["VALUATION"]
    d = receipt["valuation_diagnostics"]
    fabricated = {key: deepcopy(d[key]) for key in ("policy", "policy_id",
        "input_policy", "input_policy_id", "quality_dependency", "entry_quality_required")}
    fabricated.update(status="FAIL", return_basis="PRICE_RETURN_ONLY", reasons=["NON_POSITIVE_CURRENT_EPS"],
        observed_inputs=dict(current_eps=-1., prior_eps=10., eps_basis="DILUTED"))
    receipt.update(state="FAIL", reason_codes=["NON_POSITIVE_CURRENT_EPS"],
                   valuation_diagnostics=fabricated)
    rehash_receipt(receipt)
    # Hashes alone are not certification. Rejection cannot grant entry, and the
    # public native validator must still detect the original positive EPS.
    assert tactical_valuation_grade(receipt, current_price=30.,
        cutoff=report["envelope"]["event"]["cutoff_at"])["state"] == "BLOCKED"
    with pytest.raises(ValueError, match="NATIVE_RECEIPT_RECOMPUTATION_MISMATCH"):
        validate_native_receipts(report)


def test_existing_frozen_policy_identities_do_not_change():
    from domain.les_v7.policy_r14 import BASELINE_VERSION as previous
    from domain.les_v7.policy_r15 import BASELINE_VERSION as current
    from domain.les_v7.rule_baseline import baseline_policy
    assert identity(baseline_policy(version=previous)) == "ec1dc441e672fcd5ff4fb91dba609baa130090c5319c73a583a44e6f3bc33109"
    assert identity(baseline_policy(version=current)) == "30ff3530e27fd55f2aee4e6b40cd6568d86e513aeb1d356c657ad98d90e53086"
