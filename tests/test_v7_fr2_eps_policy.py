"""SYNTHETIC ONLY: integrated EPS/FR2 overlay, not traded-return validation."""
from copy import deepcopy

import pandas as pd
import pytest

from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.fr2_eps_valuation import INPUT_SCHEMA
from domain.les_v7.fr2_policy import apply_fr2, apply_fr2_eps, restore_original
from domain.les_v7.report import validate_report
from tests.test_v7_eps_price_receipts import source
from tests.test_v7_fr2_policy import native, request


CUTOFF = "2026-10-14T13:30:00+08:00"


def _symbol(value, symbol):
    if isinstance(value, dict):
        return {key: symbol if key == "symbol" else _symbol(child, symbol) for key, child in value.items()}
    if isinstance(value, (list, tuple)): return [_symbol(child, symbol) for child in value]
    return value


def case(*, restrictions=True):
    item = request()
    packet = item["context"]["native_evidence_source"]
    item["context"]["as_of"] = CUTOFF
    packet["cutoff"] = timestamp(CUTOFF).isoformat()
    annual = _symbol(source(current_eps=30., prior_eps=25.), "2330")
    fields = ("symbol", "market", "source", "scope", "currency", "unit", "period_end",
              "flow_basis", "available_at", "revision_id", "content_sha256", "filing_schema",
              "revenue", "operating_income", "net_income", "diluted_shares", "dividends")
    slim = {key: annual.get(key) for key in fields}
    slim["field_evidence"] = {key: annual.get("field_evidence", {}).get(key, {})
                             for key in ("revenue", "operating_income", "net_income", "diluted_shares", "dividends")}
    packet["valuation_filings"] = [slim]
    item["bars"][-1]["PE"] = 6.
    packet["market_rows"] = deepcopy(item["bars"])
    packet["market_evidence"].update(bars_sha256=identity(packet["market_rows"]),
        source_record_ids=[identity(packet["market_rows"])])
    if not restrictions: packet["restriction_evidence"] = {}
    original = native(item)
    supplement = dict(schema=INPUT_SCHEMA, original_report_id=identity(original), symbol="2330",
        market="TWSE", cutoff=timestamp(CUTOFF).isoformat(), valuation_filings=[annual])
    return original, item["context"]["model_bundle"], supplement


@pytest.fixture(scope="module")
def bound():
    return case()


def test_overlay_restores_exact_original_and_leaves_predictor_quality_and_plan_unchanged(bound):
    original, bundle, supplement = deepcopy(bound)
    before = canonical(dict(report=original, bundle=bundle, supplement=supplement))
    result = apply_fr2_eps(original, bundle, supplement)
    validate_report(result)
    assert result["diagnostics"]["fr2"]["schema"] == "les-v7-fr2-eps-overlay/v1"
    assert canonical(dict(report=original, bundle=bundle, supplement=supplement)) == before
    assert restore_original(result) == original
    for key in ("features", "quality", "reference_plan", "localization", "native_evidence", "strategy_policy"):
        assert result["diagnostics"][key] == original["diagnostics"][key]
    for key in ("model", "quality", "localization"):
        assert result["envelope"][key] == original["envelope"][key]
    for key in ("factor_values", "applicable_weights"):
        assert result["envelope"]["score"][key] == original["envelope"]["score"][key]
    assert result["diagnostics"]["fr2"]["order_authority"] is False


def test_effective_receipt_is_new_eps_method_but_original_native_receipt_is_retained(bound):
    original, bundle, supplement = bound
    result = apply_fr2_eps(original, bundle, supplement)
    before = next(g for g in original["diagnostics"]["gate_receipts"] if g["gate_id"] == "VALUATION")
    effective = next(g for g in result["diagnostics"]["gate_receipts"] if g["gate_id"] == "VALUATION")
    assert before["state"] == "UNKNOWN" and effective["state"] == "PASS"
    assert effective["input_hash"] == before["input_hash"]
    assert result["diagnostics"]["native_evidence"] == original["diagnostics"]["native_evidence"]
    for gate in result["diagnostics"]["gate_receipts"]:
        if gate["gate_id"] != "VALUATION":
            assert gate == next(g for g in original["diagnostics"]["gate_receipts"] if g["gate_id"] == gate["gate_id"])
    admission = result["diagnostics"]["fr2"]["admission"]
    assert admission["valuation"]["return_basis"] == "PRICE_RETURN_ONLY"
    assert admission["valuation"]["native_state"] == "PASS"
    # The existing unmatched price trigger is not manufactured by changing an
    # annual dependency or lowering the FR2 numerical admission boundary.
    assert result["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}


@pytest.mark.parametrize("damage", ["assessment", "supplement", "reason", "effective_receipt",
    "score", "forecast", "plan", "model", "schema", "missing_inputs"])
def test_rehashed_overlay_and_all_affected_business_fields_are_recomputed(bound, damage):
    original, bundle, supplement = bound
    result = apply_fr2_eps(original, bundle, supplement)
    overlay = result["diagnostics"]["fr2"]
    if damage == "assessment":
        overlay["admission"]["valuation"]["warning_eligible"] = True
        overlay["admission"]["assessment_id"] = identity({key: value for key, value in overlay["admission"].items() if key != "assessment_id"})
    elif damage == "supplement": overlay["valuation_inputs"]["valuation_filings"][0]["earnings_source"]["fields"]["diluted_eps"] = 999.
    elif damage == "reason": result["envelope"]["strategy"]["reason_codes"].append("FAKE_SOURCE_PASS")
    elif damage == "effective_receipt": next(g for g in result["diagnostics"]["gate_receipts"] if g["gate_id"] == "VALUATION")["state"] = "UNKNOWN"
    elif damage == "score": result["envelope"]["score"]["final_score"] = 100.
    elif damage == "forecast": result["diagnostics"]["localization"]["p_fill"] = .99
    elif damage == "plan": result["diagnostics"]["reference_plan"]["stop"] += 1.
    elif damage == "model": overlay["predictor_bundle"]["payload"]["weights"]["REVERSAL"][0] += .01
    elif damage == "schema": overlay["schema"] = "les-v7-fr2-overlay/v1"
    else: overlay.pop("valuation_inputs")
    with pytest.raises(ValueError): validate_report(result)


def test_original_fr2_path_keeps_old_annual_requirements_and_namespace(bound):
    original, bundle, supplement = bound
    before = canonical(original)
    old = apply_fr2(original, bundle)
    new = apply_fr2_eps(original, bundle, supplement)
    validate_report(old); validate_report(new)
    assert old["diagnostics"]["fr2"]["schema"] == "les-v7-fr2-overlay/v1"
    assert old["diagnostics"]["fr2"]["admission"]["valuation"]["native_state"] == "UNKNOWN"
    assert old["envelope"]["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"
    assert old["envelope"]["event"]["signal_id"] != new["envelope"]["event"]["signal_id"]
    assert canonical(original) == before


def test_unknown_restriction_is_not_a_valuation_warning_or_an_entry():
    original, bundle, supplement = case(restrictions=False)
    result = apply_fr2_eps(original, bundle, supplement)
    validate_report(result)
    assert result["diagnostics"]["fr2"]["admission"]["valuation"]["native_state"] == "PASS"
    assert next(g for g in result["diagnostics"]["gate_receipts"] if g["gate_id"] == "TRADING_STATUS")["state"] == "UNKNOWN"
    assert result["envelope"]["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"
    assert result["envelope"]["score"]["final_score"] is None


def test_invalid_stop_cannot_be_repaired_by_the_valuation_component(bound):
    original, bundle, supplement = deepcopy(bound)
    original["diagnostics"]["reference_plan"]["stop"] = original["diagnostics"]["reference_plan"]["entry"]
    supplement["original_report_id"] = identity(original)
    with pytest.raises(ValueError): apply_fr2_eps(original, bundle, supplement)


def test_allocation_is_not_an_input_to_model_or_admission_score(bound):
    from domain.les_v7.fr2_execution import assess_execution
    original, bundle, supplement = bound
    result = apply_fr2_eps(original, bundle, supplement)
    before = canonical(result)
    admission = result["diagnostics"]["fr2"]["admission"]
    for cash in (0., 1_000_000.):
        assessed = assess_execution(result, admission, dict(value=1_000_000., available_cash=cash,
            intended_weight=.25, capacity_weight=.25, positions=[], as_of=timestamp(CUTOFF).isoformat()))
        assert assessed["allocation_authorized"] is False
        assert assessed["indicative_shares"] == 0
        assert canonical(result) == before
