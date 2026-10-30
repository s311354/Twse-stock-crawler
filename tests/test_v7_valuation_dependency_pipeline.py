"""SYNTHETIC ONLY: source-readiness separation never grants entry authority."""
from copy import deepcopy

import pandas as pd
import pytest

from application.low_entry_v7_daily import project_report
from domain.les_v7.allocation import authorize_candidate
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.engine import evaluate
from domain.les_v7.native_evidence import SOURCE_SCHEMA, produce_native_receipts
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (BASELINE_POLICY, R4_BASELINE_POLICY, baseline_policy)
from domain.les_v7.valuation_input_policy import input_policy
from domain.les_v7.valuation_eps_receipt import PRODUCER, INDEPENDENT_PRODUCER
from domain.trading_restrictions import CURRENT_RESTRICTION_SCOPE, RESTRICTION_ENDPOINTS
from tests.strategies.test_v7_rule_baseline import legal_bars, unit_outcome
from tests.test_v7_eps_price_receipts import CUTOFF, source


def request():
    bars = legal_bars()
    bars = bars.assign(Open=30., High=30.5, Low=29.5, Close=30., Volume=2_000_000.,
        Amount=60_000_000., Bid=29.95, Ask=30., PE=15.)
    bars.loc[bars.index[-1], "PE"] = 6.
    rows = bars.astype(object).where(pd.notna(bars), None).to_dict("records")
    for row in rows: row["Date"] = pd.Timestamp(row["Date"]).date().isoformat()
    session = rows[-1]["Date"]
    annual = source()
    packet = dict(schema=SOURCE_SCHEMA, symbol="SYNTHETIC", market="TWSE",
        cutoff=timestamp(CUTOFF).isoformat(), expected_session=session, historical_replay=False,
        market_rows=rows, valuation_filings=[annual], market_evidence=dict(symbol="SYNTHETIC",
            market="TWSE", session_date=session, source="TWSE", price_unit="TWD", volume_unit="shares",
            amount_unit="TWD", source_final=True, bars_sha256=identity(rows),
            source_available_at=CUTOFF, source_record_ids=[identity(rows)]),
        restriction_evidence=dict(source="TWSE_OPENAPI", session_date=session, observation_date=session,
            scope=CURRENT_RESTRICTION_SCOPE, available_at=CUTOFF, restricted=False,
            endpoints=list(RESTRICTION_ENDPOINTS)))
    context = dict(symbol="SYNTHETIC", market="TWSE", as_of=CUTOFF, rule_baseline=True,
        filings=[annual], native_evidence_source=packet, external_evidence_ids=["SYNTHETIC_ONLY_NO_MARKET_REPAIR"])
    return bars, context


def test_native_known_valuation_with_missing_quality_never_allows_entry_or_account_quota():
    bars, context = request(); before = deepcopy(context)
    report = evaluate(bars, context, receipt_producer=produce_native_receipts)
    validate_report(report)
    quality = report["diagnostics"]["quality"]
    assert quality["status"] == "DATA_INSUFFICIENT" and not quality["core_inputs_pass"]
    native = report["diagnostics"]["native_evidence"]
    assert native["valuation_input_policy"] == input_policy()
    receipt = native["produced_receipts"]["VALUATION"]
    assert receipt["gate_version"] == INDEPENDENT_PRODUCER and receipt["state"] == "PASS"
    assert receipt["valuation_diagnostics"]["entry_quality_required"] is True
    assert report["envelope"]["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"
    assert report["envelope"]["score"]["final_score"] <= 61
    row = project_report(report)
    assert row["LES v7 Execution Eligible"] is False and row["建議股數"] == 0
    quota = authorize_candidate(report, dict(account_equity=1_000_000., current_cash=1_000_000.,
        positions=[], as_of=CUTOFF))
    assert quota["authorized"] is False and quota["indicative_shares"] == 0
    assert context == before


def test_saved_r4_keeps_original_quality_dependency_and_validates_without_reinterpretation():
    bars, context = request()
    old = evaluate(bars, dict(context, rule_baseline_policy_version=R4_BASELINE_POLICY),
                   receipt_producer=produce_native_receipts)
    new = evaluate(bars, context, receipt_producer=produce_native_receipts)
    validate_report(old); validate_report(new)
    old_native = old["diagnostics"]["native_evidence"]
    assert "valuation_input_policy" not in old_native
    assert old_native["produced_receipts"]["VALUATION"]["gate_version"] == PRODUCER
    assert old_native["produced_receipts"]["VALUATION"]["state"] == "UNKNOWN"
    assert old_native["produced_receipts"]["VALUATION"]["reason_codes"] == ["PRICE_VALUATION_QUALITY_INPUTS_UNVERIFIED"]
    assert old["envelope"]["quality"] == new["envelope"]["quality"]
    assert old["diagnostics"]["features"] == new["diagnostics"]["features"]
    assert old["diagnostics"]["rule_baseline"]["score_evidence"] == new["diagnostics"]["rule_baseline"]["score_evidence"]
    assert old["parameter_version"] != new["parameter_version"]
    assert identity(baseline_policy(version=R4_BASELINE_POLICY)) == "bb983b14f9dc8c21d48e0243a704a6c995e96d9a8b0f9a154897c185feb7f69e"


@pytest.mark.parametrize("status", ["DATA_INSUFFICIENT", "REJECT", "WATCH_QUALITY"])
def test_high_rule_score_with_all_gate_receipts_pass_still_requires_native_quality_pass(status):
    result = unit_outcome(quality_status=status)
    assert result["score"]["raw_score"] >= 90
    assert result["decision"] not in {"BUY", "BUY_SMALL"}
    assert result["score"]["final_score"] <= 61


@pytest.mark.parametrize("damage", ["native_policy", "missing_native_policy", "baseline_policy", "receipt"])
def test_rehashed_native_dependency_or_receipt_cannot_bypass_recomputation(damage):
    bars, context = request()
    report = evaluate(bars, context, receipt_producer=produce_native_receipts)
    native = report["diagnostics"]["native_evidence"]
    if damage == "native_policy": native["valuation_input_policy"]["entry_quality_required"] = False
    elif damage == "missing_native_policy": native.pop("valuation_input_policy")
    elif damage == "baseline_policy":
        report["diagnostics"]["rule_baseline"]["policy"]["valuation_input_policy"]["required_annual_operands"] = []
    else: native["produced_receipts"]["VALUATION"]["gate_version"] = PRODUCER
    with pytest.raises(ValueError): validate_report(report)


def test_new_input_policy_keeps_same_valuation_economics_and_technical_rules():
    old, new = baseline_policy(version=R4_BASELINE_POLICY), baseline_policy()
    assert new["policy_version"] == BASELINE_POLICY
    for key in ("valuation_policy", "quality_policy", "config", "model_spec", "score_policy",
                "context_policy", "native_risk_limits"):
        assert old[key] == new[key]
    assert "valuation_input_policy" not in old and new["valuation_input_policy"] == input_policy()
