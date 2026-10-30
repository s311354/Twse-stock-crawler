"""SYNTHETIC ONLY: FR2 mechanics do not prove market forecast/trading gains."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from domain.les_v7.contracts import identity, canonical
from domain.les_v7.engine import evaluate
from domain.les_v7.fr2_policy import apply_fr2 as apply_with_model, restore_original, classify, component
from domain.les_v7.native_evidence import produce_native_receipts
from domain.les_v7.quality_policy import r3_quality_policy
from domain.les_v7.report import validate_report
from tests.test_v7_native_evidence import inputs, annual
from tests.strategies.test_low_entry_score_v7 import bundle, seal, CUTOFF
from tests.strategies.test_v7_reported_quality import source_records


def request():
    item = inputs()
    model = bundle()
    model["payload"]["quality_policy"] = r3_quality_policy()
    # The synthetic frozen model intentionally gives a 68.29 score, between
    # old SMALL72 and new SMALL65. No production weights are altered.
    for name in model["payload"]["weights"]:
        model["payload"]["weights"][name] = [.8, .2, 0., 0., 0., 0.]
    seal(model)
    source = item["source_packet"]
    source["symbol"] = source["market_evidence"]["symbol"] = "2330"
    source["valuation_filings"] = [annual(2023, 20000.), annual(2024, 24000.), annual(2025, 28800.)]
    for row in source["valuation_filings"]:
        row["symbol"] = "2330"
        for proof in row["field_evidence"].values(): proof["symbol"] = "2330"
    return dict(bars=item["bars"].to_dict("records"), context=dict(symbol="2330", market="TWSE", as_of=CUTOFF,
        filings=source_records(), quality_policy=r3_quality_policy(), model_bundle=model, native_evidence_source=source),
        account=dict(value=1_000_000., available_cash=1_000_000., intended_weight=.25, capacity_weight=.25,
            positions=[], as_of=CUTOFF))


def native(item=None):
    item = request() if item is None else item
    return evaluate(pd.DataFrame(item["bars"]), item["context"], receipt_producer=produce_native_receipts)


def apply_fr2(original):
    return apply_with_model(original, request()["context"]["model_bundle"])


@pytest.mark.parametrize("score,expected", [(64.999,"WATCH"),(65.,"BUY_SMALL"),(74.999,"BUY_SMALL"),(75.,"BUY"),(100.,"BUY")])
def test_approved_score_boundaries(score, expected):
    decision, _, _ = classify(ready=True, hard=[], valuation_warning=False,
        quality=dict(status="PASS",quality_score=70.), final=score, starter=True, confirmed=True,
        starter_gate=True, confirmed_gate=True)
    assert decision == expected


@pytest.mark.parametrize("change", [dict(ready=False,missing=True), dict(hard=["INVALID_STOP"]),
    dict(quality=dict(status="DATA_INSUFFICIENT",quality_score=None),ready=False,missing=True),
    dict(quality=dict(status="REJECT",quality_score=100.)), dict(starter_gate=False,confirmed_gate=False),
    dict(quality=dict(status="PASS",quality_score=69.999))])
def test_high_score_does_not_override_unknown_major_risk_invalid_stop_or_trigger(change):
    args=dict(ready=True,hard=[],valuation_warning=False,quality=dict(status="PASS",quality_score=100.),
        final=100.,starter=True,confirmed=True,starter_gate=True,confirmed_gate=True)
    args.update(change)
    assert classify(**args)[0] not in {"BUY","BUY_SMALL"}


def test_verified_valuation_warning_cannot_be_buy():
    assert classify(ready=True, hard=[], valuation_warning=True,
        quality=dict(status="PASS",quality_score=100.), final=100.,starter=True,confirmed=True,
        starter_gate=True,confirmed_gate=True)[0] == "BUY_SMALL"


def test_native_prediction_plan_and_original_are_immutable_with_separate_entry_identity():
    original = native()
    before = canonical(original)
    report = apply_fr2(original)
    validate_report(report)
    assert canonical(original) == before
    assert restore_original(report) == original
    assert original["envelope"]["strategy"]["strategy_decision"] == "ARMED"
    # This price path has not passed the original reversal trigger. Lowering a
    # numerical threshold must not manufacture that independent price event.
    assert report["envelope"]["strategy"]["strategy_decision"] == "ARMED"
    for key in ("model","quality","localization"):
        assert report["envelope"][key] == original["envelope"][key]
    for key in ("features","reference_plan","localization","gate_receipts","strategy_policy","evidence_binding"):
        assert report["diagnostics"][key] == original["diagnostics"][key]
    assert report["envelope"]["event"]["evidence_id"] == original["envelope"]["event"]["evidence_id"]
    assert report["envelope"]["event"]["signal_id"] != original["envelope"]["event"]["signal_id"]
    assert report == apply_fr2(original)
    assert report["diagnostics"]["research_components"]["entry"] == component()


@pytest.mark.parametrize("damage", ["score","forecast","entry","plan","restore","component"])
def test_fr2_public_validation_reconstructs_all_business_fields(damage):
    report = apply_fr2(native())
    if damage == "score": report["envelope"]["score"]["final_score"] += .1
    elif damage == "forecast": report["diagnostics"]["localization"]["p_fill"] = .99
    elif damage == "entry": report["diagnostics"]["fr2"]["admission"]["small_allowed"] = not report["diagnostics"]["fr2"]["admission"]["small_allowed"]
    elif damage == "plan": report["diagnostics"]["reference_plan"]["stop"] += 1
    elif damage == "restore": report["diagnostics"]["fr2"]["original_report_id"] = "0"*64
    else: report["diagnostics"]["research_components"]["entry"]["component_id"] = "0"*64
    with pytest.raises(ValueError): validate_report(report)


def test_unknown_restriction_never_becomes_warning_or_buy():
    item = request()
    item["context"]["native_evidence_source"]["restriction_evidence"] = {}
    report = apply_fr2(native(item))
    validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"
    assert report["envelope"]["score"]["final_score"] is None


def test_frozen_model_reexecution_catches_rehashed_forecast_and_weights():
    original=native()
    original["diagnostics"]["localization"]["p_fill"] = .99
    original["envelope"]["localization"].update(p_fill=.99,
        forecast_result_id=identity(original["diagnostics"]["localization"]))
    # Even an otherwise rehashed native non-entry result cannot replace the
    # original predictor's probabilities under the same bundle identity.
    with pytest.raises(ValueError): apply_fr2(original)
    original=native()
    model=request()["context"]["model_bundle"]
    model["payload"]["weights"]["REVERSAL"]=[1.,0.,0.,0.,0.,0.]
    seal(model)
    with pytest.raises(ValueError,match="PREDICTOR_ID|SOURCE_BUNDLE_MISMATCH"):apply_with_model(original,model)


def test_future_filings_and_revisions_do_not_change_prior_cutoff():
    item = request()
    original = native(item)
    future = deepcopy(item["context"]["filings"][-1])
    future.update(available_at="2027-01-01T00:00:00Z",net_income=99999999.)
    item["context"]["filings"].append(future)
    # Engine's evidence cutoff is unchanged; no cache or FR2 policy can include it.
    assert native(item) == original
    assert apply_fr2(native(item)) == apply_fr2(original)


def test_unverified_dividends_are_not_replaced_by_zero_or_reason_allowlist():
    item = request()
    item["context"]["native_evidence_source"]["valuation_filings"][-1]["field_evidence"].pop("dividends")
    report = apply_fr2(native(item))
    assert report["diagnostics"]["fr2"]["admission"]["valuation_warning"] is False
    assert report["envelope"]["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"


def test_fr2_cannot_accidentally_use_legacy_trained_allocator():
    from domain.les_v7.allocation import authorize_candidate, _authorize_validated
    report=apply_fr2(native())
    for method in (authorize_candidate,_authorize_validated):
        with pytest.raises(ValueError,match="VERSIONED_EXECUTION_COMPONENT"):
            method(report,request()["account"])


def test_original_and_new_signal_namespaces_coexist_without_overwriting(tmp_path):
    from infrastructure.storage.les_v7_repository import LesV7SignalRepository
    original=native()
    fr2=apply_fr2(original)
    repository=LesV7SignalRepository(tmp_path/"isolated_signals")
    a,b=repository.publish(original),repository.publish(fr2)
    assert a["signal_id"] != b["signal_id"]
    assert repository.load(a["signal_id"]) == original
    assert repository.load(b["signal_id"]) == fr2
    assert repository.publish(fr2) == b


def test_confirmed_native_reversal_can_enter_but_missing_execution_costs_still_block():
    from tests.strategies.test_daily_v6_pipeline import daily_market
    from domain.les_v7.fr2_execution import assess_execution
    item=request()
    market=daily_market()
    data=pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-12",periods=500).strftime("%Y-%m-%d"),
        Open=[v/1000 for v in market.opens[0]], High=[v/1000 for v in market.highs[0]],
        Low=[v/1000 for v in market.lows[0]],Close=[v/1000 for v in market.closes[0]],
        Bid=[v/1000 for v in market.bids[0]],Ask=[v/1000 for v in market.asks[0]],
        Volume=market.volumes[0],Amount=market.trade_values[0],PE=market.pe_ratios[0],
        **{"TAIEX Close":market.taiex_closes[0]}))
    data.loc[data.index[-6:],"Low"]=[94.5,94.7,94.4,95.5,95.7,96.1]
    source=item["context"]["native_evidence_source"]
    source["market_rows"]=data.to_dict("records")
    source["market_evidence"].update(bars_sha256=identity(source["market_rows"]),
        source_record_ids=[identity(source["market_rows"])])
    item["bars"]=source["market_rows"]
    model=item["context"]["model_bundle"]
    for setup in model["payload"]["weights"]: model["payload"]["weights"][setup]=[1.,0.,0.,0.,0.,0.]
    seal(model)
    original=native(item)
    report=apply_with_model(original,model)
    validate_report(report)
    assert report["envelope"]["strategy"]["strategy_decision"] in {"BUY","BUY_SMALL"}
    before=deepcopy(report["envelope"]["score"])
    execution=assess_execution(report,report["diagnostics"]["fr2"]["admission"],item["account"])
    # This old synthetic bundle has no native share-size cost-path source.
    assert execution["state"] == "UNKNOWN" and execution["indicative_shares"] == 0
    assert report["envelope"]["score"] == before


def test_public_request_requires_real_explicit_account_and_no_mail(tmp_path,capsys):
    from application.fr2_research import analyze_fr2_request
    from interface.les_v7_fr2 import main
    item = request()
    payload=dict(schema="les-v7-fr2-request/v1",securities=[item])
    before=deepcopy(payload)
    result=analyze_fr2_request(payload)
    assert payload == before
    assert result["research_only"] and not result["execution_enabled"]
    report=result["reports"][0]
    validate_report(report)
    assert report["fr2_execution"]["live_order_authorized"] is False
    path=tmp_path/"request.json";path.write_text(canonical(payload),encoding="utf-8")
    assert main([str(path),"--output",str(tmp_path/"new.json"),"--html",str(tmp_path/"preview.html")]) == 0
    assert (tmp_path/"preview.html").is_file()
    with pytest.raises(SystemExit): main([str(path),"--output",str(tmp_path/"new.json")])
    payload["securities"][0].pop("account")
    with pytest.raises(ValueError,match="EXPLICIT_ACCOUNT"):analyze_fr2_request(payload)
