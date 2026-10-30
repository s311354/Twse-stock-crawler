"""SYNTHETIC ONLY: fixed rule arithmetic, not model training or real performance."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from domain.les_v7.contracts import FACTORS, GateReceipt, ReferencePlan, identity, timestamp
from domain.les_v7.engine import evaluate
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (AVAILABLE_AT, BASELINE_POLICY, R3_BASELINE_POLICY, CONFIG, baseline_policy, prepare_baseline,
    score_baseline, localization_record, select_baseline_context)
from domain.strategies.low_entry_v6_hybrid import select_context, score_factors
from tests.strategies.test_low_entry_score_v7 import bundle
from tests.strategies.test_v7_reported_quality import source_records


CUTOFF = "2026-10-14T13:30:00+08:00"


def legal_bars(count=120):
    close = np.round((110 + 5 * np.sin(np.arange(count) / 3)) * 2) / 2
    volume = 500000 + np.arange(count) % 5 * 50000
    return pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-14", periods=count).strftime("%Y-%m-%d"),
        Open=close, High=close + .5, Low=close - .5, Close=close, Volume=volume,
        Amount=close * volume, Bid=close - .5, Ask=close, PE=12., **{"TAIEX Close": np.full(count, 10000.)}))


def request():
    return legal_bars(), dict(symbol="2330", market="TWSE", as_of=CUTOFF, rule_baseline=True,
        filings=source_records(), receipts={}, external_evidence_ids=["SYNTHETIC_ONLY_NOT_MARKET_REPAIR"])


def bound_report():
    bars, context = request()
    first = evaluate(bars, context)
    for gate, inputs in first["diagnostics"]["gate_bindings"].items():
        if gate == "PRICE_DISCOVERY_PAYOFF": continue
        context["receipts"][gate] = dict(gate_id=gate, gate_version="SYNTHETIC_ONLY_EXTERNAL_PORT",
            state="PASS", input_hash=identity(inputs), source_available_at=timestamp(CUTOFF).isoformat(),
            proof_record_ids=["SYNTHETIC_ONLY"], input_ids=["SYNTHETIC_ONLY"],
            checked_at=timestamp(CUTOFF).isoformat(), reason_codes=[])
    return evaluate(bars, context)


def unit_features():
    """Handcrafted pure-factor operands; this is not a validated public report."""
    return dict(close=101., open=100.5, high=101.5, low=100.5, volume=500000., amount=50500000.,
        support=100., support_intact=True, higher_low=True, no_new_low2=True, new_low=False, new_lows3=0,
        support_break=False, undercut_reclaim=False, drawdown60=-.2, position60=.2, drawdown20=-.1,
        ema5=101., ema10=100., ema20=100., ema5_slope=.2, ema20_slope=0., atr14=2., atr_percent=2 / 101.,
        atr_expansion=.8, previous_high=100., hh20=109., hh60=110., swing_high=109.,
        rs20=.05, rs60=.1, rs_slope5=.01, resilience=.1, r5=.01, gap=.005, clv=.8, volume_ratio=1.5,
        market_ready=True, history_ready=True, market_regime="NEUTRAL", market_correction20=-.04,
        downside_capture20=.25, correction_support=100., industry_status="UNKNOWN",
        industry_rs20=None, industry_rs60=None, cmf20=.2, cmf_improving=True, obv_slope5=100000.,
        volume_ma20_strict=500000., amount_ratio=1.5, amount_ma20=50000000., rsi_change=5., rsi14=40.,
        rsi_cross40=True, macd_improving=True, rs_improving=True, break_previous_high=True,
        up_volume_ratio=1.5, vwap20=100., spread=.003, score_availability={})


def unit_outcome(features=None, *, missing_gate=None, quality_status="PASS", plan_present=True,
                 policy_version=BASELINE_POLICY):
    features = unit_features() if features is None else features
    prepared = dict(policy=baseline_policy(version=policy_version), features=features,
                    selection=select_baseline_context(features, policy_version=policy_version).record())
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL",
        101., 99., 109., 2., (101., 101.5), 1, .1, .1) if plan_present else None
    gates = [GateReceipt.bind(None, gate_id=gate, inputs={}, cutoff=CUTOFF) if gate == missing_gate else
             GateReceipt(gate, "SYNTHETIC_ONLY", "PASS", identity({}), timestamp(CUTOFF).isoformat(), (),
                         ("SYNTHETIC_ONLY",), ("SYNTHETIC_ONLY",), timestamp(CUTOFF).isoformat())
             for gate in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK")]
    return score_baseline(prepared, plan=plan, quality=dict(status=quality_status, quality_score=80.), gates=gates,
        native_features=dict(spread=.003, adv=50_000_000.), risks=[], standard=plan_present,
        starter_gate=plan_present, confirmed_gate=plan_present)


def test_fixed_policy_is_detached_and_has_no_model_head_probability_or_learned_weight():
    record = baseline_policy()
    before = identity(record)
    assert record["config_usage"] == "FACTOR_SCORING_AND_CONTEXT_ONLY_NOT_POSITION_POLICY"
    assert record["model_spec"]["factor_budgets"] == dict(structure=30, trend=10, relative_strength=20,
                                                        demand=25, momentum=5, execution=10)
    assert record["config"]["buy_score_minimum"] == 72
    assert record["config"]["reversal_starter_score_minimum"] == 62
    assert record["prediction_status"] == record["calibration_status"] == "NOT_APPLICABLE"
    assert not {"heads", "weights", "calibration"} & set(record)
    record["config"]["buy_score_minimum"] = 1.
    record["model_spec"]["factor_budgets"]["structure"] = 100
    assert identity(baseline_policy()) == before


def test_rule_pipeline_reports_real_fixed_score_without_odds_and_retains_hard_unknowns():
    bars, context = request()
    before = deepcopy(context)
    report = evaluate(bars, context)
    validate_report(report)
    assert context == before
    assert report["envelope"]["model"]["model_status"] == "RULE_BASED_BASELINE"
    assert report["envelope"]["model"]["bundle_sha256"] is None
    assert report["envelope"]["score"]["raw_score"] > 0
    assert report["envelope"]["score"]["final_score"] <= 61
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert report["diagnostics"]["localization"] == localization_record()
    assert "MODEL_ARTIFACTS_NOT_PROVIDED" not in report["envelope"]["strategy"]["reason_codes"]


def test_rule_ledger_reuses_v6_fixed_formula_with_same_v7_bound_prices():
    features = unit_features()
    result = unit_outcome(features, policy_version=R3_BASELINE_POLICY)
    direct = score_factors(features, select_context(features), dict(entry_price=101., stop_price=99.,
        target_price=109., stop_source="V7_BOUND_H20_STRUCTURE", stop_distance_atr=1.,
        stop_distance_fraction=2 / 101.), config=CONFIG)
    assert result["score_evidence"] == direct["score_evidence"]
    assert sum(row["weight"] for row in result["score_evidence"]["conditions"]) == pytest.approx(100)
    assert result["score"]["raw_score"] == pytest.approx(94.)
    assert result["decision"] == "BUY"  # Pure rules only; no real-source/portfolio/fill claim.


def test_missing_soft_indicator_gets_zero_no_renormalization():
    full = unit_outcome()
    features = unit_features(); features.update(rsi_change=None, rsi14=None)
    partial = unit_outcome(features)
    assert partial["score"]["raw_score"] == pytest.approx(full["score"]["raw_score"] - 5.)
    assert partial["score_evidence"]["total_weight"] == 100
    assert partial["score_evidence"]["coverage_percent"] == pytest.approx(89.)  # Six missing peer points plus RSI five.
    assert partial["weights"] == full["weights"]
    assert any(row["status"] == "UNKNOWN" for row in partial["score_evidence"]["conditions"] if row["group"] == "momentum")


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK"])
def test_high_score_cannot_compensate_missing_hard_receipt(gate):
    result = unit_outcome(missing_gate=gate)
    assert result["score"]["raw_score"] >= 90
    assert result["score"]["final_score"] <= 61
    assert result["decision"] == "DATA_INSUFFICIENT"


def test_quality_unknown_and_known_financial_reject_have_different_decisions():
    unknown, rejected = unit_outcome(quality_status="DATA_INSUFFICIENT"), unit_outcome(quality_status="REJECT")
    assert unknown["decision"] == "DATA_INSUFFICIENT" and unknown["score"]["final_score"] <= 61
    assert rejected["decision"] == "NO_BUY" and rejected["score"]["final_score"] == 0


def test_no_clear_context_watch_only_is_not_a_pass_for_nonexistent_plan():
    features = unit_features(); features.update(drawdown60=None, position60=None)
    result = unit_outcome(features, missing_gate="STRESS_RISK", plan_present=False)
    assert select_context(features).context == "NO_CLEAR_CONTEXT"
    assert result["decision"] == "WATCH_ONLY" and result["score"]["final_score"] <= 61
    features["market_ready"] = False
    assert unit_outcome(features, missing_gate="STRESS_RISK", plan_present=False)["decision"] == "DATA_INSUFFICIENT"


@pytest.mark.parametrize("case", ["model", "policy", "diagnostic", "availability", "boolean"])
def test_baseline_configuration_cannot_mix_or_backdate(case):
    bars, context = request()
    if case == "model": context["model_bundle"] = bundle()
    elif case == "policy": context["strategy_policy"] = {}
    elif case == "diagnostic": context["diagnostics_only"] = True
    elif case == "availability": context["as_of"] = "2026-10-13T13:30:00+08:00"
    else: context["rule_baseline"] = "true"
    with pytest.raises(ValueError): evaluate(bars, context)


@pytest.mark.parametrize("damage", ["probability", "calibration", "weights", "raw", "features", "policy", "mode", "decision"])
def test_rehashed_baseline_report_cannot_fake_trained_odds_or_alter_business_arithmetic(damage):
    report = bound_report()
    if damage == "probability": report["diagnostics"]["localization"]["p_entry"] = .99
    elif damage == "calibration": report["diagnostics"]["localization"]["calibration_version"] = "INVENTED"
    elif damage == "weights": report["envelope"]["score"]["applicable_weights"] = [1 / 6] * 6
    elif damage == "raw": report["envelope"]["score"]["raw_score"] = 100
    elif damage == "features": report["diagnostics"]["rule_baseline"]["features"]["ema5_slope"] = 123
    elif damage == "policy": report["diagnostics"]["rule_baseline"]["policy"]["config"]["buy_score_minimum"] = 1
    elif damage == "mode": report["envelope"]["model"]["model_status"] = "OFFLINE_FITTED"
    else: report["envelope"]["strategy"]["strategy_decision"] = "BUY"
    report["envelope"]["localization"]["forecast_result_id"] = identity(report["diagnostics"]["localization"])
    with pytest.raises(ValueError): validate_report(report)


def test_future_bars_and_financial_versions_do_not_change_baseline():
    bars, context = request()
    baseline = evaluate(bars, context)
    more = pd.concat([bars, bars.tail(1).assign(Date="2026-10-15", Close=1000., High=1001.)], ignore_index=True)
    context["filings"].append(dict(context["filings"][-1], available_at="2026-10-15T09:00:00+08:00", cfo=-1e9))
    assert evaluate(more, context) == baseline
