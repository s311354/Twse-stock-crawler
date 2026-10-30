"""SYNTHETIC ONLY. No trained market artifact, approval, trade or profit claims."""
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import json
import math

import numpy as np
import pandas as pd
import pytest

from application.les_v7 import analyze_request
from domain.features import SharedFeaturePipeline
from domain.les_v7.contracts import CONTEXTS, FACTORS, PRIOR, FEATURE_VERSION, MAPPING_VERSION, GateReceipt, ReferencePlan, identity, timestamp
from domain.les_v7.decision import decide, factor_vector, make_plan, rule, score_once, select_context
from domain.les_v7.features import compute_features, select_bars
from domain.les_v7.fractals import dfa, higuchi, mfdfa, lppl_shadow
from domain.les_v7.localization import FrozenBundle, Forecast, predict, localization_gate
from domain.les_v7.quality import CORE, METRICS, Metric, quality_gate, evaluate_financials
from domain.les_v7.research import fit_heads, fit_factor_weights, label_event
from domain.strategies.low_entry_score_v7 import attach_low_entry_v7_context
from domain.strategy import get_strategy

CUTOFF = "2026-10-12T13:30:00+08:00"
POLICY = dict(filing_schedule_version="SYNTHETIC_ONLY", source_mapping_id="SYNTHETIC_ONLY",
              latest_required_period="2026Q2", tax_rule="VERIFIED_TTM_EXPENSE_OVER_PRETAX",
              capex_rule="PPE_PLUS_INTANGIBLES_POSITIVE_OUTFLOW")


def quality(gaps=()):
    ledger = [Metric(key, None if key in gaps else 1., "ROIC" if key in {"R1", "A3"} and key in gaps else
                     "CAPEX" if key in {"C3", "A2"} and key in gaps else None) for key in METRICS]
    return quality_gate(ledger, core_valid={key: True for key in CORE}, safety_complete=True,
                        gap_shapes=dict(ROIC=True, CAPEX=True))


def receipt(gate, inputs, state="PASS"):
    return dict(gate_id=gate, gate_version="SYNTHETIC_ONLY", state=state, input_hash=identity(inputs),
                source_available_at=timestamp(CUTOFF).isoformat(), proof_record_ids=["SYNTHETIC_ONLY"],
                input_ids=["SYNTHETIC_ONLY"], checked_at=timestamp(CUTOFF).isoformat(), reason_codes=[])


def bundle():
    compare = lambda field, op, value: dict(field=field, op=op, value=value)
    rules = {"REVERSAL": compare("close", "gt", 0),
             "TREND_PULLBACK": compare("close", "lt", 0), "LEADERSHIP_CONTINUATION": compare("close", "lt", 0)}
    payload = dict(bundle_id="SYNTHETIC_ONLY_NOT_APPROVED", trained_until="2026-01-02T00:00:00Z",
        fit_cutoff="2026-01-01T00:00:00Z", last_label_available_at="2025-12-30T00:00:00Z",
        feature_profile=FEATURE_VERSION, mapping_version=MAPPING_VERSION,
        fractal_mode="V7_NO_FRACTAL_BASELINE", baseline_validation_report_id="SYNTHETIC_ONLY",
        feature_names=["close"], feature_bounds=[[1., 200.]], imputation=[None], supported_masks=[[True]],
        context_rules=rules, trigger_rules=dict(near_pullback_support=compare("close", "gt", 0),
                                               tight_base_or_retest=compare("close", "gt", 0)),
        quality_policy=deepcopy(POLICY),
        reference_policy=dict(policy_id="REFERENCE_H20", version="SYNTHETIC_ONLY", entry_rule="CUTOFF_CLOSE_NEXT_SESSION_BAND",
                              support_rule="STRICT_PIVOT_ELSE_PRIOR_LL20", reference_shares=1, entry_band_atr=[0., .25],
                              cost_up_per_share=.3, cost_stop_per_share=.3, leadership_rr_min=1.8),
        weights={key: list(PRIOR) for key in (*CONTEXTS, "GLOBAL")},
        heads=dict(price=[[-.02, 0.], [-.01, 0.], [-.005, 0.]], timing=[[4., 0.]] + [[0., 0.]] * 20,
                   fill=[2., 0.], rebound=[2., 0.], joint=[2., 0.]),
        calibration=dict(version="SYNTHETIC_ONLY", method="POST_PROJECTION_PLATT_TEMPERATURE_R1", projection_validated=True,
                         validation_report_id="SYNTHETIC_ONLY", timing_temperature=1., fill=[1., 0.], rebound=[1., 0.], joint=[1., 0.]),
        training_manifest=dict(upstream_predictions="STRICTLY_PAST_OOF", dataset_id="SYNTHETIC_ONLY",
            oof_prediction_ids=["SYNTHETIC_ONLY"], folds=[dict(last_training_label_available_at="2025-01-01T00:00:00Z",
                fit_cutoff="2025-01-02T00:00:00Z", prediction_at="2025-01-03T00:00:00Z")]))
    return dict(payload=payload, sha256=identity(payload))


def seal(value):
    value["sha256"] = identity(value["payload"])
    return value


def prediction_for(status="VALID"):
    return (Forecast("VALID", (98., 99., 99.5), (1.,) + (0.,) * 20, .8, .8, .7,
                     calibration_version="SYNTHETIC_ONLY") if status == "VALID" else Forecast(status))


def bars(count=100):
    dates = pd.bdate_range(end="2026-10-12", periods=count)
    c = 105 + 5 * np.sin(np.arange(count) / 3)
    return pd.DataFrame(dict(Date=dates.strftime("%Y-%m-%d"), Open=c - .1, High=c + 2, Low=c - 2,
                             Close=c, Volume=100000 + (np.arange(count) % 5) * 5000,
                             Amount=c * 100000, Bid=c - .05, Ask=c + .05, **{"TAIEX Close": np.full(count, 10000.)}))


def filings():
    rows = []
    for period in pd.period_range("2024Q3", "2026Q2", freq="Q"):
        end, start = period.end_time.date().isoformat(), period.start_time.date().isoformat()
        fields = dict(revenue=22000. if period >= pd.Period("2025Q3") else 20000., cost_of_revenue=8000.,
            operating_income=6000., net_income=4000., cfo=5000., capex=1000., intangible_capex=100., income_tax=500., pretax_income=5000.,
            basic_eps=1., interest_expense=200., parent_net_income=4500., dividends=0., share_repurchases=0.,
            total_assets=100000., non_current_assets=50000., current_assets=50000., total_liabilities=20000.,
            current_liabilities=10000., total_equity=80000., parent_equity=78000., cash=20000., interest_bearing_debt=5000.,
            accounts_receivable=1000., inventory=3000.)
        sha = identity(dict(period=str(period), fields=fields))
        row = dict(symbol="SYNTHETIC_ONLY", market="TWSE", currency="TWD", unit="thousand", scope="consolidated",
                   source="MOPS_IXBRL", content_sha256=sha, revision_id=sha, revision_basis_id="SYNTHETIC_BASIS",
                   period_end=end, available_at=(period.end_time.date() + timedelta(days=45)).isoformat() + "T10:00:00+08:00",
                   flow_basis="QUARTER", **fields)
        instant = {"total_assets", "non_current_assets", "current_assets", "total_liabilities", "current_liabilities",
                   "total_equity", "parent_equity", "cash", "interest_bearing_debt", "accounts_receivable", "inventory"}
        row["field_evidence"] = {}
        for field, value in fields.items():
            eps = field == "basic_eps"
            proof = dict({k: row[k] for k in ("symbol", "source", "scope", "currency", "content_sha256", "revision_id", "available_at")},
                         status="RESOLVED", unit="TWD/share" if eps else "thousand", start_date=None if field in instant else start,
                         end_date=end, basis="INSTANT" if field in instant else "QUARTER", contexts=[dict(
                             numeric_value=value if eps else value * 1000, unit="iso4217:TWD/xbrli:shares" if eps else "iso4217:TWD",
                             start_date=None if field in instant else start, end_date=None if field in instant else end,
                             instant=end if field in instant else None, dimensions=[], context_id="SYNTHETIC_ONLY")])
            row["field_evidence"][field] = proof
        row["adjusted_share_growth_evidence"] = dict(value=0., source_hash=sha, status="VERIFIED", policy_id="SYNTHETIC_ONLY", available_at=row["available_at"])
        rows.append(row)
    return rows


@pytest.mark.parametrize("gaps,expected,coverage", [((), 100, 1), (("R1", "A3"), 84, .87),
    (("C3", "A2"), 87, .9), (("R1", "A3", "C3", "A2"), 71, .77)])
def test_quality_budget_and_bounded_gaps(gaps, expected, coverage):
    q = quality(gaps)
    assert q["status"] == "PASS" and q["quality_score"] == expected
    assert q["coverage_total"] == coverage and q["quality_les_cap"] == 100
    assert sum(budget for domain, budget in METRICS.values()) == 100


@pytest.mark.parametrize("field", CORE)
def test_missing_mandatory_ttm_cannot_be_bounded_gap(field):
    valid = {key: key != field for key in CORE}
    q = quality_gate([Metric(key, 1.) for key in METRICS], core_valid=valid, safety_complete=True)
    assert q["status"] == "DATA_INSUFFICIENT" and q["quality_score"] is None


@pytest.mark.parametrize("score,expected", [(54.999, "AVOID"), (55, "WATCH"), (64.999, "WATCH"), (65, "ARMED"),
    (71.999, "ARMED"), (72, "BUY_SMALL"), (79.999, "BUY_SMALL"), (80, "BUY"), (100, "BUY")])
def test_unrounded_decision_boundaries(score, expected):
    f = prediction_for()
    assert decide(quality=quality(), valuation="PASS", context="REVERSAL", forecast=f, final=score,
                  starter_trigger=True, confirmed_trigger=True, starter_gate=True, confirmed_gate=True) == expected


def test_confirmed_vs_starter_quality_and_caps():
    q = quality(("R1", "A3", "C3", "A2"))
    assert decide(quality=q, valuation="PASS", context="REVERSAL", forecast=prediction_for(), final=90,
                  starter_trigger=True, starter_gate=True) == "ARMED"
    assert decide(quality=q, valuation="PASS", context="REVERSAL", forecast=prediction_for(), final=90,
                  confirmed_trigger=True, confirmed_gate=True) == "BUY"
    score = score_once([1.] * 6, PRIOR, {}, dict(confirmation=79., localization=79.))
    assert score["final_score"] == 79
    assert score_once(None, PRIOR, {}, {})["final_score"] is None


@pytest.mark.parametrize("weights", [[-.1, .2, .2, .2, .2, .3], [0.] * 6, [1.] * 5, [True] * 6])
def test_no_weight_fallback_or_renormalization(weights):
    with pytest.raises(ValueError): score_once([1.] * 6, weights, {}, {})


@pytest.mark.parametrize("forecast,required,valuation,veto,expected", [
    ("NOT_READY", True, "PASS", False, "DATA_INSUFFICIENT"), ("OOD", True, "PASS", False, "WATCH_ONLY"),
    ("VALID", False, "PASS", False, "DATA_INSUFFICIENT"), ("VALID", True, "UNKNOWN", False, "DATA_INSUFFICIENT"),
    ("VALID", True, "FAIL", False, "NO_BUY"), ("NOT_READY", False, "UNKNOWN", True, "NO_BUY")])
def test_high_score_never_compensates_hard_or_unknown_gates(forecast, required, valuation, veto, expected):
    prediction = prediction_for(forecast)
    assert decide(quality=quality(), valuation=valuation, context="REVERSAL", forecast=prediction, final=100,
                  hard_veto=veto, required_valid=required, starter_trigger=True, confirmed_trigger=True,
                  starter_gate=True, confirmed_gate=True) == expected


def test_fractal_formula_validity_and_direction_invariance():
    rng = np.random.default_rng(17)
    x = rng.normal(size=1024)
    a, b = dfa(x), dfa(-x)
    assert a.status == "ESTIMATED" and a.value == pytest.approx(b.value)
    assert .3 < a.value < .8 and a.validation_status == "NULL_ENVELOPE_NOT_PROVIDED"
    assert dfa(np.zeros(512)).value is None and dfa(x[:50]).value is None
    assert higuchi(np.arange(256.)).value == pytest.approx(1., abs=.03)
    assert higuchi(np.zeros(256)).value is None
    assert mfdfa(np.zeros(2048))["status"] == "ZERO_FLUCTUATION"
    assert lppl_shadow(np.ones(120))["status"] == "LPPL_NOT_APPLICABLE"


def test_prefix_feature_invariance_and_prior_volume_denominator():
    data = bars()
    selected = select_bars(data, CUTOFF)
    f, audit = compute_features(selected)
    more = pd.concat([data, data.tail(1).assign(Date="2026-10-13", Close=1., Low=.5)], ignore_index=True)
    assert compute_features(select_bars(more, CUTOFF)) == (f, audit)
    assert f["vr"] == pytest.approx(data.Volume.iloc[-1] / data.Volume.iloc[-21:-1].mean())
    assert f["hh20"] == data.High.iloc[-21:-1].max()
    missing = data.copy(); missing.loc[missing.index[-1], "TAIEX Close"] = np.nan
    fm, _ = compute_features(missing)
    assert fm["rs20"] is None and fm["rs60"] is None


def test_pivot_confirmation_does_not_backdate_or_accept_ties():
    data = bars(70)
    data.loc[65:69, "Low"] = [100., 99., 98., 99., 100.]
    data.loc[65:69, "High"] = 115.
    data.loc[65:69, "Open"] = data.loc[65:69, "Close"] = 105.
    before, _ = compute_features(data.iloc[:68])
    after, _ = compute_features(data)
    assert before["support"] != 98.
    assert after["higher_low"] is not None
    data.loc[68, "Low"] = 98.
    tied, _ = compute_features(data)
    assert tied["support"] != 98.


def test_context_rules_cannot_score_shop_or_read_predictions():
    rules = {c: dict(op="gt", field="close", value=0) for c in CONTEXTS}
    features = dict(close=1, return20=-.1, return5=-.01, no_low2=True, reclaim=False,
                    prior_slope20=.1, slope20=.1, ema10=2, ema20=1, support_intact=True, drawdown20=-.1,
                    rs20=.1, rs_slope=.1)
    assert select_context(features, rules)[0] == "NO_CLEAR_CONTEXT"
    rules["REVERSAL"] = dict(op="lt", field="close", value=0)
    assert select_context(features, rules)[0] == "TREND_PULLBACK"
    with pytest.raises(ValueError, match="CANNOT_READ"):
        rule(dict(op="gt", field="p_entry", value=.5), dict(p_entry=.9))
    assert rule(dict(op="gt", field="missing", value=0), {}) is None


def test_head_projections_keep_raw_predictions_and_probability_order():
    raw = bundle()
    raw["payload"]["heads"]["price"] = [[.1, 0.], [-.01, 0.], [-.1, 0.]]
    raw["payload"]["heads"]["joint"] = [4., 0.]
    raw["payload"]["heads"]["rebound"] = [1., 0.]
    model = FrozenBundle.load(seal(raw))
    f = dict(close=100., low=98., atr=2.)
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL", 100., 97., 108., 2., (100., 100.5), 1, .3, .3)
    result = predict(model, f, plan, cutoff=CUTOFF)
    assert result.status == "VALID" and result.quantiles[0] <= result.quantiles[1] <= result.quantiles[2] <= 98.
    assert result.p_entry <= result.p_rebound and math.fsum(result.timing) == pytest.approx(1.)
    assert len(result.timing) == 21 and len(result.repairs) == 2 and result.raw_log_quantiles[0] == .1
    detached = model.record(); detached["heads"]["fill"][0] = -1000
    assert predict(model, f, plan, cutoff=CUTOFF) == result


@pytest.mark.parametrize("change,pattern", [("hash", "HASH"), ("weights", "WEIGHTS"), ("oof", "OOF"),
    ("maturity", "LEAKAGE"), ("calibration", "CALIBRATION"), ("baseline", "BASELINE")])
def test_untrained_invalid_or_leaky_artifacts_are_rejected(change, pattern):
    raw = bundle()
    if change == "hash": raw["sha256"] = "0" * 64
    elif change == "weights": raw["payload"]["weights"]["REVERSAL"] = [0.] * 6
    elif change == "oof": raw["payload"]["training_manifest"]["upstream_predictions"] = "IN_SAMPLE"
    elif change == "maturity": raw["payload"]["last_label_available_at"] = "2026-02-01T00:00:00Z"
    elif change == "calibration": raw["payload"]["calibration"]["projection_validated"] = False
    else: del raw["payload"]["baseline_validation_report_id"]
    if change != "hash": seal(raw)
    with pytest.raises(ValueError, match=pattern): FrozenBundle.load(raw)


def test_future_model_and_unsupported_feature_mask_do_not_produce_odds():
    raw = bundle(); raw["payload"]["trained_until"] = "2027-01-01T00:00:00Z"
    model = FrozenBundle.load(seal(raw))
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL", 100., 97., 108., 2., (100., 101.), 1, .3, .3)
    assert predict(model, dict(close=100., low=99.), plan, cutoff=CUTOFF).status == "NOT_READY"
    model = FrozenBundle.load(bundle())
    assert predict(model, dict(close=None, low=99.), plan, cutoff=CUTOFF).status == "OOD"


def test_native_financial_proof_selection_and_latest_ttm():
    rows = filings()
    safety = GateReceipt.bind(receipt("SAFETY", {}), gate_id="SAFETY", inputs={}, cutoff=CUTOFF)
    q = evaluate_financials(rows, symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=CUTOFF, safety_receipt=safety, policy=POLICY)
    assert q["core_ttm_pass"] and q["quality_score"] is not None
    assert q["core_ttm"]["cfo"] == 20000 and q["status"] == "PASS"
    before = deepcopy(rows)
    future = deepcopy(rows[-1]); future["available_at"] = "2026-10-13T10:00:00+08:00"; future["cfo"] = -1e9
    assert evaluate_financials(rows + [future], symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=CUTOFF, safety_receipt=safety, policy=POLICY) == q
    assert rows == before


@pytest.mark.parametrize("damage", ["missing", "proof", "scope", "basis", "latest"])
def test_financial_gaps_no_skipna_no_older_revision_fallback(damage):
    rows = filings()
    if damage == "missing": rows.pop(-2)
    elif damage == "proof": rows[-1]["field_evidence"]["cfo"]["content_sha256"] = "wrong"
    elif damage == "scope": rows[-2]["scope"] = "individual"
    elif damage == "basis": rows[-1]["revision_basis_id"] = "UNRECONCILED_REVISION"
    else: rows.pop()
    q = evaluate_financials(rows, symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=CUTOFF,
         safety_receipt=GateReceipt.bind(receipt("SAFETY", {}), gate_id="SAFETY", inputs={}, cutoff=CUTOFF), policy=POLICY)
    assert q["status"] == "DATA_INSUFFICIENT" and q["quality_score"] is None


@pytest.mark.parametrize("count", [1, 2])
def test_capex_gap_scope_and_no_fcf_imputation(count):
    rows = filings()
    for row in rows[-count:]: row["capex"] = None
    q = evaluate_financials(rows, symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=CUTOFF,
        safety_receipt=GateReceipt.bind(receipt("SAFETY", {}), gate_id="SAFETY", inputs={}, cutoff=CUTOFF), policy=POLICY)
    assert q["fcf_ttm"] is None and q["capex_ttm"] is None
    assert q["status"] == ("PASS" if count == 1 else "PROVISIONAL")
    if count == 1: assert q["missing_penalty_points"] == 3


def test_cli_registry_shared_entry_and_missing_artifact_parity(tmp_path, capsys):
    from interface.les_v7 import main
    # Explicit R1 keeps historical CLI/API/plugin parity. New native calls
    # without a frozen bundle/policy deliberately opt into registered R3.
    request = dict(bars=bars().to_dict("records"), context=dict(symbol="SYNTHETIC_ONLY", market="TWSE", as_of=CUTOFF,
                                                               filings=[], quality_policy=POLICY))
    expected = analyze_request(request)
    path = tmp_path / "input.json"; path.write_text(json.dumps(request))
    output = tmp_path / "output.json"
    assert main([str(path), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert json.loads(output.read_text()) == expected
    envelope = expected["reports"][0]["envelope"]
    assert envelope["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"
    assert envelope["score"]["final_score"] is None and envelope["execution"]["live_enabled"] is False
    assert get_strategy("low_entry_score_v7").version == "7.0.0"
    history = attach_low_entry_v7_context(pd.DataFrame(request["bars"]), **request["context"])
    result = get_strategy("low_entry_score_v7").run_with_features(history, SharedFeaturePipeline().build(history))
    assert result.iloc[-1]["LES v7 Report"] == expected["reports"][0]
    assert result["LES v7 Report"].iloc[:-1].isna().all()


def test_offline_heads_are_deterministic_and_do_not_self_calibrate():
    rows = [dict(features=[i / 10], log_bottom_ratio=-.02, bottom_timing=i % 21, fill=int(i % 3 != 0),
                 rebound=int(i % 2) if i % 3 else None, joint=int(i % 2) if i % 3 else None, cutoff="2025-01-01T13:30:00+08:00",
                 feature_fit_cutoff="2024-12-01T00:00:00Z", label_available_at="2025-02-01T00:00:00Z") for i in range(12)]
    a = fit_heads(rows, fit_cutoff="2025-03-01T00:00:00Z", iterations=20)
    assert a == fit_heads(rows, fit_cutoff="2025-03-01T00:00:00Z", iterations=20)
    assert a["calibration_status"] == "NOT_PROVIDED" and a["approved_for_live"] is False
    rows[0]["label_available_at"] = "2025-04-01T00:00:00Z"
    with pytest.raises(ValueError, match="FUTURE_LABEL"): fit_heads(rows, fit_cutoff="2025-03-01T00:00:00Z")


def test_matured_labels_censoring_no_fill_and_total_risk_units():
    dates = pd.bdate_range("2026-10-12", periods=21)
    path = [dict(session=d.date().isoformat(), low=99., complete=True, coordinate_id="SYNTHETIC_ONLY",
                 available_at=d.date().isoformat() + "T13:31:00+08:00") for d in dates]
    sessions = [row["session"] for row in path]
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL", 100., 98., 108., 2., (100., 100.5), 100, .3, .3)
    assert label_event(path, cutoff=CUTOFF)["status"] == "CALENDAR_NOT_PROVIDED"
    assert label_event(path[:-1], cutoff=CUTOFF, expected_sessions=sessions)["status"] == "RIGHT_CENSORED"
    execution = dict(path_evidence_ids=["SYNTHETIC_ONLY"], path_available_at=path[-1]["available_at"],
                     cost_policy_version="SYNTHETIC_ONLY", realized_at="2026-10-14T10:00:00+08:00",
                     plan_id=plan.plan_id, policy_version=plan.policy_version, coordinate_id="SYNTHETIC_ONLY")
    result = label_event(path, cutoff=CUTOFF, expected_sessions=sessions, plan=plan,
                        fill=dict(policy_id="REFERENCE_H20", filled=False, **execution))
    assert result["timing"] == 0 and result["fill"] == 0 and result["rebound"] is None and result["outcome_r"] is None
    fill = dict(policy_id="REFERENCE_H20", filled=True, entry=100., initial_stop=98.,
                cutoff_atr=2., shares=100, filled_at="2026-10-13T09:00:00+08:00", path_order_verified=True,
                net_2r_first_hit_at="2026-10-14T10:00:00+08:00", initial_stop_first_hit_at=None, realized_net_pnl=400., **execution)
    result = label_event(path, cutoff=CUTOFF, expected_sessions=sessions, fill=fill, plan=plan)
    assert result["outcome_r"] == 2 and result["normalized_utility"] == 1 and result["joint"] == 1
    fill["initial_stop_first_hit_at"] = fill["net_2r_first_hit_at"]
    assert label_event(path, cutoff=CUTOFF, expected_sessions=sessions, fill=fill, plan=plan)["status"] == "PATH_AMBIGUOUS"
    fill["plan_id"] = "0" * 64
    with pytest.raises(ValueError, match="SEED_MISMATCH"):
        label_event(path, cutoff=CUTOFF, expected_sessions=sessions, fill=fill, plan=plan)


def test_stage_c_rejects_in_sample_and_uses_weighted_mean_huber():
    rows = [dict(factors=[i / 5] * 6, utility=i / 5, context="REVERSAL", prediction_kind="STRICTLY_PAST_OOF",
                 head_last_label_available_at="2024-01-01T00:00:00Z", head_fit_cutoff="2024-01-02T00:00:00Z",
                 cutoff="2024-02-01T00:00:00Z", label_available_at="2024-03-01T00:00:00Z") for i in range(6)]
    result = fit_factor_weights(rows, fit_cutoff="2024-04-01T00:00:00Z")
    duplicate = fit_factor_weights(rows * 2, fit_cutoff="2024-04-01T00:00:00Z")
    assert result["weights"] == duplicate["weights"] and result["loss_normalization"] == "WEIGHTED_MEAN_STANDARD_HUBER"
    rows[0]["prediction_kind"] = "IN_SAMPLE"
    with pytest.raises(ValueError, match="OOF"): fit_factor_weights(rows, fit_cutoff="2024-04-01T00:00:00Z")


def complete_request():
    """Synthetic upstream validation receipts, NOT official or live approval."""
    request = dict(bars=bars(90).to_dict("records"), context=dict(symbol="SYNTHETIC_ONLY", market="TWSE",
        as_of=CUTOFF, filings=filings(), quality_policy=deepcopy(POLICY), model_bundle=bundle(), receipts={}))
    report = analyze_request(request)["reports"][0]
    for gate in ("SAFETY", "DATA_INTEGRITY", "TRADING_STATUS"):
        request["context"]["receipts"][gate] = receipt(gate, report["diagnostics"]["gate_bindings"][gate])
    report = analyze_request(request)["reports"][0]
    for gate in ("VALUATION", "STRESS_RISK"):
        request["context"]["receipts"][gate] = receipt(gate, report["diagnostics"]["gate_bindings"][gate])
    return request


def test_complete_synthetic_native_pipeline_cli_parity_and_prefix_invariance(tmp_path, capsys):
    from interface.les_v7 import main
    request = complete_request()
    expected = analyze_request(request)
    report = expected["reports"][0]
    envelope = report["envelope"]
    assert envelope["event"]["context"] == "REVERSAL"
    assert envelope["quality"]["status"] == "PASS"
    assert envelope["localization"]["status"] == "VALID"
    assert len(envelope["score"]["factor_values"]) == 6
    assert envelope["score"]["raw_score"] == pytest.approx(100 * np.dot(PRIOR, envelope["score"]["factor_values"]))
    assert envelope["score"]["final_score"] is not None
    assert envelope["execution"]["allocated_shares"] == 0
    assert envelope["execution"]["authorization_verified"] is False
    assert all(row["raw_value"] is not None or row["value_status"] == "OBSERVED"
               for row in report["diagnostics"]["quality"]["metric_ledger"])
    path = tmp_path / "request.json"; path.write_text(json.dumps(request))
    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    original = deepcopy(request)
    request["bars"].append(dict(request["bars"][-1], Date="2026-10-13", Close=1., Low=.5))
    request["context"]["filings"].append(dict(filings()[-1], available_at="2026-10-13T00:00:00+08:00", cfo=-1e9))
    assert analyze_request(request) == expected
    assert original == complete_request()


def test_bound_safety_receipts_are_not_user_pass_flags():
    request = complete_request()
    request["context"]["receipts"]["TRADING_STATUS"]["input_hash"] = "0" * 64
    with pytest.raises(ValueError, match="BINDING_INVALID"): analyze_request(request)
    request = complete_request()
    request["context"]["receipts"].pop("TRADING_STATUS")
    envelope = analyze_request(request)["reports"][0]["envelope"]
    assert envelope["score"]["final_score"] is None
    assert envelope["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"
    request = complete_request()
    request["context"]["receipts"]["TRADING_STATUS"]["state"] = "FAIL"
    assert analyze_request(request)["reports"][0]["envelope"]["strategy"]["strategy_decision"] == "NO_BUY"


@pytest.mark.parametrize("damage", ["coefficient", "calibration", "bounds", "imputation"])
def test_bundle_rejects_coercible_but_invalid_numbers(damage):
    raw = bundle()
    if damage == "coefficient": raw["payload"]["heads"]["fill"][0] = True
    elif damage == "calibration": raw["payload"]["calibration"]["projection_validated"] = "false"
    elif damage == "bounds": raw["payload"]["feature_bounds"] = [[200., 1.]]
    else: raw["payload"]["imputation"] = [201.]
    with pytest.raises(ValueError): FrozenBundle.load(seal(raw))


def test_forecast_requires_complete_calibrated_probabilities():
    with pytest.raises(ValueError, match="COMPLETE"): Forecast("VALID")
    with pytest.raises(ValueError, match="SUM_ONE"):
        Forecast("VALID", (98., 99., 100.), (0.,) * 21, .8, .8, .7, calibration_version="SYNTHETIC_ONLY")
    with pytest.raises(ValueError, match="EXCEEDS"):
        Forecast("VALID", (98., 99., 100.), (1.,) + (0.,) * 20, .8, .2, .7, calibration_version="SYNTHETIC_ONLY")


def test_unconfigured_economic_denominator_is_not_a_filing_gap():
    rows = filings()
    for row in rows:
        row["cost_of_revenue"] = 0.
        row["field_evidence"]["cost_of_revenue"]["contexts"][0]["numeric_value"] = 0.
    q = evaluate_financials(rows, symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=CUTOFF,
        safety_receipt=GateReceipt.bind(receipt("SAFETY", {}), gate_id="SAFETY", inputs={}, cutoff=CUTOFF), policy=POLICY)
    assert q["status"] == "CONFIG_INCOMPLETE" and q["quality_score"] is None
    assert "DSO_DIO_ECONOMIC_BRANCH_NOT_CONFIGURED" in q["reason_codes"]


def test_immutable_signal_repository_idempotence_detached_read_and_conflict(tmp_path):
    import sqlite3
    from infrastructure.storage.les_v7_repository import LesV7SignalRepository
    report = analyze_request(complete_request())["reports"][0]
    repository = LesV7SignalRepository(tmp_path)
    published = repository.publish(report)
    assert repository.publish(report) == published
    loaded = repository.load(published["signal_id"])
    assert loaded == report
    loaded["envelope"]["score"]["factor_values"][0] = 0.
    assert repository.load(published["signal_id"]) == report
    changed = deepcopy(report); changed["diagnostics"]["unexpected_change"] = True
    with pytest.raises(ValueError, match="NONDETERMINISTIC_REPLAY"): repository.publish(changed)
    with sqlite3.connect(tmp_path / "les_v7_signal_index.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 1


@pytest.mark.parametrize("damage", ["symbol", "model", "plan", "quality", "quality_summary", "forecast_summary", "score", "execution"])
def test_immutable_repository_rejects_report_binding_tampering(tmp_path, damage):
    from infrastructure.storage.les_v7_repository import LesV7SignalRepository
    report = analyze_request(complete_request())["reports"][0]
    if damage == "symbol": report["symbol"] = "OTHER"
    elif damage == "model": report["model_version"] = "6.0.0"
    elif damage == "plan": report["diagnostics"]["reference_plan"]["entry"] += 1
    elif damage == "quality": report["diagnostics"]["quality"]["quality_score"] += 1
    elif damage == "quality_summary": report["envelope"]["quality"]["quality_score"] += 1
    elif damage == "forecast_summary": report["envelope"]["localization"]["p_fill"] = 1.
    elif damage == "score": report["envelope"]["score"]["raw_score"] += 1
    else: report["envelope"]["execution"]["live_enabled"] = True
    with pytest.raises(ValueError): LesV7SignalRepository(tmp_path).publish(report)
    assert not tmp_path.joinpath("les_v7_signal_index.sqlite3").exists()


def test_signal_index_rollback_and_resume(tmp_path, monkeypatch):
    import sqlite3
    from infrastructure.storage.les_v7_repository import LesV7SignalRepository
    from infrastructure.storage.report_evidence import ReportEvidenceWriter
    report = analyze_request(complete_request())["reports"][0]
    repository = LesV7SignalRepository(tmp_path)
    original = ReportEvidenceWriter.retain
    def interrupted(self, value):
        original(self, value)
        raise OSError("SYNTHETIC INTERRUPTION")
    monkeypatch.setattr(ReportEvidenceWriter, "retain", interrupted)
    with pytest.raises(OSError): repository.publish(report)
    with sqlite3.connect(tmp_path / "les_v7_signal_index.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == 0
    monkeypatch.setattr(ReportEvidenceWriter, "retain", original)
    receipt = repository.publish(report)
    assert repository.load(receipt["signal_id"]) == report


def test_batched_requests_reject_conflicting_inputs():
    request = complete_request()
    with pytest.raises(ValueError, match="DUPLICATE_SECURITY_CUTOFF"):
        analyze_request(dict(securities=[request, deepcopy(request)]))
    with pytest.raises(ValueError, match="CONFLICTING_MODEL_BUNDLES"):
        analyze_request(dict(request, model_bundle={}))


def test_source_envelope_shape_and_packaged_entry_contract():
    from pathlib import Path
    import tomllib
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "docs/les-v7-signal-envelope.schema.json").read_text())
    envelope = analyze_request(complete_request())["reports"][0]["envelope"]
    assert set(envelope) == set(schema["required"])
    for section, definition in schema["properties"].items():
        value = envelope[section]
        if "const" in definition: assert value == definition["const"]
        if definition.get("type") != "object": continue
        assert set(value) == set(definition["required"])
        for key, sub in definition["properties"].items():
            if "enum" in sub: assert value[key] in sub["enum"]
            if "const" in sub: assert value[key] == sub["const"]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert "domain.les_v7" in config["tool"]["setuptools"]["packages"]
    assert config["project"]["scripts"]["aistock-les-v7"] == "interface.les_v7:main"


def test_atomic_same_signal_concurrent_publication(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from infrastructure.storage.les_v7_repository import LesV7SignalRepository
    report = analyze_request(complete_request())["reports"][0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: LesV7SignalRepository(tmp_path).publish(report), range(2)))
    assert results[0] == results[1]


def test_stored_content_tamper_is_not_a_cache_hit(tmp_path):
    import sqlite3
    from infrastructure.storage.les_v7_repository import LesV7SignalRepository
    repository = LesV7SignalRepository(tmp_path)
    report = analyze_request(complete_request())["reports"][0]
    saved = repository.publish(report)
    with sqlite3.connect(tmp_path / "les_v7_signal_index.sqlite3") as conn:
        conn.execute("UPDATE signals SET report_hash=?", ("0" * 64,))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): repository.load(saved["signal_id"])


def test_cli_store_missing_model_is_immutable_and_still_non_executable(tmp_path, capsys):
    from interface.les_v7 import main
    path = tmp_path / "request.json"
    path.write_text(json.dumps(dict(bars=[], context=dict(symbol="SYNTHETIC_ONLY", market="TWSE", as_of=CUTOFF))))
    output, store = tmp_path / "result.json", tmp_path / "research"
    assert main([str(path), "--output", str(output), "--store", str(store)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main([str(path), "--output", str(output), "--store", str(store)]) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert first["reports"][0]["envelope"]["score"]["final_score"] is None
    assert first["reports"][0]["envelope"]["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"


@pytest.mark.parametrize("cutoff,expected", [("2026-10-12T08:00:00+08:00", "2026-10-09"),
    ("2026-10-12T13:29:59+08:00", "2026-10-09"), (CUTOFF, "2026-10-12"),
    ("2026-10-11T13:30:00+08:00", "2026-10-09")])
def test_daily_close_cutoff_boundary_is_explicit(cutoff, expected):
    selected = select_bars(bars(), cutoff)
    assert selected.Date.iloc[-1] == expected
    _, audit = compute_features(selected)
    assert audit["cutoff_session"] == expected


@pytest.mark.parametrize("damage", ["identity", "boolean", "complete", "naive"])
def test_invalid_raw_market_or_clock_is_rejected(damage):
    request = complete_request()
    if damage == "identity": request["bars"][-1]["Symbol"] = "OTHER"
    elif damage == "boolean": request["bars"][-1]["Volume"] = True
    elif damage == "complete": request["bars"][-1]["Complete"] = 1
    else: request["context"]["as_of"] = "2026-10-12T13:30:00"
    with pytest.raises(ValueError): analyze_request(request)


def test_known_negative_capex_does_not_become_approved_missing_gap():
    rows = filings()
    rows[-1]["capex"] = -1000.
    rows[-1]["field_evidence"]["capex"]["contexts"][0]["numeric_value"] = -1e6
    q = evaluate_financials(rows, symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=CUTOFF,
        safety_receipt=GateReceipt.bind(receipt("SAFETY", {}), gate_id="SAFETY", inputs={}, cutoff=CUTOFF), policy=POLICY)
    assert q["status"] == "CONFIG_INCOMPLETE" and q["quality_score"] is None
    assert "CAPEX_SIGN_MAPPING_INVALID_NOT_BOUNDED_GAP" in q["reason_codes"]


def test_partial_benchmark_is_unknown_not_invalid_json_or_neutral_rs():
    request = complete_request()
    request["context"]["receipts"] = {}
    benchmark = [{"Date": row["Date"], "Close": 10000.} for row in request["bars"][:-1]]
    request["taiex_bars"] = benchmark
    report = analyze_request(request)["reports"][0]
    assert report["diagnostics"]["features"]["rs20"] is None
    assert report["envelope"]["score"]["factor_values"] is None
    assert report["envelope"]["score"]["final_score"] is None
    assert report["envelope"]["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"
    with pytest.raises(ValueError, match="CANNOT_READ"):
        rule(dict(op="gt", field="close", other_field="p_entry"), dict(close=100., p_entry=.7))


def test_market_availability_requires_timezone_and_preserves_dataframe_json_parity():
    data = bars(90)
    data["Available At"] = pd.to_datetime(data.Date).dt.tz_localize("Asia/Taipei") + pd.Timedelta(hours=13, minutes=30)
    request = complete_request()
    request["context"]["receipts"] = {}
    report = get_strategy("low_entry_score_v7").evaluate(data, context=request["context"])
    serialized = data.copy()
    serialized["Available At"] = serialized["Available At"].map(lambda v: v.isoformat())
    request["bars"] = serialized.to_dict("records")
    assert analyze_request(request)["reports"][0] == report
    data["Available At"] = pd.to_datetime(data.Date) + pd.Timedelta(hours=13, minutes=30)
    with pytest.raises(ValueError, match="TIMEZONE_REQUIRED"): select_bars(data, CUTOFF)
