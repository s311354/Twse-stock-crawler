"""SYNTHETIC_TEST_ONLY: solver/evidence contracts, not real return validation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math

import numpy as np
import pandas as pd
import pytest

from application.les_weight_estimation import EXECUTION_FIELDS, estimate_weights, outcome_in_r
from domain.strategies.low_entry_score_v6 import LowEntryScoreV6HybridStrategy, assess_execution, attach_low_entry_v6_context, parameter_version
from domain.strategies.low_entry_v6_hybrid import NO_CLEAR_CONTEXT, PARAMETER_VERSION, score_factors, select_context
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.strategies.low_entry_v6_signals import decide, size_position, trade_plan
from domain.strategies.low_entry_v6_weights import (
    CONTEXTS, FACTORS, PRIOR, WEIGHT_KEYS, FrozenWeights, digest, fit_convex_weights,
    normalized_factors, objective_gradient, project_simplex, timestamp,
)
from infrastructure.report.low_entry_v6_renderer import setup_score_summary
from interface.les_v6 import analyze_request, main as analyze_main
from interface.les_v6_research import main as fit_main
from tests.strategies.test_low_entry_score_v6 import ACCOUNT, bars
from tests.strategies.test_low_entry_v6_hybrid import CONFIG, report_for, strong
from tests.strategies.test_low_entry_v6_outcomes import observation, packet
from tests.strategies.test_low_entry_v6_unified_decision import verified_quality


def study(count=36, holdout=0):
    source = packet(count, holdout)
    source["versions"]["parameters"] = PARAMETER_VERSION
    source["weight_estimation"] = dict(frozen_at="2026-10-02T13:30:00+08:00")
    source["signal_reports"] = {}
    for i, row in enumerate(source["outcomes"]):
        report = report_for(strong(CONTEXTS[i % 3], rsi_change=3. + (i % 3)))
        report.update(symbol=row["symbol"], decision_time=row["signal_at"],
            signal_date=timestamp(row["signal_at"]).astimezone(tz=timestamp("2026-01-01T00:00:00+08:00").tzinfo).date().isoformat(),
            market=dict(exchange="TWSE", regime="NEUTRAL"))
        row.update(versions=dict(source["versions"]), setup=report["decision"]["setup"],
            entry_decision=report["decision"]["signal"], formal_score=report["decision"]["formal_score"],
            initial_stop=report["entry"]["stop_price"], research_result_sha256=digest(report))
        source["signal_reports"][digest(report)] = report
        row["execution_evidence"] = {key: dict(status="ASSESSED", source_id="SYNTHETIC_TEST_ONLY",
            source_sha256="c"*64, detail="synthetic contract test, not a real fill", available_at=row["exit_at"])
            for key in EXECUTION_FIELDS}
        row["execution_evidence"]["next_session"]["trade_date"] = row["entry_at"][:10]
        row["execution_evidence"]["exit_protocol"]["policy_version"] = source["versions"]["execution_policy"]
    return source


@pytest.fixture(scope="module")
def artifact():
    result = estimate_weights(study())
    assert result["status"] == "ESTIMATED_RESEARCH_ONLY", result["exclusion_counts"]
    return result["artifact"]


def apply(artifact, f=None, q=None, **kwargs):
    return decide(q or verified_quality(), strong() if f is None else f, config=CONFIG,
        convex_weights=artifact, as_of="2026-10-03T13:30:00+08:00", **kwargs)


def weighted_report(artifact, f=None):
    report = report_for(f)
    d = apply(artifact, f)
    report.update(decision_time="2026-10-03T13:30:00+08:00", signal_date="2026-10-02",
        parameter_version=parameter_version(CONFIG, artifact),
        decision={k: v for k, v in d.items() if k not in {"plan", "setups"}},
        setups=dict(selection=d["selection"], selected_setup=d["setup"]))
    return report


def test_simplex_projection_and_immutable_weights(artifact):
    result = project_simplex(np.array([[1., 0, 0, 0, 0, 0], [2., -1, 1., 0, 0, 0], [0.]*6]))
    assert result[0].tolist() == [1., 0, 0, 0, 0, 0]
    assert result[1].tolist() == [1., 0, 0, 0, 0, 0]
    assert result[2] == pytest.approx(np.ones(6)/6)
    frozen = FrozenWeights.from_record(artifact)
    with pytest.raises(FrozenInstanceError):
        frozen.sha256 = "bad"
    detached = frozen.record(); detached["weights"]["GLOBAL"][0] = 100
    assert frozen.record() == artifact
    assert frozen.for_context(NO_CLEAR_CONTEXT)[0] == "GLOBAL"


def test_huber_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    x, y, weights = rng.random((12,6)), rng.random(12), project_simplex(rng.random((4,6)))
    contexts, alpha = np.arange(12)%3, np.linspace(.1, 2, 12)
    def objective(w):
        return objective_gradient(w, x, y, contexts, alpha, delta=.12, pooling=.7, prior_strength=.3)
    gradient = objective(weights)[1]
    for index in np.ndindex(weights.shape):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        assert gradient[index] == pytest.approx((objective(plus)[0]-objective(minus)[0])/2e-6, abs=1e-8)


def test_convex_solution_feasibility_convergence_and_shuffle_determinism():
    rng = np.random.default_rng(10)
    x = rng.random((90,6)); y = x @ np.array([.4,.1,.1,.1,.2,.1])
    contexts = list(CONTEXTS)*30
    result = fit_convex_weights(x, y, contexts)
    assert result["status"] == "CONVERGED" and result["convex_gap"] <= 1e-8
    assert result["objective"] < result["initial_objective"]
    for w in result["weights"].values():
        assert min(w) >= 0 and sum(w) == pytest.approx(1.)
    order = rng.permutation(90)
    shuffled = fit_convex_weights(x[order], y[order], [contexts[i] for i in order])
    assert result == shuffled


def test_convex_known_optimum_and_pooling_prior_shrinkage():
    x = np.tile(np.eye(6), (10,1)); target = np.array([.4,.2,.1,.1,.1,.1]); y=x@target
    unpooled = fit_convex_weights(x,y,["REVERSAL"]*60,pooling=0,prior_strength=0)
    assert unpooled["weights"]["REVERSAL"] == pytest.approx(target, abs=1e-7)
    pooled = fit_convex_weights(x,y,["REVERSAL"]*60,pooling=2,prior_strength=2)
    global_w = np.array(pooled["weights"]["GLOBAL"])
    assert np.linalg.norm(global_w-PRIOR) < np.linalg.norm(target-PRIOR)
    assert pooled["weights"]["TREND_PULLBACK"] == pytest.approx(global_w, abs=1e-7)


@pytest.mark.parametrize("kwargs", [dict(delta=0), dict(pooling=-1), dict(prior_strength=-1),
    dict(tolerance=math.nan), dict(max_iterations=True), dict(alpha=[-1]*6), dict(alpha=[0]*6)])
def test_invalid_fit_parameters_and_nonfinite_inputs(kwargs):
    with pytest.raises(ValueError):
        fit_convex_weights(np.eye(6), np.ones(6), ["REVERSAL"]*6, **kwargs)
    with pytest.raises(ValueError):
        fit_convex_weights(np.full((1,6), math.inf), [1], ["REVERSAL"])


@pytest.mark.parametrize("risk_units,label", [(-2,0),(-1,0),(0,1/3),(1,2/3),(2,1),(3,1)])
def test_net_cash_outcome_r_units_and_clipping(risk_units,label):
    row = dict(quantity=100, entry_notional=10000., initial_stop=98., entry_cost=15., exit_cost=45.,
        net_cash_dividends=0., exit_notional=10000+60+risk_units*200)
    r, y = outcome_in_r(row)
    assert r == pytest.approx(risk_units) and y == pytest.approx(label)
    assert outcome_in_r(dict(row, net_cash_dividends=100))[0] == pytest.approx(risk_units+.5)
    with pytest.raises(ValueError):
        outcome_in_r(dict(row, initial_stop=100))


def test_fit_has_explicit_provenance_and_research_only_authority(artifact):
    source = study(36,21); original = deepcopy(source)
    result = estimate_weights(source)
    assert source == original
    assert result["training_count"] == 36 and result["holdout_count"] == 21
    assert result["holdout_status"] == "DIAGNOSTICS_AVAILABLE_NOT_VALIDATED"
    assert result["research_only"] and not result["executable"] and not result["kelly_calibration_eligible"]
    assert not result["performance_validated"]
    assert set(artifact["weights"]) == set(WEIGHT_KEYS)
    assert artifact["source_kind"] == "SYNTHETIC_TEST_ONLY"
    assert FrozenWeights.from_record(artifact).record() == artifact


def test_holdout_labels_future_labels_and_shuffle_do_not_change_fitted_artifact():
    source = study(36,21); baseline = estimate_weights(source)
    for row in source["outcomes"][36:]:
        row["exit_notional"] = 1000000.
    future = observation(99, versions={"model":"future"})
    future.update(exit_at="2027-01-01T00:00:00+08:00",available_at="2027-01-01T00:00:00+08:00")
    source["outcomes"].append(future)
    result = estimate_weights(source)
    assert result["artifact"] == baseline["artifact"]
    assert result["holdout_huber"] != baseline["holdout_huber"]
    assert result["exclusion_counts"]["OUTCOME_UNAVAILABLE_AT_CUTOFF"] == 1
    source["outcomes"].reverse()
    assert estimate_weights(source)["artifact"] == result["artifact"]


@pytest.mark.parametrize("mutation,reason", [
    (lambda r:r.update(initial_stop=97.), "INITIAL_STOP_NOT_BOUND"),
    (lambda r:r["execution_evidence"].pop("price_limits"), "INCOMPLETE_EXECUTION_EVIDENCE"),
    (lambda r:r["execution_evidence"]["next_session"].update(trade_date="2026-01-01"), "NEXT_SESSION"),
    (lambda r:r.update(observation_weight=-1), "INVALID_OBSERVATION_WEIGHT"),
    (lambda r:r.update(research_result_sha256="f"*64), "SIGNAL_REPORT_HASH_MISMATCH"),
])
def test_missing_or_unbound_execution_is_excluded_not_synthesized(mutation,reason):
    source=study(30)
    mutation(source["outcomes"][0])
    result=estimate_weights(source)
    assert result["status"] == "UNKNOWN" and result["artifact"] is None
    assert result["training_count"] == 29
    assert any(reason in item["reason"] for item in result["exclusions"])


def test_report_factor_tamper_and_wrong_company_rejected():
    for key,value in (("symbol","wrong"),("decision_time","2026-02-01T13:30:00+08:00")):
        source=study(30); raw=source["outcomes"][0]
        report=source["signal_reports"].pop(raw["research_result_sha256"])
        report[key]=value
        raw["research_result_sha256"]=digest(report); source["signal_reports"][digest(report)]=report
        result=estimate_weights(source)
        assert result["artifact"] is None and result["exclusion_counts"]["OUTCOME_SIGNAL_IDENTITY_MISMATCH"] == 1
    source=study(30); raw=source["outcomes"][0]
    report=source["signal_reports"].pop(raw["research_result_sha256"])
    report["decision"]["budget_groups"]["trend"]["earned"] = 0.
    raw["research_result_sha256"]=digest(report); source["signal_reports"][digest(report)]=report
    assert estimate_weights(source)["artifact"] is None


def test_purging_incomplete_samples_backdating_and_no_convergence():
    source=study(30)
    source["outcomes"][0]["available_at"] = source["validation_start"]
    result=estimate_weights(source)
    assert result["artifact"] is None and result["exclusion_counts"]["PURGED_LABEL_CROSSES_VALIDATION_BOUNDARY"] == 1
    source=study(); source["weight_estimation"]["max_iterations"]=1
    assert estimate_weights(source)["reasons"] == ["OPTIMIZER_NOT_CONVERGED"]
    source["weight_estimation"]["frozen_at"]="2026-01-01T00:00:00+08:00"
    with pytest.raises(ValueError,match="BACKDATED"):
        estimate_weights(source)
    assert estimate_weights(study(0))["artifact"] is None


@pytest.mark.parametrize("context", [*CONTEXTS,NO_CLEAR_CONTEXT])
def test_one_context_vector_weight_raw_score_and_native_caps(artifact,context):
    f=strong(context, rsi_change=2.5, clv=.4)
    original=deepcopy(f); d=apply(artifact,f)
    w=d["weighting"]
    assert w["applied_context"] == (context if context in CONTEXTS else "GLOBAL")
    assert d["setup"] == select_context(f).context == context
    assert d["raw_score"] == pytest.approx(100*np.dot(w["applied_weights"],w["normalized_vector"]))
    assert sum(g["budget"] for g in d["budget_groups"].values()) == pytest.approx(100.)
    assert "setup_decisions" not in d and f==original
    native = decide(verified_quality(), f, config=CONFIG)
    assert not d["research_only"] and not d["performance_validated"]
    assert d["execution_policy"] == "NATIVE_GATES" and d["requires_execution_check"]
    assert d["risk_fraction"] == native["risk_fraction"] > 0
    assert d["position_multiplier"] == native["position_multiplier"] > 0
    assert apply(artifact,strong(context,restricted=True))["final_score"] == 0
    assert apply(artifact,strong(context,restricted=None))["safety_cap"] == 64


def test_missing_slots_fixed_normalization_never_redistributed(artifact):
    f=strong(); selection=select_context(f); plan=trade_plan(f,config=CONFIG,setup=selection.context)
    vector,coverage=normalized_factors(score_factors(f,selection,plan,config=CONFIG))
    assert vector == coverage == (1.,)*6
    partial=apply(artifact,dict(f,cmf20=None))
    full=apply(artifact,f)
    assert partial["weighting"]["normalized_vector"][3] < 1
    assert partial["weighting"]["applied_weights"] == full["weighting"]["applied_weights"]
    assert partial["raw_score"] <= full["raw_score"]
    assert apply(artifact,dict.fromkeys(f))["final_score"] is None


def test_zero_weight_allowed_no_fabricated_score_and_cost_rr_policy(artifact):
    source=deepcopy(artifact)
    source["weights"]["TREND_PULLBACK"]=[0.,0.,0.,0.,0.,1.]
    source["sha256"]=digest({k:v for k,v in source.items() if k!="sha256"})
    d=apply(source,strong(spread=None))
    assert d["raw_score"] == 90. and d["weighting"]["applied_weights"][-1] == 1
    assert d["signal"] not in {"BUY","BUY_SMALL"}
    discover=apply(artifact,strong("LEADERSHIP_CONTINUATION",swing_high=None,hh20=99.,hh60=99.))
    assert discover["plan"]["target_price"] is discover["plan"]["net_reward_risk"] is None
    assert discover["rr_policy"] == "PRICE_DISCOVERY_EXTENSION_SUPPORT"


def test_artifact_checksum_simplex_feature_version_time_and_config_rejected(artifact):
    bad=deepcopy(artifact); bad["weights"]["REVERSAL"][0]+=.1
    with pytest.raises(ValueError,match="INTEGRITY"):
        FrozenWeights.from_record(bad)
    bad["sha256"]=digest({k:v for k,v in bad.items() if k!="sha256"})
    with pytest.raises(ValueError,match="SIMPLEX"):
        FrozenWeights.from_record(bad)
    frozen=FrozenWeights.from_record(artifact)
    with pytest.raises(ValueError,match="UNAVAILABLE"):
        frozen.require_compatible(CONFIG,"2026-10-01T13:30:00+08:00")
    with pytest.raises(ValueError,match="TIMEZONE"):
        frozen.require_compatible(CONFIG,"2026-10-03")
    with pytest.raises(ValueError,match="CONFIG_MISMATCH"):
        frozen.require_compatible(replace(CONFIG,buy_fee_rate=.002),"2026-10-03T13:30:00+08:00")
    bad=deepcopy(artifact); bad["feature_model_sha256"]="f"*64
    bad["sha256"]=digest({k:v for k,v in bad.items() if k!="sha256"})
    with pytest.raises(ValueError,match="INCOMPATIBLE"):
        FrozenWeights.from_record(bad)


def execution_inputs():
    quote = dict(symbol="2330", timestamp="2026-10-05T09:10:00+08:00",
        price=100.5, bid=100.4, ask=100.5, restricted=False)
    args = dict(now="2026-10-05T09:10:30+08:00", trading_sessions=["2026-10-02", "2026-10-05"],
        portfolio=ACCOUNT)
    return quote, args


def test_report_recomputed_renderer_identity_and_native_execution_allowed(artifact):
    report=weighted_report(artifact)
    original = deepcopy(report)
    assert set(entry_gate_checks(report).values()) == {"PASS"}
    assert report_failure(report) is None
    quote, args = execution_inputs()
    execution = assess_execution(report, quote, **args)
    assert execution["action"] == "ELIGIBLE" and execution["shares"] > 0
    assert execution["is_fill"] is False and report == original
    shares = size_position(apply(artifact), portfolio_value=ACCOUNT["value"],
        available_cash=ACCOUNT["available_cash"], intended_weight=ACCOUNT["intended_weight"],
        portfolio_capacity_weight=ACCOUNT["capacity_weight"], amount_ma20=strong()["amount_ma20"], config=CONFIG)
    assert shares > 0
    rendered=setup_score_summary(report)
    assert "凍結凸估計權重" in rendered and "載入權重不額外阻擋" in rendered
    assert "績效尚未驗證" in rendered and "研究限定、不可執行" not in rendered
    assert artifact["sha256"][:12] in rendered
    report["decision"]["weighting"]["normalized_vector"][0]=.2
    assert entry_gate_checks(report)["score"] == "FAIL"


@pytest.mark.parametrize("changes", [dict(restricted=True), dict(restricted=None), dict(spread=.02),
    dict(support=80.), dict(amount_ma20=None), dict(ema5_slope=-1., ema20_slope=-1., atr_expansion=2., r5=-.2)])
def test_weighted_signals_do_not_bypass_native_caps_or_risk(artifact, changes):
    report = weighted_report(artifact, strong(**changes))
    assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}
    assert report["decision"]["risk_fraction"] == 0
    quote, args = execution_inputs()
    assert assess_execution(report, quote, **args)["action"] == "SKIP"


@pytest.mark.parametrize("changes,reason", [
    (dict(price=100.), "TRIGGER_NOT_REACHED"),
    (dict(timestamp="2026-10-05T08:00:00+08:00"), "STALE_OR_MISSING_QUOTE"),
    (dict(restricted=True), "INVALID_QUOTE_OR_RESTRICTION"),
    (dict(price=110.), "REPRICED_RISK_OR_LIQUIDITY_FAIL"),
    (dict(bid=99., ask=102.), "REPRICED_RISK_OR_LIQUIDITY_FAIL"),
])
def test_weighted_execution_still_requires_fresh_trigger_and_repriced_risk(artifact, changes, reason):
    quote, args = execution_inputs()
    assert assess_execution(weighted_report(artifact), dict(quote, **changes), **args)["reason"] == reason


def test_weighted_execution_rejects_account_quality_hard_risk_and_research_interventions(artifact):
    report = weighted_report(artifact)
    quote, args = execution_inputs()
    for portfolio in ({}, dict(ACCOUNT, available_cash=0), dict(ACCOUNT, capacity_weight=0)):
        assert assess_execution(report, quote, **dict(args, portfolio=portfolio))["reason"] == "ZERO_CAPACITY_OR_MISSING_PORTFOLIO_DATA"
    assert assess_execution(report, quote, **args, hard_risk=True)["action"] == "SKIP"
    report["quality"]["status"] = "DATA_INSUFFICIENT"
    assert assess_execution(report, quote, **args)["action"] == "SKIP"
    report = weighted_report(artifact)
    report["decision"]["research_only"] = True
    assert report_failure(report) == "RESEARCH_ONLY_SCORE_INTERVENTION"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(parameter_version=PARAMETER_VERSION),
    lambda r: r["decision"]["weighting"]["artifact"]["weights"]["GLOBAL"].__setitem__(0, .99),
    lambda r: r["decision"]["weighting"].pop("artifact"),
    lambda r: r["decision"].pop("weighting"),
])
def test_weighted_execution_checks_artifact_bound_version_before_quote(artifact, mutation):
    report = weighted_report(artifact)
    mutation(report)
    quote, args = execution_inputs()
    assert assess_execution(report, quote, **args)["reason"] == "PARAMETER_VERSION_MISMATCH"
    assert report_failure(report) is not None


def test_weighted_historical_replay_boundary_preserved_in_plugin_and_execution(artifact):
    frame = attach_low_entry_v6_context(bars(), symbol="2330", sector="ELEC",
        as_of="2026-10-03T13:30:00+08:00", filings=[], convex_weights=artifact)
    frame.attrs["les_v6_context"]["historical_replay"] = True
    generated = LowEntryScoreV6HybridStrategy(CONFIG).run_latest(frame)["LES v6 Report"].iloc[0]
    assert generated["evaluation_mode"] == "HISTORICAL_REPLAY"
    assert generated["weight_evaluation_mode"] == "AS_OF_SNAPSHOT"
    report = weighted_report(artifact)
    report["evaluation_mode"] = "HISTORICAL_REPLAY"
    assert report_failure(report) == "HISTORICAL_REPLAY_NOT_EXECUTABLE"
    assert report_failure(report, allow_replay=True) is None
    quote, args = execution_inputs()
    assert assess_execution(report, quote, **args)["reason"] == "HISTORICAL_REPLAY_NOT_EXECUTABLE"


def test_weighted_report_cannot_backdate_artifact_availability(artifact):
    report = weighted_report(artifact)
    report["decision_time"] = "2026-10-01T13:30:00+08:00"
    assert entry_gate_checks(report)["score"] == "UNKNOWN"
    assert report_failure(report) is not None
    quote, args = execution_inputs()
    assert assess_execution(report, quote, **args)["action"] == "SKIP"


def test_weighted_plugin_preserves_native_sizing_reason(artifact, monkeypatch):
    import domain.strategies.low_entry_score_v6 as module
    fields = [dict(strong(), pe=20., rsi14=40.) for _ in range(100)]
    fields[-1]["rsi14"] = 45.
    monkeypatch.setattr(module, "build_features", lambda data, features: pd.DataFrame(fields, index=data.index))
    monkeypatch.setattr(module, "evaluate_quality", lambda *args, **kwargs: verified_quality())
    def industry(data, dates, **kwargs):
        prices = pd.Series([110.] * (len(dates) - 1) + [100.], index=dates.index)
        return prices, {"status": "PASS", "source": "SYNTHETIC_TEST_ONLY"}
    monkeypatch.setattr(module, "aligned_index", industry)
    frame = attach_low_entry_v6_context(bars(), symbol="2330", sector="ELEC",
        as_of="2026-10-03T13:30:00+08:00", filings=[], convex_weights=artifact, portfolio=ACCOUNT)
    result = LowEntryScoreV6HybridStrategy(CONFIG).run_latest(frame)
    report = result["LES v6 Report"].iloc[0]
    assert report["decision"]["signal"] == "BUY"
    assert result["建議股數"].iloc[0] == report["sizing"]["indicative_shares"] > 0
    assert report["sizing"]["reason"] == "AWAIT_FRESH_EXECUTION_CHECK"


def test_selected_context_cannot_switch_to_best_weights(artifact):
    source=deepcopy(artifact)
    source["weights"]["TREND_PULLBACK"]=[0.,0.,0.,0.,1.,0.]
    source["weights"]["REVERSAL"]=[1.,0.,0.,0.,0.,0.]
    source["sha256"]=digest({k:v for k,v in source.items() if k!="sha256"})
    d=apply(source,strong(rsi_change=0.))
    assert d["selection"]["matches"] == ["TREND_PULLBACK","REVERSAL"]
    assert d["setup"] == "TREND_PULLBACK" and d["raw_score"] == 0.
    assert d["signal"] == "AVOID"


@pytest.mark.parametrize("momentum_weight,signal", [(.25,"BUY"),(.2500000001,"BUY_SMALL"),(.35,"BUY_SMALL"),(.3500000001,"WATCH")])
def test_convex_decision_uses_unrounded_final_not_display_score(artifact,momentum_weight,signal):
    source=deepcopy(artifact)
    source["weights"]["TREND_PULLBACK"]=[1-momentum_weight,0.,0.,0.,momentum_weight,0.]
    source["sha256"]=digest({k:v for k,v in source.items() if k!="sha256"})
    d=apply(source,strong(rsi_change=0.))
    assert d["raw_score"] == pytest.approx(100*(1-momentum_weight))
    assert d["signal"] == signal


def test_runtime_implementation_change_rejects_frozen_artifact(artifact,monkeypatch):
    import domain.strategies.low_entry_v6_weights as module
    monkeypatch.setattr(module,"implementation_signature",lambda:"changed")
    with pytest.raises(ValueError,match="INCOMPATIBLE"):
        FrozenWeights.from_record(artifact)


def test_future_duplicate_id_cannot_shadow_selected_training_evidence():
    source=study(30); baseline=estimate_weights(source)
    future=deepcopy(source["outcomes"][0])
    future.update(available_at="2027-01-01T13:30:00+08:00",exit_at="2027-01-01T13:30:00+08:00",
        versions={"model":"future"},initial_stop=1.)
    source["outcomes"].append(future)
    assert estimate_weights(source)["artifact"] == baseline["artifact"]


def test_cost_parameters_remain_pinned_and_utility_overflow_rejected(artifact):
    source=deepcopy(artifact); source["decision_config_sha256"]="f"*64
    source["sha256"]=digest({k:v for k,v in source.items() if k!="sha256"})
    with pytest.raises(ValueError,match="CONFIG_MISMATCH"):
        apply(source)
    with pytest.raises(ValueError,match="UTILITY"):
        outcome_in_r({},lower=1e308,upper=1e308)
    with pytest.raises(ValueError,match="NUMERIC_OVERFLOW"):
        fit_convex_weights(np.eye(6),np.ones(6),["REVERSAL"]*6,pooling=1e308)


def test_snapshot_rejects_naive_cutoff(artifact):
    frame=attach_low_entry_v6_context(bars(),symbol="2330",sector="ELEC",as_of="2026-10-03",filings=[],convex_weights=artifact)
    with pytest.raises(ValueError,match="TIMEZONE"):
        LowEntryScoreV6HybridStrategy(CONFIG).run_latest(frame)


def test_fit_cli_api_parity_and_apply_snapshot_native_plugin(artifact,tmp_path,capsys):
    source=study(); path=tmp_path/"outcomes.json"
    path.write_text(json.dumps(source),encoding="utf-8")
    assert fit_main([str(path),"--estimate-weights"]) == 0
    cli=json.loads(capsys.readouterr().out)
    assert cli == estimate_weights(source)
    data=bars(); payload=dict(score_policy="hybrid-legacy",convex_weights=artifact,
        bars=data.to_dict(orient="records"),context=dict(symbol="2330",sector="ELEC",
        as_of="2026-10-03T13:30:00+08:00",filings=[]))
    # Keep all numeric precision; only serialize exchange-local session dates.
    for record,date in zip(payload["bars"],data["Date"],strict=True):
        record["Date"]=str(date)[:10]
    result=analyze_request(payload)
    assert result["evaluation_mode"] == "CONVEX_WEIGHT_SNAPSHOT"
    assert result["execution_policy"] == "NATIVE_GATES" and result["requires_execution_check"]
    assert not result["research_only"]
    assert len(result["reports"]) == 1 and result["reports"][0]["sizing"]["indicative_shares"] == 0
    frame=attach_low_entry_v6_context(data,**payload["context"],convex_weights=artifact)
    direct=LowEntryScoreV6HybridStrategy(CONFIG).run_latest(frame)["LES v6 Report"].iloc[0]
    assert result["reports"][0] == direct
    weights=tmp_path/"weights.json"; weights.write_text(json.dumps(cli),encoding="utf-8")
    input_path=tmp_path/"input.json"; payload.pop("convex_weights")
    input_path.write_text(json.dumps(payload),encoding="utf-8")
    assert analyze_main([str(input_path),"--weights",str(weights)]) == 0
    assert json.loads(capsys.readouterr().out) == result
    with pytest.raises(ValueError,match="UNAVAILABLE"):
        LowEntryScoreV6HybridStrategy(CONFIG).run(frame)
    with pytest.raises(ValueError,match="EXPLICIT_HYBRID"):
        analyze_request(dict(payload,score_policy="unified",convex_weights=artifact))
