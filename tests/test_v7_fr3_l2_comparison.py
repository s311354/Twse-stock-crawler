"""SYNTHETIC routing/port tests only; no fitted model or efficacy claims.

Real source plans/gates and complete native report recomputation are retained
where stated. Forecast/financial doubles are explicitly inherited from the
existing FR3 public-port fixture, not evidence that an L2 model was trained.
"""
from copy import deepcopy

import pytest

from application import fr3_research as app
from domain.les_v7 import fr3_l2, fr3_plan, training
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.fr3_policy import (components, entry_policy, execution_policy, eps_policy,
    L2_ENTRY_VERSION, L2_EXECUTION_VERSION)
from examples import compare_fr3_study as study
from infrastructure.les_v7_fr3_comparison import (
    _compare_verified_inputs, compare_fr3_predictions, fr3_reference_versions,
)
from tests.test_v7_fr3_baseline_folds import archive, provider, bind_heads
from tests.test_v7_fr3_comparison import comparison_inputs, synthetic_fr3_report
from tests.test_v7_fr3_plan import reference
from tests.test_v7_fr3_research import research_request, allow_synthetic_gate_ports, native_producer
from tests.test_v7_fr3_study import routed


CREATED = "2026-10-24T05:00:00+00:00"  # Explicit SYNTHETIC policy declaration.


def l2_reference():
    return fr3_plan.policy(reference(), available_at=CREATED, version=fr3_l2.PLAN_VERSION)


def test_r1_default_component_identities_remain_exact():
    assert identity(entry_policy()) == "d7a4eda508efd559a140263422ab3f8a3d8662df5643bf6e2986f0cd6a7fed67"
    assert identity(execution_policy()) == "6b86c230e052c430d434cee1c3a21102e7751ba690ae016f6916b362cf707010"
    assert identity(eps_policy()) == "8514c10d24a2577e3cd906b1d695a19c0eb1539f2d8d10e47f566b7eea8345ec"
    assert identity(components()) == "16edbbb8f7c1e26fe2a1c5363b805f491cc033b1f8f812abeccfb6a40a955416"


def test_l2_components_bind_exact_native_structure_and_explicit_clock_without_loosening_safety():
    options = dict(reference_version=fr3_l2.PLAN_VERSION, available_at=CREATED)
    current = components(**options)
    assert current["entry"]["policy"]["version"] == L2_ENTRY_VERSION
    assert current["execution"]["policy"]["version"] == L2_EXECUTION_VERSION
    assert current["valuation"] == components()["valuation"]
    for name in ("entry", "execution"):
        row = current[name]
        assert row["component_id"] == identity(row["policy"])
        assert row["policy"]["reference_plan_version"] == fr3_l2.PLAN_VERSION
        assert row["policy"]["available_at"] == CREATED
        assert row["policy"]["structural_policy"] == fr3_l2.structural_policy()
        assert row["policy"]["structural_policy_id"] == identity(fr3_l2.structural_policy())
    for key in ("small_minimum", "minimum_quality_score", "valuation_warning_points", "maximum_stop_atr",
                "minimum_net_rr", "minimum_fill_probability", "minimum_bottom_soon_probability",
                "minimum_conditional_entry_probability", "unknown_blocks", "position_limits_affect_score"):
        assert current["entry"]["policy"][key] == entry_policy()[key]
    for key in ("total_symbol_cap", "valuation_warning_cap", "planned_symbol_nav_risk",
                "planned_portfolio_nav_risk", "maximum_holdings", "maximum_adv_fraction", "maximum_spread"):
        assert current["execution"]["policy"][key] == execution_policy()[key]
    current["entry"]["policy"]["structural_policy"]["leader_rule"]["maximum_open_opportunities"] = 100
    assert components(**options)["entry"]["policy"]["structural_policy"] == fr3_l2.structural_policy()
    with pytest.raises(ValueError, match="CREATION_TIME"):
        entry_policy(reference_version=fr3_l2.PLAN_VERSION)
    with pytest.raises(ValueError, match="UNSUPPORTED"):
        execution_policy(reference_version="UNRECOGNIZED", available_at=CREATED)


def l2_request(research_request):
    result = deepcopy(research_request)
    model = result["model_bundle"]["payload"]
    model["reference_policy"] = fr3_plan.policy(model["reference_policy"],
        available_at=CREATED, version=fr3_l2.PLAN_VERSION)
    result["model_bundle"]["sha256"] = identity(model)
    return result


def test_l2_research_reexecutes_new_bound_plan_preserves_r1_and_never_fills(research_request, monkeypatch):
    allow_synthetic_gate_ports(monkeypatch)
    old = app.analyze_fr3_request(research_request)
    request = l2_request(research_request)
    before = canonical(request)
    result = app.analyze_fr3_request(request)
    report = result["reports"][0]
    assert canonical(request) == before
    assert report["plan"]["policy_version"] == fr3_l2.PLAN_VERSION
    assert report["model_id"] != old["reports"][0]["model_id"]
    assert report["components"]["entry"]["policy"]["version"] == L2_ENTRY_VERSION
    assert report["decision"]["strategy_decision"] == "BUY_SMALL"
    assert report["decision"]["entry_policy_id"] == report["components"]["entry"]["component_id"]
    assert report["execution"]["execution_policy_id"] == report["components"]["execution"]["component_id"]
    assert not report["execution"]["fill_eligible"] and not report["execution"]["filled_shares"]
    assert not report["execution"]["live_order_authorized"]
    assert app.validate_fr3_report(report, request["securities"][0], request["model_bundle"], universe=request["universe"])
    assert app.analyze_fr3_request(research_request) == old
    damaged = deepcopy(report)
    damaged["components"] = components()
    damaged["report_id"] = identity({k:v for k,v in damaged.items() if k != "report_id"})
    with pytest.raises(ValueError, match="RECOMPUTATION"):
        app.validate_fr3_report(damaged, request["securities"][0], request["model_bundle"], universe=request["universe"])


def test_l2_high_score_does_not_replace_unknown_native_evidence(research_request, monkeypatch):
    request = l2_request(research_request)
    monkeypatch.setattr(app, "produce_native_receipts", native_producer)
    report = app.analyze_fr3_request(request)["reports"][0]
    assert report["score"]["final_score"] == pytest.approx(90.)
    assert report["decision"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert any(row["state"] == "UNKNOWN" for row in report["gate_receipts"])
    assert not report["execution"]["allocation_authorized"]


def l2_comparison():
    result = comparison_inputs()
    result.update(old_reference=reference(), new_reference=l2_reference())
    result["outcome"]["native_outcome"]["bundle_sha256"] = "SYNTHETIC_L2_MODEL"
    return result


def report_for(model, policy):
    result = synthetic_fr3_report()
    result["model_id"] = model
    result["components"] = components(reference_version=policy["version"], available_at=policy["available_at"])
    result["report_id"] = identity({k:v for k,v in result.items() if k != "report_id"})
    return result


def test_l2_comparison_keeps_complete_scope_distinct_versions_and_absent_decisions_unknown():
    source = l2_comparison()
    result = _compare_verified_inputs(**source)
    assert result["verified_decision_counts"] == {"FR3_R1": None, "FR3_L2": None}
    assert result["fixed_universe_count"] == 3
    assert len(result["stocks"]) == 3
    assert result["status"] == "NOT_ASSESSABLE"
    assert result["model_versions"]["FR3_R1"]["reference_plan_version"] == fr3_plan.PLAN_VERSION
    assert result["model_versions"]["FR3_L2"] == fr3_reference_versions(source["new_reference"])
    assert result["policy_version_changes_are_not_same_label_ablation"]
    assert "NOT_FRACTAL_ONLY_ABLATION" in result["comparison_kind"]
    assert result["after_cost_comparison"]["status"] == "NOT_ASSESSABLE"


def test_l2_reports_require_own_exact_model_and_reexecution_not_a_hash_only_buy():
    source = l2_comparison()
    old = report_for(source["original"]["sha256"], source["old_reference"])
    new = report_for(source["outcome"]["native_outcome"]["bundle_sha256"], source["new_reference"])
    unverified = _compare_verified_inputs(**source, old_reports=[old], new_reports=[new])
    assert unverified["verified_decision_counts"] == {"FR3_R1": None, "FR3_L2": None}
    assert unverified["stocks"][0]["old"]["decision_status"] == "UNVERIFIED"
    checked = []
    verified = _compare_verified_inputs(**source, old_reports=[old], new_reports=[new],
        old_report_validator=lambda r: checked.append(("R1", r["report_id"])),
        new_report_validator=lambda r: checked.append(("L2", r["report_id"])))
    assert len(checked) == 2
    assert verified["verified_decision_counts"] == {"FR3_R1": {"BUY_SMALL": 1}, "FR3_L2": {"BUY_SMALL": 1}}
    assert verified["status"] == "NOT_ASSESSABLE"  # Other stocks/evidence/accounts still missing.
    assert old["report_id"] != new["report_id"]
    assert verified["stocks"][0]["matched_cutoff_evidence_reports"][0]["decision_comparison"].endswith("NOT_SAME_PLAN_ABLATION")
    assert verified["stocks"][0]["matched_cutoff_evidence_reports"][0]["same_plan"] is False
    partial = _compare_verified_inputs(**source, old_reports=[old], new_reports=[new],
        new_report_validator=lambda _: None)
    assert partial["stocks"][0]["matched_cutoff_evidence_reports"][0]["verification"] == "UNVERIFIED"


@pytest.mark.parametrize("damage", ["model", "plan", "component", "missing_model", "same_version"])
def test_l2_comparison_rejects_cross_version_or_model_substitution(damage):
    source = l2_comparison()
    new = report_for(source["outcome"]["native_outcome"]["bundle_sha256"], source["new_reference"])
    if damage == "model": new["model_id"] = source["original"]["sha256"]
    elif damage == "plan": new["plan"] = {"policy_version": fr3_plan.PLAN_VERSION}
    elif damage == "component": new["components"] = components()
    elif damage == "missing_model": source["outcome"]["native_outcome"]["bundle_sha256"] = None
    else: source["old_reference"] = deepcopy(source["new_reference"])
    new["report_id"] = identity({k:v for k,v in new.items() if k != "report_id"})
    with pytest.raises(ValueError, match="FR3_L2_"):
        _compare_verified_inputs(**source, new_reports=[new], new_report_validator=lambda _: None)


def test_l2_study_uses_actual_plan_path_labels_not_r1_constants(routed):
    config = deepcopy(routed["config"])
    config["reference_policy"] = l2_reference()
    outcome, report, artifact = (deepcopy(routed[key]) for key in ("outcome", "report", "artifact"))
    outcome["native_outcome"]["config_id"] = identity(config)
    for value in (report, artifact): value["source_bindings"]["config_id"] = identity(config)
    report["reference_path_diagnostic"]["cost_path_policy"] = config["reference_policy"]["cost_path_policy"]
    forecasts, costs, rows = study._extract_recomputed(report, artifact, outcome, config)
    for value in (forecasts, costs, rows):
        assert value["reference_plan_version"] == fr3_l2.PLAN_VERSION
        assert value["path_version"] == fr3_l2.PATH_VERSION
        assert value["label_version"] == fr3_l2.LABEL_VERSION
    assert forecasts["after_cost_account_status"] == "NOT_ASSESSABLE"
    assert forecasts["fresh_confirmatory"] is False
    assert study.LABEL_VERSION == fr3_plan.LABEL_VERSION
    bad = deepcopy(report)
    bad["reference_path_diagnostic"]["cost_path_policy"] = reference()["cost_path_policy"]
    with pytest.raises(ValueError, match="COST_PLAN"):
        study._extract_recomputed(bad, artifact, outcome, config)


def test_l2_m0_fold_reconstruction_preserves_own_exact_past_proof(archive):
    case = deepcopy(archive)
    case["config"]["reference_policy"] = l2_reference()
    case["proof"] = training._baseline_fold_proof(case["config"], case["names"], case["bounds"], case["folds"])
    callback = provider(case)
    with pytest.raises(ValueError, match="^INSUFFICIENT_HEAD_TRAINING_ROWS$"):
        bind_heads(case, callback, 0)
    assert bind_heads(case, callback, 1) == next(iter(case["reports"].values()))["raw_heads"]
    old = deepcopy(case)
    old["config"]["reference_policy"] = reference()
    with pytest.raises(ValueError, match="PROOF_BINDING"):
        provider(old)  # A re-labelled reference cannot reuse an L2 fold proof.


def test_same_plan_l2_m0_m1_pair_rejects_old_plan_labels():
    from tests.test_v7_fr2_expansion import case
    from domain.les_v7.features import compute_features, select_bars
    from tests.test_v7_fr3_plan import CUTOFF
    from tests.test_v7_plan_r18 import neutral_bars
    from tests.test_v7_plan_r16 import synthetic_calendar
    rows, config, m0, m1 = case()
    policy = l2_reference()
    for value in (config, m0, m1): value["reference_policy"] = deepcopy(policy)
    data = neutral_bars()
    features, _ = compute_features(select_bars(data, CUTOFF))
    plan, audit = fr3_l2.make_plan("REVERSAL", features, policy, "SYNTHETIC_SOURCE",
        bars=data, cutoff=CUTOFF, session_evidence=synthetic_calendar(data))
    assert plan is not None, audit
    for row in rows:
        row["plan"] = deepcopy(plan)
    result = compare_fr3_predictions(rows, config, m0, m1)
    assert result["label_version"] == fr3_l2.LABEL_VERSION
    assert result["native_metrics"]["M0"]["label_sha256"] == result["native_metrics"]["M1"]["label_sha256"]
    assert result["after_cost_account_status"] == "NOT_ASSESSABLE"
    rows[0]["plan"] = rows[0]["plan"].original_plan()
    with pytest.raises(ValueError, match="NEW_PLAN_LABELS"):
        compare_fr3_predictions(rows, config, m0, m1)
