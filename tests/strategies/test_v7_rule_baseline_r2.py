"""SYNTHETIC ONLY: context reachability, not real performance or fills."""
from copy import deepcopy
from datetime import timedelta

import pytest

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.engine import evaluate
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (AVAILABLE_AT, BASELINE_POLICY, LEGACY_BASELINE_POLICY, R2_BASELINE_POLICY, R3_BASELINE_POLICY, R4_BASELINE_POLICY,
    baseline_policy, baseline_decision_policy, baseline_triggers, prepare_baseline,
    require_baseline_policy, select_baseline_context)
from domain.strategies.low_entry_v6_hybrid import select_context
from tests.strategies.test_v7_rule_baseline import request, unit_features, unit_outcome


def relative_leader():
    f = unit_features()
    f.update(downside_resilience=False, resilience=-.01, base_compression=True, breakout_quality=True,
        breakout_retest_confirmed=False, position60=.8, drawdown60=-.02, drawdown20=0.,
        ema10=100.5, ema20=100., ema20_slope=.2, rs_near_high=True,
        ema20_extension=.5, breakout_extension=.1)
    return f


def test_r1_policy_content_identity_is_not_rewritten_when_default_is_r2():
    expected = {"TWSE": "efa115538e0d1fc094aa00acdf20646982652a736215d2cee700a6a8916fc7b0",
        "TPEX": "5b0c96889d8f27dd864b71d0ab3d55dac3b7c8634f1ff98c6e827e486150582d"}
    for market, sha in expected.items():
        old = baseline_policy(market, version=LEGACY_BASELINE_POLICY)
        new = baseline_policy(market)
        assert identity(old) == sha
        assert new["policy_version"] == BASELINE_POLICY and identity(new) != sha
        assert new["config"] == old["config"] and new["model_spec"] == old["model_spec"]
        assert new["quality_policy"] == old["quality_policy"]
        assert new["native_risk_limits"] == old["native_risk_limits"]
        assert baseline_decision_policy(old) == "RULE_BASED_V6_HYBRID_R1"
        r2 = baseline_policy(market, version=R2_BASELINE_POLICY)
        assert baseline_decision_policy(r2) == "RULE_BASED_V6_HYBRID_R2"
        assert baseline_decision_policy(new) == "RULE_BASED_V6_HYBRID_R5"
        r3 = baseline_policy(market, version=R3_BASELINE_POLICY)
        assert baseline_decision_policy(r3) == "RULE_BASED_V6_HYBRID_R3"
        assert r3["valuation_policy"] == new["valuation_policy"]
        r4 = baseline_policy(market, version=R4_BASELINE_POLICY)
        assert baseline_decision_policy(r4) == "RULE_BASED_V6_HYBRID_R4"
        assert r4["valuation_policy"] == new["valuation_policy"]
        assert r4["score_policy"] == new["score_policy"]
        if market == "TWSE":
            assert identity(r2) == "c11868a5924c9e376d193f9e953f55fde3f7ed8d51c6c3883ed5ef29b093f72e"
            assert identity(r3) == "8a14cb74c1e384da694a5acfa743b9fd9713eadb72ddfcc11d2ff19e9ea65236"
            assert identity(r4) == "bb983b14f9dc8c21d48e0243a704a6c995e96d9a8b0f9a154897c185feb7f69e"


def test_measured_relative_leadership_can_reach_its_existing_trigger_without_rewriting_v6():
    f = relative_leader(); before = deepcopy(f)
    assert select_context(f).context == "NO_CLEAR_CONTEXT"
    assert select_baseline_context(f, policy_version=LEGACY_BASELINE_POLICY) == select_context(f)
    selected = select_baseline_context(f)
    assert selected.context == "LEADERSHIP_CONTINUATION"
    checks = dict(selected.evidence)["LEADERSHIP_CONTINUATION"]
    assert dict(checks)["RESILIENCE_OR_RELATIVE_LEADERSHIP"] is True
    trigger = baseline_triggers(dict(features=f, selection=selected.record()))
    assert trigger["starter"] is True and trigger["confirmed"] is True
    assert f == before


@pytest.mark.parametrize("field,value", [("rs20", None), ("rs20", 0.), ("rs60", -.01),
    ("rs_slope5", 0.), ("rs_near_high", False), ("rs_near_high", "True"),
    ("support_intact", False), ("no_new_low2", None), ("market_ready", False),
    ("market_ready", None), ("history_ready", False), ("base_compression", False),
    ("breakout_quality", False), ("ema20_slope", 0.), ("position60", .59)])
def test_unknown_stale_or_failed_leader_evidence_does_not_take_the_new_branch(field, value):
    f = relative_leader(); f[field] = value
    assert select_baseline_context(f).context != "LEADERSHIP_CONTINUATION"


@pytest.mark.parametrize("field", ["rs20", "rs60", "rs_slope5", "rs_near_high", "no_new_low2"])
def test_old_positive_values_with_unavailable_masks_do_not_prove_relative_leadership(field):
    f = relative_leader(); f["score_availability"][field] = False
    assert select_baseline_context(f).context != "LEADERSHIP_CONTINUATION"


def test_existing_verified_downside_branch_is_not_replaced_by_required_relative_strength():
    f = relative_leader(); f.update(downside_resilience=True, rs60=None)
    assert select_context(f).context == select_baseline_context(f).context == "LEADERSHIP_CONTINUATION"
    assert select_baseline_context(unit_features()).context == select_context(unit_features()).context


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK"])
def test_new_context_does_not_turn_unknown_hard_gate_into_buy(gate):
    outcome = unit_outcome(relative_leader(), missing_gate=gate)
    assert outcome["score_evidence"]["total_weight"] == 100
    assert outcome["score"]["final_score"] <= 61
    assert outcome["decision"] == "DATA_INSUFFICIENT"


def test_new_policy_is_not_available_at_an_old_cutoff_and_unknown_version_is_rejected():
    before = (timestamp(AVAILABLE_AT) - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="V7_BASELINE_NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(), market="TWSE", cutoff=before)
    assert require_baseline_policy(baseline_policy(version=LEGACY_BASELINE_POLICY),
        market="TWSE", cutoff=before)["policy_version"] == LEGACY_BASELINE_POLICY
    with pytest.raises(ValueError, match="V7_BASELINE_POLICY_VERSION_UNKNOWN"):
        baseline_policy(version="UNREVIEWED")


def test_old_report_recomputes_under_its_saved_policy_and_new_run_gets_distinct_identity():
    bars, context = request()
    context["rule_baseline_policy_version"] = LEGACY_BASELINE_POLICY
    old = evaluate(bars, context); validate_report(old)
    assert old["diagnostics"]["rule_baseline"]["policy_id"] == identity(
        baseline_policy(version=LEGACY_BASELINE_POLICY))
    context["rule_baseline_policy_version"] = R3_BASELINE_POLICY
    new = evaluate(bars, context); validate_report(new)
    assert old["parameter_version"] != new["parameter_version"]
    assert old["envelope"]["event"]["signal_id"] != new["envelope"]["event"]["signal_id"]
    assert old["envelope"]["event"]["evidence_id"] != new["envelope"]["event"]["evidence_id"]
    assert old["envelope"]["score"] == new["envelope"]["score"]
    assert old["envelope"]["quality"] == new["envelope"]["quality"]
    assert old["envelope"]["localization"] == new["envelope"]["localization"]


def test_saved_r1_selection_and_features_recompute_without_using_the_new_default():
    bars, context = request()
    prepared = prepare_baseline(bars, market=context["market"], cutoff=context["as_of"],
        policy_version=LEGACY_BASELINE_POLICY)
    assert prepared["selection"] == select_context(prepared["features"]).record()
    old = evaluate(bars, dict(context, rule_baseline_policy_version=LEGACY_BASELINE_POLICY))
    old["diagnostics"]["rule_baseline"]["selection"]["selected_setup"] = "LEADERSHIP_CONTINUATION"
    with pytest.raises(ValueError, match="V7_BASELINE_SOURCE_OR_FEATURE_RECOMPUTATION_MISMATCH"):
        validate_report(old)


def test_baseline_version_cannot_be_ignored_by_the_learned_or_ordinary_api():
    bars, context = request(); context.pop("rule_baseline")
    with pytest.raises(ValueError, match="V7_BASELINE_POLICY_VERSION_WITHOUT_RULE_MODE"):
        evaluate(bars, dict(context, rule_baseline_policy_version=LEGACY_BASELINE_POLICY))
