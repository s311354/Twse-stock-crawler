"""SYNTHETIC ONLY: policy separation is not trained model or trade approval."""
from copy import deepcopy

import pandas as pd
import pytest

from application.les_v7 import analyze_request
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.engine import evaluate
from domain.les_v7.localization import FrozenBundle
from domain.les_v7.report import validate_report
from domain.les_v7.strategy_policy import StrategyPolicy, POLICY_BLOCKS, make_strategy_policy
from tests.strategies.test_low_entry_score_v7 import CUTOFF, bundle, complete_request, receipt


def policy_record():
    return StrategyPolicy.from_bundle(FrozenBundle.load(bundle())).export()


def standalone_request():
    request = complete_request()
    request["context"].pop("model_bundle")
    request["context"]["strategy_policy"] = policy_record()
    request["context"]["receipts"] = {}
    return request


def evaluate_with_receipts(request):
    data, context = pd.DataFrame(request["bars"]), request["context"]
    report = evaluate(data, context)
    for gate in ("SAFETY", "DATA_INTEGRITY", "TRADING_STATUS"):
        context["receipts"][gate] = receipt(gate, report["diagnostics"]["gate_bindings"][gate])
    report = evaluate(data, context)
    for gate in ("VALUATION", "STRESS_RISK"):
        context["receipts"][gate] = receipt(gate, report["diagnostics"]["gate_bindings"][gate])
    return evaluate(data, context)


def test_legacy_bundle_complete_result_stays_byte_identical():
    # Captured before the policy separation; not regenerated from the new path.
    assert identity(analyze_request(complete_request())) == "0451f73052e734e31024cf90785d8301790026e84ae6ff937f065079437360ee"


def test_compatibility_policy_contains_no_fitted_heads_calibration_or_weights():
    original = FrozenBundle.load(bundle())
    policy = StrategyPolicy.from_bundle(original)
    record = policy.record()
    assert timestamp(record["available_at"]) == timestamp(original.record()["trained_until"])
    assert record["source_bundle_sha256"] == original.sha256
    assert set(POLICY_BLOCKS) <= set(record)
    assert not {"heads", "calibration", "weights", "imputation"} & set(record)
    record["context_rules"]["REVERSAL"]["value"] = -123
    assert policy.record()["context_rules"]["REVERSAL"]["value"] == 0
    policy.bind_bundle(original)


def test_policy_only_computes_causal_context_plan_trigger_not_probability_or_formal_score():
    request = standalone_request()
    original = deepcopy(request)
    report = evaluate_with_receipts(request)
    validate_report(report)
    assert report["envelope"]["event"]["context"] == "REVERSAL"
    assert report["diagnostics"]["reference_plan"] is not None
    assert isinstance(report["diagnostics"]["starter"], bool)
    assert report["envelope"]["localization"]["status"] == "NOT_READY"
    assert report["envelope"]["localization"]["p_entry_given_fill"] is None
    assert report["envelope"]["score"]["factor_values"] is None
    assert report["envelope"]["score"]["applicable_weights"] is None
    assert report["envelope"]["score"]["final_score"] is None
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert original["context"]["strategy_policy"] == request["context"]["strategy_policy"]


@pytest.mark.parametrize("block", POLICY_BLOCKS)
def test_model_requires_all_four_strategy_blocks_to_match(block):
    raw = policy_record()
    if block == "context_rules": raw["payload"][block]["REVERSAL"]["value"] = 1
    elif block == "trigger_rules": raw["payload"][block]["tight_base_or_retest"]["value"] = 1
    elif block == "reference_policy": raw["payload"][block]["version"] = "DIFFERENT_POLICY"
    else: raw["payload"][block]["latest_required_period"] = "2026Q1"
    raw["sha256"] = identity(raw["payload"])
    policy = StrategyPolicy.load(raw)
    with pytest.raises(ValueError, match="STRATEGY_POLICY_MODEL_BINDING_MISMATCH"):
        policy.bind_bundle(FrozenBundle.load(bundle()))


def test_policy_availability_cannot_backdate_bundle_compatibility_or_ignore_cutoff():
    raw = policy_record()
    raw["payload"]["available_at"] = "2025-01-01T00:00:00Z"
    raw["sha256"] = identity(raw["payload"])
    with pytest.raises(ValueError, match="COMPATIBILITY_CANNOT_BACKDATE"):
        StrategyPolicy.load(raw).bind_bundle(FrozenBundle.load(bundle()))
    raw["payload"]["available_at"] = "2027-01-01T00:00:00Z"
    raw["sha256"] = identity(raw["payload"])
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        StrategyPolicy.load(raw, cutoff=CUTOFF)


def test_reference_cost_source_availability_must_not_follow_policy_publication():
    raw = policy_record()
    raw["payload"]["quality_policy"]["available_at"] = "2026-01-03T00:00:00Z"
    raw["sha256"] = identity(raw["payload"])
    with pytest.raises(ValueError, match="REFERENCES_FUTURE_SOURCE"):
        StrategyPolicy.load(raw)


def test_independently_published_policy_does_not_make_future_model_usable():
    request = standalone_request()
    raw = request["context"]["strategy_policy"]
    raw["payload"].pop("source_bundle_sha256")
    raw["sha256"] = identity(raw["payload"])
    model = bundle()
    model["payload"]["trained_until"] = "2027-01-01T00:00:00Z"
    model["sha256"] = identity(model["payload"])
    request["context"]["model_bundle"] = model
    report = evaluate_with_receipts(request)
    validate_report(report)
    assert "MODEL_NOT_AVAILABLE_AT_CUTOFF" in report["diagnostics"]["localization"]["reasons"]
    assert report["envelope"]["score"]["final_score"] is None


@pytest.mark.parametrize("damage", ["policy_hash", "policy_id", "plan", "context", "trigger", "source_id", "quality"])
def test_report_validation_recomputes_policy_and_source_bindings(damage):
    report = evaluate_with_receipts(standalone_request())
    if damage == "policy_hash": report["diagnostics"]["strategy_policy"]["sha256"] = "0" * 64
    elif damage == "policy_id": report["diagnostics"]["strategy_policy_id"] = "0" * 64
    elif damage == "plan": report["diagnostics"]["reference_plan"]["entry"] += 1
    elif damage == "context": report["envelope"]["event"]["context"] = "TREND_PULLBACK"
    elif damage == "trigger": report["diagnostics"]["starter"] = not report["diagnostics"]["starter"]
    elif damage == "source_id": report["diagnostics"]["evidence_binding"]["quality_policy_id"] = "0" * 64
    else:
        report["diagnostics"]["strategy_policy"]["payload"]["quality_policy"]["latest_required_period"] = "2026Q1"
        report["diagnostics"]["strategy_policy"]["sha256"] = identity(report["diagnostics"]["strategy_policy"]["payload"])
    with pytest.raises(ValueError): validate_report(report)


def test_future_inputs_do_not_change_explicit_policy_report_or_caller():
    request = standalone_request()
    baseline = evaluate_with_receipts(request)
    original = deepcopy(request)
    request["bars"].append(dict(request["bars"][-1], Date="2026-10-13", Close=1., Low=.5))
    request["context"]["filings"].append(dict(request["context"]["filings"][-1],
        available_at="2026-10-13T00:00:00+08:00", cfo=-1e9))
    assert evaluate(pd.DataFrame(request["bars"]), request["context"]) == baseline
    assert original["context"]["strategy_policy"] == request["context"]["strategy_policy"]


def test_standalone_policy_maker_never_invents_availability_or_model_fallback():
    raw = bundle()["payload"]
    record = make_strategy_policy(policy_version="SYNTHETIC_ONLY", available_at=raw["trained_until"],
                                  **{key: raw[key] for key in POLICY_BLOCKS})
    assert "source_bundle_sha256" not in record["payload"]
    assert StrategyPolicy.load(record).export() == record
    altered = deepcopy(record); altered["payload"]["weights"] = [1 / 6] * 6
    altered["sha256"] = identity(altered["payload"])
    with pytest.raises(ValueError, match="DEPENDENCIES_INCOMPLETE"):
        StrategyPolicy.load(altered)


def test_causal_policy_rules_cannot_read_learned_predictions_or_scores():
    raw = policy_record()
    raw["payload"]["context_rules"]["REVERSAL"]["field"] = "score_final"
    raw["sha256"] = identity(raw["payload"])
    with pytest.raises(ValueError, match="CONTEXT_CANNOT_READ_FORECAST_OR_SCORE"):
        StrategyPolicy.load(raw)


def test_explicit_diagnostic_scope_cannot_publish_formal_score_even_with_fitted_bundle():
    request = complete_request()
    baseline = evaluate(pd.DataFrame(request["bars"]), request["context"])
    request["context"]["diagnostics_only"] = True
    report = evaluate(pd.DataFrame(request["bars"]), request["context"])
    validate_report(report)
    assert baseline["envelope"]["score"]["final_score"] is not None
    assert report["envelope"]["score"]["raw_score"] == baseline["envelope"]["score"]["raw_score"]
    assert report["envelope"]["score"]["final_score"] is None
    assert report["envelope"]["strategy"]["strategy_decision"] == "WATCH_ONLY"
    assert report["envelope"]["event"]["evidence_id"] == baseline["envelope"]["event"]["evidence_id"]
    assert report["envelope"]["event"]["signal_id"] != baseline["envelope"]["event"]["signal_id"]
    forged = deepcopy(report)
    forged["envelope"]["score"]["final_score"] = baseline["envelope"]["score"]["final_score"]
    with pytest.raises(ValueError, match="DIAGNOSTIC_SCOPE_CANNOT_AUTHORIZE"):
        validate_report(forged)


@pytest.mark.parametrize("value", [1, "true", None])
def test_ambiguous_diagnostic_scope_configuration_is_rejected(value):
    with pytest.raises(ValueError, match="DIAGNOSTIC_SCOPE_BOOLEAN_REQUIRED"):
        evaluate(pd.DataFrame(), dict(diagnostics_only=value))
