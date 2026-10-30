"""Shared v7 report integrity contract for storage, daily projection and renderers."""
from __future__ import annotations

from dataclasses import fields, asdict
from collections.abc import Mapping
from domain.les_v7.contracts import (FACTORS, SPEC_ID, SPEC_SHA256, FEATURE_VERSION, ReferencePlan, finite, identity,
                                    timestamp, QUALITY_REPORTED_PROFILES, parameter_version, quality_extension_schema,
                                    envelope_schema_version)
from domain.les_v7.decision import score_once, select_context, make_plan, triggers
from domain.les_v7.localization import Forecast
from domain.les_v7.allocation import validate_authorization
from domain.les_v7.strategy_policy import StrategyPolicy, policy_blocks_id


def _validate_strategy_quality(declared, quality, *, cutoff, symbol, market):
    """Bind frozen config separately from cutoff-selected runtime annotations."""
    if quality.get("quality_policy_id") is None:
        return  # Legacy R1 result has no independent quality-policy identity.
    from domain.les_v7.quality_policy import validate_quality_policy
    resolved = quality.get("resolved_policy")
    if not isinstance(resolved, Mapping) or identity(resolved) != quality["quality_policy_id"]:
        raise ValueError("V7_STRATEGY_QUALITY_POLICY_MISMATCH")
    fixed = validate_quality_policy(declared, cutoff, market=market)
    actual = validate_quality_policy(resolved, cutoff, market=market)
    annotations = {"latest_required_period", "period_selection_status", "period_selection_reasons"}
    if identity({k: v for k, v in fixed.items() if k not in annotations}) != identity(
            {k: v for k, v in actual.items() if k not in annotations}):
        raise ValueError("V7_STRATEGY_QUALITY_POLICY_MISMATCH")
    required = declared.get("latest_required_period") if isinstance(declared, Mapping) else None
    if required not in (None, "LATEST_VERIFIED_PUBLISHED_PERIOD") and required != actual.get("latest_required_period"):
        raise ValueError("V7_STRATEGY_QUALITY_PERIOD_MISMATCH")
    source = quality.get("financial_safety_source")
    if source is not None and quality.get("profile_id") in QUALITY_REPORTED_PROFILES:
        selected = validate_quality_policy(declared, cutoff, market=market, records=[source], symbol=symbol)
        if identity(selected) != identity(resolved):
            raise ValueError("V7_STRATEGY_QUALITY_PERIOD_SELECTION_MISMATCH")


def _validate_strategy_policy(report):
    """Opt-in v1 policy and v2 evidence identity; historical reports unchanged."""
    diagnostics, envelope = report["diagnostics"], report["envelope"]
    event = envelope["event"]
    record = diagnostics.get("strategy_policy")
    policy = None
    baseline = diagnostics.get("rule_baseline")
    if baseline is not None:
        if record is not None: raise ValueError("V7_BASELINE_CANNOT_MIX_EXTERNAL_STRATEGY")
        policy = baseline["policy"]
        _validate_strategy_quality(policy["quality_policy"], diagnostics["quality"], cutoff=event["cutoff_at"],
            symbol=event["security_id"], market=diagnostics["gate_bindings"]["SAFETY"]["market"])
    if record is not None:
        frozen = StrategyPolicy.load(record, cutoff=event["cutoff_at"])
        policy = frozen.record()
        if frozen.sha256 != diagnostics.get("strategy_policy_id"):
            raise ValueError("V7_STRATEGY_POLICY_ID_MISMATCH")
        q = diagnostics["quality"]
        _validate_strategy_quality(policy["quality_policy"], q, cutoff=event["cutoff_at"],
            symbol=event["security_id"], market=diagnostics["gate_bindings"]["SAFETY"]["market"])
        features = diagnostics["features"]
        selected, matched = select_context(features, policy["context_rules"])
        expected_plan = make_plan(selected, features, policy["reference_policy"], event["evidence_id"])
        trigger = triggers(selected, features, policy["trigger_rules"])
        if (event["context"] != selected or identity(diagnostics["context_matches"]) != identity(matched)
                or identity(diagnostics["reference_plan"]) != identity(asdict(expected_plan) if expected_plan else None)
                or any(diagnostics[key] != trigger[key] for key in ("starter", "confirmed", "evidence_families"))):
            raise ValueError("V7_STRATEGY_POLICY_CALCULATION_MISMATCH")
        bound = diagnostics.get("model_policy_binding")
        if envelope["model"]["bundle_sha256"] is not None:
            if (not isinstance(bound, dict) or set(bound) != {"bundle_sha256", "trained_until", "policy_blocks_id"}
                    or bound["bundle_sha256"] != envelope["model"]["bundle_sha256"]
                    or bound["policy_blocks_id"] != policy_blocks_id(policy)):
                raise ValueError("V7_STRATEGY_MODEL_POLICY_MISMATCH")
            origin = policy.get("source_bundle_sha256")
            if origin is not None and (origin != bound["bundle_sha256"]
                    or timestamp(policy["available_at"]) < timestamp(bound["trained_until"])):
                raise ValueError("V7_STRATEGY_COMPATIBILITY_POLICY_BACKDATED")
        elif bound is not None:
            raise ValueError("V7_STRATEGY_MODEL_POLICY_MISMATCH")
    elif diagnostics.get("strategy_policy_id") is not None or diagnostics.get("model_policy_binding") is not None:
        raise ValueError("V7_STRATEGY_POLICY_RECORD_MISSING")
    binding = diagnostics.get("evidence_binding")
    native = diagnostics.get("native_evidence")
    if policy is not None and "source_policy" in policy and native is None:
        raise ValueError("V7_R6_NATIVE_SOURCE_EVIDENCE_REQUIRED")
    if record is not None or native is not None or baseline is not None:
        if not isinstance(binding, dict): raise ValueError("V7_EVIDENCE_BINDING_MISSING")
        base = diagnostics["gate_bindings"]["SAFETY"]
        quality_policy = policy["quality_policy"] if policy is not None else diagnostics.get("evidence_quality_policy")
        if policy is None:
            _validate_strategy_quality(quality_policy, diagnostics["quality"], cutoff=event["cutoff_at"],
                symbol=event["security_id"], market=base["market"])
        external = binding.get("external_evidence_ids")
        if not isinstance(external, list) or any(not isinstance(item, str) or not item for item in external):
            raise ValueError("V7_EXTERNAL_EVIDENCE_IDENTITIES_INVALID")
        feature_version = policy.get("native_feature_version", FEATURE_VERSION) if baseline else FEATURE_VERSION
        expected = dict(schema="les-v7-evidence-binding/v2", **base, data_rules=feature_version,
            external_evidence_ids=external, quality_policy_id=identity(quality_policy),
            strategy_policy_id=diagnostics.get("strategy_policy_id"),
            native_source_id=identity(native["source_packet"]) if native is not None else None)
        if baseline is not None: expected["rule_baseline_policy_id"] = baseline["policy_id"]
        if (base["symbol"] != event["security_id"] or timestamp(base["cutoff"]) != timestamp(event["cutoff_at"])
                or identity(binding) != identity(expected) or identity(binding) != event["evidence_id"]):
            raise ValueError("V7_EVIDENCE_IDENTITY_BINDING_MISMATCH")
        for gate in ("DATA_INTEGRITY", "TRADING_STATUS"):
            if diagnostics["gate_bindings"][gate] != base:
                raise ValueError("V7_GATE_SOURCE_BINDING_MISMATCH")
        if diagnostics["gate_bindings"]["VALUATION"] != dict(base, price=diagnostics["features"].get("close"),
                quality_result_id=identity(diagnostics["quality"])):
            raise ValueError("V7_GATE_SOURCE_BINDING_MISMATCH")
        for gate in ("STRESS_RISK", "PRICE_DISCOVERY_PAYOFF"):
            if diagnostics["gate_bindings"][gate] != dict(base, plan=diagnostics["reference_plan"]):
                raise ValueError("V7_GATE_SOURCE_BINDING_MISMATCH")
    elif binding is not None:
        raise ValueError("V7_EVIDENCE_BINDING_WITHOUT_VERSIONED_SOURCE")
    if native is not None:
        if identity(native.get("reference_policy")) != identity(policy.get("reference_policy") if policy else None):
            raise ValueError("V7_NATIVE_REFERENCE_POLICY_BINDING_MISMATCH")
        if identity(native.get("valuation_policy")) != identity(policy.get("valuation_policy") if policy else None):
            raise ValueError("V7_NATIVE_VALUATION_POLICY_BINDING_MISMATCH")
        if identity(native.get("valuation_input_policy")) != identity(policy.get("valuation_input_policy") if policy else None):
            raise ValueError("V7_NATIVE_VALUATION_INPUT_POLICY_BINDING_MISMATCH")
        if identity(native.get("source_policy")) != identity(policy.get("source_policy") if policy else None):
            raise ValueError("V7_NATIVE_SOURCE_POLICY_BINDING_MISMATCH")
        from domain.les_v7.native_evidence import validate_native_receipts
        validate_native_receipts(report)


def _validate_model_input_profile(report):
    """Optional predictor lineage; old reports remain readable without it.

    The bundle digest binds this declaration to its source. Native reports do
    not embed the full frozen model; FR2 reports do, permitting an additional
    exact comparison with their retained predictor declaration.
    """
    diagnostics, envelope = report["diagnostics"], report["envelope"]
    if "model_input_profile" not in diagnostics:
        return
    from domain.les_v7.fractals import is_active_fractal_feature
    profile = diagnostics["model_input_profile"]
    required = {"schema", "bundle_sha256", "feature_names", "fractal_mode"}
    if isinstance(profile, dict) and profile.get("schema") == "les-v7-model-input-profile/v2":
        required.add("feature_bounds")
        optional = {"execution_feature_profile", "fill_only_feature_extension", "fill_calibration_profile"}
        if set(profile) - required - optional or not set(profile) & optional:
            raise ValueError("V7_MODEL_INPUT_PROFILE_BINDING_MISMATCH")
        required.update(set(profile) & optional)
        from domain.les_v7.execution_profile import validate_execution_profile
        validate_execution_profile(profile)
        bounds = profile.get("feature_bounds")
        if not isinstance(bounds, list) or len(bounds) != len(profile.get("feature_names", ())):
            raise ValueError("V7_MODEL_INPUT_PROFILE_INVALID")
        if "fill_calibration_profile" in profile:
            from domain.les_v7.fill_calibration import validate_profile
            validate_profile(profile["fill_calibration_profile"])
    if (not isinstance(profile, dict) or set(profile) != required
            or profile["schema"] not in {"les-v7-model-input-profile/v1", "les-v7-model-input-profile/v2"}
            or not envelope["model"].get("bundle_sha256")
            or profile["bundle_sha256"] != envelope["model"]["bundle_sha256"]):
        raise ValueError("V7_MODEL_INPUT_PROFILE_BINDING_MISMATCH")
    names, mode = profile["feature_names"], profile["fractal_mode"]
    if (not isinstance(names, list) or not 1 <= len(names) <= 64
            or any(not isinstance(name, str) or not 1 <= len(name) <= 80 for name in names)
            or len(set(names)) != len(names)
            or mode is not None and (not isinstance(mode, str) or not 1 <= len(mode) <= 80)):
        raise ValueError("V7_MODEL_INPUT_PROFILE_INVALID")
    if mode == "V7_NO_FRACTAL_BASELINE" and any(is_active_fractal_feature(name) for name in names):
        raise ValueError("V7_MODEL_INPUT_PROFILE_FRACTAL_MODE_CONFLICT")
    fr2 = diagnostics.get("fr2")
    if fr2 is not None:
        record = fr2.get("predictor_bundle", {})
        payload = record.get("payload", {})
        from domain.les_v7.localization import model_input_profile
        expected = model_input_profile(payload, record.get("sha256"))
        if profile != expected:
            raise ValueError("V7_MODEL_INPUT_PROFILE_PREDICTOR_MISMATCH")


def validate_report(report: dict) -> None:
    _validate_model_input_profile(report)
    if "fr2" in report.get("diagnostics", {}):
        from domain.les_v7.fr2_policy import validate_fr2_report
        validate_fr2_report(report)
        return
    envelope, diagnostics = report["envelope"], report["diagnostics"]
    event = envelope["event"]
    baseline = diagnostics.get("rule_baseline")
    from domain.les_v7.rule_baseline import (ENVELOPE_SCHEMA as BASELINE_ENVELOPE_SCHEMA,
        baseline_decision_policy, require_baseline_policy, validate_rule_baseline_report)
    expected_parameters = parameter_version(diagnostics["quality"].get("profile_id"))
    baseline_policy = None
    if baseline is not None:
        baseline_policy = require_baseline_policy(baseline["policy"],
            market=diagnostics["gate_bindings"]["SAFETY"]["market"], cutoff=event["cutoff_at"])
        expected_parameters += ":" + baseline_policy["policy_version"] + ":" + baseline["policy_id"]
    if (report["symbol"] != event["security_id"] or report["model_version"] != "7.0.0"
            or envelope["schema_version"] != (BASELINE_ENVELOPE_SCHEMA if baseline else envelope_schema_version(diagnostics["quality"].get("profile_id")))
            or report["parameter_version"] != expected_parameters
            or report["spec_sha256"] != SPEC_SHA256 or envelope["spec_id"] != SPEC_ID
            or report["signal_date"] != diagnostics["feature_audit"].get("cutoff_session")):
        raise ValueError("V7_REPORT_EVENT_BINDING_MISMATCH")
    timestamp(event["cutoff_at"])
    _validate_strategy_policy(report)
    scope = diagnostics.get("run_scope")
    expected_decision_policy = baseline_decision_policy(baseline_policy) if baseline else "IMP_STATE_01"
    if envelope["strategy"]["decision_policy_version"] == expected_decision_policy + ":DIAGNOSTICS_ONLY_R1":
        if (scope != dict(schema="v7-diagnostics-only/v1", diagnostics_only=True, formal_entry_enabled=False)
                or envelope["score"]["final_score"] is not None
                or envelope["strategy"]["strategy_decision"] in {"BUY", "BUY_SMALL"}
                or "DIAGNOSTICS_ONLY_NOT_FORMAL_ENTRY" not in envelope["strategy"]["reason_codes"]):
            raise ValueError("V7_DIAGNOSTIC_SCOPE_CANNOT_AUTHORIZE_FORMAL_ENTRY")
    elif scope is not None:
        raise ValueError("V7_DIAGNOSTIC_SCOPE_POLICY_MISMATCH")
    elif baseline is not None and envelope["strategy"]["decision_policy_version"] != expected_decision_policy:
        raise ValueError("V7_BASELINE_DECISION_POLICY_MISMATCH")
    expected_signal = identity(dict(evidence_id=event["evidence_id"], bundle_hash=envelope["model"]["bundle_sha256"],
                                    decision_policy_version=envelope["strategy"]["decision_policy_version"]))
    if event["signal_id"] != expected_signal: raise ValueError("V7_SIGNAL_ID_BINDING_MISMATCH")
    if (identity(diagnostics["quality"]) != envelope["quality"]["quality_result_id"]
            or identity(diagnostics["localization"]) != envelope["localization"]["forecast_result_id"]):
        raise ValueError("V7_RESULT_ID_BINDING_MISMATCH")
    q = diagnostics["quality"]
    extension = quality_extension_schema(q.get("profile_id"))
    if extension:
        if (report.get("quality_extension_schema") != extension
                or q.get("quality_policy_id") != identity(diagnostics.get("quality_policy"))):
            raise ValueError("V7_QUALITY_POLICY_BINDING_MISMATCH")
        readiness = diagnostics.get("readiness", {})
        plan = diagnostics["reference_plan"]
        f = diagnostics["features"]
        liquidity_known = bool(f.get("spread") is not None and plan and f.get("adv") and f["adv"] > 0)
        decision = envelope["strategy"]["strategy_decision"]
        from domain.les_v7.entry_policy_r9 import is_r9, quality_readiness
        expected_readiness = (quality_readiness(q, diagnostics["entry_admission"]) if is_r9(report) else
            dict(state="READY" if q["status"] == "PASS" else "BLOCKED", status=q["status"], reason_codes=q["reason_codes"]))
        if (readiness.get("quality") != expected_readiness
                or readiness.get("model") != dict(
                    state="RULE_BASED_READY" if baseline else "READY" if diagnostics["localization"]["status"] == "VALID" else "NOT_READY",
                    status=diagnostics["localization"]["status"],
                    bundle_provided=envelope["model"]["bundle_sha256"] is not None,
                    reason_codes=diagnostics["localization"]["reasons"])
                or readiness.get("execution") != dict(
                    state="READY_FOR_ACCOUNT_CHECK" if decision in {"BUY", "BUY_SMALL"} else "BLOCKED",
                    candidate_decision=decision, order_authority=False,
                    required_gates={g["gate_id"]: g["state"] for g in diagnostics["gate_receipts"]},
                    reference_plan_ready=plan is not None, liquidity_known=liquidity_known)):
            raise ValueError("V7_READINESS_BINDING_MISMATCH")
    expected_quality = dict(status=q["status"], quality_score=q["quality_score"], data_confidence=q["data_confidence"],
        coverage_total=q["coverage_total"], core_ttm_pass=q["core_ttm_pass"], gap_qualifier=q["gap_qualifier"],
        gap_penalty_points=q["missing_penalty_points"], quality_result_id=identity(q))
    if q.get("profile_id") in QUALITY_REPORTED_PROFILES:
        from domain.les_v7.quality_policy import validate_quality_policy
        from domain.les_v7.quality_reported import validate_reported_result
        policy = diagnostics.get("quality_policy")
        if (not isinstance(policy, dict) or policy.get("profile_id") != q.get("profile_id")
                or identity(validate_quality_policy(policy, event["cutoff_at"])) != identity(policy)
                or q.get("mapping_version") != policy.get("mapping_version")):
            raise ValueError("V7_REPORTED_QUALITY_POLICY_BINDING_MISMATCH")
        if (q.get("scoring_basis") != "REPORTED_SAME_PERIOD"
                or not isinstance(q.get("cutoff_at"), str)
                or timestamp(q.get("cutoff_at")) != timestamp(event["cutoff_at"])
                or not isinstance(q.get("core_reported_pass"), bool)
                or not isinstance(q.get("core_inputs_pass"), bool)
                or q.get("status") == "PASS" and not q["core_inputs_pass"]):
            raise ValueError("V7_REPORTED_QUALITY_CORE_BINDING_MISMATCH")
        financial_source = q.get("financial_safety_source")
        if financial_source is not None and (
                not isinstance(financial_source, Mapping)
                or financial_source.get("symbol") != event["security_id"]
                or financial_source.get("market") != diagnostics["gate_bindings"]["SAFETY"]["market"]):
            raise ValueError("V7_REPORTED_QUALITY_SECURITY_BINDING_MISMATCH")
        validate_reported_result(q)
        expected_quality.update(core_reported_pass=q["core_reported_pass"],
                                core_inputs_pass=q["core_inputs_pass"], scoring_basis=q["scoring_basis"])
    if identity(envelope["quality"]) != identity(expected_quality):
        raise ValueError("V7_QUALITY_SUMMARY_MISMATCH")
    prediction = None if baseline else Forecast(**{field.name: diagnostics["localization"][field.name] for field in fields(Forecast)})
    expected_forecast = dict(status=prediction.status, bottom_price_quantiles=prediction.quantiles,
        timing_pmf=prediction.timing, p_fill=prediction.p_fill, p_rebound_given_fill=prediction.p_rebound,
        p_entry_given_fill=prediction.p_entry, p_bottom_soon=prediction.p_soon,
        forecast_result_id=identity(prediction.record())) if prediction is not None else None
    if prediction is not None and identity(envelope["localization"]) != identity(expected_forecast):
        raise ValueError("V7_FORECAST_SUMMARY_MISMATCH")
    score = envelope["score"]
    if baseline is not None:
        validate_rule_baseline_report(report)
        expected_score = dict(diagnostics["diagnostic_score"])
    else:
        expected_score = score_once(score["factor_values"], score["applicable_weights"],
                                    diagnostics["penalties"], diagnostics["safety_caps"])
    if identity(expected_score) != identity(diagnostics["diagnostic_score"]):
        raise ValueError("V7_SCORE_ARITHMETIC_MISMATCH")
    if score["final_score"] is not None:
        finite(score["final_score"], "final_score", 0, 100)
        if baseline is None and (q["status"] != "PASS" or prediction.status != "VALID"): raise ValueError("V7_FORMAL_SCORE_WITH_INVALID_GATES")
        if score["final_score"] != expected_score["final_score"]: raise ValueError("V7_SCORE_ARITHMETIC_MISMATCH")
    expected_score.update(factor_order=FACTORS, factor_values=score["factor_values"],
                          applicable_weights=score["applicable_weights"], final_score=score["final_score"])
    if identity(expected_score) != identity(score): raise ValueError("V7_SCORE_ARITHMETIC_MISMATCH")
    plan = diagnostics["reference_plan"]
    if (plan is None) != (event["plan_id"] is None): raise ValueError("V7_PLAN_ID_BINDING_MISMATCH")
    if plan is not None:
        from domain.les_v7.plan_r19 import plan_from_record
        reconstructed = plan_from_record(plan)
        if reconstructed.plan_id != event["plan_id"] or reconstructed.context != event["context"]:
            raise ValueError("V7_PLAN_ID_BINDING_MISMATCH")
    if envelope["execution"] != dict(run_mode="RESEARCH", live_enabled=False, authorization_verified=False,
            execution_status=None, allocated_shares=0, client_order_id=None, order_status=None):
        raise ValueError("V7_RESEARCH_STORE_CANNOT_AUTHORIZE_ORDERS")
    decision = envelope["strategy"]["strategy_decision"]
    if decision not in {"NO_BUY", "DATA_INSUFFICIENT", "WATCH_ONLY", "AVOID", "WATCH", "ARMED", "BUY_SMALL", "BUY"}:
        raise ValueError("INVALID_V7_STRATEGY_DECISION")
    if decision in {"BUY", "BUY_SMALL"}:
        confirmed = decision == "BUY"
        from domain.les_v7.entry_policy_r9 import is_r9
        graded = is_r9(report)
        admission = diagnostics.get("entry_admission", {})
        eligible = (admission.get("buy_allowed" if confirmed else "small_allowed") is True if graded else
            q["status"] == "PASS" and q["quality_score"] >= (70 if confirmed else 75)
            and all(r["state"] == "PASS" for r in diagnostics["gate_receipts"]))
        minima = (baseline["policy"].get("unified_technical_policy", {}) if baseline else {})
        threshold = (minima.get("buy_minimum", 72) if confirmed else minima.get("small_minimum", 62)) if baseline else (80 if confirmed else 72)
        if (score["final_score"] is None or score["final_score"] < threshold
                or not eligible
                or diagnostics["confirmed" if confirmed else "starter"] is not True
                or diagnostics["confirmed_gate" if confirmed else "starter_gate"] is not True):
            raise ValueError("V7_ENTRY_DECISION_WITH_FAILED_OR_UNKNOWN_GATES")
    validate_authorization(report)
