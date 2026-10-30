"""Cheap global v7 readiness checks, before market/MOPS fetch or notifications.

This is configuration readiness only, not a guarantee of per-stock evidence,
Quality, liquidity or entry permission. API diagnostic calls remain compatible.
"""
from __future__ import annotations

import json

from domain.les_v7.contracts import timestamp, identity, QUALITY_REPORTED_PROFILES
from domain.les_v7.localization import FrozenBundle
from domain.les_v7.strategy_policy import StrategyPolicy
from domain.les_v7.quality_policy import validate_quality_policy, is_source_evidence_policy


def daily_v7_preflight(settings, *, checked_at, market="TWSE"):
    """Validate supplied model/policy without loading market or filing graphs."""
    if settings.daily_entry_model != "v7":
        return dict(state="NOT_APPLICABLE", mode="LEGACY", reason_codes=[])
    at = timestamp(checked_at)
    diagnostics = bool(settings.les_v7_diagnostics_only)
    result = dict(state="BLOCKED", mode="DIAGNOSTIC_ONLY" if diagnostics else "FULL_V7",
        checked_at=at.isoformat(), cutoff_policy="AFTER_COLLECTION_CURRENT_DAILY_ONLY",
        reason_codes=[], model_sha256=None, strategy_policy_sha256=None,
        native_evidence_enabled=bool(settings.les_v7_native_evidence),
        formal_entry_enabled=False, order_authority=False)
    try:
        if settings.les_v7_rule_baseline:
            result["mode"] = "RULE_BASED_BASELINE"
            if (settings.les_v7_bundle_json or settings.les_v7_strategy_policy_json or diagnostics
                    or getattr(settings, "les_v7_research_metadata_json", None)):
                raise ValueError("V7_RULE_BASELINE_CANNOT_MIX_MODEL_OR_CUSTOM_POLICY")
            from domain.les_v7.rule_baseline import baseline_policy, current_source_baseline_version
            policy = baseline_policy(market, **({"version": current_source_baseline_version(at)}
                if getattr(settings, "les_v7_source_repair", False) else {}))
            if timestamp(policy["available_at"]) > at:
                raise ValueError("V7_RULE_BASELINE_NOT_AVAILABLE_AT_CUTOFF")
            validate_quality_policy(policy["quality_policy"], at.isoformat(), market=market)
            if not settings.les_v7_native_evidence:
                raise ValueError("V7_DAILY_NATIVE_EVIDENCE_REQUIRED")
            result.update(state="READY", formal_entry_enabled=True,
                rule_baseline_policy_id=identity(policy),
                model_status="NOT_REQUIRED_RULE_BASED_BASELINE",
                prediction_status="NOT_APPLICABLE", calibration_status="NOT_APPLICABLE",
                probability_defaults=None, rule_baseline_policy=policy)
            result["source_repair_enabled"] = bool(getattr(settings, "les_v7_source_repair", False))
            return result
        model = FrozenBundle.load(json.loads(settings.les_v7_bundle_json)) if settings.les_v7_bundle_json else None
        policy = (StrategyPolicy.load(json.loads(settings.les_v7_strategy_policy_json), cutoff=at.isoformat())
                  if settings.les_v7_strategy_policy_json else StrategyPolicy.from_bundle(model) if model else None)
        if model:
            if timestamp(model.record()["trained_until"]) > at:
                raise ValueError("MODEL_NOT_AVAILABLE_AT_PREFLIGHT")
            result["model_sha256"] = model.sha256
            from domain.les_v7.fractals import fractal_history_requirements
            requirements = fractal_history_requirements(0)
            active = {name: requirements[name]["required_close_count"]
                      for name in model.record()["feature_names"] if name in requirements}
            if active:
                required = max(active.values())
                result["warmup"] = dict(required_valid_close_count=required,
                    active_feature_requirements=active,
                    suggested_calendar_days=1100 if required <= 518 else 2000,
                    policy="CALENDAR_WINDOW_IS_NOT_PROOF_OF_VALID_CLOSES",
                    note="700 calendar days may not supply 513/518 valid closes; native per-stock count/estimator checks remain mandatory")
        if policy:
            if timestamp(policy.record()["available_at"]) > at:
                raise ValueError("STRATEGY_POLICY_NOT_AVAILABLE_AT_PREFLIGHT")
            if model: policy.bind_bundle(model)
            quality = policy.record()["quality_policy"]
            if is_source_evidence_policy(quality):
                validated_quality = validate_quality_policy(quality, at.isoformat())
                if validated_quality.get("profile_id") not in QUALITY_REPORTED_PROFILES:
                    raise ValueError("V7_NATIVE_REPORTED_QUALITY_POLICY_REQUIRED")
            else:
                raise ValueError("V7_NATIVE_REPORTED_QUALITY_POLICY_REQUIRED")
            result["strategy_policy_sha256"] = policy.sha256
        if not model and not diagnostics:
            raise ValueError("V7_DAILY_MODEL_REQUIRED_USE_EXPLICIT_RESEARCH_OR_DIAGNOSTIC_MODE")
        if policy is None:
            raise ValueError("V7_STRATEGY_POLICY_REQUIRED")
        if not settings.les_v7_native_evidence:
            raise ValueError("V7_DAILY_NATIVE_EVIDENCE_REQUIRED")
        result.update(state="READY", formal_entry_enabled=not diagnostics)
        research_marker = getattr(settings, "les_v7_research_metadata_json", None)
        if research_marker:
            metadata = json.loads(research_marker)
            if (not isinstance(metadata, dict) or metadata.get("schema") != "les-v7-research-model-use/v1"
                    or model is None or metadata.get("bundle_sha256") != model.sha256
                    or metadata.get("research_only") is not True or metadata.get("approved_for_live") is not False
                    or metadata.get("order_authority") is not False or metadata.get("execution_action") != "NO_ACTION"
                    or not isinstance(metadata.get("symbols"), list) or not metadata["symbols"]
                    or len(metadata["symbols"]) > 2048 or len(set(metadata["symbols"])) != len(metadata["symbols"])
                    or any(not isinstance(symbol, str) or not symbol for symbol in metadata["symbols"])
                    or market != "TWSE" or diagnostics):
                raise ValueError("V7_RESEARCH_BUNDLE_PREFLIGHT_BINDING_MISMATCH")
            from domain.les_v7.fr2_policy import component as entry_component
            from domain.les_v7.fr2_execution import component as execution_component
            from domain.les_v7.valuation_price import policy as eps_price_policy
            result.update(mode="FR2_EPS_RESEARCH_ONLY", formal_entry_enabled=False,
                research_entry_enabled=True, prediction_bundle_unchanged=True,
                admission_component=entry_component(), execution_component=execution_component(),
                valuation_policy=eps_price_policy(), decision_thresholds=dict(BUY=75., BUY_SMALL=65.),
                return_basis="PRICE_RETURN_ONLY_NOT_TOTAL_RETURN")
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        result.update(state="BLOCKED", formal_entry_enabled=False)
        result["reason_codes"] = [str(error)]
    return result
