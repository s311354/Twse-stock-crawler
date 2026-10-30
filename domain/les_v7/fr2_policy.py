"""FR_ENTRY_R2: opt-in research admission, never a predictive-model rewrite.

The original native report is recoverable byte-for-business-byte. Its frozen
heads, calibration, factors and causal H20 plan remain authoritative. A new
signal identity describes this counterfactual policy, not historical deployment.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import json
import pandas as pd

from domain.les_v7.contracts import identity, canonical
from domain.les_v7.decision import score_once, vetoes

VERSION = "FR_ENTRY_R2"
PREDICTION_VERSION = "FROZEN_FIVE_HEAD_V1"
POLICY = dict(version=VERSION, buy_minimum=75., small_minimum=65., buy_quality_minimum=70.,
    small_quality_minimum=70., valuation_warning_points=5., warning_small_only=True,
    warning_reasons=["BASE_CAGR_BELOW_REQUIRED_RETURN", "BASE_FAIR_VALUE_NOT_ABOVE_CURRENT_PRICE"],
    preserve_context_trigger_localization=True, preserve_original_reference_plan=True)


def component():
    return dict(version=VERSION, component_id=identity(POLICY))


def classify(*, ready, hard, valuation_warning, quality, final, starter, confirmed,
             starter_gate, confirmed_gate, missing=False):
    """Single score-to-admission boundary; sizing is deliberately not an operand."""
    buy = bool(ready and not hard and not valuation_warning and quality["status"] == "PASS"
        and quality["quality_score"] >= POLICY["buy_quality_minimum"]
        and final is not None and final >= POLICY["buy_minimum"] and confirmed is True and confirmed_gate is True)
    small = bool(ready and not hard and quality["status"] == "PASS"
        and quality["quality_score"] >= POLICY["small_quality_minimum"]
        and final is not None and final >= POLICY["small_minimum"] and starter is True and starter_gate is True)
    if hard or quality["status"] == "REJECT": decision = "NO_BUY"
    elif not ready: decision = "DATA_INSUFFICIENT" if missing else "WATCH_ONLY"
    elif buy: decision = "BUY"
    elif small: decision = "BUY_SMALL"
    elif final is None: decision = "DATA_INSUFFICIENT"
    elif final < 55: decision = "AVOID"
    elif final < 65: decision = "WATCH"
    else: decision = "ARMED"
    return decision, buy, small


def _valuation_assessment(report):
    """Rebuild economics from the bound annual originals, never a reason whitelist alone."""
    from domain.les_v7.native_evidence import _valuation, _market_source
    d, e = report["diagnostics"], report["envelope"]
    source = d["native_evidence"]["source_packet"]
    base = d["gate_bindings"]["SAFETY"]
    rows = source["market_rows"]
    market_source = _market_source(source, symbol=base["symbol"], market=base["market"],
        cutoff=base["cutoff"], rows=rows)
    details = {}
    receipt = _valuation(source, symbol=base["symbol"], market=base["market"], cutoff=base["cutoff"],
        features=d["features"], quality=d["quality"], bars=pd.DataFrame(rows),
        binding=d["gate_bindings"]["VALUATION"], market_source=market_source, assessment_sink=details)
    if receipt != d["native_evidence"]["produced_receipts"].get("VALUATION"):
        raise ValueError("FR2_VALUATION_SOURCE_RECOMPUTATION_MISMATCH")
    snapshot = details.get("snapshot", {})
    reasons = set((receipt or {}).get("reason_codes", ()))
    warning = bool(receipt and receipt["state"] == "FAIL" and details.get("state") == "COMPLETE"
        and reasons and reasons <= set(POLICY["warning_reasons"])
        and snapshot.get("valuation_model_status") == "COMPLETE"
        and snapshot.get("bear_case_permanent_loss_acceptable") is True
        and snapshot.get("cycle_valuation_review_required") is not True)
    return dict(schema="fr2-verified-valuation/v1", native_state=(receipt or {}).get("state", "UNKNOWN"),
        warning_eligible=warning, selected_annual_ids=details.get("selected_annual_ids", []),
        scenario_id=identity(snapshot), receipt_id=identity(receipt),
        reason_codes=sorted(reasons), snapshot=snapshot)


def _derive(original, model_bundle, *, valuation_inputs=None):
    d, e = original["diagnostics"], original["envelope"]
    if (e["model"]["model_status"] != "OFFLINE_FITTED" or not e["model"]["bundle_sha256"]
            or d.get("rule_baseline") is not None or d.get("run_scope") is not None
            or d.get("native_evidence") is None or not d.get("strategy_policy")):
        raise ValueError("FR2_VERIFIED_NATIVE_FROZEN_MODEL_REPORT_REQUIRED")
    if d["native_evidence"].get("valuation_policy") is not None:
        raise ValueError("FR2_ORIGINAL_ANNUAL_VALUATION_POLICY_REQUIRED")
    _verify_trained_calculations(original, model_bundle)
    gates = {g["gate_id"]: g for g in d["gate_receipts"]}
    effective_valuation = None
    if valuation_inputs is None:
        valuation = _valuation_assessment(original)
    else:
        from dataclasses import asdict
        from domain.les_v7.contracts import GateReceipt
        from domain.les_v7.fr2_eps_valuation import assess_eps_valuation
        assessed = assess_eps_valuation(original, valuation_inputs)
        valuation = assessed["assessment"]
        effective_valuation = json.loads(canonical(asdict(GateReceipt.bind(assessed["receipt"], gate_id="VALUATION",
            inputs=d["gate_bindings"]["VALUATION"], cutoff=e["event"]["cutoff_at"]))))
        # Only this versioned research component changes. The native receipt,
        # source packet, predictor and every other safety gate stay untouched.
        gates["VALUATION"] = effective_valuation
    warning = valuation["warning_eligible"]
    val_ok = gates["VALUATION"]["state"] == "PASS" or warning
    other_ok = all(g["state"] == "PASS" for key, g in gates.items() if key != "VALUATION")
    hard = list(vetoes(d["features"]))
    hard.extend(key + "_FAIL" for key, g in gates.items() if g["state"] == "FAIL" and (key != "VALUATION" or not warning))
    # EXECUTION_FAIL also covers the unchanged plan/stop/RR/liquidity policy.
    if "EXECUTION_FAIL" in e["strategy"]["reason_codes"]:
        hard.append("EXECUTION_FAIL")
    penalties = dict(d["penalties"])
    if warning:
        penalties["valuation_warning"] = POLICY["valuation_warning_points"]  # exactly once
    caps = {k: v for k, v in d["safety_caps"].items() if k != "hard_veto"}
    if hard: caps["hard_veto"] = 0.
    score = e["score"]
    scored = score_once(score["factor_values"], score["applicable_weights"], penalties, caps)
    quality = d["quality"]
    daily_errors = bool(valuation_inputs is not None and d.get("daily_input_errors"))
    ready = bool(val_ok and other_ok and quality["status"] == "PASS"
        and d["localization"]["status"] == "VALID" and d["reference_plan"] is not None
        and d["starter"] is not None and d["confirmed"] is not None and not daily_errors)
    final = scored["final_score"] if ready else None
    decision, buy, small = classify(ready=ready, hard=hard, valuation_warning=warning,
        quality=quality, final=final, starter=d["starter"], confirmed=d["confirmed"],
        starter_gate=d["starter_gate"], confirmed_gate=d["confirmed_gate"],
        missing=daily_errors or not other_ok or not val_ok or quality["status"] in {"UNKNOWN", "CONFIG_INCOMPLETE", "DATA_INSUFFICIENT"}
            or d["localization"]["status"] in {"NOT_READY", "INVALID"})
    admission = dict(version=VERSION, policy_id=identity(POLICY), policy=deepcopy(POLICY),
        original_report_id=identity(original), evidence_id=e["event"]["evidence_id"],
        cutoff=e["event"]["cutoff_at"], bundle_sha256=e["model"]["bundle_sha256"],
        quality_result_id=e["quality"]["quality_result_id"], valuation=valuation,
        valuation_warning=warning, valuation_admission="ALLOWED_WITH_WARNING" if warning else gates["VALUATION"]["state"],
        buy_allowed=buy, small_allowed=small, decision=decision, final_score=final,
        hard_vetoes=sorted(set(hard)), scope="COUNTERFACTUAL_RESEARCH_ONLY_NOT_DEPLOYMENT")
    admission["assessment_id"] = identity(admission)
    return admission, dict(scored, final_score=final), penalties, caps, effective_valuation


def _verify_trained_calculations(report, model_bundle):
    """Do not trust an old non-entry report's booleans to authorize a new entry."""
    from dataclasses import fields
    from domain.les_v7.localization import Forecast, FrozenBundle, predict, localization_gate
    from domain.les_v7.decision import factor_vector
    from domain.les_v7.plan_r19 import plan_from_record
    d, e = report["diagnostics"], report["envelope"]
    bundle = FrozenBundle.load(model_bundle)
    from domain.les_v7.strategy_policy import StrategyPolicy
    StrategyPolicy.load(d["strategy_policy"], cutoff=e["event"]["cutoff_at"]).bind_bundle(bundle)
    if bundle.sha256 != e["model"]["bundle_sha256"]:
        raise ValueError("FR2_ORIGINAL_PREDICTOR_ID_MISMATCH")
    raw_plan, f = d["reference_plan"], d["features"]
    plan = plan_from_record(raw_plan) if raw_plan else None
    forecast = Forecast(**{field.name: d["localization"][field.name] for field in fields(Forecast)})
    setup = e["event"]["context"]
    prediction_features = f
    from domain.les_v7.execution_profile import validate_execution_profile
    partition = validate_execution_profile(bundle.record())
    if partition is not None:
        # Native validation recomputes these inputs from retained cutoff-only
        # history.  Only the predictor sees the extension; all factors, plans,
        # valuation and execution gates below retain their original base inputs.
        from domain.les_v7.native_evidence import validate_native_receipts
        validate_native_receipts(report)
        extension = d.get("execution_features")
        if not isinstance(extension, dict) or set(extension) != set(partition[1]):
            raise ValueError("FR2_EXECUTION_FEATURE_PROFILE_MISMATCH")
        prediction_features = dict(f, **extension)
    expected_forecast = predict(bundle, prediction_features, plan, cutoff=e["event"]["cutoff_at"])
    if (canonical(expected_forecast.record()) != canonical(d["localization"])
            or canonical(bundle.record().get("weights", {}).get(setup)) != canonical(e["score"]["applicable_weights"])):
        raise ValueError("FR2_FROZEN_PREDICTOR_RECOMPUTATION_MISMATCH")
    policy = d["strategy_policy"]["payload"]["reference_policy"]
    minimum = 2. if setup == "REVERSAL" else 1.8 if setup == "TREND_PULLBACK" else policy.get("leadership_rr_min")
    gates = {g["gate_id"]: g["state"] for g in d["gate_receipts"]}
    known = bool(plan and f.get("adv") and f["adv"] > 0 and f.get("spread") is not None)
    liquid = bool(known and f["spread"] <= .005 and plan.entry * plan.reference_shares / f["adv"] <= .01)
    standard = bool(plan and liquid and gates.get("STRESS_RISK") == "PASS" and plan.net_rr is not None
        and minimum is not None and plan.net_rr >= minimum and plan.stop_atr <= 1.5)
    starter = bool(standard and plan.stop_atr <= (1.2 if setup == "REVERSAL" else 1.3)
        and localization_gate(forecast, plan, confirmed=False))
    confirmed = bool(standard and localization_gate(forecast, plan, confirmed=True))
    if (d["starter_gate"], d["confirmed_gate"]) != (starter, confirmed):
        raise ValueError("FR2_ORIGINAL_EXECUTION_RECOMPUTATION_MISMATCH")
    if plan and forecast.status == "VALID":
        factors, ledger = factor_vector(setup, f, forecast, plan, leadership_rr_min=minimum)
        if (canonical(factors) != canonical(e["score"]["factor_values"])
                or canonical(ledger) != canonical(d["factor_ledger"])):
            raise ValueError("FR2_ORIGINAL_FACTOR_RECOMPUTATION_MISMATCH")


def _apply_validated(original, model_bundle, *, valuation_inputs=None):
    from domain.les_v7.fr2_execution import component as execution_component
    admission, scored, penalties, caps, effective_valuation = _derive(original, model_bundle,
        valuation_inputs=valuation_inputs)
    result = deepcopy(original)
    d, e = result["diagnostics"], result["envelope"]
    # Small reversible overlay, not another copy of the financial evidence graph.
    previous = dict(score=deepcopy(e["score"]), strategy=deepcopy(e["strategy"]),
        signal_id=e["event"]["signal_id"], diagnostic_score=deepcopy(d["diagnostic_score"]),
        penalties=deepcopy(d["penalties"]), safety_caps=deepcopy(d["safety_caps"]),
        readiness_execution=deepcopy(d.get("readiness", {}).get("execution")))
    d["research_components"] = dict(prediction=dict(version=PREDICTION_VERSION,
        component_id=e["model"]["bundle_sha256"]), entry=component(), execution=execution_component())
    if valuation_inputs is not None:
        from domain.les_v7.valuation_price import POLICY_VERSION
        assessment = admission["valuation"]
        previous["gate_receipts"] = deepcopy(d["gate_receipts"])
        d["research_components"]["valuation"] = dict(version=POLICY_VERSION,
            component_id=assessment["valuation_policy_id"],
            input_policy_id=assessment["valuation_input_policy_id"],
            valuation_inputs_id=assessment["valuation_inputs_id"],
            receipt_id=assessment["receipt_id"], return_basis="PRICE_RETURN_ONLY")
        d["gate_receipts"] = [effective_valuation if g["gate_id"] == "VALUATION" else g
                              for g in d["gate_receipts"]]
    d["fr2"] = dict(schema="les-v7-fr2-overlay/v1", original=previous,
        predictor_bundle=deepcopy(model_bundle),
        original_report_id=identity(original), admission=admission,
        components_id=identity(d["research_components"]), order_authority=False)
    if valuation_inputs is not None:
        d["fr2"].update(schema="les-v7-fr2-eps-overlay/v1",
            valuation_inputs=deepcopy(valuation_inputs))
    d.update(diagnostic_score=deepcopy(scored), penalties=penalties, safety_caps=caps)
    e["score"].update(scored)
    reasons = e["strategy"]["reason_codes"]
    if valuation_inputs is not None:
        old_valuation = next(g for g in previous["gate_receipts"] if g["gate_id"] == "VALUATION")
        # Retire reasons emitted by the superseded method, but do not erase an
        # identically named reason still supplied by another safety component.
        retired = set(old_valuation["reasons"]) | {"VALUATION_" + old_valuation["state"]}
        retained = set(d["quality"]["reason_codes"]) | set(d["localization"]["reasons"])
        retained.update(reason for g in d["gate_receipts"] if g["gate_id"] != "VALUATION"
                        for reason in g["reasons"])
        reasons = [r for r in reasons if r not in retired or r in retained]
        reasons += list(effective_valuation["reasons"]) + [POLICY_VERSION]
        if effective_valuation["state"] != "PASS":
            reasons.append("VALUATION_" + effective_valuation["state"])
    e["strategy"].update(strategy_decision=admission["decision"], decision_policy_version=VERSION,
        reason_codes=sorted(set(reasons + [VERSION]
            + (["VALUATION_WARNING_SMALL_ONLY"] if admission["valuation_warning"] else []))))
    e["event"]["signal_id"] = identity(dict(evidence_id=e["event"]["evidence_id"],
        bundle_hash=e["model"]["bundle_sha256"], components_id=identity(d["research_components"]),
        decision_policy_version=VERSION))
    if "readiness" in d:
        d["readiness"]["execution"].update(state="READY_FOR_ACCOUNT_CHECK" if admission["decision"] in {"BUY", "BUY_SMALL"} else "BLOCKED",
            candidate_decision=admission["decision"])
        if valuation_inputs is not None:
            d["readiness"]["execution"]["required_gates"] = {g["gate_id"]: g["state"] for g in d["gate_receipts"]}
    return result


def apply_fr2(original, model_bundle):
    """Public admission entry: validates once; no cash/position authority is inferred."""
    from domain.les_v7.report import validate_report
    if "pae_authorization" in original or "fr2_execution" in original or "fr2" in original.get("diagnostics", {}):
        raise ValueError("FR2_UNALLOCATED_ORIGINAL_REPORT_REQUIRED")
    validate_report(original)
    return _apply_validated(original, model_bundle)


def apply_fr2_eps(original, model_bundle, valuation_inputs):
    """Opt-in price-only valuation plus FR2; never change the frozen bundle.

    Supplemental facts must bind to the exact cutoff-visible annual originals.
    An independently identified overlay retains the entire native result for
    restoration and validates every business field by recomputation.
    """
    from domain.les_v7.report import validate_report
    if "pae_authorization" in original or "fr2_execution" in original or "fr2" in original.get("diagnostics", {}):
        raise ValueError("FR2_UNALLOCATED_ORIGINAL_REPORT_REQUIRED")
    validate_report(original)
    return _apply_validated(original, model_bundle, valuation_inputs=valuation_inputs)


def restore_original(report):
    result = deepcopy(report)
    result.pop("fr2_execution", None)
    if "pae_authorization" in result: raise ValueError("FR2_LEGACY_AUTHORIZATION_FORBIDDEN")
    d, e = result["diagnostics"], result["envelope"]
    overlay = d.pop("fr2")
    d.pop("research_components")
    before = overlay["original"]
    e["score"], e["strategy"] = before["score"], before["strategy"]
    e["event"]["signal_id"] = before["signal_id"]
    for key in ("diagnostic_score", "penalties", "safety_caps"): d[key] = before[key]
    if "gate_receipts" in before: d["gate_receipts"] = before["gate_receipts"]
    if "readiness" in d: d["readiness"]["execution"] = before["readiness_execution"]
    if identity(result) != overlay["original_report_id"]:
        raise ValueError("FR2_ORIGINAL_REPORT_ID_MISMATCH")
    return result


def validate_fr2_report(report):
    """Independently reconstruct original proof, admission, arithmetic and allocation."""
    from domain.les_v7.report import validate_report
    original = restore_original(report)
    validate_report(original)
    overlay = report["diagnostics"]["fr2"]
    if overlay["schema"] not in {"les-v7-fr2-overlay/v1", "les-v7-fr2-eps-overlay/v1"}:
        raise ValueError("FR2_OVERLAY_SCHEMA_UNKNOWN")
    eps = overlay["schema"] == "les-v7-fr2-eps-overlay/v1"
    if eps != ("valuation_inputs" in overlay):
        raise ValueError("FR2_VALUATION_OVERLAY_INPUTS_REQUIRED")
    expected = _apply_validated(original, overlay["predictor_bundle"],
        valuation_inputs=overlay.get("valuation_inputs") if eps else None)
    actual = {k: v for k, v in report.items() if k != "fr2_execution"}
    if canonical(actual) != canonical(expected): raise ValueError("FR2_REPORT_RECOMPUTATION_MISMATCH")
    if "fr2_execution" in report:
        from domain.les_v7.fr2_execution import validate_execution_assessment
        validate_execution_assessment(report, report["diagnostics"]["fr2"]["admission"],
            report["fr2_execution"], validate_native=False)
