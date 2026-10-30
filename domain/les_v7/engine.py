"""Pure v7 daily composition. External receipts are bound ports, never UI switches."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from collections.abc import Mapping
import json
import pandas as pd

from domain.features import FeatureFrame
from domain.les_v7.contracts import (SPEC_ID, SPEC_SHA256, FEATURE_VERSION, MAPPING_VERSION,
                                    FACTORS, GateReceipt, canonical, identity, timestamp,
                                    QUALITY_REPORTED_PROFILES, parameter_version, quality_extension_schema,
                                    envelope_schema_version)
from domain.les_v7.features import select_bars, compute_features
from domain.les_v7.quality import evaluate_financials
from domain.les_v7.localization import FrozenBundle, predict, localization_gate, model_input_profile
from domain.les_v7.decision import select_context, make_plan, triggers, vetoes, factor_vector, score_once, decide
from domain.les_v7.strategy_policy import StrategyPolicy, policy_blocks_id
from domain.les_v7.policy_r13 import BASELINE_VERSION as R13_BASELINE_VERSION, risk_limits
from domain.les_v7.policy_r14 import BASELINE_VERSION as R14_BASELINE_VERSION
from domain.les_v7.policy_r15 import BASELINE_VERSION as R15_BASELINE_VERSION
from domain.les_v7.policy_r16 import BASELINE_VERSION as R16_BASELINE_VERSION
from domain.les_v7.policy_r17 import BASELINE_VERSION as R17_BASELINE_VERSION
from domain.les_v7.policy_r18 import BASELINE_VERSION as R18_BASELINE_VERSION
from domain.les_v7.policy_r19 import BASELINE_VERSION as R19_BASELINE_VERSION
from domain.les_v7.rule_baseline import (prepare_baseline, baseline_triggers, score_baseline,
    localization_record, baseline_decision_policy, GRADED_POLICIES, RECOVERY_POLICIES,
    INDUSTRY_POLICIES, RISK_BOUNDED_POLICIES, LIMIT_POLICIES, CONDITIONAL_POLICIES,
    ENVELOPE_SCHEMA as BASELINE_ENVELOPE_SCHEMA)


def evaluate(data: pd.DataFrame, context: Mapping, *, shared: FeatureFrame | None = None,
             receipt_producer=None) -> dict:
    # Mutable public callers retain complete isolation. Native daily packets
    # implement immutable deepcopy and share source-bound proof nodes instead;
    # cutoff selection/receipt validation below still execute on every call.
    context = deepcopy(context)
    if "diagnostics_only" in context and type(context["diagnostics_only"]) is not bool:
        raise ValueError("V7_DIAGNOSTIC_SCOPE_BOOLEAN_REQUIRED")
    if "rule_baseline" in context and type(context["rule_baseline"]) is not bool:
        raise ValueError("V7_RULE_BASELINE_BOOLEAN_REQUIRED")
    rule_baseline = context.get("rule_baseline") is True
    diagnostics_only = context.get("diagnostics_only") is True
    if rule_baseline and diagnostics_only:
        raise ValueError("V7_RULE_BASELINE_CANNOT_MIX_DIAGNOSTICS_ONLY")
    if not rule_baseline and "rule_baseline_policy_version" in context:
        raise ValueError("V7_BASELINE_POLICY_VERSION_WITHOUT_RULE_MODE")
    decision_policy_version = "IMP_STATE_01"
    if diagnostics_only: decision_policy_version += ":DIAGNOSTICS_ONLY_R1"
    cutoff = timestamp(context["as_of"]).isoformat()
    symbol, market = context["symbol"], context["market"]
    if (not isinstance(symbol, str) or not symbol.strip() or symbol != symbol.strip()
            or market not in {"TWSE", "TPEX"}): raise ValueError("EXPLICIT_SECURITY_AND_MARKET_REQUIRED")
    bars = select_bars(data, cutoff)
    for column in ("Symbol", "symbol", "stock_no", "證券代號"):
        if column in bars and not bars[column].map(str).eq(symbol).all():
            raise ValueError("BAR_SECURITY_IDENTITY_MISMATCH")
    if "Market" in bars and not bars["Market"].eq(market).all():
        raise ValueError("BAR_MARKET_IDENTITY_MISMATCH")
    filings = [r for r in context.get("filings", []) if timestamp(r["available_at"]) <= timestamp(cutoff)]
    market_rows = bars.astype(object).where(pd.notna(bars), None).to_dict("records")
    # Dates are canonical strings; evidence IDs are independent of future bars and weights.
    for row in market_rows:
        row["Date"] = pd.Timestamp(row["Date"]).date().isoformat()
    supplied_model = context.get("model_bundle")
    bundle = FrozenBundle.load(supplied_model) if supplied_model is not None else None
    model = bundle.record() if bundle is not None else {}
    explicit_policy = context.get("strategy_policy")
    if rule_baseline and (supplied_model is not None or explicit_policy is not None):
        raise ValueError("V7_RULE_BASELINE_CANNOT_MIX_TRAINED_OR_EXTERNAL_POLICY")
    native_source = json.loads(canonical(context.get("native_evidence_source", {}))) if receipt_producer is not None else None
    industry_input = None
    if rule_baseline and context.get("rule_baseline_policy_version") in INDUSTRY_POLICIES:
        if native_source is None:
            raise ValueError("V7_R11_NATIVE_SOURCE_REQUIRED")
        industry_input = native_source.setdefault("industry_input", dict(symbol=symbol, sector="", benchmark=None))
        if not isinstance(industry_input, Mapping) or industry_input.get("symbol") != symbol:
            raise ValueError("V7_INDUSTRY_SECURITY_BINDING_MISMATCH")
    baseline = prepare_baseline(bars, market=market, cutoff=cutoff, industry_input=industry_input,
        session_evidence=(native_source or {}).get("calendar_evidence"),
        **({"policy_version": context["rule_baseline_policy_version"]} if "rule_baseline_policy_version" in context else {})) if rule_baseline else None
    if baseline is not None: decision_policy_version = baseline_decision_policy(baseline["policy"])
    strategy_policy = StrategyPolicy.load(explicit_policy, cutoff=cutoff) if explicit_policy is not None else None
    if receipt_producer is not None and strategy_policy is None and bundle is not None:
        strategy_policy = StrategyPolicy.load(StrategyPolicy.from_bundle(bundle).export(), cutoff=cutoff)
    if strategy_policy is not None and bundle is not None: strategy_policy.bind_bundle(bundle)
    policy = baseline["policy"] if baseline else strategy_policy.record() if strategy_policy is not None else model
    feature_version = policy.get("native_feature_version", FEATURE_VERSION) if baseline else FEATURE_VERSION
    if "source_policy" in policy:
        from domain.les_v7.native_evidence import produce_native_receipts
        if receipt_producer is not produce_native_receipts or context.get("receipts"):
            raise ValueError("V7_R6_NATIVE_SOURCE_PRODUCER_REQUIRED_NO_PREBOUND_RECEIPTS")
    quality_policy = context.get("quality_policy")
    if baseline is not None:
        if quality_policy is not None and identity(quality_policy) != identity(policy["quality_policy"]):
            raise ValueError("V7_BASELINE_QUALITY_POLICY_MISMATCH")
        quality_policy = policy["quality_policy"]
    if strategy_policy is not None:
        if quality_policy is not None and identity(quality_policy) != identity(policy["quality_policy"]):
            raise ValueError("QUALITY_POLICY_STRATEGY_MISMATCH")
        quality_policy = policy["quality_policy"]
    base_binding = dict(symbol=symbol, market=market, cutoff=cutoff,
                        bars_hash=identity(market_rows), financials_hash=identity(filings))
    source = dict(symbol=symbol, market=market, cutoff=cutoff, bars=market_rows, filings=filings,
                  data_rules=feature_version, external_evidence=context.get("external_evidence_ids", []),
                  quality_policy=context.get("quality_policy"))
    evidence_binding = None
    if strategy_policy is not None or receipt_producer is not None or baseline is not None:
        evidence_binding = dict(schema="les-v7-evidence-binding/v2", **base_binding,
            data_rules=feature_version, external_evidence_ids=list(context.get("external_evidence_ids", ())),
            quality_policy_id=identity(quality_policy), strategy_policy_id=strategy_policy.sha256 if strategy_policy else None,
            native_source_id=identity(native_source) if native_source is not None else None)
        if baseline is not None: evidence_binding["rule_baseline_policy_id"] = baseline["policy_id"]
    evidence_id = identity(evidence_binding if evidence_binding is not None else source)
    if shared is not None and len(bars) != len(data): shared = None
    f, feature_audit = compute_features(bars, shared=shared, shadow=context.get("shadow_fractals") is True,
        **(dict(feature_version=feature_version) if feature_version != FEATURE_VERSION else {}))
    receipts = context.get("receipts", {})
    safety = GateReceipt.bind(receipts.get("SAFETY"), gate_id="SAFETY", inputs=base_binding, cutoff=cutoff)
    # The supplied policy may retain frozen tuple arrays even though the model
    # loader returns standard JSON lists. Compare the same JSON representation;
    # a real policy value change must still fail the bundle binding.
    if bundle is not None and json.loads(canonical(quality_policy)) != model["quality_policy"]:
        raise ValueError("QUALITY_POLICY_BUNDLE_MISMATCH")
    quality = evaluate_financials(filings, symbol=symbol, market=market, cutoff=cutoff,
                                 safety_receipt=safety, policy=quality_policy)
    valuation_binding = dict(base_binding, price=f.get("close"), quality_result_id=identity(quality))
    selected, matched = ((baseline["selection"]["selected_setup"], baseline["selection"]["matches"])
                         if baseline else select_context(f, policy.get("context_rules")))
    structure = None
    is_r10 = baseline is not None and policy["policy_version"] in RECOVERY_POLICIES
    if is_r10:
        from domain.les_v7.plan_r10 import make_plan as structure_plan
        plan, structure = structure_plan(selected, f, policy["reference_policy"], evidence_id, bars=bars, cutoff=cutoff,
            **(dict(session_evidence=(native_source or {}).get("calendar_evidence"))
               if policy["policy_version"] in CONDITIONAL_POLICIES else {}))
    else:
        plan = make_plan(selected, f, policy.get("reference_policy"), evidence_id)
    # Execution inputs affect only the declared fill-head extension. Quality,
    # plans, triggers, gate receipts and the original feature audit stay causal
    # base computations; their historical contract is not silently replaced.
    prediction_features, execution_inputs, execution_audit = f, None, None
    if bundle is not None:
        from domain.les_v7.execution_profile import validate_execution_profile, execution_feature_provider
        partition = validate_execution_profile(model)
        if partition is not None:
            provider = execution_feature_provider(model)
            extra, execution_audit = provider._from_selected_bars(bars, base_features=f, plan=plan)
            execution_inputs = {name: extra[name] for name in partition[1]}
            prediction_features = dict(f, **execution_inputs)
    forecast = predict(bundle, prediction_features, plan, cutoff=cutoff)
    trigger = baseline_triggers(baseline) if baseline else triggers(selected, f, policy.get("trigger_rules"))
    risks = list(vetoes(f))
    risk_binding = dict(base_binding, plan=asdict(plan) if plan else None)
    bindings = dict(SAFETY=base_binding, DATA_INTEGRITY=base_binding, TRADING_STATUS=base_binding,
                    VALUATION=valuation_binding, STRESS_RISK=risk_binding, PRICE_DISCOVERY_PAYOFF=risk_binding)
    produced = None
    if receipt_producer is not None:
        if quality.get("profile_id") not in QUALITY_REPORTED_PROFILES:
            raise ValueError("NATIVE_RECEIPTS_REQUIRE_INDEPENDENT_REPORTED_QUALITY")
        # A pure producer sees one detached cutoff selection, not the mutable
        # caller graph. No second feature/Quality/context/plan calculation.
        producer_bars = bars.copy(deep=True)
        producer_bars.attrs.clear()
        produced = receipt_producer(symbol=symbol, market=market, cutoff=cutoff, bars=producer_bars,
            features=json.loads(canonical(f)), quality=json.loads(canonical(quality)),
            plan=asdict(plan) if plan else None, risks=tuple(risks),
            bindings=json.loads(canonical(bindings)), source_packet=deepcopy(native_source),
            reference_policy=json.loads(canonical(policy.get("reference_policy"))),
            **({"valuation_policy": json.loads(canonical(policy["valuation_policy"]))}
               if "valuation_policy" in policy else {}),
            **({"valuation_input_policy": json.loads(canonical(policy["valuation_input_policy"]))}
               if "valuation_input_policy" in policy else {}),
            **({"source_policy": json.loads(canonical(policy["source_policy"]))}
               if "source_policy" in policy else {}),
            **(dict(feature_version=feature_version) if feature_version != FEATURE_VERSION else {}))
        if not isinstance(produced, Mapping) or set(produced) - set(bindings):
            raise ValueError("NATIVE_RECEIPT_PRODUCER_RESULT_INVALID")
        merged = dict(receipts)
        for gate, receipt in produced.items():
            if gate in merged and identity(merged[gate]) != identity(receipt):
                raise ValueError("NATIVE_RECEIPT_CONFLICT:" + gate)
            merged[gate] = receipt
        receipts = merged
        safety = GateReceipt.bind(receipts.get("SAFETY"), gate_id="SAFETY", inputs=base_binding, cutoff=cutoff)
    valuation = GateReceipt.bind(receipts.get("VALUATION"), gate_id="VALUATION", inputs=valuation_binding, cutoff=cutoff)
    integrity = GateReceipt.bind(receipts.get("DATA_INTEGRITY"), gate_id="DATA_INTEGRITY", inputs=base_binding, cutoff=cutoff)
    restrictions = GateReceipt.bind(receipts.get("TRADING_STATUS"), gate_id="TRADING_STATUS", inputs=base_binding, cutoff=cutoff)
    risk = GateReceipt.bind(receipts.get("STRESS_RISK"), gate_id="STRESS_RISK", inputs=risk_binding, cutoff=cutoff)
    gates = [integrity, restrictions, safety, valuation, risk]
    if any(g.state == "FAIL" for g in gates): risks.extend(g.gate_id + "_FAIL" for g in gates if g.state == "FAIL")
    participation = plan.entry * plan.reference_shares / f["adv"] if plan and f.get("adv") and f["adv"] > 0 else None
    liquidity_known = f.get("spread") is not None and participation is not None
    liquidity = liquidity_known and f["spread"] <= .005 and participation <= .01
    if liquidity_known and not liquidity: risks.append("EXECUTION_FAIL")
    is_r17 = baseline is not None and policy["policy_version"] == R17_BASELINE_VERSION
    is_r19 = baseline is not None and policy["policy_version"] in RISK_BOUNDED_POLICIES
    is_r18 = baseline is not None and policy["policy_version"] in LIMIT_POLICIES
    is_r16 = baseline is not None and policy["policy_version"] == R16_BASELINE_VERSION
    is_r15 = baseline is not None and policy["policy_version"] in {R15_BASELINE_VERSION, *CONDITIONAL_POLICIES}
    is_r13 = baseline is not None and policy["policy_version"] in {R13_BASELINE_VERSION, R14_BASELINE_VERSION, R15_BASELINE_VERSION, *CONDITIONAL_POLICIES}
    rr_min = (risk_limits()["minimum_net_rr"].get(selected) if is_r13 else
        2. if selected == "REVERSAL" else 1.8 if selected == "TREND_PULLBACK" else policy.get("reference_policy", {}).get("leadership_rr_min"))
    payoff = GateReceipt.bind(receipts.get("PRICE_DISCOVERY_PAYOFF"), gate_id="PRICE_DISCOVERY_PAYOFF", inputs=risk_binding, cutoff=cutoff)
    discovery = (selected == "LEADERSHIP_CONTINUATION" and plan is not None and plan.target is None
                 and policy.get("reference_policy", {}).get("price_discovery_policy_id") is not None and payoff.state == "PASS")
    execution_assessment = None
    if is_r18:
        from domain.les_v7.risk_r18 import assess_execution, require_assessment
        if is_r19:
            from domain.les_v7.risk_r19 import assess_execution, require_assessment
        execution_assessment = ((produced or {}).get("STRESS_RISK") or {}).get("execution_assessment")
        if execution_assessment is None:
            if plan is not None and risk.state == "PASS":
                raise ValueError("R18_EXECUTION_ASSESSMENT_REQUIRED")
            execution_assessment = assess_execution(plan, policy["reference_policy"], f)
        execution_assessment = require_assessment(execution_assessment, plan, policy["reference_policy"], f)
        close_risk = execution_assessment["tactical"]["close"]
        if execution_assessment["state"] == "FAIL": risks.append("EXECUTION_FAIL")
        standard = bool(plan and execution_assessment["state"] == "PASS" and risk.state == "PASS")
    elif is_r13:
        from domain.les_v7.plan_r10 import evaluate_entry
        close_risk = evaluate_entry(plan, plan.entry if plan else None, policy["reference_policy"],
            **(dict(starter=True) if is_r15 else {}))
        if close_risk["state"] == "FAIL": risks.append("EXECUTION_FAIL")
        standard = bool(plan and liquidity and risk.state == "PASS" and close_risk["state"] == "PASS")
    else:
        if plan and (plan.stop_atr > 1.5 or plan.net_rr is not None and rr_min is not None and plan.net_rr < rr_min):
            risks.append("EXECUTION_FAIL")
        standard = bool(plan and liquidity and risk.state == "PASS" and (discovery or plan.net_rr is not None and rr_min is not None and plan.net_rr >= rr_min) and plan.stop_atr <= 1.5)
    starter_distance = 1.2 if selected == "REVERSAL" else 1.3
    starter_gate = bool(standard and plan.stop_atr <= starter_distance and localization_gate(forecast, plan, confirmed=False))
    confirmed_gate = bool(standard and localization_gate(forecast, plan, confirmed=True))
    if baseline is not None:
        starter_gate = bool(standard and plan.stop_atr <= starter_distance)
        confirmed_gate = bool(standard)
    if is_r18:
        starter_gate = bool(standard and execution_assessment["tactical"]["state"] == "PASS")
        confirmed_gate = bool(standard and execution_assessment["core"]["state"] == "PASS")
    elif is_r10 and plan is not None:
        from domain.les_v7.plan_r10 import validate_band
        confirmed_gate = bool(standard and validate_band(plan, policy["reference_policy"])["state"] == "PASS")
        starter_gate = bool(standard and validate_band(plan, policy["reference_policy"], starter=True)["state"] == "PASS")
    factors, factor_audit = (factor_vector(selected, f, forecast, plan, leadership_rr_min=rr_min, price_discovery=discovery)
                             if plan and forecast.status == "VALID" else (None, {}))
    weights = model.get("weights", {}).get(selected)
    penalties = {}
    # Penalties require bound, positive evidence, not missing defaults.
    if valuation.state == "PASS" and "NEAR_VALUE_UPPER_BOUND" in valuation.reasons: penalties["valuation_warning"] = 5.
    if "RISK_OFF_RECOVERY" in integrity.reasons: penalties["market"] = 5.
    caps = {}
    if trigger["confirmed"] is not True: caps["confirmation"] = 79.
    if forecast.p_entry is not None and .5 <= forecast.p_entry < .6: caps["localization"] = 79.
    if forecast.status != "VALID" or selected == "NO_CLEAR_CONTEXT": caps["model_or_context"] = 71.
    if risks: caps["hard_veto"] = 0.
    required_valid = bool(plan and liquidity_known and all(g.state == "PASS" for g in gates)
                          and trigger["starter"] is not None and trigger["confirmed"] is not None)
    scored = score_once(factors, weights, penalties, caps) if weights is not None else score_once(None, (), {}, {})
    # Formal null is retained for mandatory evidence failure, not a fake zero score.
    formal = scored["final_score"] if required_valid and quality["status"] == "PASS" and not diagnostics_only else None
    decision = decide(quality=quality, valuation=valuation.state, context=selected, forecast=forecast,
        final=formal, hard_veto=bool(risks), required_valid=required_valid,
        starter_trigger=trigger["starter"], confirmed_trigger=trigger["confirmed"],
        starter_gate=starter_gate, confirmed_gate=confirmed_gate)
    if (not risks and bundle is not None and selected == "NO_CLEAR_CONTEXT" and quality["status"] == "PASS"
            and all(g.state == "PASS" for g in (integrity, restrictions, safety, valuation))):
        decision = "WATCH_ONLY"  # No global head/plan is invented for ambiguous context.
    if (not risks and bundle is not None and selected == "LEADERSHIP_CONTINUATION" and plan and plan.target is None
            and not discovery and quality["status"] == "PASS" and forecast.status == "VALID"
            and all(g.state == "PASS" for g in (integrity, restrictions, safety, valuation))):
        decision = "WATCH_ONLY"
    baseline_outcome = None
    if baseline is not None:
        baseline_outcome = score_baseline(baseline, plan=plan, quality=quality, gates=gates,
            native_features=f, risks=risks, standard=standard,
            starter_gate=starter_gate, confirmed_gate=confirmed_gate,
            **(dict(valuation_receipt=produced.get("VALUATION") if produced else None, entry_binding=base_binding)
               if policy["policy_version"] in GRADED_POLICIES else {}),
            **(dict(execution_assessment=execution_assessment) if is_r18 else {}))
        scored = baseline_outcome["score"]
        factors, weights = baseline_outcome["factors"], baseline_outcome["weights"]
        penalties, caps, factor_audit = (baseline_outcome["penalties"], baseline_outcome["caps"],
                                        baseline_outcome["factor_ledger"])
        formal = None if diagnostics_only else scored["final_score"]
        decision = baseline_outcome["decision"]
    if (diagnostics_only and not risks and not (baseline_outcome and baseline_outcome["hard_vetoes"])
            and quality["status"] == "PASS" and all(g.state == "PASS" for g in gates)):
        decision = "WATCH_ONLY"
    forecast_record = localization_record() if baseline is not None else forecast.record()
    reasons = sorted(set(quality["reason_codes"] + list(forecast_record["reasons"]) + list(risks) + trigger["reasons"]
                         + [reason for g in gates for reason in g.reasons]
                         + [g.gate_id + "_" + g.state for g in gates if g.state != "PASS"]))
    if bundle is None and baseline is None: reasons.append("MODEL_ARTIFACTS_NOT_PROVIDED")
    if baseline_outcome is not None:
        reasons.extend(baseline_outcome["caps"])
        reasons.extend(baseline_outcome["hard_vetoes"])
        reasons.extend(baseline_outcome["reasons"])
    if not liquidity_known and (not is_r15 or plan is not None): reasons.append("CUTOFF_LIQUIDITY_UNKNOWN")
    if plan is None: reasons.append("REFERENCE_PLAN_NOT_READY")
    if factors is None: reasons.append("FACTOR_VECTOR_INCOMPLETE")
    if not discovery and plan is not None and plan.target is None: reasons.append("TARGET_UNKNOWN_PRICE_DISCOVERY_POLICY_NOT_BOUND")
    if diagnostics_only: reasons.append("DIAGNOSTICS_ONLY_NOT_FORMAL_ENTRY")
    signal_id = identity(dict(evidence_id=evidence_id, bundle_hash=bundle.sha256 if bundle else None, decision_policy_version=decision_policy_version))
    qid = identity(quality)
    fid = identity(forecast_record)
    envelope = dict(schema_version=BASELINE_ENVELOPE_SCHEMA if baseline else envelope_schema_version(quality.get("profile_id")), spec_id=SPEC_ID,
        model=dict(model_status="RULE_BASED_BASELINE" if baseline else "OFFLINE_FITTED" if bundle else "RESEARCH_SPEC", status_basis="HASH_BOUND_RUNTIME_FACT_NOT_DOCUMENT_DECLARATION",
                   bundle_sha256=bundle.sha256 if bundle else None, trained_artifacts_status="NOT_APPLICABLE" if baseline else "BOUND_VERIFIED" if bundle else "NOT_PROVIDED",
                   validation_evidence_status="NOT_APPLICABLE" if baseline else "NOT_PROVIDED"),
        event=dict(security_id=symbol, cutoff_at=cutoff, evidence_id=evidence_id, plan_id=plan.plan_id if plan else None, signal_id=signal_id, context=selected),
        quality=dict(status=quality["status"], quality_score=quality["quality_score"], data_confidence=quality["data_confidence"],
                     coverage_total=quality["coverage_total"], core_ttm_pass=quality["core_ttm_pass"], gap_qualifier=quality["gap_qualifier"],
                     gap_penalty_points=quality["missing_penalty_points"], quality_result_id=qid),
        localization=dict(status=forecast_record["status"], bottom_price_quantiles=forecast_record["quantiles"], timing_pmf=forecast_record["timing"],
                          p_fill=forecast_record["p_fill"], p_rebound_given_fill=forecast_record["p_rebound"], p_entry_given_fill=forecast_record["p_entry"],
                          p_bottom_soon=forecast_record["p_bottom_soon"], forecast_result_id=fid),
        score=dict(factor_order=FACTORS, factor_values=factors, applicable_weights=weights, **dict(scored, final_score=formal)),
        strategy=dict(strategy_decision=decision, reason_codes=sorted(set(reasons)), decision_policy_version=decision_policy_version),
        execution=dict(run_mode="RESEARCH", live_enabled=False, authorization_verified=False, execution_status=None,
                       allocated_shares=0, client_order_id=None, order_status=None))
    result = dict(symbol=symbol, signal_date=feature_audit.get("cutoff_session"), model_version="7.0.0",
                parameter_version=(parameter_version(quality.get("profile_id")) + ":" + baseline["policy"]["policy_version"] + ":" + baseline["policy_id"]
                                   if baseline else parameter_version(quality.get("profile_id"))), spec_sha256=SPEC_SHA256, envelope=envelope,
                diagnostics=dict(features=f, feature_audit=feature_audit, quality=quality, context_matches=matched,
                    reference_plan=asdict(plan) if plan else None, localization=forecast_record,
                    factor_ledger=factor_audit, penalties=penalties, safety_caps=caps,
                    diagnostic_score=scored, starter=trigger["starter"], confirmed=trigger["confirmed"],
                    starter_gate=starter_gate, confirmed_gate=confirmed_gate, evidence_families=trigger["evidence_families"],
                    gate_receipts=[asdict(g) for g in gates],
                    gate_bindings=dict(SAFETY=base_binding, DATA_INTEGRITY=base_binding, TRADING_STATUS=base_binding,
                        VALUATION=valuation_binding, STRESS_RISK=risk_binding, PRICE_DISCOVERY_PAYOFF=risk_binding)),
                research_only=True, requires_next_session_execution_check=True)
    if bundle is not None:
        # Presentation lineage describes the bound predictor inputs, not all
        # diagnostics computed along the way (which can include unused DFA).
        result["diagnostics"]["model_input_profile"] = model_input_profile(model, bundle.sha256)
    if execution_inputs is not None:
        result["diagnostics"].update(execution_features=execution_inputs,
            execution_feature_audit=execution_audit)
    extension = quality_extension_schema(quality.get("profile_id"))
    if is_r10:
        result["diagnostics"]["structure_plan"] = structure
    if is_r18:
        result["diagnostics"]["execution_assessment"] = execution_assessment
    if baseline is not None:
        result["diagnostics"]["rule_baseline"] = dict(baseline,
            score_evidence=baseline_outcome["score_evidence"], hard_vetoes=baseline_outcome["hard_vetoes"])
        if "technical_vetoes" in baseline_outcome:
            result["diagnostics"]["rule_baseline"]["technical_vetoes"] = baseline_outcome["technical_vetoes"]
        if "entry_admission" in baseline_outcome:
            result["diagnostics"]["entry_admission"] = baseline_outcome["entry_admission"]
        if is_r15:
            route_key = "r19_route_evaluation" if is_r19 else "r18_route_evaluation" if is_r18 else "r17_route_evaluation" if is_r17 else "r16_route_evaluation" if is_r16 else "r15_route_evaluation"
            result["diagnostics"][route_key] = baseline_outcome[route_key]
            if decision in {"BUY", "BUY_SMALL"}:
                from domain.les_v7.plan_r10 import bind_entry_route
                result["diagnostics"]["reference_entry_route"] = bind_entry_route(plan,
                    policy["reference_policy"], strategy_decision=decision,
                    risk_route=baseline_outcome[route_key]["selected_route"],
                    setup_variant=baseline["selection"].get("setup_variant"))
    if diagnostics_only:
        result["diagnostics"]["run_scope"] = dict(schema="v7-diagnostics-only/v1", diagnostics_only=True,
                                                   formal_entry_enabled=False)
    if evidence_binding is not None:
        result["diagnostics"]["evidence_binding"] = evidence_binding
        if baseline is not None or strategy_policy is None:
            result["diagnostics"]["evidence_quality_policy"] = quality_policy
    if strategy_policy is not None:
        result["diagnostics"]["strategy_policy"] = strategy_policy.export()
        result["diagnostics"]["strategy_policy_id"] = strategy_policy.sha256
        result["diagnostics"]["model_policy_binding"] = (dict(bundle_sha256=bundle.sha256,
            trained_until=model["trained_until"], policy_blocks_id=policy_blocks_id(model)) if bundle else None)
    if receipt_producer is not None:
        result["diagnostics"]["native_evidence"] = dict(schema="les-v7-native-evidence/v1",
            source_packet=native_source, source_id=identity(native_source), produced_receipts=produced,
            reference_policy=policy.get("reference_policy"))
        if "valuation_policy" in policy:
            result["diagnostics"]["native_evidence"]["valuation_policy"] = policy["valuation_policy"]
        if "valuation_input_policy" in policy:
            result["diagnostics"]["native_evidence"]["valuation_input_policy"] = policy["valuation_input_policy"]
        if "source_policy" in policy:
            result["diagnostics"]["native_evidence"]["source_policy"] = policy["source_policy"]
    if extension:
        result["quality_extension_schema"] = extension
        result["diagnostics"]["quality_policy"] = quality.get("resolved_policy", quality_policy)
        result["diagnostics"]["readiness"] = dict(
            quality=dict(state="READY" if quality["status"] == "PASS" else "BLOCKED",
                         status=quality["status"], reason_codes=quality["reason_codes"]),
            model=dict(state="RULE_BASED_READY" if baseline else "READY" if forecast.status == "VALID" else "NOT_READY",
                       status=forecast_record["status"], bundle_provided=bundle is not None,
                       reason_codes=list(forecast_record["reasons"])),
            execution=dict(state="READY_FOR_ACCOUNT_CHECK" if decision in {"BUY", "BUY_SMALL"} else "BLOCKED",
                           candidate_decision=decision, order_authority=False,
                           required_gates={g.gate_id: g.state for g in gates},
                           reference_plan_ready=plan is not None, liquidity_known=liquidity_known))
        if baseline_outcome is not None and "entry_admission" in baseline_outcome:
            from domain.les_v7.entry_policy_r9 import quality_readiness
            result["diagnostics"]["readiness"]["quality"] = quality_readiness(quality, baseline_outcome["entry_admission"])
        if quality.get("profile_id") in QUALITY_REPORTED_PROFILES:
            result["envelope"]["quality"].update(
                core_reported_pass=quality["core_reported_pass"],
                core_inputs_pass=quality["core_inputs_pass"], scoring_basis=quality["scoring_basis"])
    return json.loads(canonical(result))  # CLI/API/export share the same JSON-native contract.
