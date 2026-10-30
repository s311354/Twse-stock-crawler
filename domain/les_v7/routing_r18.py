"""R18 fixed score and one verified execution assessment, without size feedback."""
from collections.abc import Mapping

from domain.les_v7.contracts import FACTORS
from domain.les_v7.decision import vetoes
from domain.les_v7.entry_policy_r18 import assess
from domain.les_v7.policy_r18 import SUPPORT_RESISTANCE_LIMIT
from domain.les_v7.risk_r18 import require_assessment
from domain.les_v7.routing_r15 import _score
from domain.les_v7.routing_r17 import _structure_unknowns, _structure_failures
from domain.strategies.low_entry_v6_signals import hard_vetoes, veto_evidence_complete


def score_routes(prepared, *, selection, features, scored, trigger, plan, quality,
                 gates, native_features, risks, starter_gate, confirmed_gate,
                 valuation_receipt, entry_binding, config, execution_assessment):
    if not isinstance(entry_binding, Mapping): raise ValueError("R18_SOURCE_BINDING_REQUIRED")
    execution = require_assessment(execution_assessment, plan,
        prepared["policy"]["reference_policy"], native_features)
    variant = prepared["selection"].get("setup_variant")
    admission = assess(quality, gates, valuation_receipt, symbol=entry_binding["symbol"],
        cutoff=entry_binding["cutoff"], current_price=native_features.get("close"), setup_variant=variant)
    common = {"RISK_OFF": config.risk_off_score_increment} if features.get("market_regime") == "RISK_OFF" else {}
    core_penalties, tactical_penalties = dict(common), dict(common)
    if admission["core_points"]: core_penalties["VALUATION_PRICE_WARNING"] = admission["core_points"]
    grade = admission["tactical_grade"]
    if grade["price_points"] + grade["bear_points"] != admission["tactical_points"]:
        raise ValueError("R18_WARNING_COMPONENT_SUM_MISMATCH")
    if grade["price_points"]: tactical_penalties["R18_VALUATION_PRICE_WARNING"] = grade["price_points"]
    if grade["bear_points"]: tactical_penalties["R18_BEAR_SCENARIO_WARNING"] = grade["bear_points"]
    ledger = scored["score_evidence"]
    core, tactical = _score(ledger, core_penalties), _score(ledger, tactical_penalties)
    states = {g.gate_id: g.state for g in gates}
    extra_vetoes = list(hard_vetoes(features))
    technical_vetoes = sorted(set(vetoes(native_features)) | set(extra_vetoes))
    technical_ready = bool(features.get("history_ready") and features.get("market_ready")
        and veto_evidence_complete(features) and tactical["final_score"] is not None)
    unknowns = [name for name in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION")
                if states.get(name) == "UNKNOWN"]
    if not admission["quality"]["evidence_ready"]: unknowns.append("QUALITY")
    if grade["state"] == "UNKNOWN": unknowns.append("VALUATION")
    if not technical_ready: unknowns.append("TECHNICAL_INPUTS")
    if plan is None: unknowns.extend(_structure_unknowns(prepared))
    failures = _structure_failures(prepared) if plan is None else []
    effective_risks = [r for r in risks if r != "VALUATION_FAIL"]
    blocked = bool(effective_risks or technical_vetoes or failures or admission["state"] == "BLOCKED")
    allowed = not blocked and not unknowns and admission["state"] == "ALLOWED" and technical_ready and plan is not None
    limits = prepared["policy"]["unified_technical_policy"]
    # No second pattern/strength/trigger boolean test: structure is in the plan,
    # strength in the fixed ledger, execution in the bound assessment exactly once.
    buy = bool(allowed and not variant and admission["core_buy_allowed"]
        and execution["core"]["state"] == "PASS" and core["final_score"] >= limits["buy_minimum"])
    small = bool(allowed and admission["tactical_small_allowed"]
        and execution["tactical"]["state"] == "PASS" and tactical["final_score"] >= limits["small_minimum"])
    route = "CORE" if buy else "TACTICAL"
    contract, penalties = (core, core_penalties) if buy else (tactical, tactical_penalties)
    neutral = variant == SUPPORT_RESISTANCE_LIMIT
    if blocked: decision = "NO_BUY"
    elif unknowns: decision = "DATA_INSUFFICIENT"
    elif not plan or selection.context == "NO_CLEAR_CONTEXT" and not neutral: decision = "WATCH_ONLY"
    elif admission["state"] == "UNKNOWN": decision = "DATA_INSUFFICIENT"
    elif buy: decision = "BUY"
    elif small: decision = "BUY_SMALL"
    elif contract["final_score"] < 55: decision = "AVOID"
    elif contract["final_score"] < limits["small_minimum"]: decision = "WATCH"
    else: decision = "ARMED"
    setup_state = "SUPPORT_RESISTANCE" if neutral else "NO_MATCH" if selection.context == "NO_CLEAR_CONTEXT" else "SELECTED"
    plan_state = ("UNKNOWN" if unknowns and plan is None else "NOT_APPLICABLE_NO_SETUP" if setup_state == "NO_MATCH" else
        "WAIT_STRUCTURE" if plan is None else "INVALID" if execution["tactical"]["state"] == "FAIL" else
        "READY" if execution["tactical"]["state"] == "PASS" else "UNKNOWN")
    routes = dict(schema="les-v7-core-tactical-routing/v4", selected_route=route,
        source_state="FAIL" if blocked else "UNKNOWN" if unknowns else "VERIFIED",
        source_unknowns=sorted(set(unknowns)), source_failures=failures, setup_state=setup_state,
        plan_state=plan_state, setup_variant=variant, execution_assessment_id=execution["assessment_id"],
        core=dict(score=core, penalties=core_penalties, admission=admission["core_grade"],
            close_risk=execution["core"]["close"], band_risk=execution["core"]["band"], buy_eligible=buy),
        tactical=dict(score=tactical, penalties=tactical_penalties, admission=grade,
            close_risk=execution["tactical"]["close"], band_risk=execution["tactical"]["band"], small_eligible=small),
        technical_confirmation="EXISTING_FIXED_FACTOR_LEDGER_ONLY", position_limits_affect_score=False,
        source_failures_compensable=False)
    values = tuple(ledger["components"][name]["earned_points"] / ledger["components"][name]["total_weight"]
                   for name in FACTORS) if contract["raw_score"] is not None else None
    weights = tuple(prepared["policy"]["model_spec"]["factor_budgets"][name] / 100 for name in FACTORS)
    return dict(score=contract, factors=values, weights=weights, penalties=penalties, caps={},
        decision=decision, score_evidence=ledger, factor_ledger=scored["factor_budgets"], trigger=trigger,
        hard_vetoes=extra_vetoes, technical_vetoes=technical_vetoes, entry_admission=admission,
        r18_route_evaluation=routes, reasons=["R18_" + route + "_SCORE", "R18_" + setup_state,
            "R18_" + plan_state, grade["reason"], *failures])
