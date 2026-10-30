"""One preselected setup, explicit core/tactical score routes, no sizing feedback.

Core BUY uses the retained R14 economics/risk. Tactical deductions cannot
upgrade it. Both computations and their source states are retained for audit;
only the selected route supplies the single published Final Score.
"""
from collections.abc import Mapping

from domain.les_v7.contracts import FACTORS
from domain.les_v7.decision import vetoes
from domain.les_v7.plan_r10 import evaluate_entry, validate_band
from domain.les_v7.rule_score_r8 import compose_penalty_score
from domain.strategies.low_entry_v6_signals import hard_vetoes, veto_evidence_complete


SCHEMA = "les-v7-core-tactical-routing/v1"


def _score(ledger, penalties):
    outcome = compose_penalty_score(ledger, [dict(reason=k, points=v) for k, v in penalties.items()])
    return dict(raw_score=outcome["raw_score"], adjusted_score=outcome["adjusted_score"],
        final_score=outcome["final_score"], total_penalty=outcome["penalty"], effective_cap=None)


def score_routes(prepared, *, selection, features, scored, trigger, plan, quality,
                 gates, native_features, risks, starter_gate, confirmed_gate,
                 valuation_receipt, entry_binding, config):
    from domain.les_v7.entry_policy_r15 import assess
    from domain.les_v7.policy_r16 import BASELINE_VERSION as R16, EARLY_PULLBACK, PRICE_DISCOVERY
    r16 = prepared["policy"]["policy_version"] == R16
    if r16:
        from domain.les_v7.entry_policy_r16 import assess
    if not isinstance(entry_binding, Mapping):
        raise ValueError("R15_SOURCE_BINDING_REQUIRED")
    variant = prepared["selection"].get("setup_variant")
    admission = assess(quality, gates, valuation_receipt, symbol=entry_binding["symbol"],
        cutoff=entry_binding["cutoff"], current_price=native_features.get("close"), setup_variant=variant)
    ledger = scored["score_evidence"]
    common = {"RISK_OFF": config.risk_off_score_increment} if features.get("market_regime") == "RISK_OFF" else {}
    deductions = prepared["policy"]["score_adjustment_policy"]["deductions"]
    if selection.context == "NO_CLEAR_CONTEXT":
        common["BASELINE_NO_CLEAR_CONTEXT"] = deductions["no_clear_context"]
    elif trigger["starter"] is not True and trigger["confirmed"] is not True:
        common["BASELINE_TRIGGER_NOT_READY"] = deductions["trigger_not_ready"]
    elif trigger["confirmed"] is not True and not (r16 and trigger["starter"] is True
            and variant in {EARLY_PULLBACK, PRICE_DISCOVERY}):
        common["BASELINE_NOT_CONFIRMED"] = deductions["unconfirmed_starter"]
    core_penalties = dict(common)
    tactical_penalties = dict(common)
    if admission["core_points"] > 0:
        core_penalties["VALUATION_PRICE_WARNING"] = admission["core_points"]
    # Keep the two declared warning components separately auditable. Their
    # sum is exactly tactical_points, never a hidden position-size penalty.
    grade = admission["tactical_grade"]
    price_points = grade["price_points"]
    bear_points = grade["bear_points"]
    if price_points + bear_points != admission["tactical_points"]:
        raise ValueError("R15_WARNING_COMPONENT_SUM_MISMATCH")
    if price_points: tactical_penalties["R15_VALUATION_PRICE_WARNING"] = price_points
    if bear_points: tactical_penalties["R15_BEAR_SCENARIO_WARNING"] = bear_points
    core, tactical = _score(ledger, core_penalties), _score(ledger, tactical_penalties)
    ref = prepared["policy"]["reference_policy"]
    core_check = evaluate_entry(plan, plan.entry if plan else None, ref)
    small_check = evaluate_entry(plan, plan.entry if plan else None, ref, starter=True)
    core_band, small_band = validate_band(plan, ref), validate_band(plan, ref, starter=True)
    states = {g.gate_id: g.state for g in gates}
    extra_vetoes = list(hard_vetoes(features))
    technical_vetoes = sorted(set(vetoes(native_features)) | set(extra_vetoes))
    effective_risks = [r for r in risks if r != "VALUATION_FAIL"]
    technical_ready = bool(features.get("history_ready") and features.get("market_ready")
                           and veto_evidence_complete(features) and tactical["final_score"] is not None)
    source_unknowns = sorted([name for name in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION")
                              if states.get(name) == "UNKNOWN"])
    if not admission["quality"]["evidence_ready"]: source_unknowns.append("QUALITY")
    if admission["tactical_grade"]["state"] == "UNKNOWN" and "VALUATION" not in source_unknowns:
        source_unknowns.append("VALUATION")
    if not technical_ready: source_unknowns.append("TECHNICAL_INPUTS")
    blocked = bool(effective_risks or technical_vetoes or admission["state"] == "BLOCKED")
    core_buy = bool(not blocked and admission["state"] == "ALLOWED" and not variant
        and technical_ready and plan and admission["core_buy_allowed"] and confirmed_gate
        and core_check["state"] == core_band["state"] == "PASS"
        and trigger["confirmed"] is True and core["final_score"] >= 72)
    small = bool(not blocked and admission["state"] == "ALLOWED" and technical_ready and plan
        and admission["tactical_small_allowed"] and starter_gate
        and small_check["state"] == small_band["state"] == "PASS"
        and trigger["starter"] is True and tactical["final_score"] >= 62)
    selected_route = "CORE" if core_buy else "TACTICAL"
    contract = core if core_buy else tactical
    penalties = core_penalties if core_buy else tactical_penalties
    if blocked: decision = "NO_BUY"
    elif technical_ready and selection.context == "NO_CLEAR_CONTEXT": decision = "WATCH_ONLY"
    elif source_unknowns: decision = "DATA_INSUFFICIENT"
    elif not plan: decision = "WATCH_ONLY"  # Known unconfirmed structure is not a certified plan.
    elif admission["state"] == "UNKNOWN": decision = "DATA_INSUFFICIENT"
    elif core_buy: decision = "BUY"
    elif small: decision = "BUY_SMALL"
    elif contract["final_score"] < 55: decision = "AVOID"
    elif contract["final_score"] < 62: decision = "WATCH"
    else: decision = "ARMED"
    source_state = "FAIL" if blocked else "UNKNOWN" if source_unknowns else "VERIFIED"
    setup_state = "NO_MATCH" if selection.context == "NO_CLEAR_CONTEXT" else "SELECTED"
    plan_state = ("NOT_APPLICABLE_NO_SETUP" if setup_state == "NO_MATCH" else
        "WAIT_STRUCTURE" if plan is None else "INVALID" if small_check["state"] == "FAIL" else
        "UNKNOWN" if small_check["state"] != "PASS" or small_band["state"] == "UNKNOWN" else
        "READY" if small_band["state"] == "PASS" else "INVALID")
    routes = dict(schema="les-v7-core-tactical-routing/v2" if r16 else SCHEMA, selected_route=selected_route, source_state=source_state,
        source_unknowns=sorted(set(source_unknowns)), setup_state=setup_state, plan_state=plan_state,
        setup_variant=variant, core=dict(score=core, penalties=core_penalties,
            admission=admission["core_grade"], close_risk=core_check, band_risk=core_band,
            buy_eligible=core_buy),
        tactical=dict(score=tactical, penalties=tactical_penalties, admission=grade,
            close_risk=small_check, band_risk=small_band, small_eligible=small),
        position_limits_affect_score=False, source_failures_compensable=False)
    values = tuple(ledger["components"][name]["earned_points"] / ledger["components"][name]["total_weight"]
                   for name in FACTORS) if contract["raw_score"] is not None else None
    weights = tuple(prepared["policy"]["model_spec"]["factor_budgets"][name] / 100 for name in FACTORS)
    return dict(score=contract, factors=values, weights=weights, penalties=penalties, caps={},
        decision=decision, score_evidence=ledger, factor_ledger=scored["factor_budgets"], trigger=trigger,
        hard_vetoes=extra_vetoes, technical_vetoes=technical_vetoes, entry_admission=admission,
        **{("r16_route_evaluation" if r16 else "r15_route_evaluation"): routes},
        reasons=[("R16_" if r16 else "R15_") + selected_route + "_SCORE",
            ("R16_" if r16 else "R15_") + setup_state,
            ("R16_" if r16 else "R15_") + plan_state, admission["tactical_grade"]["reason"]])
