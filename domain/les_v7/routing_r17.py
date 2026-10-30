"""One fixed factor ledger; technical strength is not a second admission gate.

Causal setup/anchors, verified sources and the same R16 economics remain
required. Known weak indicators earn their original lower factor points, not
another trigger penalty. Sizing never changes the score.
"""
from collections.abc import Mapping

from domain.les_v7.contracts import FACTORS
from domain.les_v7.decision import vetoes
from domain.les_v7.entry_policy_r17 import assess
from domain.les_v7.plan_r10 import evaluate_entry, validate_band
from domain.les_v7.routing_r15 import _score
from domain.strategies.low_entry_v6_signals import hard_vetoes, veto_evidence_complete


def _structure_unknowns(prepared):
    """A missing source must not be presented as a known unfinished pattern."""
    audit = prepared["selection"].get("short_base_audit") or {}
    unknowns = []
    def inspect(value):
        if not isinstance(value, Mapping):
            return
        coverage = value.get("session_coverage")
        if isinstance(coverage, Mapping) and coverage.get("state") == "UNKNOWN":
            unknowns.append("OFFICIAL_SESSION_COVERAGE")
        if value.get("state") == "UNKNOWN":
            unknowns.append("STRUCTURE_INPUTS_UNKNOWN")
            for key, state in value.get("checks", {}).items():
                if state is None:
                    unknowns.append("STRUCTURE_" + key)
        for key in ("early_pullback", "price_discovery", "support_audit", "candidate_audit"):
            inspect(value.get(key))
    inspect(audit)
    return sorted(set(unknowns))


def _structure_failures(prepared):
    audit = prepared["selection"].get("short_base_audit") or {}
    failures = []
    def inspect(value):
        if not isinstance(value, Mapping): return
        coverage = value.get("session_coverage")
        if isinstance(coverage, Mapping) and coverage.get("state") == "FAIL":
            failures.append("OFFICIAL_SESSION_COVERAGE_FAIL")
        for key in ("early_pullback", "price_discovery", "support_audit", "candidate_audit"):
            inspect(value.get(key))
    inspect(audit)
    return sorted(set(failures))


def score_routes(prepared, *, selection, features, scored, trigger, plan, quality,
                 gates, native_features, risks, starter_gate, confirmed_gate,
                 valuation_receipt, entry_binding, config):
    if not isinstance(entry_binding, Mapping):
        raise ValueError("R17_SOURCE_BINDING_REQUIRED")
    variant = prepared["selection"].get("setup_variant")
    admission = assess(quality, gates, valuation_receipt, symbol=entry_binding["symbol"],
        cutoff=entry_binding["cutoff"], current_price=native_features.get("close"), setup_variant=variant)
    ledger = scored["score_evidence"]
    common = {"RISK_OFF": config.risk_off_score_increment} if features.get("market_regime") == "RISK_OFF" else {}
    core_penalties, tactical_penalties = dict(common), dict(common)
    if admission["core_points"]:
        core_penalties["VALUATION_PRICE_WARNING"] = admission["core_points"]
    grade = admission["tactical_grade"]
    if grade["price_points"] + grade["bear_points"] != admission["tactical_points"]:
        raise ValueError("R17_WARNING_COMPONENT_SUM_MISMATCH")
    if grade["price_points"]:
        tactical_penalties["R15_VALUATION_PRICE_WARNING"] = grade["price_points"]
    if grade["bear_points"]:
        tactical_penalties["R15_BEAR_SCENARIO_WARNING"] = grade["bear_points"]
    core, tactical = _score(ledger, core_penalties), _score(ledger, tactical_penalties)
    ref = prepared["policy"]["reference_policy"]
    core_check, small_check = (evaluate_entry(plan, plan.entry if plan else None, ref, starter=s)
                               for s in (False, True))
    core_band, small_band = (validate_band(plan, ref, starter=s) for s in (False, True))
    states = {g.gate_id: g.state for g in gates}
    extra_vetoes = list(hard_vetoes(features))
    technical_vetoes = sorted(set(vetoes(native_features)) | set(extra_vetoes))
    effective_risks = [r for r in risks if r != "VALUATION_FAIL"]
    technical_ready = bool(features.get("history_ready") and features.get("market_ready")
        and veto_evidence_complete(features) and tactical["final_score"] is not None)
    source_unknowns = [name for name in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION")
                       if states.get(name) == "UNKNOWN"]
    if not admission["quality"]["evidence_ready"]: source_unknowns.append("QUALITY")
    if grade["state"] == "UNKNOWN": source_unknowns.append("VALUATION")
    if not technical_ready: source_unknowns.append("TECHNICAL_INPUTS")
    if plan is None: source_unknowns.extend(_structure_unknowns(prepared))
    source_failures = _structure_failures(prepared) if plan is None else []
    blocked = bool(effective_risks or technical_vetoes or source_failures or admission["state"] == "BLOCKED")
    # No EMA/volume/CLV/confirmation AND test remains here. All existing
    # risk/source/Quality checks still execute independently of factor points.
    core_buy = bool(not blocked and not source_unknowns and admission["state"] == "ALLOWED"
        and not variant and technical_ready and plan and admission["core_buy_allowed"]
        and confirmed_gate and core_check["state"] == core_band["state"] == "PASS"
        and core["final_score"] >= 72)
    small = bool(not blocked and not source_unknowns and admission["state"] == "ALLOWED"
        and technical_ready and plan and admission["tactical_small_allowed"] and starter_gate
        and small_check["state"] == small_band["state"] == "PASS" and tactical["final_score"] >= 62)
    selected_route = "CORE" if core_buy else "TACTICAL"
    contract, penalties = (core, core_penalties) if core_buy else (tactical, tactical_penalties)
    if blocked: decision = "NO_BUY"
    elif source_unknowns: decision = "DATA_INSUFFICIENT"
    elif selection.context == "NO_CLEAR_CONTEXT" or not plan: decision = "WATCH_ONLY"
    elif admission["state"] == "UNKNOWN": decision = "DATA_INSUFFICIENT"
    elif core_buy: decision = "BUY"
    elif small: decision = "BUY_SMALL"
    elif contract["final_score"] < 55: decision = "AVOID"
    elif contract["final_score"] < 62: decision = "WATCH"
    else: decision = "ARMED"
    setup_state = "NO_MATCH" if selection.context == "NO_CLEAR_CONTEXT" else "SELECTED"
    plan_state = ("UNKNOWN" if source_unknowns and plan is None else
        "NOT_APPLICABLE_NO_SETUP" if setup_state == "NO_MATCH" else "WAIT_STRUCTURE" if plan is None else
        "INVALID" if small_check["state"] == "FAIL" or small_band["state"] == "FAIL" else
        "READY" if small_check["state"] == small_band["state"] == "PASS" else "UNKNOWN")
    routes = dict(schema="les-v7-core-tactical-routing/v3", selected_route=selected_route,
        source_state="FAIL" if blocked else "UNKNOWN" if source_unknowns else "VERIFIED",
        source_unknowns=sorted(set(source_unknowns)), setup_state=setup_state, plan_state=plan_state,
        source_failures=source_failures,
        setup_variant=variant, core=dict(score=core, penalties=core_penalties,
            admission=admission["core_grade"], close_risk=core_check, band_risk=core_band, buy_eligible=core_buy),
        tactical=dict(score=tactical, penalties=tactical_penalties, admission=grade,
            close_risk=small_check, band_risk=small_band, small_eligible=small),
        technical_confirmation="EXISTING_FIXED_FACTOR_LEDGER_ONLY",
        position_limits_affect_score=False, source_failures_compensable=False)
    values = tuple(ledger["components"][name]["earned_points"] / ledger["components"][name]["total_weight"]
                   for name in FACTORS) if contract["raw_score"] is not None else None
    weights = tuple(prepared["policy"]["model_spec"]["factor_budgets"][name] / 100 for name in FACTORS)
    return dict(score=contract, factors=values, weights=weights, penalties=penalties, caps={},
        decision=decision, score_evidence=ledger, factor_ledger=scored["factor_budgets"], trigger=trigger,
        hard_vetoes=extra_vetoes, technical_vetoes=technical_vetoes, entry_admission=admission,
        r17_route_evaluation=routes,
        reasons=["R17_" + selected_route + "_SCORE", "R17_" + setup_state,
                 "R17_" + plan_state, grade["reason"], *source_failures])
