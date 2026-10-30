"""R17 causal plans with one technical score, not a second strength veto.

Reuse immutable R16 mathematics, source selection helpers, costs and legal
ticks. Only the approved soft-condition interpretation is versioned here.
There is no score bonus, risk/valuation relaxation, RR-selected support or fill.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Decimal

from domain.les_v7 import plan_r16 as r16
from domain.les_v7.context_r15 import VARIANT as LEGACY_FLOOR_VARIANT
from domain.les_v7.context_r17 import select_context, price_structure_features, policy as context_policy
from domain.les_v7.contracts import CONTEXTS, identity, timestamp
from domain.les_v7.policy_r17 import PLAN_VERSION, AVAILABLE_AT, EARLY_PULLBACK, PRICE_DISCOVERY
from domain.les_v7.reference_path import _legal, _round_price, estimate_reference_costs
from domain.les_v7.support_r17 import assess_floor
from domain.les_v7.session_coverage_r17 import verify_window
from domain.strategies.low_entry_v6_hybrid import ContextSelection, NO_CLEAR_CONTEXT, known_features

A_VARIANT, C_VARIANT = EARLY_PULLBACK, PRICE_DISCOVERY
NEW_VARIANTS = frozenset({A_VARIANT, C_VARIANT})
SCHEMA = "les-v7-r17-structure-selection/v1"
PLAN_SCHEMA = "les-v7-setup-plan-evidence/v3"
ROUTE_SCHEMA = "les-v7-reference-entry-route/v3"
A_ENTRY_BASIS, CLOSE_ENTRY_BASIS = r16.A_ENTRY_BASIS, r16.CLOSE_ENTRY_BASIS
OBSERVED_TARGET, PROJECTED_TARGET = r16.OBSERVED_TARGET, r16.PROJECTED_TARGET
_TO_R16 = {A_VARIANT: r16.A_VARIANT, C_VARIANT: r16.C_VARIANT}
_HARD_A = frozenset({"HISTORY_READY", "MARKET_READY", "FULL_PRIOR_20", "FLOOR_AFTER_IMPULSE",
    "TWO_CONSECUTIVE_PRIOR_CLOSES", "SIGNAL_HOLDS_FLOOR", "CLOSE_ABOVE_FLOOR"})
_HARD_C = frozenset({"HISTORY_READY", "MARKET_READY", "FULL_PRIOR_120", "ABOVE_PRIOR_120_HIGH", "SUPPORT_READY"})


def _selection_policy(value, cutoff=None, *, reference_policy=None):
    """Resolve an explicitly bound new selection policy, never a global default.

    Older reference policies have no marker and retain their exact selection
    outputs. The marker travels with the reference-policy identity so the
    source producer and reconstruction cannot silently use different rules.
    """
    recorded = reference_policy.get("source_selection_policy") if reference_policy is not None else None
    if value is None:
        value = recorded
    elif reference_policy is not None and value != recorded:
        raise ValueError("R22_SOURCE_SELECTION_POLICY_MISMATCH")
    if value is None:
        return None
    from domain.les_v7.policy_r22 import AVAILABLE_AT, SOURCE_SELECTION_VERSION
    if value != SOURCE_SELECTION_VERSION:
        raise ValueError("R22_SOURCE_SELECTION_POLICY_MISMATCH")
    if (cutoff is not None and timestamp(cutoff) < timestamp(AVAILABLE_AT)
            or reference_policy is not None
            and timestamp(reference_policy["available_at"]) < timestamp(AVAILABLE_AT)):
        raise ValueError("R22_SOURCE_SELECTION_BEFORE_RELEASE")
    return value


def structural_policy():
    return dict(version=PLAN_VERSION, context_policy=context_policy(),
        hard_a=sorted(_HARD_A), hard_c=sorted(_HARD_C),
        soft="KNOWN_PATTERN_STRENGTH_ONLY_IN_EXISTING_SIX_FACTOR_SCORE_NO_EXTRA_VETO_OR_BONUS",
        missing="REQUIRED_SOURCE_OR_TECHNICAL_OPERANDS_REMAIN_UNKNOWN",
        floor_method="CAUSAL_FROZEN_PRIOR5_FLOOR_INDEPENDENT_OF_PATTERN_STRENGTH_R17",
        support_priority="ORIGINAL_CONFIRMED_SUPPORT_BEFORE_FIXED_FLOOR_NO_RISK_RESCUE",
        risk_math="UNCHANGED_R16_CORE_AND_TACTICAL_RR_ATR_FEES_BANDS_AND_CAPS",
        target_projection_multiple=.5, stop_buffer_atr=.3,
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", expiry="NEXT_SESSION_CLOSE",
        performance_validated=False)


def policy(reference_policy, *, available_at):
    if timestamp(available_at) < timestamp(AVAILABLE_AT):
        raise ValueError("R17_REFERENCE_POLICY_BEFORE_RELEASE")
    result = r16.policy(reference_policy, available_at=available_at)
    result.update(version=PLAN_VERSION, support_rule="CAUSAL_STRUCTURES_SINGLE_SOFT_SCORE_R17",
        conditional_entry="NEXT_OFFICIAL_SESSION_OPEN_IN_STRUCTURAL_BAND_R17")
    result["structural_rules"]["r17"] = structural_policy()
    return result


def _as_r16_policy(reference_policy):
    result = deepcopy(dict(reference_policy))
    result.update(version=r16.PLAN_VERSION, support_rule="CAUSAL_FIXED_PRIORITY_STRUCTURES_R16",
        conditional_entry="NEXT_OFFICIAL_SESSION_OPEN_IN_STRUCTURAL_BAND_R16")
    result["structural_rules"].pop("r17", None)
    return result


def require_policy(reference_policy, cutoff=None):
    _selection_policy(None, cutoff, reference_policy=reference_policy)
    if (reference_policy.get("version") != PLAN_VERSION
            or reference_policy.get("support_rule") != "CAUSAL_STRUCTURES_SINGLE_SOFT_SCORE_R17"
            or reference_policy.get("conditional_entry") != "NEXT_OFFICIAL_SESSION_OPEN_IN_STRUCTURAL_BAND_R17"
            or reference_policy.get("structural_rules", {}).get("r17") != structural_policy()):
        raise ValueError("R17_REFERENCE_POLICY_MISMATCH")
    if timestamp(reference_policy["available_at"]) < timestamp(AVAILABLE_AT):
        raise ValueError("R17_REFERENCE_POLICY_BEFORE_RELEASE")
    r16.require_policy(_as_r16_policy(reference_policy), cutoff)
    return reference_policy


def _as_r16_plan(plan):
    values = asdict(plan)
    values.update(policy_version=r16.PLAN_VERSION, setup_variant=_TO_R16.get(plan.setup_variant, plan.setup_variant))
    return r16.R16ReferencePlan(**values)


@dataclass(frozen=True)
class R17ReferencePlan(r16.R16ReferencePlan):
    def __post_init__(self):
        if self.policy_version != PLAN_VERSION or self.setup_variant not in {None, LEGACY_FLOOR_VARIANT, *NEW_VARIANTS}:
            raise ValueError("R17_PLAN_VERSION_OR_VARIANT_INVALID")
        # Reuse the exact source-free shape/tick/basis invariants. This proxy
        # is not passed to a source producer or advertised as an old report.
        _as_r16_plan(self)


def plan_from_record(record):
    if record is None: return None
    if record.get("policy_version") != PLAN_VERSION: return r16.plan_from_record(record)
    return R17ReferencePlan(**dict(record, band=tuple(record["band"])))


def evaluate_entry(plan, observed_entry, reference_policy, *, starter=False):
    require_policy(reference_policy)
    if plan is None: return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    if not isinstance(plan, R17ReferencePlan): raise ValueError("R17_SOURCE_BOUND_PLAN_REQUIRED")
    return r16.evaluate_entry(_as_r16_plan(plan), observed_entry, _as_r16_policy(reference_policy), starter=starter)


def validate_band(plan, reference_policy, *, starter=False):
    require_policy(reference_policy)
    if plan is None: return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    if not isinstance(plan, R17ReferencePlan): raise ValueError("R17_SOURCE_BOUND_PLAN_REQUIRED")
    return r16.validate_band(_as_r16_plan(plan), _as_r16_policy(reference_policy), starter=starter)


def _regrade(audit, variant, hard, *, selection_policy=None):
    selection_policy = _selection_policy(selection_policy, audit.get("cutoff"))
    result = deepcopy(audit)
    checks = result.get("checks", {})
    hard_checks = {name: checks.get(name) for name in sorted(hard)}
    soft_checks = {name: value for name, value in checks.items() if name not in hard}
    # A failed known strength gets its ordinary lower technical score, not a
    # second veto. An unmeasurable required operand is not turned into PASS.
    unknowns = [name for name, value in checks.items() if value is None]
    failed = [name for name, value in hard_checks.items() if value is False]
    missing_hard = [name for name, value in hard_checks.items() if value is None]
    required_unknowns = missing_hard if selection_policy is not None else unknowns
    state = "UNKNOWN" if required_unknowns or missing_hard else "FAIL" if failed else "PASS"
    result.update(schema=SCHEMA, policy_id=identity(structural_policy()), state=state,
        candidate_variant=variant, setup_variant=variant if state == "PASS" else None,
        hard_checks=hard_checks, soft_checks=soft_checks,
        reasons=list(dict.fromkeys([*required_unknowns, *missing_hard, *failed])),
        soft_warnings=[name for name, value in soft_checks.items() if value is False],
        soft_effect="EXISTING_FACTOR_LEDGER_ONLY_NO_EXTRA_POINTS_OR_BOOLEAN_VETO")
    if selection_policy is not None:
        result.update(source_selection_policy=selection_policy,
            soft_unknowns=[name for name, value in soft_checks.items() if value is None],
            soft_missing_policy="RETAIN_DIAGNOSTICS_NO_DEFAULT_POINTS_NO_STRUCTURAL_VETO")
    result["retained_arithmetic_operands_id"] = result.get("operands_id")
    result["operands_id"] = identity(dict(arithmetic=result["retained_arithmetic_operands_id"],
        session_coverage=result.get("session_coverage"), support=result.get("support"),
        source_floor=result.get("support_audit"), classification_policy=result["policy_id"],
        hard_checks=hard_checks, soft_checks=soft_checks))
    result["r17_assessment_id"] = identity(result)
    return result


def assess_early_pullback(bars, features, *, cutoff, impulse_high, session_evidence=None, market=None,
                          selection_policy=None):
    selection_policy = _selection_policy(selection_policy, cutoff)
    original = r16.assess_early_pullback(bars, features, cutoff=cutoff, impulse_high=impulse_high,
        session_evidence=session_evidence, market=market)
    if original.get("checks"):
        _, rows = r16._rows(bars, cutoff)
        coverage = verify_window(session_evidence, market=market, cutoff=cutoff,
            observed_sessions=[row["Date"] for row in rows[-21:]], prior_sessions=20)
        original["session_coverage"] = coverage
        original["checks"]["FULL_PRIOR_20"] = (True if coverage["state"] == "PASS" else
            False if coverage["state"] == "FAIL" else None)
    return _regrade(original, A_VARIANT, _HARD_A, selection_policy=selection_policy)


def assess_price_discovery(bars, features, *, cutoff, support=None, support_method=None,
                           session_evidence=None, market=None, selection_policy=None):
    selection_policy = _selection_policy(selection_policy, cutoff)
    floor = None
    if support is None:
        floor = assess_floor(bars, cutoff=cutoff, market=market, session_evidence=session_evidence)
        support = floor.get("support")
        support_method = floor["method"]
    # Supplying the causal support bypasses R16's complete R15 short-pattern
    # helper. The floor's own source/chronology outcome is imposed below.
    result = r16.assess_price_discovery(bars, features, cutoff=cutoff, support=support,
        support_method=support_method, session_evidence=session_evidence, market=market)
    if result.get("checks"):
        _, rows = r16._rows(bars, cutoff)
        coverage = verify_window(session_evidence, market=market, cutoff=cutoff,
            observed_sessions=[row["Date"] for row in rows[-121:]], prior_sessions=120)
        result["session_coverage"] = coverage
        result["checks"]["FULL_PRIOR_120"] = (True if coverage["state"] == "PASS" else
            False if coverage["state"] == "FAIL" else None)
    if floor is not None:
        result["support_audit"] = floor
        result.setdefault("checks", {})["SUPPORT_READY"] = (True if floor["state"] == "PASS" else
            False if floor["state"] == "FAIL" else None)
    return _regrade(result, C_VARIANT, _HARD_C, selection_policy=selection_policy)


def prepare_selection(initial, features, *, bars, cutoff, session_evidence=None, market=None,
                      selection_policy=None):
    selection_policy = _selection_policy(selection_policy, cutoff)
    from domain.les_v7.plan_r10 import _pivots, _support, _resistance
    if not isinstance(initial, ContextSelection): raise ValueError("R17_ORIGINAL_CONTEXT_SELECTION_REQUIRED")
    audit = dict(schema=SCHEMA, policy_id=identity(structural_policy()), cutoff=timestamp(cutoff).isoformat(),
        original_selection=initial.record(), state="UNKNOWN", setup_variant=None, candidate_variant=None,
        reasons=[], support=None, priority="SOURCE_ANCHORS_THEN_EXACT_A_THEN_C_THEN_CAUSAL_FLOOR",
        soft_effect="EXISTING_FACTOR_LEDGER_ONLY_NO_EXTRA_POINTS_OR_BOOLEAN_VETO")
    if bars is None or bars.empty:
        audit["reasons"] = ["COMPLETED_SOURCE_BARS_MISSING"]
        return initial, audit
    _, all_rows = r16._rows(bars, cutoff)
    rows = all_rows[-124:]
    if len(rows) < 5:
        audit["reasons"] = ["CONFIRMED_STRUCTURE_HISTORY_MISSING"]
        return initial, audit
    f = known_features(features)
    atr, close = r16._num(f.get("atr14")), r16._num(f.get("close"))
    if atr is None or atr <= 0 or close is None or close != rows[-1]["Close"]:
        audit["reasons"] = ["SOURCE_CLOSE_OR_ATR_MISSING_OR_CONFLICTING"]
        return initial, audit
    lows, highs = _pivots(rows, "Low", low=True), _pivots(rows, "High", low=False)
    support, missing, method = _support(rows, lows, highs, initial.context, atr) if initial.context in CONTEXTS else (None, "NO_SELECTED_SETUP", None)
    overhead = _resistance(rows, highs, close, max(0, len(rows) - 120))
    audit.update(bars_id=identity(rows), original_support=deepcopy(support),
        original_support_missing=missing, original_resistance=deepcopy(overhead), support_method=method)
    if support is not None and overhead is not None:
        audit.update(state="NOT_APPLICABLE", support=deepcopy(support), reasons=["ORIGINAL_SOURCE_ANCHORS_PRESERVED"])
        return initial, audit
    if initial.context == "TREND_PULLBACK" and missing == "SAME_PULLBACK_CONFIRMED_LOW_MISSING":
        impulse = [point for point in highs if point["index"] >= max(0, len(rows) - 60)][-1]
        result = assess_early_pullback(bars, features, cutoff=cutoff, impulse_high=impulse,
            session_evidence=session_evidence, market=market, selection_policy=selection_policy)
        result.update(original_selection=initial.record(), original_support_missing=missing,
            original_resistance=deepcopy(overhead), priority=audit["priority"])
        return initial, result
    if overhead is None:
        if initial.context == NO_CLEAR_CONTEXT:
            support, _, method = _support(rows, lows, highs, "LEADERSHIP_CONTINUATION", atr)
        result = assess_price_discovery(bars, features, cutoff=cutoff, support=support,
            support_method=method, session_evidence=session_evidence, market=market,
            selection_policy=selection_policy)
        result.update(original_selection=initial.record(), original_support_missing=missing,
            original_resistance=None, priority=audit["priority"])
        if result["state"] == "PASS" and initial.context == NO_CLEAR_CONTEXT:
            checks = tuple((name, tuple(result["hard_checks"].items()) if name == "LEADERSHIP_CONTINUATION" else items)
                for name, items in initial.evidence)
            return ContextSelection("LEADERSHIP_CONTINUATION", checks), result
        return initial, result
    # A known price breakout and an independently verified floor may retain
    # the existing starter route without the old R15 EMA/RS/volume predicates.
    if support is None and initial.context in {NO_CLEAR_CONTEXT, "LEADERSHIP_CONTINUATION"}:
        floor = assess_floor(bars, cutoff=cutoff, market=market, session_evidence=session_evidence)
        breakout = close > max(row["High"] for row in all_rows[-11:-1]) if len(all_rows) >= 11 else None
        audit.update(support_audit=floor, support=deepcopy(floor.get("support")),
            checks=dict(CAUSAL_FLOOR=floor["state"] == "PASS" if floor["state"] != "UNKNOWN" else None,
                        ACTUAL_PRIOR10_BREAKOUT=breakout),
            state="UNKNOWN" if floor["state"] == "UNKNOWN" or breakout is None else
                "PASS" if floor["state"] == "PASS" and breakout else "FAIL")
        if audit["state"] == "PASS":
            audit.update(setup_variant=LEGACY_FLOOR_VARIANT, support_method=floor["method"])
            checks = tuple((name, tuple(audit["checks"].items()) if name == "LEADERSHIP_CONTINUATION" else items)
                for name, items in initial.evidence)
            return ContextSelection("LEADERSHIP_CONTINUATION", checks), audit
        return initial, audit
    audit.update(state="NOT_APPLICABLE", reasons=[missing or "ORIGINAL_CONTEXT_PRESERVED"])
    return initial, audit


def _native_selection(bars, reference_policy, cutoff, *, session_evidence=None, selection_policy=None):
    selection_policy = _selection_policy(selection_policy, cutoff, reference_policy=reference_policy)
    from domain.les_v7.rule_baseline import prepare_baseline, RELAXED_BASELINE_POLICY
    prepared = prepare_baseline(r16.select_bars(bars, cutoff), market=reference_policy["cost_path_policy"]["market"],
        cutoff=cutoff, policy_version=RELAXED_BASELINE_POLICY)
    prepared["features"] = price_structure_features(bars, prepared["features"], cutoff=cutoff)
    return prepare_selection(select_context(prepared["features"]), prepared["features"], bars=bars,
        cutoff=cutoff, session_evidence=session_evidence, market=reference_policy["cost_path_policy"]["market"],
        selection_policy=selection_policy)


def make_plan(context, features, reference_policy, evidence_id, *, bars=None, cutoff,
              setup_variant=None, session_evidence=None, selection_policy=None):
    from domain.les_v7.plan_r10 import _pivots, _support, _resistance
    require_policy(reference_policy, cutoff)
    selection_policy = _selection_policy(selection_policy, cutoff, reference_policy=reference_policy)
    if bars is None or bars.empty:
        return None, dict(schema=PLAN_SCHEMA, state="UNKNOWN", reasons=["SOURCE_BOUND_COMPLETED_BARS_MISSING"])
    selection, audit = _native_selection(bars, reference_policy, cutoff, session_evidence=session_evidence,
        selection_policy=selection_policy)
    variant = audit.get("setup_variant")
    if selection.context != context or setup_variant is not None and setup_variant != variant:
        raise ValueError("R17_SOURCE_BOUND_CONTEXT_VARIANT_MISMATCH")
    sidecar = dict(schema=PLAN_SCHEMA, policy_id=identity(reference_policy), evidence_id=evidence_id,
        cutoff=timestamp(cutoff).isoformat(), context=context, state="UNKNOWN", reasons=[], anchors=None,
        bars_id=audit.get("bars_id"), setup_variant=variant, short_base_audit=audit, conditional_entry=None,
        source_verification="INHERITED_NATIVE_EVIDENCE_BINDING_NOT_CERTIFIED_HERE")
    if context not in CONTEXTS:
        sidecar["reasons"] = ["NO_SELECTED_CAUSAL_SETUP"]
        return None, sidecar
    if audit["state"] == "UNKNOWN" or audit.get("candidate_variant") is not None and audit["state"] != "PASS":
        sidecar["reasons"] = audit["reasons"]
        return None, sidecar
    _, all_rows = r16._rows(bars, cutoff)
    rows, last = all_rows[-124:], all_rows[-1]
    atr, close = r16._num(features.get("atr")), r16._num(features.get("close"))
    _, source_atr = r16._atr(r16.select_bars(bars, cutoff))
    if atr is None or close is None or close != last["Close"] or atr != source_atr:
        raise ValueError("R17_PLAN_SOURCE_CLOSE_OR_ATR_MISMATCH")
    lows, highs = _pivots(rows, "Low", low=True), _pivots(rows, "High", low=False)
    if variant is not None:
        support, method = deepcopy(audit.get("support")), audit.get("support_method", variant)
    else:
        support, missing, method = _support(rows, lows, highs, context, atr)
        if support is None:
            sidecar["reasons"] = [missing]
            return None, sidecar
    if support is None:
        sidecar["reasons"] = ["CAUSAL_SUPPORT_MISSING"]
        return None, sidecar
    support["index"] = next(index for index, row in enumerate(rows) if row["Date"] == support["session"])
    overhead = _resistance(rows, highs, close, max(0, len(rows) - 120))
    stop = support["value"] - .3 * atr
    if not 0 < stop < close or last["Low"] < support["value"]:
        sidecar["reasons"] = ["SELECTED_STRUCTURE_BROKEN_OR_STOP_INVALID"]
        return None, sidecar
    if any(row["Low"] <= stop for row in rows[support["index"] + 1:]):
        sidecar["reasons"] = ["SELECTED_STRUCTURE_STOP_ALREADY_VIOLATED"]
        return None, sidecar
    target = audit["metrics"]["target"] if variant == C_VARIANT else overhead["value"] if overhead else None
    lower, upper = ((support["value"] + .5 * atr, support["value"] + atr) if variant == A_VARIANT else
                    (close, close + .25 * atr))
    band = tuple(float(_round_price(Decimal(str(value)), direction)) for value, direction in ((lower, "up"), (upper, "down")))
    entry = band[1] if variant == A_VARIANT else close
    basis, target_method = (A_ENTRY_BASIS if variant == A_VARIANT else CLOSE_ENTRY_BASIS,
        PROJECTED_TARGET if variant == C_VARIANT else OBSERVED_TARGET)
    sidecar.update(entry_basis=basis, signal_close=close, target_method=target_method,
        anchors=dict(support=support, support_method=method, resistance=deepcopy(overhead),
            target_projection=dict(method=PROJECTED_TARGET, high20=audit["metrics"]["prior20_high"],
                low20=audit["metrics"]["prior20_low"], multiple=.5, value=target) if variant == C_VARIANT else None,
            stop_buffer_atr=.3, selection_order="SETUP_ANCHORS_BEFORE_COSTS_OR_RR"))
    if band[0] > band[1] or band[0] <= stop or target is not None and target <= entry:
        sidecar["reasons"] = ["NO_VALID_EXECUTABLE_BAND_OR_FIXED_TARGET"]
        return None, sidecar
    structure_id = identity(dict(selection=audit, anchors=sidecar["anchors"], entry_basis=basis,
        signal_close=close, band=list(band)))
    shares = reference_policy["reference_shares"]
    costs = estimate_reference_costs(entry, stop, target, shares, reference_policy["cost_path_policy"])
    plan = R17ReferencePlan("REFERENCE_H20", PLAN_VERSION, evidence_id, context, entry, stop, target, atr,
        band, shares, costs["cost_up_per_share"], costs["cost_stop_per_share"], entry_basis=basis,
        signal_close=close, setup_variant=variant, target_method=target_method, structure_id=structure_id)
    # Ordinary close plans retain the existing R15 contiguous pass-prefix
    # construction. New A/C bands remain fixed in full, never RR-selected.
    prefix = None
    if variant not in NEW_VARIANTS:
        from domain.les_v7.plan_r10 import _pass_prefix
        old = r16._plain_plan(_as_r16_plan(plan))
        old_policy = r16._as_r15_policy(_as_r16_policy(reference_policy))
        core = evaluate_entry(plan, plan.entry, reference_policy)
        old, prefix = _pass_prefix(old, old_policy, starter=variant is not None or core["state"] != "PASS")
        values = asdict(plan); values["band"] = old.band
        plan = R17ReferencePlan(**values)
    sidecar.update(state="VERIFIED_ANCHORS" if target is not None else "PARTIAL",
        reasons=[] if target is not None else ["EVIDENCE_DERIVED_RESISTANCE_MISSING"],
        plan_id=plan.plan_id, structure_id=structure_id, formal_entry_basis=basis,
        entry_band_selection=prefix, close_evaluation=evaluate_entry(plan, plan.entry, reference_policy),
        band_evaluation=validate_band(plan, reference_policy),
        starter_band_evaluation=validate_band(plan, reference_policy, starter=True),
        conditional_entry=dict(state="UNFILLED", entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND",
            expiry="NEXT_SESSION_CLOSE", band=list(plan.band), entry_basis=basis,
            signal_close=close, live_order_authorized=False))
    return plan, sidecar


def bind_entry_route(plan, reference_policy, *, strategy_decision, risk_route, setup_variant=None):
    require_policy(reference_policy)
    if not isinstance(plan, R17ReferencePlan): raise ValueError("R17_SOURCE_BOUND_PLAN_REQUIRED")
    old_variant = _TO_R16.get(setup_variant, setup_variant)
    result = r16.bind_entry_route(_as_r16_plan(plan), _as_r16_policy(reference_policy),
        strategy_decision=strategy_decision, risk_route=risk_route, setup_variant=old_variant)
    result = {key: value for key, value in result.items() if key != "route_id"}
    result.update(schema=ROUTE_SCHEMA, plan_id=plan.plan_id,
        reference_policy_id=identity(reference_policy), reference_policy=deepcopy(reference_policy),
        setup_variant=plan.setup_variant)
    return dict(result, route_id=identity(result))
