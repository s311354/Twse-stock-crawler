"""R18 source-bound support/resistance LIMIT plans, never assumed fills.

R17 remains the first-choice causal plan.  The neutral fallback fixes the
latest independently confirmed support and nearest observed resistance before
costs or RR are inspected.  Its public context remains NO_CLEAR_CONTEXT; the
three historical pattern labels are not fabricated to grant eligibility.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from decimal import Decimal

from domain.les_v7 import plan_r17 as previous
from domain.les_v7.contracts import CONTEXTS, identity, timestamp
from domain.les_v7.context_r17 import price_structure_features, select_context
from domain.les_v7.policy_r18 import AVAILABLE_AT, PLAN_VERSION, SUPPORT_RESISTANCE
from domain.les_v7.reference_path import _legal, _round_price, _tick, estimate_reference_costs
from domain.les_v7.session_coverage_r17 import verify_window
from domain.strategies.low_entry_v6_hybrid import ContextSelection, NO_CLEAR_CONTEXT, known_features

A_VARIANT, C_VARIANT = previous.A_VARIANT, previous.C_VARIANT
NEUTRAL_VARIANT = SUPPORT_RESISTANCE
NEW_VARIANTS = previous.NEW_VARIANTS | {NEUTRAL_VARIANT}
SCHEMA = "les-v7-r18-structure-selection/v1"
PLAN_SCHEMA = "les-v7-setup-plan-evidence/v4"
ROUTE_SCHEMA = "les-v7-reference-entry-route/v4"
A_ENTRY_BASIS, CLOSE_ENTRY_BASIS = previous.A_ENTRY_BASIS, previous.CLOSE_ENTRY_BASIS
OBSERVED_TARGET, PROJECTED_TARGET = previous.OBSERVED_TARGET, previous.PROJECTED_TARGET


def structural_policy():
    return dict(version=PLAN_VERSION, previous=previous.structural_policy(),
        neutral_variant=NEUTRAL_VARIANT, neutral_context=NO_CLEAR_CONTEXT,
        selection="KEEP_EXISTING_CAUSAL_PLAN_ELSE_LATEST_CONFIRMED_SUPPORT_AND_NEAREST_RESISTANCE",
        support_lookback=60, resistance_lookback=120, pivot_left=2, pivot_right=2,
        anchor_confirmation="TWO_RIGHT_COMPLETED_SESSIONS_BEFORE_SIGNAL_SESSION",
        official_window="VERIFIED_PRIOR120_AND_SIGNAL_SESSION",
        anchor_trading="POSITIVE_VOLUME_ON_ANCHORS_CONFIRMATIONS_AND_SIGNAL",
        support_must_remain_unbroken=True, target_projection_allowed=False,
        entry_band_atr=[.5, 1.], entry_band_cap="ACTUAL_SIGNAL_CLOSE_BEFORE_RR",
        stop_buffer_atr=.3, tactical_maximum_stop_atr=2.5, tactical_minimum_net_rr=1.2,
        neutral_route="BUY_SMALL_TACTICAL_ONLY", entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND",
        order_type="LIMIT", expiry="NEXT_SESSION_CLOSE", position_limits_affect_score=False,
        performance_validated=False)


def policy(reference_policy, *, available_at):
    if timestamp(available_at) < timestamp(AVAILABLE_AT):
        raise ValueError("R18_REFERENCE_POLICY_BEFORE_RELEASE")
    result = previous.policy(reference_policy, available_at=available_at)
    result.update(version=PLAN_VERSION, support_rule="SOURCE_BOUND_SUPPORT_RESISTANCE_LIMIT_R18",
        conditional_entry="NEXT_OFFICIAL_SESSION_OPEN_IN_FIXED_LIMIT_BAND_R18")
    result["structural_rules"]["r18"] = structural_policy()
    return result


def _as_previous_policy(reference_policy):
    result = deepcopy(dict(reference_policy))
    result.update(version=previous.PLAN_VERSION, support_rule="CAUSAL_STRUCTURES_SINGLE_SOFT_SCORE_R17",
        conditional_entry="NEXT_OFFICIAL_SESSION_OPEN_IN_STRUCTURAL_BAND_R17")
    result["structural_rules"].pop("r18", None)
    return result


def require_policy(reference_policy, cutoff=None):
    previous._selection_policy(None, cutoff, reference_policy=reference_policy)
    if (reference_policy.get("version") != PLAN_VERSION
            or reference_policy.get("support_rule") != "SOURCE_BOUND_SUPPORT_RESISTANCE_LIMIT_R18"
            or reference_policy.get("conditional_entry") != "NEXT_OFFICIAL_SESSION_OPEN_IN_FIXED_LIMIT_BAND_R18"
            or reference_policy.get("structural_rules", {}).get("r18") != structural_policy()):
        raise ValueError("R18_REFERENCE_POLICY_MISMATCH")
    if timestamp(reference_policy["available_at"]) < timestamp(AVAILABLE_AT):
        raise ValueError("R18_REFERENCE_POLICY_BEFORE_RELEASE")
    previous.require_policy(_as_previous_policy(reference_policy), cutoff)
    return reference_policy


def _as_previous_plan(plan):
    values = asdict(plan)
    values["policy_version"] = previous.PLAN_VERSION
    if plan.setup_variant == NEUTRAL_VARIANT:
        # An arithmetic-only proxy uses the existing conditional-band shape;
        # neither source selection nor the public report is relabeled.
        values.update(context="TREND_PULLBACK", setup_variant=previous.A_VARIANT)
    return previous.R17ReferencePlan(**values)


@dataclass(frozen=True)
class R18ReferencePlan(previous.R17ReferencePlan):
    def __post_init__(self):
        if self.policy_version != PLAN_VERSION:
            raise ValueError("R18_PLAN_VERSION_REQUIRED")
        if self.setup_variant == NEUTRAL_VARIANT:
            if self.context != NO_CLEAR_CONTEXT:
                raise ValueError("R18_NEUTRAL_CONTEXT_MUST_NOT_BE_RELABELLED")
        elif self.context not in CONTEXTS:
            raise ValueError("R18_ORIGINAL_CONTEXT_REQUIRED")
        _as_previous_plan(self)


def plan_from_record(record):
    if record is None:
        return None
    if record.get("policy_version") != PLAN_VERSION:
        return previous.plan_from_record(record)
    return R18ReferencePlan(**dict(record, band=tuple(record["band"])))


def evaluate_entry(plan, observed_entry, reference_policy, *, starter=False):
    require_policy(reference_policy)
    if plan is None:
        return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    if not isinstance(plan, R18ReferencePlan):
        raise ValueError("R18_SOURCE_BOUND_PLAN_REQUIRED")
    if plan.setup_variant == NEUTRAL_VARIANT and not starter:
        return dict(state="FAIL", reasons=["R18_NEUTRAL_REQUIRES_TACTICAL_SMALL"])
    result = previous.evaluate_entry(_as_previous_plan(plan), observed_entry,
        _as_previous_policy(reference_policy), starter=starter)
    if not starter or "costs" not in result:
        return result
    # Reuse identical fees and legal ticks, but not R17's obsolete thresholds.
    decimal = lambda value: Decimal(str(value))
    entry = decimal(result["entry"])
    costs = result["costs"]
    stop_atr = (entry - decimal(plan.stop)) / decimal(plan.atr)
    rr = (decimal(plan.target) - entry - decimal(costs["cost_up_per_share"])) / (
        entry - decimal(plan.stop) + decimal(costs["cost_stop_per_share"]))
    rules = reference_policy["structural_rules"]["r18"]
    maximum, minimum = rules["tactical_maximum_stop_atr"], rules["tactical_minimum_net_rr"]
    reasons = []
    if stop_atr > decimal(maximum): reasons.append("FIXED_STRUCTURE_STOP_ATR_EXCEEDED")
    if rr < decimal(minimum): reasons.append("FIXED_RESISTANCE_NET_RR_BELOW_MINIMUM")
    return dict(result, state="FAIL" if reasons else "PASS", reasons=reasons,
        stop_atr=float(stop_atr), net_rr=float(rr), maximum_stop_atr=maximum, minimum_net_rr=minimum)


def validate_band(plan, reference_policy, *, starter=False):
    require_policy(reference_policy)
    if plan is None:
        return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    low, high = (_round_price(Decimal(str(value)), direction)
        for value, direction in zip(plan.band, ("up", "down")))
    if low > high or low <= Decimal(str(plan.stop)):
        return dict(state="FAIL", reasons=["NO_VALID_EXECUTABLE_ENTRY_BAND"])
    result = evaluate_entry(plan, float(high), reference_policy, starter=starter)
    return dict(result, band=list(plan.band), executable_band=[float(low), float(high)],
        evaluation_basis="HIGHEST_PERMITTED_LEGAL_ENTRY_TICK")


def _neutral(initial, features, *, bars, cutoff, session_evidence, market, prior_audit,
             selection_policy=None):
    selection_policy = previous._selection_policy(selection_policy, cutoff)
    from domain.les_v7.plan_r10 import _pivots, _resistance
    audit = dict(schema=SCHEMA, policy_id=identity(structural_policy()),
        cutoff=timestamp(cutoff).isoformat(), original_selection=initial.record(),
        previous_selection=deepcopy(prior_audit), state="UNKNOWN", setup_variant=None,
        candidate_variant=NEUTRAL_VARIANT, support=None, resistance=None, reasons=[],
        priority="ORIGINAL_CAUSAL_PLAN_THEN_NEUTRAL_FIXED_ANCHORS_BEFORE_RR")
    _, all_rows = previous.r16._rows(bars, cutoff)
    rows = all_rows[-125:]
    audit["bars_id"] = identity(rows)
    if len(rows) < 121:
        audit["reasons"] = ["R18_PRIOR_120_SESSION_HISTORY_MISSING"]
        return initial, audit
    coverage = verify_window(session_evidence, market=market, cutoff=cutoff,
        observed_sessions=[row["Date"] for row in rows[-121:]], prior_sessions=120)
    audit["session_coverage"] = coverage
    if coverage["state"] != "PASS":
        audit.update(state=coverage["state"], reasons=["R18_OFFICIAL_SESSION_COVERAGE_" + coverage["state"]])
        return initial, audit
    if rows[-1].get("Volume") is not None and rows[-1]["Volume"] <= 0:
        audit.update(state="FAIL", reasons=["R18_ANCHOR_OR_CONFIRMATION_DID_NOT_TRADE"])
        return initial, audit
    if selection_policy is not None:
        f = known_features(features)
        atr, close = previous.r16._num(f.get("atr14")), previous.r16._num(f.get("close"))
        if atr is None or atr <= 0 or close is None or close != rows[-1]["Close"]:
            audit["reasons"] = ["SOURCE_CLOSE_OR_ATR_MISSING_OR_CONFLICTING"]
            return initial, audit
        if rows[-1].get("Volume") is None:
            audit["reasons"] = ["R18_ANCHOR_TRADING_EVIDENCE_MISSING"]
            return initial, audit
        if not _legal(Decimal(str(close))):
            audit.update(state="FAIL", reasons=["R18_SIGNAL_CLOSE_OUTSIDE_LEGAL_PRICE_GRID"])
            return initial, audit
    prior = rows[:-1]
    lows = _pivots(prior, "Low", low=True)
    candidates = [point for point in lows if point["index"] >= max(0, len(prior) - 60)]
    if not candidates:
        audit["reasons"] = ["R18_INDEPENDENTLY_CONFIRMED_SUPPORT_MISSING"]
        if selection_policy is not None:
            audit.update(state="NO_MATCH", source_selection_policy=selection_policy,
                reasons=["R22_NO_CONFIRMED_SUPPORT_IN_VERIFIED_WINDOW"],
                source_coverage="VERIFIED_NO_MATCHING_STRUCTURAL_ANCHOR")
        return initial, audit
    support = deepcopy(candidates[-1])
    resistance = _resistance(rows, _pivots(prior, "High", low=False), rows[-1]["Close"],
        max(0, len(prior) - 120))
    audit.update(support=support, resistance=deepcopy(resistance))
    anchor_rows = {len(rows) - 1, support["index"], support["index"] + 1, support["index"] + 2}
    if resistance is not None:
        anchor_rows.update((resistance["index"], resistance["index"] + 1, resistance["index"] + 2))
    volumes = [rows[index].get("Volume") for index in sorted(anchor_rows)]
    if any(value is None for value in volumes):
        audit.update(state="UNKNOWN", reasons=["R18_ANCHOR_TRADING_EVIDENCE_MISSING"])
        return initial, audit
    if any(value <= 0 for value in volumes):
        audit.update(state="FAIL", reasons=["R18_ANCHOR_OR_CONFIRMATION_DID_NOT_TRADE"])
        return initial, audit
    f = known_features(features)
    atr, close = previous.r16._num(f.get("atr14")), previous.r16._num(f.get("close"))
    if atr is None or atr <= 0 or close is None or close != rows[-1]["Close"]:
        audit["reasons"] = ["SOURCE_CLOSE_OR_ATR_MISSING_OR_CONFLICTING"]
        return initial, audit
    lower = support["value"] + .5 * atr
    upper = min(support["value"] + atr, close)
    band = tuple(float(_round_price(Decimal(str(value)), direction))
        for value, direction in ((lower, "up"), (upper, "down")))
    stop = support["value"] - .3 * atr
    audit.update(support=support, resistance=deepcopy(resistance), band=list(band),
        support_method="LATEST_PRIOR60_INDEPENDENTLY_CONFIRMED_UNBROKEN_LOW",
        metrics=dict(signal_atr=atr, signal_close=close, stop=stop,
            target=resistance["value"] if resistance is not None else None))
    if not _legal(Decimal(str(close))):
        audit.update(state="FAIL", reasons=["R18_SIGNAL_CLOSE_OUTSIDE_LEGAL_PRICE_GRID"])
    elif any(row["Low"] < support["value"] for row in rows[support["index"] + 1:]):
        audit.update(state="FAIL", reasons=["R18_CONFIRMED_SUPPORT_BROKEN"])
    elif not 0 < stop < band[0] <= band[1] <= close:
        audit.update(state="FAIL", reasons=["R18_FIXED_LIMIT_BAND_INVALID"])
    elif resistance is None:
        audit["reasons"] = ["R18_OBSERVED_UNBROKEN_RESISTANCE_MISSING"]
        if selection_policy is not None:
            audit.update(state="NO_MATCH", source_selection_policy=selection_policy,
                reasons=["R22_NO_OBSERVED_OVERHEAD_RESISTANCE_IN_VERIFIED_WINDOW"],
                source_coverage="VERIFIED_NO_MATCHING_STRUCTURAL_ANCHOR")
    elif resistance["value"] <= band[1]:
        audit.update(state="FAIL", reasons=["R18_FIXED_RESISTANCE_NOT_ABOVE_LIMIT"])
    else:
        audit.update(state="PASS", setup_variant=NEUTRAL_VARIANT)
    return (ContextSelection(NO_CLEAR_CONTEXT, initial.evidence) if audit["state"] == "PASS" else initial), audit


def prepare_selection(initial, features, *, bars, cutoff, session_evidence=None, market=None,
                      selection_policy=None):
    selection_policy = previous._selection_policy(selection_policy, cutoff)
    if not isinstance(initial, ContextSelection):
        raise ValueError("R18_ORIGINAL_CONTEXT_SELECTION_REQUIRED")
    selected, audit = previous.prepare_selection(initial, features, bars=bars, cutoff=cutoff,
        session_evidence=session_evidence, market=market, selection_policy=selection_policy)
    # Never replace a selected causal structure to improve its risk ratio.
    if audit.get("setup_variant") is not None or (
            selected.context in CONTEXTS and audit.get("support") is not None
            and audit.get("original_resistance") is not None):
        return selected, audit
    if bars is None or bars.empty:
        return selected, audit
    return _neutral(initial, features, bars=bars, cutoff=cutoff,
        session_evidence=session_evidence, market=market, prior_audit=audit,
        selection_policy=selection_policy)


def _native_selection(bars, reference_policy, cutoff, *, session_evidence=None, selection_policy=None):
    selection_policy = previous._selection_policy(selection_policy, cutoff, reference_policy=reference_policy)
    from domain.les_v7.rule_baseline import prepare_baseline, RELAXED_BASELINE_POLICY
    market = reference_policy["cost_path_policy"]["market"]
    prepared = prepare_baseline(previous.r16.select_bars(bars, cutoff), market=market,
        cutoff=cutoff, policy_version=RELAXED_BASELINE_POLICY)
    features = price_structure_features(bars, prepared["features"], cutoff=cutoff)
    return prepare_selection(select_context(features), features, bars=bars, cutoff=cutoff,
        session_evidence=session_evidence, market=market, selection_policy=selection_policy)


def _prefix(plan, reference_policy, *, starter):
    """Retain the predeclared close band, clipping only its contiguous prefix."""
    check = evaluate_entry(plan, plan.entry, reference_policy, starter=starter)
    audit = dict(policy="R18_CONTIGUOUS_LEGAL_PASS_PREFIX", original_band=list(plan.band),
        state=check["state"], close_evaluation=check, checked_legal_ticks=0, first_nonpass=None,
        executable_prefix=None)
    if check["state"] != "PASS":
        audit["reason"] = "CUTOFF_CLOSE_MUST_PASS_NO_LOWER_PRICE_SUBSTITUTION"
        return plan, audit
    price, last, passed = Decimal(str(plan.entry)), Decimal(str(plan.band[1])), Decimal(str(plan.entry))
    while price <= last:
        result = evaluate_entry(plan, float(price), reference_policy, starter=starter)
        audit["checked_legal_ticks"] += 1
        if result["state"] != "PASS":
            audit["first_nonpass"] = dict(entry=float(price), check=result)
            break
        passed = price
        price += _tick(price)
    updated = replace(plan, band=(plan.entry, float(passed)))
    audit.update(state="PASS", executable_prefix=list(updated.band), clipped=updated.band != plan.band)
    return updated, audit


def make_plan(context, features, reference_policy, evidence_id, *, bars=None, cutoff,
              setup_variant=None, session_evidence=None, selection_policy=None):
    require_policy(reference_policy, cutoff)
    selection_policy = previous._selection_policy(selection_policy, cutoff, reference_policy=reference_policy)
    if bars is None or bars.empty:
        return None, dict(schema=PLAN_SCHEMA, state="UNKNOWN", reasons=["SOURCE_BOUND_COMPLETED_BARS_MISSING"])
    selected, audit = _native_selection(bars, reference_policy, cutoff, session_evidence=session_evidence,
        selection_policy=selection_policy)
    variant = audit.get("setup_variant")
    if selected.context != context or setup_variant is not None and setup_variant != variant:
        raise ValueError("R18_SOURCE_BOUND_CONTEXT_VARIANT_MISMATCH")
    if audit.get("candidate_variant") == NEUTRAL_VARIANT and audit["state"] != "PASS":
        return None, dict(schema=PLAN_SCHEMA, policy_id=identity(reference_policy), evidence_id=evidence_id,
            cutoff=timestamp(cutoff).isoformat(), context=context, state=audit["state"],
            reasons=list(audit["reasons"]), anchors=None, bars_id=audit.get("bars_id"),
            setup_variant=None, short_base_audit=audit, r18_selection=audit,
            conditional_entry=None, source_verification="INHERITED_NATIVE_EVIDENCE_BINDING_NOT_CERTIFIED_HERE")
    if variant != NEUTRAL_VARIANT:
        plan, sidecar = previous.make_plan(context, features, _as_previous_policy(reference_policy),
            evidence_id, bars=bars, cutoff=cutoff, setup_variant=variant, session_evidence=session_evidence,
            selection_policy=selection_policy)
        sidecar.update(schema=PLAN_SCHEMA, policy_id=identity(reference_policy), r18_selection=audit)
        if plan is None:
            return None, sidecar
        values = asdict(plan)
        prefix = sidecar.get("entry_band_selection")
        if prefix is not None:
            values["band"] = tuple(prefix["original_band"])
        values["policy_version"] = PLAN_VERSION
        plan = R18ReferencePlan(**values)
        if variant not in previous.NEW_VARIANTS:
            core = evaluate_entry(plan, plan.entry, reference_policy)
            plan, prefix = _prefix(plan, reference_policy, starter=variant is not None or core["state"] != "PASS")
            sidecar["entry_band_selection"] = prefix
    else:
        _, atr = previous.r16._atr(previous.r16.select_bars(bars, cutoff))
        if features.get("close") != audit["metrics"]["signal_close"] or features.get("atr") != atr:
            raise ValueError("R18_PLAN_SOURCE_CLOSE_OR_ATR_MISMATCH")
        support, resistance = deepcopy(audit["support"]), deepcopy(audit["resistance"])
        entry, stop, target = audit["band"][1], audit["metrics"]["stop"], resistance["value"]
        anchors = dict(support=support, support_method=audit["support_method"], resistance=resistance,
            target_projection=None, stop_buffer_atr=.3, selection_order="FIXED_ANCHORS_AND_BAND_BEFORE_COSTS_OR_RR")
        structure_id = identity(dict(selection=audit, anchors=anchors, band=audit["band"]))
        shares = reference_policy["reference_shares"]
        costs = estimate_reference_costs(entry, stop, target, shares, reference_policy["cost_path_policy"])
        plan = R18ReferencePlan("REFERENCE_H20", PLAN_VERSION, evidence_id, NO_CLEAR_CONTEXT,
            entry, stop, target, atr, tuple(audit["band"]), shares,
            costs["cost_up_per_share"], costs["cost_stop_per_share"], entry_basis=A_ENTRY_BASIS,
            signal_close=audit["metrics"]["signal_close"], setup_variant=NEUTRAL_VARIANT,
            target_method=OBSERVED_TARGET, structure_id=structure_id)
        sidecar = dict(schema=PLAN_SCHEMA, policy_id=identity(reference_policy), evidence_id=evidence_id,
            cutoff=timestamp(cutoff).isoformat(), context=context, state="VERIFIED_ANCHORS", reasons=[],
            anchors=anchors, bars_id=audit["bars_id"], setup_variant=variant,
            short_base_audit=audit, r18_selection=audit, entry_basis=plan.entry_basis,
            signal_close=plan.signal_close, target_method=plan.target_method,
            source_verification="INHERITED_NATIVE_EVIDENCE_BINDING_NOT_CERTIFIED_HERE")
    sidecar.update(plan_id=plan.plan_id, structure_id=plan.structure_id,
        formal_entry_basis=plan.entry_basis, close_evaluation=evaluate_entry(plan, plan.entry, reference_policy),
        band_evaluation=validate_band(plan, reference_policy),
        starter_band_evaluation=validate_band(plan, reference_policy, starter=True),
        conditional_entry=dict(state="UNFILLED", order_type="LIMIT",
            entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", expiry="NEXT_SESSION_CLOSE",
            band=list(plan.band), entry_basis=plan.entry_basis, signal_close=plan.signal_close,
            live_order_authorized=False, assumes_fill=False))
    return plan, sidecar


def bind_entry_route(plan, reference_policy, *, strategy_decision, risk_route, setup_variant=None):
    require_policy(reference_policy)
    if not isinstance(plan, R18ReferencePlan) or reference_policy.get("reference_shares") != plan.reference_shares:
        raise ValueError("R18_SOURCE_BOUND_PLAN_AND_REFERENCE_SIZE_REQUIRED")
    plan_from_record(asdict(plan))
    if not all(_legal(Decimal(str(value))) for value in (*plan.band, plan.entry)):
        raise ValueError("R18_ENTRY_BAND_LEGAL_TICKS_REQUIRED")
    if setup_variant is not None and setup_variant != plan.setup_variant:
        raise ValueError("R18_ENTRY_ROUTE_VARIANT_MISMATCH")
    if strategy_decision not in {"BUY", "BUY_SMALL"} or risk_route not in {"CORE", "TACTICAL"}:
        raise ValueError("R18_ENTRY_ROUTE_DECISION_OR_ROUTE_INVALID")
    if plan.setup_variant is not None and (strategy_decision != "BUY_SMALL" or risk_route != "TACTICAL"):
        raise ValueError("R18_VARIANT_REQUIRES_TACTICAL_SMALL")
    if strategy_decision == "BUY" and risk_route != "CORE":
        raise ValueError("R18_BUY_REQUIRES_CORE")
    if (evaluate_entry(plan, plan.entry, reference_policy, starter=risk_route == "TACTICAL")["state"] != "PASS"
            or validate_band(plan, reference_policy, starter=risk_route == "TACTICAL")["state"] != "PASS"):
        raise ValueError("R18_ENTRY_ROUTE_NATIVE_RISK_NOT_PASS")
    result = dict(schema=ROUTE_SCHEMA, plan_id=plan.plan_id, reference_policy_id=identity(reference_policy),
        reference_policy=deepcopy(reference_policy), risk_route=risk_route, strategy_decision=strategy_decision,
        setup_variant=plan.setup_variant, entry_basis=plan.entry_basis, signal_close=plan.signal_close,
        target_method=plan.target_method, structure_id=plan.structure_id,
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", order_type="LIMIT", expiry="NEXT_SESSION_CLOSE",
        authority="REFERENCE_RESEARCH_ROUTE_NOT_LIVE_ORDER_AUTHORITY")
    return dict(result, route_id=identity(result))
