"""R10 causal, setup-bound H20 plans; no price or risk qualification hacks.

The cutoff close remains the formal reference entry. A lower structural entry
zone is only an UNFILLED diagnostic, never a replacement close, signal or fill.
Stop and resistance are selected before costs/RR are calculated. A resistance
is broken only by a subsequent completed close above it. Missing anchors stay
unknown; no +2R target or generic older-support fallback is manufactured.

The existing ReferencePlan contract and legal-grid/fee implementation are reused.
All new anchor/conditional evidence lives in a separate, compact sidecar.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal

from domain.les_v7.contracts import CONTEXTS, ReferencePlan, finite, identity, timestamp
from domain.les_v7.features import select_bars
from domain.les_v7.reference_path import (_legal, _round_price, _tick, estimate_reference_costs,
                                         validate_reference_path_policy)
from domain.les_v7.policy_r13 import (AVAILABLE_AT as R13_AVAILABLE_AT,
                                     PLAN_VERSION as R13_VERSION, risk_limits)

VERSION = "V7_SETUP_STRUCTURE_H20_R10"
SCHEMA = "les-v7-setup-plan-evidence/v1"
R14_VERSION = "V7_SETUP_STRUCTURE_H20_R14"
R14_BAND_POLICY = "CONTIGUOUS_LEGAL_TICK_PASS_PREFIX_CLOSE_MUST_PASS_R14"
R15_VERSION = "V7_SETUP_STRUCTURE_H20_R15"
R15_BAND_POLICY = "CORE_ELSE_TACTICAL_CONTIGUOUS_PASS_PREFIX_CLOSE_MUST_PASS_R15"
R15_ROUTE_SCHEMA = "les-v7-reference-entry-route/v1"
_RULES = dict(support_lookback=60, resistance_lookback=120, pivot_left=2,
              pivot_right=2, base_sessions=20, maximum_base_width_atr=4.,
              retest_tolerance_atr=.25, stop_buffer_atr=.3,
              conditional_zone_atr=.5, maximum_stop_atr=1.5,
              starter_stop_atr=dict(REVERSAL=1.2, TREND_PULLBACK=1.3,
                                    LEADERSHIP_CONTINUATION=1.3),
              minimum_net_rr=dict(REVERSAL=2., TREND_PULLBACK=1.8,
                                  LEADERSHIP_CONTINUATION=1.8))


def _rules(version):
    """Keep the causal construction fixed; version only declared risk limits."""
    if version not in {VERSION, R13_VERSION, R14_VERSION, R15_VERSION}:
        raise ValueError("R10_REFERENCE_POLICY_MISMATCH")
    result = deepcopy(_RULES)
    if version in {R13_VERSION, R14_VERSION, R15_VERSION}:
        limits = risk_limits()
        for key in ("maximum_stop_atr", "starter_stop_atr", "minimum_net_rr"):
            result[key] = deepcopy(limits[key])
    if version == R14_VERSION:
        from domain.les_v7.context_r14 import context_policy
        result.update(early_short_base_policy=context_policy(), entry_band_policy=R14_BAND_POLICY)
    if version == R15_VERSION:
        from domain.les_v7.context_r15 import context_policy
        from domain.les_v7.policy_r15 import tactical_risk_limits
        tactical = tactical_risk_limits()
        result.update(early_short_base_policy=context_policy(), entry_band_policy=R15_BAND_POLICY,
            starter_stop_atr={name: tactical["maximum_stop_atr"] for name in CONTEXTS},
            starter_minimum_net_rr={name: tactical["minimum_net_rr"] for name in CONTEXTS})
    return result


def policy(reference_policy, *, available_at, version=VERSION):
    """Version a declared old reference/cost policy without altering old data."""
    if version == "V7_RISK_BOUNDED_ENTRY_R19":
        from domain.les_v7.plan_r19 import policy as bounded_policy
        return bounded_policy(reference_policy, available_at=available_at)
    if version == "V7_SUPPORT_RESISTANCE_LIMIT_R18":
        from domain.les_v7.plan_r18 import policy as limit_policy
        return limit_policy(reference_policy, available_at=available_at)
    if version == "V7_SETUP_STRUCTURE_H20_R17":
        from domain.les_v7.plan_r17 import policy as unified_policy
        return unified_policy(reference_policy, available_at=available_at)
    if version == "V7_SETUP_STRUCTURE_H20_R16":
        from domain.les_v7.plan_r16 import policy as r16_policy
        return r16_policy(reference_policy, available_at=available_at)
    timestamp(available_at)
    rules = _rules(version)
    if version == R13_VERSION and timestamp(available_at) < timestamp(R13_AVAILABLE_AT):
        raise ValueError("R13_REFERENCE_POLICY_BEFORE_RELEASE")
    if version == R14_VERSION:
        from domain.les_v7.policy_r14 import AVAILABLE_AT
        if timestamp(available_at) < timestamp(AVAILABLE_AT):
            raise ValueError("R14_REFERENCE_POLICY_BEFORE_RELEASE")
    if version == R15_VERSION:
        from domain.les_v7.policy_r15 import AVAILABLE_AT
        if timestamp(available_at) < timestamp(AVAILABLE_AT):
            raise ValueError("R15_REFERENCE_POLICY_BEFORE_RELEASE")
    result = deepcopy(dict(reference_policy))
    if result.get("policy_id") != "REFERENCE_H20":
        raise ValueError("REFERENCE_H20_POLICY_REQUIRED")
    result.update(version=version, available_at=available_at,
                  entry_rule="CUTOFF_CLOSE_NEXT_SESSION_BAND",
                  support_rule="SETUP_BOUND_CAUSAL_PIVOTS_R10",
                  structural_rules=rules, conditional_entry="UNFILLED_DIAGNOSTIC_ONLY")
    result["cost_path_policy"]["entry_guard_policy"] = "R10_REPRICE_NATIVE_STOP_AND_RESISTANCE"
    validate_reference_path_policy(result.get("cost_path_policy"))
    return result


def _require_policy(reference_policy, cutoff=None):
    if reference_policy.get("version") == "V7_RISK_BOUNDED_ENTRY_R19":
        from domain.les_v7.plan_r19 import require_policy
        return require_policy(reference_policy, cutoff)
    if reference_policy.get("version") == "V7_SUPPORT_RESISTANCE_LIMIT_R18":
        from domain.les_v7.plan_r18 import require_policy
        return require_policy(reference_policy, cutoff)
    if reference_policy.get("version") == "V7_SETUP_STRUCTURE_H20_R17":
        from domain.les_v7.plan_r17 import require_policy
        return require_policy(reference_policy, cutoff)
    if reference_policy.get("version") == "V7_SETUP_STRUCTURE_H20_R16":
        from domain.les_v7.plan_r16 import require_policy
        return require_policy(reference_policy, cutoff)
    version = reference_policy.get("version")
    rules = _rules(version)
    if (reference_policy.get("policy_id") != "REFERENCE_H20"
            or reference_policy.get("entry_rule") != "CUTOFF_CLOSE_NEXT_SESSION_BAND"
            or reference_policy.get("support_rule") != "SETUP_BOUND_CAUSAL_PIVOTS_R10"
            or reference_policy.get("conditional_entry") != "UNFILLED_DIAGNOSTIC_ONLY"
            or reference_policy.get("structural_rules") != rules):
        raise ValueError("R10_REFERENCE_POLICY_MISMATCH")
    timestamp(reference_policy.get("available_at"))
    if version == R13_VERSION and timestamp(reference_policy["available_at"]) < timestamp(R13_AVAILABLE_AT):
        raise ValueError("R13_REFERENCE_POLICY_BEFORE_RELEASE")
    if version == R14_VERSION:
        from domain.les_v7.policy_r14 import AVAILABLE_AT
        if timestamp(reference_policy["available_at"]) < timestamp(AVAILABLE_AT):
            raise ValueError("R14_REFERENCE_POLICY_BEFORE_RELEASE")
    if version == R15_VERSION:
        from domain.les_v7.policy_r15 import AVAILABLE_AT
        if timestamp(reference_policy["available_at"]) < timestamp(AVAILABLE_AT):
            raise ValueError("R15_REFERENCE_POLICY_BEFORE_RELEASE")
    path = validate_reference_path_policy(reference_policy.get("cost_path_policy"))
    if path.get("entry_guard_policy") != "R10_REPRICE_NATIVE_STOP_AND_RESISTANCE":
        raise ValueError("R10_ACTUAL_ENTRY_GUARD_REQUIRED")
    if cutoff is not None:
        if max(timestamp(reference_policy["available_at"]), timestamp(path["available_at"]),
               timestamp(path["cost_policy"]["available_at"])) > timestamp(cutoff):
            raise ValueError("R10_REFERENCE_POLICY_NOT_AVAILABLE_AT_CUTOFF")
    return reference_policy


def _pivots(rows, key, *, low):
    """Strict 2-left/2-right pivots, confirmed by the second right close."""
    output = []
    for i in range(2, len(rows) - 2):
        value = rows[i][key]
        peers = [rows[j][key] for j in (i - 2, i - 1, i + 1, i + 2)]
        if all(value < other if low else value > other for other in peers):
            output.append(dict(index=i, session=rows[i]["Date"], value=value,
                               confirmed_at_session=rows[i + 2]["Date"],
                               method="STRICT_2_LEFT_2_RIGHT_COMPLETED_CLOSE"))
    return output


def _resistance(rows, highs, entry, lower):
    eligible = [point for point in highs if point["index"] >= lower
                and point["value"] > entry
                and all(row["Close"] <= point["value"] for row in rows[point["index"] + 1:])]
    # The nearest unbroken overhead level is compulsory, not the best-RR level.
    return min(eligible, key=lambda p: (p["value"], -p["index"]), default=None)


def _support(rows, lows, highs, context, atr):
    lower = max(0, len(rows) - _RULES["support_lookback"])
    candidates = [point for point in lows if point["index"] >= lower]
    if not candidates:
        return None, "CONFIRMED_SETUP_LOW_MISSING", None
    if context == "REVERSAL":
        return candidates[-1], None, "CONFIRMED_BOTTOM_LOW"
    if context == "TREND_PULLBACK":
        impulse = [point for point in highs if point["index"] >= lower]
        if not impulse:
            return None, "PRIOR_IMPULSE_HIGH_MISSING", None
        high = impulse[-1]
        post_impulse = [point for point in candidates if point["index"] > high["index"]]
        if not post_impulse:
            return None, "SAME_PULLBACK_CONFIRMED_LOW_MISSING", None
        support = deepcopy(post_impulse[-1])
        support["impulse_high"] = deepcopy(high)
        return support, None, "SAME_IMPULSE_CONFIRMED_PULLBACK_LOW"
    # A retest must follow an observed close breakout, have its own confirmed
    # low near the old H20 level, and finish above that level. No future retest.
    for point in reversed(candidates):
        for i in range(point["index"] - 1, max(20, lower) - 1, -1):
            prior_high = max(row["High"] for row in rows[i - 20:i])
            if (rows[i]["Close"] > prior_high
                    and abs(point["value"] - prior_high) <= .25 * atr
                    and rows[-1]["Close"] >= prior_high):
                support = deepcopy(point)
                support["breakout"] = dict(session=rows[i]["Date"], level=prior_high,
                                            method="CLOSE_ABOVE_PRIOR_20_SESSION_HIGH")
                return support, None, "INDEPENDENTLY_CONFIRMED_BREAKOUT_RETEST_LOW"
    base_start = max(0, len(rows) - _RULES["base_sessions"])
    base = rows[base_start:]
    width = max(row["High"] for row in base) - min(row["Low"] for row in base)
    base_lows = [point for point in candidates if point["index"] >= base_start]
    if width > 4. * atr or not base_lows:
        return None, "CONFIRMED_CONSOLIDATION_BASE_OR_RETEST_MISSING", None
    return min(base_lows, key=lambda p: (p["value"], -p["index"])), None, "CONFIRMED_20_SESSION_BASE_FLOOR"


def evaluate_entry(plan, observed_entry, reference_policy, *, starter=False):
    """Reprice exact fees at one actual legal price; no account authorization."""
    if reference_policy.get("version") == "V7_RISK_BOUNDED_ENTRY_R19":
        from domain.les_v7.plan_r19 import evaluate_entry as bounded_evaluate
        return bounded_evaluate(plan, observed_entry, reference_policy, starter=starter)
    if reference_policy.get("version") == "V7_SUPPORT_RESISTANCE_LIMIT_R18":
        from domain.les_v7.plan_r18 import evaluate_entry as limit_evaluate
        return limit_evaluate(plan, observed_entry, reference_policy, starter=starter)
    if reference_policy.get("version") == "V7_SETUP_STRUCTURE_H20_R17":
        from domain.les_v7.plan_r17 import evaluate_entry as unified_evaluate
        return unified_evaluate(plan, observed_entry, reference_policy, starter=starter)
    if reference_policy.get("version") == "V7_SETUP_STRUCTURE_H20_R16":
        from domain.les_v7.plan_r16 import evaluate_entry as r16_evaluate
        return r16_evaluate(plan, observed_entry, reference_policy, starter=starter)
    _require_policy(reference_policy)
    if plan is None:
        return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    if plan.policy_version != reference_policy["version"]:
        raise ValueError("R10_PLAN_VERSION_REQUIRED")
    try:
        entry = finite(observed_entry, "observed_entry", 0)
    except ValueError:
        return dict(state="UNKNOWN", reasons=["OBSERVED_ENTRY_NOT_VERIFIED"])
    if not _legal(Decimal(str(entry))):
        return dict(state="FAIL", reasons=["ENTRY_OUTSIDE_LEGAL_PRICE_GRID"], entry=entry)
    if entry <= plan.stop:
        return dict(state="FAIL", reasons=["ENTRY_NOT_ABOVE_FIXED_STOP"], entry=entry)
    if plan.target is None:
        return dict(state="UNKNOWN", reasons=["EVIDENCE_DERIVED_RESISTANCE_MISSING"], entry=entry)
    if entry >= plan.target:
        return dict(state="FAIL", reasons=["ENTRY_NOT_BELOW_FIXED_RESISTANCE"], entry=entry)
    costs = estimate_reference_costs(entry, plan.stop, plan.target, plan.reference_shares,
                                     reference_policy["cost_path_policy"])
    risk = entry - plan.stop + costs["cost_stop_per_share"]
    rr = (plan.target - entry - costs["cost_up_per_share"]) / risk
    stop_atr = (entry - plan.stop) / plan.atr
    rules = reference_policy["structural_rules"]
    maximum = rules["starter_stop_atr"][plan.context] if starter else rules["maximum_stop_atr"]
    minimum_rr = (rules["starter_minimum_net_rr"][plan.context]
                  if reference_policy["version"] == R15_VERSION and starter
                  else rules["minimum_net_rr"][plan.context])
    # Preserve R10's exact arithmetic/output. For the new policy, inclusive
    # limits compare declared decimal operands, not subtraction noise such as
    # (111 - 103.8) / 4 == 1.8000000000000007. Fees remain the bound estimates.
    stop_failed = stop_atr > maximum
    rr_failed = rr < minimum_rr
    if reference_policy["version"] in {R13_VERSION, R14_VERSION, R15_VERSION}:
        decimal = lambda value: Decimal(str(value))
        exact_stop = (decimal(entry) - decimal(plan.stop)) / decimal(plan.atr)
        exact_risk = decimal(entry) - decimal(plan.stop) + decimal(costs["cost_stop_per_share"])
        exact_rr = (decimal(plan.target) - decimal(entry) - decimal(costs["cost_up_per_share"])) / exact_risk
        stop_atr, rr = float(exact_stop), float(exact_rr)
        stop_failed = exact_stop > decimal(maximum)
        rr_failed = exact_rr < decimal(minimum_rr)
    reasons = []
    if stop_failed: reasons.append("FIXED_STRUCTURE_STOP_ATR_EXCEEDED")
    if rr_failed: reasons.append("FIXED_RESISTANCE_NET_RR_BELOW_MINIMUM")
    return dict(state="FAIL" if reasons else "PASS", reasons=reasons, entry=entry,
                stop_atr=stop_atr, maximum_stop_atr=maximum, net_rr=rr,
                minimum_net_rr=minimum_rr, costs=costs)


def validate_band(plan, reference_policy, *, starter=False):
    """Validate the worst executable tick in the complete permitted band."""
    if reference_policy.get("version") == "V7_RISK_BOUNDED_ENTRY_R19":
        from domain.les_v7.plan_r19 import validate_band as bounded_validate
        return bounded_validate(plan, reference_policy, starter=starter)
    if reference_policy.get("version") == "V7_SUPPORT_RESISTANCE_LIMIT_R18":
        from domain.les_v7.plan_r18 import validate_band as limit_validate
        return limit_validate(plan, reference_policy, starter=starter)
    if reference_policy.get("version") == "V7_SETUP_STRUCTURE_H20_R17":
        from domain.les_v7.plan_r17 import validate_band as unified_validate
        return unified_validate(plan, reference_policy, starter=starter)
    if reference_policy.get("version") == "V7_SETUP_STRUCTURE_H20_R16":
        from domain.les_v7.plan_r16 import validate_band as r16_validate
        return r16_validate(plan, reference_policy, starter=starter)
    _require_policy(reference_policy)
    if plan is None:
        return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    low = _round_price(Decimal(str(plan.band[0])), "up")
    high = _round_price(Decimal(str(plan.band[1])), "down")
    if low > high or low <= Decimal(str(plan.stop)):
        return dict(state="FAIL", reasons=["NO_VALID_EXECUTABLE_ENTRY_BAND"])
    result = evaluate_entry(plan, float(high), reference_policy, starter=starter)
    result.update(band=list(plan.band), executable_band=[float(low), float(high)],
                  evaluation_basis="HIGHEST_PERMITTED_LEGAL_ENTRY_TICK")
    return result


def _pass_prefix(plan, reference_policy, *, starter=False):
    """Keep only the initial all-PASS legal ticks, never search past a failure.

    The close must itself pass. A failed close retains its failed diagnostic
    plan for audit; it never acquires a lower replacement entry or authorization.
    """
    check = evaluate_entry(plan, plan.entry, reference_policy, starter=starter)
    diagnostic = dict(policy=R15_BAND_POLICY if reference_policy["version"] == R15_VERSION else R14_BAND_POLICY,
        original_band=list(plan.band),
        state=check["state"], close_evaluation=check, executable_prefix=None,
        checked_legal_ticks=0, first_nonpass=None,
        selection_basis="INITIAL_CONTIGUOUS_LEGAL_TICKS_WITH_FIXED_STOP_AND_RESISTANCE")
    if check["state"] != "PASS":
        diagnostic["reason"] = "CUTOFF_CLOSE_MUST_PASS_NO_LOWER_PRICE_SUBSTITUTION"
        return plan, diagnostic
    price = Decimal(str(plan.entry))
    last = _round_price(Decimal(str(plan.band[1])), "down")
    passed = price
    while price <= last:
        observed = evaluate_entry(plan, float(price), reference_policy, starter=starter)
        diagnostic["checked_legal_ticks"] += 1
        if observed["state"] != "PASS":
            diagnostic["first_nonpass"] = dict(entry=float(price), check=observed)
            break
        passed = price
        price += _tick(price)
    updated = replace(plan, band=(plan.entry, float(passed)))
    diagnostic.update(state="PASS", executable_prefix=list(updated.band),
        clipped=updated.band[1] != plan.band[1])
    return updated, diagnostic


def _r14_context(bars, reference_policy, cutoff):
    """Re-derive the unique variant independently of saved pass/subtype flags."""
    from domain.les_v7.rule_baseline import prepare_baseline, select_baseline_context, RELAXED_BASELINE_POLICY
    from domain.les_v7.context_r14 import select_context
    prepared = prepare_baseline(bars, market=reference_policy["cost_path_policy"]["market"],
        cutoff=cutoff, policy_version=RELAXED_BASELINE_POLICY)
    original = select_baseline_context(prepared["features"], policy_version=RELAXED_BASELINE_POLICY)
    return select_context(original, prepared["features"], bars=bars, cutoff=cutoff)


def _r15_context(bars, reference_policy, cutoff):
    from domain.les_v7.rule_baseline import prepare_baseline, select_baseline_context, RELAXED_BASELINE_POLICY
    from domain.les_v7.context_r15 import select_context
    prepared = prepare_baseline(bars, market=reference_policy["cost_path_policy"]["market"],
        cutoff=cutoff, policy_version=RELAXED_BASELINE_POLICY)
    original = select_baseline_context(prepared["features"], policy_version=RELAXED_BASELINE_POLICY)
    return select_context(original, prepared["features"], bars=bars, cutoff=cutoff)


def bind_entry_route(plan, reference_policy, *, strategy_decision, risk_route, setup_variant=None):
    """Bind a validated research route, never grant live order authority.

    The whole small policy is retained because a path-only minimal policy
    cannot reproduce the content identity of a full native reference policy.
    """
    if reference_policy.get("version") == "V7_RISK_BOUNDED_ENTRY_R19":
        from domain.les_v7.plan_r19 import bind_entry_route as bounded_bind
        return bounded_bind(plan, reference_policy, strategy_decision=strategy_decision,
            risk_route=risk_route, setup_variant=setup_variant)
    if reference_policy.get("version") == "V7_SUPPORT_RESISTANCE_LIMIT_R18":
        from domain.les_v7.plan_r18 import bind_entry_route as limit_bind
        return limit_bind(plan, reference_policy, strategy_decision=strategy_decision,
            risk_route=risk_route, setup_variant=setup_variant)
    if reference_policy.get("version") == "V7_SETUP_STRUCTURE_H20_R17":
        from domain.les_v7.plan_r17 import bind_entry_route as unified_bind
        return unified_bind(plan, reference_policy, strategy_decision=strategy_decision,
            risk_route=risk_route, setup_variant=setup_variant)
    if reference_policy.get("version") == "V7_SETUP_STRUCTURE_H20_R16":
        from domain.les_v7.plan_r16 import bind_entry_route as r16_bind
        return r16_bind(plan, reference_policy, strategy_decision=strategy_decision,
            risk_route=risk_route, setup_variant=setup_variant)
    _require_policy(reference_policy)
    from domain.les_v7.context_r15 import VARIANT
    if reference_policy["version"] != R15_VERSION or plan is None or plan.policy_version != R15_VERSION:
        raise ValueError("R15_ENTRY_ROUTE_REQUIRES_R15_PLAN_AND_POLICY")
    if reference_policy.get("reference_shares") != plan.reference_shares:
        raise ValueError("R15_ENTRY_ROUTE_REFERENCE_SIZE_MISMATCH")
    if strategy_decision not in {"BUY", "BUY_SMALL"} or risk_route not in {"CORE", "TACTICAL"}:
        raise ValueError("R15_ENTRY_ROUTE_DECISION_OR_ROUTE_INVALID")
    if setup_variant not in {None, VARIANT}:
        raise ValueError("R15_ENTRY_ROUTE_VARIANT_INVALID")
    if strategy_decision == "BUY" and (risk_route != "CORE" or setup_variant is not None):
        raise ValueError("R15_BUY_REQUIRES_CORE_ROUTE_WITHOUT_TACTICAL_VARIANT")
    if setup_variant is not None and (strategy_decision != "BUY_SMALL" or risk_route != "TACTICAL"):
        raise ValueError("R15_EARLY_VARIANT_REQUIRES_TACTICAL_SMALL")
    check = validate_band(plan, reference_policy, starter=risk_route == "TACTICAL")
    close = evaluate_entry(plan, plan.entry, reference_policy, starter=risk_route == "TACTICAL")
    if close["state"] != "PASS" or check["state"] != "PASS":
        raise ValueError("R15_ENTRY_ROUTE_NATIVE_RISK_NOT_PASS")
    result = dict(schema=R15_ROUTE_SCHEMA, plan_id=plan.plan_id,
        reference_policy_id=identity(reference_policy), reference_policy=deepcopy(reference_policy),
        risk_route=risk_route, strategy_decision=strategy_decision, setup_variant=setup_variant,
        authority="REFERENCE_RESEARCH_ROUTE_NOT_LIVE_ORDER_AUTHORITY")
    return dict(result, route_id=identity(result))


def _conditional(plan, support, reference_policy, *, starter=False):
    lower = _round_price(Decimal(str(support)), "up")
    upper = _round_price(Decimal(str(min(support + .5 * plan.atr, plan.entry))), "down")
    if lower <= Decimal(str(plan.stop)) or lower > upper:
        return None
    if evaluate_entry(plan, float(lower), reference_policy, starter=starter)["state"] != "PASS":
        return None
    if evaluate_entry(plan, float(upper), reference_policy, starter=starter)["state"] != "PASS":
        # Monotone legal-grid RR search holds anchors fixed. The zone existed
        # before this test; failure cannot relocate its support/resistance.
        good, bad = lower, upper
        while True:
            middle = _round_price((good + bad) / 2, "down")
            if middle <= good: break
            if evaluate_entry(plan, float(middle), reference_policy, starter=starter)["state"] == "PASS":
                good = middle
            else:
                bad = middle
        upper = good
    return dict(state="UNFILLED", band=[float(lower), float(upper)],
                zone_basis="FIXED_SUPPORT_TO_SUPPORT_PLUS_HALF_ATR",
                qualifies_current_close=False, live_order_authorized=False,
                requires_new_price_trigger_and_account_execution_checks=True,
                worst_entry=evaluate_entry(plan, float(upper), reference_policy, starter=starter))


def make_plan(context, features, reference_policy, evidence_id, *, bars=None, cutoff, setup_variant=None,
              session_evidence=None):
    """Return (old-contract plan, compact source-bound R10 anchor sidecar)."""
    if reference_policy.get("version") == "V7_RISK_BOUNDED_ENTRY_R19":
        from domain.les_v7.plan_r19 import make_plan as bounded_make
        return bounded_make(context, features, reference_policy, evidence_id,
            bars=bars, cutoff=cutoff, setup_variant=setup_variant, session_evidence=session_evidence)
    if reference_policy.get("version") == "V7_SUPPORT_RESISTANCE_LIMIT_R18":
        from domain.les_v7.plan_r18 import make_plan as limit_make
        return limit_make(context, features, reference_policy, evidence_id,
            bars=bars, cutoff=cutoff, setup_variant=setup_variant, session_evidence=session_evidence)
    if reference_policy.get("version") == "V7_SETUP_STRUCTURE_H20_R17":
        from domain.les_v7.plan_r17 import make_plan as unified_make
        return unified_make(context, features, reference_policy, evidence_id,
            bars=bars, cutoff=cutoff, setup_variant=setup_variant, session_evidence=session_evidence)
    if reference_policy.get("version") == "V7_SETUP_STRUCTURE_H20_R16":
        from domain.les_v7.plan_r16 import make_plan as r16_make
        return r16_make(context, features, reference_policy, evidence_id,
            bars=bars, cutoff=cutoff, setup_variant=setup_variant, session_evidence=session_evidence)
    _require_policy(reference_policy, cutoff)
    sidecar = dict(schema=SCHEMA, policy_id=identity(reference_policy), evidence_id=evidence_id,
                   cutoff=timestamp(cutoff).isoformat(), context=context, state="UNKNOWN",
                   reasons=[], anchors=None, bars_id=None, conditional_entry=None,
                   source_verification="INHERITED_NATIVE_EVIDENCE_BINDING_NOT_CERTIFIED_HERE")
    if context not in CONTEXTS:
        sidecar["reasons"] = ["NO_SELECTED_SETUP"]
        return None, sidecar
    if bars is None or bars.empty:
        sidecar["reasons"] = ["SOURCE_BOUND_COMPLETED_BARS_MISSING"]
        return None, sidecar
    r14_audit = None
    if reference_policy["version"] in {R14_VERSION, R15_VERSION}:
        selector = _r15_context if reference_policy["version"] == R15_VERSION else _r14_context
        selection, r14_audit = selector(bars, reference_policy, cutoff)
        derived_variant = r14_audit.get("setup_variant")
        if selection.context != context or setup_variant is not None and setup_variant != derived_variant:
            raise ValueError(("R15" if reference_policy["version"] == R15_VERSION else "R14")
                             + "_SOURCE_BOUND_CONTEXT_VARIANT_MISMATCH")
        setup_variant = derived_variant
        sidecar.update(setup_variant=setup_variant, short_base_audit=r14_audit)
    elif setup_variant is not None:
        raise ValueError("R14_SETUP_VARIANT_REQUIRES_R14_POLICY")
    selected = select_bars(bars, cutoff).tail(124)
    rows = selected[["Date", "Open", "High", "Low", "Close", "Volume"]].to_dict("records")
    for row in rows:
        row["Date"] = str(row["Date"])[:10]
        for key in ("Open", "High", "Low", "Close", "Volume"): row[key] = float(row[key])
    sidecar["bars_id"] = identity(rows)
    if len(rows) < 5:
        sidecar["reasons"] = ["CONFIRMED_STRUCTURE_HISTORY_MISSING"]
        return None, sidecar
    entry, atr = features.get("close"), features.get("atr")
    if entry is None or atr is None or not 0 < finite(atr):
        sidecar["reasons"] = ["REFERENCE_CLOSE_OR_ATR_MISSING"]
        return None, sidecar
    if finite(entry) != rows[-1]["Close"]:
        raise ValueError("R10_REFERENCE_CLOSE_DOES_NOT_MATCH_CUTOFF_BAR")
    lows, highs = _pivots(rows, "Low", low=True), _pivots(rows, "High", low=False)
    if setup_variant is not None:
        if reference_policy["version"] == R15_VERSION:
            from domain.les_v7.context_r15 import SUPPORT_METHOD, VARIANT
        else:
            from domain.les_v7.context_r14 import SUPPORT_METHOD, VARIANT
        if setup_variant != VARIANT or context != "LEADERSHIP_CONTINUATION" or r14_audit["state"] != "PASS":
            raise ValueError("R14_SHORT_BASE_SOURCE_VERIFICATION_REQUIRED")
        if finite(atr) != r14_audit["metrics"]["signal_atr"]:
            raise ValueError("R14_SHORT_BASE_ATR_SOURCE_MISMATCH")
        support = deepcopy(r14_audit["support"])
        support["index"] = next(i for i, row in enumerate(rows) if row["Date"] == support["session"])
        missing, method = None, SUPPORT_METHOD
    else:
        support, missing, method = _support(rows, lows, highs, context, atr)
    if support is None:
        sidecar["reasons"] = [missing]
        return None, sidecar
    overhead = _resistance(rows, highs, entry, max(0, len(rows) - 120))
    target = overhead["value"] if overhead else None
    stop = support["value"] - .3 * atr
    sidecar["anchors"] = dict(support=deepcopy(support), support_method=method,
                               resistance=deepcopy(overhead), stop_buffer_atr=.3,
                               selection_order="SETUP_ANCHORS_BEFORE_COSTS_OR_RR")
    if not 0 < stop < entry or entry < support["value"]:
        sidecar["reasons"] = ["SELECTED_STRUCTURE_BROKEN_OR_STOP_INVALID"]
        return None, sidecar
    if any(row["Low"] <= stop for row in rows[support["index"] + 1:]):
        sidecar["reasons"] = ["SELECTED_STRUCTURE_STOP_ALREADY_VIOLATED"]
        return None, sidecar
    shares = reference_policy.get("reference_shares")
    if type(shares) is not int or shares <= 0: raise ValueError("REFERENCE_SHARES_REQUIRED")
    band_low, band_high = reference_policy["entry_band_atr"]
    if finite(band_low) != 0 or finite(band_high) != .25:
        raise ValueError("R10_CLOSE_BAND_REQUIRED")
    costs = estimate_reference_costs(entry, stop, target, shares, reference_policy["cost_path_policy"])
    plan = ReferencePlan("REFERENCE_H20", reference_policy["version"], evidence_id, context, entry, stop, target, atr,
                         (entry, entry + .25 * atr), shares,
                         costs["cost_up_per_share"], costs["cost_stop_per_share"])
    if reference_policy["version"] == R14_VERSION:
        plan, prefix = _pass_prefix(plan, reference_policy)
        sidecar["entry_band_selection"] = prefix
    if reference_policy["version"] == R15_VERSION:
        core = evaluate_entry(plan, entry, reference_policy)
        tactical = evaluate_entry(plan, entry, reference_policy, starter=True)
        route = "CORE" if core["state"] == "PASS" and setup_variant is None else "TACTICAL"
        plan, prefix = _pass_prefix(plan, reference_policy, starter=route == "TACTICAL")
        sidecar.update(entry_band_selection=prefix,
            r15_risk=dict(core=core, tactical=tactical, band_route=route,
                core_band=validate_band(plan, reference_policy),
                tactical_band=validate_band(plan, reference_policy, starter=True)))
    sidecar.update(state="VERIFIED_ANCHORS" if target is not None else "PARTIAL",
                   reasons=[] if target is not None else ["EVIDENCE_DERIVED_RESISTANCE_MISSING"],
                   plan_id=plan.plan_id, formal_entry_basis="CUTOFF_CLOSE_NOT_CONDITIONAL_PRICE",
                   close_evaluation=evaluate_entry(plan, entry, reference_policy),
                   band_evaluation=validate_band(plan, reference_policy),
                   starter_band_evaluation=validate_band(plan, reference_policy, starter=True))
    if target is not None:
        conditional = _conditional(plan, support["value"], reference_policy)
        if conditional is not None:
            conditional["starter"] = _conditional(plan, support["value"], reference_policy, starter=True)
        sidecar["conditional_entry"] = conditional
    return plan, sidecar
