"""Causal R19 entry bands, widened only while native costs and risk still pass.

R18 still chooses the source-bound structure, stop and target.  R19 expands
only the upper limit, stops at the first non-passing legal tick, and never
repairs a failed original plan or uses subsequent prices to choose the band.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from decimal import Decimal

from domain.les_v7 import plan_r18 as previous
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.policy_r19 import AVAILABLE_AT, PLAN_VERSION
from domain.les_v7.reference_path import _legal, _round_price, _tick, estimate_reference_costs

A_VARIANT, C_VARIANT = previous.A_VARIANT, previous.C_VARIANT
NEUTRAL_VARIANT, NEW_VARIANTS = previous.NEUTRAL_VARIANT, previous.NEW_VARIANTS
A_ENTRY_BASIS, CLOSE_ENTRY_BASIS = previous.A_ENTRY_BASIS, previous.CLOSE_ENTRY_BASIS
OBSERVED_TARGET, PROJECTED_TARGET = previous.OBSERVED_TARGET, previous.PROJECTED_TARGET
prepare_selection = previous.prepare_selection
PLAN_SCHEMA = "les-v7-setup-plan-evidence/v5"
ROUTE_SCHEMA = "les-v7-reference-entry-route/v5"


def structural_policy():
    return dict(version=PLAN_VERSION, selection="UNCHANGED_R18_SOURCE_STRUCTURES",
        original_lower_limit="UNCHANGED_R18",
        planned_upper_limit="MAX_ORIGINAL_UPPER_AND_SIGNAL_CLOSE_PLUS_ONE_ATR",
        upper_extension_atr=1., upper_selection="CONTIGUOUS_LEGAL_PASS_PREFIX_FROM_ORIGINAL_UPPER",
        risk_limits="UNCHANGED_R18_ROUTE_SPECIFIC_STOP_ATR_AND_COST_NET_RR",
        route_selection="ORIGINAL_CORE_IF_BOTH_CLOSE_AND_BAND_PASS_ELSE_TACTICAL",
        failed_original_plan="NO_RESCUE_OR_REANCHORING", stop_and_target="UNCHANGED_R18",
        features="COMPLETED_SIGNAL_TIME_BARS_ONLY", future_prices_used=False,
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", expiry="NEXT_SESSION_CLOSE",
        sizing_basis="HIGHEST_PERMITTED_LEGAL_ENTRY_TICK", performance_validated=False)


def policy(reference_policy, *, available_at):
    if timestamp(available_at) < timestamp(AVAILABLE_AT):
        raise ValueError("R19_REFERENCE_POLICY_BEFORE_RELEASE")
    result = previous.policy(reference_policy, available_at=available_at)
    result.update(version=PLAN_VERSION, support_rule="R18_FIXED_STRUCTURES_RISK_BOUNDED_ENTRY_R19",
        conditional_entry="NEXT_OFFICIAL_SESSION_OPEN_IN_RISK_BOUNDED_BAND_R19")
    result["structural_rules"]["r19"] = structural_policy()
    return result


def _as_previous_policy(reference_policy):
    result = deepcopy(dict(reference_policy))
    result.update(version=previous.PLAN_VERSION, support_rule="SOURCE_BOUND_SUPPORT_RESISTANCE_LIMIT_R18",
        conditional_entry="NEXT_OFFICIAL_SESSION_OPEN_IN_FIXED_LIMIT_BAND_R18")
    result["structural_rules"].pop("r19", None)
    return result


def require_policy(reference_policy, cutoff=None):
    previous.previous._selection_policy(None, cutoff, reference_policy=reference_policy)
    if (reference_policy.get("version") != PLAN_VERSION
            or reference_policy.get("support_rule") != "R18_FIXED_STRUCTURES_RISK_BOUNDED_ENTRY_R19"
            or reference_policy.get("conditional_entry") != "NEXT_OFFICIAL_SESSION_OPEN_IN_RISK_BOUNDED_BAND_R19"
            or reference_policy.get("structural_rules", {}).get("r19") != structural_policy()):
        raise ValueError("R19_REFERENCE_POLICY_MISMATCH")
    if timestamp(reference_policy["available_at"]) < timestamp(AVAILABLE_AT):
        raise ValueError("R19_REFERENCE_POLICY_BEFORE_RELEASE")
    previous.require_policy(_as_previous_policy(reference_policy), cutoff)
    return reference_policy


def _as_previous_plan(plan):
    return previous.R18ReferencePlan(**dict(asdict(plan), policy_version=previous.PLAN_VERSION))


@dataclass(frozen=True)
class R19ReferencePlan(previous.R18ReferencePlan):
    def __post_init__(self):
        if self.policy_version != PLAN_VERSION:
            raise ValueError("R19_PLAN_VERSION_REQUIRED")
        _as_previous_plan(self)


def plan_from_record(record):
    if record is None:
        return None
    from domain.les_v7.fr3_plan import is_fr3_reference
    if is_fr3_reference({"version": record.get("policy_version")}):
        from domain.les_v7.fr3_plan import plan_from_record as fr3_from_record
        return fr3_from_record(record)
    if record.get("policy_version") != PLAN_VERSION:
        return previous.plan_from_record(record)
    return R19ReferencePlan(**dict(record, band=tuple(record["band"])))


def evaluate_entry(plan, observed_entry, reference_policy, *, starter=False):
    require_policy(reference_policy)
    if plan is None:
        return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    if not isinstance(plan, R19ReferencePlan):
        raise ValueError("R19_SOURCE_BOUND_PLAN_REQUIRED")
    return previous.evaluate_entry(_as_previous_plan(plan), observed_entry,
        _as_previous_policy(reference_policy), starter=starter)


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


def _extend(plan, reference_policy):
    """Extend an already accepted route; never search beyond its first failure."""
    core = evaluate_entry(plan, plan.entry, reference_policy)
    core_band = validate_band(plan, reference_policy)
    starter = bool(plan.setup_variant is not None or core["state"] != "PASS" or core_band["state"] != "PASS")
    origin = evaluate_entry(plan, plan.entry, reference_policy, starter=starter)
    upper = validate_band(plan, reference_policy, starter=starter)
    audit = dict(schema="les-v7-risk-bounded-entry-extension/v1", policy=structural_policy(),
        original_band=list(plan.band), original_entry=plan.entry, route="TACTICAL" if starter else "CORE",
        origin_evaluation=origin, original_band_evaluation=upper,
        planned_upper=None, checked_legal_ticks=0, first_nonpass=None,
        extended=False, state="FAIL" if "FAIL" in {origin["state"], upper["state"]} else "UNKNOWN")
    if origin["state"] != "PASS" or upper["state"] != "PASS":
        audit.update(reason="ORIGINAL_REFERENCE_AND_BAND_MUST_PASS_NO_RESCUE", executable_band=list(plan.band))
        return plan, audit
    ceiling = _round_price(max(Decimal(str(plan.band[1])),
        Decimal(str(plan.signal_close)) + Decimal(str(plan.atr))), "down")
    audit["planned_upper"] = float(ceiling)
    price = Decimal(str(plan.band[1]))
    passed = price
    while price < ceiling:
        price += _tick(price)
        if price > ceiling:
            break
        result = evaluate_entry(plan, float(price), reference_policy, starter=starter)
        audit["checked_legal_ticks"] += 1
        if result["state"] != "PASS":
            audit["first_nonpass"] = dict(entry=float(price), check=result)
            break
        passed = price
    band = (plan.band[0], float(passed))
    entry = band[1] if plan.entry_basis == A_ENTRY_BASIS else plan.entry
    costs = estimate_reference_costs(entry, plan.stop, plan.target, plan.reference_shares,
        reference_policy["cost_path_policy"])
    updated = replace(plan, band=band, entry=entry, cost_up=costs["cost_up_per_share"],
        cost_stop=costs["cost_stop_per_share"], structure_id=identity(dict(
            original_structure_id=plan.structure_id, extension_policy=identity(structural_policy()), band=band)))
    audit.update(state="PASS", executable_band=list(band), extended=band != plan.band,
        clipping_reason="FIRST_NONPASS_LEGAL_TICK" if audit["first_nonpass"] else "SIGNAL_CLOSE_PLUS_ONE_ATR_LIMIT")
    return updated, audit


def make_plan(context, features, reference_policy, evidence_id, *, bars=None, cutoff,
              setup_variant=None, session_evidence=None, selection_policy=None):
    require_policy(reference_policy, cutoff)
    selection_policy = previous.previous._selection_policy(selection_policy, cutoff, reference_policy=reference_policy)
    original, sidecar = previous.make_plan(context, features, _as_previous_policy(reference_policy),
        evidence_id, bars=bars, cutoff=cutoff, setup_variant=setup_variant, session_evidence=session_evidence,
        selection_policy=selection_policy)
    sidecar = deepcopy(sidecar)
    sidecar.update(schema=PLAN_SCHEMA, policy_id=identity(reference_policy))
    if original is None:
        return None, sidecar
    plan = R19ReferencePlan(**dict(asdict(original), policy_version=PLAN_VERSION))
    plan, extension = _extend(plan, reference_policy)
    sidecar.update(plan_id=plan.plan_id, structure_id=plan.structure_id,
        r19_entry_band_extension=extension, formal_entry_basis=plan.entry_basis,
        close_evaluation=evaluate_entry(plan, plan.entry, reference_policy),
        band_evaluation=validate_band(plan, reference_policy),
        starter_band_evaluation=validate_band(plan, reference_policy, starter=True),
        conditional_entry=dict(state="UNFILLED", order_type="LIMIT",
            entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", expiry="NEXT_SESSION_CLOSE",
            band=list(plan.band), entry_basis=plan.entry_basis, signal_close=plan.signal_close,
            live_order_authorized=False, assumes_fill=False))
    return plan, sidecar


def bind_entry_route(plan, reference_policy, *, strategy_decision, risk_route, setup_variant=None):
    require_policy(reference_policy)
    if not isinstance(plan, R19ReferencePlan) or reference_policy.get("reference_shares") != plan.reference_shares:
        raise ValueError("R19_SOURCE_BOUND_PLAN_AND_REFERENCE_SIZE_REQUIRED")
    plan_from_record(asdict(plan))
    if not all(_legal(Decimal(str(value))) for value in (*plan.band, plan.entry)):
        raise ValueError("R19_ENTRY_BAND_LEGAL_TICKS_REQUIRED")
    if setup_variant is not None and setup_variant != plan.setup_variant:
        raise ValueError("R19_ENTRY_ROUTE_VARIANT_MISMATCH")
    if strategy_decision not in {"BUY", "BUY_SMALL"} or risk_route not in {"CORE", "TACTICAL"}:
        raise ValueError("R19_ENTRY_ROUTE_DECISION_OR_ROUTE_INVALID")
    if plan.setup_variant is not None and (strategy_decision != "BUY_SMALL" or risk_route != "TACTICAL"):
        raise ValueError("R19_VARIANT_REQUIRES_TACTICAL_SMALL")
    if strategy_decision == "BUY" and risk_route != "CORE":
        raise ValueError("R19_BUY_REQUIRES_CORE")
    if (evaluate_entry(plan, plan.entry, reference_policy, starter=risk_route == "TACTICAL")["state"] != "PASS"
            or validate_band(plan, reference_policy, starter=risk_route == "TACTICAL")["state"] != "PASS"):
        raise ValueError("R19_ENTRY_ROUTE_NATIVE_RISK_NOT_PASS")
    result = dict(schema=ROUTE_SCHEMA, plan_id=plan.plan_id, reference_policy_id=identity(reference_policy),
        reference_policy=deepcopy(reference_policy), risk_route=risk_route, strategy_decision=strategy_decision,
        setup_variant=plan.setup_variant, entry_basis=plan.entry_basis, signal_close=plan.signal_close,
        target_method=plan.target_method, structure_id=plan.structure_id,
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", order_type="LIMIT", expiry="NEXT_SESSION_CLOSE",
        authority="REFERENCE_RESEARCH_ROUTE_NOT_LIVE_ORDER_AUTHORITY")
    return dict(result, route_id=identity(result))
