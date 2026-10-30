"""Opt-in FR3-L2: a cutoff-frozen leader band and three causal open chances.

The R1 source plan must pass before this route exists. Only its highest entry
tick and waiting policy change; original support, stop, resistance and H20 are
immutable. This is a new label/policy identity, never old calibrated odds.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
import json

from domain.les_v7 import fr3_plan as r1
from domain.les_v7.contracts import ReferencePlan, canonical, finite, identity, timestamp
from domain.les_v7.reference_path import _legal, _round_price, estimate_reference_costs

PLAN_VERSION = "FR3_SOURCE_BOUND_LIMIT_H20_L2"
PATH_VERSION = "FR3_H20_OPEN_LIMIT3_FIXED_TARGET_COST_AFTER_L2"
LABEL_VERSION = "FR3_L2_FIXED_TARGET_NEAR_BOTTOM_GIVEN_FILL_R1"
LEADER = "LEADERSHIP_CONTINUATION"
ENTRY_RULE = "CONTEXT_BOUND_OFFICIAL_SESSION_OPENS_IN_FROZEN_BAND"
LEADER_ENTRY_RULE = "FIRST_ELIGIBLE_OF_NEXT_THREE_OFFICIAL_OPENS_IN_BAND"
EXPIRY = "THIRD_OFFICIAL_SESSION_CLOSE"
WAIT_RULE = "PRIOR_COMPLETED_SESSION_LOW_LT_SUPPORT_OR_HIGH_GTE_TARGET"


def structural_policy():
    rules = r1.structural_policy()
    rules.update(version=PLAN_VERSION, label_version=LABEL_VERSION,
        entry_band_cap="LEADER_CUTOFF_LEGAL_PREFIX_CLOSE_PLUS_025_ATR_STOP_PLUS_25_ATR_NET_RR_12",
        entry_rule=ENTRY_RULE, expiry="CONTEXT_BOUND_ONE_OR_THREE_OFFICIAL_SESSION_CLOSE",
        leader_rule=dict(context=LEADER, maximum_open_opportunities=3,
            entry_rule=LEADER_ENTRY_RULE, expiry=EXPIRY, wait_cancellation=WAIT_RULE,
            cancellation_observation="AFTER_UNFILLED_OPEN_AT_COMPLETED_SESSION_CLOSE",
            original_plan_must_pass=True, alternate_anchor_allowed=False,
            lower_band="UNCHANGED_R1", stop="UNCHANGED_R1", target="UNCHANGED_R1",
            upper_cap_close_atr=.25, maximum_stop_atr=2.5, minimum_net_rr=1.2),
        other_context_rule="UNCHANGED_R1_BAND_AND_NEXT_OPEN_ONLY",
        horizon_rule="SIGNAL_PLUS_20_OFFICIAL_SESSIONS_NOT_FILL_PLUS_20")
    return rules


def path_policy(base, *, available_at):
    from domain.les_v7.reference_path import validate_reference_path_policy
    result = r1.path_policy(base, available_at=available_at)
    result.update(version=PATH_VERSION, entry_rule=ENTRY_RULE,
        label_version=LABEL_VERSION, waiting_policy=WAIT_RULE,
        maximum_leader_open_opportunities=3, horizon_anchor="SIGNAL_SESSION")
    return validate_reference_path_policy(result)


def policy(base, *, available_at):
    timestamp(available_at)
    # Deliberately require the caller's real creation clock; no backdated
    # module default or wall-clock-dependent hidden configuration.
    result = r1.policy(_r1_base(base), available_at=available_at)
    result.update(version=PLAN_VERSION, entry_rule=ENTRY_RULE,
        conditional_entry="CONTEXT_BOUND_UNFILLED_OFFICIAL_OPEN_LIMIT",
        structural_rules=structural_policy(),
        cost_path_policy=path_policy(_r1_base(base)["cost_path_policy"], available_at=available_at))
    return validate_policy(result)


def _r1_base(reference):
    """Detach a policy and remove only this version's explicit path additions."""
    result = deepcopy(dict(reference))
    path = result["cost_path_policy"]
    for key in ("waiting_policy", "maximum_leader_open_opportunities", "horizon_anchor"):
        path.pop(key, None)
    if path.get("version") == PATH_VERSION:
        path.update(version=r1.PATH_VERSION, entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND",
            label_version=r1.LABEL_VERSION)
    return result


def r1_policy(reference):
    return r1.policy(_r1_base(reference), available_at=reference["available_at"])


def validate_policy(reference, cutoff=None):
    from domain.les_v7.reference_path import validate_reference_path_policy
    if (not isinstance(reference, Mapping) or reference.get("policy_id") != "REFERENCE_H20"
            or reference.get("version") != PLAN_VERSION
            or reference.get("temporal_policy") != r1.TEMPORAL_POLICY
            or reference.get("historical_deployment_claim") is not False
            or reference.get("support_rule") != "FR3_R18_FIXED_SUPPORT_RESISTANCE"
            or reference.get("entry_rule") != ENTRY_RULE
            or reference.get("conditional_entry") != "CONTEXT_BOUND_UNFILLED_OFFICIAL_OPEN_LIMIT"
            or reference.get("structural_rules") != structural_policy()):
        raise ValueError("FR3_L2_REFERENCE_POLICY_MISMATCH")
    if timestamp(reference["approved_at"]) != timestamp(reference["available_at"]):
        raise ValueError("FR3_POLICY_APPROVAL_CLOCK_MISMATCH")
    r1.validate_policy(r1_policy(reference), cutoff)
    path = validate_reference_path_policy(dict(reference["cost_path_policy"]))
    if (path["version"] != PATH_VERSION or timestamp(path["available_at"]) != timestamp(reference["available_at"])):
        raise ValueError("FR3_L2_PATH_POLICY_BINDING_MISMATCH")
    return reference


@dataclass(frozen=True)
class FR3L2ReferencePlan(ReferencePlan):
    entry_basis: str = r1.ENTRY_BASIS
    signal_close: float = 0.
    setup_variant: str | None = None
    target_method: str = r1.TARGET_METHOD
    structure_id: str = ""
    parent_plan_id: str = ""
    original_band: tuple[float, float] = (0., 0.)
    original_cost_up: float = 0.
    original_cost_stop: float = 0.
    original_structure_id: str = ""
    source_structure_json: str = ""

    def __post_init__(self):
        super().__post_init__()
        if (self.policy_version != PLAN_VERSION or self.entry_basis != r1.ENTRY_BASIS
                or type(self.band) is not tuple or type(self.original_band) is not tuple
                or self.entry != self.band[1] or self.setup_variant is not None
                or self.target_method != r1.TARGET_METHOD or self.target is None
                or not all(_legal(Decimal(str(p))) for p in (*self.band, self.entry, self.target))
                or self.band[0] != self.original_band[0] or self.band[1] < self.original_band[1]):
            raise ValueError("FR3_L2_FIXED_SOURCE_PLAN_INVALID")
        old = self.original_plan()
        try:
            source = json.loads(self.source_structure_json)
            support = finite(source["support"]["value"])
            if (canonical(source) != self.source_structure_json
                    or identity(source) != self.original_structure_id
                    or source["band"] != list(self.original_band)
                    or source["resistance"]["value"] != self.target
                    or support - .3*self.atr != self.stop
                    or old.plan_id != self.parent_plan_id
                    or self.structure_id != identity(self.structure_binding())):
                raise ValueError("FR3_L2_ORIGINAL_SOURCE_BINDING_INVALID")
        except (KeyError, TypeError, json.JSONDecodeError):
            raise ValueError("FR3_L2_ORIGINAL_SOURCE_BINDING_INVALID") from None
        if self.context != LEADER and self.band != self.original_band:
            raise ValueError("FR3_L2_NONLEADER_BAND_CHANGED")
        if self.context == LEADER and Decimal(str(self.band[1])) > max(
                Decimal(str(self.original_band[1])), _round_price(min(
                    Decimal(str(self.signal_close)) + Decimal(".25")*Decimal(str(self.atr)),
                    Decimal(str(self.stop)) + Decimal("2.5")*Decimal(str(self.atr))), "down")):
            raise ValueError("FR3_L2_CUTOFF_CAP_EXCEEDED")

    @property
    def support(self):
        return json.loads(self.source_structure_json)["support"]["value"]

    def original_plan(self):
        return r1.FR3ReferencePlan("REFERENCE_H20", r1.PLAN_VERSION, self.evidence_id, self.context,
            self.original_band[1], self.stop, self.target, self.atr, self.original_band,
            self.reference_shares, self.original_cost_up, self.original_cost_stop,
            signal_close=self.signal_close, structure_id=self.original_structure_id)

    def structure_binding(self):
        return dict(policy_version=PLAN_VERSION, parent_plan_id=self.parent_plan_id,
            original_structure_id=self.original_structure_id, source_structure_json=self.source_structure_json,
            band=list(self.band), stop=self.stop, target=self.target, context=self.context)


def plan_from_record(record):
    return FR3L2ReferencePlan(**dict(record, band=tuple(record["band"]), original_band=tuple(record["original_band"])))


def _risk(entry, old, reference):
    entry = Decimal(str(entry))
    if entry <= Decimal(str(old.stop)) or entry >= Decimal(str(old.target)):
        return dict(state="FAIL", reasons=["FR3_ENTRY_OUTSIDE_FIXED_PLAN"])
    costs = estimate_reference_costs(float(entry), old.stop, old.target, old.reference_shares, reference["cost_path_policy"])
    with localcontext() as ctx:
        ctx.prec = 50
        distance = (entry-Decimal(str(old.stop)))/Decimal(str(old.atr))
        rr = (Decimal(str(old.target))-entry-Decimal(str(costs["cost_up_per_share"]))) / (
            entry-Decimal(str(old.stop))+Decimal(str(costs["cost_stop_per_share"])))
    reasons = []
    if distance > Decimal("2.5"):
        reasons.append("FR3_FIXED_STOP_ATR_EXCEEDED")
    if rr < Decimal("1.2"):
        reasons.append("FR3_FIXED_TARGET_NET_RR_BELOW_MINIMUM")
    return dict(state="FAIL" if reasons else "PASS", reasons=reasons, entry=float(entry),
        stop_atr=float(distance), net_rr=float(rr), maximum_stop_atr=2.5, minimum_net_rr=1.2, costs=costs)


def maximum_upper(old, reference):
    """Cutoff-only maximal passing legal prefix, with fixed original barriers.

    On the legal tick grid, fixed shares/fees imply nondecreasing buy costs,
    decreasing net reward and increasing net risk. Hence the passing RR set
    above the old upper is a prefix; future opens never choose this boundary.
    """
    if r1.validate_band(old, r1_policy(reference))["state"] != "PASS":
        raise ValueError("FR3_L2_ORIGINAL_PLAN_NOT_PASS")
    if old.context != LEADER:
        return old.band[1]
    with localcontext() as ctx:
        ctx.prec = 50
        lower = Decimal(str(old.band[1]))
        upper = max(lower, _round_price(min(Decimal(str(old.signal_close))+Decimal(".25")*Decimal(str(old.atr)),
            Decimal(str(old.stop))+Decimal("2.5")*Decimal(str(old.atr))), "down"))
        if _risk(upper, old, reference)["state"] == "PASS":
            return float(upper)
        while True:
            middle = _round_price((lower+upper)/2, "down")
            if middle <= lower:
                return float(lower)
            if middle >= upper:
                raise ValueError("FR3_L2_GRID_SEARCH_PRECISION_EXHAUSTED")
            if _risk(middle, old, reference)["state"] == "PASS":
                lower = middle
            else:
                upper = middle


def make_plan(context, features, reference, evidence_id, *, bars=None, cutoff, setup_variant=None, session_evidence=None):
    validate_policy(reference, cutoff)
    old, audit = r1.make_plan(context, features, r1_policy(reference), evidence_id, bars=bars,
        cutoff=cutoff, setup_variant=setup_variant, session_evidence=session_evidence)
    audit = deepcopy(audit)
    audit.update(schema="les-v7-fr3-l2-reference-plan-evidence/v1", policy_id=identity(reference),
        policy_version=PLAN_VERSION, label_version=LABEL_VERSION)
    if old is None:
        return None, audit
    original_check = r1.validate_band(old, r1_policy(reference))
    audit.update(original_plan_id=old.plan_id, original_band_evaluation=original_check)
    if original_check["state"] != "PASS":
        audit.update(state="FAIL", reasons=["FR3_L2_ORIGINAL_PLAN_NOT_PASS"])
        return None, audit
    high = maximum_upper(old, reference)
    band = (old.band[0], high)
    costs = estimate_reference_costs(high, old.stop, old.target, old.reference_shares, reference["cost_path_policy"])
    source = dict(support=audit["support"], resistance=audit["resistance"], band=list(old.band),
        selection="LATEST_CONFIRMED_SUPPORT_NEAREST_OBSERVED_RESISTANCE_BEFORE_RR")
    values = dict(asdict(old), policy_version=PLAN_VERSION, entry=high, band=band,
        cost_up=costs["cost_up_per_share"], cost_stop=costs["cost_stop_per_share"],
        expiry=EXPIRY if context == LEADER else "NEXT_SESSION_CLOSE", parent_plan_id=old.plan_id,
        original_band=old.band, original_cost_up=old.cost_up, original_cost_stop=old.cost_stop,
        original_structure_id=old.structure_id, source_structure_json=canonical(source))
    binding = {key: values[key] for key in ("policy_version", "parent_plan_id", "original_structure_id", "source_structure_json", "stop", "target", "context")}
    binding["band"] = list(band)
    values["structure_id"] = identity(binding)
    plan = FR3L2ReferencePlan(**values)
    audit.update(plan_id=plan.plan_id, structure_id=plan.structure_id, band=list(plan.band),
        conditional_entry=dict(state="UNFILLED", assumes_fill=False, live_order_authorized=False,
            order_type="LIMIT", entry_rule=LEADER_ENTRY_RULE if context == LEADER else "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND",
            expiry=plan.expiry), band_evaluation=r1.validate_band(plan, reference))
    return plan, audit


def evaluate_entry(plan, observed_entry, reference, *, starter=True):
    validate_policy(reference)
    if plan is None:
        return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    if type(plan) is not FR3L2ReferencePlan:
        raise ValueError("FR3_L2_SOURCE_BOUND_PLAN_REQUIRED")
    if not starter:
        return dict(state="FAIL", reasons=["FR3_CONDITIONAL_LIMIT_SMALL_ONLY"])
    entry = Decimal(str(finite(observed_entry, "FR3_observed_entry", 0)))
    if not _legal(entry) or not Decimal(str(plan.band[0])) <= entry <= Decimal(str(plan.band[1])):
        return dict(state="FAIL", reasons=["FR3_ENTRY_OUTSIDE_FIXED_PLAN"])
    return _risk(entry, plan, reference)


def bind_entry_route(plan, reference, *, strategy_decision="BUY_SMALL", risk_route="TACTICAL", setup_variant=None):
    validate_policy(reference)
    if (type(plan) is not FR3L2ReferencePlan or strategy_decision != "BUY_SMALL" or risk_route != "TACTICAL"
            or setup_variant is not None or plan.reference_shares != reference["reference_shares"]):
        raise ValueError("FR3_L2_LIMIT_REQUIRES_TACTICAL_SMALL_SOURCE_PLAN")
    plan_from_record(asdict(plan))
    old = plan.original_plan()
    original_costs = estimate_reference_costs(old.entry, old.stop, old.target, old.reference_shares, reference["cost_path_policy"])
    current_costs = estimate_reference_costs(plan.entry, plan.stop, plan.target, plan.reference_shares, reference["cost_path_policy"])
    if (plan.band[1] != maximum_upper(old, reference)
            or (old.cost_up, old.cost_stop) != (original_costs["cost_up_per_share"], original_costs["cost_stop_per_share"])
            or (plan.cost_up, plan.cost_stop) != (current_costs["cost_up_per_share"], current_costs["cost_stop_per_share"])
            or evaluate_entry(plan, plan.entry, reference)["state"] != "PASS"):
        raise ValueError("FR3_L2_FROZEN_MAXIMAL_BAND_OR_RISK_INVALID")
    result = dict(schema="les-v7-fr3-l2-reference-entry-route/v1", plan_id=plan.plan_id,
        reference_policy_id=identity(reference), reference_policy=deepcopy(reference),
        strategy_decision=strategy_decision, risk_route=risk_route, setup_variant=None,
        entry_basis=plan.entry_basis, signal_close=plan.signal_close, target_method=plan.target_method,
        structure_id=plan.structure_id, original_plan_id=plan.parent_plan_id,
        entry_rule=LEADER_ENTRY_RULE if plan.context == LEADER else "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND",
        order_type="LIMIT", expiry=plan.expiry,
        maximum_open_opportunities=3 if plan.context == LEADER else 1,
        cancellation_support=plan.support, cancellation_target=plan.target,
        waiting_policy=WAIT_RULE, horizon_anchor="SIGNAL_SESSION", horizon=20,
        authority="REFERENCE_RESEARCH_ROUTE_NOT_LIVE_ORDER_AUTHORITY")
    return dict(result, route_id=identity(result))
