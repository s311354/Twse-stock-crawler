"""FR3 fixed-source support/resistance limits for retrospective research.

The R18 anchor rules are shared across the original three contexts.  Source
anchors and the limit band are fixed before costs or reward/risk are checked;
there is no stop rescue, target projection, alternate support or assumed fill.
The policy creation clock is real metadata, never historical market evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal

from domain.les_v7.contracts import CONTEXTS, ReferencePlan, finite, identity, timestamp
from domain.les_v7.features import select_bars
from domain.les_v7.reference_path import _legal, _round_price, estimate_reference_costs

PLAN_VERSION = "FR3_SOURCE_BOUND_LIMIT_H20_R1"
PATH_VERSION = "FR3_H20_OPEN_ONLY_FIXED_TARGET_COST_AFTER_R1"
LABEL_VERSION = "FR3_FIXED_TARGET_NEAR_BOTTOM_GIVEN_FILL_R1"
L2_PLAN_VERSION = "FR3_SOURCE_BOUND_LIMIT_H20_L2"
SUPPORTED_PLAN_VERSIONS = (PLAN_VERSION, L2_PLAN_VERSION)
SCHEMA = "les-v7-fr3-reference-plan-evidence/v1"
ROUTE_SCHEMA = "les-v7-fr3-reference-entry-route/v1"
ENTRY_BASIS = "STRUCTURAL_CONDITIONAL_BAND"
TARGET_METHOD = "CONFIRMED_UNBROKEN_OVERHEAD_RESISTANCE"
TEMPORAL_POLICY = "RETROSPECTIVE_FIXED_RULES_NOT_DEPLOYMENT"
FINAL_CALENDAR_POLICY = "FINAL_OFFICIAL_CALENDAR_RECONSTRUCTION_R1"


def structural_policy():
    return dict(version=PLAN_VERSION, source_selection="R18_FIXED_SUPPORT_RESISTANCE_ANCHORS",
        support_lookback=60, resistance_lookback=120, pivot_left=2, pivot_right=2,
        anchor_confirmation="TWO_RIGHT_COMPLETED_SESSIONS_BEFORE_SIGNAL_SESSION",
        support_must_remain_unbroken=True, target_projection_allowed=False,
        entry_band_atr=[.5, 1.], entry_band_cap="ACTUAL_SIGNAL_CLOSE_BEFORE_RR",
        stop_buffer_atr=.3, maximum_stop_atr=2.5, minimum_net_rr=1.2,
        context_policy="KEEP_ORIGINAL_THREE_CONTEXTS_NO_NEUTRAL_WEIGHT_FALLBACK",
        route="BUY_SMALL_ONLY", order_type="LIMIT",
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", expiry="NEXT_SESSION_CLOSE",
        target_rule="FIXED_SOURCE_RESISTANCE_BEFORE_STOP_ELSE_H20_CLOSE",
        label_version=LABEL_VERSION, historical_deployment_claim=False,
        performance_validated=False)


def is_fr3_reference(reference_policy):
    return isinstance(reference_policy, Mapping) and reference_policy.get("version") in SUPPORTED_PLAN_VERSIONS


def path_policy(base, *, available_at):
    """Keep declared fees/grid/coordinates; version the outcome barrier only."""
    from domain.les_v7.reference_path import validate_reference_path_policy
    result = validate_reference_path_policy(dict(base))
    timestamp(available_at)
    result.update(version=PATH_VERSION, available_at=available_at,
        exit_rule="FIRST_STOP_OR_FIXED_TARGET_ELSE_H20_CLOSE",
        entry_guard_policy="FR3_REPRICE_FIXED_STOP_AND_RESISTANCE",
        target_policy="FIXED_CAUSAL_PLAN_TARGET_NO_2R_SUBSTITUTION",
        label_version=LABEL_VERSION, historical_deployment_claim=False)
    return validate_reference_path_policy(result)


def policy(base, *, available_at, version=PLAN_VERSION):
    """An explicit new research policy, never a rewritten old bundle."""
    if version == L2_PLAN_VERSION:
        from domain.les_v7.fr3_l2 import policy as l2_policy
        return l2_policy(base, available_at=available_at)
    if version != PLAN_VERSION:
        raise ValueError("FR3_PLAN_VERSION_REQUIRED")
    if base.get("policy_id") != "REFERENCE_H20":
        raise ValueError("FR3_REFERENCE_H20_REQUIRED")
    timestamp(available_at)
    result = deepcopy(dict(base))
    result.update(version=PLAN_VERSION, available_at=available_at, approved_at=available_at,
        temporal_policy=TEMPORAL_POLICY, historical_deployment_claim=False,
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND",
        support_rule="FR3_R18_FIXED_SUPPORT_RESISTANCE",
        conditional_entry="UNFILLED_NEXT_OFFICIAL_OPEN_LIMIT",
        structural_rules=structural_policy(),
        cost_path_policy=path_policy(base["cost_path_policy"], available_at=available_at))
    return validate_policy(result)


def validate_policy(reference_policy, cutoff=None):
    if isinstance(reference_policy, Mapping) and reference_policy.get("version") == L2_PLAN_VERSION:
        from domain.les_v7.fr3_l2 import validate_policy as l2_validate
        return l2_validate(reference_policy, cutoff)
    from domain.les_v7.reference_path import validate_reference_path_policy
    if (not isinstance(reference_policy, Mapping)
            or reference_policy.get("policy_id") != "REFERENCE_H20"
            or reference_policy.get("version") != PLAN_VERSION
            or reference_policy.get("temporal_policy") != TEMPORAL_POLICY
            or reference_policy.get("historical_deployment_claim") is not False
            or reference_policy.get("entry_rule") != "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND"
            or reference_policy.get("support_rule") != "FR3_R18_FIXED_SUPPORT_RESISTANCE"
            or reference_policy.get("conditional_entry") != "UNFILLED_NEXT_OFFICIAL_OPEN_LIMIT"
            or reference_policy.get("structural_rules") != structural_policy()):
        raise ValueError("FR3_REFERENCE_POLICY_MISMATCH")
    created = timestamp(reference_policy["available_at"])
    if timestamp(reference_policy["approved_at"]) != created:
        raise ValueError("FR3_POLICY_APPROVAL_CLOCK_MISMATCH")
    # ``cutoff`` belongs to causal observations. A newly declared retrospective
    # rule may be computed for old dates, but is never claimed deployed then.
    if cutoff is not None:
        timestamp(cutoff)
    shares = reference_policy.get("reference_shares")
    if type(shares) is not int or shares <= 0:
        raise ValueError("FR3_POSITIVE_REFERENCE_SHARES_REQUIRED")
    path = validate_reference_path_policy(dict(reference_policy["cost_path_policy"]))
    if (path.get("version") != PATH_VERSION
            or path.get("entry_guard_policy") != "FR3_REPRICE_FIXED_STOP_AND_RESISTANCE"
            or timestamp(path["available_at"]) != created):
        raise ValueError("FR3_PATH_POLICY_BINDING_MISMATCH")
    return reference_policy


require_policy = validate_policy


@dataclass(frozen=True)
class FR3ReferencePlan(ReferencePlan):
    entry_basis: str = ENTRY_BASIS
    signal_close: float = 0.
    setup_variant: str | None = None
    target_method: str = TARGET_METHOD
    structure_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        if (self.policy_version != PLAN_VERSION or self.entry_basis != ENTRY_BASIS
                or self.entry != self.band[1] or self.band[1] > finite(self.signal_close)
                or self.setup_variant is not None or self.target is None
                or not _legal(Decimal(str(self.target)))
                or self.target_method != TARGET_METHOD or len(self.structure_id) != 64
                or any(c not in "0123456789abcdef" for c in self.structure_id)
                or not all(_legal(Decimal(str(p))) for p in (*self.band, self.entry))):
            raise ValueError("FR3_FIXED_SOURCE_PLAN_INVALID")


def plan_from_record(record):
    if record is None:
        return None
    if record.get("policy_version") == L2_PLAN_VERSION:
        from domain.les_v7.fr3_l2 import plan_from_record as l2_plan
        return l2_plan(record)
    if record.get("policy_version") != PLAN_VERSION:
        raise ValueError("FR3_PLAN_VERSION_REQUIRED")
    return FR3ReferencePlan(**dict(record, band=tuple(record["band"])))


def verify_window(calendar, *, market, cutoff, observed_sessions):
    """Native-normalized calendars or the existing original-proof contract.

    The source normalizer verifies raw official documents before publishing
    normalized calendars. This function checks their full sequence, source
    lineage and availability; it does not certify arbitrary caller JSON.
    """
    if not isinstance(calendar, Mapping) or "availability_by_date" not in calendar:
        from domain.les_v7.session_coverage_r17 import verify_window as original_window
        return original_window(calendar, market=market, cutoff=cutoff,
            observed_sessions=observed_sessions, prior_sessions=120)
    audit = dict(schema="les-v7-fr3-native-calendar-window/v1", state="UNKNOWN", reasons=[],
        source_id=identity(calendar), historical_information_set_certified=False,
        source_authenticity="INHERITED_NATIVE_SOURCE_NORMALIZATION_NOT_CERTIFIED_HERE")
    try:
        if (calendar.get("source") != market + "_OFFICIAL"
                or not calendar.get("evidence_ids") or len(observed_sessions) != 121
                or list(observed_sessions) != sorted(set(observed_sessions))):
            raise ValueError("FR3_NATIVE_CALENDAR_SCOPE_INVALID")
        first, last = (date.fromisoformat(v) for v in (observed_sessions[0], observed_sessions[-1]))
        if (last-first).days > 1000 or timestamp(last.isoformat()+"T13:30:00+08:00") > timestamp(cutoff):
            raise ValueError("FR3_NATIVE_CALENDAR_WINDOW_INVALID")
        needed = [(first+timedelta(days=i)).isoformat() for i in range((last-first).days+1)]
        days = calendar.get("days", {})
        if any(days.get(d) not in {"OPEN", "CLOSED"} for d in needed):
            raise ValueError("FR3_NATIVE_CALENDAR_UNKNOWN_DATES")
        expected = [d for d in needed if days[d] == "OPEN"]
        if expected != list(observed_sessions):
            audit.update(state="FAIL", reasons=["FR3_OFFICIAL_SESSION_SEQUENCE_MISMATCH"])
            return audit
        if any(not calendar["availability_by_date"].get(d)
                or timestamp(calendar["availability_by_date"][d]) > timestamp(cutoff) for d in needed):
            raise ValueError("FR3_NATIVE_CALENDAR_AVAILABILITY_UNKNOWN_OR_FUTURE")
        final = calendar.get("availability_policy") == FINAL_CALENDAR_POLICY
        if final:
            if (calendar.get("historical_information_set_certified") is not False
                    or calendar.get("original_historical_available_at") is not None
                    or any(not calendar.get("source_versions_by_date", {}).get(d) for d in needed)):
                raise ValueError("FR3_FINAL_CALENDAR_LINEAGE_REQUIRED")
        elif not calendar.get("publication_events"):
            raise ValueError("FR3_NATIVE_CALENDAR_PUBLICATION_LINEAGE_REQUIRED")
        audit.update(state="PASS", expected_sessions=expected,
            availability_basis="OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_NOT_PIT" if final
                else "NATIVE_VERIFIED_PUBLICATION_SELECTION", evidence_ids=list(calendar["evidence_ids"]))
    except (KeyError, TypeError, ValueError) as error:
        audit["reasons"] = [str(error)]
    return audit


def make_plan(context, features, reference_policy, evidence_id, *, bars=None, cutoff,
              setup_variant=None, session_evidence=None):
    if isinstance(reference_policy, Mapping) and reference_policy.get("version") == L2_PLAN_VERSION:
        from domain.les_v7.fr3_l2 import make_plan as l2_make
        return l2_make(context, features, reference_policy, evidence_id, bars=bars, cutoff=cutoff,
            setup_variant=setup_variant, session_evidence=session_evidence)
    validate_policy(reference_policy, cutoff)
    audit = dict(schema=SCHEMA, state="UNKNOWN", reasons=[], policy_id=identity(reference_policy),
        evidence_id=evidence_id, cutoff=timestamp(cutoff).isoformat(), context=context,
        policy_created_at=reference_policy["available_at"], historical_deployment_claim=False)
    if context not in CONTEXTS:
        audit["reasons"] = ["FR3_NO_TRAINED_CONTEXT_WEIGHTS"]
        return None, audit
    if setup_variant is not None:
        raise ValueError("FR3_NO_PATTERN_RELABEL_OR_SETUP_VARIANT")
    if bars is None or bars.empty:
        audit["reasons"] = ["SOURCE_BOUND_COMPLETED_BARS_MISSING"]
        return None, audit
    all_selected = select_bars(bars, cutoff)
    selected = all_selected.tail(121)
    if len(selected) != 121:
        audit["reasons"] = ["FR3_PRIOR_120_SESSION_HISTORY_MISSING"]
        return None, audit
    rows = selected.astype(object).where(selected.notna(), None).to_dict("records")
    for row in rows:
        row["Date"] = str(row["Date"])[:10]
    audit["bars_id"] = identity(rows)
    coverage = verify_window(session_evidence, market=reference_policy["cost_path_policy"]["market"],
        cutoff=cutoff, observed_sessions=[row["Date"] for row in rows])
    audit["session_coverage"] = coverage
    if coverage["state"] != "PASS":
        audit.update(state=coverage["state"], reasons=["FR3_OFFICIAL_SESSION_COVERAGE_"+coverage["state"]])
        return None, audit
    from domain.les_v7.plan_r10 import _pivots, _resistance
    prior = rows[:-1]
    supports = [p for p in _pivots(prior, "Low", low=True) if p["index"] >= len(prior)-60]
    if not supports:
        audit["reasons"] = ["FR3_INDEPENDENTLY_CONFIRMED_SUPPORT_MISSING"]
        return None, audit
    support = supports[-1]
    close, atr = features.get("close"), features.get("atr")
    from domain.les_v7.plan_r16 import _atr
    _, source_atr = _atr(all_selected)
    if (close is None or atr is None or atr <= 0 or close != rows[-1]["Close"]
            or atr != source_atr):
        audit["reasons"] = ["FR3_SOURCE_CLOSE_OR_ATR_MISSING_OR_CONFLICTING"]
        return None, audit
    resistance = _resistance(rows, _pivots(prior, "High", low=False), close, 0)
    audit.update(support=deepcopy(support), resistance=deepcopy(resistance))
    if resistance is None:
        audit["reasons"] = ["FR3_OBSERVED_RESISTANCE_MISSING"]
        return None, audit
    anchors = {len(rows)-1, support["index"], support["index"]+1, support["index"]+2,
        resistance["index"], resistance["index"]+1, resistance["index"]+2}
    if any(rows[i].get("Volume") is None for i in anchors):
        audit["reasons"] = ["FR3_ANCHOR_TRADING_EVIDENCE_MISSING"]
        return None, audit
    if any(rows[i]["Volume"] <= 0 for i in anchors):
        audit.update(state="FAIL", reasons=["FR3_ANCHOR_OR_CONFIRMATION_DID_NOT_TRADE"])
        return None, audit
    band = tuple(float(_round_price(Decimal(str(v)), direction)) for v, direction in (
        (support["value"]+.5*atr, "up"), (min(support["value"]+atr, close), "down")))
    stop = support["value"]-.3*atr
    audit.update(band=list(band), stop=stop, target=resistance["value"])
    if (any(row["Low"] < support["value"] for row in rows[support["index"]+1:])
            or not 0 < stop < band[0] <= band[1] <= close or resistance["value"] <= band[1]):
        audit.update(state="FAIL", reasons=["FR3_FIXED_STRUCTURE_OR_LIMIT_INVALID"])
        return None, audit
    shares = reference_policy["reference_shares"]
    costs = estimate_reference_costs(band[1], stop, resistance["value"], shares, reference_policy["cost_path_policy"])
    structure_id = identity(dict(support=support, resistance=resistance, band=list(band),
        selection="LATEST_CONFIRMED_SUPPORT_NEAREST_OBSERVED_RESISTANCE_BEFORE_RR"))
    plan = FR3ReferencePlan("REFERENCE_H20", PLAN_VERSION, evidence_id, context, band[1], stop,
        resistance["value"], atr, band, shares, costs["cost_up_per_share"], costs["cost_stop_per_share"],
        signal_close=close, structure_id=structure_id)
    audit.update(state="VERIFIED_ANCHORS", plan_id=plan.plan_id, structure_id=structure_id,
        conditional_entry=dict(state="UNFILLED", assumes_fill=False, live_order_authorized=False,
            order_type="LIMIT", entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", expiry="NEXT_SESSION_CLOSE"),
        band_evaluation=validate_band(plan, reference_policy))
    return plan, audit


def evaluate_entry(plan, observed_entry, reference_policy, *, starter=True):
    if isinstance(reference_policy, Mapping) and reference_policy.get("version") == L2_PLAN_VERSION:
        from domain.les_v7.fr3_l2 import evaluate_entry as l2_evaluate
        return l2_evaluate(plan, observed_entry, reference_policy, starter=starter)
    validate_policy(reference_policy)
    if plan is None:
        return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    if not isinstance(plan, FR3ReferencePlan):
        raise ValueError("FR3_SOURCE_BOUND_PLAN_REQUIRED")
    if not starter:
        return dict(state="FAIL", reasons=["FR3_CONDITIONAL_LIMIT_SMALL_ONLY"])
    entry = Decimal(str(finite(observed_entry, "FR3_observed_entry", 0)))
    if (not _legal(entry) or not Decimal(str(plan.band[0])) <= entry <= Decimal(str(plan.band[1]))
            or entry <= Decimal(str(plan.stop)) or plan.target is None or entry >= Decimal(str(plan.target))):
        return dict(state="FAIL", reasons=["FR3_ENTRY_OUTSIDE_FIXED_PLAN"])
    costs = estimate_reference_costs(float(entry), plan.stop, plan.target,
        plan.reference_shares, reference_policy["cost_path_policy"])
    stop_atr = (entry-Decimal(str(plan.stop)))/Decimal(str(plan.atr))
    rr = (Decimal(str(plan.target))-entry-Decimal(str(costs["cost_up_per_share"]))) / (
        entry-Decimal(str(plan.stop))+Decimal(str(costs["cost_stop_per_share"])))
    reasons = []
    if stop_atr > Decimal("2.5"):
        reasons.append("FR3_FIXED_STOP_ATR_EXCEEDED")
    if rr < Decimal("1.2"):
        reasons.append("FR3_FIXED_TARGET_NET_RR_BELOW_MINIMUM")
    return dict(state="FAIL" if reasons else "PASS", reasons=reasons, entry=float(entry),
        stop_atr=float(stop_atr), net_rr=float(rr), maximum_stop_atr=2.5, minimum_net_rr=1.2, costs=costs)


def validate_band(plan, reference_policy, *, starter=True):
    if plan is None:
        return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    result = evaluate_entry(plan, plan.band[1], reference_policy, starter=starter)
    return dict(result, band=list(plan.band), executable_band=list(plan.band),
        evaluation_basis="HIGHEST_PERMITTED_LEGAL_ENTRY_TICK")


def bind_entry_route(plan, reference_policy, *, strategy_decision="BUY_SMALL", risk_route="TACTICAL", setup_variant=None):
    if isinstance(reference_policy, Mapping) and reference_policy.get("version") == L2_PLAN_VERSION:
        from domain.les_v7.fr3_l2 import bind_entry_route as l2_bind
        return l2_bind(plan, reference_policy, strategy_decision=strategy_decision,
            risk_route=risk_route, setup_variant=setup_variant)
    validate_policy(reference_policy)
    if (not isinstance(plan, FR3ReferencePlan) or strategy_decision != "BUY_SMALL"
            or risk_route != "TACTICAL" or setup_variant is not None
            or plan.reference_shares != reference_policy["reference_shares"]):
        raise ValueError("FR3_LIMIT_REQUIRES_TACTICAL_SMALL_SOURCE_PLAN")
    plan_from_record(asdict(plan))
    if validate_band(plan, reference_policy)["state"] != "PASS":
        raise ValueError("FR3_ENTRY_ROUTE_NATIVE_RISK_NOT_PASS")
    result = dict(schema=ROUTE_SCHEMA, plan_id=plan.plan_id, reference_policy_id=identity(reference_policy),
        reference_policy=deepcopy(reference_policy), strategy_decision=strategy_decision, risk_route=risk_route,
        setup_variant=None, entry_basis=plan.entry_basis, signal_close=plan.signal_close,
        target_method=plan.target_method, structure_id=plan.structure_id,
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", order_type="LIMIT", expiry="NEXT_SESSION_CLOSE",
        authority="REFERENCE_RESEARCH_ROUTE_NOT_LIVE_ORDER_AUTHORITY")
    return dict(result, route_id=identity(result))
