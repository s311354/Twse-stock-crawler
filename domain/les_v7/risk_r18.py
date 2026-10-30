"""One complete source-bound R18 execution assessment per native production.

Consumers verify its binding rather than repeat eligibility arithmetic. Public
report validation still independently regenerates the native source receipt;
this is not a cached PASS, cross-account result, or live execution permission.
"""
from collections.abc import Mapping
from copy import deepcopy

from domain.les_v7.contracts import finite, identity

SCHEMA = "les-v7-r18-execution-assessment/v1"


def _operands(plan, reference_policy, features):
    return dict(plan_id=plan.plan_id if plan else None,
        reference_policy_id=identity(reference_policy),
        spread=features.get("spread"), adv=features.get("adv"))


def assess_execution(plan, reference_policy, features):
    from domain.les_v7.plan_r18 import evaluate_entry
    operands = _operands(plan, reference_policy, features)
    spread, adv = operands["spread"], operands["adv"]
    for value in (spread, adv):
        if value is not None: finite(value)
    participation = plan.entry * plan.reference_shares / adv if plan and adv is not None and adv > 0 else None
    known = spread is not None and spread >= 0 and participation is not None
    liquidity = dict(state="PASS" if known and spread <= .005 and participation <= .01 else
        "FAIL" if known else "UNKNOWN", known=known, spread=spread, adv=adv,
        participation=participation, maximum_spread=.005, maximum_reference_order_adv=.01)
    # Equal close/band prices are repriced once. The dictionaries exposed below
    # are detached, so a consumer cannot mutate another route's assessment.
    checks = {}
    def check(starter, entry):
        key = starter, entry
        if key not in checks:
            checks[key] = evaluate_entry(plan, entry, reference_policy, starter=starter)
        return deepcopy(checks[key])
    routes = {}
    for name, starter in (("core", False), ("tactical", True)):
        close = check(starter, plan.entry if plan else None)
        band = check(starter, plan.band[1] if plan else None)
        if plan:
            band.update(band=list(plan.band), executable_band=list(plan.band),
                        evaluation_basis="HIGHEST_PERMITTED_LEGAL_ENTRY_TICK")
        states = (close["state"], band["state"], liquidity["state"])
        routes[name] = dict(close=close, band=band,
            state="FAIL" if "FAIL" in states else "UNKNOWN" if "UNKNOWN" in states else "PASS")
    result = dict(schema=SCHEMA, operands=operands, liquidity=liquidity, **routes,
        state=routes["tactical"]["state"],
        pass_scope="TACTICAL_SMALL_MINIMUM_NOT_CORE_BUY_AUTHORIZATION",
        complete_validation="SOURCE_RECONSTRUCTION_PLUS_FIXED_PLAN_COST_GRID_LIQUIDITY_CORE_AND_TACTICAL")
    return dict(result, assessment_id=identity(result))


def require_assessment(value, plan, reference_policy, features):
    if (not isinstance(value, Mapping) or value.get("schema") != SCHEMA
            or value.get("operands") != _operands(plan, reference_policy, features)
            or value.get("assessment_id") != identity({k: v for k, v in value.items() if k != "assessment_id"})):
        raise ValueError("R18_EXECUTION_ASSESSMENT_BINDING_MISMATCH")
    return deepcopy(dict(value))
