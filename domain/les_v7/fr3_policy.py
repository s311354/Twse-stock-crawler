"""Explicit FR3 research policies; original model/daily policies are unchanged.

EPS-R2 deliberately reuses the approved R13 *annual-only* arithmetic.  Neither
interim EPS annualization nor guessed dividends/share counts are introduced.
Publication, evidence availability and historical feature cutoffs are separate.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from domain.les_v7.contracts import identity, timestamp

AVAILABLE_AT = "2026-10-23T15:21:04Z"
EPS_VERSION = "FR3_REPORTED_ANNUAL_EPS_PRICE_R2"
INPUT_VERSION = "FR3_VERIFIED_ANNUAL_EPS_INPUT_R2"
ENTRY_VERSION = "FR_ENTRY_R3"
EXECUTION_VERSION = "FR_EXEC_R3"
L2_ENTRY_VERSION = "FR_ENTRY_R3_L2"
L2_EXECUTION_VERSION = "FR_EXEC_R3_L2"


def _l2_binding(reference_version, available_at):
    """Resolve one explicit new plan; absent version retains original bytes."""
    from domain.les_v7.fr3_plan import PLAN_VERSION
    if reference_version is None or reference_version == PLAN_VERSION:
        return None
    from domain.les_v7.fr3_l2 import PLAN_VERSION as l2, structural_policy
    if reference_version != l2:
        raise ValueError("FR3_ADMISSION_REFERENCE_VERSION_UNSUPPORTED")
    if available_at is None:
        raise ValueError("FR3_L2_EXPLICIT_POLICY_CREATION_TIME_REQUIRED")
    created = timestamp(available_at).isoformat()
    structure = structural_policy()
    return dict(available_at=created, reference_plan_version=l2,
        structural_policy=structure, structural_policy_id=identity(structure))


def eps_policy():
    from domain.les_v7.valuation_relaxed_r13 import policy as original
    result = original()
    result.update(schema="les-v7-fr3-eps-price/v1", policy_version=EPS_VERSION,
        available_at=AVAILABLE_AT, native_arithmetic_policy_id=identity(original()),
        selected_method="VERIFIED_ANNUAL_ONLY_NO_RECOVERY_OR_PEER_OVERRIDE",
        research_only=True, approved_for_live=False)
    return result


def evaluate_eps_r2(**inputs):
    from domain.les_v7.valuation_relaxed_r13 import evaluate_annual_price_valuation
    result = evaluate_annual_price_valuation(**inputs)
    fixed = eps_policy()
    result.update(schema=fixed["schema"], policy_version=EPS_VERSION,
                  policy=fixed, policy_id=identity(fixed),
                  selected_method=fixed["selected_method"])
    return result


def entry_policy(*, reference_version=None, available_at=None):
    result = dict(schema="les-v7-fr3-entry-policy/v1", version=ENTRY_VERSION,
        available_at=AVAILABLE_AT, maximum_decision="BUY_SMALL", small_minimum=65.,
        minimum_quality_score=70., valuation_warning_points=5.,
        maximum_stop_atr=2.5, minimum_net_rr=1.2,
        minimum_fill_probability=.5, minimum_bottom_soon_probability=.5,
        minimum_conditional_entry_probability=.5,
        maximum_bottom_interval_atr=2., maximum_entry_minus_bottom_atr=1.,
        retain_native_technical_vetoes=True, limit_band_not_a_fill=True,
        admission_trigger="VERIFIED_CONDITIONAL_LIMIT_PLAN_NO_DUPLICATE_CLOSE_TRIGGER",
        unknown_blocks=True, position_limits_affect_score=False)
    bound = _l2_binding(reference_version, available_at)
    if bound is not None:
        result.update(schema="les-v7-fr3-entry-policy/v2", version=L2_ENTRY_VERSION, **bound)
    return result


def execution_policy(*, reference_version=None, available_at=None):
    result = dict(schema="les-v7-fr3-execution-policy/v1", version=EXECUTION_VERSION,
        available_at=AVAILABLE_AT, total_symbol_cap=.01, valuation_warning_cap=.005,
        planned_symbol_nav_risk=.0005, planned_portfolio_nav_risk=.01,
        maximum_holdings=4, maximum_adv_fraction=.01, maximum_spread=.005,
        fill_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_FIXED_LIMIT_BAND",
        expiry="NEXT_SESSION_CLOSE", live_order_authorized=False)
    bound = _l2_binding(reference_version, available_at)
    if bound is not None:
        result.update(schema="les-v7-fr3-execution-policy/v2", version=L2_EXECUTION_VERSION,
            fill_rule="LEADER_FIRST_ELIGIBLE_OPEN_IN_NEXT_THREE_OFFICIAL_SESSIONS_OTHERS_NEXT_OPEN",
            expiry="LEADER_THIRD_OFFICIAL_SESSION_CLOSE_OTHERS_NEXT_SESSION_CLOSE", **bound)
    return result


def valuation_admission(receipt):
    """Grade verified economics, never interpret missing facts as a warning."""
    if not isinstance(receipt, Mapping):
        return dict(state="UNKNOWN", warning=False, penalty=0., cap_multiplier=0.)
    state = receipt.get("state", "UNKNOWN")
    snapshot = receipt.get("valuation_diagnostics", {})
    if snapshot.get("policy_id") != identity(eps_policy()):
        return dict(state="UNKNOWN", warning=False, penalty=0., cap_multiplier=0.)
    checks = snapshot.get("gate_checks")
    required = {"BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE",
                "BEAR_LOSS_LTE_MAXIMUM", "BEAR_LOSS_LTE_HARD_MAXIMUM"}
    complete = bool(snapshot.get("scenarios") and isinstance(checks, Mapping) and set(checks) == required
                    and all(type(v) is bool for v in checks.values()))
    if state == "PASS" and complete and all(checks.values()):
        return dict(state="PASS", warning=False, penalty=0., cap_multiplier=1.)
    if state == "PASS":
        return dict(state="UNKNOWN", warning=False, penalty=0., cap_multiplier=0.)
    failed = {key for key, value in (checks or {}).items() if value is False}
    allowed = {"BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE",
               "BEAR_LOSS_LTE_MAXIMUM"}
    warning = bool(state == "FAIL" and complete and failed and failed <= allowed
        and checks.get("BEAR_LOSS_LTE_HARD_MAXIMUM") is True)
    return dict(state="ALLOWED_WITH_WARNING" if warning else state,
        warning=warning, penalty=5. if warning else 0., cap_multiplier=.5 if warning else 0.)


def components(*, reference_version=None, available_at=None):
    result = {}
    for name, value in (("entry", entry_policy(reference_version=reference_version, available_at=available_at)),
                        ("execution", execution_policy(reference_version=reference_version, available_at=available_at)),
                        ("valuation", eps_policy())):
        result[name] = dict(policy=deepcopy(value), component_id=identity(value))
    return result
