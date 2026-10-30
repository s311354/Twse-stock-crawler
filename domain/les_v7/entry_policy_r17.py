"""R17 preserves R16 source/Quality/valuation grades and absolute route limits."""
from domain.les_v7.contracts import identity
from domain.les_v7.policy_r17 import (ADMISSION_VERSION, EARLY_PULLBACK,
                                     PRICE_DISCOVERY, route_limits)


def policy():
    from domain.les_v7.entry_policy_r15 import policy as previous
    return dict(previous(), version=ADMISSION_VERSION,
        valuation_semantics="UNCHANGED_SOURCE_BOUND_R15",
        conditional_routes={v: route_limits(v) for v in (EARLY_PULLBACK, PRICE_DISCOVERY)},
        no_sizing_feedback_to_score=True)


def assess(quality, gates, valuation_receipt, *, symbol, cutoff, current_price,
           setup_variant=None):
    from domain.les_v7.entry_policy_r15 import assess as previous, TACTICAL_SETUP
    if setup_variant not in {None, TACTICAL_SETUP, EARLY_PULLBACK, PRICE_DISCOVERY}:
        raise ValueError("R17_UNKNOWN_SETUP_VARIANT")
    new_route = setup_variant in {EARLY_PULLBACK, PRICE_DISCOVERY}
    result = previous(quality, gates, valuation_receipt, symbol=symbol, cutoff=cutoff,
        current_price=current_price, setup_variant=TACTICAL_SETUP if new_route else setup_variant)
    result = {k: v for k, v in result.items() if k != "assessment_id"}
    limits = route_limits(setup_variant)
    if new_route:
        result.update(core_buy_allowed=False, buy_allowed=False, core_buy_cap=0., buy_total_cap=0.)
        for key in ("small_total_cap", "tactical_total_cap", "core_small_cap"):
            result[key] = min(result[key], limits["maximum_total_weight"])
    result.update(policy_version=ADMISSION_VERSION, policy_id=identity(policy()), admission_policy_id=identity(policy()),
        setup_variant=setup_variant, route_total_cap=limits["maximum_total_weight"],
        tactical_risk_fraction=limits["total_plan_loss_fraction"], risk_fraction=limits["total_plan_loss_fraction"],
        position_limits_affect_score=False)
    return dict(result, assessment_id=identity(result))
