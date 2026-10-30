"""R18 admission reuses verified R15 sources without weakening hard gates.

Only tactical-small Quality and economic warning deductions change. The
original core economics, source verdicts and absolute loss budgets are retained;
position limits never feed back into the LES score.
"""
from domain.les_v7.contracts import identity
from domain.les_v7.policy_r18 import (ADMISSION_VERSION, EARLY_PULLBACK,
    PRICE_DISCOVERY, SUPPORT_RESISTANCE_LIMIT, route_limits)


def policy():
    from domain.les_v7.entry_policy_r15 import policy as previous
    fixed = previous()
    tactical = dict(fixed["tactical"], quality_minimum=60., bear_warning_points=3.,
        price_bands=[dict(row, points=points) for row, points in
                     zip(fixed["tactical"]["price_bands"], (2., 4., 6.))])
    return dict(fixed, version=ADMISSION_VERSION, tactical=tactical,
        core_policy="UNCHANGED_R17_CORE_QUALITY_VALUATION_AND_PRICE_PENALTIES",
        valuation_semantics="VERIFIED_SOURCE_BOUND_R15_R18_TACTICAL_DEDUCTIONS",
        conditional_routes={v: route_limits(v) for v in
                            (EARLY_PULLBACK, PRICE_DISCOVERY, SUPPORT_RESISTANCE_LIMIT)},
        no_sizing_feedback_to_score=True)


def valuation_grade(receipt, *, current_price):
    """Keep core assumptions and grade; bind the new admission policy identity."""
    from domain.les_v7.entry_policy_r15 import valuation_grade as previous
    return dict(previous(receipt, current_price=current_price),
                admission_policy_id=identity(policy()))


def tactical_valuation_grade(receipt, *, current_price, cutoff=None):
    """Reprice only an exactly verified economic warning, never a source gap."""
    from domain.les_v7.entry_policy_r15 import tactical_valuation_grade as previous
    grade = dict(previous(receipt, current_price=current_price, cutoff=cutoff),
                 admission_policy_id=identity(policy()))
    if grade["state"] not in {"ALLOWED", "ALLOWED_WITH_WARNING"}:
        return grade
    tactical = policy()["tactical"]
    bands = {row["name"]: row for row in tactical["price_bands"]}
    price_points = 0. if grade["band"] == "PASS" else bands[grade["band"]]["points"]
    bear_points = tactical["bear_warning_points"] if grade["bear_warning"] else 0.
    return dict(grade, price_points=price_points, bear_points=bear_points,
                points=price_points + bear_points)


def assess(quality, gates, valuation_receipt, *, symbol, cutoff, current_price,
           setup_variant=None):
    """Return independent core and tactical routes sharing one native gate set."""
    from domain.les_v7.entry_policy_r9 import _assess
    from domain.les_v7.entry_policy_r15 import TACTICAL_SETUP

    conditional = {EARLY_PULLBACK, PRICE_DISCOVERY, SUPPORT_RESISTANCE_LIMIT}
    if setup_variant not in {None, TACTICAL_SETUP, *conditional}:
        raise ValueError("R18_UNKNOWN_SETUP_VARIANT")
    fixed = policy()
    core_grade = valuation_grade(valuation_receipt, current_price=current_price)
    tactical_grade = tactical_valuation_grade(valuation_receipt,
        current_price=current_price, cutoff=cutoff)
    core = _assess(quality, gates, valuation_receipt, symbol=symbol, cutoff=cutoff,
        current_price=current_price, admission_policy=fixed, valuation_result=core_grade)
    tactical = _assess(quality, gates, valuation_receipt, symbol=symbol, cutoff=cutoff,
        current_price=current_price, admission_policy=fixed, valuation_result=tactical_grade)
    restricted_core = setup_variant is not None or core_grade["method_small_only"]
    if restricted_core:
        core.update(buy_allowed=False, buy_total_cap=0.,
            small_total_cap=min(core["quality"]["small_cap"], .025) * core_grade["multiplier"])
        # Core's existing Quality floor does not inherit the tactical concession.
        if core["quality"]["score"] is None or core["quality"]["score"] < 65:
            core["small_allowed"] = False
    tactical_ready = (tactical["small_allowed"] and tactical["quality"]["score"] is not None
                      and tactical["quality"]["score"] >= fixed["tactical"]["quality_minimum"])
    limits = route_limits(setup_variant)
    tactical_cap = min(tactical["quality"]["small_cap"], tactical_grade["total_cap"],
                       limits["maximum_total_weight"])
    state = tactical["state"]
    if state == "ALLOWED" and not tactical_ready:
        state = "BLOCKED"
    if setup_variant in conditional:
        core["small_total_cap"] = min(core["small_total_cap"], limits["maximum_total_weight"])
    result = {key: value for key, value in tactical.items() if key != "assessment_id"}
    result.update(state=state, valuation=tactical_grade, setup_variant=setup_variant,
        policy_version=ADMISSION_VERSION, policy_id=identity(fixed), admission_policy_id=identity(fixed),
        restricted_starter=True, cap_basis="MINIMUM_ABSOLUTE_TOTAL_CAPS_LESS_HELD_EXPOSURE",
        risk_fraction=limits["total_plan_loss_fraction"], route_total_cap=limits["maximum_total_weight"],
        core_grade=core_grade, core_points=core_grade["points"], core_state=core["state"],
        core_buy_allowed=core["buy_allowed"], core_small_allowed=core["small_allowed"],
        core_buy_cap=core["buy_total_cap"], core_small_cap=core["small_total_cap"],
        core_risk_fraction=.00125 if restricted_core else None,
        tactical_grade=tactical_grade, tactical_points=tactical_grade["points"],
        tactical_small_allowed=tactical_ready, tactical_total_cap=tactical_cap,
        tactical_risk_fraction=limits["total_plan_loss_fraction"],
        buy_allowed=core["buy_allowed"], small_allowed=tactical_ready,
        buy_total_cap=core["buy_total_cap"], small_total_cap=tactical_cap,
        position_limits_affect_score=False)
    return dict(result, assessment_id=identity(result))
