"""R14 source-bound starter subtypes and total caps; size never changes score."""
from collections.abc import Mapping

from domain.les_v7.contracts import identity
from domain.les_v7.policy_r14 import ADMISSION_VERSION, VALUATION_VERSION


def policy():
    from domain.les_v7.entry_policy_r13 import policy as previous
    return dict(previous(), version=ADMISSION_VERSION, valuation_method=VALUATION_VERSION,
        restricted_method="RECOVERY_PEER_ONLY_R14", restricted_quality_minimum=65.,
        restricted_total_cap=.025, restricted_risk_fraction=.00125,
        restricted_setup="EARLY_SHORT_BASE_BREAKOUT_R14",
        method_position_multiplier=.5, signal_expiry="NEXT_OFFICIAL_SESSION_CLOSE")


def valuation_grade(receipt, *, current_price):
    from domain.les_v7.entry_policy_r13 import _valuation_grade
    from domain.les_v7.valuation_recovery_r14 import (
        evaluate_annual_price_valuation, evaluate_recovery_price_valuation,
    )
    grade = _valuation_grade(receipt, current_price=current_price, admission_policy=policy(),
        valuation_version=VALUATION_VERSION, annual_evaluator=evaluate_annual_price_valuation,
        scenario_evaluator=evaluate_recovery_price_valuation)
    diagnostic = receipt.get("valuation_diagnostics", {}) if isinstance(receipt, Mapping) else {}
    # _valuation_grade independently recalculates all expected diagnostic values.
    # Only an allowed, exact R14 receipt may activate a method sizing rule.
    restricted = grade["state"] in {"ALLOWED", "ALLOWED_WITH_WARNING"} and diagnostic.get("method_small_only") is True
    multiplier = .5 if restricted else 1.
    grade.update(method_small_only=restricted, method_position_multiplier=multiplier)
    if restricted:
        grade.update(small_only=True, multiplier=grade["multiplier"] * multiplier)
    return grade


def assess(quality, gates, valuation_receipt, *, symbol, cutoff, current_price, setup_variant=None):
    from domain.les_v7.entry_policy_r9 import _assess
    if setup_variant not in (None, policy()["restricted_setup"]):
        raise ValueError("R14_UNKNOWN_SETUP_VARIANT")
    result = _assess(quality, gates, valuation_receipt, symbol=symbol, cutoff=cutoff,
        current_price=current_price, admission_policy=policy(),
        valuation_result=valuation_grade(valuation_receipt, current_price=current_price))
    result.pop("assessment_id")
    restricted = setup_variant is not None or result["valuation"]["method_small_only"]
    result.update(setup_variant=setup_variant, restricted_starter=restricted,
        cap_basis="TOTAL_POSITION_AFTER_ALL_MULTIPLIERS_LESS_HELD_EXPOSURE",
        risk_fraction=.00125 if restricted else None)
    if restricted:
        # Caps apply before, then multiply by, price/bear/method limits. They
        # are not incremental quotas and must not be applied a second time.
        result.update(buy_allowed=False, buy_total_cap=0.,
            small_total_cap=min(result["quality"]["small_cap"], .025) * result["valuation"]["multiplier"])
        if result["quality"]["score"] is None or result["quality"]["score"] < 65:
            result.update(small_allowed=False)
            if result["state"] == "ALLOWED": result["state"] = "BLOCKED"
    return dict(result, assessment_id=identity(result))
