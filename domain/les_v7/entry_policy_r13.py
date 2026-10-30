"""R13 graded scenario admission; known warnings reduce size, not source truth."""
from collections.abc import Mapping
from decimal import Decimal, ROUND_HALF_UP

from domain.les_v7.contracts import identity
from domain.les_v7.policy_r13 import ADMISSION_VERSION, VALUATION_VERSION


def policy():
    from domain.les_v7.entry_policy_r9 import policy as previous
    return dict(previous(), version=ADMISSION_VERSION, maximum_bear_loss=.45,
        normal_bear_loss=.35, bear_warning_small_only=True, bear_warning_position_multiplier=.5,
        valuation_method=VALUATION_VERSION,
        bear_metric="DISCOUNTED_SCENARIO_VALUE_SHORTFALL_NOT_DRAWDOWN_OR_LOSS_PROBABILITY")


def valuation_grade(receipt, *, current_price):
    from domain.les_v7.valuation_relaxed_r13 import (
        evaluate_annual_price_valuation, evaluate_relaxed_price_valuation,
    )
    return _valuation_grade(receipt, current_price=current_price, admission_policy=policy(),
        valuation_version=VALUATION_VERSION, annual_evaluator=evaluate_annual_price_valuation,
        scenario_evaluator=evaluate_relaxed_price_valuation)


def _valuation_grade(receipt, *, current_price, admission_policy, valuation_version,
                     annual_evaluator, scenario_evaluator):
    """Shared exact grading, with each caller binding its complete policy/math.

    No caller-supplied PASS flags, position limits or scenario values are trusted.
    R13's public function and its historical result remain unchanged.
    """
    from domain.les_v7.entry_policy_r9 import _number
    grade = dict(native_state=receipt.get("state") if isinstance(receipt, Mapping) else "UNKNOWN",
        state="UNKNOWN", band=None, points=0., multiplier=0., small_only=False,
        price_ratio=None, reason="VALUATION_VERIFIED_SCENARIOS_REQUIRED",
        bear_warning=False, bear_value_shortfall=None, admission_policy_id=identity(admission_policy))
    if not isinstance(receipt, Mapping) or receipt.get("gate_id") != "VALUATION":
        return grade
    diagnostic = receipt.get("valuation_diagnostics")
    if not isinstance(diagnostic, Mapping) or not isinstance(diagnostic.get("inputs"), Mapping):
        if receipt.get("state") == "FAIL": grade.update(state="BLOCKED", reason="VALUATION_PREMISE_FAIL")
        return grade
    try:
        annual = annual_evaluator(**dict(diagnostic["inputs"]))
        expected = scenario_evaluator(legacy_result=annual,
            comparables=diagnostic.get("comparables", {}), recovery=diagnostic.get("recovery"))
    except (TypeError, ValueError, OverflowError):
        return grade
    if (diagnostic.get("policy_version") != valuation_version
            or any(identity(diagnostic.get(key)) != identity(value) for key, value in expected.items())
            or receipt.get("state") != expected["status"]
            or _number(current_price) is None or current_price != expected["inputs"].get("current_price")):
        return grade
    if not expected.get("scenarios") or not expected.get("gate_checks"):
        grade.update(state="BLOCKED" if expected["status"] == "FAIL" else "UNKNOWN", reason="VALUATION_PREMISE_NOT_VALID")
        return grade
    checks = expected["gate_checks"]
    grade["bear_value_shortfall"] = expected.get("bear_permanent_loss")
    if checks.get("BEAR_LOSS_LTE_HARD_MAXIMUM") is not True:
        grade.update(state="BLOCKED", reason="VALUATION_BEAR_LOSS_EXCEEDED")
        return grade
    warning = checks.get("BEAR_LOSS_LTE_MAXIMUM") is not True
    grade["bear_warning"] = warning
    allowed = {"BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE", "BEAR_LOSS_LTE_MAXIMUM"}
    if expected["status"] not in {"PASS", "FAIL"} or not set(expected["reasons"]) <= allowed:
        grade.update(state="BLOCKED" if expected["status"] == "FAIL" else "UNKNOWN", reason="VALUATION_NON_PRICE_FAILURE")
        return grade
    price_warning = any(not checks.get(key) for key in
                        ("BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE"))
    points, multiplier, small_only, band, ratio = 0., 1., warning, "PASS", None
    if price_warning:
        price = Decimal(str(expected["inputs"]["current_price"]))
        fair = Decimal(str(expected["scenarios"]["BASE"]["fair_price"]))
        ratio = (price / fair).quantize(Decimal(".000000000001"), rounding=ROUND_HALF_UP)
        selected = next(row for row in admission_policy["valuation_bands"] if row["maximum_ratio"] is None
                        or ratio <= Decimal(str(row["maximum_ratio"])))
        points, multiplier, band = selected["points"], selected["multiplier"], selected["name"]
        small_only = small_only or selected["name"] == "HIGH"
    if warning:
        multiplier *= admission_policy["bear_warning_position_multiplier"]
    grade.update(state="ALLOWED_WITH_WARNING" if warning or price_warning else "ALLOWED",
        band=band, points=points, multiplier=multiplier, small_only=small_only,
        price_ratio=float(ratio) if ratio is not None else None,
        reason=("VALUATION_PRICE_AND_BEAR_WARNING" if warning and price_warning else
                "VALUATION_BEAR_WARNING" if warning else "VALUATION_PRICE_WARNING" if price_warning else "VALUATION_NATIVE_PASS"))
    return grade


def assess(quality, gates, valuation_receipt, *, symbol, cutoff, current_price):
    from domain.les_v7.entry_policy_r9 import _assess
    return _assess(quality, gates, valuation_receipt, symbol=symbol, cutoff=cutoff,
        current_price=current_price, admission_policy=policy(),
        valuation_result=valuation_grade(valuation_receipt, current_price=current_price))
