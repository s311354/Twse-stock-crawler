"""R15 core admission and separately scored tactical-small total caps.

Native valuation FAIL is retained. Only exact, complete economic scenarios may
receive tactical price penalties; no unknown premise or hard safety gate is
compensable. Sizing never feeds back into the LES score.
"""
from collections.abc import Mapping
from decimal import Decimal, ROUND_HALF_UP

from domain.les_v7.contracts import finite, identity, timestamp
from domain.les_v7.policy_r15 import ADMISSION_VERSION, VALUATION_VERSION


TACTICAL_SETUP = "EARLY_SHORT_BASE_BREAKOUT_R15"
EARLY_EPS_FAILURE_SCHEMA = "les-v7-source-bound-early-eps-failure/v1"
# Diagnostic code capability, not an investment-policy parameter. Earlier
# retained R15 reports must still reproduce their original UNKNOWN assessment.
EARLY_EPS_FAILURE_AVAILABLE_AT = "2026-10-16T10:39:50Z"


def policy():
    from domain.les_v7.entry_policy_r14 import policy as previous
    return dict(previous(), version=ADMISSION_VERSION, valuation_method=VALUATION_VERSION,
        restricted_method="RECOVERY_PEER_ONLY_R15", restricted_setup=TACTICAL_SETUP,
        core_policy="R14_QUALITY_VALUATION_ADMISSION_AND_ORIGINAL_PRICE_PENALTIES",
        tactical=dict(signal="BUY_SMALL_ONLY", quality_minimum=65.,
            minimum_data_confidence=65., minimum_coverage=.75,
            total_weight_cap=.01, total_plan_loss_fraction=.0005,
            price_bands=[dict(name="MILD", maximum_ratio=1.10, points=3., total_cap=.01),
                         dict(name="MODERATE", maximum_ratio=1.25, points=5., total_cap=.0075),
                         dict(name="HIGH", maximum_ratio=None, points=8., total_cap=.005)],
            bear_warning_above=.35, bear_warning_points=4., bear_warning_total_cap=.0025,
            peer_only_total_cap=.0125,
            cap_basis="MINIMUM_ABSOLUTE_TOTAL_CAPS_LESS_EXISTING_EXPOSURE_NOT_MULTIPLIERS",
            noncompensable="UNKNOWN_NONPRICE_PREMISE_MAJOR_FINANCIAL_AND_HARD_GATES"))


def valuation_grade(receipt, *, current_price):
    """Core-only grade: unchanged R14 economic admission and penalty behavior."""
    from domain.les_v7.entry_policy_r13 import _valuation_grade
    from domain.les_v7.valuation_recovery_r15 import evaluate_annual_price_valuation, evaluate_recovery_price_valuation
    grade = _valuation_grade(receipt, current_price=current_price, admission_policy=policy(),
        valuation_version=VALUATION_VERSION, annual_evaluator=evaluate_annual_price_valuation,
        scenario_evaluator=evaluate_recovery_price_valuation)
    d = receipt.get("valuation_diagnostics", {}) if isinstance(receipt, Mapping) else {}
    restricted = grade["state"] in {"ALLOWED", "ALLOWED_WITH_WARNING"} and d.get("method_small_only") is True
    grade.update(method_small_only=restricted, method_position_multiplier=.5 if restricted else 1.)
    if restricted:
        grade.update(small_only=True, multiplier=grade["multiplier"] * .5)
    return grade


def _verified_scenarios(receipt, *, current_price):
    from domain.les_v7.entry_policy_r9 import _number
    from domain.les_v7.valuation_recovery_r15 import evaluate_annual_price_valuation, evaluate_recovery_price_valuation
    if not isinstance(receipt, Mapping) or receipt.get("gate_id") != "VALUATION":
        return None
    d = receipt.get("valuation_diagnostics")
    if not isinstance(d, Mapping) or not isinstance(d.get("inputs"), Mapping):
        return None
    try:
        annual = evaluate_annual_price_valuation(**dict(d["inputs"]))
        expected = evaluate_recovery_price_valuation(legacy_result=annual,
            comparables=d.get("comparables", {}), recovery=d.get("recovery"))
    except (TypeError, ValueError, OverflowError):
        return None
    if (d.get("policy_version") != VALUATION_VERSION
            or any(identity(d.get(key)) != identity(value) for key, value in expected.items())
            or receipt.get("state") != expected["status"]
            or _number(current_price) is None or current_price != expected["inputs"].get("current_price")):
        return None
    return expected


def _verified_early_eps_failure(receipt, *, cutoff):
    """Recognize only the native producer's existing, bound EPS rejection.

    This is a rejection-only receipt check, not original-source certification.
    The native report validator still regenerates the receipt from the retained
    source packet. No scenarios or missing monetary operands are invented, and
    the producer bytes/policy identity remain unchanged for old archives.
    """
    from domain.les_v7.valuation_eps_receipt import TACTICAL_PRODUCER
    from domain.les_v7.valuation_recovery_r15 import policy as valuation_policy

    if (not isinstance(receipt, Mapping) or receipt.get("gate_id") != "VALUATION"
            or receipt.get("gate_version") != TACTICAL_PRODUCER
            or receipt.get("state") != "FAIL" or cutoff is None):
        return None
    d = receipt.get("valuation_diagnostics")
    if (not isinstance(d, Mapping) or d.get("status") != "FAIL"
            or d.get("return_basis") != "PRICE_RETURN_ONLY"
            or "inputs" in d or "scenarios" in d or "gate_checks" in d):
        return None
    observed = d.get("observed_inputs")
    if (not isinstance(observed, Mapping)
            or set(observed) != {"current_eps", "prior_eps", "eps_basis"}
            or observed.get("eps_basis") not in {"BASIC", "DILUTED"}):
        return None
    try:
        fixed = valuation_policy()
        if (identity(d.get("policy")) != identity(fixed)
                or d.get("policy_id") != identity(fixed)):
            return None
        checked, available = timestamp(receipt["checked_at"]), timestamp(receipt["source_available_at"])
        if (checked != timestamp(cutoff) or checked < timestamp(EARLY_EPS_FAILURE_AVAILABLE_AT)
                or not timestamp(fixed["available_at"]) <= available <= checked):
            return None
        values = {key: finite(observed[key]) for key in ("current_eps", "prior_eps")}
        reasons = ["NON_POSITIVE_" + key.upper() for key, value in values.items() if value <= 0]
        if not reasons or d.get("reasons") != reasons or receipt.get("reason_codes") != sorted(reasons):
            return None
        source_ids, proof_ids = receipt.get("input_ids"), receipt.get("proof_record_ids")
        ids = [*(source_ids or ()), *(proof_ids or ()), receipt.get("input_hash")]
        if (not isinstance(source_ids, (list, tuple)) or len(source_ids) != 1
                or not isinstance(proof_ids, (list, tuple)) or len(proof_ids) != 1
                or any(not isinstance(value, str) or len(value) != 64
                       or any(ch not in "0123456789abcdef" for ch in value) for value in ids)):
            return None
        source_policy_keys = {"input_policy", "input_policy_id", "quality_dependency", "entry_quality_required"}
        if source_policy_keys & set(d):
            from domain.les_v7.valuation_input_policy import require_input_policy
            from domain.les_v7.policy_r15 import INPUT_VERSION
            if not source_policy_keys <= set(d):
                return None
            source_policy = require_input_policy(d["input_policy"], cutoff=cutoff)
            if (source_policy["policy_version"] != INPUT_VERSION
                    or d["input_policy_id"] != identity(source_policy)
                    or d["quality_dependency"] != source_policy["quality_dependency"]
                    or d["entry_quality_required"] is not True):
                return None
        proof = dict(producer=TACTICAL_PRODUCER, gate_id="VALUATION", source_id=source_ids[0],
                     input_hash=receipt["input_hash"], result=d)
        if proof_ids[0] != identity(proof):
            return None
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return None
    return dict(schema=EARLY_EPS_FAILURE_SCHEMA, reason_codes=reasons,
                source_id=source_ids[0], proof_id=proof_ids[0], input_hash=receipt["input_hash"],
                checked_at=checked.isoformat())


def tactical_valuation_grade(receipt, *, current_price, cutoff=None):
    fixed = policy(); tactical = fixed["tactical"]
    grade = dict(native_state=receipt.get("state") if isinstance(receipt, Mapping) else "UNKNOWN",
        state="UNKNOWN", band=None, points=0., multiplier=0., small_only=True,
        price_points=0., bear_points=0.,
        price_ratio=None, reason="VALUATION_VERIFIED_SCENARIOS_REQUIRED",
        bear_warning=False, bear_value_shortfall=None, total_cap=0.,
        price_total_cap=0., bear_total_cap=None, method_total_cap=None,
        method_small_only=False, method_position_multiplier=1.,
        admission_policy_id=identity(fixed), route="TACTICAL_SMALL")
    expected = _verified_scenarios(receipt, current_price=current_price)
    if expected is None:
        failed = _verified_early_eps_failure(receipt, cutoff=cutoff)
        if failed is not None:
            grade.update(state="BLOCKED", reason="VALUATION_VERIFIED_EPS_PREREQUISITE_FAIL",
                         prerequisite_failure_schema=EARLY_EPS_FAILURE_SCHEMA,
                         prerequisite_failure_id=identity(failed))
        return grade
    checks = expected.get("gate_checks")
    if not expected.get("scenarios") or not checks:
        grade.update(state="BLOCKED" if expected["status"] == "FAIL" else "UNKNOWN",
                     reason="VALUATION_PREMISE_NOT_VALID")
        return grade
    allowed = {"BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE",
               "BEAR_LOSS_LTE_MAXIMUM", "BEAR_LOSS_LTE_HARD_MAXIMUM"}
    if expected["status"] not in {"PASS", "FAIL"} or not set(expected["reasons"]) <= allowed:
        grade.update(state="BLOCKED" if expected["status"] == "FAIL" else "UNKNOWN",
                     reason="VALUATION_NON_PRICE_FAILURE")
        return grade
    price_warning = any(checks.get(key) is not True for key in
        ("BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE"))
    price_points, price_cap, band, ratio = 0., tactical["total_weight_cap"], "PASS", None
    if price_warning:
        price = Decimal(str(expected["inputs"]["current_price"]))
        fair = Decimal(str(expected["scenarios"]["BASE"]["fair_price"]))
        ratio = (price / fair).quantize(Decimal(".000000000001"), rounding=ROUND_HALF_UP)
        selected = next(row for row in tactical["price_bands"] if row["maximum_ratio"] is None
            or ratio <= Decimal(str(row["maximum_ratio"])))
        price_points, price_cap, band = selected["points"], selected["total_cap"], selected["name"]
    bear_warning = checks.get("BEAR_LOSS_LTE_MAXIMUM") is not True
    bear_cap = tactical["bear_warning_total_cap"] if bear_warning else None
    peer_only = expected.get("method_small_only") is True
    method_cap = tactical["peer_only_total_cap"] if peer_only else None
    caps = [tactical["total_weight_cap"], price_cap, *([bear_cap] if bear_cap is not None else []),
            *([method_cap] if method_cap is not None else [])]
    grade.update(state="ALLOWED_WITH_WARNING" if price_warning or bear_warning else "ALLOWED",
        band=band, points=price_points + (tactical["bear_warning_points"] if bear_warning else 0.),
        price_points=price_points, bear_points=tactical["bear_warning_points"] if bear_warning else 0.,
        multiplier=1., price_ratio=float(ratio) if ratio is not None else None,
        price_total_cap=price_cap, bear_total_cap=bear_cap, method_total_cap=method_cap,
        total_cap=min(caps), bear_warning=bear_warning,
        bear_value_shortfall=expected.get("bear_permanent_loss"), method_small_only=peer_only,
        reason="TACTICAL_VERIFIED_ECONOMIC_WARNINGS" if price_warning or bear_warning else "TACTICAL_NATIVE_PASS")
    return grade


def assess(quality, gates, valuation_receipt, *, symbol, cutoff, current_price, setup_variant=None):
    """Return both score routes; the caller must never use tactical points for BUY."""
    from domain.les_v7.entry_policy_r9 import _assess
    if setup_variant not in (None, TACTICAL_SETUP):
        raise ValueError("R15_UNKNOWN_SETUP_VARIANT")
    fixed = policy()
    core_grade = valuation_grade(valuation_receipt, current_price=current_price)
    tactical_grade = tactical_valuation_grade(valuation_receipt, current_price=current_price, cutoff=cutoff)
    core = _assess(quality, gates, valuation_receipt, symbol=symbol, cutoff=cutoff,
        current_price=current_price, admission_policy=fixed, valuation_result=core_grade)
    tactical = _assess(quality, gates, valuation_receipt, symbol=symbol, cutoff=cutoff,
        current_price=current_price, admission_policy=fixed, valuation_result=tactical_grade)
    restricted_core = setup_variant is not None or core_grade["method_small_only"]
    if restricted_core:
        core.update(buy_allowed=False, buy_total_cap=0.,
            small_total_cap=min(core["quality"]["small_cap"], .025) * core_grade["multiplier"])
        if core["quality"]["score"] is None or core["quality"]["score"] < 65:
            core["small_allowed"] = False
    tactical_ready = (tactical["small_allowed"] and tactical["quality"]["score"] is not None
                      and tactical["quality"]["score"] >= fixed["tactical"]["quality_minimum"])
    tactical_cap = min(tactical["quality"]["small_cap"], tactical_grade["total_cap"])
    # The common state excludes only compensable economic valuation failures;
    # required unknown evidence and financial/hard failures still block both.
    state = tactical["state"]
    if state == "ALLOWED" and not tactical_ready:
        state = "BLOCKED"
    result = {key: value for key, value in tactical.items() if key != "assessment_id"}
    result.update(state=state, valuation=tactical_grade, setup_variant=setup_variant,
        restricted_starter=True, cap_basis="MINIMUM_ABSOLUTE_TOTAL_CAPS_LESS_HELD_EXPOSURE",
        risk_fraction=.0005, core_grade=core_grade, core_points=core_grade["points"],
        core_state=core["state"], core_buy_allowed=core["buy_allowed"], core_small_allowed=core["small_allowed"],
        core_buy_cap=core["buy_total_cap"], core_small_cap=core["small_total_cap"],
        core_risk_fraction=.00125 if restricted_core else None,
        tactical_grade=tactical_grade, tactical_points=tactical_grade["points"],
        tactical_small_allowed=tactical_ready, tactical_total_cap=tactical_cap,
        tactical_risk_fraction=.0005,
        buy_allowed=core["buy_allowed"], small_allowed=tactical_ready,
        buy_total_cap=core["buy_total_cap"], small_total_cap=tactical_cap)
    return dict(result, assessment_id=identity(result))
