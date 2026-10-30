"""User-approved, versioned admission and total-position limits, not probability.

Original Quality and valuation verdicts stay intact. Only verified economic
valuation failures are compensable; unknown evidence and risk failures are not.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
from math import isfinite

from domain.les_v7.contracts import canonical, identity
from domain.strategies.portfolio_allocation_v2_contracts import PortfolioSecurityV2

VERSION = "V7_GRADED_ENTRY_AND_TOTAL_CAP_R1"
SCHEMA = "les-v7-entry-admission/v1"


def policy():
    return dict(version=VERSION, quality_buy_minimum=65., quality_small_minimum=60.,
        minimum_data_confidence=65., minimum_coverage=.75,
        quality_tiers=[dict(name="HIGH", minimum=75., buy_cap=.20, small_cap=.05),
                       dict(name="STANDARD", minimum=65., buy_cap=.10, small_cap=.025),
                       dict(name="LIMITED", minimum=60., buy_cap=0., small_cap=.02)],
        valuation_bands=[dict(name="MILD", maximum_ratio=1.10, points=5., multiplier=.75),
                         dict(name="MODERATE", maximum_ratio=1.25, points=10., multiplier=.50),
                         dict(name="HIGH", maximum_ratio=None, points=15., multiplier=.25)],
        high_valuation_small_only=True, maximum_bear_loss=.35,
        ratio_comparison="DECIMAL_RATIO_12_PLACES_HALF_UP_BANDS_ONLY_NOT_BEAR_GATE",
        raw_verdicts="RETAINED_NOT_RELABELLED_PASS", required_sources="UNCHANGED_VERIFIED_R6",
        cap_basis="TOTAL_POSITION_WEIGHT_NOT_INCREMENTAL_OR_SCORE",
        small_cap_already_scaled=True, performance_validated=False,
        pae_cer="UNCHANGED_VERIFIED_SIDECAR_INPUTS_NO_INVENTED_CONFIDENCE")


def _number(value):
    return float(value) if not isinstance(value, bool) and isinstance(value, (float, int)) and isfinite(value) else None


def quality_tier(quality):
    qs, dc, coverage = (_number(quality.get(k)) for k in ("quality_score", "data_confidence", "coverage_total"))
    rejected = quality.get("status") == "REJECT" or quality.get("financial_safety", {}).get("state") == "FAIL"
    ready = bool(not rejected and quality.get("status") in {"PASS", "WATCH_QUALITY"}
        and quality.get("profile_id") in {"QG_V7_REPORTED_PERIOD_R6", "QG_V7_REPORTED_PERIOD_R7"}
        and quality.get("core_inputs_pass") is True and quality.get("core_reported_pass") is True
        and quality.get("mandatory_safety_pass") is True and quality.get("native_quality_complete") is True
        and quality.get("financial_safety", {}).get("state") == "PASS"
        and qs is not None and 0 <= qs <= 100 and dc is not None and 65 <= dc <= 100
        and coverage is not None and .75 <= coverage <= 1.)
    tier = next((row for row in policy()["quality_tiers"] if ready and qs >= row["minimum"]), None)
    return dict(native_status=quality.get("status"), score=qs, evidence_ready=ready,
        state="REJECT" if rejected else "UNKNOWN" if not ready else "READY" if tier else "BELOW_MINIMUM",
        tier=tier["name"] if tier else None, buy_cap=tier["buy_cap"] if tier else 0.,
        small_cap=tier["small_cap"] if tier else 0.)


def valuation_grade(receipt, *, current_price):
    from domain.les_v7.policy_r15 import VALUATION_VERSION as r15_version
    diagnostic = receipt.get("valuation_diagnostics") if isinstance(receipt, Mapping) else None
    if isinstance(diagnostic, Mapping) and diagnostic.get("policy_version") == r15_version:
        from domain.les_v7.entry_policy_r15 import valuation_grade as revised
        return revised(receipt, current_price=current_price)
    from domain.les_v7.policy_r14 import VALUATION_VERSION as r14_version
    diagnostic = receipt.get("valuation_diagnostics") if isinstance(receipt, Mapping) else None
    if isinstance(diagnostic, Mapping) and diagnostic.get("policy_version") == r14_version:
        from domain.les_v7.entry_policy_r14 import valuation_grade as revised
        return revised(receipt, current_price=current_price)
    from domain.les_v7.policy_r13 import VALUATION_VERSION
    diagnostic = receipt.get("valuation_diagnostics") if isinstance(receipt, Mapping) else None
    if isinstance(diagnostic, Mapping) and diagnostic.get("policy_version") == VALUATION_VERSION:
        from domain.les_v7.entry_policy_r13 import valuation_grade as revised
        return revised(receipt, current_price=current_price)
    grade = dict(native_state=receipt.get("state") if isinstance(receipt, Mapping) else "UNKNOWN",
        state="UNKNOWN", band=None, points=0., multiplier=0., small_only=False,
        price_ratio=None, reason="VALUATION_VERIFIED_SCENARIOS_REQUIRED")
    if not isinstance(receipt, Mapping) or receipt.get("gate_id") != "VALUATION": return grade
    diagnostic = receipt.get("valuation_diagnostics")
    if not isinstance(diagnostic, Mapping) or not isinstance(diagnostic.get("inputs"), Mapping):
        if receipt.get("state") == "FAIL": grade.update(state="BLOCKED", reason="VALUATION_PREMISE_FAIL")
        return grade
    from domain.les_v7.valuation_price import evaluate_price_valuation
    try:
        expected = evaluate_price_valuation(**dict(diagnostic["inputs"]))
        if diagnostic.get("policy_version") == "V7_RECOVERY_INDUSTRY_PRICE_RETURN_R10":
            from domain.les_v7.valuation_recovery_r10 import evaluate_recovery_price_valuation
            expected = evaluate_recovery_price_valuation(legacy_result=expected,
                comparables=diagnostic.get("comparables", {}), recovery=diagnostic.get("recovery"))
    except (TypeError, ValueError, OverflowError):
        return grade
    if (any(identity(diagnostic.get(key)) != identity(value) for key, value in expected.items())
            or receipt.get("state") != expected["status"]
            or _number(current_price) is None or current_price != expected["inputs"].get("current_price")):
        return grade
    if not expected.get("scenarios") or not expected.get("gate_checks"):
        grade.update(state="BLOCKED" if expected["status"] == "FAIL" else "UNKNOWN", reason="VALUATION_PREMISE_NOT_VALID")
        return grade
    if expected["gate_checks"].get("BEAR_LOSS_LTE_MAXIMUM") is not True:
        grade.update(state="BLOCKED", reason="VALUATION_BEAR_LOSS_EXCEEDED")
        return grade
    if expected["status"] == "PASS":
        grade.update(state="ALLOWED", band="PASS", multiplier=1., reason="VALUATION_NATIVE_PASS")
        return grade
    if (expected["status"] != "FAIL" or not set(expected["reasons"]) <= {
            "BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE"}):
        grade.update(state="BLOCKED", reason="VALUATION_NON_PRICE_FAILURE")
        return grade
    price = Decimal(str(expected["inputs"]["current_price"]))
    fair = Decimal(str(expected["scenarios"]["BASE"]["fair_price"]))
    ratio = (price / fair).quantize(Decimal(".000000000001"), rounding=ROUND_HALF_UP)
    band = next(row for row in policy()["valuation_bands"] if row["maximum_ratio"] is None
                or ratio <= Decimal(str(row["maximum_ratio"])))
    grade.update(state="ALLOWED_WITH_WARNING", band=band["name"], points=band["points"],
        multiplier=band["multiplier"], small_only=band["name"] == "HIGH",
        price_ratio=float(ratio), reason="VALUATION_PRICE_WARNING")
    return grade


def assess(quality, gates, valuation_receipt, *, symbol, cutoff, current_price):
    return _assess(quality, gates, valuation_receipt, symbol=symbol, cutoff=cutoff, current_price=current_price,
        admission_policy=policy(), valuation_result=valuation_grade(valuation_receipt, current_price=current_price))


def _assess(quality, gates, valuation_receipt, *, symbol, cutoff, current_price, admission_policy, valuation_result):
    """Shared gate/cap arithmetic; caller must bind the declared policy family."""
    q, v = quality_tier(quality), valuation_result
    states = {g.gate_id: g.state for g in gates}
    required = {"DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK"}
    complete = len(gates) == 5 and set(states) == required
    hard = {g: state for g, state in states.items() if g != "VALUATION"}
    if not complete or any(state == "UNKNOWN" for state in hard.values()) or q["state"] == "UNKNOWN" or v["state"] == "UNKNOWN":
        state = "UNKNOWN"
    else: state = "ALLOWED"
    if (any(s == "FAIL" for s in hard.values()) or q["state"] in {"REJECT", "BELOW_MINIMUM"}
            or v["state"] == "BLOCKED"):
        state = "BLOCKED"
    # A valuation result must be exactly the native receipt referenced by its gate.
    if not isinstance(valuation_receipt, Mapping) or states.get("VALUATION") != valuation_receipt.get("state"):
        state = "UNKNOWN" if state != "BLOCKED" else state
    if v.get("admission_policy_id") not in (None, identity(admission_policy)):
        state = "UNKNOWN" if state != "BLOCKED" else state
    record = dict(schema=SCHEMA, policy_id=identity(admission_policy), symbol=symbol, cutoff=cutoff,
        quality_result_id=identity(quality), gate_receipts_id=identity([asdict(g) for g in gates]),
        valuation_receipt_id=identity(valuation_receipt), quality=q, valuation=v, state=state,
        buy_allowed=state == "ALLOWED" and q["buy_cap"] > 0 and not v["small_only"],
        small_allowed=state == "ALLOWED" and q["small_cap"] > 0,
        buy_total_cap=q["buy_cap"] * v["multiplier"],
        small_total_cap=q["small_cap"] * v["multiplier"])
    return dict(record, assessment_id=identity(record))


def is_r9(report):
    """Compatibility name for the unchanged R9 admission used by R9--R11."""
    from domain.les_v7.rule_baseline import GRADED_POLICIES
    return (report.get("diagnostics", {}).get("rule_baseline") or {}).get("policy", {}).get(
        "policy_version") in GRADED_POLICIES


def cap_for(admission, signal):
    return admission["buy_total_cap"] if signal == "BUY" else admission["small_total_cap"] if signal == "BUY_SMALL" else 0.


def quality_readiness(quality, admission):
    tier = admission["quality"]
    ready = tier["evidence_ready"] and tier["small_cap"] > 0
    return dict(state=("READY" if quality["status"] == "PASS" else "READY_WITH_GRADED_LIMIT") if ready else "BLOCKED",
        status=quality["status"], reason_codes=quality["reason_codes"],
        tier=tier["tier"], entry_admission_id=admission["assessment_id"])


@dataclass(frozen=True)
class R9PortfolioSecurityV2(PortfolioSecurityV2):
    """Explicit extension; old dataclass representations and hashes stay intact."""
    entry_admission_json: str = ""

    def __post_init__(self):
        super().__post_init__()
        row = json.loads(self.entry_admission_json)
        from domain.les_v7.entry_policy_r13 import policy as revised_policy
        from domain.les_v7.entry_policy_r14 import policy as adaptive_policy
        from domain.les_v7.entry_policy_r15 import policy as tactical_policy
        from domain.les_v7.entry_policy_r16 import policy as conditional_policy
        from domain.les_v7.entry_policy_r17 import policy as unified_policy
        from domain.les_v7.entry_policy_r18 import policy as limit_policy
        if (row.get("schema") != SCHEMA or row.get("policy_id") not in {identity(policy()), identity(revised_policy()), identity(adaptive_policy()), identity(tactical_policy()), identity(conditional_policy()), identity(unified_policy()), identity(limit_policy())}
                or row.get("assessment_id") != identity({k: v for k, v in row.items() if k != "assessment_id"})
                or row.get("symbol") != self.stock_id
                or row["quality"]["native_status"] != self.quality_status
                or row["quality"]["score"] != self.quality_score
                or row["valuation"]["native_state"] != self.valuation_gate):
            raise ValueError("PAE_R9_ADMISSION_BINDING_MISMATCH")


def pae_admission(security):
    return json.loads(security.entry_admission_json) if isinstance(security, R9PortfolioSecurityV2) else None
