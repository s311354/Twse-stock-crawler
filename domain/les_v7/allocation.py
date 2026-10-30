"""Account-bound PAE candidate authorization; never broker/order authorization."""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, ROUND_FLOOR
from types import MappingProxyType

from domain.les_v7.contracts import finite, identity, timestamp


PAE_AUTHORIZATION_POLICY = "LES_V7_PAE_SIGNAL_AUTHORIZATION_R1"
REQUIRED_GATES = {"SAFETY", "DATA_INTEGRITY", "TRADING_STATUS", "VALUATION", "STRESS_RISK"}
POLICY = MappingProxyType(dict(full_risk_fraction=.005, starter_multiplier=.25,
              initial_position_cap=.25, maximum_order_adv=.01, maximum_spread=.005))


def account_positions(values: object) -> list[dict] | None:
    """Normalize explicit position identities/quantities without inferring cash."""
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise ValueError("V7_PAE_POSITIONS_REQUIRED")
    rows, seen = [], set()
    for position in values:
        symbol = position.get("symbol") if isinstance(position, Mapping) else None
        if not isinstance(symbol, str) or not symbol.strip() or symbol != symbol.strip() or symbol in seen:
            raise ValueError("V7_PAE_INVALID_OR_DUPLICATE_POSITION")
        seen.add(symbol)
        shares = None if position.get("shares") is None else finite(position["shares"], "shares", 0)
        if shares is not None and (shares <= 0 or not shares.is_integer()):
            raise ValueError("V7_PAE_INVALID_POSITION_SHARES")
        rows.append(dict(symbol=symbol, shares=shares))
    return sorted(rows, key=lambda row: row["symbol"])


def authorize_candidate(report: Mapping, account: Mapping | None) -> dict:
    if "fr2" in report.get("diagnostics", {}):
        raise ValueError("FR2_USE_VERSIONED_EXECUTION_COMPONENT")
    from domain.les_v7.report import validate_report
    # Revalidate the native signal, independently of any old account quota.
    validate_report({key: value for key, value in report.items() if key != "pae_authorization"})
    return _authorize_validated(report, account)


def _authorize_validated(report: Mapping, account: Mapping | None) -> dict:
    """Derive an indicative incremental quota from an already validated signal.

    Use the existing causal plan and its bound cost assumptions, never LES as a
    win probability. PAE can reduce/reject the quota. Gap stress, actual order
    size support and next-session quotes still require execution revalidation.
    """
    envelope, diagnostics = report["envelope"], report["diagnostics"]
    if "fr2" in diagnostics:
        raise ValueError("FR2_USE_VERSIONED_EXECUTION_COMPONENT")
    event = envelope["event"]
    cutoff = timestamp(event["cutoff_at"]).isoformat()
    account = account or {}
    snapshot = {key: None if account.get(key) is None else finite(account[key], key, 0)
                for key in ("value", "available_cash", "intended_weight", "capacity_weight")}
    snapshot["positions"] = account_positions(account.get("positions"))
    snapshot["as_of"] = timestamp(account.get("as_of", cutoff)).isoformat()
    if snapshot["as_of"] != cutoff:
        raise ValueError("V7_PAE_ACCOUNT_CUTOFF_MISMATCH")
    if (snapshot["value"] is not None and snapshot["value"] <= 0
            or any(snapshot[k] is not None and snapshot[k] > 1 for k in ("intended_weight", "capacity_weight"))
            or snapshot["value"] is not None and snapshot["available_cash"] is not None
            and snapshot["available_cash"] > snapshot["value"]):
        raise ValueError("V7_PAE_INVALID_ACCOUNT")

    reasons = []
    signal = envelope["strategy"]["strategy_decision"]
    if signal not in {"BUY", "BUY_SMALL"}: reasons.append("SIGNAL_NOT_EXECUTABLE")
    gates = diagnostics["gate_receipts"]
    from domain.les_v7.entry_policy_r9 import is_r9, cap_for
    graded = is_r9(report)
    admission = diagnostics.get("entry_admission", {})
    from domain.les_v7.policy_r14 import BASELINE_VERSION as r14_version
    from domain.les_v7.policy_r15 import BASELINE_VERSION as r15_version
    from domain.les_v7.policy_r16 import BASELINE_VERSION as r16_version
    from domain.les_v7.policy_r17 import BASELINE_VERSION as r17_version
    from domain.les_v7.policy_r18 import BASELINE_VERSION as r18_version
    from domain.les_v7.policy_r19 import BASELINE_VERSION as r19_version
    r14 = (diagnostics.get("rule_baseline") or {}).get("policy", {}).get("policy_version") == r14_version
    r17 = (diagnostics.get("rule_baseline") or {}).get("policy", {}).get("policy_version") == r17_version
    r18 = (diagnostics.get("rule_baseline") or {}).get("policy", {}).get("policy_version") == r18_version
    r19 = (diagnostics.get("rule_baseline") or {}).get("policy", {}).get("policy_version") == r19_version
    r16 = (diagnostics.get("rule_baseline") or {}).get("policy", {}).get("policy_version") in {r16_version, r17_version, r18_version, r19_version}
    r15 = (diagnostics.get("rule_baseline") or {}).get("policy", {}).get("policy_version") in {r15_version, r16_version, r17_version, r18_version, r19_version}
    if graded and admission.get("buy_allowed" if signal == "BUY" else "small_allowed") is not True:
        reasons.append("R9_ENTRY_ADMISSION_NOT_ALLOWED")
    if (len(gates) != len(REQUIRED_GATES) or {g["gate_id"] for g in gates} != REQUIRED_GATES
            or any(g["state"] != "PASS" for g in gates if not graded or g["gate_id"] != "VALUATION")):
        reasons.append("V7_HARD_GATES_NOT_PASS")
    if diagnostics.get("daily_input_errors"): reasons.append("DAILY_INPUT_INVALID")
    if any(snapshot[key] is None for key in ("value", "available_cash", "intended_weight", "capacity_weight")):
        reasons.append("ACCOUNT_INPUTS_MISSING")
    if snapshot["positions"] is None or any(position["shares"] is None for position in snapshot["positions"]):
        reasons.append("ACCOUNT_POSITIONS_UNKNOWN")
    plan, features = diagnostics["reference_plan"], diagnostics["features"]
    if plan is None: reasons.append("REFERENCE_PLAN_MISSING")
    adv, spread = features.get("adv"), features.get("spread")
    if adv is None or spread is None or finite(adv, "adv", 0) <= 0 or not 0 <= finite(spread) <= POLICY["maximum_spread"]:
        reasons.append("LIQUIDITY_NOT_PASS")

    shares, full_shares, fraction = 0, 0, 0.
    entry_limit = risk_per_share = None
    maximum_total_weight = current_weight = incremental_capacity = None
    exposure_price = held_risk_per_share = None
    if not reasons:
        number = lambda value: Decimal(str(value))
        equity, cash, intended, capacity = (number(snapshot[key]) for key in
            ("value", "available_cash", "intended_weight", "capacity_weight"))
        # Budget the worst allowed band price and all bound costs conservatively.
        entry = number(max(plan["entry"], plan["band"][1]))
        cost_up, cost_stop = plan["cost_up"], plan["cost_stop"]
        if r14 or r15:
            from domain.les_v7.reference_path import estimate_reference_costs
            bound = diagnostics["rule_baseline"]["policy"]["reference_policy"]
            repriced = estimate_reference_costs(float(entry), plan["stop"], plan["target"],
                plan["reference_shares"], bound["cost_path_policy"])
            cost_up, cost_stop = repriced["cost_up_per_share"], repriced["cost_stop_per_share"]
        entry_cost = entry + number(cost_up)
        loss = entry - number(plan["stop"]) + number(cost_stop)
        if loss <= 0 or entry_cost <= 0: raise ValueError("V7_PAE_INVALID_PLAN_RISK")
        exposure_unit, held_loss = entry_cost, loss
        if r16:
            mark = number(plan["signal_close"])
            # A lower conditional price cannot buy more mark-to-market
            # exposure than the declared total cap. Held shares also consume
            # their true-close-to-fixed-stop risk, not a cheaper future entry.
            exposure_unit = max(entry_cost, mark)
            held_entry = max(entry, mark)
            held_cost = estimate_reference_costs(float(held_entry), plan["stop"], plan["target"],
                plan["reference_shares"], bound["cost_path_policy"])
            held_loss = held_entry - number(plan["stop"]) + number(held_cost["cost_stop_per_share"])
            exposure_price, held_risk_per_share = float(exposure_unit), float(held_loss)
        if graded:
            total = min(number(cap_for(admission, signal)), intended, capacity)
            held = sum(number(p["shares"]) for p in snapshot["positions"] if p["symbol"] == event["security_id"])
            current = held * number(plan.get("signal_close", plan["entry"])) / equity
            remaining = max(Decimal(0), total - current)
            # Approved caps are already signal-sized. Retain the old SMALL
            # risk budget, but never quarter the cash/liquidity/weight cap again.
            def quota(risk_fraction):
                return max(0, int(min(max(Decimal(0), equity * risk_fraction - held * held_loss) / loss,
                    cash / entry_cost, equity * remaining / exposure_unit,
                    number(adv) * number(POLICY["maximum_order_adv"]) / entry).to_integral_value(rounding=ROUND_FLOOR)))
            full_risk = number(POLICY["full_risk_fraction"])
            full_shares = quota(full_risk)
            small_risk = number(admission["tactical_risk_fraction"]) if r15 else full_risk * number(POLICY["starter_multiplier"])
            shares = quota(small_risk) if signal == "BUY_SMALL" else full_shares
            maximum_total_weight, current_weight, incremental_capacity = map(float, (total, current, remaining))
        else:
            full_shares = int(min(equity * number(POLICY["full_risk_fraction"]) / loss,
                cash / entry_cost, equity * min(number(POLICY["initial_position_cap"]), intended, capacity) / entry_cost,
                number(adv) * number(POLICY["maximum_order_adv"]) / entry).to_integral_value(rounding=ROUND_FLOOR))
            shares = (int((number(full_shares) * number(POLICY["starter_multiplier"])).to_integral_value(rounding=ROUND_FLOOR))
                      if signal == "BUY_SMALL" else full_shares)
        # PAE weights are valued at the signal close, so that a weight cannot
        # imply more shares than this quota merely because the band is higher.
        fraction = float(number(shares) * number(plan.get("signal_close", plan["entry"])) / equity)
        entry_limit, risk_per_share = float(entry), float(loss)
        if shares <= 0: reasons.append("NO_CASH_OR_RISK_CAPACITY")
    result = dict(policy_version=PAE_AUTHORIZATION_POLICY, policy=POLICY.copy(),
        scope="PAE_CANDIDATE", signal_id=event["signal_id"], plan_id=event["plan_id"],
        signal_sha256=identity(envelope), gate_evidence_sha256=identity(gates),
        account=snapshot, account_sha256=identity(snapshot),
        authorized=not reasons and shares > 0, full_shares=full_shares, indicative_shares=shares,
        position_fraction=fraction, sizing_entry_limit=entry_limit, bound_plan_loss_per_share=risk_per_share,
        risk_basis="BOUND_REFERENCE_PLAN_COSTS_AT_ENTRY_BAND_LIMIT",
        requires_next_session_execution_check=True, live_order_authorized=False,
        reason_codes=reasons or ["V7_SIGNAL_AUTHORIZED_FOR_PAE"])
    if graded:
        result.update(policy_version="LES_V7_PAE_SIGNAL_AUTHORIZATION_R2",
            entry_admission_id=admission["assessment_id"],
            maximum_total_weight=maximum_total_weight, current_weight=current_weight,
            incremental_weight_capacity=incremental_capacity,
            position_cap_basis="R9_TOTAL_CAP_LESS_CURRENT_HOLDING",
            small_sizing="DIRECT_FINAL_CAP_NO_REPEAT_MULTIPLIER",
            small_risk_fraction=POLICY["full_risk_fraction"] * POLICY["starter_multiplier"])
        if r14:
            result.update(policy_version="LES_V7_PAE_SIGNAL_AUTHORIZATION_R3",
                risk_basis="R14_REPRICED_BOUND_COSTS_AT_ENTRY_BAND_LIMIT",
                position_cap_basis="R14_TOTAL_CAP_LESS_HELD_EXPOSURE_AND_RISK")
        if r15:
            result.update(policy_version="LES_V7_PAE_SIGNAL_AUTHORIZATION_R4",
                risk_basis="R15_REPRICED_BOUND_COSTS_AT_ENTRY_BAND_LIMIT",
                position_cap_basis="R15_MIN_TOTAL_CAPS_LESS_HELD_EXPOSURE_AND_RISK",
                small_risk_fraction=admission["tactical_risk_fraction"],
                reference_entry_route_id=(diagnostics.get("reference_entry_route") or {}).get("route_id"))
        if r16:
            result.update(policy_version="LES_V7_PAE_SIGNAL_AUTHORIZATION_R5",
                risk_basis="R16_REPRICED_BOUND_COSTS_AT_ENTRY_BAND_LIMIT",
                position_cap_basis="R16_MIN_TOTAL_CAPS_LESS_SIGNAL_CLOSE_EXPOSURE_AND_PLAN_RISK",
                conditional_entry_basis=(plan or {}).get("entry_basis"),
                exposure_price=exposure_price, bound_held_loss_per_share=held_risk_per_share,
                execution_scope="NEXT_VERIFIED_OFFICIAL_SESSION_OPEN_IN_BAND_ONLY")
            if r17:
                result.update(policy_version="LES_V7_PAE_SIGNAL_AUTHORIZATION_R6",
                    risk_basis="R17_REPRICED_BOUND_COSTS_AT_ENTRY_BAND_LIMIT",
                    position_cap_basis="R17_MIN_TOTAL_CAPS_LESS_SIGNAL_CLOSE_EXPOSURE_AND_PLAN_RISK")
            if r18:
                result.update(policy_version="LES_V7_PAE_SIGNAL_AUTHORIZATION_R7",
                    risk_basis="R18_REPRICED_BOUND_COSTS_AT_ENTRY_BAND_LIMIT",
                    position_cap_basis="R18_MIN_TOTAL_CAPS_LESS_SIGNAL_CLOSE_EXPOSURE_AND_PLAN_RISK")
            if r19:
                result.update(policy_version="LES_V7_PAE_SIGNAL_AUTHORIZATION_R8",
                    risk_basis="R19_REPRICED_BOUND_COSTS_AT_ENTRY_BAND_LIMIT",
                    position_cap_basis="R19_MIN_TOTAL_CAPS_LESS_SIGNAL_CLOSE_EXPOSURE_AND_PLAN_RISK")
    if envelope["model"].get("model_status") == "RULE_BASED_BASELINE":
        # Old quota arithmetic is retained; R9 has explicit total-position caps.
        # Bind the separately
        # validated, uncalibrated rule policy without inventing Kelly odds or
        # describing its signal as a fitted-model prediction. Old reports keep
        # their exact authorization body and content identity.
        baseline = diagnostics.get("rule_baseline", {})
        if (not isinstance(baseline.get("policy_id"), str) or not baseline["policy_id"]
                or envelope["model"].get("bundle_sha256") is not None
                or envelope["model"].get("trained_artifacts_status") != "NOT_APPLICABLE"
                or envelope["localization"].get("status") != "NOT_APPLICABLE"
                or any(envelope["localization"].get(key) is not None for key in
                       ("bottom_price_quantiles", "timing_pmf", "p_fill", "p_rebound_given_fill",
                        "p_entry_given_fill", "p_bottom_soon"))):
            raise ValueError("V7_BASELINE_AUTHORIZATION_POLICY_BINDING_INVALID")
        result["signal_basis"] = dict(mode="RULE_BASED_BASELINE", rule_policy_id=baseline["policy_id"],
            model_version=report["model_version"], parameter_version=report["parameter_version"],
            calibration_status="NOT_APPLICABLE", order_authority=False)
    return dict(result, authorization_id=identity(result))


def validate_authorization(report: Mapping) -> None:
    record = report.get("pae_authorization")
    if record is not None and record != _authorize_validated(report, record["account"]):
        raise ValueError("V7_PAE_AUTHORIZATION_BINDING_MISMATCH")
