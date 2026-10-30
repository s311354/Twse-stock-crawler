"""FR_EXEC_R1: research-only, account-bound allocation and open-price checks.

The predictor and original H20 plan are inputs, never edited here. A receipt
is a reproducible assessment of supplied evidence, not source certification,
a broker authorization, a fill, or a modification to an account ledger.
Existing-position risk inputs are reviewed caller assertions with exact hash,
source, quantity, cost and cutoff bindings; a ``verified=True`` flag is not a
substitute. Missing evidence leaves allocation UNKNOWN without changing LES.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_FLOOR
import json
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

from domain.les_v7.contracts import canonical, finite, identity, timestamp

VERSION = "FR_EXEC_R1"
SCHEMA = "les-v7-fr2-execution/v1"
REQUIRED_GATES = frozenset({"SAFETY", "DATA_INTEGRITY", "TRADING_STATUS", "VALUATION", "STRESS_RISK"})
TAIPEI = ZoneInfo("Asia/Taipei")


def component():
    """Return a detached, stable component identity, not a mutable global."""
    body = dict(schema="les-v7-fr2-execution-component/v1", version=VERSION,
        total_symbol_caps=dict(BUY=.05, BUY_SMALL=.01, VALUATION_WARNING=.005),
        planned_nav_risk=dict(BUY=.0025, BUY_SMALL=.0005), portfolio_planned_risk=.01,
        maximum_holdings=4, nav_reconciliation_tolerance=.01,
        maximum_order_adv=.01, maximum_spread=.005,
        entry_rule="ORIGINAL_H20_NEXT_OFFICIAL_SESSION_OPEN_IN_BAND",
        sizing="TOTAL_CAP_LESS_CURRENT_EXPOSURE_NO_REPEAT_SMALL_MULTIPLIER",
        risk="MARK_TO_VERIFIED_STOP_PLUS_BOUND_COSTS_NOT_GAP_LOSS_GUARANTEE",
        authority="RESEARCH_ONLY_NEVER_LIVE_ORDER")
    return dict(body, component_id=identity(body))


def _id(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("FR2_INVALID_" + name)
    return value


def _number(value, name, minimum=0, maximum=None):
    return Decimal(str(finite(value, "FR2_" + name, minimum, maximum)))


def _position(position, *, cutoff):
    if not isinstance(position, Mapping):
        raise ValueError("FR2_POSITION_REQUIRED")
    symbol = position.get("symbol")
    if not isinstance(symbol, str) or not symbol or symbol.strip() != symbol:
        raise ValueError("FR2_POSITION_SYMBOL_REQUIRED")
    shares = position.get("shares")
    if type(shares) is not int or shares <= 0:
        raise ValueError("FR2_POSITIVE_INTEGER_POSITION_SHARES_REQUIRED")
    result = dict(symbol=symbol, shares=shares, risk_evidence=None, risk_evidence_id=None)
    proof = position.get("risk_evidence")
    if proof is None:
        return result
    proof = json.loads(canonical(proof))
    if (proof.get("schema") != "les-v7-fr2-position-risk/v1"
            or proof.get("symbol") != symbol or proof.get("shares") != shares
            or proof.get("market") not in {"TWSE", "TPEX"}
            or proof.get("currency") != "TWD" or proof.get("price_unit") != "TWD"
            or proof.get("quantity_unit") != "shares"
            or timestamp(proof["as_of"]) != cutoff
            or timestamp(proof["source_available_at"]) > cutoff
            or not proof.get("reviewed_by")
            or not isinstance(proof.get("source_evidence_ids"), list) or not proof["source_evidence_ids"]
            or any(not isinstance(v, str) or not v for v in proof["source_evidence_ids"])):
        raise ValueError("FR2_POSITION_RISK_EVIDENCE_BINDING_INVALID")
    _id(proof.get("cost_policy_id"), "POSITION_COST_POLICY_ID")
    claimed = position.get("risk_evidence_id")
    if claimed != identity(proof):
        raise ValueError("FR2_POSITION_RISK_EVIDENCE_HASH_MISMATCH")
    mark = _number(proof.get("mark_price"), "mark_price")
    stop = _number(proof.get("stop_price"), "stop_price")
    _number(proof.get("cost_stop_per_share"), "cost_stop_per_share")
    if not 0 < stop < mark:
        raise ValueError("FR2_POSITION_STOP_INVALID")
    result.update(risk_evidence=proof, risk_evidence_id=claimed)
    return result


def _account(account, *, cutoff):
    if not isinstance(account, Mapping):
        return None
    result = dict(account)
    for key in ("value", "available_cash", "intended_weight", "capacity_weight"):
        if result.get(key) is None:
            return None
        _number(result[key], key, maximum=1 if key.endswith("weight") else None)
    if (result["value"] <= 0 or result["available_cash"] > result["value"]
            or timestamp(result["as_of"]) != cutoff):
        raise ValueError("FR2_INVALID_ACCOUNT_OR_CUTOFF")
    if not isinstance(result.get("positions"), list):
        return None
    rows = [_position(row, cutoff=cutoff) for row in result["positions"]]
    if len(rows) > 4 or len({row["symbol"] for row in rows}) != len(rows):
        raise ValueError("FR2_INVALID_OR_DUPLICATE_POSITIONS")
    # Retain the public replay account's empty-cash/NAV rule. When all held
    # marks are evidenced, also reconcile NAV rather than allowing inflated
    # unexplained equity to enlarge risk budgets. No other assets are inferred.
    if not rows or all(row["risk_evidence"] is not None for row in rows):
        marked_assets = sum((Decimal(row["shares"]) * _number(row["risk_evidence"]["mark_price"], "mark_price")
            for row in rows), Decimal(0))
        if abs(_number(result["available_cash"], "available_cash") + marked_assets
                - _number(result["value"], "value")) > Decimal(".01"):
            raise ValueError("FR2_ACCOUNT_NAV_CASH_HOLDING_MARKS_MISMATCH")
    return dict(value=float(result["value"]), available_cash=float(result["available_cash"]),
        intended_weight=float(result["intended_weight"]), capacity_weight=float(result["capacity_weight"]),
        as_of=cutoff.isoformat(), positions=sorted(rows, key=lambda row: row["symbol"]))


def _reference_policy(report):
    strategy = report.get("diagnostics", {}).get("strategy_policy", {})
    ref = strategy.get("payload", strategy.get("policy", strategy)).get("reference_policy", {})
    if not ref:
        ref = report.get("diagnostics", {}).get("reference_policy", {})
    return ref


def _cost_policy(report):
    return _reference_policy(report).get("cost_path_policy")


def _validate_native(report):
    from domain.les_v7.report import validate_report
    validate_report({key: value for key, value in report.items() if key != "fr2_execution"})


def assess_execution(report, admission, account, *, quote=None, validate_native=True):
    """Build one detached assessment; quote omission never simulates a fill.

    ``validate_native=False`` is reserved for the native report validator after
    its complete source/model/admission checks, avoiding recursive validation.
    Other public callers retain the default full native validation.
    """
    if validate_native:
        _validate_native(report)
    return _assess_validated(report, admission, account, quote=quote)


def _assess_validated(report, admission, account, *, quote=None):
    envelope, diagnostics = report["envelope"], report["diagnostics"]
    event = envelope["event"]
    cutoff = timestamp(event["cutoff_at"])
    admission = json.loads(canonical(admission))
    if (admission.get("version", admission.get("policy_version")) != "FR_ENTRY_R2"
            or admission.get("assessment_id") != identity({key: value for key, value in admission.items()
                if key != "assessment_id"})):
        raise ValueError("FR2_ADMISSION_REQUIRED")
    bound_admission = (diagnostics.get("fr2") or {}).get("admission")
    if (canonical(bound_admission) != canonical(admission)
            or admission.get("decision") != envelope["strategy"]["strategy_decision"]
            or "cutoff" in admission and timestamp(admission["cutoff"]) != cutoff
            or "evidence_id" in admission and admission["evidence_id"] != event["evidence_id"]
            or "bundle_sha256" in admission and admission["bundle_sha256"] != envelope.get("model", {}).get("bundle_sha256")
            or "quality_result_id" in admission and admission["quality_result_id"] != envelope.get("quality", {}).get("quality_result_id")):
        raise ValueError("FR2_EXECUTION_ADMISSION_BINDING_MISMATCH")
    signal = admission.get("decision")
    warning = admission.get("valuation_warning") is True
    gates = diagnostics["gate_receipts"]
    states = {row["gate_id"]: row["state"] for row in gates}
    reasons = []
    unknown = False
    if signal not in {"BUY", "BUY_SMALL"}:
        reasons.append("SIGNAL_NOT_EXECUTABLE")
    elif admission.get("buy_allowed" if signal == "BUY" else "small_allowed") is not True:
        reasons.append("ADMISSION_NOT_ALLOWED")
    if (len(gates) != len(REQUIRED_GATES) or set(states) != REQUIRED_GATES
            or any(state != "PASS" for gate, state in states.items() if gate != "VALUATION")
            or states.get("VALUATION") != "PASS" and not (warning and states.get("VALUATION") == "FAIL")):
        reasons.append("HARD_GATES_NOT_PASS")
        unknown |= set(states) != REQUIRED_GATES or any(state in {"UNKNOWN", "NOT_APPLICABLE"} for state in states.values())
    if warning and signal != "BUY_SMALL":
        reasons.append("VALUATION_WARNING_SMALL_ONLY")
    if diagnostics["quality"].get("status") != "PASS":
        reasons.append("QUALITY_NOT_PASS")
        unknown |= diagnostics["quality"].get("status") == "UNKNOWN"
    if diagnostics.get("daily_input_errors"):
        reasons.append("DAILY_INPUT_INVALID")
    try:
        snapshot = _account(account, cutoff=cutoff)
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        snapshot = None
        reasons.append("ACCOUNT_INVALID:" + str(error))
    if snapshot is None:
        reasons.append("ACCOUNT_INPUTS_UNKNOWN")
        unknown = True
    elif any(row["risk_evidence"] is None for row in snapshot["positions"]):
        reasons.append("EXISTING_POSITION_RISK_UNKNOWN")
        unknown = True
    if snapshot is not None and len(snapshot["positions"]) == 4 and event["security_id"] not in {
            row["symbol"] for row in snapshot["positions"]}:
        reasons.append("PORTFOLIO_FOUR_POSITIONS_NO_NEW_SYMBOL_SLOT")

    plan, features, path_policy = diagnostics.get("reference_plan"), diagnostics["features"], _cost_policy(report)
    parsed_plan = None
    try:
        from domain.les_v7.plan_r19 import plan_from_record
        from domain.les_v7.reference_path import validate_reference_path_policy
        parsed_plan = plan_from_record(plan)
        if parsed_plan is None:
            raise ValueError("ORIGINAL_H20_PLAN_MISSING")
        if parsed_plan.plan_id != event["plan_id"] or not parsed_plan.target:
            raise ValueError("ORIGINAL_H20_PLAN_BINDING_INVALID")
        path_policy = validate_reference_path_policy(path_policy)
        if path_policy["market"] != diagnostics["gate_bindings"]["SAFETY"]["market"]:
            raise ValueError("PLAN_COST_MARKET_MISMATCH")
        if not (path_policy["cost_policy"]["effective_start"] <= report["signal_date"]
                <= path_policy["cost_policy"]["effective_end"]):
            raise ValueError("PLAN_COSTS_NOT_EFFECTIVE_FOR_SIGNAL_SESSION")
    except (ValueError, TypeError, KeyError) as error:
        reasons.append("PLAN_OR_BOUND_COSTS_INVALID:" + str(error))
        unknown = True
    try:
        adv = _number(features.get("adv"), "adv")
        spread = _number(features.get("spread"), "spread", maximum=.005)
        if adv <= 0:
            raise ValueError("ADV_MUST_BE_POSITIVE")
    except (ValueError, TypeError) as error:
        reasons.append("LIQUIDITY_UNKNOWN_OR_FAILED:" + str(error))
        unknown |= features.get("adv") is None or features.get("spread") is None
        adv, spread = Decimal(0), None

    shares = 0
    total_cap = current_weight = current_risk = portfolio_risk = incremental_weight = None
    sizing_entry = stop_risk = entry_cash = None
    if not reasons:
        from domain.les_v7.reference_path import estimate_reference_costs, _round_price
        equity = _number(snapshot["value"], "value")
        cash = _number(snapshot["available_cash"], "available_cash")
        # Original H20 bands can be off-grid. Budget their highest legal
        # executable price; do not round the original band/anchor outward.
        entry = _round_price(_number(plan["band"][1], "entry_band_high"), "down")
        mark = _number(features["close"], "close")
        exposure = max(entry, mark)
        stop = _number(plan["stop"], "stop")
        if entry < _number(plan["band"][0], "entry_band_low") or not 0 < stop < entry:
            reasons.append("INVALID_STOP_AT_ENTRY_BAND_LIMIT")
        else:
            cap = Decimal(".005") if warning else Decimal(".05") if signal == "BUY" else Decimal(".01")
            total_cap = min(cap, _number(snapshot["intended_weight"], "intended_weight"),
                _number(snapshot["capacity_weight"], "capacity_weight"))
            budget = equity * (Decimal(".0025") if signal == "BUY" else Decimal(".0005"))
            held_exposure = held_risk = all_risk = Decimal(0)
            for position in snapshot["positions"]:
                proof = position["risk_evidence"]
                quantity = Decimal(position["shares"])
                position_mark = _number(proof["mark_price"], "position_mark")
                risk = quantity * (position_mark - _number(proof["stop_price"], "position_stop")
                    + _number(proof["cost_stop_per_share"], "position_cost"))
                all_risk += risk
                if position["symbol"] == event["security_id"]:
                    held_exposure += quantity * max(mark, position_mark)
                    held_risk += risk
            remaining = max(Decimal(0), equity * total_cap - held_exposure)
            loss_budget = max(Decimal(0), min(budget - held_risk, equity * Decimal(".01") - all_risk))
            upper = max(0, int(min(remaining / exposure, cash / entry,
                adv * Decimal(".01") / entry).to_integral_value(rounding=ROUND_FLOOR)))

            def amounts(quantity):
                costs = estimate_reference_costs(float(entry), float(stop), plan["target"], quantity, path_policy)
                return (entry * quantity + Decimal(costs["detail"]["buy_fee"]),
                        (entry - stop + Decimal(costs["detail"]["cost_stop_per_share"])) * quantity)

            # Cent rounding and minimum fees are recomputed for the actual
            # quantity, not copied from a one-share prediction reference.
            low, high = 0, upper
            while low < high:
                mid = (low + high + 1) // 2
                debit, loss = amounts(mid)
                if debit <= cash and debit <= remaining and loss <= loss_budget:
                    low = mid
                else:
                    high = mid - 1
            shares = low
            if shares:
                entry_cash, stop_risk = map(float, amounts(shares))
            else:
                reasons.append("NO_CASH_OR_RISK_CAPACITY")
            current_weight, current_risk, portfolio_risk, incremental_weight = map(float,
                (held_exposure / equity, held_risk / equity, all_risk / equity, remaining / equity))
            total_cap, sizing_entry = float(total_cap), float(entry)

    quote_result = _quote_check(report, quote, shares=shares, signal=signal, path_policy=path_policy) if quote is not None else dict(state="NOT_PROVIDED", eligible=False,
        reason_codes=["NEXT_SESSION_PRICE_CHECK_REQUIRED"])
    result = dict(schema=SCHEMA, version=VERSION, component_id=component()["component_id"],
        scope="RESEARCH_INDICATIVE_ALLOCATION_NOT_FILL", signal=signal, admission_id=admission["assessment_id"],
        signal_id=event["signal_id"], evidence_id=event["evidence_id"], plan_id=event["plan_id"],
        cutoff_at=cutoff.isoformat(), envelope_id=identity(envelope), gate_evidence_id=identity(gates),
        admission=admission, account=deepcopy(snapshot), account_id=identity(snapshot),
        supplied_account=deepcopy(account),
        supplied_account_id=identity(account), cost_path_policy_id=identity(path_policy),
        maximum_total_weight=total_cap, current_weight=current_weight, incremental_weight_capacity=incremental_weight,
        existing_symbol_risk_fraction=current_risk, existing_portfolio_risk_fraction=portfolio_risk,
        indicative_shares=shares, sizing_entry_limit=sizing_entry, planned_cash_debit=entry_cash,
        planned_stop_loss=stop_risk, quote=deepcopy(quote), quote_assessment=quote_result,
        state="UNKNOWN" if unknown else "PASS" if not reasons else "FAIL",
        allocation_authorized=not reasons and shares > 0,
        price_condition_satisfied=quote_result["eligible"],
        fill_eligible=not reasons and shares > 0 and quote_result["eligible"],
        live_order_authorized=False, filled_shares=0,
        reason_codes=reasons or ["RESEARCH_ALLOCATION_VALID_NEXT_SESSION_PRICE_STILL_REQUIRED"])
    return dict(result, assessment_id=identity(result))


def _quote_check(report, quote, *, shares, signal, path_policy):
    """Verify the retained official-open assertion; never infer intraday fills."""
    result = dict(state="UNKNOWN", eligible=False, reason_codes=[])
    try:
        from domain.evidence_publication import calendar_at, verified_document, verify_publication
        from domain.trading_calendar import OfficialAnnualCalendar
        from domain.les_v7.session_coverage_r16 import MARKET_HOSTS
        event, diagnostics = report["envelope"]["event"], report["diagnostics"]
        market = diagnostics["gate_bindings"]["SAFETY"]["market"]
        payload, proof = quote["payload"], quote["publication_proof"]
        if urlparse(proof["document"]["url"]).hostname not in MARKET_HOSTS[market]:
            raise ValueError("QUOTE_SOURCE_MARKET_MISMATCH")
        at = timestamp(quote["checked_at"])
        session = date.fromisoformat(payload["session"])
        cutoff = timestamp(event["cutoff_at"])
        opened = datetime.combine(session, time(9), TAIPEI)
        binding = dict(kind="FR2_OFFICIAL_OPEN_QUOTE", symbol=event["security_id"], market=market,
            session=session.isoformat(), content_sha256=identity(payload))
        # The reviewed source document must itself retain this exact payload;
        # a pass flag, arbitrary row hash, or unaudited extraction is not proof.
        body = verified_document(proof["document"])
        if (json.loads(body) != payload or payload.get("symbol") != event["security_id"]
                or payload.get("market") != market or payload.get("phase") != "OFFICIAL_OPEN"
                or payload.get("price_unit") != "TWD" or payload.get("volume_unit") != "shares"
                or timestamp(payload["observed_at"]) != opened
                or at < opened or cutoff >= opened
                or timestamp(verify_publication(proof, binding=binding)) > at
                or timestamp(proof["document"]["retrieved_at"]) > at):
            raise ValueError("OFFICIAL_OPEN_QUOTE_BINDING_INVALID")
        calendar = deepcopy(quote["calendar_evidence"])
        first = date.fromisoformat(report["signal_date"])
        if not first < session or (session - first).days > 31 or calendar.get("source", "").split("_")[0] != market:
            raise ValueError("QUOTE_CALENDAR_SCOPE_INVALID")
        calendar["days"] = {(first + timedelta(days=i)).isoformat(): "UNKNOWN"
            for i in range((session - first).days + 1)}
        def verify_calendar(source_proof, *, binding):
            if urlparse(source_proof["document"]["url"]).hostname not in MARKET_HOSTS[market]:
                raise ValueError("QUOTE_CALENDAR_SOURCE_MARKET_MISMATCH")
            return verify_publication(source_proof, binding=binding)

        days, excluded = calendar_at(calendar, at, strict=True, annual_calendar=OfficialAnnualCalendar,
            verifier=verify_calendar)
        if excluded or "UNKNOWN" in days.values() or days[first.isoformat()] != "OPEN":
            raise ValueError("QUOTE_OFFICIAL_SESSION_SEQUENCE_UNKNOWN")
        following = [day for day, state in sorted(days.items()) if day > first.isoformat() and state == "OPEN"]
        if not following or following[0] != session.isoformat():
            raise ValueError("QUOTE_NOT_NEXT_OFFICIAL_SESSION")
        price = finite(payload.get("price"), "quote_price", 0)
        bid, ask = finite(payload.get("bid"), "bid", 0), finite(payload.get("ask"), "ask", 0)
        if not 0 < bid <= ask or price <= 0 or (ask - bid) / price > .005:
            raise ValueError("QUOTE_SPREAD_INVALID")
        plan = diagnostics["reference_plan"]
        if plan is None:
            raise ValueError("ORIGINAL_H20_PLAN_MISSING")
        if not plan["band"][0] <= price <= plan["band"][1]:
            return dict(state="FAIL", eligible=False, reason_codes=["OFFICIAL_OPEN_OUTSIDE_ORIGINAL_ENTRY_BAND"])
        if price <= plan["stop"] or plan.get("target") is None or price >= plan["target"]:
            return dict(state="FAIL", eligible=False, reason_codes=["OFFICIAL_OPEN_INVALIDATES_STOP_OR_TARGET"])
        if not (path_policy["cost_policy"]["effective_start"] <= session.isoformat()
                <= path_policy["cost_policy"]["effective_end"]):
            raise ValueError("BOUND_COSTS_NOT_EFFECTIVE_FOR_QUOTE_SESSION")
        if shares <= 0:
            raise ValueError("EXPLICIT_ACCOUNT_ALLOCATION_REQUIRED_FOR_PRICE_CHECK")
        # Retain, rather than relax, the trained model's existing distance/RR
        # thresholds. Actual-size fees can invalidate an otherwise valid plan.
        from domain.les_v7.reference_path import estimate_reference_costs
        estimated = estimate_reference_costs(price, plan["stop"], plan["target"], shares, path_policy)
        risk = price - plan["stop"] + estimated["cost_stop_per_share"]
        rr = (plan["target"] - price - estimated["cost_up_per_share"]) / risk
        stop_atr = (price - plan["stop"]) / plan["atr"]
        distance = 1.5 if signal == "BUY" else 1.2 if event["context"] == "REVERSAL" else 1.3
        minimum_rr = (2. if event["context"] == "REVERSAL" else 1.8
            if event["context"] == "TREND_PULLBACK" else
            finite(_reference_policy(report).get("leadership_rr_min"), "leadership_rr_min", 0))
        if stop_atr > distance or rr < minimum_rr:
            return dict(state="FAIL", eligible=False, reason_codes=["OFFICIAL_OPEN_RISK_OR_NET_RR_FAILED"],
                actual_size_net_rr=rr, actual_open_stop_atr=stop_atr)
        return dict(state="PASS", eligible=True, reason_codes=["RETAINED_NEXT_OFFICIAL_OPEN_PRICE_CONDITION_MET"],
            observation_scope="RETAINED_SOURCE_ASSERTION_NOT_LIVE_QUOTE_OR_ACTUAL_FILL",
            proof_id=identity(proof), calendar_id=identity(quote["calendar_evidence"]))
    except (KeyError, ValueError, TypeError, OverflowError) as error:
        result["reason_codes"] = ["QUOTE_EVIDENCE_UNKNOWN:" + str(error)]
        return result


def validate_execution_assessment(report, admission, assessment, *, validate_native=True):
    """Independently rebuild the complete assessment, including account inputs."""
    if validate_native:
        _validate_native(report)
    rebuilt = _assess_validated(report, admission, assessment.get("supplied_account"), quote=assessment.get("quote"))
    if identity(rebuilt) != identity(assessment):
        raise ValueError("FR2_EXECUTION_ASSESSMENT_BINDING_MISMATCH")
