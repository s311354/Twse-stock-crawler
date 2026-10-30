"""Deterministic H20 daily-bar reference paths, not a broker or account engine.

Only the observed next official session open can fill. Daily ranges cannot
prove which of two opposing barriers was crossed first. Range-hit timestamps
are *close bounds*, with explicit interval/precision receipts, never invented
intraday observations. No-day-trade exits start in the session after entry.
The policy, costs and raw price-coordinate coverage are explicit research
inputs; this module does not certify official source authenticity.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, localcontext
from datetime import date
from zoneinfo import ZoneInfo

from domain.les_v7.contracts import ReferencePlan, identity, timestamp

PATH_SCHEMA = "les-v7-reference-path-policy/v1"
PATH_VERSION = "H20_OPEN_ONLY_NO_DAY_TRADE_COST_AFTER_R1"
PRECISION = "SESSION_RANGE_BOUND_NOT_INTRADAY"
COORDINATE_POLICY = "RAW_UNADJUSTED_NO_ACTIONS_COVERED"
FINAL_COORDINATE_POLICY = "OFFICIAL_FINAL_ACTIONS_RECONSTRUCTION_R1"
GRID_VERSION = "ORDINARY_STOCK_TICK_R1"
_GRID = tuple((Decimal(upper), Decimal(tick)) for upper, tick in
              (("10", ".01"), ("50", ".05"), ("100", ".1"), ("500", ".5"), ("1000", "1")))


def _decimal(value, name, *, positive=False, rate=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(name + ": FINITE_DECIMAL_REQUIRED")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(name + ": FINITE_DECIMAL_REQUIRED") from None
    if not number.is_finite() or number < 0 or positive and number <= 0 or rate and number >= Decimal(".1"):
        raise ValueError(name + ": DECIMAL_OUT_OF_RANGE")
    return number


def _ids(value):
    return (isinstance(value, list) and bool(value)
            and all(isinstance(item, str) and item and item == item.strip() for item in value)
            and len(set(value)) == len(value))


def _session(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("CANONICAL_SESSION_REQUIRED")
    return value


def _clock(session, time="13:30:00"):
    return timestamp(session + "T" + time + "+08:00").isoformat()


def _tick(price):
    return next((tick for upper, tick in _GRID if price < upper), Decimal("5"))


def _legal(price):
    return price > 0 and price % _tick(price) == 0


def _round_price(price, direction):
    """Round model barriers only; observed OHLC is never rounded into evidence."""
    rounding = ROUND_FLOOR if direction == "down" else ROUND_CEILING
    result = (price / _tick(price)).to_integral_value(rounding=rounding) * _tick(price)
    if not _legal(result):
        raise ValueError("MODEL_BARRIER_OUTSIDE_LEGAL_PRICE_GRID")
    return result


def validate_reference_path_policy(policy):
    """Validate a bounded, source-bound research policy without a fallback."""
    fr3_l2 = isinstance(policy, dict) and policy.get("version") == "FR3_H20_OPEN_LIMIT3_FIXED_TARGET_COST_AFTER_L2"
    fr3 = isinstance(policy, dict) and policy.get("version") in {
        "FR3_H20_OPEN_ONLY_FIXED_TARGET_COST_AFTER_R1", "FR3_H20_OPEN_LIMIT3_FIXED_TARGET_COST_AFTER_L2"}
    expected = dict(schema=PATH_SCHEMA, version=PATH_VERSION, instrument_type="ORDINARY_STOCK",
                    entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND",
                    exit_rule="FIRST_STOP_OR_NET_2R_ELSE_H20_CLOSE",
                    exit_activation="NEXT_SESSION_AFTER_FILL", shares_rule="NATIVE_REFERENCE_SHARES",
                    price_grid=GRID_VERSION)
    if fr3:
        expected.update(version="FR3_H20_OPEN_ONLY_FIXED_TARGET_COST_AFTER_R1",
            exit_rule="FIRST_STOP_OR_FIXED_TARGET_ELSE_H20_CLOSE",
            entry_guard_policy="FR3_REPRICE_FIXED_STOP_AND_RESISTANCE",
            target_policy="FIXED_CAUSAL_PLAN_TARGET_NO_2R_SUBSTITUTION",
            label_version="FR3_FIXED_TARGET_NEAR_BOTTOM_GIVEN_FILL_R1",
            historical_deployment_claim=False)
    if fr3_l2:
        expected.update(version="FR3_H20_OPEN_LIMIT3_FIXED_TARGET_COST_AFTER_L2",
            entry_rule="CONTEXT_BOUND_OFFICIAL_SESSION_OPENS_IN_FROZEN_BAND",
            label_version="FR3_L2_FIXED_TARGET_NEAR_BOTTOM_GIVEN_FILL_R1",
            waiting_policy="PRIOR_COMPLETED_SESSION_LOW_LT_SUPPORT_OR_HIGH_GTE_TARGET",
            maximum_leader_open_opportunities=3, horizon_anchor="SIGNAL_SESSION")
    if (not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected.items())
            or policy.get("market") not in {"TWSE", "TPEX"}
            or policy.get("coordinate_policy") not in {COORDINATE_POLICY, FINAL_COORDINATE_POLICY}
            or not _ids(policy.get("source_evidence_ids")) or not policy.get("available_at")):
        raise ValueError("EXPLICIT_REFERENCE_PATH_POLICY_REQUIRED")
    if _decimal(policy.get("slippage_bps"), "slippage_bps") != 0:
        raise ValueError("ONLY_EXPLICIT_ZERO_SLIPPAGE_REFERENCE_SUPPORTED")
    if policy["coordinate_policy"] == FINAL_COORDINATE_POLICY and policy["market"] != "TWSE":
        raise ValueError("FINAL_ACTIONS_POLICY_TWSE_ONLY")
    timestamp(policy["available_at"])
    cost = policy.get("cost_policy")
    if (not isinstance(cost, dict) or not isinstance(cost.get("version"), str) or not cost["version"]
            or cost.get("rounding") != "ROUND_HALF_UP" or not _ids(cost.get("evidence_ids"))
            or not cost.get("available_at")):
        raise ValueError("SOURCE_BOUND_COST_POLICY_REQUIRED")
    timestamp(cost["available_at"])
    if _decimal(cost.get("currency_quantum"), "currency_quantum", positive=True) != Decimal(".01"):
        raise ValueError("TWD_CENT_COST_QUANTUM_REQUIRED")
    for key in ("buy_rate", "sell_rate"):
        _decimal(cost.get(key), key, rate=True)
    for key in ("minimum_buy_fee", "minimum_sell_fee"):
        _decimal(cost.get(key), key)
    if (not cost.get("effective_start") or not cost.get("effective_end")
            or _session(cost["effective_start"]) > _session(cost["effective_end"])):
        raise ValueError("COST_ASSUMPTION_EFFECTIVE_RANGE_REQUIRED")
    # Cost rates are declared research assumptions, not an assertion that the
    # model/policy had been deployed at historical event time.
    return deepcopy(policy)


def _fee(notional, side, cost):
    amount = max(notional * _decimal(cost[side + "_rate"], side + "_rate", rate=True),
                 _decimal(cost["minimum_" + side + "_fee"], "minimum_fee"))
    return amount.quantize(Decimal(".01"), rounding=ROUND_HALF_UP)


def _cash(price, shares, side, cost):
    notional = price * shares
    fee = _fee(notional, side, cost)
    return notional + fee if side == "buy" else notional - fee


def _net_2r_price(entry, native_stop, executable_stop, shares, cost):
    """Smallest legal price delivering two native gross reference-risk units.

    Search legal prices directly: cent-rounded fees create downward jumps in
    off-grid sell proceeds, so continuous bisection followed by upward rounding
    can miss the cash threshold. Legal-grid proceeds are nondecreasing: each
    positive integer-share tick increases notional by at least one cent and
    the declared fee rate is below 10%. The bracket retains a failing legal
    lower price and passing legal upper price until they are adjacent.
    This outcome barrier does not replace the evidence-derived resistance.
    """
    debit = _cash(entry, shares, "buy", cost)
    risk = shares * (entry - native_stop)
    if risk <= 0:
        raise ValueError("POSITIVE_NATIVE_REFERENCE_RISK_REQUIRED")
    required = debit + 2 * risk
    sell_rate = _decimal(cost["sell_rate"], "sell_rate", rate=True)
    minimum_fee = _decimal(cost["minimum_sell_fee"], "minimum_sell_fee")
    # max(rate * notional, minimum) rounded to cents is at most rate *
    # notional + minimum + .005. The extra cent makes the upper bracket
    # conservative without changing the exact required proceeds or fees.
    lower = entry
    upper = _round_price((required + minimum_fee + Decimal(".01")) /
                         (shares * (1 - sell_rate)), "up")
    if _cash(upper, shares, "sell", cost) < required:
        raise ValueError("NET_2R_DISCRETE_CASH_THRESHOLD_FAILED")
    while True:
        middle = _round_price((lower + upper) / 2, "down")
        if middle <= lower:
            break
        if middle >= upper:
            raise ValueError("NET_2R_GRID_SEARCH_PRECISION_EXHAUSTED")
        if _cash(middle, shares, "sell", cost) >= required:
            upper = middle
        else:
            lower = middle
    target = upper
    if _cash(target, shares, "sell", cost) < required:
        raise ValueError("NET_2R_DISCRETE_CASH_THRESHOLD_FAILED")
    return target, debit, risk, debit - _cash(executable_stop, shares, "sell", cost)


def estimate_reference_costs(entry, stop, target, shares, policy):
    """Cutoff-only planning costs from the declared path fee/grid policy.

    No path or future outcome is used. The adverse floor gap at the sell stop
    belongs to ``cost_stop``; a favorable target-rounding gap is not credited.
    A missing resistance target uses entry only to estimate unused up fees,
    never manufactures a +2R resistance or a reward/risk qualification.
    """
    policy = validate_reference_path_policy(policy)
    if type(shares) is not int or shares <= 0:
        raise ValueError("POSITIVE_REFERENCE_SHARES_REQUIRED")
    with localcontext() as ctx:
        ctx.prec = 50
        entry = _decimal(entry, "reference_entry", positive=True)
        stop = _decimal(stop, "reference_stop", positive=True)
        if stop >= entry or not _legal(entry):
            raise ValueError("LEGAL_ENTRY_AND_POSITIVE_NATIVE_RISK_REQUIRED")
        target_value = entry if target is None else _decimal(target, "reference_target", positive=True)
        if target is not None and target_value <= entry:
            raise ValueError("RESISTANCE_TARGET_MUST_EXCEED_ENTRY")
        executable_stop = _round_price(stop, "down")
        executable_target = _round_price(target_value, "up")
        cost = policy["cost_policy"]
        buy_fee = _fee(entry * shares, "buy", cost)
        stop_fee = _fee(executable_stop * shares, "sell", cost)
        up_fee = _fee(executable_target * shares, "sell", cost)
        rounding_loss = (stop - executable_stop) * shares
        cost_up, cost_stop = (buy_fee + up_fee) / shares, (buy_fee + stop_fee + rounding_loss) / shares
        return dict(cost_up_per_share=float(cost_up), cost_stop_per_share=float(cost_stop),
                    cost_policy_id=identity(cost), path_policy_id=identity(policy),
                    detail=dict(schema="les-v7-reference-cost-estimate/v1", basis="CUTOFF_ONLY_DECLARED_ASSUMPTIONS",
                                currency="TWD", shares=shares, entry=str(entry), native_stop=str(stop),
                                executable_stop=str(executable_stop), native_target=str(target_value) if target is not None else None,
                                executable_target=str(executable_target), target_present=target is not None,
                                buy_fee=str(buy_fee), stop_sell_fee=str(stop_fee), target_sell_fee=str(up_fee),
                                stop_rounding_loss=str(rounding_loss), cost_up_per_share=str(cost_up),
                                cost_stop_per_share=str(cost_stop), counterfactual_policy_created_at=policy["available_at"]))


def path_window(history_packet, expected_sessions):
    """Detach source-bound OHLC path rows without inventing missing sessions.

    ``bars`` use the existing training Date/OHLC/Volume/Available At/Complete
    convention; ``bar_lineage`` binds each date to official source-row IDs.
    All bar/lineage dates, including future dates outside this window, must be
    unique. A missing row is an explicit incomplete placeholder, not a fill.
    No shared evidence references escape and no caller data is modified.
    """
    if (not isinstance(history_packet, dict) or not history_packet.get("coordinate_id")
            or not isinstance(history_packet.get("bars"), list)
            or not isinstance(history_packet.get("bar_lineage"), list)
            or not isinstance(expected_sessions, list) or not 1 <= len(expected_sessions) <= 21
            or expected_sessions != sorted(set(expected_sessions))):
        raise ValueError("SOURCE_BOUND_PATH_HISTORY_REQUIRED")
    for session in expected_sessions:
        _session(session)
    bars, lineage = {}, {}
    for bar in history_packet["bars"]:
        if not isinstance(bar, dict):
            raise ValueError("PATH_BAR_RECORD_REQUIRED")
        session = _session(bar.get("Date"))
        if session in bars:
            raise ValueError("DUPLICATE_PATH_BAR_SESSION")
        bars[session] = bar
    for record in history_packet["bar_lineage"]:
        if not isinstance(record, dict):
            raise ValueError("PATH_BAR_LINEAGE_RECORD_REQUIRED")
        session = _session(record.get("date"))
        if session in lineage:
            raise ValueError("DUPLICATE_PATH_BAR_LINEAGE_SESSION")
        lineage[session] = record
    output = []
    for session in expected_sessions:
        row, proof = bars.get(session), lineage.get(session)
        if row is None:
            output.append(dict(session=session, complete=False,
                               coordinate_id=history_packet["coordinate_id"], evidence_ids=[]))
            continue
        result = dict(session=session, coordinate_id=history_packet["coordinate_id"],
                      complete=row.get("Complete"), available_at=row.get("Available At"),
                      evidence_ids=deepcopy(proof.get("source_row_ids", [])) if proof else [])
        result.update({name: deepcopy(row.get(source)) for name, source in
                       (("open", "Open"), ("high", "High"), ("low", "Low"),
                        ("close", "Close"), ("volume", "Volume"))})
        if "Corporate Action State" in row:
            result["corporate_action_state"] = deepcopy(row["Corporate Action State"])
        output.append(result)
    return output


def _marker(session, phase, evidence_ids, price):
    return dict(session=session, phase=phase,
                at=_clock(session, "09:00:00" if phase == "OPEN" else "13:30:00"),
                time_precision="OBSERVED_SESSION_OPEN" if phase == "OPEN" else PRECISION,
                interval_start=_clock(session, "09:00:00"), interval_end=_clock(session),
                price=str(price), evidence_ids=list(evidence_ids))


def evaluate_reference_path(plan: ReferencePlan, window: list[dict], policy: dict, *,
                            expected_sessions: list[str], cutoff: str, coordinate_id: str,
                            source_evidence_ids: list[str], corporate_action_coverage: dict | None = None,
                            entry_route: dict | None = None) -> dict:
    """Return execution evidence plus diagnostics; never change a plan/account.

    ``window`` is signal session plus 20 following official sessions. Each row
    requires complete OHLCV, source IDs, availability and matching coordinate.
    Corporate-action coverage must prove no price-coordinate-changing action
    throughout the window. Missing evidence returns DATA_GAP, not false no-fill.
    Policy provenance binds retrospective research assumptions; it is not a
    historical deployment or source-authenticity certificate.
    """
    if not isinstance(plan, ReferencePlan):
        raise ValueError("NATIVE_REFERENCE_PLAN_REQUIRED")
    policy = validate_reference_path_policy(policy)
    from domain.les_v7.fr3_plan import PLAN_VERSION as fr3_plan_version, PATH_VERSION as fr3_r1_path_version
    from domain.les_v7.fr3_l2 import (PLAN_VERSION as l2_plan_version, PATH_VERSION as l2_path_version,
        LABEL_VERSION as l2_label_version, LEADER, LEADER_ENTRY_RULE, EXPIRY as l2_expiry)
    fr3_l2 = plan.policy_version == l2_plan_version
    fr3 = plan.policy_version in {fr3_plan_version, l2_plan_version}
    fr3_path_version = l2_path_version if fr3_l2 else fr3_r1_path_version
    fr3_label_version = l2_label_version if fr3_l2 else "FR3_FIXED_TARGET_NEAR_BOTTOM_GIVEN_FILL_R1"
    if (fr3 and policy["version"] != fr3_path_version
            or not fr3 and policy["version"] in {fr3_r1_path_version, l2_path_version}):
        raise ValueError("FR3_REFERENCE_PLAN_PATH_VERSION_MISMATCH")
    cutoff_at = timestamp(cutoff)
    session = cutoff_at.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
    if cutoff_at != timestamp(_clock(session)):
        raise ValueError("SIGNAL_SESSION_CLOSE_CUTOFF_REQUIRED")
    if (not isinstance(expected_sessions, list) or len(expected_sessions) != 21
            or expected_sessions != sorted(set(expected_sessions))
            or expected_sessions[0] != session):
        raise ValueError("OFFICIAL_H20_SESSION_SEQUENCE_REQUIRED")
    for value in expected_sessions:
        _session(value)
    if (policy["cost_policy"]["effective_start"] > expected_sessions[1]
            or policy["cost_policy"]["effective_end"] < expected_sessions[-1]):
        raise ValueError("COST_ASSUMPTION_DOES_NOT_COVER_H20_EXECUTION")
    if not coordinate_id or not _ids(source_evidence_ids):
        raise ValueError("RAW_WINDOW_SOURCE_BINDING_REQUIRED")
    diagnostic = dict(schema="les-v7-reference-path-diagnostic/v1", reference_plan_id=plan.plan_id,
        path_policy_id=identity(policy), source_evidence_ids=list(source_evidence_ids),
        raw_window_sha256=identity(window), coordinate_id=coordinate_id, expected_sessions=list(expected_sessions),
        time_precision=PRECISION, research_only=True, approved_for_live=False,
        source_authenticity="SOURCE_AUTHENTICITY_NOT_CERTIFIED", reason_codes=[])

    def gap(reason):
        return dict(status="DATA_GAP", execution=None,
                    diagnostic=dict(diagnostic, reason_codes=[reason]))

    from domain.les_v7.plan_r10 import R15_VERSION as r15_plan_version
    from domain.les_v7.policy_r16 import PLAN_VERSION as r16_plan_version
    from domain.les_v7.policy_r17 import PLAN_VERSION as r17_plan_version
    from domain.les_v7.policy_r18 import PLAN_VERSION as r18_plan_version
    from domain.les_v7.policy_r19 import PLAN_VERSION as r19_plan_version
    conditional_plan = plan.policy_version in {r16_plan_version, r17_plan_version, r18_plan_version, r19_plan_version, fr3_plan_version, l2_plan_version}
    conditional_prefix = ("FR3" if fr3 else "R19" if plan.policy_version == r19_plan_version else
                          "R18" if plan.policy_version == r18_plan_version else "R17" if plan.policy_version == r17_plan_version else "R16")
    route_policy, verified_route = None, None
    if plan.policy_version == r15_plan_version:
        if not isinstance(entry_route, dict):
            return gap("R15_ENTRY_ROUTE_BINDING_REQUIRED")
        try:
            from domain.les_v7.plan_r10 import bind_entry_route
            route_policy = entry_route["reference_policy"]
            if (not isinstance(route_policy, dict)
                    or identity(route_policy.get("cost_path_policy")) != identity(policy)):
                return gap("R15_ENTRY_ROUTE_PATH_POLICY_MISMATCH")
            expected_route = bind_entry_route(plan, route_policy,
                strategy_decision=entry_route["strategy_decision"],
                risk_route=entry_route["risk_route"], setup_variant=entry_route.get("setup_variant"))
            if identity(entry_route) != identity(expected_route):
                return gap("R15_ENTRY_ROUTE_BINDING_MISMATCH")
            verified_route = expected_route
        except (KeyError, TypeError, ValueError, AttributeError):
            return gap("R15_ENTRY_ROUTE_BINDING_INVALID")
    if conditional_plan:
        if not isinstance(entry_route, dict):
            return gap(conditional_prefix + "_ENTRY_ROUTE_BINDING_REQUIRED")
        try:
            from domain.les_v7.plan_r16 import R16ReferencePlan, bind_entry_route
            plan_type = R16ReferencePlan
            if plan.policy_version == r17_plan_version:
                from domain.les_v7.plan_r17 import R17ReferencePlan, bind_entry_route
                plan_type = R17ReferencePlan
            if plan.policy_version == r18_plan_version:
                from domain.les_v7.plan_r18 import R18ReferencePlan, bind_entry_route
                plan_type = R18ReferencePlan
            if plan.policy_version == r19_plan_version:
                from domain.les_v7.plan_r19 import R19ReferencePlan, bind_entry_route
                plan_type = R19ReferencePlan
            if fr3:
                from domain.les_v7.fr3_plan import FR3ReferencePlan, bind_entry_route
                plan_type = FR3ReferencePlan
            if fr3_l2:
                from domain.les_v7.fr3_l2 import FR3L2ReferencePlan
                plan_type = FR3L2ReferencePlan
            if not isinstance(plan, plan_type):
                return gap(conditional_prefix + "_SOURCE_BOUND_PLAN_REQUIRED")
            route_policy = entry_route["reference_policy"]
            if (not isinstance(route_policy, dict)
                    or identity(route_policy.get("cost_path_policy")) != identity(policy)):
                return gap(conditional_prefix + "_ENTRY_ROUTE_PATH_POLICY_MISMATCH")
            expected_route = bind_entry_route(plan, route_policy,
                strategy_decision=entry_route["strategy_decision"],
                risk_route=entry_route["risk_route"], setup_variant=entry_route.get("setup_variant"))
            expected_rule = LEADER_ENTRY_RULE if fr3_l2 and plan.context == LEADER else "NEXT_OFFICIAL_SESSION_OPEN_IN_BAND"
            expected_expiry = l2_expiry if fr3_l2 and plan.context == LEADER else "NEXT_SESSION_CLOSE"
            if (identity(entry_route) != identity(expected_route)
                    or expected_route.get("entry_rule") != expected_rule
                    or expected_route.get("expiry") != expected_expiry):
                return gap(conditional_prefix + "_ENTRY_ROUTE_BINDING_MISMATCH")
            # A plan's band and its actual observation are independently
            # required to lie on the declared legal grid. R16 never rounds an
            # observed open into an eligible conditional price.
            if any(not _legal(_decimal(value, "band_price", positive=True)) for value in plan.band):
                return gap(conditional_prefix + "_ILLEGAL_CONDITIONAL_BAND")
            basis = expected_route.get("entry_basis")
            if (basis not in {"CUTOFF_CLOSE", "STRUCTURAL_CONDITIONAL_BAND"}
                    or basis == "STRUCTURAL_CONDITIONAL_BAND" and plan.entry != plan.band[1]
                    or basis == "CUTOFF_CLOSE" and plan.entry != plan.signal_close):
                return gap(conditional_prefix + "_ENTRY_BASIS_INVALID")
            verified_route = expected_route
        except (KeyError, TypeError, ValueError, AttributeError):
            return gap(conditional_prefix + "_ENTRY_ROUTE_BINDING_INVALID")
    elif (isinstance(plan.policy_version, str)
            and plan.policy_version.startswith("V7_SETUP_STRUCTURE_H20_R16")):
        return gap("UNSUPPORTED_R16_REFERENCE_PLAN_VERSION")
    elif (isinstance(plan.policy_version, str)
            and plan.policy_version.startswith("V7_SETUP_STRUCTURE_H20_R17")):
        return gap("UNSUPPORTED_R17_REFERENCE_PLAN_VERSION")
    elif (isinstance(plan.policy_version, str)
            and plan.policy_version.startswith("V7_SUPPORT_RESISTANCE_LIMIT_R18")):
        return gap("UNSUPPORTED_R18_REFERENCE_PLAN_VERSION")
    elif (isinstance(plan.policy_version, str)
            and plan.policy_version.startswith("V7_RISK_BOUNDED_ENTRY_R19")):
        return gap("UNSUPPORTED_R19_REFERENCE_PLAN_VERSION")
    elif (isinstance(plan.policy_version, str)
            and plan.policy_version.startswith("FR3_SOURCE_BOUND_LIMIT_H20_")):
        return gap("UNSUPPORTED_FR3_REFERENCE_PLAN_VERSION")

    if not isinstance(window, list) or len(window) != 21 or [r.get("session") for r in window] != expected_sessions:
        return gap("INCOMPLETE_OR_CONFLICTING_H20_WINDOW")
    coverage = corporate_action_coverage
    if (not isinstance(coverage, dict) or coverage.get("schema") != "les-v7-coordinate-coverage/v1"
            or coverage.get("policy") != policy["coordinate_policy"]
            or coverage.get("status") != "VERIFIED_NO_ACTIONS"
            or coverage.get("coordinate_id") != coordinate_id or coverage.get("market") != policy["market"]
            or not isinstance(coverage.get("start_session"), str)
            or not isinstance(coverage.get("end_session"), str)
            or coverage["start_session"] > expected_sessions[0]
            or coverage["end_session"] < expected_sessions[-1]
            or not _ids(coverage.get("evidence_ids")) or not coverage.get("available_at")):
        return gap("CORPORATE_ACTION_OR_COORDINATE_COVERAGE_UNVERIFIED")
    try:
        _session(coverage["start_session"])
        _session(coverage["end_session"])
        observed_at = timestamp(coverage["available_at"])
        coverage_at = observed_at
        if policy["coordinate_policy"] == FINAL_COORDINATE_POLICY:
            if (coverage.get("historical_information_set_certified") is not False
                    or coverage.get("historical_available_at") is not None
                    or not coverage.get("actions_packet_id")
                    or coverage.get("category_scope") != ["EX_RIGHTS_AND_DIVIDENDS",
                        "CAPITAL_REDUCTION_RESUMPTION", "PAR_VALUE_CHANGE_RESUMPTION"]):
                return gap("FINAL_ACTIONS_RECONSTRUCTION_NOT_EXPLICIT")
            # Declared retrospective label completion, not a changed receipt.
            coverage_at = timestamp(_clock(expected_sessions[-1]))
    except (ValueError, TypeError):
        return gap("CORPORATE_ACTION_COVERAGE_TIME_INVALID")
    if observed_at < timestamp(_clock(coverage["end_session"])):
        return gap("CORPORATE_ACTION_COVERAGE_PRECEDES_WINDOW_COMPLETION")
    diagnostic["coordinate_coverage_sha256"] = identity(coverage)
    rows, availability, lineage = [], [], list(source_evidence_ids)
    with localcontext() as ctx:
        ctx.prec = 50
        for raw in window:
            if (raw.get("complete") is not True or raw.get("coordinate_id") != coordinate_id
                    or not _ids(raw.get("evidence_ids"))):
                return gap("INCOMPLETE_OR_UNBOUND_PATH_BAR")
            if raw.get("corporate_action_state") not in {None, "NO_ACTION"}:
                return gap("CORPORATE_ACTION_PATH_REQUIRES_VERIFIED_RECOORDINATION")
            try:
                values = {name: _decimal(raw[name], "path_" + name, positive=True)
                          for name in ("open", "high", "low", "close")}
                volume = _decimal(raw["volume"], "path_volume")
                available = timestamp(raw["available_at"])
            except (KeyError, ValueError, TypeError):
                return gap("INVALID_OR_MISSING_PATH_OHLCV_OR_AVAILABILITY")
            if (values["high"] < max(values["open"], values["close"])
                    or values["low"] > min(values["open"], values["close"])
                    or values["low"] > values["high"] or any(not _legal(v) for v in values.values())
                    or available < timestamp(_clock(raw["session"]))):
                return gap("CONFLICTING_OR_ILLEGAL_PATH_BAR")
            if volume <= 0:
                return gap("ZERO_VOLUME_OR_POTENTIAL_SUSPENSION")
            rows.append(dict(raw, **values))
            availability.append(available)
            lineage.extend(raw["evidence_ids"])
        # Today's fixed counterfactual policy is build metadata, NOT late
        # market evidence. Its creation time must not relabel historical
        # outcomes as if the policy had been deployed then or alter OOF age.
        if (conditional_plan
                and _decimal(plan.signal_close, "signal_close", positive=True) != rows[0]["close"]):
            return gap(conditional_prefix + "_SIGNAL_CLOSE_SOURCE_MISMATCH")
        path_available_at = max(*availability, coverage_at)
        lineage.extend(coverage["evidence_ids"])
        lineage.extend(policy["source_evidence_ids"])
        lineage.extend(policy["cost_policy"]["evidence_ids"])
        execution = dict(policy_id=plan.policy_id, policy_version=plan.policy_version, plan_id=plan.plan_id,
            path_policy_version=PATH_VERSION, path_policy_id=identity(policy), coordinate_id=coordinate_id,
            path_evidence_ids=sorted(set(lineage)), path_available_at=path_available_at.isoformat(),
            raw_window_sha256=diagnostic["raw_window_sha256"],
            coordinate_coverage_sha256=diagnostic["coordinate_coverage_sha256"],
            cost_policy_version=policy["cost_policy"]["version"], cost_policy_id=identity(policy["cost_policy"]),
            counterfactual_policy_created_at=policy["available_at"],
            counterfactual_cost_policy_created_at=policy["cost_policy"]["available_at"],
            historical_deployment_claim=False,
            time_precision=PRECISION, research_only=True, approved_for_live=False,
            exit_activation="NEXT_SESSION_AFTER_FILL", slippage_assumption="EXPLICIT_ZERO_NOT_CALIBRATED",
            liquidity_assumption="DAILY_OPEN_REFERENCE_NO_DEPTH_PROOF",
            capacity_guarantee=False, partial_fills_modeled=False,
            path_order_verified=True, filled=False)
        if fr3:
            execution.update(path_policy_version=fr3_path_version,
                label_version=fr3_label_version,
                native_plan_target=plan.target,
                target_policy="FIXED_CAUSAL_PLAN_TARGET_NO_2R_SUBSTITUTION")
        if policy["coordinate_policy"] == FINAL_COORDINATE_POLICY:
            execution.update(reference_evidence_policy=FINAL_COORDINATE_POLICY,
                source_observed_at=observed_at.isoformat(), original_historical_available_at=None,
                label_availability_basis="RETROSPECTIVE_H20_WINDOW_CLOSE_NOT_ORIGINAL_PUBLICATION",
                historical_information_set_certified=False)
        fill_index = 1
        entry = rows[fill_index]["open"]
        if conditional_plan:
            execution.update(reference_entry_route=verified_route,
                conditional_entry=dict(entry_basis=verified_route["entry_basis"],
                    setup_variant=verified_route.get("setup_variant"),
                    signal_close=verified_route["signal_close"], target_method=verified_route["target_method"],
                    entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", expiry="NEXT_SESSION_CLOSE",
                    eligible_session=expected_sessions[1], expires_at=_clock(expected_sessions[1]),
                    observed_open=float(entry), observation="OBSERVED_SESSION_OPEN",
                    intraday_touch_authorizes_fill=False, price_condition_met=False,
                    state="EXPIRED_AT_NEXT_SESSION_CLOSE"))
        if fr3_l2:
            # Select an opening before observing that session's future range.
            # Only an unfilled, subsequently completed session can cancel a
            # later opening. The signal-anchored H20 window never moves.
            maximum = 3 if plan.context == LEADER else 1
            conditional = execution["conditional_entry"]
            conditional.update(entry_rule=verified_route["entry_rule"], expiry=plan.expiry,
                eligible_sessions=expected_sessions[1:maximum+1],
                expires_at=_clock(expected_sessions[maximum]), waiting_observations=[],
                maximum_open_opportunities=maximum,
                state="EXPIRED_AT_LAST_ELIGIBLE_SESSION_CLOSE")
            chosen, cancelled = None, None
            for candidate in range(1, maximum+1):
                opening = rows[candidate]["open"]
                in_band = _decimal(plan.band[0], "band_low", positive=True) <= opening <= _decimal(plan.band[1], "band_high", positive=True)
                observation = dict(session=expected_sessions[candidate], open=float(opening),
                    observation="OBSERVED_SESSION_OPEN", price_condition_met=in_band,
                    evidence_ids=list(rows[candidate]["evidence_ids"]))
                conditional["waiting_observations"].append(observation)
                if in_band:
                    chosen = candidate
                    break
                # This close is used only for subsequent opens. Never revoke
                # an opening fill based on lows/highs observed later that day.
                if candidate < maximum:
                    broken = rows[candidate]["low"] < _decimal(plan.support, "original_support", positive=True)
                    target_seen = rows[candidate]["high"] >= _decimal(plan.target, "original_target", positive=True)
                    observation.update(completed_at=_clock(expected_sessions[candidate]),
                        support_broken=broken, target_seen=target_seen)
                    if broken or target_seen:
                        cancelled = "FR3_L2_WAIT_SUPPORT_BROKEN" if broken else "FR3_L2_WAIT_TARGET_ALREADY_REACHED"
                        conditional.update(state="CANCELLED_BEFORE_SUBSEQUENT_OPEN",
                            cancelled_at=_clock(expected_sessions[candidate]), cancellation_reason=cancelled)
                        break
            if chosen is None:
                reason = cancelled or "FR3_L2_NO_ELIGIBLE_OPEN_IN_FROZEN_BAND"
                return dict(status="VERIFIED_NO_FILL", execution=dict(execution, no_fill_reason=reason),
                    diagnostic=dict(diagnostic, reason_codes=[reason]))
            fill_index = chosen
            entry = rows[fill_index]["open"]
            conditional.update(eligible_session=expected_sessions[fill_index], observed_open=float(entry),
                filled_opportunity=fill_index)
        if not _decimal(plan.band[0], "band_low", positive=True) <= entry <= _decimal(plan.band[1], "band_high", positive=True):
            return dict(status="VERIFIED_NO_FILL", execution=dict(execution, no_fill_reason="NEXT_OPEN_OUTSIDE_NATIVE_BAND"),
                        diagnostic=dict(diagnostic, reason_codes=["NEXT_OPEN_OUTSIDE_NATIVE_BAND"]))
        if conditional_plan:
            execution["conditional_entry"]["price_condition_met"] = True
            execution["conditional_entry"]["state"] = "PRICE_CONFIRMED_PENDING_REFERENCE_RISK_CHECK"
        stop = _decimal(plan.stop, "native_stop", positive=True)
        if entry <= stop:
            return dict(status="VERIFIED_NO_FILL", execution=dict(execution, no_fill_reason="NEXT_OPEN_NOT_ABOVE_NATIVE_STOP"),
                        diagnostic=dict(diagnostic, reason_codes=["NEXT_OPEN_NOT_ABOVE_NATIVE_STOP"]))
        from domain.les_v7.policy_r13 import PLAN_VERSION as r13_plan_version
        from domain.les_v7.plan_r10 import R14_VERSION as r14_plan_version
        if plan.policy_version in {"V7_SETUP_STRUCTURE_H20_R10", r13_plan_version, r14_plan_version,
                                    r15_plan_version, r16_plan_version, r17_plan_version, r18_plan_version, r19_plan_version, fr3_plan_version, l2_plan_version}:
            from domain.les_v7.plan_r10 import evaluate_entry, policy as structure_policy
            if fr3:
                from domain.les_v7.fr3_plan import evaluate_entry
            prefix = ("FR3" if fr3 else "R19" if plan.policy_version == r19_plan_version else
                      "R18" if plan.policy_version == r18_plan_version else
                      "R17" if plan.policy_version == r17_plan_version else
                      "R16" if plan.policy_version == r16_plan_version else
                      "R15" if plan.policy_version == r15_plan_version else
                      "R14" if plan.policy_version == r14_plan_version else
                      "R13" if plan.policy_version == r13_plan_version else "R10")
            if policy.get("entry_guard_policy") != ("FR3_REPRICE_FIXED_STOP_AND_RESISTANCE" if fr3
                    else "R10_REPRICE_NATIVE_STOP_AND_RESISTANCE"):
                return gap(prefix + "_ACTUAL_ENTRY_GUARD_MISSING")
            bound = route_policy if plan.policy_version in {r15_plan_version, r16_plan_version, r17_plan_version, r18_plan_version, r19_plan_version, fr3_plan_version, l2_plan_version} else structure_policy(
                dict(policy_id="REFERENCE_H20", reference_shares=plan.reference_shares,
                    entry_band_atr=[0., .25], cost_path_policy=policy), available_at=policy["available_at"],
                    version=plan.policy_version)
            tactical = verified_route is not None and verified_route["risk_route"] == "TACTICAL"
            check = evaluate_entry(plan, float(entry), bound,
                **(dict(starter=tactical) if plan.policy_version in {r15_plan_version, r16_plan_version, r17_plan_version, r18_plan_version, r19_plan_version, fr3_plan_version, l2_plan_version} else {}))
            if check["state"] != "PASS":
                if conditional_plan:
                    execution["conditional_entry"]["state"] = "EXPIRED_REFERENCE_RISK_NOT_PASS"
                return dict(status="VERIFIED_NO_FILL", execution=dict(execution,
                    no_fill_reason=prefix + "_ACTUAL_ENTRY_RISK_NOT_PASS", entry_risk_check=check),
                    diagnostic=dict(diagnostic, reason_codes=[prefix + "_ACTUAL_ENTRY_RISK_NOT_PASS"]))
            execution["entry_risk_check"] = check
            if verified_route is not None:
                execution["reference_entry_route"] = verified_route
        shares, cost = plan.reference_shares, policy["cost_policy"]
        if _decimal(rows[fill_index]["volume"], "entry_volume") < shares:
            return gap("DAILY_VOLUME_BELOW_REFERENCE_ENTRY_SIZE")
        sell_stop = _round_price(stop, "down")
        target, debit, risk, net_stop_loss = _net_2r_price(entry, stop, sell_stop, shares, cost)
        net_2r_diagnostic = target
        if fr3:
            target = _decimal(plan.target, "fixed_reference_target", positive=True)
            if not _legal(target) or target <= entry:
                return gap("FR3_FIXED_REFERENCE_TARGET_INVALID")
        target_reason = "FIXED_TARGET" if fr3 else "NET_2R"
        target_hit_key = "reference_target_first_hit_at" if fr3 else "net_2r_first_hit_at"
        execution.update(filled=True, entry=float(entry), initial_stop=plan.stop, cutoff_atr=plan.atr,
            shares=shares, filled_at=_clock(rows[fill_index]["session"], "09:00:00"),
            initial_stop_first_hit_at=None, net_2r_first_hit_at=None,
            hit_markers={}, executable_stop=str(sell_stop), executable_net_2r=str(target),
            native_resistance_target=plan.target,
            reference_sizing=dict(rule="NATIVE_REFERENCE_SHARES", shares=shares, reference_plan_id=plan.plan_id),
            cost_arithmetic=dict(currency="TWD", quantum="0.01", arithmetic="DECIMAL_50_PRECISION",
                                 buy_debit=str(debit), native_gross_reference_risk=str(risk),
                                 net_stop_loss=str(net_stop_loss), net_2r_basis="NET_PNL_OVER_NATIVE_GROSS_RISK",
                                 buy_fee=str(_fee(entry * shares, "buy", cost))))
        if conditional_plan:
            execution["conditional_entry"]["state"] = "REFERENCE_FILLED_AT_ELIGIBLE_OPEN"
            execution.update(native_plan_target=plan.target, target_method=plan.target_method,
                native_resistance_target=(plan.target if plan.target_method ==
                    "CONFIRMED_UNBROKEN_OVERHEAD_RESISTANCE" else None))
        diagnostic.update(executable_stop=str(sell_stop), executable_net_2r=str(target),
                          ignored_entry_session_range=True)
        if fr3:
            execution.update(path_policy_version=fr3_path_version,
                label_version=fr3_label_version,
                reference_target_first_hit_at=None, executable_target=str(target),
                executable_net_2r=str(net_2r_diagnostic),
                net_2r_diagnostic_only=True, target_policy="FIXED_CAUSAL_PLAN_TARGET_NO_2R_SUBSTITUTION")
            diagnostic.update(executable_target=str(target), executable_net_2r=str(net_2r_diagnostic),
                net_2r_diagnostic_only=True)
        exit_price, marker, reason = None, None, None
        for row in rows[fill_index+1:]:
            stop_open, target_open = row["open"] <= sell_stop, row["open"] >= target
            if stop_open or target_open:
                # The official opening observation precedes all remaining
                # session-range extrema; never fill a gap at a model threshold.
                reason = "STOP" if stop_open else target_reason
                exit_price, marker = row["open"], _marker(row["session"], "OPEN", row["evidence_ids"], row["open"])
            else:
                stop_hit, target_hit = row["low"] <= sell_stop, row["high"] >= target
                if stop_hit and target_hit:
                    if _decimal(row["volume"], "exit_volume") < shares:
                        return gap("DAILY_VOLUME_BELOW_REFERENCE_EXIT_SIZE")
                    markers = {"STOP": _marker(row["session"], "SESSION_RANGE", row["evidence_ids"], sell_stop),
                               target_reason: _marker(row["session"], "SESSION_RANGE", row["evidence_ids"], target)}
                    execution.update(path_order_verified=False, hit_markers=markers,
                        initial_stop_first_hit_at=markers["STOP"]["at"],
                        realized_net_pnl=None, path_status="PATH_AMBIGUOUS")
                    execution[target_hit_key] = markers[target_reason]["at"]
                    return dict(status="PATH_AMBIGUOUS", execution=execution,
                                diagnostic=dict(diagnostic, reason_codes=["SAME_SESSION_OPPOSING_BARRIERS_ORDER_UNKNOWN"]))
                if stop_hit or target_hit:
                    reason = "STOP" if stop_hit else target_reason
                    exit_price = sell_stop if stop_hit else target
                    marker = _marker(row["session"], "SESSION_RANGE", row["evidence_ids"], exit_price)
            if reason:
                if _decimal(row["volume"], "exit_volume") < shares:
                    return gap("DAILY_VOLUME_BELOW_REFERENCE_EXIT_SIZE")
                break
        if exit_price is None:
            reason, exit_price = "H20_CLOSE", rows[-1]["close"]
            marker = _marker(rows[-1]["session"], "CLOSE", rows[-1]["evidence_ids"], exit_price)
            marker["time_precision"] = "OBSERVED_SESSION_CLOSE"
            if _decimal(rows[-1]["volume"], "exit_volume") < shares:
                return gap("DAILY_VOLUME_BELOW_REFERENCE_EXIT_SIZE")
        proceeds = _cash(exit_price, shares, "sell", cost)
        pnl = proceeds - debit
        execution.update(path_status="VERIFIED_FILLED", exit_reason=reason, exit_price=float(exit_price),
                         realized_net_pnl=float(pnl), realized_at=marker["at"],
                         hit_markers={reason: marker},
                         cost_arithmetic=dict(execution["cost_arithmetic"], sell_proceeds=str(proceeds),
                             sell_fee=str(_fee(exit_price * shares, "sell", cost)), realized_net_pnl=str(pnl)))
        if reason != "H20_CLOSE":
            execution["initial_stop_first_hit_at" if reason == "STOP" else target_hit_key] = marker["at"]
        return dict(status="VERIFIED_FILLED", execution=execution, diagnostic=diagnostic)
